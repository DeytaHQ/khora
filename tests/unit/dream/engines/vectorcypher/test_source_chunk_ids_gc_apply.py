"""Apply-mode unit tests for source_chunk_ids GC handler (#668).

The Postgres + SQLite SQL paths use a thin in-process stub
(:class:`_FakeSession`) so we can pin the handler's contract without
spinning up a database. The handler emits raw SQLAlchemy ``text()``
statements; the fake captures them and serves curated query results.

Integration smoke against the real DB is deferred to
``tests/integration/dream/test_orchestrator_apply_e2e.py`` (covered by
the orchestrator test fixtures from #699).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.engines.vectorcypher.source_chunk_ids_gc import (
    apply_vectorcypher_source_chunk_ids_gc,
)
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeSession:
    """Captures SQL statements and serves curated rows.

    The handler always issues:
      * 1x SELECT for the current ``source_chunk_ids`` value (per entity).
      * 1x UPDATE to rewrite the array.

    Tests prime ``select_responses`` keyed by entity_id and assert on
    ``update_calls`` afterwards.
    """

    dialect_name: str = "postgresql"
    select_responses: dict[UUID, list[Any]] = field(default_factory=dict)
    update_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    select_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    bind: Any = None

    def __post_init__(self) -> None:
        # Mirror real AsyncSession.bind.dialect.name access path.
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=self.dialect_name))

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        text_str = str(stmt)
        params = params or {}
        upper = text_str.lstrip().upper()
        if upper.startswith("UPDATE"):
            self.update_calls.append((text_str, params))
            return SimpleNamespace(rowcount=1)
        # SELECT path: handler asks for the current source_chunk_ids of
        # one entity_id (param "eid"). On SQLite the value is the hex
        # string form of the UUID; normalize before lookup.
        self.select_calls.append((text_str, params))
        raw = params.get("eid") or params.get("entity_id")
        entity_id = _coerce_uuid(raw)
        rows = self.select_responses.get(entity_id, [])
        return _FakeResult(rows)


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def first(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class _FakeCoordinator:
    """Just a marker — the handler does not call coordinator methods directly."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_op(
    entity_id: UUID,
    *,
    dead: list[UUID],
    after: list[UUID],
    before_length: int | None = None,
) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="mutation",
        op_type=OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC,
        inputs=(
            {
                "entity_id": str(entity_id),
                "before_length": before_length if before_length is not None else len(dead) + len(after),
                "dead_uuids": [str(u) for u in dead],
            },
        ),
        outputs=(
            {
                "after_array": [str(u) for u in after],
                "after_length": len(after),
            },
        ),
        decision="planned",
        rationale="GC dead refs.",
        started_at=datetime.now(UTC),
        duration_ms=0.1,
        namespace_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# Tests — Postgres path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postgres_happy_path_rewrites_array_and_returns_undo() -> None:
    entity_id = uuid4()
    survivors = [uuid4(), uuid4()]
    dead = [uuid4(), uuid4()]
    full = survivors + dead

    op = _build_op(entity_id, dead=dead, after=survivors, before_length=4)
    session = _FakeSession(
        dialect_name="postgresql",
        select_responses={
            entity_id: [SimpleNamespace(source_chunk_ids=list(full))],
        },
    )

    undo = await apply_vectorcypher_source_chunk_ids_gc(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    assert undo.op_id == op.op_id
    assert undo.op_type == str(op.op_type)
    assert "chunk_id" not in undo.before  # safety floor: never use this key
    entries = undo.before["entities"]
    assert len(entries) == 1
    entry = entries[0]
    assert UUID(entry["entity_id"]) == entity_id
    assert [UUID(u) for u in entry["previous_source_chunk_ids"]] == full

    assert len(session.update_calls) == 1
    sql, params = session.update_calls[0]
    assert "UPDATE entities" in sql
    assert "source_chunk_ids" in sql
    assert params["eid"] == entity_id
    # New array is the survivor set (UUIDs, not strings).
    new_array = params.get("new_array")
    assert [UUID(str(u)) for u in new_array] == survivors


@pytest.mark.asyncio
async def test_idempotent_when_array_already_matches_plan() -> None:
    """If the current array already equals the planned after-array, no UPDATE fires."""
    entity_id = uuid4()
    survivors = [uuid4(), uuid4()]
    dead = [uuid4()]
    op = _build_op(entity_id, dead=dead, after=survivors)

    # Current state already matches the planned `after_array` — handler
    # treats this as a noop on replay.
    session = _FakeSession(
        dialect_name="postgresql",
        select_responses={
            entity_id: [SimpleNamespace(source_chunk_ids=list(survivors))],
        },
    )

    undo = await apply_vectorcypher_source_chunk_ids_gc(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    assert undo.before.get("noop") is True
    assert session.update_calls == []


@pytest.mark.asyncio
async def test_sqlite_path_writes_json_string_back() -> None:
    entity_id = uuid4()
    survivors = [uuid4(), uuid4()]
    dead = [uuid4()]
    full = survivors + dead

    op = _build_op(entity_id, dead=dead, after=survivors)
    # SQLite stores the column as JSON-text.
    session = _FakeSession(
        dialect_name="sqlite",
        select_responses={
            entity_id: [SimpleNamespace(source_chunk_ids=json.dumps([str(u) for u in full]))],
        },
    )

    undo = await apply_vectorcypher_source_chunk_ids_gc(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    assert "entities" in undo.before
    assert "chunk_id" not in undo.before

    assert len(session.update_calls) == 1
    _, params = session.update_calls[0]
    new_array = params["new_array"]
    # SQLite path writes JSON-text — round-trip cleanly.
    decoded = json.loads(new_array)
    assert [UUID(s) for s in decoded] == survivors


@pytest.mark.asyncio
async def test_undo_round_trip_can_restore_previous_array() -> None:
    """The undo snapshot must be sufficient to reconstruct the array pre-apply."""
    entity_id = uuid4()
    survivors = [uuid4(), uuid4(), uuid4()]
    dead = [uuid4(), uuid4()]
    full = survivors + dead

    op = _build_op(entity_id, dead=dead, after=survivors, before_length=5)
    session = _FakeSession(
        dialect_name="postgresql",
        select_responses={
            entity_id: [SimpleNamespace(source_chunk_ids=list(full))],
        },
    )

    undo = await apply_vectorcypher_source_chunk_ids_gc(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    # Reconstruct what an undoer would write back.
    entry = undo.before["entities"][0]
    restored = [UUID(u) for u in entry["previous_source_chunk_ids"]]
    assert restored == full


@pytest.mark.asyncio
async def test_handler_does_not_touch_documents_table() -> None:
    """Every SQL statement issued must target ``entities`` (never ``documents``)."""
    entity_id = uuid4()
    op = _build_op(entity_id, dead=[uuid4()], after=[uuid4()])
    session = _FakeSession(
        dialect_name="postgresql",
        select_responses={
            entity_id: [SimpleNamespace(source_chunk_ids=[uuid4(), uuid4()])],
        },
    )

    await apply_vectorcypher_source_chunk_ids_gc(op, coordinator=_FakeCoordinator(), session=session)

    for sql, _ in session.select_calls + session.update_calls:
        upper = sql.upper()
        assert " DOCUMENTS " not in upper and "FROM DOCUMENTS" not in upper
        assert "UPDATE DOCUMENTS" not in upper


@pytest.mark.asyncio
async def test_undo_record_carries_no_chunk_id_top_level_key() -> None:
    """Safety floor: a top-level 'chunk_id' key trips DreamForbiddenOpError."""
    entity_id = uuid4()
    op = _build_op(entity_id, dead=[uuid4()], after=[uuid4()])
    session = _FakeSession(
        dialect_name="postgresql",
        select_responses={
            entity_id: [SimpleNamespace(source_chunk_ids=[uuid4(), uuid4()])],
        },
    )

    undo = await apply_vectorcypher_source_chunk_ids_gc(op, coordinator=_FakeCoordinator(), session=session)
    assert "chunk_id" not in undo.before
