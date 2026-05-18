"""Unit tests for the vectorcypher edge-pruning dream op (#671, Phase 5.2).

Planner: pure SELECT — finds relationships matching the three-conjunct
predicate (``confidence < threshold`` AND ``valid_to IS NULL`` AND every
``source_chunk_ids`` UUID is dead), one op per candidate.

Apply: stamps ``valid_to = now()`` and captures the pre-state
``confidence`` and ``valid_to`` so undo can clear ``valid_to``.

The Postgres + SQLite SQL paths use a thin in-process stub
(:class:`_FakeSession`) — no real database needed for unit tests.
Integration is deferred to the orchestrator end-to-end fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.engines.vectorcypher.prune_edges import (
    apply_vectorcypher_prune_edges,
    plan_vectorcypher_prune_edges,
)
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeSession:
    """Captures SQL statements and serves curated rows."""

    dialect_name: str = "postgresql"
    select_rows: list[Any] = field(default_factory=list)
    pre_state_rows: dict[UUID, Any] = field(default_factory=dict)
    update_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    select_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    update_rowcount: int = 1
    bind: Any = None

    def __post_init__(self) -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=self.dialect_name))

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        text_str = str(stmt)
        params = params or {}
        upper = text_str.lstrip().upper()
        if upper.startswith("UPDATE"):
            self.update_calls.append((text_str, params))
            return SimpleNamespace(rowcount=self.update_rowcount)
        self.select_calls.append((text_str, params))
        # Pre-state lookup for the apply handler (single relationship)
        rid = params.get("rid")
        if rid is not None and self.pre_state_rows:
            row = self.pre_state_rows.get(_coerce_uuid(rid))
            return _FakeResult([row] if row is not None else [])
        return _FakeResult(list(self.select_rows))


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
    """Yields a transaction whose ``session`` is a :class:`_FakeSession`."""

    def __init__(self, session: _FakeSession, namespace_id: UUID) -> None:
        self._session = session
        self._namespace_id = namespace_id

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        del namespace_id
        return self._namespace_id

    def transaction(self) -> _FakeTxnCtx:
        return _FakeTxnCtx(self._session)


class _FakeTxnCtx:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> _FakeTxnCtx:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    rel_id: UUID,
    confidence: float = 0.2,
    valid_to: datetime | None = None,
    relationship_type: str = "ASSOCIATED_WITH",
    source_chunk_ids: list[UUID] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=rel_id,
        confidence=confidence,
        valid_to=valid_to,
        relationship_type=relationship_type,
        source_chunk_ids=source_chunk_ids or [],
    )


def _build_op(
    rel_id: UUID,
    *,
    confidence: float = 0.2,
    relationship_type: str = "ASSOCIATED_WITH",
) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="mutation",
        op_type=OpKind.VECTORCYPHER_PRUNE_EDGES,
        inputs=(
            {
                "relationship_id": str(rel_id),
                "relationship_type": relationship_type,
                "confidence": confidence,
                "threshold": 0.4,
            },
        ),
        outputs=(),
        decision="planned",
        rationale="orphan low-confidence edge",
        started_at=datetime.now(UTC),
        namespace_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_emits_one_op_per_matching_relationship() -> None:
    ns = uuid4()
    rel_a = uuid4()
    rel_b = uuid4()
    session = _FakeSession(
        select_rows=[
            _row(rel_id=rel_a, confidence=0.1),
            _row(rel_id=rel_b, confidence=0.35),
        ],
    )
    coordinator = _FakeCoordinator(session, ns)

    ops = await plan_vectorcypher_prune_edges(ns, coordinator=coordinator)

    assert len(ops) == 2
    rel_ids = {UUID(op.inputs[0]["relationship_id"]) for op in ops}
    assert rel_ids == {rel_a, rel_b}
    for op in ops:
        assert op.op_type == OpKind.VECTORCYPHER_PRUNE_EDGES
        assert op.decision == "planned"
        assert op.phase == "mutation"
        assert op.inputs[0]["threshold"] == 0.4
        assert op.inputs[0]["relationship_type"] == "ASSOCIATED_WITH"


@pytest.mark.asyncio
async def test_planner_default_predicate_list_is_associated_with_only() -> None:
    """Operators must opt in explicitly to broader pruning."""
    ns = uuid4()
    session = _FakeSession(select_rows=[])
    coordinator = _FakeCoordinator(session, ns)

    await plan_vectorcypher_prune_edges(ns, coordinator=coordinator)

    assert len(session.select_calls) == 1
    sql, params = session.select_calls[0]
    # Default predicate list is exactly ["ASSOCIATED_WITH"] — no other types
    # are queried unless the caller passes them.
    rel_types = params.get("rel_types")
    assert list(rel_types) == ["ASSOCIATED_WITH"]


@pytest.mark.asyncio
async def test_planner_respects_target_predicates_override() -> None:
    ns = uuid4()
    session = _FakeSession(select_rows=[])
    coordinator = _FakeCoordinator(session, ns)

    await plan_vectorcypher_prune_edges(
        ns,
        coordinator=coordinator,
        target_predicates=("ASSOCIATED_WITH", "MENTIONS"),
    )

    sql, params = session.select_calls[0]
    assert set(params["rel_types"]) == {"ASSOCIATED_WITH", "MENTIONS"}


@pytest.mark.asyncio
async def test_planner_respects_threshold_override() -> None:
    ns = uuid4()
    session = _FakeSession(select_rows=[_row(rel_id=uuid4(), confidence=0.5)])
    coordinator = _FakeCoordinator(session, ns)

    ops = await plan_vectorcypher_prune_edges(
        ns,
        coordinator=coordinator,
        confidence_threshold=0.7,
    )

    assert len(ops) == 1
    assert ops[0].inputs[0]["threshold"] == 0.7
    _, params = session.select_calls[0]
    assert params["threshold"] == 0.7


@pytest.mark.asyncio
async def test_planner_apply_mode_raises() -> None:
    ns = uuid4()
    session = _FakeSession()
    coordinator = _FakeCoordinator(session, ns)

    with pytest.raises(NotImplementedError):
        await plan_vectorcypher_prune_edges(
            ns,
            coordinator=coordinator,
            mode="apply",
        )


@pytest.mark.asyncio
async def test_planner_emits_empty_when_no_candidates() -> None:
    ns = uuid4()
    session = _FakeSession(select_rows=[])
    coordinator = _FakeCoordinator(session, ns)

    ops = await plan_vectorcypher_prune_edges(ns, coordinator=coordinator)
    assert ops == ()


# ---------------------------------------------------------------------------
# Apply handler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_stamps_valid_to_and_captures_pre_state() -> None:
    rel_id = uuid4()
    op = _build_op(rel_id, confidence=0.25)
    session = _FakeSession(
        pre_state_rows={
            rel_id: SimpleNamespace(confidence=0.25, valid_to=None),
        },
    )

    undo = await apply_vectorcypher_prune_edges(
        op,
        coordinator=_FakeCoordinator(session, uuid4()),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    assert undo.op_id == op.op_id
    assert undo.op_type == str(op.op_type)

    entries = undo.before["relationships"]
    assert len(entries) == 1
    entry = entries[0]
    assert UUID(entry["relationship_id"]) == rel_id
    assert entry["previous_confidence"] == 0.25
    assert entry["previous_valid_to"] is None

    assert len(session.update_calls) == 1
    sql, params = session.update_calls[0]
    assert "UPDATE relationships" in sql
    assert "valid_to" in sql
    assert params["rid"] == rel_id
    assert isinstance(params["ts"], datetime)


@pytest.mark.asyncio
async def test_apply_is_idempotent_when_already_pruned() -> None:
    """If valid_to is already set, no UPDATE fires."""
    rel_id = uuid4()
    already = datetime.now(UTC)
    op = _build_op(rel_id)
    session = _FakeSession(
        pre_state_rows={
            rel_id: SimpleNamespace(confidence=0.2, valid_to=already),
        },
    )

    undo = await apply_vectorcypher_prune_edges(
        op,
        coordinator=_FakeCoordinator(session, uuid4()),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    assert undo.before.get("noop") is True
    assert session.update_calls == []


@pytest.mark.asyncio
async def test_apply_skips_missing_relationship() -> None:
    """If the relationship row vanished between plan and apply, return a noop."""
    rel_id = uuid4()
    op = _build_op(rel_id)
    session = _FakeSession(pre_state_rows={})  # no row found

    undo = await apply_vectorcypher_prune_edges(
        op,
        coordinator=_FakeCoordinator(session, uuid4()),
        session=session,
    )

    assert undo.before.get("noop") is True
    assert session.update_calls == []


@pytest.mark.asyncio
async def test_undo_round_trip_can_clear_valid_to() -> None:
    """The undo snapshot is sufficient to reverse the soft-delete."""
    rel_id = uuid4()
    op = _build_op(rel_id, confidence=0.3)
    session = _FakeSession(
        pre_state_rows={rel_id: SimpleNamespace(confidence=0.3, valid_to=None)},
    )

    undo = await apply_vectorcypher_prune_edges(
        op,
        coordinator=_FakeCoordinator(session, uuid4()),
        session=session,
    )

    entry = undo.before["relationships"][0]
    # The undoer would write: UPDATE relationships SET valid_to = previous_valid_to
    # confidence is captured so the audit log can show the prune rationale.
    assert entry["previous_valid_to"] is None
    assert entry["previous_confidence"] == 0.3


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_dream_config_has_prune_edges_knobs() -> None:
    from khora.dream.config import DreamConfig

    cfg = DreamConfig()
    assert cfg.prune_edges_enabled is False
    assert cfg.prune_edges_target_predicates == ["ASSOCIATED_WITH"]
    assert cfg.prune_edges_confidence_threshold == 0.4


def test_opkind_has_vectorcypher_prune_edges() -> None:
    assert OpKind.VECTORCYPHER_PRUNE_EDGES == "vectorcypher_prune_edges"
