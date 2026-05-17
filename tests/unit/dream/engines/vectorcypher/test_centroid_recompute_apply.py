"""Apply-mode unit tests for centroid_recompute handler (#668).

The handler is Postgres-only in v0.15 — SQLite raises
:class:`DreamForbiddenOpError` (LanceDB owns vectors off-row and the
vector backend's ``update_entity_embedding`` commits out-of-band,
violating the orchestrator's caller-owned-transaction contract). The
SQLite-aware path lands in v0.16.

These unit tests pin three branches:

  * ``skip_singleton`` / ``skip_multimodal`` — handler returns a noop
    :class:`UndoRecord` with no DB activity.
  * ``centroid`` — handler reads current embedding, computes weighted
    mean, writes via ``UPDATE entities`` on the same session.
  * ``re_embed`` — without an injected embedder, the handler raises
    :class:`DreamForbiddenOpError`; with one, it calls the embedder and
    writes the new vector.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.engines.vectorcypher.centroid_recompute import (
    apply_vectorcypher_centroid_recompute,
)
from khora.dream.exceptions import DreamForbiddenOpError
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, dialect_name: str = "postgresql") -> None:
        self.dialect_name = dialect_name
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
        self.update_calls: list[tuple[str, dict[str, Any]]] = []
        self.select_calls: list[tuple[str, dict[str, Any]]] = []
        self.select_responses: dict[UUID, Any] = {}

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        text_str = str(stmt)
        params = params or {}
        upper = text_str.lstrip().upper()
        if upper.startswith("UPDATE"):
            self.update_calls.append((text_str, params))
            return SimpleNamespace(rowcount=1)
        self.select_calls.append((text_str, params))
        eid = params.get("eid") or params.get("entity_id")
        try:
            key = eid if isinstance(eid, UUID) else UUID(str(eid))
        except (TypeError, ValueError):
            key = None
        row = self.select_responses.get(key)
        return _Result(row)


class _Result:
    def __init__(self, row: Any) -> None:
        self._row = row

    def first(self) -> Any:
        return self._row


class _FakeCoordinator:
    """Marker only — apply handler writes via session, not coordinator."""


# ---------------------------------------------------------------------------
# Op builders
# ---------------------------------------------------------------------------


def _op_centroid(
    canonical_id: UUID,
    member_ids: list[UUID],
    new_vec: list[float],
) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="mutation",
        op_type=OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
        inputs=(
            {
                "cluster_size": len(member_ids),
                "member_ids": [str(m) for m in member_ids],
                "canonical_entity_id": str(canonical_id),
            },
        ),
        outputs=(
            {
                "new_embedding_vector": new_vec,
                "source_count": len(member_ids),
            },
        ),
        decision="centroid",
        rationale="lev within threshold",
        started_at=datetime.now(UTC),
        duration_ms=0.5,
        namespace_id=uuid4(),
    )


def _op_re_embed(canonical_id: UUID, name: str = "OpenAI") -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="mutation",
        op_type=OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
        inputs=(
            {
                "cluster_size": 2,
                "member_ids": [str(canonical_id), str(uuid4())],
                "canonical_entity_id": str(canonical_id),
            },
        ),
        outputs=(
            {
                "new_embedding_text": name,
                "embedding_model": "text-embedding-3-small",
                "source_count": 2,
            },
        ),
        decision="re_embed",
        rationale="lev out of threshold",
        started_at=datetime.now(UTC),
        duration_ms=0.5,
        namespace_id=uuid4(),
    )


def _op_skip(decision: str) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="mutation",
        op_type=OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
        inputs=({"cluster_size": 1, "member_ids": [str(uuid4())]},),
        outputs=(),
        decision=decision,
        rationale="skip",
        started_at=datetime.now(UTC),
        duration_ms=0.0,
        namespace_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["skip_singleton", "skip_multimodal"])
async def test_skip_branches_return_noop_undo_no_db_writes(decision: str) -> None:
    session = _FakeSession()
    op = _op_skip(decision)

    undo = await apply_vectorcypher_centroid_recompute(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    assert undo.before == {"noop": True}
    assert session.update_calls == []
    assert "chunk_id" not in undo.before


@pytest.mark.asyncio
async def test_centroid_branch_updates_and_returns_previous_embedding() -> None:
    canonical_id = uuid4()
    member_ids = [canonical_id, uuid4(), uuid4()]
    new_vec = [0.1, 0.2, 0.3]
    previous = [0.7, 0.7, 0.0]

    session = _FakeSession(dialect_name="postgresql")
    session.select_responses[canonical_id] = SimpleNamespace(embedding=previous)

    op = _op_centroid(canonical_id, member_ids, new_vec)

    undo = await apply_vectorcypher_centroid_recompute(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    assert "chunk_id" not in undo.before
    cluster_entries = undo.before["clusters"]
    assert len(cluster_entries) == 1
    entry = cluster_entries[0]
    assert UUID(entry["entity_id"]) == canonical_id
    assert entry["previous_embedding"] == previous

    # Exactly one UPDATE was issued, against ``entities``.
    assert len(session.update_calls) == 1
    sql, params = session.update_calls[0]
    assert "UPDATE entities" in sql
    assert params["eid"] == canonical_id
    assert params["embedding"] == new_vec


@pytest.mark.asyncio
async def test_re_embed_without_embedder_raises_forbidden() -> None:
    canonical_id = uuid4()
    op = _op_re_embed(canonical_id)
    session = _FakeSession(dialect_name="postgresql")

    with pytest.raises(DreamForbiddenOpError, match="embedder"):
        await apply_vectorcypher_centroid_recompute(
            op,
            coordinator=_FakeCoordinator(),
            session=session,
        )

    assert session.update_calls == []


@pytest.mark.asyncio
async def test_re_embed_with_embedder_writes_new_vector() -> None:
    canonical_id = uuid4()
    op = _op_re_embed(canonical_id, name="International Business Machines")
    session = _FakeSession(dialect_name="postgresql")
    previous = [0.0, 0.0, 0.0]
    session.select_responses[canonical_id] = SimpleNamespace(embedding=previous)

    calls: list[str] = []

    async def fake_embedder(text_value: str) -> list[float]:
        calls.append(text_value)
        return [0.6, 0.8, 0.0]

    undo = await apply_vectorcypher_centroid_recompute(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
        embedder=fake_embedder,
    )

    assert isinstance(undo, UndoRecord)
    assert calls == ["International Business Machines"]
    assert len(session.update_calls) == 1
    _, params = session.update_calls[0]
    # Re-embed must L2-normalize before write — [0.6, 0.8, 0.0] already
    # has norm 1.0, so post-normalize equals the input.
    assert params["embedding"] == pytest.approx([0.6, 0.8, 0.0])
    assert undo.before["clusters"][0]["previous_embedding"] == previous


@pytest.mark.asyncio
async def test_centroid_replay_is_idempotent() -> None:
    """When the canonical's embedding already matches the planned vector, no UPDATE."""
    canonical_id = uuid4()
    new_vec = [1.0, 0.0, 0.0]
    session = _FakeSession(dialect_name="postgresql")
    # Current embedding equals planned vector — handler treats as noop.
    session.select_responses[canonical_id] = SimpleNamespace(embedding=list(new_vec))
    op = _op_centroid(canonical_id, [canonical_id, uuid4()], new_vec)

    undo = await apply_vectorcypher_centroid_recompute(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert undo.before.get("noop") is True
    assert session.update_calls == []


@pytest.mark.asyncio
async def test_sqlite_dialect_raises_forbidden() -> None:
    """SQLite path is gated until v0.16."""
    canonical_id = uuid4()
    op = _op_centroid(canonical_id, [canonical_id, uuid4()], [0.1, 0.0, 0.0])
    session = _FakeSession(dialect_name="sqlite")

    with pytest.raises(DreamForbiddenOpError, match="Postgres-only"):
        await apply_vectorcypher_centroid_recompute(
            op,
            coordinator=_FakeCoordinator(),
            session=session,
        )


@pytest.mark.asyncio
async def test_handler_does_not_touch_documents_table() -> None:
    canonical_id = uuid4()
    session = _FakeSession()
    session.select_responses[canonical_id] = SimpleNamespace(embedding=[0.1, 0.2, 0.3])
    op = _op_centroid(canonical_id, [canonical_id, uuid4()], [0.9, 0.1, 0.0])

    await apply_vectorcypher_centroid_recompute(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    for sql, _ in session.select_calls + session.update_calls:
        upper = sql.upper()
        assert "DOCUMENTS" not in upper
