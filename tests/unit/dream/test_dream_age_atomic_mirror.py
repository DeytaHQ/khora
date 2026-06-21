"""AGE atomic same-transaction dream mirror routing (#1307).

AGE is an Apache-AGE extension in the SAME Postgres DB as the dream-apply SQL,
so its soft-delete mirror runs IN the apply transaction (atomic with the PG
commit) instead of post-commit eventual-consistency. These are mock-driven
orchestrator-routing assertions: a fake AGE-shaped graph backend records the
verb call (and the session it ran on) so we can assert the orchestrator

  - calls ``soft_invalidate_relationships_batch`` with ``session=txn.session``
    (the apply transaction's session - the atomicity guarantee),
  - does NOT queue a ``graph_mirror_pending`` row for AGE (no reconcile needed),
  - does NOT also run the post-commit ``_mirror_dream_op`` for AGE,
  - and still routes a genuinely-remote backend (no ``mirror_in_transaction``)
    through the post-commit / pre-mark path unchanged.

The repo's pgvector/pg17 image does not ship AGE, so a live cross-store atomic
prune is out of scope here (matches #1279, which covered AGE mock-only); the
in-tx routing + session threading is what these tests pin.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.config import DreamConfig
from khora.dream.orchestrator import DreamOrchestrator
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.dream.runstore import GraphMirrorPending

_NS = uuid4()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _AGEFakeGraph:
    """AGE-shaped backend: advertises in-tx mirror + the flat-soft-delete kinds.

    Records every ``soft_invalidate_relationships_batch`` call (including the
    session it ran on) so the test can assert the orchestrator threaded the
    apply transaction's session.
    """

    def __init__(self) -> None:
        self.invalidated: list[dict[str, Any]] = []

    def mirror_in_transaction(self) -> bool:
        return True

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        return frozenset({OpKind.VECTORCYPHER_PRUNE_EDGES, OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE})

    async def soft_invalidate_relationships_batch(
        self,
        relationship_ids: list[UUID],
        *,
        namespace_id: UUID,
        invalidated_at: datetime,
        session: Any | None = None,
    ) -> int:
        self.invalidated.append(
            {
                "ids": list(relationship_ids),
                "namespace_id": namespace_id,
                "invalidated_at": invalidated_at,
                "session": session,
            }
        )
        return len(relationship_ids)


class _RemoteFakeGraph:
    """Genuinely-remote backend: no ``mirror_in_transaction`` -> post-commit path."""

    def __init__(self) -> None:
        self.invalidated: list[dict[str, Any]] = []

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        return frozenset({OpKind.VECTORCYPHER_PRUNE_EDGES})

    async def soft_invalidate_relationships_batch(
        self, relationship_ids: list[UUID], *, namespace_id: UUID, invalidated_at: datetime
    ) -> int:
        self.invalidated.append({"ids": list(relationship_ids), "namespace_id": namespace_id})
        return len(relationship_ids)


class _SessionStub:
    """Apply-transaction session stub. ``bind=None`` keeps the dialect gate happy."""

    def __init__(self) -> None:
        self.bind = None

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(first=lambda: None, all=lambda: [])


class _TxnCtx:
    def __init__(self, coordinator: _Coordinator) -> None:
        self._coordinator = coordinator

    async def __aenter__(self) -> Any:
        return SimpleNamespace(session=self._coordinator.session)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Emulate the real coordinator.transaction(): commit on clean exit,
        # roll back on exception. The AGE mirror ran on this session, so a
        # rollback here unwinds the soft-delete WITH the PG apply (atomicity).
        if exc_type is None:
            self._coordinator.committed = True
        else:
            self._coordinator.rolled_back = True
        return None


class _Coordinator:
    def __init__(self, graph: Any) -> None:
        self._graph = graph
        self.session = _SessionStub()
        self.committed = False
        self.rolled_back = False

    def transaction(self) -> _TxnCtx:
        return _TxnCtx(self)

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        # Idempotent (returns input) - keeps the in-tx mirror's namespace resolve
        # honest without standing up a real namespace-version table.
        return namespace_id


class _RunStore:
    """Records graph_mirror_pending marks so the test can assert AGE queues none."""

    def __init__(self) -> None:
        self.marked: list[GraphMirrorPending] = []

    async def advance_checkpoint(self, run_id: UUID, seq: int, *, session: Any | None = None) -> None:
        del run_id, seq, session

    async def mark_graph_mirror_pending(
        self, run_id: UUID, entry: GraphMirrorPending, *, session: Any | None = None
    ) -> None:
        del run_id, session
        self.marked.append(entry)

    async def clear_graph_mirror_pending(self, run_id: UUID, op_seq: int) -> None:
        del run_id, op_seq

    async def get_open_graph_mirror_pending(self, namespace_id: UUID) -> list[tuple[UUID, GraphMirrorPending]]:
        del namespace_id
        return []


class _FakeKB:
    def __init__(self, coordinator: _Coordinator) -> None:
        self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
        self._engine_name = "stub"
        self.storage = coordinator


def _orch(graph: Any, *, run_store: _RunStore | None = None) -> DreamOrchestrator:
    orch = DreamOrchestrator(_FakeKB(_Coordinator(graph)), DreamConfig(enabled=True), sinks=[])
    if run_store is not None:
        orch._run_store_cache = run_store  # type: ignore[attr-defined]
        orch._run_store_resolved = True  # type: ignore[attr-defined]
    return orch


def _prune_undo(rel_ids: list[UUID]) -> UndoRecord:
    return UndoRecord(
        op_id=uuid4(),
        op_type=str(OpKind.VECTORCYPHER_PRUNE_EDGES),
        before={"relationships": [{"relationship_id": str(r)} for r in rel_ids]},
        applied_at=datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC),
    )


def _prune_op() -> DreamOp:
    return DreamOp(op_id=uuid4(), phase="apply", op_type=OpKind.VECTORCYPHER_PRUNE_EDGES, namespace_id=_NS)


# ---------------------------------------------------------------------------
# _apply_one_op routes AGE in-tx
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_one_op_mirrors_age_on_txn_session() -> None:
    """The AGE soft-delete runs on the apply transaction's session (atomic)."""
    graph = _AGEFakeGraph()
    store = _RunStore()
    orch = _orch(graph, run_store=store)

    rel_ids = [uuid4(), uuid4()]

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del op, coordinator, session
        return _prune_undo(rel_ids)

    run_id = uuid4()
    await orch._apply_one_op(run_id=run_id, seq=0, op=_prune_op(), handler=_handler, namespace_id=_NS)

    # The verb fired with the pruned ids, namespace-scoped, on the txn session.
    assert len(graph.invalidated) == 1
    call = graph.invalidated[0]
    assert call["ids"] == rel_ids
    assert call["namespace_id"] == _NS
    assert call["session"] is orch._kb.storage.session  # the apply transaction session
    # No graph_mirror_pending row queued for AGE (atomic, no reconcile needed).
    assert store.marked == []


@pytest.mark.asyncio
async def test_apply_one_op_commits_on_clean_age_mirror() -> None:
    """A successful AGE in-tx mirror lets the apply transaction commit."""
    graph = _AGEFakeGraph()
    orch = _orch(graph, run_store=_RunStore())

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del op, coordinator, session
        return _prune_undo([uuid4()])

    await orch._apply_one_op(run_id=uuid4(), seq=0, op=_prune_op(), handler=_handler, namespace_id=_NS)
    assert orch._kb.storage.committed is True
    assert orch._kb.storage.rolled_back is False


@pytest.mark.asyncio
async def test_apply_one_op_age_mirror_failure_rolls_back_atomically() -> None:
    """An AGE mirror error propagates and rolls the WHOLE apply op back - the PG
    soft-delete is not committed without the graph SET (atomicity, #1307)."""
    graph = _AGEFakeGraph()

    async def _boom(*_a: Any, **_k: Any) -> int:
        raise RuntimeError("age cypher boom")

    graph.soft_invalidate_relationships_batch = _boom  # type: ignore[method-assign]
    store = _RunStore()
    orch = _orch(graph, run_store=store)

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del op, coordinator, session
        return _prune_undo([uuid4()])

    with pytest.raises(RuntimeError, match="age cypher boom"):
        await orch._apply_one_op(run_id=uuid4(), seq=0, op=_prune_op(), handler=_handler, namespace_id=_NS)

    # The transaction rolled back (PG apply + graph SET both unwound), and no
    # pending row was queued (AGE never uses the reconciler).
    assert orch._kb.storage.rolled_back is True
    assert orch._kb.storage.committed is False
    assert store.marked == []


@pytest.mark.asyncio
async def test_apply_one_op_age_noop_apply_marks_nothing() -> None:
    """An already-pruned (empty-target) AGE op mirrors nothing and queues nothing."""
    graph = _AGEFakeGraph()
    store = _RunStore()
    orch = _orch(graph, run_store=store)

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del op, coordinator, session
        return _prune_undo([])  # no relationships -> no targets

    await orch._apply_one_op(run_id=uuid4(), seq=0, op=_prune_op(), handler=_handler, namespace_id=_NS)
    assert graph.invalidated == []
    assert store.marked == []


@pytest.mark.asyncio
async def test_apply_one_op_remote_backend_premarks_pending() -> None:
    """A genuinely-remote backend keeps the pre-mark path (NOT in-tx)."""
    graph = _RemoteFakeGraph()
    store = _RunStore()
    orch = _orch(graph, run_store=store)

    rel_ids = [uuid4()]

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del op, coordinator, session
        return _prune_undo(rel_ids)

    await orch._apply_one_op(run_id=uuid4(), seq=0, op=_prune_op(), handler=_handler, namespace_id=_NS)

    # Remote backend does NOT mirror in-tx; it pre-marks for the post-commit path.
    assert graph.invalidated == []
    assert len(store.marked) == 1
    assert store.marked[0].op_type == str(OpKind.VECTORCYPHER_PRUNE_EDGES)


# ---------------------------------------------------------------------------
# Direct _mirror_dream_op_in_tx unit behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mirror_in_tx_returns_false_for_remote_backend() -> None:
    orch = _orch(_RemoteFakeGraph())
    session = object()
    handled = await orch._mirror_dream_op_in_tx(session, _NS, _prune_op(), _prune_undo([uuid4()]))
    assert handled is False


@pytest.mark.asyncio
async def test_mirror_in_tx_owns_routing_even_for_unmirrorable_kind() -> None:
    """An in-tx backend owns the routing for a kind it can't mirror, so the
    caller must not also queue a pending row (returns True, mirrors nothing)."""
    graph = _AGEFakeGraph()
    orch = _orch(graph)
    op = DreamOp(op_id=uuid4(), phase="apply", op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES, namespace_id=_NS)
    undo = UndoRecord(op_id=op.op_id, op_type=str(op.op_type), before={"merges": []}, applied_at=datetime.now(UTC))
    handled = await orch._mirror_dream_op_in_tx(object(), _NS, op, undo)
    assert handled is True
    assert graph.invalidated == []


@pytest.mark.asyncio
async def test_mirrors_in_transaction_probe() -> None:
    assert _orch(_AGEFakeGraph())._mirrors_in_transaction() is True
    assert _orch(_RemoteFakeGraph())._mirrors_in_transaction() is False
