"""PG+Neo4j divergence warning for vectorcypher apply (#875).

The four vectorcypher mutation apply handlers
(``apply_vectorcypher_dedupe_entities`` and siblings) mutate the
relational store only - none of them touch a configured graph backend.
When the coordinator carries a graph backend (Neo4j / Memgraph /
Neptune / AGE), the post-apply state diverges: the SQL row is
soft-deleted / rewritten but the graph mirror still reflects the
pre-apply shape.

This module verifies that the orchestrator emits a ``logger.warning``
on each such apply so operators get an honest signal until the
graph-store mirror lands in a future release. The warning is gated on
``coordinator._graph is not None`` to avoid noise on the PG-only stacks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from loguru import logger

from khora.dream.orchestrator import (
    _POSTGRES_ONLY_OP_KINDS,
    _warn_graph_divergence,
)
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GraphStub:
    """Marker object so ``getattr(coordinator, '_graph', None)`` returns truthy."""


class _PgCoordinator:
    """Coordinator stub with optional graph slot and a postgresql session."""

    def __init__(self, *, graph: Any | None = None) -> None:
        self.session = SimpleNamespace(
            bind=SimpleNamespace(dialect=SimpleNamespace(name="postgresql")),
        )
        self._graph = graph

    def transaction(self) -> Any:
        return _PgTxn(self)


class _PgTxn:
    def __init__(self, coordinator: _PgCoordinator) -> None:
        self._coordinator = coordinator

    async def __aenter__(self) -> Any:
        return SimpleNamespace(session=self._coordinator.session)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def _op(op_type: OpKind) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=op_type,
        decision="planned",
        rationale="test",
        outputs=({"merges": []},),
        started_at=datetime.now(UTC),
        duration_ms=1.0,
        namespace_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# Direct helper tests - cheaper than wiring the orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_kind_str", sorted(_POSTGRES_ONLY_OP_KINDS))
def test_warn_graph_divergence_fires_when_graph_attached(op_kind_str: str) -> None:
    """Every gated op kind triggers a WARNING when ``coordinator._graph`` is set."""
    coordinator = _PgCoordinator(graph=_GraphStub())
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        _warn_graph_divergence(coordinator, op_kind_str)
    finally:
        logger.remove(sink_id)

    joined = "\n".join(messages)
    assert op_kind_str in joined
    assert "graph store will not reflect" in joined
    assert "relational store only" in joined


def test_warn_graph_divergence_silent_when_no_graph() -> None:
    """No graph backend means no warning - this is the PG-only happy path."""
    coordinator = _PgCoordinator(graph=None)
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        _warn_graph_divergence(coordinator, OpKind.VECTORCYPHER_DEDUPE_ENTITIES)
    finally:
        logger.remove(sink_id)

    assert messages == []


def test_warn_graph_divergence_ignores_non_gated_ops() -> None:
    """Audit / non-gated ops don't trigger the divergence warning even with a graph attached."""
    coordinator = _PgCoordinator(graph=_GraphStub())
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        _warn_graph_divergence(coordinator, OpKind.CHRONICLE_TOMBSTONE_AUDIT)
        _warn_graph_divergence(coordinator, OpKind.VECTORCYPHER_NORMALIZE_SCHEMA)
    finally:
        logger.remove(sink_id)

    assert messages == []


# ---------------------------------------------------------------------------
# Orchestrator-level: warning fires once per gated apply invocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_one_op_emits_divergence_warning_with_graph() -> None:
    """Running ``_apply_one_op`` on a PG+graph coordinator emits the warning."""
    from khora.dream.config import DreamConfig
    from khora.dream.orchestrator import DreamOrchestrator

    class _FakeKB:
        def __init__(self, coordinator: _PgCoordinator) -> None:
            self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
            self._engine_name = "stub"
            self.storage = coordinator

    captured_session: list[Any] = []

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del coordinator
        captured_session.append(session)
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={"entity_id": str(uuid4())},
            applied_at=datetime.now(UTC),
        )

    coordinator = _PgCoordinator(graph=_GraphStub())
    kb = _FakeKB(coordinator)
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    op = _op(OpKind.VECTORCYPHER_PRUNE_EDGES)

    # Patch out checkpoint persistence (it requires a real PG session
    # response shape we're not faking here).
    async def _noop_checkpoint(_session: Any, _run_id: UUID, _seq: int) -> None:
        return None

    orch._record_committed_in_session = _noop_checkpoint  # type: ignore[assignment]

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        undo = await orch._apply_one_op(run_id=uuid4(), seq=0, op=op, handler=_handler)
    finally:
        logger.remove(sink_id)

    assert isinstance(undo, UndoRecord)
    joined = "\n".join(messages)
    assert "vectorcypher_prune_edges" in joined
    assert "graph store will not reflect" in joined
    # The handler still ran (the warning does not abort the apply).
    assert len(captured_session) == 1


@pytest.mark.asyncio
async def test_apply_one_op_silent_when_no_graph_configured() -> None:
    """No graph backend means no warning even when the gated op kind runs."""
    from khora.dream.config import DreamConfig
    from khora.dream.orchestrator import DreamOrchestrator

    class _FakeKB:
        def __init__(self, coordinator: _PgCoordinator) -> None:
            self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
            self._engine_name = "stub"
            self.storage = coordinator

    async def _handler(op: DreamOp, **_kwargs: Any) -> UndoRecord:
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={},
            applied_at=datetime.now(UTC),
        )

    coordinator = _PgCoordinator(graph=None)
    kb = _FakeKB(coordinator)
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    async def _noop_checkpoint(_session: Any, _run_id: UUID, _seq: int) -> None:
        return None

    orch._record_committed_in_session = _noop_checkpoint  # type: ignore[assignment]

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await orch._apply_one_op(
            run_id=uuid4(),
            seq=0,
            op=_op(OpKind.VECTORCYPHER_DEDUPE_ENTITIES),
            handler=_handler,
        )
    finally:
        logger.remove(sink_id)

    # The divergence warning is the only one we expect from this path; any
    # other WARNING is incidental but the divergence string MUST NOT appear.
    joined = "\n".join(messages)
    assert "graph store will not reflect" not in joined
