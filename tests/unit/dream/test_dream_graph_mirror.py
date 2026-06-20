"""Neo4j dream tombstone-mirror: post-commit translation + reconciler (#1272).

Replaces the obsolete ``_warn_graph_divergence`` accounting (the divergence is
no longer warned-about-and-deferred; it is mirrored). These are mock-driven
verb-level assertions: a fake graph backend records the verb calls so we can
assert the committed PG soft-deletes are translated onto the graph
``valid_until`` via the #1271 verbs, that an unsupported op kind records a skip
(not a silent divergence), and that a mirror failure after the PG commit queues
the op for the reconciler and surfaces a degradation.

The live pg+Neo4j cross-store invariant lives in the integration suite
(``tests/integration/dream/test_neo4j_dream_mirror_integration.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.config import DreamConfig
from khora.dream.graph_mirror import (
    extract_mirror_targets,
    mirror_payload,
    targets_from_payload,
)
from khora.dream.orchestrator import DreamOrchestrator
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.dream.runstore import GraphMirrorPending

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Records the #1271 verb calls; advertises the mirrorable op kinds."""

    def __init__(self, *, fail_invalidate: bool = False, fail_retire: bool = False) -> None:
        self.retired: list[dict[str, Any]] = []
        self.invalidated: list[dict[str, Any]] = []
        self._fail_invalidate = fail_invalidate
        self._fail_retire = fail_retire

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        return frozenset(
            {
                OpKind.VECTORCYPHER_PRUNE_EDGES,
                OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
                OpKind.VECTORCYPHER_NORMALIZE_SCHEMA,
            }
        )

    async def soft_retire_entities_batch(
        self, entity_ids: list[UUID], *, namespace_id: UUID, retired_at: datetime, reason: str = "dream_consolidated"
    ) -> int:
        if self._fail_retire:
            raise RuntimeError("neo4j retire boom")
        self.retired.append({"ids": list(entity_ids), "namespace_id": namespace_id, "retired_at": retired_at})
        return len(entity_ids)

    async def soft_invalidate_relationships_batch(
        self, relationship_ids: list[UUID], *, namespace_id: UUID, invalidated_at: datetime
    ) -> int:
        if self._fail_invalidate:
            raise RuntimeError("neo4j invalidate boom")
        self.invalidated.append(
            {"ids": list(relationship_ids), "namespace_id": namespace_id, "invalidated_at": invalidated_at}
        )
        return len(relationship_ids)


class _NoMirrorGraph:
    """A graph backend that advertises no mirror support (GraphBackendBase default)."""

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        return frozenset()


class _Coordinator:
    def __init__(self, graph: Any | None) -> None:
        self._graph = graph


class _RunStore:
    """In-memory graph_mirror_pending store keyed by (run_id, op_seq)."""

    def __init__(self) -> None:
        # (run_id, op_seq) -> entry, plus the namespace each run belongs to.
        self.pending: dict[tuple[UUID, int], GraphMirrorPending] = {}
        self.namespaces: dict[UUID, UUID] = {}

    def bind_run(self, run_id: UUID, namespace_id: UUID) -> None:
        self.namespaces[run_id] = namespace_id

    async def mark_graph_mirror_pending(
        self, run_id: UUID, entry: GraphMirrorPending, *, session: Any | None = None
    ) -> None:
        del session  # the fake has no transaction to enroll
        self.pending[(run_id, entry.op_seq)] = entry

    async def get_graph_mirror_pending(self, run_id: UUID) -> list[GraphMirrorPending]:
        return [entry for (rid, _seq), entry in self.pending.items() if rid == run_id]

    async def get_open_graph_mirror_pending(self, namespace_id: UUID) -> list[tuple[UUID, GraphMirrorPending]]:
        return [(rid, entry) for (rid, _seq), entry in self.pending.items() if self.namespaces.get(rid) == namespace_id]

    async def clear_graph_mirror_pending(self, run_id: UUID, op_seq: int) -> None:
        self.pending.pop((run_id, op_seq), None)


class _FakeKB:
    def __init__(self, coordinator: _Coordinator) -> None:
        self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
        self._engine_name = "stub"
        self.storage = coordinator


def _orch(graph: Any | None, *, run_store: _RunStore | None = None) -> DreamOrchestrator:
    orch = DreamOrchestrator(_FakeKB(_Coordinator(graph)), DreamConfig(enabled=True), sinks=[])
    if run_store is not None:
        orch._run_store_cache = run_store  # type: ignore[attr-defined]
        orch._run_store_resolved = True  # type: ignore[attr-defined]
    return orch


def _prune_op() -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_PRUNE_EDGES,
        namespace_id=uuid4(),
    )


def _dedupe_op() -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        namespace_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# Pure translation
# ---------------------------------------------------------------------------


def test_extract_targets_prune_edges() -> None:
    rel_id = uuid4()
    undo = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_prune_edges",
        before={"relationships": [{"relationship_id": str(rel_id), "previous_valid_to": None}]},
        applied_at=datetime.now(UTC),
    )
    targets = extract_mirror_targets("vectorcypher_prune_edges", undo)
    assert targets["invalidate_relationship_ids"] == [rel_id]
    assert targets["retire_entity_ids"] == []


def test_extract_targets_prune_noop() -> None:
    undo = UndoRecord(op_id=uuid4(), op_type="vectorcypher_prune_edges", before={"noop": True}, applied_at=None)
    targets = extract_mirror_targets("vectorcypher_prune_edges", undo)
    assert targets["invalidate_relationship_ids"] == []
    assert targets["retire_entity_ids"] == []


def test_extract_targets_dedupe_entity_and_self_loop() -> None:
    absorbed = uuid4()
    self_loop = uuid4()
    rewritten = uuid4()
    undo = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_dedupe_entities",
        before={
            "merges": [
                {
                    "canonical_id": str(uuid4()),
                    "absorbed_id": str(absorbed),
                    "self_loops_invalidated": [str(self_loop)],
                    # An endpoint rewrite is #1273 - it must NOT be mirrored here.
                    "previous_relationships": [{"id": str(rewritten)}],
                    "applied": True,
                }
            ]
        },
        applied_at=datetime.now(UTC),
    )
    targets = extract_mirror_targets("vectorcypher_dedupe_entities", undo)
    assert targets["retire_entity_ids"] == [absorbed]
    assert targets["invalidate_relationship_ids"] == [self_loop]
    # The rewritten edge id is not a mirror target (endpoint rewrite = #1273).
    assert rewritten not in targets["invalidate_relationship_ids"]


def test_extract_targets_dedupe_skips_unapplied_merge() -> None:
    undo = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_dedupe_entities",
        before={"merges": [{"absorbed_id": str(uuid4()), "self_loops_invalidated": [], "applied": False}]},
        applied_at=datetime.now(UTC),
    )
    targets = extract_mirror_targets("vectorcypher_dedupe_entities", undo)
    assert targets["retire_entity_ids"] == []


def test_payload_round_trips() -> None:
    absorbed = uuid4()
    self_loop = uuid4()
    op = _dedupe_op()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type="vectorcypher_dedupe_entities",
        before={
            "merges": [{"absorbed_id": str(absorbed), "self_loops_invalidated": [str(self_loop)], "applied": True}]
        },
        applied_at=datetime.now(UTC),
    )
    payload = mirror_payload(op, undo)
    recovered = targets_from_payload(payload)
    assert recovered["retire_entity_ids"] == [absorbed]
    assert recovered["invalidate_relationship_ids"] == [self_loop]


# ---------------------------------------------------------------------------
# Orchestrator-level mirror
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mirror_prune_calls_invalidate_verb() -> None:
    graph = _FakeGraph()
    orch = _orch(graph)
    op = _prune_op()
    rel_id = uuid4()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"relationships": [{"relationship_id": str(rel_id)}]},
        applied_at=datetime.now(UTC),
    )
    ns = uuid4()
    degradation = await orch._mirror_dream_op(uuid4(), 0, ns, op, undo)
    assert degradation is None
    assert len(graph.invalidated) == 1
    assert graph.invalidated[0]["ids"] == [rel_id]
    assert graph.invalidated[0]["namespace_id"] == ns


@pytest.mark.asyncio
async def test_mirror_dedupe_calls_retire_and_invalidate() -> None:
    graph = _FakeGraph()
    orch = _orch(graph)
    op = _dedupe_op()
    absorbed = uuid4()
    self_loop = uuid4()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={
            "merges": [{"absorbed_id": str(absorbed), "self_loops_invalidated": [str(self_loop)], "applied": True}]
        },
        applied_at=datetime.now(UTC),
    )
    degradation = await orch._mirror_dream_op(uuid4(), 0, uuid4(), op, undo)
    assert degradation is None
    assert graph.retired[0]["ids"] == [absorbed]
    assert graph.invalidated[0]["ids"] == [self_loop]


@pytest.mark.asyncio
async def test_mirror_noop_when_no_graph() -> None:
    orch = _orch(None)
    op = _prune_op()
    undo = UndoRecord(op_id=op.op_id, op_type=str(op.op_type), before={"noop": True}, applied_at=None)
    assert await orch._mirror_dream_op(uuid4(), 0, uuid4(), op, undo) is None


@pytest.mark.asyncio
async def test_mirror_skip_when_backend_unsupported() -> None:
    """A backend that advertises no mirror support surfaces a structured skip (#1292)."""
    graph = _NoMirrorGraph()
    orch = _orch(graph)
    op = _prune_op()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"relationships": [{"relationship_id": str(uuid4())}]},
        applied_at=datetime.now(UTC),
    )
    # The skip is returned (not silently swallowed) so the apply loop threads it
    # onto DreamResult.metadata - no exception, but not None either.
    record = await orch._mirror_dream_op(uuid4(), 0, uuid4(), op, undo)
    assert record is not None
    assert record["reason"] == "graph_mirror_unsupported_op_kind"


@pytest.mark.asyncio
async def test_mirror_failure_queues_pending_and_degrades() -> None:
    graph = _FakeGraph(fail_invalidate=True)
    store = _RunStore()
    orch = _orch(graph, run_store=store)
    op = _prune_op()
    rel_id = uuid4()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"relationships": [{"relationship_id": str(rel_id)}]},
        applied_at=datetime.now(UTC),
    )
    run_id = uuid4()
    degradation = await orch._mirror_dream_op(run_id, 3, uuid4(), op, undo)
    assert degradation is not None
    assert degradation["reason"] == "graph_mirror_failed_after_pg_commit"
    assert degradation["exception"] == "RuntimeError"
    # The op is queued for the reconciler under its (run_id, op_seq) key.
    assert (run_id, 3) in store.pending
    assert store.pending[(run_id, 3)].op_type == "vectorcypher_prune_edges"
    assert str(rel_id) in store.pending[(run_id, 3)].payload["invalidate_relationship_ids"]


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconciler_re_mirrors_and_clears() -> None:
    """The drain re-mirrors a queued op (idempotent) and clears it on success."""
    graph = _FakeGraph()
    store = _RunStore()
    rel_id = uuid4()
    run_id = uuid4()
    ns = uuid4()
    store.bind_run(run_id, ns)
    store.pending[(run_id, 3)] = GraphMirrorPending(
        op_seq=3,
        op_id=uuid4(),
        op_type="vectorcypher_prune_edges",
        payload={"retire_entity_ids": [], "invalidate_relationship_ids": [str(rel_id)], "applied_at": None},
    )
    orch = _orch(graph, run_store=store)
    degradations = await orch._drain_graph_mirror_pending(run_id, ns)
    assert degradations == []
    assert graph.invalidated[0]["ids"] == [rel_id]
    # Cleared on success.
    assert (run_id, 3) not in store.pending


@pytest.mark.asyncio
async def test_reconciler_keeps_pending_on_repeated_failure() -> None:
    graph = _FakeGraph(fail_invalidate=True)
    store = _RunStore()
    run_id = uuid4()
    ns = uuid4()
    store.bind_run(run_id, ns)
    store.pending[(run_id, 3)] = GraphMirrorPending(
        op_seq=3,
        op_id=uuid4(),
        op_type="vectorcypher_prune_edges",
        payload={"retire_entity_ids": [], "invalidate_relationship_ids": [str(uuid4())], "applied_at": None},
    )
    orch = _orch(graph, run_store=store)
    degradations = await orch._drain_graph_mirror_pending(run_id, ns)
    assert len(degradations) == 1
    assert degradations[0]["reason"] == "graph_mirror_reconcile_failed"
    # Still queued for the next attempt.
    assert (run_id, 3) in store.pending


# ---------------------------------------------------------------------------
# #1292 gap 1: crash-durable pending + namespace-scoped drain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_mark_persists_pending_before_mirror_attempt() -> None:
    """The pending row is written INSIDE the apply tx, before any mirror runs.

    Closes the hard-crash window: a process death between the PG commit and the
    mirror leaves a durable pending row the reconciler can drain.
    """
    graph = _FakeGraph()
    store = _RunStore()
    orch = _orch(graph, run_store=store)
    op = _prune_op()
    rel_id = uuid4()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"relationships": [{"relationship_id": str(rel_id)}]},
        applied_at=datetime.now(UTC),
    )
    run_id = uuid4()
    # The apply loop calls this inside the checkpoint transaction (session=None
    # in the fake). It must persist the pending entry pre-emptively.
    await orch._pre_mark_graph_mirror_pending(None, run_id, 5, op, undo)
    assert (run_id, 5) in store.pending
    assert store.pending[(run_id, 5)].op_type == "vectorcypher_prune_edges"
    assert str(rel_id) in store.pending[(run_id, 5)].payload["invalidate_relationship_ids"]


@pytest.mark.asyncio
async def test_pre_mark_skips_when_no_targets() -> None:
    """A no-op apply (no mirror targets) leaves nothing queued."""
    graph = _FakeGraph()
    store = _RunStore()
    orch = _orch(graph, run_store=store)
    op = _prune_op()
    undo = UndoRecord(op_id=op.op_id, op_type=str(op.op_type), before={"noop": True}, applied_at=None)
    run_id = uuid4()
    await orch._pre_mark_graph_mirror_pending(None, run_id, 0, op, undo)
    assert store.pending == {}


@pytest.mark.asyncio
async def test_mirror_clears_pending_on_success() -> None:
    """A successful mirror clears the pre-emptively-marked pending row."""
    graph = _FakeGraph()
    store = _RunStore()
    orch = _orch(graph, run_store=store)
    op = _prune_op()
    rel_id = uuid4()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"relationships": [{"relationship_id": str(rel_id)}]},
        applied_at=datetime.now(UTC),
    )
    run_id = uuid4()
    ns = uuid4()
    await orch._pre_mark_graph_mirror_pending(None, run_id, 1, op, undo)
    assert (run_id, 1) in store.pending
    degradation = await orch._mirror_dream_op(run_id, 1, ns, op, undo)
    assert degradation is None
    # Mirror succeeded -> pending cleared.
    assert (run_id, 1) not in store.pending


@pytest.mark.asyncio
async def test_drain_heals_prior_run_in_same_namespace() -> None:
    """#1292 gap 1: a NEW run (different run_id) drains a prior run's pending op.

    On origin/main the drain reads only the current run_id, so a crash-left
    pending op from run A is never retried by run B. This asserts the
    namespace-scoped drain heals it.
    """
    graph = _FakeGraph()
    store = _RunStore()
    ns = uuid4()
    prior_run = uuid4()
    new_run = uuid4()
    store.bind_run(prior_run, ns)
    store.bind_run(new_run, ns)
    rel_id = uuid4()
    # A pending op left by the PRIOR run (e.g. crash before mirror).
    store.pending[(prior_run, 0)] = GraphMirrorPending(
        op_seq=0,
        op_id=uuid4(),
        op_type="vectorcypher_prune_edges",
        payload={"retire_entity_ids": [], "invalidate_relationship_ids": [str(rel_id)], "applied_at": None},
    )
    orch = _orch(graph, run_store=store)
    # The NEW run's drain heals the prior run's op.
    degradations = await orch._drain_graph_mirror_pending(new_run, ns)
    assert degradations == []
    assert graph.invalidated[0]["ids"] == [rel_id]
    # The prior run's pending entry is cleared under ITS run_id.
    assert (prior_run, 0) not in store.pending


# ---------------------------------------------------------------------------
# #1292 gap 2: unsupported-mirror skip surfaces as a structured record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mirror_skip_returns_structured_skip_reason() -> None:
    """A backend that cannot mirror an op kind surfaces a SkipReason, not just a log."""
    graph = _NoMirrorGraph()
    orch = _orch(graph)
    op = _prune_op()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"relationships": [{"relationship_id": str(uuid4())}]},
        applied_at=datetime.now(UTC),
    )
    record = await orch._mirror_dream_op(uuid4(), 0, uuid4(), op, undo)
    assert record is not None, "unsupported-mirror skip must surface on the result (ADR-001)"
    assert record["reason"] == "graph_mirror_unsupported_op_kind"
    assert record["component"] == "dream.graph_mirror"
    assert "vectorcypher_prune_edges" in record["detail"]


# ---------------------------------------------------------------------------
# #1292 gap 3: namespace-resolve failure degrades (no silent fallback)
# ---------------------------------------------------------------------------


class _ResolverErrorCoordinator(_Coordinator):
    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        raise RuntimeError("resolver boom")


def _orch_with_coordinator(coordinator: _Coordinator, *, run_store: _RunStore | None = None) -> DreamOrchestrator:
    kb = _FakeKB(coordinator)
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
    if run_store is not None:
        orch._run_store_cache = run_store  # type: ignore[attr-defined]
        orch._run_store_resolved = True  # type: ignore[attr-defined]
    return orch


@pytest.mark.asyncio
async def test_resolver_failure_degrades_and_queues() -> None:
    """A resolver error in the mirror path degrades + queues; it does not silently fall back."""
    graph = _FakeGraph()
    store = _RunStore()
    orch = _orch_with_coordinator(_ResolverErrorCoordinator(graph), run_store=store)
    op = _prune_op()
    rel_id = uuid4()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"relationships": [{"relationship_id": str(rel_id)}]},
        applied_at=datetime.now(UTC),
    )
    run_id = uuid4()
    degradation = await orch._mirror_dream_op(run_id, 0, uuid4(), op, undo)
    assert degradation is not None
    assert degradation["reason"] == "graph_mirror_failed_after_pg_commit"
    # No mirror verb was called (the resolve raised before the graph write).
    assert graph.invalidated == []
    # Queued for the reconciler.
    assert (run_id, 0) in store.pending


@pytest.mark.asyncio
async def test_resolve_namespace_raises_on_resolver_error() -> None:
    """_resolve_namespace_for_mirror propagates resolver errors (no silent fallback)."""
    orch = _orch_with_coordinator(_ResolverErrorCoordinator(_FakeGraph()))
    with pytest.raises(RuntimeError):
        await orch._resolve_namespace_for_mirror(uuid4())


@pytest.mark.asyncio
async def test_resolve_namespace_idempotent_without_resolver() -> None:
    """No resolver method -> the input is returned unchanged (idempotent)."""
    orch = _orch(_FakeGraph())  # plain _Coordinator has no resolve_namespace
    ns = uuid4()
    assert await orch._resolve_namespace_for_mirror(ns) == ns


# ---------------------------------------------------------------------------
# #1292 gap 4: pending-read swallow records a degradation
# ---------------------------------------------------------------------------


class _PendingReadErrorStore(_RunStore):
    async def get_open_graph_mirror_pending(self, namespace_id: UUID) -> list[tuple[UUID, GraphMirrorPending]]:
        raise RuntimeError("pending read boom")


@pytest.mark.asyncio
async def test_drain_pending_read_error_degrades() -> None:
    """A failing pending read records a Degradation instead of silently returning []."""
    graph = _FakeGraph()
    store = _PendingReadErrorStore()
    orch = _orch(graph, run_store=store)
    degradations = await orch._drain_graph_mirror_pending(uuid4(), uuid4())
    assert len(degradations) == 1
    assert degradations[0]["reason"] == "graph_mirror_pending_read_failed"
    assert degradations[0]["component"] == "dream.graph_mirror.reconcile"
