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
    """In-memory graph_mirror_pending store keyed by op_seq."""

    def __init__(self) -> None:
        self.pending: dict[int, GraphMirrorPending] = {}

    async def mark_graph_mirror_pending(self, run_id: UUID, entry: GraphMirrorPending) -> None:
        self.pending[entry.op_seq] = entry

    async def get_graph_mirror_pending(self, run_id: UUID) -> list[GraphMirrorPending]:
        return list(self.pending.values())

    async def clear_graph_mirror_pending(self, run_id: UUID, op_seq: int) -> None:
        self.pending.pop(op_seq, None)


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
    """A backend that advertises no mirror support records a skip, not a silent divergence."""
    graph = _NoMirrorGraph()
    orch = _orch(graph)
    op = _prune_op()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"relationships": [{"relationship_id": str(uuid4())}]},
        applied_at=datetime.now(UTC),
    )
    # No verb to call - returns None (skip recorded via log), no exception.
    assert await orch._mirror_dream_op(uuid4(), 0, uuid4(), op, undo) is None


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
    # The op is queued for the reconciler under its op_seq.
    assert 3 in store.pending
    assert store.pending[3].op_type == "vectorcypher_prune_edges"
    assert str(rel_id) in store.pending[3].payload["invalidate_relationship_ids"]


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
    store.pending[3] = GraphMirrorPending(
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
    assert 3 not in store.pending


@pytest.mark.asyncio
async def test_reconciler_keeps_pending_on_repeated_failure() -> None:
    graph = _FakeGraph(fail_invalidate=True)
    store = _RunStore()
    run_id = uuid4()
    store.pending[3] = GraphMirrorPending(
        op_seq=3,
        op_id=uuid4(),
        op_type="vectorcypher_prune_edges",
        payload={"retire_entity_ids": [], "invalidate_relationship_ids": [str(uuid4())], "applied_at": None},
    )
    orch = _orch(graph, run_store=store)
    degradations = await orch._drain_graph_mirror_pending(run_id, uuid4())
    assert len(degradations) == 1
    assert degradations[0]["reason"] == "graph_mirror_reconcile_failed"
    # Still queued for the next attempt.
    assert 3 in store.pending
