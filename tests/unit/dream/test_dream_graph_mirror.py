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

    def __init__(self, *, fail_invalidate: bool = False, fail_retire: bool = False, fail_rewrite: bool = False) -> None:
        self.retired: list[dict[str, Any]] = []
        self.invalidated: list[dict[str, Any]] = []
        self.rewritten: list[dict[str, Any]] = []
        self._fail_invalidate = fail_invalidate
        self._fail_retire = fail_retire
        self._fail_rewrite = fail_rewrite

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

    async def rewrite_relationship_endpoints_batch(
        self, rewrites: list[dict[str, Any]], *, namespace_id: UUID, rewritten_at: datetime
    ) -> int:
        if self._fail_rewrite:
            raise RuntimeError("neo4j rewrite boom")
        self.rewritten.append({"rewrites": list(rewrites), "namespace_id": namespace_id, "rewritten_at": rewritten_at})
        return len(rewrites)


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
    canonical = uuid4()
    absorbed = uuid4()
    self_loop = uuid4()
    rewritten = uuid4()
    other = uuid4()
    undo = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_dedupe_entities",
        before={
            "merges": [
                {
                    "canonical_id": str(canonical),
                    "absorbed_id": str(absorbed),
                    "self_loops_invalidated": [str(self_loop)],
                    # Incident edge: absorbed -> other re-points to canonical -> other (#1273).
                    "previous_relationships": [
                        {
                            "id": str(rewritten),
                            "source_entity_id": str(absorbed),
                            "target_entity_id": str(other),
                            "relationship_type": "RELATES_TO",
                        }
                    ],
                    "applied": True,
                }
            ]
        },
        applied_at=datetime.now(UTC),
    )
    targets = extract_mirror_targets("vectorcypher_dedupe_entities", undo)
    assert targets["retire_entity_ids"] == [absorbed]
    assert targets["invalidate_relationship_ids"] == [self_loop]
    # The incident edge is re-pointed to the canonical (#1273), not invalidated.
    assert rewritten not in targets["invalidate_relationship_ids"]
    assert targets["rewrite_relationships"] == [
        {
            "relationship_id": str(rewritten),
            "source_entity_id": str(canonical),
            "target_entity_id": str(other),
            "relationship_type": "RELATES_TO",
        }
    ]


def test_extract_targets_dedupe_repoints_incident_edges() -> None:
    """Both directions of an incident edge re-point to the canonical (#1273)."""
    canonical = uuid4()
    absorbed = uuid4()
    neighbor = uuid4()
    out_edge = uuid4()  # absorbed -> neighbor
    in_edge = uuid4()  # neighbor -> absorbed
    undo = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_dedupe_entities",
        before={
            "merges": [
                {
                    "canonical_id": str(canonical),
                    "absorbed_id": str(absorbed),
                    "self_loops_invalidated": [],
                    "previous_relationships": [
                        {
                            "id": str(out_edge),
                            "source_entity_id": str(absorbed),
                            "target_entity_id": str(neighbor),
                            "relationship_type": "KNOWS",
                        },
                        {
                            "id": str(in_edge),
                            "source_entity_id": str(neighbor),
                            "target_entity_id": str(absorbed),
                            "relationship_type": "KNOWS",
                        },
                    ],
                    "applied": True,
                }
            ]
        },
        applied_at=datetime.now(UTC),
    )
    targets = extract_mirror_targets("vectorcypher_dedupe_entities", undo)
    rewrites = {rw["relationship_id"]: rw for rw in targets["rewrite_relationships"]}
    assert rewrites[str(out_edge)]["source_entity_id"] == str(canonical)
    assert rewrites[str(out_edge)]["target_entity_id"] == str(neighbor)
    assert rewrites[str(in_edge)]["source_entity_id"] == str(neighbor)
    assert rewrites[str(in_edge)]["target_entity_id"] == str(canonical)


def test_extract_targets_dedupe_self_loop_excluded_from_rewrite() -> None:
    """An edge listed in self_loops_invalidated is never also a rewrite target."""
    canonical = uuid4()
    absorbed = uuid4()
    loop = uuid4()  # canonical -> absorbed, becomes canonical -> canonical
    undo = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_dedupe_entities",
        before={
            "merges": [
                {
                    "canonical_id": str(canonical),
                    "absorbed_id": str(absorbed),
                    "self_loops_invalidated": [str(loop)],
                    "previous_relationships": [
                        {
                            "id": str(loop),
                            "source_entity_id": str(canonical),
                            "target_entity_id": str(absorbed),
                            "relationship_type": "RELATES_TO",
                        }
                    ],
                    "applied": True,
                }
            ]
        },
        applied_at=datetime.now(UTC),
    )
    targets = extract_mirror_targets("vectorcypher_dedupe_entities", undo)
    assert targets["invalidate_relationship_ids"] == [loop]
    assert targets["rewrite_relationships"] == []


def test_extract_targets_dedupe_transitive_collapses_to_one_canonical() -> None:
    """A->B and B's edges fold to a single canonical; no edge points at an
    absorbed intermediate (#806 id-remap, global map across merges)."""
    canonical = uuid4()
    b = uuid4()  # absorbed
    c = uuid4()  # absorbed in a second merge entry of the same op
    neighbor = uuid4()
    edge_to_b = uuid4()  # neighbor -> b  (b absorbed)
    edge_b_to_c = uuid4()  # b -> c  (both absorbed -> becomes canonical self-loop)
    undo = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_dedupe_entities",
        before={
            "merges": [
                {
                    "canonical_id": str(canonical),
                    "absorbed_id": str(b),
                    "self_loops_invalidated": [],
                    "previous_relationships": [
                        {
                            "id": str(edge_to_b),
                            "source_entity_id": str(neighbor),
                            "target_entity_id": str(b),
                            "relationship_type": "KNOWS",
                        },
                        {
                            "id": str(edge_b_to_c),
                            "source_entity_id": str(b),
                            "target_entity_id": str(c),
                            "relationship_type": "KNOWS",
                        },
                    ],
                    "applied": True,
                },
                {
                    "canonical_id": str(canonical),
                    "absorbed_id": str(c),
                    "self_loops_invalidated": [],
                    "previous_relationships": [],
                    "applied": True,
                },
            ]
        },
        applied_at=datetime.now(UTC),
    )
    targets = extract_mirror_targets("vectorcypher_dedupe_entities", undo)
    assert set(targets["retire_entity_ids"]) == {b, c}
    rewrites = {rw["relationship_id"]: rw for rw in targets["rewrite_relationships"]}
    # neighbor -> b re-points to neighbor -> canonical (no edge at the absorbed b).
    assert rewrites[str(edge_to_b)]["target_entity_id"] == str(canonical)
    # b -> c: both endpoints map to canonical -> a cross-component self-loop the
    # PG handler did not list. Mirrored as an invalidate, never a rewrite.
    assert str(edge_b_to_c) not in rewrites
    assert edge_b_to_c in targets["invalidate_relationship_ids"]


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
    canonical = uuid4()
    absorbed = uuid4()
    self_loop = uuid4()
    incident = uuid4()
    neighbor = uuid4()
    op = _dedupe_op()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type="vectorcypher_dedupe_entities",
        before={
            "merges": [
                {
                    "canonical_id": str(canonical),
                    "absorbed_id": str(absorbed),
                    "self_loops_invalidated": [str(self_loop)],
                    "previous_relationships": [
                        {
                            "id": str(incident),
                            "source_entity_id": str(absorbed),
                            "target_entity_id": str(neighbor),
                            "relationship_type": "RELATES_TO",
                        }
                    ],
                    "applied": True,
                }
            ]
        },
        applied_at=datetime.now(UTC),
    )
    payload = mirror_payload(op, undo)
    # JSON-serialisable (the reconciler persists this verbatim).
    import json

    json.dumps(payload)
    recovered = targets_from_payload(payload)
    assert recovered["retire_entity_ids"] == [absorbed]
    assert recovered["invalidate_relationship_ids"] == [self_loop]
    assert recovered["rewrite_relationships"] == [
        {
            "relationship_id": str(incident),
            "source_entity_id": str(canonical),
            "target_entity_id": str(neighbor),
            "relationship_type": "RELATES_TO",
        }
    ]


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
async def test_mirror_dedupe_calls_rewrite_endpoints() -> None:
    """The dedupe mirror re-points incident edges via the rewrite verb (#1273)."""
    graph = _FakeGraph()
    ns = uuid4()
    orch = _orch(graph)
    op = _dedupe_op()
    canonical = uuid4()
    absorbed = uuid4()
    neighbor = uuid4()
    incident = uuid4()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={
            "merges": [
                {
                    "canonical_id": str(canonical),
                    "absorbed_id": str(absorbed),
                    "self_loops_invalidated": [],
                    "previous_relationships": [
                        {
                            "id": str(incident),
                            "source_entity_id": str(absorbed),
                            "target_entity_id": str(neighbor),
                            "relationship_type": "RELATES_TO",
                        }
                    ],
                    "applied": True,
                }
            ]
        },
        applied_at=datetime.now(UTC),
    )
    degradation = await orch._mirror_dream_op(uuid4(), 0, ns, op, undo)
    assert degradation is None
    assert graph.retired[0]["ids"] == [absorbed]
    assert len(graph.rewritten) == 1
    assert graph.rewritten[0]["namespace_id"] == ns
    assert graph.rewritten[0]["rewrites"] == [
        {
            "relationship_id": str(incident),
            "source_entity_id": str(canonical),
            "target_entity_id": str(neighbor),
            "relationship_type": "RELATES_TO",
        }
    ]


@pytest.mark.asyncio
async def test_mirror_rewrite_failure_queues_pending_and_degrades() -> None:
    """A rewrite-verb failure after the PG commit queues the op + degrades."""
    graph = _FakeGraph(fail_rewrite=True)
    store = _RunStore()
    orch = _orch(graph, run_store=store)
    op = _dedupe_op()
    canonical = uuid4()
    absorbed = uuid4()
    neighbor = uuid4()
    incident = uuid4()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={
            "merges": [
                {
                    "canonical_id": str(canonical),
                    "absorbed_id": str(absorbed),
                    "self_loops_invalidated": [],
                    "previous_relationships": [
                        {
                            "id": str(incident),
                            "source_entity_id": str(absorbed),
                            "target_entity_id": str(neighbor),
                            "relationship_type": "RELATES_TO",
                        }
                    ],
                    "applied": True,
                }
            ]
        },
        applied_at=datetime.now(UTC),
    )
    degradation = await orch._mirror_dream_op(uuid4(), 7, uuid4(), op, undo)
    assert degradation is not None
    assert degradation["reason"] == "graph_mirror_failed_after_pg_commit"
    # The rewrite payload survives into the pending slot for the reconciler.
    assert 7 in store.pending
    rewrites = store.pending[7].payload["rewrite_relationships"]
    assert rewrites[0]["relationship_id"] == str(incident)
    assert rewrites[0]["source_entity_id"] == str(canonical)


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
