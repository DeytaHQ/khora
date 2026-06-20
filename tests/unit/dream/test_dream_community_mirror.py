"""Dream community materialization mirror: translation + dispatch (#1276).

The GraphRAG payoff. The dream ``community_summary`` op persists LLM-grounded
summaries to PG; the post-commit mirror materializes them into the graph as
:Community nodes + [:HAS_MEMBER] edges so they are queryable at recall.

These are mock-driven verb-level assertions: a fake graph backend records the
materialize verb call so we can assert the committed PG community is translated
into the graph-materialization verb via the #1271 capability seam, that a
backend lacking the capability records a skip (not a silent divergence), that a
no-op community apply mirrors nothing, and that a mirror failure after the PG
commit queues the op for the reconciler.

The live pg+Neo4j cross-store assertion lives in the integration suite
(``tests/integration/dream/test_neo4j_dream_mirror_integration.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.core.models import CommunityNode
from khora.dream.config import DreamConfig
from khora.dream.graph_mirror import (
    communities_from_payload,
    extract_community_targets,
    mirror_payload,
)
from khora.dream.orchestrator import DreamOrchestrator
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.dream.runstore import GraphMirrorPending


class _FakeGraph:
    """Records the #1276 materialize verb call; advertises the community op kind."""

    def __init__(self, *, fail_materialize: bool = False) -> None:
        self.materialized: list[dict[str, Any]] = []
        self._fail_materialize = fail_materialize

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        return frozenset({OpKind.VECTORCYPHER_COMMUNITY_SUMMARY})

    async def materialize_communities_batch(
        self,
        communities: list[CommunityNode],
        *,
        namespace_id: UUID,
        materialized_at: datetime,
    ) -> int:
        if self._fail_materialize:
            raise RuntimeError("neo4j materialize boom")
        self.materialized.append(
            {"communities": list(communities), "namespace_id": namespace_id, "at": materialized_at}
        )
        return len(communities)


class _NoMirrorGraph:
    def supports_dream_mirror(self) -> frozenset[OpKind]:
        return frozenset()


class _Coordinator:
    def __init__(self, graph: Any | None) -> None:
        self._graph = graph


class _RunStore:
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


def _community_op() -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_COMMUNITY_SUMMARY,
        namespace_id=uuid4(),
    )


def _persisted_undo(op: DreamOp, community_id: UUID, member_ids: list[UUID]) -> UndoRecord:
    return UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={
            "community_id": str(community_id),
            "kept_claims": 2,
            "dropped_claims": 0,
            "summary_text": "Alice and Bob co-founded Acme.",
            "member_ids": [str(m) for m in member_ids],
            "summary_depth": 1,
        },
        applied_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Pure translation
# ---------------------------------------------------------------------------


def test_extract_community_targets_persisted() -> None:
    community_id = uuid4()
    m1, m2 = uuid4(), uuid4()
    op = _community_op()
    nodes = extract_community_targets(str(op.op_type), _persisted_undo(op, community_id, [m1, m2]))
    assert len(nodes) == 1
    assert nodes[0].id == community_id
    assert nodes[0].summary == "Alice and Bob co-founded Acme."
    assert nodes[0].member_ids == [m1, m2]
    assert nodes[0].summary_depth == 1


def test_extract_community_targets_noop() -> None:
    op = _community_op()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"noop": True, "reason": "already_live"},
        applied_at=None,
    )
    assert extract_community_targets(str(op.op_type), undo) == []


def test_extract_community_targets_wrong_op_kind() -> None:
    undo = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_prune_edges",
        before={"relationships": [{"relationship_id": str(uuid4())}]},
        applied_at=None,
    )
    assert extract_community_targets("vectorcypher_prune_edges", undo) == []


def test_community_payload_round_trips() -> None:
    community_id = uuid4()
    m1, m2 = uuid4(), uuid4()
    op = _community_op()
    payload = mirror_payload(op, _persisted_undo(op, community_id, [m1, m2]))
    recovered = communities_from_payload(payload)
    assert len(recovered) == 1
    assert recovered[0].id == community_id
    assert recovered[0].member_ids == [m1, m2]
    assert recovered[0].summary == "Alice and Bob co-founded Acme."


# ---------------------------------------------------------------------------
# Orchestrator-level mirror
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mirror_community_calls_materialize_verb() -> None:
    graph = _FakeGraph()
    orch = _orch(graph)
    op = _community_op()
    community_id = uuid4()
    m1, m2 = uuid4(), uuid4()
    ns = uuid4()
    degradation = await orch._mirror_dream_op(uuid4(), 0, ns, op, _persisted_undo(op, community_id, [m1, m2]))
    assert degradation is None
    assert len(graph.materialized) == 1
    nodes = graph.materialized[0]["communities"]
    assert nodes[0].id == community_id
    assert nodes[0].member_ids == [m1, m2]
    assert graph.materialized[0]["namespace_id"] == ns


@pytest.mark.asyncio
async def test_mirror_community_noop_mirrors_nothing() -> None:
    graph = _FakeGraph()
    orch = _orch(graph)
    op = _community_op()
    undo = UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"noop": True, "reason": "no_grounded_claims"},
        applied_at=None,
    )
    assert await orch._mirror_dream_op(uuid4(), 0, uuid4(), op, undo) is None
    assert graph.materialized == []


@pytest.mark.asyncio
async def test_mirror_community_skip_when_backend_unsupported() -> None:
    """A backend that advertises no community-mirror support records a skip, not a divergence."""
    graph = _NoMirrorGraph()
    orch = _orch(graph)
    op = _community_op()
    undo = _persisted_undo(op, uuid4(), [uuid4()])
    assert await orch._mirror_dream_op(uuid4(), 0, uuid4(), op, undo) is None


@pytest.mark.asyncio
async def test_mirror_community_failure_queues_pending_and_degrades() -> None:
    graph = _FakeGraph(fail_materialize=True)
    store = _RunStore()
    orch = _orch(graph, run_store=store)
    op = _community_op()
    community_id = uuid4()
    undo = _persisted_undo(op, community_id, [uuid4()])
    run_id = uuid4()
    degradation = await orch._mirror_dream_op(run_id, 7, uuid4(), op, undo)
    assert degradation is not None
    assert degradation["reason"] == "graph_mirror_failed_after_pg_commit"
    assert 7 in store.pending
    assert store.pending[7].op_type == "vectorcypher_community_summary"
    assert store.pending[7].payload["communities"][0]["id"] == str(community_id)


@pytest.mark.asyncio
async def test_reconciler_re_materializes_community_and_clears() -> None:
    graph = _FakeGraph()
    store = _RunStore()
    run_id = uuid4()
    ns = uuid4()
    community_id = uuid4()
    store.pending[7] = GraphMirrorPending(
        op_seq=7,
        op_id=uuid4(),
        op_type="vectorcypher_community_summary",
        payload={
            "retire_entity_ids": [],
            "invalidate_relationship_ids": [],
            "communities": [
                {"id": str(community_id), "summary": "s", "member_ids": [str(uuid4())], "summary_depth": 1}
            ],
            "applied_at": None,
        },
    )
    orch = _orch(graph, run_store=store)
    degradations = await orch._drain_graph_mirror_pending(run_id, ns)
    assert degradations == []
    assert graph.materialized[0]["communities"][0].id == community_id
    assert 7 not in store.pending
