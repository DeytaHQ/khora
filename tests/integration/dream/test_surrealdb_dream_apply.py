"""SurrealDB-unified dream-apply: native path + flat soft-delete mirror (#1280).

The Phase-4 SurrealDB leg of the dream-on-graph umbrella (#1282). Before #1280
``coordinator.transaction()`` raised ``RuntimeError("No SQL backend ...")`` on a
SurrealDB-unified stack and the SQL apply handlers ran with ``session=None``
(crash / silent no-op). This module proves the SurrealQL-native apply path:

* ``vectorcypher_prune_edges``      — full ``kb.dream`` planner -> native apply,
  edge flat-soft-deleted, hidden from the live read, second pass is a no-op.
* ``vectorcypher_dedupe_entities``  — self-loop subset: absorbed entity retired +
  its incident edge (the self-loop) flat-soft-deleted; replay is idempotent.
* an unsupported op kind (centroid recompute) — declared unsupported with a
  structured ``surrealdb_native_apply_required`` skip (ADR-001), never crashing.

The store is unified (graph == vector == relational), so the live "graph"
read set IS the ground truth; the invariant we assert is internal convergence:
the soft-deleted rows disappear from ``list_entities`` / ``list_relationships``
and no live edge points at a retired entity, with idempotent replay.

Runs fully EMBEDDED (``memory://``) — no Docker.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.config import KhoraConfig  # noqa: E402
from khora.core.models import Entity, Relationship  # noqa: E402
from khora.dream.config import DreamConfig  # noqa: E402
from khora.dream.orchestrator import DreamOrchestrator, _surrealdb_connection  # noqa: E402
from khora.dream.plan import DreamOp, DreamPlan, DreamScope, OpKind  # noqa: E402
from khora.khora import Khora  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_EMBED_DIM = 4


@pytest.fixture
def _surrealdb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh in-memory SurrealDB-unified stack, with PG/Neo4j env leaks cleared."""
    for leaked in ("KHORA_DATABASE_URL", "KHORA_NEO4J_URL"):
        monkeypatch.delenv(leaked, raising=False)
    monkeypatch.setenv("KHORA_STORAGE_BACKEND", "surrealdb")
    monkeypatch.setenv("KHORA_STORAGE_SURREALDB_MODE", "memory")


@pytest.fixture
async def kb(_surrealdb_env: None) -> AsyncIterator[Khora]:
    config = KhoraConfig()
    config.storage.embedding_dimension = _EMBED_DIM
    config.llm.embedding_dimension = _EMBED_DIM
    config.dream = DreamConfig(
        enabled=True,
        prune_edges_enabled=True,
        prune_edges_target_predicates=["ASSOCIATED_WITH"],
        prune_edges_confidence_threshold=0.4,
    )
    khora = Khora(config, run_migrations=False)
    await khora.connect()
    try:
        yield khora
    finally:
        await khora.disconnect()


async def _live_entity_ids(kb: Khora, ns_row_id) -> set:
    return {e.id for e in await kb.storage.list_entities(ns_row_id, limit=1000)}


async def _live_relationship_ids(kb: Khora, ns_row_id) -> set:
    return {r.id for r in await kb.storage.list_relationships(ns_row_id, limit=1000)}


async def _seed_entity(kb: Khora, ns_row_id, name: str):
    ent = Entity(namespace_id=ns_row_id, name=name, entity_type="ORG", description=name)
    await kb.storage.create_entity(ent)
    return ent.id


async def _seed_edge(kb: Khora, ns_row_id, src, tgt, rel_type: str, *, confidence: float = 0.9, chunk_ids=None):
    rel = Relationship(
        namespace_id=ns_row_id,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description="edge",
        confidence=confidence,
        source_chunk_ids=list(chunk_ids or []),
    )
    await kb.storage.create_relationship(rel)
    return rel.id


async def test_surrealdb_unified_is_detected(kb: Khora) -> None:
    """Sanity: the coordinator is the unified-SurrealDB shape (no SQL session)."""
    conn = _surrealdb_connection(kb.storage)
    assert conn is not None, "expected a shared SurrealDBConnection on the unified stack"
    assert conn._mode == "memory"
    with pytest.raises(RuntimeError, match="No SQL backend"):
        async with kb.storage.transaction():
            pass


async def test_prune_edges_native_apply_converges_and_is_idempotent(kb: Khora) -> None:
    """prune_edges: full ``kb.dream`` apply soft-deletes the edge; it disappears
    from the live read and a second dream run applies ZERO ops."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    a = await _seed_entity(kb, ns_row_id, f"alice-{uuid4().hex[:8]}")
    b = await _seed_entity(kb, ns_row_id, f"bob-{uuid4().hex[:8]}")
    # Low confidence + a dead source-chunk id (no matching chunk row) satisfies
    # the three-conjunct prune predicate.
    rel_id = await _seed_edge(kb, ns_row_id, a, b, "ASSOCIATED_WITH", confidence=0.1, chunk_ids=[uuid4()])

    pre = await _live_relationship_ids(kb, ns_row_id)
    assert rel_id in pre

    result = await kb.dream(
        ns.namespace_id,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_PRUNE_EDGES,)),
    )
    assert sum(op.applied for op in result.ops) == 1, result.ops
    assert not result.metadata.get("degradations"), result.metadata

    post = await _live_relationship_ids(kb, ns_row_id)
    assert rel_id not in post, "pruned edge must be hidden from the live read"

    # Idempotent convergence: nothing left to prune.
    result2 = await kb.dream(
        ns.namespace_id,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_PRUNE_EDGES,)),
    )
    assert sum(op.applied for op in result2.ops) == 0, result2.ops
    assert await _live_relationship_ids(kb, ns_row_id) == post


async def test_dedupe_self_loop_native_apply_converges_and_replay_is_idempotent(kb: Khora) -> None:
    """dedupe self-loop: absorbed entity retired + the canonical<->absorbed edge
    (a self-loop after merge) soft-deleted; the live set is internally consistent
    and a replay of the same op leaves it unchanged."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    canonical = await _seed_entity(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}")
    loop = await _seed_edge(kb, ns_row_id, canonical, absorbed, "RELATES_TO")

    pre_ents = await _live_entity_ids(kb, ns_row_id)
    pre_rels = await _live_relationship_ids(kb, ns_row_id)
    assert {canonical, absorbed} <= pre_ents
    assert loop in pre_rels

    orch = DreamOrchestrator(kb, kb._config.dream, sinks=[])
    run_id = uuid4()
    await orch._init_run_row(run_id, ns_row_id, "apply")
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=({"merges": [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}]},),
        namespace_id=ns_row_id,
    )
    plan = DreamPlan(plan_id=uuid4(), namespace_id=ns_row_id, ops=(op,))
    result = await orch._apply_phase(run_id, plan, cancel_flag=asyncio.Event(), on_progress=None)
    assert sum(o.applied for o in result.ops) == 1, result.ops
    assert not result.metadata.get("degradations"), result.metadata

    ents = await _live_entity_ids(kb, ns_row_id)
    rels = await _live_relationship_ids(kb, ns_row_id)
    assert absorbed not in ents
    assert canonical in ents
    assert loop not in rels
    # Internal consistency: no live edge points at the retired entity.
    live_rels = await kb.storage.list_relationships(ns_row_id, limit=1000)
    assert all(r.source_entity_id in ents and r.target_entity_id in ents for r in live_rels)

    # Idempotent replay (same op_id) leaves the live sets unchanged.
    run_id2 = uuid4()
    await orch._init_run_row(run_id2, ns_row_id, "apply")
    plan2 = DreamPlan(plan_id=uuid4(), namespace_id=ns_row_id, ops=(op,))
    await orch._apply_phase(run_id2, plan2, cancel_flag=asyncio.Event(), on_progress=None)
    assert await _live_entity_ids(kb, ns_row_id) == ents
    assert await _live_relationship_ids(kb, ns_row_id) == rels


async def test_dedupe_multi_incident_edges_batched_in_one_round_trip(kb: Khora) -> None:
    """dedupe with several incident edges: all are flat-soft-deleted in a single
    batched round-trip (unique indexed bind names, no collision) and no live edge
    points at the retired entity."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    canonical = await _seed_entity(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}")
    n1 = await _seed_entity(kb, ns_row_id, f"vendor-{uuid4().hex[:8]}")
    n2 = await _seed_entity(kb, ns_row_id, f"client-{uuid4().hex[:8]}")
    loop = await _seed_edge(kb, ns_row_id, canonical, absorbed, "RELATES_TO")
    out_edge = await _seed_edge(kb, ns_row_id, absorbed, n1, "SUPPLIES")
    in_edge = await _seed_edge(kb, ns_row_id, n2, absorbed, "PAYS")

    conn = _surrealdb_connection(kb.storage)
    assert conn is not None, "expected unified SurrealDB connection"
    from khora.dream.engines.vectorcypher.surrealdb_apply import apply_surrealdb_op

    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=({"merges": [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}]},),
        namespace_id=ns_row_id,
    )
    undo = await apply_surrealdb_op(op, conn=conn)

    ents = await _live_entity_ids(kb, ns_row_id)
    rels = await _live_relationship_ids(kb, ns_row_id)
    assert absorbed not in ents
    assert {loop, out_edge, in_edge}.isdisjoint(rels), "every incident edge must be soft-deleted"
    assert len(undo.before["merges"][0]["edges_retired"]) == 3
    live = await kb.storage.list_relationships(ns_row_id, limit=1000)
    assert all(r.source_entity_id in ents and r.target_entity_id in ents for r in live)


async def test_dedupe_repoints_incident_edges_onto_canonical_and_replay_is_noop(kb: Khora) -> None:
    """dedupe re-pointing (#1303): each non-self-loop incident edge of the
    absorbed entity is re-pointed onto the canonical - the old edge is
    soft-deleted and an equivalent live edge to/from the canonical is created
    with the same type + properties. No live edge touches the absorbed entity,
    the canonical carries the re-pointed edges, and a replay is a no-op."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    canonical = await _seed_entity(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}")
    n1 = await _seed_entity(kb, ns_row_id, f"vendor-{uuid4().hex[:8]}")
    n2 = await _seed_entity(kb, ns_row_id, f"client-{uuid4().hex[:8]}")
    # canonical<->absorbed collapses to a self-loop on merge: soft-deleted, NOT
    # re-pointed. The out/in edges to third parties ARE re-pointed.
    loop = await _seed_edge(kb, ns_row_id, canonical, absorbed, "RELATES_TO")
    out_edge = await _seed_edge(kb, ns_row_id, absorbed, n1, "SUPPLIES", confidence=0.77)
    in_edge = await _seed_edge(kb, ns_row_id, n2, absorbed, "PAYS", confidence=0.55)

    conn = _surrealdb_connection(kb.storage)
    assert conn is not None, "expected unified SurrealDB connection"
    from khora.dream.engines.vectorcypher.surrealdb_apply import apply_surrealdb_op

    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=({"merges": [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}]},),
        namespace_id=ns_row_id,
    )
    undo = await apply_surrealdb_op(op, conn=conn)

    ents = await _live_entity_ids(kb, ns_row_id)
    rels = await _live_relationship_ids(kb, ns_row_id)
    assert absorbed not in ents
    assert canonical in ents
    # Every original incident edge id is soft-deleted (the re-pointed edges
    # carry fresh rel_ids).
    assert {loop, out_edge, in_edge}.isdisjoint(rels)

    live = await kb.storage.list_relationships(ns_row_id, limit=1000)
    # Internal consistency: no live edge points at the retired entity.
    assert all(r.source_entity_id in ents and r.target_entity_id in ents for r in live)

    # The canonical now carries the re-pointed edges, preserving type +
    # properties (the self-loop is NOT re-pointed).
    out_repointed = [r for r in live if r.source_entity_id == canonical and r.target_entity_id == n1]
    in_repointed = [r for r in live if r.source_entity_id == n2 and r.target_entity_id == canonical]
    assert len(out_repointed) == 1, "absorbed->n1 must be re-pointed to canonical->n1"
    assert len(in_repointed) == 1, "n2->absorbed must be re-pointed to n2->canonical"
    assert out_repointed[0].relationship_type == "SUPPLIES"
    assert out_repointed[0].confidence == 0.77
    assert in_repointed[0].relationship_type == "PAYS"
    assert in_repointed[0].confidence == 0.55
    # No self-loop on the canonical was created for the collapsed loop edge.
    assert not [r for r in live if r.source_entity_id == canonical and r.target_entity_id == canonical]

    undo_entry = undo.before["merges"][0]
    assert len(undo_entry["edges_repointed"]) == 2

    # Idempotent replay: a second apply of the same op creates no duplicate
    # re-pointed edge and leaves the live sets unchanged (the absorbed entity
    # has no live incident edges left, so nothing is re-pointed again).
    await apply_surrealdb_op(op, conn=conn)
    assert await _live_entity_ids(kb, ns_row_id) == ents
    assert await _live_relationship_ids(kb, ns_row_id) == rels


async def test_unsupported_op_is_skip_declared_not_crashed(kb: Khora) -> None:
    """An op with no SurrealQL-native handler is declared unsupported with a
    structured ``surrealdb_native_apply_required`` skip (ADR-001) - it advances
    the checkpoint (applied=0, skipped=1) and never crashes the run."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    orch = DreamOrchestrator(kb, kb._config.dream, sinks=[])
    run_id = uuid4()
    await orch._init_run_row(run_id, ns_row_id, "apply")
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
        inputs=({"entity_id": str(uuid4())},),
        namespace_id=ns_row_id,
    )
    plan = DreamPlan(plan_id=uuid4(), namespace_id=ns_row_id, ops=(op,))
    result = await orch._apply_phase(run_id, plan, cancel_flag=asyncio.Event(), on_progress=None)

    summary = next(o for o in result.ops if o.op_type == str(OpKind.VECTORCYPHER_CENTROID_RECOMPUTE))
    assert summary.applied == 0
    assert summary.skipped == 1
    reasons = result.metadata.get("skip_reasons") or []
    assert any(r["reason"] == "surrealdb_native_apply_required" for r in reasons), reasons
