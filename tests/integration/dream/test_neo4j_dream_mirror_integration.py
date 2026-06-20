"""Cross-store live-set invariant for the Neo4j dream tombstone-mirror (#1272).

The leg deferred from #1268: after a dream apply + post-commit mirror on a real
pg+Neo4j stack, the graph-preferring ``list_relationships`` / ``list_entities``
and the PG ground-truth live set must be byte-identical, and pruned / merged
self-loop edges must be invisible to graph recall.

This FAILS on origin/main (no mirror, gated read filter): PG soft-deletes the
row but Neo4j still returns the live edge. It PASSES once the mirror stamps the
graph ``valid_until`` and the read filter becomes unconditional.

How to run locally::

    make dev   # starts postgres (5434) + neo4j (7688) via compose
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
    KHORA_NEO4J_URL=bolt://localhost:7688 \\
    KHORA_NEO4J_USERNAME=neo4j KHORA_NEO4J_PASSWORD=pleaseletmein \\
        uv run pytest tests/integration/dream/test_neo4j_dream_mirror_integration.py -v
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from khora.config.schema import KhoraConfig
from khora.core.models.entity import Entity, Relationship
from khora.dream.config import DreamConfig
from khora.dream.engines.vectorcypher.dedupe_entities import apply_vectorcypher_dedupe_entities
from khora.dream.plan import DreamOp, DreamScope, OpKind
from khora.dream.runstore import select_run_store
from khora.khora import Khora

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
NEO4J_URL = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")


def _reachable(url: str, default_port: int) -> bool:
    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_reachable(DATABASE_URL, 5432) and _reachable(NEO4J_URL, 7687)),
        reason="pg+neo4j not reachable (run `make dev`)",
    ),
]

EMBED_DIM = 4


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=NEO4J_URL)
    config.storage.neo4j_user = NEO4J_USER
    config.storage.neo4j_password = NEO4J_PASSWORD
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.dream = DreamConfig(
        enabled=True,
        prune_edges_enabled=True,
        prune_edges_target_predicates=["ASSOCIATED_WITH"],
        prune_edges_confidence_threshold=0.4,
    )
    kb = Khora(config, run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        await kb.disconnect()


def _graph_backend(kb: Khora):
    graph = kb.storage.graph
    return getattr(graph, "_backend", graph)


async def _seed_entity_both(kb: Khora, ns_row_id: UUID, name: str) -> UUID:
    """Seed one entity into PG (entities table) and Neo4j with a matching id.

    Ingest resolves the stable namespace id to the row id before any write, so
    both stores carry ``ns_row_id`` on ``namespace_id`` (the FK target). We do
    the same here. ``coordinator.create_entity`` writes vector-first (the
    ``entities`` table recall reads) then graph, so one call lands the same id
    in both stores.
    """
    ent = Entity(namespace_id=ns_row_id, name=name, entity_type="PERSON", description=name)
    await kb.storage.create_entity(ent)
    return ent.id


async def _live_pg_relationship_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    async with kb.storage.transaction() as txn:
        rows = (
            await txn.session.execute(
                text(
                    "SELECT id FROM relationships WHERE namespace_id = :ns "
                    "AND valid_to IS NULL AND invalidated_at IS NULL "
                    "AND (valid_until IS NULL OR valid_until > now())"
                ),
                {"ns": ns_row_id},
            )
        ).all()
    return {str(r.id) for r in rows}


async def _live_pg_entity_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    async with kb.storage.transaction() as txn:
        rows = (
            await txn.session.execute(
                text(
                    "SELECT id FROM entities WHERE namespace_id = :ns AND (valid_until IS NULL OR valid_until > now())"
                ),
                {"ns": ns_row_id},
            )
        ).all()
    return {str(r.id) for r in rows}


async def _graph_relationship_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    rels = await kb.storage.list_relationships(ns_row_id, limit=1000)
    return {str(r.id) for r in rels}


async def _graph_entity_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    ents = await kb.storage.list_entities(ns_row_id, limit=1000)
    return {str(e.id) for e in ents}


@pytest.mark.asyncio
async def test_prune_edge_converges_pg_and_neo4j(kb: Khora) -> None:
    """A pruned ASSOCIATED_WITH edge is invisible to both PG and graph recall."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    a = await _seed_entity_both(kb, ns_row_id, f"alice-{uuid4().hex[:8]}")
    b = await _seed_entity_both(kb, ns_row_id, f"bob-{uuid4().hex[:8]}")

    rel = Relationship(
        namespace_id=ns_row_id,
        source_entity_id=a,
        target_entity_id=b,
        relationship_type="ASSOCIATED_WITH",
        description="co-occurrence",
        confidence=0.1,
    )
    await kb.storage.create_relationship(rel)
    # PG relationships row: low confidence, valid_to NULL, dead chunk ids (the
    # three-conjunct prune predicate). dead chunk id => no live chunks row.
    async with kb.storage.transaction() as txn:
        await txn.session.execute(
            text(
                "INSERT INTO relationships (id, namespace_id, source_entity_id, target_entity_id, "
                "relationship_type, description, properties, source_document_ids, source_chunk_ids, "
                "confidence, weight, metadata, created_at, updated_at) "
                "VALUES (:id, :ns, :src, :tgt, 'ASSOCIATED_WITH', '', '{}'::jsonb, '{}', "
                "ARRAY[:dead]::uuid[], 0.1, 1.0, '{}'::jsonb, now(), now())"
            ),
            {"id": rel.id, "ns": ns_row_id, "src": a, "tgt": b, "dead": uuid4()},
        )

    # Pre-apply: the edge is live in both stores.
    assert str(rel.id) in await _live_pg_relationship_ids(kb, ns_row_id)
    assert str(rel.id) in await _graph_relationship_ids(kb, ns_row_id)

    # Apply the prune op end-to-end (planner -> apply -> post-commit mirror).
    result = await kb.dream(
        ns_stable,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_PRUNE_EDGES,)),
    )
    # No graph-mirror degradation expected on the happy path.
    assert not result.metadata.get("degradations"), result.metadata.get("degradations")

    pg_live = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_live = await _graph_relationship_ids(kb, ns_row_id)
    # The pruned edge is invisible to PG recall ...
    assert str(rel.id) not in pg_live
    # ... and to graph recall (this is the leg that fails on origin/main).
    assert str(rel.id) not in graph_live
    # The live sets are byte-identical.
    assert pg_live == graph_live


@pytest.mark.asyncio
async def test_dedupe_self_loop_and_absorbed_entity_converge(kb: Khora) -> None:
    """dedupe soft-retires the absorbed entity + invalidates the self-loop in both stores."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}")

    # An edge canonical -> absorbed becomes a self-loop after the merge.
    rel = Relationship(
        namespace_id=ns_row_id,
        source_entity_id=canonical,
        target_entity_id=absorbed,
        relationship_type="RELATES_TO",
        description="will become a self-loop",
        confidence=0.9,
    )
    await kb.storage.create_relationship(rel)
    async with kb.storage.transaction() as txn:
        await txn.session.execute(
            text(
                "INSERT INTO relationships (id, namespace_id, source_entity_id, target_entity_id, "
                "relationship_type, description, properties, source_document_ids, source_chunk_ids, "
                "confidence, weight, metadata, created_at, updated_at) "
                "VALUES (:id, :ns, :src, :tgt, 'RELATES_TO', '', '{}'::jsonb, '{}', '{}', "
                "0.9, 1.0, '{}'::jsonb, now(), now())"
            ),
            {"id": rel.id, "ns": ns_row_id, "src": canonical, "tgt": absorbed},
        )

    # Drive the dedupe apply handler + the post-commit mirror directly (the
    # planner->apply bridge with embeddings is #1265; here we assert the #1272
    # mirror surface against a real pg+neo4j stack).
    orch = _orchestrator(kb)
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=({"merges": [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}]},),
        namespace_id=ns_row_id,
    )
    async with kb.storage.transaction() as txn:
        undo = await apply_vectorcypher_dedupe_entities(op, coordinator=kb.storage, session=txn.session)
    degradation = await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo)
    assert degradation is None

    pg_rel_live = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_rel_live = await _graph_relationship_ids(kb, ns_row_id)
    pg_ent_live = await _live_pg_entity_ids(kb, ns_row_id)
    graph_ent_live = await _graph_entity_ids(kb, ns_row_id)

    # The self-loop edge is invalidated in both stores ...
    assert str(rel.id) not in pg_rel_live
    assert str(rel.id) not in graph_rel_live
    assert pg_rel_live == graph_rel_live
    # ... and the absorbed entity is soft-retired in both stores.
    assert str(absorbed) not in pg_ent_live
    assert str(absorbed) not in graph_ent_live
    assert str(canonical) in pg_ent_live
    assert str(canonical) in graph_ent_live
    assert pg_ent_live == graph_ent_live


@pytest.mark.asyncio
async def test_crash_between_commit_and_mirror_reconciles(kb: Khora) -> None:
    """A mirror that raises once after the PG commit is healed by the reconciler."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    a = await _seed_entity_both(kb, ns_row_id, f"c-{uuid4().hex[:8]}")
    b = await _seed_entity_both(kb, ns_row_id, f"d-{uuid4().hex[:8]}")
    rel = Relationship(
        namespace_id=ns_row_id,
        source_entity_id=a,
        target_entity_id=b,
        relationship_type="ASSOCIATED_WITH",
        description="co-occurrence",
        confidence=0.1,
    )
    await kb.storage.create_relationship(rel)
    async with kb.storage.transaction() as txn:
        await txn.session.execute(
            text(
                "INSERT INTO relationships (id, namespace_id, source_entity_id, target_entity_id, "
                "relationship_type, description, properties, source_document_ids, source_chunk_ids, "
                "confidence, weight, metadata, created_at, updated_at) "
                "VALUES (:id, :ns, :src, :tgt, 'ASSOCIATED_WITH', '', '{}'::jsonb, '{}', "
                "ARRAY[:dead]::uuid[], 0.1, 1.0, '{}'::jsonb, now(), now())"
            ),
            {"id": rel.id, "ns": ns_row_id, "src": a, "tgt": b, "dead": uuid4()},
        )

    orch = _orchestrator(kb)
    run_id = uuid4()
    # A khora_dream_runs row must exist for graph_mirror_pending to persist.
    run_store = select_run_store(kb.storage)
    assert run_store is not None
    await run_store.record_run(run_id, ns_row_id, mode="apply", trigger="manual")

    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_PRUNE_EDGES,
        inputs=({"relationship_id": str(rel.id)},),
        namespace_id=ns_row_id,
    )

    # Simulate the crash window: PG commits, but the mirror raises once.
    from khora.dream.engines.vectorcypher.prune_edges import apply_vectorcypher_prune_edges

    async with kb.storage.transaction() as txn:
        undo = await apply_vectorcypher_prune_edges(op, coordinator=kb.storage, session=txn.session)

    graph = _graph_backend(kb)
    original = graph.soft_invalidate_relationships_batch
    calls = {"n": 0}

    async def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash-window mirror failure")
        return await original(*args, **kwargs)

    graph.soft_invalidate_relationships_batch = _flaky  # type: ignore[assignment]
    try:
        degradation = await orch._mirror_dream_op(run_id, 0, ns_row_id, op, undo)
        assert degradation is not None  # mirror failed, queued for reconcile
        # PG is ahead of the graph now: edge gone from PG, still live in graph.
        assert str(rel.id) not in await _live_pg_relationship_ids(kb, ns_row_id)
        assert str(rel.id) in await _graph_relationship_ids(kb, ns_row_id)

        # The reconciler drains the queued op (second call succeeds).
        degradations = await orch._drain_graph_mirror_pending(run_id, ns_row_id)
        assert degradations == []
    finally:
        graph.soft_invalidate_relationships_batch = original  # type: ignore[assignment]

    # Converged: invisible in both, byte-identical live sets.
    pg_live = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_live = await _graph_relationship_ids(kb, ns_row_id)
    assert str(rel.id) not in pg_live
    assert str(rel.id) not in graph_live
    assert pg_live == graph_live


def _orchestrator(kb: Khora):
    from khora.dream.orchestrator import DreamOrchestrator

    return DreamOrchestrator(kb, kb._config.dream, sinks=[])
