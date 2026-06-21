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


async def _pg_relationship_endpoints(kb: Khora, ns_row_id: UUID) -> dict[str, tuple[str, str]]:
    """Map each live PG relationship id -> (source_entity_id, target_entity_id)."""
    async with kb.storage.transaction() as txn:
        rows = (
            await txn.session.execute(
                text(
                    "SELECT id, source_entity_id, target_entity_id FROM relationships "
                    "WHERE namespace_id = :ns AND valid_to IS NULL AND invalidated_at IS NULL "
                    "AND (valid_until IS NULL OR valid_until > now())"
                ),
                {"ns": ns_row_id},
            )
        ).all()
    return {str(r.id): (str(r.source_entity_id), str(r.target_entity_id)) for r in rows}


async def _graph_relationship_endpoints(kb: Khora, ns_row_id: UUID) -> dict[str, tuple[str, str]]:
    """Map each live graph relationship id -> (source_entity_id, target_entity_id)."""
    rels = await kb.storage.list_relationships(ns_row_id, limit=1000)
    return {str(r.id): (str(r.source_entity_id), str(r.target_entity_id)) for r in rels}


async def _insert_pg_relationship(
    kb: Khora, ns_row_id: UUID, rel_id: UUID, src: UUID, tgt: UUID, rel_type: str
) -> None:
    async with kb.storage.transaction() as txn:
        await txn.session.execute(
            text(
                "INSERT INTO relationships (id, namespace_id, source_entity_id, target_entity_id, "
                "relationship_type, description, properties, source_document_ids, source_chunk_ids, "
                "confidence, weight, metadata, created_at, updated_at) "
                "VALUES (:id, :ns, :src, :tgt, :rt, '', '{}'::jsonb, '{}', '{}', "
                "0.9, 1.0, '{}'::jsonb, now(), now())"
            ),
            {"id": rel_id, "ns": ns_row_id, "src": src, "tgt": tgt, "rt": rel_type},
        )


async def _seed_edge_both(kb: Khora, ns_row_id: UUID, src: UUID, tgt: UUID, rel_type: str) -> UUID:
    """Create one edge in both stores (graph via create_relationship, PG via SQL)."""
    rel = Relationship(
        namespace_id=ns_row_id,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description="incident",
        confidence=0.9,
    )
    await kb.storage.create_relationship(rel)
    await _insert_pg_relationship(kb, ns_row_id, rel.id, src, tgt, rel_type)
    return rel.id


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
async def test_dedupe_entity_merge_repoints_incident_edges(kb: Khora) -> None:
    """Entity-merge re-points every incident edge from A and B onto the canonical
    in BOTH stores (#1273): endpoint parity + cross-store live-set parity."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}")
    neighbor1 = await _seed_entity_both(kb, ns_row_id, f"vendor-{uuid4().hex[:8]}")
    neighbor2 = await _seed_entity_both(kb, ns_row_id, f"client-{uuid4().hex[:8]}")

    # Edges incident to the absorbed entity, both directions.
    out_edge = await _seed_edge_both(kb, ns_row_id, absorbed, neighbor1, "SUPPLIES")
    in_edge = await _seed_edge_both(kb, ns_row_id, neighbor2, absorbed, "PAYS")
    # An edge already on the canonical must be left untouched.
    canon_edge = await _seed_edge_both(kb, ns_row_id, canonical, neighbor1, "SUPPLIES")

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

    pg_eps = await _pg_relationship_endpoints(kb, ns_row_id)
    graph_eps = await _graph_relationship_endpoints(kb, ns_row_id)

    # Every incident edge re-points to the canonical, identically in both stores.
    assert pg_eps[str(out_edge)] == (str(canonical), str(neighbor1))
    assert graph_eps[str(out_edge)] == (str(canonical), str(neighbor1))
    assert pg_eps[str(in_edge)] == (str(neighbor2), str(canonical))
    assert graph_eps[str(in_edge)] == (str(neighbor2), str(canonical))
    # The pre-existing canonical edge is untouched.
    assert pg_eps[str(canon_edge)] == (str(canonical), str(neighbor1))
    assert graph_eps[str(canon_edge)] == (str(canonical), str(neighbor1))
    # No live edge points at the retired absorbed node, in either store.
    assert all(str(absorbed) not in eps for eps in pg_eps.values())
    assert all(str(absorbed) not in eps for eps in graph_eps.values())
    # Endpoint maps are byte-identical across stores.
    assert pg_eps == graph_eps

    # The absorbed node is retired in both stores; the canonical survives.
    pg_ent_live = await _live_pg_entity_ids(kb, ns_row_id)
    graph_ent_live = await _graph_entity_ids(kb, ns_row_id)
    assert str(absorbed) not in pg_ent_live
    assert str(absorbed) not in graph_ent_live
    assert str(canonical) in pg_ent_live
    assert str(canonical) in graph_ent_live


@pytest.mark.asyncio
async def test_dedupe_transitive_merge_collapses_to_one_canonical(kb: Khora) -> None:
    """A->B->C in one op collapses to a single canonical with no edge pointing at
    a retired intermediate, in both stores (#806 id-remap)."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    canonical = await _seed_entity_both(kb, ns_row_id, f"a-{uuid4().hex[:8]}")
    b = await _seed_entity_both(kb, ns_row_id, f"b-{uuid4().hex[:8]}")
    c = await _seed_entity_both(kb, ns_row_id, f"c-{uuid4().hex[:8]}")
    neighbor = await _seed_entity_both(kb, ns_row_id, f"n-{uuid4().hex[:8]}")

    edge_to_b = await _seed_edge_both(kb, ns_row_id, neighbor, b, "KNOWS")
    edge_b_to_c = await _seed_edge_both(kb, ns_row_id, b, c, "KNOWS")

    orch = _orchestrator(kb)
    # The Phase-1 planner emits one component with a single canonical absorbing
    # both B and C (two merge entries sharing canonical_id).
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=(
            {
                "merges": [
                    {"canonical_id": str(canonical), "absorbed_id": str(b)},
                    {"canonical_id": str(canonical), "absorbed_id": str(c)},
                ]
            },
        ),
        namespace_id=ns_row_id,
    )
    async with kb.storage.transaction() as txn:
        undo = await apply_vectorcypher_dedupe_entities(op, coordinator=kb.storage, session=txn.session)
    degradation = await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo)
    assert degradation is None

    pg_eps = await _pg_relationship_endpoints(kb, ns_row_id)
    graph_eps = await _graph_relationship_endpoints(kb, ns_row_id)

    # neighbor -> b re-points to neighbor -> canonical.
    assert pg_eps[str(edge_to_b)] == (str(neighbor), str(canonical))
    assert graph_eps[str(edge_to_b)] == (str(neighbor), str(canonical))
    # b -> c becomes canonical -> canonical (self-loop) and is invalidated in both.
    assert str(edge_b_to_c) not in pg_eps
    assert str(edge_b_to_c) not in graph_eps
    # No edge points at a retired intermediate (B or C).
    for eps in list(pg_eps.values()) + list(graph_eps.values()):
        assert str(b) not in eps
        assert str(c) not in eps
    assert pg_eps == graph_eps

    pg_ent_live = await _live_pg_entity_ids(kb, ns_row_id)
    graph_ent_live = await _graph_entity_ids(kb, ns_row_id)
    assert str(b) not in pg_ent_live and str(c) not in pg_ent_live
    assert str(b) not in graph_ent_live and str(c) not in graph_ent_live
    assert str(canonical) in pg_ent_live and str(canonical) in graph_ent_live


@pytest.mark.asyncio
async def test_dedupe_duplicate_key_after_repoint_no_violation(kb: Khora) -> None:
    """Two edges that collapse to the same (src, tgt, type) after re-pointing do
    not raise a constraint violation; both re-point to the canonical in lockstep."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme2-{uuid4().hex[:8]}")
    neighbor = await _seed_entity_both(kb, ns_row_id, f"vendor-{uuid4().hex[:8]}")

    # canonical -> neighbor AND absorbed -> neighbor, same type: after the merge
    # both are canonical -> neighbor:SUPPLIES (duplicate key).
    canon_edge = await _seed_edge_both(kb, ns_row_id, canonical, neighbor, "SUPPLIES")
    absorbed_edge = await _seed_edge_both(kb, ns_row_id, absorbed, neighbor, "SUPPLIES")

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
    # No constraint violation on the graph mirror (Neo4j allows parallel edges;
    # there is no relationships UNIQUE on PG).
    degradation = await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo)
    assert degradation is None

    pg_eps = await _pg_relationship_endpoints(kb, ns_row_id)
    graph_eps = await _graph_relationship_endpoints(kb, ns_row_id)
    # Both edges now point canonical -> neighbor, in both stores, by id.
    assert pg_eps[str(canon_edge)] == (str(canonical), str(neighbor))
    assert pg_eps[str(absorbed_edge)] == (str(canonical), str(neighbor))
    assert graph_eps[str(canon_edge)] == (str(canonical), str(neighbor))
    assert graph_eps[str(absorbed_edge)] == (str(canonical), str(neighbor))
    assert pg_eps == graph_eps


@pytest.mark.asyncio
async def test_dedupe_entity_merge_mirror_idempotent_replay(kb: Khora) -> None:
    """Re-running the mirror from the same undo is a no-op (idempotent / convergent)."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme2-{uuid4().hex[:8]}")
    neighbor = await _seed_entity_both(kb, ns_row_id, f"vendor-{uuid4().hex[:8]}")
    edge = await _seed_edge_both(kb, ns_row_id, absorbed, neighbor, "SUPPLIES")

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

    assert await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo) is None
    graph_eps_first = await _graph_relationship_endpoints(kb, ns_row_id)
    # Replay the mirror with the SAME undo: convergent, no duplicate, no shift.
    assert await orch._mirror_dream_op(uuid4(), 1, ns_row_id, op, undo) is None
    graph_eps_second = await _graph_relationship_endpoints(kb, ns_row_id)

    assert graph_eps_first == graph_eps_second
    assert graph_eps_second[str(edge)] == (str(canonical), str(neighbor))


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


@pytest.mark.asyncio
async def test_hard_crash_after_commit_heals_via_later_run(kb: Khora) -> None:
    """#1292 gap 1: a hard crash after the PG commit (before the mirror even runs)
    leaves a DURABLE pending row, and a LATER run with a fresh run_id - draining
    by namespace - re-mirrors it.

    On origin/main the pending row is only written inside ``_mirror_dream_op``'s
    except handler, so a crash before that point leaves NO row; and the drain
    reads only the current run_id, so a new run never retries it. This asserts
    both halves of the fix: the durable pre-mark (inside the apply tx) and the
    namespace-scoped drain (across run_ids).
    """
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    a = await _seed_entity_both(kb, ns_row_id, f"e-{uuid4().hex[:8]}")
    b = await _seed_entity_both(kb, ns_row_id, f"f-{uuid4().hex[:8]}")
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

    from khora.dream.engines.registry import get_apply_handler

    # --- Run A: commit the op + durable pre-mark, then "crash" (never mirror) ---
    orch_a = _orchestrator(kb)
    run_a = uuid4()
    run_store = select_run_store(kb.storage)
    assert run_store is not None
    await run_store.record_run(run_a, ns_stable, mode="apply", trigger="manual")

    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_PRUNE_EDGES,
        inputs=({"relationship_id": str(rel.id)},),
        namespace_id=ns_row_id,
    )
    handler = get_apply_handler(op.op_type)
    assert handler is not None
    # _apply_one_op commits the PG apply + checkpoint AND the durable pending
    # pre-mark in one transaction. We stop here - simulating a process death
    # before the post-commit mirror runs.
    await orch_a._apply_one_op(run_id=run_a, seq=0, op=op, handler=handler, namespace_id=ns_stable)

    # PG soft-deleted the edge; the graph still shows it (mirror never ran).
    assert str(rel.id) not in await _live_pg_relationship_ids(kb, ns_row_id)
    assert str(rel.id) in await _graph_relationship_ids(kb, ns_row_id)
    # The crash-durable pending row exists under run A.
    pending = await run_store.get_graph_mirror_pending(run_a)
    assert any(p.op_seq == 0 for p in pending), "no durable pending row after the commit (crash window open)"

    # --- Run B: a fresh run drains the prior run's pending op by namespace ---
    orch_b = _orchestrator(kb)
    run_b = uuid4()
    await run_store.record_run(run_b, ns_stable, mode="apply", trigger="manual")
    degradations = await orch_b._drain_graph_mirror_pending(run_b, ns_stable)
    assert degradations == [], degradations

    # Converged: invisible in both stores, byte-identical live sets.
    pg_live = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_live = await _graph_relationship_ids(kb, ns_row_id)
    assert str(rel.id) not in pg_live
    assert str(rel.id) not in graph_live
    assert pg_live == graph_live
    # Run A's pending row is cleared (drained under its own run_id).
    assert await run_store.get_graph_mirror_pending(run_a) == []


def _orchestrator(kb: Khora):
    from khora.dream.orchestrator import DreamOrchestrator

    return DreamOrchestrator(kb, kb._config.dream, sinks=[])


# ---------------------------------------------------------------------------
# Community materialization (#1276 - the GraphRAG payoff)
# ---------------------------------------------------------------------------


async def _apply_community_op(kb: Khora, ns_row_id: UUID, community_id: UUID, members: list[UUID]):
    """Drive the community_summary apply handler with a mocked LLM, then return the undo.

    The planner->apply path needs >=5 entities + an LLM call; here we invoke the
    apply handler directly (as the dedupe test does for #1273's surface) with a
    deterministic grounded summary so the assertion targets the #1276 mirror.
    """
    import json

    import khora.dream.engines.vectorcypher.community_summary as mod
    from khora.dream.engines.vectorcypher.community_summary import apply_vectorcypher_community_summary

    summary_json = json.dumps(
        {
            "text": "Alice and Bob collaborate within the community.",
            "claims": [{"text": "they collaborate", "cited_entity_ids": [str(members[0]), str(members[1])]}],
        }
    )

    async def _fake_acompletion(prompt, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return summary_json

    original = mod.acompletion
    mod.acompletion = _fake_acompletion
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_COMMUNITY_SUMMARY,
        inputs=(
            {
                "community_id": str(community_id),
                "member_ids": [str(m) for m in members],
                "member_names": [f"e{i}" for i in range(len(members))],
                "member_types": ["PERSON"] * len(members),
                "relationship_modes": {"WORKS_WITH": 1},
            },
        ),
        namespace_id=ns_row_id,
    )
    try:
        async with kb.storage.transaction() as txn:
            return op, await apply_vectorcypher_community_summary(op, coordinator=kb.storage, session=txn.session)
    finally:
        mod.acompletion = original


@pytest.mark.asyncio
async def test_community_materializes_to_graph_and_is_queryable(kb: Khora) -> None:
    """A dream community is materialized as a :Community node + HAS_MEMBER edges, queryable at recall."""
    from uuid import uuid5

    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    members = [await _seed_entity_both(kb, ns_row_id, f"m{i}-{uuid4().hex[:8]}") for i in range(3)]
    community_id = uuid5(ns_row_id, ",".join(sorted(str(m) for m in members)))

    op, undo = await _apply_community_op(kb, ns_row_id, community_id, members)
    assert undo.before.get("kept_claims", 0) >= 1, undo.before

    orch = _orchestrator(kb)
    degradation = await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo)
    assert degradation is None

    # The :Community node + HAS_MEMBER edges exist in the graph.
    communities = await kb.storage.get_communities(ns_row_id, limit=100)
    by_id = {c.id: c for c in communities}
    assert community_id in by_id
    com = by_id[community_id]
    assert "collaborate" in com.summary
    assert set(com.member_ids) == set(members)

    # The entity-anchored recall reader returns the same community.
    via_entity = await kb.storage.get_entity_communities([members[0]], namespace_id=ns_row_id)
    assert community_id in {c.id for c in via_entity}

    # The Khora-level recall accessor surfaces it via the stable namespace id.
    via_khora = await kb.get_communities(namespace=ns.namespace_id)
    assert community_id in {c.id for c in via_khora}


@pytest.mark.asyncio
async def test_community_materialization_idempotent_on_community_id(kb: Khora) -> None:
    """A second mirror of the same community_id creates no duplicate :Community node."""
    from uuid import uuid5

    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    members = [await _seed_entity_both(kb, ns_row_id, f"d{i}-{uuid4().hex[:8]}") for i in range(3)]
    community_id = uuid5(ns_row_id, ",".join(sorted(str(m) for m in members)))

    op, undo = await _apply_community_op(kb, ns_row_id, community_id, members)
    orch = _orchestrator(kb)

    # Mirror twice (the second is a reconciler-shaped replay).
    assert await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo) is None
    assert await orch._mirror_dream_op(uuid4(), 1, ns_row_id, op, undo) is None

    communities = await kb.storage.get_communities(ns_row_id, limit=100)
    matching = [c for c in communities if c.id == community_id]
    # MERGE on community_id -> exactly one node, no duplicate.
    assert len(matching) == 1, matching


@pytest.mark.asyncio
async def test_community_materializes_node_even_with_unresolved_member(kb: Khora) -> None:
    """A community whose member id has no graph node still materializes its :Community node."""
    from uuid import uuid5

    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    real = await _seed_entity_both(kb, ns_row_id, f"r-{uuid4().hex[:8]}")
    ghost = uuid4()  # never seeded into the graph
    members = [real, ghost]
    community_id = uuid5(ns_row_id, ",".join(sorted(str(m) for m in members)))

    op, undo = await _apply_community_op(kb, ns_row_id, community_id, members)
    orch = _orchestrator(kb)
    assert await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo) is None

    communities = await kb.storage.get_communities(ns_row_id, limit=100)
    by_id = {c.id: c for c in communities}
    # The :Community node exists even though one member has no graph node.
    assert community_id in by_id
    # Only the real member resolves a HAS_MEMBER edge.
    via_real = await kb.storage.get_entity_communities([real], namespace_id=ns_row_id)
    assert community_id in {c.id for c in via_real}
    via_ghost = await kb.storage.get_entity_communities([ghost], namespace_id=ns_row_id)
    assert community_id not in {c.id for c in via_ghost}


@pytest.mark.asyncio
async def test_community_skip_when_backend_lacks_capability(kb: Khora) -> None:
    """A backend that does not advertise the community op kind records a structured skip, not a divergence."""

    class _NoCommunityGraph:
        def supports_dream_mirror(self):  # noqa: ANN202
            return frozenset()

    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)
    members = [await _seed_entity_both(kb, ns_row_id, f"s{i}-{uuid4().hex[:8]}") for i in range(3)]
    community_id = uuid4()
    op, undo = await _apply_community_op(kb, ns_row_id, community_id, members)

    orch = _orchestrator(kb)
    real_graph = kb.storage._graph
    kb.storage._graph = _NoCommunityGraph()  # type: ignore[assignment]
    try:
        # A structured skip is surfaced (ADR-001), no exception, nothing materialized.
        record = await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo)
        assert record is not None, "unsupported-mirror skip must surface on the result (ADR-001)"
        assert record["reason"] == "graph_mirror_unsupported_op_kind"
        assert record["component"] == "dream.graph_mirror"
    finally:
        kb.storage._graph = real_graph  # type: ignore[assignment]

    # Nothing was materialized for this community.
    communities = await kb.storage.get_communities(ns_row_id, limit=100)
    assert community_id not in {c.id for c in communities}
