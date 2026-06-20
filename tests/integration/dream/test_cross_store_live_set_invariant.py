"""Cross-store live-set permutation invariant - the master gate (#1268).

The phase-1 "master gate" for the dream-on-graph umbrella (#1282). It asserts
the single property the Phase-2 mirror had to satisfy: after a dream apply on a
real pg+Neo4j stack, the **graph-preferring** read path
(``coordinator.list_entities`` / ``list_relationships``, which prefer the graph
backend when one is configured) and the **PG ground-truth** live set return
byte-identical id sets - and a second convergence pass emits no further work.

One assertion shape, exercised across every mirrorable op kind:

  - ``vectorcypher_prune_edges``                  (soft-invalidate edge)
  - ``vectorcypher_dedupe_entities`` self-loop    (retire entity + invalidate loop)
  - ``vectorcypher_dedupe_entities`` entity-merge  (id-remap + incident re-point)
  - ``vectorcypher_community_summary``            (additive :Community materialize)

plus ``vectorcypher_normalize_schema``, whose graph-label relabel is NOT
mirrored (deferred): the gate proves it is a *documented* skip, not a silent
divergence - the live id sets stay byte-identical (a relabel removes no rows)
and the mirror surfaces a structured skip on the result (ADR-001).

The issue's original acceptance criteria expected this to FAIL on prune /
dedupe-self-loop / normalize_schema, documenting the live divergence bug. That
was written pre-mirror. With #1272 (Neo4j tombstone-mirror), #1273 (entity-merge
re-pointing), #1276 (community materialization) and #1274 (runstore) all landed
on main, every mirrorable op kind now CONVERGES; this is the blocking gate that
keeps them converged.

How to run locally::

    make dev   # starts postgres (5434) + neo4j (7688) via compose
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
    KHORA_NEO4J_URL=bolt://localhost:7688 \\
    KHORA_NEO4J_USERNAME=neo4j KHORA_NEO4J_PASSWORD=pleaseletmein \\
        uv run pytest tests/integration/dream/test_cross_store_live_set_invariant.py -v
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse
from uuid import UUID, uuid4, uuid5

import pytest
from sqlalchemy import text

from khora.config.schema import KhoraConfig
from khora.core.models.entity import Entity, Relationship
from khora.dream.config import DreamConfig
from khora.dream.engines.vectorcypher.dedupe_entities import apply_vectorcypher_dedupe_entities
from khora.dream.plan import DreamOp, DreamScope, OpKind
from khora.khora import Khora

# --- Connection wiring (copied from test_neo4j_dream_mirror_integration.py) ---
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


# --- Seed + live-set helpers (copied from test_neo4j_dream_mirror_integration.py) ---


async def _seed_entity_both(kb: Khora, ns_row_id: UUID, name: str) -> UUID:
    """Seed one entity into PG (entities table) and Neo4j with a matching id."""
    ent = Entity(namespace_id=ns_row_id, name=name, entity_type="PERSON", description=name)
    await kb.storage.create_entity(ent)
    return ent.id


async def _insert_pg_relationship(
    kb: Khora, ns_row_id: UUID, rel_id: UUID, src: UUID, tgt: UUID, rel_type: str, *, confidence: float = 0.9
) -> None:
    async with kb.storage.transaction() as txn:
        await txn.session.execute(
            text(
                "INSERT INTO relationships (id, namespace_id, source_entity_id, target_entity_id, "
                "relationship_type, description, properties, source_document_ids, source_chunk_ids, "
                "confidence, weight, metadata, created_at, updated_at) "
                "VALUES (:id, :ns, :src, :tgt, :rt, '', '{}'::jsonb, '{}', '{}', "
                ":conf, 1.0, '{}'::jsonb, now(), now())"
            ),
            {"id": rel_id, "ns": ns_row_id, "src": src, "tgt": tgt, "rt": rel_type, "conf": confidence},
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


def _orchestrator(kb: Khora):
    from khora.dream.orchestrator import DreamOrchestrator

    return DreamOrchestrator(kb, kb._config.dream, sinks=[])


async def _assert_live_sets_byte_identical(kb: Khora, ns_row_id: UUID) -> tuple[set[str], set[str]]:
    """The master-gate invariant: graph-preferring live set == PG ground-truth.

    Returns ``(entity_ids, relationship_ids)`` (the agreed-upon sets) so callers
    can make op-specific membership assertions on top of the parity check.
    """
    pg_ents = await _live_pg_entity_ids(kb, ns_row_id)
    graph_ents = await _graph_entity_ids(kb, ns_row_id)
    pg_rels = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_rels = await _graph_relationship_ids(kb, ns_row_id)
    assert graph_ents == pg_ents, (
        f"entity live-set divergence: graph-only={graph_ents - pg_ents} pg-only={pg_ents - graph_ents}"
    )
    assert graph_rels == pg_rels, (
        f"relationship live-set divergence: graph-only={graph_rels - pg_rels} pg-only={pg_rels - graph_rels}"
    )
    return pg_ents, pg_rels


async def _apply_dedupe_op(kb: Khora, ns_row_id: UUID, merges: list[dict[str, str]]):
    """Drive the dedupe apply handler + the post-commit mirror; return ``(op, undo)``.

    The planner->apply bridge needs embeddings (#1265), so the dedupe / community
    legs invoke the apply handler directly (as the #1273 mirror tests do) and
    assert the cross-store invariant on the result. The op_id stays stable across
    a replay so the mirror is byte-for-byte idempotent.
    """
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=({"merges": merges},),
        namespace_id=ns_row_id,
    )
    async with kb.storage.transaction() as txn:
        undo = await apply_vectorcypher_dedupe_entities(op, coordinator=kb.storage, session=txn.session)
    orch = _orchestrator(kb)
    assert await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo) is None
    return op, undo


async def _apply_community_op(kb: Khora, ns_row_id: UUID, community_id: UUID, members: list[UUID]):
    """Drive the community_summary apply handler with a mocked LLM; return ``(op, undo)``."""
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
            undo = await apply_vectorcypher_community_summary(op, coordinator=kb.storage, session=txn.session)
    finally:
        mod.acompletion = original
    return op, undo


# ---------------------------------------------------------------------------
# The master gate: one invariant, every mirrorable op kind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prune_full_pipeline_converges_and_is_idempotent(kb: Khora) -> None:
    """prune_edges: end-to-end ``kb.dream`` apply converges both stores; a second
    dream run emits ZERO applied ops (idempotent convergence)."""
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
    # PG row matching the three-conjunct prune predicate (low conf, valid_to NULL,
    # dead chunk id with no live chunks row).
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

    # Live in both stores before the run.
    _, pre_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert str(rel.id) in pre_rels

    # First apply via the public dream entry point (planner -> apply -> mirror).
    result = await kb.dream(
        ns_stable,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_PRUNE_EDGES,)),
    )
    assert not result.metadata.get("degradations"), result.metadata.get("degradations")
    assert sum(op.applied for op in result.ops) == 1, result.ops

    # Invariant: live sets byte-identical, and the pruned edge is gone from both.
    _, post_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert str(rel.id) not in post_rels

    # Idempotent convergence: a second dream run finds nothing left to prune.
    result2 = await kb.dream(
        ns_stable,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_PRUNE_EDGES,)),
    )
    assert not result2.metadata.get("degradations"), result2.metadata.get("degradations")
    assert sum(op.applied for op in result2.ops) == 0, result2.ops
    # Still byte-identical and unchanged after the no-op second pass.
    _, post_rels2 = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert post_rels2 == post_rels


@pytest.mark.asyncio
async def test_dedupe_self_loop_converges_and_replay_is_idempotent(kb: Khora) -> None:
    """dedupe self-loop: absorbed entity retired + self-loop invalidated in both
    stores; a mirror replay (same undo) leaves the live sets unchanged."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}")

    # canonical -> absorbed becomes a self-loop after the merge.
    loop = await _seed_edge_both(kb, ns_row_id, canonical, absorbed, "RELATES_TO")

    pre_ents, pre_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert {str(canonical), str(absorbed)} <= pre_ents
    assert str(loop) in pre_rels

    op, undo = await _apply_dedupe_op(kb, ns_row_id, [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}])

    ents, rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert str(absorbed) not in ents
    assert str(canonical) in ents
    assert str(loop) not in rels

    # Idempotent convergence: replay the mirror from the same undo - no shift.
    orch = _orchestrator(kb)
    assert await orch._mirror_dream_op(uuid4(), 1, ns_row_id, op, undo) is None
    ents2, rels2 = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert (ents2, rels2) == (ents, rels)


@pytest.mark.asyncio
async def test_dedupe_entity_merge_converges_and_replay_is_idempotent(kb: Khora) -> None:
    """dedupe entity-merge: incident edges re-point to the canonical (#1273); live
    sets byte-identical, no edge points at the retired node, and a replay is a
    no-op."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}")
    neighbor1 = await _seed_entity_both(kb, ns_row_id, f"vendor-{uuid4().hex[:8]}")
    neighbor2 = await _seed_entity_both(kb, ns_row_id, f"client-{uuid4().hex[:8]}")

    out_edge = await _seed_edge_both(kb, ns_row_id, absorbed, neighbor1, "SUPPLIES")
    in_edge = await _seed_edge_both(kb, ns_row_id, neighbor2, absorbed, "PAYS")
    canon_edge = await _seed_edge_both(kb, ns_row_id, canonical, neighbor1, "SUPPLIES")

    await _assert_live_sets_byte_identical(kb, ns_row_id)

    op, undo = await _apply_dedupe_op(kb, ns_row_id, [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}])

    ents, rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert str(absorbed) not in ents
    assert str(canonical) in ents
    # All three edges survive (re-pointing keeps them live), by id, in both stores.
    assert {str(out_edge), str(in_edge), str(canon_edge)} <= rels
    # No live edge points at the retired absorbed node, in EITHER store.
    pg_eps = await _pg_relationship_endpoints(kb, ns_row_id)
    graph_eps = await _graph_relationship_endpoints(kb, ns_row_id)
    assert pg_eps == graph_eps
    assert all(str(absorbed) not in eps for eps in pg_eps.values())
    assert pg_eps[str(out_edge)] == (str(canonical), str(neighbor1))
    assert pg_eps[str(in_edge)] == (str(neighbor2), str(canonical))

    # Idempotent convergence: replay the mirror - endpoints and live sets unchanged.
    orch = _orchestrator(kb)
    assert await orch._mirror_dream_op(uuid4(), 1, ns_row_id, op, undo) is None
    ents2, rels2 = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert (ents2, rels2) == (ents, rels)
    assert await _graph_relationship_endpoints(kb, ns_row_id) == graph_eps


@pytest.mark.asyncio
async def test_community_materializes_without_disturbing_live_sets(kb: Khora) -> None:
    """community_summary: an additive :Community materialization leaves the entity
    / relationship live sets byte-identical across stores, is queryable, and a
    second mirror creates no duplicate node."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    members = [await _seed_entity_both(kb, ns_row_id, f"m{i}-{uuid4().hex[:8]}") for i in range(3)]
    community_id = uuid5(ns_row_id, ",".join(sorted(str(m) for m in members)))

    pre_ents, pre_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)

    op, undo = await _apply_community_op(kb, ns_row_id, community_id, members)
    orch = _orchestrator(kb)
    assert await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo) is None

    # The :Community node is queryable, and the underlying live sets are unchanged.
    communities = await kb.storage.get_communities(ns_row_id, limit=100)
    assert community_id in {c.id for c in communities}
    ents, rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert ents == pre_ents
    assert rels == pre_rels

    # Idempotent convergence: a second mirror is a no-op (MERGE on community_id).
    assert await orch._mirror_dream_op(uuid4(), 1, ns_row_id, op, undo) is None
    matching = [c for c in await kb.storage.get_communities(ns_row_id, limit=100) if c.id == community_id]
    assert len(matching) == 1, matching
    ents2, rels2 = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert (ents2, rels2) == (ents, rels)


@pytest.mark.asyncio
async def test_normalize_schema_relabel_is_documented_skip_not_silent_divergence(kb: Khora) -> None:
    """normalize_schema: its graph-label relabel is NOT mirrored (deferred). The
    live id sets stay byte-identical (a relabel removes no rows), and the mirror
    surfaces a STRUCTURED skip on the result - not a silent divergence (ADR-001)."""
    from khora.dream.engines.vectorcypher.normalize_schema import apply_vectorcypher_normalize_schema

    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    ent = await _seed_entity_both(kb, ns_row_id, f"org-{uuid4().hex[:8]}")
    pre_ents, pre_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)

    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_NORMALIZE_SCHEMA,
        outputs=(
            {
                "entity_renames": [{"id": str(ent), "old_type": "PERSON", "new_type": "ORGANIZATION"}],
                "relationship_renames": [],
            },
        ),
        namespace_id=ns_row_id,
    )
    async with kb.storage.transaction() as txn:
        undo = await apply_vectorcypher_normalize_schema(op, coordinator=kb.storage, session=txn.session)

    orch = _orchestrator(kb)
    record = await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo)
    # A structured skip - the divergence (graph label not relabeled) is ACCOUNTED
    # for on the result, not silently dropped.
    assert record is not None, "normalize_schema must surface a structured skip (ADR-001)"
    assert record["component"] == "dream.graph_mirror"
    assert record["reason"] == "graph_mirror_unsupported_op_kind"

    # The live id sets are byte-identical: the relabel removed no rows. (The PG
    # type label changed; the GRAPH label is intentionally out of scope for the
    # id-set invariant - that mirror is deferred.)
    post_ents, post_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert post_ents == pre_ents
    assert post_rels == pre_rels
    assert str(ent) in post_ents


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
