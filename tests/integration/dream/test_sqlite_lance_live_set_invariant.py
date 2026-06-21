"""Embedded-stack live-set invariant for dream-on-graph (#1277, phase 4).

The sqlite_lance analog of the pg+Neo4j master gate in
``test_cross_store_live_set_invariant.py`` (#1268). On the embedded stack the
graph backend *is* the SQLite store the SQL apply handler writes to (one store,
no separate mirror), so the invariant collapses to a single property: after a
dream apply, the graph-preferring read path
(``coordinator.list_entities`` / ``list_relationships``) and the
ground-truth live set computed with the same soft-delete predicate the pgvector
backend uses return byte-identical id sets - and a second dream run emits ZERO
applied ops.

This is the regression guard for the two things #1277 lifted:

  * the dialect gate (#875): dedupe / prune_edges / normalize_schema apply
    handlers are now UUID-bind-safe on SQLite and run instead of being skipped;
  * the embedded read path: ``list_entities`` / ``list_relationships`` honor
    ``valid_to`` / ``invalidated_at`` / ``valid_until`` so retired / invalidated
    rows drop from the live set.

No Docker: the fixture stack runs in-process on sqlite_lance + LanceDB.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._helpers import embedded_khora, install_mock_llm  # noqa: E402
from khora.core.models.entity import Entity, Relationship  # noqa: E402
from khora.dream.config import DreamConfig  # noqa: E402
from khora.dream.engines.vectorcypher.dedupe_entities import (  # noqa: E402
    apply_vectorcypher_dedupe_entities,
)
from khora.dream.plan import DreamOp, DreamScope, OpKind  # noqa: E402
from khora.khora import Khora  # noqa: E402

pytestmark = pytest.mark.embedded

EMBED_DIM = 8


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    install_mock_llm(dim=EMBED_DIM)
    async with embedded_khora(embedding_dimension=EMBED_DIM, engine="vectorcypher") as instance:
        instance._config.dream = DreamConfig(
            enabled=True,
            prune_edges_enabled=True,
            prune_edges_target_predicates=["ASSOCIATED_WITH"],
            prune_edges_confidence_threshold=0.4,
        )
        yield instance


# --- Seed helpers --------------------------------------------------------------


async def _seed_entity(kb: Khora, ns_row_id: UUID, name: str, entity_type: str = "PERSON") -> UUID:
    ent = Entity(namespace_id=ns_row_id, name=name, entity_type=entity_type, description=name)
    await kb.storage.create_entity(ent)
    return ent.id


async def _seed_edge(
    kb: Khora,
    ns_row_id: UUID,
    src: UUID,
    tgt: UUID,
    rel_type: str,
    *,
    confidence: float = 0.9,
    source_chunk_ids: list[UUID] | None = None,
) -> UUID:
    rel = Relationship(
        namespace_id=ns_row_id,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description="edge",
        confidence=confidence,
        source_chunk_ids=source_chunk_ids or [],
    )
    await kb.storage.create_relationship(rel)
    return rel.id


# --- Ground-truth live sets (same predicate the pgvector backend filters on) ---


async def _truth_entity_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    async with kb.storage.transaction() as txn:
        rows = (
            await txn.session.execute(
                text(
                    "SELECT id FROM entities WHERE namespace_id = :ns "
                    "AND (valid_until IS NULL OR datetime(valid_until) > datetime('now'))"
                ),
                {"ns": ns_row_id.hex},
            )
        ).all()
    return {str(UUID(r.id)) for r in rows}


async def _truth_relationship_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    async with kb.storage.transaction() as txn:
        rows = (
            await txn.session.execute(
                text(
                    "SELECT id FROM relationships WHERE namespace_id = :ns "
                    "AND valid_to IS NULL AND invalidated_at IS NULL "
                    "AND (valid_until IS NULL OR datetime(valid_until) > datetime('now'))"
                ),
                {"ns": ns_row_id.hex},
            )
        ).all()
    return {str(UUID(r.id)) for r in rows}


async def _graph_entity_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    return {str(e.id) for e in await kb.storage.list_entities(ns_row_id, limit=1000)}


async def _graph_relationship_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    return {str(r.id) for r in await kb.storage.list_relationships(ns_row_id, limit=1000)}


async def _assert_live_sets_byte_identical(kb: Khora, ns_row_id: UUID) -> tuple[set[str], set[str]]:
    """The embedded invariant: graph-preferring live set == ground-truth live set."""
    truth_ents = await _truth_entity_ids(kb, ns_row_id)
    graph_ents = await _graph_entity_ids(kb, ns_row_id)
    truth_rels = await _truth_relationship_ids(kb, ns_row_id)
    graph_rels = await _graph_relationship_ids(kb, ns_row_id)
    assert graph_ents == truth_ents, (
        f"entity live-set divergence: graph-only={graph_ents - truth_ents} truth-only={truth_ents - graph_ents}"
    )
    assert graph_rels == truth_rels, (
        f"relationship live-set divergence: graph-only={graph_rels - truth_rels} truth-only={truth_rels - graph_rels}"
    )
    return truth_ents, truth_rels


# ---------------------------------------------------------------------------
# The gate: prune + dedupe-self-loop converge on the embedded stack
# ---------------------------------------------------------------------------


async def test_prune_full_pipeline_converges_and_is_idempotent(kb: Khora) -> None:
    """prune_edges: ``kb.dream`` apply drops the orphan edge from the live set;
    the live sets stay byte-identical and a second run applies ZERO ops."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    a = await _seed_entity(kb, ns_row_id, f"alice-{uuid4().hex[:8]}")
    b = await _seed_entity(kb, ns_row_id, f"bob-{uuid4().hex[:8]}")
    # Orphan edge: low confidence, valid_to NULL, every source chunk dead (no
    # live chunks row exists for the random id) - matches the prune predicate.
    rel = await _seed_edge(kb, ns_row_id, a, b, "ASSOCIATED_WITH", confidence=0.1, source_chunk_ids=[uuid4()])

    _, pre_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert str(rel) in pre_rels

    result = await kb.dream(
        ns_stable,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_PRUNE_EDGES,)),
    )
    # The op ran (gate lifted) and was applied, not skipped.
    assert sum(op.applied for op in result.ops) == 1, result.ops
    assert sum(op.skipped for op in result.ops) == 0, result.ops

    _, post_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert str(rel) not in post_rels

    # Idempotent convergence: a second run finds nothing left to prune.
    result2 = await kb.dream(
        ns_stable,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_PRUNE_EDGES,)),
    )
    assert sum(op.applied for op in result2.ops) == 0, result2.ops
    _, post_rels2 = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert post_rels2 == post_rels


async def test_dedupe_self_loop_converges_and_replay_is_idempotent(kb: Khora) -> None:
    """dedupe self-loop: the absorbed entity is retired and the self-loop is
    invalidated; both drop from the live set, the sets stay byte-identical, and a
    replay of the same merge changes nothing."""
    ns = await kb.create_namespace()
    ns_row_id = await kb.storage.resolve_namespace(ns.namespace_id)

    canonical = await _seed_entity(kb, ns_row_id, f"acme-{uuid4().hex[:8]}", "ORGANIZATION")
    absorbed = await _seed_entity(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}", "ORGANIZATION")
    # canonical -> absorbed becomes a self-loop once absorbed folds into canonical.
    loop = await _seed_edge(kb, ns_row_id, canonical, absorbed, "RELATES_TO")

    pre_ents, pre_rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert {str(canonical), str(absorbed)} <= pre_ents
    assert str(loop) in pre_rels

    merges = [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}]

    def _dedupe_op() -> DreamOp:
        return DreamOp(
            op_id=uuid4(),
            phase="apply",
            op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
            outputs=({"merges": merges},),
            namespace_id=ns_row_id,
        )

    async with kb.storage.transaction() as txn:
        await apply_vectorcypher_dedupe_entities(_dedupe_op(), coordinator=kb.storage, session=txn.session)

    ents, rels = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert str(absorbed) not in ents
    assert str(canonical) in ents
    assert str(loop) not in rels

    # Idempotent convergence: replay the same merge - the live sets do not shift.
    async with kb.storage.transaction() as txn:
        await apply_vectorcypher_dedupe_entities(_dedupe_op(), coordinator=kb.storage, session=txn.session)
    ents2, rels2 = await _assert_live_sets_byte_identical(kb, ns_row_id)
    assert (ents2, rels2) == (ents, rels)
