"""Unit tests for the Personalized PageRank retrieval helpers (#542).

Covers the pure functions in ``khora.engines.vectorcypher.ppr_retrieval``
plus the async ``ppr_retrieve_chunks`` orchestrator driven by a mocked
``StorageCoordinator``. Live Neo4j / pgvector isn't required: PPR runs
on the entity graph via ``coordinator.list_entities`` /
``list_relationships`` and scores chunks via ``Entity.source_chunk_ids``
+ ``get_chunks_batch``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk, ChunkMetadata, Entity, Relationship
from khora.engines.vectorcypher.ppr_retrieval import (
    build_personalization_vector,
    build_ppr_graph,
    ppr_retrieve_chunks,
    score_chunks_via_ppr,
)


def _make_entity(name: str, chunk_ids: list[UUID]) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=uuid4(),
        name=name,
        source_chunk_ids=list(chunk_ids),
    )


def _make_chunk(content: str) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        metadata=ChunkMetadata(),
    )


# ---------------------------------------------------------------------------
# Personalization vector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPersonalizationVector:
    def test_seeds_present_entities(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        idx = {a: 0, b: 1, c: 2}
        vec = build_personalization_vector([(a, 0.9), (c, 0.5)], idx)
        assert vec == pytest.approx([0.9, 0.0, 0.5])

    def test_drops_unknown_entities(self) -> None:
        a, b = uuid4(), uuid4()
        rogue = uuid4()  # not in idx
        idx = {a: 0, b: 1}
        vec = build_personalization_vector([(a, 0.7), (rogue, 9.9)], idx)
        assert vec == pytest.approx([0.7, 0.0])

    def test_returns_all_zeros_when_no_overlap(self) -> None:
        """No surviving seeds — caller's signal to fall back to vector-only."""
        a = uuid4()
        idx = {a: 0, uuid4(): 1}
        rogue = uuid4()
        vec = build_personalization_vector([(rogue, 0.5)], idx)
        assert sum(vec) == 0.0

    def test_negative_scores_clamped_to_small_positive(self) -> None:
        """Bogus negative scores should not crash or zero out the seed."""
        a = uuid4()
        idx = {a: 0}
        vec = build_personalization_vector([(a, -1.0)], idx)
        assert vec[0] > 0.0


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPPRGraph:
    def test_edges_are_bidirectional(self) -> None:
        e1, e2, e3 = _make_entity("A", []), _make_entity("B", []), _make_entity("C", [])
        edges = [(e1.id, e2.id, 1.0), (e2.id, e3.id, 2.0)]
        graph = build_ppr_graph([e1, e2, e3], edges)
        # 2 input edges → 4 directed edges in the dense form
        assert len(graph.edges) == 4

    def test_drops_self_loops_and_dangling_endpoints(self) -> None:
        e1, e2 = _make_entity("A", []), _make_entity("B", [])
        rogue = uuid4()
        edges = [
            (e1.id, e1.id, 1.0),  # self-loop
            (e1.id, rogue, 1.0),  # dangling
            (e1.id, e2.id, 1.0),  # valid
        ]
        graph = build_ppr_graph([e1, e2], edges)
        # Only the valid edge survives → 2 directed entries
        assert len(graph.edges) == 2


# ---------------------------------------------------------------------------
# Chunk scoring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoreChunksViaPPR:
    def test_chunks_ranked_by_summed_pr_mass(self) -> None:
        cid_a, cid_b, cid_c = uuid4(), uuid4(), uuid4()
        e1 = _make_entity("A", [cid_a, cid_b])
        e2 = _make_entity("B", [cid_a])
        e3 = _make_entity("C", [cid_c])
        pr = [0.6, 0.3, 0.1]
        ranked = score_chunks_via_ppr(pr, [e1, e2, e3], top_entities=3)
        # cid_a: 0.6 + 0.3 = 0.9 ; cid_b: 0.6 ; cid_c: 0.1
        assert ranked[0][0] == cid_a
        assert ranked[1][0] == cid_b
        assert ranked[2][0] == cid_c
        assert ranked[0][1] == pytest.approx(0.9)

    def test_top_entities_cap_is_respected(self) -> None:
        chunks = [uuid4() for _ in range(5)]
        entities = [_make_entity(f"E{i}", [chunks[i]]) for i in range(5)]
        pr = [0.4, 0.3, 0.2, 0.05, 0.05]
        ranked = score_chunks_via_ppr(pr, entities, top_entities=2)
        # Only the top-2 entities contribute → 2 chunks scored.
        assert len(ranked) == 2
        assert {cid for cid, _ in ranked} == {chunks[0], chunks[1]}

    def test_returns_empty_on_degenerate_input(self) -> None:
        assert score_chunks_via_ppr([], [], top_entities=5) == []

    def test_similarity_blend_boosts_but_preserves_order(self) -> None:
        cid_a, cid_b = uuid4(), uuid4()
        e1 = _make_entity("A", [cid_a])
        e2 = _make_entity("B", [cid_b])
        pr = [0.8, 0.2]
        sim = {cid_a: 0.5, cid_b: 0.9}
        ranked = score_chunks_via_ppr(pr, [e1, e2], top_entities=2, chunk_similarity=sim)
        # PPR mass still dominates: cid_a (0.8 * 1.5 = 1.2) > cid_b (0.2 * 1.9 = 0.38)
        assert ranked[0][0] == cid_a


# ---------------------------------------------------------------------------
# Async orchestrator (mocked storage)
# ---------------------------------------------------------------------------


def _mock_storage(
    *,
    entities: list[Entity],
    relationships: list[Relationship],
    chunks_map: dict[UUID, Chunk],
) -> MagicMock:
    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=entities)
    storage.list_relationships = AsyncMock(return_value=relationships)
    storage.get_chunks_batch = AsyncMock(return_value=chunks_map)
    return storage


def _make_relationship(src: UUID, tgt: UUID, weight: float = 1.0) -> Relationship:
    return Relationship(
        id=uuid4(),
        namespace_id=uuid4(),
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type="RELATES_TO",
        weight=weight,
    )


@pytest.mark.unit
class TestPPRRetrieveChunks:
    @pytest.mark.asyncio
    async def test_returns_chunks_scored_by_ppr(self) -> None:
        ns = uuid4()
        chunk_a, chunk_b = _make_chunk("Alice met Bob"), _make_chunk("Bob met Carol")
        e_alice = _make_entity("Alice", [chunk_a.id])
        e_bob = _make_entity("Bob", [chunk_a.id, chunk_b.id])
        e_carol = _make_entity("Carol", [chunk_b.id])

        rels = [
            _make_relationship(e_alice.id, e_bob.id),
            _make_relationship(e_bob.id, e_carol.id),
        ]
        chunks_map = {chunk_a.id: chunk_a, chunk_b.id: chunk_b}
        storage = _mock_storage(
            entities=[e_alice, e_bob, e_carol],
            relationships=rels,
            chunks_map=chunks_map,
        )

        results, entity_scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(e_alice.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        # Alice is the seed → her PR is highest, so chunk_a (which she
        # mentions) outranks chunk_b in the top result.
        assert results
        assert results[0][0] == chunk_a.id
        assert set(entity_scores.keys()) == {e_alice.id, e_bob.id, e_carol.id}
        # PR mass is a probability distribution → sums to ~1.
        assert sum(entity_scores.values()) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_empty_entry_entities_returns_fallback_sentinel(self) -> None:
        ns = uuid4()
        storage = _mock_storage(entities=[], relationships=[], chunks_map={})
        results, scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[],
            damping=0.85,
            max_iter=10,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        assert results == []
        assert scores == {}
        # Short-circuit: list_entities not called because entry_entities empty.
        storage.list_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_entity_graph_falls_back(self) -> None:
        ns = uuid4()
        storage = _mock_storage(entities=[], relationships=[], chunks_map={})
        results, scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(uuid4(), 1.0)],
            damping=0.85,
            max_iter=10,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        assert results == []
        assert scores == {}

    @pytest.mark.asyncio
    async def test_no_seed_overlap_falls_back(self) -> None:
        """Entry entities all unknown to the namespace graph → fallback."""
        ns = uuid4()
        e1 = _make_entity("A", [])
        storage = _mock_storage(entities=[e1], relationships=[], chunks_map={})
        results, scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(uuid4(), 1.0)],  # not in entity list
            damping=0.85,
            max_iter=10,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        assert results == []
        assert scores == {}

    @pytest.mark.asyncio
    async def test_top_k_limit_respected(self) -> None:
        """Returned chunks never exceed ``limit`` even with many candidates."""
        ns = uuid4()
        chunks = [_make_chunk(f"c{i}") for i in range(8)]
        # 8 separate entities → 8 candidate chunks.
        entities = [_make_entity(f"E{i}", [chunks[i].id]) for i in range(8)]
        rels = [_make_relationship(entities[i].id, entities[i + 1].id) for i in range(7)]
        chunks_map = {c.id: c for c in chunks}
        storage = _mock_storage(entities=entities, relationships=rels, chunks_map=chunks_map)
        results, _ = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(entities[0].id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=3,
        )
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_deterministic_for_fixed_seed(self) -> None:
        """Two runs with identical inputs must produce identical rankings."""
        ns = uuid4()
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        entities = [_make_entity(f"E{i}", [chunks[i].id]) for i in range(5)]
        rels = [
            _make_relationship(entities[0].id, entities[1].id),
            _make_relationship(entities[1].id, entities[2].id),
            _make_relationship(entities[2].id, entities[3].id),
            _make_relationship(entities[3].id, entities[4].id),
        ]
        chunks_map = {c.id: c for c in chunks}
        storage1 = _mock_storage(entities=entities, relationships=rels, chunks_map=chunks_map)
        storage2 = _mock_storage(entities=entities, relationships=rels, chunks_map=chunks_map)

        results1, scores1 = await ppr_retrieve_chunks(
            storage=storage1,
            namespace_id=ns,
            entry_entities=[(entities[0].id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        results2, scores2 = await ppr_retrieve_chunks(
            storage=storage2,
            namespace_id=ns,
            entry_entities=[(entities[0].id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        assert [cid for cid, _, _ in results1] == [cid for cid, _, _ in results2]
        for eid in scores1:
            assert scores1[eid] == pytest.approx(scores2[eid])
