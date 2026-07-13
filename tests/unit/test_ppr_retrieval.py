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

from khora.core.diagnostics import Degradation
from khora.core.models import Chunk, Entity, Relationship
from khora.engines.vectorcypher import ppr_retrieval
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


# ---------------------------------------------------------------------------
# #1373 — seed-anchored augmentation above the slice cap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeedAnchoredAugmentation:
    @pytest.mark.asyncio
    async def test_seeds_survive_when_namespace_exceeds_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The #1373 invariant: when the global slice hits its cap and excludes
        the query seeds, augmentation re-includes them so personalization is
        non-zero and the graph channel returns chunks."""
        # Tiny cap so the test is fast: 3 filler entities fill the entity slice.
        monkeypatch.setattr(ppr_retrieval, "_MAX_ENTITIES_FOR_PPR", 3)
        ns = uuid4()

        # The slice is cap-many fillers — none of them are the seed.
        fillers = [_make_entity(f"aaa_filler_{i}", []) for i in range(3)]

        chunk = _make_chunk("Marie Curie discovered radium")
        seed = _make_entity("marie curie", [chunk.id])
        neighbor = _make_entity("radium", [chunk.id])
        seed_rel = _make_relationship(seed.id, neighbor.id)

        storage = _mock_storage(entities=fillers, relationships=[], chunks_map={chunk.id: chunk})
        # Augmentation reaches for the seeds (pgvector fallback present) and the
        # 1-hop neighborhood (graph backend).
        storage.get_entities_batch = AsyncMock(return_value={seed.id: seed, neighbor.id: neighbor})
        storage.get_entity_relationships = AsyncMock(return_value=[seed_rel])

        degradations: list[Degradation] = []
        results, entity_scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(seed.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
            out_degradations=degradations,
        )

        # The seed survived into the graph → it carries PR mass → its chunk is
        # returned. No degradation on the happy (augmented) path.
        assert seed.id in entity_scores
        assert entity_scores[seed.id] > 0.0
        assert results
        assert results[0][0] == chunk.id
        assert degradations == []
        storage.get_entities_batch.assert_awaited_once()
        # One gather call per unique seed id.
        storage.get_entity_relationships.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_below_cap_does_not_augment(self) -> None:
        """Below the cap behavior is byte-identical to pre-#1373: no extra
        round-trips (get_entities_batch / get_entity_relationships untouched)."""
        ns = uuid4()
        chunk = _make_chunk("Alice met Bob")
        e_alice = _make_entity("Alice", [chunk.id])
        e_bob = _make_entity("Bob", [chunk.id])
        rels = [_make_relationship(e_alice.id, e_bob.id)]
        storage = _mock_storage(
            entities=[e_alice, e_bob],
            relationships=rels,
            chunks_map={chunk.id: chunk},
        )
        # Stub the augmentation primitives so an accidental call is detectable.
        storage.get_entities_batch = AsyncMock(return_value={})
        storage.get_entity_relationships = AsyncMock(return_value=[])

        results, _ = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(e_alice.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        assert results
        storage.get_entities_batch.assert_not_called()
        storage.get_entity_relationships.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_seed_overlap_records_degradation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A genuine degenerate case above the cap (seeds vanished between
        vector resolution and the graph read) records an ADR-001 Degradation."""
        monkeypatch.setattr(ppr_retrieval, "_MAX_ENTITIES_FOR_PPR", 2)
        ns = uuid4()
        fillers = [_make_entity(f"f{i}", []) for i in range(2)]
        storage = _mock_storage(entities=fillers, relationships=[], chunks_map={})
        # Seeds no longer resolvable: batch returns nothing, neighborhood empty.
        storage.get_entities_batch = AsyncMock(return_value={})
        storage.get_entity_relationships = AsyncMock(return_value=[])

        degradations: list[Degradation] = []
        results, scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(uuid4(), 1.0)],  # not in the slice and not resolvable
            damping=0.85,
            max_iter=10,
            tol=1e-5,
            top_entities=10,
            limit=10,
            out_degradations=degradations,
        )
        assert results == []
        assert scores == {}
        assert len(degradations) == 1
        assert degradations[0]["component"] == "vectorcypher.ppr"
        assert degradations[0]["reason"] == "no_seed_overlap"

    @pytest.mark.asyncio
    async def test_empty_graph_channel_records_degradation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PPR runs but no scored entity carries a source chunk (graph-less +
        isolated seed) → empty_graph_channel degradation, not a silent drop."""
        monkeypatch.setattr(ppr_retrieval, "_MAX_ENTITIES_FOR_PPR", 2)
        ns = uuid4()
        fillers = [_make_entity(f"f{i}", []) for i in range(2)]
        # Seed resolves but has no source_chunk_ids and no neighborhood edges.
        seed = _make_entity("isolated_seed", [])
        storage = _mock_storage(entities=fillers, relationships=[], chunks_map={})
        storage.get_entities_batch = AsyncMock(return_value={seed.id: seed})
        storage.get_entity_relationships = AsyncMock(return_value=[])

        degradations: list[Degradation] = []
        results, scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(seed.id, 1.0)],
            damping=0.85,
            max_iter=10,
            tol=1e-5,
            top_entities=10,
            limit=10,
            out_degradations=degradations,
        )
        # Seed got PR mass (personalization non-zero) but no chunk to score.
        assert seed.id in scores
        assert results == []
        assert len(degradations) == 1
        assert degradations[0]["reason"] == "empty_graph_channel"

    @pytest.mark.asyncio
    async def test_seed_neighborhood_fetch_error_degrades_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A transient get_entity_relationships error for a seed must not abort
        recall — the augmentation skips that seed's edges and PPR still runs."""
        monkeypatch.setattr(ppr_retrieval, "_MAX_ENTITIES_FOR_PPR", 2)
        ns = uuid4()
        fillers = [_make_entity(f"f{i}", []) for i in range(2)]
        chunk = _make_chunk("Marie Curie discovered radium")
        seed = _make_entity("marie curie", [chunk.id])
        storage = _mock_storage(entities=fillers, relationships=[], chunks_map={chunk.id: chunk})
        storage.get_entities_batch = AsyncMock(return_value={seed.id: seed})
        # The seed's neighborhood fetch raises — must be swallowed per-seed.
        storage.get_entity_relationships = AsyncMock(side_effect=RuntimeError("transient neo4j"))

        results, entity_scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(seed.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        # No crash; the seed still survives (batch fetch) and carries its chunk.
        assert seed.id in entity_scores
        assert results
        assert results[0][0] == chunk.id

    @pytest.mark.asyncio
    async def test_augmentation_never_shrinks_base_slice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_neighborhood_entities below the slice size must NOT trim the base
        slice — the effective bound is max(bound, len(slice)) (#1378 review)."""
        monkeypatch.setattr(ppr_retrieval, "_MAX_ENTITIES_FOR_PPR", 5)
        ns = uuid4()
        chunk = _make_chunk("seed chunk")
        # 5 filler slice entities (hits the cap) + 1 seed pulled in via batch.
        fillers = [_make_entity(f"aaa_{i}", []) for i in range(5)]
        seed = _make_entity("zzz_seed", [chunk.id])
        storage = _mock_storage(entities=fillers, relationships=[], chunks_map={chunk.id: chunk})
        storage.get_entities_batch = AsyncMock(return_value={seed.id: seed})
        storage.get_entity_relationships = AsyncMock(return_value=[])

        results, entity_scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(seed.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
            # Bound below the slice size — must be clamped up to len(slice)=5.
            max_neighborhood_entities=2,
        )
        # Effective bound = max(2, len(slice)=5) = 5, NOT 2: the base slice is
        # never shrunk by augmentation. Seeds are kept first, so the seed always
        # survives and 5 entities carry PR mass (vs. 2 under the buggy trim).
        assert len(entity_scores) == 5
        assert seed.id in entity_scores
        assert results
        assert results[0][0] == chunk.id

    @pytest.mark.asyncio
    async def test_chunk_hydration_empty_records_degradation(self) -> None:
        """PPR scores chunk ids but get_chunks_batch returns nothing (chunk
        store / entity-graph divergence, the #1372 symptom) → degradation."""
        ns = uuid4()
        chunk = _make_chunk("Alice met Bob")
        e_alice = _make_entity("Alice", [chunk.id])
        e_bob = _make_entity("Bob", [chunk.id])
        rels = [_make_relationship(e_alice.id, e_bob.id)]
        # get_chunks_batch returns {} despite chunk_ids being non-empty.
        storage = _mock_storage(entities=[e_alice, e_bob], relationships=rels, chunks_map={})

        degradations: list[Degradation] = []
        results, scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(e_alice.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
            out_degradations=degradations,
        )
        assert results == []
        # Entity scores still returned (PPR ran), but the hydration was empty.
        assert e_alice.id in scores
        assert len(degradations) == 1
        assert degradations[0]["reason"] == "chunk_hydration_empty"


# ---------------------------------------------------------------------------
# #1476 — base-graph-slice cache keyed on the namespace write-epoch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPPRGraphSliceCache:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        ppr_retrieval._clear_graph_slice_cache()
        yield
        ppr_retrieval._clear_graph_slice_cache()

    def _fixture(self):
        chunk = _make_chunk("Alice met Bob")
        e_alice = _make_entity("Alice", [chunk.id])
        e_bob = _make_entity("Bob", [chunk.id])
        rels = [_make_relationship(e_alice.id, e_bob.id)]
        return e_alice, e_bob, rels, {chunk.id: chunk}

    async def _run(self, storage, ns, seed_id, epoch):
        return await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(seed_id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
            graph_cache_epoch=epoch,
        )

    @pytest.mark.asyncio
    async def test_same_epoch_hits_cache_skips_second_db_fetch(self) -> None:
        ns = uuid4()
        e_alice, e_bob, rels, chunks_map = self._fixture()
        storage = _mock_storage(entities=[e_alice, e_bob], relationships=rels, chunks_map=chunks_map)

        r1, s1 = await self._run(storage, ns, e_alice.id, epoch=1)
        r2, s2 = await self._run(storage, ns, e_alice.id, epoch=1)

        # Second call is a cache hit: the base slice is not re-fetched.
        assert storage.list_entities.await_count == 1
        assert storage.list_relationships.await_count == 1
        # Identical results from cache.
        assert [cid for cid, _, _ in r1] == [cid for cid, _, _ in r2]
        assert s1.keys() == s2.keys()

    @pytest.mark.asyncio
    async def test_new_epoch_invalidates_and_refetches(self) -> None:
        ns = uuid4()
        e_alice, e_bob, rels, chunks_map = self._fixture()
        storage = _mock_storage(entities=[e_alice, e_bob], relationships=rels, chunks_map=chunks_map)

        await self._run(storage, ns, e_alice.id, epoch=1)
        await self._run(storage, ns, e_alice.id, epoch=2)  # write bumped the epoch

        # A new epoch is a distinct key → both calls re-fetch the slice.
        assert storage.list_entities.await_count == 2
        assert storage.list_relationships.await_count == 2

    @pytest.mark.asyncio
    async def test_none_epoch_disables_cache(self) -> None:
        ns = uuid4()
        e_alice, e_bob, rels, chunks_map = self._fixture()
        storage = _mock_storage(entities=[e_alice, e_bob], relationships=rels, chunks_map=chunks_map)

        await self._run(storage, ns, e_alice.id, epoch=None)
        await self._run(storage, ns, e_alice.id, epoch=None)

        # No epoch → caching off → every recall re-fetches (legacy behaviour).
        assert storage.list_entities.await_count == 2
        assert storage.list_relationships.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_slice_is_not_cached(self) -> None:
        ns = uuid4()
        storage = _mock_storage(entities=[], relationships=[], chunks_map={})

        # Empty entity graph → early fallback, nothing cached.
        await self._run(storage, ns, uuid4(), epoch=1)
        await self._run(storage, ns, uuid4(), epoch=1)

        # Both calls hit the DB (an empty slice is never stored).
        assert storage.list_entities.await_count == 2

    @pytest.mark.asyncio
    async def test_lru_evicts_oldest_epoch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ppr_retrieval, "_GRAPH_SLICE_CACHE_MAX", 2)
        ns = uuid4()
        e_alice, e_bob, rels, chunks_map = self._fixture()
        storage = _mock_storage(entities=[e_alice, e_bob], relationships=rels, chunks_map=chunks_map)

        # Fill the cache past its cap: epochs 1, 2, 3 → epoch 1 is evicted.
        await self._run(storage, ns, e_alice.id, epoch=1)
        await self._run(storage, ns, e_alice.id, epoch=2)
        await self._run(storage, ns, e_alice.id, epoch=3)
        base = storage.list_entities.await_count  # 3 misses so far
        # epoch 1 was evicted → re-fetch; epoch 3 is still cached → hit.
        await self._run(storage, ns, e_alice.id, epoch=1)
        await self._run(storage, ns, e_alice.id, epoch=3)
        assert storage.list_entities.await_count == base + 1


# ---------------------------------------------------------------------------
# #1476 — HippoRAG-2 recognition-filtered seeding (quality experiment)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecognitionFilterSeeds:
    def test_drops_seed_with_no_query_relevant_chunk(self) -> None:
        c_rel, c_irrel = uuid4(), uuid4()
        e_relevant = _make_entity("relevant", [c_rel])
        e_noise = _make_entity("noise", [c_irrel])
        entities_by_id = {e_relevant.id: e_relevant, e_noise.id: e_noise}
        chunk_similarity = {c_rel: 0.8, c_irrel: 0.05}

        out = ppr_retrieval.recognition_filter_seeds(
            [(e_relevant.id, 0.9), (e_noise.id, 0.9)],
            entities_by_id,
            chunk_similarity,
            min_recognition=0.3,
        )
        assert out == [(e_relevant.id, 0.9)]

    def test_never_zeros_seeds(self) -> None:
        c = uuid4()
        e = _make_entity("e", [c])
        # Below threshold → would drop everything → falls back to unfiltered.
        out = ppr_retrieval.recognition_filter_seeds(
            [(e.id, 0.9)],
            {e.id: e},
            {c: 0.01},
            min_recognition=0.5,
        )
        assert out == [(e.id, 0.9)]

    def test_no_chunk_similarity_is_noop(self) -> None:
        e = _make_entity("e", [uuid4()])
        seeds = [(e.id, 0.9)]
        out = ppr_retrieval.recognition_filter_seeds(seeds, {e.id: e}, {}, min_recognition=0.3)
        assert out == seeds

    def test_unknown_entity_is_kept(self) -> None:
        unknown = uuid4()
        out = ppr_retrieval.recognition_filter_seeds(
            [(unknown, 0.9)],
            {},  # not in the graph slice
            {uuid4(): 0.9},
            min_recognition=0.3,
        )
        assert out == [(unknown, 0.9)]

    @pytest.mark.asyncio
    async def test_end_to_end_filters_noise_seed(self) -> None:
        ns = uuid4()
        c_rel = _make_chunk("query relevant evidence")
        c_noise = _make_chunk("unrelated")
        e_rel = _make_entity("relevant", [c_rel.id])
        e_noise = _make_entity("noise", [c_noise.id])
        rels = [_make_relationship(e_rel.id, e_noise.id)]
        storage = _mock_storage(
            entities=[e_rel, e_noise],
            relationships=rels,
            chunks_map={c_rel.id: c_rel, c_noise.id: c_noise},
        )

        # Both seeds resolve by name-cosine, but only e_rel has a query-relevant
        # chunk. Recognition filtering drops e_noise from the seed set.
        results, scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(e_rel.id, 0.9), (e_noise.id, 0.9)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
            chunk_similarity={c_rel.id: 0.8, c_noise.id: 0.02},
            recognition_filter=True,
            recognition_min_similarity=0.3,
        )
        assert results
        # e_rel was the only seed → it carries the most PR mass.
        assert scores[e_rel.id] > scores[e_noise.id]
