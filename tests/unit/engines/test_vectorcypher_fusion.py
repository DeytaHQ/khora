"""Unit tests for VectorCypher RRF fusion utilities."""

from __future__ import annotations

import random
from types import SimpleNamespace
from uuid import uuid4

from khora.engines.vectorcypher.fusion import (
    FusedResult,
    apply_coherence_boost,
    apply_recency_boost,
    attach_relevance_scores,
    bigram_coherence_score,
    normalize_scores,
    reciprocal_rank_fusion,
    weighted_rrf,
    weighted_rrf_normalized,
)


class TestFusedResult:
    """Tests for FusedResult dataclass."""

    def test_create_fused_result(self) -> None:
        """Test creating a FusedResult with all fields."""
        item_id = uuid4()
        result = FusedResult(
            item_id=item_id,
            item={"content": "test"},
            rrf_score=0.5,
            vector_rank=1,
            graph_rank=2,
            vector_score=0.9,
            graph_score=0.7,
        )
        assert result.item_id == item_id
        assert result.rrf_score == 0.5
        assert result.vector_rank == 1
        assert result.graph_rank == 2

    def test_fused_result_defaults(self) -> None:
        """Test FusedResult default values for optional fields."""
        result = FusedResult(item_id=uuid4(), item="test", rrf_score=0.5)
        assert result.vector_rank is None
        assert result.graph_rank is None
        assert result.vector_score is None
        assert result.graph_score is None


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""

    def test_single_list(self) -> None:
        """Test RRF with a single result list."""
        id1, id2 = uuid4(), uuid4()
        results = reciprocal_rank_fusion(
            [(id1, 0.9, "item1"), (id2, 0.7, "item2")],
            k=60,
        )
        assert len(results) == 2
        assert results[0].item_id == id1
        assert results[0].rrf_score > results[1].rrf_score

    def test_two_lists_shared_item(self) -> None:
        """Test RRF with overlapping items across two lists."""
        shared_id = uuid4()
        id_a = uuid4()
        id_b = uuid4()

        vector_results = [(shared_id, 0.9, "shared"), (id_a, 0.7, "a")]
        graph_results = [(shared_id, 0.8, "shared"), (id_b, 0.6, "b")]

        fused = reciprocal_rank_fusion(vector_results, graph_results, k=60)

        # Shared item should rank highest (appears in both)
        assert fused[0].item_id == shared_id
        assert fused[0].vector_rank == 1
        assert fused[0].graph_rank == 1

    def test_empty_lists(self) -> None:
        """Test RRF with empty input lists."""
        fused = reciprocal_rank_fusion([], [], k=60)
        assert fused == []

    def test_k_parameter(self) -> None:
        """Test that k parameter affects scoring."""
        id1 = uuid4()
        # With k=1, rank 1 -> 1/(1+1) = 0.5
        fused_k1 = reciprocal_rank_fusion([(id1, 0.9, "item")], k=1)
        # With k=100, rank 1 -> 1/(100+1) ≈ 0.0099
        fused_k100 = reciprocal_rank_fusion([(id1, 0.9, "item")], k=100)

        assert fused_k1[0].rrf_score > fused_k100[0].rrf_score


class TestWeightedRrf:
    """Tests for weighted_rrf function."""

    def test_basic_fusion(self) -> None:
        """Test basic weighted RRF fusion."""
        id1 = uuid4()
        id2 = uuid4()

        vector_results = [(id1, 0.9, "v1")]
        graph_results = [(id2, 0.8, "g1")]

        fused = weighted_rrf(
            vector_results,
            graph_results,
            k=60,
            vector_weight=0.6,
            graph_weight=0.4,
        )

        assert len(fused) == 2
        # Vector item should have higher score with higher weight
        scores = {r.item_id: r.rrf_score for r in fused}
        assert scores[id1] > scores[id2]

    def test_equal_weights(self) -> None:
        """Test weighted RRF with equal weights."""
        id1, id2 = uuid4(), uuid4()
        vector_results = [(id1, 0.9, "v1")]
        graph_results = [(id2, 0.8, "g1")]

        fused = weighted_rrf(
            vector_results,
            graph_results,
            k=60,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        # Same rank, same weight -> same score
        assert len(fused) == 2
        assert abs(fused[0].rrf_score - fused[1].rrf_score) < 1e-10

    def test_graph_heavy_weights(self) -> None:
        """Test weighted RRF with graph-heavy weights."""
        id1, id2 = uuid4(), uuid4()
        vector_results = [(id1, 0.9, "v1")]
        graph_results = [(id2, 0.8, "g1")]

        fused = weighted_rrf(
            vector_results,
            graph_results,
            k=60,
            vector_weight=0.2,
            graph_weight=0.8,
        )

        # Graph item should rank higher
        assert fused[0].item_id == id2

    def test_empty_vector_results(self) -> None:
        """Test weighted RRF with no vector results."""
        id1 = uuid4()
        fused = weighted_rrf([], [(id1, 0.8, "g1")], k=60)
        assert len(fused) == 1
        assert fused[0].item_id == id1
        assert fused[0].vector_rank is None
        assert fused[0].graph_rank == 1

    def test_empty_graph_results(self) -> None:
        """Test weighted RRF with no graph results."""
        id1 = uuid4()
        fused = weighted_rrf([(id1, 0.9, "v1")], [], k=60)
        assert len(fused) == 1
        assert fused[0].graph_rank is None
        assert fused[0].vector_rank == 1

    def test_both_empty(self) -> None:
        """Test weighted RRF with both inputs empty."""
        fused = weighted_rrf([], [], k=60)
        assert fused == []

    def test_overlapping_items_boost(self) -> None:
        """Test that items in both lists get score boost."""
        shared_id = uuid4()
        only_vec = uuid4()

        vector_results = [(shared_id, 0.9, "shared"), (only_vec, 0.7, "v_only")]
        graph_results = [(shared_id, 0.8, "shared")]

        fused = weighted_rrf(vector_results, graph_results, k=60)

        scores = {r.item_id: r.rrf_score for r in fused}
        assert scores[shared_id] > scores[only_vec]


class TestNormalizeScores:
    """Tests for normalize_scores function."""

    def test_basic_normalization(self) -> None:
        """Test that scores are normalized to [0, 1]."""
        results = [
            FusedResult(item_id=uuid4(), item="a", rrf_score=0.1),
            FusedResult(item_id=uuid4(), item="b", rrf_score=0.5),
            FusedResult(item_id=uuid4(), item="c", rrf_score=0.3),
        ]

        normalized = normalize_scores(results)

        # Min score -> 0, Max score -> 1
        scores = [r.rrf_score for r in normalized]
        assert min(scores) == 0.0
        assert max(scores) == 1.0

    def test_empty_list(self) -> None:
        """Test normalization of empty list."""
        result = normalize_scores([])
        assert result == []

    def test_single_item(self) -> None:
        """Test normalization of single item list."""
        results = [FusedResult(item_id=uuid4(), item="a", rrf_score=0.5)]
        normalized = normalize_scores(results)
        assert normalized[0].rrf_score == 1.0

    def test_identical_scores(self) -> None:
        """Test normalization when all scores are equal."""
        results = [
            FusedResult(item_id=uuid4(), item="a", rrf_score=0.5),
            FusedResult(item_id=uuid4(), item="b", rrf_score=0.5),
        ]
        normalized = normalize_scores(results)
        assert all(r.rrf_score == 1.0 for r in normalized)


class TestAttachRelevanceScores:
    """Tests for attach_relevance_scores (#811).

    The reported score must be an ABSOLUTE relevance measure (raw vector cosine),
    not a per-result-set min-max rescaling that forces top=1.0 / bottom=0.0.
    """

    def test_offtopic_top_is_not_one(self) -> None:
        """An off-topic result set has all-low cosines; the top must NOT be 1.0."""
        # Simulate a sourdough query against an IT-support corpus: every chunk's
        # raw cosine sits in the noise band, but RRF still orders them.
        results = [
            FusedResult(item_id=uuid4(), item="t1", rrf_score=0.0166, vector_score=0.11),
            FusedResult(item_id=uuid4(), item="t2", rrf_score=0.0161, vector_score=0.09),
            FusedResult(item_id=uuid4(), item="t3", rrf_score=0.0159, vector_score=0.06),
        ]

        attach_relevance_scores(results)

        scores = [r.rrf_score for r in results]
        # min-max would have produced exactly 1.0 and 0.0 - the bug.
        assert scores[0] != 1.0
        assert scores[-1] != 0.0
        # Reported value is the absolute cosine, in the low band.
        assert scores == [0.11, 0.09, 0.06]
        # A relevance threshold drops every chunk on this off-topic query.
        assert all(s < 0.3 for s in scores)

    def test_scores_comparable_across_result_sets(self) -> None:
        """Two different result sets keep their absolute scales (not rescaled)."""
        offtopic = [
            FusedResult(item_id=uuid4(), item="a", rrf_score=0.016, vector_score=0.10),
            FusedResult(item_id=uuid4(), item="b", rrf_score=0.015, vector_score=0.05),
        ]
        ontopic = [
            FusedResult(item_id=uuid4(), item="c", rrf_score=0.016, vector_score=0.88),
            FusedResult(item_id=uuid4(), item="d", rrf_score=0.015, vector_score=0.71),
        ]

        attach_relevance_scores(offtopic)
        attach_relevance_scores(ontopic)

        # The on-topic top outscores the off-topic top - impossible under
        # per-result-set min-max (both would be exactly 1.0).
        assert ontopic[0].rrf_score > offtopic[0].rrf_score
        assert offtopic[0].rrf_score == 0.10
        assert ontopic[0].rrf_score == 0.88

    def test_ordering_is_preserved(self) -> None:
        """Order is decided by rrf_score upstream; this must not re-sort."""
        # Deliberately make cosine NON-monotonic with the existing order (a
        # reranker legitimately reorders vs pure cosine). The order must stay.
        ids = [uuid4() for _ in range(3)]
        results = [
            FusedResult(item_id=ids[0], item="first", rrf_score=0.9, vector_score=0.40),
            FusedResult(item_id=ids[1], item="second", rrf_score=0.5, vector_score=0.95),
            FusedResult(item_id=ids[2], item="third", rrf_score=0.1, vector_score=0.60),
        ]

        attach_relevance_scores(results)

        assert [r.item_id for r in results] == ids
        assert [r.rrf_score for r in results] == [0.40, 0.95, 0.60]

    def test_graph_only_without_embedding_reports_zero(self) -> None:
        """A graph-only chunk must NEVER report the graph channel's score (#1433).

        The graph score is a mentions-count rank input (>= 1.0, unbounded), not a
        bounded relevance value. Without an embedding to compute a cosine from,
        the display score is 0.0 - "no vector-relevance measurement".
        """
        results = [
            FusedResult(item_id=uuid4(), item="g", rrf_score=0.02, vector_score=None, graph_score=3.6),
        ]
        attach_relevance_scores(results)
        assert results[0].rrf_score == 0.0

    def test_graph_only_with_embedding_computes_cosine(self) -> None:
        """A graph-only chunk carrying an embedding reports its true cosine (#1433)."""
        parallel = SimpleNamespace(content="g1", embedding=[1.0, 0.0])
        orthogonal = SimpleNamespace(content="g2", embedding=[0.0, 1.0])
        results = [
            FusedResult(item_id=uuid4(), item=parallel, rrf_score=0.02, graph_score=5.0),
            FusedResult(item_id=uuid4(), item=orthogonal, rrf_score=0.01, graph_score=9.0),
        ]
        attach_relevance_scores(results, query_embedding=[1.0, 0.0])
        assert abs(results[0].rrf_score - 1.0) < 1e-6
        assert abs(results[1].rrf_score - 0.0) < 1e-6

    def test_raw_cosine_map_wins_over_hybrid_rrf_vector_score(self) -> None:
        """With a raw-cosine map, the display is the cosine, not the tuple score.

        On HYBRID stores the fusion tuple score (mirrored into ``vector_score``)
        is the store-internal RRF blend (~1/(60+rank) = ~0.016), not a cosine.
        The caller threads the true cosine via ``raw_cosine_by_id`` (#1433).
        """
        cid = uuid4()
        results = [
            FusedResult(item_id=cid, item="v", rrf_score=0.0164, vector_score=0.0164),
        ]
        attach_relevance_scores(results, raw_cosine_by_id={cid: 0.73})
        assert results[0].rrf_score == 0.73

    def test_negative_cosine_clamped_to_zero_on_all_branches(self) -> None:
        """A negative cosine (opposite-direction embedding) displays as 0.0 on
        the map-lookup and legacy vector_score branches, matching the
        embedding-computed branch - bounded relevance, not signed similarity."""
        cid = uuid4()
        mapped = [FusedResult(item_id=cid, item="m", rrf_score=0.02, vector_score=0.01)]
        attach_relevance_scores(mapped, raw_cosine_by_id={cid: -0.4})
        assert mapped[0].rrf_score == 0.0

        legacy = [FusedResult(item_id=uuid4(), item="l", rrf_score=0.02, vector_score=-0.2)]
        attach_relevance_scores(legacy)
        assert legacy[0].rrf_score == 0.0

    def test_mixed_vector_graph_display_bounded(self) -> None:
        """Graph-only display never exceeds a vector chunk's cosine unless its
        own computed cosine actually does; sorting by score does not move the
        top-RANKED chunk on a representative fixture (#1433).

        Pre-#1433, graph-only chunks carried mentions-scale scores (>= 1.0 after
        the engine clamp), so a caller sorting by score promoted them above
        every genuinely-matching vector chunk.
        """
        v1, v2, g1 = uuid4(), uuid4(), uuid4()
        results = [
            FusedResult(item_id=v1, item="v1", rrf_score=0.030, vector_score=0.0164),
            FusedResult(item_id=v2, item="v2", rrf_score=0.020, vector_score=0.0161),
            FusedResult(item_id=g1, item="g1", rrf_score=0.015, graph_score=3.6),
        ]
        attach_relevance_scores(results, raw_cosine_by_id={v1: 0.72, v2: 0.55})

        by_id = {r.item_id: r.rrf_score for r in results}
        # Graph-only display (no embedding -> 0.0) stays below the vector cosines.
        assert by_id[g1] == 0.0
        assert by_id[g1] <= by_id[v1]
        assert by_id[g1] <= by_id[v2]
        # A caller sorting by display score keeps the rank-1 chunk on top
        # (pre-fix the graph-only chunk's clamped 1.0 jumped the queue).
        resorted = sorted(results, key=lambda r: r.rrf_score, reverse=True)
        assert resorted[0].item_id == v1

    def test_order_parity_attach_never_reorders(self) -> None:
        """ORDER-PARITY (#1433): for a fixed fused fixture, attach changes only
        the display VALUES - the returned chunk order is byte-identical to the
        fusion-determined rank order."""
        ids = [uuid4() for _ in range(4)]
        chunks = {i: SimpleNamespace(content=f"chunk {i}") for i in ids}
        vector_results = [(ids[0], 0.0164, chunks[ids[0]]), (ids[1], 0.0161, chunks[ids[1]])]
        graph_results = [(ids[2], 4.2, chunks[ids[2]]), (ids[0], 2.0, chunks[ids[0]]), (ids[3], 1.0, chunks[ids[3]])]

        fused = weighted_rrf_normalized(vector_results, graph_results, k=60)
        order_before = [r.item_id for r in fused]

        attach_relevance_scores(
            fused,
            raw_cosine_by_id={ids[0]: 0.81, ids[1]: 0.64},
            query_embedding=[0.1, 0.2],
        )

        assert [r.item_id for r in fused] == order_before

    def test_no_signal_keeps_rrf_score(self) -> None:
        """With neither vector nor graph score, the existing score is untouched."""
        results = [
            FusedResult(item_id=uuid4(), item="x", rrf_score=0.123, vector_score=None, graph_score=None),
        ]
        attach_relevance_scores(results)
        assert results[0].rrf_score == 0.123

    def test_empty_list(self) -> None:
        """Empty input returns empty."""
        assert attach_relevance_scores([]) == []


class TestWeightedRrfNormalized:
    """Tests for weighted_rrf_normalized function."""

    def test_basic_normalized_fusion(self) -> None:
        """Test basic normalized RRF fusion."""
        id1, id2 = uuid4(), uuid4()
        vector_results = [(id1, 0.9, "v1"), (id2, 0.3, "v2")]
        graph_results = [(id2, 5.0, "g2"), (id1, 1.0, "g1")]

        fused = weighted_rrf_normalized(
            vector_results,
            graph_results,
            k=60,
            vector_weight=0.6,
            graph_weight=0.4,
        )

        assert len(fused) == 2
        assert all(r.rrf_score > 0 for r in fused)

    def test_tiebreaking_with_normalization(self) -> None:
        """Test that normalized scores provide tiebreaking for same-rank items."""
        id1, id2 = uuid4(), uuid4()

        # Both rank 1 in their respective lists, but different raw scores
        vector_results = [(id1, 0.95, "high")]
        graph_results = [(id2, 0.50, "low")]

        fused = weighted_rrf_normalized(
            vector_results,
            graph_results,
            k=60,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        # Both should have RRF component equal, but normalized score contribution
        # for id1 (normalized to 1.0 since only item) should differ from id2
        assert len(fused) == 2

    def test_empty_inputs(self) -> None:
        """Test normalized fusion with empty inputs."""
        fused = weighted_rrf_normalized([], [], k=60)
        assert fused == []

    def test_single_source_vector(self) -> None:
        """Test normalized fusion with only vector results."""
        id1, id2 = uuid4(), uuid4()
        vector_results = [(id1, 0.9, "v1"), (id2, 0.5, "v2")]

        fused = weighted_rrf_normalized(vector_results, [], k=60)

        assert len(fused) == 2
        assert fused[0].item_id == id1
        assert fused[0].graph_rank is None

    def test_single_source_graph(self) -> None:
        """Test normalized fusion with only graph results."""
        id1 = uuid4()
        fused = weighted_rrf_normalized([], [(id1, 3.0, "g1")], k=60)

        assert len(fused) == 1
        assert fused[0].vector_rank is None

    def test_score_normalization_effect(self) -> None:
        """Test that normalization balances different score distributions."""
        id1, id2, id3 = uuid4(), uuid4(), uuid4()

        # Vector: small range [0.8, 0.9]
        # Graph: large range [1.0, 100.0]
        vector_results = [(id1, 0.9, "v1"), (id2, 0.8, "v2")]
        graph_results = [(id3, 100.0, "g1"), (id1, 1.0, "g2")]

        fused = weighted_rrf_normalized(
            vector_results,
            graph_results,
            k=60,
            vector_weight=0.5,
            graph_weight=0.5,
        )

        # All three items should be present
        assert len(fused) == 3

    def test_preserves_rank_info(self) -> None:
        """Test that rank info is preserved after fusion."""
        id1 = uuid4()
        vector_results = [(id1, 0.9, "item")]
        graph_results = [(id1, 0.8, "item")]

        fused = weighted_rrf_normalized(vector_results, graph_results, k=60)

        assert fused[0].vector_rank == 1
        assert fused[0].graph_rank == 1
        assert fused[0].vector_score == 0.9
        assert fused[0].graph_score == 0.8


class TestApplyRecencyBoost:
    """Tests for apply_recency_boost function."""

    def test_basic_recency_boost(self) -> None:
        """Test that multiplicative recency boosts reorder results."""
        old_id = uuid4()
        new_id = uuid4()

        results = [
            FusedResult(item_id=old_id, item="old", rrf_score=0.9),
            FusedResult(item_id=new_id, item="new", rrf_score=0.8),
        ]

        recency_scores = {old_id: 0.1, new_id: 1.0}  # new is more recent

        boosted = apply_recency_boost(results, recency_scores, recency_weight=0.5)

        # Multiplicative: score *= max(recency, 0.5) ** (2.0 * weight)
        # old: 0.9 * max(0.1, 0.5)^(2.0*0.5) = 0.9 * 0.5^1.0 = 0.45
        # new: 0.8 * max(1.0, 0.5)^(2.0*0.5) = 0.8 * 1.0^1.0 = 0.80
        assert boosted[0].item_id == new_id

    def test_zero_recency_weight(self) -> None:
        """Test that zero weight preserves original order."""
        id1, id2 = uuid4(), uuid4()
        results = [
            FusedResult(item_id=id1, item="first", rrf_score=0.9),
            FusedResult(item_id=id2, item="second", rrf_score=0.8),
        ]

        boosted = apply_recency_boost(results, {}, recency_weight=0.0)
        assert boosted[0].item_id == id1
        # With weight=0, exponent is 0 so multiplier is 1.0
        assert abs(boosted[0].rrf_score - 0.9) < 1e-10

    def test_missing_recency_score_uses_floor(self) -> None:
        """Test items with missing recency scores get the floor (0.5)."""
        id1 = uuid4()
        results = [FusedResult(item_id=id1, item="test", rrf_score=0.9)]

        boosted = apply_recency_boost(results, {}, recency_weight=0.2)

        # Multiplicative: 0.9 * 0.5^(2.0*0.2) = 0.9 * 0.5^0.4
        expected = 0.9 * (0.5**0.4)
        assert abs(boosted[0].rrf_score - expected) < 1e-10

    def test_empty_results(self) -> None:
        """Test recency boost on empty results."""
        boosted = apply_recency_boost([], {}, recency_weight=0.2)
        assert boosted == []

    def test_recency_never_inflates_above_rrf(self) -> None:
        """Multiplicative recency should never increase a score above its original RRF."""
        id1 = uuid4()
        original_rrf = 0.5
        results = [FusedResult(item_id=id1, item="test", rrf_score=original_rrf)]
        recency_scores = {id1: 1.0}  # maximum recency

        boosted = apply_recency_boost(results, recency_scores, recency_weight=0.8)

        # max(1.0, 0.5)^(2.0*0.8) = 1.0^1.6 = 1.0, so score stays at 0.5
        assert boosted[0].rrf_score <= original_rrf + 1e-10


class TestBigramCoherenceScore:
    """Tests for bigram_coherence_score function."""

    def test_short_text_returns_one(self) -> None:
        """Text with < 6 words is too short to assess, returns 1.0."""
        assert bigram_coherence_score("hello world") == 1.0
        assert bigram_coherence_score("one two three four five") == 1.0

    def test_empty_string_returns_one(self) -> None:
        """Empty string has < 6 words, returns 1.0."""
        assert bigram_coherence_score("") == 1.0

    def test_coherent_text_scores_high(self) -> None:
        """Natural English text with proper article/preposition patterns scores high."""
        text = "The cat sat on the mat in the garden by the old house"
        score = bigram_coherence_score(text)
        assert score > 0.8

    def test_shuffled_text_scores_lower(self) -> None:
        """Shuffled text scores strictly lower than its coherent source."""
        text = "The researchers found that the results of the experiment were in the expected range and the data from the survey confirmed the hypothesis about the impact on the population"
        coherent_score = bigram_coherence_score(text)

        words = text.split()
        rng = random.Random(42)
        rng.shuffle(words)
        shuffled_score = bigram_coherence_score(" ".join(words))

        assert shuffled_score < coherent_score

    def test_no_function_words_returns_one(self) -> None:
        """Text with only content words (no articles/prepositions) has total == 0."""
        text = "quantum computing neural networks machine learning algorithms data"
        score = bigram_coherence_score(text)
        assert score == 1.0


class TestApplyCoherenceBoost:
    """Tests for apply_coherence_boost function."""

    def test_coherent_items_maintain_order(self) -> None:
        """Items with coherent content preserve their relative ranking."""
        id1, id2 = uuid4(), uuid4()
        results = [
            FusedResult(
                item_id=id1,
                item=SimpleNamespace(content="The report discusses the impact of the new policy on trade"),
                rrf_score=0.9,
            ),
            FusedResult(
                item_id=id2,
                item=SimpleNamespace(content="A study found that the results were in the expected range"),
                rrf_score=0.8,
            ),
        ]

        boosted = apply_coherence_boost(results, coherence_weight=0.1)
        assert boosted[0].item_id == id1
        assert boosted[1].item_id == id2

    def test_shuffled_item_gets_demoted(self) -> None:
        """An item with shuffled content ranks lower than one with coherent content."""
        coherent_id, shuffled_id = uuid4(), uuid4()
        coherent_text = "The report discusses the impact of the new policy on trade"
        words = coherent_text.split()
        rng = random.Random(99)
        rng.shuffle(words)
        shuffled_text = " ".join(words)

        results = [
            FusedResult(
                item_id=shuffled_id,
                item=SimpleNamespace(content=shuffled_text),
                rrf_score=0.8,
            ),
            FusedResult(
                item_id=coherent_id,
                item=SimpleNamespace(content=coherent_text),
                rrf_score=0.8,
            ),
        ]

        boosted = apply_coherence_boost(results, coherence_weight=0.3)
        assert boosted[0].item_id == coherent_id

    def test_zero_weight_is_noop(self) -> None:
        """coherence_weight=0.0 should not change scores."""
        id1, id2 = uuid4(), uuid4()
        results = [
            FusedResult(item_id=id1, item=SimpleNamespace(content="anything"), rrf_score=0.9),
            FusedResult(item_id=id2, item=SimpleNamespace(content="whatever"), rrf_score=0.8),
        ]

        boosted = apply_coherence_boost(results, coherence_weight=0.0)
        assert boosted[0].rrf_score == 0.9
        assert boosted[1].rrf_score == 0.8

    def test_items_without_content_attribute(self) -> None:
        """Items lacking a .content attribute get coherence 1.0 (treated as neutral)."""
        id1 = uuid4()
        results = [
            FusedResult(item_id=id1, item={"no_content": True}, rrf_score=0.9),
        ]

        boosted = apply_coherence_boost(results, coherence_weight=0.2)
        # coherence=1.0 for empty content (< 6 words): (1-0.2)*0.9 + 0.2*1.0 = 0.92
        assert abs(boosted[0].rrf_score - 0.92) < 1e-10

    def test_coherence_does_not_override_relevance_at_rrf_scale_1056(self) -> None:
        """Regression for #1056: at raw weighted-RRF scale, a default
        coherence_weight=0.1 must NOT demote the relevance winner.

        The retriever applies coherence to raw RRF scores (~0.02 at top, k=60).
        Blending those against coherence in [0, 1] lets the nominal-10% term
        dominate, demoting a more-relevant-but-less-fluent chunk below a filler.
        The fix normalizes fused scores to [0, 1] BEFORE the coherence blend
        (the order the retriever now uses). This test pins both halves: the raw
        order is corrupted, the normalized order is not.
        """
        winner_id, filler_id = uuid4(), uuid4()
        # Less-fluent but clearly more relevant (rank-1 in both channels).
        winner_text = "revenue in of the company and the and the field team"  # coherence 0.6
        # Fully "coherent" filler that loses on relevance.
        filler_text = "The report discusses the impact of the new policy on trade"  # coherence 1.0

        def _fused() -> list[FusedResult]:
            return [
                FusedResult(item_id=winner_id, item=SimpleNamespace(content=winner_text), rrf_score=0.02639),
                FusedResult(item_id=filler_id, item=SimpleNamespace(content=filler_text), rrf_score=0.00909),
            ]

        # Buggy order (coherence on RAW rrf scores): the winner is demoted.
        raw = apply_coherence_boost(_fused(), coherence_weight=0.1)
        raw.sort(key=lambda r: r.rrf_score, reverse=True)
        assert raw[0].item_id == filler_id, "expected the raw-scale bug to demote the winner"

        # Fixed order (normalize to [0,1] BEFORE the blend): winner stays on top.
        fixed = apply_coherence_boost(normalize_scores(_fused()), coherence_weight=0.1)
        fixed.sort(key=lambda r: r.rrf_score, reverse=True)
        assert fixed[0].item_id == winner_id, "normalize-before-blend must keep the relevance winner on top"
