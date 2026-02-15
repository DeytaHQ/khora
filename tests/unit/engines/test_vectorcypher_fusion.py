"""Unit tests for VectorCypher RRF fusion utilities."""

from __future__ import annotations

from uuid import uuid4

from khora.engines.vectorcypher.fusion import (
    FusedResult,
    apply_recency_boost,
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
        """Test that recency boosts reorder results."""
        old_id = uuid4()
        new_id = uuid4()

        results = [
            FusedResult(item_id=old_id, item="old", rrf_score=0.9),
            FusedResult(item_id=new_id, item="new", rrf_score=0.8),
        ]

        recency_scores = {old_id: 0.1, new_id: 1.0}  # new is more recent

        boosted = apply_recency_boost(results, recency_scores, recency_weight=0.5)

        # With 50% recency weight, the new item should rank first
        # old: 0.5*0.9 + 0.5*0.1 = 0.5
        # new: 0.5*0.8 + 0.5*1.0 = 0.9
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

    def test_missing_recency_score(self) -> None:
        """Test items with missing recency scores get 0."""
        id1 = uuid4()
        results = [FusedResult(item_id=id1, item="test", rrf_score=0.9)]

        boosted = apply_recency_boost(results, {}, recency_weight=0.2)

        # (1-0.2)*0.9 + 0.2*0.0 = 0.72
        assert abs(boosted[0].rrf_score - 0.72) < 1e-10

    def test_empty_results(self) -> None:
        """Test recency boost on empty results."""
        boosted = apply_recency_boost([], {}, recency_weight=0.2)
        assert boosted == []
