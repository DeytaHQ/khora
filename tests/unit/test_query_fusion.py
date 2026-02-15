"""Unit tests for query/fusion.py — Reciprocal Rank Fusion."""

from __future__ import annotations

from khora.query.fusion import RankedItem, combine_with_weights, reciprocal_rank_fusion


class TestRankedItem:
    """Tests for the RankedItem dataclass."""

    def test_create(self) -> None:
        """Test basic creation with defaults."""
        item = RankedItem(item="doc1", score=0.9, source="vector")
        assert item.item == "doc1"
        assert item.score == 0.9
        assert item.source == "vector"
        assert item.rank == 0

    def test_custom_rank(self) -> None:
        """Test creation with custom rank."""
        item = RankedItem(item="doc1", score=0.5, source="graph", rank=3)
        assert item.rank == 3


class TestReciprocalRankFusion:
    """Tests for the reciprocal_rank_fusion function."""

    def test_empty_input(self) -> None:
        """Empty ranked_lists returns empty result."""
        assert reciprocal_rank_fusion({}) == []

    def test_all_empty_lists(self) -> None:
        """All sources provide empty lists."""
        result = reciprocal_rank_fusion({"vector": [], "graph": []})
        assert result == []

    def test_single_source(self) -> None:
        """Single source preserves ranking order."""
        ranked = {"vector": [("a", 0.9), ("b", 0.7), ("c", 0.5)]}
        result = reciprocal_rank_fusion(ranked)
        items = [item for item, _ in result]
        assert items == ["a", "b", "c"]

    def test_multiple_sources_no_overlap(self) -> None:
        """Multiple sources with no overlapping items."""
        ranked = {
            "vector": [("a", 0.9), ("b", 0.7)],
            "graph": [("c", 0.8), ("d", 0.6)],
        }
        result = reciprocal_rank_fusion(ranked)
        items = {item for item, _ in result}
        assert items == {"a", "b", "c", "d"}

    def test_overlap_dedup(self) -> None:
        """Overlapping items accumulate RRF scores and appear once."""
        ranked = {
            "vector": [("a", 0.9), ("b", 0.7)],
            "graph": [("a", 0.8), ("c", 0.6)],
        }
        result = reciprocal_rank_fusion(ranked)
        items = [item for item, _ in result]
        assert items.count("a") == 1
        # "a" appears in both sources, so it should rank highest
        assert items[0] == "a"

    def test_custom_k(self) -> None:
        """Custom k parameter changes score distribution."""
        ranked = {"v": [("a", 1.0), ("b", 0.5)]}
        result_k10 = reciprocal_rank_fusion(ranked, k=10)
        result_k100 = reciprocal_rank_fusion(ranked, k=100)
        # With higher k, scores are more evenly distributed
        score_diff_k10 = result_k10[0][1] - result_k10[1][1]
        score_diff_k100 = result_k100[0][1] - result_k100[1][1]
        assert score_diff_k10 > score_diff_k100

    def test_weights(self) -> None:
        """Custom weights affect source contribution."""
        ranked = {
            "vector": [("a", 0.9)],
            "graph": [("b", 0.9)],
        }
        # Give all weight to graph
        result = reciprocal_rank_fusion(ranked, weights={"vector": 0.0, "graph": 1.0})
        items = [item for item, _ in result]
        # "b" from graph should rank higher
        assert items[0] == "b"

    def test_zero_weights_fallback(self) -> None:
        """All-zero weights fall back to equal weights."""
        ranked = {"v": [("a", 0.9)], "g": [("b", 0.8)]}
        result = reciprocal_rank_fusion(ranked, weights={"v": 0.0, "g": 0.0})
        assert len(result) == 2

    def test_weight_normalization(self) -> None:
        """Weights are normalized so absolute values don't matter."""
        ranked = {
            "v": [("a", 0.9), ("b", 0.7)],
            "g": [("c", 0.8)],
        }
        result1 = reciprocal_rank_fusion(ranked, weights={"v": 1.0, "g": 1.0})
        result2 = reciprocal_rank_fusion(ranked, weights={"v": 100.0, "g": 100.0})
        # Same relative weights → same ranking
        items1 = [item for item, _ in result1]
        items2 = [item for item, _ in result2]
        assert items1 == items2

    def test_id_extractor(self) -> None:
        """Custom id_extractor for dedup with complex items."""
        ranked = {
            "v": [({"id": 1, "text": "hello"}, 0.9)],
            "g": [({"id": 1, "text": "hello"}, 0.8)],
        }
        result = reciprocal_rank_fusion(ranked, id_extractor=lambda x: x["id"])
        assert len(result) == 1

    def test_ranking_order_descending(self) -> None:
        """Results are sorted by score descending."""
        ranked = {"v": [("low", 0.1), ("mid", 0.5), ("high", 0.9)]}
        result = reciprocal_rank_fusion(ranked)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_formula_correctness(self) -> None:
        """Verify RRF score includes rank-based + normalized score contribution."""
        ranked = {"v": [("a", 0.9)]}
        k = 60
        result = reciprocal_rank_fusion(ranked, k=k)
        # Single source, weight=1.0 (normalized), rank=1
        # weighted_rrf_normalized adds a small normalized score contribution
        # for tiebreaking: weight * norm_score * 0.01
        # Single item normalizes to 1.0, so contribution = 1.0 * 1.0 * 0.01
        rrf_component = 1.0 / (k + 1)
        norm_contribution = 1.0 * 1.0 * 0.01
        expected_score = rrf_component + norm_contribution
        assert abs(result[0][1] - expected_score) < 1e-10


class TestCombineWithWeights:
    """Tests for the combine_with_weights function."""

    def test_empty_results(self) -> None:
        """Empty input returns empty output."""
        result = combine_with_weights([], [])
        assert result == []

    def test_single_list(self) -> None:
        """Single result list preserves ranking."""
        results = [[("a", 0.9), ("b", 0.5)]]
        result = combine_with_weights(results, [1.0])
        items = [item for item, _ in result]
        assert items == ["a", "b"]

    def test_overlap_accumulation(self) -> None:
        """Overlapping items accumulate weighted scores."""
        results = [
            [("a", 0.8), ("b", 0.4)],
            [("a", 0.6), ("c", 0.9)],
        ]
        result = combine_with_weights(results, [0.5, 0.5])
        items = [item for item, _ in result]
        # "a" has score from both lists
        assert "a" in items

    def test_zero_weights_fallback(self) -> None:
        """All-zero weights fall back to equal weights."""
        results = [[("a", 0.9)]]
        result = combine_with_weights(results, [0.0])
        assert len(result) == 1

    def test_id_extractor(self) -> None:
        """Custom id_extractor for complex items."""
        results = [
            [({"id": 1}, 0.9)],
            [({"id": 1}, 0.7)],
        ]
        result = combine_with_weights(results, [1.0, 1.0], id_extractor=lambda x: x["id"])
        assert len(result) == 1

    def test_descending_order(self) -> None:
        """Results are sorted by score descending."""
        results = [[("a", 0.1), ("b", 0.9), ("c", 0.5)]]
        result = combine_with_weights(results, [1.0])
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)
