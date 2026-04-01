"""Unit tests for pluggable fusion strategies."""

from __future__ import annotations

import pytest

from khora.query.fusion_strategies import (
    CombMNZStrategy,
    FusionResult,
    RRFStrategy,
    WeightedSumStrategy,
    _min_max_normalize,
    _z_score_normalize,
    create_fusion_strategy,
    list_fusion_strategies,
)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalization:
    def test_min_max_basic(self) -> None:
        result = _min_max_normalize([1.0, 2.0, 3.0])
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_min_max_single(self) -> None:
        result = _min_max_normalize([5.0])
        assert result == [0.5]

    def test_min_max_same_values(self) -> None:
        result = _min_max_normalize([3.0, 3.0, 3.0])
        assert result == [0.5, 0.5, 0.5]

    def test_min_max_empty(self) -> None:
        assert _min_max_normalize([]) == []

    def test_z_score_basic(self) -> None:
        result = _z_score_normalize([1.0, 2.0, 3.0])
        assert len(result) == 3
        assert result[0] < result[1] < result[2]  # order preserved
        assert min(result) == pytest.approx(0.0)
        assert max(result) == pytest.approx(1.0)

    def test_z_score_preserves_order(self) -> None:
        # Z-score preserves relative ordering
        scores = [0.9, 0.91, 0.92, 0.93, 0.1]  # 0.1 is outlier
        z_result = _z_score_normalize(scores)
        # Order preserved: 0.1 should be lowest, 0.93 highest
        assert z_result[4] < z_result[0]  # outlier is lowest
        assert z_result[0] < z_result[3]  # 0.9 < 0.93
        # All values in [0, 1]
        assert all(0.0 <= v <= 1.0 for v in z_result)

    def test_z_score_single(self) -> None:
        assert _z_score_normalize([5.0]) == [0.5]

    def test_z_score_empty(self) -> None:
        assert _z_score_normalize([]) == []


# ---------------------------------------------------------------------------
# RRF Strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRRFStrategy:
    def test_basic_fusion(self) -> None:
        strategy = RRFStrategy(k=60)
        result = strategy.fuse(
            {"vector": [("a", 0.9), ("b", 0.8)], "keyword": [("b", 0.7), ("c", 0.6)]},
            id_extractor=lambda x: x,
        )
        assert isinstance(result, FusionResult)
        assert len(result.items) >= 2
        assert result.metadata["strategy"] == "rrf"

    def test_empty_input(self) -> None:
        strategy = RRFStrategy()
        result = strategy.fuse({})
        assert result.items == []

    def test_single_source(self) -> None:
        strategy = RRFStrategy()
        result = strategy.fuse({"vector": [("a", 0.9), ("b", 0.5)]})
        assert len(result.items) == 2

    def test_weights_applied(self) -> None:
        strategy = RRFStrategy()
        result = strategy.fuse(
            {"vector": [("a", 0.9)], "keyword": [("b", 0.9)]},
            weights={"vector": 0.8, "keyword": 0.2},
        )
        # "a" from vector (weight 0.8) should score higher than "b" from keyword (0.2)
        ids = [item for item, _ in result.items]
        assert ids[0] == "a"


# ---------------------------------------------------------------------------
# Weighted Sum Strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWeightedSumStrategy:
    def test_basic_fusion(self) -> None:
        strategy = WeightedSumStrategy()
        result = strategy.fuse(
            {"vector": [("a", 0.9), ("b", 0.5)], "keyword": [("b", 0.8), ("c", 0.3)]},
            weights={"vector": 0.6, "keyword": 0.4},
            id_extractor=lambda x: x,
        )
        assert isinstance(result, FusionResult)
        assert result.metadata["strategy"] == "weighted_sum"
        # "b" appears in both sources — should rank high
        ids = [item for item, _ in result.items]
        assert "b" in ids

    def test_z_score_normalization(self) -> None:
        strategy = WeightedSumStrategy(normalization="z_score")
        result = strategy.fuse(
            {"vector": [("a", 0.95), ("b", 0.50)]},
            id_extractor=lambda x: x,
        )
        assert result.metadata["normalization"] == "z_score"

    def test_min_max_normalization(self) -> None:
        strategy = WeightedSumStrategy(normalization="min_max")
        result = strategy.fuse(
            {"vector": [("a", 0.95), ("b", 0.50)]},
            id_extractor=lambda x: x,
        )
        assert result.metadata["normalization"] == "min_max"

    def test_empty_input(self) -> None:
        strategy = WeightedSumStrategy()
        result = strategy.fuse({})
        assert result.items == []


# ---------------------------------------------------------------------------
# CombMNZ Strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombMNZStrategy:
    def test_multi_source_boost(self) -> None:
        strategy = CombMNZStrategy()
        result = strategy.fuse(
            {
                "vector": [("a", 0.9), ("b", 0.5)],
                "keyword": [("b", 0.8), ("c", 0.3)],
                "graph": [("b", 0.7)],
            },
            id_extractor=lambda x: x,
        )
        # "b" appears in all 3 sources — CombMNZ should rank it first
        ids = [item for item, _ in result.items]
        assert ids[0] == "b"

    def test_single_source_no_boost(self) -> None:
        strategy = CombMNZStrategy()
        result = strategy.fuse(
            {"vector": [("a", 0.9), ("b", 0.5)]},
            id_extractor=lambda x: x,
        )
        # With single source, count=1 for all — just sum of scores
        assert len(result.items) == 2

    def test_metadata(self) -> None:
        strategy = CombMNZStrategy()
        result = strategy.fuse({"v": [("a", 1.0)]}, id_extractor=lambda x: x)
        assert result.metadata["strategy"] == "combmnz"

    def test_empty_input(self) -> None:
        strategy = CombMNZStrategy()
        result = strategy.fuse({})
        assert result.items == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFusionFactory:
    def test_create_rrf(self) -> None:
        s = create_fusion_strategy("rrf", k=40)
        assert s.name == "rrf"

    def test_create_weighted_sum(self) -> None:
        s = create_fusion_strategy("weighted_sum", normalization="min_max")
        assert s.name == "weighted_sum"

    def test_create_combmnz(self) -> None:
        s = create_fusion_strategy("combmnz")
        assert s.name == "combmnz"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            create_fusion_strategy("nonexistent")

    def test_list_strategies(self) -> None:
        strategies = list_fusion_strategies()
        assert "rrf" in strategies
        assert "weighted_sum" in strategies
        assert "combmnz" in strategies
