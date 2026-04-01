"""Pluggable fusion strategies for combining multi-source search results.

Provides a protocol-based architecture for swapping between different
fusion algorithms without changing engine or query engine code.

Available strategies:
- ``rrf``: Reciprocal Rank Fusion (default, current behavior)
- ``weighted_sum``: Normalized weighted linear combination
- ``combmnz``: CombMNZ — boosts items appearing in multiple sources

Usage::

    from khora.query.fusion_strategies import create_fusion_strategy

    strategy = create_fusion_strategy("rrf", k=60)
    fused = strategy.fuse(ranked_lists, weights={"vector": 0.5, "graph": 0.3})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class FusionResult:
    """Result of a fusion operation."""

    items: list[tuple[Any, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FusionStrategy(Protocol):
    """Protocol for fusion strategies."""

    @property
    def name(self) -> str: ...

    def fuse(
        self,
        ranked_lists: dict[str, list[tuple[Any, float]]],
        *,
        weights: dict[str, float] | None = None,
        id_extractor: Any = None,
    ) -> FusionResult: ...


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    if rng == 0:
        return [0.5] * len(scores)
    return [(s - mn) / rng for s in scores]


def _z_score_normalize(scores: list[float]) -> list[float]:
    """Z-score normalize scores (mean=0, std=1), then shift to [0, 1] range.

    More robust to outliers than min-max. A single extreme value doesn't
    destroy granularity for the rest of the distribution.
    """
    if not scores or len(scores) < 2:
        return [0.5] * len(scores)

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance) if variance > 0 else 1.0

    if std == 0:
        return [0.5] * len(scores)

    z_scores = [(s - mean) / std for s in scores]

    # Shift to [0, 1] using sigmoid-like clamping
    mn, mx = min(z_scores), max(z_scores)
    rng = mx - mn
    if rng == 0:
        return [0.5] * len(scores)
    return [(z - mn) / rng for z in z_scores]


# ---------------------------------------------------------------------------
# RRF Strategy (default)
# ---------------------------------------------------------------------------


class RRFStrategy:
    """Reciprocal Rank Fusion.

    score(item) = Σ weight[source] / (k + rank_in_source)

    The default and battle-tested fusion algorithm. Ignores score magnitudes,
    only uses ranks — fair across sources with different score distributions.
    """

    def __init__(self, *, k: int = 60) -> None:
        self._k = k

    @property
    def name(self) -> str:
        return "rrf"

    def fuse(
        self,
        ranked_lists: dict[str, list[tuple[Any, float]]],
        *,
        weights: dict[str, float] | None = None,
        id_extractor: Any = None,
    ) -> FusionResult:
        from .fusion import reciprocal_rank_fusion

        _id_fn = id_extractor or (lambda x: x)
        items = reciprocal_rank_fusion(
            ranked_lists,
            k=self._k,
            weights=weights,
            id_extractor=_id_fn,
        )
        return FusionResult(
            items=items,
            metadata={"strategy": "rrf", "k": self._k},
        )


# ---------------------------------------------------------------------------
# Weighted Sum Strategy
# ---------------------------------------------------------------------------


class WeightedSumStrategy:
    """Normalized weighted linear sum of scores.

    Normalizes each source's scores independently, then combines:
      final_score = Σ weight[source] * normalized_score[source]

    Better than RRF when score magnitudes are meaningful (e.g., calibrated
    similarity scores). Supports min-max or z-score normalization.
    """

    def __init__(self, *, normalization: str = "z_score") -> None:
        """
        Args:
            normalization: "min_max" or "z_score" (default: z_score).
        """
        self._normalization = normalization

    @property
    def name(self) -> str:
        return "weighted_sum"

    def fuse(
        self,
        ranked_lists: dict[str, list[tuple[Any, float]]],
        *,
        weights: dict[str, float] | None = None,
        id_extractor: Any = None,
    ) -> FusionResult:
        if not ranked_lists:
            return FusionResult(items=[], metadata={"strategy": "weighted_sum"})

        _id_fn = id_extractor or (lambda x: x)

        # Default equal weights
        if weights is None:
            weights = {s: 1.0 for s in ranked_lists}

        # Normalize weights to sum to 1
        total = sum(weights.get(s, 1.0) for s in ranked_lists)
        norm_weights = {s: weights.get(s, 1.0) / total for s in ranked_lists} if total > 0 else {}

        # Choose normalization function
        normalize_fn = _z_score_normalize if self._normalization == "z_score" else _min_max_normalize

        # Normalize and aggregate
        item_scores: dict[Any, float] = {}
        items_by_id: dict[Any, Any] = {}

        for source, items in ranked_lists.items():
            if not items:
                continue
            w = norm_weights.get(source, 0.0)
            raw_scores = [s for _, s in items]
            norm_scores = normalize_fn(raw_scores)

            for (item, _raw), norm_s in zip(items, norm_scores):
                item_id = _id_fn(item)
                if item_id not in items_by_id:
                    items_by_id[item_id] = item
                    item_scores[item_id] = 0.0
                item_scores[item_id] += w * norm_s

        # Sort descending
        sorted_ids = sorted(item_scores, key=item_scores.get, reverse=True)
        result = [(items_by_id[iid], item_scores[iid]) for iid in sorted_ids]

        return FusionResult(
            items=result,
            metadata={
                "strategy": "weighted_sum",
                "normalization": self._normalization,
            },
        )


# ---------------------------------------------------------------------------
# CombMNZ Strategy
# ---------------------------------------------------------------------------


class CombMNZStrategy:
    """CombMNZ: Combined Maximum Normalization.

    score(item) = count_sources_containing(item) * Σ normalized_scores

    Boosts items appearing in multiple sources (consensus signal).
    Good for entity-centric queries where agreement across vector,
    graph, and keyword searches indicates high relevance.
    """

    def __init__(self, *, normalization: str = "z_score") -> None:
        self._normalization = normalization

    @property
    def name(self) -> str:
        return "combmnz"

    def fuse(
        self,
        ranked_lists: dict[str, list[tuple[Any, float]]],
        *,
        weights: dict[str, float] | None = None,
        id_extractor: Any = None,
    ) -> FusionResult:
        if not ranked_lists:
            return FusionResult(items=[], metadata={"strategy": "combmnz"})

        _id_fn = id_extractor or (lambda x: x)
        normalize_fn = _z_score_normalize if self._normalization == "z_score" else _min_max_normalize

        # Normalize each source and accumulate
        item_data: dict[Any, dict[str, Any]] = {}

        for source, items in ranked_lists.items():
            if not items:
                continue
            raw_scores = [s for _, s in items]
            norm_scores = normalize_fn(raw_scores)

            for (item, _raw), norm_s in zip(items, norm_scores):
                item_id = _id_fn(item)
                if item_id not in item_data:
                    item_data[item_id] = {"item": item, "sources": set(), "sum_score": 0.0}
                item_data[item_id]["sources"].add(source)
                item_data[item_id]["sum_score"] += norm_s

        # CombMNZ: count * sum
        result = [(d["item"], len(d["sources"]) * d["sum_score"]) for d in item_data.values()]
        result.sort(key=lambda x: x[1], reverse=True)

        return FusionResult(
            items=result,
            metadata={"strategy": "combmnz", "normalization": self._normalization},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, type] = {
    "rrf": RRFStrategy,
    "weighted_sum": WeightedSumStrategy,
    "combmnz": CombMNZStrategy,
}


def register_fusion_strategy(name: str, cls: type) -> None:
    """Register a custom fusion strategy.

    Args:
        name: Strategy name (used in config)
        cls: Class implementing FusionStrategy protocol
    """
    _STRATEGY_REGISTRY[name] = cls


def create_fusion_strategy(name: str = "rrf", **kwargs: Any) -> FusionStrategy:
    """Create a fusion strategy by name.

    Args:
        name: Strategy name ("rrf", "weighted_sum", "combmnz")
        **kwargs: Strategy-specific parameters

    Returns:
        FusionStrategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(_STRATEGY_REGISTRY)
        raise ValueError(f"Unknown fusion strategy: {name!r}. Available: {available}")
    return cls(**kwargs)


def list_fusion_strategies() -> list[str]:
    """List available fusion strategy names."""
    return list(_STRATEGY_REGISTRY)
