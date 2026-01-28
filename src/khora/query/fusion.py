"""Reciprocal Rank Fusion for combining search results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class RankedItem:
    """An item with its score and source."""

    item: Any
    score: float
    source: str  # Which search method produced this result
    rank: int = 0  # Rank within its source


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[Any, float]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    id_extractor: callable = lambda x: x,
) -> list[tuple[Any, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF is a simple but effective method for combining results from
    multiple search systems without requiring score calibration.

    RRF score = sum(weight[source] / (k + rank[source]))

    Args:
        ranked_lists: Dict of source name to list of (item, score) tuples
        k: RRF parameter (default 60, higher = more even distribution)
        weights: Optional weights for each source
        id_extractor: Function to extract ID from item for deduplication

    Returns:
        List of (item, rrf_score) tuples sorted by score descending
    """
    if not ranked_lists:
        return []

    # Filter out empty lists
    ranked_lists = {k: v for k, v in ranked_lists.items() if v}
    if not ranked_lists:
        return []

    # Default equal weights
    if weights is None:
        weights = {source: 1.0 for source in ranked_lists}

    # Normalize weights
    total_weight = sum(weights.get(s, 1.0) for s in ranked_lists)
    if total_weight == 0:
        # If all weights are zero, use equal weights
        total_weight = len(ranked_lists)
        normalized_weights = {s: 1.0 / total_weight for s in ranked_lists}
    else:
        normalized_weights = {s: weights.get(s, 1.0) / total_weight for s in ranked_lists}

    # Calculate RRF scores
    rrf_scores: dict[Any, float] = {}
    items_by_id: dict[Any, Any] = {}

    for source, ranked_list in ranked_lists.items():
        weight = normalized_weights.get(source, 1.0)

        for rank, (item, _score) in enumerate(ranked_list, start=1):
            item_id = id_extractor(item)

            # RRF formula: weight / (k + rank)
            rrf_contribution = weight / (k + rank)

            if item_id in rrf_scores:
                rrf_scores[item_id] += rrf_contribution
            else:
                rrf_scores[item_id] = rrf_contribution
                items_by_id[item_id] = item

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [(items_by_id[item_id], rrf_scores[item_id]) for item_id in sorted_ids]


def combine_with_weights(
    results: list[list[tuple[Any, float]]],
    weights: list[float],
    *,
    id_extractor: callable = lambda x: x,
) -> list[tuple[Any, float]]:
    """Combine results using simple weighted scoring.

    Args:
        results: List of ranked result lists
        weights: Weight for each result list
        id_extractor: Function to extract ID from item

    Returns:
        Combined and sorted results
    """
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        # If all weights are zero, use equal weights
        normalized_weights = [1.0 / len(weights) for _ in weights] if weights else []
    else:
        normalized_weights = [w / total_weight for w in weights]

    # Combine scores
    combined_scores: dict[Any, float] = {}
    items_by_id: dict[Any, Any] = {}

    for result_list, weight in zip(results, normalized_weights):
        for item, score in result_list:
            item_id = id_extractor(item)
            weighted_score = score * weight

            if item_id in combined_scores:
                combined_scores[item_id] += weighted_score
            else:
                combined_scores[item_id] = weighted_score
                items_by_id[item_id] = item

    # Sort by combined score
    sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

    return [(items_by_id[item_id], combined_scores[item_id]) for item_id in sorted_ids]
