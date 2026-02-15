"""Reciprocal Rank Fusion for combining search results.

This module delegates to :mod:`khora.engines.vectorcypher.fusion` for the
core RRF implementation while preserving a simplified public API that
accepts generic ``(item, score)`` tuples keyed by source name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import uuid4

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

    Delegates to :func:`khora.engines.vectorcypher.fusion.weighted_rrf_normalized`
    for the core RRF computation with score normalization.

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
        total_weight = len(ranked_lists)
        normalized_weights = {s: 1.0 / total_weight for s in ranked_lists}
    else:
        normalized_weights = {s: weights.get(s, 1.0) / total_weight for s in ranked_lists}

    from khora.engines.vectorcypher.fusion import weighted_rrf_normalized as _weighted_rrf_normalized

    # Assign stable synthetic UUIDs per item for deduplication and map
    # the generic (item, score) tuples into the vectorcypher format
    # (UUID, score, item).
    id_to_uuid: dict[Any, Any] = {}  # id_extractor(item) -> UUID
    items_by_uuid: dict[Any, Any] = {}  # UUID -> original item

    source_names = list(ranked_lists.keys())

    # Build two canonical lists: first two sources map to vector/graph slots,
    # additional sources are folded in via sequential calls.
    all_converted: list[list[tuple[Any, float, Any]]] = []
    for source in source_names:
        converted: list[tuple[Any, float, Any]] = []
        for item, score in ranked_lists[source]:
            item_id = id_extractor(item)
            if item_id not in id_to_uuid:
                id_to_uuid[item_id] = uuid4()
            uid = id_to_uuid[item_id]
            items_by_uuid[uid] = item
            converted.append((uid, score, item))
        all_converted.append(converted)

    # Fuse pairwise: start with first source, merge subsequent sources
    if len(all_converted) == 1:
        source = source_names[0]
        w = normalized_weights.get(source, 1.0)
        fused = _weighted_rrf_normalized(
            all_converted[0],
            [],
            k=k,
            vector_weight=w,
            graph_weight=0.0,
        )
    else:
        # First pair
        s0, s1 = source_names[0], source_names[1]
        w0 = normalized_weights.get(s0, 1.0)
        w1 = normalized_weights.get(s1, 1.0)
        fused = _weighted_rrf_normalized(
            all_converted[0],
            all_converted[1],
            k=k,
            vector_weight=w0,
            graph_weight=w1,
        )

        # Fold in additional sources (rare: most callers use 2 sources)
        for i in range(2, len(all_converted)):
            si = source_names[i]
            wi = normalized_weights.get(si, 1.0)
            # Convert current fused results back to list format
            current = [(r.item_id, r.rrf_score, r.item) for r in fused]
            fused = _weighted_rrf_normalized(
                current,
                all_converted[i],
                k=k,
                vector_weight=1.0,
                graph_weight=wi,
            )

    # Map back to (original_item, score) tuples
    return [(items_by_uuid[r.item_id], r.rrf_score) for r in fused]


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
