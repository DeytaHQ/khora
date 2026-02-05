"""RRF (Reciprocal Rank Fusion) utilities for combining search results.

This module provides utilities for fusing vector and graph search results
using Reciprocal Rank Fusion, a simple but effective method for combining
ranked lists from different retrieval systems.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import UUID

T = TypeVar("T")


@dataclass
class FusedResult:
    """Result after RRF fusion."""

    item_id: UUID
    item: Any
    rrf_score: float
    vector_rank: int | None = None
    graph_rank: int | None = None
    vector_score: float | None = None
    graph_score: float | None = None


def reciprocal_rank_fusion(
    *result_lists: list[tuple[UUID, float, Any]],
    k: int = 60,
) -> list[FusedResult]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each list where item appears.

    Args:
        *result_lists: Variable number of result lists, each containing
                       tuples of (item_id, score, item_data)
        k: Constant to prevent high-ranked items from dominating (default: 60)

    Returns:
        List of FusedResult sorted by RRF score (highest first)
    """
    # Track RRF scores and source information
    rrf_scores: dict[UUID, float] = defaultdict(float)
    items: dict[UUID, Any] = {}
    vector_ranks: dict[UUID, int] = {}
    graph_ranks: dict[UUID, int] = {}
    vector_scores: dict[UUID, float] = {}
    graph_scores: dict[UUID, float] = {}

    for list_idx, results in enumerate(result_lists):
        for rank, (item_id, score, item) in enumerate(results, start=1):
            rrf_scores[item_id] += 1.0 / (k + rank)
            items[item_id] = item

            # Track source-specific information
            if list_idx == 0:  # Vector results
                vector_ranks[item_id] = rank
                vector_scores[item_id] = score
            elif list_idx == 1:  # Graph results
                graph_ranks[item_id] = rank
                graph_scores[item_id] = score

    # Build fused results
    fused = [
        FusedResult(
            item_id=item_id,
            item=items[item_id],
            rrf_score=rrf_score,
            vector_rank=vector_ranks.get(item_id),
            graph_rank=graph_ranks.get(item_id),
            vector_score=vector_scores.get(item_id),
            graph_score=graph_scores.get(item_id),
        )
        for item_id, rrf_score in rrf_scores.items()
    ]

    # Sort by RRF score descending
    fused.sort(key=lambda x: x.rrf_score, reverse=True)

    return fused


def weighted_rrf(
    vector_results: list[tuple[UUID, float, Any]],
    graph_results: list[tuple[UUID, float, Any]],
    *,
    k: int = 60,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list[FusedResult]:
    """Weighted RRF fusion with configurable weights for each source.

    Extends RRF with weights to emphasize one source over another.

    Args:
        vector_results: Results from vector search (item_id, score, item)
        graph_results: Results from graph traversal (item_id, score, item)
        k: RRF constant (default: 60)
        vector_weight: Weight for vector results (default: 0.6)
        graph_weight: Weight for graph results (default: 0.4)

    Returns:
        List of FusedResult sorted by weighted RRF score
    """
    # Track weighted RRF scores
    rrf_scores: dict[UUID, float] = defaultdict(float)
    items: dict[UUID, Any] = {}
    vector_ranks: dict[UUID, int] = {}
    graph_ranks: dict[UUID, int] = {}
    vector_scores: dict[UUID, float] = {}
    graph_scores: dict[UUID, float] = {}

    # Process vector results with weight
    for rank, (item_id, score, item) in enumerate(vector_results, start=1):
        rrf_scores[item_id] += vector_weight / (k + rank)
        items[item_id] = item
        vector_ranks[item_id] = rank
        vector_scores[item_id] = score

    # Process graph results with weight
    for rank, (item_id, score, item) in enumerate(graph_results, start=1):
        rrf_scores[item_id] += graph_weight / (k + rank)
        if item_id not in items:
            items[item_id] = item
        graph_ranks[item_id] = rank
        graph_scores[item_id] = score

    # Build fused results
    fused = [
        FusedResult(
            item_id=item_id,
            item=items[item_id],
            rrf_score=rrf_score,
            vector_rank=vector_ranks.get(item_id),
            graph_rank=graph_ranks.get(item_id),
            vector_score=vector_scores.get(item_id),
            graph_score=graph_scores.get(item_id),
        )
        for item_id, rrf_score in rrf_scores.items()
    ]

    # Sort by weighted RRF score descending
    fused.sort(key=lambda x: x.rrf_score, reverse=True)

    return fused


def normalize_scores(results: list[FusedResult]) -> list[FusedResult]:
    """Normalize RRF scores to [0, 1] range.

    Args:
        results: List of FusedResult to normalize

    Returns:
        Same list with normalized rrf_score values
    """
    if not results:
        return results

    max_score = max(r.rrf_score for r in results)
    min_score = min(r.rrf_score for r in results)

    if max_score == min_score:
        for r in results:
            r.rrf_score = 1.0
    else:
        for r in results:
            r.rrf_score = (r.rrf_score - min_score) / (max_score - min_score)

    return results


def apply_recency_boost(
    results: list[FusedResult],
    recency_scores: dict[UUID, float],
    *,
    recency_weight: float = 0.2,
) -> list[FusedResult]:
    """Apply recency boost to fused results.

    Combines RRF score with a recency score to boost more recent results.

    Args:
        results: List of FusedResult
        recency_scores: Map of item_id -> recency score (0-1, higher = more recent)
        recency_weight: Weight for recency (default: 0.2)

    Returns:
        Results with adjusted scores, re-sorted
    """
    for r in results:
        recency = recency_scores.get(r.item_id, 0.0)
        # Blend RRF score with recency
        r.rrf_score = (1 - recency_weight) * r.rrf_score + recency_weight * recency

    # Re-sort by adjusted score
    results.sort(key=lambda x: x.rrf_score, reverse=True)

    return results


__all__ = [
    "FusedResult",
    "apply_recency_boost",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "weighted_rrf",
]
