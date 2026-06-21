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


def attach_relevance_scores(results: list[FusedResult]) -> list[FusedResult]:
    """Replace each result's score with an absolute relevance measure.

    The final ranking is decided by ``rrf_score`` (after fusion + all boosts +
    reranking), so this preserves the existing order. It only rewrites the score
    VALUE reported to callers, swapping the per-result-set min-max normalized RRF
    score (which forces the top to 1.0 and the bottom to 0.0 regardless of how
    relevant they are) for the raw vector cosine similarity captured pre-fusion.

    Cosine similarity is absolute and comparable across queries, so an off-topic
    top result reads as low (e.g. ~0.1) instead of 1.0 and callers can threshold
    on it. Graph-only chunks (no vector hit) fall back to the graph score; if
    neither is present the existing ``rrf_score`` is kept unchanged.

    The list is NOT re-sorted - callers rely on the boost/rerank-determined order.

    Args:
        results: List of FusedResult, already sorted by final ``rrf_score``

    Returns:
        Same list, in the same order, with ``rrf_score`` set to absolute relevance
    """
    for r in results:
        if r.vector_score is not None:
            r.rrf_score = r.vector_score
        elif r.graph_score is not None:
            r.rrf_score = r.graph_score
    return results


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1] range.

    Args:
        scores: List of scores to normalize

    Returns:
        Normalized scores in same order
    """
    if not scores:
        return scores

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def weighted_rrf_normalized(
    vector_results: list[tuple[UUID, float, Any]],
    graph_results: list[tuple[UUID, float, Any]],
    *,
    k: int = 60,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list[FusedResult]:
    """Weighted RRF with score normalization before fusion.

    This improved fusion method:
    1. Normalizes vector and graph scores to [0, 1] before computing RRF
    2. Uses normalized scores as a tiebreaker for same-rank items
    3. Produces more balanced fusion when score distributions differ

    The normalization prevents one source from dominating when its scores
    are on a different scale (e.g., cosine similarity 0-1 vs graph scores 1-N).

    Args:
        vector_results: Results from vector search (item_id, score, item)
        graph_results: Results from graph traversal (item_id, score, item)
        k: RRF constant (default: 60)
        vector_weight: Weight for vector results (default: 0.6)
        graph_weight: Weight for graph results (default: 0.4)

    Returns:
        List of FusedResult sorted by weighted normalized RRF score
    """
    # Track scores and item data
    rrf_scores: dict[UUID, float] = defaultdict(float)
    score_contributions: dict[UUID, float] = defaultdict(float)  # Normalized score contributions
    items: dict[UUID, Any] = {}
    vector_ranks: dict[UUID, int] = {}
    graph_ranks: dict[UUID, int] = {}
    vector_scores: dict[UUID, float] = {}
    graph_scores: dict[UUID, float] = {}

    # Normalize vector scores
    if vector_results:
        raw_vector_scores = [score for _, score, _ in vector_results]
        normalized_vector_scores = _min_max_normalize(raw_vector_scores)

        for rank, ((item_id, score, item), norm_score) in enumerate(
            zip(vector_results, normalized_vector_scores), start=1
        ):
            # RRF contribution
            rrf_scores[item_id] += vector_weight / (k + rank)
            # Add small normalized score contribution for tiebreaking
            # Scale down to not dominate RRF (max contribution ~0.01 per source)
            score_contributions[item_id] += vector_weight * norm_score * 0.01
            items[item_id] = item
            vector_ranks[item_id] = rank
            vector_scores[item_id] = score

    # Normalize graph scores
    if graph_results:
        raw_graph_scores = [score for _, score, _ in graph_results]
        normalized_graph_scores = _min_max_normalize(raw_graph_scores)

        for rank, ((item_id, score, item), norm_score) in enumerate(
            zip(graph_results, normalized_graph_scores), start=1
        ):
            # RRF contribution
            rrf_scores[item_id] += graph_weight / (k + rank)
            # Add normalized score contribution for tiebreaking
            score_contributions[item_id] += graph_weight * norm_score * 0.01
            if item_id not in items:
                items[item_id] = item
            graph_ranks[item_id] = rank
            graph_scores[item_id] = score

    # Combine RRF with normalized score contributions
    final_scores = {item_id: rrf_scores[item_id] + score_contributions[item_id] for item_id in rrf_scores}

    # Build fused results
    fused = [
        FusedResult(
            item_id=item_id,
            item=items[item_id],
            rrf_score=final_score,
            vector_rank=vector_ranks.get(item_id),
            graph_rank=graph_ranks.get(item_id),
            vector_score=vector_scores.get(item_id),
            graph_score=graph_scores.get(item_id),
        )
        for item_id, final_score in final_scores.items()
    ]

    # Sort by final score descending
    fused.sort(key=lambda x: x.rrf_score, reverse=True)

    return fused


def weighted_rrf_nlist(
    sources: list[tuple[list[tuple[UUID, float, Any]], float]],
    *,
    k: int = 60,
) -> list[FusedResult]:
    """Weighted RRF over N sources in a single accumulation pass.

    ``sources`` is a list of ``(results, weight)`` pairs. Unlike folding sources
    in pairwise (which re-ranks an already-fused list and applies the wrong
    weight to it), every source contributes ``weight / (k + rank)`` exactly once
    into a shared accumulator, so the configured weights are honored. Each
    source's scores are min-max normalized for a small tiebreak contribution,
    matching :func:`weighted_rrf_normalized`.
    """
    rrf_scores: dict[UUID, float] = defaultdict(float)
    score_contributions: dict[UUID, float] = defaultdict(float)
    items: dict[UUID, Any] = {}

    for results, weight in sources:
        if not results:
            continue
        raw_scores = [score for _, score, _ in results]
        normalized = _min_max_normalize(raw_scores)
        for rank, ((item_id, _score, item), norm_score) in enumerate(zip(results, normalized), start=1):
            rrf_scores[item_id] += weight / (k + rank)
            score_contributions[item_id] += weight * norm_score * 0.01
            if item_id not in items:
                items[item_id] = item

    final_scores = {item_id: rrf_scores[item_id] + score_contributions[item_id] for item_id in rrf_scores}
    fused = [
        FusedResult(item_id=item_id, item=items[item_id], rrf_score=final_score)
        for item_id, final_score in final_scores.items()
    ]
    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    return fused


def apply_recency_boost(
    results: list[FusedResult],
    recency_scores: dict[UUID, float],
    *,
    recency_weight: float = 0.2,
    recency_floor: float = 0.5,
) -> list[FusedResult]:
    """Apply multiplicative recency boost to fused results.

    Uses a multiplicative formula so recency attenuates stale results without
    inflating scores above the original RRF value.  This prevents the abstention
    regression where high recency weights pushed irrelevant-but-recent chunks
    above score thresholds.

    Formula: ``rrf_score *= max(recency, floor) ** (exponent * recency_weight)``

    - recency_floor prevents zeroing-out chunks that lack timestamps.
      Per-category floors allow stronger discrimination for temporal queries
      (e.g. 0.3 for STATE_QUERY/RECENCY) while protecting non-temporal ones (0.5).
    - The exponent (2.0) controls decay steepness.
    - recency_weight scales influence: NONE (0.2) barely changes ranking,
      STATE_QUERY (0.6) penalises stale chunks significantly.

    Args:
        results: List of FusedResult
        recency_scores: Map of item_id -> recency score (0-1, higher = more recent)
        recency_weight: Weight for recency (default: 0.2)
        recency_floor: Minimum recency to prevent zero-multiplication (default: 0.5)

    Returns:
        Results with adjusted scores, re-sorted
    """
    _EXPONENT = 2.0  # steepness of decay curve

    for r in results:
        recency = recency_scores.get(r.item_id, recency_floor)
        clamped = max(recency, recency_floor)
        # Multiplicative: old chunks get attenuated, recent chunks stay near 1.0
        r.rrf_score *= clamped ** (_EXPONENT * recency_weight)

    # Re-sort by adjusted score
    results.sort(key=lambda x: x.rrf_score, reverse=True)

    return results


def bigram_coherence_score(text: str) -> float:
    """Score text coherence based on word transition patterns.

    Natural text has predictable transitions: articles before nouns,
    prepositions before noun phrases. Word-shuffled text breaks these patterns.

    Returns value in [0.0, 1.0] where 1.0 = coherent natural text.
    """
    words = text.lower().split()
    n = len(words)
    if n < 6:
        return 1.0  # Too short to assess

    _ARTICLES = frozenset({"the", "a", "an", "this", "that", "these", "those"})
    _PREPOSITIONS = frozenset({"in", "of", "to", "for", "with", "on", "at", "by", "from", "about"})
    _FUNCTION = _ARTICLES | _PREPOSITIONS | frozenset({"is", "are", "was", "were", "and", "or", "but", "not"})

    coherent = 0
    total = 0
    for i in range(n - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 in _ARTICLES:
            total += 1
            if w2 not in _FUNCTION:
                coherent += 1
        elif w1 in _PREPOSITIONS:
            total += 1
            if w2 in _ARTICLES or w2 not in _PREPOSITIONS:
                coherent += 1

    if total == 0:
        return 1.0

    return coherent / total


def apply_coherence_boost(
    results: list[FusedResult],
    *,
    coherence_weight: float = 0.1,
) -> list[FusedResult]:
    """Apply coherence-based re-ranking to penalize word-shuffled confounders.

    Computes a bigram coherence score for each chunk's content and blends
    it with the existing RRF score. Shuffled text scores low, natural text
    scores high.

    Args:
        results: List of FusedResult (items must have .content attribute)
        coherence_weight: Weight for coherence signal (default: 0.1)

    Returns:
        Results with adjusted scores, re-sorted
    """
    for r in results:
        content = getattr(r.item, "content", None) or ""
        coherence = bigram_coherence_score(content)
        r.rrf_score = (1 - coherence_weight) * r.rrf_score + coherence_weight * coherence

    results.sort(key=lambda x: x.rrf_score, reverse=True)
    return results


__all__ = [
    "FusedResult",
    "apply_coherence_boost",
    "apply_recency_boost",
    "attach_relevance_scores",
    "bigram_coherence_score",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "weighted_rrf",
    "weighted_rrf_normalized",
]
