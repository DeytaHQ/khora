"""Score normalization helpers for the public ``RecallChunk.score`` contract.

``RecallChunk.score`` is defined as a min-max normalized rank in [0, 1] on
every engine - the top chunk in the returned set always gets 1.0, the bottom
always gets 0.0 (when there are 2+ chunks). With one chunk, score = 1.0.

Prior to v0.17.1 the three engines exposed three different scales:

- VectorCypher: min-max normalized RRF rank in [0, 1] (the correct one).
- Chronicle: post-rerank fused score on an arbitrary scale (cross-encoder +
  temporal decay + version + RRF).
- Skeleton: raw cosine or BM25 ``combined_score`` on an arbitrary scale.

Reporter Damir caught this on #834. This helper is the single normalization
shape Chronicle and Skeleton both use to comply with the unified contract.
"""

from __future__ import annotations


def min_max_normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1].

    Args:
        scores: Raw scores in the order they will be returned to the caller.

    Returns:
        A list the same length as ``scores`` where the maximum maps to 1.0,
        the minimum to 0.0, and intermediate values to their linear position
        between the two. Single-element and tied-score lists collapse to
        ``[1.0, ...]`` so the public contract ("top chunk = 1.0") still holds.
    """
    if not scores:
        return scores

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


__all__ = ["min_max_normalize"]
