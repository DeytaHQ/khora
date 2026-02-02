"""Accelerated operations with graceful fallbacks.

Provides optimized implementations of CPU-intensive operations using
numpy (cosine similarity) and rapidfuzz (string similarity). Falls back
to pure-Python implementations when those libraries are not available.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

try:
    from rapidfuzz.distance import Levenshtein as _rf_lev  # type: ignore[unresolved-import]
    from rapidfuzz.fuzz import ratio as _rf_ratio  # type: ignore[unresolved-import]

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses numpy when available (~50-200x faster for 1536-dim vectors).
    """
    if len(vec1) != len(vec2):
        return 0.0

    if _HAS_NUMPY:
        a = np.asarray(vec1, dtype=np.float32)
        b = np.asarray(vec2, dtype=np.float32)
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    # Pure-Python fallback
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def batch_cosine_similarity(
    query: list[float],
    candidates: list[list[float]],
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """Compute cosine similarity between a query vector and a matrix of candidates.

    Returns (index, similarity) pairs above threshold, sorted descending.
    Uses numpy batch matmul when available.
    """
    if not candidates:
        return []

    if _HAS_NUMPY:
        q = np.asarray(query, dtype=np.float32)
        mat = np.asarray(candidates, dtype=np.float32)

        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        norms = np.linalg.norm(mat, axis=1)
        # Avoid division by zero
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        sims = (mat @ q) / (safe_norms * q_norm)
        # Zero out entries where candidate norm was zero
        sims = np.where(norms == 0.0, 0.0, sims)

        results = []
        for i in range(len(sims)):
            s = float(sims[i])
            if s >= threshold:
                results.append((i, s))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # Pure-Python fallback
    results = []
    for i, cand in enumerate(candidates):
        s = cosine_similarity(query, cand)
        if s >= threshold:
            results.append((i, s))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Levenshtein similarity
# ---------------------------------------------------------------------------


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1.0 = identical).

    Uses rapidfuzz when available (~5-10x faster).
    """
    a, b = s1.lower(), s2.lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    if _HAS_RAPIDFUZZ:
        return _rf_lev.normalized_similarity(a, b)

    # Pure-Python single-row DP fallback
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr

    distance = prev[lb]
    return 1.0 - (distance / max(la, lb))


# ---------------------------------------------------------------------------
# Sequence matching (SequenceMatcher replacement)
# ---------------------------------------------------------------------------


def sequence_match_ratio(s1: str, s2: str) -> float:
    """Compute sequence match ratio between two strings.

    Uses rapidfuzz.fuzz.ratio when available (~3-10x faster than
    difflib.SequenceMatcher). Returns a float in [0.0, 1.0].
    """
    if _HAS_RAPIDFUZZ:
        return _rf_ratio(s1, s2) / 100.0

    from difflib import SequenceMatcher

    return SequenceMatcher(None, s1, s2).ratio()
