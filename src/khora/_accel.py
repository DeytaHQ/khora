"""Accelerated operations with graceful fallbacks.

Provides optimized implementations of CPU-intensive operations.
Three-tier acceleration: Rust (khora-accel) → NumPy/RapidFuzz → Pure Python.

Control the backend via the KHORA_ACCEL_BACKEND environment variable:
  - unset: auto-detect fastest available (default)
  - "rust": use Rust if available, fall through otherwise
  - "numpy": skip Rust, use NumPy/RapidFuzz
  - "python": force pure Python (useful for debugging/testing)
"""

from __future__ import annotations

import heapq
import math
import os
import re
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Runtime backend override
# ---------------------------------------------------------------------------

_FORCE_BACKEND = os.environ.get("KHORA_ACCEL_BACKEND")

# ---------------------------------------------------------------------------
# Tier 0: Rust native acceleration (fastest)
# ---------------------------------------------------------------------------

# Single source of truth for the Rust symbols khora depends on, mapping the
# local binding name used by the kernels below to the attribute name exported
# by the khora-accel wheel. The CI guard in
# tests/unit/test_accel_symbol_drift.py reuses this table to assert a built
# wheel exports every symbol, so it can't drift from what the kernels call.
_RUST_SYMBOLS: dict[str, str] = {
    "RustBM25Index": "RustBM25Index",
    "_rust_batch_cosine": "batch_cosine_similarity",
    "_rust_batch_dot_product": "batch_dot_product",
    "_rust_batch_levenshtein": "batch_levenshtein",
    "_rust_batch_recency_scores": "batch_recency_scores",
    "_rust_batch_score_stats": "batch_score_stats",
    "_rust_batch_sequence_match": "batch_sequence_match",
    "_rust_batch_temporal_filter": "batch_temporal_filter",
    "_rust_block_and_score_pairs": "block_and_score_pairs",
    "_rust_build_chunk_edges": "build_chunk_edges",
    "_rust_configure_thread_pool": "configure_thread_pool",
    "_rust_cosine": "cosine_similarity",
    "_rust_deduplicate_chunks": "deduplicate_chunks",
    "_rust_detect_communities": "detect_communities",
    "_rust_detect_temporal_category": "detect_temporal_category",
    "_rust_detect_temporal_category_with_confidence": "detect_temporal_category_with_confidence",
    "_rust_extract_keywords": "extract_keywords",
    "_rust_extract_keywords_batch": "extract_keywords_batch",
    "_rust_levenshtein": "levenshtein_similarity",
    "_rust_mmr_diversity_select": "mmr_diversity_select",
    "_rust_normalize_embeddings_batch": "normalize_embeddings_batch",
    "_rust_normalize_entity_name": "normalize_entity_name",
    "_rust_normalize_entity_names_batch": "normalize_entity_names_batch",
    "_rust_normalize_scores": "normalize_scores",
    "_rust_pagerank": "pagerank",
    "_rust_pairwise_cosine": "pairwise_cosine_above_threshold",
    "_rust_rrf": "reciprocal_rank_fusion",
    "_rust_resolve_entities_batch": "resolve_entities_batch",
    "_rust_resolve_entities_enhanced": "resolve_entities_enhanced",
    "_rust_score_entropy": "score_entropy",
    "_rust_sequence_match": "sequence_match_ratio",
    "_rust_weighted_rrf": "weighted_rrf",
    "_rust_weighted_rrf_normalized": "weighted_rrf_normalized",
    "_rust_weighted_rrf_normalized_with_diagnostics": "weighted_rrf_normalized_with_diagnostics",
    "_rust_weighted_rrf_normalized_with_provenance": "weighted_rrf_normalized_with_provenance",
}

# Import the wheel once, then bind each symbol individually via getattr. A stale
# or partial wheel that is missing one newly-added symbol degrades ONLY the
# kernels that need that symbol (they fall back to numpy / pure-Python) instead
# of disabling every Rust kernel. ``_HAS_RUST`` reflects "the wheel imported at
# all"; each kernel additionally gates on its own ``_rust_*`` binding being
# non-None. Bindings are explicit assignments (not a ``globals()`` loop) so
# static analysis can see them; ``_RUST_SYMBOLS`` above stays the single source
# of truth that the CI drift guard checks against.
_khora_accel: object | None
try:
    import khora_accel as _imported_khora_accel

    _khora_accel = _imported_khora_accel
except ImportError:
    _khora_accel = None

_HAS_RUST = _khora_accel is not None

RustBM25Index = getattr(_khora_accel, "RustBM25Index", None)
_rust_batch_cosine = getattr(_khora_accel, "batch_cosine_similarity", None)
_rust_batch_dot_product = getattr(_khora_accel, "batch_dot_product", None)
_rust_batch_levenshtein = getattr(_khora_accel, "batch_levenshtein", None)
_rust_batch_recency_scores = getattr(_khora_accel, "batch_recency_scores", None)
_rust_batch_score_stats = getattr(_khora_accel, "batch_score_stats", None)
_rust_batch_sequence_match = getattr(_khora_accel, "batch_sequence_match", None)
_rust_batch_temporal_filter = getattr(_khora_accel, "batch_temporal_filter", None)
_rust_block_and_score_pairs = getattr(_khora_accel, "block_and_score_pairs", None)
_rust_build_chunk_edges = getattr(_khora_accel, "build_chunk_edges", None)
_rust_configure_thread_pool = getattr(_khora_accel, "configure_thread_pool", None)
_rust_cosine = getattr(_khora_accel, "cosine_similarity", None)
_rust_deduplicate_chunks = getattr(_khora_accel, "deduplicate_chunks", None)
_rust_detect_communities = getattr(_khora_accel, "detect_communities", None)
_rust_detect_temporal_category = getattr(_khora_accel, "detect_temporal_category", None)
_rust_detect_temporal_category_with_confidence = getattr(_khora_accel, "detect_temporal_category_with_confidence", None)
_rust_extract_keywords = getattr(_khora_accel, "extract_keywords", None)
_rust_extract_keywords_batch = getattr(_khora_accel, "extract_keywords_batch", None)
_rust_levenshtein = getattr(_khora_accel, "levenshtein_similarity", None)
_rust_mmr_diversity_select = getattr(_khora_accel, "mmr_diversity_select", None)
_rust_normalize_embeddings_batch = getattr(_khora_accel, "normalize_embeddings_batch", None)
_rust_normalize_entity_name = getattr(_khora_accel, "normalize_entity_name", None)
_rust_normalize_entity_names_batch = getattr(_khora_accel, "normalize_entity_names_batch", None)
_rust_normalize_scores = getattr(_khora_accel, "normalize_scores", None)
_rust_pagerank = getattr(_khora_accel, "pagerank", None)
_rust_pairwise_cosine = getattr(_khora_accel, "pairwise_cosine_above_threshold", None)
_rust_rrf = getattr(_khora_accel, "reciprocal_rank_fusion", None)
_rust_resolve_entities_batch = getattr(_khora_accel, "resolve_entities_batch", None)
_rust_resolve_entities_enhanced = getattr(_khora_accel, "resolve_entities_enhanced", None)
_rust_score_entropy = getattr(_khora_accel, "score_entropy", None)
_rust_sequence_match = getattr(_khora_accel, "sequence_match_ratio", None)
_rust_weighted_rrf = getattr(_khora_accel, "weighted_rrf", None)
_rust_weighted_rrf_normalized = getattr(_khora_accel, "weighted_rrf_normalized", None)
_rust_weighted_rrf_normalized_with_diagnostics = getattr(_khora_accel, "weighted_rrf_normalized_with_diagnostics", None)
_rust_weighted_rrf_normalized_with_provenance = getattr(_khora_accel, "weighted_rrf_normalized_with_provenance", None)

# When the wheel is present but partially importable (stale/partial build), warn
# and name the missing symbols so the operator knows their wheel is stale. A
# wholly-absent wheel is the normal no-accel install and only logs INFO below.
# Suppressed when the operator has explicitly forced Rust off (numpy/python):
# they opted out, so the stale-wheel notice would just be misleading noise.
if _HAS_RUST and _FORCE_BACKEND not in ("numpy", "python"):
    _missing_rust_symbols = sorted(attr for attr in _RUST_SYMBOLS.values() if getattr(_khora_accel, attr, None) is None)
    if _missing_rust_symbols:
        logger.warning(
            "khora-accel wheel is present but missing {} expected symbol(s): {}. "
            "The affected kernels will use the numpy/pure-Python fallback. This "
            "usually means the installed wheel is stale - rebuild/reinstall "
            "khora-accel to restore full acceleration.",
            len(_missing_rust_symbols),
            ", ".join(_missing_rust_symbols),
        )

# ---------------------------------------------------------------------------
# Tier 1: NumPy / RapidFuzz (existing)
# ---------------------------------------------------------------------------

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

try:
    from rapidfuzz.distance import JaroWinkler as _rf_jw
    from rapidfuzz.distance import Levenshtein as _rf_lev

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False

# ---------------------------------------------------------------------------
# Apply runtime backend override
# ---------------------------------------------------------------------------

if _FORCE_BACKEND == "python":
    _HAS_RUST = False
    _HAS_NUMPY = False
    _HAS_RAPIDFUZZ = False
    for _name in _RUST_SYMBOLS:
        globals()[_name] = None
elif _FORCE_BACKEND == "numpy":
    _HAS_RUST = False
    for _name in _RUST_SYMBOLS:
        globals()[_name] = None
# "rust" or unset: use auto-detected fastest path

# ---------------------------------------------------------------------------
# Log active backend
# ---------------------------------------------------------------------------

if _HAS_RUST:
    logger.info("Khora acceleration backend: rust (khora-accel)")
elif _HAS_NUMPY:
    logger.info("Khora acceleration backend: numpy/rapidfuzz")
else:
    logger.info("Khora acceleration backend: pure python")


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses Rust > numpy > pure Python, depending on availability.
    """
    if _HAS_RUST and _rust_cosine is not None:
        return _rust_cosine(vec1, vec2)

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
    """
    if len(candidates) == 0:
        return []

    if _HAS_RUST and _rust_batch_cosine is not None and _HAS_NUMPY:
        q = np.asarray(query, dtype=np.float32)
        mat = np.asarray(candidates, dtype=np.float32)
        return _rust_batch_cosine(q, mat, threshold)

    if _HAS_NUMPY:
        q = np.asarray(query, dtype=np.float32)
        mat = np.asarray(candidates, dtype=np.float32)

        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        norms = np.linalg.norm(mat, axis=1)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        sims = (mat @ q) / (safe_norms * q_norm)
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


def pairwise_cosine_above_threshold(
    embeddings: list[list[float]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """All-pairs cosine similarity above a threshold.

    Returns (i, j, similarity) triples where i < j and similarity >= threshold.
    Uses Rust (rayon parallel) > numpy > pure Python.
    """
    if _HAS_RUST and _rust_pairwise_cosine is not None and _HAS_NUMPY:
        mat = np.asarray(embeddings, dtype=np.float32)
        return _rust_pairwise_cosine(mat, threshold)

    n = len(embeddings)
    if n < 2:
        return []

    if _HAS_NUMPY:
        mat = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1)
        results = []
        for i in range(n):
            if norms[i] == 0.0:
                continue
            for j in range(i + 1, n):
                if norms[j] == 0.0:
                    continue
                sim = float(np.dot(mat[i], mat[j]) / (norms[i] * norms[j]))
                if sim >= threshold:
                    results.append((i, j, sim))
        return results

    # Pure-Python fallback
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                results.append((i, j, sim))
    return results


# ---------------------------------------------------------------------------
# Pairwise similarity with name-token-prefix blocking
# (dream-phase cross-batch entity resolution, see issue #663)
# ---------------------------------------------------------------------------

_TOKEN_PREFIX_LEN = 3


def _name_prefix_keys(name: str) -> set[str]:
    """Token-prefix blocking keys for an entity name.

    Lowercase, split on every non-alphanumeric character, drop tokens of
    length < 2, then take the first three characters of each surviving
    token. Mirrors the Rust implementation byte-for-byte.
    """
    keys: set[str] = set()
    current: list[str] = []
    for ch in name:
        if ch.isalnum():
            current.append(ch.lower())
        elif current:
            if len(current) >= 2:
                keys.add("".join(current[:_TOKEN_PREFIX_LEN]))
            current.clear()
    if current and len(current) >= 2:
        keys.add("".join(current[:_TOKEN_PREFIX_LEN]))
    return keys


def block_and_score_pairs(
    embeddings,  # numpy ndarray or 2D sequence
    names: list[str],
    threshold: float,
    *,
    name_token_blocking: bool = True,
) -> list[tuple[int, int, float]]:
    """Pairwise cosine similarity with optional name-token-prefix blocking.

    For cross-batch entity resolution: returns every `(i, j, similarity)`
    triple (with `i < j`) where the cosine similarity between
    `embeddings[i]` and `embeddings[j]` is at least `threshold`, optionally
    filtered to pairs whose entity names share at least one blocking key.

    A blocking key is the first three characters of a token, where tokens
    are lowercase, alphanumeric, length >= 2 substrings of a name. With
    `name_token_blocking=True`, this cuts the candidate set from
    O(N^2) to roughly O(N * average_block_size), ~10-100x on a realistic
    name distribution.

    Args:
        embeddings: `(N, D)` pre-normalised float32 matrix. Pre-normalised
            means cosine == dot product — the kernel does not re-normalise.
        names: length-N list of entity names, one per row. Names are not
            re-normalised — case and punctuation are tokenised raw.
        threshold: cosine similarity threshold in [-1.0, 1.0]. Only pairs
            with similarity >= threshold are returned.
        name_token_blocking: when True (default), apply token-prefix
            blocking. When False, the output is identical to
            `pairwise_cosine_above_threshold(embeddings, threshold)` over
            pre-normalised embeddings.

    Returns:
        `[(i, j, similarity), ...]` with `i < j`. No deterministic global
        sort — callers that need one should sort the result.

    Raises:
        ValueError: if `embeddings.ndim != 2` or `len(names) != N`.

    Uses Rust (rayon parallel) > numpy > pure Python.
    """
    if _HAS_RUST and _rust_block_and_score_pairs is not None and _HAS_NUMPY:
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got ndim={mat.ndim}")
        if len(names) != mat.shape[0]:
            raise ValueError(f"names length ({len(names)}) does not match embeddings rows ({mat.shape[0]})")
        return _rust_block_and_score_pairs(mat, names, float(threshold), bool(name_token_blocking))

    # Numpy / pure-Python fallback
    if _HAS_NUMPY:
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got ndim={mat.ndim}")
        n = mat.shape[0]
    else:
        mat = embeddings  # treat as list-of-lists
        n = len(mat)
    if len(names) != n:
        raise ValueError(f"names length ({len(names)}) does not match embeddings rows ({n})")
    if n < 2:
        return []

    if name_token_blocking:
        row_keys = [_name_prefix_keys(name) for name in names]
        inverted: dict[str, list[int]] = {}
        for i, keys in enumerate(row_keys):
            for k in keys:
                inverted.setdefault(k, []).append(i)
        results: list[tuple[int, int, float]] = []
        for i in range(n):
            keys = row_keys[i]
            if not keys:
                continue
            candidates: set[int] = set()
            for k in keys:
                for j in inverted.get(k, ()):
                    if j > i:
                        candidates.add(j)
            for j in sorted(candidates):
                if _HAS_NUMPY:
                    sim = float(np.dot(mat[i], mat[j]))
                else:
                    sim = sum(a * b for a, b in zip(mat[i], mat[j]))
                if sim >= threshold:
                    results.append((i, j, sim))
        return results

    # Unblocked path: parity with pairwise_cosine_above_threshold over
    # pre-normalised embeddings (dot == cosine).
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            if _HAS_NUMPY:
                sim = float(np.dot(mat[i], mat[j]))
            else:
                sim = sum(a * b for a, b in zip(mat[i], mat[j]))
            if sim >= threshold:
                results.append((i, j, sim))
    return results


# ---------------------------------------------------------------------------
# Levenshtein similarity
# ---------------------------------------------------------------------------


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1.0 = identical).

    Uses Rust > rapidfuzz > pure Python.
    """
    if _HAS_RUST and _rust_levenshtein is not None:
        return _rust_levenshtein(s1, s2)

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

    Uses Rust > rapidfuzz > difflib, in order of preference.

    All backends use Jaro-Winkler similarity so scores agree regardless of which
    is active: entity-resolution thresholds are calibrated for Jaro-Winkler, and
    the rapidfuzz fallback previously used fuzz.ratio (Indel/LCS), which scored
    prefix-overlapping name variants lower and left them unmerged. The difflib
    last resort still diverges slightly (no Jaro-Winkler in the stdlib).
    """
    if _HAS_RUST and _rust_sequence_match is not None:
        return _rust_sequence_match(s1, s2)

    if _HAS_RAPIDFUZZ:
        return _rf_jw.similarity(s1, s2)

    from difflib import SequenceMatcher

    return SequenceMatcher(None, s1, s2).ratio()


# ---------------------------------------------------------------------------
# Batch string operations
# ---------------------------------------------------------------------------


def batch_levenshtein(
    query: str,
    candidates: list[str],
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """Score query against all candidates using Levenshtein similarity.

    Returns (index, similarity) pairs above threshold, sorted descending.
    Uses Rust parallelism when available, otherwise falls back to serial loop.
    """
    if _HAS_RUST and _rust_batch_levenshtein is not None:
        return _rust_batch_levenshtein(query, candidates, threshold)

    results = []
    for i, cand in enumerate(candidates):
        s = levenshtein_similarity(query, cand)
        if s >= threshold:
            results.append((i, s))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def batch_sequence_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """Score query against all candidates using sequence match ratio.

    Returns (index, similarity) pairs above threshold, sorted descending.
    Uses Rust parallelism when available, otherwise falls back to serial loop.
    """
    if _HAS_RUST and _rust_batch_sequence_match is not None:
        return _rust_batch_sequence_match(query, candidates, threshold)

    results = []
    for i, cand in enumerate(candidates):
        s = sequence_match_ratio(query, cand)
        if s >= threshold:
            results.append((i, s))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------


def _top_k_indices(scores: list[float], k: int) -> list[int]:
    """Indices of the top-``k`` scored nodes: score descending, ties by index.

    ``heapq.nsmallest(k, ..., key=lambda i: (-scores[i], i))`` is the documented
    O(n log k) equivalent of ``sorted(range(n), key=lambda i: (-scores[i], i))[:k]``
    — cheap enough to run every iteration. Kept byte-for-byte equivalent to the
    Rust ``top_k_indices`` helper (which uses ``select_nth`` for the same result)
    so the Rust and pure-Python paths stop the #1476 early-stop on the same
    iteration.
    """
    k = min(k, len(scores))
    if k <= 0:
        return []
    return heapq.nsmallest(k, range(len(scores)), key=lambda i: (-scores[i], i))


def pagerank(
    n: int,
    edges: list[tuple[int, int, float]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    personalization: list[float] | None = None,
    rank_k: int | None = None,
    stable_iters: int = 3,
) -> list[float]:
    """Compute PageRank scores on a weighted directed graph.

    When ``personalization`` is provided, computes Personalized PageRank
    (PPR) — the teleport distribution becomes the supplied vector rather
    than uniform. Negatives are clipped to 0; if the result sums to 0
    (or the length doesn't match ``n``), falls back to uniform — never
    raises, so a query-time caller never crashes on a malformed seed.

    Args:
        n: Number of nodes (IDs are 0..n-1).
        edges: (src, dst, weight) triples.
        damping: Damping factor (typically 0.85).
        max_iter: Maximum iterations.
        tol: Convergence threshold.
        personalization: Optional seed distribution of length ``n``.
            None / mismatched length / all-zero → uniform (standard PageRank).
        rank_k: Optional top-k rank-stability early-stop (#1476). When set, the
            power iteration also halts once the ordering of the top-``rank_k``
            nodes (score desc, index asc) is unchanged for ``stable_iters``
            consecutive iterations. ``None`` keeps the pure global-L1 behaviour.
        stable_iters: Patience for the ``rank_k`` early-stop. Ignored when
            ``rank_k`` is ``None`` or ``0``.

    Returns:
        List of length n with PageRank scores indexed by node ID.
    """
    if _HAS_RUST and _rust_pagerank is not None:
        # Only forward the personalization arg when set so an older
        # khora-accel wheel that still has the 5-arg signature keeps
        # working until it's rebuilt. New wheels accept either path.
        if rank_k is not None:
            # Early-stop requires the #1476 kernel (8-arg signature). If the
            # installed wheel predates it, the call raises TypeError and we fall
            # through to the pre-#1476 call below (correct, just no early-stop).
            try:
                return _rust_pagerank(n, edges, damping, max_iter, tol, personalization, rank_k, stable_iters)
            except TypeError:
                pass
        if personalization is None:
            return _rust_pagerank(n, edges, damping, max_iter, tol)
        return _rust_pagerank(n, edges, damping, max_iter, tol, personalization)

    # Pure-Python fallback
    if n == 0:
        return []

    # Resolve teleport distribution `p` (mirrors the Rust validation).
    uniform = 1.0 / n
    if personalization is not None and len(personalization) == n:
        clipped = [x if x > 0.0 else 0.0 for x in personalization]
        total = sum(clipped)
        p = [x / total for x in clipped] if total > 0.0 else [uniform] * n
    else:
        p = [uniform] * n

    incoming: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    out_degree: list[float] = [0.0] * n

    for src, dst, weight in edges:
        if 0 <= src < n and 0 <= dst < n:
            incoming[dst].append((src, weight))
            out_degree[src] += weight

    scores = list(p)  # init from p so the first iteration is already seeded

    # Top-k rank-stability early-stop state (#1476). Mirrors pagerank.rs.
    capped_rank_k = min(rank_k, n) if rank_k is not None else None
    prev_top: list[int] = []
    stable_count = 0

    for _ in range(max_iter):
        new_scores = [0.0] * n
        diff = 0.0

        for node in range(n):
            contrib = 0.0
            for src, weight in incoming[node]:
                if out_degree[src] > 0:
                    contrib += scores[src] * weight / out_degree[src]
            new_score = (1.0 - damping) * p[node] + damping * contrib
            diff += abs(new_score - scores[node])
            new_scores[node] = new_score

        scores = new_scores
        if diff < tol:
            break

        # Halt once the top-k ordering has been stable for `stable_iters`
        # consecutive iterations. Checked after the global-L1 test so
        # rank_k=None reproduces the pre-#1476 behaviour exactly.
        if capped_rank_k is not None and capped_rank_k > 0 and stable_iters > 0:
            top = _top_k_indices(scores, capped_rank_k)
            if top == prev_top:
                stable_count += 1
                if stable_count >= stable_iters:
                    break
            else:
                stable_count = 0
            prev_top = top

    return scores


def build_chunk_edges(
    n_chunks: int,
    keyword_chunk_ids: list[list[int]],
    idf_scores: list[float],
) -> list[tuple[int, int, float]]:
    """Build chunk-to-chunk edges from keyword co-occurrence.

    For each keyword, creates bidirectional edges among all chunks sharing
    that keyword, weighted by the keyword's IDF score.

    Args:
        n_chunks: Total number of chunks.
        keyword_chunk_ids: For each keyword, the list of chunk indices containing it.
        idf_scores: IDF score per keyword (parallel to keyword_chunk_ids).

    Returns:
        Flat edge list of (src, dst, weight) triples (bidirectional).
    """
    if _HAS_RUST and _rust_build_chunk_edges is not None:
        return _rust_build_chunk_edges(n_chunks, keyword_chunk_ids, idf_scores)

    # Pure-Python fallback
    edges: list[tuple[int, int, float]] = []
    for keyword_idx, chunk_ids in enumerate(keyword_chunk_ids):
        weight = idf_scores[keyword_idx] if keyword_idx < len(idf_scores) else 0.0
        for i in range(len(chunk_ids)):
            cid1 = chunk_ids[i]
            if cid1 >= n_chunks:
                continue
            for j in range(i + 1, len(chunk_ids)):
                cid2 = chunk_ids[j]
                if cid2 >= n_chunks:
                    continue
                edges.append((cid1, cid2, weight))
                edges.append((cid2, cid1, weight))
    return edges


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_SKELETON_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "we",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "can",
    }
)

_KEYWORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")


def extract_keywords(content: str) -> list[str]:
    """Extract unique keywords from content.

    Tokenises with ``\\b[a-zA-Z]{3,}\\b``, removes stopwords, deduplicates.
    Uses Rust when available, otherwise pure Python.
    """
    if _HAS_RUST and _rust_extract_keywords is not None:
        return _rust_extract_keywords(content)

    lower = content.lower()
    seen: set[str] = set()
    keywords: list[str] = []
    for m in _KEYWORD_RE.finditer(lower):
        word = m.group()
        if word not in _SKELETON_STOPWORDS and word not in seen:
            seen.add(word)
            keywords.append(word)
    return keywords


def extract_keywords_batch(contents: list[str]) -> list[list[str]]:
    """Batch keyword extraction.

    Uses Rust (rayon parallel) when available, otherwise serial Python.
    """
    if _HAS_RUST and _rust_extract_keywords_batch is not None:
        return _rust_extract_keywords_batch(contents)

    return [extract_keywords(c) for c in contents]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (low-level, string-ID based)
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Basic Reciprocal Rank Fusion over string ID lists.

    For each list the score contribution is ``1 / (k + rank)`` where rank
    is 1-indexed.  Returns ``(id, score)`` pairs sorted descending.

    Note: The higher-level ``engines.vectorcypher.fusion`` module wraps this
    with rich FusedResult metadata tracking. Use this for raw score computation.
    """
    if _HAS_RUST and _rust_rrf is not None:
        return _rust_rrf(ranked_lists, k)

    scores: dict[str, float] = {}
    for item_list in ranked_lists:
        for rank_0, item_id in enumerate(item_list):
            rank = rank_0 + 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return results


def weighted_rrf(
    ranked_lists: list[tuple[float, list[str]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion.

    Each entry is ``(weight, [id, ...])``.  Score contribution per item:
    ``weight / (k + rank)`` where rank is 1-indexed.
    Returns ``(id, score)`` pairs sorted descending.
    """
    if _HAS_RUST and _rust_weighted_rrf is not None:
        return _rust_weighted_rrf(ranked_lists, k)

    scores: dict[str, float] = {}
    for weight, item_list in ranked_lists:
        for rank_0, item_id in enumerate(item_list):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank_0 + 1)

    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return results


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to ``[0, 1]``.

    If all scores are identical, returns a list of ``1.0``.
    """
    if _HAS_RUST and _rust_normalize_scores is not None:
        return _rust_normalize_scores(scores)

    if not scores:
        return scores

    min_s = min(scores)
    max_s = max(scores)
    if abs(max_s - min_s) < 1e-15:
        return [1.0] * len(scores)
    rng = max_s - min_s
    return [(s - min_s) / rng for s in scores]


def weighted_rrf_normalized(
    vector_results: list[tuple[str, float]],
    graph_results: list[tuple[str, float]],
    k: int = 60,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list[tuple[str, float]]:
    """Weighted RRF with score normalization. Rust > Python fallback."""
    if _HAS_RUST and _rust_weighted_rrf_normalized is not None:
        return _rust_weighted_rrf_normalized(vector_results, graph_results, k, vector_weight, graph_weight)

    # Python fallback - same logic as Rust
    scores: dict[str, float] = {}
    contributions: dict[str, float] = {}

    if vector_results:
        raw = [s for _, s in vector_results]
        normalized = normalize_scores(raw)
        for rank_0, ((item_id, _), norm) in enumerate(zip(vector_results, normalized)):
            scores[item_id] = scores.get(item_id, 0.0) + vector_weight / (k + rank_0 + 1)
            contributions[item_id] = contributions.get(item_id, 0.0) + vector_weight * norm * 0.01

    if graph_results:
        raw = [s for _, s in graph_results]
        normalized = normalize_scores(raw)
        for rank_0, ((item_id, _), norm) in enumerate(zip(graph_results, normalized)):
            scores[item_id] = scores.get(item_id, 0.0) + graph_weight / (k + rank_0 + 1)
            contributions[item_id] = contributions.get(item_id, 0.0) + graph_weight * norm * 0.01

    results = [(id, scores[id] + contributions.get(id, 0.0)) for id in scores]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def weighted_rrf_normalized_with_provenance(
    vector_results: list[tuple[str, float]],
    graph_results: list[tuple[str, float]],
    k: int = 60,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list[tuple[str, float, int]]:
    """Weighted RRF with score normalization and source provenance.

    Returns ``(id, combined_score, source_bitmap)`` tuples where
    *source_bitmap* is a ``u8`` flag: 1 = vector only, 2 = graph only,
    3 = both.  Rust > Python fallback.
    """
    if _HAS_RUST and _rust_weighted_rrf_normalized_with_provenance is not None:
        return _rust_weighted_rrf_normalized_with_provenance(
            vector_results, graph_results, k, vector_weight, graph_weight
        )

    # Python fallback — same logic as Rust
    scores: dict[str, float] = {}
    contributions: dict[str, float] = {}
    sources: dict[str, int] = {}

    if vector_results:
        raw = [s for _, s in vector_results]
        normalized = normalize_scores(raw)
        for rank_0, ((item_id, _), norm) in enumerate(zip(vector_results, normalized)):
            scores[item_id] = scores.get(item_id, 0.0) + vector_weight / (k + rank_0 + 1)
            contributions[item_id] = contributions.get(item_id, 0.0) + vector_weight * norm * 0.01
            sources[item_id] = sources.get(item_id, 0) | 0x01

    if graph_results:
        raw = [s for _, s in graph_results]
        normalized = normalize_scores(raw)
        for rank_0, ((item_id, _), norm) in enumerate(zip(graph_results, normalized)):
            scores[item_id] = scores.get(item_id, 0.0) + graph_weight / (k + rank_0 + 1)
            contributions[item_id] = contributions.get(item_id, 0.0) + graph_weight * norm * 0.01
            sources[item_id] = sources.get(item_id, 0) | 0x02

    results = [(id_, scores[id_] + contributions.get(id_, 0.0), sources.get(id_, 0)) for id_ in scores]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def weighted_rrf_normalized_with_diagnostics(
    vector_results: list[tuple[str, float]],
    graph_results: list[tuple[str, float]],
    k: int = 60,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list[tuple[str, float, int, int, int, float, float, float, float]]:
    """Weighted RRF with full per-result diagnostics.

    Returns ``(id, score, source, vector_rank, graph_rank,
    vector_norm_score, graph_norm_score, vector_rrf_contrib,
    graph_rrf_contrib)`` tuples sorted descending by score.
    Rust > Python fallback.
    """
    if _HAS_RUST and _rust_weighted_rrf_normalized_with_diagnostics is not None:
        return _rust_weighted_rrf_normalized_with_diagnostics(
            vector_results, graph_results, k, vector_weight, graph_weight
        )

    # Python fallback
    scores: dict[str, float] = {}
    sources: dict[str, int] = {}
    v_ranks: dict[str, int] = {}
    g_ranks: dict[str, int] = {}
    v_norms: dict[str, float] = {}
    g_norms: dict[str, float] = {}
    v_contribs: dict[str, float] = {}
    g_contribs: dict[str, float] = {}

    if vector_results:
        raw = [s for _, s in vector_results]
        normalized = normalize_scores(raw)
        for rank_0, ((item_id, _), norm) in enumerate(zip(vector_results, normalized)):
            rank = rank_0 + 1
            rrf_c = vector_weight / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_c + vector_weight * norm * 0.01
            sources[item_id] = sources.get(item_id, 0) | 0x01
            v_ranks[item_id] = rank
            v_norms[item_id] = norm
            v_contribs[item_id] = rrf_c

    if graph_results:
        raw = [s for _, s in graph_results]
        normalized = normalize_scores(raw)
        for rank_0, ((item_id, _), norm) in enumerate(zip(graph_results, normalized)):
            rank = rank_0 + 1
            rrf_c = graph_weight / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_c + graph_weight * norm * 0.01
            sources[item_id] = sources.get(item_id, 0) | 0x02
            g_ranks[item_id] = rank
            g_norms[item_id] = norm
            g_contribs[item_id] = rrf_c

    results = [
        (
            id_,
            scores[id_],
            sources.get(id_, 0),
            v_ranks.get(id_, 0),
            g_ranks.get(id_, 0),
            v_norms.get(id_, 0.0),
            g_norms.get(id_, 0.0),
            v_contribs.get(id_, 0.0),
            g_contribs.get(id_, 0.0),
        )
        for id_ in scores
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def batch_score_stats(
    scores: list[float],
) -> tuple[float, float, float, float, float]:
    """Compute score statistics: (mean, std_dev, min, max, median).

    Rust > Python fallback. Empty input returns all zeros.
    """
    if _HAS_RUST and _rust_batch_score_stats is not None:
        return _rust_batch_score_stats(scores)

    if not scores:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std_dev = math.sqrt(variance)
    min_s = min(scores)
    max_s = max(scores)
    sorted_s = sorted(scores)
    if n % 2 == 0:
        median = (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2.0
    else:
        median = sorted_s[n // 2]
    return (mean, std_dev, min_s, max_s, median)


def score_entropy(scores: list[float]) -> float:
    """Compute Shannon entropy of a score distribution.

    Normalizes scores to a probability distribution and returns
    ``-sum(p * ln(p))``. Higher entropy = more uniform = less confident.
    Rust > Python fallback. Returns 0.0 for empty or all-zero inputs.
    """
    if _HAS_RUST and _rust_score_entropy is not None:
        return _rust_score_entropy(scores)

    if not scores:
        return 0.0

    total = sum(s for s in scores if s > 0)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for s in scores:
        if s > 0:
            p = s / total
            entropy -= p * math.log(p)
    return entropy


# ---------------------------------------------------------------------------
# Entity name normalization
# ---------------------------------------------------------------------------

_HONORIFICS = ("mr.", "mrs.", "ms.", "dr.", "prof.", "sir ", "lord ", "lady ")


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for dedup. Rust > Python."""
    if _HAS_RUST and _rust_normalize_entity_name is not None:
        return _rust_normalize_entity_name(name)
    result = name.lower()
    for h in _HONORIFICS:
        if result.startswith(h):
            result = result[len(h) :]
            break
    result = " ".join(result.split())
    result = result.strip().strip(".,;:!?\"'()-[]{}").strip()
    return result


def normalize_entity_names_batch(names: list[str]) -> list[str]:
    """Batch normalize entity names. Rust > Python."""
    if _HAS_RUST and _rust_normalize_entity_names_batch is not None:
        return _rust_normalize_entity_names_batch(names)
    return [normalize_entity_name(n) for n in names]


# ---------------------------------------------------------------------------
# Entity resolution (batch)
# ---------------------------------------------------------------------------


def resolve_entities_batch(
    new_names: list[str],
    existing_names: list[str],
    existing_aliases: list[list[str]],
    threshold: float = 0.85,
) -> list[tuple[int, float, str] | None]:
    """Resolve a batch of new entity names against existing entities.

    For each new name, attempts matching in order:
    1. Exact match — case-insensitive against existing names
    2. Alias match — case-insensitive against each entity's aliases
    3. Fuzzy match — normalized Levenshtein above *threshold*

    Returns a list parallel to *new_names*. Each element is either
    ``(existing_index, score, match_type)`` or ``None``.
    """
    if _HAS_RUST and _rust_resolve_entities_batch is not None:
        return _rust_resolve_entities_batch(new_names, existing_names, existing_aliases, threshold)

    # Pure-Python fallback
    existing_lower = [n.lower() for n in existing_names]
    aliases_lower = [[a.lower() for a in aliases] for aliases in existing_aliases]

    results: list[tuple[int, float, str] | None] = []
    for new_name in new_names:
        query = new_name.lower()
        matched = False

        # Step 1: Exact name match
        for idx, existing in enumerate(existing_lower):
            if query == existing:
                results.append((idx, 1.0, "exact"))
                matched = True
                break

        if matched:
            continue

        # Step 2: Alias match
        for idx, aliases in enumerate(aliases_lower):
            for alias in aliases:
                if query == alias:
                    results.append((idx, 1.0, "alias"))
                    matched = True
                    break
            if matched:
                break

        if matched:
            continue

        # Step 3: Fuzzy match
        best_idx = None
        best_score = threshold
        for idx, existing in enumerate(existing_lower):
            if not query or not existing:
                continue
            sim = levenshtein_similarity(query, existing)
            if sim > best_score:
                best_score = sim
                best_idx = idx

        if best_idx is not None:
            results.append((best_idx, best_score, "fuzzy"))
        else:
            results.append(None)

    return results


def resolve_entities_enhanced(
    new_names: list[str],
    new_types: list[str],
    existing_names: list[str],
    existing_aliases: list[list[str]],
    existing_types: list[str],
    type_thresholds: dict[str, float] | None = None,
    default_threshold: float = 0.85,
) -> list[tuple[int, float, str] | None]:
    """Enhanced entity resolution using Jaro-Winkler + token overlap + per-type thresholds.

    For each new entity (name + type), attempts matching against same-type existing entities:
    1. Exact match — case-insensitive name equality (score 1.0)
    2. Alias match — case-insensitive alias equality (score 1.0)
    3. Enhanced fuzzy — combined Jaro-Winkler (0.6) + token overlap (0.4),
       checked against the per-type threshold.

    Note: Thresholds (PERSON 0.92, DATE 0.95, default 0.85) are calibrated for
    Jaro-Winkler scores produced by the Rust backend. The Python fallback also
    uses Jaro-Winkler via :func:`sequence_match_ratio`, so thresholds are
    consistent across backends.

    Args:
        new_names: Names of new entities to resolve.
        new_types: Entity types parallel to *new_names* (e.g. "PERSON").
        existing_names: Names of existing entities.
        existing_aliases: Aliases for each existing entity.
        existing_types: Types of existing entities (parallel to *existing_names*).
        type_thresholds: Per-type merge thresholds (e.g. {"PERSON": 0.92}).
        default_threshold: Fallback threshold for types not in *type_thresholds*.

    Returns:
        List parallel to *new_names*. Each element is either
        ``(existing_index, score, match_type)`` or ``None``.
    """
    type_thresholds = type_thresholds or {}

    if _HAS_RUST and _rust_resolve_entities_enhanced is not None:
        keys = list(type_thresholds.keys())
        vals = [type_thresholds[k] for k in keys]
        return _rust_resolve_entities_enhanced(
            new_names,
            new_types,
            existing_names,
            existing_aliases,
            existing_types,
            keys,
            vals,
            default_threshold,
        )

    # Pure-Python fallback
    existing_lower = [n.lower() for n in existing_names]
    aliases_lower = [[a.lower() for a in aliases] for aliases in existing_aliases]
    existing_types_upper = [t.upper() for t in existing_types]

    results: list[tuple[int, float, str] | None] = []
    for new_name, new_type in zip(new_names, new_types):
        query = new_name.lower()
        query_type = new_type.upper()
        threshold = type_thresholds.get(query_type, default_threshold)
        matched = False

        # Step 1: Exact name match (same type)
        for idx, existing in enumerate(existing_lower):
            if existing_types_upper[idx] == query_type and query == existing:
                results.append((idx, 1.0, "exact"))
                matched = True
                break

        if matched:
            continue

        # Step 2: Alias match (same type)
        for idx, aliases in enumerate(aliases_lower):
            if existing_types_upper[idx] != query_type:
                continue
            for alias in aliases:
                if query == alias:
                    results.append((idx, 1.0, "alias"))
                    matched = True
                    break
            if matched:
                break

        if matched:
            continue

        # Step 3: Enhanced fuzzy — Jaro-Winkler + token overlap
        best_idx = None
        best_score = threshold
        for idx, existing in enumerate(existing_lower):
            if existing_types_upper[idx] != query_type:
                continue
            if not query or not existing:
                continue
            jw = sequence_match_ratio(query, existing)  # Jaro-Winkler via _accel
            # Token overlap
            q_tokens = set(query.split())
            e_tokens = set(existing.split())
            if q_tokens and e_tokens:
                tok = len(q_tokens & e_tokens) / max(len(q_tokens), len(e_tokens))
            else:
                tok = 0.0
            combined = 0.6 * jw + 0.4 * tok
            if combined > best_score:
                best_score = combined
                best_idx = idx

        if best_idx is not None:
            results.append((best_idx, best_score, "fuzzy"))
        else:
            results.append(None)

    return results


# ---------------------------------------------------------------------------
# Temporal filtering (batch operations)
# ---------------------------------------------------------------------------

_SECONDS_PER_DAY = 86400.0
_LN_HALF = -0.6931471805599453  # math.log(0.5)


def batch_temporal_filter(
    timestamps_secs: list[float],
    operator: str,
    start_secs: float | None = None,
    end_secs: float | None = None,
) -> list[bool]:
    """Batch temporal filter: test each timestamp against a temporal range.

    Args:
        timestamps_secs: Epoch seconds for each item.
        operator: One of "before", "after", "between".
        start_secs: Start boundary (epoch seconds), or None.
        end_secs: End boundary (epoch seconds), or None.

    Returns:
        Boolean mask — True if the timestamp matches the filter.
    """
    if _HAS_RUST and _rust_batch_temporal_filter is not None:
        return _rust_batch_temporal_filter(timestamps_secs, operator, start_secs, end_secs)

    # Pure-Python fallback
    results: list[bool] = []
    for ts in timestamps_secs:
        if operator == "before":
            results.append(end_secs is None or ts < end_secs)
        elif operator == "after":
            results.append(start_secs is None or ts > start_secs)
        elif operator == "between":
            after_start = start_secs is None or ts >= start_secs
            before_end = end_secs is None or ts <= end_secs
            results.append(after_start and before_end)
        else:
            results.append(True)
    return results


def batch_recency_scores(
    timestamps_secs: list[float],
    now_secs: float,
    decay_days: float,
    recency_weight: float,
) -> list[float]:
    """Batch recency scores using exponential decay.

    For each timestamp computes:
        (1 - recency_weight) + recency_weight * 0.5^(age_days / decay_days)

    Args:
        timestamps_secs: Epoch seconds for each item.
        now_secs: Current time as epoch seconds.
        decay_days: Half-life in days.
        recency_weight: Blending weight (0 = no bias, 1 = full decay).

    Returns:
        Recency score per timestamp.
    """
    if _HAS_RUST and _rust_batch_recency_scores is not None:
        return _rust_batch_recency_scores(timestamps_secs, now_secs, decay_days, recency_weight)

    # Pure-Python fallback
    if recency_weight == 0.0:
        return [1.0] * len(timestamps_secs)

    base = 1.0 - recency_weight
    decay_factor = _LN_HALF / decay_days if decay_days > 0 else 0.0

    scores: list[float] = []
    for ts in timestamps_secs:
        # Clamp future timestamps (clock skew, deliberate forward-dating) to
        # age=0 so a forward-dated chunk gets full freshness rather than
        # decay > 1.0 from math.exp(positive). Mirrors the `max(0, ...)`
        # clamp in chronicle/engine.py's `_apply_temporal_decay`.
        age_days = max(0.0, (now_secs - ts) / _SECONDS_PER_DAY)
        decay = math.exp(decay_factor * age_days)
        scores.append(base + recency_weight * decay)
    return scores


# ---------------------------------------------------------------------------
# Temporal category detection
# ---------------------------------------------------------------------------
#
# This dictionary is intentionally English-only. The multilingual / paraphrase
# gap from #981 (German queries and paraphrased-English queries collapsing to
# NONE) is handled by the opt-in Tier-2 LLM semantic fallback in
# ``khora.query.temporal_detection.TemporalDetector.detect_async`` -- NOT by
# growing per-language keyword lists here. Decision (#981): a multilingual
# keyword dictionary is NOT warranted. Reasons: (1) it would need a separate
# curated phrase list per language, kept in lockstep across the Rust kernel and
# this Python fallback, which is unbounded maintenance for a long tail of
# languages; (2) it cannot resolve paraphrases ("evolved over time", "timeline
# of milestones") in any language, which the LLM tier handles for free; (3) the
# LLM tier classifies into the correct six-way category, which a keyword list
# only approximates. Keep new English whole-word synonyms here (cheap, Tier-1);
# route everything else to Tier-2.

TEMPORAL_DICTIONARY: dict[int, list[str]] = {
    1: [  # EXPLICIT
        "when ",
        "before ",
        "after ",
        "during ",
        "since ",
        "until ",
        "yesterday",
        "today",
        " ago",
        "january",
        "february",
        "march",
        "april",
        # "may" is whole-word ambiguous (modal verb / proper name); gated to
        # temporal context by _may_is_temporal (#981).
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "last week",
        "last month",
        "last year",
        "last night",
        "last time",
    ],
    2: [  # STATE_QUERY
        "currently",
        "right now",
        "at present",
        "presently",
        "these days",
        "nowadays",
        "at this point",
        "at the moment",
        # Implicit temporal patterns for conversational memory.
        # These catch questions about current state that lack explicit temporal keywords.
        " does he still",
        " does she still",
        " do they still",
        " is he still",
        " is she still",
        " are they still",
        " does it still",
        " is it still",
        " am i still",
        " do i still",
        "'s current ",
        " current job",
        " current role",
        " current position",
        " live now",
        " work now",
        " working now",
        " living now",
        " doing now",
        # Enterprise domain compound current-state patterns.
        # Generic " current " was too broad — triggered on recency lookups.
        " current status",
        " current stage",
        " current state",
        " current health",
        " current deal",
        " current project",
        " current plan",
        " current team",
        # Bare " active " was whole-word ambiguous ("the active variable" —
        # math, not temporal); narrowed to state-query noun phrases (#981).
        "active deal",
        "active deals",
        "active project",
        "active projects",
        "active since",
        "currently active",
        "who is the ",
        "who are the ",
        "up-to-date",
        "up to date",
        "authoritative",
        "most reliable",
        "official record",
    ],
    3: [  # ORDINAL
        "first ",
        " earliest",
        "which came",
        "what came",
        "preceding",
        "following ",
        "subsequent",
        "in what order",
        "chronological",
        "what order did",
        "what sequence",
        "before or after",
        "happened first",
        "closed first",
        "created first",
        "came first",
        "started first",
        "completed first",
    ],
    4: [  # AGGREGATE
        "how many times",
        "how many total",
        "all instances",
        "every time",
        "in total",
        "count of",
        "number of times",
        "how often ",
    ],
    5: [  # RECENCY
        "most recent",
        "newest",
        "recently",
        "latest ",
        # Bare "just " was whole-word ambiguous ("just confirm" — adverb
        # meaning "only/simply"); narrowed to "just"+recency phrases (#981).
        "just now",
        "just released",
        "just announced",
        "just shipped",
        "just launched",
        "just published",
        "just landed",
        "just happened",
        "just finished",
        "just completed",
        "just started",
        "just arrived",
    ],
    6: [  # CHANGE
        "changed",
        "switched",
        "moved to",
        "used to",
        "no longer",
        "anymore",
        "former ",
        "previous ",
        "ex-",
        "updated",
        "replaced",
        "went from",
        "transitioned",
        "turned into ",
        "switched to ",
        "became ",
        "converted to ",
        "went back to ",
    ],
}


# "may" only reads as the month name (EXPLICIT) in a temporal context — when
# preceded by a temporal preposition / number or followed by a number (#981).
# This mirrors the Rust kernel's `may_is_temporal`: bare "may" (modal verb) and
# "May Corp" (proper name) do NOT classify as temporal. Temporal prepositions
# kept in lockstep with `MAY_TEMPORAL_PREPS` in rust/khora-accel/src/temporal.rs.
_MAY_TEMPORAL_RE = re.compile(
    r"\b(?:in|on|by|since|until|before|after|during|early|late|\d+)\b\W*\bmay\b"
    r"|\bmay\s+\d",
    re.IGNORECASE,
)


def _compile_temporal_term(term: str) -> re.Pattern[str]:
    """Compile a dictionary term into a word-boundary-aware regex.

    The dictionary historically used leading/trailing spaces as crude word
    boundary anchors ("when ", " active ", " does she still"). Plain
    substring matching let keywords match inside other words ("changed"
    inside "unchanged", "march" inside "marched"), producing false-positive
    temporal categories (#981). We strip those padding spaces and anchor with
    ``\\b`` only on the side where the term begins/ends with a word character,
    so internal-space phrases and hyphenated terms ("up-to-date", "ex-") keep
    matching while substrings of larger words no longer do.

    The whole-word ambiguous "may" (month vs. modal verb vs. proper name) is
    special-cased to require a temporal context (#981), in parity with the Rust
    kernel's ``may_is_temporal``.
    """
    core = term.strip()
    if core == "may":
        return _MAY_TEMPORAL_RE
    left = r"\b" if core[:1].isalnum() else ""
    right = r"\b" if core[-1:].isalnum() else ""
    return re.compile(left + re.escape(core) + right, re.IGNORECASE)


_TEMPORAL_REGEXES: dict[int, list[re.Pattern[str]]] = {
    cat: [_compile_temporal_term(term) for term in terms] for cat, terms in TEMPORAL_DICTIONARY.items()
}


def detect_temporal_category(query: str) -> int:
    """Detect temporal category of a query.

    Returns category ID: 0=NONE, 1=EXPLICIT, 2=STATE_QUERY, 3=ORDINAL,
    4=AGGREGATE, 5=RECENCY, 6=CHANGE.

    Uses Rust Aho-Corasick when available, otherwise Python word-boundary
    regex matching (parity with the Rust kernel).
    """
    if _HAS_RUST and _rust_detect_temporal_category is not None:
        return _rust_detect_temporal_category(query)

    # Python fallback: word-boundary regex matching (keywords match only as
    # whole words/phrases, not as substrings of larger words).
    best_cat = 0
    for cat, regexes in _TEMPORAL_REGEXES.items():
        for rx in regexes:
            if rx.search(query):
                best_cat = max(best_cat, cat)
                break
    return best_cat


_DATE_PATTERN = re.compile(r"\b\d{4}[-/]\d{1,2}|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b")


def detect_temporal_category_with_confidence(
    query: str,
) -> tuple[int, float, list[str]]:
    """Detect temporal category with confidence score and matched terms.

    Returns ``(category_id, confidence, matched_terms)`` where:
    - ``category_id``: 0=NONE through 6=CHANGE
    - ``confidence``: 0.0–1.0 based on match count and strength
    - ``matched_terms``: list of matched keyword strings

    Uses Rust Aho-Corasick when available, otherwise Python word-boundary
    regex matching (parity with the Rust kernel).
    """
    if _HAS_RUST and _rust_detect_temporal_category_with_confidence is not None:
        return _rust_detect_temporal_category_with_confidence(query)

    # Python fallback: word-boundary regex matching (parity with the Rust
    # kernel and with detect_temporal_category above).
    best_cat = 0
    matched_terms: list[str] = []
    matched_cats: set[int] = set()

    for cat, terms in TEMPORAL_DICTIONARY.items():
        for term, rx in zip(terms, _TEMPORAL_REGEXES[cat], strict=True):
            if rx.search(query):
                if cat > best_cat:
                    best_cat = cat
                matched_cats.add(cat)
                if term not in matched_terms:
                    matched_terms.append(term)

    if best_cat == 0:
        return (0, 0.0, [])

    n_matches = len(matched_terms)
    n_cats = len(matched_cats)
    if n_matches == 1:
        confidence = 0.6
    elif n_matches == 2:
        confidence = 0.85 if n_cats > 1 else 0.8
    else:
        confidence = 0.95

    if _DATE_PATTERN.search(query):
        confidence = min(confidence + 0.1, 1.0)

    return (best_cat, confidence, matched_terms)


# ---------------------------------------------------------------------------
# Embedding normalization and dot product
# ---------------------------------------------------------------------------


def normalize_embeddings_batch(
    vectors: list[list[float]],
) -> list[list[float]]:
    """L2-normalize a batch of embedding vectors.

    Each vector is divided by its L2 norm. Zero vectors are returned as-is.
    Uses Rust (rayon parallel) > numpy > pure Python.

    Args:
        vectors: List of embedding vectors.

    Returns:
        List of L2-normalized vectors.
    """
    if _HAS_RUST and _rust_normalize_embeddings_batch is not None:
        return _rust_normalize_embeddings_batch(vectors)

    if _HAS_NUMPY:
        result = []
        for vec in vectors:
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm == 0.0:
                result.append(vec)
            else:
                result.append((arr / norm).tolist())
        return result

    # Pure-Python fallback
    result = []
    for vec in vectors:
        sq_sum = sum(v * v for v in vec)
        if sq_sum == 0.0:
            result.append(list(vec))
        else:
            norm = math.sqrt(sq_sum)
            result.append([v / norm for v in vec])
    return result


def batch_dot_product(
    query: list[float],
    candidates: list[list[float]],
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """Batch dot product: one query against N candidates (pre-normalized).

    For pre-normalized vectors, dot product equals cosine similarity but
    skips norm computation.

    Returns (index, score) pairs above threshold, sorted descending.
    """
    if len(candidates) == 0:
        return []

    if _HAS_RUST and _rust_batch_dot_product is not None and _HAS_NUMPY:
        q = np.asarray(query, dtype=np.float32)
        mat = np.asarray(candidates, dtype=np.float32)
        return _rust_batch_dot_product(q, mat, threshold)

    if _HAS_NUMPY:
        q = np.asarray(query, dtype=np.float32)
        mat = np.asarray(candidates, dtype=np.float32)
        dots = mat @ q
        results = []
        for i in range(len(dots)):
            s = float(dots[i])
            if s >= threshold:
                results.append((i, s))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # Pure-Python fallback
    results = []
    for i, cand in enumerate(candidates):
        dot = sum(a * b for a, b in zip(query, cand))
        if dot >= threshold:
            results.append((i, dot))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# MMR diversity selection
# ---------------------------------------------------------------------------


def detect_communities(
    n: int,
    edges: list[tuple[int, int, float]],
    resolution: float = 1.0,
    max_iter: int = 10,
) -> list[int]:
    """Detect communities using Louvain-style modularity optimization.

    Args:
        n: Number of nodes (IDs are 0..n-1).
        edges: (src, dst, weight) triples (undirected — provide both directions).
        resolution: Modularity resolution parameter (higher = smaller communities).
        max_iter: Maximum optimization passes.

    Returns:
        Community ID per node (0-indexed, -1 for isolated nodes).
    """
    if _HAS_RUST and _rust_detect_communities is not None:
        return _rust_detect_communities(n, edges, resolution, max_iter)

    # Pure-Python fallback (same Louvain algorithm)
    if n == 0:
        return []

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    strengths: list[float] = [0.0] * n
    total_weight = 0.0

    for src, dst, weight in edges:
        if 0 <= src < n and 0 <= dst < n and src != dst:
            adj[src].append((dst, weight))
            strengths[src] += weight
            total_weight += weight

    m = total_weight / 2.0
    if m == 0.0:
        return [-1] * n

    community: list[int] = list(range(n))
    sigma_tot: list[float] = list(strengths)

    for _ in range(max_iter):
        moved = False

        for i in range(n):
            if strengths[i] == 0.0:
                continue

            ki = strengths[i]
            ci = community[i]

            sigma_tot[ci] -= ki

            k_in: dict[int, float] = {}
            for nb, w in adj[i]:
                cnb = community[nb]
                k_in[cnb] = k_in.get(cnb, 0.0) + w

            k_i_in_ci = k_in.get(ci, 0.0)
            gain_ci = k_i_in_ci / m - resolution * sigma_tot[ci] * ki / (2.0 * m * m)

            best_c = ci
            best_gain = gain_ci

            for c, k_i_in_c in k_in.items():
                if c == ci:
                    continue
                gain = k_i_in_c / m - resolution * sigma_tot[c] * ki / (2.0 * m * m)
                # Deterministic tie-break on the smallest community id (#1131),
                # matching the Rust kernel so both backends agree exactly.
                if gain > best_gain or (gain == best_gain and c < best_c):
                    best_gain = gain
                    best_c = c

            community[i] = best_c
            sigma_tot[best_c] += ki

            if best_c != ci:
                moved = True

        if not moved:
            break

    id_map: dict[int, int] = {}
    result: list[int] = [-1] * n

    for i in range(n):
        if strengths[i] > 0.0:
            c = community[i]
            if c not in id_map:
                id_map[c] = len(id_map)
            result[i] = id_map[c]

    return result


def mmr_diversity_select(
    embeddings: list[list[float]],
    scores: list[float],
    lambda_param: float,
    k: int,
) -> list[int]:
    """Greedy Maximal Marginal Relevance diversity selection.

    Iteratively picks the candidate maximising:
        lambda * relevance - (1 - lambda) * max_similarity_to_selected

    Embeddings should be pre-normalized (dot product = cosine similarity).
    Uses Rust > numpy > pure Python.

    Args:
        embeddings: One embedding vector per candidate.
        scores: Relevance score per candidate.
        lambda_param: Tradeoff (0 = pure diversity, 1 = pure relevance).
        k: Number of items to select.

    Returns:
        Indices of selected items in selection order.
    """
    n = len(embeddings)
    if n == 0 or k == 0:
        return []
    k = min(k, n)

    if _HAS_RUST and _rust_mmr_diversity_select is not None and _HAS_NUMPY:
        mat = np.asarray(embeddings, dtype=np.float32)
        return _rust_mmr_diversity_select(mat, scores, lambda_param, k)

    # NumPy fallback
    if _HAS_NUMPY:
        mat = np.asarray(embeddings, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)
        available = [True] * n
        selected: list[int] = []
        max_sim = np.full(n, -np.inf, dtype=np.float32)
        one_minus_lambda = 1.0 - lambda_param

        for _ in range(k):
            best_idx = -1
            best_mmr = float("-inf")

            for i in range(n):
                if not available[i]:
                    continue
                sim_to_sel = 0.0 if not selected else max(0.0, float(max_sim[i]))
                mmr = lambda_param * float(scores_arr[i]) - one_minus_lambda * sim_to_sel
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            if best_idx < 0:
                break

            available[best_idx] = False
            selected.append(best_idx)

            # Update max similarities
            dots = mat @ mat[best_idx]
            for i in range(n):
                if available[i] and dots[i] > max_sim[i]:
                    max_sim[i] = dots[i]

        return selected

    # Pure-Python fallback
    available = [True] * n
    selected = []
    max_sim = [float("-inf")] * n
    one_minus_lambda = 1.0 - lambda_param

    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    for _ in range(k):
        best_idx = -1
        best_mmr = float("-inf")

        for i in range(n):
            if not available[i]:
                continue
            sim_to_sel = 0.0 if not selected else max(0.0, max_sim[i])
            mmr = lambda_param * scores[i] - one_minus_lambda * sim_to_sel
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx < 0:
            break

        available[best_idx] = False
        selected.append(best_idx)

        # Update max similarities
        for i in range(n):
            if available[i]:
                d = _dot(embeddings[best_idx], embeddings[i])
                if d > max_sim[i]:
                    max_sim[i] = d

    return selected


# ---------------------------------------------------------------------------
# Thread pool configuration
# ---------------------------------------------------------------------------


def configure_thread_pool(num_threads: int = 0, *, mode: str = "query") -> None:
    """Configure the Rust rayon global thread pool.

    Call once during initialization to control parallelism in Rust-accelerated
    operations.  ``0`` means auto with mode-based defaults:
      - ``"query"`` (default): ``num_cpus / 2`` — lower latency for concurrent queries
      - ``"ingest"``: ``num_cpus * 3 / 4`` — higher throughput for batch ingestion
    Subsequent calls are no-ops (rayon only allows one global pool).

    Falls back to a no-op when the Rust accelerator is unavailable.

    Args:
        num_threads: Number of threads for the rayon pool.  0 = auto.
        mode: Workload mode — ``"query"`` or ``"ingest"``.
    """
    if _HAS_RUST and _rust_configure_thread_pool is not None:
        _rust_configure_thread_pool(num_threads, mode)
    else:
        logger.debug("configure_thread_pool: Rust accel unavailable, skipping")


# ---------------------------------------------------------------------------
# Chunk deduplication (MinHash-based near-duplicate detection)
# ---------------------------------------------------------------------------


def _py_deduplicate_chunks(
    chunks: list[str],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> list[tuple[int, int | None]]:
    """Pure-Python fallback for chunk deduplication.

    Uses character-level n-gram Jaccard similarity (slower but functional).
    """
    import hashlib

    n = len(chunks)
    if n == 0:
        return []

    def _char_ngrams(text: str, ng: int = 5) -> set[str]:
        t = text.lower()
        if len(t) < ng:
            return {t} if t else set()
        return {t[i : i + ng] for i in range(len(t) - ng + 1)}

    def _minhash(shingles: set[str], num_perm: int) -> list[int]:
        if not shingles:
            return [2**63 - 1] * num_perm
        sig = []
        for seed in range(num_perm):
            min_h = min(
                int(hashlib.md5(f"{seed}:{s}".encode(), usedforsecurity=False).hexdigest()[:16], 16) for s in shingles
            )
            sig.append(min_h)
        return sig

    # Compute signatures
    shingle_sets = [_char_ngrams(c) for c in chunks]
    signatures = [_minhash(s, num_perm) for s in shingle_sets]

    duplicate_of: list[int | None] = [None] * n

    for i in range(n):
        if duplicate_of[i] is not None:
            continue
        for j in range(i + 1, n):
            if duplicate_of[j] is not None:
                continue
            # Estimate similarity from signatures
            matching = sum(1 for a, b in zip(signatures[i], signatures[j]) if a == b)
            sim = matching / num_perm
            if sim >= threshold:
                duplicate_of[j] = i

    return [(i, duplicate_of[i]) for i in range(n)]


def deduplicate_chunks(
    chunks: list[str],
    threshold: float = 0.85,
    num_perm: int = 64,
) -> list[tuple[int, int | None]]:
    """Detect near-duplicate text chunks using MinHash similarity.

    Returns a list of ``(chunk_index, duplicate_of_index)`` tuples.
    ``duplicate_of_index`` is ``None`` for unique (canonical) chunks, or the
    index of the earlier chunk this one duplicates.

    Uses Rust MinHash + LSH banding when available, falling back to a pure-Python
    character n-gram implementation.

    Args:
        chunks: Text chunks to deduplicate.
        threshold: Jaccard similarity threshold (0.0–1.0). Default 0.85.
        num_perm: Number of MinHash permutations. Default 64.
    """
    if _HAS_RUST and _rust_deduplicate_chunks is not None:
        return _rust_deduplicate_chunks(chunks, threshold, num_perm)
    return _py_deduplicate_chunks(chunks, threshold, num_perm)
