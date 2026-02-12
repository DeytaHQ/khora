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

try:
    from khora_accel import (
        RustBM25Index,
    )
    from khora_accel import batch_cosine_similarity as _rust_batch_cosine
    from khora_accel import batch_levenshtein as _rust_batch_levenshtein
    from khora_accel import batch_sequence_match as _rust_batch_sequence_match
    from khora_accel import build_chunk_edges as _rust_build_chunk_edges
    from khora_accel import cosine_similarity as _rust_cosine
    from khora_accel import extract_keywords as _rust_extract_keywords
    from khora_accel import extract_keywords_batch as _rust_extract_keywords_batch
    from khora_accel import levenshtein_similarity as _rust_levenshtein
    from khora_accel import normalize_entity_name as _rust_normalize_entity_name
    from khora_accel import normalize_entity_names_batch as _rust_normalize_entity_names_batch
    from khora_accel import normalize_scores as _rust_normalize_scores
    from khora_accel import pagerank as _rust_pagerank
    from khora_accel import pairwise_cosine_above_threshold as _rust_pairwise_cosine
    from khora_accel import reciprocal_rank_fusion as _rust_rrf
    from khora_accel import resolve_entities_batch as _rust_resolve_entities_batch
    from khora_accel import resolve_entities_enhanced as _rust_resolve_entities_enhanced
    from khora_accel import sequence_match_ratio as _rust_sequence_match
    from khora_accel import weighted_rrf as _rust_weighted_rrf
    from khora_accel import weighted_rrf_normalized as _rust_weighted_rrf_normalized

    _HAS_RUST = True
except ImportError:  # pragma: no cover
    _HAS_RUST = False
    RustBM25Index = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Tier 1: NumPy / RapidFuzz (existing)
# ---------------------------------------------------------------------------

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

try:
    from rapidfuzz.distance import Levenshtein as _rf_lev
    from rapidfuzz.fuzz import ratio as _rf_ratio

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
    RustBM25Index = None  # type: ignore[assignment]
elif _FORCE_BACKEND == "numpy":
    _HAS_RUST = False
    RustBM25Index = None  # type: ignore[assignment]
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
    if _HAS_RUST:
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

    if _HAS_RUST and _HAS_NUMPY:
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
    if _HAS_RUST and _HAS_NUMPY:
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
# Levenshtein similarity
# ---------------------------------------------------------------------------


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1.0 = identical).

    Uses Rust > rapidfuzz > pure Python.
    """
    if _HAS_RUST:
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

    Uses Rust > rapidfuzz > difflib.
    """
    if _HAS_RUST:
        return _rust_sequence_match(s1, s2)

    if _HAS_RAPIDFUZZ:
        return _rf_ratio(s1, s2) / 100.0

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
    if _HAS_RUST:
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
    if _HAS_RUST:
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


def pagerank(
    n: int,
    edges: list[tuple[int, int, float]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> list[float]:
    """Compute PageRank scores on a weighted directed graph.

    Args:
        n: Number of nodes (IDs are 0..n-1).
        edges: (src, dst, weight) triples.
        damping: Damping factor (typically 0.85).
        max_iter: Maximum iterations.
        tol: Convergence threshold.

    Returns:
        List of length n with PageRank scores indexed by node ID.
    """
    if _HAS_RUST:
        return _rust_pagerank(n, edges, damping, max_iter, tol)

    # Pure-Python fallback
    if n == 0:
        return []

    incoming: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    out_degree: list[float] = [0.0] * n

    for src, dst, weight in edges:
        if 0 <= src < n and 0 <= dst < n:
            incoming[dst].append((src, weight))
            out_degree[src] += weight

    base = (1.0 - damping) / n
    scores = [1.0 / n] * n

    for _ in range(max_iter):
        new_scores = [0.0] * n
        diff = 0.0

        for node in range(n):
            contrib = 0.0
            for src, weight in incoming[node]:
                if out_degree[src] > 0:
                    contrib += scores[src] * weight / out_degree[src]
            new_score = base + damping * contrib
            diff += abs(new_score - scores[node])
            new_scores[node] = new_score

        scores = new_scores
        if diff < tol:
            break

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
    if _HAS_RUST:
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
    if _HAS_RUST:
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
    if _HAS_RUST:
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
    if _HAS_RUST:
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
    if _HAS_RUST:
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
    if _HAS_RUST:
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
    if _HAS_RUST:
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


# ---------------------------------------------------------------------------
# Entity name normalization
# ---------------------------------------------------------------------------

_HONORIFICS = ("mr.", "mrs.", "ms.", "dr.", "prof.", "sir ", "lord ", "lady ")


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for dedup. Rust > Python."""
    if _HAS_RUST:
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
    if _HAS_RUST:
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
    if _HAS_RUST:
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

    if _HAS_RUST:
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
