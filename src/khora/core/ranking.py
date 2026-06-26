"""PageRank-based core chunk selection (KET-RAG inspired).

Identifies the "core" chunks of a document set via a keyword-chunk bipartite
graph and PageRank, so an upstream caller can run full LLM extraction on only
the highest-signal fraction of chunks.

Strict-leaf module: imports only the standard library, ``khora.core``, and
``khora._accel`` (the Rust/Python acceleration boundary). It must never import
from ``khora.engines`` — the engines depend on this util, not the reverse.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger

if TYPE_CHECKING:
    from khora.core.temporal import TemporalChunk


@dataclass
class CoreSelection:
    """Result of a core-chunk selection.

    Attributes:
        core_ids: Chunk IDs selected as core, ordered by descending PageRank
            score (stable for ties).
        scores: PageRank score per chunk ID, for every input chunk.
    """

    core_ids: list[UUID]
    scores: dict[UUID, float]


def select_core_chunks(
    chunks: list[TemporalChunk],
    core_ratio: float,
    *,
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> CoreSelection:
    """Select core chunks via a keyword-chunk PageRank graph.

    1. Extract keywords from each chunk (fast, no LLM).
    2. Build a keyword-chunk bipartite graph weighted by keyword IDF.
    3. Run PageRank to score chunks.
    4. Select the top ``core_ratio`` fraction (at least one).

    Args:
        chunks: Chunks to rank (must carry ``.id`` and ``.content``).
        core_ratio: Fraction of chunks to mark as core.
        damping_factor: PageRank damping factor.
        max_iterations: Maximum PageRank iterations.
        convergence_threshold: PageRank convergence threshold.
        tokenizer: Optional keyword tokenizer ``(text) -> list[str]``. Defaults
            to ``khora._accel.extract_keywords`` (ASCII-only). Pass a
            multilingual tokenizer to make selection work on non-Latin scripts.

    Returns:
        A :class:`CoreSelection` with the core IDs (score-descending) and the
        per-chunk score map. Empty input yields an empty selection.
    """
    if not chunks:
        return CoreSelection([], {})

    from khora._accel import build_chunk_edges, extract_keywords, pagerank

    extract_kw = tokenizer if tokenizer is not None else extract_keywords

    # Build the keyword -> chunk-ids map in first-seen order; each keyword
    # accumulates the chunk ids it appears in. Mirrors the original
    # SkeletonIndexer.add_chunk insertion semantics.
    keywords: dict[str, list[UUID]] = {}
    for chunk in chunks:
        kw_set = set(extract_kw(chunk.content))
        for keyword in kw_set:
            if keyword not in keywords:
                keywords[keyword] = []
            keywords[keyword].append(chunk.id)

    # Map chunk ids to integer indices for accelerated computation.
    chunk_ids = [chunk.id for chunk in chunks]
    n = len(chunk_ids)
    chunk_idx = {cid: i for i, cid in enumerate(chunk_ids)}

    # IDF per keyword: log(n_docs / (1 + df)) + 1.
    n_docs = n
    keyword_list = list(keywords.values())
    keyword_chunk_ids = [[chunk_idx[cid] for cid in kw_chunk_ids if cid in chunk_idx] for kw_chunk_ids in keyword_list]
    idf_scores = [math.log(n_docs / (1 + len(kw_chunk_ids))) + 1 for kw_chunk_ids in keyword_list]

    # Build edges and run PageRank via _accel (Rust or Python fallback).
    edges = build_chunk_edges(n, keyword_chunk_ids, idf_scores)
    raw_scores = pagerank(n, edges, damping_factor, max_iterations, convergence_threshold)

    scores = {cid: raw_scores[i] for i, cid in enumerate(chunk_ids)}

    # Select the top N% by score (stable sort, descending).
    sorted_ids = sorted(chunk_ids, key=lambda cid: scores[cid], reverse=True)
    n_core = max(1, int(n * core_ratio))
    core_ids = sorted_ids[:n_core]

    logger.info(f"Core selection: {len(core_ids)}/{n} core chunks ({len(core_ids) / n * 100:.1f}%)")

    return CoreSelection(core_ids=core_ids, scores=scores)


def select_core_chunk_ids(
    chunks: list[TemporalChunk],
    core_ratio: float,
    *,
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> list[UUID]:
    """Return just the core chunk IDs (thin wrapper over :func:`select_core_chunks`)."""
    return select_core_chunks(
        chunks,
        core_ratio,
        damping_factor=damping_factor,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        tokenizer=tokenizer,
    ).core_ids


__all__ = ["CoreSelection", "select_core_chunk_ids", "select_core_chunks"]
