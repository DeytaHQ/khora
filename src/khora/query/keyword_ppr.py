"""KET-RAG keyword-chunk PageRank retrieval channel (#1391).

Experimental, opt-in alternative to BM25 for the lexical recall slot, selected
by ``config.query.lexical_channel == "keyword_ppr"``. The ingest path persists a
keyword -> chunk bipartite (the ``keyword_chunks`` edge table); this module runs
a per-query personalized PageRank over the induced chunk graph, seeded on the
chunks that contain the query's keywords.

The chunk-graph construction mirrors ``khora.core.ranking.select_core_chunks``:
each keyword links the chunks it appears in (``build_chunk_edges``), weighted by
the keyword's stored IDF, and ``_accel.pagerank`` ranks chunks. The
personalization vector seeds chunks containing any query keyword, weighted by
that keyword's IDF (normalized).

Degrade-safe: no query keywords, no edges loaded, or no seed overlap -> ``[]``.
The caller (the retriever lexical slot) then fuses an empty channel, same as a
BM25 channel that matched nothing.

References:
- KET-RAG (the keyword-chunk skeleton): the same machinery used at ingest for
  core-chunk selection, re-used here as a query-time retrieval channel.
- An earlier synthetic-corpus spike found ~no marginal recall over BM25, so this
  is default-off and meant to be A/B'd on real data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger

from khora._accel import build_chunk_edges, pagerank

if TYPE_CHECKING:
    from collections.abc import Callable

    from khora.storage.coordinator import StorageCoordinator


async def keyword_ppr_retrieve_chunks(
    storage: StorageCoordinator,
    namespace_id: UUID,
    query_text: str,
    *,
    tokenizer: Callable[[str], list[str]],
    damping: float,
    max_iter: int,
    tol: float,
    limit: int,
    max_edges: int,
) -> list[tuple[UUID, float]]:
    """Rank chunks for ``query_text`` via keyword-chunk personalized PageRank.

    Args:
        storage: Coordinator exposing ``get_keyword_chunk_edges``.
        namespace_id: Namespace to retrieve within (stable id; the coordinator
            resolves it).
        query_text: The raw query.
        tokenizer: Keyword tokenizer ``(text) -> list[str]`` - the same
            multilingual tokenizer used at ingest, so query and stored keywords
            match.
        damping: PageRank damping factor.
        max_iter: PageRank max iterations.
        tol: PageRank convergence tolerance.
        limit: Max chunks to return.
        max_edges: Cap on keyword -> chunk edges loaded (bounds per-query cost).

    Returns:
        ``[(chunk_id, score), ...]`` sorted by score descending, capped at
        ``limit``. Empty on any degenerate condition (no keywords / no edges /
        no seed overlap).
    """
    query_keywords = set(tokenizer(query_text))
    if not query_keywords:
        return []

    edges_rows = await storage.get_keyword_chunk_edges(namespace_id, limit=max_edges)
    if not edges_rows:
        logger.debug("keyword_ppr: no keyword_chunk edges for namespace; returning empty channel")
        return []

    # Build the keyword -> chunk-ids map (first-seen order), the per-chunk index,
    # and the per-keyword IDF (stored at ingest; same value on every row for a
    # keyword, so first-seen wins). Mirrors core.ranking.select_core_chunks'
    # construction of keyword_chunk_ids / idf_scores.
    chunk_index: dict[UUID, int] = {}
    keyword_to_chunks: dict[str, list[int]] = {}
    keyword_idf: dict[str, float] = {}
    for keyword, chunk_id, idf in edges_rows:
        idx = chunk_index.get(chunk_id)
        if idx is None:
            idx = len(chunk_index)
            chunk_index[chunk_id] = idx
        keyword_to_chunks.setdefault(keyword, []).append(idx)
        keyword_idf.setdefault(keyword, idf)

    n_chunks = len(chunk_index)
    keyword_list = list(keyword_to_chunks.keys())
    keyword_chunk_ids = [keyword_to_chunks[kw] for kw in keyword_list]
    idf_scores = [keyword_idf[kw] for kw in keyword_list]

    # Personalization: seed chunks containing any query keyword, weighted by that
    # keyword's IDF. No seed overlap -> empty channel.
    personalization = [0.0] * n_chunks
    seeded = False
    for keyword in query_keywords:
        chunk_idxs = keyword_to_chunks.get(keyword)
        if not chunk_idxs:
            continue
        weight = keyword_idf[keyword]
        for idx in chunk_idxs:
            personalization[idx] += weight
            seeded = True
    if not seeded:
        logger.debug("keyword_ppr: no query keyword overlapped the namespace bipartite; empty channel")
        return []

    total = sum(personalization)
    if total > 0:
        personalization = [p / total for p in personalization]

    edges = build_chunk_edges(n_chunks, keyword_chunk_ids, idf_scores)
    scores = pagerank(n_chunks, edges, damping, max_iter, tol, personalization)

    idx_to_chunk = {idx: cid for cid, idx in chunk_index.items()}
    ranked = sorted(range(n_chunks), key=lambda i: scores[i], reverse=True)
    return [(idx_to_chunk[i], scores[i]) for i in ranked[:limit]]


__all__ = ["keyword_ppr_retrieve_chunks"]
