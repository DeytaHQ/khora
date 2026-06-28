"""Ingest-time keyword -> chunk edge persistence for the keyword_ppr channel (#1391).

Gated step run after chunks are persisted at ingest, only when
``config.query.lexical_channel == "keyword_ppr"``. Extracts keywords per chunk
with the multilingual tokenizer, computes per-batch approximate IDF (mirroring
``khora.core.ranking.select_core_chunks``), and bulk-inserts the edges via
``StorageCoordinator.upsert_keyword_chunk_edges``.

Default ``bm25`` deployments never call this (zero write cost). A write failure
degrades per ADR-001: WARNING log + a ``Degradation`` appended to the supplied
diagnostics dict. No metric counter is emitted (would trip the telemetry-contract
drift gate); the ``Degradation`` record satisfies ADR-001 - same choice as
``ppr_retrieval._record_ppr_degradation``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.core.diagnostics import Degradation
from khora.extraction.tokenize import tokenize_multilingual

if TYPE_CHECKING:
    from khora.storage.coordinator import StorageCoordinator
    from khora.storage.temporal import TemporalChunk


def build_keyword_chunk_edges_from_keywords(
    chunk_keywords: list[tuple[UUID, set[str]]],
) -> list[tuple[str, UUID, float]]:
    """Build ``(keyword, chunk_id, idf)`` edges from per-chunk keyword sets.

    Mirrors the keyword -> chunk-ids map + IDF formula in
    ``khora.core.ranking.select_core_chunks``: each keyword's IDF is
    ``log(n_chunks / (1 + df)) + 1`` where ``df`` is the number of chunks the
    keyword appears in and ``n_chunks`` is the batch size (chunk-as-document).
    Approximate by design - computed per ingest batch, not over the whole
    namespace - which is fine for the experimental channel.

    Takes ``(chunk_id, keyword_set)`` pairs rather than chunks so the caller can
    tokenize per window and release the embedded ``TemporalChunk``s (and their
    1536-dim payloads) without retaining the whole document in memory - keeping
    the ``max_chunks_in_flight`` bound. Returns one edge per (keyword, chunk).
    """
    if not chunk_keywords:
        return []

    # keyword -> chunk ids it appears in, mirroring select_core_chunks'
    # insertion semantics. Iterate keywords SORTED so the produced edge-list
    # order is deterministic (a set's iteration order is hash-seed dependent,
    # which would make the edge list - and exact-list tests - order-flaky and
    # break the byte-identical guarantee between the two entry points).
    keyword_to_chunks: dict[str, list[UUID]] = {}
    for chunk_id, keywords in chunk_keywords:
        for keyword in sorted(keywords):
            keyword_to_chunks.setdefault(keyword, []).append(chunk_id)

    n_chunks = len(chunk_keywords)
    edges: list[tuple[str, UUID, float]] = []
    for keyword, chunk_ids in keyword_to_chunks.items():
        idf = math.log(n_chunks / (1 + len(chunk_ids))) + 1
        for chunk_id in chunk_ids:
            edges.append((keyword, chunk_id, idf))
    return edges


def build_keyword_chunk_edges(chunks: list[TemporalChunk]) -> list[tuple[str, UUID, float]]:
    """Build ``(keyword, chunk_id, idf)`` edges for a batch of chunks.

    Tokenizes each chunk and delegates to
    :func:`build_keyword_chunk_edges_from_keywords`. Chunks must carry ``.id``
    (assigned post-persist) and ``.content``.
    """
    return build_keyword_chunk_edges_from_keywords(
        [(chunk.id, set(tokenize_multilingual(chunk.content))) for chunk in chunks]
    )


async def _persist_edges(
    storage: StorageCoordinator,
    namespace_id: UUID,
    edges: list[tuple[str, UUID, float]],
    out_diagnostics: dict[str, Any] | None,
) -> None:
    if not edges:
        return
    try:
        await storage.upsert_keyword_chunk_edges(namespace_id, edges)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            "keyword_ppr edge write failed, continuing without keyword_chunks for this batch",
        )
        if out_diagnostics is not None:
            out_diagnostics.setdefault("degradations", []).append(
                Degradation(
                    component="vectorcypher.keyword_edges",
                    reason="keyword_chunk_write_failed",
                    detail=str(exc)[:200] or None,
                    exception=type(exc).__name__,
                )
            )


async def persist_keyword_chunk_edges(
    storage: StorageCoordinator,
    namespace_id: UUID,
    chunks: list[TemporalChunk],
    *,
    out_diagnostics: dict[str, Any] | None = None,
) -> None:
    """Extract keywords + persist keyword -> chunk edges for ``chunks`` (#1391).

    Called from the VectorCypher persist sites only when the keyword_ppr channel
    is enabled. The chunks are already durable, so a write failure degrades
    (WARNING + ADR-001 ``Degradation``) rather than aborting ingest.
    """
    await _persist_edges(storage, namespace_id, build_keyword_chunk_edges(chunks), out_diagnostics)


async def persist_keyword_chunk_edges_from_keywords(
    storage: StorageCoordinator,
    namespace_id: UUID,
    chunk_keywords: list[tuple[UUID, set[str]]],
    *,
    out_diagnostics: dict[str, Any] | None = None,
) -> None:
    """Persist edges from a precomputed ``(chunk_id, keyword_set)`` snapshot (#1391).

    Same contract as :func:`persist_keyword_chunk_edges` but document-scoped IDF
    is computed over a lightweight keyword-set snapshot, so the windowed ingest
    path can tokenize + release each window's embedded chunks instead of
    retaining the whole document (preserving the ``max_chunks_in_flight`` bound).
    """
    await _persist_edges(
        storage, namespace_id, build_keyword_chunk_edges_from_keywords(chunk_keywords), out_diagnostics
    )


__all__ = [
    "build_keyword_chunk_edges",
    "build_keyword_chunk_edges_from_keywords",
    "persist_keyword_chunk_edges",
    "persist_keyword_chunk_edges_from_keywords",
]
