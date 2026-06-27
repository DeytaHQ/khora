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


def build_keyword_chunk_edges(chunks: list[TemporalChunk]) -> list[tuple[str, UUID, float]]:
    """Build ``(keyword, chunk_id, idf)`` edges for a batch of chunks.

    Mirrors the keyword -> chunk-ids map + IDF formula in
    ``khora.core.ranking.select_core_chunks``: each keyword's IDF is
    ``log(n_chunks / (1 + df)) + 1`` where ``df`` is the number of chunks the
    keyword appears in and ``n_chunks`` is the batch size (chunk-as-document).
    Approximate by design - computed per ingest batch, not over the whole
    namespace - which is fine for the experimental channel.

    Chunks must carry ``.id`` (assigned post-persist) and ``.content``. Returns
    one edge per (keyword, chunk) pair.
    """
    if not chunks:
        return []

    # keyword -> chunk ids it appears in (first-seen order), mirroring
    # select_core_chunks' insertion semantics.
    keyword_to_chunks: dict[str, list[UUID]] = {}
    for chunk in chunks:
        for keyword in set(tokenize_multilingual(chunk.content)):
            keyword_to_chunks.setdefault(keyword, []).append(chunk.id)

    n_chunks = len(chunks)
    edges: list[tuple[str, UUID, float]] = []
    for keyword, chunk_ids in keyword_to_chunks.items():
        idf = math.log(n_chunks / (1 + len(chunk_ids))) + 1
        for chunk_id in chunk_ids:
            edges.append((keyword, chunk_id, idf))
    return edges


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
    edges = build_keyword_chunk_edges(chunks)
    if not edges:
        return
    try:
        await storage.upsert_keyword_chunk_edges(namespace_id, edges)
    except Exception as exc:
        logger.warning(
            "keyword_ppr edge write failed, continuing without keyword_chunks for this batch",
            exc_info=True,
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


__all__ = ["build_keyword_chunk_edges", "persist_keyword_chunk_edges"]
