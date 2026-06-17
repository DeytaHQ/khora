"""Neutral temporal data types for chunk storage.

Strict-leaf module: imports only the standard library and
``khora.core.models``. It must never pull an engine, a DB driver, or the
recall-filter machinery into ``sys.modules`` — so callers can depend on the
temporal data shapes without paying the cost of importing a backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.core.models.document import Chunk

if TYPE_CHECKING:
    from khora.core.models.document import Document


@dataclass
class TemporalChunk:
    """Chunk with temporal metadata for the Skeleton engine."""

    id: UUID
    namespace_id: UUID
    document_id: UUID
    content: str
    embedding: list[float] | None = None

    # Temporal fields
    occurred_at: datetime | None = None  # When the event/fact happened
    created_at: datetime | None = None  # When the chunk was created

    # Metadata for filtering
    source_system: str | None = None
    author: str | None = None
    channel: str | None = None
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    chunker_info: dict[str, Any] = field(default_factory=dict)

    # Denormalized document-grained fields (copied from the parent document)
    # for deterministic recall filters. All nullable. ``source_timestamp`` is
    # the producer's verbatim event time, distinct from ``occurred_at``.
    source_type: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    source_timestamp: datetime | None = None
    external_id: str | None = None
    content_type: str | None = None
    source: str | None = None
    title: str | None = None


def document_denorm_fields(document: Document) -> dict[str, Any]:
    """Return the denormalized document-grained chunk fields from a Document.

    Copies the eight provenance fields off the typed Document so chunk
    builders can stamp them onto a :class:`TemporalChunk` without repeating
    the mapping at every write site. ``source_timestamp`` is the producer's
    verbatim time, kept distinct from the chunk event-time ``occurred_at``.
    """
    return {
        "source_type": document.source_type,
        "source_name": document.source_name,
        "source_url": document.source_url,
        "source_timestamp": document.source_timestamp,
        "external_id": document.external_id,
        "content_type": document.content_type,
        "source": document.source,
        "title": document.title,
    }


@dataclass
class TemporalSearchResult:
    """Result from a temporal search."""

    chunk: TemporalChunk
    similarity: float
    bm25_score: float | None = None  # Only for hybrid search
    combined_score: float | None = None  # After fusion


@dataclass
class ChunkTemporalFilter:
    """Filter for temporal queries."""

    # Time range filters
    occurred_after: datetime | None = None
    occurred_before: datetime | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None

    # Keyword filters
    source_system: str | None = None
    author: str | None = None
    channel: str | None = None
    tags: list[str] | None = None  # Chunks must have ALL of these tags

    # Additional structured filters (key -> value or operator dict)
    # Example: {"confidence": {"gte": 0.8}, "metadata.priority": {"eq": "high"}}
    additional: dict[str, Any] = field(default_factory=dict)


def temporal_chunk_to_chunk(tc: TemporalChunk) -> Chunk:
    """Adapt a ``TemporalChunk`` (skeleton storage) to a public ``Chunk``.

    Preserves fields the retriever and rerankers depend on:
    ``chunker_info`` (#800), ``created_at`` (#810), and ``session_id``
    (#620 — stamped into ``TemporalChunk.metadata`` by the engines).
    Surfaces ``occurred_at`` (the chunk event-time) and ``source_timestamp``
    (the producer's verbatim time) as distinct values; the recall projection
    applies the event-time-then-producer-time fallback downstream.
    """
    md = tc.metadata or {}
    sid_raw = md.get("session_id")
    session_id: UUID | None
    if isinstance(sid_raw, UUID):
        session_id = sid_raw
    elif sid_raw:
        try:
            session_id = UUID(str(sid_raw))
        except (TypeError, ValueError):
            session_id = None
    else:
        session_id = None

    return Chunk(
        id=tc.id,
        namespace_id=tc.namespace_id,
        document_id=tc.document_id,
        content=tc.content,
        chunk_index=int(md.get("chunk_index", 0) or 0),
        start_char=int(md.get("start_char", 0) or 0),
        end_char=int(md.get("end_char", 0) or 0),
        token_count=int(md.get("token_count", 0) or 0),
        metadata=md,
        chunker_info=tc.chunker_info or {},
        embedding=tc.embedding,
        embedding_model=str(md.get("embedding_model", "") or ""),
        created_at=tc.created_at or datetime.now(UTC),
        # Carry the chunk event-time and the producer's verbatim time as
        # distinct values; the recall projection applies the fallback.
        occurred_at=tc.occurred_at,
        source_timestamp=tc.source_timestamp,
        session_id=session_id,
    )


__all__ = [
    "ChunkTemporalFilter",
    "TemporalChunk",
    "TemporalSearchResult",
    "document_denorm_fields",
    "temporal_chunk_to_chunk",
]
