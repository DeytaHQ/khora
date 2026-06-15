"""Backend implementations for the Skeleton engine.

The Skeleton engine supports multiple backends for temporal vector storage:
- pgvector: PostgreSQL+pgvector (default, no additional infrastructure)
- weaviate: Weaviate (advanced hybrid search, multi-field filtering)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from khora.core.models.document import Chunk

if TYPE_CHECKING:
    from khora.config import KhoraConfig
    from khora.core.models.document import Document
    from khora.filter.ast import FilterNode
    from khora.filter.report import ChannelPlan


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
class TemporalFilter:
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


class TemporalVectorStore(Protocol):
    """Protocol for temporal vector storage backends.

    Backends implement temporal-aware chunk storage with:
    - Vector similarity search
    - Structured field filtering
    - Optional hybrid search (BM25 + vector)
    """

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the backend."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the backend."""
        ...

    @abstractmethod
    async def create_chunk(self, chunk: TemporalChunk) -> TemporalChunk:
        """Store a chunk with temporal metadata."""
        ...

    @abstractmethod
    async def create_chunks_batch(self, chunks: list[TemporalChunk]) -> list[TemporalChunk]:
        """Store multiple chunks in batch."""
        ...

    @abstractmethod
    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> TemporalChunk | None:
        """Get a chunk by ID."""
        ...

    @abstractmethod
    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:
        """Delete a chunk by ID."""
        ...

    @abstractmethod
    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:
        """Delete all chunks for a document."""
        ...

    @abstractmethod
    async def search(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        temporal_filter: TemporalFilter | None = None,
        hybrid_alpha: float | None = None,  # None = vector only, 0-1 = blend
        query_text: str | None = None,  # Required for hybrid search
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
    ) -> list[TemporalSearchResult]:
        """Search for similar chunks with temporal filtering.

        Args:
            namespace_id: Namespace to search within
            query_embedding: Query vector
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            temporal_filter: Structured temporal and metadata filters
            hybrid_alpha: Blend factor for hybrid search (0=BM25 only, 1=vector only)
            query_text: Original query text for BM25 (required if hybrid_alpha is set)
            filter_ast: Canonical recall-filter AST. The pgvector backend
                compiles it to a WHERE predicate; the other backends accept
                it for protocol parity and ignore it (no compilation yet).
            filter_plan_out: Optional per-call sink for the honest filter-pushdown
                plan (#1069). When provided, the backend appends exactly one
                :class:`~khora.filter.report.ChannelPlan` describing how *this*
                call's ``filter_ast`` was pushed down vs. post-filtered. The caller
                passes a fresh list per call and reads back ``[0]``, so the report
                is race-free under concurrent recalls on a shared store — no mutable
                instance state is involved. Backends that do not report pushdown
                leave it untouched.

        Returns:
            List of matching chunks with similarity scores
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        ...

    # Optional capability — backends that store ingested chunk content
    # in their own table override this to expose a BM25 / full-text path
    # for the StorageCoordinator dispatch. Default returns an empty list
    # so backends without a fulltext-search table fall through to the
    # relational ``chunks``-table reader on the coordinator.
    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Full-text search over the temporal store's chunk table.

        Returns ``[]`` by default; backends with a populated fulltext
        column override this.

        ``filter_ast`` is the canonical recall-filter AST. The pgvector
        backend compiles it to the SAME ``khora_chunks`` WHERE predicate the
        vector path uses; the other backends accept it for protocol parity
        and ignore it (no compilation yet).

        ``filter_plan_out`` is the optional per-call sink for the honest
        filter-pushdown plan, mirroring :meth:`search`. When provided, the
        backend appends exactly one :class:`~khora.filter.report.ChannelPlan`
        built from THIS call's actual fulltext compile — so a raise-mode
        backend reports every leaf pushed while a split-mode backend
        (sqlite_lance) reports the pushed/post-filtered partition its WHERE
        + in-memory re-check actually produced. The default (return ``[]``)
        leaves it untouched.
        """
        return []

    # Optional capability — backends that store ingested chunk content in
    # their own table override this to expose a pure-recency channel for the
    # VectorCypher recency path. Default returns an empty list so backends
    # without a recency-capable table contribute nothing to the channel.
    async def search_recent_chunks(
        self,
        namespace_id: UUID,
        limit: int,
        *,
        created_after: datetime | None = None,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
    ) -> list[tuple[Chunk, float | None]]:
        """Return the ``limit`` most-recent chunks in the namespace.

        Pure recency sort with no semantic gate — the caller (the
        VectorCypher recency channel) applies the cosine relevance floor.
        The recency axis is ``COALESCE(occurred_at, source_timestamp,
        created_at)`` (event-time → producer-time → ingest-time), narrowed
        from above by the optional ``created_after`` bound on the same axis.

        ``filter_ast`` is the canonical recall-filter AST. The pgvector
        backend compiles it to the SAME ``khora_chunks`` WHERE predicate the
        vector path uses; the other backends accept it for protocol parity
        and ignore it (no compilation yet).

        ``filter_plan_out`` is the optional per-call sink for the honest
        filter-pushdown plan, mirroring :meth:`search` / :meth:`search_fulltext`.
        When provided, the backend appends exactly one
        :class:`~khora.filter.report.ChannelPlan` built from THIS call's actual
        compile — a raise-mode backend reports every leaf pushed. The default
        (return ``[]``) leaves it untouched.

        Returns ``(chunk, None)`` tuples; the ``None`` signals "no cosine
        score available" so the caller branches on it. The default empty
        list means the backend does not support the recency channel
        (sqlite_lance support tracked in GitHub issue #1182).
        """
        return []


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


def create_temporal_store(
    backend: str,
    config: KhoraConfig,
    *,
    weaviate_url: str | Any | None = None,
    turbopuffer_config: str | Any | None = None,
    surrealdb_config: Any | None = None,
    surrealdb_connection: Any | None = None,
    engine: Any | None = None,
    sqlite_lance_handle: Any | None = None,
) -> TemporalVectorStore:
    """Create a temporal vector store backend.

    Args:
        backend: Backend type ("pgvector", "weaviate", "surrealdb", or "sqlite_lance")
        config: Khora configuration
        weaviate_url: Either a Weaviate connection URL (str, self-hosted)
            or a :class:`khora.engines.skeleton.backends.weaviate.WeaviateBackendConfig`
            (cloud / auth / custom-port). Required for the weaviate backend.
        surrealdb_config: SurrealDBConfig instance (optional, falls back to config.storage.surrealdb)
        surrealdb_connection: Shared ``SurrealDBConnection`` (surrealdb backend only).
            When provided, the temporal store reuses this connection instead
            of opening its own. Required on ``surrealkv://`` (embedded) mode
            because surrealkv allows only one open handle per directory —
            opening a second handle raises
            ``InternalError: Invalid revision 0 for type Value`` on the
            first write (see issue #718). Mirrors the vectorcypher wiring.
        engine: Optional shared SQLAlchemy AsyncEngine (pgvector backend only).
            When provided, the temporal store reuses this engine instead of
            creating a private connection pool.
        sqlite_lance_handle: Shared ``EmbeddedStorageHandle`` (sqlite_lance backend only).
            The Skeleton engine extracts this from the unified
            ``StorageCoordinator``'s vector adapter so the temporal store and
            the coordinator share one aiosqlite + LanceDB pair.

    Returns:
        Configured TemporalVectorStore implementation
    """
    if backend == "pgvector":
        from khora.engines.skeleton.backends.pgvector import PgVectorTemporalStore

        return PgVectorTemporalStore(config, engine=engine)
    elif backend == "weaviate":
        if not weaviate_url:
            raise ValueError("weaviate_url is required for weaviate backend")
        from khora.engines.skeleton.backends.weaviate import WeaviateTemporalStore

        return WeaviateTemporalStore(config, weaviate_url)
    elif backend == "turbopuffer":
        if not turbopuffer_config:
            raise ValueError(
                "turbopuffer_config is required for turbopuffer backend (api-key str or TurbopufferBackendConfig)"
            )
        from khora.engines.skeleton.backends.turbopuffer import TurbopufferTemporalStore

        return TurbopufferTemporalStore(config, turbopuffer_config)
    elif backend == "surrealdb":
        from khora.engines.skeleton.backends.surrealdb import SurrealDBTemporalStore

        return SurrealDBTemporalStore(
            config,
            surrealdb_config=surrealdb_config,
            connection=surrealdb_connection,
        )
    elif backend == "sqlite_lance":
        if sqlite_lance_handle is None:
            raise ValueError(
                "sqlite_lance_handle is required for sqlite_lance backend "
                "(extracted from the unified StorageCoordinator)"
            )
        from khora.engines.skeleton.backends.sqlite_lance import SQLiteLanceTemporalStore

        return SQLiteLanceTemporalStore(sqlite_lance_handle)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Available: pgvector, weaviate, turbopuffer, surrealdb, sqlite_lance"
        )


__all__ = [
    "TemporalChunk",
    "TemporalFilter",
    "TemporalSearchResult",
    "TemporalVectorStore",
    "create_temporal_store",
    "document_denorm_fields",
    "temporal_chunk_to_chunk",
]
