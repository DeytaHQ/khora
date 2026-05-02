"""Backend implementations for the Skeleton engine.

The Skeleton engine supports multiple backends for temporal vector storage:
- pgvector: PostgreSQL+pgvector (default, no additional infrastructure)
- weaviate: Weaviate (advanced hybrid search, multi-field filtering)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from khora.config import KhoraConfig


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

        Returns:
            List of matching chunks with similarity scores
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        ...


def create_temporal_store(
    backend: str,
    config: KhoraConfig,
    *,
    weaviate_url: str | None = None,
    surrealdb_config: Any | None = None,
    engine: Any | None = None,
    sqlite_lance_handle: Any | None = None,
) -> TemporalVectorStore:
    """Create a temporal vector store backend.

    Args:
        backend: Backend type ("pgvector", "weaviate", "surrealdb", or "sqlite_lance")
        config: Khora configuration
        weaviate_url: Weaviate URL (required for weaviate backend)
        surrealdb_config: SurrealDBConfig instance (optional, falls back to config.storage.surrealdb)
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
    elif backend == "surrealdb":
        from khora.engines.skeleton.backends.surrealdb import SurrealDBTemporalStore

        return SurrealDBTemporalStore(config, surrealdb_config=surrealdb_config)
    elif backend == "sqlite_lance":
        if sqlite_lance_handle is None:
            raise ValueError(
                "sqlite_lance_handle is required for sqlite_lance backend "
                "(extracted from the unified StorageCoordinator)"
            )
        from khora.engines.skeleton.backends.sqlite_lance import SQLiteLanceTemporalStore

        return SQLiteLanceTemporalStore(sqlite_lance_handle)
    else:
        raise ValueError(f"Unknown backend: {backend}. Available: pgvector, weaviate, surrealdb, sqlite_lance")


__all__ = [
    "TemporalChunk",
    "TemporalFilter",
    "TemporalSearchResult",
    "TemporalVectorStore",
    "create_temporal_store",
]
