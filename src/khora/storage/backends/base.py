"""Abstract protocols for storage backends.

These protocols define the interface that all storage backends must implement,
enabling dependency injection and easy testing with mocks.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable
from uuid import UUID

T = TypeVar("T")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.core.models import (
        Chunk,
        Document,
        Entity,
        Episode,
        MemoryEvent,
        MemoryNamespace,
        Relationship,
    )
    from khora.core.models.document import DocumentSource


@dataclass(frozen=True)
class PaginatedResult(Generic[T]):
    """Paginated query result with total count."""

    items: list[T]
    total: int
    limit: int
    offset: int


@runtime_checkable
class RelationalBackendProtocol(Protocol):
    """Protocol for relational database backends (PostgreSQL).

    Handles storage of documents, tenancy data, ACLs, and sync checkpoints.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connections."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        ...

    # Namespace operations
    @abstractmethod
    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        """Resolve a stable namespace_id to the active version's row id."""
        ...

    @abstractmethod
    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace."""
        ...

    @abstractmethod
    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        ...

    @abstractmethod
    async def list_namespaces(
        self, *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> PaginatedResult[MemoryNamespace]:
        """List namespaces with pagination."""
        ...

    @abstractmethod
    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update a namespace."""
        ...

    @abstractmethod
    async def create_namespace_version(
        self,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        """Create a new version of a namespace.

        Args:
            previous_version: The previous version to supersede (if any)

        Returns:
            New namespace version
        """
        ...

    @abstractmethod
    async def deactivate_namespace(self, namespace_id: UUID) -> None:
        """Mark a namespace version as inactive.

        Args:
            namespace_id: ID of the namespace to deactivate
        """
        ...

    # Document operations
    @abstractmethod
    async def create_document(self, document: Document) -> Document:
        """Create a new document."""
        ...

    @abstractmethod
    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""
        ...

    @abstractmethod
    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents in a namespace."""
        ...

    @abstractmethod
    async def update_document(self, document: Document) -> Document:
        """Update a document."""
        ...

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document."""
        ...

    @abstractmethod
    async def count_documents(self, namespace_id: UUID) -> int:
        """Count documents in a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            Total number of documents. Returns 0 if namespace is empty.
        """
        ...

    @abstractmethod
    async def get_last_activity_at(self, namespace_id: UUID) -> datetime | None:
        """Get the most recent document creation timestamp in a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            datetime: Timestamp of the most recently created document (UTC)
            None: If the namespace has no documents
        """
        ...

    async def get_document_stats(self, namespace_id: UUID) -> tuple[int, datetime | None]:
        """Count documents and get last activity in a single query.

        Returns (count, last_activity_at). Backends may override for efficiency;
        the default falls back to two separate calls.
        """
        count = await self.count_documents(namespace_id)
        last_activity = await self.get_last_activity_at(namespace_id)
        return count, last_activity

    @abstractmethod
    async def get_document_by_checksum(self, namespace_id: UUID, checksum: str) -> Document | None:
        """Get a document by its content checksum (for deduplication)."""
        ...

    async def get_documents_batch(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query.

        Returns dictionary mapping document ID to Document object.
        """
        ...

    async def get_document_sources_batch(self, document_ids: list[UUID]) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution.

        Returns a column-limited projection (no content, processing stats,
        or mutable state) for display and linking purposes.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            Dictionary mapping document ID to DocumentSource
        """
        ...

    # Sync checkpoint operations
    @abstractmethod
    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        """Get the last sync checkpoint for a source."""
        ...

    @abstractmethod
    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """Set the sync checkpoint for a source."""
        ...

    def _get_session(self) -> Any:
        """Get a database session (provided by AsyncSessionMixin)."""
        ...


@runtime_checkable
class VectorBackendProtocol(Protocol):
    """Protocol for vector database backends (pgvector).

    Handles storage and retrieval of embeddings for semantic search.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connections."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        ...

    # Chunk operations
    @abstractmethod
    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create a new chunk with its embedding."""
        ...

    @abstractmethod
    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks in a batch."""
        ...

    @abstractmethod
    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Get a chunk by ID."""
        ...

    @abstractmethod
    async def get_chunks_batch(self, chunk_ids: list[UUID]) -> dict[UUID, Chunk]:
        """Get multiple chunks by ID in a single query.

        Args:
            chunk_ids: List of chunk IDs to fetch

        Returns:
            Dictionary mapping chunk ID to Chunk (only for existing chunks)
        """
        ...

    @abstractmethod
    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Get all chunks for a document."""
        ...

    @abstractmethod
    async def delete_chunks_by_document(self, document_id: UUID, *, session: AsyncSession | None = None) -> int:
        """Delete all chunks for a document."""
        ...

    @abstractmethod
    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        filter_document_ids: list[UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity.

        Returns list of (chunk, similarity_score) tuples.
        """
        ...

    # Entity operations (for vector search via PostgreSQL)
    @abstractmethod
    async def create_entity(self, entity: Entity) -> None:
        """Create an entity record in PostgreSQL for vector search."""
        ...

    @abstractmethod
    async def update_entity(self, entity: Entity) -> None:
        """Update an entity record in PostgreSQL."""
        ...

    @abstractmethod
    async def entity_exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists in PostgreSQL."""
        ...

    @abstractmethod
    async def update_entity_embedding(self, entity_id: UUID, embedding: list[float], model: str) -> None:
        """Update the embedding for an entity."""
        ...

    async def update_entity_embeddings_batch(self, updates: list[tuple[UUID, list[float], str]]) -> int:
        """Update embeddings for multiple entities in a single transaction."""
        ...

    @abstractmethod
    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        """Search for similar entities by embedding."""
        ...

    @abstractmethod
    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",
    ) -> list[tuple[Chunk, float]]:
        """Search chunks using PostgreSQL full-text search.

        Uses ts_rank on the content_tsv generated column.

        Returns list of (chunk, rank_score) tuples.
        """
        ...

    # Aggregate operations (optional — have default implementations in VectorBackendBase)

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Count chunks in a namespace."""
        ...

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        """List chunks in a namespace."""
        ...


@runtime_checkable
class GraphBackendProtocol(Protocol):
    """Protocol for graph database backends (Neo4j).

    Handles storage and traversal of the knowledge graph.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connections."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        ...

    # Entity operations
    @abstractmethod
    async def create_entity(self, entity: Entity) -> Entity:
        """Create an entity node in the graph."""
        ...

    @abstractmethod
    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        ...

    @abstractmethod
    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        """Get an entity by name and type (for deduplication)."""
        ...

    @abstractmethod
    async def update_entity(self, entity: Entity) -> Entity:
        """Update an entity."""
        ...

    @abstractmethod
    async def delete_entity(self, entity_id: UUID) -> bool:
        """Delete an entity and its relationships."""
        ...

    @abstractmethod
    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities in a namespace."""
        ...

    # Relationship operations
    @abstractmethod
    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a relationship between entities."""
        ...

    @abstractmethod
    async def get_relationship(self, relationship_id: UUID) -> Relationship | None:
        """Get a relationship by ID."""
        ...

    @abstractmethod
    async def delete_relationship(self, relationship_id: UUID) -> bool:
        """Delete a relationship."""
        ...

    @abstractmethod
    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        ...

    @abstractmethod
    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        """List all relationships in a namespace."""
        ...

    # Episode operations
    @abstractmethod
    async def create_episode(self, episode: Episode) -> Episode:
        """Create an episode node."""
        ...

    @abstractmethod
    async def get_episode(self, episode_id: UUID) -> Episode | None:
        """Get an episode by ID."""
        ...

    @abstractmethod
    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """List episodes in a time range."""
        ...

    # Graph traversal
    @abstractmethod
    async def find_paths(
        self,
        namespace_id: UUID,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find paths between two entities."""
        ...

    @abstractmethod
    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity up to a certain depth."""
        ...

    @abstractmethod
    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        """Search entities by attribute value."""
        ...

    # Batch and aggregate operations (optional — have default implementations in GraphBackendBase)

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query.

        Returns dictionary mapping entity ID to Entity object.
        """
        ...

    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Get neighborhoods for multiple entities.

        Returns dictionary mapping entity ID to neighborhood data.
        """
        ...

    async def count_entities(self, namespace_id: UUID) -> int:
        """Count entities in a namespace."""
        ...

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using MERGE semantics.

        For each entity, creates it if new or updates if existing
        (matched by name + type within namespace).

        Returns list of (entity, is_new) tuples.
        """
        ...

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
    ) -> int:
        """Batch create relationships.

        Returns the number of relationships created.
        """
        ...


@runtime_checkable
class EventStoreProtocol(Protocol):
    """Protocol for event store backends.

    Handles the append-only event log for event sourcing.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the store."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connections."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the store is healthy."""
        ...

    @abstractmethod
    async def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append an event to the log."""
        ...

    @abstractmethod
    async def append_events_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events in a batch."""
        ...

    @abstractmethod
    async def get_events(
        self,
        namespace_id: UUID,
        *,
        event_types: list[str] | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEvent]:
        """Query events from the log."""
        ...

    @abstractmethod
    async def get_events_for_resource(
        self,
        resource_type: str,
        resource_id: UUID,
        *,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Get all events for a specific resource."""
        ...

    @abstractmethod
    async def get_latest_event(
        self,
        resource_type: str,
        resource_id: UUID,
    ) -> MemoryEvent | None:
        """Get the latest event for a resource."""
        ...

    @abstractmethod
    async def count_events(
        self,
        namespace_id: UUID,
        *,
        event_types: list[str] | None = None,
        after: datetime | None = None,
    ) -> int:
        """Count events matching criteria."""
        ...
