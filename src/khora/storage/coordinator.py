"""Storage coordinator that orchestrates all backends.

The coordinator provides a unified interface to all storage backends
(PostgreSQL, pgvector, Neo4j) and handles cross-cutting concerns like
transaction coordination and consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.core.models import (
    Chunk,
    Document,
    Entity,
    Episode,
    MemoryEvent,
    MemoryNamespace,
    Organization,
    Relationship,
    Workspace,
)

if TYPE_CHECKING:
    from .backends.base import (
        EventStoreProtocol,
        GraphBackendProtocol,
        RelationalBackendProtocol,
        VectorBackendProtocol,
    )


@dataclass
class StorageHealth:
    """Health status of all storage backends."""

    relational: bool = False
    vector: bool = False
    graph: bool = False
    event_store: bool = False

    @property
    def is_healthy(self) -> bool:
        """Check if all backends are healthy."""
        return self.relational and self.vector

    @property
    def summary(self) -> dict[str, bool]:
        """Get health summary as a dictionary."""
        return {
            "relational": self.relational,
            "vector": self.vector,
            "graph": self.graph,
            "event_store": self.event_store,
        }


@dataclass
class StorageCoordinator:
    """Coordinates operations across all storage backends.

    Provides a unified interface for storage operations and handles
    cross-cutting concerns like transaction management and consistency.
    """

    relational: RelationalBackendProtocol | None = None
    vector: VectorBackendProtocol | None = None
    graph: GraphBackendProtocol | None = None
    event_store: EventStoreProtocol | None = None

    _connected: bool = field(default=False, init=False)

    async def connect(self) -> None:
        """Connect all configured backends."""
        if self._connected:
            return

        logger.info("Connecting storage backends...")

        if self.relational:
            await self.relational.connect()
        if self.vector:
            await self.vector.connect()
        if self.graph:
            await self.graph.connect()
        if self.event_store:
            await self.event_store.connect()

        self._connected = True
        logger.info("Storage backends connected")

    async def disconnect(self) -> None:
        """Disconnect all backends."""
        if not self._connected:
            return

        logger.info("Disconnecting storage backends...")

        if self.event_store:
            await self.event_store.disconnect()
        if self.graph:
            await self.graph.disconnect()
        if self.vector:
            await self.vector.disconnect()
        if self.relational:
            await self.relational.disconnect()

        self._connected = False
        logger.info("Storage backends disconnected")

    async def health_check(self) -> StorageHealth:
        """Check health of all backends."""
        health = StorageHealth()

        if self.relational:
            health.relational = await self.relational.is_healthy()
        if self.vector:
            health.vector = await self.vector.is_healthy()
        if self.graph:
            health.graph = await self.graph.is_healthy()
        if self.event_store:
            health.event_store = await self.event_store.is_healthy()

        return health

    # =========================================================================
    # Tenancy operations (delegated to relational)
    # =========================================================================

    async def create_organization(self, org: Organization) -> Organization:
        """Create a new organization."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.create_organization(org)

    async def get_organization(self, org_id: UUID) -> Organization | None:
        """Get an organization by ID."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_organization(org_id)

    async def get_organization_by_slug(self, slug: str) -> Organization | None:
        """Get an organization by slug."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_organization_by_slug(slug)

    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create a new workspace."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.create_workspace(workspace)

    async def get_workspace(self, workspace_id: UUID) -> Workspace | None:
        """Get a workspace by ID."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_workspace(workspace_id)

    async def list_workspaces(self, organization_id: UUID) -> list[Workspace]:
        """List all workspaces in an organization."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.list_workspaces(organization_id)

    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.create_namespace(namespace)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_namespace(namespace_id)

    async def get_namespace_by_slug(self, workspace_id: UUID, slug: str) -> MemoryNamespace | None:
        """Get a namespace by workspace ID and slug."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_namespace_by_slug(workspace_id, slug)

    async def list_namespaces(self, workspace_id: UUID) -> list[MemoryNamespace]:
        """List all namespaces in a workspace."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.list_namespaces(workspace_id)

    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update a namespace."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.update_namespace(namespace)

    # =========================================================================
    # Document operations (delegated to relational)
    # =========================================================================

    async def create_document(self, document: Document) -> Document:
        """Create a new document."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.create_document(document)

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_document(document_id)

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents in a namespace."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.list_documents(namespace_id, status=status, limit=limit, offset=offset)

    async def update_document(self, document: Document) -> Document:
        """Update a document."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.update_document(document)

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and its chunks."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")

        # Delete chunks first
        if self.vector:
            await self.vector.delete_chunks_by_document(document_id)

        return await self.relational.delete_document(document_id)

    async def get_document_by_checksum(self, namespace_id: UUID, checksum: str) -> Document | None:
        """Get a document by its content checksum."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_document_by_checksum(namespace_id, checksum)

    # =========================================================================
    # Chunk operations (delegated to vector)
    # =========================================================================

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create a new chunk with embedding."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.create_chunk(chunk)

    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks in a batch."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.create_chunks_batch(chunks)

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Get a chunk by ID."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.get_chunk(chunk_id)

    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Get all chunks for a document."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.get_chunks_by_document(document_id)

    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        filter_document_ids: list[UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.search_similar(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
            filter_document_ids=filter_document_ids,
        )

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Count chunks in a namespace."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.count_chunks(namespace_id)

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Chunk]:
        """List chunks in a namespace.

        Args:
            namespace_id: Namespace ID
            limit: Maximum chunks to return
            offset: Offset for pagination

        Returns:
            List of chunks
        """
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.list_chunks(namespace_id, limit=limit, offset=offset)

    async def count_entities(self, namespace_id: UUID) -> int:
        """Count entities in a namespace."""
        if self.graph:
            return await self.graph.count_entities(namespace_id)
        return 0

    # =========================================================================
    # Entity operations (cross-backend)
    # =========================================================================

    async def create_entity(self, entity: Entity) -> Entity:
        """Create an entity in both graph and relational stores."""
        # Store in graph for relationships and traversal
        if self.graph:
            entity = await self.graph.create_entity(entity)
        return entity

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        if self.graph:
            return await self.graph.get_entity(entity_id)
        return None

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        """Get an entity by name and type."""
        if self.graph:
            return await self.graph.get_entity_by_name(namespace_id, name, entity_type)
        return None

    async def update_entity(self, entity: Entity) -> Entity:
        """Update an entity."""
        if self.graph:
            entity = await self.graph.update_entity(entity)
        return entity

    async def delete_entity(self, entity_id: UUID) -> bool:
        """Delete an entity."""
        if self.graph:
            return await self.graph.delete_entity(entity_id)
        return False

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities in a namespace."""
        if self.graph:
            return await self.graph.list_entities(namespace_id, entity_type=entity_type, limit=limit, offset=offset)
        return []

    async def update_entity_embedding(self, entity_id: UUID, embedding: list[float], model: str) -> None:
        """Update the embedding for an entity."""
        if self.vector:
            await self.vector.update_entity_embedding(entity_id, embedding, model)

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        """Search for similar entities."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.search_similar_entities(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
        )

    # =========================================================================
    # Relationship operations (delegated to graph)
    # =========================================================================

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a relationship between entities."""
        if not self.graph:
            raise RuntimeError("Graph backend not configured")
        return await self.graph.create_relationship(relationship)

    async def get_relationship(self, relationship_id: UUID) -> Relationship | None:
        """Get a relationship by ID."""
        if self.graph:
            return await self.graph.get_relationship(relationship_id)
        return None

    async def delete_relationship(self, relationship_id: UUID) -> bool:
        """Delete a relationship."""
        if self.graph:
            return await self.graph.delete_relationship(relationship_id)
        return False

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        if self.graph:
            return await self.graph.get_entity_relationships(
                entity_id, direction=direction, relationship_types=relationship_types, limit=limit
            )
        return []

    # =========================================================================
    # Episode operations (delegated to graph)
    # =========================================================================

    async def create_episode(self, episode: Episode) -> Episode:
        """Create an episode."""
        if not self.graph:
            raise RuntimeError("Graph backend not configured")
        return await self.graph.create_episode(episode)

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        """Get an episode by ID."""
        if self.graph:
            return await self.graph.get_episode(episode_id)
        return None

    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """List episodes in a time range."""
        if self.graph:
            return await self.graph.list_episodes(namespace_id, start_time=start_time, end_time=end_time, limit=limit)
        return []

    # =========================================================================
    # Graph traversal (delegated to graph)
    # =========================================================================

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
        if self.graph:
            return await self.graph.find_paths(
                namespace_id,
                source_entity_id,
                target_entity_id,
                max_depth=max_depth,
                relationship_types=relationship_types,
            )
        return []

    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity."""
        if self.graph:
            return await self.graph.get_neighborhood(
                entity_id, depth=depth, relationship_types=relationship_types, limit=limit
            )
        return {"entities": [], "relationships": []}

    # =========================================================================
    # Batch operations (optimized for parallel fetching)
    # =========================================================================

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query.

        Args:
            entity_ids: List of entity IDs to fetch

        Returns:
            Dictionary mapping entity ID to Entity object
        """
        if not entity_ids:
            return {}
        if self.graph:
            return await self.graph.get_entities_batch(entity_ids)
        return {}

    async def get_documents_batch(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            Dictionary mapping document ID to Document object
        """
        if not document_ids:
            return {}
        if self.relational:
            return await self.relational.get_documents_batch(document_ids)
        return {}

    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Get neighborhoods for multiple entities in a single query.

        Args:
            entity_ids: List of entity IDs
            depth: Max traversal depth
            relationship_types: Optional relationship type filter
            limit_per_entity: Max nodes per entity neighborhood

        Returns:
            Dictionary mapping entity ID to neighborhood data
        """
        if not entity_ids:
            return {}
        if self.graph:
            return await self.graph.get_neighborhoods_batch(
                entity_ids,
                depth=depth,
                relationship_types=relationship_types,
                limit_per_entity=limit_per_entity,
            )
        return {}

    # =========================================================================
    # Event operations (delegated to event store)
    # =========================================================================

    async def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append an event to the log."""
        if not self.event_store:
            raise RuntimeError("Event store not configured")
        return await self.event_store.append_event(event)

    async def append_events_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events in a batch."""
        if not self.event_store:
            raise RuntimeError("Event store not configured")
        return await self.event_store.append_events_batch(events)

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
        if not self.event_store:
            raise RuntimeError("Event store not configured")
        return await self.event_store.get_events(
            namespace_id,
            event_types=event_types,
            resource_type=resource_type,
            resource_id=resource_id,
            after=after,
            before=before,
            limit=limit,
            offset=offset,
        )

    # =========================================================================
    # Sync checkpoint operations (delegated to relational)
    # =========================================================================

    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        """Get the last sync checkpoint for a source."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_sync_checkpoint(namespace_id, source)

    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """Set the sync checkpoint for a source."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        await self.relational.set_sync_checkpoint(namespace_id, source, checkpoint)
