"""Storage coordinator that orchestrates all backends.

The coordinator provides a unified interface to all storage backends
(PostgreSQL, pgvector, Neo4j) and handles cross-cutting concerns like
transaction coordination and consistency.
"""

from __future__ import annotations

import asyncio
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
        """Connect all configured backends in parallel.

        All backend connections are initiated concurrently using asyncio.gather
        for faster startup when multiple backends are configured.
        """
        if self._connected:
            return

        logger.info("Connecting storage backends...")

        # Build list of connection tasks to run in parallel
        tasks = []
        if self.relational:
            tasks.append(self.relational.connect())
        if self.vector:
            tasks.append(self.vector.connect())
        if self.graph:
            tasks.append(self.graph.connect())
        if self.event_store:
            tasks.append(self.event_store.connect())

        # Connect all backends concurrently
        if tasks:
            await asyncio.gather(*tasks)

        self._connected = True
        logger.info("All storage backends connected")

    async def disconnect(self) -> None:
        """Disconnect all backends in parallel.

        All backend disconnections are initiated concurrently for faster shutdown.
        """
        if not self._connected:
            return

        logger.info("Disconnecting storage backends...")

        # Build list of disconnection tasks to run in parallel
        tasks = []
        if self.event_store:
            tasks.append(self.event_store.disconnect())
        if self.graph:
            tasks.append(self.graph.disconnect())
        if self.vector:
            tasks.append(self.vector.disconnect())
        if self.relational:
            tasks.append(self.relational.disconnect())

        # Disconnect all backends concurrently
        if tasks:
            await asyncio.gather(*tasks)

        self._connected = False
        logger.info("All storage backends disconnected")

    async def health_check(self) -> StorageHealth:
        """Check health of all backends (parallel)."""
        health = StorageHealth()

        # Build list of health check coroutines to run in parallel
        checks: list[tuple[str, Any]] = []
        if self.relational:
            checks.append(("relational", self.relational.is_healthy()))
        if self.vector:
            checks.append(("vector", self.vector.is_healthy()))
        if self.graph:
            checks.append(("graph", self.graph.is_healthy()))
        if self.event_store:
            checks.append(("event_store", self.event_store.is_healthy()))

        if checks:
            results = await asyncio.gather(*[coro for _, coro in checks], return_exceptions=True)
            for (name, _), result in zip(checks, results):
                # Treat exceptions as unhealthy
                setattr(health, name, result is True)

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

    async def create_namespace_version(
        self,
        workspace_id: UUID,
        slug: str,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        """Create a new version of a namespace.

        If previous_version is provided, increments its version number and links to it.
        The previous version is marked as inactive.

        Args:
            workspace_id: Workspace ID
            slug: Namespace slug
            previous_version: The previous version to supersede (if any)

        Returns:
            New namespace version
        """
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.create_namespace_version(workspace_id, slug, previous_version=previous_version)

    async def deactivate_namespace(self, namespace_id: UUID) -> None:
        """Mark a namespace version as inactive.

        Args:
            namespace_id: ID of the namespace to deactivate
        """
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        await self.relational.deactivate_namespace(namespace_id)

    # =========================================================================
    # Document operations (delegated to relational)
    # =========================================================================

    async def create_document(self, document: Document) -> Document:
        """Create a new document."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        import time as _time

        _t0 = _time.perf_counter()
        result = await self.relational.create_document(document)
        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="postgresql",
            operation="create_document",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=1,
            namespace_id=document.namespace_id,
        )
        return result

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

    async def get_documents_by_checksums(self, namespace_id: UUID, checksums: list[str]) -> dict[str, Document]:
        """Fetch documents by content checksums in a single query.

        Used for batch deduplication to avoid N serial DB queries.

        Args:
            namespace_id: Namespace to search in
            checksums: List of content checksums to look up

        Returns:
            Dictionary mapping checksum to Document (only for existing documents)
        """
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_documents_by_checksums(namespace_id, checksums)  # type: ignore[unresolved-attribute]

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
        import time as _time

        _t0 = _time.perf_counter()
        result = await self.vector.create_chunks_batch(chunks)
        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="pgvector",
            operation="create_chunks_batch",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=len(chunks),
            namespace_id=chunks[0].namespace_id if chunks else None,
        )
        return result

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

    async def get_chunks_batch(self, chunk_ids: list[UUID]) -> dict[UUID, Chunk]:
        """Fetch multiple chunks by ID in a single query.

        Args:
            chunk_ids: List of chunk IDs to fetch

        Returns:
            Dictionary mapping chunk ID to Chunk (only for existing chunks)
        """
        if not chunk_ids:
            return {}
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.get_chunks_batch(chunk_ids)

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
        import time as _time

        _t0 = _time.perf_counter()
        result = await self.vector.search_similar(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
            filter_document_ids=filter_document_ids,
        )
        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="pgvector",
            operation="search_similar_chunks",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=len(result),
            namespace_id=namespace_id,
        )
        return result

    async def search_fulltext_chunks(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",
    ) -> list[tuple[Chunk, float]]:
        """Search chunks using PostgreSQL full-text search."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        import time as _time

        _t0 = _time.perf_counter()
        result = await self.vector.search_fulltext(
            namespace_id,
            query_text,
            limit=limit,
            language=language,
        )
        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="pgvector",
            operation="search_fulltext_chunks",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=len(result),
            namespace_id=namespace_id,
        )
        return result

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
        """Create an entity in both graph and vector stores (parallel)."""
        import time as _time

        _t0 = _time.perf_counter()
        # Parallel writes to graph + vector, matching update_entity pattern
        if self.graph and self.vector:
            graph_result, _ = await asyncio.gather(
                self.graph.create_entity(entity),
                self.vector.create_entity(entity),
            )
            entity = graph_result
        elif self.graph:
            entity = await self.graph.create_entity(entity)
        elif self.vector:
            await self.vector.create_entity(entity)
        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="graph+vector",
            operation="create_entity",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=1,
            namespace_id=entity.namespace_id,
        )
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
        """Update an entity in both graph and vector stores (parallel)."""
        if self.graph and self.vector:
            graph_result, _ = await asyncio.gather(
                self.graph.update_entity(entity),
                self.vector.update_entity(entity),
            )
            return graph_result
        if self.graph:
            return await self.graph.update_entity(entity)
        if self.vector:
            await self.vector.update_entity(entity)
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

    async def update_entity_embeddings_batch(self, updates: list[tuple[UUID, list[float], str]]) -> int:
        """Update embeddings for multiple entities in a single transaction."""
        if self.vector and hasattr(self.vector, "update_entity_embeddings_batch"):
            return await self.vector.update_entity_embeddings_batch(updates)
        # Fallback to individual updates (sequential)
        if self.vector:
            for entity_id, embedding, model in updates:
                await self.vector.update_entity_embedding(entity_id, embedding, model)
            return len(updates)
        return 0

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
        import time as _time

        _t0 = _time.perf_counter()
        result = await self.vector.search_similar_entities(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
        )
        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="pgvector",
            operation="search_similar_entities",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=len(result),
            namespace_id=namespace_id,
        )
        return result

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 200,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities across graph and vector backends.

        Uses MERGE semantics: creates new entities, updates existing ones
        matched by (namespace_id, name, entity_type).

        Returns list of (entity, is_new) tuples.
        """
        if not entities:
            return []

        import time as _time

        _t0 = _time.perf_counter()

        results: list[tuple[Entity, bool]] = []

        # Upsert in graph and vector backends in parallel
        has_graph = self.graph and hasattr(self.graph, "upsert_entities_batch")
        has_vector = self.vector and hasattr(self.vector, "upsert_entities_batch")
        logger.debug(
            f"upsert_entities_batch: {len(entities)} entities, " f"has_graph={has_graph}, has_vector={has_vector}"
        )

        if has_graph and has_vector:
            graph_results, _ = await asyncio.gather(
                self.graph.upsert_entities_batch(namespace_id, entities, batch_size=batch_size),
                self.vector.upsert_entities_batch(namespace_id, entities, batch_size=batch_size),  # type: ignore[unresolved-attribute]
            )
            results = graph_results
        elif has_graph:
            results = await self.graph.upsert_entities_batch(namespace_id, entities, batch_size=batch_size)
        elif has_vector:
            results = await self.vector.upsert_entities_batch(namespace_id, entities, batch_size=batch_size)  # type: ignore[unresolved-attribute]

        # Fallback: if no backend returned results, create synthetic results
        # to ensure callers always get one result per input entity
        if not results:
            logger.debug(f"upsert_entities_batch: using fallback synthetic results for {len(entities)} entities")
            results = [(entity, True) for entity in entities]

        logger.debug(f"upsert_entities_batch: returning {len(results)} results for {len(entities)} input entities")

        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="graph+vector",
            operation="upsert_entities_batch",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=len(entities),
            namespace_id=namespace_id,
        )
        return results

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 50,
    ) -> int:
        """Batch create relationships in the graph backend.

        Returns the number of relationships created.
        """
        if not relationships:
            return 0

        import time as _time

        _t0 = _time.perf_counter()

        count = 0
        if self.graph and hasattr(self.graph, "create_relationships_batch"):
            count = await self.graph.create_relationships_batch(relationships, batch_size=batch_size)

        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="graph",
            operation="create_relationships_batch",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=count,
            namespace_id=relationships[0].namespace_id if relationships else None,
        )
        return count

    # =========================================================================
    # Relationship operations (delegated to graph)
    # =========================================================================

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a relationship between entities."""
        if not self.graph:
            raise RuntimeError("Graph backend not configured")
        import time as _time

        _t0 = _time.perf_counter()
        result = await self.graph.create_relationship(relationship)
        from khora.telemetry import get_collector

        get_collector().record_storage_op(
            backend="graph",
            operation="create_relationship",
            latency_ms=(_time.perf_counter() - _t0) * 1000,
            record_count=1,
            namespace_id=relationship.namespace_id,
        )
        return result

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

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        """List all relationships in a namespace."""
        if self.graph:
            return await self.graph.list_relationships(
                namespace_id, relationship_type=relationship_type, limit=limit, offset=offset
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
            import time as _time

            _t0 = _time.perf_counter()
            result = await self.graph.get_neighborhoods_batch(
                entity_ids,
                depth=depth,
                relationship_types=relationship_types,
                limit_per_entity=limit_per_entity,
            )
            from khora.telemetry import get_collector

            get_collector().record_storage_op(
                backend="graph",
                operation="get_neighborhoods_batch",
                latency_ms=(_time.perf_counter() - _t0) * 1000,
                record_count=len(result),
                # No namespace_id available — entity_ids don't carry namespace info
            )
            return result
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
