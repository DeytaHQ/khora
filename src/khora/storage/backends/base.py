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
    from khora.core.models.recall import DocumentProjection
    from khora.dream.plan import OpKind
    from khora.filter.ast import FilterNode


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
    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id``.

        Returns ``None`` if the document does not exist OR belongs to a
        different namespace — the caller's namespace is the authority.
        The ``namespace_id`` filter prevents cross-tenant document access
        by id (IDOR).
        """
        ...

    @abstractmethod
    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        updated_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents in a namespace."""
        ...

    @abstractmethod
    async def claim_orphaned_documents(
        self,
        namespace_id: UUID,
        *,
        pending_before: datetime,
        processing_before: datetime,
        limit: int = 100,
    ) -> list[Document]:
        """Atomically claim stale orphaned documents for crash recovery.

        Selects documents that are either ``pending`` and older than
        ``pending_before`` OR ``processing`` and older than
        ``processing_before``, flips the claimed rows to ``processing`` (with a
        fresh ``updated_at``), and returns them. On PostgreSQL the claim is
        serialized with ``FOR UPDATE SKIP LOCKED`` so concurrent recovery loops
        never claim the same document. SQLite (single-writer) and SurrealDB
        perform the same claim without row locking.

        Unlike :meth:`list_documents` (a pure read), this method mutates state.
        """
        ...

    @abstractmethod
    async def update_document(self, document: Document) -> Document:
        """Update a document."""
        ...

    @abstractmethod
    async def delete_document(self, document_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a document, scoped to ``namespace_id``.

        Returns ``False`` if the document does not exist OR belongs to a
        different namespace — the caller's namespace is the authority. The
        ``namespace_id`` filter prevents cross-tenant deletion by id.
        """
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

    @abstractmethod
    async def get_document_by_external_id(
        self,
        external_id: str | None,
        *,
        namespace_id: UUID,
    ) -> Document | None:
        """Get a document by its caller-supplied external_id.

        Unlike ``get_document_by_checksum``, this lookup does NOT filter by
        status — it returns ``COMPLETED``, ``PROCESSING``, and ``FAILED`` rows
        so callers can self-heal a failed extraction on the next replace
        against the same ``external_id``.

        Returns ``None`` immediately if ``external_id`` is ``None`` (guard).
        """
        ...

    @abstractmethod
    async def get_documents_by_external_ids(
        self,
        external_ids: list[str],
        *,
        namespace_id: UUID,
    ) -> dict[str, Document]:
        """Batch equivalent of :meth:`get_document_by_external_id`.

        Returns a mapping of ``external_id -> Document`` for every external_id
        that currently resolves to a row within the namespace. Like the single
        lookup, this does NOT filter by status (self-heal).
        ``None`` / empty entries in ``external_ids`` are skipped.

        Empty input returns ``{}`` immediately.
        """
        ...

    async def get_documents_batch(self, document_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query, scoped to ``namespace_id``.

        Documents belonging to any other namespace are silently dropped
        from the result to prevent cross-tenant IDOR (IDOR family).

        Returns dictionary mapping document ID to Document object.
        """
        ...

    async def get_document_sources_batch(
        self, document_ids: list[UUID], *, namespace_id: UUID
    ) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution,
        scoped to ``namespace_id``.

        Returns a column-limited projection (no content, processing stats,
        or mutable state) for display and linking purposes. Documents in
        other namespaces are silently dropped from the result (IDOR family).

        Args:
            document_ids: List of document IDs to fetch
            namespace_id: Caller's namespace; documents belonging to any
                other namespace are silently dropped from the result.

        Returns:
            Dictionary mapping document ID to DocumentSource
        """
        ...

    async def get_document_projections_batch(
        self,
        document_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> dict[UUID, DocumentProjection]:
        """Fetch full ``DocumentProjection`` rows for recall responses.

        Returns the typed projection shape used by ``Khora.recall()``:
        ``id``, ``created_at``, ``source_type``, ``title``, ``external_id``,
        ``source``, ``source_name``, ``source_url``, ``content_type``,
        ``source_timestamp``, ``metadata``.

        Distinct from ``get_document_sources_batch`` (which returns the
        narrower ``DocumentSource`` for entity-source attribution) so the
        two consumers can evolve their column sets independently.

        Args:
            document_ids: List of document IDs to fetch
            namespace_id: Namespace scope — rows from other namespaces are
                filtered at the query layer (security close-out).

        Returns:
            Dictionary mapping document ID to DocumentProjection
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
    async def get_chunk(self, chunk_id: UUID, *, namespace_id: UUID) -> Chunk | None:
        """Get a chunk by ID, scoped to ``namespace_id``.

        Returns ``None`` if the chunk does not exist OR belongs to a
        different namespace — the caller's namespace is the authority.
        The ``namespace_id`` filter prevents cross-tenant chunk access
        by id (IDOR).
        """
        ...

    @abstractmethod
    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        """Get multiple chunks by ID in a single query, scoped to ``namespace_id``.

        Args:
            chunk_ids: List of chunk IDs to fetch.
            namespace_id: Caller's namespace; chunks belonging to any
                other namespace are silently dropped from the result
                to prevent cross-tenant IDOR.

        Returns:
            Dictionary mapping chunk ID to Chunk (only for existing
            chunks within ``namespace_id``).
        """
        ...

    @abstractmethod
    async def get_chunks_by_document(self, document_id: UUID, *, namespace_id: UUID) -> list[Chunk]:
        """Get all chunks for a document, scoped to ``namespace_id``.

        Returns an empty list when the document does not belong to the
        caller's namespace. The namespace filter prevents cross-tenant
        chunk access by document id.
        """
        ...

    @abstractmethod
    async def delete_chunks_by_document(
        self,
        document_id: UUID,
        *,
        namespace_id: UUID,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete all chunks for a document, scoped to ``namespace_id``.

        When *session* is provided the caller owns the transaction —
        no commit is issued.  When ``None``, a private session is used
        and committed automatically. The ``namespace_id`` filter prevents
        cross-tenant deletion by document id.
        """
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
    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> None:
        """Update an entity record in PostgreSQL, scoped to ``namespace_id``.

        Updates are skipped silently when the entity belongs to a different
        namespace — prevents cross-tenant entity mutation by id.
        """
        ...

    @abstractmethod
    async def entity_exists(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Check if an entity exists in PostgreSQL within ``namespace_id``.

        Returns ``False`` if the entity does not exist OR belongs to a
        different namespace. The ``namespace_id`` filter prevents
        cross-tenant entity-existence enumeration (IDOR).
        """
        ...

    @abstractmethod
    async def update_entity_embedding(
        self,
        entity_id: UUID,
        embedding: list[float],
        model: str,
        *,
        namespace_id: UUID,
    ) -> None:
        """Update the embedding for an entity, scoped to ``namespace_id``.

        Updates are skipped silently when the entity belongs to a different
        namespace — prevents cross-tenant embedding mutation.
        """
        ...

    async def update_entity_embeddings_batch(
        self,
        updates: list[tuple[UUID, list[float], str]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Update embeddings for multiple entities in a single transaction.

        Updates are restricted to the caller's namespace; ids outside it
        are silently skipped from the count.
        """
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
        filter_ast: FilterNode | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search chunks using PostgreSQL full-text search.

        Uses ts_rank on the content_tsv generated column.

        Returns list of (chunk, rank_score) tuples.

        ``filter_ast`` is the canonical recall-filter AST. The relational
        ``chunks`` table lacks the denormalized filter columns, so backends
        REFUSE under an active filter (return ``[]``) rather than smuggle
        unfiltered rows; the filtered BM25 path is the ``khora_chunks``
        temporal store.
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
    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id``.

        Returns ``None`` if the entity does not exist OR belongs to a
        different namespace. Prevents cross-tenant entity access by id
        (IDOR).
        """
        ...

    @abstractmethod
    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        """Get an entity by name and type (for deduplication)."""
        ...

    @abstractmethod
    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
        """Update an entity, scoped to ``namespace_id``.

        Updates are skipped when the entity belongs to a different
        namespace — prevents cross-tenant entity mutation by id.
        """
        ...

    @abstractmethod
    async def delete_entity(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete an entity and its relationships, scoped to ``namespace_id``.

        Returns ``False`` if the entity does not exist OR belongs to a
        different namespace. Prevents cross-tenant deletion by id.
        """
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
    async def get_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> Relationship | None:
        """Get a relationship by ID, scoped to ``namespace_id``.

        Returns ``None`` if the relationship does not exist OR belongs to
        a different namespace. Prevents cross-tenant relationship access
        by id (IDOR).
        """
        ...

    @abstractmethod
    async def delete_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a relationship, scoped to ``namespace_id``.

        Returns ``False`` if the relationship does not exist OR belongs to
        a different namespace. Prevents cross-tenant deletion by id.
        """
        ...

    @abstractmethod
    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity, scoped to ``namespace_id``.

        Returns an empty list if the entity does not belong to the
        caller's namespace. Edges that cross into other namespaces are
        excluded from the result. Prevents cross-tenant subgraph leakage
        (IDOR family).
        """
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
    async def get_episode(self, episode_id: UUID, *, namespace_id: UUID) -> Episode | None:
        """Get an episode by ID, scoped to ``namespace_id``.

        Returns ``None`` if the episode does not exist OR belongs to a
        different namespace. Prevents cross-tenant episode access by id
        (IDOR).
        """
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
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        namespace_id: UUID,
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
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity up to a certain depth,
        scoped to ``namespace_id``.

        The seed entity is verified to belong to ``namespace_id``; the
        traversal MUST NOT cross into other namespaces. Returns an empty
        structure when the seed is in a different namespace. Prevents
        cross-tenant subgraph leakage (IDOR family).
        """
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

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query, scoped to ``namespace_id``.

        Entities belonging to any other namespace are silently dropped
        from the result to prevent cross-tenant IDOR (IDOR family).

        Returns dictionary mapping entity ID to Entity object.
        """
        ...

    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Get neighborhoods for multiple entities, scoped to ``namespace_id``.

        Seed entities outside ``namespace_id`` are silently dropped; the
        traversal MUST NOT cross into other namespaces. Prevents
        cross-tenant subgraph leakage (IDOR family).

        Returns dictionary mapping entity ID to neighborhood data.
        """
        ...

    async def count_entities(self, namespace_id: UUID) -> int:
        """Count entities in a namespace."""
        ...

    async def count_relationships(self, namespace_id: UUID) -> int:
        """Count relationships in a namespace."""
        ...

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 100,
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using MERGE semantics.

        For each entity, creates it if new or updates if existing
        (matched by name + type within namespace). On match the input
        entity's ``id`` MUST be synced in place to the stored id so
        relationship endpoints resolve (the #806 id-remap contract).

        Returns list of (entity, is_new) tuples.
        """
        ...

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 100,
    ) -> int:
        """Batch create relationships.

        Returns the number of relationships created.
        """
        ...

    # Dream bi-temporal mirror verbs (#1271) — optional. The dream-apply
    # phase mirrors its PG-side soft-delete / rewrite / relabel to the graph
    # through these. They are dream-predicate-keyed (confidence + chunk
    # liveness, document-independent), NOT the document-replace-shaped
    # ``retire_orphaned_*`` primitives. ``GraphBackendBase`` provides a
    # capability-gated default that raises ``DreamBackendUnsupported`` so a
    # backend without native support degrades to a structured skip_reason
    # rather than silently no-op-ing or hard-deleting. The mirror wiring into
    # the orchestrator is #1272; this seam only declares the contract.

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        """The ``OpKind`` values this backend can mirror to the graph.

        The dream orchestrator (#1272) intersects the plan's op kinds with
        this set; ops outside it record a structured skip_reason instead of
        diverging the two stores. Empty set = no graph-mirror support.
        """
        ...

    async def soft_invalidate_relationships_batch(
        self,
        relationship_ids: list[UUID],
        *,
        namespace_id: UUID,
        invalidated_at: datetime,
    ) -> int:
        """Soft-delete relationships by id by stamping ``valid_until``.

        Mirrors ``prune_edges`` (the dream predicate: low-confidence +
        chunk-dead edges). Idempotent by id (only edges with a null
        ``valid_until`` are touched), namespace-scoped. Never hard-deletes.
        Returns the number of edges actually invalidated.
        """
        ...

    async def soft_retire_entities_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        retired_at: datetime,
        reason: str = "dream_consolidated",
    ) -> int:
        """Soft-retire entities by id, snapshotting the pre-state.

        Mirrors the absorbed-entity soft-delete in ``dedupe_entities``:
        snapshots the live node into a version record and stamps
        ``valid_until`` / ``version_valid_to`` on the original. Idempotent
        by id (only still-live entities are retired), namespace-scoped.
        Never hard-deletes. Returns the number of entities actually retired.
        """
        ...

    async def rewrite_relationship_endpoints_batch(
        self,
        rewrites: list[dict[str, Any]],
        *,
        namespace_id: UUID,
        rewritten_at: datetime,
    ) -> int:
        """Re-point relationship endpoints by id.

        Mirrors the absorbed-endpoint rewrite in ``dedupe_entities``. Each
        dict carries ``relationship_id``, ``source_entity_id``,
        ``target_entity_id`` (the post-rewrite endpoints), and
        ``relationship_type`` (the Cypher edge label - sanitized by backends
        that store types as labels). Idempotent by id, namespace-scoped.
        Returns the number of edges actually re-pointed.
        """
        ...

    async def rename_types_batch(
        self,
        renames: list[dict[str, str]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Relabel relationship types (Cypher edge labels).

        Mirrors ``normalize_schema``. Each dict carries ``old_type`` and
        ``new_type``. The relationship type is a Cypher edge label and CANNOT
        be ``$``-parameterized, so backends MUST route both ends through the
        shared ``sanitize_cypher_label`` hard-validation. Namespace-scoped.
        Returns the number of edges relabeled.
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
        namespace_id: UUID,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Get all events for a specific resource, scoped to ``namespace_id``.

        Returns an empty list if the resource belongs to a different
        namespace. Prevents cross-tenant audit-log leakage (the IDOR family /
        the IDOR family family).
        """
        ...

    @abstractmethod
    async def get_latest_event(
        self,
        resource_type: str,
        resource_id: UUID,
        *,
        namespace_id: UUID,
    ) -> MemoryEvent | None:
        """Get the latest event for a resource, scoped to ``namespace_id``.

        Returns ``None`` if the resource belongs to a different namespace.
        Prevents cross-tenant audit-log leakage (the IDOR family / the IDOR family family).
        """
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
