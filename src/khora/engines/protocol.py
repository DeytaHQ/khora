"""Engine protocol defining the interface all memory engines must implement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from uuid import UUID

    from khora.core.models import Document, Entity, MemoryNamespace
    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig
    from khora.filter import FilterNode
    from khora.khora import BatchResult, RecallResult, RememberResult, Stats
    from khora.query import SearchMode


@runtime_checkable
class MemoryEngineProtocol(Protocol):
    """Protocol all memory engines must implement.

    Engines encapsulate the full implementation of memory storage and retrieval.
    The Khora facade delegates all operations to the configured engine.
    """

    # Each engine declares the ``SearchMode`` values it implements honestly.
    # ``recall()`` raises ``EngineCapabilityError`` for any mode outside this
    # set rather than silently degrading. See ``khora.exceptions``.
    supported_modes: ClassVar[frozenset[SearchMode]]

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Connect to all storage backends."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components.

        Returns:
            Dict with status and component health details
        """
        ...

    # =========================================================================
    # Core Operations (REQUIRED)
    # =========================================================================

    async def remember(
        self,
        content: str,
        namespace_id: UUID,
        *,
        title: str = "",
        source: str = "",
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        external_id: str | None = None,
    ) -> RememberResult:
        """Store content in the memory engine.

        Args:
            content: Content to remember
            namespace_id: Target namespace UUID
            title: Optional title for the content
            source: Optional source identifier
            source_type: Provenance category (e.g. "library", "api", "file").
            source_name: Optional provider-level identifier (e.g. "slack", "linear").
            source_url: Optional original-source URL.
            source_timestamp: Optional original-source timestamp. When provided,
                persists directly to ``Document.source_timestamp``.
            metadata: Optional metadata
            skill_name: Extraction skill to use
            entity_types: Required entity types to extract
            relationship_types: Required relationship types to extract
            expertise: Optional expertise config for domain-specific extraction
            extraction_config_hash: Optional hash of the extraction config for change detection
            chunk_strategy: Override chunking strategy for this call only.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.
            external_id: Optional caller-supplied external identifier for the document.
                Must be None or a non-blank string (max 512 chars).
                Raises ValueError if constraints are violated.

        Returns:
            RememberResult with details
        """
        ...

    async def recall(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
        mode: SearchMode = ...,
        min_similarity: float = 0.0,
        # Temporal parameters (optional — engines may ignore these)
        temporal_filter: Any | None = None,
        recency_bias: float | None = None,
        filter_ast: FilterNode | None = None,
    ) -> RecallResult:
        """Recall memories relevant to a query.

        Args:
            query: Query text
            namespace_id: Namespace to search in
            limit: Maximum results to return
            mode: Search mode (VECTOR, GRAPH, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            temporal_filter: Optional temporal filter for time-scoped retrieval.
                Type varies by engine (e.g. TemporalFilter for Skeleton).
                Engines that do not support temporal filtering may ignore this.
            recency_bias: Optional recency bias weight (0.0–1.0).
                Engines that do not support recency biasing may ignore this.
            filter_ast: Optional canonical recall-filter AST. Engines that do
                not filter may ignore it.

        Returns:
            RecallResult with matched memories
        """
        ...

    async def forget(self, document_id: UUID, namespace_id: UUID | None) -> bool:
        """Remove a memory from the engine.

        Args:
            document_id: ID of the document to remove
            namespace_id: Namespace for verification (optional)

        Returns:
            True if deleted, False if not found
        """
        ...

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        max_concurrent: int = 10,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
    ) -> BatchResult:
        """Store multiple documents with automatic optimization.

        Args:
            documents: List of document dicts with keys: content, title, source,
                source_type, source_name, source_url, source_timestamp, metadata,
                external_id (optional caller-supplied external identifier).
                Top-level source_type/source_name/source_url/source_timestamp
                per-doc keys override the corresponding kwargs.
            namespace_id: Target namespace UUID
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Deduplicate entities across documents
            infer_relationships: Infer relationships after ingestion
            on_progress: Callback(processed_count, total_count) for progress updates
            entity_types: Required entity types to extract
            relationship_types: Required relationship types to extract
            expertise: Optional expertise config for domain-specific extraction
            extraction_config_hash: Optional hash of the extraction config for change detection
            chunk_strategy: Override chunking strategy for this call only.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.

        Returns:
            BatchResult with aggregated statistics
        """
        ...

    # =========================================================================
    # Namespace Operations
    # =========================================================================

    async def create_namespace(
        self,
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace.

        Args:
            config_overrides: Optional configuration overrides

        Returns:
            Created MemoryNamespace
        """
        ...

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID.

        Args:
            namespace_id: Namespace UUID

        Returns:
            MemoryNamespace or None if not found
        """
        ...

    # =========================================================================
    # Entity Operations (optional - return empty/None if not supported)
    # =========================================================================

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to a namespace.

        Args:
            entity_id: Entity UUID
            namespace_id: Required — returns None when the entity belongs to a
                different namespace (prevents cross-tenant IDOR)

        Returns:
            Entity or None if not found / cross-namespace
        """
        ...

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List entities in a namespace.

        Args:
            namespace_id: Namespace UUID
            entity_type: Optional filter by entity type
            limit: Maximum entities to return

        Returns:
            List of Entities
        """
        ...

    async def find_related_entities(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity.

        Args:
            entity_id: Source entity UUID
            namespace_id: Namespace UUID
            max_depth: Maximum traversal depth
            limit: Maximum entities to return

        Returns:
            List of (Entity, score) tuples
        """
        ...

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id``.

        Args:
            document_id: Document UUID
            namespace_id: Caller's namespace; out-of-namespace rows return ``None``
                (IDOR).

        Returns:
            Document or None if not found (or not in this namespace)
        """
        ...

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
    ) -> list[Document]:
        """List documents in a namespace.

        Args:
            namespace_id: Namespace UUID
            limit: Maximum documents to return

        Returns:
            List of Documents
        """
        ...

    async def search_entities(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity.

        Args:
            query: Search query text
            namespace_id: Namespace UUID
            limit: Maximum entities to return

        Returns:
            List of matching Entities (most similar first)
        """
        ...

    # =========================================================================
    # Stats
    # =========================================================================

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts and last activity time for a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            Stats with:
            - documents: Count of documents in the namespace
            - chunks: Count of document chunks
            - entities: Count of extracted entities
            - relationships: Count of entity relationships
            - last_activity_at: Timestamp of most recent document (None if namespace is empty)
        """
        ...
