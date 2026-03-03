"""Engine protocol defining the interface all memory engines must implement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from khora.core.models import Document, Entity, MemoryNamespace
    from khora.memory_lake import BatchResult, RecallResult, RememberResult, Stats
    from khora.query import SearchMode


@runtime_checkable
class MemoryEngineProtocol(Protocol):
    """Protocol all memory engines must implement.

    Engines encapsulate the full implementation of memory storage and retrieval.
    The MemoryLake facade delegates all operations to the configured engine.
    """

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
        source_tool: str = "",
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
    ) -> RememberResult:
        """Store content in the memory engine.

        Args:
            content: Content to remember
            namespace_id: Target namespace UUID
            title: Optional title for the content
            source: Optional source identifier
            source_tool: Canonical SaaS tool identifier (e.g. "slack", "linear")
            metadata: Optional metadata
            skill_name: Extraction skill to use

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
        agentic: bool = False,
        raw: bool = False,
    ) -> RecallResult:
        """Recall memories relevant to a query.

        Args:
            query: Query text
            namespace_id: Namespace to search in
            limit: Maximum results to return
            mode: Search mode (VECTOR, GRAPH, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            agentic: If True, use multi-step agentic search
            raw: If True, skip all LLM features

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
    ) -> BatchResult:
        """Store multiple documents with automatic optimization.

        Args:
            documents: List of document dicts with keys: content, title, source, source_tool, metadata
            namespace_id: Target namespace UUID
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Deduplicate entities across documents
            infer_relationships: Infer relationships after ingestion
            on_progress: Callback(processed_count, total_count) for progress updates

        Returns:
            BatchResult with aggregated statistics
        """
        ...

    # =========================================================================
    # Namespace Operations
    # =========================================================================

    async def get_or_create_default_namespace(self) -> UUID:
        """Get or create a default namespace for simple usage.

        Returns:
            Default namespace UUID
        """
        ...

    async def create_namespace(
        self,
        name: str,
        workspace_id: UUID,
        *,
        description: str = "",
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace.

        Args:
            name: Namespace name
            workspace_id: Parent workspace ID
            description: Optional description
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

    async def ensure_namespace(
        self,
        name: str,
        *,
        description: str = "",
    ) -> UUID:
        """Get or create a namespace by name.

        Creates the default organization and workspace if they don't exist.

        Args:
            name: Namespace name (will be slugified)
            description: Optional description

        Returns:
            Namespace UUID
        """
        ...

    # =========================================================================
    # Entity Operations (optional - return empty/None if not supported)
    # =========================================================================

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: Entity UUID

        Returns:
            Entity or None if not found
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

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID.

        Args:
            document_id: Document UUID

        Returns:
            Document or None if not found
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
        """Get document/chunk/entity/relationship counts for a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            Stats with document/chunk/entity/relationship counts
        """
        ...
