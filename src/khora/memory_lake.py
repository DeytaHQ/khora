"""MemoryLake - Primary API for Khora Memory Lake.

This is the main entry point for using Khora as a library.
Provides a simple, unified interface for memory storage and retrieval.

The MemoryLake class is a thin facade that delegates to pluggable engines.
The default engine is "graphrag" which uses knowledge graphs, vectors, and LLM extraction.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, load_config
from khora.core.models import Chunk, Document, Entity, MemoryNamespace
from khora.query import SearchMode
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from khora.engines.protocol import MemoryEngineProtocol
    from khora.storage import StorageConfig, StorageCoordinator


@dataclass(slots=True, frozen=True)
class RememberResult:
    """Result of a remember operation."""

    document_id: UUID
    namespace_id: UUID
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BatchResult:
    """Result of remember_batch() operation."""

    total: int
    processed: int
    skipped: int
    failed: int
    chunks: int
    entities: int
    relationships: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Stats:
    """Namespace statistics."""

    documents: int
    chunks: int
    entities: int
    relationships: int


@dataclass(slots=True, frozen=True)
class RecallResult:
    """Result of a recall operation."""

    query: str
    namespace_id: UUID
    chunks: list[tuple[Chunk, float]]
    entities: list[tuple[Entity, float]]
    context_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryLake:
    """Primary interface for Khora Memory Lake.

    Provides a simple API for storing and retrieving memories:
    - remember(): Store content in the memory lake
    - recall(): Retrieve relevant memories for a query
    - forget(): Remove memories

    Can be used as a context manager for automatic connection handling.

    The MemoryLake is a facade that delegates to pluggable engines.
    The default engine is "graphrag" which uses knowledge graphs and vector embeddings.

    Usage:
        # Simplest - from env vars (KHORA_DATABASE_URL)
        async with MemoryLake() as lake:
            await lake.remember("Important fact...", namespace="my-ns")

        # Common - explicit database URL
        async with MemoryLake("postgresql://localhost/mydb") as lake:
            results = await lake.recall("What do I know about...", namespace="my-ns")

        # With graph backend
        async with MemoryLake("postgresql://...", graph_url="bolt://localhost:7687") as lake:
            ...

        # Explicit engine selection (same as default)
        async with MemoryLake("postgresql://...", engine="graphrag") as lake:
            ...

        # Full config
        async with MemoryLake(KhoraConfig(...)) as lake:
            ...
    """

    def __init__(
        self,
        database_url: str | KhoraConfig | None = None,
        *,
        engine: str = "graphrag",
        graph_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        storage_config: StorageConfig | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Memory Lake.

        Args:
            database_url: PostgreSQL URL, or full KhoraConfig, or None (reads KHORA_DATABASE_URL from env)
            engine: Engine to use (default: "graphrag")
            graph_url: Optional Neo4j/graph database URL (bolt://user:pass@host:port)
            embedding_model: Embedding model to use (default: text-embedding-3-small)
            storage_config: Storage configuration (derived from config if None) - deprecated
            engine_kwargs: Additional keyword arguments forwarded to the engine constructor
                (e.g., vectorcypher_config=VectorCypherConfig(...))

        Examples:
            # Simplest - from env vars
            lake = MemoryLake()

            # Common - explicit database
            lake = MemoryLake("postgresql://localhost/mydb")

            # With graph
            lake = MemoryLake("postgresql://...", graph_url="bolt://...")

            # Explicit engine selection
            lake = MemoryLake("postgresql://...", engine="graphrag")

            # Full config
            lake = MemoryLake(KhoraConfig(...))
        """
        # Handle overloaded first argument
        if isinstance(database_url, KhoraConfig):
            self._config = database_url
        elif isinstance(database_url, str):
            # Build config from URL parameters
            self._config = KhoraConfig(
                database_url=database_url,
                neo4j_url=graph_url,
            )
            # Override embedding model if non-default
            if embedding_model != "text-embedding-3-small":
                self._config.llm.embedding_model = embedding_model
        else:
            # None - load from env/file
            self._config = load_config()
            # Apply overrides if provided
            if graph_url:
                self._config.neo4j_url = graph_url
            if embedding_model != "text-embedding-3-small":
                self._config.llm.embedding_model = embedding_model

        # Store for deferred engine creation
        self._engine_name = engine
        self._storage_config = storage_config  # for backwards compat
        self._engine_kwargs = engine_kwargs or {}
        self._engine: MemoryEngineProtocol | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting Memory Lake...")

        from khora.engines import create_engine

        self._engine = create_engine(
            self._engine_name,
            self._config,
            storage_config=self._storage_config,
            **self._engine_kwargs,
        )
        await self._engine.connect()

        self._connected = True
        logger.info("Memory Lake connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting Memory Lake...")

        if self._engine:
            await self._engine.disconnect()
            self._engine = None

        self._connected = False
        logger.info("Memory Lake disconnected")

    async def __aenter__(self) -> MemoryLake:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def _get_engine(self) -> MemoryEngineProtocol:
        """Get the engine (internal use)."""
        if self._engine is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")
        return self._engine

    @property
    def storage(self) -> StorageCoordinator:
        """Get the storage coordinator for admin/management operations.

        Provides direct access to the underlying storage coordinator for
        managing namespaces and other administrative tasks not covered
        by the high-level API.

        For common operations, prefer the MemoryLake convenience methods:
        - lake.get_document() for document retrieval
        - lake.list_documents() for document listing
        - lake.search_entities() for entity search
        - lake.stats() for namespace statistics
        """
        engine = self._get_engine()
        if hasattr(engine, "_storage") and engine._storage:
            return engine._storage  # type: ignore[invalid-return-type]
        raise AttributeError("Current engine does not expose storage")

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def create_namespace(
        self,
        name: str,
        *,
        description: str = "",
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace.

        Args:
            name: Namespace name
            description: Optional description
            config_overrides: Optional configuration overrides

        Returns:
            Created MemoryNamespace
        """
        return await self._get_engine().create_namespace(
            name,
            description=description,
            config_overrides=config_overrides,
        )

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        return await self._get_engine().get_namespace(namespace_id)

    async def get_or_create_default_namespace(self) -> UUID:
        """Get or create a default namespace for simple usage."""
        return await self._get_engine().get_or_create_default_namespace()

    # =========================================================================
    # Core API: remember, recall, forget
    # =========================================================================

    async def remember(
        self,
        content: str,
        *,
        namespace: str | UUID | None = None,
        title: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        entity_types: list[str],
        relationship_types: list[str],
    ) -> RememberResult:
        """Store content in the memory lake.

        This is the primary method for adding memories. It:
        1. Creates a document
        2. Chunks the content
        3. Generates embeddings
        4. Extracts entities and relationships

        Args:
            content: Content to remember
            namespace: Namespace name, ID, or None for default
            title: Optional title for the content
            source: Optional source identifier
            metadata: Optional metadata
            skill_name: Extraction skill to use
            entity_types: Required entity types to extract
            relationship_types: Required relationship types to extract

        Returns:
            RememberResult with details
        """
        from khora.telemetry.context import clear_trace_id, ensure_trace_id

        ensure_trace_id()
        try:
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.remember", namespace_id=str(namespace_id), content_length=len(content)):
                return await self._get_engine().remember(
                    content,
                    namespace_id,
                    title=title,
                    source=source,
                    metadata=metadata,
                    skill_name=skill_name,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                )
        finally:
            clear_trace_id()

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str | UUID | None = None,
        skill_name: str = "general_entities",
        max_concurrent: int = 10,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> BatchResult:
        """Store multiple documents with automatic optimization.

        Handles internally:
        - Shared embedder with LRU cache (reused across batches)
        - Entity deduplication via EntityIndex
        - Multi-phase resolution (smart mode)
        - Relationship inference

        This is more efficient than calling remember() for each document
        as it processes documents in parallel with controlled concurrency
        and shares resources across documents.

        Args:
            documents: List of document dicts with keys:
                - content: str (required)
                - title: str (optional)
                - source: str (optional)
                - metadata: dict (optional)
            namespace: Namespace name, ID, or None for default
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Deduplicate entities across documents (default: True)
            infer_relationships: Infer relationships after ingestion (default: True)
            on_progress: Callback(processed_count, total_count) for progress updates
            entity_types: Required entity types to extract
            relationship_types: Required relationship types to extract

        Returns:
            BatchResult with aggregated statistics
        """
        from khora.telemetry.context import clear_trace_id, ensure_trace_id

        ensure_trace_id()
        try:
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.remember_batch", namespace_id=str(namespace_id), batch_size=len(documents)):
                return await self._get_engine().remember_batch(
                    documents,
                    namespace_id,
                    skill_name=skill_name,
                    max_concurrent=max_concurrent,
                    deduplicate=deduplicate,
                    infer_relationships=infer_relationships,
                    on_progress=on_progress,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                )
        finally:
            clear_trace_id()

    async def recall(
        self,
        query: str,
        *,
        namespace: str | UUID | None = None,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        agentic: bool = False,
        raw: bool = False,
    ) -> RecallResult:
        """Recall memories relevant to a query.

        This is the primary method for retrieving memories. It:
        1. Uses LLM to understand query (entities, temporal refs, etc.)
        2. Searches across vector, graph, and keyword indexes
        3. Fuses results using Reciprocal Rank Fusion
        4. Returns ranked results

        When agentic=True, uses multi-step exploration:
        1. Initial comprehensive search with query understanding
        2. Executes pre-computed follow-up queries for deeper exploration
        3. Explores under-represented sources
        4. Returns combined results with full trace

        When raw=True, skips all LLM features:
        - Query understanding
        - Entity linking
        - Reranking
        - HyDE expansion
        This is useful for benchmarks and simple searches.

        Args:
            query: Query text
            namespace: Namespace name, ID, or None for default
            limit: Maximum results to return
            mode: Search mode (VECTOR, GRAPH, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            agentic: If True, use multi-step agentic search (default: False)
            raw: If True, skip all LLM features (default: False)

        Returns:
            RecallResult with matched memories
        """
        from khora.telemetry.context import clear_trace_id, ensure_trace_id

        ensure_trace_id()
        try:
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.recall", namespace_id=str(namespace_id), query=query):
                return await self._get_engine().recall(
                    query,
                    namespace_id,
                    limit=limit,
                    mode=mode,
                    min_similarity=min_similarity,
                    agentic=agentic,
                    raw=raw,
                )
        finally:
            clear_trace_id()

    async def forget(
        self,
        document_id: UUID,
        *,
        namespace: str | UUID | None = None,
    ) -> bool:
        """Remove a memory from the lake.

        Args:
            document_id: ID of the document to remove
            namespace: Namespace for verification (optional)

        Returns:
            True if deleted, False if not found
        """
        namespace_id = None
        if namespace:
            namespace_id = await self._resolve_namespace(namespace)

        with trace_span(
            "khora.forget",
            namespace_id=str(namespace_id) if namespace_id else "",
            document_id=str(document_id),
        ):
            return await self._get_engine().forget(document_id, namespace_id)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        return await self._get_engine().get_entity(entity_id)

    async def list_entities(
        self,
        *,
        namespace: str | UUID | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List entities in a namespace."""
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().list_entities(namespace_id, entity_type=entity_type, limit=limit)

    async def find_related_entities(
        self,
        entity_id: UUID,
        *,
        namespace: str | UUID | None = None,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity."""
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().find_related_entities(
            entity_id,
            namespace_id,
            max_depth=max_depth,
            limit=limit,
        )

    # =========================================================================
    # Document Operations (Convenience Methods)
    # =========================================================================

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID.

        Args:
            document_id: Document UUID

        Returns:
            Document or None if not found
        """
        return await self._get_engine().get_document(document_id)

    async def list_documents(
        self,
        *,
        namespace: str | UUID | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """List documents in a namespace.

        Args:
            namespace: Namespace name, ID, or None for default
            limit: Maximum documents to return

        Returns:
            List of Documents
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().list_documents(namespace_id, limit=limit)

    async def search_entities(
        self,
        query: str,
        *,
        namespace: str | UUID | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity.

        Args:
            query: Search query text
            namespace: Namespace name, ID, or None for default
            limit: Maximum entities to return

        Returns:
            List of matching Entities (most similar first)
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().search_entities(query, namespace_id, limit=limit)

    async def stats(self, *, namespace: str | UUID | None = None) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace.

        Args:
            namespace: Namespace name, ID, or None for default

        Returns:
            Stats with document/chunk/entity/relationship counts
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().stats(namespace_id)

    async def ensure_namespace(
        self,
        name: str,
        *,
        description: str = "",
    ) -> UUID:
        """Get or create a namespace by name.

        This is a convenience method for simple usage where you just want
        a namespace by name without managing it explicitly.

        Args:
            name: Namespace name (will be slugified)
            description: Optional description

        Returns:
            Namespace UUID
        """
        return await self._get_engine().ensure_namespace(name, description=description)

    # =========================================================================
    # Helpers
    # =========================================================================

    async def _resolve_namespace(self, namespace: str | UUID | None) -> UUID:
        """Resolve a namespace reference to a UUID."""
        if namespace is None:
            return await self.get_or_create_default_namespace()

        if isinstance(namespace, UUID):
            return namespace

        # Try to parse as UUID
        try:
            return UUID(namespace)
        except ValueError:
            pass

        # Look up by slug (globally unique)
        engine = self._get_engine()
        if not hasattr(engine, "_storage") or engine._storage is None:
            raise RuntimeError("Engine does not support namespace lookup by slug")

        storage = engine._storage
        ns = await storage.get_namespace_by_slug(namespace)  # type: ignore[unresolved-attribute]
        if ns:
            return ns.id

        raise ValueError(f"Namespace not found: {namespace}")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        if not self._connected or self._engine is None:
            return {"status": "disconnected"}

        return await self._engine.health_check()


# Convenience function for one-off usage
@asynccontextmanager
async def memory_lake(
    config: KhoraConfig | None = None,
) -> AsyncGenerator[MemoryLake]:
    """Context manager for one-off Memory Lake usage.

    Usage:
        async with memory_lake() as lake:
            await lake.remember("Hello, world!")
            result = await lake.recall("greeting")
    """
    lake = MemoryLake(config)
    try:
        await lake.connect()
        yield lake
    finally:
        await lake.disconnect()
