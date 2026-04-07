"""MemoryLake - Primary API for Khora Memory Lake.

This is the main entry point for using Khora as a library.
Provides a simple, unified interface for memory storage and retrieval.

The MemoryLake class is a thin facade that delegates to pluggable engines.
The default engine is "graphrag" which uses knowledge graphs, vectors, and LLM extraction.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, load_config
from khora.core.models import Chunk, Document, Entity, MemoryNamespace
from khora.query import SearchMode
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from khora.core.models import Relationship
    from khora.engines.protocol import MemoryEngineProtocol
    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig
    from khora.storage import StorageConfig, StorageCoordinator


# DYT-645: LLMUsage is a public API type consumed by Poros (DYT-650) and Peras (DYT-651).
# Changes to field names or types require coordination with those projects.
@dataclass(slots=True, frozen=True)
class LLMUsage:
    """A single LLM API call's token usage.

    Read-only value object — Khora produces it, consumers read it.
    """

    operation: str
    """Logical operation name (e.g. "entity_extraction", "embedding")."""
    model: str
    """Model identifier (e.g. "gpt-4o", "text-embedding-3-small")."""
    prompt_tokens: int
    completion_tokens: int
    """0 for embeddings."""
    total_tokens: int
    latency_ms: float
    batch_size: int = 1
    """>1 for embedding batches."""


@dataclass(slots=True, frozen=True)
class RememberResult:
    """Result of a remember operation."""

    document_id: UUID
    namespace_id: UUID
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_usage: list[LLMUsage] = field(default_factory=list)


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
    llm_usage: list[LLMUsage] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class Stats:
    """Namespace statistics."""

    documents: int
    chunks: int
    entities: int
    relationships: int


@dataclass(slots=True, frozen=True)
class RecallResult:
    """Result of a recall operation.

    Attributes:
        query: The original query string.
        namespace_id: Namespace the recall was executed against.
        chunks: Scored chunk tuples ``(Chunk, score)``.
        entities: Scored entity tuples ``(Entity, score)``.
        context_text: Pre-formatted text for LLM context.  When relationships
            are present, includes a ``--- Relationships ---`` section.
        metadata: Engine-specific metadata dict.
        relationships: Scored relationship tuples ``(Relationship, score)``.
            Populated only by the VectorCypher engine; empty list for other
            engines.
    """

    query: str
    namespace_id: UUID
    chunks: list[tuple[Chunk, float]]
    entities: list[tuple[Entity, float]]
    context_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    relationships: list[tuple[Relationship, float]] = field(default_factory=list)
    llm_usage: list[LLMUsage] = field(default_factory=list)


class MemoryLake:
    """Primary interface for Khora Memory Lake.

    Provides a simple API for storing and retrieving memories:
    - remember(): Store content in the memory lake
    - recall(): Retrieve relevant memories for a query
    - forget(): Remove memories
    - create_namespace(): Create a new memory namespace
    - get_namespace_by_stable_id(): Get a namespace by its stable ID

    Can be used as a context manager for automatic connection handling.

    The MemoryLake is a facade that delegates to pluggable engines.
    The default engine is "graphrag" which uses knowledge graphs and vector embeddings.

    Usage:
        # Simplest - from env vars (KHORA_DATABASE_URL)
        async with MemoryLake() as lake:
            await lake.remember("Important fact...", namespace=namespace_id,
                entity_types=["PERSON", "CONCEPT"], relationship_types=["RELATES_TO"])

        # Common - explicit database URL
        async with MemoryLake("postgresql://localhost/mydb") as lake:
            results = await lake.recall("What do I know about...", namespace=namespace_id)

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
        run_migrations: bool = False,
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
            run_migrations: If True, run Alembic migrations during connect() (default: False)

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
        self._run_migrations = run_migrations
        self._engine: MemoryEngineProtocol | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting Memory Lake...")

        if self._run_migrations:
            from khora.db.session import run_migrations as _run_migrations

            db_url = self._config.database_url
            result = await _run_migrations(db_url)
            if not result.success:
                raise RuntimeError(f"Database migration failed: {result.error}")

        from khora.engines import create_engine

        self._engine = create_engine(
            self._engine_name,
            self._config,
            storage_config=self._storage_config,
            **self._engine_kwargs,
        )
        await self._engine.connect()

        # Wire hook dispatcher into the storage coordinator so the
        # ingestion pipeline can dispatch events without knowing about MemoryLake.
        storage = getattr(self._engine, "_storage", None)
        if storage is not None:
            try:
                storage._hook_dispatcher = self._get_hook_dispatcher()
            except (AttributeError, TypeError):
                pass  # Mock or non-standard engine — hooks won't fire

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
        *,
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace.

        Args:
            config_overrides: Optional configuration overrides

        Returns:
            Created MemoryNamespace
        """
        return await self._get_engine().create_namespace(
            config_overrides=config_overrides,
        )

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        return await self._get_engine().get_namespace(namespace_id)

    async def get_namespace_by_stable_id(self, namespace_id: str | UUID) -> MemoryNamespace | None:
        """Get a namespace by its stable namespace_id.

        Unlike get_namespace() which takes a row-level id, this accepts
        the stable namespace_id (shared across versions) and resolves it
        to the active version before fetching.

        Args:
            namespace_id: The stable namespace identifier (UUID or string)

        Returns:
            MemoryNamespace, or None if the resolved namespace row is not found

        Raises:
            ValueError: If no active namespace version exists for the given namespace_id
        """
        resolved_id = await self._resolve_namespace(namespace_id)
        return await self._get_engine().get_namespace(resolved_id)

    # =========================================================================
    # Core API: remember, recall, forget
    # =========================================================================

    async def remember(
        self,
        content: str,
        *,
        namespace: str | UUID,
        title: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
    ) -> RememberResult:
        """Store content in the memory lake.

        This is the primary method for adding memories. It:
        1. Creates a document
        2. Chunks the content
        3. Generates embeddings
        4. Extracts entities and relationships

        Args:
            content: Content to remember
            namespace: Namespace UUID (as UUID or string)
            title: Optional title for the content
            source: Optional source identifier
            metadata: Optional metadata
            skill_name: Extraction skill to use
            entity_types: Required entity types to extract
            relationship_types: Required relationship types to extract
            expertise: Optional expertise config for domain-specific extraction
            extraction_config_hash: Optional hash of the extraction config for change detection
            chunk_strategy: Override chunking strategy for this call only.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.

        Returns:
            RememberResult with details
        """
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        try:
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.remember", namespace_id=str(namespace_id), content_length=len(content)):
                # NOTE: expertise and extraction_config_hash are always forwarded,
                # even when None. Custom engines registered via register_engine()
                # must accept these kwargs to remain compatible (ADR-022).
                result = await self._get_engine().remember(
                    content,
                    namespace_id,
                    title=title,
                    source=source,
                    metadata=metadata,
                    skill_name=skill_name,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    expertise=expertise,
                    extraction_config_hash=extraction_config_hash,
                    chunk_strategy=chunk_strategy,
                )
                return replace(result, llm_usage=collect_usage())
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str | UUID,
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
            namespace: Namespace UUID (as UUID or string)
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Deduplicate entities across documents (default: True)
            infer_relationships: Infer relationships after ingestion (default: True)
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
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        try:
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.remember_batch", namespace_id=str(namespace_id), batch_size=len(documents)):
                # NOTE: see remember() comment re: custom engine compatibility
                result = await self._get_engine().remember_batch(
                    documents,
                    namespace_id,
                    skill_name=skill_name,
                    max_concurrent=max_concurrent,
                    deduplicate=deduplicate,
                    infer_relationships=infer_relationships,
                    on_progress=on_progress,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    expertise=expertise,
                    extraction_config_hash=extraction_config_hash,
                    chunk_strategy=chunk_strategy,
                )
                return replace(result, llm_usage=collect_usage())
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()

    async def recall(
        self,
        query: str,
        *,
        namespace: str | UUID,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        agentic: bool = False,
        raw: bool = False,
        include_sources: bool = False,
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
            namespace: Namespace UUID (as UUID or string)
            limit: Maximum results to return
            mode: Search mode (VECTOR, GRAPH, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            agentic: If True, use multi-step agentic search (default: False)
            raw: If True, skip all LLM features (default: False)
            include_sources: If True, populate source document metadata on
                returned chunks and entities (default: False)

        Returns:
            RecallResult with matched memories.  When using the VectorCypher
            engine, ``relationships`` contains scored relationship tuples and
            ``context_text`` includes a ``--- Relationships ---`` section.
        """
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        try:
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.recall", namespace_id=str(namespace_id), query=query):
                result = await self._get_engine().recall(
                    query,
                    namespace_id,
                    limit=limit,
                    mode=mode,
                    min_similarity=min_similarity,
                    agentic=agentic,
                    raw=raw,
                )
                if include_sources:
                    await self._populate_sources(result.chunks, result.entities, result.relationships)
                return replace(result, llm_usage=collect_usage())
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()

    async def forget(
        self,
        document_id: UUID,
        *,
        namespace: str | UUID,
    ) -> bool:
        """Remove a memory from the lake.

        Args:
            document_id: ID of the document to remove
            namespace: Namespace UUID (as UUID or string)

        Returns:
            True if deleted, False if not found
        """
        namespace_id = await self._resolve_namespace(namespace)

        with trace_span(
            "khora.forget",
            namespace_id=str(namespace_id),
            document_id=str(document_id),
        ):
            return await self._get_engine().forget(document_id, namespace_id)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entity(
        self,
        entity_id: UUID,
        *,
        include_sources: bool = False,
    ) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: Entity UUID to retrieve
            include_sources: If True, populate source document metadata on
                the returned entity (default: False)

        Returns:
            Entity if found, else None
        """
        entity = await self._get_engine().get_entity(entity_id)
        if entity is not None and include_sources:
            await self._populate_sources([], [entity], [])
        return entity

    async def list_entities(
        self,
        *,
        namespace: str | UUID,
        entity_type: str | None = None,
        limit: int = 100,
        include_sources: bool = False,
    ) -> list[Entity]:
        """List entities in a namespace.

        Args:
            namespace: Namespace UUID (as UUID or string)
            entity_type: Optional entity type filter
            limit: Maximum entities to return
            include_sources: If True, populate source document metadata on
                returned entities (default: False)

        Returns:
            List of Entity objects
        """
        namespace_id = await self._resolve_namespace(namespace)
        entities = await self._get_engine().list_entities(namespace_id, entity_type=entity_type, limit=limit)
        if include_sources:
            await self._populate_sources([], entities, [])
        return entities

    async def find_related_entities(
        self,
        entity_id: UUID,
        *,
        namespace: str | UUID,
        max_depth: int = 2,
        limit: int = 20,
        include_sources: bool = False,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity.

        Args:
            entity_id: Entity UUID to find related entities for
            namespace: Namespace UUID (as UUID or string)
            max_depth: Maximum graph traversal depth
            limit: Maximum entities to return
            include_sources: If True, populate source document metadata on
                returned entities (default: False)

        Returns:
            List of (Entity, score) tuples
        """
        namespace_id = await self._resolve_namespace(namespace)
        results = await self._get_engine().find_related_entities(
            entity_id,
            namespace_id,
            max_depth=max_depth,
            limit=limit,
        )
        if include_sources:
            await self._populate_sources([], results, [])
        return results

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
        namespace: str | UUID,
        limit: int = 100,
    ) -> list[Document]:
        """List documents in a namespace.

        Args:
            namespace: Namespace UUID (as UUID or string)
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
        namespace: str | UUID,
        limit: int = 10,
        include_sources: bool = False,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity.

        Args:
            query: Search query text
            namespace: Namespace UUID (as UUID or string)
            limit: Maximum entities to return
            include_sources: If True, populate source document metadata on
                returned entities (default: False)

        Returns:
            List of matching Entities (most similar first)
        """
        namespace_id = await self._resolve_namespace(namespace)
        entities = await self._get_engine().search_entities(query, namespace_id, limit=limit)
        if include_sources:
            await self._populate_sources([], entities, [])
        return entities

    async def stats(self, *, namespace: str | UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace.

        Args:
            namespace: Namespace UUID (as UUID or string)

        Returns:
            Stats with document/chunk/entity/relationship counts
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().stats(namespace_id)

    # =========================================================================
    # Helpers
    # =========================================================================

    async def _populate_sources(
        self,
        chunks: list[tuple[Chunk, float]],
        entities: list[tuple[Entity, float]] | list[Entity],
        relationships: list[tuple[Relationship, float]],
    ) -> None:
        """Batch-fetch document sources and populate entity/chunk/relationship fields **in-place**.

        ``entities`` accepts either ``list[Entity]`` or
        ``list[tuple[Entity, float]]`` (entity, score pairs).  The method
        unwraps tuples transparently.

        Collects unique document IDs from *chunks*, *entities*, and
        *relationships*, fetches lightweight metadata via batched SELECTs
        (chunked at 1 000 IDs), then populates ``chunk.source_document``,
        ``entity.source_documents``, and ``relationship.source_documents``
        on the provided objects.  No value is returned; callers observe
        changes through the mutated inputs.
        """
        # Collect unique doc IDs
        doc_ids: set[UUID] = set()
        for chunk, _score in chunks:
            doc_ids.add(chunk.document_id)
        for item in entities:
            entity = item[0] if isinstance(item, tuple) else item
            doc_ids.update(entity.source_document_ids)
        for rel, _score in relationships:
            doc_ids.update(rel.source_document_ids)

        if not doc_ids:
            return

        sorted_ids = sorted(doc_ids)
        sources: dict = {}
        for i in range(0, len(sorted_ids), 1000):
            batch = sorted_ids[i : i + 1000]
            sources.update(await self.storage.get_document_sources_batch(batch))

        # Populate chunks
        for chunk, _score in chunks:
            chunk.source_document = sources.get(chunk.document_id)

        # Populate entities
        for item in entities:
            entity = item[0] if isinstance(item, tuple) else item
            entity_sources = {did: sources[did] for did in entity.source_document_ids if did in sources}
            # None means either "sources not fetched" (include_sources=False) or
            # "all source documents deleted".  Callers distinguish via the
            # include_sources flag they passed.
            entity.source_documents = entity_sources if entity_sources else None

        # Populate relationships
        for rel, _score in relationships:
            rel_sources = {did: sources[did] for did in rel.source_document_ids if did in sources}
            rel.source_documents = rel_sources if rel_sources else None

    # ------------------------------------------------------------------
    # Semantic hooks (subscription API)
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: str,
        callback: Callable[..., Any],
        *,
        filter: Any | None = None,
        namespace_id: UUID | None = None,
    ) -> UUID:
        """Subscribe to extraction events with optional semantic filtering.

        Registers an async callback that fires during document ingestion
        when an event of the given type occurs. Optionally, attach a
        ``SemanticFilter`` to narrow matches by entity type, embedding
        similarity, or LLM evaluation.

        Args:
            event_type: Event type string or ``EventType`` enum
                (e.g., ``"entity.created"``, ``EventType.ENTITY_CREATED``).
            callback: Async function ``async def handler(event: MemoryEvent) -> None``.
            filter: Optional ``SemanticFilter`` for type/embedding/LLM gating.
            namespace_id: Scope to a specific namespace (None = all).

        Returns:
            Subscription UUID for later ``unsubscribe()``.

        Example::

            async def on_entity(event):
                print(f"New entity: {event.data.get('name')}")

            sub_id = lake.subscribe("entity.created", on_entity)
            await lake.remember("Acme Corp announced...", ...)
            lake.unsubscribe(sub_id)
        """
        return self._get_hook_dispatcher().subscribe(
            event_type,
            callback,
            filter=filter,
            namespace_id=namespace_id,
        )

    def unsubscribe(self, subscription_id: UUID) -> bool:
        """Remove a hook subscription.

        Returns True if found and removed, False otherwise.
        """
        return self._get_hook_dispatcher().unsubscribe(subscription_id)

    def _get_hook_dispatcher(self) -> Any:
        """Lazy-initialize the hook dispatcher."""
        if not hasattr(self, "_hook_dispatcher"):
            from khora.hooks.dispatcher import HookDispatcher

            hooks_config = getattr(self._config, "hooks", None)
            max_concurrent = 10
            if hooks_config:
                max_concurrent = getattr(hooks_config, "max_concurrent_callbacks", 10)
            self._hook_dispatcher = HookDispatcher(max_concurrent=max_concurrent)
        return self._hook_dispatcher

    @property
    def hooks(self) -> Any:
        """Access the hook dispatcher directly for advanced usage."""
        return self._get_hook_dispatcher()

    async def _dispatch_hook(self, event: Any) -> None:
        """Dispatch an event to hook subscribers (internal, called by engines)."""
        dispatcher = self._get_hook_dispatcher()
        if dispatcher.subscription_count > 0:
            await dispatcher.dispatch(event)

    async def _resolve_namespace(self, namespace: str | UUID) -> UUID:
        """Resolve a namespace_id to the active version's row-level id.

        Accepts a stable namespace_id (UUID or string) and resolves it to
        the row-level id of the currently active version via DB lookup.
        """
        if isinstance(namespace, str):
            try:
                namespace = UUID(namespace)
            except ValueError:
                raise ValueError(f"Invalid namespace: {namespace!r}. Must be a valid UUID.")

        return await self.storage.resolve_namespace(namespace)

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
