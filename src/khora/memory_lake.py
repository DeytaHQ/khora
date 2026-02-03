"""MemoryLake - Primary API for Khora Memory Lake.

This is the main entry point for using Khora as a library.
Provides a simple, unified interface for memory storage and retrieval.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig, load_config
from khora.core.models import Document, DocumentMetadata, Entity, MemoryNamespace, Organization, Workspace
from khora.extraction.embedders import LiteLLMEmbedder
from khora.query import HybridQueryEngine, QueryConfig, SearchMode
from khora.storage import StorageConfig, StorageCoordinator, create_storage_coordinator

if TYPE_CHECKING:
    pass


@dataclass
class RememberResult:
    """Result of a remember operation."""

    document_id: UUID
    namespace_id: UUID
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of remember_batch() operation."""

    total: int
    processed: int
    skipped: int
    failed: int
    chunks: int
    entities: int
    relationships: int


@dataclass
class Stats:
    """Namespace statistics."""

    documents: int
    chunks: int
    entities: int
    relationships: int


@dataclass
class RecallResult:
    """Result of a recall operation."""

    query: str
    namespace_id: UUID
    chunks: list[tuple[Any, float]]
    entities: list[tuple[Any, float]]
    context_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryLake:
    """Primary interface for Khora Memory Lake.

    Provides a simple API for storing and retrieving memories:
    - remember(): Store content in the memory lake
    - recall(): Retrieve relevant memories for a query
    - forget(): Remove memories

    Can be used as a context manager for automatic connection handling.

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

        # Full config
        async with MemoryLake(KhoraConfig(...)) as lake:
            ...
    """

    def __init__(
        self,
        database_url: str | KhoraConfig | None = None,
        *,
        graph_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        storage_config: StorageConfig | None = None,
    ) -> None:
        """Initialize the Memory Lake.

        Args:
            database_url: PostgreSQL URL, or full KhoraConfig, or None (reads KHORA_DATABASE_URL from env)
            graph_url: Optional Neo4j/graph database URL (bolt://user:pass@host:port)
            embedding_model: Embedding model to use (default: text-embedding-3-small)
            storage_config: Storage configuration (derived from config if None) - deprecated

        Examples:
            # Simplest - from env vars
            lake = MemoryLake()

            # Common - explicit database
            lake = MemoryLake("postgresql://localhost/mydb")

            # With graph
            lake = MemoryLake("postgresql://...", graph_url="bolt://...")

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

        # Set up storage config
        if storage_config:
            self._storage_config = storage_config
        else:
            postgresql_url = self._config.get_postgresql_url()
            graph_config = self._config.get_graph_config()
            vector_config = self._config.get_vector_config()
            self._storage_config = StorageConfig(
                postgresql_url=postgresql_url,
                pgvector_url=postgresql_url,  # pgvector uses same database as relational
                neo4j_url=self._config.get_neo4j_url(),
                neo4j_user=self._config.get_neo4j_user(),
                neo4j_password=self._config.get_neo4j_password(),
                neo4j_database=self._config.get_neo4j_database(),
                pgvector_embedding_dimension=self._config.storage.embedding_dimension,
                graph_config=graph_config,
                vector_config=vector_config,
            )

        self._storage: StorageCoordinator | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._query_engine: HybridQueryEngine | None = None
        self._connected = False

        # Default namespace for simple usage
        self._default_namespace_id: UUID | None = None

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting Memory Lake...")

        # Create and connect storage
        self._storage = create_storage_coordinator(self._storage_config)
        await self._storage.connect()

        # Create embedder
        llm_config = LiteLLMConfig(
            model=self._config.llm.model,
            embedding_model=self._config.llm.embedding_model,
            embedding_dimension=self._config.llm.embedding_dimension,
            timeout=self._config.llm.timeout,
            max_retries=self._config.llm.max_retries,
        )
        self._embedder = LiteLLMEmbedder.from_config(llm_config)

        # Create query engine
        self._query_engine = HybridQueryEngine(
            storage=self._storage,
            embedder=self._embedder,
        )

        # Initialize telemetry (no-op if KHORA_TELEMETRY_DATABASE_URL not set)
        from khora.telemetry import init_telemetry
        from khora.telemetry.config import TelemetryConfig

        telemetry_cfg = TelemetryConfig(
            database_url=self._config.telemetry_database_url,
            service_name=self._config.telemetry_service_name,
        )
        await init_telemetry(telemetry_cfg)

        self._connected = True
        logger.info("Memory Lake connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting Memory Lake...")

        # Shutdown telemetry
        from khora.telemetry import shutdown_telemetry

        await shutdown_telemetry()

        if self._storage:
            await self._storage.disconnect()
            self._storage = None

        self._embedder = None
        self._query_engine = None
        self._connected = False

        logger.info("Memory Lake disconnected")

    async def __aenter__(self) -> MemoryLake:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    @property
    def storage(self) -> StorageCoordinator:
        """Get the storage coordinator.

        .. deprecated::
            Direct access to storage is deprecated. Use MemoryLake methods instead:
            - lake.get_document() instead of lake.storage.get_document()
            - lake.list_documents() instead of lake.storage.list_documents()
            - lake.search_entities() instead of lake.storage.search_entities()
            - lake.stats() instead of querying storage directly
        """
        warnings.warn(
            "lake.storage is deprecated. Use lake.get_document(), lake.list_documents(), "
            "lake.search_entities(), lake.stats() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._storage is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")
        return self._storage

    @property
    def query_engine(self) -> HybridQueryEngine:
        """Get the query engine.

        .. deprecated::
            Direct access to query_engine is deprecated. Use lake.recall() instead.
            For unprocessed search without LLM features, use lake.recall(query, raw=True).
        """
        warnings.warn(
            "lake.query_engine is deprecated. Use lake.recall() instead. "
            "For raw search without LLM features, use lake.recall(query, raw=True).",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._query_engine is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")
        return self._query_engine

    def _get_storage(self) -> StorageCoordinator:
        """Get storage coordinator (internal use, no deprecation warning)."""
        if self._storage is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")
        return self._storage

    def _get_query_engine(self) -> HybridQueryEngine:
        """Get query engine (internal use, no deprecation warning)."""
        if self._query_engine is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")
        return self._query_engine

    # =========================================================================
    # Namespace Management
    # =========================================================================

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
        namespace = MemoryNamespace(
            workspace_id=workspace_id,
            name=name,
            description=description,
            config_overrides=config_overrides or {},
        )
        return await self._get_storage().create_namespace(namespace)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        return await self._get_storage().get_namespace(namespace_id)

    async def get_or_create_default_namespace(self) -> UUID:
        """Get or create a default namespace for simple usage."""
        if self._default_namespace_id:
            return self._default_namespace_id

        # Try to find existing default namespace
        # For simplicity, we'll create a default org/workspace/namespace
        storage = self._get_storage()
        default_org = await storage.get_organization_by_slug("default")
        if not default_org:
            default_org = await storage.create_organization(Organization(name="Default", slug="default"))

        workspaces = await storage.list_workspaces(default_org.id)
        if workspaces:
            default_workspace = workspaces[0]
        else:
            default_workspace = await storage.create_workspace(
                Workspace(
                    organization_id=default_org.id,
                    name="Default",
                    slug="default",
                )
            )

        namespaces = await storage.list_namespaces(default_workspace.id)
        if namespaces:
            default_namespace = namespaces[0]
        else:
            default_namespace = await storage.create_namespace(
                MemoryNamespace(
                    workspace_id=default_workspace.id,
                    name="Default",
                    slug="default",
                )
            )

        self._default_namespace_id = default_namespace.id
        return self._default_namespace_id

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

        Returns:
            RememberResult with details
        """
        from khora.telemetry.context import clear_trace_id, ensure_trace_id

        ensure_trace_id()
        try:
            return await self._remember_inner(
                content,
                namespace=namespace,
                title=title,
                source=source,
                metadata=metadata,
                skill_name=skill_name,
            )
        finally:
            clear_trace_id()

    async def _remember_inner(
        self,
        content: str,
        *,
        namespace: str | UUID | None = None,
        title: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
    ) -> RememberResult:
        """Internal remember implementation with trace context already set."""
        # Resolve namespace
        namespace_id = await self._resolve_namespace(namespace)

        # Compute checksum
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

        storage = self._get_storage()

        # Check for duplicate - skip if any document with same checksum exists
        existing = await storage.get_document_by_checksum(namespace_id, checksum)
        if existing:
            logger.debug(f"Document already exists (checksum={checksum[:8]}..., status={existing.status})")
            return RememberResult(
                document_id=existing.id,
                namespace_id=namespace_id,
                chunks_created=existing.chunk_count,
                entities_extracted=existing.entity_count,
                relationships_created=0,
                metadata={"duplicate": True, "status": str(existing.status)},
            )

        # Create document
        doc_metadata = DocumentMetadata(
            title=title,
            source=source,
            source_type="api",
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            custom=metadata or {},
        )
        document = Document(
            namespace_id=namespace_id,
            content=content,
            metadata=doc_metadata,
        )
        document = await storage.create_document(document)

        # Process through pipeline
        from khora.pipelines.flows.ingest import process_document

        result = await process_document(
            document,
            storage,
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
        )

        return RememberResult(
            document_id=document.id,
            namespace_id=namespace_id,
            chunks_created=result["chunks"],
            entities_extracted=result["entities"],
            relationships_created=result["relationships"],
        )

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str | UUID | None = None,
        skill_name: str = "general_entities",
        max_concurrent: int = 5,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
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

        Returns:
            BatchResult with aggregated statistics
        """
        from khora.telemetry.context import clear_trace_id, ensure_trace_id

        ensure_trace_id()
        try:
            return await self._remember_batch_inner(
                documents,
                namespace=namespace,
                skill_name=skill_name,
                max_concurrent=max_concurrent,
                deduplicate=deduplicate,
                infer_relationships=infer_relationships,
                on_progress=on_progress,
            )
        finally:
            clear_trace_id()

    async def _remember_batch_inner(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str | UUID | None = None,
        skill_name: str = "general_entities",
        max_concurrent: int = 5,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Internal remember_batch implementation using ingest_documents for shared EntityIndex."""
        if not documents:
            return BatchResult(
                total=0,
                processed=0,
                skipped=0,
                failed=0,
                chunks=0,
                entities=0,
                relationships=0,
            )

        namespace_id = await self._resolve_namespace(namespace)

        # Build doc dicts for ingest_documents
        doc_inputs = []
        for doc_data in documents:
            doc_inputs.append(
                {
                    "content": doc_data.get("content", ""),
                    "title": doc_data.get("title", ""),
                    "source": doc_data.get("source", ""),
                    "source_type": "api",
                    "metadata": doc_data.get("metadata", {}),
                }
            )

        from khora.pipelines.flows.ingest import ingest_documents

        # Create a shared embedder for efficiency (uses LRU cache internally)
        shared_embedder = LiteLLMEmbedder(model=self._config.llm.embedding_model)

        # Create shared EntityIndex for cross-document deduplication if enabled
        shared_entity_index = None
        if deduplicate:
            from khora.extraction.expansion.entity_index import EntityIndex

            shared_entity_index = EntityIndex()

            # Optionally preload existing entities for dedup against stored data
            existing_entities = await self._get_storage().list_entities(namespace_id, limit=50000)
            for entity in existing_entities:
                shared_entity_index.add(entity)

            if existing_entities:
                logger.debug(f"Preloaded {len(existing_entities)} existing entities into EntityIndex")

        result = await ingest_documents(
            namespace_id,
            doc_inputs,
            self._get_storage(),
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
            max_concurrent_documents=max_concurrent,
            shared_embedder=shared_embedder,
            shared_entity_index=shared_entity_index,
            enable_expansion=infer_relationships,
        )

        # Call progress callback if provided
        if on_progress:
            processed = result.get("processed_documents", 0)
            total = result.get("total_documents", len(documents))
            on_progress(processed, total)

        # Build BatchResult from aggregated stats
        return BatchResult(
            total=result.get("total_documents", len(documents)),
            processed=result.get("processed_documents", 0),
            skipped=result.get("skipped_documents", 0),
            failed=result.get("failed_documents", 0),
            chunks=result.get("total_chunks", 0),
            entities=result.get("total_entities", 0),
            relationships=result.get("total_relationships", 0) + result.get("total_inferred_relationships", 0),
        )

    async def remember_batch_legacy(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str | UUID | None = None,
        skill_name: str = "general_entities",
        max_concurrent: int = 5,
    ) -> list[RememberResult]:
        """Store multiple documents - legacy version returning list of RememberResult.

        .. deprecated::
            Use remember_batch() which returns BatchResult with aggregated stats.
            This method is kept for backwards compatibility.
        """
        warnings.warn(
            "remember_batch_legacy() is deprecated. Use remember_batch() which returns BatchResult.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not documents:
            return []

        namespace_id = await self._resolve_namespace(namespace)

        doc_inputs = []
        for doc_data in documents:
            doc_inputs.append(
                {
                    "content": doc_data.get("content", ""),
                    "title": doc_data.get("title", ""),
                    "source": doc_data.get("source", ""),
                    "source_type": "api",
                    "metadata": doc_data.get("metadata", {}),
                }
            )

        from khora.pipelines.flows.ingest import ingest_documents

        result = await ingest_documents(
            namespace_id,
            doc_inputs,
            self._get_storage(),
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
            max_concurrent_documents=max_concurrent,
        )

        per_doc = result.get("per_document_results", [])
        final_results: list[RememberResult] = []
        for doc_result in per_doc:
            final_results.append(
                RememberResult(
                    document_id=UUID(doc_result["document_id"]),
                    namespace_id=namespace_id,
                    chunks_created=doc_result.get("chunks", 0),
                    entities_extracted=doc_result.get("entities", 0),
                    relationships_created=doc_result.get("relationships", 0),
                )
            )

        failed_count = result.get("failed_documents", 0)
        for _ in range(failed_count):
            final_results.append(
                RememberResult(
                    document_id=UUID("00000000-0000-0000-0000-000000000000"),
                    namespace_id=namespace_id,
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_created=0,
                    metadata={"failed": True},
                )
            )

        return final_results

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
            return await self._recall_inner(
                query,
                namespace=namespace,
                limit=limit,
                mode=mode,
                min_similarity=min_similarity,
                agentic=agentic,
                raw=raw,
            )
        finally:
            clear_trace_id()

    async def _recall_inner(
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
        """Internal recall implementation with trace context already set."""
        namespace_id = await self._resolve_namespace(namespace)

        config = QueryConfig(
            mode=mode,
            max_chunks=limit,
            max_entities=limit,
            min_chunk_similarity=min_similarity,
            min_entity_similarity=min_similarity,
        )

        # Raw mode: disable all LLM features
        if raw:
            config.enable_query_understanding = False
            config.enable_query_expansion = False
            config.enable_entity_extraction = False
            config.enable_temporal_detection = False
            config.enable_entity_linking = False
            config.enable_reranking = False
            config.enable_hyde = False

        result = await self._get_query_engine().query(query, namespace_id, config=config, agentic=agentic)

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            chunks=result.chunks,
            entities=result.entities,
            context_text=result.get_context_text(max_chunks=limit),
            metadata=result.metadata,
        )

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
        storage = self._get_storage()

        # Verify namespace if provided
        if namespace:
            namespace_id = await self._resolve_namespace(namespace)
            document = await storage.get_document(document_id)
            if document and document.namespace_id != namespace_id:
                logger.warning(f"Document {document_id} not in namespace {namespace_id}")
                return False

        return await storage.delete_document(document_id)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        return await self._get_storage().get_entity(entity_id)

    async def list_entities(
        self,
        *,
        namespace: str | UUID | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List entities in a namespace."""
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_storage().list_entities(namespace_id, entity_type=entity_type, limit=limit)

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
        return await self._get_query_engine().find_related_entities(
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
        return await self._get_storage().get_document(document_id)

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
        return await self._get_storage().list_documents(namespace_id, limit=limit)

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

        # Embed the query
        if self._embedder is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")

        query_embedding = await self._embedder.embed(query)

        # Search similar entities
        entity_ids_scores = await self._get_storage().search_similar_entities(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=0.0,
        )

        # Fetch full entities
        storage = self._get_storage()
        entities = []
        for entity_id, _score in entity_ids_scores:
            entity = await storage.get_entity(entity_id)
            if entity:
                entities.append(entity)

        return entities

    async def stats(self, *, namespace: str | UUID | None = None) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace.

        Args:
            namespace: Namespace name, ID, or None for default

        Returns:
            Stats with document/chunk/entity/relationship counts
        """
        namespace_id = await self._resolve_namespace(namespace)
        storage = self._get_storage()

        # Get counts
        documents = await storage.list_documents(namespace_id, limit=0)
        doc_count = len(documents) if documents else 0

        # For more accurate counts, query directly
        # These may vary by backend implementation
        try:
            doc_count = await storage.count_documents(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        try:
            chunk_count = await storage.count_chunks(namespace_id)
        except (AttributeError, NotImplementedError):
            chunks = await storage.list_chunks(namespace_id, limit=0)
            chunk_count = len(chunks) if chunks else 0

        try:
            entity_count = await storage.count_entities(namespace_id)
        except (AttributeError, NotImplementedError):
            entities = await storage.list_entities(namespace_id, limit=0)
            entity_count = len(entities) if entities else 0

        try:
            relationship_count = await storage.count_relationships(namespace_id)
        except (AttributeError, NotImplementedError):
            rels = await storage.list_relationships(namespace_id, limit=0)
            relationship_count = len(rels) if rels else 0

        return Stats(
            documents=doc_count,
            chunks=chunk_count,
            entities=entity_count,
            relationships=relationship_count,
        )

    async def ensure_namespace(
        self,
        name: str,
        *,
        description: str = "",
    ) -> UUID:
        """Get or create a namespace by name.

        Creates the default organization and workspace if they don't exist.
        This is a convenience method for simple usage where you just want
        a namespace by name without managing the full hierarchy.

        Args:
            name: Namespace name (will be slugified)
            description: Optional description

        Returns:
            Namespace UUID
        """
        storage = self._get_storage()

        # Ensure default org/workspace exists
        await self.get_or_create_default_namespace()

        # Get default workspace
        default_org = await storage.get_organization_by_slug("default")
        if not default_org:
            raise RuntimeError("Default organization not found")

        workspaces = await storage.list_workspaces(default_org.id)
        if not workspaces:
            raise RuntimeError("Default workspace not found")

        default_workspace = workspaces[0]

        # Try to find namespace by slug
        slug = name.lower().replace(" ", "-")
        existing_ns = await storage.get_namespace_by_slug(default_workspace.id, slug)
        if existing_ns:
            return existing_ns.id

        # Create new namespace
        new_ns = await storage.create_namespace(
            MemoryNamespace(
                workspace_id=default_workspace.id,
                name=name,
                slug=slug,
                description=description,
            )
        )
        return new_ns.id

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

        # Look up by slug in default workspace
        storage = self._get_storage()
        default_ns_id = await self.get_or_create_default_namespace()
        default_ns = await storage.get_namespace(default_ns_id)
        if default_ns:
            ns = await storage.get_namespace_by_slug(default_ns.workspace_id, namespace)
            if ns:
                return ns.id

        raise ValueError(f"Namespace not found: {namespace}")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        if not self._connected:
            return {"status": "disconnected"}

        storage_health = await self._get_storage().health_check()

        return {
            "status": "healthy" if storage_health.is_healthy else "degraded",
            "storage": storage_health.summary,
        }


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
    lake = MemoryLake(config=config)
    try:
        await lake.connect()
        yield lake
    finally:
        await lake.disconnect()
