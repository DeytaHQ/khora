"""GraphRAG engine implementation.

This is the default memory engine for Khora, providing:
- Knowledge graph storage (Neo4j, Kuzu, Memgraph, ArcadeDB)
- Vector embeddings (pgvector, ArcadeDB)
- LLM-based entity extraction
- Hybrid search (vector + graph + keyword)
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.models import Document, DocumentMetadata, Entity, MemoryNamespace, Organization, Workspace
from khora.extraction.embedders import LiteLLMEmbedder
from khora.memory_lake import BatchResult, RecallResult, RememberResult, Stats
from khora.query import HybridQueryEngine, QueryConfig, SearchMode
from khora.storage import StorageConfig, StorageCoordinator, create_storage_coordinator

if TYPE_CHECKING:
    pass


class GraphRAGEngine:
    """GraphRAG engine - full-featured engine using knowledge graphs, vectors, and LLM extraction.

    This is the default engine for MemoryLake. It provides:
    - Document chunking (fixed, semantic, recursive, conversation-aware)
    - Entity and relationship extraction using LLMs
    - Vector similarity search via pgvector
    - Graph traversal via configurable graph backends
    - Hybrid search with RRF fusion
    - Query understanding and entity linking
    - Neural reranking
    """

    def __init__(
        self,
        config: KhoraConfig,
        *,
        storage_config: StorageConfig | None = None,
    ) -> None:
        """Initialize the GraphRAG engine.

        Args:
            config: KhoraConfig instance
            storage_config: Storage configuration (derived from config if None) - deprecated
        """
        self._config = config

        # Set up storage config
        if storage_config:
            self._storage_config = storage_config
        else:
            postgresql_url = config.get_postgresql_url()
            graph_config = config.get_graph_config()
            vector_config = config.get_vector_config()
            self._storage_config = StorageConfig(
                postgresql_url=postgresql_url,
                pgvector_url=postgresql_url,  # pgvector uses same database as relational
                neo4j_url=config.get_neo4j_url(),
                neo4j_user=config.get_neo4j_user(),
                neo4j_password=config.get_neo4j_password(),
                neo4j_database=config.get_neo4j_database(),
                pgvector_embedding_dimension=config.storage.embedding_dimension,
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

        logger.info("Connecting GraphRAG engine...")

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
        logger.info("GraphRAG engine connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting GraphRAG engine...")

        # Shutdown telemetry
        from khora.telemetry import shutdown_telemetry

        await shutdown_telemetry()

        if self._storage:
            await self._storage.disconnect()
            self._storage = None

        self._embedder = None
        self._query_engine = None
        self._connected = False

        logger.info("GraphRAG engine disconnected")

    def _get_storage(self) -> StorageCoordinator:
        """Get storage coordinator (internal use)."""
        if self._storage is None:
            raise RuntimeError("GraphRAG engine not connected. Call connect() first.")
        return self._storage

    def _get_query_engine(self) -> HybridQueryEngine:
        """Get query engine (internal use)."""
        if self._query_engine is None:
            raise RuntimeError("GraphRAG engine not connected. Call connect() first.")
        return self._query_engine

    # =========================================================================
    # Core API: remember, recall, forget
    # =========================================================================

    async def remember(
        self,
        content: str,
        namespace_id: UUID,
        *,
        title: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
    ) -> RememberResult:
        """Store content in the memory engine."""
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

    async def recall(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        agentic: bool = False,
        raw: bool = False,
    ) -> RecallResult:
        """Recall memories relevant to a query."""
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

    async def forget(self, document_id: UUID, namespace_id: UUID | None) -> bool:
        """Remove a memory from the engine."""
        storage = self._get_storage()

        # Verify namespace if provided
        if namespace_id:
            document = await storage.get_document(document_id)
            if document and document.namespace_id != namespace_id:
                logger.warning(f"Document {document_id} not in namespace {namespace_id}")
                return False

        return await storage.delete_document(document_id)

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        max_concurrent: int = 5,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Store multiple documents with automatic optimization."""
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

    # =========================================================================
    # Namespace Management
    # =========================================================================

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

    async def create_namespace(
        self,
        name: str,
        workspace_id: UUID,
        *,
        description: str = "",
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace."""
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

    async def ensure_namespace(
        self,
        name: str,
        *,
        description: str = "",
    ) -> UUID:
        """Get or create a namespace by name."""
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
    # Entity Operations
    # =========================================================================

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        return await self._get_storage().get_entity(entity_id)

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List entities in a namespace."""
        return await self._get_storage().list_entities(namespace_id, entity_type=entity_type, limit=limit)

    async def find_related_entities(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity."""
        return await self._get_query_engine().find_related_entities(
            entity_id,
            namespace_id,
            max_depth=max_depth,
            limit=limit,
        )

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""
        return await self._get_storage().get_document(document_id)

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
    ) -> list[Document]:
        """List documents in a namespace."""
        return await self._get_storage().list_documents(namespace_id, limit=limit)

    async def search_entities(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity."""
        # Embed the query
        if self._embedder is None:
            raise RuntimeError("GraphRAG engine not connected. Call connect() first.")

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

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace."""
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

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        if not self._connected:
            return {"status": "disconnected"}

        storage_health = await self._get_storage().health_check()

        return {
            "status": "healthy" if storage_health.is_healthy else "degraded",
            "storage": storage_health.summary,
        }
