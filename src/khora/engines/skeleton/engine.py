"""Skeleton Construction engine - temporal-first memory engine.

This engine is optimized for:
- Temporal queries with structured field filtering
- Fast and cost-efficient ingestion
- High-precision retrieval with bi-temporal model
- Multiple backends (pgvector, Weaviate)
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.models import Document, DocumentMetadata, Entity, MemoryNamespace, Organization, Workspace
from khora.extraction.embedders import LiteLLMEmbedder
from khora.memory_lake import BatchResult, RecallResult, RememberResult, Stats
from khora.query import SearchMode
from khora.storage import StorageConfig, StorageCoordinator, create_storage_coordinator

from .backends import TemporalChunk, TemporalFilter, TemporalVectorStore, create_temporal_store

if TYPE_CHECKING:
    pass


class SkeletonConstructionEngine:
    """Skeleton Construction engine - temporal-first, cost-efficient memory engine.

    Key features:
    - Bi-temporal model: Track occurrence time vs ingestion time
    - Hierarchical time graph: Year → Quarter → Month → Week → Day
    - Structured field filtering: Filter on occurred_at, not just created_at
    - Multiple backends: PostgreSQL+pgvector (default) and Weaviate (advanced)
    - Cost optimization: Skeleton-based indexing with lazy expansion

    Usage:
        # Default backend (pgvector)
        engine = SkeletonConstructionEngine(config)
        await engine.connect()

        # Weaviate backend
        engine = SkeletonConstructionEngine(config, backend="weaviate", weaviate_url="http://localhost:8080")
        await engine.connect()
    """

    def __init__(
        self,
        config: KhoraConfig,
        *,
        storage_config: StorageConfig | None = None,
        backend: str = "pgvector",
        weaviate_url: str | None = None,
    ) -> None:
        """Initialize the Skeleton Construction engine.

        Args:
            config: KhoraConfig instance
            storage_config: Storage configuration (deprecated, derived from config)
            backend: Backend type ("pgvector" or "weaviate")
            weaviate_url: Weaviate URL (required for weaviate backend)
        """
        self._config = config
        self._backend_type = backend
        self._weaviate_url = weaviate_url

        # Build storage config
        if storage_config:
            self._storage_config = storage_config
        else:
            postgresql_url = config.get_postgresql_url()
            graph_config = config.get_graph_config()

            # Only use legacy neo4j_url fields when graph_config is explicitly set
            # This prevents environment variables (KHORA_NEO4J_URL) from overriding
            # the explicit graph_config=None setting
            storage_kwargs: dict[str, Any] = {
                "postgresql_url": postgresql_url,
                "pgvector_url": postgresql_url,
                "pgvector_embedding_dimension": config.storage.embedding_dimension,
                "graph_config": graph_config,
                "vector_config": config.get_vector_config(),
            }
            if graph_config is not None:
                # Only pass legacy neo4j fields when graph is configured
                storage_kwargs["neo4j_url"] = config.get_neo4j_url()
                storage_kwargs["neo4j_user"] = config.get_neo4j_user()
                storage_kwargs["neo4j_password"] = config.get_neo4j_password()
                storage_kwargs["neo4j_database"] = config.get_neo4j_database()

            self._storage_config = StorageConfig(**storage_kwargs)

        self._storage: StorageCoordinator | None = None
        self._temporal_store: TemporalVectorStore | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._connected = False

        # Default namespace for simple usage
        self._default_namespace_id: UUID | None = None

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info(f"Connecting Skeleton Construction engine (backend={self._backend_type})...")

        # Create and connect relational storage (for documents, namespaces, etc.)
        self._storage = create_storage_coordinator(self._storage_config)
        await self._storage.connect()

        # Create and connect temporal vector store
        self._temporal_store = create_temporal_store(
            self._backend_type,
            self._config,
            weaviate_url=self._weaviate_url,
        )
        await self._temporal_store.connect()

        # Create embedder
        llm_config = LiteLLMConfig(
            model=self._config.llm.model,
            embedding_model=self._config.llm.embedding_model,
            embedding_dimension=self._config.llm.embedding_dimension,
            timeout=self._config.llm.timeout,
            max_retries=self._config.llm.max_retries,
        )
        self._embedder = LiteLLMEmbedder.from_config(llm_config)

        # Initialize telemetry
        from khora.telemetry import init_telemetry
        from khora.telemetry.config import TelemetryConfig

        telemetry_cfg = TelemetryConfig(
            database_url=self._config.telemetry_database_url,
            service_name=self._config.telemetry_service_name,
        )
        await init_telemetry(telemetry_cfg)

        self._connected = True
        logger.info("Skeleton Construction engine connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting Skeleton Construction engine...")

        # Shutdown telemetry
        from khora.telemetry import shutdown_telemetry

        await shutdown_telemetry()

        if self._temporal_store:
            await self._temporal_store.disconnect()
            self._temporal_store = None

        if self._storage:
            await self._storage.disconnect()
            self._storage = None

        self._embedder = None
        self._connected = False

        logger.info("Skeleton Construction engine disconnected")

    def _get_storage(self) -> StorageCoordinator:
        """Get storage coordinator (internal use)."""
        if self._storage is None:
            raise RuntimeError("Skeleton Construction engine not connected. Call connect() first.")
        return self._storage

    def _get_temporal_store(self) -> TemporalVectorStore:
        """Get temporal store (internal use)."""
        if self._temporal_store is None:
            raise RuntimeError("Skeleton Construction engine not connected. Call connect() first.")
        return self._temporal_store

    def _get_embedder(self) -> LiteLLMEmbedder:
        """Get embedder (internal use)."""
        if self._embedder is None:
            raise RuntimeError("Skeleton Construction engine not connected. Call connect() first.")
        return self._embedder

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
        occurred_at: datetime | None = None,
    ) -> RememberResult:
        """Store content in the memory engine.

        Args:
            content: Content to store
            namespace_id: Namespace to store in
            title: Document title
            source: Document source
            metadata: Additional metadata
            skill_name: Extraction skill (default: general_entities)
            occurred_at: When this content/event occurred (default: now)

        Returns:
            RememberResult with document_id and counts
        """
        # Compute checksum
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

        storage = self._get_storage()

        # Check for duplicate
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

        # Create document in relational storage
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

        # Process through simplified pipeline (no full KG extraction)
        chunks_created, entities_extracted, relationships_created = await self._process_document(
            document,
            skill_name=skill_name,
            occurred_at=occurred_at or datetime.now(UTC),
        )

        return RememberResult(
            document_id=document.id,
            namespace_id=namespace_id,
            chunks_created=chunks_created,
            entities_extracted=entities_extracted,
            relationships_created=relationships_created,
        )

    async def _process_document(
        self,
        document: Document,
        *,
        skill_name: str,
        occurred_at: datetime,
    ) -> tuple[int, int, int]:
        """Process a document into chunks (simplified pipeline).

        Unlike GraphRAG, this focuses on fast chunking and embedding without
        full entity extraction. Entity extraction can be done lazily on retrieval.
        """
        from khora.pipelines.chunking import create_chunker  # type: ignore[unresolved-import]
        from khora.pipelines.chunking.config import ChunkerConfig  # type: ignore[unresolved-import]

        storage = self._get_storage()
        embedder = self._get_embedder()
        temporal_store = self._get_temporal_store()

        # Create chunker
        chunker_config = ChunkerConfig(
            strategy=self._config.pipeline.chunking_strategy,
            chunk_size=self._config.pipeline.chunk_size,
            chunk_overlap=self._config.pipeline.chunk_overlap,
        )
        chunker = create_chunker(chunker_config)

        # Chunk the document (run in thread to avoid blocking event loop during
        # CPU-bound tiktoken operations)
        raw_chunks = await asyncio.to_thread(chunker.chunk, document.content)

        if not raw_chunks:
            # Mark document as processed with 0 chunks
            document.mark_completed(0, 0)
            await storage.update_document(document)
            return 0, 0, 0

        # Embed chunks in batch
        chunk_texts = [c.content for c in raw_chunks]
        embeddings = await embedder.embed_batch(chunk_texts)

        # Extract metadata for filtering (source_system, author, channel, etc.)
        doc_metadata = document.metadata.custom if document.metadata else {}

        # Create temporal chunks
        temporal_chunks = []
        for i, (raw_chunk, embedding) in enumerate(zip(raw_chunks, embeddings)):
            temporal_chunk = TemporalChunk(
                id=None,  # Will be assigned
                namespace_id=document.namespace_id,
                document_id=document.id,
                content=raw_chunk.content,
                embedding=embedding,
                occurred_at=occurred_at,
                created_at=datetime.now(UTC),
                source_system=doc_metadata.get("source_system"),
                author=doc_metadata.get("author"),
                channel=doc_metadata.get("channel"),
                tags=doc_metadata.get("tags", []),
                confidence=1.0,
                metadata={
                    "chunk_index": i,
                    "start_char": raw_chunk.start_char if hasattr(raw_chunk, "start_char") else 0,
                    "end_char": raw_chunk.end_char if hasattr(raw_chunk, "end_char") else len(raw_chunk.content),
                },
            )
            temporal_chunks.append(temporal_chunk)

        # Store in temporal store
        stored_chunks = await temporal_store.create_chunks_batch(temporal_chunks)

        # Update document status
        document.mark_completed(len(stored_chunks), 0)
        await storage.update_document(document)

        logger.debug(f"Processed document {document.id}: {len(stored_chunks)} chunks")

        return len(stored_chunks), 0, 0

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
        # Khora-specific parameters
        temporal_filter: TemporalFilter | None = None,
        temporal_reference: datetime | None = None,
        hybrid_alpha: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RecallResult:
        """Recall memories relevant to a query.

        Args:
            query: Query text
            namespace_id: Namespace to search
            limit: Maximum number of results
            mode: Search mode (VECTOR, KEYWORD, HYBRID)
            min_similarity: Minimum similarity threshold
            agentic: Whether to use agentic mode
            raw: Disable all LLM features
            temporal_filter: Structured temporal filter
            temporal_reference: Reference point for relative time (e.g., message timestamp)
            hybrid_alpha: Blend factor for hybrid search (0=BM25, 1=vector)
            filters: Additional structured filters (converted to TemporalFilter)

        Returns:
            RecallResult with chunks and context
        """
        embedder = self._get_embedder()
        temporal_store = self._get_temporal_store()

        # Embed the query
        query_embedding = await embedder.embed(query)

        # Build temporal filter from filters dict if provided
        if filters and not temporal_filter:
            temporal_filter = self._build_temporal_filter_from_dict(filters)

        # Handle relative time references
        if temporal_reference and temporal_filter:
            temporal_filter = self._adjust_relative_time(temporal_filter, temporal_reference)

        # Determine hybrid alpha based on mode
        if hybrid_alpha is None:
            if mode == SearchMode.VECTOR:
                hybrid_alpha = 1.0  # Pure vector
            elif mode == SearchMode.KEYWORD:  # type: ignore[unresolved-attribute]
                hybrid_alpha = 0.0  # Pure BM25
            else:  # HYBRID
                hybrid_alpha = 0.7  # Default blend

        # Perform search
        results = await temporal_store.search(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
            temporal_filter=temporal_filter,
            hybrid_alpha=hybrid_alpha,
            query_text=query,
        )

        # Build context text
        context_parts = []
        chunks_with_scores = []
        for result in results:
            context_parts.append(result.chunk.content)
            # Convert TemporalChunk to a simple dict for RecallResult
            chunk_dict = {
                "id": result.chunk.id,
                "content": result.chunk.content,
                "document_id": result.chunk.document_id,
                "occurred_at": result.chunk.occurred_at.isoformat() if result.chunk.occurred_at else None,
                "metadata": result.chunk.metadata,
            }
            chunks_with_scores.append((chunk_dict, result.combined_score or result.similarity))

        context_text = "\n\n---\n\n".join(context_parts[:limit])

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            chunks=chunks_with_scores,
            entities=[],  # Skeleton engine focuses on chunks, not entities
            context_text=context_text,
            metadata={
                "backend": self._backend_type,
                "hybrid_alpha": hybrid_alpha,
                "temporal_filter": str(temporal_filter) if temporal_filter else None,
            },
        )

    def _build_temporal_filter_from_dict(self, filters: dict[str, Any]) -> TemporalFilter:
        """Convert a filters dict to a TemporalFilter.

        Example:
            filters = {
                "occurred_at": {"gte": "2024-01-01", "lt": "2024-02-01"},
                "author": {"eq": "alice"},
                "source_system": {"eq": "slack"},
            }
        """

        tf = TemporalFilter()

        for key, value in filters.items():
            if not isinstance(value, dict):
                value = {"eq": value}

            if key == "occurred_at":
                if "gte" in value:
                    tf.occurred_after = self._parse_datetime(value["gte"])
                if "gt" in value:
                    tf.occurred_after = self._parse_datetime(value["gt"])
                if "lt" in value:
                    tf.occurred_before = self._parse_datetime(value["lt"])
                if "lte" in value:
                    tf.occurred_before = self._parse_datetime(value["lte"])
            elif key == "created_at":
                if "gte" in value:
                    tf.created_after = self._parse_datetime(value["gte"])
                if "gt" in value:
                    tf.created_after = self._parse_datetime(value["gt"])
                if "lt" in value:
                    tf.created_before = self._parse_datetime(value["lt"])
                if "lte" in value:
                    tf.created_before = self._parse_datetime(value["lte"])
            elif key == "source_system":
                tf.source_system = value.get(
                    "eq", value.get("in", [None])[0] if isinstance(value.get("in"), list) else None
                )
            elif key == "author":
                tf.author = value.get("eq")
            elif key == "channel":
                tf.channel = value.get("eq")
            elif key == "tags":
                if "contains" in value:
                    tf.tags = value["contains"]
                elif "eq" in value:
                    tf.tags = [value["eq"]] if isinstance(value["eq"], str) else value["eq"]
            else:
                tf.additional[key] = value

        return tf

    def _parse_datetime(self, value: Any) -> datetime:
        """Parse a datetime value from various formats."""
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value
        if isinstance(value, str):
            # Date only (try this first to avoid fromisoformat without tz)
            try:
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError:
                pass
            # ISO format with timezone
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except ValueError:
                pass
        raise ValueError(f"Cannot parse datetime: {value}")

    def _adjust_relative_time(
        self,
        temporal_filter: TemporalFilter,
        reference: datetime,
    ) -> TemporalFilter:
        """Adjust temporal filter for relative time references.

        This enables queries like "yesterday" to be relative to the message timestamp,
        not the current time.
        """
        # If filter already has absolute times, don't adjust
        # This is a placeholder for more sophisticated relative time handling
        return temporal_filter

    async def forget(self, document_id: UUID, namespace_id: UUID | None) -> bool:
        """Remove a memory from the engine."""
        storage = self._get_storage()
        temporal_store = self._get_temporal_store()

        # Verify namespace if provided
        if namespace_id:
            document = await storage.get_document(document_id)
            if document and document.namespace_id != namespace_id:
                logger.warning(f"Document {document_id} not in namespace {namespace_id}")
                return False

        # Delete from temporal store
        ns_id = namespace_id or (await storage.get_document(document_id)).namespace_id
        await temporal_store.delete_chunks_by_document(document_id, ns_id)

        # Delete from relational storage
        return await storage.delete_document(document_id)

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        max_concurrent: int = 20,
        deduplicate: bool = True,
        infer_relationships: bool = False,  # Not used in Skeleton Construction engine
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Store multiple documents with automatic optimization.

        The Skeleton Construction engine uses a simplified pipeline:
        - Fast chunking without full entity extraction
        - Batch embedding for cost efficiency
        - Parallel processing with semaphore control

        Args:
            documents: List of document dicts with 'content', 'title', 'source', 'metadata'
            namespace_id: Namespace to store documents in
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing (default 10)
            deduplicate: Whether to skip duplicate documents
            infer_relationships: Not used in Skeleton engine
            on_progress: Callback for progress updates (completed, total)

        Returns:
            BatchResult with processing statistics
        """
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

        storage = self._get_storage()
        total = len(documents)

        # Track results with thread-safe counters
        results: dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0, "chunks": 0}
        results_lock = asyncio.Lock()
        progress_count = 0
        progress_lock = asyncio.Lock()

        # Compute checksums for all documents upfront
        doc_checksums: list[str] = []
        for doc_data in documents:
            content = doc_data.get("content", "")
            checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
            doc_checksums.append(checksum)

        # Batch lookup existing documents by checksum (single query instead of N queries)
        existing_docs: dict[str, Any] = {}
        if deduplicate:
            existing_docs = await storage.get_documents_by_checksums(namespace_id, doc_checksums)

        # Track checksums we've started processing (for intra-batch deduplication)
        checksums_in_flight: set[str] = set()
        checksums_lock = asyncio.Lock()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_document(doc_data: dict[str, Any], checksum: str) -> None:
            """Process a single document with semaphore control."""
            nonlocal progress_count

            # Check for duplicate within the batch (before acquiring semaphore)
            async with checksums_lock:
                if checksum in checksums_in_flight:
                    async with results_lock:
                        results["skipped"] += 1
                    return
                checksums_in_flight.add(checksum)

            # Check if already exists in storage (using pre-fetched batch result)
            if deduplicate and checksum in existing_docs:
                async with results_lock:
                    results["skipped"] += 1
                # Update progress
                if on_progress:
                    async with progress_lock:
                        progress_count += 1
                        on_progress(progress_count, total)
                return

            async with semaphore:
                try:
                    # Extract occurred_at from metadata if present
                    doc_metadata = doc_data.get("metadata", {})
                    occurred_at = None
                    if "occurred_at" in doc_metadata:
                        occurred_at = self._parse_datetime(doc_metadata["occurred_at"])

                    content = doc_data.get("content", "")
                    result = await self.remember(
                        content,
                        namespace_id,
                        title=doc_data.get("title", ""),
                        source=doc_data.get("source", ""),
                        metadata=doc_metadata,
                        skill_name=skill_name,
                        occurred_at=occurred_at,
                    )

                    async with results_lock:
                        if result.metadata.get("duplicate"):
                            results["skipped"] += 1
                        else:
                            results["processed"] += 1
                            results["chunks"] += result.chunks_created

                except Exception as e:
                    logger.error(f"Failed to process document: {e}")
                    async with results_lock:
                        results["failed"] += 1

            # Update progress
            if on_progress:
                async with progress_lock:
                    progress_count += 1
                    on_progress(progress_count, total)

        # Process all documents concurrently with semaphore control
        await asyncio.gather(*[process_document(doc, checksum) for doc, checksum in zip(documents, doc_checksums)])

        return BatchResult(
            total=total,
            processed=results["processed"],
            skipped=results["skipped"],
            failed=results["failed"],
            chunks=results["chunks"],
            entities=0,
            relationships=0,
        )

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def get_or_create_default_namespace(self) -> UUID:
        """Get or create a default namespace for simple usage."""
        if self._default_namespace_id:
            return self._default_namespace_id

        storage = self._get_storage()

        # Try to find existing default namespace
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
    # Entity Operations (minimal for Khora engine)
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
        """Find entities related to a given entity.

        Note: Skeleton Construction engine focuses on temporal chunk retrieval.
        For full entity graph traversal, use the GraphRAG engine.
        """
        # Return empty for now - Khora focuses on chunks
        return []

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
        """Search entities by query text.

        Note: Skeleton Construction engine focuses on temporal chunk retrieval.
        For full entity search, use the GraphRAG engine.
        """
        # Return empty for now - Khora focuses on chunks
        return []

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace."""
        storage = self._get_storage()

        # Get counts
        try:
            doc_count = await storage.count_documents(namespace_id)  # type: ignore[unresolved-attribute]
        except (AttributeError, NotImplementedError):
            documents = await storage.list_documents(namespace_id, limit=0)
            doc_count = len(documents) if documents else 0

        try:
            chunk_count = await storage.count_chunks(namespace_id)
        except (AttributeError, NotImplementedError):
            chunk_count = 0  # Skeleton engine chunks are in temporal_store

        try:
            entity_count = await storage.count_entities(namespace_id)
        except (AttributeError, NotImplementedError):
            entity_count = 0

        try:
            relationship_count = await storage.count_relationships(namespace_id)  # type: ignore[unresolved-attribute]
        except (AttributeError, NotImplementedError):
            relationship_count = 0

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
        temporal_health = await self._get_temporal_store().health_check()

        all_healthy = storage_health.is_healthy and temporal_health.get("status") == "healthy"

        return {
            "status": "healthy" if all_healthy else "degraded",
            "storage": storage_health.summary,
            "temporal_store": temporal_health,
            "backend": self._backend_type,
        }


__all__ = ["SkeletonConstructionEngine"]
