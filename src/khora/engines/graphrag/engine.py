"""GraphRAG engine implementation.

This is the default memory engine for Khora, providing:
- Knowledge graph storage (Neo4j, Memgraph, SurrealDB)
- Vector embeddings (pgvector, SurrealDB)
- LLM-based entity extraction
- Hybrid search (vector + graph + keyword)
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.models import Document, DocumentMetadata, Entity, MemoryNamespace
from khora.engines._storage_config import build_storage_config
from khora.extraction.embedders import LiteLLMEmbedder
from khora.memory_lake import BatchResult, RecallResult, RememberResult, Stats
from khora.query import HybridQueryEngine, QueryConfig, SearchMode
from khora.query.temporal_detection import (
    TemporalCategory,
    TemporalDetector,
    TemporalSignal,
    get_retrieval_params,
)
from khora.storage import StorageConfig, StorageCoordinator, create_storage_coordinator
from khora.telemetry import trace

if TYPE_CHECKING:
    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig


@dataclass
class ExtractionQualityMetrics:
    """Track extraction quality for monitoring.

    This dataclass captures metrics about the quality of entity and
    relationship extraction during document ingestion, enabling
    monitoring and quality control.
    """

    total_chunks: int = 0
    chunks_with_entities: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    avg_entities_per_chunk: float = 0.0
    avg_confidence: float = 0.0
    entity_type_distribution: dict[str, int] = field(default_factory=dict)
    low_confidence_entities: int = 0
    extraction_time_ms: float = 0.0

    def compute_averages(self) -> None:
        """Compute average metrics from totals."""
        if self.total_chunks > 0:
            self.avg_entities_per_chunk = self.total_entities / self.total_chunks

    def log_quality_summary(self, document_id: UUID) -> None:
        """Log a quality summary for monitoring."""
        if self.total_chunks == 0:
            logger.debug(f"Document {document_id}: no chunks processed")
            return

        entity_ratio = self.chunks_with_entities / self.total_chunks if self.total_chunks > 0 else 0
        logger.info(
            f"Document {document_id} extraction quality: "
            f"{self.total_entities} entities from {self.chunks_with_entities}/{self.total_chunks} chunks "
            f"({entity_ratio:.1%} coverage), "
            f"{self.total_relationships} relationships, "
            f"avg confidence: {self.avg_confidence:.2f}"
        )

        if self.low_confidence_entities > 0:
            logger.warning(f"Document {document_id}: {self.low_confidence_entities} low-confidence entities detected")

        if entity_ratio < 0.1 and self.total_chunks > 5:
            logger.warning(
                f"Document {document_id}: low entity extraction rate ({entity_ratio:.1%}). "
                "Consider reviewing extraction skill or content quality."
            )


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

        # Build storage config (shared helper handles SurrealDB, pool_pre_ping, etc.)
        self._storage_config = storage_config or build_storage_config(config)

        self._storage: StorageCoordinator | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._query_engine: HybridQueryEngine | None = None
        self._connected = False

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

    @trace("khora.graphrag.remember")
    async def remember(
        self,
        content: str,
        namespace_id: UUID,
        *,
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
        """Store content in the memory engine.

        Processes document through the ingestion pipeline with parallel
        chunking, embedding, and entity extraction for optimal performance.

        Args:
            content: The document text to store.
            namespace_id: Target namespace UUID.
            title: Optional document title.
            source: Optional source identifier.
            metadata: Optional custom metadata dict.
            skill_name: Extraction skill to use.
            entity_types: Allowed entity types for extraction.
            relationship_types: Allowed relationship types for extraction.
            expertise: Optional expertise configuration.
            extraction_config_hash: Optional hash for extraction config dedup.
            chunk_strategy: Override chunking strategy for this call.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.

        Returns:
            RememberResult with document_id, counts, and timing metrics in metadata.
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # Compute checksum
        start = time.perf_counter()
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        timings["checksum_ms"] = (time.perf_counter() - start) * 1000

        storage = self._get_storage()

        # Check for duplicate - skip if any document with same checksum exists.
        # NOTE: dedup is content-only; changing extraction_config_hash without
        # changing content will still hit this early return. Re-extraction on
        # config change would require a force_reprocess flag or composite key.
        start = time.perf_counter()
        existing = await storage.get_document_by_checksum(namespace_id, checksum)
        timings["dedup_check_ms"] = (time.perf_counter() - start) * 1000

        if existing:
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000
            logger.debug(f"Document already exists (checksum={checksum[:8]}..., status={existing.status})")
            return RememberResult(
                document_id=existing.id,
                namespace_id=namespace_id,
                chunks_created=existing.chunk_count,
                entities_extracted=existing.entity_count,
                relationships_created=0,
                metadata={"duplicate": True, "status": str(existing.status), "timings": timings},
            )

        # Create document
        start = time.perf_counter()
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
            extraction_config_hash=extraction_config_hash,
        )
        document = await storage.create_document(document)
        timings["document_create_ms"] = (time.perf_counter() - start) * 1000

        # Process through pipeline (handles chunking, embedding, extraction in parallel)
        from khora.pipelines.flows.ingest import process_document

        start = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
            entity_types=entity_types,
            relationship_types=relationship_types,
            expertise=expertise,
        )
        if chunk_strategy is not None:
            kwargs["chunk_strategy"] = chunk_strategy
        result = await process_document(document, storage, **kwargs)
        timings["pipeline_ms"] = (time.perf_counter() - start) * 1000
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        # Track extraction quality metrics
        quality_metrics = ExtractionQualityMetrics(
            total_chunks=result["chunks"],
            total_entities=result["entities"],
            total_relationships=result["relationships"],
            extraction_time_ms=timings.get("pipeline_ms", 0),
        )
        quality_metrics.compute_averages()
        quality_metrics.log_quality_summary(document.id)

        # Log performance summary
        logger.debug(
            f"remember() completed: {result['chunks']} chunks, {result['entities']} entities, "
            f"{result['relationships']} relationships in {timings['total_ms']:.1f}ms"
        )

        # Invalidate query caches so new documents are visible to searches
        self._get_query_engine().invalidate_caches(namespace_id)

        return RememberResult(
            document_id=document.id,
            namespace_id=namespace_id,
            chunks_created=result["chunks"],
            entities_extracted=result["entities"],
            relationships_created=result["relationships"],
            metadata={
                "timings": timings,
                "quality_metrics": {
                    "avg_entities_per_chunk": quality_metrics.avg_entities_per_chunk,
                    "extraction_time_ms": quality_metrics.extraction_time_ms,
                },
            },
        )

    @trace("khora.graphrag.recall")
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
        temporal_filter: Any | None = None,
        recency_bias: float | None = None,
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
            config.enable_hyde = "never"

        # Auto-detect temporal category when no explicit filter and not raw mode
        temporal_signal: TemporalSignal | None = None
        if temporal_filter is None and not raw:
            detector = TemporalDetector()
            temporal_signal = detector.detect(query)
            params = get_retrieval_params(temporal_signal)

            if temporal_signal.is_temporal:
                config.enable_temporal_detection = True
            if temporal_signal.category in (TemporalCategory.RECENCY, TemporalCategory.STATE_QUERY):
                config.apply_recency_bias = True
                config.recency_weight = params.recency_weight

        # Apply explicit recency_bias override
        if recency_bias is not None:
            config.apply_recency_bias = True
            config.recency_weight = recency_bias

        result = await self._get_query_engine().query(query, namespace_id, config=config, agentic=agentic)

        metadata = dict(result.metadata)
        if temporal_signal is not None:
            metadata["temporal_category"] = temporal_signal.category.value
            metadata["temporal_confidence"] = temporal_signal.confidence

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            chunks=result.chunks,
            entities=result.entities,
            context_text=result.get_context_text(max_chunks=limit),
            metadata=metadata,
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

    @trace("khora.graphrag.remember_batch")
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
    ) -> BatchResult:
        """Store multiple documents with automatic optimization.

        Processes documents in parallel with configurable concurrency.
        Uses shared embedder and entity index for efficiency.

        Args:
            documents: List of document dicts with content, title, source, metadata.
            namespace_id: Target namespace UUID.
            skill_name: Extraction skill to use.
            max_concurrent: Maximum number of documents to process in parallel.
            deduplicate: Whether to deduplicate entities across documents.
            infer_relationships: Whether to infer relationships between entities.
            on_progress: Optional callback for progress reporting.
            entity_types: Allowed entity types for extraction.
            relationship_types: Allowed relationship types for extraction.
            expertise: Optional expertise configuration.
            extraction_config_hash: Optional hash for extraction config dedup.
            chunk_strategy: Override chunking strategy for this call.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.

        Returns:
            BatchResult with counts and timing metrics.
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

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
        start = time.perf_counter()
        doc_inputs = []
        for doc_data in documents:
            entry: dict[str, Any] = {
                "content": doc_data.get("content", ""),
                "title": doc_data.get("title", ""),
                "source": doc_data.get("source", ""),
                "source_type": "api",
                "metadata": doc_data.get("metadata", {}),
            }
            if extraction_config_hash is not None:
                entry["extraction_config_hash"] = extraction_config_hash
            doc_inputs.append(entry)
        timings["prepare_inputs_ms"] = (time.perf_counter() - start) * 1000

        from khora.pipelines.flows.ingest import ingest_documents

        # Create a shared embedder for efficiency (uses LRU cache internally)
        shared_embedder = LiteLLMEmbedder(model=self._config.llm.embedding_model)

        # Create shared EntityIndex for cross-document deduplication if enabled
        shared_entity_index = None
        if deduplicate:
            start = time.perf_counter()
            from khora.extraction.expansion.entity_index import EntityIndex

            shared_entity_index = EntityIndex()

            # Optionally preload existing entities for dedup against stored data
            existing_entities = await self._get_storage().list_entities(namespace_id, limit=50000)
            for entity in existing_entities:
                shared_entity_index.add(entity)

            timings["entity_preload_ms"] = (time.perf_counter() - start) * 1000
            if existing_entities:
                logger.debug(f"Preloaded {len(existing_entities)} existing entities into EntityIndex")

        # Determine expansion: expertise.expansion.enabled takes precedence
        effective_expansion = infer_relationships
        if expertise is not None and expertise.expansion.enabled:
            effective_expansion = True

        start = time.perf_counter()
        ingest_kwargs: dict[str, Any] = dict(
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            extraction_model=self._config.llm.extraction_model or self._config.llm.model,
            max_concurrent_documents=max_concurrent,
            shared_embedder=shared_embedder,
            shared_entity_index=shared_entity_index,
            enable_expansion=effective_expansion,
            entity_types=entity_types,
            relationship_types=relationship_types,
            expertise=expertise,
        )
        if chunk_strategy is not None:
            ingest_kwargs["chunk_strategy"] = chunk_strategy
        result = await ingest_documents(namespace_id, doc_inputs, self._get_storage(), **ingest_kwargs)
        timings["ingest_pipeline_ms"] = (time.perf_counter() - start) * 1000
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        # Calculate throughput metrics
        processed = result.get("processed_documents", 0)
        if processed > 0 and timings["total_ms"] > 0:
            timings["docs_per_second"] = processed / (timings["total_ms"] / 1000)
            timings["avg_doc_ms"] = timings["ingest_pipeline_ms"] / processed

        logger.info(
            f"remember_batch() completed: {processed}/{len(documents)} docs, "
            f"{result.get('total_chunks', 0)} chunks, {result.get('total_entities', 0)} entities "
            f"in {timings['total_ms']:.1f}ms ({timings.get('docs_per_second', 0):.1f} docs/sec)"
        )

        # Invalidate query caches so new documents are visible to searches
        self._get_query_engine().invalidate_caches(namespace_id)

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
            metadata={"timings": timings},
        )

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def create_namespace(
        self,
        *,
        config_overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace."""
        namespace = MemoryNamespace(
            config_overrides=config_overrides or {},
            metadata=metadata or {},
        )
        return await self._get_storage().create_namespace(namespace)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        return await self._get_storage().get_namespace(namespace_id)

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

    @trace("khora.find_related_entities", result=lambda r: {"result_count": len(r)})
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

    @trace("khora.search_entities", exclude={"query"}, result=lambda r: {"result_count": len(r)})
    async def search_entities(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity.

        Uses batch entity fetching to avoid N+1 queries for better performance.
        """
        # Embed the query
        if self._embedder is None:
            raise RuntimeError("GraphRAG engine not connected. Call connect() first.")

        query_embedding = await self._embedder.embed(query)

        # Search similar entities
        storage = self._get_storage()
        entity_ids_scores = await storage.search_similar_entities(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=0.0,
        )

        if not entity_ids_scores:
            return []

        # Batch fetch all entities in a single query (avoids N+1)
        entity_ids = [entity_id for entity_id, _ in entity_ids_scores]
        entities_map = await storage.get_entities_batch(entity_ids)

        # Return entities in score order, filtering out any that weren't found
        return [entities_map[eid] for eid, _score in entity_ids_scores if eid in entities_map]

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace."""
        storage = self._get_storage()

        doc_count = 0
        chunk_count = 0
        entity_count = 0
        relationship_count = 0
        last_activity_at = None

        try:
            doc_count, last_activity_at = await storage.get_document_stats(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        try:
            chunk_count = await storage.count_chunks(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        try:
            entity_count = await storage.count_entities(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        try:
            relationship_count = await storage.count_relationships(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        return Stats(
            documents=doc_count,
            chunks=chunk_count,
            entities=entity_count,
            relationships=relationship_count,
            last_activity_at=last_activity_at,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components and dependencies.

        Returns a dict with:
        - status: 'healthy', 'degraded', or 'disconnected'
        - checks: Individual component check results
        - engine: Engine name for identification
        """
        if not self._connected:
            return {"status": "disconnected", "engine": "graphrag"}

        health: dict[str, Any] = {
            "engine": "graphrag",
            "status": "healthy",
            "checks": {},
        }

        # Check storage coordinator (PostgreSQL + backends)
        try:
            storage_health = await self._get_storage().health_check()
            health["checks"]["storage"] = storage_health.summary
            if not storage_health.is_healthy:
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["storage"] = f"error: {e}"
            health["status"] = "degraded"

        # Check embedder availability
        try:
            if self._embedder is not None:
                # Simple test - just verify the embedder is configured
                health["checks"]["embedder"] = "ok"
            else:
                health["checks"]["embedder"] = "not configured"
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["embedder"] = f"error: {e}"
            health["status"] = "degraded"

        # Check query engine
        try:
            if self._query_engine is not None:
                health["checks"]["query_engine"] = "ok"
            else:
                health["checks"]["query_engine"] = "not configured"
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["query_engine"] = f"error: {e}"
            health["status"] = "degraded"

        return health
