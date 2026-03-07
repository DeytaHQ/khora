"""VectorCypher engine - hybrid vector+graph retrieval with temporal support.

This engine implements the VectorCypher retrieval paradigm inspired by Graph RAG 2026:
- Vector search to find entry entities (pgvector)
- Cypher traversal to expand relationships (Neo4j)
- Query routing to optimize simple vs complex queries
- HippoRAG 2 dual-node architecture (Entity + Chunk nodes)
- Skeleton-based construction (KET-RAG) - full KG extraction for top chunks (default 70%)
- Bi-temporal edges (Graphiti-style) - occurred_at vs ingested_at with invalidation

Target: Sub-300ms P95 for simple queries, sub-800ms for complex multi-hop queries.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger
from neo4j import AsyncGraphDatabase

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.models import (
    Chunk,
    Document,
    DocumentMetadata,
    Entity,
    MemoryNamespace,
    Relationship,
)
from khora.engines.skeleton.backends import TemporalChunk, TemporalFilter, create_temporal_store
from khora.engines.skeleton.skeleton import SkeletonIndexer
from khora.extraction.embedders import LiteLLMEmbedder
from khora.memory_lake import BatchResult, RecallResult, RememberResult, Stats
from khora.query import SearchMode
from khora.storage import StorageConfig, create_storage_coordinator
from khora.telemetry import trace, trace_span

from .dual_nodes import DualNodeManager, EntityChunkLink
from .retriever import RetrieverConfig, VectorCypherRetriever
from .router import QueryComplexityRouter, RouterConfig
from .temporal_detection import TemporalDetector, TemporalSignal

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from khora.extraction.skills import ExpertiseConfig
    from khora.storage import StorageCoordinator


_MAX_COOCCURRENCE_PER_CHUNK = 15


def _build_cooccurrence_relationships(
    entities: list[Entity],
    namespace_id: UUID,
    existing_relationships: list[Relationship],
) -> list[Relationship]:
    """Create ASSOCIATED_WITH edges between entities sharing the same chunk.

    This mirrors the co-occurrence logic in ``pipelines/flows/ingest.py:369-405``
    to ensure VectorCypher builds equally dense graphs.  Capped at
    ``_MAX_COOCCURRENCE_PER_CHUNK`` per chunk to prevent quadratic explosion.
    """
    # Build chunk → entities map
    chunk_entity_map: dict[UUID, list[Entity]] = {}
    for entity in entities:
        for chunk_id in entity.source_chunk_ids:
            chunk_entity_map.setdefault(chunk_id, []).append(entity)

    # Collect existing pairs to avoid duplicates
    existing_pairs: set[tuple[UUID, UUID]] = set()
    for r in existing_relationships:
        pair = (min(r.source_entity_id, r.target_entity_id), max(r.source_entity_id, r.target_entity_id))
        existing_pairs.add(pair)

    cooccurrence_rels: list[Relationship] = []
    for chunk_id, chunk_entities in chunk_entity_map.items():
        if len(chunk_entities) < 2:
            continue
        chunk_count = 0
        for i, e1 in enumerate(chunk_entities):
            for e2 in chunk_entities[i + 1 :]:
                pair = (min(e1.id, e2.id), max(e1.id, e2.id))
                if pair in existing_pairs:
                    continue
                existing_pairs.add(pair)
                cooccurrence_rels.append(
                    Relationship(
                        source_entity_id=e1.id,
                        target_entity_id=e2.id,
                        relationship_type="ASSOCIATED_WITH",
                        namespace_id=namespace_id,
                        description="Co-occurs in same chunk",
                        properties={},
                        confidence=0.4,
                    )
                )
                chunk_count += 1
                if chunk_count >= _MAX_COOCCURRENCE_PER_CHUNK:
                    break
            if chunk_count >= _MAX_COOCCURRENCE_PER_CHUNK:
                break

    if cooccurrence_rels:
        logger.debug(f"VectorCypher: added {len(cooccurrence_rels)} co-occurrence edges")

    return cooccurrence_rels


@dataclass(slots=True)
class ExtractionQualityMetrics:
    """Track extraction quality for monitoring."""

    total_chunks: int = 0
    chunks_with_entities: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    avg_entities_per_chunk: float = 0.0
    avg_confidence: float = 0.0
    entity_type_distribution: dict[str, int] = field(default_factory=dict)

    def compute_averages(self) -> None:
        """Compute average metrics from totals."""
        if self.total_chunks > 0:
            self.avg_entities_per_chunk = self.total_entities / self.total_chunks


@dataclass(slots=True)
class VectorCypherConfig:
    """VectorCypher-specific configuration."""

    # Routing
    routing_enabled: bool = True
    routing_use_llm: bool = False

    # Skeleton indexing
    skeleton_core_ratio: float = 0.70  # 70% get full KG extraction (increased for denser graphs)

    # Graph traversal
    graph_default_depth: int = 2
    graph_max_depth: int = 4
    graph_max_entry_entities: int = 10

    # Fusion
    fusion_rrf_k: int = 60
    fusion_vector_weight: float = 0.6
    fusion_graph_weight: float = 0.4
    fusion_simple_vector_weight: float = 0.8
    fusion_simple_graph_weight: float = 0.2
    fusion_complex_vector_weight: float = 0.4
    fusion_complex_graph_weight: float = 0.6

    # Temporal
    temporal_recency_weight: float = 0.2
    temporal_recency_decay_days: int = 30
    recency_decay_type: str = "exponential"  # "linear" or "exponential"

    # Query caching
    query_cache_ttl_seconds: int = 300  # 5 min TTL
    query_cache_max_size: int = 100

    # Extraction concurrency (aligned with ingest pipeline's default of 20)
    max_concurrent_extractions: int = 20

    # Streaming pipeline (A-1: batch entity storage across documents)
    streaming_pipeline: bool = True
    enable_smart_resolution: bool = True

    # Skip LLM entity extraction for short messages (conversation batches).
    # Messages with all chunks ≤ this token count rely on BM25 + vector search instead.
    min_extraction_tokens: int = 50

    # Lazy entity expansion (recovers graph signal for non-core chunks)
    lazy_entity_expansion: bool = True

    # Search thresholds
    fusion_hybrid_alpha: float = 0.7
    retriever_min_entity_similarity: float = 0.3


class VectorCypherEngine:
    """VectorCypher engine - hybrid vector+graph retrieval with temporal support.

    Key features:
    - Dual retrieval: Vector similarity (pgvector) + graph traversal (Neo4j)
    - Smart routing: Route queries to optimal path (simple vs complex)
    - Skeleton indexing: Full KG extraction only for core chunks (configurable ratio)
    - Bi-temporal model: Track occurred_at vs ingested_at
    - RRF fusion: Combine vector and graph scores

    Requirements:
    - PostgreSQL with pgvector extension
    - Neo4j (required, not optional like in SkeletonEngine)

    Usage:
        engine = VectorCypherEngine(config)
        await engine.connect()

        # Store with temporal context
        result = await engine.remember(
            "Meeting with John about Q1 planning",
            namespace_id,
            occurred_at=datetime(2024, 1, 15),
        )

        # Retrieve with hybrid search
        result = await engine.recall(
            "What did we discuss with John about planning?",
            namespace_id,
            graph_depth=2,
        )
    """

    def __init__(
        self,
        config: KhoraConfig,
        *,
        storage_config: StorageConfig | None = None,
        vectorcypher_config: VectorCypherConfig | None = None,
    ) -> None:
        """Initialize the VectorCypher engine.

        Args:
            config: KhoraConfig instance
            storage_config: Storage configuration (deprecated, derived from config)
            vectorcypher_config: VectorCypher-specific configuration
        """
        self._config = config
        self._vc_config = vectorcypher_config or VectorCypherConfig()

        # Build storage config
        if storage_config:
            self._storage_config = storage_config
        else:
            postgresql_url = config.get_postgresql_url()
            graph_config = config.get_graph_config()

            storage_kwargs: dict[str, Any] = {
                "postgresql_url": postgresql_url,
                "pgvector_url": postgresql_url,
                "postgresql_pool_size": config.storage.postgresql_pool_size,
                "postgresql_max_overflow": config.storage.postgresql_max_overflow,
                "pgvector_embedding_dimension": config.storage.embedding_dimension,
                "pgvector_use_halfvec": config.storage.use_halfvec,
                "graph_config": graph_config,
                "vector_config": config.get_vector_config(),
            }
            if graph_config is not None:
                storage_kwargs["neo4j_url"] = config.get_neo4j_url()
                storage_kwargs["neo4j_user"] = config.get_neo4j_user()
                storage_kwargs["neo4j_password"] = config.get_neo4j_password()
                storage_kwargs["neo4j_database"] = config.get_neo4j_database()

            self._storage_config = StorageConfig(**storage_kwargs)

        # Component instances (initialized on connect)
        self._storage: StorageCoordinator | None = None
        self._temporal_store = None
        self._neo4j_driver: AsyncDriver | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._retriever: VectorCypherRetriever | None = None
        self._dual_nodes: DualNodeManager | None = None
        self._router: QueryComplexityRouter | None = None
        self._connected = False

        # Default namespace cache
        self._default_namespace_id: UUID | None = None

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting VectorCypher engine...")

        # Connect to Neo4j (required for VectorCypher) — single shared driver
        neo4j_url = self._config.get_neo4j_url()
        if not neo4j_url:
            raise ValueError(
                "Neo4j URL is required for VectorCypher engine. Set GENESIS_NEO4J_URL or configure graph_config."
            )

        neo4j_cfg = self._config.get_graph_config()
        pool_size = getattr(neo4j_cfg, "max_connection_pool_size", 100) if neo4j_cfg else 100
        self._neo4j_driver = AsyncGraphDatabase.driver(
            neo4j_url,
            auth=(self._config.get_neo4j_user(), self._config.get_neo4j_password()),
            max_connection_pool_size=pool_size,
        )
        await self._neo4j_driver.verify_connectivity()

        # Create and connect relational storage, sharing the Neo4j driver
        # so only one connection pool is used for the entire engine.
        self._storage = create_storage_coordinator(self._storage_config)

        from khora.storage.backends.neo4j import Neo4jBackend

        neo4j_database = self._config.get_neo4j_database() or "neo4j"
        if self._storage.graph is not None:
            self._storage.graph = Neo4jBackend.from_driver(
                self._neo4j_driver,
                database=neo4j_database,
                entity_write_concurrency=getattr(neo4j_cfg, "entity_write_concurrency", 12),
                relationship_write_concurrency=getattr(neo4j_cfg, "relationship_write_concurrency", 8),
            )

        await self._storage.connect()

        # Create and connect temporal vector store (pgvector)
        self._temporal_store = create_temporal_store("pgvector", self._config)
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

        # Initialize dual node manager
        self._dual_nodes = DualNodeManager(self._neo4j_driver, neo4j_database)
        await self._dual_nodes.ensure_indexes()

        # Initialize router
        router_config = RouterConfig(
            enabled=self._vc_config.routing_enabled,
            use_llm=self._vc_config.routing_use_llm,
            moderate_depth=1,
            complex_depth=self._vc_config.graph_default_depth,
        )
        self._router = QueryComplexityRouter(router_config)

        # Initialize retriever
        retriever_config = RetrieverConfig(
            default_depth=self._vc_config.graph_default_depth,
            max_depth=self._vc_config.graph_max_depth,
            max_entry_entities=self._vc_config.graph_max_entry_entities,
            rrf_k=self._vc_config.fusion_rrf_k,
            vector_weight=self._vc_config.fusion_vector_weight,
            graph_weight=self._vc_config.fusion_graph_weight,
            simple_vector_weight=self._vc_config.fusion_simple_vector_weight,
            simple_graph_weight=self._vc_config.fusion_simple_graph_weight,
            complex_vector_weight=self._vc_config.fusion_complex_vector_weight,
            complex_graph_weight=self._vc_config.fusion_complex_graph_weight,
            recency_weight=self._vc_config.temporal_recency_weight,
            recency_decay_days=self._vc_config.temporal_recency_decay_days,
            recency_decay_type=self._vc_config.recency_decay_type,
            min_entity_similarity=self._vc_config.retriever_min_entity_similarity,
            hybrid_alpha=self._vc_config.fusion_hybrid_alpha,
            query_cache_ttl_seconds=self._vc_config.query_cache_ttl_seconds,
            query_cache_max_size=self._vc_config.query_cache_max_size,
            lazy_entity_expansion=self._vc_config.lazy_entity_expansion,
        )
        self._retriever = VectorCypherRetriever(
            vector_store=self._temporal_store,
            neo4j_driver=self._neo4j_driver,
            embedder=self._embedder,
            database=neo4j_database,
            config=retriever_config,
            storage=self._storage,
        )

        # Initialize telemetry
        from khora.telemetry import init_telemetry
        from khora.telemetry.config import TelemetryConfig

        telemetry_cfg = TelemetryConfig(
            database_url=self._config.telemetry_database_url,
            service_name=self._config.telemetry_service_name,
        )
        await init_telemetry(telemetry_cfg)

        self._connected = True
        logger.info("VectorCypher engine connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting VectorCypher engine...")

        # Shutdown telemetry
        from khora.telemetry import shutdown_telemetry

        await shutdown_telemetry()

        if self._neo4j_driver:
            await self._neo4j_driver.close()
            self._neo4j_driver = None

        if self._temporal_store:
            await self._temporal_store.disconnect()
            self._temporal_store = None

        if self._storage:
            await self._storage.disconnect()
            self._storage = None

        self._embedder = None
        self._retriever = None
        self._dual_nodes = None
        self._router = None
        self._connected = False

        logger.info("VectorCypher engine disconnected")

    def _get_storage(self) -> StorageCoordinator:
        if self._storage is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._storage

    def _get_temporal_store(self):
        if self._temporal_store is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._temporal_store

    def _get_embedder(self) -> LiteLLMEmbedder:
        if self._embedder is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._embedder

    def _get_retriever(self) -> VectorCypherRetriever:
        if self._retriever is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._retriever

    def _get_dual_nodes(self) -> DualNodeManager:
        if self._dual_nodes is None:
            raise RuntimeError("VectorCypher engine not connected. Call connect() first.")
        return self._dual_nodes

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
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        occurred_at: datetime | None = None,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> RememberResult:
        """Store content in the memory engine.

        Args:
            content: Content to store
            namespace_id: Namespace to store in
            title: Document title
            source: Document source
            metadata: Additional metadata
            skill_name: Extraction skill to use
            expertise: ExpertiseConfig, expertise name string, or file path
            extraction_model: LLM model for entity extraction (default: config model)
            occurred_at: When this content/event occurred (default: now)

        Returns:
            RememberResult with document_id and counts
        """
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

        # Process document
        chunks_created, entities_extracted, relationships_created = await self._process_document(
            document,
            skill_name=skill_name,
            expertise=expertise,
            extraction_model=extraction_model,
            occurred_at=occurred_at or datetime.now(UTC),
            entity_types=entity_types,
            relationship_types=relationship_types,
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
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        occurred_at: datetime,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> tuple[int, int, int]:
        """Process a document into chunks with skeleton-based entity extraction.

        The VectorCypher pipeline:
        1. Chunk the document
        2. Embed all chunks
        3. Run skeleton indexing to identify core chunks (configurable, default 70%)
        4. Extract entities only from core chunks
        5. Store chunks in pgvector and create Chunk nodes in Neo4j
        6. Link entities to chunks via MENTIONED_IN
        """
        from khora.pipelines.chunking import create_chunker  # type: ignore[unresolved-import]
        from khora.pipelines.chunking.config import ChunkerConfig  # type: ignore[unresolved-import]

        with trace_span(
            "khora.vectorcypher.process_document",
            namespace_id=str(document.namespace_id),
            document_id=str(document.id),
        ) as span:
            storage = self._get_storage()
            embedder = self._get_embedder()
            temporal_store = self._get_temporal_store()
            dual_nodes = self._get_dual_nodes()

            # Create chunker
            chunker_config = ChunkerConfig(
                strategy=self._config.pipeline.chunking_strategy,
                chunk_size=self._config.pipeline.chunk_size,
                chunk_overlap=self._config.pipeline.chunk_overlap,
            )
            chunker = create_chunker(chunker_config)

            # Chunk the document
            raw_chunks = await asyncio.to_thread(chunker.chunk, document.content)

            if not raw_chunks:
                document.mark_completed(0, 0)
                await storage.update_document(document)
                span.set_attribute("chunk_count", 0)
                return 0, 0, 0

            # Embed chunks in batch
            chunk_texts = [c.content for c in raw_chunks]
            embeddings = await embedder.embed_batch(chunk_texts)

            # Extract metadata
            doc_metadata = document.metadata.custom if document.metadata else {}

            # Create temporal chunks
            temporal_chunks = []
            for i, (raw_chunk, embedding) in enumerate(zip(raw_chunks, embeddings)):
                temporal_chunk = TemporalChunk(
                    id=None,
                    namespace_id=document.namespace_id,
                    document_id=document.id,
                    content=raw_chunk.content,
                    embedding=embedding,
                    occurred_at=occurred_at,
                    created_at=datetime.now(UTC),
                    source_system=doc_metadata.get("source_system"),
                    author=doc_metadata.get("author"),
                    channel=doc_metadata.get("channel") or doc_metadata.get("thread_id"),
                    tags=doc_metadata.get("tags", []),
                    confidence=1.0,
                    metadata={
                        "chunk_index": i,
                        "start_char": raw_chunk.start_char if hasattr(raw_chunk, "start_char") else 0,
                        "end_char": raw_chunk.end_char if hasattr(raw_chunk, "end_char") else len(raw_chunk.content),
                        **{k: v for k, v in doc_metadata.items() if isinstance(v, (str, int, float, bool))},
                    },
                )
                temporal_chunks.append(temporal_chunk)

            # Store in pgvector
            stored_chunks = await temporal_store.create_chunks_batch(temporal_chunks)

            # Update temporal_chunks with assigned IDs
            for i, stored in enumerate(stored_chunks):
                temporal_chunks[i].id = stored.id

            # Create Chunk nodes in Neo4j
            await dual_nodes.create_chunk_nodes_batch(temporal_chunks, document.namespace_id)

            # Skeleton-based entity extraction (for core chunks only)
            entities_extracted = 0
            relationships_created = 0

            if self._config.pipeline.extract_entities:
                entities_extracted, relationships_created = await self._run_skeleton_extraction(
                    temporal_chunks,
                    document.namespace_id,
                    skill_name=skill_name,
                    expertise=expertise,
                    extraction_model=extraction_model,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                )

            # Update document status
            document.mark_completed(len(stored_chunks), entities_extracted)
            await storage.update_document(document)

            logger.debug(
                f"Processed document {document.id}: {len(stored_chunks)} chunks, "
                f"{entities_extracted} entities, {relationships_created} relationships"
            )

            span.set_attribute("chunk_count", len(stored_chunks))
            span.set_attribute("entities_extracted", entities_extracted)
            span.set_attribute("relationships_created", relationships_created)
            return len(stored_chunks), entities_extracted, relationships_created

    async def _run_skeleton_extraction(
        self,
        chunks: list[TemporalChunk],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> tuple[int, int]:
        """Run skeleton-based entity extraction on core chunks only.

        Uses the skeleton indexer to identify the top core chunks (by PageRank),
        then runs full LLM extraction on those chunks and stores the results
        in both Neo4j (entities, relationships, MENTIONED_IN links) and pgvector.

        Args:
            chunks: All chunks from the document
            namespace_id: Namespace ID
            skill_name: Extraction skill to use
            expertise: ExpertiseConfig, expertise name string, or file path
            extraction_model: LLM model for extraction (default: config model)

        Returns:
            Tuple of (entities_extracted, relationships_created)
        """
        from khora.pipelines.tasks.extract import extract_entities

        if not chunks:
            return 0, 0

        with trace_span(
            "khora.vectorcypher.skeleton_extraction",
            namespace_id=str(namespace_id),
            total_chunks=len(chunks),
        ) as span:
            # Skip skeleton overhead for small documents (≤2 chunks)
            if len(chunks) <= 2:
                core_ids = {c.id for c in chunks}
            else:
                skeleton = SkeletonIndexer(core_ratio=self._vc_config.skeleton_core_ratio)
                skeleton.add_chunks_batch(chunks)
                core_ids = await asyncio.to_thread(skeleton.build_skeleton)

            logger.debug(f"Skeleton indexing: {len(core_ids)}/{len(chunks)} core chunks")
            span.set_attribute("core_chunks", len(core_ids))

            if not core_ids:
                return 0, 0

            # Get core chunks
            core_temporal_chunks = [c for c in chunks if c.id in core_ids]

            # Convert TemporalChunk -> Chunk for extract_entities()
            chunk_objects = []
            for tc in core_temporal_chunks:
                chunk_objects.append(
                    Chunk(
                        id=tc.id,
                        namespace_id=tc.namespace_id,
                        document_id=tc.document_id,
                        content=tc.content,
                        created_at=tc.created_at or tc.occurred_at,
                    )
                )

            # Run LLM extraction on core chunks
            model = extraction_model or self._config.llm.model
            entities, relationships = await extract_entities(
                chunk_objects,
                skill_name=skill_name,
                expertise=expertise,
                model=model,
                max_concurrent=self._vc_config.max_concurrent_extractions,
                entity_types=entity_types,
                relationship_types=relationship_types,
            )

            if not entities:
                span.set_attribute("entities_extracted", 0)
                return 0, 0

            # Compute entity embeddings (matching ingest pipeline format)
            embedder = self._get_embedder()
            entity_texts = [f"{e.name}: {e.description}" if e.description else e.name for e in entities]
            entity_embeddings = await embedder.embed_batch(entity_texts)
            for entity, embedding in zip(entities, entity_embeddings):
                entity.embedding = embedding
                entity.embedding_model = embedder.model_name

            storage = self._get_storage()
            dual_nodes = self._get_dual_nodes()

            # Store entities in Neo4j + pgvector
            await storage.upsert_entities_batch(namespace_id, entities)

            # Create co-occurrence relationships between entities in the same chunk
            cooccurrence_rels = _build_cooccurrence_relationships(entities, namespace_id, relationships)
            if cooccurrence_rels:
                relationships = list(relationships) + cooccurrence_rels

            # Store relationships in Neo4j
            relationships_created = 0
            if relationships:
                relationships_created = await storage.create_relationships_batch(relationships)

            # Build entity-chunk links from source_chunk_ids
            entity_chunk_links: list[EntityChunkLink] = []
            for entity in entities:
                for chunk_id in entity.source_chunk_ids:
                    entity_chunk_links.append(
                        EntityChunkLink(
                            entity_id=entity.id,
                            chunk_id=chunk_id,
                        )
                    )

            # Create MENTIONED_IN edges in Neo4j
            if entity_chunk_links:
                await dual_nodes.link_entities_to_chunks_batch(entity_chunk_links)

            logger.debug(
                f"Skeleton extraction: {len(entities)} entities, "
                f"{relationships_created} relationships from {len(core_temporal_chunks)} core chunks"
            )

            span.set_attribute("entities_extracted", len(entities))
            span.set_attribute("relationships_created", relationships_created)
            return len(entities), relationships_created

    async def _run_skeleton_extraction_deferred(
        self,
        chunks: list[TemporalChunk],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> tuple[list[Entity], list[Relationship], list[EntityChunkLink]]:
        """Run skeleton extraction but return results instead of storing.

        Same as _run_skeleton_extraction but defers storage so the caller
        can accumulate entities across multiple documents for batch storage.

        Returns:
            Tuple of (entities_with_embeddings, relationships, entity_chunk_links)
        """
        from khora.pipelines.tasks.extract import extract_entities

        if not chunks:
            return [], [], []

        # Skip LLM entity extraction for short conversation messages.
        # Short messages (≤ min_extraction_tokens per chunk) produce low-quality
        # extraction results. BM25 + vector search handles them better.
        min_tokens = self._vc_config.min_extraction_tokens
        if min_tokens > 0 and all(len(c.content.split()) <= min_tokens for c in chunks):
            logger.debug(f"Skipping entity extraction for {len(chunks)} short chunks " f"(all ≤ {min_tokens} tokens)")
            return [], [], []

        # Skip skeleton overhead for small documents (≤2 chunks)
        if len(chunks) <= 2:
            core_ids = {c.id for c in chunks}
        else:
            skeleton = SkeletonIndexer(core_ratio=self._vc_config.skeleton_core_ratio)
            skeleton.add_chunks_batch(chunks)
            core_ids = await asyncio.to_thread(skeleton.build_skeleton)

        logger.debug(f"Skeleton indexing (deferred): {len(core_ids)}/{len(chunks)} core chunks")

        if not core_ids:
            return [], [], []

        core_temporal_chunks = [c for c in chunks if c.id in core_ids]
        chunk_objects = [
            Chunk(
                id=tc.id,
                namespace_id=tc.namespace_id,
                document_id=tc.document_id,
                content=tc.content,
                created_at=tc.created_at or tc.occurred_at,
            )
            for tc in core_temporal_chunks
        ]

        model = extraction_model or self._config.llm.model
        entities, relationships = await extract_entities(
            chunk_objects,
            skill_name=skill_name,
            expertise=expertise,
            model=model,
            max_concurrent=self._vc_config.max_concurrent_extractions,
            entity_types=entity_types,
            relationship_types=relationship_types,
        )

        if not entities:
            return [], [], []

        # Compute entity embeddings
        embedder = self._get_embedder()
        entity_texts = [f"{e.name}: {e.description}" if e.description else e.name for e in entities]
        entity_embeddings = await embedder.embed_batch(entity_texts)
        for entity, embedding in zip(entities, entity_embeddings):
            entity.embedding = embedding
            entity.embedding_model = embedder.model_name

        # Create co-occurrence relationships between entities in the same chunk
        cooccurrence_rels = _build_cooccurrence_relationships(entities, namespace_id, relationships)
        if cooccurrence_rels:
            relationships = list(relationships) + cooccurrence_rels

        # Build entity-chunk links
        entity_chunk_links: list[EntityChunkLink] = []
        for entity in entities:
            for chunk_id in entity.source_chunk_ids:
                entity_chunk_links.append(EntityChunkLink(entity_id=entity.id, chunk_id=chunk_id))

        return entities, relationships, entity_chunk_links

    async def _process_document_streaming(
        self,
        document: Document,
        *,
        skill_name: str,
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        occurred_at: datetime,
        embedding_text_override: str | None = None,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> tuple[int, list[Entity], list[Relationship], list[EntityChunkLink]]:
        """Process a document, returning entities for deferred batch storage.

        Same as _process_document but returns entities/rels/links instead of
        storing them, allowing the caller to batch across documents.

        Args:
            embedding_text_override: If provided, use this text for embedding
                instead of the raw chunk content. The original content is still
                stored in the chunk (preserves substring-based metrics).

        Returns:
            Tuple of (chunks_created, entities, relationships, entity_chunk_links)
        """
        from khora.pipelines.chunking import create_chunker  # type: ignore[unresolved-import]
        from khora.pipelines.chunking.config import ChunkerConfig  # type: ignore[unresolved-import]

        storage = self._get_storage()
        embedder = self._get_embedder()
        temporal_store = self._get_temporal_store()
        dual_nodes = self._get_dual_nodes()

        chunker_config = ChunkerConfig(
            strategy=self._config.pipeline.chunking_strategy,
            chunk_size=self._config.pipeline.chunk_size,
            chunk_overlap=self._config.pipeline.chunk_overlap,
        )
        chunker = create_chunker(chunker_config)
        raw_chunks = await asyncio.to_thread(chunker.chunk, document.content)

        if not raw_chunks:
            document.mark_completed(0, 0)
            await storage.update_document(document)
            return 0, [], [], []

        # WS3: Use enriched text for embedding if provided (conversation context),
        # but store original content in the chunk for answer_accuracy matching.
        if embedding_text_override:
            embed_texts = [embedding_text_override]
        else:
            embed_texts = [c.content for c in raw_chunks]
        embeddings = await embedder.embed_batch(embed_texts)
        doc_metadata = document.metadata.custom if document.metadata else {}

        temporal_chunks = []
        for i, (raw_chunk, embedding) in enumerate(zip(raw_chunks, embeddings)):
            temporal_chunk = TemporalChunk(
                id=None,
                namespace_id=document.namespace_id,
                document_id=document.id,
                content=raw_chunk.content,
                embedding=embedding,
                occurred_at=occurred_at,
                created_at=datetime.now(UTC),
                source_system=doc_metadata.get("source_system"),
                author=doc_metadata.get("author"),
                channel=doc_metadata.get("channel") or doc_metadata.get("thread_id"),
                tags=doc_metadata.get("tags", []),
                confidence=1.0,
                metadata={
                    "chunk_index": i,
                    "start_char": raw_chunk.start_char if hasattr(raw_chunk, "start_char") else 0,
                    "end_char": raw_chunk.end_char if hasattr(raw_chunk, "end_char") else len(raw_chunk.content),
                    **{k: v for k, v in doc_metadata.items() if isinstance(v, (str, int, float, bool))},
                },
            )
            temporal_chunks.append(temporal_chunk)

        # Store chunks in pgvector
        stored_chunks = await temporal_store.create_chunks_batch(temporal_chunks)
        for i, stored in enumerate(stored_chunks):
            temporal_chunks[i].id = stored.id

        # Create Chunk nodes in Neo4j
        await dual_nodes.create_chunk_nodes_batch(temporal_chunks, document.namespace_id)

        # Deferred skeleton extraction — returns entities instead of storing
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        entity_chunk_links: list[EntityChunkLink] = []

        if self._config.pipeline.extract_entities:
            entities, relationships, entity_chunk_links = await self._run_skeleton_extraction_deferred(
                temporal_chunks,
                document.namespace_id,
                skill_name=skill_name,
                expertise=expertise,
                extraction_model=extraction_model,
                entity_types=entity_types,
                relationship_types=relationship_types,
            )

        return len(stored_chunks), entities, relationships, entity_chunk_links

    def _validate_recall_results(
        self,
        chunks: list[tuple[Chunk, float]],
        query: str,
        *,
        min_content_length: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Validate and filter retrieval results.

        Removes duplicates, filters out empty content, ensures scores are normalized,
        and logs quality warnings.

        Args:
            chunks: List of (chunk, score) tuples
            query: Original query text for logging context
            min_content_length: Minimum content length to accept

        Returns:
            Validated and filtered list of (chunk, score) tuples
        """
        validated: list[tuple[Chunk, float]] = []
        seen_ids: set[UUID] = set()
        empty_count = 0
        duplicate_count = 0

        for chunk, score in chunks:
            if not isinstance(chunk, Chunk):
                logger.warning(f"Skipping non-Chunk result: {type(chunk)}")
                continue

            # Skip duplicates
            if chunk.id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(chunk.id)

            # Skip empty content
            if not chunk.content or len(chunk.content.strip()) < min_content_length:
                empty_count += 1
                continue

            # Normalize score to [0, 1]
            normalized_score = min(max(score, 0.0), 1.0)

            validated.append((chunk, normalized_score))

        # Log quality warnings
        if duplicate_count > 0:
            logger.debug(f"Recall validation: removed {duplicate_count} duplicate chunks for query: {query[:50]}...")
        if empty_count > 0:
            logger.warning(f"Recall validation: filtered {empty_count} empty/short chunks for query: {query[:50]}...")

        return validated

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
        # VectorCypher-specific parameters
        temporal_filter: TemporalFilter | None = None,
        graph_depth: int | None = None,
        hybrid_alpha: float | None = None,
    ) -> RecallResult:
        """Recall memories relevant to a query using VectorCypher.

        Args:
            query: Query text
            namespace_id: Namespace to search
            limit: Maximum number of results
            mode: Search mode (VECTOR, GRAPH, HYBRID)
            min_similarity: Minimum similarity threshold
            agentic: Whether to use agentic mode
            raw: Disable all LLM features
            temporal_filter: Temporal constraints
            graph_depth: Override graph traversal depth
            hybrid_alpha: Blend factor (0=graph, 1=vector)

        Returns:
            RecallResult with chunks, entities, and context
        """
        retriever = self._get_retriever()

        # Cascade temporal detection: Aho-Corasick dictionary → (optional) semantic
        # Replaces the old regex + dateparser approach with categorized signals.
        temporal_signal: TemporalSignal | None = None
        if temporal_filter is None:
            detector = TemporalDetector()
            temporal_signal = detector.detect(query)
            # EXPLICIT category produces a date-range TemporalFilter for pushdown
            if temporal_signal.temporal_filter is not None:
                temporal_filter = temporal_signal.temporal_filter

        # Use VectorCypher retriever
        result = await retriever.retrieve(
            query=query,
            namespace_id=namespace_id,
            temporal_filter=temporal_filter,
            temporal_signal=temporal_signal,
            graph_depth=graph_depth,
            limit=limit,
        )

        # Validate and filter retrieval results
        validated_chunks = self._validate_recall_results(result.chunks, query)

        # Build context text from validated chunks
        context_parts = []
        for chunk, score in validated_chunks:
            context_parts.append(chunk.content)

        context_text = "\n\n---\n\n".join(context_parts[:limit])

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            chunks=validated_chunks,
            entities=result.entities,
            context_text=context_text,
            metadata={
                "engine": "vectorcypher",
                "routing": result.routing_decision.complexity.value,
                "use_graph": result.routing_decision.use_graph,
                "graph_depth": result.routing_decision.graph_depth,
                "raw_chunk_count": len(result.chunks),
                "validated_chunk_count": len(validated_chunks),
                **result.metadata,
            },
        )

    async def forget(self, document_id: UUID, namespace_id: UUID | None) -> bool:
        """Remove a memory from the engine."""
        storage = self._get_storage()
        temporal_store = self._get_temporal_store()
        dual_nodes = self._get_dual_nodes()

        # Verify namespace if provided
        if namespace_id:
            document = await storage.get_document(document_id)
            if document and document.namespace_id != namespace_id:
                logger.warning(f"Document {document_id} not in namespace {namespace_id}")
                return False

        ns_id = namespace_id
        if not ns_id:
            document = await storage.get_document(document_id)
            if document:
                ns_id = document.namespace_id
            else:
                return False

        # Delete from Neo4j (Chunk nodes and relationships)
        await dual_nodes.delete_chunks_by_document(document_id, ns_id)

        # Delete from pgvector
        await temporal_store.delete_chunks_by_document(document_id, ns_id)

        # Delete from relational storage
        return await storage.delete_document(document_id)

    @staticmethod
    def _build_conversation_context(
        sorted_docs: list[dict[str, Any]],
    ) -> dict[int, str]:
        """Build context-enriched embedding text for conversation messages.

        For each message, creates text that includes ±2 neighboring messages
        as context. This helps embeddings capture conversational flow.

        Args:
            sorted_docs: Documents sorted by occurred_at timestamp.

        Returns:
            Dict mapping document index → enriched text for embedding.
        """
        context_map: dict[int, str] = {}
        n = len(sorted_docs)
        for i in range(n):
            parts: list[str] = []
            # ±2 neighbor context window
            for j in range(max(0, i - 2), min(n, i + 3)):
                if j == i:
                    continue
                neighbor = sorted_docs[j]
                author = neighbor.get("metadata", {}).get("author", "")
                content = neighbor.get("content", "")[:100]
                prefix = "prev" if j < i else "next"
                parts.append(f"{prefix}: {author}: {content}")

            current = sorted_docs[i].get("content", "")
            if parts:
                context_str = " | ".join(parts)
                context_map[i] = f"[Context: {context_str}]\n{current}"
            else:
                context_map[i] = current
        return context_map

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        namespace_id: UUID,
        *,
        skill_name: str = "general_entities",
        expertise: ExpertiseConfig | str | None = None,
        extraction_model: str | None = None,
        max_concurrent: int = 20,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> BatchResult:
        """Store multiple documents with automatic optimization.

        Args:
            documents: List of document dicts with 'content', 'title', 'source', 'metadata'
            namespace_id: Namespace to store documents in
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Whether to skip duplicate documents
            infer_relationships: Whether to infer relationships
            on_progress: Callback for progress updates

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

        # Track results
        results: dict[str, int] = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
        }
        results_lock = asyncio.Lock()
        progress_count = 0
        progress_lock = asyncio.Lock()

        # Compute checksums
        doc_checksums: list[str] = []
        for doc_data in documents:
            content = doc_data.get("content", "")
            checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
            doc_checksums.append(checksum)

        # Batch lookup existing documents
        existing_docs: dict[str, Any] = {}
        if deduplicate:
            existing_docs = await storage.get_documents_by_checksums(namespace_id, doc_checksums)

        # Track in-flight checksums for intra-batch deduplication
        checksums_in_flight: set[str] = set()
        checksums_lock = asyncio.Lock()

        semaphore = asyncio.Semaphore(max_concurrent)

        use_streaming = self._vc_config.streaming_pipeline

        # WS3: Detect conversation mode and build context-enriched embedding text.
        # Conversation mode: >50% of docs have occurred_at AND avg content < 200 chars.
        conversation_context: dict[int, str] = {}
        docs_with_ts = sum(1 for d in documents if "occurred_at" in d.get("metadata", {}))
        avg_content_len = sum(len(d.get("content", "")) for d in documents) / max(len(documents), 1)
        if docs_with_ts > len(documents) * 0.5 and avg_content_len < 200:
            # Sort by occurred_at for context windowing
            indexed_docs = list(enumerate(documents))
            indexed_docs.sort(
                key=lambda x: x[1].get("metadata", {}).get("occurred_at", ""),
            )
            sorted_docs = [d for _, d in indexed_docs]
            conversation_context = self._build_conversation_context(sorted_docs)
            # Map from original index to context text
            orig_to_sorted = {orig_idx: sort_idx for sort_idx, (orig_idx, _) in enumerate(indexed_docs)}
            context_by_orig: dict[int, str] = {
                orig_idx: conversation_context[sort_idx]
                for orig_idx, sort_idx in orig_to_sorted.items()
                if sort_idx in conversation_context
            }
            logger.info(
                f"Conversation mode detected: {docs_with_ts}/{len(documents)} docs with timestamps, "
                f"avg content {avg_content_len:.0f} chars, enriching embeddings"
            )
        else:
            context_by_orig = {}

        # Streaming pipeline: accumulate entities across documents for batch storage
        all_entities: list[Entity] = []
        all_relationships: list[Relationship] = []
        all_entity_chunk_links: list[EntityChunkLink] = []
        entity_lock = asyncio.Lock()

        async def process_document(doc_data: dict[str, Any], checksum: str, doc_index: int = 0) -> None:
            nonlocal progress_count

            # Check for intra-batch duplicate
            async with checksums_lock:
                if checksum in checksums_in_flight:
                    async with results_lock:
                        results["skipped"] += 1
                    return
                checksums_in_flight.add(checksum)

            # Check if already exists
            if deduplicate and checksum in existing_docs:
                async with results_lock:
                    results["skipped"] += 1
                if on_progress:
                    async with progress_lock:
                        progress_count += 1
                        on_progress(progress_count, total)
                return

            async with semaphore:
                try:
                    doc_metadata = doc_data.get("metadata", {})
                    occurred_at = None
                    if "occurred_at" in doc_metadata:
                        occurred_at = self._parse_datetime(doc_metadata["occurred_at"])

                    if use_streaming:
                        # Streaming path: create document + process, defer entity storage
                        document = Document(
                            namespace_id=namespace_id,
                            content=doc_data.get("content", ""),
                            metadata=DocumentMetadata(
                                title=doc_data.get("title", ""),
                                source=doc_data.get("source", ""),
                                checksum=checksum,
                                source_type="api",
                                custom=doc_metadata,
                            ),
                        )
                        document = await storage.create_document(document)

                        chunks_created, entities, rels, links = await self._process_document_streaming(
                            document,
                            skill_name=skill_name,
                            expertise=expertise,
                            extraction_model=extraction_model,
                            occurred_at=occurred_at or datetime.now(UTC),
                            embedding_text_override=context_by_orig.get(doc_index),
                            entity_types=entity_types,
                            relationship_types=relationship_types,
                        )

                        # Accumulate entities for batch storage
                        if entities or rels or links:
                            async with entity_lock:
                                all_entities.extend(entities)
                                all_relationships.extend(rels)
                                all_entity_chunk_links.extend(links)

                        # Update document status
                        document.mark_completed(chunks_created, len(entities))
                        await storage.update_document(document)

                        async with results_lock:
                            results["processed"] += 1
                            results["chunks"] += chunks_created
                            results["entities"] += len(entities)
                            results["relationships"] += len(rels)
                    else:
                        # Legacy path: per-document storage
                        result = await self.remember(
                            doc_data.get("content", ""),
                            namespace_id,
                            title=doc_data.get("title", ""),
                            source=doc_data.get("source", ""),
                            metadata=doc_metadata,
                            skill_name=skill_name,
                            expertise=expertise,
                            extraction_model=extraction_model,
                            occurred_at=occurred_at,
                            entity_types=entity_types,
                            relationship_types=relationship_types,
                        )

                        async with results_lock:
                            if result.metadata.get("duplicate"):
                                results["skipped"] += 1
                            else:
                                results["processed"] += 1
                                results["chunks"] += result.chunks_created
                                results["entities"] += result.entities_extracted
                                results["relationships"] += result.relationships_created

                except Exception as e:
                    logger.error(f"Failed to process document: {e}")
                    async with results_lock:
                        results["failed"] += 1

            if on_progress:
                async with progress_lock:
                    progress_count += 1
                    on_progress(progress_count, total)

        # Phase 2: Process all documents in parallel
        await asyncio.gather(
            *[process_document(doc, checksum, idx) for idx, (doc, checksum) in enumerate(zip(documents, doc_checksums))]
        )

        # Phase 3: Batch entity storage (streaming pipeline only)
        if use_streaming and all_entities:
            dual_nodes = self._get_dual_nodes()

            # Cross-document entity dedup by normalized name:type
            if self._vc_config.enable_smart_resolution:
                from khora._accel import normalize_entity_name

                deduped: dict[str, Entity] = {}
                for entity in all_entities:
                    key = f"{normalize_entity_name(entity.name)}:{entity.entity_type}"
                    if key in deduped:
                        existing = deduped[key]
                        existing.mention_count += entity.mention_count
                        for doc_id in entity.source_document_ids:
                            if doc_id not in existing.source_document_ids:
                                existing.source_document_ids.append(doc_id)
                        for chunk_id in entity.source_chunk_ids:
                            if chunk_id not in existing.source_chunk_ids:
                                existing.source_chunk_ids.append(chunk_id)
                    else:
                        deduped[key] = entity
                all_entities = list(deduped.values())
                logger.debug(f"Cross-document dedup: {len(deduped)} unique entities")

            # Single batch upsert to Neo4j + pgvector
            await storage.upsert_entities_batch(namespace_id, all_entities)

            if all_relationships:
                await storage.create_relationships_batch(all_relationships)

            if all_entity_chunk_links:
                await dual_nodes.link_entities_to_chunks_batch(all_entity_chunk_links)

            logger.info(
                f"Streaming pipeline batch store: {len(all_entities)} entities, "
                f"{len(all_relationships)} relationships, {len(all_entity_chunk_links)} links"
            )

        return BatchResult(
            total=total,
            processed=results["processed"],
            skipped=results["skipped"],
            failed=results["failed"],
            chunks=results["chunks"],
            entities=results["entities"],
            relationships=results["relationships"],
        )

    # Compiled regex for lightweight temporal keyword detection
    _TEMPORAL_KW_RE = re.compile(
        r"\b(when|before|after|during|since|until|last\s+(?:week|month|year|night|time)"
        r"|yesterday|today|recently|earlier|latest|newest|oldest|first|most\s+recent"
        r"|in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"|in\s+\d{4}|on\s+\d{1,2}[/\-]|ago)\b",
        re.IGNORECASE,
    )
    _DATE_EXTRACT_RE = re.compile(
        r"(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})"
        r"|(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{1,2},?\s+\d{4}\b)"
        r"|(\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{4}\b)",
        re.IGNORECASE,
    )

    def _detect_temporal_filter(self, query: str) -> TemporalFilter | None:
        """Lightweight regex-based temporal detection — no LLM call.

        Returns a TemporalFilter if temporal keywords and parseable dates
        are found, otherwise None. Cost: ~0.25ms.
        """
        if not self._TEMPORAL_KW_RE.search(query):
            return None

        # Try to extract an explicit date from the query
        date_match = self._DATE_EXTRACT_RE.search(query)
        if date_match:
            date_str = date_match.group(0)
            try:
                parsed_dt = self._parse_datetime(date_str)
                # "before" / "after" / default to "around that date" (±30 days)
                query_lower = query.lower()
                if "before" in query_lower:
                    return TemporalFilter(occurred_before=parsed_dt)
                elif "after" in query_lower or "since" in query_lower:
                    return TemporalFilter(occurred_after=parsed_dt)
                else:
                    # Within ±30 days of the mentioned date
                    from datetime import timedelta

                    return TemporalFilter(
                        occurred_after=parsed_dt - timedelta(days=30),
                        occurred_before=parsed_dt + timedelta(days=30),
                    )
            except ValueError:
                pass

        # Temporal keywords detected but no parseable date — signal to retriever
        # via a marker filter that enables recency boosting
        return None

    def _parse_datetime(self, value: Any) -> datetime:
        """Parse a datetime value from various formats."""
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError:
                pass
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except ValueError:
                pass
            # LongMemEval format: "2023/04/10 (Mon) 17:50"
            for fmt in (
                "%Y/%m/%d (%a) %H:%M",
                "%Y/%m/%d %H:%M",
                "%Y/%m/%d",
                "%B %d, %Y",
            ):
                try:
                    return datetime.strptime(value, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue
            # Last-resort: dateparser (handles a wide variety of natural-language dates)
            try:
                import dateparser

                dt = dateparser.parse(value, settings={"RETURN_AS_TIMEZONE_AWARE": True})
                if dt is not None:
                    return dt
            except Exception:
                pass
        raise ValueError(f"Cannot parse datetime: {value}")

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def get_or_create_default_namespace(self) -> UUID:
        """Get or create a default namespace for simple usage."""
        if self._default_namespace_id:
            return self._default_namespace_id

        storage = self._get_storage()

        # Try to find existing default namespace by slug
        default_namespace = await storage.get_namespace_by_slug("default")
        if not default_namespace:
            default_namespace = await storage.create_namespace(
                MemoryNamespace(
                    name="Default",
                    slug="default",
                )
            )

        self._default_namespace_id = default_namespace.id
        return self._default_namespace_id

    async def create_namespace(
        self,
        name: str,
        *,
        description: str = "",
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace."""
        namespace = MemoryNamespace(
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

        # Try to find namespace by slug
        slug = name.lower().replace(" ", "-")
        existing_ns = await storage.get_namespace_by_slug(slug)
        if existing_ns:
            return existing_ns.id

        new_ns = await storage.create_namespace(
            MemoryNamespace(
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

    @trace("khora.find_related_entities", result=lambda r: {"result_count": len(r)})
    async def find_related_entities(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity via graph traversal."""
        dual_nodes = self._get_dual_nodes()

        neighborhoods = await dual_nodes.get_entity_neighborhoods(
            entity_ids=[entity_id],
            namespace_id=namespace_id,
            depth=max_depth,
            limit_per_entity=limit,
        )

        results: list[tuple[Entity, float]] = []
        entity_infos = neighborhoods.get(str(entity_id), [])

        for info in entity_infos[:limit]:
            entity = await self._get_storage().get_entity(UUID(info["id"]))
            if entity:
                score = 1.0 / (1 + info.get("distance", 1))
                results.append((entity, score))

        return results

    @trace("khora.search_entities", exclude={"query"}, result=lambda r: {"result_count": len(r)})
    async def search_entities(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity."""
        embedder = self._get_embedder()
        query_embedding = await embedder.embed(query)

        # Search via storage coordinator
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

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace."""
        storage = self._get_storage()
        dual_nodes = self._get_dual_nodes()

        try:
            doc_count = await storage.count_documents(namespace_id)  # type: ignore[unresolved-attribute]
        except (AttributeError, NotImplementedError):
            documents = await storage.list_documents(namespace_id, limit=0)
            doc_count = len(documents) if documents else 0

        # Get chunk count from Neo4j
        chunk_count = await dual_nodes.count_chunks(namespace_id)

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

        # Check Neo4j
        neo4j_healthy = False
        if self._neo4j_driver:
            try:
                await self._neo4j_driver.verify_connectivity()
                neo4j_healthy = True
            except Exception:
                pass

        all_healthy = storage_health.is_healthy and temporal_health.get("status") == "healthy" and neo4j_healthy

        return {
            "status": "healthy" if all_healthy else "degraded",
            "storage": storage_health.summary,
            "temporal_store": temporal_health,
            "neo4j": "healthy" if neo4j_healthy else "unhealthy",
            "engine": "vectorcypher",
        }


__all__ = ["ExtractionQualityMetrics", "VectorCypherConfig", "VectorCypherEngine"]
