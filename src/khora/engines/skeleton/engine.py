"""Skeleton Construction engine - temporal-first memory engine.

This engine is optimized for:
- Temporal queries with structured field filtering
- Fast and cost-efficient ingestion
- High-precision retrieval with bi-temporal model
- Multiple backends (pgvector, Weaviate, SurrealDB, sqlite_lance)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.models import (
    Chunk,
    Document,
    Entity,
    MemoryNamespace,
)
from khora.core.models.recall import DocumentProjection, RecallChunk
from khora.core.recall_scoring import min_max_normalize
from khora.core.temporal import (
    ChunkTemporalFilter,
    TemporalChunk,
    document_denorm_fields,
)
from khora.engines._storage_config import build_storage_config
from khora.exceptions import EngineCapabilityError, UnsupportedEngineKwargError
from khora.extraction.embedders import LiteLLMEmbedder
from khora.filter.report import ChannelPlan, build_filter_report
from khora.khora import BatchResult, RecallResult, RememberResult, Stats
from khora.query import SearchMode
from khora.storage import StorageConfig, StorageCoordinator, create_storage_coordinator
from khora.storage.temporal import TemporalVectorStore
from khora.telemetry import trace, trace_span

if TYPE_CHECKING:
    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig
    from khora.filter import FilterNode


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

        # Weaviate backend (configure via env or config.storage.weaviate first)
        #   export KHORA_STORAGE_WEAVIATE_URL=http://localhost:8080
        engine = SkeletonConstructionEngine(config, backend="weaviate")
        await engine.connect()
    """

    # #833: Skeleton supports VECTOR (semantic), HYBRID (vector + BM25 via
    # the temporal store), and KEYWORD (pure BM25). GRAPH is unsupported -
    # Skeleton has no graph backend and its temporal edges are not wired
    # into recall(). ALL is unsupported because Skeleton has no additional
    # channels beyond what HYBRID already covers, so ALL would either
    # silently equal HYBRID or mislead callers about extra signal sources.
    supported_modes: ClassVar[frozenset[SearchMode]] = frozenset(
        {SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.KEYWORD}
    )

    def __init__(
        self,
        config: KhoraConfig,
        *,
        storage_config: StorageConfig | None = None,
        backend: str = "pgvector",
    ) -> None:
        """Initialize the Skeleton Construction engine.

        Args:
            config: KhoraConfig instance
            storage_config: Storage configuration (deprecated, derived from config)
            backend: Backend type ("pgvector", "weaviate", "turbopuffer",
                "surrealdb", or "sqlite_lance"). The ``weaviate`` and
                ``turbopuffer`` backends read their connection details from
                ``config.storage.weaviate`` / ``config.storage.turbopuffer``.
        """
        self._config = config
        # Auto-detect unified backends from config when not explicitly set
        if backend == "pgvector" and config.storage.backend == "surrealdb":
            backend = "surrealdb"
        elif backend == "pgvector" and config.storage.backend == "sqlite_lance":
            backend = "sqlite_lance"
        self._backend_type = backend

        # Build storage config — skip graph backend (skeleton is vector + BM25 only)
        self._storage_config = storage_config or build_storage_config(config, skip_graph=True)

        self._storage: StorageCoordinator | None = None
        self._temporal_store: TemporalVectorStore | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info(f"Connecting Skeleton Construction engine (backend={self._backend_type})...")

        # Create and connect relational storage (for documents, namespaces, etc.)
        self._storage = create_storage_coordinator(self._storage_config)
        await self._storage.connect()

        # Create and connect temporal vector store. The coordinator gathers the
        # per-backend shared resource (PG engine / EmbeddedStorageHandle /
        # SurrealDBConnection) so the store reuses existing connections, and
        # returns an already-connected store.
        self._temporal_store = await self._storage.temporal_store(self._backend_type, self._config)

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

    @trace("khora.skeleton.remember")
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
        occurred_at: datetime | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | str | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        external_id: str | None = None,
    ) -> RememberResult:
        """Store content in the memory engine.

        Args:
            content: Content to store
            namespace_id: Namespace to store in
            title: Document title
            source: Document source
            metadata: Additional metadata
            skill_name: Extraction skill (default: general_entities)
            occurred_at: When this content/event occurred. Resolution order:
                explicit ``occurred_at`` kwarg > ``metadata["occurred_at"]`` >
                ``source_timestamp`` > ``datetime.now(UTC)``.
            chunk_strategy: Override chunking strategy for this call.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.
            chunk_size: Override target chunk size (in tokens) for this call.
                When None (default), uses the configured pipeline default.

        Returns:
            RememberResult with document_id and counts

        Raises:
            UnsupportedEngineKwargError: When ``entity_types`` or
                ``relationship_types`` is non-empty, or ``expertise`` is
                non-None. The Skeleton engine deliberately skips typed
                entity / relationship extraction for cost efficiency;
                silently dropping these kwargs would hide a caller bug
                (#890, #1431). Pass empty lists / ``None`` when targeting
                Skeleton.
        """
        # #890: Skeleton does not extract typed entities or relationships.
        # Silently dropping the kwargs would mislead callers into thinking
        # the engine respected the type whitelist. Refuse non-empty values
        # so the caller picks a different engine or drops the kwargs.
        if entity_types:
            raise UnsupportedEngineKwargError(
                "skeleton",
                "entity_types",
                "Skeleton engine does not extract typed entities. "
                "Use VectorCypher or Chronicle for typed entity extraction.",
            )
        if relationship_types:
            raise UnsupportedEngineKwargError(
                "skeleton",
                "relationship_types",
                "Skeleton engine does not extract typed relationships. "
                "Use VectorCypher or Chronicle for typed relationship extraction.",
            )
        # #1431: expertise is the third ontology-guidance kwarg in the same
        # family; accepting-and-ignoring it was the leftover inconsistency
        # after #890 made the other two loud.
        if expertise is not None:
            raise UnsupportedEngineKwargError(
                "skeleton",
                "expertise",
                "Skeleton engine does not perform ontology-guided extraction. "
                "Use VectorCypher or Chronicle for expertise-guided extraction.",
            )

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
                relationships_created=existing.relationship_count,
                metadata={"duplicate": True, "status": str(existing.status)},
            )

        # Create document in relational storage
        document = Document(
            namespace_id=namespace_id,
            content=content,
            title=title or None,
            source=source or None,
            source_type=source_type,
            source_name=source_name or None,
            source_url=source_url or None,
            source_timestamp=source_timestamp,
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            metadata=dict(metadata or {}),
            extraction_config_hash=extraction_config_hash,
            external_id=external_id,
        )
        document = await storage.create_document(document)

        # Resolve occurred_at: explicit kwarg wins, then metadata["occurred_at"]
        # (parity with remember_batch), then the user-supplied source_timestamp
        # (parity with the relational Document.source_timestamp field), finally
        # fall back to now(). Fixes #856 - source_timestamp was previously
        # ignored on the chunk side even though it was persisted on Document.
        if occurred_at is None:
            if metadata and "occurred_at" in metadata:
                try:
                    occurred_at = self._parse_datetime(metadata["occurred_at"])
                except ValueError:
                    pass
            if occurred_at is None and source_timestamp is not None:
                occurred_at = source_timestamp
            if occurred_at is None:
                occurred_at = datetime.now(UTC)

        # Process through simplified pipeline (no full KG extraction)
        chunks_created, entities_extracted, relationships_created = await self._process_document(
            document,
            skill_name=skill_name,
            occurred_at=occurred_at,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
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
        selective_embedding: bool = False,
        importance_ratio: float = 0.7,
        chunk_strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
    ) -> tuple[int, int, int]:
        """Process a document into chunks (simplified pipeline).

        Unlike VectorCypher, this focuses on fast chunking and embedding without
        full entity extraction. Entity extraction can be done lazily on retrieval.
        """
        from khora.extraction.chunkers import create_chunker

        storage = self._get_storage()
        embedder = self._get_embedder()
        temporal_store = self._get_temporal_store()

        # Create chunker
        strategy = chunk_strategy if chunk_strategy is not None else self._config.pipeline.chunking_strategy
        chunker = create_chunker(
            strategy=strategy,
            chunk_size=chunk_size if chunk_size is not None else self._config.pipeline.chunk_size,
            chunk_overlap=self._config.pipeline.chunk_overlap,
        )

        # Chunk the document (run in thread to avoid blocking event loop during
        # CPU-bound tiktoken operations)
        with trace_span("khora.skeleton.chunk") as span:
            raw_chunks = await asyncio.to_thread(chunker.chunk, document.content)
            span.set_attribute("chunk_count", len(raw_chunks))

        if not raw_chunks:
            # Mark document as processed with 0 chunks
            document.mark_completed(0, 0)
            await storage.update_document(document)
            return 0, 0, 0

        # Select chunks for embedding based on importance scoring
        if selective_embedding:
            from khora.extraction.importance import ChunkImportanceScorer

            scorer = ChunkImportanceScorer()
            embed_chunks, skip_chunks = scorer.select_for_extraction(raw_chunks, ratio=importance_ratio)
            # Only embed selected chunks, store all with None embedding for skipped
        else:
            embed_chunks = raw_chunks

        # Embed selected chunks in batch
        chunk_texts = [c.content for c in embed_chunks]
        with trace_span("khora.skeleton.embed") as span:
            embeddings = await embedder.embed_batch(chunk_texts)
            span.set_attribute("embedding_count", len(embeddings))

        # Build a mapping from chunk content to embedding for selected chunks
        embed_map: dict[int, list[float]] = {}
        for idx, (chunk, embedding) in enumerate(zip(embed_chunks, embeddings)):
            # Use id of the raw_chunk object to map back
            embed_map[id(chunk)] = embedding

        # Extract metadata for filtering (source_system, author, channel, etc.)
        doc_metadata = document.metadata or {}

        # Create temporal chunks (all chunks, with None embedding for skipped)
        temporal_chunks = []
        for i, raw_chunk in enumerate(raw_chunks):
            embedding = embed_map.get(id(raw_chunk))
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
                metadata=dict(doc_metadata),
                chunker_info={
                    **dict(raw_chunk.metadata),
                    "chunk_index": i,
                    "start_char": raw_chunk.start_char if hasattr(raw_chunk, "start_char") else 0,
                    "end_char": raw_chunk.end_char if hasattr(raw_chunk, "end_char") else len(raw_chunk.content),
                    "token_count": raw_chunk.token_count if hasattr(raw_chunk, "token_count") else 0,
                },
                **document_denorm_fields(document),
            )
            temporal_chunks.append(temporal_chunk)

        # Store in temporal store
        stored_chunks = await temporal_store.create_chunks_batch(temporal_chunks)

        # Update document status
        document.mark_completed(len(stored_chunks), 0)
        await storage.update_document(document)

        logger.debug(f"Processed document {document.id}: {len(stored_chunks)} chunks")

        return len(stored_chunks), 0, 0

    @trace("khora.skeleton.recall")
    async def recall(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        # Khora-specific parameters
        temporal_filter: ChunkTemporalFilter | None = None,
        temporal_reference: datetime | None = None,
        hybrid_alpha: float | None = None,
        filters: dict[str, Any] | None = None,
        recency_bias: float | None = None,
        filter_ast: FilterNode | None = None,
    ) -> RecallResult:
        """Recall memories relevant to a query.

        Args:
            query: Query text
            namespace_id: Namespace to search
            limit: Maximum number of results
            mode: Search mode (VECTOR, KEYWORD, HYBRID)
            min_similarity: Minimum similarity threshold
            temporal_filter: Structured temporal filter
            temporal_reference: Reference point for relative time (e.g., message timestamp)
            hybrid_alpha: Blend factor for hybrid search (0=BM25, 1=vector)
            filters: Additional structured filters (converted to ChunkTemporalFilter)
            filter_ast: Canonical recall-filter AST. Threaded through to the
                temporal store's ``search``; the pgvector backend compiles it
                to a WHERE predicate, the other backends accept-and-ignore it.
            recency_bias: Accepted only as ``None`` for protocol parity.
                Skeleton does not implement temporal decay on the recall
                path; passing a non-None value silently no-ops in pre-fix
                builds. Now raises ``UnsupportedEngineKwargError`` so the
                caller routes to an engine that honors decay (Chronicle).

        Returns:
            RecallResult with chunks and context

        Raises:
            EngineCapabilityError: When ``mode`` is not in
                ``supported_modes``.
            UnsupportedEngineKwargError: When ``recency_bias`` is non-None.
                Skeleton does not apply temporal decay at recall time
                (#891).
        """
        # #833: validate the mode contract before doing any storage work.
        if mode not in self.supported_modes:
            raise EngineCapabilityError("skeleton", mode, self.supported_modes)

        # #891: Skeleton does not implement recency decay on recall.
        # Accepting recency_bias silently misleads callers into thinking
        # the engine applied a decay function. None is a no-op (accepted
        # for protocol parity); any concrete value must route to an engine
        # that actually applies decay (e.g. Chronicle).
        if recency_bias is not None:
            raise UnsupportedEngineKwargError(
                "skeleton",
                "recency_bias",
                "Skeleton engine does not apply temporal decay on recall. Use Chronicle for recency-weighted recall.",
            )

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
            elif mode == SearchMode.KEYWORD:
                hybrid_alpha = 0.0  # Pure BM25
            else:  # HYBRID
                hybrid_alpha = 0.7  # Default blend

        # Per-call sink for the honest filter-pushdown plan (#1069). A fresh list
        # per recall keeps the report race-free under concurrent recalls on a
        # shared store: the backend appends the ChannelPlan it built from the SAME
        # compile this call's WHERE ran, with no mutable instance state to clobber.
        filter_plan_sink: list[ChannelPlan] = []

        # Perform search
        results = await temporal_store.search(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
            temporal_filter=temporal_filter,
            hybrid_alpha=hybrid_alpha,
            query_text=query,
            filter_ast=filter_ast,
            filter_plan_out=filter_plan_sink,
        )

        chunks_with_scores: list[tuple[Chunk, float]] = []
        for result in results:
            chunk = Chunk(
                id=result.chunk.id,
                namespace_id=result.chunk.namespace_id,
                document_id=result.chunk.document_id,
                content=result.chunk.content,
                metadata={
                    "occurred_at": result.chunk.occurred_at.isoformat() if result.chunk.occurred_at else None,
                    **(result.chunk.metadata or {}),
                },
                created_at=result.chunk.created_at or result.chunk.occurred_at or datetime.now(UTC),
                occurred_at=result.chunk.occurred_at,
                source_timestamp=result.chunk.source_timestamp,
            )
            chunks_with_scores.append((chunk, result.combined_score or result.similarity))

        # #834: ``RecallChunk.score`` is a min-max normalized rank in [0, 1]
        # across all engines. Skeleton's raw ``combined_score or similarity``
        # is either raw cosine (VECTOR / HYBRID) or unbounded BM25 (KEYWORD);
        # min-max collapses both into the documented top=1.0 / bottom=0.0 shape.
        normalized_chunk_scores = min_max_normalize([s for _, s in chunks_with_scores])
        recall_chunks = [
            RecallChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=score,
                created_at=chunk.created_at,
                occurred_at=(chunk.occurred_at if chunk.occurred_at is not None else chunk.source_timestamp),
                chunker_info=chunk.chunker_info or {},
            )
            for (chunk, _), score in zip(chunks_with_scores, normalized_chunk_scores, strict=False)
        ]

        # Document stubs — fuller projections land with the recall-method rewrite.
        seen_doc_ids: set[UUID] = set()
        documents: list[DocumentProjection] = []
        for chunk, _ in chunks_with_scores:
            if chunk.document_id in seen_doc_ids:
                continue
            seen_doc_ids.add(chunk.document_id)
            src = chunk.source_document
            documents.append(
                DocumentProjection(
                    id=chunk.document_id,
                    created_at=chunk.created_at,
                    source_type=(src.source_type if src and src.source_type else "library"),
                    title=(src.title if src and src.title else None),
                    source=(src.source if src and src.source else None),
                    source_timestamp=(src.source_timestamp if src else None),
                    metadata=dict(chunk.metadata or {}),
                )
            )

        engine_info: dict[str, Any] = {
            "engine": "skeleton",
            "backend": self._backend_type,
            "hybrid_alpha": hybrid_alpha,
            "temporal_filter": str(temporal_filter) if temporal_filter else None,
        }
        # Report honest filter pushdown (#1069). The carrier is the ChannelPlan
        # the backend appended to ``filter_plan_sink`` from the SAME compile its
        # search just performed — never a backend-name check and never a re-compile
        # here, so the report cannot drift from what the WHERE clause actually
        # pushed. The fresh per-call sink (created above the search) makes this
        # race-free under concurrent recalls. The single configured backend is one
        # channel, keyed by ``self._backend_type``. ``build_filter_report``
        # enumerates the filter's leaves and partitions them per the plan; a
        # no-filter / constraint-free recall yields an empty plan and an all-False
        # report. The carrier is written on every recall.
        backend_plan = filter_plan_sink[0] if filter_plan_sink else ChannelPlan()
        engine_info["filter"] = build_filter_report(
            filter_ast,
            {self._backend_type: backend_plan},
            # Skeleton emits chunks only (no entities / relationships), and the
            # single backend channel gates the chunk surface, so covered_surfaces
            # defaults to {"chunks"} and the surface-coverage rule is inert here.
            surface_sizes={
                "chunks": len(recall_chunks),
                "entities": 0,
                "relationships": 0,
            },
        ).model_dump(mode="json")

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            documents=documents,
            chunks=recall_chunks,
            entities=[],  # Skeleton engine focuses on chunks, not entities
            relationships=[],
            engine_info=engine_info,
        )

    def _build_temporal_filter_from_dict(self, filters: dict[str, Any]) -> ChunkTemporalFilter:
        """Convert a filters dict to a ChunkTemporalFilter.

        Example:
            filters = {
                "occurred_at": {"gte": "2024-01-01", "lt": "2024-02-01"},
                "author": {"eq": "alice"},
                "source_system": {"eq": "slack"},
            }
        """

        tf = ChunkTemporalFilter()

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
        temporal_filter: ChunkTemporalFilter,
        reference: datetime,
    ) -> ChunkTemporalFilter:
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

        # namespace_id is required for IDOR-safe lookup (IDOR family). Callers
        # going through Khora.forget always resolve it before calling here;
        # bail loudly rather than allow a cross-tenant id probe.
        if namespace_id is None:
            logger.warning(f"Cannot forget document {document_id}: namespace_id is required")
            return False

        # Verify the document exists in this namespace before doing any work.
        document = await storage.get_document(document_id, namespace_id=namespace_id)
        if document is None:
            return False

        # Delete from temporal store
        await temporal_store.delete_chunks_by_document(document_id, namespace_id)

        # Delete from relational storage
        return await storage.delete_document(document_id, namespace_id=namespace_id)

    @trace("khora.skeleton.remember_batch")
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
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | str | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        bulk_mode: bool = False,
    ) -> BatchResult:
        """Store multiple documents with staged batch pipeline.

        Staged architecture (no entity extraction -- skeleton's identity):
          Stage 0: Batch dedup (checksum lookup + intra-batch dedup)
          Stage 1: Create documents + chunk in parallel
          Stage 2: Batch embed ALL chunks in one API call
          Stage 3: Build TemporalChunks + store batch
          Stage 4: Update document statuses

        Args:
            documents: List of document dicts with 'content', 'title', 'source', 'metadata'
            namespace_id: Namespace to store documents in
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Whether to skip duplicate documents
            infer_relationships: Not used in Skeleton engine (protocol compliance)
            on_progress: Callback for progress updates (completed, total)
            entity_types: Not used in Skeleton engine (protocol compliance)
            relationship_types: Not used in Skeleton engine (protocol compliance)
            expertise: Not used in Skeleton engine (protocol compliance)
            extraction_config_hash: Persisted for change-detection workflows
            chunk_strategy: Override chunking strategy for this call.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.
            chunk_size: Override target chunk size (in tokens) for this call.
                When None (default), uses the configured pipeline default.
            bulk_mode: When True, defer HNSW index creation for faster bulk loads

        Returns:
            BatchResult with processing statistics and timing metrics
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

        from khora.extraction.chunkers import create_chunker

        storage = self._get_storage()
        embedder = self._get_embedder()
        temporal_store = self._get_temporal_store()
        total = len(documents)

        progress_count = 0

        def _report_progress(n: int = 1) -> None:
            nonlocal progress_count
            if on_progress:
                progress_count += n
                on_progress(progress_count, total)

        if bulk_mode:
            from khora.storage.optimize import prepare_for_bulk_load

            await prepare_for_bulk_load(storage)

        try:
            # ── Stage 0: Batch dedup ──────────────────────────────────────
            start = time.perf_counter()

            # Compute checksums for all documents upfront
            doc_checksums: list[str] = []
            for doc_data in documents:
                content = doc_data.get("content", "")
                checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
                doc_checksums.append(checksum)

            # Batch lookup existing documents by checksum (single query instead of N)
            existing_docs: dict[str, Any] = {}
            if deduplicate:
                existing_docs = await storage.get_documents_by_checksums(namespace_id, doc_checksums)

            # Filter to non-duplicate documents with intra-batch dedup
            checksums_in_flight: set[str] = set()
            new_indices: list[int] = []
            skipped = 0
            for i, checksum in enumerate(doc_checksums):
                if deduplicate and checksum in existing_docs:
                    skipped += 1
                    _report_progress()
                    continue
                if checksum in checksums_in_flight:
                    skipped += 1
                    _report_progress()
                    continue
                checksums_in_flight.add(checksum)
                new_indices.append(i)

            timings["dedup_ms"] = (time.perf_counter() - start) * 1000
            logger.debug(f"Stage 0 dedup: {len(new_indices)} new, {skipped} skipped in {timings['dedup_ms']:.1f}ms")

            if not new_indices:
                timings["total_ms"] = (time.perf_counter() - total_start) * 1000
                return BatchResult(
                    total=total,
                    processed=0,
                    skipped=skipped,
                    failed=0,
                    chunks=0,
                    entities=0,
                    relationships=0,
                    metadata={"timings": timings},
                )

            # ── Stage 1: Create documents + chunk in parallel ────────────
            start = time.perf_counter()

            strategy = chunk_strategy if chunk_strategy is not None else self._config.pipeline.chunking_strategy
            chunker = create_chunker(
                strategy=strategy,
                chunk_size=chunk_size if chunk_size is not None else self._config.pipeline.chunk_size,
                chunk_overlap=self._config.pipeline.chunk_overlap,
            )

            # Create Document records for all new docs
            created_docs: list[Document] = []
            doc_metas: list[dict[str, Any]] = []
            occurred_ats: list[datetime] = []
            failed = 0

            for i in new_indices:
                doc_data = documents[i]
                checksum = doc_checksums[i]
                content = doc_data.get("content", "")
                doc_metadata = doc_data.get("metadata", {})

                # Parse occurred_at - resolution order:
                # metadata["occurred_at"] > source_timestamp (per-doc or batch
                # fallback) > now(). Fixes #856 - source_timestamp was ignored
                # on the chunk side even though it was persisted on Document.
                occurred_at: datetime | None = None
                if "occurred_at" in doc_metadata:
                    try:
                        occurred_at = self._parse_datetime(doc_metadata["occurred_at"])
                    except ValueError:
                        pass
                if occurred_at is None:
                    doc_source_ts = doc_data.get("source_timestamp", source_timestamp)
                    if doc_source_ts is not None:
                        occurred_at = doc_source_ts
                if occurred_at is None:
                    occurred_at = datetime.now(UTC)

                document = Document(
                    namespace_id=namespace_id,
                    content=content,
                    title=doc_data.get("title") or None,
                    source=doc_data.get("source") or None,
                    source_type=doc_data.get("source_type", source_type),
                    source_name=doc_data.get("source_name", source_name) or None,
                    source_url=doc_data.get("source_url", source_url) or None,
                    source_timestamp=doc_data.get("source_timestamp", source_timestamp),
                    checksum=checksum,
                    size_bytes=len(content.encode("utf-8")),
                    metadata=dict(doc_metadata),
                    extraction_config_hash=extraction_config_hash,
                    external_id=doc_data.get("external_id"),
                )
                try:
                    document = await storage.create_document(document)
                    created_docs.append(document)
                    doc_metas.append(doc_metadata)
                    occurred_ats.append(occurred_at)
                except Exception as e:
                    logger.error(f"Failed to create document: {e}")
                    failed += 1
                    _report_progress()

            # Chunk all documents in parallel (CPU-bound tiktoken runs in threads),
            # bounded by max_concurrent so the documented throttle is honored (#935).
            chunk_sem = asyncio.Semaphore(max_concurrent)

            async def _chunk_bounded(content: str) -> Any:
                async with chunk_sem:
                    return await asyncio.to_thread(chunker.chunk, content)

            with trace_span("khora.skeleton.batch_chunk") as span:
                chunk_tasks = [_chunk_bounded(doc.content) for doc in created_docs]
                all_raw_chunks = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                span.set_attribute("doc_count", len(created_docs))

            timings["create_and_chunk_ms"] = (time.perf_counter() - start) * 1000
            logger.debug(f"Stage 1 create+chunk: {len(created_docs)} docs in {timings['create_and_chunk_ms']:.1f}ms")

            # Build per-document chunk lists, tracking which docs had errors
            per_doc_chunks: list[list[Any]] = []
            docs_to_process: list[int] = []  # indices into created_docs
            for doc_idx, raw_result in enumerate(all_raw_chunks):
                if isinstance(raw_result, BaseException):
                    logger.error(f"Chunking failed for doc {created_docs[doc_idx].id}: {raw_result}")
                    failed += 1
                    _report_progress()
                    per_doc_chunks.append([])
                else:
                    per_doc_chunks.append(raw_result)
                    if raw_result:
                        docs_to_process.append(doc_idx)

            # ── Stage 2: Batch embed ALL chunks in one API call ──────────
            start = time.perf_counter()

            # Collect all chunk texts with a mapping back to (doc_idx, chunk_idx)
            all_chunk_texts: list[str] = []
            chunk_map: list[tuple[int, int]] = []  # (doc_idx, chunk_idx_within_doc)
            for doc_idx in docs_to_process:
                for chunk_idx, raw_chunk in enumerate(per_doc_chunks[doc_idx]):
                    all_chunk_texts.append(raw_chunk.content)
                    chunk_map.append((doc_idx, chunk_idx))

            all_embeddings: list[list[float]] = []
            if all_chunk_texts:
                with trace_span("khora.skeleton.batch_embed") as span:
                    all_embeddings = await embedder.embed_batch(all_chunk_texts)
                    span.set_attribute("chunk_count", len(all_chunk_texts))

            timings["embed_ms"] = (time.perf_counter() - start) * 1000
            logger.debug(f"Stage 2 embed: {len(all_chunk_texts)} chunks in {timings['embed_ms']:.1f}ms")

            # ── Stage 3: Build TemporalChunks + store batch ──────────────
            start = time.perf_counter()

            temporal_chunks: list[TemporalChunk] = []
            for emb_idx, (doc_idx, chunk_idx) in enumerate(chunk_map):
                doc = created_docs[doc_idx]
                raw_chunk = per_doc_chunks[doc_idx][chunk_idx]
                embedding = all_embeddings[emb_idx]
                doc_custom = doc_metas[doc_idx]
                occurred_at = occurred_ats[doc_idx]

                temporal_chunk = TemporalChunk(
                    id=None,
                    namespace_id=doc.namespace_id,
                    document_id=doc.id,
                    content=raw_chunk.content,
                    embedding=embedding,
                    occurred_at=occurred_at,
                    created_at=datetime.now(UTC),
                    source_system=doc_custom.get("source_system"),
                    author=doc_custom.get("author"),
                    channel=doc_custom.get("channel"),
                    tags=doc_custom.get("tags", []),
                    confidence=1.0,
                    metadata=dict(doc_custom),
                    chunker_info={
                        **dict(raw_chunk.metadata),
                        "chunk_index": chunk_idx,
                        "start_char": (raw_chunk.start_char if hasattr(raw_chunk, "start_char") else 0),
                        "end_char": (raw_chunk.end_char if hasattr(raw_chunk, "end_char") else len(raw_chunk.content)),
                        "token_count": (raw_chunk.token_count if hasattr(raw_chunk, "token_count") else 0),
                    },
                    **document_denorm_fields(doc),
                )
                temporal_chunks.append(temporal_chunk)

            stored_chunks: list[Any] = []
            if temporal_chunks:
                with trace_span("khora.skeleton.batch_store") as span:
                    stored_chunks = await temporal_store.create_chunks_batch(temporal_chunks)
                    span.set_attribute("stored_count", len(stored_chunks))

            timings["store_ms"] = (time.perf_counter() - start) * 1000
            logger.debug(f"Stage 3 store: {len(stored_chunks)} chunks in {timings['store_ms']:.1f}ms")

            # ── Stage 4: Update document statuses ─────────────────────────
            start = time.perf_counter()

            # Count chunks per document for status update
            chunks_per_doc: dict[int, int] = {}
            for doc_idx, _ in chunk_map:
                chunks_per_doc[doc_idx] = chunks_per_doc.get(doc_idx, 0) + 1

            processed = 0
            for doc_idx, doc in enumerate(created_docs):
                if isinstance(all_raw_chunks[doc_idx], BaseException):
                    continue
                chunk_count = chunks_per_doc.get(doc_idx, 0)
                doc.mark_completed(chunk_count, 0)
                try:
                    await storage.update_document(doc)
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to update document {doc.id}: {e}")
                    failed += 1
                _report_progress()

            timings["status_update_ms"] = (time.perf_counter() - start) * 1000
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000

            total_chunks = len(stored_chunks)

            # Calculate throughput metrics
            if processed > 0 and timings["total_ms"] > 0:
                timings["docs_per_second"] = processed / (timings["total_ms"] / 1000)
                timings["avg_doc_ms"] = timings["total_ms"] / processed
                timings["chunks_per_second"] = total_chunks / (timings["total_ms"] / 1000)

            logger.info(
                f"remember_batch() completed: {processed}/{total} docs, "
                f"{total_chunks} chunks in {timings['total_ms']:.1f}ms "
                f"({timings.get('docs_per_second', 0):.1f} docs/sec)"
            )

            return BatchResult(
                total=total,
                processed=processed,
                skipped=skipped,
                failed=failed,
                chunks=total_chunks,
                entities=0,
                relationships=0,
                metadata={"timings": timings},
            )
        finally:
            if bulk_mode:
                from khora.storage.optimize import ensure_hnsw_indexes

                await ensure_hnsw_indexes(storage)

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
    # Entity Operations (minimal for Khora engine)
    # =========================================================================

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id``."""
        return await self._get_storage().get_entity(entity_id, namespace_id=namespace_id)

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
        """Not supported by Skeleton Construction engine.

        The Skeleton engine focuses on temporal chunk retrieval without
        maintaining entity graphs. Use VectorCypher for entity graph
        traversal.
        """
        raise NotImplementedError(
            "find_related_entities is not supported by the Skeleton Construction engine. "
            "Use VectorCypher for entity graph operations."
        )

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id`` (IDOR family)."""
        return await self._get_storage().get_document(document_id, namespace_id=namespace_id)

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
        """Not supported by Skeleton Construction engine.

        The Skeleton engine focuses on temporal chunk retrieval without
        maintaining entity graphs. Use VectorCypher for entity search.
        """
        raise NotImplementedError(
            "search_entities is not supported by the Skeleton Construction engine. "
            "Use VectorCypher for entity graph operations."
        )

    async def stats(self, namespace_id: UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace.

        Skeleton writes chunks to the temporal store's ``khora_chunks`` table,
        NOT the relational ``chunks`` table, so chunk counting is routed to the
        temporal store's ``count_chunks`` method when available (#1070).
        Entity/relationship counts are always 0 (Skeleton skips graph extraction).
        """
        storage = self._get_storage()

        doc_count = 0
        last_activity_at = None

        try:
            doc_count, last_activity_at = await storage.get_document_stats(namespace_id)
        except (AttributeError, NotImplementedError):
            pass

        # Route chunk count through the temporal store when it exposes count_chunks.
        # Skeleton writes to ``khora_chunks``, not the relational ``chunks`` table,
        # so the coordinator's count_chunks (which reads ``chunks``) returns 0 (#1070).
        chunk_count = 0
        metadata: dict[str, Any] = {}
        temporal_store = self._temporal_store
        temporal_count_fn = getattr(temporal_store, "count_chunks", None) if temporal_store else None
        if temporal_count_fn is not None:
            try:
                chunk_count = await temporal_count_fn(namespace_id)
            except Exception as exc:  # noqa: BLE001
                from khora.telemetry.metrics import metric_counter

                logger.warning("stats: skeleton.count_chunks (temporal) failed, reporting 0", exc_info=True)
                metric_counter(
                    "khora.stats.counter_failed_total",
                    description="stats() counters that could not run.",
                ).add(1, attributes={"engine": "skeleton", "counter": "chunks"})
                metadata["errors"] = [
                    {
                        "component": "skeleton.stats.count_chunks",
                        "reason": "counter_unavailable",
                        "exception": type(exc).__name__,
                        "detail": str(exc) or None,
                    }
                ]

        return Stats(
            documents=doc_count,
            chunks=chunk_count,
            entities=0,
            relationships=0,
            last_activity_at=last_activity_at,
            metadata=metadata,
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
