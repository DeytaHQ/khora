"""Chronicle engine — temporal-semantic memory for benchmark-optimized recall.

This engine is designed for high-accuracy memory retrieval benchmarks
(LongMemEval, LoCoMo, BEAM). Unlike GraphRAG, it requires no graph database —
all storage is PostgreSQL + pgvector. Unlike Skeleton, it performs full entity
extraction and 4-channel retrieval with temporal decay scoring.

Implements:
- Full ingest pipeline (chunking, embedding, entity extraction)
- 4-channel parallel retrieval: semantic + BM25 + temporal + entity co-occurrence
- Reciprocal Rank Fusion across all channels
- Temporal decay scoring (Ebbinghaus forgetting curve)
- Event decomposition (SVO tuples with datetime ranges)
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig
from khora.core.models import Chunk, Document, DocumentMetadata, Entity, MemoryNamespace
from khora.engines._storage_config import build_storage_config
from khora.extraction.embedders import LiteLLMEmbedder
from khora.memory_lake import BatchResult, RecallResult, RememberResult, Stats
from khora.query import SearchMode
from khora.query.fusion import reciprocal_rank_fusion
from khora.storage import StorageConfig, StorageCoordinator, create_storage_coordinator
from khora.telemetry import trace

if TYPE_CHECKING:
    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig


# ---------------------------------------------------------------------------
# Temporal decay helpers
# ---------------------------------------------------------------------------


def _ebbinghaus_decay(age_hours: float, *, half_life_hours: float = 168.0) -> float:
    """Compute a retention factor using an Ebbinghaus-inspired forgetting curve.

    R(t) = exp(-t / tau) where tau = half_life / ln(2).

    With the default half-life of 168 hours (7 days), a memory retains ~50 %
    strength after one week, ~25 % after two weeks, etc.

    Returns a value in (0, 1].
    """
    if age_hours <= 0:
        return 1.0
    tau = half_life_hours / math.log(2)
    return math.exp(-age_hours / tau)


def _apply_temporal_decay(
    chunks_with_scores: list[tuple[Chunk, float]],
    *,
    decay_weight: float = 0.15,
    half_life_hours: float = 168.0,
    reference_time: datetime | None = None,
) -> list[tuple[Chunk, float]]:
    """Re-score chunks by blending relevance score with temporal decay.

    final_score = (1 - decay_weight) * relevance + decay_weight * retention

    Uses Rust-accelerated ``batch_recency_scores`` from ``khora._accel``
    when available (~10x faster than per-item Python loop for large batches).
    Falls back to per-item computation otherwise.
    """
    if not chunks_with_scores or decay_weight <= 0:
        return chunks_with_scores

    now = reference_time or datetime.now(UTC)
    now_secs = now.timestamp()
    decay_days = half_life_hours / 24.0

    # Collect timestamps
    timestamps: list[float] = []
    for chunk, _score in chunks_with_scores:
        created = chunk.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        timestamps.append(created.timestamp())

    # Batch compute recency scores via Rust/NumPy/Python acceleration
    from khora._accel import batch_recency_scores

    recency_multipliers = batch_recency_scores(
        timestamps,
        now_secs,
        decay_days,
        decay_weight,
    )

    # Blend: relevance * recency_multiplier
    rescored: list[tuple[Chunk, float]] = [
        (chunk, relevance * mult) for (chunk, relevance), mult in zip(chunks_with_scores, recency_multipliers)
    ]
    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return rescored


class ChronicleEngine:
    """Chronicle engine — temporal-semantic memory for benchmark-optimized recall.

    Key features:
    - Full entity extraction via shared ingest pipeline
    - 4-channel parallel retrieval (Phase 1: semantic + BM25; temporal + entity stubbed)
    - Reciprocal Rank Fusion for multi-channel result merging
    - Ebbinghaus temporal decay scoring
    - No graph database required — PostgreSQL + pgvector only

    Usage:
        engine = ChronicleEngine(config)
        await engine.connect()

        # Or via MemoryLake facade:
        async with MemoryLake(db_url, engine="chronicle") as lake:
            await lake.remember(content, namespace=ns_id,
                entity_types=["PERSON"], relationship_types=["KNOWS"])
            result = await lake.recall("query", namespace=ns_id)
    """

    def __init__(
        self,
        config: KhoraConfig,
        *,
        storage_config: StorageConfig | None = None,
    ) -> None:
        """Initialize the Chronicle engine.

        Args:
            config: KhoraConfig instance
            storage_config: Storage configuration (derived from config if None) - deprecated
        """
        self._config = config

        # Build storage config — skip graph backend (chronicle is PostgreSQL + pgvector only)
        self._storage_config = storage_config or build_storage_config(config, skip_graph=True)

        self._storage: StorageCoordinator | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._connected = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting Chronicle engine...")

        # Create and connect storage coordinator
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

        # Initialize telemetry (no-op if KHORA_TELEMETRY_DATABASE_URL not set)
        from khora.telemetry import init_telemetry
        from khora.telemetry.config import TelemetryConfig

        telemetry_cfg = TelemetryConfig(
            database_url=self._config.telemetry_database_url,
            service_name=self._config.telemetry_service_name,
        )
        await init_telemetry(telemetry_cfg)

        self._connected = True
        logger.info("Chronicle engine connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting Chronicle engine...")

        # Shutdown telemetry
        from khora.telemetry import shutdown_telemetry

        await shutdown_telemetry()

        if self._storage:
            await self._storage.disconnect()
            self._storage = None

        self._embedder = None
        self._connected = False

        logger.info("Chronicle engine disconnected")

    def _get_storage(self) -> StorageCoordinator:
        """Get storage coordinator, raising if not connected."""
        if self._storage is None:
            raise RuntimeError("Chronicle engine not connected. Call connect() first.")
        return self._storage

    def _get_embedder(self) -> LiteLLMEmbedder:
        """Get embedder, raising if not connected."""
        if self._embedder is None:
            raise RuntimeError("Chronicle engine not connected. Call connect() first.")
        return self._embedder

    # =========================================================================
    # Core API: remember, recall, forget
    # =========================================================================

    @trace("khora.chronicle.remember")
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

        Uses the shared ingest pipeline for chunking, embedding, and entity
        extraction — identical to GraphRAG's pipeline for maximum extraction
        quality.

        Args:
            content: Content to remember
            namespace_id: Target namespace UUID
            title: Optional title
            source: Optional source identifier
            metadata: Optional metadata dict
            skill_name: Extraction skill to use
            entity_types: Entity types to extract
            relationship_types: Relationship types to extract
            expertise: Optional expertise config (ADR-022)
            extraction_config_hash: Optional hash for change detection
            chunk_strategy: Override chunking strategy for this call

        Returns:
            RememberResult with document_id and counts
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # Compute content checksum
        start = time.perf_counter()
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        timings["checksum_ms"] = (time.perf_counter() - start) * 1000

        storage = self._get_storage()

        # Dedup check
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

        # Create document record
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

        # Process through shared ingest pipeline (chunking, embedding, extraction)
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

        logger.debug(
            f"remember() completed: {result['chunks']} chunks, {result['entities']} entities, "
            f"{result['relationships']} relationships in {timings['total_ms']:.1f}ms"
        )

        return RememberResult(
            document_id=document.id,
            namespace_id=namespace_id,
            chunks_created=result["chunks"],
            entities_extracted=result["entities"],
            relationships_created=result["relationships"],
            metadata={"timings": timings},
        )

    @trace("khora.chronicle.recall")
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
        """Recall memories using 4-channel parallel retrieval with RRF fusion.

        Phase 1 channels:
          1. Semantic (vector similarity via pgvector)
          2. BM25 (PostgreSQL full-text search)
          3. Temporal — stubbed, returns empty (Phase 2)
          4. Entity — stubbed, returns empty (Phase 2)

        Results are fused via Reciprocal Rank Fusion and then re-scored
        with Ebbinghaus temporal decay.

        Args:
            query: Query text
            namespace_id: Namespace to search
            limit: Maximum results
            mode: Search mode (VECTOR, KEYWORD, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            agentic: Reserved for future multi-step search
            raw: If True, skip LLM features (temporal decay still applies)
            temporal_filter: Reserved for Phase 2 temporal filtering
            recency_bias: Override temporal decay weight (0.0-1.0)

        Returns:
            RecallResult with fused and decay-scored chunks
        """
        storage = self._get_storage()
        embedder = self._get_embedder()
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # Read Chronicle-specific config from QuerySettings (with safe defaults)
        qs = getattr(self._config, "query", None)
        _overfetch = getattr(qs, "chronicle_overfetch_multiplier", 4) if qs else 4
        _rrf_w_semantic = getattr(qs, "chronicle_rrf_semantic_weight", 1.0) if qs else 1.0
        _rrf_w_bm25 = getattr(qs, "chronicle_rrf_bm25_weight", 0.8) if qs else 0.8
        _rrf_w_temporal = getattr(qs, "chronicle_rrf_temporal_weight", 0.9) if qs else 0.9
        _rrf_w_entity = getattr(qs, "chronicle_rrf_entity_weight", 0.85) if qs else 0.85
        _cfg_decay = getattr(qs, "chronicle_decay_weight", 0.25) if qs else 0.25
        overfetch_limit = limit * _overfetch

        # ── Channel 1: Semantic (vector similarity) ──────────────────────
        semantic_results: list[tuple[Chunk, float]] = []
        if mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
            start = time.perf_counter()
            query_embedding = await embedder.embed(query)
            timings["embed_ms"] = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            try:
                semantic_results = await storage.search_similar_chunks(
                    namespace_id,
                    query_embedding,
                    limit=overfetch_limit,
                    min_similarity=min_similarity,
                )
            except RuntimeError:
                # Vector backend not configured
                logger.debug("Vector backend not available for semantic search")
            timings["semantic_ms"] = (time.perf_counter() - start) * 1000

        # ── Channel 2: BM25 (full-text search) ──────────────────────────
        bm25_results: list[tuple[Chunk, float]] = []
        if mode in (SearchMode.HYBRID, SearchMode.ALL):
            start = time.perf_counter()
            try:
                bm25_results = await storage.search_fulltext_chunks(
                    namespace_id,
                    query,
                    limit=overfetch_limit,
                )
            except RuntimeError:
                # Vector backend not configured (fulltext lives on same backend)
                logger.debug("Fulltext backend not available for BM25 search")
            timings["bm25_ms"] = (time.perf_counter() - start) * 1000

        # ── Channel 3: Temporal (time-scoped retrieval) ─────────────────
        temporal_results: list[tuple[Chunk, float]] = []
        if mode in (SearchMode.HYBRID, SearchMode.ALL):
            start = time.perf_counter()
            try:
                temporal_results = await self._temporal_channel(
                    namespace_id,
                    query,
                    query_embedding,
                    overfetch_limit,
                    temporal_filter,
                )
            except Exception:
                logger.debug("Temporal channel failed")
            timings["temporal_ms"] = (time.perf_counter() - start) * 1000

        # ── Channel 4: Entity co-occurrence ─────────────────────────────
        entity_results: list[tuple[Chunk, float]] = []
        if mode in (SearchMode.HYBRID, SearchMode.ALL) and query_embedding is not None:
            start = time.perf_counter()
            try:
                entity_results = await self._entity_channel(
                    namespace_id,
                    query,
                    query_embedding,
                    overfetch_limit,
                )
            except Exception:
                logger.debug("Entity channel failed")
            timings["entity_ms"] = (time.perf_counter() - start) * 1000

        # ── Fusion via Reciprocal Rank Fusion ────────────────────────────
        start = time.perf_counter()

        ranked_lists: dict[str, list[tuple[Any, float]]] = {}
        weights: dict[str, float] = {}

        if semantic_results:
            ranked_lists["semantic"] = semantic_results
            weights["semantic"] = _rrf_w_semantic
        if bm25_results:
            ranked_lists["bm25"] = bm25_results
            weights["bm25"] = _rrf_w_bm25
        if temporal_results:
            ranked_lists["temporal"] = temporal_results
            weights["temporal"] = _rrf_w_temporal
        if entity_results:
            ranked_lists["entity"] = entity_results
            weights["entity"] = _rrf_w_entity

        if ranked_lists:
            fused: list[tuple[Any, float]] = reciprocal_rank_fusion(
                ranked_lists,
                weights=weights,
                id_extractor=lambda chunk: chunk.id,
            )
            chunks_with_scores: list[tuple[Chunk, float]] = fused[:overfetch_limit]
        elif semantic_results:
            # Fallback: only one channel had results
            chunks_with_scores = semantic_results[:overfetch_limit]
        elif bm25_results:
            chunks_with_scores = bm25_results[:overfetch_limit]
        else:
            chunks_with_scores = []

        timings["fusion_ms"] = (time.perf_counter() - start) * 1000

        # ── Temporal decay scoring ───────────────────────────────────────
        start = time.perf_counter()
        decay_weight = recency_bias if recency_bias is not None else _cfg_decay
        chunks_with_scores = _apply_temporal_decay(
            chunks_with_scores,
            decay_weight=decay_weight,
        )
        timings["decay_ms"] = (time.perf_counter() - start) * 1000

        # Trim to requested limit
        chunks_with_scores = chunks_with_scores[:limit]

        # ── Build context text ───────────────────────────────────────────
        context_parts = [chunk.content for chunk, _score in chunks_with_scores]
        context_text = "\n\n---\n\n".join(context_parts[:limit])

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        logger.debug(
            f"recall() completed: {len(chunks_with_scores)} chunks "
            f"(semantic={len(semantic_results)}, bm25={len(bm25_results)}) "
            f"in {timings['total_ms']:.1f}ms"
        )

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            chunks=chunks_with_scores,
            entities=[],  # Phase 2: entity-level results
            context_text=context_text,
            metadata={
                "engine": "chronicle",
                "channels": {
                    "semantic": len(semantic_results),
                    "bm25": len(bm25_results),
                    "temporal": len(temporal_results),
                    "entity": len(entity_results),
                },
                "decay_weight": decay_weight,
                "timings": timings,
            },
        )

    # ------------------------------------------------------------------
    # Retrieval channels (Phase 2)
    # ------------------------------------------------------------------

    async def _temporal_channel(
        self,
        namespace_id: UUID,
        query: str,
        query_embedding: list[float] | None,
        limit: int,
        temporal_filter: Any | None,
    ) -> list[tuple[Chunk, float]]:
        """Channel 3: Time-scoped chunk retrieval.

        Searches for chunks within a temporal window and scores them by
        both semantic relevance and temporal proximity to the query's
        referenced time.
        """
        storage = self._get_storage()

        # Determine time bounds from temporal_filter or default to recent
        created_after = None
        created_before = None
        if temporal_filter is not None:
            created_after = getattr(temporal_filter, "start_time", None)
            created_before = getattr(temporal_filter, "end_time", None)

        if created_after is None and created_before is None:
            # Use configurable temporal window (0 = unlimited — let decay handle scoring)
            qs = getattr(self._config, "query", None)
            window_days = getattr(qs, "chronicle_temporal_window_days", 0.0) if qs else 0.0
            if window_days > 0:
                from datetime import timedelta

                created_after = datetime.now(UTC) - timedelta(days=window_days)

        # Use semantic search with temporal bounds
        if query_embedding is None:
            return []

        try:
            results = await storage.search_similar_chunks(
                namespace_id,
                query_embedding,
                limit=limit,
                created_after=created_after,
                created_before=created_before,
            )
        except Exception:
            return []

        # Boost chunks that are temporally closer to now (recency signal)
        now = datetime.now(UTC)
        scored = []
        for chunk, sim in results:
            chunk_time = getattr(chunk, "source_timestamp", None) or chunk.created_at
            if chunk_time:
                hours_old = max(0, (now - chunk_time).total_seconds() / 3600)
                temporal_boost = _ebbinghaus_decay(hours_old, half_life_hours=72)  # 3-day half-life
                blended = sim * 0.6 + temporal_boost * 0.4
            else:
                blended = sim
            scored.append((chunk, blended))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    async def _entity_channel(
        self,
        namespace_id: UUID,
        query: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[tuple[Chunk, float]]:
        """Channel 4: Entity co-occurrence retrieval.

        Finds entities similar to the query, then retrieves chunks that
        mention those entities. Provides a "follow the entities" signal
        complementary to pure semantic search.
        """
        storage = self._get_storage()

        # Step 1: Find entities similar to the query
        try:
            entity_results = await storage.search_similar_entities(
                namespace_id,
                query_embedding,
                limit=10,
            )
        except Exception:
            return []

        if not entity_results:
            return []

        # Step 2: Get the source chunk IDs from matching entities
        entity_ids = [eid for eid, _score in entity_results]
        entity_scores = {eid: score for eid, score in entity_results}

        try:
            entities = await storage.get_entities_batch(entity_ids)
        except Exception:
            return []

        # Collect chunk IDs from entity sources, weighted by entity similarity
        chunk_scores: dict[UUID, float] = {}
        for eid, entity in entities.items():
            escore = entity_scores.get(eid, 0.0)
            for cid in entity.source_chunk_ids:
                chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), escore)

        if not chunk_scores:
            return []

        # Step 3: Fetch the actual chunks
        chunk_ids = sorted(chunk_scores, key=lambda k: chunk_scores.get(k, 0.0), reverse=True)[:limit]
        try:
            chunks_map = await storage.get_chunks_batch(chunk_ids)
        except Exception:
            return []

        results = []
        for cid in chunk_ids:
            chunk = chunks_map.get(cid)
            if chunk:
                results.append((chunk, chunk_scores[cid]))

        return results

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

    @trace("khora.chronicle.remember_batch")
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
        """Store multiple documents via the shared ingest pipeline.

        Delegates to the same ``ingest_documents`` pipeline used by GraphRAG
        for full entity extraction, deduplication, and optional expansion.

        Args:
            documents: List of document dicts with content, title, source, metadata
            namespace_id: Target namespace UUID
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Deduplicate entities across documents
            infer_relationships: Infer relationships after ingestion
            on_progress: Callback(processed_count, total_count)
            entity_types: Entity types to extract
            relationship_types: Relationship types to extract
            expertise: Optional expertise config (ADR-022)
            extraction_config_hash: Hash for change detection
            chunk_strategy: Override chunking strategy

        Returns:
            BatchResult with aggregated statistics
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

        # Build doc inputs for ingest_documents
        start = time.perf_counter()
        doc_inputs: list[dict[str, Any]] = []
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

        # Create shared embedder
        shared_embedder = LiteLLMEmbedder(model=self._config.llm.embedding_model)

        # Optional cross-document entity deduplication
        shared_entity_index = None
        if deduplicate:
            start = time.perf_counter()
            from khora.extraction.expansion.entity_index import EntityIndex

            shared_entity_index = EntityIndex()

            existing_entities = await self._get_storage().list_entities(namespace_id, limit=50000)
            for entity in existing_entities:
                shared_entity_index.add(entity)

            timings["entity_preload_ms"] = (time.perf_counter() - start) * 1000
            if existing_entities:
                logger.debug(f"Preloaded {len(existing_entities)} existing entities into EntityIndex")

        # Determine expansion
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

        processed = result.get("processed_documents", 0)
        if processed > 0 and timings["total_ms"] > 0:
            timings["docs_per_second"] = processed / (timings["total_ms"] / 1000)
            timings["avg_doc_ms"] = timings["ingest_pipeline_ms"] / processed

        logger.info(
            f"remember_batch() completed: {processed}/{len(documents)} docs, "
            f"{result.get('total_chunks', 0)} chunks, {result.get('total_entities', 0)} entities "
            f"in {timings['total_ms']:.1f}ms ({timings.get('docs_per_second', 0):.1f} docs/sec)"
        )

        if on_progress:
            on_progress(
                result.get("processed_documents", 0),
                result.get("total_documents", len(documents)),
            )

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
    ) -> MemoryNamespace:
        """Create a new memory namespace."""
        namespace = MemoryNamespace(
            config_overrides=config_overrides or {},
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

    async def find_related_entities(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity.

        Chronicle is designed for PostgreSQL-only operation without a graph
        backend. Returns an empty list. Use GraphRAG or VectorCypher for
        graph-based entity traversal.
        """
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

    @trace("khora.chronicle.search_entities", exclude={"query"}, result=lambda r: {"result_count": len(r)})
    async def search_entities(
        self,
        query: str,
        namespace_id: UUID,
        *,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity.

        Uses batch entity fetching to avoid N+1 queries.
        """
        embedder = self._get_embedder()
        storage = self._get_storage()

        query_embedding = await embedder.embed(query)

        entity_ids_scores = await storage.search_similar_entities(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=0.0,
        )

        if not entity_ids_scores:
            return []

        entity_ids = [entity_id for entity_id, _ in entity_ids_scores]
        entities_map = await storage.get_entities_batch(entity_ids)

        return [entities_map[eid] for eid, _score in entity_ids_scores if eid in entities_map]

    # =========================================================================
    # Stats
    # =========================================================================

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
        """Check health of all components.

        Returns a dict with:
        - status: 'healthy', 'degraded', or 'disconnected'
        - engine: Engine name
        - checks: Individual component results
        """
        if not self._connected:
            return {"status": "disconnected", "engine": "chronicle"}

        health: dict[str, Any] = {
            "engine": "chronicle",
            "status": "healthy",
            "checks": {},
        }

        # Check storage (PostgreSQL + pgvector)
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
                health["checks"]["embedder"] = "ok"
            else:
                health["checks"]["embedder"] = "not configured"
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["embedder"] = f"error: {e}"
            health["status"] = "degraded"

        return health
