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

import asyncio
import hashlib
import math
import re
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


# ---------------------------------------------------------------------------
# Version-aware scoring helpers
# ---------------------------------------------------------------------------

# Patterns that signal the user wants the latest/current state
_VERSION_INTENT_PATTERNS = re.compile(
    r"\b(current|currently|latest|newest|most\s+recent|up[- ]?to[- ]?date|now|today|present"
    r"|status|state|active|existing)\b",
    re.IGNORECASE,
)


def _has_version_intent(query: str) -> bool:
    """Return True when the query signals interest in the latest version/state."""
    return _VERSION_INTENT_PATTERNS.search(query) is not None


def _apply_version_scoring(
    chunks_with_scores: list[tuple[Chunk, float]],
    query: str,
) -> list[tuple[Chunk, float]]:
    """Penalize superseded document versions so the latest state floats up.

    Only activates when:
      - The query contains temporal/state intent keywords
      - At least one chunk carries ``version`` metadata

    Chunks are grouped by entity (first ``entity_refs`` entry, falling back
    to document title stored in ``chunk.metadata.custom``).  Within each
    group the maximum version is identified, and older versions receive a
    soft penalty::

        score *= (version / max_version) ** 0.5

    The square-root exponent ensures old versions are demoted but not
    aggressively filtered out.
    """
    if not chunks_with_scores:
        return chunks_with_scores

    # Gate: only apply when the query signals latest/current intent
    if not _has_version_intent(query):
        return chunks_with_scores

    # Collect version info — bail quickly if no chunk has a version
    has_any_version = False
    chunk_meta: list[tuple[str, int | None]] = []  # (group_key, version)
    for chunk, _score in chunks_with_scores:
        custom = chunk.metadata.custom if chunk.metadata else {}
        version = custom.get("version")
        if version is not None:
            has_any_version = True
            try:
                version = int(version)
            except (TypeError, ValueError):
                version = None

        # Group key: first entity_ref, or title, or document_id
        entity_refs = custom.get("entity_refs")
        if entity_refs and isinstance(entity_refs, list) and len(entity_refs) > 0:
            group_key = str(entity_refs[0])
        else:
            group_key = custom.get("title", str(chunk.document_id))

        chunk_meta.append((group_key, version))

    if not has_any_version:
        return chunks_with_scores

    # Build max-version lookup per entity group
    max_version: dict[str, int] = {}
    for group_key, version in chunk_meta:
        if version is not None:
            if group_key not in max_version or version > max_version[group_key]:
                max_version[group_key] = version

    # Re-score
    rescored: list[tuple[Chunk, float]] = []
    for (chunk, score), (group_key, version) in zip(chunks_with_scores, chunk_meta):
        if version is not None and group_key in max_version and max_version[group_key] > 0:
            penalty = (version / max_version[group_key]) ** 0.5
            score = score * penalty
        rescored.append((chunk, score))

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
        _cfg_half_life = getattr(qs, "temporal_half_life_hours", 168.0) if qs else 168.0
        overfetch_limit = limit * _overfetch

        # Resolve relative dates ("last 7 days") to SQL-pushdown filter
        # when the caller didn't supply an explicit temporal_filter.
        if temporal_filter is None:
            from khora.query.temporal_detection import TemporalDetector
            from khora.query.temporal_resolver import resolve_temporal_filter, to_query_temporal_filter

            _detector = TemporalDetector()
            _signal = _detector.detect(query)
            if _signal.is_temporal:
                _skeleton_filter = resolve_temporal_filter(query, _signal)
                if _skeleton_filter is not None:
                    temporal_filter = to_query_temporal_filter(_skeleton_filter)

        # ── Phase 1: Embed query + BM25 in parallel ───────────────────
        # BM25 needs only the query text (no embedding), so start it
        # concurrently with embedding to save one round-trip of latency.
        query_embedding: list[float] | None = None
        bm25_results: list[tuple[Chunk, float]] = []

        bm25_task: asyncio.Task[list[tuple[Chunk, float]]] | None = None
        if mode in (SearchMode.HYBRID, SearchMode.ALL):
            bm25_task = asyncio.create_task(
                storage.search_fulltext_chunks(
                    namespace_id,
                    query,
                    limit=overfetch_limit,
                )
            )

        if mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
            start = time.perf_counter()
            query_embedding = await embedder.embed(query)
            timings["embed_ms"] = (time.perf_counter() - start) * 1000

        # Collect BM25 result (already running in background)
        if bm25_task is not None:
            start = time.perf_counter()
            try:
                bm25_results = await bm25_task
            except RuntimeError:
                logger.debug("Fulltext backend not available for BM25 search")
            except Exception:
                logger.debug("BM25 channel failed")
            timings["bm25_ms"] = (time.perf_counter() - start) * 1000

        # ── Phase 2: Semantic + Temporal + Entity in parallel ───────────
        # All three need the embedding, so they run after Phase 1 completes.
        semantic_results: list[tuple[Chunk, float]] = []
        temporal_results: list[tuple[Chunk, float]] = []
        entity_results: list[tuple[Chunk, float]] = []

        # chronicle_temporal_window_days semantics:
        #   -1  = disable temporal channel entirely
        #    0  = unlimited window (search ALL data with recency-primary scoring)
        #   >0  = N-day window filter
        _temporal_window_days = getattr(qs, "chronicle_temporal_window_days", 0.0) if qs else 0.0
        run_temporal = mode in (SearchMode.HYBRID, SearchMode.ALL) and (
            _temporal_window_days >= 0 or temporal_filter is not None
        )

        channel_coros: list[tuple[str, Any]] = []

        if mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL) and query_embedding is not None:
            channel_coros.append(
                (
                    "semantic",
                    storage.search_similar_chunks(
                        namespace_id,
                        query_embedding,
                        limit=overfetch_limit,
                        min_similarity=min_similarity,
                    ),
                )
            )

        if run_temporal and query_embedding is not None:
            channel_coros.append(
                (
                    "temporal",
                    self._temporal_channel(
                        namespace_id,
                        query,
                        query_embedding,
                        overfetch_limit,
                        temporal_filter,
                    ),
                )
            )

        if mode in (SearchMode.HYBRID, SearchMode.ALL) and query_embedding is not None:
            channel_coros.append(
                (
                    "entity",
                    self._entity_channel(
                        namespace_id,
                        query,
                        query_embedding,
                        overfetch_limit,
                    ),
                )
            )

        if channel_coros:
            start = time.perf_counter()
            gathered = await asyncio.gather(
                *[coro for _name, coro in channel_coros],
                return_exceptions=True,
            )
            elapsed = (time.perf_counter() - start) * 1000

            for (name, _coro), result in zip(channel_coros, gathered):
                if isinstance(result, BaseException):
                    logger.debug(f"{name.capitalize()} channel failed: {result}")
                    timings[f"{name}_ms"] = elapsed
                    continue
                timings[f"{name}_ms"] = elapsed
                if name == "semantic":
                    semantic_results = result
                elif name == "temporal":
                    temporal_results = result
                elif name == "entity":
                    entity_results = result

        # Capture max raw cosine similarity for abstention signals
        max_raw_cosine = max((score for _, score in semantic_results), default=0.0) if semantic_results else 0.0

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
            half_life_hours=_cfg_half_life,
        )
        timings["decay_ms"] = (time.perf_counter() - start) * 1000

        # ── Version-aware scoring ───────────────────────────────────────
        # Penalize superseded document versions so the latest state
        # surfaces first.  Only fires when the query has temporal/state
        # intent and chunks carry version metadata.
        start = time.perf_counter()
        chunks_with_scores = _apply_version_scoring(chunks_with_scores, query)
        timings["version_scoring_ms"] = (time.perf_counter() - start) * 1000

        # ── Cross-encoder reranking (post-fusion) ───────────────────────
        _enable_reranking = getattr(qs, "enable_reranking", False) if qs else False
        if _enable_reranking and chunks_with_scores:
            start = time.perf_counter()
            _reranking_model = getattr(qs, "reranking_model", None) if qs else None
            _reranking_top_n = getattr(qs, "reranking_top_n", 30) if qs else 30
            try:
                from khora.query.reranking import rerank_chunks

                reranked = await rerank_chunks(
                    query,
                    chunks_with_scores[:_reranking_top_n],
                    method="cross_encoder",
                    top_k=limit,
                    model=_reranking_model,
                )
                chunks_with_scores = reranked
            except Exception as e:
                logger.warning("Chronicle cross-encoder reranking failed: %s", e)
            timings["reranking_ms"] = (time.perf_counter() - start) * 1000

        # Trim to requested limit
        chunks_with_scores = chunks_with_scores[:limit]

        # ── Build context text ───────────────────────────────────────────
        context_parts = [chunk.content for chunk, _score in chunks_with_scores]
        context_text = "\n\n---\n\n".join(context_parts[:limit])

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000

        logger.debug(
            f"recall() completed: {len(chunks_with_scores)} chunks "
            f"(semantic={len(semantic_results)}, bm25={len(bm25_results)}, "
            f"temporal={len(temporal_results)}, entity={len(entity_results)}) "
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
                "max_raw_vector_score": max_raw_cosine,
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

        # Detect timestamp collapse: if all chunks were created within ~1 hour
        # (e.g., benchmark batch ingestion), recency scoring is pure noise.
        # Fall back to semantic-only scores so the channel doesn't pollute RRF.
        if results:
            times = []
            for chunk, _ in results:
                ct = getattr(chunk, "source_timestamp", None) or chunk.created_at
                if ct:
                    if ct.tzinfo is None:
                        ct = ct.replace(tzinfo=UTC)
                    times.append(ct.timestamp())
            if len(times) > 1:
                import statistics

                if statistics.stdev(times) < 3600:  # < 1 hour spread
                    return results  # Pure semantic scores

        # Balanced scoring: 60% semantic, 40% recency — gives RRF
        # a meaningfully different ranking without drowning out relevance.
        now = datetime.now(UTC)
        scored = []
        for chunk, sim in results:
            chunk_time = getattr(chunk, "source_timestamp", None) or chunk.created_at
            if chunk_time:
                if chunk_time.tzinfo is None:
                    chunk_time = chunk_time.replace(tzinfo=UTC)
                hours_old = max(0, (now - chunk_time).total_seconds() / 3600)
                recency_factor = _ebbinghaus_decay(hours_old, half_life_hours=72)  # 3-day half-life
                blended = sim * 0.6 + recency_factor * 0.4
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
        except Exception as e:
            logger.warning("Entity channel: search_similar_entities failed: %s", e)
            return []

        if not entity_results:
            logger.debug("Entity channel: no similar entities found")
            return []

        logger.debug("Entity channel: found %d similar entities", len(entity_results))

        # Step 2: Get the source chunk IDs from matching entities
        entity_ids = [eid for eid, _score in entity_results]
        entity_scores = {eid: score for eid, score in entity_results}

        try:
            entities = await storage.get_entities_batch(entity_ids)
        except Exception as e:
            logger.warning("Entity channel: get_entities_batch failed for %d IDs: %s", len(entity_ids), e)
            return []

        logger.debug("Entity channel: resolved %d/%d entities", len(entities), len(entity_ids))

        # Collect chunk IDs from entity sources, weighted by entity similarity
        chunk_scores: dict[UUID, float] = {}
        for eid, entity in entities.items():
            escore = entity_scores.get(eid, 0.0)
            for cid in entity.source_chunk_ids:
                chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), escore)

        if not chunk_scores:
            logger.debug("Entity channel: no source chunks from matched entities")
            return []

        # Step 3: Fetch the actual chunks
        chunk_ids = sorted(chunk_scores, key=lambda k: chunk_scores.get(k, 0.0), reverse=True)[:limit]
        try:
            chunks_map = await storage.get_chunks_batch(chunk_ids)
        except Exception as e:
            logger.warning("Entity channel: get_chunks_batch failed for %d IDs: %s", len(chunk_ids), e)
            return []

        # Semantic relevance gate: filter out entity-adjacent chunks that
        # share entity mentions but are semantically irrelevant to the query.
        results = []
        for cid in chunk_ids:
            chunk = chunks_map.get(cid)
            if not chunk:
                continue
            if query_embedding is not None and chunk.embedding is not None:
                # Compute cosine similarity between query and chunk
                try:
                    dot = sum(a * b for a, b in zip(query_embedding, chunk.embedding))
                    norm_q = sum(a * a for a in query_embedding) ** 0.5
                    norm_c = sum(a * a for a in chunk.embedding) ** 0.5
                    sim = dot / (norm_q * norm_c) if norm_q > 0 and norm_c > 0 else 0.0
                    if sim < 0.3:
                        continue  # Below relevance threshold
                    results.append((chunk, chunk_scores[cid] * sim))
                except Exception:
                    results.append((chunk, chunk_scores[cid]))
            else:
                results.append((chunk, chunk_scores[cid]))

        logger.debug("Entity channel: returning %d chunks (after relevance gate)", len(results))
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
        extraction_batch_size: int | None = None,
        extraction_max_tokens: int | None = None,
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
            extraction_batch_size: Max texts per LLM extraction call (None = pipeline default)
            extraction_max_tokens: Max tokens for extraction LLM calls (None = pipeline default)

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
            extraction_timeout=self._config.llm.timeout,
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
        if extraction_batch_size is not None:
            ingest_kwargs["extraction_batch_size"] = extraction_batch_size
        if extraction_max_tokens is not None:
            ingest_kwargs["extraction_max_tokens"] = extraction_max_tokens
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
