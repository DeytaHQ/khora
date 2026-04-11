"""VectorCypher retriever - hybrid vector+graph retrieval.

Implements the VectorCypher retrieval pipeline:
1. Vector search to find entry entities (pgvector)
2. Cypher traversal to expand relationships (Neo4j)
3. Chunk retrieval via MENTIONED_IN relationships
4. RRF fusion to combine vector and graph scores

Performance optimizations:
- Parallel execution of independent operations (vector chunk search + entity path)
- Batch entity neighborhood fetching via UNWIND
- Normalized score fusion for better ranking
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger
from neo4j.exceptions import Neo4jError

from khora.core.models import Chunk, ChunkMetadata, Entity, Relationship
from khora.telemetry import trace_span

from .dual_nodes import DualNodeManager
from .fusion import (
    FusedResult,
    apply_coherence_boost,
    apply_recency_boost,
    normalize_scores,
    weighted_rrf,
    weighted_rrf_normalized,
)
from .router import QueryComplexity, QueryComplexityRouter, RouterConfig, RoutingDecision
from .temporal_detection import (
    RETRIEVAL_PARAMS,
    RetrievalParams,
    TemporalCategory,
    TemporalSignal,
    get_retrieval_params,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from khora.engines.skeleton.backends import TemporalFilter, TemporalVectorStore
    from khora.extraction.embedders import EmbedderProtocol  # type: ignore[unresolved-import]
    from khora.query.reranking import CrossEncoderReranker, LLMReranker
    from khora.storage import StorageCoordinator


@dataclass
class VectorCypherResult:
    """Result from VectorCypher retrieval."""

    chunks: list[tuple[Chunk, float]]
    entities: list[tuple[Entity, float]]
    routing_decision: RoutingDecision
    relationships: list[tuple[Relationship, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""

    # Graph traversal settings
    default_depth: int = 2
    max_depth: int = 4
    max_entry_entities: int = 10

    # Adaptive depth settings
    adaptive_depth_enabled: bool = True
    adaptive_depth_high_entity_threshold: int = 10  # Shallow depth if >= this many entities
    adaptive_depth_low_entity_threshold: int = 2  # Deeper depth if <= this many entities

    # Fusion settings
    rrf_k: int = 60
    vector_weight: float = 0.6
    graph_weight: float = 0.4

    # Per-complexity fusion overrides (used when routing is enabled)
    simple_vector_weight: float = 0.8
    simple_graph_weight: float = 0.2
    complex_vector_weight: float = 0.4
    complex_graph_weight: float = 0.6

    # Temporal fusion overrides (used when temporal signal is detected)
    temporal_vector_weight: float = 0.3
    temporal_graph_weight: float = 0.7

    # Temporal settings
    recency_weight: float = 0.2
    recency_decay_days: int = 30
    recency_decay_type: str = "exponential"  # "linear" or "exponential"

    # Coherence scoring (penalizes word-shuffled confounders)
    coherence_weight: float = 0.1

    # Search thresholds
    min_entity_similarity: float = 0.3
    hybrid_alpha: float = 0.7

    # Query caching
    query_cache_ttl_seconds: int = 0  # 0 = disabled
    query_cache_max_size: int = 100

    # Lazy entity expansion
    lazy_entity_expansion: bool = False
    skeleton_core_ratio: float = 0.70  # Skip lazy expansion when > 0.6

    # BM25 channel (independent full-text search alongside vector + graph)
    enable_bm25_channel: bool = False
    bm25_weight: float = 0.3
    bm25_top_k: int = 50  # How many BM25 results to fetch

    # Cross-encoder reranking
    enable_reranking: bool = False
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranking_top_n: int = 50  # Candidates to feed to the cross-encoder
    reranking_blend_weight: float = 0.7  # Rerank vs original score blend

    # LLM reranking (applied after cross-encoder, only for temporal queries)
    enable_llm_reranking: bool = False
    llm_reranking_model: str = "gpt-4o-mini"
    llm_reranking_top_n: int = 5
    llm_reranking_confidence_threshold: float = 0.1  # Skip LLM reranking when cross-encoder gap >= this

    # Session-aware parallel retrieval for cross-session temporal queries.
    # When enabled AND the query is temporal AND entry entities span multiple
    # sessions, fans out parallel per-session vector searches instead of a
    # single global search.  Improves session_crossing_recall.
    enable_session_aware_search: bool = False

    # Limits
    max_chunks: int = 50
    max_entities: int = 30
    max_relationships: int = 90  # ~3x max_entities


def _extract_occurred_at(item: Any) -> str | None:
    """Extract occurred_at string from a Chunk or dict item."""
    if isinstance(item, Chunk):
        custom = item.metadata.custom if item.metadata else {}
        return custom.get("occurred_at")
    elif isinstance(item, dict):
        return item.get("occurred_at")
    return None


class VectorCypherRetriever:
    """Hybrid retriever combining vector search with Cypher graph traversal.

    The retrieval pipeline:
    1. Route query to determine search strategy
    2. Vector search for entry entities via pgvector
    3. (If complex) Expand entities via Neo4j Cypher queries
    4. Fetch chunks connected to entities via MENTIONED_IN
    5. Apply RRF fusion to combine results
    6. Apply temporal recency boost
    """

    def __init__(
        self,
        vector_store: TemporalVectorStore,
        neo4j_driver: AsyncDriver | None,
        embedder: EmbedderProtocol,
        *,
        database: str = "neo4j",
        config: RetrieverConfig | None = None,
        router_config: RouterConfig | None = None,
        storage: StorageCoordinator | None = None,
        neo4j_query_timeout: float | None = None,
    ):
        """Initialize the retriever.

        Args:
            vector_store: pgvector temporal store for chunk search
            neo4j_driver: Neo4j async driver for graph traversal
            embedder: Embedder for query embedding
            database: Neo4j database name
            config: Retriever configuration
            router_config: Router configuration (optional, for LLM routing etc.)
            storage: Storage coordinator for entity vector search via pgvector
            neo4j_query_timeout: Optional per-transaction timeout in seconds
                forwarded to the underlying ``DualNodeManager`` to bound
                ``get_entity_neighborhoods``. ``None`` disables the timeout.
        """
        self._vector_store = vector_store
        self._neo4j_driver = neo4j_driver
        self._embedder = embedder
        self._database = database
        self._config = config or RetrieverConfig()
        self._storage = storage

        # Initialize router with config, syncing adaptive depth settings
        if router_config is None:
            router_config = RouterConfig(
                adaptive_depth_enabled=self._config.adaptive_depth_enabled,
                adaptive_depth_high_entity_threshold=self._config.adaptive_depth_high_entity_threshold,
                adaptive_depth_low_entity_threshold=self._config.adaptive_depth_low_entity_threshold,
                complex_depth=self._config.default_depth,
            )
        self._router = QueryComplexityRouter(router_config)
        self._dual_nodes = (
            DualNodeManager(neo4j_driver, database, query_timeout=neo4j_query_timeout) if neo4j_driver else None
        )

        # Query result cache (LRU + TTL)
        self._cache: dict[str, tuple[float, VectorCypherResult]] = {}
        self._cache_ttl = self._config.query_cache_ttl_seconds
        self._cache_max_size = self._config.query_cache_max_size

        # Lazy entity expansion cache: chunk_id -> expansion_score (0 = no match)
        self._expansion_cache: dict[UUID, float] = {}

        # Cached cross-encoder reranker (lazy-init on first use, reused across queries)
        self._reranker: CrossEncoderReranker | None = None

        # Cached LLM reranker for temporal queries (lazy-init on first use)
        self._llm_reranker: LLMReranker | None = None

    async def retrieve(
        self,
        query: str,
        namespace_id: UUID,
        *,
        temporal_filter: TemporalFilter | None = None,
        temporal_signal: TemporalSignal | None = None,
        graph_depth: int | None = None,
        limit: int | None = None,
    ) -> VectorCypherResult:
        """Retrieve relevant chunks using VectorCypher hybrid approach.

        Args:
            query: User query
            namespace_id: Namespace to search
            temporal_filter: Optional temporal constraints
            temporal_signal: Optional temporal detection signal (drives recency/sort behavior)
            graph_depth: Override for graph traversal depth
            limit: Maximum chunks to return

        Returns:
            VectorCypherResult with chunks, entities, and metadata
        """
        with trace_span("khora.vectorcypher.retrieve", namespace_id=str(namespace_id)) as span:
            limit = limit or self._config.max_chunks

            # Resolve retrieval parameters from temporal signal
            params = (
                get_retrieval_params(temporal_signal) if temporal_signal else RETRIEVAL_PARAMS[TemporalCategory.NONE]
            )
            if temporal_signal and temporal_signal.is_temporal:
                span.set_attribute("temporal_category", temporal_signal.category.value)
                logger.debug(
                    f"Temporal signal: {temporal_signal.category.value} (confidence={temporal_signal.confidence:.2f})"
                )

            # Cache check
            cache_key = ""
            if self._cache_ttl > 0:
                cache_key = hashlib.md5(
                    f"{query}:{namespace_id}:{temporal_filter}:{graph_depth}:{limit}".encode(),
                    usedforsecurity=False,
                ).hexdigest()

                if cache_key in self._cache:
                    cached_time, cached_result = self._cache[cache_key]
                    if time.monotonic() - cached_time < self._cache_ttl:
                        cached_result.metadata["cache_hit"] = True
                        span.set_attribute("cache_hit", True)
                        return cached_result
                    else:
                        del self._cache[cache_key]

            # Step 1: Route query to determine strategy
            with trace_span("khora.vectorcypher.route") as route_span:
                routing = await self._router.route(query, temporal_signal=temporal_signal)
                route_span.set_attribute("complexity", routing.complexity.value)
                route_span.set_attribute("use_graph", routing.use_graph)
            logger.debug(f"Query routing: {routing.complexity.value} (use_graph={routing.use_graph})")
            span.set_attribute("routing_complexity", routing.complexity.value)

            # Step 2: Embed the query
            with trace_span("khora.vectorcypher.embed_query") as embed_span:
                embed_span.set_attribute("model", self._embedder.model_name)
                embed_span.set_attribute("dimension", self._embedder.dimension)
                embed_span.set_attribute("text_length", len(query))
                _stats = getattr(self._embedder, "cache_stats", None)
                _pre_hits = _stats["hits"] if isinstance(_stats, dict) else None
                query_embedding = await self._embedder.embed(query)
                if _pre_hits is not None:
                    _post_hits = self._embedder.cache_stats["hits"]
                    embed_span.set_attribute("cache_hit", _post_hits > _pre_hits)

            # Step 3: Vector search for entry points
            if routing.complexity == QueryComplexity.SIMPLE:
                # Simple path: direct chunk retrieval
                result = await self._simple_retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    namespace_id=namespace_id,
                    temporal_filter=temporal_filter,
                    limit=limit,
                    routing=routing,
                    effective_recency=params.recency_weight,
                    decay_days_override=params.decay_days_override,
                    temporal_sort=params.temporal_sort,
                    recency_floor=params.recency_floor,
                    temporal_signal=temporal_signal,
                )
            else:
                # Complex/moderate path: VectorCypher with parallel execution
                # Wrap in try/except for graceful fallback on graph failures
                try:
                    result = await self._vectorcypher_retrieve(
                        query=query,
                        query_embedding=query_embedding,
                        namespace_id=namespace_id,
                        temporal_filter=temporal_filter,
                        graph_depth=graph_depth,
                        limit=limit,
                        routing=routing,
                        temporal_params=params,
                        temporal_signal=temporal_signal,
                    )
                except Neo4jError as e:
                    logger.warning(f"Graph search failed, falling back to vector-only: {e}")
                    result = await self._vector_only_fallback(
                        query=query,
                        query_embedding=query_embedding,
                        namespace_id=namespace_id,
                        temporal_filter=temporal_filter,
                        limit=limit,
                        routing=routing,
                        effective_recency=params.recency_weight,
                        decay_days_override=params.decay_days_override,
                        temporal_sort=params.temporal_sort,
                        recency_floor=params.recency_floor,
                        temporal_signal=temporal_signal,
                    )

            # Store in cache
            if self._cache_ttl > 0 and cache_key:
                if len(self._cache) >= self._cache_max_size:
                    oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                    del self._cache[oldest_key]
                self._cache[cache_key] = (time.monotonic(), result)

            span.set_attribute("chunk_count", len(result.chunks))
            span.set_attribute("entity_count", len(result.entities))
            return result

    async def _vectorcypher_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
        graph_depth: int | None,
        limit: int,
        routing: RoutingDecision,
        *,
        temporal_params: RetrievalParams | None = None,
        temporal_signal: TemporalSignal | None = None,
    ) -> VectorCypherResult:
        """Internal VectorCypher retrieval with graph traversal.

        This is the main VectorCypher path that combines vector and graph search.
        Separated from retrieve() to enable clean fallback handling.

        Implements adaptive depth: adjusts graph traversal depth based on the
        number of entry entities found. More entities = shallower depth (to avoid
        explosion), fewer entities = deeper depth (to find more context).

        Bi-temporal versioning:
        - EXPLICIT temporal queries with a date filter narrow entities to those
          valid at the target date via ``version_valid_from``/``version_valid_to``.
        - CHANGE temporal queries traverse ``[:SUPERSEDES]`` edges to surface
          entity version history for comparison.
        """
        _tp = temporal_params or RETRIEVAL_PARAMS[TemporalCategory.NONE]
        base_depth = graph_depth or routing.graph_depth
        entry_limit = routing.suggested_entry_limit

        # OPTIMIZATION: Start vector chunk search immediately in parallel
        # This operation doesn't depend on entity search results.
        # When the BM25 channel is active, use pure vector (hybrid_alpha=1.0)
        # to avoid double-counting BM25 (once in the vector blend, once as
        # its own independent channel).
        effective_hybrid_alpha = 1.0 if self._config.enable_bm25_channel else None
        vector_chunks_task = asyncio.create_task(
            self._vector_search_chunks(
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                query_text=query,
                limit=limit,
                hybrid_alpha_override=effective_hybrid_alpha,
            )
        )

        # Launch BM25 search in parallel with vector search (independent channel)
        bm25_chunks_task: asyncio.Task[list[tuple[UUID, float, Chunk]]] | None = None
        if self._config.enable_bm25_channel and self._storage:
            bm25_chunks_task = asyncio.create_task(
                self._bm25_search_chunks(
                    query=query,
                    namespace_id=namespace_id,
                    limit=self._config.bm25_top_k,
                )
            )

        # Step 3a: Find entry entities via vector search (runs in parallel with vector_chunks_task)
        entry_entities = await self._vector_search_entities(
            query_embedding=query_embedding,
            namespace_id=namespace_id,
            limit=entry_limit,
        )

        if not entry_entities:
            logger.debug("No entry entities found, falling back to simple retrieval")
            # Cancel the parallel tasks since we're taking a different path
            vector_chunks_task.cancel()
            try:
                await vector_chunks_task
            except asyncio.CancelledError:
                pass
            if bm25_chunks_task is not None:
                bm25_chunks_task.cancel()
                try:
                    await bm25_chunks_task
                except asyncio.CancelledError:
                    pass
            return await self._simple_retrieve(
                query=query,
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                limit=limit,
                routing=routing,
                effective_recency=_tp.recency_weight,
                decay_days_override=_tp.decay_days_override,
                temporal_sort=_tp.temporal_sort,
                recency_floor=_tp.recency_floor,
                temporal_signal=temporal_signal,
            )

        # Step 3b: Session-aware parallel retrieval
        # When enabled and the query is temporal, discover which sessions the
        # entry entities belong to. If they span multiple sessions, cancel the
        # single global vector search and fan out parallel per-session searches
        # to improve session_crossing_recall.
        session_aware_activated = False
        _session_aware_chunks: list[tuple[UUID, float, Chunk]] | None = None
        if (
            self._config.enable_session_aware_search
            and self._dual_nodes is not None
            and temporal_signal
            and temporal_signal.is_temporal
            and len(entry_entities) >= 2
        ):
            with trace_span(
                "khora.vectorcypher.session_discovery",
                entity_count=len(entry_entities),
            ) as sa_span:
                try:
                    entity_channels = await self._dual_nodes.get_entity_channels(
                        entity_ids=[str(e[0]) for e in entry_entities],
                        namespace_id=str(namespace_id),
                    )
                    sa_span.set_attribute("channel_count", len(entity_channels))
                except Exception as e:
                    logger.warning(f"Session discovery failed, using global search: {e}")
                    entity_channels = []

            if len(entity_channels) >= 2:
                # Cancel the original global vector search
                vector_chunks_task.cancel()
                try:
                    await vector_chunks_task
                except asyncio.CancelledError:
                    pass

                # Fan out per-session vector searches + one unscoped fallback
                session_aware_activated = True
                per_session_limit = max(3, limit // len(entity_channels))
                logger.info(
                    f"Session-aware search: {len(entity_channels)} sessions, " f"{per_session_limit} chunks/session"
                )

                from khora.engines.skeleton.backends import TemporalFilter as _TF

                session_tasks: list[asyncio.Task[list[tuple[UUID, float, Chunk]]]] = []
                for ch in entity_channels:
                    # Build a per-session temporal filter, preserving any existing
                    # time-range constraints from the original filter.
                    if temporal_filter is not None:
                        session_tf = _TF(
                            occurred_after=temporal_filter.occurred_after,
                            occurred_before=temporal_filter.occurred_before,
                            created_after=temporal_filter.created_after,
                            created_before=temporal_filter.created_before,
                            source_system=temporal_filter.source_system,
                            author=temporal_filter.author,
                            channel=ch,
                            tags=temporal_filter.tags,
                            additional=temporal_filter.additional,
                        )
                    else:
                        session_tf = _TF(channel=ch)

                    session_tasks.append(
                        asyncio.create_task(
                            self._vector_search_chunks(
                                query_embedding=query_embedding,
                                namespace_id=namespace_id,
                                temporal_filter=session_tf,
                                query_text=query,
                                limit=per_session_limit,
                                hybrid_alpha_override=effective_hybrid_alpha,
                            )
                        )
                    )

                # Also keep one unscoped search as fallback (in case sessions
                # are incomplete or the query spans non-entity sessions)
                fallback_limit = max(3, limit // 3)
                session_tasks.append(
                    asyncio.create_task(
                        self._vector_search_chunks(
                            query_embedding=query_embedding,
                            namespace_id=namespace_id,
                            temporal_filter=temporal_filter,
                            query_text=query,
                            limit=fallback_limit,
                            hybrid_alpha_override=effective_hybrid_alpha,
                        )
                    )
                )

                # Gather all per-session results
                all_session_results = await asyncio.gather(*session_tasks, return_exceptions=True)

                # Merge and deduplicate by chunk_id, keeping the best score
                merged: dict[UUID, tuple[UUID, float, Chunk]] = {}
                for i, result in enumerate(all_session_results):
                    if isinstance(result, Exception):
                        ch_label = entity_channels[i] if i < len(entity_channels) else "fallback"
                        logger.warning(f"Session search failed for channel={ch_label}: {result}")
                        continue
                    for chunk_id, score, chunk in result:
                        if chunk_id not in merged or score > merged[chunk_id][1]:
                            merged[chunk_id] = (chunk_id, score, chunk)

                # Store merged results; we'll use _session_aware_chunks instead
                # of awaiting vector_chunks_task in Step 6.
                _session_aware_chunks = list(merged.values())
                logger.info(
                    f"Session-aware search merged {len(merged)} unique chunks "
                    f"from {len(entity_channels)} sessions + fallback"
                )

        # Compute adaptive depth based on entry entity count
        # This prevents explosion when many entities are found
        depth = self._router.compute_adaptive_depth(
            entry_entity_count=len(entry_entities),
            base_depth=base_depth,
        )

        # Step 4: Cypher expand to find related entities
        # For temporal queries (STATE_QUERY/RECENCY/CHANGE), prefer currently-valid
        # entities by filtering out those whose valid_until has passed.
        expanded_entities, entity_info_map = await self._cypher_expand(
            entry_entity_ids=[e[0] for e in entry_entities],
            namespace_id=namespace_id,
            depth=depth,
            prefer_current=_tp.temporal_sort,
        )

        # Step 4b: Bi-temporal version filtering
        # For EXPLICIT temporal queries with a parsed date, narrow entities to
        # those whose version was valid at the target date.
        version_history: list[dict[str, Any]] | None = None
        all_entity_ids = list({e[0] for e in entry_entities} | expanded_entities.keys())

        if temporal_signal and temporal_signal.is_temporal:
            if temporal_signal.category == TemporalCategory.EXPLICIT and temporal_signal.temporal_filter is not None:
                # Derive a target date from the temporal filter
                tf = temporal_signal.temporal_filter
                target_date = getattr(tf, "occurred_before", None) or getattr(tf, "occurred_after", None)
                if target_date is not None:
                    with trace_span("khora.vectorcypher.version_filter", target_date=target_date.isoformat()):
                        all_entity_ids = await self._version_filter_entities(
                            entity_ids=all_entity_ids,
                            namespace_id=namespace_id,
                            target_date=target_date,
                        )

            elif temporal_signal.category == TemporalCategory.CHANGE:
                # For CHANGE queries, fetch version history via SUPERSEDES edges
                with trace_span("khora.vectorcypher.version_history", entity_count=len(all_entity_ids)):
                    version_history = await self._fetch_version_history(
                        entity_ids=all_entity_ids,
                        namespace_id=namespace_id,
                    )

        # Step 5: Fetch chunks from all entities
        graph_chunks = await self._fetch_chunks_from_entities(
            entity_ids=all_entity_ids,
            namespace_id=namespace_id,
            temporal_filter=temporal_filter,
            limit=limit * 2,  # Fetch more for fusion
            temporal_sort=_tp.temporal_sort,
            prefer_current=_tp.temporal_sort,
        )

        # Step 6: Wait for parallel vector chunk search to complete
        # This was started at the beginning and may already be done.
        # If session-aware search produced results, use those instead.
        if _session_aware_chunks is not None:
            vector_chunks = _session_aware_chunks
            # The original task was already cancelled; no need to await.
        else:
            vector_chunks = await vector_chunks_task

        # Fallback: if temporal filter was too restrictive, re-run without it.
        # SKIP fallback when the temporal signal is EXPLICIT with a parsed date —
        # sparse results are the correct signal (the data may not exist for that
        # time window, which is important for abstention on unanswerable queries).
        is_explicit_with_date = (
            temporal_signal
            and temporal_signal.category == TemporalCategory.EXPLICIT
            and temporal_signal.temporal_filter is not None
        )
        if temporal_filter and len(vector_chunks) < limit // 2 and not is_explicit_with_date:
            logger.debug(f"Temporal filter too restrictive ({len(vector_chunks)} results), falling back to unfiltered")
            vector_chunks = await self._vector_search_chunks(
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=None,
                query_text=query,
                limit=limit,
                hybrid_alpha_override=effective_hybrid_alpha,
            )

        # Await BM25 results (also launched in parallel at the beginning)
        bm25_chunks: list[tuple[UUID, float, Chunk]] = []
        if bm25_chunks_task is not None:
            try:
                bm25_chunks = await bm25_chunks_task
            except Exception as e:
                logger.warning(f"BM25 channel failed, continuing without: {e}")

        # Step 6a: Temporal query decomposition for CHANGE queries
        # Runs a second vector search focused on the "current state" sub-query
        # to ensure both past and present evidence are retrieved. The original
        # query naturally retrieves past-state chunks ("used to", "previously"),
        # while the decomposed sub-query targets current-state chunks.
        if temporal_signal and temporal_signal.category == TemporalCategory.CHANGE and version_history:
            current_state_query = self._decompose_change_query(query)
            if current_state_query and current_state_query != query:
                with trace_span("khora.vectorcypher.change_decomposition", sub_query=current_state_query):
                    sub_embedding = await self._embedder.embed(current_state_query)
                    sub_vector_chunks = await self._vector_search_chunks(
                        query_embedding=sub_embedding,
                        namespace_id=namespace_id,
                        temporal_filter=None,  # No temporal filter — want current state
                        query_text=current_state_query,
                        limit=limit,
                    )
                    # Merge sub-query results, deduplicating by chunk ID
                    existing_ids = {c[0] for c in vector_chunks}
                    new_chunks = [c for c in sub_vector_chunks if c[0] not in existing_ids]
                    if new_chunks:
                        vector_chunks = vector_chunks + new_chunks
                        logger.debug(
                            f"CHANGE decomposition added {len(new_chunks)} chunks "
                            f"from sub-query: {current_state_query[:60]}"
                        )

        # Step 6b: Lazy entity expansion for vector-only chunks
        # Recovers graph coverage lost from low skeleton_core_ratio by doing
        # lightweight keyword matching (no LLM) on chunks without MENTIONED_IN edges
        if self._config.lazy_entity_expansion and vector_chunks and self._config.skeleton_core_ratio <= 0.6:
            graph_chunk_ids = {c[0] for c in graph_chunks}
            vector_only = [c for c in vector_chunks if c[0] not in graph_chunk_ids]
            if vector_only:
                expanded = self._lazy_expand_chunks(vector_only, entry_entities, entity_info_map)
                if expanded:
                    graph_chunks = graph_chunks + expanded
                    logger.debug(f"Lazy expansion added {len(expanded)} chunks to graph results")

        # Step 7: RRF fusion with score normalization and dynamic weights
        fused_results = self._fuse_results(
            vector_chunks=vector_chunks,
            graph_chunks=graph_chunks,
            bm25_chunks=bm25_chunks if bm25_chunks else None,
            use_normalization=True,
            routing=routing,
            is_temporal=_tp.recency_weight > 0.2,
        )

        # Step 8: Apply recency boost driven by temporal signal category
        effective_recency = _tp.recency_weight
        # WS4: Also boost when explicit temporal filter is active
        if temporal_filter is not None and effective_recency > 0:
            effective_recency = max(effective_recency, 0.4)
        if effective_recency > 0:
            with trace_span("khora.vectorcypher.recency_boost", chunk_count=len(fused_results)):
                recency_scores = self._calculate_recency_scores(
                    fused_results, decay_days_override=_tp.decay_days_override
                )
                fused_results = apply_recency_boost(
                    fused_results,
                    recency_scores,
                    recency_weight=effective_recency,
                    recency_floor=_tp.recency_floor,
                )

        # Step 8b: Apply coherence scoring to penalize word-shuffled confounders
        if self._config.coherence_weight > 0:
            with trace_span("khora.vectorcypher.coherence_boost", chunk_count=len(fused_results)):
                fused_results = apply_coherence_boost(
                    fused_results,
                    coherence_weight=self._config.coherence_weight,
                )

        # Step 8c: Cross-encoder reranking (after boosts, before version scoring)
        if self._config.enable_reranking:
            with trace_span("khora.vectorcypher.reranking", candidate_count=len(fused_results)):
                fused_results = await self._apply_reranking(query, fused_results, limit)

        # Step 8d: LLM reranking of top-N for temporal queries (after cross-encoder)
        # Skip when cross-encoder is already confident (large gap between #1 and #2).
        if self._config.enable_llm_reranking and temporal_signal and temporal_signal.is_temporal:
            _skip_llm = False
            if len(fused_results) >= 2:
                gap = fused_results[0].rrf_score - fused_results[1].rrf_score
                if gap >= self._config.llm_reranking_confidence_threshold:
                    _skip_llm = True
                    logger.debug(
                        f"Skipping LLM reranking: cross-encoder gap {gap:.4f} >= "
                        f"threshold {self._config.llm_reranking_confidence_threshold}"
                    )
            if not _skip_llm:
                with trace_span("khora.vectorcypher.llm_reranking", candidate_count=len(fused_results)):
                    fused_results = await self._apply_llm_reranking(query, fused_results, limit)

        # Step 8e: Version-aware scoring — the FINAL score adjustment.
        # Applied after ALL reranking (cross-encoder + LLM) so nothing can
        # undo the version preference. The LLM reranker provides valuable
        # content understanding but has no temporal awareness; version scoring
        # layers recency preference on top of the LLM's relevance baseline.
        # CHANGE excluded: needs both old and new versions for comparison.
        # ORDINAL excluded: needs full version history for ordering.
        if temporal_signal and temporal_signal.category in (
            TemporalCategory.STATE_QUERY,
            TemporalCategory.RECENCY,
        ):
            from collections import defaultdict

            entity_versions: dict[str, int] = defaultdict(int)  # entity -> max version
            chunk_versions: dict[UUID, int] = {}  # chunk_id -> version

            for r in fused_results:
                meta = r.item.metadata if hasattr(r.item, "metadata") and r.item.metadata else {}
                if isinstance(meta, ChunkMetadata):
                    meta = meta.custom or {}
                if isinstance(meta, dict):
                    version = meta.get("version") or meta.get("entity_version", 0)
                    if version:
                        chunk_versions[r.item_id] = int(version)
                        for ref in meta.get("entity_refs") or []:
                            entity_versions[ref] = max(entity_versions[ref], int(version))

            if entity_versions:
                _VERSION_DECAY = 0.7  # Stronger penalty: v1/v5 → 0.44 (was 0.6 with 0.5)
                for r in fused_results:
                    v = chunk_versions.get(r.item_id, 0)
                    if v > 0:
                        meta = r.item.metadata if hasattr(r.item, "metadata") and r.item.metadata else {}
                        if isinstance(meta, ChunkMetadata):
                            meta = meta.custom or {}
                        if isinstance(meta, dict):
                            for ref in meta.get("entity_refs") or []:
                                max_v = entity_versions.get(ref, v)
                                if max_v > v:
                                    ratio = v / max_v
                                    r.rrf_score *= 1.0 - _VERSION_DECAY * (1.0 - ratio)
                                    break

        # Normalize scores to [0,1] — ensures consistent score scale for
        # downstream consumers (abstention detection, adapter score reporting).
        # This is a single normalization of the final fused+boosted scores,
        # matching what _simple_retrieve does at its exit.
        fused_results = normalize_scores(fused_results)

        # Build result
        chunk_results = [(r.item, r.rrf_score) for r in fused_results[:limit]]

        # Classify each chunk by which search method(s) found it
        vector_only_ids: list[UUID] = []
        graph_only_ids: list[UUID] = []
        both_ids: list[UUID] = []
        for r in fused_results[:limit]:
            has_vector = r.vector_rank is not None
            has_graph = r.graph_rank is not None
            if has_vector and has_graph:
                both_ids.append(r.item_id)
            elif has_vector:
                vector_only_ids.append(r.item_id)
            elif has_graph:
                graph_only_ids.append(r.item_id)

        vector_ids = vector_only_ids + both_ids
        graph_ids = graph_only_ids + both_ids

        # Entity IDs are discovered via vector similarity then expanded via graph,
        # so they are attributed to "graph" (the graph expansion is what surfaces them)
        entity_ids_str = [str(eid) for eid, _ in entry_entities[: self._config.max_entities]]

        search_methods = {
            "chunk_overlap": {
                "vector_only": {"ids": [str(id) for id in vector_only_ids], "count": len(vector_only_ids)},
                "graph_only": {"ids": [str(id) for id in graph_only_ids], "count": len(graph_only_ids)},
                "vector_and_graph": {"ids": [str(id) for id in both_ids], "count": len(both_ids)},
            },
            "entity_overlap": {
                "vector_and_graph": {"ids": entity_ids_str, "count": len(entity_ids_str)},
                "vector_only": {"ids": [], "count": 0},
                "graph_only": {"ids": [], "count": 0},
            },
            "by_method": {
                "vector": {"chunk_ids": [str(id) for id in vector_ids], "count": len(vector_ids)},
                "graph": {"chunk_ids": [str(id) for id in graph_ids], "count": len(graph_ids)},
            },
        }

        # Collect all entity IDs: entry entities (score 1.0) + expanded (score from graph distance).
        # Entry entities come first (higher relevance), expanded follow sorted by score desc.
        all_entity_scores: list[tuple[UUID, float]] = []
        seen_ids: set[UUID] = set()
        for eid, score in entry_entities:
            if eid not in seen_ids:
                all_entity_scores.append((eid, score))
                seen_ids.add(eid)
        for eid, score in sorted(expanded_entities.items(), key=lambda x: x[1], reverse=True):
            if eid not in seen_ids:
                all_entity_scores.append((eid, score))
                seen_ids.add(eid)

        # Cap total entities to max_entities
        all_entity_scores = all_entity_scores[: self._config.max_entities]

        # OPTIMIZATION: Fire entity batch-fetch (PostgreSQL) and relationship
        # fetch (Neo4j) in parallel — they hit different databases and both
        # only need the final entity ID list computed above.
        entity_ids_to_fetch = [eid for eid, _ in all_entity_scores]
        entity_ids_str = [str(eid) for eid, _ in all_entity_scores]

        # Start relationship fetch immediately (doesn't need full Entity objects)
        if self._dual_nodes is not None:
            rels_task = asyncio.create_task(
                self._dual_nodes.get_relationships_between(
                    entity_ids_str,
                    str(namespace_id),
                    limit=self._config.max_relationships,
                )
            )
        else:
            # SurrealDB: no dual node manager, fetch via storage coordinator
            async def _fetch_rels_from_storage() -> list:
                if not self._storage or not self._storage.graph:
                    return []
                rels = []
                for eid in entity_ids_to_fetch[:10]:
                    try:
                        entity_rels = await self._storage.graph.get_entity_relationships(eid, limit=20)
                        for r in entity_rels:
                            rels.append(
                                {
                                    "id": str(r.id),
                                    "source_entity_id": str(r.source_entity_id),
                                    "target_entity_id": str(r.target_entity_id),
                                    "relationship_type": r.relationship_type,
                                    "description": r.description or "",
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Failed to fetch relationships for entity {eid}: {e}")
                return rels

            rels_task = asyncio.create_task(_fetch_rels_from_storage())

        # Batch-fetch full entities from storage in parallel
        entity_results: list[tuple[Entity, float]] = []

        if entity_ids_to_fetch and self._storage:
            try:
                entities_map = await self._storage.get_entities_batch(entity_ids_to_fetch)
                for eid, score in all_entity_scores:
                    if eid in entities_map:
                        entity_results.append((entities_map[eid], score))
                    else:
                        # Fallback: use info from graph expansion
                        info = entity_info_map.get(str(eid), {})
                        entity = Entity(
                            id=eid,
                            namespace_id=namespace_id,
                            name=info.get("name", ""),
                            entity_type=info.get("entity_type", ""),
                            description=info.get("description", ""),
                            source_tool=info.get("source_tool", ""),
                        )
                        entity_results.append((entity, score))
            except Exception as e:
                logger.warning(f"Failed to batch-fetch entities, using stubs: {e}")
                # Fall back to stub construction
                for eid, score in all_entity_scores:
                    info = entity_info_map.get(str(eid), {})
                    entity = Entity(
                        id=eid,
                        namespace_id=namespace_id,
                        name=info.get("name", ""),
                        entity_type=info.get("entity_type", ""),
                        description=info.get("description", ""),
                        source_tool=info.get("source_tool", ""),
                    )
                    entity_results.append((entity, score))
        else:
            # No storage available or no entities to fetch
            for eid, score in all_entity_scores:
                info = entity_info_map.get(str(eid), {})
                entity = Entity(
                    id=eid,
                    namespace_id=namespace_id,
                    name=info.get("name", ""),
                    entity_type=info.get("entity_type", ""),
                    description=info.get("description", ""),
                    source_tool=info.get("source_tool", ""),
                )
                entity_results.append((entity, score))

        # Await the parallel relationship fetch
        try:
            raw_rels = await rels_task
        except Exception:
            logger.warning("Relationship fetch failed, continuing without relationships", exc_info=True)
            raw_rels = []
        entity_scores_by_id: dict[UUID, float] = {entity.id: score for entity, score in entity_results}
        entity_names_by_id: dict[UUID, str] = {entity.id: entity.name for entity, _ in entity_results}
        relationships: list[tuple[Relationship, float]] = []
        for raw in raw_rels:
            src_id = UUID(raw["source_entity_id"])
            tgt_id = UUID(raw["target_entity_id"])
            rel_score = (entity_scores_by_id.get(src_id, 0.0) + entity_scores_by_id.get(tgt_id, 0.0)) / 2
            rel = Relationship(
                id=UUID(raw["id"]) if raw.get("id") else uuid4(),
                namespace_id=namespace_id,
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                relationship_type=raw.get("relationship_type", "RELATES_TO"),
                description=raw.get("description", "") or "",
                source_entity_name=entity_names_by_id.get(src_id, ""),
                target_entity_name=entity_names_by_id.get(tgt_id, ""),
                source_document_ids=[UUID(d) for d in (raw.get("source_document_ids") or [])],
                source_chunk_ids=[UUID(c) for c in (raw.get("source_chunk_ids") or [])],
                confidence=raw.get("confidence") if raw.get("confidence") is not None else 1.0,
                weight=raw.get("weight") if raw.get("weight") is not None else 1.0,
            )
            relationships.append((rel, rel_score))
        relationships.sort(key=lambda x: x[1], reverse=True)

        return VectorCypherResult(
            chunks=chunk_results,
            entities=entity_results,
            relationships=relationships,
            routing_decision=routing,
            metadata={
                "entry_entities": len(entry_entities),
                "expanded_entities": len(expanded_entities),
                "graph_depth": depth,
                "base_depth": base_depth,
                "adaptive_depth_applied": depth != base_depth,
                "total_chunks_before_fusion": len(graph_chunks) + len(vector_chunks) + len(bm25_chunks),
                "routing_confidence": routing.confidence,
                # Fusion telemetry
                "vector_chunk_count": len(vector_chunks),
                "graph_chunk_count": len(graph_chunks),
                "bm25_chunk_count": len(bm25_chunks),
                "is_temporal": _tp.recency_weight > 0.2,
                "recency_weight": _tp.recency_weight,
                "effective_recency": effective_recency,
                # Session-aware search telemetry
                "session_aware_activated": session_aware_activated,
                # Bi-temporal entity version history (populated for CHANGE queries)
                "version_history": version_history,
                # Search provenance: which method(s) found each chunk
                "search_methods": search_methods,
            },
        )

    async def _vector_only_fallback(
        self,
        query: str,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
        limit: int,
        routing: RoutingDecision,
        *,
        effective_recency: float = 0.0,
        decay_days_override: int | None = None,
        temporal_sort: bool = False,
        recency_floor: float = 0.5,
        temporal_signal: TemporalSignal | None = None,
    ) -> VectorCypherResult:
        """Fallback to vector-only search when graph operations fail.

        This provides graceful degradation when Neo4j is unavailable or
        returns errors. Results are still useful, just without graph expansion.
        """
        logger.info("Using vector-only fallback due to graph search failure")

        # Use the simple retrieval path which only needs pgvector
        result = await self._simple_retrieve(
            query=query,
            query_embedding=query_embedding,
            namespace_id=namespace_id,
            temporal_filter=temporal_filter,
            limit=limit,
            routing=routing,
            effective_recency=effective_recency,
            decay_days_override=decay_days_override,
            temporal_sort=temporal_sort,
            recency_floor=recency_floor,
            temporal_signal=temporal_signal,
        )

        # Update metadata to indicate fallback was used
        result.metadata["fallback_mode"] = "vector_only"
        result.metadata["graph_unavailable"] = True

        return result

    async def _apply_reranking(
        self,
        query: str,
        fused_results: list[FusedResult],
        limit: int,
    ) -> list[FusedResult]:
        """Apply cross-encoder reranking to fused results.

        Takes the top-N candidates (configured via reranking_top_n), scores them
        with the cross-encoder model, and returns re-ordered results. The reranker
        blends cross-encoder scores with original RRF scores using
        reranking_blend_weight.

        Falls back to original ordering on any error.

        Args:
            query: Original query text
            fused_results: Fused results after recency/coherence boosts
            limit: Final number of results to return

        Returns:
            Re-ordered FusedResult list (may be shorter than input)
        """
        if not fused_results:
            return fused_results

        from khora.query.reranking import CrossEncoderReranker, RerankCandidate

        top_n = min(self._config.reranking_top_n, len(fused_results))
        candidates_to_rerank = fused_results[:top_n]
        remainder = fused_results[top_n:]

        # Normalize original scores to [0,1] before passing to the reranker
        # so the 0.3 original_score blend is meaningful (raw RRF scores are
        # ~0.01-0.02 which would make the blend effectively zero).
        raw_scores = [r.rrf_score for r in candidates_to_rerank]
        score_min = min(raw_scores) if raw_scores else 0.0
        score_max = max(raw_scores) if raw_scores else 1.0
        score_range = score_max - score_min
        candidates = []
        for r in candidates_to_rerank:
            # Build content with optional temporal prefix for cross-encoder context
            chunk_content = r.item.content if hasattr(r.item, "content") else str(r.item)
            if hasattr(r.item, "metadata") and r.item.metadata:
                meta = r.item.metadata
                # Support both ChunkMetadata objects and plain dicts
                raw = meta.custom if hasattr(meta, "custom") else (meta if isinstance(meta, dict) else {})
                if raw:
                    prefix_parts = []
                    session_id = raw.get("session_id") or raw.get("conversation_id")
                    if session_id:
                        prefix_parts.append(f"Session: {session_id}")
                    occurred_at = raw.get("occurred_at") or raw.get("source_timestamp")
                    if occurred_at:
                        prefix_parts.append(f"Date: {str(occurred_at)[:10]}")
                    if prefix_parts:
                        chunk_content = f"[{', '.join(prefix_parts)}] {chunk_content}"
            candidates.append(
                RerankCandidate(
                    item=r,
                    original_score=(r.rrf_score - score_min) / score_range if score_range > 1e-9 else 0.5,
                    content=chunk_content,
                    metadata=r.item.metadata if hasattr(r.item, "metadata") else {},
                )
            )

        try:
            if self._reranker is None:
                self._reranker = CrossEncoderReranker(model_name=self._config.reranking_model)
            results = await self._reranker.rerank(
                query, candidates, top_k=top_n, blend_weight=self._config.reranking_blend_weight
            )

            # Map reranked scores back onto FusedResult objects
            reranked: list[FusedResult] = []
            for rr in results:
                fused = rr.item  # The original FusedResult
                fused.rrf_score = rr.final_score
                reranked.append(fused)

            # Append any remainder that wasn't reranked (already sorted by original score)
            reranked.extend(remainder)
            logger.debug(f"Cross-encoder reranking applied: {top_n} candidates scored, " f"returning top {limit}")
            return reranked[:limit]
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed, keeping original order: {e}")
            return fused_results

    async def _apply_llm_reranking(
        self,
        query: str,
        fused_results: list[FusedResult],
        limit: int,
    ) -> list[FusedResult]:
        """Apply LLM reranking to the top-N fused results for temporal queries.

        Similar to ``_apply_reranking`` but uses the LLM-based reranker which
        understands temporal context better than the cross-encoder.  Only the
        top ``llm_reranking_top_n`` candidates are sent to the LLM; the
        remainder is appended unchanged.

        Args:
            query: Original query text
            fused_results: Fused results (already cross-encoder reranked if enabled)
            limit: Final number of results to return

        Returns:
            Re-ordered FusedResult list
        """
        if not fused_results:
            return fused_results

        from khora.query.reranking import LLMReranker, RerankCandidate

        top_n = min(self._config.llm_reranking_top_n, len(fused_results))
        candidates_to_rerank = fused_results[:top_n]
        remainder = fused_results[top_n:]

        # Normalize original scores to [0,1] for blending
        raw_scores = [r.rrf_score for r in candidates_to_rerank]
        score_min = min(raw_scores) if raw_scores else 0.0
        score_max = max(raw_scores) if raw_scores else 1.0
        score_range = score_max - score_min

        candidates = []
        for r in candidates_to_rerank:
            # Build content with temporal metadata prefix
            chunk_content = r.item.content if hasattr(r.item, "content") else str(r.item)
            if hasattr(r.item, "metadata") and r.item.metadata:
                meta = r.item.metadata
                raw = meta.custom if hasattr(meta, "custom") else (meta if isinstance(meta, dict) else {})
                if raw:
                    prefix_parts = []
                    session_id = raw.get("session_id") or raw.get("conversation_id")
                    if session_id:
                        prefix_parts.append(f"Session: {session_id}")
                    occurred_at = raw.get("occurred_at") or raw.get("source_timestamp")
                    if occurred_at:
                        prefix_parts.append(f"Date: {str(occurred_at)[:10]}")
                    if prefix_parts:
                        chunk_content = f"[{', '.join(prefix_parts)}] {chunk_content}"
            candidates.append(
                RerankCandidate(
                    item=r,
                    original_score=(r.rrf_score - score_min) / score_range if score_range > 1e-9 else 0.5,
                    content=chunk_content,
                    metadata=r.item.metadata if hasattr(r.item, "metadata") else {},
                )
            )

        try:
            if self._llm_reranker is None:
                self._llm_reranker = LLMReranker(model=self._config.llm_reranking_model)
            results = await self._llm_reranker.rerank(query, candidates, top_k=top_n, blend_weight=0.7)

            reranked: list[FusedResult] = []
            for rr in results:
                fused = rr.item  # The original FusedResult
                fused.rrf_score = rr.final_score
                reranked.append(fused)

            reranked.extend(remainder)
            logger.debug(f"LLM reranking applied: {top_n} candidates scored for temporal query")
            return reranked[:limit]
        except Exception as e:
            logger.warning(f"LLM reranking failed, keeping current order: {e}")
            return fused_results

    async def _simple_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
        limit: int,
        routing: RoutingDecision,
        *,
        effective_recency: float = 0.0,
        decay_days_override: int | None = None,
        temporal_sort: bool = False,
        recency_floor: float = 0.5,
        temporal_signal: TemporalSignal | None = None,
    ) -> VectorCypherResult:
        """Simple retrieval path - vector search only.

        For SIMPLE-routed queries, uses a lower hybrid_alpha (0.5) to give
        BM25 equal weight — lexical overlap is stronger for factual queries.

        When temporal_sort is True, results are re-sorted by occurred_at DESC
        after recency boosting so that the most recent chunks surface first
        (matches the graph-path behaviour for temporal categories).
        """
        with trace_span("khora.vectorcypher.simple_retrieve", namespace_id=str(namespace_id)) as span:
            # When BM25 channel is active, use pure vector (hybrid_alpha=1.0)
            # to avoid double-counting BM25 in both the vector blend and the
            # independent channel. Otherwise, lower alpha for SIMPLE queries
            # to boost the pgvector-internal BM25 signal.
            if self._config.enable_bm25_channel:
                effective_alpha = 1.0
            else:
                effective_alpha = self._config.hybrid_alpha
                if routing.complexity == QueryComplexity.SIMPLE:
                    effective_alpha = min(effective_alpha, 0.5)

            # Launch BM25 search in parallel with vector search
            bm25_task: asyncio.Task[list[tuple[UUID, float, Chunk]]] | None = None
            if self._config.enable_bm25_channel and self._storage:
                bm25_task = asyncio.create_task(
                    self._bm25_search_chunks(
                        query=query,
                        namespace_id=namespace_id,
                        limit=self._config.bm25_top_k,
                    )
                )

            results = await self._vector_store.search(
                namespace_id=namespace_id,
                query_embedding=query_embedding,
                limit=limit,
                temporal_filter=temporal_filter,
                hybrid_alpha=effective_alpha,
                query_text=query,
            )

            chunk_results: list[tuple[Chunk, float]] = []
            for r in results:
                chunk = Chunk(
                    id=r.chunk.id,
                    namespace_id=r.chunk.namespace_id,
                    document_id=r.chunk.document_id,
                    content=r.chunk.content,
                    metadata=ChunkMetadata(
                        custom={
                            "occurred_at": r.chunk.occurred_at.isoformat() if r.chunk.occurred_at else None,
                            **(r.chunk.metadata or {}),
                        }
                    ),
                    created_at=r.chunk.created_at or r.chunk.occurred_at,
                )
                chunk_results.append((chunk, r.combined_score or r.similarity))

            # Fuse with BM25 results if the channel is active
            simple_bm25_count = 0
            if bm25_task is not None:
                try:
                    bm25_results = await bm25_task
                except Exception as e:
                    logger.warning(f"BM25 channel failed in simple path: {e}")
                    bm25_results = []

                simple_bm25_count = len(bm25_results)
                if bm25_results:
                    from khora.query.fusion import reciprocal_rank_fusion as _nlist_rrf

                    bm25_weight = self._config.bm25_weight
                    # Convert vector results to (item, score) format
                    ranked_lists: dict[str, list[tuple[Chunk, float]]] = {
                        "vector": [(c, s) for c, s in chunk_results],
                        "bm25": [(bm25_chunk, bm25_score) for _cid, bm25_score, bm25_chunk in bm25_results],
                    }
                    weights: dict[str, float] = {"vector": 1.0, "bm25": bm25_weight}
                    fused_raw = _nlist_rrf(
                        ranked_lists,
                        weights=weights,
                        k=self._config.rrf_k,
                        id_extractor=lambda c: c.id,
                    )
                    chunk_results = list(fused_raw[:limit])
                    logger.debug(
                        f"Simple path BM25 fusion: {len(bm25_results)} BM25 + "
                        f"{len(chunk_results)} vector -> {len(fused_raw)} fused"
                    )

            # Apply recency boost to simple path (was previously missing)
            if effective_recency > 0 and chunk_results:
                fused = [FusedResult(item=c, rrf_score=s, item_id=c.id) for c, s in chunk_results]
                with trace_span("khora.vectorcypher.recency_boost", chunk_count=len(fused)):
                    recency_scores = self._calculate_recency_scores(fused, decay_days_override=decay_days_override)
                    fused = apply_recency_boost(
                        fused, recency_scores, recency_weight=effective_recency, recency_floor=recency_floor
                    )
                chunk_results = [(r.item, r.rrf_score) for r in fused]

            # Cross-encoder reranking (after recency boost, before version scoring)
            if self._config.enable_reranking and chunk_results:
                fused = [FusedResult(item=c, rrf_score=s, item_id=c.id) for c, s in chunk_results]
                with trace_span("khora.vectorcypher.reranking", candidate_count=len(fused)):
                    fused = await self._apply_reranking(query, fused, limit)
                chunk_results = [(r.item, r.rrf_score) for r in fused]

            # LLM reranking of top-N for temporal queries (after cross-encoder)
            # Skip when cross-encoder is already confident (large gap between #1 and #2).
            if self._config.enable_llm_reranking and temporal_signal and temporal_signal.is_temporal and chunk_results:
                _skip_llm_simple = False
                if len(chunk_results) >= 2:
                    _gap = chunk_results[0][1] - chunk_results[1][1]
                    if _gap >= self._config.llm_reranking_confidence_threshold:
                        _skip_llm_simple = True
                if not _skip_llm_simple:
                    fused = [FusedResult(item=c, rrf_score=s, item_id=c.id) for c, s in chunk_results]
                    with trace_span("khora.vectorcypher.llm_reranking", candidate_count=len(fused)):
                        fused = await self._apply_llm_reranking(query, fused, limit)
                    chunk_results = [(r.item, r.rrf_score) for r in fused]

            # Version-aware scoring — the FINAL score adjustment after ALL reranking.
            # CHANGE/ORDINAL excluded: need full version history.
            if (
                temporal_signal
                and temporal_signal.category
                in (
                    TemporalCategory.STATE_QUERY,
                    TemporalCategory.RECENCY,
                )
                and chunk_results
            ):
                from collections import defaultdict as _defaultdict

                _entity_versions: dict[str, int] = _defaultdict(int)
                _chunk_versions: dict[UUID, int] = {}

                for c, _s in chunk_results:
                    meta = c.metadata.custom if c.metadata and isinstance(c.metadata, ChunkMetadata) else {}
                    if isinstance(meta, dict):
                        version = meta.get("version") or meta.get("entity_version", 0)
                        if version:
                            _chunk_versions[c.id] = int(version)
                            for ref in meta.get("entity_refs") or []:
                                _entity_versions[ref] = max(_entity_versions[ref], int(version))

                if _entity_versions:
                    _VERSION_DECAY = 0.7  # Stronger penalty: v1/v5 → 0.44 (was 0.6 with 0.5)
                    updated = []
                    for c, s in chunk_results:
                        v = _chunk_versions.get(c.id, 0)
                        if v > 0:
                            meta = c.metadata.custom if c.metadata and isinstance(c.metadata, ChunkMetadata) else {}
                            if isinstance(meta, dict):
                                for ref in meta.get("entity_refs") or []:
                                    max_v = _entity_versions.get(ref, v)
                                    if max_v > v:
                                        ratio = v / max_v
                                        s *= 1.0 - _VERSION_DECAY * (1.0 - ratio)
                                        break
                        updated.append((c, s))
                    chunk_results = updated

            # Apply temporal sort: re-order by occurred_at DESC so the most
            # recent chunks rank first. This mirrors the graph path's
            # temporal_sort and is critical for STATE_QUERY/RECENCY/CHANGE.
            #
            # Skip when cross-encoder reranking is active: the reranker already
            # captures semantic relevance order, and the recency boost (above)
            # provides temporal discrimination. A hard re-sort by timestamp
            # would override the reranker's carefully computed ranking.
            if temporal_sort and chunk_results and not self._config.enable_reranking:
                from datetime import datetime as _dt

                def _ts(pair: tuple[Chunk, float]) -> _dt:
                    occ = (pair[0].metadata.custom or {}).get("occurred_at") if pair[0].metadata else None
                    if occ:
                        try:
                            return _dt.fromisoformat(occ)
                        except (ValueError, TypeError):
                            pass
                    return pair[0].created_at or _dt.min

                # ORDINAL queries ("first", "which came earlier") need ascending
                # order; all other temporal categories use descending (most recent first).
                sort_descending = True
                if temporal_signal and temporal_signal.category == TemporalCategory.ORDINAL:
                    sort_descending = False

                chunk_results.sort(key=_ts, reverse=sort_descending)

            # Normalize scores to [0,1] — matches complex path behavior
            if chunk_results:
                fused = [FusedResult(item=c, rrf_score=s, item_id=c.id) for c, s in chunk_results]
                fused = normalize_scores(fused)
                chunk_results = [(r.item, r.rrf_score) for r in fused]

            span.set_attribute("chunk_count", len(chunk_results))

            # All chunks come from vector search in simple mode
            all_ids = [str(c.id) for c, _ in chunk_results]
            search_methods = {
                "chunk_overlap": {
                    "vector_only": {"ids": all_ids, "count": len(all_ids)},
                    "graph_only": {"ids": [], "count": 0},
                    "vector_and_graph": {"ids": [], "count": 0},
                },
                "by_method": {
                    "vector": {"chunk_ids": all_ids, "count": len(all_ids)},
                    "graph": {"chunk_ids": [], "count": 0},
                },
            }

            return VectorCypherResult(
                chunks=chunk_results,
                entities=[],
                routing_decision=routing,
                metadata={
                    "search_mode": "simple_vector" if not simple_bm25_count else "simple_vector_bm25",
                    "routing_confidence": routing.confidence,
                    "vector_chunk_count": len(chunk_results),
                    "graph_chunk_count": 0,
                    "bm25_chunk_count": simple_bm25_count,
                    "effective_recency": effective_recency,
                    "temporal_sort": temporal_sort,
                    # Search provenance: all chunks from vector in simple mode
                    "search_methods": search_methods,
                },
            )

    async def _vector_search_entities(
        self,
        query_embedding: list[float],
        namespace_id: UUID,
        limit: int,
    ) -> list[tuple[UUID, float]]:
        """Search for entry entities using vector similarity via pgvector HNSW."""
        if not self._storage:
            logger.warning("Storage coordinator not available for entity vector search")
            return []

        with trace_span("khora.vectorcypher.vector_search_entities", namespace_id=str(namespace_id)) as span:
            try:
                results = await self._storage.search_similar_entities(
                    namespace_id,
                    query_embedding,
                    limit=limit,
                    min_similarity=self._config.min_entity_similarity,
                )
                span.set_attribute("entity_count", len(results))
                return results
            except Exception as e:
                logger.warning(f"Entity vector search failed: {e}")
                return []

    async def _cypher_expand(
        self,
        entry_entity_ids: list[UUID],
        namespace_id: UUID,
        depth: int,
        *,
        prefer_current: bool = False,
    ) -> tuple[dict[UUID, float], dict[str, dict[str, str]]]:
        """Expand entry entities to find related entities via graph traversal.

        Args:
            entry_entity_ids: Starting entity IDs
            namespace_id: Namespace constraint
            depth: Maximum traversal depth
            prefer_current: When True, filter out expired entities (for temporal queries)

        Returns:
            Tuple of:
            - Dict mapping entity_id -> relevance score
            - Dict mapping entity_id_str -> {name, entity_type} for all discovered entities
        """
        if not entry_entity_ids:
            return {}, {}

        with trace_span("khora.vectorcypher.cypher_expand", entry_count=len(entry_entity_ids), depth=depth) as span:
            depth = min(max(1, depth), self._config.max_depth)

            # Get neighborhoods from dual node manager (Neo4j) or storage coordinator (SurrealDB)
            if self._dual_nodes is not None:
                neighborhoods = await self._dual_nodes.get_entity_neighborhoods(
                    entity_ids=entry_entity_ids,
                    namespace_id=namespace_id,
                    depth=depth,
                    limit_per_entity=20,
                    prefer_current=prefer_current,
                )
            elif self._storage and self._storage.graph:
                raw_neighborhoods = await self._storage.get_neighborhoods_batch(
                    entry_entity_ids,
                    depth=depth,
                    limit_per_entity=20,
                )
                # Normalize: get_neighborhoods_batch returns
                # {UUID: {"entities": [...], "relationships": [...]}}
                # but the scoring loop expects {UUID: [{"id":..., "distance":..., ...}]}
                neighborhoods = {}
                for eid, data in raw_neighborhoods.items():
                    entities_list = data.get("entities", []) if isinstance(data, dict) else data
                    normalized = []
                    for i, entity_data in enumerate(entities_list if isinstance(entities_list, list) else []):
                        if isinstance(entity_data, dict):
                            entity_data.setdefault("distance", i + 1)
                            normalized.append(entity_data)
                    neighborhoods[eid] = normalized
            else:
                neighborhoods = {}

            # Score entities by distance from entry points and collect entity info
            entity_scores: dict[UUID, float] = {}
            entity_info_map: dict[str, dict[str, str]] = {}

            for source_id, related in neighborhoods.items():
                for entity_info in related:
                    # Handle both bare UUIDs (Neo4j) and record IDs like "entity:⟨uuid⟩" (SurrealDB)
                    raw_id = entity_info["id"]
                    try:
                        entity_id = UUID(str(raw_id)) if not isinstance(raw_id, UUID) else raw_id
                    except ValueError:
                        # SurrealDB record ID — extract UUID from "table:⟨uuid⟩"
                        import re

                        m = re.search(r"[0-9a-fA-F\-]{36}", str(raw_id))
                        entity_id = UUID(m.group(0)) if m else UUID(int=0)
                    distance = entity_info.get("distance", 1)
                    # Score decreases with distance
                    score = 1.0 / (1 + distance)

                    if entity_id in entity_scores:
                        # Take max score if entity reached multiple ways
                        entity_scores[entity_id] = max(entity_scores[entity_id], score)
                    else:
                        entity_scores[entity_id] = score

                    # Capture name, type, description, source_tool (zero-cost, data already fetched)
                    eid_str = str(entity_id)
                    if eid_str not in entity_info_map:
                        entity_info_map[eid_str] = {
                            "name": entity_info.get("name", ""),
                            "entity_type": entity_info.get("entity_type", ""),
                            "description": entity_info.get("description", ""),
                            "source_tool": entity_info.get("source_tool", ""),
                        }

            span.set_attribute("expanded_entity_count", len(entity_scores))
            return entity_scores, entity_info_map

    @staticmethod
    def _decompose_change_query(query: str) -> str | None:
        """Decompose a CHANGE query into a current-state sub-query.

        Rewrites temporal-change phrasing into a present-tense question about
        the entity's current state, so a second vector search retrieves
        up-to-date evidence alongside the historical evidence from the
        original query.

        Examples:
            "What did Alice used to play?" → "What does Alice play now?"
            "Does she still work at Google?" → "Where does she work now?"
            "He switched from piano to guitar" → "What instrument does he play now?"
        """
        import re

        q = query.strip()
        ql = q.lower()

        # Pattern: "used to <verb>" → "currently <verb>"
        m = re.search(r"(\w+)\s+used\s+to\s+(.+?)(?:\?|$)", ql)
        if m:
            subject = m.group(1)
            rest = m.group(2).rstrip("? .")
            return f"What does {subject} {rest} now?"

        # Pattern: "still <verb>" → current state question
        m = re.search(r"(?:does|do|is)\s+(\w+)\s+still\s+(.+?)(?:\?|$)", ql)
        if m:
            subject = m.group(1)
            rest = m.group(2).rstrip("? .")
            return f"What does {subject} {rest} now?"

        # Pattern: "switched from X to Y" / "changed from X to Y"
        m = re.search(r"(\w+)\s+(?:switched|changed|moved|transitioned)\s+(?:from\s+.+?\s+)?to\s+(.+?)(?:\?|$)", ql)
        if m:
            subject = m.group(1)
            new_state = m.group(2).rstrip("? .")
            return f"What is {subject} {new_state} now?"

        # Pattern: "no longer" → ask about current state
        m = re.search(r"(\w+)\s+(?:is|was)\s+no\s+longer\s+(.+?)(?:\?|$)", ql)
        if m:
            subject = m.group(1)
            old_state = m.group(2).rstrip("? .")
            return f"What is {subject} doing instead of {old_state}?"

        # Fallback: prepend "currently" to make it present-focused
        if any(kw in ql for kw in ("used to", "still", "previously", "before", "changed", "switched")):
            # Strip common change keywords and add "currently"
            cleaned = re.sub(
                r"\b(used to|still|previously|formerly|no longer)\b",
                "currently",
                ql,
                count=1,
            )
            return cleaned.strip()

        return None

    async def _version_filter_entities(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        target_date: datetime,
    ) -> list[UUID]:
        """Filter entities to those valid at a specific point in time (bi-temporal).

        Uses ``version_valid_from`` / ``version_valid_to`` properties on
        Entity nodes.  Entities without version properties are treated as
        always-valid (backward-compatible).

        Also checks :EntityVersion snapshot nodes reachable via SUPERSEDES
        edges, returning the snapshot ID when the current entity was not
        yet valid at the target date but a prior version was.

        Args:
            entity_ids: Candidate entity IDs
            namespace_id: Namespace constraint
            target_date: The point-in-time to query for

        Returns:
            Filtered list of entity IDs (may include EntityVersion IDs)
            that were valid at ``target_date``
        """
        if not entity_ids:
            return []

        # SurrealDB unified backend — no Neo4j driver, skip version filtering
        if self._neo4j_driver is None:
            return list(entity_ids)

        # First: keep current Entity nodes that are valid at target_date,
        # OR that have no version properties (backward-compatible).
        # Second: for entities not valid at target_date, check if a prior
        # EntityVersion was valid via SUPERSEDES edges.
        query = """
        UNWIND $entity_ids AS eid
        MATCH (e:Entity {id: eid, namespace_id: $namespace_id})
        OPTIONAL MATCH (e)-[:SUPERSEDES]->(ev:EntityVersion)
        WHERE ev.namespace_id = $namespace_id
          AND (ev.version_valid_from IS NULL OR ev.version_valid_from <= $target_date)
          AND (ev.version_valid_to IS NULL OR ev.version_valid_to > $target_date)
        WITH e, collect(ev.id) AS version_ids
        WITH e, version_ids,
             CASE
               WHEN e.version_valid_from IS NULL THEN true
               WHEN e.version_valid_from <= $target_date
                    AND (e.version_valid_to IS NULL OR e.version_valid_to > $target_date)
               THEN true
               ELSE false
             END AS current_valid
        WHERE current_valid OR size(version_ids) > 0
        RETURN CASE WHEN current_valid THEN e.id
                    ELSE version_ids[0]
               END AS id
        """

        async with self._neo4j_driver.session(database=self._database) as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    entity_ids=[str(eid) for eid in entity_ids],
                    namespace_id=str(namespace_id),
                    target_date=target_date.isoformat(),
                )
                return [record.data() async for record in result]

            records = await session.execute_read(_work)

        filtered = [UUID(r["id"]) for r in records if r["id"]]
        logger.debug(
            f"Version filter at {target_date.isoformat()}: {len(entity_ids)} candidates -> {len(filtered)} valid"
        )
        return filtered

    async def _fetch_version_history(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
    ) -> list[dict[str, Any]]:
        """Traverse SUPERSEDES edges to retrieve version history for entities.

        Used for CHANGE-category temporal queries ("what did X used to be?",
        "how has Y changed?").

        Args:
            entity_ids: Entity IDs to get version history for
            namespace_id: Namespace constraint

        Returns:
            List of dicts with ``current_*`` and ``previous_*`` fields
            representing the version transition chain.
        """
        if not entity_ids or self._neo4j_driver is None:
            return []

        query = """
        UNWIND $entity_ids AS eid
        MATCH (current:Entity {id: eid, namespace_id: $namespace_id})
        OPTIONAL MATCH (current)-[s:SUPERSEDES]->(prev:EntityVersion)
        RETURN current.id AS current_id,
               current.name AS name,
               current.entity_type AS entity_type,
               current.attributes AS current_attributes,
               current.version_valid_from AS current_valid_from,
               current.version_valid_to AS current_valid_to,
               prev.id AS previous_id,
               prev.attributes AS previous_attributes,
               prev.version_valid_from AS previous_valid_from,
               prev.version_valid_to AS previous_valid_to,
               s.superseded_at AS superseded_at
        ORDER BY current.name, s.superseded_at DESC
        """

        async with self._neo4j_driver.session(database=self._database) as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    entity_ids=[str(eid) for eid in entity_ids],
                    namespace_id=str(namespace_id),
                )
                return [record.data() async for record in result]

            records = await session.execute_read(_work)

        logger.debug(f"Version history: {len(records)} version records for {len(entity_ids)} entities")
        return records

    async def _fetch_chunks_from_entities(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
        limit: int,
        *,
        temporal_sort: bool = False,
        prefer_current: bool = False,
    ) -> list[tuple[UUID, float, Chunk]]:
        """Fetch chunks connected to entities via MENTIONED_IN.

        Args:
            entity_ids: Entity IDs to fetch chunks for
            namespace_id: Namespace constraint
            temporal_filter: Optional temporal constraints
            limit: Maximum chunks to return
            temporal_sort: If True, sort by occurred_at DESC (for temporal queries)
            prefer_current: When True, filter out expired entities

        Returns:
            List of (chunk_id, score, chunk) tuples
        """
        with trace_span(
            "khora.vectorcypher.fetch_entity_chunks",
            entity_count=len(entity_ids),
            namespace_id=str(namespace_id),
        ) as span:
            if self._dual_nodes is not None:
                chunk_records = await self._dual_nodes.get_chunks_by_entities(
                    entity_ids=entity_ids,
                    namespace_id=namespace_id,
                    temporal_filter=temporal_filter,
                    temporal_sort=temporal_sort,
                    prefer_current=prefer_current,
                    limit=limit,
                )
            elif self._storage:
                # SurrealDB fallback: get chunks via entity source_chunk_ids
                chunk_records = []
                try:
                    entities_map = await self._storage.get_entities_batch(entity_ids)
                    all_chunk_ids: list[UUID] = []
                    for entity in entities_map.values():
                        all_chunk_ids.extend(entity.source_chunk_ids[:5])
                    if all_chunk_ids:
                        chunks_map = await self._storage.get_chunks_batch(all_chunk_ids)
                        for cid, chunk in chunks_map.items():
                            chunk_records.append(
                                {
                                    "chunk_id": str(cid),
                                    "content": chunk.content,
                                    "embedding": chunk.embedding,
                                    "total_mentions": 1,
                                    "entity_ids": [],
                                    "occurred_at": getattr(chunk, "source_timestamp", None),
                                }
                            )
                except Exception as e:
                    logger.warning(f"SurrealDB chunk fetch fallback failed: {e}")
                    chunk_records = []
            else:
                chunk_records = []

            results: list[tuple[UUID, float, Chunk]] = []
            for record in chunk_records:
                chunk_id = UUID(record["chunk_id"])
                # Score based on mention count and entity coverage
                score = float(record.get("total_mentions", 1))
                entity_count = len(record.get("entity_ids", []))
                score = score * (1 + 0.1 * entity_count)  # Boost for multiple entity connections

                chunk = Chunk(
                    id=chunk_id,
                    namespace_id=namespace_id,
                    document_id=UUID(record["document_id"]),
                    content=record["content"],
                    metadata=ChunkMetadata(
                        custom={
                            "occurred_at": record.get("occurred_at"),
                            "connected_entities": record.get("entity_ids", []),
                            **(record.get("metadata") or {}),
                        }
                    ),
                )
                results.append((chunk_id, score, chunk))

            span.set_attribute("chunk_count", len(results))
            return results

    async def _vector_search_chunks(
        self,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
        query_text: str,
        limit: int,
        *,
        hybrid_alpha_override: float | None = None,
    ) -> list[tuple[UUID, float, Chunk]]:
        """Direct vector search on chunks via pgvector.

        Args:
            query_embedding: Query embedding
            namespace_id: Namespace to search
            temporal_filter: Temporal constraints
            query_text: Original query text for hybrid search
            limit: Maximum results
            hybrid_alpha_override: If set, overrides the configured hybrid_alpha.
                                   Used to force pure vector (1.0) when the BM25
                                   channel is active to avoid double-counting.

        Returns:
            List of (chunk_id, score, chunk) tuples
        """
        effective_alpha = hybrid_alpha_override if hybrid_alpha_override is not None else self._config.hybrid_alpha
        with trace_span("khora.vectorcypher.vector_search_chunks", namespace_id=str(namespace_id)) as span:
            results = await self._vector_store.search(
                namespace_id=namespace_id,
                query_embedding=query_embedding,
                limit=limit,
                temporal_filter=temporal_filter,
                hybrid_alpha=effective_alpha,
                query_text=query_text,
            )

            span.set_attribute("chunk_count", len(results))
            return [
                (
                    r.chunk.id,
                    r.combined_score or r.similarity,
                    Chunk(
                        id=r.chunk.id,
                        namespace_id=r.chunk.namespace_id,
                        document_id=r.chunk.document_id,
                        content=r.chunk.content,
                        metadata=ChunkMetadata(
                            custom={
                                "occurred_at": r.chunk.occurred_at.isoformat() if r.chunk.occurred_at else None,
                                **(r.chunk.metadata or {}),
                            }
                        ),
                        created_at=r.chunk.created_at or r.chunk.occurred_at,
                    ),
                )
                for r in results
            ]

    async def _bm25_search_chunks(
        self,
        query: str,
        namespace_id: UUID,
        limit: int,
    ) -> list[tuple[UUID, float, Chunk]]:
        """Full-text BM25 search on chunks via StorageCoordinator.

        Mirrors the pattern of ``_vector_search_chunks`` but uses PostgreSQL
        full-text search (``ts_rank``) instead of vector similarity.

        Args:
            query: Original query text
            namespace_id: Namespace to search
            limit: Maximum results

        Returns:
            List of (chunk_id, score, chunk) tuples
        """
        if not self._storage:
            logger.debug("Storage coordinator not available for BM25 search")
            return []

        with trace_span("khora.vectorcypher.bm25_search_chunks", namespace_id=str(namespace_id)) as span:
            try:
                results = await self._storage.search_fulltext_chunks(
                    namespace_id,
                    query,
                    limit=limit,
                )
                span.set_attribute("chunk_count", len(results))
                logger.debug(f"BM25 channel returned {len(results)} chunks")
                return [
                    (
                        chunk.id,
                        score,
                        chunk,
                    )
                    for chunk, score in results
                ]
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
                return []

    def _lazy_expand_chunks(
        self,
        vector_only_chunks: list[tuple[UUID, float, Chunk]],
        entry_entities: list[tuple[UUID, float]],
        entity_info_map: dict[str, dict[str, str]],
    ) -> list[tuple[UUID, float, Chunk]]:
        """Expand vector-only chunks by keyword matching against known entities.

        For chunks retrieved via vector search that have no MENTIONED_IN edges,
        extract keywords and match them against entity names. This recovers
        graph signal for chunks that weren't covered by skeleton extraction.

        Results are cached per chunk_id so repeated retrievals are fast.
        """
        from khora._accel import extract_keywords

        # Build lowercased entity name set
        entity_names: set[str] = set()
        for eid, _ in entry_entities:
            info = entity_info_map.get(str(eid), {})
            name = info.get("name", "").lower().strip()
            if name:
                entity_names.add(name)

        if not entity_names:
            return []

        results: list[tuple[UUID, float, Chunk]] = []
        for chunk_id, _vec_score, chunk in vector_only_chunks:
            # Check cache first
            if chunk_id in self._expansion_cache:
                cached_score = self._expansion_cache[chunk_id]
                if cached_score > 0:
                    results.append((chunk_id, cached_score, chunk))
                continue

            content = chunk.content
            if not content:
                self._expansion_cache[chunk_id] = 0.0
                continue

            keywords = {kw.lower() for kw in extract_keywords(content)}
            matches = keywords & entity_names
            if matches:
                # Weak signal: 0.5 per matched entity name
                expansion_score = len(matches) * 0.5
                results.append((chunk_id, expansion_score, chunk))
                self._expansion_cache[chunk_id] = expansion_score
            else:
                self._expansion_cache[chunk_id] = 0.0

        return results

    def _fuse_results(
        self,
        vector_chunks: list[tuple[UUID, float, Chunk]],
        graph_chunks: list[tuple[UUID, float, Chunk]],
        *,
        bm25_chunks: list[tuple[UUID, float, Chunk]] | None = None,
        use_normalization: bool = False,
        routing: RoutingDecision | None = None,
        is_temporal: bool = False,
    ) -> list[FusedResult]:
        """Fuse vector, graph, and optionally BM25 results using weighted RRF.

        When ``bm25_chunks`` is provided (BM25 channel active), uses the N-list
        ``reciprocal_rank_fusion`` from :mod:`khora.query.fusion` to fuse all
        three channels. Otherwise falls back to the 2-list
        ``weighted_rrf_normalized`` for vector+graph fusion.

        Args:
            vector_chunks: Results from vector search
            graph_chunks: Results from graph traversal
            bm25_chunks: Results from BM25 full-text search (optional)
            use_normalization: If True, normalize scores before fusion for better ranking
            routing: If provided, adjust weights based on query complexity
            is_temporal: If True, use temporal fusion weights (graph-heavy)

        Returns:
            Fused and sorted results
        """
        with trace_span(
            "khora.vectorcypher.rrf_fusion",
            vector_count=len(vector_chunks),
            graph_count=len(graph_chunks),
        ) as span:
            # Dynamic fusion weights based on query complexity
            vector_weight = self._config.vector_weight
            graph_weight = self._config.graph_weight
            if is_temporal:
                # Temporal queries benefit from graph-heavy fusion:
                # graph traversal surfaces temporally-related entities and their chunks
                vector_weight = self._config.temporal_vector_weight
                graph_weight = self._config.temporal_graph_weight

                # DYT-470: Adapt fusion weights when graph returns zero/few results.
                # When graph retrieval yields nothing (common for short conversational
                # messages without entity extraction), using graph-heavy weights (0.3/0.7)
                # dilutes good vector results and hurts ranking.
                graph_count = len(graph_chunks)
                if graph_count == 0:
                    vector_weight = 0.85
                    graph_weight = 0.15
                    logger.debug(
                        "Adaptive fusion: graph empty, using vector-heavy weights (%.2f/%.2f)",
                        vector_weight,
                        graph_weight,
                    )
                elif graph_count < 3:
                    vector_weight = self._config.vector_weight  # default 0.6
                    graph_weight = self._config.graph_weight  # default 0.4
                    logger.debug(
                        "Adaptive fusion: sparse graph (%d chunks), using moderate weights (%.2f/%.2f)",
                        graph_count,
                        vector_weight,
                        graph_weight,
                    )
            elif routing is not None:
                if routing.complexity == QueryComplexity.SIMPLE:
                    vector_weight = self._config.simple_vector_weight
                    graph_weight = self._config.simple_graph_weight
                elif routing.complexity == QueryComplexity.COMPLEX:
                    vector_weight = self._config.complex_vector_weight
                    graph_weight = self._config.complex_graph_weight

            span.set_attribute("vector_weight", vector_weight)
            span.set_attribute("graph_weight", graph_weight)

            # ── 3-channel fusion (vector + graph + BM25) ────────────────
            if bm25_chunks:
                from khora.query.fusion import reciprocal_rank_fusion as _nlist_rrf

                bm25_weight = self._config.bm25_weight
                span.set_attribute("bm25_weight", bm25_weight)
                span.set_attribute("bm25_count", len(bm25_chunks))

                # Build ranked lists in the (item, score) format expected by
                # the N-list reciprocal_rank_fusion.
                ranked_lists: dict[str, list[tuple[Chunk, float]]] = {}
                if vector_chunks:
                    ranked_lists["vector"] = [(chunk, score) for _cid, score, chunk in vector_chunks]
                if graph_chunks:
                    ranked_lists["graph"] = [(chunk, score) for _cid, score, chunk in graph_chunks]
                ranked_lists["bm25"] = [(chunk, score) for _cid, score, chunk in bm25_chunks]

                weights: dict[str, float] = {
                    "vector": vector_weight,
                    "graph": graph_weight,
                    "bm25": bm25_weight,
                }

                fused_raw: list[tuple[Chunk, float]] = _nlist_rrf(
                    ranked_lists,
                    weights=weights,
                    k=self._config.rrf_k,
                    id_extractor=lambda chunk: chunk.id,
                )

                # Convert (Chunk, rrf_score) tuples to FusedResult objects.
                # The N-list RRF doesn't populate per-source ranks, so we
                # build lookup maps to back-fill vector/graph provenance.
                vector_rank_map: dict[UUID, int] = {}
                vector_score_map: dict[UUID, float] = {}
                for rank, (cid, score, _chunk) in enumerate(vector_chunks, start=1):
                    vector_rank_map[cid] = rank
                    vector_score_map[cid] = score

                graph_rank_map: dict[UUID, int] = {}
                graph_score_map: dict[UUID, float] = {}
                for rank, (cid, score, _chunk) in enumerate(graph_chunks, start=1):
                    graph_rank_map[cid] = rank
                    graph_score_map[cid] = score

                return [
                    FusedResult(
                        item_id=chunk.id,
                        item=chunk,
                        rrf_score=rrf_score,
                        vector_rank=vector_rank_map.get(chunk.id),
                        graph_rank=graph_rank_map.get(chunk.id),
                        vector_score=vector_score_map.get(chunk.id),
                        graph_score=graph_score_map.get(chunk.id),
                    )
                    for chunk, rrf_score in fused_raw
                ]

            # ── 2-channel fusion (vector + graph) ──────────────────────
            if use_normalization:
                return weighted_rrf_normalized(
                    vector_results=vector_chunks,
                    graph_results=graph_chunks,
                    k=self._config.rrf_k,
                    vector_weight=vector_weight,
                    graph_weight=graph_weight,
                )
            return weighted_rrf(
                vector_results=vector_chunks,
                graph_results=graph_chunks,
                k=self._config.rrf_k,
                vector_weight=vector_weight,
                graph_weight=graph_weight,
            )

    def _calculate_recency_scores(
        self,
        results: list[FusedResult],
        *,
        decay_days_override: int | None = None,
    ) -> dict[UUID, float]:
        """Calculate recency scores for temporal boosting.

        Uses *relative* recency: the reference time is the newest occurred_at
        in the result set (not wall-clock time).  This ensures benchmark data
        from any era produces meaningful discrimination and that live data
        (where max ≈ now) behaves identically to the old implementation.

        Args:
            results: Fused results with items containing occurred_at
            decay_days_override: Override for decay_days (e.g. 7 for RECENCY category)

        Returns:
            Dict mapping item_id -> recency score (0-1)
        """
        decay_days = decay_days_override or self._config.recency_decay_days
        scores: dict[UUID, float] = {}

        # First pass: extract all occurred_at timestamps and find the max
        parsed_times: dict[UUID, datetime] = {}
        for r in results:
            occurred_at_str = _extract_occurred_at(r.item)
            if occurred_at_str:
                try:
                    occurred_at = datetime.fromisoformat(occurred_at_str.replace("Z", "+00:00"))
                    if occurred_at.tzinfo is None:
                        occurred_at = occurred_at.replace(tzinfo=UTC)
                    parsed_times[r.item_id] = occurred_at
                except (ValueError, TypeError):
                    pass

        # Relative reference: newest item in result set, fallback to wall-clock
        now = max(parsed_times.values()) if parsed_times else datetime.now(UTC)

        # Second pass: compute recency scores relative to reference time
        for r in results:
            if r.item_id in parsed_times:
                days_old = (now - parsed_times[r.item_id]).total_seconds() / 86400.0
                if self._config.recency_decay_type == "exponential":
                    half_life_lambda = math.log(2) / decay_days
                    recency = math.exp(-half_life_lambda * days_old)
                else:
                    recency = max(0.0, 1.0 - (days_old / decay_days))
                scores[r.item_id] = recency
            else:
                scores[r.item_id] = 0.5  # Default for missing/unparseable dates

        return scores


__all__ = [
    "RetrieverConfig",
    "VectorCypherResult",
    "VectorCypherRetriever",
]
