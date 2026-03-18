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

    # Limits
    max_chunks: int = 50
    max_entities: int = 30


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
        neo4j_driver: AsyncDriver,
        embedder: EmbedderProtocol,
        *,
        database: str = "neo4j",
        config: RetrieverConfig | None = None,
        router_config: RouterConfig | None = None,
        storage: StorageCoordinator | None = None,
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
        self._dual_nodes = DualNodeManager(neo4j_driver, database)

        # Query result cache (LRU + TTL)
        self._cache: dict[str, tuple[float, VectorCypherResult]] = {}
        self._cache_ttl = self._config.query_cache_ttl_seconds
        self._cache_max_size = self._config.query_cache_max_size

        # Lazy entity expansion cache: chunk_id -> expansion_score (0 = no match)
        self._expansion_cache: dict[UUID, float] = {}

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
                    f"{query}:{namespace_id}:{temporal_filter}:{graph_depth}:{limit}".encode()
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
                routing = await self._router.route(query)
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
        # This operation doesn't depend on entity search results
        vector_chunks_task = asyncio.create_task(
            self._vector_search_chunks(
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                query_text=query,
                limit=limit,
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
            # Cancel the parallel task since we're taking a different path
            vector_chunks_task.cancel()
            try:
                await vector_chunks_task
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
        # This was started at the beginning and may already be done
        vector_chunks = await vector_chunks_task

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

        # Normalize scores
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

        # Batch-fetch full entities from storage instead of constructing stubs
        entity_ids_to_fetch = [eid for eid, _ in all_entity_scores]
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

        # Closed-world relationship fetch: query Neo4j with only the final
        # entity IDs so the DB never returns relationships we'd discard.
        result_entity_ids_str = [str(entity.id) for entity, _ in entity_results]
        raw_rels = await self._dual_nodes.get_relationships_between(
            result_entity_ids_str,
            str(namespace_id),
        )
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
                "total_chunks_before_fusion": len(graph_chunks) + len(vector_chunks),
                "routing_confidence": routing.confidence,
                # Fusion telemetry
                "vector_chunk_count": len(vector_chunks),
                "graph_chunk_count": len(graph_chunks),
                "is_temporal": _tp.recency_weight > 0.2,
                "recency_weight": _tp.recency_weight,
                "effective_recency": effective_recency,
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
        )

        # Update metadata to indicate fallback was used
        result.metadata["fallback_mode"] = "vector_only"
        result.metadata["graph_unavailable"] = True

        return result

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
    ) -> VectorCypherResult:
        """Simple retrieval path - vector search only.

        For SIMPLE-routed queries, uses a lower hybrid_alpha (0.5) to give
        BM25 equal weight — lexical overlap is stronger for factual queries.

        When temporal_sort is True, results are re-sorted by occurred_at DESC
        after recency boosting so that the most recent chunks surface first
        (matches the graph-path behaviour for temporal categories).
        """
        with trace_span("khora.vectorcypher.simple_retrieve", namespace_id=str(namespace_id)) as span:
            # WS8: Lower alpha for SIMPLE queries to boost BM25 signal
            effective_alpha = self._config.hybrid_alpha
            if routing.complexity == QueryComplexity.SIMPLE:
                effective_alpha = min(effective_alpha, 0.5)

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

            # Apply recency boost to simple path (was previously missing)
            if effective_recency > 0 and chunk_results:
                fused = [FusedResult(item=c, rrf_score=s, item_id=c.id) for c, s in chunk_results]
                with trace_span("khora.vectorcypher.recency_boost", chunk_count=len(fused)):
                    recency_scores = self._calculate_recency_scores(fused, decay_days_override=decay_days_override)
                    fused = apply_recency_boost(
                        fused, recency_scores, recency_weight=effective_recency, recency_floor=recency_floor
                    )
                chunk_results = [(r.item, r.rrf_score) for r in fused]

            # Apply temporal sort: re-order by occurred_at DESC so the most
            # recent chunks rank first. This mirrors the graph path's
            # temporal_sort and is critical for STATE_QUERY/RECENCY/CHANGE.
            if temporal_sort and chunk_results:
                from datetime import datetime as _dt

                def _ts(pair: tuple[Chunk, float]) -> _dt:
                    occ = (pair[0].metadata.custom or {}).get("occurred_at") if pair[0].metadata else None
                    if occ:
                        try:
                            return _dt.fromisoformat(occ)
                        except (ValueError, TypeError):
                            pass
                    return pair[0].created_at or _dt.min

                chunk_results.sort(key=_ts, reverse=True)

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
                    "search_mode": "simple_vector",
                    "routing_confidence": routing.confidence,
                    "vector_chunk_count": len(chunk_results),
                    "graph_chunk_count": 0,
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

            # Get neighborhoods from dual node manager
            neighborhoods = await self._dual_nodes.get_entity_neighborhoods(
                entity_ids=entry_entity_ids,
                namespace_id=namespace_id,
                depth=depth,
                limit_per_entity=20,
                prefer_current=prefer_current,
            )

            # Score entities by distance from entry points and collect entity info
            entity_scores: dict[UUID, float] = {}
            entity_info_map: dict[str, dict[str, str]] = {}

            for source_id, related in neighborhoods.items():
                for entity_info in related:
                    entity_id = UUID(entity_info["id"])
                    distance = entity_info["distance"]
                    # Score decreases with distance
                    score = 1.0 / (1 + distance)

                    if entity_id in entity_scores:
                        # Take max score if entity reached multiple ways
                        entity_scores[entity_id] = max(entity_scores[entity_id], score)
                    else:
                        entity_scores[entity_id] = score

                    # Capture name, type, description, source_tool (zero-cost, data already fetched)
                    if entity_info["id"] not in entity_info_map:
                        entity_info_map[entity_info["id"]] = {
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
            f"Version filter at {target_date.isoformat()}: " f"{len(entity_ids)} candidates -> {len(filtered)} valid"
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
        if not entity_ids:
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
            chunk_records = await self._dual_nodes.get_chunks_by_entities(
                entity_ids=entity_ids,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                temporal_sort=temporal_sort,
                prefer_current=prefer_current,
                limit=limit,
            )

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
    ) -> list[tuple[UUID, float, Chunk]]:
        """Direct vector search on chunks via pgvector.

        Args:
            query_embedding: Query embedding
            namespace_id: Namespace to search
            temporal_filter: Temporal constraints
            query_text: Original query text for hybrid search
            limit: Maximum results

        Returns:
            List of (chunk_id, score, chunk) tuples
        """
        with trace_span("khora.vectorcypher.vector_search_chunks", namespace_id=str(namespace_id)) as span:
            results = await self._vector_store.search(
                namespace_id=namespace_id,
                query_embedding=query_embedding,
                limit=limit,
                temporal_filter=temporal_filter,
                hybrid_alpha=self._config.hybrid_alpha,
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
        use_normalization: bool = False,
        routing: RoutingDecision | None = None,
        is_temporal: bool = False,
    ) -> list[FusedResult]:
        """Fuse vector and graph results using weighted RRF.

        Args:
            vector_chunks: Results from vector search
            graph_chunks: Results from graph traversal
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
