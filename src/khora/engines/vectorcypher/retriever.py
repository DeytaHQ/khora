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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger
from neo4j.exceptions import Neo4jError

from .dual_nodes import DualNodeManager
from .fusion import (
    FusedResult,
    apply_recency_boost,
    normalize_scores,
    weighted_rrf,
    weighted_rrf_normalized,
)
from .router import QueryComplexity, QueryComplexityRouter, RouterConfig, RoutingDecision

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from khora.engines.skeleton.backends import TemporalFilter, TemporalVectorStore
    from khora.extraction.embedders import EmbedderProtocol  # type: ignore[unresolved-import]
    from khora.storage import StorageCoordinator


@dataclass
class VectorCypherResult:
    """Result from VectorCypher retrieval."""

    chunks: list[tuple[dict[str, Any], float]]
    entities: list[tuple[dict[str, Any], float]]
    routing_decision: RoutingDecision
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

    # Temporal settings
    recency_weight: float = 0.2
    recency_decay_days: int = 30

    # Limits
    max_chunks: int = 50
    max_entities: int = 30


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

    async def retrieve(
        self,
        query: str,
        namespace_id: UUID,
        *,
        temporal_filter: TemporalFilter | None = None,
        graph_depth: int | None = None,
        limit: int | None = None,
    ) -> VectorCypherResult:
        """Retrieve relevant chunks using VectorCypher hybrid approach.

        Args:
            query: User query
            namespace_id: Namespace to search
            temporal_filter: Optional temporal constraints
            graph_depth: Override for graph traversal depth
            limit: Maximum chunks to return

        Returns:
            VectorCypherResult with chunks, entities, and metadata
        """
        limit = limit or self._config.max_chunks

        # Step 1: Route query to determine strategy
        routing = await self._router.route(query)
        logger.debug(f"Query routing: {routing.complexity.value} (use_graph={routing.use_graph})")

        # Step 2: Embed the query
        query_embedding = await self._embedder.embed(query)

        # Step 3: Vector search for entry points
        if routing.complexity == QueryComplexity.SIMPLE:
            # Simple path: direct chunk retrieval
            return await self._simple_retrieve(
                query=query,
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                limit=limit,
                routing=routing,
            )

        # Complex/moderate path: VectorCypher with parallel execution
        # Wrap in try/except for graceful fallback on graph failures
        try:
            return await self._vectorcypher_retrieve(
                query=query,
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                graph_depth=graph_depth,
                limit=limit,
                routing=routing,
            )
        except Neo4jError as e:
            logger.warning(f"Graph search failed, falling back to vector-only: {e}")
            return await self._vector_only_fallback(
                query=query,
                query_embedding=query_embedding,
                namespace_id=namespace_id,
                temporal_filter=temporal_filter,
                limit=limit,
                routing=routing,
            )

    async def _vectorcypher_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
        graph_depth: int | None,
        limit: int,
        routing: RoutingDecision,
    ) -> VectorCypherResult:
        """Internal VectorCypher retrieval with graph traversal.

        This is the main VectorCypher path that combines vector and graph search.
        Separated from retrieve() to enable clean fallback handling.

        Implements adaptive depth: adjusts graph traversal depth based on the
        number of entry entities found. More entities = shallower depth (to avoid
        explosion), fewer entities = deeper depth (to find more context).
        """
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
            )

        # Compute adaptive depth based on entry entity count
        # This prevents explosion when many entities are found
        depth = self._router.compute_adaptive_depth(
            entry_entity_count=len(entry_entities),
            base_depth=base_depth,
        )

        # Step 4: Cypher expand to find related entities
        expanded_entities, entity_info_map = await self._cypher_expand(
            entry_entity_ids=[e[0] for e in entry_entities],
            namespace_id=namespace_id,
            depth=depth,
        )

        # Step 5: Fetch chunks from all entities
        all_entity_ids = list({e[0] for e in entry_entities} | expanded_entities.keys())

        graph_chunks = await self._fetch_chunks_from_entities(
            entity_ids=all_entity_ids,
            namespace_id=namespace_id,
            temporal_filter=temporal_filter,
            limit=limit * 2,  # Fetch more for fusion
        )

        # Step 6: Wait for parallel vector chunk search to complete
        # This was started at the beginning and may already be done
        vector_chunks = await vector_chunks_task

        # Step 7: RRF fusion with score normalization and dynamic weights
        fused_results = self._fuse_results(
            vector_chunks=vector_chunks,
            graph_chunks=graph_chunks,
            use_normalization=True,
            routing=routing,
        )

        # Step 8: Apply recency boost if temporal data available
        if self._config.recency_weight > 0:
            recency_scores = self._calculate_recency_scores(fused_results)
            fused_results = apply_recency_boost(
                fused_results,
                recency_scores,
                recency_weight=self._config.recency_weight,
            )

        # Normalize scores
        fused_results = normalize_scores(fused_results)

        # Build result
        chunk_results = [(r.item, r.rrf_score) for r in fused_results[:limit]]

        # Build entity results with name/type from graph neighborhoods
        entity_results = []
        for eid, score in entry_entities[: self._config.max_entities]:
            info = entity_info_map.get(str(eid), {})
            entity_results.append(
                (
                    {
                        "id": str(eid),
                        "name": info.get("name", ""),
                        "entity_type": info.get("entity_type", ""),
                        "score": score,
                    },
                    score,
                )
            )

        return VectorCypherResult(
            chunks=chunk_results,
            entities=entity_results,
            routing_decision=routing,
            metadata={
                "entry_entities": len(entry_entities),
                "expanded_entities": len(expanded_entities),
                "graph_depth": depth,
                "base_depth": base_depth,
                "adaptive_depth_applied": depth != base_depth,
                "total_chunks_before_fusion": len(graph_chunks) + len(vector_chunks),
                "routing_confidence": routing.confidence,
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
    ) -> VectorCypherResult:
        """Simple retrieval path - vector search only."""
        results = await self._vector_store.search(
            namespace_id=namespace_id,
            query_embedding=query_embedding,
            limit=limit,
            temporal_filter=temporal_filter,
            hybrid_alpha=0.7,  # Default hybrid
            query_text=query,
        )

        chunk_results = []
        for r in results:
            chunk_dict = {
                "id": str(r.chunk.id),
                "content": r.chunk.content,
                "document_id": str(r.chunk.document_id),
                "occurred_at": r.chunk.occurred_at.isoformat() if r.chunk.occurred_at else None,
                "metadata": r.chunk.metadata,
            }
            chunk_results.append((chunk_dict, r.combined_score or r.similarity))

        return VectorCypherResult(
            chunks=chunk_results,
            entities=[],
            routing_decision=routing,
            metadata={"search_mode": "simple_vector"},
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

        try:
            return await self._storage.search_similar_entities(
                namespace_id,
                query_embedding,
                limit=limit,
                min_similarity=0.3,
            )
        except Exception as e:
            logger.warning(f"Entity vector search failed: {e}")
            return []

    async def _cypher_expand(
        self,
        entry_entity_ids: list[UUID],
        namespace_id: UUID,
        depth: int,
    ) -> tuple[dict[UUID, float], dict[str, dict[str, str]]]:
        """Expand entry entities to find related entities via graph traversal.

        Args:
            entry_entity_ids: Starting entity IDs
            namespace_id: Namespace constraint
            depth: Maximum traversal depth

        Returns:
            Tuple of:
            - Dict mapping entity_id -> relevance score
            - Dict mapping entity_id_str -> {name, entity_type} for all discovered entities
        """
        if not entry_entity_ids:
            return {}, {}

        depth = min(max(1, depth), self._config.max_depth)

        # Get neighborhoods from dual node manager
        neighborhoods = await self._dual_nodes.get_entity_neighborhoods(
            entity_ids=entry_entity_ids,
            namespace_id=namespace_id,
            depth=depth,
            limit_per_entity=20,
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

                # Capture name and type (zero-cost, data already fetched)
                if entity_info["id"] not in entity_info_map:
                    entity_info_map[entity_info["id"]] = {
                        "name": entity_info.get("name", ""),
                        "entity_type": entity_info.get("entity_type", ""),
                    }

        return entity_scores, entity_info_map

    async def _fetch_chunks_from_entities(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
        limit: int,
    ) -> list[tuple[UUID, float, dict[str, Any]]]:
        """Fetch chunks connected to entities via MENTIONED_IN.

        Args:
            entity_ids: Entity IDs to fetch chunks for
            namespace_id: Namespace constraint
            temporal_filter: Optional temporal constraints
            limit: Maximum chunks to return

        Returns:
            List of (chunk_id, score, chunk_data) tuples
        """
        chunk_records = await self._dual_nodes.get_chunks_by_entities(
            entity_ids=entity_ids,
            namespace_id=namespace_id,
            temporal_filter=temporal_filter,
            limit=limit,
        )

        results = []
        for record in chunk_records:
            chunk_id = UUID(record["chunk_id"])
            # Score based on mention count and entity coverage
            score = float(record.get("total_mentions", 1))
            entity_count = len(record.get("entity_ids", []))
            score = score * (1 + 0.1 * entity_count)  # Boost for multiple entity connections

            chunk_data = {
                "id": record["chunk_id"],
                "content": record["content"],
                "document_id": record["document_id"],
                "occurred_at": record.get("occurred_at"),
                "metadata": record.get("metadata", {}),
                "connected_entities": record.get("entity_ids", []),
            }
            results.append((chunk_id, score, chunk_data))

        return results

    async def _vector_search_chunks(
        self,
        query_embedding: list[float],
        namespace_id: UUID,
        temporal_filter: TemporalFilter | None,
        query_text: str,
        limit: int,
    ) -> list[tuple[UUID, float, dict[str, Any]]]:
        """Direct vector search on chunks via pgvector.

        Args:
            query_embedding: Query embedding
            namespace_id: Namespace to search
            temporal_filter: Temporal constraints
            query_text: Original query text for hybrid search
            limit: Maximum results

        Returns:
            List of (chunk_id, score, chunk_data) tuples
        """
        results = await self._vector_store.search(
            namespace_id=namespace_id,
            query_embedding=query_embedding,
            limit=limit,
            temporal_filter=temporal_filter,
            hybrid_alpha=0.7,
            query_text=query_text,
        )

        return [
            (
                r.chunk.id,
                r.combined_score or r.similarity,
                {
                    "id": str(r.chunk.id),
                    "content": r.chunk.content,
                    "document_id": str(r.chunk.document_id),
                    "occurred_at": r.chunk.occurred_at.isoformat() if r.chunk.occurred_at else None,
                    "metadata": r.chunk.metadata,
                },
            )
            for r in results
        ]

    def _fuse_results(
        self,
        vector_chunks: list[tuple[UUID, float, dict[str, Any]]],
        graph_chunks: list[tuple[UUID, float, dict[str, Any]]],
        *,
        use_normalization: bool = False,
        routing: RoutingDecision | None = None,
    ) -> list[FusedResult]:
        """Fuse vector and graph results using weighted RRF.

        Args:
            vector_chunks: Results from vector search
            graph_chunks: Results from graph traversal
            use_normalization: If True, normalize scores before fusion for better ranking
            routing: If provided, adjust weights based on query complexity

        Returns:
            Fused and sorted results
        """
        # Dynamic fusion weights based on query complexity
        vector_weight = self._config.vector_weight
        graph_weight = self._config.graph_weight
        if routing is not None:
            if routing.complexity == QueryComplexity.SIMPLE:
                vector_weight, graph_weight = 0.8, 0.2
            elif routing.complexity == QueryComplexity.COMPLEX:
                vector_weight, graph_weight = 0.4, 0.6

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
    ) -> dict[UUID, float]:
        """Calculate recency scores for temporal boosting.

        Args:
            results: Fused results with items containing occurred_at

        Returns:
            Dict mapping item_id -> recency score (0-1)
        """
        now = datetime.now(UTC)
        decay_days = self._config.recency_decay_days
        scores: dict[UUID, float] = {}

        for r in results:
            item = r.item
            if isinstance(item, dict):
                occurred_at_str = item.get("occurred_at")
                if occurred_at_str:
                    try:
                        occurred_at = datetime.fromisoformat(occurred_at_str.replace("Z", "+00:00"))
                        if occurred_at.tzinfo is None:
                            occurred_at = occurred_at.replace(tzinfo=UTC)
                        days_old = (now - occurred_at).days
                        # Exponential decay
                        recency = max(0.0, 1.0 - (days_old / decay_days))
                        scores[r.item_id] = recency
                    except (ValueError, TypeError):
                        scores[r.item_id] = 0.5  # Default for unparseable dates
                else:
                    scores[r.item_id] = 0.5  # Default for missing dates
            else:
                scores[r.item_id] = 0.5

        return scores


__all__ = [
    "RetrieverConfig",
    "VectorCypherResult",
    "VectorCypherRetriever",
]
