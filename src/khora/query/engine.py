"""Hybrid query engine for Khora Memory Lake.

Combines vector search, graph traversal, and keyword search
with configurable fusion weights.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from .fusion import reciprocal_rank_fusion
from .temporal import TemporalFilter, TemporalQuery

if TYPE_CHECKING:
    from khora.acl import ACLContext
    from khora.core.models import Chunk, Entity
    from khora.extraction.embedders import Embedder
    from khora.storage import StorageCoordinator


class SearchMode(Enum):
    """Search mode for the query engine."""

    VECTOR = auto()  # Vector similarity only
    GRAPH = auto()  # Graph traversal only
    HYBRID = auto()  # Combine vector and graph
    ALL = auto()  # Vector, graph, and keyword


@dataclass
class QueryResult:
    """Result from a query."""

    chunks: list[tuple[Chunk, float]] = field(default_factory=list)
    entities: list[tuple[Entity, float]] = field(default_factory=list)
    graph_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_chunks(self) -> list[Chunk]:
        """Get top chunks without scores."""
        return [chunk for chunk, _ in self.chunks]

    @property
    def top_entities(self) -> list[Entity]:
        """Get top entities without scores."""
        return [entity for entity, _ in self.entities]

    def get_context_text(self, max_chunks: int = 5) -> str:
        """Get concatenated text from top chunks for LLM context."""
        texts = []
        for chunk, score in self.chunks[:max_chunks]:
            texts.append(chunk.content)
        return "\n\n---\n\n".join(texts)


@dataclass
class QueryConfig:
    """Configuration for query execution."""

    # Search mode
    mode: SearchMode = SearchMode.HYBRID

    # Result limits
    max_chunks: int = 10
    max_entities: int = 10
    max_graph_depth: int = 2

    # Similarity thresholds
    min_chunk_similarity: float = 0.5
    min_entity_similarity: float = 0.5

    # Fusion weights
    vector_weight: float = 0.6
    graph_weight: float = 0.3
    keyword_weight: float = 0.1

    # RRF parameter
    rrf_k: int = 60

    # Temporal settings
    apply_recency_bias: bool = False
    recency_weight: float = 0.2
    recency_decay_days: float = 30.0


class HybridQueryEngine:
    """Hybrid query engine combining multiple search methods.

    Supports:
    - Vector similarity search on chunks and entities
    - Graph traversal for related entities
    - Keyword search (via pgvector full-text)
    - Reciprocal Rank Fusion for combining results
    - Temporal filtering and recency bias
    """

    def __init__(
        self,
        storage: StorageCoordinator,
        embedder: Embedder | None = None,
        config: QueryConfig | None = None,
    ) -> None:
        """Initialize the query engine.

        Args:
            storage: StorageCoordinator for data access
            embedder: Embedder for query embedding
            config: Query configuration
        """
        self._storage = storage
        self._embedder = embedder
        self._config = config or QueryConfig()

    async def query(
        self,
        query_text: str,
        namespace_id: UUID,
        *,
        config: QueryConfig | None = None,
        temporal_filter: TemporalFilter | None = None,
        context: ACLContext | None = None,
    ) -> QueryResult:
        """Execute a hybrid query.

        Args:
            query_text: Query text
            namespace_id: Namespace to search in
            config: Optional query config override
            temporal_filter: Optional temporal filter
            context: Optional ACL context for permission filtering

        Returns:
            QueryResult with matched chunks and entities
        """
        cfg = config or self._config

        logger.debug(f"Executing query: {query_text[:50]}... (mode={cfg.mode.name})")

        # Get query embedding
        query_embedding = None
        if self._embedder and cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
            query_embedding = await self._embedder.embed(query_text)

        # Execute searches in parallel based on mode
        tasks = []

        if cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL) and query_embedding:
            tasks.append(self._vector_search(namespace_id, query_embedding, cfg))

        if cfg.mode in (SearchMode.GRAPH, SearchMode.HYBRID, SearchMode.ALL):
            tasks.append(self._graph_search(namespace_id, query_text, query_embedding, cfg))

        if cfg.mode == SearchMode.ALL:
            tasks.append(self._keyword_search(namespace_id, query_text, cfg))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        chunk_results: dict[str, list[tuple[Any, float]]] = {}
        entity_results: dict[str, list[tuple[Any, float]]] = {}
        graph_context = {}

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search {i} failed: {result}")
                continue

            if isinstance(result, dict):
                if "chunks" in result:
                    source = result.get("source", f"search_{i}")
                    chunk_results[source] = result["chunks"]
                if "entities" in result:
                    source = result.get("source", f"search_{i}")
                    entity_results[source] = result["entities"]
                if "graph_context" in result:
                    graph_context.update(result["graph_context"])

        # Apply RRF fusion
        fused_chunks = []
        if chunk_results:
            weights = {
                "vector": cfg.vector_weight,
                "graph": cfg.graph_weight,
                "keyword": cfg.keyword_weight,
            }
            fused_chunks = reciprocal_rank_fusion(
                chunk_results,
                k=cfg.rrf_k,
                weights=weights,
                id_extractor=lambda c: str(c.id),
            )

        fused_entities = []
        if entity_results:
            weights = {
                "vector": cfg.vector_weight,
                "graph": cfg.graph_weight,
            }
            fused_entities = reciprocal_rank_fusion(
                entity_results,
                k=cfg.rrf_k,
                weights=weights,
                id_extractor=lambda e: str(e.id),
            )

        # Apply temporal filter
        if temporal_filter:
            fused_chunks = [(c, s) for c, s in fused_chunks if temporal_filter.matches(c.created_at)]

        # Apply recency bias
        if cfg.apply_recency_bias:
            temporal_query = TemporalQuery(query_text).with_recency_bias(
                cfg.recency_weight,
                cfg.recency_decay_days,
            )
            fused_chunks = [(c, s * temporal_query.calculate_recency_score(c.created_at)) for c, s in fused_chunks]
            # Re-sort after recency adjustment
            fused_chunks.sort(key=lambda x: x[1], reverse=True)

        # Limit results
        fused_chunks = fused_chunks[: cfg.max_chunks]
        fused_entities = fused_entities[: cfg.max_entities]

        return QueryResult(
            chunks=fused_chunks,
            entities=fused_entities,
            graph_context=graph_context,
            metadata={
                "query": query_text,
                "mode": cfg.mode.name,
                "namespace_id": str(namespace_id),
            },
        )

    async def _vector_search(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        config: QueryConfig,
    ) -> dict[str, Any]:
        """Perform vector similarity search."""
        # Search chunks
        chunk_results = await self._storage.search_similar_chunks(
            namespace_id,
            query_embedding,
            limit=config.max_chunks * 2,  # Get extra for fusion
            min_similarity=config.min_chunk_similarity,
        )

        # Search entities
        entity_ids_scores = await self._storage.search_similar_entities(
            namespace_id,
            query_embedding,
            limit=config.max_entities * 2,
            min_similarity=config.min_entity_similarity,
        )

        # Fetch full entities
        entities = []
        for entity_id, score in entity_ids_scores:
            entity = await self._storage.get_entity(entity_id)
            if entity:
                entities.append((entity, score))

        return {
            "source": "vector",
            "chunks": chunk_results,
            "entities": entities,
        }

    async def _graph_search(
        self,
        namespace_id: UUID,
        query_text: str,
        query_embedding: list[float] | None,
        config: QueryConfig,
    ) -> dict[str, Any]:
        """Perform graph-based search."""
        entities = []
        graph_context = {}

        # If we have embedding, find similar entities first
        if query_embedding:
            entity_ids_scores = await self._storage.search_similar_entities(
                namespace_id,
                query_embedding,
                limit=5,  # Top entities for graph expansion
                min_similarity=config.min_entity_similarity,
            )

            # Expand neighborhood for top entities
            for entity_id, score in entity_ids_scores[:3]:
                entity = await self._storage.get_entity(entity_id)
                if entity:
                    entities.append((entity, score))

                    # Get neighborhood
                    neighborhood = await self._storage.get_neighborhood(
                        entity_id,
                        depth=config.max_graph_depth,
                        limit=20,
                    )
                    graph_context[str(entity_id)] = neighborhood

        # Get related chunks through entities
        chunks = []
        for entity, score in entities:
            # Get chunks that mention this entity
            for chunk_id in entity.source_chunk_ids[:5]:
                chunk = await self._storage.get_chunk(chunk_id)
                if chunk:
                    # Score based on entity score and mention count
                    chunk_score = score * (1 + 0.1 * entity.mention_count)
                    chunks.append((chunk, chunk_score))

        return {
            "source": "graph",
            "chunks": chunks,
            "entities": entities,
            "graph_context": graph_context,
        }

    async def _keyword_search(
        self,
        namespace_id: UUID,
        query_text: str,
        config: QueryConfig,
    ) -> dict[str, Any]:
        """Perform keyword-based search.

        Note: This is a placeholder. Full implementation would use
        PostgreSQL full-text search or similar.
        """
        # For now, return empty results
        # Full implementation would query documents/chunks with full-text search
        return {
            "source": "keyword",
            "chunks": [],
            "entities": [],
        }

    async def find_related_entities(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity through the graph.

        Args:
            entity_id: Starting entity
            namespace_id: Namespace to search in
            max_depth: Maximum relationship depth
            limit: Maximum entities to return

        Returns:
            List of (entity, relevance_score) tuples
        """
        neighborhood = await self._storage.get_neighborhood(
            entity_id,
            depth=max_depth,
            limit=limit,
        )

        entities = []
        for node in neighborhood.get("entities", []):
            entity = await self._storage.get_entity(UUID(node["id"]))
            if entity:
                # Score based on path length (shorter = higher score)
                # This is simplified - full impl would consider actual path lengths
                score = 1.0 / (1 + len(neighborhood.get("relationships", [])))
                entities.append((entity, score))

        return entities

    async def temporal_query(
        self,
        query: TemporalQuery,
        namespace_id: UUID,
        *,
        config: QueryConfig | None = None,
    ) -> QueryResult:
        """Execute a query with temporal context.

        Args:
            query: TemporalQuery with filters and settings
            namespace_id: Namespace to search in
            config: Optional query config override

        Returns:
            QueryResult with temporal filtering applied
        """
        cfg = config or QueryConfig()

        # Apply temporal settings to config
        if query.recency_weight > 0:
            cfg.apply_recency_bias = True
            cfg.recency_weight = query.recency_weight
            cfg.recency_decay_days = query.decay_days

        # Get context filter
        temporal_filter = None
        if query.filters:
            temporal_filter = query.filters[0]  # Use first filter for now
        elif query.context_window_days:
            temporal_filter = query.get_context_filter()

        return await self.query(
            query.query,
            namespace_id,
            config=cfg,
            temporal_filter=temporal_filter,
        )
