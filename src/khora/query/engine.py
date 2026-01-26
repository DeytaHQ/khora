"""Hybrid query engine for Khora Memory Lake.

Combines vector search, graph traversal, and keyword search
with configurable fusion weights. Now enhanced with:
- LLM-based query understanding
- Entity linking
- BM25 keyword search
- Neural reranking
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from .fusion import reciprocal_rank_fusion
from .keyword import KeywordSearcher, normalize_bm25_score
from .linking import EntityLinker, LinkingResult
from .reranking import RerankCandidate, create_reranker
from .temporal import TemporalFilter, TemporalQuery
from .understanding import QueryUnderstanding, UnderstandingResult

if TYPE_CHECKING:
    from khora.acl import ACLContext
    from khora.config.llm import LiteLLMConfig
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
    min_chunk_similarity: float = 0.3
    min_entity_similarity: float = 0.3

    # Fusion weights
    vector_weight: float = 0.5
    graph_weight: float = 0.3
    keyword_weight: float = 0.2

    # RRF parameter
    rrf_k: int = 60

    # Temporal settings
    apply_recency_bias: bool = False
    recency_weight: float = 0.2
    recency_decay_days: float = 30.0

    # Query understanding settings
    enable_query_understanding: bool = True
    enable_query_expansion: bool = True
    enable_entity_extraction: bool = True
    enable_temporal_detection: bool = True

    # Entity linking settings
    enable_entity_linking: bool = True
    entity_linking_fuzzy_threshold: float = 0.8
    entity_linking_embedding_threshold: float = 0.7
    entity_linking_max_candidates: int = 5

    # Reranking settings
    enable_reranking: bool = False
    reranking_method: str = "cross_encoder"
    reranking_top_n: int = 50
    reranking_final_k: int = 10

    # Keyword search settings
    enable_keyword_search: bool = True
    keyword_search_method: str = "bm25"

    @classmethod
    def from_settings(cls, settings: Any) -> QueryConfig:
        """Create QueryConfig from QuerySettings.

        Args:
            settings: QuerySettings from KhoraConfig

        Returns:
            QueryConfig instance
        """
        mode_map = {
            "vector": SearchMode.VECTOR,
            "graph": SearchMode.GRAPH,
            "hybrid": SearchMode.HYBRID,
            "all": SearchMode.ALL,
        }

        return cls(
            mode=mode_map.get(settings.default_mode.lower(), SearchMode.HYBRID),
            min_chunk_similarity=settings.min_chunk_similarity,
            min_entity_similarity=settings.min_entity_similarity,
            vector_weight=settings.vector_weight,
            graph_weight=settings.graph_weight,
            keyword_weight=settings.keyword_weight,
            apply_recency_bias=settings.apply_recency_bias,
            recency_weight=settings.recency_weight,
            recency_decay_days=settings.recency_decay_days,
            # Query understanding
            enable_query_understanding=settings.understanding.enabled,
            enable_query_expansion=settings.understanding.expand_query,
            enable_entity_extraction=settings.understanding.extract_entities,
            enable_temporal_detection=settings.understanding.detect_temporal,
            # Entity linking
            enable_entity_linking=settings.entity_linking.enabled,
            entity_linking_fuzzy_threshold=settings.entity_linking.fuzzy_threshold,
            entity_linking_embedding_threshold=settings.entity_linking.embedding_threshold,
            entity_linking_max_candidates=settings.entity_linking.max_candidates,
            # Reranking
            enable_reranking=settings.reranking.enabled,
            reranking_method=settings.reranking.method,
            reranking_top_n=settings.reranking.top_n,
            reranking_final_k=settings.reranking.final_k,
            # Keyword search
            enable_keyword_search=settings.keyword_search.enabled,
            keyword_search_method=settings.keyword_search.method,
        )


class HybridQueryEngine:
    """Hybrid query engine combining multiple search methods.

    Supports:
    - Vector similarity search on chunks and entities
    - Graph traversal for related entities
    - BM25 keyword search
    - Reciprocal Rank Fusion for combining results
    - Temporal filtering and recency bias
    - LLM-based query understanding
    - Entity linking
    - Neural reranking
    """

    def __init__(
        self,
        storage: StorageCoordinator,
        embedder: Embedder | None = None,
        config: QueryConfig | None = None,
        llm_config: LiteLLMConfig | None = None,
    ) -> None:
        """Initialize the query engine.

        Args:
            storage: StorageCoordinator for data access
            embedder: Embedder for query embedding
            config: Query configuration
            llm_config: LLM configuration for understanding/reranking
        """
        self._storage = storage
        self._embedder = embedder
        self._config = config or QueryConfig()
        self._llm_config = llm_config

        # Initialize query understanding
        self._query_understanding = QueryUnderstanding(llm_config=llm_config)

        # Entity linker (created per-query with embedder)
        self._entity_linker: EntityLinker | None = None

        # Keyword searcher (built per namespace)
        self._keyword_searchers: dict[str, KeywordSearcher] = {}

    async def query(
        self,
        query_text: str,
        namespace_id: UUID,
        *,
        config: QueryConfig | None = None,
        temporal_filter: TemporalFilter | None = None,
        context: ACLContext | None = None,
    ) -> QueryResult:
        """Execute a hybrid query with optional enhanced pipeline.

        The query pipeline:
        1. Query Understanding (optional) - Extract intent, entities, temporal refs
        2. Entity Linking (optional) - Link mentions to stored entities
        3. Multi-source Search - Vector, graph, keyword (BM25)
        4. RRF Fusion - Combine results
        5. Temporal Filtering - Apply time constraints
        6. Reranking (optional) - Neural re-ranking
        7. Final Limiting - Return top results

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

        # Initialize metadata
        metadata: dict[str, Any] = {
            "query": query_text,
            "mode": cfg.mode.name,
            "namespace_id": str(namespace_id),
        }

        # Step 1: Query Understanding
        understanding: UnderstandingResult | None = None
        if cfg.enable_query_understanding:
            try:
                understanding = await self._query_understanding.understand(
                    query_text,
                    expand_query=cfg.enable_query_expansion,
                    extract_entities=cfg.enable_entity_extraction,
                    detect_temporal=cfg.enable_temporal_detection,
                )
                metadata["understanding"] = {
                    "intent": understanding.intent.name,
                    "entities": [e.name for e in understanding.entities],
                    "temporal": understanding.has_temporal,
                    "expanded_queries": understanding.expanded_queries,
                    "keywords": understanding.keywords,
                }
                logger.debug(
                    f"Query understanding: intent={understanding.intent.name}, entities={len(understanding.entities)}"
                )
            except Exception as e:
                logger.warning(f"Query understanding failed: {e}")

        # Step 2: Entity Linking
        linking_result: LinkingResult | None = None
        linked_entity_ids: list[UUID] = []
        if cfg.enable_entity_linking and understanding and understanding.entities:
            try:
                linker = EntityLinker(
                    self._storage,
                    self._embedder,
                    fuzzy_threshold=cfg.entity_linking_fuzzy_threshold,
                    embedding_threshold=cfg.entity_linking_embedding_threshold,
                    max_candidates=cfg.entity_linking_max_candidates,
                )
                linking_result = await linker.link(understanding.entities, namespace_id)
                linked_entity_ids = linking_result.get_linked_entity_ids()
                metadata["entity_linking"] = {
                    "total_mentions": linking_result.total_mentions,
                    "linked_count": linking_result.linked_count,
                    "success_rate": linking_result.success_rate,
                }
                logger.debug(f"Entity linking: {linking_result.linked_count}/{linking_result.total_mentions} linked")
            except Exception as e:
                logger.warning(f"Entity linking failed: {e}")

        # Determine queries to search (original + expansions)
        queries_to_search = [query_text]
        if understanding and cfg.enable_query_expansion:
            queries_to_search.extend(understanding.expanded_queries[:2])  # Limit expansions

        # Step 3: Execute searches
        all_chunk_results: dict[str, list[tuple[Any, float]]] = {}
        all_entity_results: dict[str, list[tuple[Any, float]]] = {}
        graph_context: dict[str, Any] = {}

        for i, q in enumerate(queries_to_search):
            suffix = "" if i == 0 else f"_exp{i}"

            # Get query embedding
            query_embedding = None
            if self._embedder and cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
                query_embedding = await self._embedder.embed(q)

            # Execute searches in parallel based on mode
            tasks = []

            if cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL) and query_embedding:
                tasks.append(self._vector_search(namespace_id, query_embedding, cfg))

            if cfg.mode in (SearchMode.GRAPH, SearchMode.HYBRID, SearchMode.ALL):
                tasks.append(self._graph_search(namespace_id, q, query_embedding, cfg, linked_entity_ids))

            if cfg.mode == SearchMode.ALL and cfg.enable_keyword_search:
                # Use BM25 keyword search with extracted keywords if available
                keywords = understanding.keywords if understanding else None
                tasks.append(self._keyword_search_bm25(namespace_id, q, cfg, keywords))

            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Search {j} failed: {result}")
                    continue

                if isinstance(result, dict):
                    if "chunks" in result:
                        source = result.get("source", f"search_{j}") + suffix
                        all_chunk_results[source] = result["chunks"]
                    if "entities" in result:
                        source = result.get("source", f"search_{j}") + suffix
                        all_entity_results[source] = result["entities"]
                    if "graph_context" in result:
                        graph_context.update(result["graph_context"])

        # Step 4: Apply RRF fusion
        fused_chunks = []
        if all_chunk_results:
            weights = {
                "vector": cfg.vector_weight,
                "graph": cfg.graph_weight,
                "keyword": cfg.keyword_weight,
            }
            # Add weights for expanded query results
            for key in all_chunk_results:
                if "_exp" in key:
                    base_source = key.split("_exp")[0]
                    weights[key] = weights.get(base_source, cfg.vector_weight) * 0.7  # Discount expansions

            fused_chunks = reciprocal_rank_fusion(
                all_chunk_results,
                k=cfg.rrf_k,
                weights=weights,
                id_extractor=lambda c: str(c.id),
            )

        fused_entities = []
        if all_entity_results:
            weights = {
                "vector": cfg.vector_weight,
                "graph": cfg.graph_weight,
            }
            fused_entities = reciprocal_rank_fusion(
                all_entity_results,
                k=cfg.rrf_k,
                weights=weights,
                id_extractor=lambda e: str(e.id),
            )

        # Boost linked entities
        if linked_entity_ids:
            boosted_entities = []
            for entity, score in fused_entities:
                if entity.id in linked_entity_ids:
                    boosted_entities.append((entity, score * 1.5))  # 50% boost
                else:
                    boosted_entities.append((entity, score))
            fused_entities = sorted(boosted_entities, key=lambda x: x[1], reverse=True)

        # Step 5: Apply temporal filter
        if temporal_filter:
            fused_chunks = [(c, s) for c, s in fused_chunks if temporal_filter.matches(c.created_at)]

        # Apply recency bias
        if cfg.apply_recency_bias:
            temporal_query = TemporalQuery(query_text).with_recency_bias(
                cfg.recency_weight,
                cfg.recency_decay_days,
            )
            fused_chunks = [(c, s * temporal_query.calculate_recency_score(c.created_at)) for c, s in fused_chunks]
            fused_chunks.sort(key=lambda x: x[1], reverse=True)

        # Step 6: Reranking (optional)
        if cfg.enable_reranking and fused_chunks:
            try:
                reranker = create_reranker(
                    method=cfg.reranking_method,
                    llm_config=self._llm_config,
                )
                candidates = [
                    RerankCandidate(
                        item=chunk,
                        original_score=score,
                        content=chunk.content,
                        metadata=chunk.metadata,
                    )
                    for chunk, score in fused_chunks[: cfg.reranking_top_n]
                ]
                reranked = await reranker.rerank(query_text, candidates, top_k=cfg.reranking_final_k)
                fused_chunks = [(r.item, r.final_score) for r in reranked]
                metadata["reranking"] = {"method": cfg.reranking_method, "reranked_count": len(fused_chunks)}
                logger.debug(f"Reranked {len(candidates)} candidates to {len(fused_chunks)} results")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        # Step 7: Limit results
        fused_chunks = fused_chunks[: cfg.max_chunks]
        fused_entities = fused_entities[: cfg.max_entities]

        return QueryResult(
            chunks=fused_chunks,
            entities=fused_entities,
            graph_context=graph_context,
            metadata=metadata,
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
        linked_entity_ids: list[UUID] | None = None,
    ) -> dict[str, Any]:
        """Perform graph-based search.

        Args:
            namespace_id: Namespace to search in
            query_text: Query text
            query_embedding: Query embedding (optional)
            config: Query configuration
            linked_entity_ids: Entity IDs from entity linking (optional)

        Returns:
            Dict with chunks, entities, and graph context
        """
        entities = []
        graph_context = {}
        seen_entity_ids = set()

        # Start with linked entities if available (high priority)
        if linked_entity_ids:
            for entity_id in linked_entity_ids[:5]:
                if entity_id in seen_entity_ids:
                    continue
                entity = await self._storage.get_entity(entity_id)
                if entity:
                    entities.append((entity, 1.0))  # High confidence from linking
                    seen_entity_ids.add(entity_id)

                    # Get neighborhood for linked entities
                    try:
                        neighborhood = await self._storage.get_neighborhood(
                            entity_id,
                            depth=config.max_graph_depth,
                            limit=20,
                        )
                        graph_context[str(entity_id)] = neighborhood
                    except Exception as e:
                        logger.debug(f"Failed to get neighborhood for {entity_id}: {e}")

        # Also find similar entities via embedding
        if query_embedding:
            entity_ids_scores = await self._storage.search_similar_entities(
                namespace_id,
                query_embedding,
                limit=5,
                min_similarity=config.min_entity_similarity,
            )

            # Expand neighborhood for top entities
            for entity_id, score in entity_ids_scores[:3]:
                if entity_id in seen_entity_ids:
                    continue
                entity = await self._storage.get_entity(entity_id)
                if entity:
                    entities.append((entity, score))
                    seen_entity_ids.add(entity_id)

                    # Get neighborhood
                    try:
                        neighborhood = await self._storage.get_neighborhood(
                            entity_id,
                            depth=config.max_graph_depth,
                            limit=20,
                        )
                        graph_context[str(entity_id)] = neighborhood
                    except Exception as e:
                        logger.debug(f"Failed to get neighborhood for {entity_id}: {e}")

        # Get related chunks through entities
        chunks = []
        seen_chunk_ids = set()
        for entity, score in entities:
            # Get chunks that mention this entity
            for chunk_id in entity.source_chunk_ids[:5]:
                if chunk_id in seen_chunk_ids:
                    continue
                chunk = await self._storage.get_chunk(chunk_id)
                if chunk:
                    # Score based on entity score and mention count
                    chunk_score = score * (1 + 0.1 * min(entity.mention_count, 10))
                    chunks.append((chunk, chunk_score))
                    seen_chunk_ids.add(chunk_id)

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
        """Perform keyword-based search (legacy, returns empty).

        Use _keyword_search_bm25 for actual BM25-based search.
        """
        return {
            "source": "keyword",
            "chunks": [],
            "entities": [],
        }

    async def _keyword_search_bm25(
        self,
        namespace_id: UUID,
        query_text: str,
        config: QueryConfig,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform BM25-based keyword search.

        Args:
            namespace_id: Namespace to search in
            query_text: Query text
            config: Query configuration
            keywords: Optional pre-extracted keywords from query understanding

        Returns:
            Dict with chunks and entities
        """
        ns_key = str(namespace_id)

        # Build or get keyword index for this namespace
        if ns_key not in self._keyword_searchers:
            try:
                # Fetch all chunks for the namespace (up to a limit)
                chunks = await self._storage.list_chunks(
                    namespace_id,
                    limit=10000,  # Reasonable limit for in-memory index
                )
                if chunks:
                    searcher = KeywordSearcher(
                        use_stemming=True,
                        remove_stopwords=True,
                    )
                    searcher.index_chunks(chunks)
                    self._keyword_searchers[ns_key] = searcher
                    logger.debug(f"Built BM25 index with {len(chunks)} chunks")
                else:
                    logger.debug("No chunks to index for keyword search")
                    return {"source": "keyword", "chunks": [], "entities": []}
            except Exception as e:
                logger.warning(f"Failed to build keyword index: {e}")
                return {"source": "keyword", "chunks": [], "entities": []}

        searcher = self._keyword_searchers.get(ns_key)
        if not searcher:
            return {"source": "keyword", "chunks": [], "entities": []}

        try:
            # Use keywords if available, otherwise use query text
            if keywords:
                results = searcher.search_with_keywords(
                    keywords,
                    limit=config.max_chunks * 2,
                    min_score=0.1,
                )
            else:
                results = searcher.search(
                    query_text,
                    limit=config.max_chunks * 2,
                    min_score=0.1,
                )

            # Normalize BM25 scores to 0-1 range
            normalized_results = [(chunk, normalize_bm25_score(score)) for chunk, score in results]

            return {
                "source": "keyword",
                "chunks": normalized_results,
                "entities": [],  # Keyword search doesn't directly find entities
            }
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return {"source": "keyword", "chunks": [], "entities": []}

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
