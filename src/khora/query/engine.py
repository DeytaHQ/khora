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
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from .fusion import reciprocal_rank_fusion
from .keyword import KeywordSearcher, normalize_bm25_score
from .linking import EntityLinker, LinkingResult
from .metrics import SearchMetrics
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
class SearchMethodStats:
    """Statistics for a single search method."""

    chunk_count: int = 0
    entity_count: int = 0
    chunk_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)

    # Score statistics
    min_score: float = 0.0
    max_score: float = 0.0
    avg_score: float = 0.0

    # Timing (in milliseconds)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunks": {
                "count": self.chunk_count,
                "ids": self.chunk_ids,  # All IDs for proper attribution
            },
            "entities": {
                "count": self.entity_count,
                "ids": self.entity_ids,  # All IDs for proper attribution
            },
            "scores": {
                "min": round(self.min_score, 4),
                "max": round(self.max_score, 4),
                "avg": round(self.avg_score, 4),
            },
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class SearchMethodContribution:
    """Tracks which search methods contributed to results with detailed statistics."""

    # Per-method statistics
    vector: SearchMethodStats = field(default_factory=SearchMethodStats)
    graph: SearchMethodStats = field(default_factory=SearchMethodStats)
    keyword: SearchMethodStats = field(default_factory=SearchMethodStats)

    # Overlap analysis
    vector_only_chunks: list[str] = field(default_factory=list)  # Chunks found ONLY by vector
    graph_only_chunks: list[str] = field(default_factory=list)  # Chunks found ONLY by graph
    keyword_only_chunks: list[str] = field(default_factory=list)  # Chunks found ONLY by keyword
    vector_graph_overlap: list[str] = field(default_factory=list)  # Chunks found by both vector AND graph
    vector_keyword_overlap: list[str] = field(default_factory=list)  # Chunks found by both vector AND keyword
    graph_keyword_overlap: list[str] = field(default_factory=list)  # Chunks found by both graph AND keyword
    all_methods_overlap: list[str] = field(default_factory=list)  # Chunks found by ALL methods

    # Entity overlap
    vector_only_entities: list[str] = field(default_factory=list)
    graph_only_entities: list[str] = field(default_factory=list)
    vector_graph_entity_overlap: list[str] = field(default_factory=list)

    # Total timing
    total_search_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0

    def compute_overlaps(self) -> None:
        """Compute overlap statistics from per-method chunk/entity lists."""
        vector_set = set(self.vector.chunk_ids)
        graph_set = set(self.graph.chunk_ids)
        keyword_set = set(self.keyword.chunk_ids)

        # Chunk overlaps
        self.vector_graph_overlap = list(vector_set & graph_set)
        self.vector_keyword_overlap = list(vector_set & keyword_set)
        self.graph_keyword_overlap = list(graph_set & keyword_set)
        self.all_methods_overlap = list(vector_set & graph_set & keyword_set)

        # Exclusive chunks
        self.vector_only_chunks = list(vector_set - graph_set - keyword_set)
        self.graph_only_chunks = list(graph_set - vector_set - keyword_set)
        self.keyword_only_chunks = list(keyword_set - vector_set - graph_set)

        # Entity overlaps
        vector_entities = set(self.vector.entity_ids)
        graph_entities = set(self.graph.entity_ids)
        self.vector_graph_entity_overlap = list(vector_entities & graph_entities)
        self.vector_only_entities = list(vector_entities - graph_entities)
        self.graph_only_entities = list(graph_entities - vector_entities)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with comprehensive statistics."""
        total_unique_chunks = len(set(self.vector.chunk_ids) | set(self.graph.chunk_ids) | set(self.keyword.chunk_ids))
        total_unique_entities = len(set(self.vector.entity_ids) | set(self.graph.entity_ids))

        return {
            "summary": {
                "total_unique_chunks": total_unique_chunks,
                "total_unique_entities": total_unique_entities,
                "total_search_latency_ms": round(self.total_search_latency_ms, 2),
                "fusion_latency_ms": round(self.fusion_latency_ms, 2),
            },
            "by_method": {
                "vector": self.vector.to_dict(),
                "graph": self.graph.to_dict(),
                "keyword": self.keyword.to_dict(),
            },
            "chunk_overlap": {
                "vector_only": {"count": len(self.vector_only_chunks), "ids": self.vector_only_chunks},
                "graph_only": {"count": len(self.graph_only_chunks), "ids": self.graph_only_chunks},
                "keyword_only": {"count": len(self.keyword_only_chunks), "ids": self.keyword_only_chunks},
                "vector_and_graph": {"count": len(self.vector_graph_overlap), "ids": self.vector_graph_overlap},
                "vector_and_keyword": {
                    "count": len(self.vector_keyword_overlap),
                    "ids": self.vector_keyword_overlap,
                },
                "graph_and_keyword": {"count": len(self.graph_keyword_overlap), "ids": self.graph_keyword_overlap},
                "all_three_methods": {"count": len(self.all_methods_overlap), "ids": self.all_methods_overlap},
            },
            "entity_overlap": {
                "vector_only": {"count": len(self.vector_only_entities), "ids": self.vector_only_entities},
                "graph_only": {"count": len(self.graph_only_entities), "ids": self.graph_only_entities},
                "vector_and_graph": {
                    "count": len(self.vector_graph_entity_overlap),
                    "ids": self.vector_graph_entity_overlap,
                },
            },
        }

    # Legacy property for backwards compatibility
    @property
    def vector_chunks(self) -> list[str]:
        return self.vector.chunk_ids

    @property
    def graph_chunks(self) -> list[str]:
        return self.graph.chunk_ids

    @property
    def keyword_chunks(self) -> list[str]:
        return self.keyword.chunk_ids


@dataclass
class GraphTraversalInfo:
    """Information about graph elements triggered during search."""

    entities_searched: list[str] = field(default_factory=list)  # Entity names searched
    entities_linked: list[str] = field(default_factory=list)  # Entities linked from query
    relationships_traversed: list[tuple[str, str, str]] = field(default_factory=list)  # (from, rel, to)
    neighborhood_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities_searched": self.entities_searched[:20],
            "entities_linked": self.entities_linked[:10],
            "relationships_traversed": [
                {"from": f, "relationship": r, "to": t} for f, r, t in self.relationships_traversed[:20]
            ],
            "neighborhood_depth": self.neighborhood_depth,
        }


@dataclass
class TemporalInfo:
    """Information about temporal filtering applied."""

    detected: bool = False
    filter_applied: bool = False
    time_start: Any = None  # datetime or None
    time_end: Any = None  # datetime or None
    reference_text: str = ""  # Original temporal reference like "last 7 days"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detected": self.detected,
            "filter_applied": self.filter_applied,
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "reference_text": self.reference_text,
        }


@dataclass
class QueryResult:
    """Result from a query with enhanced metadata."""

    chunks: list[tuple[Chunk, float]] = field(default_factory=list)
    entities: list[tuple[Entity, float]] = field(default_factory=list)
    graph_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Enhanced tracking
    search_contributions: SearchMethodContribution | None = None
    graph_info: GraphTraversalInfo | None = None
    temporal_info: TemporalInfo | None = None

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

    def get_full_metadata(self) -> dict[str, Any]:
        """Get complete metadata including search method contributions."""
        result = dict(self.metadata)
        if self.search_contributions:
            result["search_methods"] = self.search_contributions.to_dict()
        if self.graph_info:
            result["graph_traversal"] = self.graph_info.to_dict()
        if self.temporal_info:
            result["temporal"] = self.temporal_info.to_dict()
        return result


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
    enable_reranking: bool = True
    reranking_method: str = "cross_encoder"
    reranking_top_n: int = 50
    reranking_final_k: int = 10

    # Keyword search settings
    enable_keyword_search: bool = True
    keyword_search_method: str = "fulltext"

    # HyDE settings
    enable_hyde: bool = False
    hyde_num_hypotheticals: int = 1

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
            # HyDE
            enable_hyde=settings.enable_hyde,
            hyde_num_hypotheticals=settings.hyde_num_hypotheticals,
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

        # HyDE expander
        self._hyde_expander: HyDEExpander | None = None
        if self._config.enable_hyde and self._embedder:
            from .hyde import HyDEExpander

            self._hyde_expander = HyDEExpander(
                self._embedder,
                llm_config=llm_config,
                num_hypotheticals=self._config.hyde_num_hypotheticals,
            )

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
        agentic: bool = False,
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
            agentic: If True, use agentic two-step exploration

        Returns:
            QueryResult with matched chunks and entities
        """
        # Agentic mode - use two-step exploration agent
        if agentic:
            from .agentic import AgenticSearchAgent

            agent = AgenticSearchAgent(self, self._llm_config)
            agentic_result = await agent.search(query_text, namespace_id, config)

            # Convert to QueryResult
            return QueryResult(
                chunks=[(c, s) for c, s, _ in agentic_result.chunks],
                entities=agentic_result.entities,
                metadata={
                    "agentic": True,
                    "summary": agentic_result.summary,
                    "trace": agentic_result.trace.to_dict() if agentic_result.trace else None,
                    **agentic_result.metadata,
                },
            )

        cfg = config or self._config

        logger.debug(f"Executing query: {query_text[:50]}... (mode={cfg.mode.name})")

        # Initialize metrics
        metrics = SearchMetrics()
        metrics.total_timer.start()
        metrics.features = {
            "query_understanding": cfg.enable_query_understanding,
            "entity_linking": cfg.enable_entity_linking,
            "reranking": cfg.enable_reranking,
            "hyde": cfg.enable_hyde,
            "keyword_method": cfg.keyword_search_method,
        }

        # Initialize tracking objects
        search_contributions = SearchMethodContribution()
        graph_info = GraphTraversalInfo()
        temporal_info = TemporalInfo()

        # Initialize metadata
        metadata: dict[str, Any] = {
            "query": query_text,
            "mode": cfg.mode.name,
            "namespace_id": str(namespace_id),
        }

        # Step 1: Query Understanding
        metrics.understanding_timer.start()
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
                    "answer_type": understanding.answer_type.name,
                    "entities": [e.name for e in understanding.entities],
                    "entity_aliases": {e.name: e.aliases for e in understanding.entities if e.aliases},
                    "relationships": [
                        {"from": r.from_entity, "type": r.relationship_type, "to": r.to_entity}
                        for r in understanding.relationships
                    ],
                    "temporal": understanding.has_temporal,
                    "expanded_queries": understanding.expanded_queries,
                    "keywords": understanding.keywords,
                    "source_priority": {
                        "slack": understanding.source_priority.slack,
                        "linear": understanding.source_priority.linear,
                        "notion": understanding.source_priority.notion,
                        "attio": understanding.source_priority.attio,
                        "gong": understanding.source_priority.gong,
                        "github": understanding.source_priority.github,
                        "bamboohr": understanding.source_priority.bamboohr,
                    },
                    "search_strategy": {
                        "vector_weight": understanding.search_strategy.vector_weight,
                        "graph_weight": understanding.search_strategy.graph_weight,
                        "keyword_weight": understanding.search_strategy.keyword_weight,
                        "reasoning": understanding.search_strategy.strategy_reasoning,
                    },
                    "complexity_score": understanding.complexity_score,
                    "requires_multi_step": understanding.requires_multi_step,
                    "follow_up_queries": [fq.query for fq in understanding.follow_up_queries],
                    "reasoning": understanding.reasoning,
                }

                # Apply LLM-recommended search strategy weights
                if understanding.search_strategy:
                    cfg.vector_weight = understanding.search_strategy.vector_weight
                    cfg.graph_weight = understanding.search_strategy.graph_weight
                    cfg.keyword_weight = understanding.search_strategy.keyword_weight
                    cfg.max_graph_depth = understanding.search_strategy.graph_depth

                logger.debug(
                    f"Query understanding: intent={understanding.intent.name}, "
                    f"entities={len(understanding.entities)}, complexity={understanding.complexity_score:.2f}"
                )

                # Extract temporal information and create filter if detected
                if understanding.has_temporal and understanding.temporal_references:
                    temporal_info.detected = True
                    for temp_ref in understanding.temporal_references:
                        temporal_info.reference_text = temp_ref.text
                        if temp_ref.start_date:
                            temporal_info.time_start = temp_ref.start_date
                        if temp_ref.end_date:
                            temporal_info.time_end = temp_ref.end_date

                    # Create temporal filter if not already provided
                    if not temporal_filter and (temporal_info.time_start or temporal_info.time_end):
                        temporal_filter = TemporalFilter(
                            start_date=temporal_info.time_start,
                            end_date=temporal_info.time_end,
                        )
                        temporal_info.filter_applied = True
                        logger.debug(f"Temporal filter applied: {temporal_info.time_start} to {temporal_info.time_end}")

            except Exception as e:
                logger.warning(f"Query understanding failed: {e}")

        metrics.understanding_timer.stop()

        # Step 2: Entity Linking
        metrics.linking_timer.start()
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

                # Track linked entity names for graph info
                for linked in linking_result.linked_entities:
                    if linked.entity:
                        graph_info.entities_linked.append(linked.entity.name)

                metadata["entity_linking"] = {
                    "total_mentions": linking_result.total_mentions,
                    "linked_count": linking_result.linked_count,
                    "success_rate": linking_result.success_rate,
                    "linked_entities": graph_info.entities_linked,
                }
                logger.debug(f"Entity linking: {linking_result.linked_count}/{linking_result.total_mentions} linked")
            except Exception as e:
                logger.warning(f"Entity linking failed: {e}")

        metrics.linking_timer.stop()

        # Determine queries to search (original + expansions)
        queries_to_search = [query_text]
        if understanding and cfg.enable_query_expansion:
            queries_to_search.extend(understanding.expanded_queries[:2])  # Limit expansions

        # Step 3: Execute searches
        all_chunk_results: dict[str, list[tuple[Any, float]]] = {}
        all_entity_results: dict[str, list[tuple[Any, float]]] = {}
        graph_context: dict[str, Any] = {}

        metrics.search_timer.start()
        search_start_time = time.perf_counter()

        for i, q in enumerate(queries_to_search):
            suffix = "" if i == 0 else f"_exp{i}"

            # Get query embedding
            query_embedding = None
            if self._embedder and cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
                query_embedding = await self._embedder.embed(q)

                # Apply HyDE expansion (only on the original query, not expansions)
                if query_embedding is not None and self._hyde_expander and i == 0:
                    query_embedding = await self._hyde_expander.expand_query_embedding(q, query_embedding)
                    metadata["hyde_applied"] = True

            # Execute searches in parallel based on mode
            tasks = []
            task_types = []  # Track which task is which for timing

            if cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL) and query_embedding is not None:
                tasks.append(self._timed_search(self._vector_search(namespace_id, query_embedding, cfg), "vector"))
                task_types.append("vector")

            if cfg.mode in (SearchMode.GRAPH, SearchMode.HYBRID, SearchMode.ALL):
                tasks.append(
                    self._timed_search(
                        self._graph_search(namespace_id, q, query_embedding, cfg, linked_entity_ids), "graph"
                    )
                )
                task_types.append("graph")

            if cfg.mode == SearchMode.ALL and cfg.enable_keyword_search:
                keywords = understanding.keywords if understanding else None
                if cfg.keyword_search_method == "fulltext":
                    tasks.append(self._timed_search(self._keyword_search_fulltext(namespace_id, q, cfg), "keyword"))
                else:
                    tasks.append(
                        self._timed_search(self._keyword_search_bm25(namespace_id, q, cfg, keywords), "keyword")
                    )
                task_types.append("keyword")

            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and track contributions with detailed stats
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Search {j} failed: {result}")
                    continue

                if isinstance(result, dict):
                    source_type = result.get("source", f"search_{j}")
                    latency_ms = result.get("latency_ms", 0.0)

                    if "chunks" in result:
                        source = source_type + suffix
                        all_chunk_results[source] = result["chunks"]

                        # Track contributions by search method with detailed stats
                        chunk_ids = [str(c.id) for c, _ in result["chunks"]]
                        scores = [s for _, s in result["chunks"]]

                        if source_type == "vector":
                            search_contributions.vector.chunk_count += len(result["chunks"])
                            search_contributions.vector.chunk_ids.extend(chunk_ids)
                            search_contributions.vector.latency_ms = latency_ms
                            if scores:
                                search_contributions.vector.min_score = min(scores)
                                search_contributions.vector.max_score = max(scores)
                                search_contributions.vector.avg_score = sum(scores) / len(scores)
                        elif source_type == "graph":
                            search_contributions.graph.chunk_count += len(result["chunks"])
                            search_contributions.graph.chunk_ids.extend(chunk_ids)
                            search_contributions.graph.latency_ms = latency_ms
                            if scores:
                                search_contributions.graph.min_score = min(scores)
                                search_contributions.graph.max_score = max(scores)
                                search_contributions.graph.avg_score = sum(scores) / len(scores)
                        elif source_type == "keyword":
                            search_contributions.keyword.chunk_count += len(result["chunks"])
                            search_contributions.keyword.chunk_ids.extend(chunk_ids)
                            search_contributions.keyword.latency_ms = latency_ms
                            if scores:
                                search_contributions.keyword.min_score = min(scores)
                                search_contributions.keyword.max_score = max(scores)
                                search_contributions.keyword.avg_score = sum(scores) / len(scores)

                    if "entities" in result:
                        source = source_type + suffix
                        all_entity_results[source] = result["entities"]

                        # Track entity stats
                        entity_ids = [str(e.id) for e, _ in result["entities"]]
                        if source_type == "vector":
                            search_contributions.vector.entity_count += len(result["entities"])
                            search_contributions.vector.entity_ids.extend(entity_ids)
                        elif source_type == "graph":
                            search_contributions.graph.entity_count += len(result["entities"])
                            search_contributions.graph.entity_ids.extend(entity_ids)
                            # Also track entity names for graph info
                            for entity, _ in result["entities"]:
                                graph_info.entities_searched.append(entity.name)

                    if "graph_context" in result:
                        graph_context.update(result["graph_context"])

                        # Track relationships from graph context
                        if "relationships" in result.get("graph_context", {}):
                            for rel in result["graph_context"]["relationships"]:
                                if isinstance(rel, dict):
                                    graph_info.relationships_traversed.append(
                                        (rel.get("from", ""), rel.get("type", ""), rel.get("to", ""))
                                    )

        # Record total search time
        search_contributions.total_search_latency_ms = (time.perf_counter() - search_start_time) * 1000
        metrics.search_timer.stop()

        # Populate per-source counts into metrics
        metrics.vector_chunk_count = search_contributions.vector.chunk_count
        metrics.graph_chunk_count = search_contributions.graph.chunk_count
        metrics.keyword_chunk_count = search_contributions.keyword.chunk_count
        metrics.vector_entity_count = search_contributions.vector.entity_count
        metrics.graph_entity_count = search_contributions.graph.entity_count

        # Step 4: Apply RRF fusion
        metrics.fusion_timer.start()
        fusion_start_time = time.perf_counter()

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

        search_contributions.fusion_latency_ms = (time.perf_counter() - fusion_start_time) * 1000
        metrics.fusion_timer.stop()
        metrics.fused_chunk_count = len(fused_chunks)
        metrics.fused_entity_count = len(fused_entities)

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
        metrics.reranking_timer.start()
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

        metrics.reranking_timer.stop()

        # Step 7: Limit results
        fused_chunks = fused_chunks[: cfg.max_chunks]
        fused_entities = fused_entities[: cfg.max_entities]

        # Update graph info with depth used
        graph_info.neighborhood_depth = cfg.max_graph_depth

        # Compute overlap statistics
        search_contributions.compute_overlaps()

        # Finalize metrics
        metrics.final_chunk_count = len(fused_chunks)
        metrics.final_entity_count = len(fused_entities)
        metrics.set_chunk_scores([s for _, s in fused_chunks])
        metrics.total_timer.stop()
        metrics.log()

        # Add search method info to metadata
        metadata["search_methods"] = search_contributions.to_dict()
        metadata["graph_traversal"] = graph_info.to_dict()
        metadata["temporal"] = temporal_info.to_dict()
        metadata["metrics"] = metrics.to_dict()

        return QueryResult(
            chunks=fused_chunks,
            entities=fused_entities,
            graph_context=graph_context,
            metadata=metadata,
            search_contributions=search_contributions,
            graph_info=graph_info,
            temporal_info=temporal_info,
        )

    async def _timed_search(
        self,
        search_coro: Any,
        source_type: str,
    ) -> dict[str, Any]:
        """Wrap a search coroutine with timing instrumentation.

        Args:
            search_coro: The search coroutine to execute
            source_type: The type of search (vector, graph, keyword)

        Returns:
            Search result dict with latency_ms added
        """
        start_time = time.perf_counter()
        result = await search_coro
        latency_ms = (time.perf_counter() - start_time) * 1000

        if isinstance(result, dict):
            result["latency_ms"] = latency_ms
        return result

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

        # Fetch full entities in batch (optimization: single query instead of N queries)
        entity_ids = [eid for eid, _ in entity_ids_scores]
        entities_map = await self._storage.get_entities_batch(entity_ids)
        entities = [(entities_map[eid], score) for eid, score in entity_ids_scores if eid in entities_map]

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
        seen_entity_ids: set[UUID] = set()

        # Collect all entity IDs we need to fetch
        all_entity_ids_to_fetch: list[UUID] = []
        linked_scores: dict[UUID, float] = {}
        similar_scores: dict[UUID, float] = {}

        # Linked entities (high priority)
        if linked_entity_ids:
            for entity_id in linked_entity_ids[:5]:
                if entity_id not in seen_entity_ids:
                    all_entity_ids_to_fetch.append(entity_id)
                    linked_scores[entity_id] = 1.0  # High confidence from linking
                    seen_entity_ids.add(entity_id)

        # Similar entities via embedding
        if query_embedding is not None:
            entity_ids_scores = await self._storage.search_similar_entities(
                namespace_id,
                query_embedding,
                limit=5,
                min_similarity=config.min_entity_similarity,
            )

            for entity_id, score in entity_ids_scores[:3]:
                if entity_id not in seen_entity_ids:
                    all_entity_ids_to_fetch.append(entity_id)
                    similar_scores[entity_id] = score
                    seen_entity_ids.add(entity_id)

        # Batch fetch all entities and neighborhoods in parallel
        if all_entity_ids_to_fetch:
            # Fetch entities and neighborhoods in parallel
            entities_map, neighborhoods = await asyncio.gather(
                self._storage.get_entities_batch(all_entity_ids_to_fetch),
                self._storage.get_neighborhoods_batch(
                    all_entity_ids_to_fetch,
                    depth=config.max_graph_depth,
                    limit_per_entity=20,
                ),
            )

            # Process results maintaining priority order
            for entity_id in all_entity_ids_to_fetch:
                if entity_id in entities_map:
                    entity = entities_map[entity_id]
                    score = linked_scores.get(entity_id) or similar_scores.get(entity_id, 0.5)
                    entities.append((entity, score))

                    # Add neighborhood to context
                    if entity_id in neighborhoods:
                        graph_context[str(entity_id)] = neighborhoods[entity_id]

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

    async def _keyword_search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        config: QueryConfig,
    ) -> dict[str, Any]:
        """Perform PostgreSQL full-text search using tsvector/tsquery.

        Unlike BM25, this runs entirely in PostgreSQL using the GIN-indexed
        content_tsv column, with no chunk count limit.
        """
        try:
            results = await self._storage.search_fulltext_chunks(
                namespace_id,
                query_text,
                limit=config.max_chunks * 2,
            )

            # Normalize ts_rank scores to 0-1 range
            if results:
                max_score = max(s for _, s in results) or 1.0
                normalized = [(chunk, score / max_score) for chunk, score in results]
            else:
                normalized = []

            return {
                "source": "keyword",
                "chunks": normalized,
                "entities": [],
            }
        except Exception as e:
            logger.warning(f"Full-text search failed: {e}")
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
