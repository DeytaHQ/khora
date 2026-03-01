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

# Regex for simple-query detection heuristic
import re
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
from .understanding import QueryIntent, QueryUnderstanding, UnderstandingResult

_TEMPORAL_PATTERN = re.compile(
    r"\b(yesterday|today|last\s+\w+|this\s+\w+|ago|since|before|after|"
    r"recent(?:ly)?|q[1-4]|20\d{2}|january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\b",
    re.IGNORECASE,
)

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
        """Get concatenated text from top chunks for LLM context.

        Groups chunks by document title/source for better readability.
        """
        # Group chunks by title
        groups: dict[str, list[str]] = {}
        for chunk, score in self.chunks[:max_chunks]:
            title = self._extract_chunk_title(chunk)
            groups.setdefault(title, []).append(chunk.content)

        sections = []
        for title, contents in groups.items():
            if title:
                sections.append(f"--- From: {title} ---\n" + "\n\n".join(contents))
            else:
                sections.extend(contents)
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _extract_chunk_title(chunk: Any) -> str:
        """Extract title from chunk metadata."""
        meta = getattr(chunk, "metadata", None)
        if meta is None:
            return ""
        # DocumentMetadata object with .title attribute
        title = getattr(meta, "title", "")
        if title:
            return title
        # dict-style metadata
        if isinstance(meta, dict):
            title = meta.get("title", "")
            if title:
                return title
        # Try custom dict
        custom = getattr(meta, "custom", None)
        if isinstance(custom, dict):
            return custom.get("title", "")
        return ""

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
    min_chunk_similarity: float = 0.05
    min_entity_similarity: float = 0.05

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
    entity_linking_fuzzy_threshold: float = 0.6
    entity_linking_embedding_threshold: float = 0.4
    entity_linking_max_candidates: int = 5

    # Reranking settings
    enable_reranking: bool = True
    reranking_method: str = "cross_encoder"
    reranking_model: str | None = None
    reranking_top_n: int = 50
    reranking_final_k: int = 10

    # Keyword search settings
    enable_keyword_search: bool = True
    keyword_search_method: str = "fulltext"

    # HyDE settings
    enable_hyde: str = "auto"
    hyde_num_hypotheticals: int = 1

    # Multi-stage ranking pipeline settings
    enable_multi_stage: bool = True
    stage1_recall_limit: int = 200
    stage3_filter_limit: int = 50
    stage4_rerank_limit: int = 50
    enable_diversity: bool = True
    diversity_lambda: float = 0.5

    # Narrative coherence scoring
    enable_narrative_coherence: bool = True
    coherence_boost_per_entity: float = 0.2
    coherence_max_boost: float = 0.5
    coherence_isolation_penalty: float = 0.15

    # Two-tier temporal resolver
    enable_temporal_resolver: bool = True
    temporal_resolver_strategy: str = "hybrid"
    temporal_sql_pushdown: bool = True
    temporal_date_validation: bool = True

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
            enable_query_understanding=settings.enable_understanding,
            enable_query_expansion=settings.understanding_expand_query,
            enable_entity_extraction=settings.understanding_extract_entities,
            enable_temporal_detection=settings.understanding_detect_temporal,
            # Entity linking
            enable_entity_linking=settings.enable_entity_linking,
            entity_linking_fuzzy_threshold=settings.entity_linking_fuzzy_threshold,
            entity_linking_embedding_threshold=settings.entity_linking_embedding_threshold,
            entity_linking_max_candidates=settings.entity_linking_max_candidates,
            # Reranking
            enable_reranking=settings.enable_reranking,
            reranking_method=settings.reranking_method,
            reranking_model=settings.reranking_model,
            reranking_top_n=settings.reranking_top_n,
            reranking_final_k=settings.reranking_final_k,
            # Keyword search
            enable_keyword_search=settings.enable_keyword_search,
            keyword_search_method=settings.keyword_search_method,
            # HyDE — settings.enable_hyde is already normalized to str by the validator
            enable_hyde=(
                settings.enable_hyde
                if isinstance(settings.enable_hyde, str)
                else ("always" if settings.enable_hyde else "never")
            ),
            hyde_num_hypotheticals=settings.hyde_num_hypotheticals,
            # Multi-stage ranking pipeline
            enable_multi_stage=settings.enable_multi_stage,
            stage1_recall_limit=settings.stage1_recall_limit,
            stage3_filter_limit=settings.stage3_filter_limit,
            stage4_rerank_limit=settings.stage4_rerank_limit,
            enable_diversity=settings.enable_diversity,
            diversity_lambda=settings.diversity_lambda,
            # Two-tier temporal resolver
            enable_temporal_resolver=getattr(settings, "enable_temporal_resolver", True),
            temporal_resolver_strategy=getattr(settings, "temporal_resolver_strategy", "hybrid"),
            temporal_sql_pushdown=getattr(settings, "temporal_sql_pushdown", True),
            temporal_date_validation=getattr(settings, "temporal_date_validation", True),
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

        # HyDE expander — create if mode allows HyDE (auto or always)
        self._hyde_expander: HyDEExpander | None = None  # type: ignore[unresolved-reference]
        if self._config.enable_hyde in ("auto", "always") and self._embedder:
            from .hyde import HyDEExpander

            self._hyde_expander = HyDEExpander(
                self._embedder,
                llm_config=llm_config,
                num_hypotheticals=self._config.hyde_num_hypotheticals,
            )

        # Query cache
        from .cache import QueryCache

        self._cache = QueryCache()

        # Understanding cache (keyed by normalized query + namespace)
        self._understanding_cache = QueryCache(max_size=500, ttl_seconds=600)

        # Keyword searcher (built per namespace)
        self._keyword_searchers: dict[str, KeywordSearcher] = {}

        # Cached rerankers (keyed by method name) so model is loaded once
        self._rerankers: dict[str, Any] = {}

        # Per-query entity similarity cache to avoid duplicate DB queries
        # when both _vector_search and _graph_search need entity similarities.
        # Keyed by id(embedding), cleared at the start of each query().
        self._entity_similarity_cache: dict[int, asyncio.Task[Any]] = {}

    def invalidate_caches(self, namespace_id: UUID) -> None:
        """Invalidate BM25 keyword index and query caches for a namespace.

        Call this after ingesting new documents so stale results are not served.

        Args:
            namespace_id: Namespace whose caches should be cleared.
        """
        ns_key = str(namespace_id)
        self._keyword_searchers.pop(ns_key, None)
        # QueryCache keys are hashed with namespace_id; the invalidate() method
        # currently clears the whole cache as a safe fallback.  Re-creating the
        # cache object achieves the same effect without needing an event loop.
        self._cache = type(self._cache)(
            max_size=self._cache._max_size,
            ttl_seconds=int(self._cache._ttl.total_seconds()),
        )
        self._understanding_cache = type(self._understanding_cache)(
            max_size=self._understanding_cache._max_size,
            ttl_seconds=int(self._understanding_cache._ttl.total_seconds()),
        )
        logger.debug(f"Invalidated caches for namespace {namespace_id}")

    async def query(
        self,
        query_text: str,
        namespace_id: UUID,
        *,
        config: QueryConfig | None = None,
        temporal_filter: TemporalFilter | None = None,
        context: ACLContext | None = None,
        agentic: bool = False,
        _lightweight_understanding: bool | None = None,
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

        # Clear per-query entity similarity cache
        self._entity_similarity_cache.clear()

        # Check cache
        cached = await self._cache.get(query_text, namespace_id, cfg.mode.name)
        if cached is not None:
            logger.debug(f"Cache hit for query: {query_text[:50]}...")
            from khora.telemetry import get_collector as _get_tc

            _get_tc().record_pipeline_stage(
                pipeline="query",
                stage="cache_lookup",
                latency_ms=0.0,
                output_count=len(cached.chunks),
                namespace_id=namespace_id,
                metadata={"hit": True},
            )
            return cached

        logger.debug(f"Executing query: {query_text[:50]}... (mode={cfg.mode.name})")

        from uuid import uuid4 as _uuid4

        from khora.telemetry.instrument import pipeline_stage

        _run_id = _uuid4()

        # Initialize metrics
        metrics = SearchMetrics()
        metrics.total_timer.start()
        metrics.features = {  # type: ignore[invalid-assignment]
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

        # Fast temporal resolution (dateparser, runs in <1ms)
        if cfg.enable_temporal_resolver and cfg.temporal_resolver_strategy in ("dateparser", "hybrid"):
            from .temporal_resolver import ResolvedRange, TemporalResolver

            resolver = TemporalResolver()
            fast_result = resolver.resolve_fast(query_text)
            if fast_result and (fast_result.start or fast_result.end):
                if cfg.temporal_date_validation:
                    validated = resolver.validate_dates(fast_result.start, fast_result.end)
                    fast_result = ResolvedRange(
                        start=validated[0],
                        end=validated[1],
                        confidence=fast_result.confidence,
                        expression=fast_result.expression,
                        source=fast_result.source,
                    )
                if fast_result.start or fast_result.end:
                    temporal_filter = TemporalFilter(
                        start_date=fast_result.start,
                        end_date=fast_result.end,
                    )
                    temporal_info.detected = True
                    temporal_info.time_start = fast_result.start
                    temporal_info.time_end = fast_result.end
                    temporal_info.filter_applied = True
                    metadata["temporal_resolver"] = {
                        "source": fast_result.source,
                        "confidence": fast_result.confidence,
                        "expression": fast_result.expression,
                    }
                    logger.debug(f"Fast temporal resolver: {fast_result.start} to {fast_result.end}")

        # Step 1: Query Understanding
        # Check cache FIRST (fastest path), then heuristic, then LLM call
        metrics.understanding_timer.start()
        understanding: UnderstandingResult | None = None
        if cfg.enable_query_understanding:
            # Check understanding cache before running heuristic (cache lookup is faster)
            cached_understanding = await self._understanding_cache.get(query_text, namespace_id, "understanding")
            if cached_understanding is not None:
                understanding = cached_understanding
                logger.debug(f"Understanding cache hit for: {query_text[:50]}...")
            elif not self._is_simple_query(query_text):
                # Only call LLM for complex queries that aren't cached
                try:
                    # Use lightweight prompt unless caller explicitly opted out
                    use_lightweight = _lightweight_understanding if _lightweight_understanding is not None else True
                    async with pipeline_stage("query", "understanding", _run_id, namespace_id=namespace_id):
                        understanding_coro = self._query_understanding.understand(
                            query_text,
                            expand_query=cfg.enable_query_expansion,
                            extract_entities=cfg.enable_entity_extraction,
                            detect_temporal=cfg.enable_temporal_detection,
                            lightweight=use_lightweight,
                        )
                        try:
                            understanding = await asyncio.wait_for(understanding_coro, timeout=2.0)
                        except TimeoutError:
                            logger.warning("Query understanding timed out after 2s, using heuristic fallback")
                            understanding = self._heuristic_understanding(query_text)
                    # Cache the understanding result
                    if understanding is not None:
                        await self._understanding_cache.set(query_text, namespace_id, "understanding", understanding)
                except Exception as e:
                    logger.warning(f"Query understanding failed: {e}")

            # Apply understanding results (works for both cached and fresh)
            if understanding is not None:
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

                # Adaptive top-k: reduce evidence for focused queries, but keep
                # a generous floor to avoid losing evidence chunks.
                if (
                    understanding.complexity_score < 0.3
                    and not understanding.requires_multi_step
                    and cfg.max_chunks > 8
                ):
                    cfg.max_chunks = 8
                    cfg.max_entities = 8
                    cfg.min_chunk_similarity = max(cfg.min_chunk_similarity, 0.25)
                    cfg.min_entity_similarity = max(cfg.min_entity_similarity, 0.25)
                    metadata["adaptive_top_k"] = {"reduced": True, "reason": "very_focused"}
                    logger.debug(
                        f"Adaptive top-k: very_focused to {cfg.max_chunks} "
                        f"(complexity={understanding.complexity_score:.2f})"
                    )
                elif (
                    understanding.complexity_score < 0.5
                    and not understanding.requires_multi_step
                    and cfg.max_chunks > 8
                ):
                    cfg.max_chunks = 8
                    cfg.max_entities = 8
                    cfg.min_chunk_similarity = max(cfg.min_chunk_similarity, 0.15)
                    cfg.min_entity_similarity = max(cfg.min_entity_similarity, 0.15)
                    metadata["adaptive_top_k"] = {"reduced": True, "reason": "single_topic"}
                    logger.debug(
                        f"Adaptive top-k: reduced to {cfg.max_chunks} (complexity={understanding.complexity_score:.2f})"
                    )

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

        # Auto-enable recency bias when temporal intent is detected
        if temporal_info.detected and not cfg.apply_recency_bias:
            cfg.apply_recency_bias = True
            cfg.recency_weight = max(cfg.recency_weight, 0.2)
            metadata["auto_recency_bias"] = True

        metrics.understanding_timer.stop()

        # Step 2: Entity Linking
        metrics.linking_timer.start()
        linking_result: LinkingResult | None = None
        linked_entity_ids: list[UUID] = []
        if cfg.enable_entity_linking and understanding and understanding.entities:
            try:
                async with pipeline_stage("query", "entity_linking", _run_id, namespace_id=namespace_id):
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

        # Get query embedding for the main query
        query_embedding = None
        if self._embedder and cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
            query_embedding = await self._embedder.embed(query_text)

            # Apply HyDE expansion based on mode
            if query_embedding is not None and self._hyde_expander:
                should_hyde = False
                if cfg.enable_hyde == "always":
                    should_hyde = True
                elif cfg.enable_hyde == "auto" and understanding is not None:
                    should_hyde = understanding.complexity_score > 0.6 or understanding.intent == QueryIntent.TEMPORAL
                if should_hyde:
                    query_embedding = await self._hyde_expander.expand_query_embedding(query_text, query_embedding)
                    metadata["hyde_applied"] = True

        # Step 3-7: Execute search pipeline (multi-stage or legacy)
        if cfg.enable_multi_stage:
            # Use multi-stage ranking pipeline for improved quality
            metrics.multi_stage_enabled = True
            metadata["multi_stage_enabled"] = True

            fused_chunks, fused_entities, graph_context, search_contributions = await self._multi_stage_search(
                query_text=query_text,
                namespace_id=namespace_id,
                query_embedding=query_embedding,
                config=cfg,
                understanding=understanding,
                linked_entity_ids=linked_entity_ids,
                temporal_filter=temporal_filter,
                metrics=metrics,
                graph_info=graph_info,
            )

            # Populate per-source counts into metrics from contributions
            metrics.vector_chunk_count = search_contributions.vector.chunk_count
            metrics.graph_chunk_count = search_contributions.graph.chunk_count
            metrics.keyword_chunk_count = search_contributions.keyword.chunk_count
            metrics.vector_entity_count = search_contributions.vector.entity_count
            metrics.graph_entity_count = search_contributions.graph.entity_count
            metrics.fused_chunk_count = metrics.stage2_normalized_count
            metrics.fused_entity_count = len(fused_entities)

            # Boost linked entities
            if linked_entity_ids and fused_entities:
                boosted_entities = []
                for entity, score in fused_entities:
                    if entity.id in linked_entity_ids:
                        boosted_entities.append((entity, score * 1.5))  # 50% boost
                    else:
                        boosted_entities.append((entity, score))
                fused_entities = sorted(boosted_entities, key=lambda x: x[1], reverse=True)

            # Compute overlap statistics
            search_contributions.compute_overlaps()

        else:
            # Legacy single-stage pipeline
            metadata["multi_stage_enabled"] = False

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

            # Pre-compute expanded query embeddings in parallel (2.7)
            expanded_embeddings: dict[int, list[float] | None] = {0: query_embedding}
            if (
                self._embedder
                and len(queries_to_search) > 1
                and cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL)
            ):
                embed_tasks = [self._embedder.embed(queries_to_search[i]) for i in range(1, len(queries_to_search))]
                embed_results = await asyncio.gather(*embed_tasks, return_exceptions=True)
                for idx, result in enumerate(embed_results):
                    if isinstance(result, BaseException):
                        logger.warning(f"Failed to embed expanded query {idx + 1}: {result}")
                        expanded_embeddings[idx + 1] = None
                    else:
                        expanded_embeddings[idx + 1] = result

            for i, q in enumerate(queries_to_search):
                suffix = "" if i == 0 else f"_exp{i}"

                # Use pre-computed embedding
                current_embedding = expanded_embeddings.get(i)

                # Execute searches in parallel based on mode
                tasks = []
                task_types = []  # Track which task is which for timing

                legacy_sql_temporal = temporal_filter if cfg.temporal_sql_pushdown else None

                if cfg.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL) and current_embedding is not None:
                    tasks.append(
                        self._timed_search(
                            self._vector_search(
                                namespace_id, current_embedding, cfg, temporal_filter=legacy_sql_temporal
                            ),
                            "vector",
                        )
                    )
                    task_types.append("vector")

                if cfg.mode in (SearchMode.GRAPH, SearchMode.HYBRID, SearchMode.ALL):
                    tasks.append(
                        self._timed_search(
                            self._graph_search(namespace_id, q, current_embedding, cfg, linked_entity_ids), "graph"
                        )
                    )
                    task_types.append("graph")

                if cfg.mode in (SearchMode.HYBRID, SearchMode.ALL) and cfg.enable_keyword_search:
                    keywords = understanding.keywords if understanding else None
                    if cfg.keyword_search_method == "fulltext":
                        tasks.append(
                            self._timed_search(
                                self._keyword_search_fulltext(
                                    namespace_id, q, cfg, temporal_filter=legacy_sql_temporal
                                ),
                                "keyword",
                            )
                        )
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

            # Record search stage telemetry
            from khora.telemetry import get_collector as _get_collector

            _get_collector().record_pipeline_stage(
                pipeline="query",
                stage="search",
                run_id=_run_id,
                latency_ms=search_contributions.total_search_latency_ms,
                output_count=sum(len(v) for v in all_chunk_results.values()),
                namespace_id=namespace_id,
            )

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

            _get_collector().record_pipeline_stage(
                pipeline="query",
                stage="fusion",
                run_id=_run_id,
                latency_ms=search_contributions.fusion_latency_ms,
                input_count=sum(len(v) for v in all_chunk_results.values()),
                output_count=len(fused_chunks) + len(fused_entities),
                namespace_id=namespace_id,
            )
            metrics.fused_chunk_count = len(fused_chunks)
            metrics.fused_entity_count = len(fused_entities)

            # Zero-result fallback: retry with no threshold and keyword search
            if not fused_chunks:
                logger.info("Zero results after fusion — attempting fallback search")
                fallback_chunks: dict[str, list[tuple[Any, float]]] = {}

                # Fallback 1: vector search with no similarity threshold
                if query_embedding is not None:
                    try:
                        fb_vector = await self._vector_search(
                            namespace_id,
                            query_embedding,
                            QueryConfig(
                                max_chunks=cfg.max_chunks,
                                max_entities=cfg.max_entities,
                                min_chunk_similarity=0.0,
                                min_entity_similarity=0.0,
                            ),
                        )
                        if fb_vector["chunks"]:
                            fallback_chunks["vector"] = fb_vector["chunks"]
                            if fb_vector.get("entities"):
                                all_entity_results["vector_fallback"] = fb_vector["entities"]
                    except Exception as e:
                        logger.warning(f"Fallback vector search failed: {e}")

                # Fallback 2: keyword search (if not already run)
                if not search_contributions.keyword.chunk_count:
                    try:
                        fb_keyword = await self._keyword_search_fulltext(namespace_id, query_text, cfg)
                        if fb_keyword["chunks"]:
                            fallback_chunks["keyword"] = fb_keyword["chunks"]
                    except Exception as e:
                        logger.warning(f"Fallback keyword search failed: {e}")

                if fallback_chunks:
                    fused_chunks = reciprocal_rank_fusion(
                        fallback_chunks,
                        k=cfg.rrf_k,
                        weights={"vector": cfg.vector_weight, "keyword": cfg.keyword_weight},
                        id_extractor=lambda c: str(c.id),
                    )
                    metrics.fused_chunk_count = len(fused_chunks)
                    logger.info(f"Fallback recovered {len(fused_chunks)} chunks")

            # Apply source priority boosting from query understanding
            if understanding and understanding.source_priority:
                fused_chunks = self._apply_source_priority(fused_chunks, understanding)
                fused_entities = self._apply_source_priority_entities(fused_entities, understanding)

            # Apply attribute-aware scoring boost for entities
            if understanding and understanding.keywords and fused_entities:
                fused_entities = [
                    (entity, score + self._attribute_relevance_boost(entity, understanding.keywords))
                    for entity, score in fused_entities
                ]
                fused_entities.sort(key=lambda x: x[1], reverse=True)

            # Boost linked entities
            if linked_entity_ids:
                boosted_entities = []
                for entity, score in fused_entities:
                    if entity.id in linked_entity_ids:
                        boosted_entities.append((entity, score * 1.5))  # 50% boost
                    else:
                        boosted_entities.append((entity, score))
                fused_entities = sorted(boosted_entities, key=lambda x: x[1], reverse=True)

            # Step 5: Apply soft temporal scoring (exponential decay outside window)
            if temporal_filter:
                fused_chunks = self._soft_temporal_score(fused_chunks, temporal_filter)

            # Apply recency bias (batch-accelerated via Rust)
            if cfg.apply_recency_bias:
                from .temporal import batch_apply_recency

                fused_chunks = batch_apply_recency(fused_chunks, cfg.recency_weight, cfg.recency_decay_days)

            # Step 6: Reranking (optional, skip for small result sets)
            metrics.reranking_timer.start()
            if cfg.enable_reranking and len(fused_chunks) >= 5:
                try:
                    if cfg.reranking_method not in self._rerankers:
                        self._rerankers[cfg.reranking_method] = create_reranker(
                            method=cfg.reranking_method,
                            model=cfg.reranking_model,
                            llm_config=self._llm_config,
                        )
                    reranker = self._rerankers[cfg.reranking_method]
                    candidates = [
                        RerankCandidate(
                            item=chunk,
                            original_score=score,
                            content=chunk.content,
                            metadata=chunk.metadata,
                        )
                        for chunk, score in fused_chunks[: cfg.reranking_top_n]
                    ]
                    async with pipeline_stage(
                        "query",
                        "reranking",
                        _run_id,
                        namespace_id=namespace_id,
                        input_count=len(candidates),
                        extra_metadata={"method": cfg.reranking_method},
                    ) as _rerank_ctx:
                        reranked = await reranker.rerank(query_text, candidates, top_k=cfg.reranking_final_k)
                        _rerank_ctx["output_count"] = len(reranked)
                    fused_chunks = [(r.item, r.final_score) for r in reranked]
                    metadata["reranking"] = {"method": cfg.reranking_method, "reranked_count": len(fused_chunks)}
                    logger.debug(f"Reranked {len(candidates)} candidates to {len(fused_chunks)} results")
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")

            metrics.reranking_timer.stop()

            # Step 6.5: Narrative coherence scoring
            if cfg.enable_narrative_coherence and len(fused_chunks) >= 3:
                fused_chunks = self._apply_narrative_coherence(fused_chunks, fused_entities, cfg)

            # Step 7: Limit results
            fused_chunks = fused_chunks[: cfg.max_chunks]
            fused_entities = fused_entities[: cfg.max_entities]

            # Compute overlap statistics
            search_contributions.compute_overlaps()

        # Update graph info with depth used
        graph_info.neighborhood_depth = cfg.max_graph_depth

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

        # Cross-session expansion: when few results are found or temporal
        # intent is detected, search adjacent sessions to improve recall
        # across session boundaries.
        _has_temporal_intent = understanding is not None and getattr(understanding, "has_temporal", False)
        if len(fused_chunks) < 2 or _has_temporal_intent:
            expanded = await self._expand_adjacent_sessions(fused_chunks, namespace_id, understanding=understanding)
            if expanded:
                fused_chunks = expanded
                metadata["session_expansion"] = True

        result = QueryResult(
            chunks=fused_chunks,
            entities=fused_entities,
            graph_context=graph_context,
            metadata=metadata,
            search_contributions=search_contributions,
            graph_info=graph_info,
            temporal_info=temporal_info,
        )

        # Cache the result
        await self._cache.set(query_text, namespace_id, cfg.mode.name, result)

        return result

    async def _expand_adjacent_sessions(
        self,
        chunks: list[tuple[Any, float]],
        namespace_id: UUID,
        *,
        understanding: UnderstandingResult | None = None,
    ) -> list[tuple[Any, float]] | None:
        """Expand search to sessions sharing entities with the current results.

        Uses entity-based linking: extracts entity names from the query
        understanding and current results, then searches for other chunks
        mentioning those entities. Falls back to sequential session_id ± 1
        adjacency if entity-based expansion finds nothing.

        Args:
            chunks: Current (sparse) result set.
            namespace_id: Namespace to search in.
            understanding: Query understanding result for entity names.

        Returns:
            Expanded chunk list, or ``None`` if expansion was not applicable.
        """
        if not chunks:
            return None

        existing_ids = {str(c.id) for c, _ in chunks}

        # --- Entity-based expansion ---
        # Collect entity names from query understanding
        entity_names: list[str] = []
        if understanding and understanding.entities:
            for e in understanding.entities:
                entity_names.append(e.name)
                entity_names.extend(e.aliases)

        if entity_names:
            entity_chunks: list[tuple[Any, float]] = []
            # Search for chunks mentioning query entities (limit queries to avoid unbounded work)
            for name in entity_names[:5]:
                try:
                    results = await self._storage.search_fulltext_chunks(
                        namespace_id,
                        name,
                        limit=5,
                    )
                    for chunk_or_tuple in results:
                        if isinstance(chunk_or_tuple, tuple):
                            chunk_obj, score = chunk_or_tuple
                        else:
                            chunk_obj, score = chunk_or_tuple, 0.3
                        if str(chunk_obj.id) not in existing_ids:
                            entity_chunks.append((chunk_obj, score * 0.7))
                            existing_ids.add(str(chunk_obj.id))
                except Exception:
                    continue

            if entity_chunks:
                expanded = list(chunks) + entity_chunks
                logger.debug(
                    f"Session expansion (entity-based): added {len(entity_chunks)} chunks "
                    f"for entities {entity_names[:5]}"
                )
                return expanded

        # --- Fallback: sequential session_id ± 1 ---
        session_ids: set[int] = set()
        for chunk, _ in chunks:
            meta = getattr(chunk, "metadata", None)
            if meta is None:
                continue
            if isinstance(meta, dict):
                sid = meta.get("session_id") or (meta.get("custom", {}) or {}).get("session_id")
            else:
                custom = getattr(meta, "custom", None)
                sid = custom.get("session_id") if isinstance(custom, dict) else None
            if sid is not None:
                session_ids.add(int(sid))

        if not session_ids:
            return None

        adjacent: set[int] = set()
        for sid in session_ids:
            adjacent.add(sid - 1)
            adjacent.add(sid + 1)
        adjacent -= session_ids

        if not adjacent:
            return None

        try:
            adjacent_chunks = await self._storage.search_chunks_by_metadata(  # type: ignore[unresolved-attribute]
                namespace_id,
                metadata_filter={"session_id": list(adjacent)},
                limit=10,
            )
        except (AttributeError, NotImplementedError):
            logger.debug("Session expansion skipped: storage does not support metadata search")
            return None
        except Exception as e:
            logger.debug(f"Session expansion failed: {e}")
            return None

        if not adjacent_chunks:
            return None

        expanded = list(chunks)
        for adj_chunk in adjacent_chunks:
            if isinstance(adj_chunk, tuple):
                chunk_obj, score = adj_chunk
            else:
                chunk_obj, score = adj_chunk, 0.3
            if str(chunk_obj.id) not in existing_ids:
                expanded.append((chunk_obj, score * 0.8))
                existing_ids.add(str(chunk_obj.id))

        logger.debug(
            f"Session expansion (sequential): added {len(expanded) - len(chunks)} chunks "
            f"from adjacent sessions {adjacent}"
        )
        return expanded

    def _apply_narrative_coherence(
        self,
        chunks: list[tuple[Any, float]],
        entities: list[tuple[Any, float]],
        config: QueryConfig,
    ) -> list[tuple[Any, float]]:
        """Apply narrative coherence scoring to retrieved chunks.

        Chunks that share entities with other retrieved chunks get a coherence
        boost (they belong to the same narrative).  Isolated chunks that don't
        share entities with any other retrieved chunk are penalized (they may
        be cross-narrative contamination).

        Args:
            chunks: Retrieved (chunk, score) pairs.
            entities: Retrieved (entity, score) pairs.
            config: Query configuration with coherence settings.

        Returns:
            Re-scored and re-sorted chunk list.
        """
        if not chunks or not entities:
            return chunks

        # Build a map: chunk_id → set of entity names associated with that chunk
        chunk_entity_map: dict[str, set[str]] = {}
        for entity, _ in entities:
            for chunk_id in getattr(entity, "source_chunk_ids", []):
                chunk_entity_map.setdefault(str(chunk_id), set()).add(entity.name)

        # For each retrieved chunk, find which entities it's associated with
        chunk_entities: list[set[str]] = []
        for chunk, _ in chunks:
            cid = str(chunk.id)
            chunk_entities.append(chunk_entity_map.get(cid, set()))

        # Count entity overlaps between each chunk and the rest of the result set
        scored: list[tuple[Any, float]] = []
        for i, (chunk, score) in enumerate(chunks):
            my_entities = chunk_entities[i]
            if not my_entities:
                scored.append((chunk, score))
                continue

            # Count how many other chunks share at least one entity
            shared_count = 0
            for j, other_entities in enumerate(chunk_entities):
                if i != j and my_entities & other_entities:
                    shared_count += 1

            if shared_count > 0:
                # Coherence boost: more shared entities = higher boost (capped)
                boost = min(
                    shared_count * config.coherence_boost_per_entity,
                    config.coherence_max_boost,
                )
                scored.append((chunk, score * (1 + boost)))
            else:
                # Isolation penalty: chunk has entities but none overlap with results
                scored.append((chunk, score * (1 - config.coherence_isolation_penalty)))

        # Re-sort by adjusted score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

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

    async def _cached_entity_search(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        limit: int,
        min_similarity: float,
    ) -> list[tuple[UUID, float]]:
        """Search similar entities with per-query deduplication.

        Uses a shared asyncio.Task so concurrent calls from _vector_search
        and _graph_search within the same asyncio.gather share a single
        database query.
        """
        cache_key = id(query_embedding)
        if cache_key not in self._entity_similarity_cache:
            max_limit = max(limit, 20)
            self._entity_similarity_cache[cache_key] = asyncio.ensure_future(
                self._storage.search_similar_entities(
                    namespace_id,
                    query_embedding,
                    limit=max_limit,
                    min_similarity=min_similarity,
                )
            )
        results = await self._entity_similarity_cache[cache_key]
        return results[:limit]

    async def _vector_search(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        config: QueryConfig,
        temporal_filter: TemporalFilter | None = None,
    ) -> dict[str, Any]:
        """Perform vector similarity search."""
        # Extract temporal bounds for SQL pushdown
        created_after = None
        created_before = None
        if temporal_filter:
            start, end = temporal_filter.get_effective_times()
            created_after = start
            created_before = end

        # Search chunks
        chunk_results = await self._storage.search_similar_chunks(
            namespace_id,
            query_embedding,
            limit=config.max_chunks * 2,  # Get extra for fusion
            min_similarity=config.min_chunk_similarity,
            created_after=created_after,
            created_before=created_before,
        )

        # Search entities (uses per-query cache to deduplicate with _graph_search)
        entity_ids_scores = await self._cached_entity_search(
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
            for entity_id in linked_entity_ids[:10]:
                if entity_id not in seen_entity_ids:
                    all_entity_ids_to_fetch.append(entity_id)
                    linked_scores[entity_id] = 1.0  # High confidence from linking
                    seen_entity_ids.add(entity_id)

        # Similar entities via embedding (uses per-query cache to deduplicate with _vector_search)
        if query_embedding is not None:
            entity_ids_scores = await self._cached_entity_search(
                namespace_id,
                query_embedding,
                limit=10,
                min_similarity=config.min_entity_similarity,
            )

            for entity_id, score in entity_ids_scores[:8]:
                if entity_id not in seen_entity_ids:
                    all_entity_ids_to_fetch.append(entity_id)
                    similar_scores[entity_id] = score
                    seen_entity_ids.add(entity_id)

            # Fallback: if no entities found, retry with no threshold
            # (not cached — different parameters, only fires when primary returns empty)
            if not all_entity_ids_to_fetch and config.min_entity_similarity > 0:
                entity_ids_scores = await self._storage.search_similar_entities(
                    namespace_id,
                    query_embedding,
                    limit=5,
                    min_similarity=0.0,
                )
                for entity_id, score in entity_ids_scores[:5]:
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

        # Get related chunks through entities - batch fetch to avoid N+1
        chunks = []

        # Collect all unique chunk IDs with their associated entity/score info
        chunk_ids_to_fetch: list[UUID] = []
        chunk_id_to_info: dict[UUID, tuple[Any, float]] = {}
        for entity, score in entities:
            for chunk_id in entity.source_chunk_ids[:5]:
                if chunk_id not in chunk_id_to_info:
                    chunk_ids_to_fetch.append(chunk_id)
                    chunk_id_to_info[chunk_id] = (entity, score)

        # Batch fetch all chunks in a single query
        if chunk_ids_to_fetch:
            from khora._accel import batch_dot_product

            chunks_map = await self._storage.get_chunks_batch(chunk_ids_to_fetch)

            _min_graph_sim = 0.3

            # Separate chunks with/without embeddings for batch processing
            if query_embedding is not None:
                # Collect chunks that have embeddings for batch dot product
                embeddable_ids: list[Any] = []
                embeddable_embeddings: list[list[float]] = []
                no_embedding_ids: list[Any] = []

                for chunk_id, chunk in chunks_map.items():
                    if chunk.embedding is not None:
                        embeddable_ids.append(chunk_id)
                        embeddable_embeddings.append(chunk.embedding)
                    else:
                        no_embedding_ids.append(chunk_id)

                # Batch compute similarities for all chunks with embeddings
                sim_results: dict[Any, float] = {}
                if embeddable_embeddings:
                    dot_scores = batch_dot_product(query_embedding, embeddable_embeddings, threshold=0.0)
                    for idx, sim in dot_scores:
                        sim_results[embeddable_ids[idx]] = sim

                # Score chunks: blend query similarity with entity score
                for chunk_id, chunk in chunks_map.items():
                    entity, score = chunk_id_to_info[chunk_id]
                    entity_score = score * (1 + 0.1 * min(entity.mention_count, 10))

                    if chunk_id in sim_results:
                        query_sim = sim_results[chunk_id]
                        if query_sim < _min_graph_sim:
                            continue
                        # Blend: 60% query similarity + 40% entity score
                        chunk_score = 0.6 * query_sim + 0.4 * entity_score
                    else:
                        # No embedding — fall back to entity score only
                        chunk_score = entity_score

                    chunks.append((chunk, chunk_score))
            else:
                # No query embedding — use entity scores only
                for chunk_id, chunk in chunks_map.items():
                    entity, score = chunk_id_to_info[chunk_id]
                    chunk_score = score * (1 + 0.1 * min(entity.mention_count, 10))
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
        temporal_filter: TemporalFilter | None = None,
    ) -> dict[str, Any]:
        """Perform PostgreSQL full-text search using tsvector/tsquery.

        Unlike BM25, this runs entirely in PostgreSQL using the GIN-indexed
        content_tsv column, with no chunk count limit.
        """
        try:
            from khora.telemetry import get_collector as _get_tc

            created_after = None
            created_before = None
            if temporal_filter:
                start, end = temporal_filter.get_effective_times()
                created_after = start
                created_before = end

            _kw_start = time.perf_counter()
            results = await self._storage.search_fulltext_chunks(
                namespace_id,
                query_text,
                limit=config.max_chunks * 2,
                created_after=created_after,
                created_before=created_before,
            )
            _get_tc().record_pipeline_stage(
                pipeline="query",
                stage="keyword_search",
                latency_ms=(time.perf_counter() - _kw_start) * 1000,
                output_count=len(results),
                namespace_id=namespace_id,
                metadata={"method": "fulltext"},
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

    def _apply_source_priority(
        self,
        results: list[tuple[Any, float]],
        understanding: UnderstandingResult,
    ) -> list[tuple[Any, float]]:
        """Boost or demote chunk results based on source priority weights.

        Reads source_tool from chunk -> document metadata and applies
        a multiplicative boost based on the query understanding's source_priority.

        Args:
            results: List of (chunk, score) tuples
            understanding: Query understanding with source_priority

        Returns:
            Re-sorted list of (chunk, score) tuples
        """
        sp = understanding.source_priority
        source_filters = set(understanding.source_filters)

        boosted = []
        for chunk, score in results:
            source_tool = ""
            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "custom"):
                source_tool = chunk.metadata.custom.get("source_tool", "")

            if not source_tool:
                boosted.append((chunk, score))
                continue

            # Filter out results from excluded sources
            if source_tool in source_filters:
                continue

            weight = getattr(sp, source_tool, 1.0) if hasattr(sp, source_tool) else 1.0
            # Apply as multiplicative boost — never fully zero out
            adjusted_score = score * (0.5 + 0.5 * weight)
            boosted.append((chunk, adjusted_score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def _apply_source_priority_entities(
        self,
        results: list[tuple[Any, float]],
        understanding: UnderstandingResult,
    ) -> list[tuple[Any, float]]:
        """Boost or demote entity results based on source priority weights.

        Args:
            results: List of (entity, score) tuples
            understanding: Query understanding with source_priority

        Returns:
            Re-sorted list of (entity, score) tuples
        """
        sp = understanding.source_priority
        source_filters = set(understanding.source_filters)

        boosted = []
        for entity, score in results:
            source_tool = getattr(entity, "source_tool", "")

            if not source_tool:
                boosted.append((entity, score))
                continue

            if source_tool in source_filters:
                continue

            weight = getattr(sp, source_tool, 1.0) if hasattr(sp, source_tool) else 1.0
            adjusted_score = score * (0.5 + 0.5 * weight)
            boosted.append((entity, adjusted_score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    @staticmethod
    def _is_simple_query(query_text: str) -> bool:
        """Check if a query is simple enough to skip LLM understanding.

        Simple queries are short, lack temporal references, and don't contain
        complex entity mentions. Skipping understanding saves ~14s per query.

        Args:
            query_text: The query text

        Returns:
            True if the query should skip understanding
        """
        words = query_text.split()
        if len(words) > 8:
            return False

        # Has temporal references
        if _TEMPORAL_PATTERN.search(query_text):
            return False

        # Contains quoted phrases (specific entity searches)
        if '"' in query_text or "'" in query_text:
            return False

        # Contains comparison words
        comparison_words = {"compare", "versus", "vs", "difference", "between", "similar"}
        if comparison_words & {w.lower().strip("?.,!") for w in words}:
            return False

        return True

    @staticmethod
    def _heuristic_understanding(query_text: str) -> UnderstandingResult | None:
        """Build a lightweight UnderstandingResult from heuristics when LLM times out."""
        from .understanding import AnswerType, QueryIntent, TemporalReference

        has_temporal = bool(_TEMPORAL_PATTERN.search(query_text))
        temporal_refs: list[TemporalReference] = []
        if has_temporal:
            try:
                from .temporal_resolver import TemporalResolver

                resolver = TemporalResolver()
                for m in _TEMPORAL_PATTERN.finditer(query_text):
                    resolved = resolver.resolve_fast(m.group())
                    temporal_refs.append(
                        TemporalReference(
                            type="relative",
                            text=m.group(),
                            start_date=resolved.start if resolved else None,
                            end_date=resolved.end if resolved else None,
                        )
                    )
            except Exception:
                # Fallback: temporal pattern detected but no dates resolved
                for m in _TEMPORAL_PATTERN.finditer(query_text):
                    temporal_refs.append(TemporalReference(type="relative", text=m.group()))

        return UnderstandingResult(
            original_query=query_text,
            intent=QueryIntent.TEMPORAL if has_temporal else QueryIntent.SEARCH,
            answer_type=AnswerType.UNKNOWN,
            temporal_references=temporal_refs,
            keywords=re.findall(r"\b\w{3,}\b", query_text.lower()),
            complexity_score=0.5,
            reasoning="heuristic fallback (LLM timeout)",
        )

    @staticmethod
    def _attribute_relevance_boost(entity: Any, keywords: list[str]) -> float:
        """Score boost based on entity attribute value matches with query keywords.

        When a query like "urgent tickets assigned to Alice" matches entities
        with priority: "urgent" and assignee: "Alice" in their attributes,
        those entities get a relevance boost.

        Args:
            entity: Entity with attributes dict
            keywords: Keywords from query understanding

        Returns:
            Additional score boost (0.0 to 0.3, capped)
        """
        attributes = getattr(entity, "attributes", None)
        if not attributes or not isinstance(attributes, dict):
            return 0.0

        boost = 0.0
        for value in attributes.values():
            value_str = str(value).lower()
            for keyword in keywords:
                if keyword.lower() in value_str:
                    boost += 0.1
        return min(boost, 0.3)  # Cap at 0.3

    # -------------------------------------------------------------------------
    # Temporal Re-ranking
    # -------------------------------------------------------------------------

    _RECENCY_KEYWORDS = frozenset({"recent", "recently", "latest", "last", "newest", "new", "current"})
    _EARLIEST_KEYWORDS = frozenset({"first", "earliest", "before", "oldest", "original", "initial"})

    def _apply_temporal_reranking(
        self,
        chunks: list[tuple[Any, float]],
        understanding: Any,
    ) -> list[tuple[Any, float]]:
        """Blend relevance scores with temporal position when temporal intent is detected.

        Args:
            chunks: Scored chunks (already sorted by relevance).
            understanding: UnderstandingResult with temporal_references.

        Returns:
            Re-scored and re-sorted chunk list.
        """
        temporal_weight = 0.3

        # Determine sort direction from temporal reference text
        ascending = False  # default: most recent first (descending)
        refs_text = " ".join(ref.text.lower() for ref in getattr(understanding, "temporal_references", []))
        if self._EARLIEST_KEYWORDS & set(refs_text.split()):
            ascending = True

        # Sort a copy by created_at to determine temporal rank.
        # Use a tuple key so chunks missing created_at sort to the end
        # instead of raising TypeError when compared with datetimes.
        sorted_by_time = sorted(
            chunks,
            key=lambda x: (
                (1, getattr(x[0], "created_at", None)) if getattr(x[0], "created_at", None) is not None else (0, 0)
            ),
            reverse=not ascending,
        )
        n = len(sorted_by_time)
        # Map chunk id -> temporal position score (best temporal match = 1.0)
        temporal_scores: dict[Any, float] = {}
        for rank, (chunk, _score) in enumerate(sorted_by_time):
            temporal_scores[id(chunk)] = 1.0 - (rank / max(n - 1, 1))

        # Blend
        blended = []
        for chunk, relevance_score in chunks:
            t_score = temporal_scores.get(id(chunk), 0.0)
            blended_score = (1 - temporal_weight) * relevance_score + temporal_weight * t_score
            blended.append((chunk, blended_score))
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended

    @staticmethod
    def _soft_temporal_score(
        chunks: list[tuple[Any, float]],
        temporal_filter: TemporalFilter,
    ) -> list[tuple[Any, float]]:
        """Apply soft temporal scoring with exponential decay.

        Chunks inside the temporal window keep full score.
        Chunks outside get exponential decay based on distance.
        Chunks >30 days outside are hard-filtered.
        """
        import math

        from .temporal import TemporalFilter as _TF
        from .temporal import _dt_to_epoch

        start, end = temporal_filter.get_effective_times()
        start_secs = _dt_to_epoch(_TF._normalize_tz(start)) if start else None
        end_secs = _dt_to_epoch(_TF._normalize_tz(end)) if end else None

        _SECS_PER_DAY = 86400.0
        _HARD_CUTOFF_DAYS = 30.0
        _HALF_LIFE_HOURS = 24.0
        _DECAY_FACTOR = -0.6931471805599453 / (_HALF_LIFE_HOURS * 3600.0)

        result: list[tuple[Any, float]] = []
        for chunk, score in chunks:
            created_at = getattr(chunk, "created_at", None)
            if created_at is None:
                result.append((chunk, score))
                continue

            ts = _dt_to_epoch(_TF._normalize_tz(created_at))
            if ts is None:
                result.append((chunk, score))
                continue

            # Check if inside window
            inside = True
            secs_outside = 0.0
            if start_secs is not None and ts < start_secs:
                inside = False
                secs_outside = start_secs - ts
            elif end_secs is not None and ts > end_secs:
                inside = False
                secs_outside = ts - end_secs

            if inside:
                result.append((chunk, score))
            else:
                days_outside = secs_outside / _SECS_PER_DAY
                if days_outside > _HARD_CUTOFF_DAYS:
                    continue
                decay = math.exp(_DECAY_FACTOR * secs_outside)
                result.append((chunk, score * decay))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    @staticmethod
    def _apply_entity_presence_scoring(
        chunks: list[tuple[Any, float]],
        understanding: Any,
    ) -> list[tuple[Any, float]]:
        """Apply entity-presence penalty for confounder rejection.

        For each chunk, check how many query-mentioned entities appear in
        the chunk text (case-insensitive). Apply multiplicative penalty
        based on entity match ratio.
        """
        entity_names: list[str] = []
        for e in understanding.entities:
            entity_names.append(e.name.lower())
            for alias in getattr(e, "aliases", []) or []:
                entity_names.append(alias.lower())

        if not entity_names:
            return chunks

        result: list[tuple[Any, float]] = []
        for chunk, score in chunks:
            content = getattr(chunk, "content", "")
            if not content or not entity_names:
                result.append((chunk, score))
                continue

            content_lower = content.lower()
            # Count how many unique query entities appear in the chunk
            unique_entities = {e.name.lower() for e in understanding.entities}
            matched = 0
            for ename in unique_entities:
                if ename in content_lower:
                    matched += 1
                    continue
                # Check aliases
                aliases = []
                for e in understanding.entities:
                    if e.name.lower() == ename:
                        aliases = [a.lower() for a in (getattr(e, "aliases", []) or [])]
                        break
                if any(a in content_lower for a in aliases):
                    matched += 1

            if len(unique_entities) > 0:
                match_ratio = matched / len(unique_entities)
                penalty = max(0.5, match_ratio)
                result.append((chunk, score * penalty))
            else:
                result.append((chunk, score))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # -------------------------------------------------------------------------
    # Multi-Stage Ranking Pipeline
    # -------------------------------------------------------------------------

    async def _multi_stage_search(
        self,
        query_text: str,
        namespace_id: UUID,
        query_embedding: list[float] | None,
        config: QueryConfig,
        understanding: Any | None,
        linked_entity_ids: list[UUID],
        temporal_filter: TemporalFilter | None,
        metrics: SearchMetrics,
        graph_info: GraphTraversalInfo,
    ) -> tuple[list[tuple[Any, float]], list[tuple[Any, float]], dict[str, Any], SearchMethodContribution]:
        """Multi-stage ranking pipeline for improved search quality.

        Pipeline stages:
        1. Broad Recall - Fast retrieval of 100-200 candidates from all sources
        2. Score Normalization & Fusion - Normalize scores to [0,1] and apply weighted RRF
        3. Lightweight Filtering - Apply temporal filters and source priority, reduce to top 50
        4. Neural Reranking - Cross-encoder reranking on top candidates (expensive)
        5. Diversity & Final Selection - Optional MMR-style diversity, return top 10

        Args:
            query_text: Query text
            namespace_id: Namespace to search in
            query_embedding: Query embedding (optional)
            config: Query configuration
            understanding: Query understanding result (optional)
            linked_entity_ids: Entity IDs from entity linking
            temporal_filter: Optional temporal filter
            metrics: SearchMetrics for tracking
            graph_info: GraphTraversalInfo for tracking

        Returns:
            Tuple of (chunks, entities, graph_context, search_contributions)
        """
        from khora.telemetry import get_collector as _get_collector

        search_contributions = SearchMethodContribution()
        graph_context: dict[str, Any] = {}

        # Stage 1: Broad Recall
        metrics.stage1_recall_timer.start()
        stage1_chunks, stage1_entities, stage1_graph_context, stage1_contributions = await self._stage1_recall(
            query_text=query_text,
            namespace_id=namespace_id,
            query_embedding=query_embedding,
            config=config,
            understanding=understanding,
            linked_entity_ids=linked_entity_ids,
            graph_info=graph_info,
            temporal_filter=temporal_filter,
        )
        metrics.stage1_recall_timer.stop()
        metrics.stage1_candidate_count = sum(len(v) for v in stage1_chunks.values())
        graph_context.update(stage1_graph_context)
        search_contributions = stage1_contributions

        _get_collector().record_pipeline_stage(
            pipeline="query",
            stage="multi_stage_1_recall",
            latency_ms=metrics.stage1_recall_timer.elapsed_ms,
            output_count=metrics.stage1_candidate_count,
            namespace_id=namespace_id,
        )

        logger.debug(
            f"Stage 1 (Recall): {metrics.stage1_candidate_count} chunks, "
            f"{sum(len(v) for v in stage1_entities.values())} entities"
        )

        if not stage1_chunks:
            fused_entities = []
            if stage1_entities:
                weights = {"vector": config.vector_weight, "graph": config.graph_weight}
                fused_entities = reciprocal_rank_fusion(
                    stage1_entities,
                    k=config.rrf_k,
                    weights=weights,
                    id_extractor=lambda e: str(e.id),
                )
            return [], fused_entities, graph_context, search_contributions

        # Stage 2: Score Normalization & Fusion
        metrics.stage2_normalize_timer.start()
        stage2_chunks, stage2_entities = self._stage2_normalize_fuse(
            stage1_chunks=stage1_chunks,
            stage1_entities=stage1_entities,
            config=config,
        )
        metrics.stage2_normalize_timer.stop()
        metrics.stage2_normalized_count = len(stage2_chunks)

        _get_collector().record_pipeline_stage(
            pipeline="query",
            stage="multi_stage_2_normalize",
            latency_ms=metrics.stage2_normalize_timer.elapsed_ms,
            input_count=metrics.stage1_candidate_count,
            output_count=len(stage2_chunks),
            namespace_id=namespace_id,
        )

        logger.debug(f"Stage 2 (Normalize/Fuse): {len(stage2_chunks)} chunks")

        # Stage 3: Lightweight Filtering
        metrics.stage3_filter_timer.start()
        stage3_chunks, stage3_entities = self._stage3_filter(
            chunks=stage2_chunks,
            entities=stage2_entities,
            config=config,
            understanding=understanding,
            temporal_filter=temporal_filter,
        )
        metrics.stage3_filter_timer.stop()
        metrics.stage3_filtered_count = len(stage3_chunks)

        _get_collector().record_pipeline_stage(
            pipeline="query",
            stage="multi_stage_3_filter",
            latency_ms=metrics.stage3_filter_timer.elapsed_ms,
            input_count=len(stage2_chunks),
            output_count=len(stage3_chunks),
            namespace_id=namespace_id,
        )

        logger.debug(f"Stage 3 (Filter): {len(stage3_chunks)} chunks")

        if not stage3_chunks:
            return [], stage3_entities, graph_context, search_contributions

        # Stage 4: Neural Reranking
        metrics.stage4_rerank_timer.start()
        stage4_chunks = await self._stage4_rerank(
            chunks=stage3_chunks,
            query_text=query_text,
            config=config,
        )
        metrics.stage4_rerank_timer.stop()
        metrics.stage4_reranked_count = len(stage4_chunks)

        _get_collector().record_pipeline_stage(
            pipeline="query",
            stage="multi_stage_4_rerank",
            latency_ms=metrics.stage4_rerank_timer.elapsed_ms,
            input_count=len(stage3_chunks),
            output_count=len(stage4_chunks),
            namespace_id=namespace_id,
            metadata={"method": config.reranking_method if config.enable_reranking else "none"},
        )

        logger.debug(f"Stage 4 (Rerank): {len(stage4_chunks)} chunks")

        # Stage 4.5: Narrative coherence scoring
        if config.enable_narrative_coherence and len(stage4_chunks) >= 3:
            stage4_chunks = self._apply_narrative_coherence(stage4_chunks, stage3_entities, config)

        # Stage 5: Diversity & Final Selection
        metrics.stage5_diversity_timer.start()
        stage5_chunks, stage5_entities = self._stage5_diversity(
            chunks=stage4_chunks,
            entities=stage3_entities,
            query_embedding=query_embedding,
            config=config,
        )
        metrics.stage5_diversity_timer.stop()
        metrics.stage5_final_count = len(stage5_chunks)

        _get_collector().record_pipeline_stage(
            pipeline="query",
            stage="multi_stage_5_diversity",
            latency_ms=metrics.stage5_diversity_timer.elapsed_ms,
            input_count=len(stage4_chunks),
            output_count=len(stage5_chunks),
            namespace_id=namespace_id,
            metadata={"diversity_enabled": config.enable_diversity},
        )

        logger.debug(f"Stage 5 (Diversity/Final): {len(stage5_chunks)} chunks")

        return stage5_chunks, stage5_entities, graph_context, search_contributions

    async def _stage1_recall(
        self,
        query_text: str,
        namespace_id: UUID,
        query_embedding: list[float] | None,
        config: QueryConfig,
        understanding: Any | None,
        linked_entity_ids: list[UUID],
        graph_info: GraphTraversalInfo,
        temporal_filter: TemporalFilter | None = None,
    ) -> tuple[
        dict[str, list[tuple[Any, float]]],
        dict[str, list[tuple[Any, float]]],
        dict[str, Any],
        SearchMethodContribution,
    ]:
        """Stage 1: Broad recall from all search sources.

        Retrieves a large candidate set (100-200) using fast search methods.
        Uses higher limits than final output to ensure we don't miss relevant results.

        Returns:
            Tuple of (chunk_results_by_source, entity_results_by_source, graph_context, contributions)
        """
        search_contributions = SearchMethodContribution()
        all_chunk_results: dict[str, list[tuple[Any, float]]] = {}
        all_entity_results: dict[str, list[tuple[Any, float]]] = {}
        graph_context: dict[str, Any] = {}

        # Calculate per-source limits based on stage1_recall_limit
        # Distribute recall limit across sources with vector getting the most
        recall_limit = config.stage1_recall_limit
        vector_limit = int(recall_limit * 0.5)  # 50% from vector
        graph_limit = int(recall_limit * 0.3)  # 30% from graph
        keyword_limit = int(recall_limit * 0.3)  # 30% from keyword (overlap allowed)

        # Determine queries to search (original + expansions)
        queries_to_search = [query_text]
        if understanding and config.enable_query_expansion:
            queries_to_search.extend(understanding.expanded_queries[:2])

        search_start_time = time.perf_counter()

        # Pre-compute expanded query embeddings in parallel (2.7)
        expanded_embeddings: dict[int, list[float] | None] = {0: query_embedding}
        if (
            self._embedder
            and len(queries_to_search) > 1
            and config.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL)
        ):
            embed_tasks = [self._embedder.embed(queries_to_search[i]) for i in range(1, len(queries_to_search))]
            embed_results = await asyncio.gather(*embed_tasks, return_exceptions=True)
            for idx, result in enumerate(embed_results):
                if isinstance(result, BaseException):
                    logger.warning(f"Failed to embed expanded query {idx + 1}: {result}")
                    expanded_embeddings[idx + 1] = None
                else:
                    expanded_embeddings[idx + 1] = result

        for i, q in enumerate(queries_to_search):
            suffix = "" if i == 0 else f"_exp{i}"

            # Use pre-computed embedding
            current_embedding = expanded_embeddings.get(i)

            # Execute searches in parallel
            tasks = []
            task_types = []

            sql_temporal = temporal_filter if config.temporal_sql_pushdown else None

            if config.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL) and current_embedding is not None:
                # Create a temporary config with higher limits for broad recall
                recall_cfg = QueryConfig(
                    max_chunks=vector_limit,
                    max_entities=config.max_entities * 2,
                    min_chunk_similarity=config.min_chunk_similarity * 0.5,  # Lower threshold for recall
                    min_entity_similarity=config.min_entity_similarity * 0.5,
                )
                tasks.append(
                    self._timed_search(
                        self._vector_search(namespace_id, current_embedding, recall_cfg, temporal_filter=sql_temporal),
                        "vector",
                    )
                )
                task_types.append("vector")

            if config.mode in (SearchMode.GRAPH, SearchMode.HYBRID, SearchMode.ALL):
                recall_cfg = QueryConfig(
                    max_chunks=graph_limit,
                    max_entities=config.max_entities * 2,
                    max_graph_depth=config.max_graph_depth,
                    min_chunk_similarity=config.min_chunk_similarity * 0.5,
                    min_entity_similarity=config.min_entity_similarity * 0.5,
                )
                tasks.append(
                    self._timed_search(
                        self._graph_search(namespace_id, q, current_embedding, recall_cfg, linked_entity_ids),
                        "graph",
                    )
                )
                task_types.append("graph")

            if config.mode in (SearchMode.HYBRID, SearchMode.ALL) and config.enable_keyword_search:
                keywords = understanding.keywords if understanding else None
                if config.keyword_search_method == "fulltext":
                    # Create config with higher limit for recall
                    recall_cfg = QueryConfig(max_chunks=keyword_limit)
                    tasks.append(
                        self._timed_search(
                            self._keyword_search_fulltext(namespace_id, q, recall_cfg, temporal_filter=sql_temporal),
                            "keyword",
                        )
                    )
                else:
                    recall_cfg = QueryConfig(max_chunks=keyword_limit)
                    tasks.append(
                        self._timed_search(self._keyword_search_bm25(namespace_id, q, recall_cfg, keywords), "keyword")
                    )
                task_types.append("keyword")

            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Stage 1 search {j} failed: {result}")
                    continue

                if isinstance(result, dict):
                    source_type = result.get("source", f"search_{j}")
                    latency_ms = result.get("latency_ms", 0.0)

                    if "chunks" in result:
                        source = source_type + suffix
                        all_chunk_results[source] = result["chunks"]

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

                        entity_ids = [str(e.id) for e, _ in result["entities"]]
                        if source_type == "vector":
                            search_contributions.vector.entity_count += len(result["entities"])
                            search_contributions.vector.entity_ids.extend(entity_ids)
                        elif source_type == "graph":
                            search_contributions.graph.entity_count += len(result["entities"])
                            search_contributions.graph.entity_ids.extend(entity_ids)
                            for entity, _ in result["entities"]:
                                graph_info.entities_searched.append(entity.name)

                    if "graph_context" in result:
                        graph_context.update(result["graph_context"])
                        if "relationships" in result.get("graph_context", {}):
                            for rel in result["graph_context"]["relationships"]:
                                if isinstance(rel, dict):
                                    graph_info.relationships_traversed.append(
                                        (rel.get("from", ""), rel.get("type", ""), rel.get("to", ""))
                                    )

        search_contributions.total_search_latency_ms = (time.perf_counter() - search_start_time) * 1000

        return all_chunk_results, all_entity_results, graph_context, search_contributions

    def _stage2_normalize_fuse(
        self,
        stage1_chunks: dict[str, list[tuple[Any, float]]],
        stage1_entities: dict[str, list[tuple[Any, float]]],
        config: QueryConfig,
    ) -> tuple[list[tuple[Any, float]], list[tuple[Any, float]]]:
        """Stage 2: Score normalization and fusion.

        Normalizes all scores to [0,1] range and applies weighted RRF fusion.
        This ensures fair comparison across different search methods.

        Returns:
            Tuple of (fused_chunks, fused_entities)
        """
        # Normalize scores within each source to [0,1]
        normalized_chunks: dict[str, list[tuple[Any, float]]] = {}
        for source, results in stage1_chunks.items():
            if not results:
                continue
            scores = [s for _, s in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score

            if score_range > 0:
                normalized_chunks[source] = [(chunk, (score - min_score) / score_range) for chunk, score in results]
            else:
                # All same score - use 0.5
                normalized_chunks[source] = [(chunk, 0.5) for chunk, _ in results]

        normalized_entities: dict[str, list[tuple[Any, float]]] = {}
        for source, results in stage1_entities.items():
            if not results:
                continue
            scores = [s for _, s in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score

            if score_range > 0:
                normalized_entities[source] = [(entity, (score - min_score) / score_range) for entity, score in results]
            else:
                normalized_entities[source] = [(entity, 0.5) for entity, _ in results]

        # Apply weighted RRF fusion
        fused_chunks = []
        if normalized_chunks:
            weights = {
                "vector": config.vector_weight,
                "graph": config.graph_weight,
                "keyword": config.keyword_weight,
            }
            # Add weights for expanded query results
            for key in normalized_chunks:
                if "_exp" in key:
                    base_source = key.split("_exp")[0]
                    weights[key] = weights.get(base_source, config.vector_weight) * 0.7

            fused_chunks = reciprocal_rank_fusion(
                normalized_chunks,
                k=config.rrf_k,
                weights=weights,
                id_extractor=lambda c: str(c.id),
            )

        fused_entities = []
        if normalized_entities:
            weights = {
                "vector": config.vector_weight,
                "graph": config.graph_weight,
            }
            fused_entities = reciprocal_rank_fusion(
                normalized_entities,
                k=config.rrf_k,
                weights=weights,
                id_extractor=lambda e: str(e.id),
            )

        return fused_chunks, fused_entities

    def _stage3_filter(
        self,
        chunks: list[tuple[Any, float]],
        entities: list[tuple[Any, float]],
        config: QueryConfig,
        understanding: Any | None,
        temporal_filter: TemporalFilter | None,
    ) -> tuple[list[tuple[Any, float]], list[tuple[Any, float]]]:
        """Stage 3: Lightweight filtering.

        Applies temporal filters and source priority boosting to reduce
        the candidate set before expensive reranking.

        Returns:
            Tuple of (filtered_chunks, filtered_entities)
        """
        filtered_chunks = chunks

        # Apply soft temporal scoring instead of hard filtering.
        # Chunks inside the window get full score; chunks outside get
        # exponential decay.  Only hard-filter chunks >30 days outside.
        if temporal_filter:
            filtered_chunks = self._soft_temporal_score(filtered_chunks, temporal_filter)

        # Apply recency bias (batch-accelerated via Rust)
        if config.apply_recency_bias:
            from .temporal import batch_apply_recency

            filtered_chunks = batch_apply_recency(filtered_chunks, config.recency_weight, config.recency_decay_days)

        # Temporal re-ranking: blend relevance with temporal position when
        # the query understanding detects temporal intent.
        if understanding and getattr(understanding, "has_temporal", False) and filtered_chunks:
            filtered_chunks = self._apply_temporal_reranking(filtered_chunks, understanding)

        # Apply source priority boosting
        if understanding and understanding.source_priority:
            filtered_chunks = self._apply_source_priority(filtered_chunks, understanding)
            entities = self._apply_source_priority_entities(entities, understanding)

        # Apply attribute-aware scoring boost for entities
        if understanding and understanding.keywords and entities:
            entities = [
                (entity, score + self._attribute_relevance_boost(entity, understanding.keywords))
                for entity, score in entities
            ]
            entities.sort(key=lambda x: x[1], reverse=True)

        # Entity-presence scoring for confounder rejection:
        # penalize chunks that don't mention query-referenced entities.
        if understanding and understanding.entities and filtered_chunks:
            filtered_chunks = self._apply_entity_presence_scoring(filtered_chunks, understanding)

        # Limit to stage3_filter_limit
        filtered_chunks = filtered_chunks[: config.stage3_filter_limit]
        entities = entities[: config.max_entities]

        return filtered_chunks, entities

    async def _stage4_rerank(
        self,
        chunks: list[tuple[Any, float]],
        query_text: str,
        config: QueryConfig,
    ) -> list[tuple[Any, float]]:
        """Stage 4: Neural reranking.

        Applies cross-encoder or LLM-based reranking to the filtered candidates.
        This is the expensive step, only applied to top candidates from Stage 3.

        Returns:
            Reranked chunks with updated scores
        """
        if not config.enable_reranking or len(chunks) < 3:
            return chunks

        try:
            # Get or create reranker
            if config.reranking_method not in self._rerankers:
                self._rerankers[config.reranking_method] = create_reranker(
                    method=config.reranking_method,
                    model=config.reranking_model,
                    llm_config=self._llm_config,
                )
            reranker = self._rerankers[config.reranking_method]

            # Limit candidates to stage4_rerank_limit
            candidates_to_rerank = chunks[: config.stage4_rerank_limit]

            candidates = [
                RerankCandidate(
                    item=chunk,
                    original_score=score,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )
                for chunk, score in candidates_to_rerank
            ]

            # Rerank - use max_chunks as final limit
            reranked = await reranker.rerank(
                query_text,
                candidates,
                top_k=config.max_chunks * 2,  # Keep extra for diversity stage
            )

            return [(r.item, r.final_score) for r in reranked]

        except Exception as e:
            logger.warning(f"Stage 4 reranking failed: {e}")
            return chunks

    def _stage5_diversity(
        self,
        chunks: list[tuple[Any, float]],
        entities: list[tuple[Any, float]],
        query_embedding: list[float] | None,
        config: QueryConfig,
    ) -> tuple[list[tuple[Any, float]], list[tuple[Any, float]]]:
        """Stage 5: Diversity selection and final limiting.

        Optionally applies MMR-style diversity to ensure varied results.
        Then applies final limit to return requested number of results.

        Returns:
            Tuple of (final_chunks, final_entities)
        """
        final_chunks = chunks

        if config.enable_diversity and query_embedding is not None and len(chunks) > config.max_chunks:
            # Apply MMR-style diversity selection
            final_chunks = self._mmr_diversity_select(
                chunks=chunks,
                query_embedding=query_embedding,
                k=config.max_chunks,
                lambda_param=config.diversity_lambda,
            )
        else:
            # Just limit to max_chunks
            final_chunks = chunks[: config.max_chunks]

        final_entities = entities[: config.max_entities]

        return final_chunks, final_entities

    def _mmr_diversity_select(
        self,
        chunks: list[tuple[Any, float]],
        query_embedding: list[float],
        k: int,
        lambda_param: float = 0.5,
    ) -> list[tuple[Any, float]]:
        """Maximal Marginal Relevance selection for diversity.

        Balances relevance to query with diversity among selected results.
        Uses Rust-accelerated MMR when available via ``_accel.mmr_diversity_select``.

        Args:
            chunks: Candidate chunks with scores
            query_embedding: Query embedding for relevance
            k: Number of results to select
            lambda_param: Tradeoff between relevance (1.0) and diversity (0.0)

        Returns:
            Selected chunks with adjusted scores
        """
        from khora._accel import batch_dot_product as _bdp
        from khora._accel import mmr_diversity_select, normalize_embeddings_batch

        if len(chunks) <= k:
            return chunks

        # Extract embeddings from chunks (if available)
        chunk_embeddings: list[list[float] | None] = []
        for chunk, _ in chunks:
            embedding = getattr(chunk, "embedding", None)
            if embedding is None and hasattr(chunk, "metadata"):
                embedding = getattr(chunk.metadata, "embedding", None)
            chunk_embeddings.append(embedding)

        # If no embeddings available, fall back to score-based selection
        if all(e is None for e in chunk_embeddings):
            return chunks[:k]

        # For candidates missing embeddings, use the query embedding as a
        # placeholder (their relevance score will dominate the MMR calc).
        filled_embeddings: list[list[float]] = [emb if emb is not None else query_embedding for emb in chunk_embeddings]

        # Compute relevance scores as cosine similarity to query.
        # Pre-normalize all embeddings (including query) so dot product = cosine.
        all_vectors = [query_embedding] + filled_embeddings
        normalized = normalize_embeddings_batch(all_vectors)
        norm_query = normalized[0]
        norm_embeddings = normalized[1:]

        # Relevance = batch dot product (Rust/NumPy accelerated)
        dot_results = _bdp(norm_query, norm_embeddings, threshold=0.0)
        scores = [0.0] * len(norm_embeddings)
        for idx, sim in dot_results:
            scores[idx] = sim

        # Run MMR selection on pre-normalized embeddings
        selected_indices = mmr_diversity_select(norm_embeddings, scores, lambda_param, k)

        return [chunks[i] for i in selected_indices]

    async def find_related_entities(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity through the graph.

        Uses batch entity fetching to avoid N+1 queries for better performance.

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

        entity_nodes = neighborhood.get("entities", [])
        if not entity_nodes:
            return []

        # Collect all entity IDs for batch fetch
        entity_ids = [UUID(node["id"]) for node in entity_nodes]

        # Batch fetch all entities in a single query (avoids N+1)
        entities_map = await self._storage.get_entities_batch(entity_ids)

        # Score based on path length (shorter = higher score)
        # This is simplified - full impl would consider actual path lengths
        base_score = 1.0 / (1 + len(neighborhood.get("relationships", [])))

        entities = []
        for node in entity_nodes:
            eid = UUID(node["id"])
            if eid in entities_map:
                entities.append((entities_map[eid], base_score))

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

    async def warm_cache(
        self,
        namespace_id: UUID,
        queries: list[str] | None = None,
        *,
        config: QueryConfig | None = None,
        include_entity_based: bool = True,
        max_entity_queries: int = 20,
    ) -> dict[str, Any]:
        """Pre-warm the query cache with common query patterns.

        This is useful for reducing latency on frequently-used queries or
        for pre-loading cache after namespace updates.

        Args:
            namespace_id: Namespace to warm cache for
            queries: Explicit list of queries to execute
            config: Optional query config
            include_entity_based: Generate queries based on top entities
            max_entity_queries: Max entity-based queries to generate

        Returns:
            Summary of cache warming results
        """
        cfg = config or self._config
        warmed_queries: list[str] = []
        errors: list[str] = []

        # Collect all queries to warm
        all_queries = list(queries or [])

        # Generate entity-based queries if requested
        if include_entity_based:
            try:
                # Get top entities by mention count
                entities = await self._storage.list_entities(
                    namespace_id,
                    limit=max_entity_queries * 2,
                )
                # Sort by mention count and take top N
                entities.sort(key=lambda e: e.mention_count, reverse=True)
                for entity in entities[:max_entity_queries]:
                    # Generate natural queries about entities
                    all_queries.append(entity.name)
                    if entity.description:
                        all_queries.append(f"What is {entity.name}?")
            except Exception as e:
                logger.warning(f"Failed to fetch entities for cache warming: {e}")

        # Execute queries to populate cache
        for query_text in all_queries:
            try:
                # Execute query with understanding disabled for speed
                await self.query(
                    query_text,
                    namespace_id,
                    config=QueryConfig(
                        mode=cfg.mode,
                        max_chunks=cfg.max_chunks,
                        max_entities=cfg.max_entities,
                        enable_query_understanding=False,
                        enable_reranking=False,
                    ),
                )
                warmed_queries.append(query_text)
            except Exception as e:
                errors.append(f"{query_text}: {e}")
                logger.debug(f"Cache warming failed for '{query_text}': {e}")

        logger.info(f"Cache warming complete: {len(warmed_queries)} queries cached, {len(errors)} errors")

        return {
            "namespace_id": str(namespace_id),
            "queries_warmed": len(warmed_queries),
            "errors": len(errors),
            "cache_stats": self._cache.stats,
        }

    async def warm_keyword_index(
        self,
        namespace_id: UUID,
        max_chunks: int = 10000,
    ) -> dict[str, Any]:
        """Pre-build the BM25 keyword index for a namespace.

        This is useful for reducing first-query latency when keyword
        search is enabled.

        Args:
            namespace_id: Namespace to build index for
            max_chunks: Maximum chunks to index

        Returns:
            Summary of indexing results
        """
        ns_key = str(namespace_id)

        if ns_key in self._keyword_searchers:
            return {
                "namespace_id": ns_key,
                "status": "already_indexed",
                "chunk_count": (
                    len(self._keyword_searchers[ns_key]._chunks)
                    if hasattr(self._keyword_searchers[ns_key], "_chunks")
                    else 0
                ),
            }

        try:
            chunks = await self._storage.list_chunks(namespace_id, limit=max_chunks)
            if chunks:
                searcher = KeywordSearcher(use_stemming=True, remove_stopwords=True)
                searcher.index_chunks(chunks)
                self._keyword_searchers[ns_key] = searcher
                logger.info(f"Pre-built BM25 index with {len(chunks)} chunks for {namespace_id}")
                return {
                    "namespace_id": ns_key,
                    "status": "indexed",
                    "chunk_count": len(chunks),
                }
            return {
                "namespace_id": ns_key,
                "status": "no_chunks",
                "chunk_count": 0,
            }
        except Exception as e:
            logger.warning(f"Failed to pre-build keyword index: {e}")
            return {
                "namespace_id": ns_key,
                "status": "error",
                "error": str(e),
            }
