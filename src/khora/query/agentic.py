"""Agentic search for Khora.

Provides a two-step exploration agent that:
1. Performs initial search with comprehensive query understanding (single LLM call)
2. Uses pre-computed follow-up queries from understanding for deeper exploration
3. Explores multiple sources even if initial hits are concentrated
4. Maintains full trace log of search reasoning

The key efficiency gain: ALL LLM extraction happens in the initial query understanding
call. Follow-up queries are pre-computed, eliminating additional LLM round-trips.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

if TYPE_CHECKING:
    from khora.config.llm import LiteLLMConfig
    from khora.core.models import Chunk, Entity
    from khora.query.engine import HybridQueryEngine, QueryConfig, QueryResult
    from khora.query.understanding import UnderstandingResult


@dataclass
class SearchStep:
    """A single step in the agentic search process."""

    step_number: int
    query: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Results summary
    total_chunks: int = 0
    total_entities: int = 0
    sources_hit: dict[str, int] = field(default_factory=dict)

    # Search method contributions (counts for display)
    vector_hits: int = 0
    graph_hits: int = 0
    keyword_hits: int = 0

    # Full search method data with chunk IDs (for attribution)
    search_methods_data: dict[str, Any] = field(default_factory=dict)

    # Graph elements triggered
    entities_linked: list[str] = field(default_factory=list)
    relationships_traversed: list[tuple[str, str, str]] = field(default_factory=list)

    # Temporal info
    temporal_filter_applied: bool = False
    time_range: tuple[datetime | None, datetime | None] | None = None


@dataclass
class AgenticSearchTrace:
    """Full trace of an agentic search session."""

    session_id: str
    original_query: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Query understanding (from single LLM call)
    understanding_reasoning: str = ""
    complexity_score: float = 0.0
    source_priority: dict[str, float] = field(default_factory=dict)

    steps: list[SearchStep] = field(default_factory=list)

    # Final summary
    summary: str = ""
    total_unique_chunks: int = 0
    total_unique_entities: int = 0
    sources_explored: dict[str, int] = field(default_factory=dict)

    def add_step(self, step: SearchStep) -> None:
        """Add a search step to the trace."""
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for logging/storage."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "understanding_reasoning": self.understanding_reasoning,
            "complexity_score": self.complexity_score,
            "source_priority": self.source_priority,
            "steps": [
                {
                    "step_number": s.step_number,
                    "query": s.query,
                    "reasoning": s.reasoning,
                    "timestamp": s.timestamp.isoformat(),
                    "total_chunks": s.total_chunks,
                    "total_entities": s.total_entities,
                    "sources_hit": s.sources_hit,
                    "search_contributions": {
                        "vector": s.vector_hits,
                        "graph": s.graph_hits,
                        "keyword": s.keyword_hits,
                    },
                    "entities_linked": s.entities_linked,
                    "relationships_traversed": [
                        {"from": f, "type": r, "to": t} for f, r, t in s.relationships_traversed
                    ],
                    "temporal_filter_applied": s.temporal_filter_applied,
                    "time_range": (
                        [
                            s.time_range[0].isoformat() if s.time_range and s.time_range[0] else None,
                            s.time_range[1].isoformat() if s.time_range and s.time_range[1] else None,
                        ]
                        if s.time_range
                        else None
                    ),
                }
                for s in self.steps
            ],
            "summary": self.summary,
            "total_unique_chunks": self.total_unique_chunks,
            "total_unique_entities": self.total_unique_entities,
            "sources_explored": self.sources_explored,
        }


@dataclass
class AgenticSearchStepResult:
    """Yielded by search_stream for each completed step."""

    step_number: int
    query: str
    chunks: list[tuple[Chunk, float, str]]
    entities: list[tuple[Entity, float]]
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticSearchResult:
    """Result from agentic search."""

    # Combined results from all steps
    chunks: list[tuple[Chunk, float, str]] = field(default_factory=list)
    entities: list[tuple[Entity, float]] = field(default_factory=list)

    # Summary (generated without additional LLM call if possible)
    summary: str = ""

    # Full trace
    trace: AgenticSearchTrace | None = None

    # Query understanding (from single LLM call)
    understanding: UnderstandingResult | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class AgenticSearchAgent:
    """Two-step exploration agent for deep search.

    EFFICIENCY: All LLM extraction happens in the initial query understanding.
    The understanding result includes:
    - Pre-computed follow-up queries with reasoning
    - Source priority recommendations
    - Search strategy (weights for vector/graph/keyword)
    - Complexity assessment

    This eliminates additional LLM calls during exploration.
    """

    def __init__(
        self,
        engine: HybridQueryEngine,
        llm_config: LiteLLMConfig | None = None,
    ) -> None:
        """Initialize the agentic search agent.

        Args:
            engine: The hybrid query engine to use
            llm_config: LLM configuration (used by engine's query understanding)
        """
        self._engine = engine
        self._llm_config = llm_config

    async def search(
        self,
        query: str,
        namespace_id: UUID,
        config: QueryConfig | None = None,
        max_steps: int = 3,
    ) -> AgenticSearchResult:
        """Perform agentic search with multi-step exploration.

        The search process:
        1. Initial query with comprehensive understanding (single LLM call)
        2. Execute pre-computed follow-up queries (no additional LLM calls)
        3. Merge and rank all results

        Args:
            query: Original search query
            namespace_id: Namespace to search in
            config: Query configuration
            max_steps: Maximum exploration steps (default 3)

        Returns:
            AgenticSearchResult with combined results and trace
        """
        import uuid

        trace = AgenticSearchTrace(
            session_id=str(uuid.uuid4()),
            original_query=query,
        )

        all_chunks: dict[str, tuple[Chunk, float, str]] = {}
        all_entities: dict[str, tuple[Entity, float]] = {}
        understanding: UnderstandingResult | None = None

        # Step 1: Initial search (this triggers comprehensive query understanding)
        logger.info(f"Agentic search step 1: '{query[:50]}...'")

        step1_result = await self._engine.query(
            query,
            namespace_id,
            config=config,
            _lightweight_understanding=False,  # Full prompt for agentic step 1 (needs follow-ups)
        )

        # Extract understanding from result metadata
        if "understanding" in step1_result.metadata:
            understanding_data = step1_result.metadata["understanding"]
            trace.understanding_reasoning = understanding_data.get("reasoning", "")
            trace.complexity_score = understanding_data.get("complexity_score", 0.5)
            trace.source_priority = understanding_data.get("source_priority", {})

        # Analyze and record step 1
        step1 = self._analyze_results(step1_result, query, 1, "Initial comprehensive search")
        trace.add_step(step1)

        # Collect results (batch fetch document sources for all chunks)
        chunk_sources = await self._get_chunk_sources_batch(step1_result.chunks)
        for chunk, score in step1_result.chunks:
            source = chunk_sources.get(str(chunk.id), "unknown")
            all_chunks[str(chunk.id)] = (chunk, score, source)

        for entity, score in step1_result.entities:
            all_entities[str(entity.id)] = (entity, score)

        # Step 2+: Execute pre-computed follow-up queries
        if max_steps >= 2 and "understanding" in step1_result.metadata:
            follow_up_queries = step1_result.metadata["understanding"].get("follow_up_queries", [])

            # Also check for queries generated from result analysis
            additional_follow_ups = self._generate_additional_follow_ups(step1_result, step1)
            all_follow_ups = list(follow_up_queries) + additional_follow_ups

            for i, follow_up in enumerate(all_follow_ups[: max_steps - 1]):
                if isinstance(follow_up, dict):
                    fq_query = follow_up.get("query", "")
                    fq_reasoning = follow_up.get("reasoning", f"Follow-up query {i + 1}")
                else:
                    fq_query = str(follow_up)
                    fq_reasoning = f"Pre-computed follow-up query {i + 1}"

                if not fq_query:
                    continue

                logger.info(f"Agentic search step {i + 2}: '{fq_query[:50]}...'")

                step_result = await self._engine.query(
                    fq_query,
                    namespace_id,
                    config=config,
                )

                step = self._analyze_results(step_result, fq_query, i + 2, fq_reasoning)
                trace.add_step(step)

                # Collect new results (keep higher scores) - batch fetch sources
                step_chunk_sources = await self._get_chunk_sources_batch(step_result.chunks)
                for chunk, score in step_result.chunks:
                    chunk_id = str(chunk.id)
                    if chunk_id not in all_chunks or all_chunks[chunk_id][1] < score:
                        source = step_chunk_sources.get(chunk_id, "unknown")
                        all_chunks[chunk_id] = (chunk, score, source)

                for entity, score in step_result.entities:
                    entity_id = str(entity.id)
                    if entity_id not in all_entities or all_entities[entity_id][1] < score:
                        all_entities[entity_id] = (entity, score)

        # Generate summary (without additional LLM call)
        summary = self._generate_summary_fast(query, all_chunks, all_entities, trace)

        # Finalize trace
        trace.completed_at = datetime.utcnow()
        trace.summary = summary
        trace.total_unique_chunks = len(all_chunks)
        trace.total_unique_entities = len(all_entities)

        # Count sources
        for _, (_, _, source) in all_chunks.items():
            trace.sources_explored[source] = trace.sources_explored.get(source, 0) + 1

        # Sort results by score
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x[1], reverse=True)
        sorted_entities = sorted(all_entities.values(), key=lambda x: x[1], reverse=True)

        # Aggregate search_methods from all steps for attribution
        aggregated_search_methods = self._aggregate_search_methods(trace.steps)

        return AgenticSearchResult(
            chunks=sorted_chunks,
            entities=sorted_entities,
            summary=summary,
            trace=trace,
            understanding=understanding,
            metadata={
                "original_query": query,
                "total_steps": len(trace.steps),
                "sources_explored": trace.sources_explored,
                "complexity_score": trace.complexity_score,
                "search_methods": aggregated_search_methods,
            },
        )

    async def search_speculative(
        self,
        query: str,
        namespace_id: UUID,
        config: QueryConfig | None = None,
        max_steps: int = 3,
    ) -> AgenticSearchResult:
        """Perform agentic search with speculative follow-up execution.

        Executes the main query, then runs all follow-up queries in parallel
        (instead of sequentially), reducing total latency.

        Args:
            query: Original search query
            namespace_id: Namespace to search in
            config: Query configuration
            max_steps: Maximum exploration steps

        Returns:
            AgenticSearchResult with combined results
        """
        import asyncio
        import uuid as uuid_mod

        trace = AgenticSearchTrace(
            session_id=str(uuid_mod.uuid4()),
            original_query=query,
        )

        all_chunks: dict[str, tuple[Chunk, float, str]] = {}
        all_entities: dict[str, tuple[Entity, float]] = {}

        # Step 1: Initial search (triggers query understanding)
        logger.info(f"Speculative search step 1: '{query[:50]}...'")
        step1_result = await self._engine.query(query, namespace_id, config=config, _lightweight_understanding=False)

        if "understanding" in step1_result.metadata:
            ud = step1_result.metadata["understanding"]
            trace.understanding_reasoning = ud.get("reasoning", "")
            trace.complexity_score = ud.get("complexity_score", 0.5)
            trace.source_priority = ud.get("source_priority", {})

        step1 = self._analyze_results(step1_result, query, 1, "Initial comprehensive search")
        trace.add_step(step1)

        chunk_sources = await self._get_chunk_sources_batch(step1_result.chunks)
        for chunk, score in step1_result.chunks:
            all_chunks[str(chunk.id)] = (chunk, score, chunk_sources.get(str(chunk.id), "unknown"))
        for entity, score in step1_result.entities:
            all_entities[str(entity.id)] = (entity, score)

        # Collect all follow-up queries
        follow_up_queries = step1_result.metadata.get("understanding", {}).get("follow_up_queries", [])
        additional = self._generate_additional_follow_ups(step1_result, step1)
        all_follow_ups = list(follow_up_queries) + additional
        follow_ups_to_run = all_follow_ups[: max_steps - 1]

        if follow_ups_to_run:
            # Execute ALL follow-ups in parallel
            async def run_follow_up(i: int, follow_up: Any) -> tuple[int, QueryResult, str]:
                fq_query = follow_up.get("query", "") if isinstance(follow_up, dict) else str(follow_up)
                fq_reasoning = (
                    follow_up.get("reasoning", f"Follow-up {i + 2}")
                    if isinstance(follow_up, dict)
                    else f"Follow-up {i + 2}"
                )
                logger.info(f"Speculative search step {i + 2}: '{fq_query[:50]}...'")
                result = await self._engine.query(fq_query, namespace_id, config=config)
                return i, result, fq_reasoning

            tasks = [
                run_follow_up(i, fu)
                for i, fu in enumerate(follow_ups_to_run)
                if (fu.get("query", "") if isinstance(fu, dict) else str(fu))
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Speculative follow-up failed: {r}")
                    continue
                i, step_result, reasoning = r  # type: ignore[not-iterable]
                fq_query = follow_ups_to_run[i]
                fq_query_text = fq_query.get("query", "") if isinstance(fq_query, dict) else str(fq_query)

                step = self._analyze_results(step_result, fq_query_text, i + 2, reasoning)
                trace.add_step(step)

                step_sources = await self._get_chunk_sources_batch(step_result.chunks)
                for chunk, score in step_result.chunks:
                    cid = str(chunk.id)
                    if cid not in all_chunks or all_chunks[cid][1] < score:
                        all_chunks[cid] = (chunk, score, step_sources.get(cid, "unknown"))
                for entity, score in step_result.entities:
                    eid = str(entity.id)
                    if eid not in all_entities or all_entities[eid][1] < score:
                        all_entities[eid] = (entity, score)

        summary = self._generate_summary_fast(query, all_chunks, all_entities, trace)
        trace.completed_at = datetime.utcnow()
        trace.summary = summary
        trace.total_unique_chunks = len(all_chunks)
        trace.total_unique_entities = len(all_entities)

        for _, (_, _, source) in all_chunks.items():
            trace.sources_explored[source] = trace.sources_explored.get(source, 0) + 1

        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x[1], reverse=True)
        sorted_entities = sorted(all_entities.values(), key=lambda x: x[1], reverse=True)
        aggregated_search_methods = self._aggregate_search_methods(trace.steps)

        return AgenticSearchResult(
            chunks=sorted_chunks,
            entities=sorted_entities,
            summary=summary,
            trace=trace,
            understanding=None,
            metadata={
                "original_query": query,
                "total_steps": len(trace.steps),
                "sources_explored": trace.sources_explored,
                "complexity_score": trace.complexity_score,
                "search_methods": aggregated_search_methods,
                "speculative": True,
            },
        )

    async def search_stream(
        self,
        query: str,
        namespace_id: UUID,
        config: QueryConfig | None = None,
        max_steps: int = 3,
    ) -> AsyncGenerator[AgenticSearchStepResult]:
        """Stream search results as each step completes.

        Yields partial results after each exploration step,
        allowing callers to display incremental progress.

        Args:
            query: Original search query
            namespace_id: Namespace to search in
            config: Query configuration
            max_steps: Maximum exploration steps

        Yields:
            AgenticSearchStepResult for each completed step
        """
        all_chunks: dict[str, tuple[Chunk, float, str]] = {}
        all_entities: dict[str, tuple[Entity, float]] = {}

        # Step 1: Initial search
        logger.info(f"Agentic stream step 1: '{query[:50]}...'")
        step1_result = await self._engine.query(query, namespace_id, config=config, _lightweight_understanding=False)

        chunk_sources = await self._get_chunk_sources_batch(step1_result.chunks)
        for chunk, score in step1_result.chunks:
            source = chunk_sources.get(str(chunk.id), "unknown")
            all_chunks[str(chunk.id)] = (chunk, score, source)
        for entity, score in step1_result.entities:
            all_entities[str(entity.id)] = (entity, score)

        is_last = max_steps <= 1
        yield AgenticSearchStepResult(
            step_number=1,
            query=query,
            chunks=sorted(all_chunks.values(), key=lambda x: x[1], reverse=True),
            entities=sorted(all_entities.values(), key=lambda x: x[1], reverse=True),
            is_final=is_last,
        )

        if is_last:
            return

        # Follow-up steps
        follow_up_queries = step1_result.metadata.get("understanding", {}).get("follow_up_queries", [])
        step1_analysis = self._analyze_results(step1_result, query, 1, "Initial search")
        additional = self._generate_additional_follow_ups(step1_result, step1_analysis)
        all_follow_ups = list(follow_up_queries) + additional

        for i, follow_up in enumerate(all_follow_ups[: max_steps - 1]):
            fq_query = follow_up.get("query", "") if isinstance(follow_up, dict) else str(follow_up)
            if not fq_query:
                continue

            logger.info(f"Agentic stream step {i + 2}: '{fq_query[:50]}...'")
            step_result = await self._engine.query(fq_query, namespace_id, config=config)

            step_sources = await self._get_chunk_sources_batch(step_result.chunks)
            for chunk, score in step_result.chunks:
                cid = str(chunk.id)
                if cid not in all_chunks or all_chunks[cid][1] < score:
                    all_chunks[cid] = (chunk, score, step_sources.get(cid, "unknown"))
            for entity, score in step_result.entities:
                eid = str(entity.id)
                if eid not in all_entities or all_entities[eid][1] < score:
                    all_entities[eid] = (entity, score)

            is_last = i + 2 >= max_steps or i + 1 >= len(all_follow_ups[: max_steps - 1])
            yield AgenticSearchStepResult(
                step_number=i + 2,
                query=fq_query,
                chunks=sorted(all_chunks.values(), key=lambda x: x[1], reverse=True),
                entities=sorted(all_entities.values(), key=lambda x: x[1], reverse=True),
                is_final=is_last,
            )

    def _analyze_results(
        self,
        result: QueryResult,
        query: str,
        step_number: int,
        reasoning: str,
    ) -> SearchStep:
        """Analyze search results and create a step record."""
        step = SearchStep(
            step_number=step_number,
            query=query,
            reasoning=reasoning,
        )

        step.total_chunks = len(result.chunks)
        step.total_entities = len(result.entities)

        # Extract search method contributions
        if result.search_contributions:
            step.vector_hits = result.search_contributions.vector.chunk_count
            step.graph_hits = result.search_contributions.graph.chunk_count
            step.keyword_hits = result.search_contributions.keyword.chunk_count
            # Store full search methods data for attribution
            step.search_methods_data = result.search_contributions.to_dict()

        # Extract graph info
        if result.graph_info:
            step.entities_linked = result.graph_info.entities_linked
            step.relationships_traversed = result.graph_info.relationships_traversed

        # Extract temporal info
        if result.temporal_info:
            step.temporal_filter_applied = result.temporal_info.filter_applied
            if result.temporal_info.time_start or result.temporal_info.time_end:
                step.time_range = (result.temporal_info.time_start, result.temporal_info.time_end)

        return step

    def _generate_additional_follow_ups(
        self,
        result: QueryResult,
        analysis: SearchStep,
    ) -> list[dict[str, Any]]:
        """Generate additional follow-up queries based on result analysis.

        These are computed locally without LLM calls, based on:
        - Under-represented sources
        - High-scoring entities found
        """
        follow_ups = []

        # Check for source imbalance
        if analysis.sources_hit:
            total_hits = sum(analysis.sources_hit.values())
            if total_hits > 0:
                dominant_source = max(analysis.sources_hit.items(), key=lambda x: x[1])
                if dominant_source[1] / total_hits > 0.8:
                    # One source dominates - target others
                    for source in ["linear", "notion", "attio", "gong"]:
                        if source != dominant_source[0] and analysis.sources_hit.get(source, 0) == 0:
                            follow_ups.append(
                                {
                                    "query": f"{analysis.query} {source}",
                                    "reasoning": f"Targeting under-represented source: {source}",
                                }
                            )
                            break

        # Explore top entities
        if result.entities and len(result.entities) > 0:
            top_entity = result.entities[0][0]
            follow_ups.append(
                {
                    "query": f"{top_entity.name} context details",
                    "reasoning": f"Exploring top entity: {top_entity.name}",
                }
            )

        return follow_ups[:2]  # Limit to 2 additional

    async def _get_chunk_source(self, chunk: Chunk, namespace_id: UUID) -> str:
        """Get the source system for a chunk."""
        try:
            doc = await self._engine._storage.get_document(chunk.document_id)
            if doc:
                source = (doc.metadata or {}).get("source_system", "")
                if not source and doc.source:
                    source = doc.source.split("/")[0]
                return source or "unknown"
        except Exception as e:
            logger.debug(f"Failed to get source system for chunk {chunk.id}: {e}")
        return "unknown"

    async def _get_chunk_sources_batch(self, chunks: list[tuple[Chunk, float]]) -> dict[str, str]:
        """Get source systems for multiple chunks in a single batch query.

        Args:
            chunks: List of (chunk, score) tuples

        Returns:
            Dictionary mapping chunk ID to source string
        """
        if not chunks:
            return {}

        # Collect unique document IDs
        doc_ids = list({chunk.document_id for chunk, _ in chunks})

        # Batch fetch all documents
        docs_map = await self._engine._storage.get_documents_batch(doc_ids)

        # Build chunk_id -> source mapping
        sources: dict[str, str] = {}
        for chunk, _ in chunks:
            doc = docs_map.get(chunk.document_id)
            if doc:
                source = (doc.metadata or {}).get("source_system", "")
                if not source and doc.source:
                    source = doc.source.split("/")[0]
                sources[str(chunk.id)] = source or "unknown"
            else:
                sources[str(chunk.id)] = "unknown"

        return sources

    def _aggregate_search_methods(self, steps: list[SearchStep]) -> dict[str, Any]:
        """Aggregate search methods data from all steps for attribution.

        Combines the chunk/entity IDs from all steps so results can be
        attributed to their source search method(s).
        """
        # Aggregate all chunk and entity IDs by method
        all_vector_chunks: set[str] = set()
        all_graph_chunks: set[str] = set()
        all_keyword_chunks: set[str] = set()
        all_vector_entities: set[str] = set()
        all_graph_entities: set[str] = set()

        for step in steps:
            if not step.search_methods_data:
                continue

            by_method = step.search_methods_data.get("by_method", {})

            # Collect chunk IDs from each method
            vector_data = by_method.get("vector", {})
            all_vector_chunks.update(vector_data.get("chunks", {}).get("ids", []))
            all_vector_entities.update(vector_data.get("entities", {}).get("ids", []))

            graph_data = by_method.get("graph", {})
            all_graph_chunks.update(graph_data.get("chunks", {}).get("ids", []))
            all_graph_entities.update(graph_data.get("entities", {}).get("ids", []))

            keyword_data = by_method.get("keyword", {})
            all_keyword_chunks.update(keyword_data.get("chunks", {}).get("ids", []))

        # Compute overlaps
        vector_graph_overlap = all_vector_chunks & all_graph_chunks
        vector_keyword_overlap = all_vector_chunks & all_keyword_chunks
        graph_keyword_overlap = all_graph_chunks & all_keyword_chunks
        all_methods_overlap = all_vector_chunks & all_graph_chunks & all_keyword_chunks

        vector_only = all_vector_chunks - all_graph_chunks - all_keyword_chunks
        graph_only = all_graph_chunks - all_vector_chunks - all_keyword_chunks
        keyword_only = all_keyword_chunks - all_vector_chunks - all_graph_chunks

        vector_graph_entity_overlap = all_vector_entities & all_graph_entities
        vector_only_entities = all_vector_entities - all_graph_entities
        graph_only_entities = all_graph_entities - all_vector_entities

        return {
            "chunk_overlap": {
                "vector_only": {"count": len(vector_only), "ids": list(vector_only)},
                "graph_only": {"count": len(graph_only), "ids": list(graph_only)},
                "keyword_only": {"count": len(keyword_only), "ids": list(keyword_only)},
                "vector_and_graph": {"count": len(vector_graph_overlap), "ids": list(vector_graph_overlap)},
                "vector_and_keyword": {"count": len(vector_keyword_overlap), "ids": list(vector_keyword_overlap)},
                "graph_and_keyword": {"count": len(graph_keyword_overlap), "ids": list(graph_keyword_overlap)},
                "all_three_methods": {"count": len(all_methods_overlap), "ids": list(all_methods_overlap)},
            },
            "entity_overlap": {
                "vector_only": {"count": len(vector_only_entities), "ids": list(vector_only_entities)},
                "graph_only": {"count": len(graph_only_entities), "ids": list(graph_only_entities)},
                "vector_and_graph": {
                    "count": len(vector_graph_entity_overlap),
                    "ids": list(vector_graph_entity_overlap),
                },
            },
        }

    def _generate_summary_fast(
        self,
        query: str,
        chunks: dict[str, tuple[Chunk, float, str]],
        entities: dict[str, tuple[Entity, float]],
        trace: AgenticSearchTrace,
    ) -> str:
        """Generate a summary without additional LLM call.

        Uses structured data from the search to create a useful summary.
        """
        sources = {}
        for _, (_, _, source) in chunks.items():
            sources[source] = sources.get(source, 0) + 1

        source_parts = [f"{s}: {c}" for s, c in sorted(sources.items(), key=lambda x: -x[1])]
        source_summary = ", ".join(source_parts) if source_parts else "none"

        entity_names = [e.name for e, _ in sorted(entities.values(), key=lambda x: -x[1])[:5]]
        entity_summary = ", ".join(entity_names) if entity_names else "none found"

        # Build structured summary
        parts = [
            f"Found {len(chunks)} results across {len(sources)} sources ({source_summary}).",
        ]

        if entity_names:
            parts.append(f"Key entities: {entity_summary}.")

        parts.append(f"Explored in {len(trace.steps)} steps.")

        if trace.complexity_score > 0.7:
            parts.append("Query was identified as complex, requiring multi-step exploration.")

        return " ".join(parts)
