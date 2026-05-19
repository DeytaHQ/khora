"""Coverage-driven tests for ``khora.query.agentic``.

The orchestrator drives an injected ``HybridQueryEngine`` plus its
storage. Both are fully mocked. Tests exercise:
 * the dataclass shapes (SearchStep, AgenticSearchTrace.to_dict)
 * the local helpers (_analyze_results, _generate_additional_follow_ups,
   _get_chunk_source, _get_chunk_sources_batch, _aggregate_search_methods,
   _generate_summary_fast)
 * the public ``search``, ``search_speculative``, ``search_stream``
   entry points with a stub engine
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models.document import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
)
from khora.core.models.entity import Entity
from khora.query.agentic import (
    AgenticSearchAgent,
    AgenticSearchTrace,
    SearchStep,
)
from khora.query.engine import (
    GraphTraversalInfo,
    QueryResult,
    SearchMethodContribution,
    SearchMethodStats,
    TemporalInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(doc_id: UUID | None = None, content: str = "x") -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=doc_id or uuid4(),
        content=content,
        metadata=ChunkMetadata(),
    )


def _entity(name: str = "Alice") -> Entity:
    return Entity(id=uuid4(), name=name, entity_type="PERSON")


def _doc(
    *,
    doc_id: UUID,
    source_system: str = "",
    source: str = "",
) -> Document:
    meta = DocumentMetadata()
    meta.source = source
    if source_system:
        meta.custom = {"source_system": source_system}
    return Document(id=doc_id, metadata=meta)


def _query_result(
    *,
    chunks: list[tuple[Chunk, float]] | None = None,
    entities: list[tuple[Entity, float]] | None = None,
    understanding: dict | None = None,
    search_contrib: SearchMethodContribution | None = None,
    graph_info: GraphTraversalInfo | None = None,
    temporal_info: TemporalInfo | None = None,
) -> QueryResult:
    meta: dict = {}
    if understanding is not None:
        meta["understanding"] = understanding
    return QueryResult(
        chunks=chunks or [],
        entities=entities or [],
        metadata=meta,
        search_contributions=search_contrib,
        graph_info=graph_info,
        temporal_info=temporal_info,
    )


def _make_engine_with_results(results: list[QueryResult]) -> MagicMock:
    """Build a fake engine whose `.query(...)` returns successive results."""
    engine = MagicMock()
    engine.query = AsyncMock(side_effect=results)
    engine._storage = MagicMock()
    engine._storage.get_document = AsyncMock(return_value=None)
    engine._storage.get_documents_batch = AsyncMock(return_value={})
    return engine


# ---------------------------------------------------------------------------
# SearchStep / AgenticSearchTrace dataclasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraceToDict:
    def test_round_trips_minimal(self) -> None:
        trace = AgenticSearchTrace(session_id="s1", original_query="q")
        trace.steps.append(SearchStep(step_number=1, query="q", reasoning="r"))
        d = trace.to_dict()
        assert d["session_id"] == "s1"
        assert d["steps"][0]["query"] == "q"
        assert d["steps"][0]["time_range"] is None

    def test_includes_completed_at_and_time_range(self) -> None:
        ts = datetime(2026, 5, 18, tzinfo=UTC)
        trace = AgenticSearchTrace(
            session_id="s2",
            original_query="q",
            completed_at=ts,
        )
        step = SearchStep(
            step_number=1,
            query="q",
            reasoning="r",
            time_range=(ts, ts),
            relationships_traversed=[("a", "REL", "b")],
        )
        trace.steps.append(step)
        d = trace.to_dict()
        assert d["completed_at"] == ts.isoformat()
        assert d["steps"][0]["time_range"][0] == ts.isoformat()
        assert d["steps"][0]["relationships_traversed"][0]["from"] == "a"


# ---------------------------------------------------------------------------
# _analyze_results
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeResults:
    def test_extracts_basic_counts(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        result = _query_result(chunks=[(_chunk(), 0.5)], entities=[(_entity(), 0.9)])
        step = agent._analyze_results(result, "q", 1, "r")
        assert step.total_chunks == 1
        assert step.total_entities == 1
        assert step.step_number == 1

    def test_extracts_search_contributions(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        contrib = SearchMethodContribution(
            vector=SearchMethodStats(chunk_count=3),
            graph=SearchMethodStats(chunk_count=2),
            keyword=SearchMethodStats(chunk_count=1),
        )
        result = _query_result(search_contrib=contrib)
        step = agent._analyze_results(result, "q", 1, "r")
        assert step.vector_hits == 3
        assert step.graph_hits == 2
        assert step.keyword_hits == 1
        assert step.search_methods_data  # populated

    def test_extracts_graph_info(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        gi = GraphTraversalInfo(
            entities_linked=["alice"],
            relationships_traversed=[("alice", "WORKS_ON", "phoenix")],
        )
        result = _query_result(graph_info=gi)
        step = agent._analyze_results(result, "q", 1, "r")
        assert step.entities_linked == ["alice"]
        assert step.relationships_traversed == [("alice", "WORKS_ON", "phoenix")]

    def test_extracts_temporal_info(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        ti = TemporalInfo(
            detected=True,
            filter_applied=True,
            time_start=ts,
            time_end=ts,
        )
        result = _query_result(temporal_info=ti)
        step = agent._analyze_results(result, "q", 1, "r")
        assert step.temporal_filter_applied is True
        assert step.time_range == (ts, ts)


# ---------------------------------------------------------------------------
# _generate_additional_follow_ups
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateAdditionalFollowUps:
    def test_targets_under_represented_source_when_one_dominates(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        analysis = SearchStep(
            step_number=1,
            query="q",
            reasoning="r",
            sources_hit={"slack": 10},
        )
        result = _query_result()  # no top entity
        out = agent._generate_additional_follow_ups(result, analysis)
        assert len(out) >= 1
        # One of the candidate sources should appear
        assert any(
            "linear" in f["query"] or "notion" in f["query"] or "attio" in f["query"] or "gong" in f["query"]
            for f in out
        )

    def test_explores_top_entity(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        analysis = SearchStep(step_number=1, query="q", reasoning="r")
        e = _entity("Phoenix")
        result = _query_result(entities=[(e, 0.9)])
        out = agent._generate_additional_follow_ups(result, analysis)
        assert any("Phoenix" in f["query"] for f in out)

    def test_caps_at_two(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        analysis = SearchStep(step_number=1, query="q", reasoning="r", sources_hit={"slack": 10})
        e = _entity("Top")
        result = _query_result(entities=[(e, 0.9)])
        out = agent._generate_additional_follow_ups(result, analysis)
        assert len(out) <= 2


# ---------------------------------------------------------------------------
# _get_chunk_source / _get_chunk_sources_batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkSources:
    async def test_get_chunk_source_uses_source_system_when_present(
        self,
    ) -> None:
        c = _chunk()
        doc = _doc(doc_id=c.document_id, source_system="slack")
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.get_document = AsyncMock(return_value=doc)
        agent = AgenticSearchAgent(engine=engine)
        out = await agent._get_chunk_source(c, uuid4())
        assert out == "slack"

    async def test_get_chunk_source_falls_back_to_source_field(self) -> None:
        c = _chunk()
        doc = _doc(doc_id=c.document_id, source="github/api/x")
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.get_document = AsyncMock(return_value=doc)
        agent = AgenticSearchAgent(engine=engine)
        out = await agent._get_chunk_source(c, uuid4())
        assert out == "github"

    async def test_get_chunk_source_returns_unknown_on_missing_doc(self) -> None:
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.get_document = AsyncMock(return_value=None)
        agent = AgenticSearchAgent(engine=engine)
        out = await agent._get_chunk_source(_chunk(), uuid4())
        assert out == "unknown"

    async def test_get_chunk_source_swallows_exception(self) -> None:
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.get_document = AsyncMock(side_effect=RuntimeError("db boom"))
        agent = AgenticSearchAgent(engine=engine)
        out = await agent._get_chunk_source(_chunk(), uuid4())
        assert out == "unknown"

    async def test_get_chunk_sources_batch_empty(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        out = await agent._get_chunk_sources_batch([])
        assert out == {}

    async def test_get_chunk_sources_batch_populates_map(self) -> None:
        c1 = _chunk()
        c2 = _chunk()
        doc1 = _doc(doc_id=c1.document_id, source_system="slack")
        doc2 = _doc(doc_id=c2.document_id, source="github/x")
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.get_documents_batch = AsyncMock(return_value={c1.document_id: doc1, c2.document_id: doc2})
        agent = AgenticSearchAgent(engine=engine)
        out = await agent._get_chunk_sources_batch([(c1, 0.5), (c2, 0.5)])
        assert out[str(c1.id)] == "slack"
        assert out[str(c2.id)] == "github"

    async def test_get_chunk_sources_batch_unknown_when_doc_missing(
        self,
    ) -> None:
        c = _chunk()
        engine = MagicMock()
        engine._storage = MagicMock()
        engine._storage.get_documents_batch = AsyncMock(return_value={})
        agent = AgenticSearchAgent(engine=engine)
        out = await agent._get_chunk_sources_batch([(c, 0.5)])
        assert out[str(c.id)] == "unknown"


# ---------------------------------------------------------------------------
# _aggregate_search_methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAggregateSearchMethods:
    def test_empty_steps_returns_zero_counts(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        out = agent._aggregate_search_methods([])
        assert out["chunk_overlap"]["vector_only"]["count"] == 0
        assert out["entity_overlap"]["vector_only"]["count"] == 0

    def test_aggregates_chunk_ids_across_steps(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        step = SearchStep(
            step_number=1,
            query="q",
            reasoning="r",
            search_methods_data={
                "by_method": {
                    "vector": {
                        "chunks": {"ids": ["c1", "c2"]},
                        "entities": {"ids": ["e1"]},
                    },
                    "graph": {
                        "chunks": {"ids": ["c2", "c3"]},
                        "entities": {"ids": ["e2"]},
                    },
                    "keyword": {
                        "chunks": {"ids": ["c3"]},
                        "entities": {"ids": []},
                    },
                }
            },
        )
        out = agent._aggregate_search_methods([step])
        co = out["chunk_overlap"]
        assert co["vector_only"]["count"] == 1  # c1
        assert co["vector_and_graph"]["count"] == 1  # c2
        assert co["graph_and_keyword"]["count"] == 1  # c3
        # Entity overlap
        eo = out["entity_overlap"]
        assert eo["vector_only"]["count"] == 1
        assert eo["graph_only"]["count"] == 1

    def test_skips_step_with_no_search_methods_data(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        step = SearchStep(step_number=1, query="q", reasoning="r")
        out = agent._aggregate_search_methods([step])
        assert out["chunk_overlap"]["vector_only"]["count"] == 0


# ---------------------------------------------------------------------------
# _generate_summary_fast
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateSummaryFast:
    def test_includes_source_counts(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        c = _chunk()
        chunks = {str(c.id): (c, 0.9, "slack")}
        entities: dict = {}
        trace = AgenticSearchTrace(session_id="s", original_query="q")
        trace.steps.append(SearchStep(step_number=1, query="q", reasoning="r"))
        out = agent._generate_summary_fast("q", chunks, entities, trace)
        assert "slack: 1" in out
        assert "1 step" in out

    def test_flags_complex_queries(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        trace = AgenticSearchTrace(session_id="s", original_query="q", complexity_score=0.9)
        out = agent._generate_summary_fast("q", {}, {}, trace)
        assert "complex" in out

    def test_lists_top_entities(self) -> None:
        agent = AgenticSearchAgent(engine=MagicMock())
        e1 = _entity("Alpha")
        e2 = _entity("Beta")
        entities = {str(e1.id): (e1, 0.9), str(e2.id): (e2, 0.5)}
        trace = AgenticSearchTrace(session_id="s", original_query="q")
        out = agent._generate_summary_fast("q", {}, entities, trace)
        assert "Alpha" in out


# ---------------------------------------------------------------------------
# search() — end-to-end (engine fully mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchEndToEnd:
    async def test_single_step_search(self) -> None:
        c = _chunk()
        e = _entity("Phoenix")
        result1 = _query_result(
            chunks=[(c, 0.9)],
            entities=[(e, 0.8)],
            understanding={
                "reasoning": "simple",
                "complexity_score": 0.2,
                "source_priority": {"slack": 0.9},
                "follow_up_queries": [],
            },
        )
        engine = _make_engine_with_results([result1])
        agent = AgenticSearchAgent(engine=engine)
        out = await agent.search("hi", namespace_id=uuid4(), max_steps=1)
        assert len(out.chunks) == 1
        assert out.trace is not None
        assert out.trace.complexity_score == 0.2

    async def test_multi_step_search_with_follow_ups(self) -> None:
        c1 = _chunk()
        e1 = _entity("Phoenix")
        c2 = _chunk()
        result1 = _query_result(
            chunks=[(c1, 0.9)],
            entities=[(e1, 0.8)],
            understanding={
                "reasoning": "complex",
                "complexity_score": 0.8,
                "source_priority": {"slack": 0.9},
                "follow_up_queries": [{"query": "deeper", "reasoning": "more"}],
            },
        )
        result2 = _query_result(
            chunks=[(c2, 0.7)],
            entities=[],
            understanding=None,
        )
        engine = _make_engine_with_results([result1, result2])
        agent = AgenticSearchAgent(engine=engine)
        out = await agent.search("hi", namespace_id=uuid4(), max_steps=2)
        # Both chunks present
        ids = {chunk.id for chunk, _, _ in out.chunks}
        assert c1.id in ids
        assert c2.id in ids
        # Trace has 2 steps
        assert len(out.trace.steps) == 2

    async def test_skips_empty_follow_up_query(self) -> None:
        c = _chunk()
        result1 = _query_result(
            chunks=[(c, 0.9)],
            understanding={
                "reasoning": "",
                "complexity_score": 0.1,
                "source_priority": {},
                "follow_up_queries": [{"query": "", "reasoning": "skip me"}],
            },
        )
        # Only one engine.query() call expected since the empty follow-up is skipped
        engine = _make_engine_with_results([result1])
        agent = AgenticSearchAgent(engine=engine)
        out = await agent.search("hi", namespace_id=uuid4(), max_steps=3)
        assert len(out.trace.steps) == 1


@pytest.mark.unit
class TestSearchSpeculative:
    async def test_runs_followups_in_parallel(self) -> None:
        c1 = _chunk()
        c2 = _chunk()
        c3 = _chunk()
        result1 = _query_result(
            chunks=[(c1, 0.9)],
            understanding={
                "reasoning": "",
                "complexity_score": 0.5,
                "source_priority": {},
                "follow_up_queries": [
                    {"query": "a", "reasoning": "x"},
                    {"query": "b", "reasoning": "y"},
                ],
            },
        )
        result2 = _query_result(chunks=[(c2, 0.7)])
        result3 = _query_result(chunks=[(c3, 0.6)])
        engine = _make_engine_with_results([result1, result2, result3])
        agent = AgenticSearchAgent(engine=engine)
        out = await agent.search_speculative("hi", namespace_id=uuid4(), max_steps=3)
        ids = {chunk.id for chunk, _, _ in out.chunks}
        assert c1.id in ids
        assert c2.id in ids
        assert c3.id in ids
        assert out.metadata["speculative"] is True

    async def test_handles_followup_exception(self) -> None:
        c1 = _chunk()
        result1 = _query_result(
            chunks=[(c1, 0.9)],
            understanding={
                "reasoning": "",
                "complexity_score": 0.5,
                "source_priority": {},
                "follow_up_queries": [{"query": "a", "reasoning": "x"}],
            },
        )
        engine = MagicMock()
        # First call returns result1; second call raises
        engine.query = AsyncMock(side_effect=[result1, RuntimeError("boom")])
        engine._storage = MagicMock()
        engine._storage.get_documents_batch = AsyncMock(return_value={})
        agent = AgenticSearchAgent(engine=engine)
        out = await agent.search_speculative("q", namespace_id=uuid4(), max_steps=2)
        # One step recorded (step 1), follow-up failed silently
        assert len(out.trace.steps) == 1


@pytest.mark.unit
class TestSearchStream:
    async def test_yields_only_first_step_when_max_steps_is_one(self) -> None:
        c = _chunk()
        engine = _make_engine_with_results([_query_result(chunks=[(c, 0.9)])])
        agent = AgenticSearchAgent(engine=engine)
        steps = []
        async for sr in agent.search_stream("q", uuid4(), max_steps=1):
            steps.append(sr)
        assert len(steps) == 1
        assert steps[0].is_final is True

    async def test_yields_multiple_steps(self) -> None:
        c1 = _chunk()
        c2 = _chunk()
        engine = _make_engine_with_results(
            [
                _query_result(
                    chunks=[(c1, 0.9)],
                    understanding={"follow_up_queries": [{"query": "more", "reasoning": "x"}]},
                ),
                _query_result(chunks=[(c2, 0.5)]),
            ]
        )
        agent = AgenticSearchAgent(engine=engine)
        steps = []
        async for sr in agent.search_stream("q", uuid4(), max_steps=2):
            steps.append(sr)
        assert len(steps) == 2
        assert steps[-1].is_final is True

    async def test_skips_empty_follow_up_in_stream(self) -> None:
        c1 = _chunk()
        engine = _make_engine_with_results(
            [
                _query_result(
                    chunks=[(c1, 0.9)],
                    understanding={"follow_up_queries": [{"query": "", "reasoning": "skip"}]},
                )
            ]
        )
        agent = AgenticSearchAgent(engine=engine)
        steps = []
        async for sr in agent.search_stream("q", uuid4(), max_steps=2):
            steps.append(sr)
        # Only first step — empty follow-up skipped
        assert len(steps) == 1
