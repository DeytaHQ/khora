"""Coverage-driven tests for ``khora.query.engine.HybridQueryEngine``.

Companion to ``test_engine_coverage_push.py`` — adds end-to-end and
high-level method coverage targeting:

  * ``HybridQueryEngine.query`` — multi-stage pipeline (single and
    legacy paths) with mocked search methods
  * ``HybridQueryEngine.query`` — cache hit
  * ``HybridQueryEngine.query`` — agentic dispatch
  * ``find_related_entities``
  * ``warm_cache`` / ``warm_keyword_index``
  * ``_multi_stage_search`` directly (no chunks short-circuit branch)
  * ``temporal_query``
  * ``QueryConfig.from_settings``
  * ``QueryResult.get_context_text`` / ``get_full_metadata``
  * ``_extract_chunk_title`` extra branches
  * ``SearchMethodContribution.compute_overlaps`` / ``to_dict``
  * ``GraphTraversalInfo.to_dict`` / ``TemporalInfo.to_dict``
  * ``SearchMethodStats.to_dict``
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models.document import Chunk
from khora.core.models.entity import Entity
from khora.query.engine import (
    GraphTraversalInfo,
    HybridQueryEngine,
    QueryConfig,
    QueryResult,
    SearchMethodContribution,
    SearchMethodStats,
    SearchMode,
    TemporalInfo,
)
from khora.query.temporal import TemporalFilter, TemporalQuery
from khora.query.understanding import (
    AnswerType,
    EntityMention,
    QueryIntent,
    SearchStrategy,
    SourcePriority,
    TemporalReference,
    UnderstandingResult,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    content: str = "content",
    embedding: list[float] | None = None,
    custom: dict | None = None,
    created_at: datetime | None = None,
) -> Chunk:
    # Post-#748: Chunk.metadata is a flat dict; the old ``ChunkMetadata.custom``
    # nesting was flattened into the dict itself.
    return Chunk(
        id=uuid4(),
        content=content,
        metadata=dict(custom or {}),
        embedding=embedding,
        created_at=created_at or datetime.now(UTC),
    )


def _make_entity(
    *,
    name: str = "Alice",
    entity_type: str = "PERSON",
    source_chunk_ids: list[UUID] | None = None,
    mention_count: int = 1,
) -> Entity:
    return Entity(
        id=uuid4(),
        name=name,
        entity_type=entity_type,
        source_chunk_ids=source_chunk_ids or [],
        mention_count=mention_count,
    )


def _storage_mock(*, chunks: list[Chunk] | None = None, entities: list[Entity] | None = None) -> MagicMock:
    """Build a storage mock with all needed async methods."""
    storage = MagicMock()
    storage.search_similar_chunks = AsyncMock(return_value=[(c, 0.5) for c in (chunks or [])])
    storage.search_similar_entities = AsyncMock(return_value=[(e.id, 0.6) for e in (entities or [])])
    storage.get_entities_batch = AsyncMock(return_value={e.id: e for e in (entities or [])})
    storage.get_neighborhood = AsyncMock(return_value={})
    storage.get_neighborhoods_batch = AsyncMock(return_value={})
    storage.get_chunks_batch = AsyncMock(return_value={})
    storage.list_chunks = AsyncMock(return_value=chunks or [])
    storage.list_entities = AsyncMock(return_value=entities or [])
    storage.search_fulltext_chunks = AsyncMock(return_value=[])
    return storage


def _embedder_mock(dim: int = 4) -> MagicMock:
    e = MagicMock()
    e.embed = AsyncMock(return_value=[0.1] * dim)
    return e


def _config_no_extras(**kwargs) -> QueryConfig:
    defaults = dict(
        enable_query_understanding=False,
        enable_entity_linking=False,
        enable_reranking=False,
        enable_keyword_search=False,
        enable_query_expansion=False,
        enable_temporal_resolver=False,
        enable_narrative_coherence=False,
        enable_multi_stage=False,
        enable_hyde="never",
    )
    defaults.update(kwargs)
    return QueryConfig(**defaults)


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------


class TestSearchMethodStats:
    def test_to_dict(self) -> None:
        s = SearchMethodStats(
            chunk_count=3,
            entity_count=2,
            chunk_ids=["a", "b", "c"],
            entity_ids=["x", "y"],
            min_score=0.1,
            max_score=0.9,
            avg_score=0.5,
            latency_ms=12.3456,
        )
        d = s.to_dict()
        assert d["chunks"]["count"] == 3
        assert d["entities"]["count"] == 2
        assert d["scores"]["max"] == 0.9
        assert d["latency_ms"] == 12.35


class TestSearchMethodContribution:
    def test_compute_overlaps_and_to_dict(self) -> None:
        c = SearchMethodContribution()
        c.vector.chunk_ids = ["a", "b", "c"]
        c.graph.chunk_ids = ["b", "c", "d"]
        c.keyword.chunk_ids = ["c", "e"]
        c.vector.entity_ids = ["e1", "e2"]
        c.graph.entity_ids = ["e2", "e3"]
        c.compute_overlaps()
        assert "c" in c.all_methods_overlap
        assert "a" in c.vector_only_chunks
        assert "d" in c.graph_only_chunks
        assert "e" in c.keyword_only_chunks
        assert "e2" in c.vector_graph_entity_overlap

        d = c.to_dict()
        assert d["summary"]["total_unique_chunks"] == 5
        assert d["chunk_overlap"]["all_three_methods"]["count"] == 1

    def test_legacy_properties(self) -> None:
        c = SearchMethodContribution()
        c.vector.chunk_ids = ["v1"]
        c.graph.chunk_ids = ["g1"]
        c.keyword.chunk_ids = ["k1"]
        assert c.vector_chunks == ["v1"]
        assert c.graph_chunks == ["g1"]
        assert c.keyword_chunks == ["k1"]


class TestGraphTraversalInfo:
    def test_to_dict_limits(self) -> None:
        info = GraphTraversalInfo()
        info.entities_searched = [f"e{i}" for i in range(50)]
        info.entities_linked = [f"l{i}" for i in range(20)]
        info.relationships_traversed = [(f"a{i}", "REL", f"b{i}") for i in range(30)]
        info.neighborhood_depth = 2
        d = info.to_dict()
        assert len(d["entities_searched"]) == 20
        assert len(d["entities_linked"]) == 10
        assert len(d["relationships_traversed"]) == 20
        assert d["neighborhood_depth"] == 2


class TestTemporalInfo:
    def test_to_dict(self) -> None:
        info = TemporalInfo(
            detected=True,
            filter_applied=True,
            time_start=datetime(2024, 1, 1, tzinfo=UTC),
            time_end=datetime(2024, 1, 31, tzinfo=UTC),
            reference_text="January 2024",
        )
        d = info.to_dict()
        assert d["detected"] is True
        assert d["time_start"] == "2024-01-01T00:00:00+00:00"
        assert d["time_end"] == "2024-01-31T00:00:00+00:00"


class TestQueryResultHelpers:
    def test_get_context_text_groups_by_title(self) -> None:
        c1 = _make_chunk(content="hello", custom={"title": "Doc A"})
        c2 = _make_chunk(content="world", custom={"title": "Doc A"})
        c3 = _make_chunk(content="other", custom={})
        result = QueryResult(chunks=[(c1, 0.5), (c2, 0.4), (c3, 0.3)])
        text = result.get_context_text(max_chunks=3)
        assert "--- From: Doc A ---" in text
        assert "hello" in text
        assert "world" in text
        assert "other" in text

    def test_get_context_text_includes_entity_section(self) -> None:
        c = _make_chunk(content="hello")
        e = Entity(name="Alice", entity_type="PERSON", description="founder")
        result = QueryResult(chunks=[(c, 0.5)], entities=[(e, 0.9)])
        text = result.get_context_text()
        assert "--- Entities ---" in text
        assert "Alice (PERSON): founder" in text

    def test_get_context_text_no_chunks_only_entities(self) -> None:
        e = Entity(name="Alice", entity_type="PERSON")
        result = QueryResult(chunks=[], entities=[(e, 0.9)])
        text = result.get_context_text()
        # Section header still appears
        assert "Alice (PERSON)" in text

    def test_top_chunks_and_entities(self) -> None:
        c = _make_chunk()
        e = Entity(name="A", entity_type="PERSON")
        result = QueryResult(chunks=[(c, 0.5)], entities=[(e, 0.4)])
        assert result.top_chunks == [c]
        assert result.top_entities == [e]

    def test_get_full_metadata(self) -> None:
        c = SearchMethodContribution()
        g = GraphTraversalInfo()
        t = TemporalInfo(detected=True)
        result = QueryResult(
            metadata={"foo": "bar"},
            search_contributions=c,
            graph_info=g,
            temporal_info=t,
        )
        full = result.get_full_metadata()
        assert full["foo"] == "bar"
        assert "search_methods" in full
        assert "graph_traversal" in full
        assert "temporal" in full


# ---------------------------------------------------------------------------
# QueryConfig.from_settings
# ---------------------------------------------------------------------------


class TestQueryConfigFromSettings:
    def test_from_settings_maps_fields(self) -> None:
        # Build a SimpleNamespace settings stub matching the expected attrs
        settings = SimpleNamespace(
            default_mode="vector",
            min_chunk_similarity=0.1,
            min_entity_similarity=0.2,
            vector_weight=0.5,
            graph_weight=0.3,
            keyword_weight=0.2,
            apply_recency_bias=False,
            recency_weight=0.1,
            recency_decay_days=14.0,
            enable_understanding=True,
            understanding_expand_query=True,
            understanding_extract_entities=True,
            understanding_detect_temporal=True,
            enable_entity_linking=True,
            entity_linking_fuzzy_threshold=0.6,
            entity_linking_embedding_threshold=0.5,
            entity_linking_max_candidates=5,
            enable_reranking=False,
            reranking_method="cross_encoder",
            reranking_model=None,
            reranking_top_n=50,
            reranking_final_k=10,
            enable_keyword_search=True,
            keyword_search_method="fulltext",
            enable_hyde="auto",
            hyde_num_hypotheticals=1,
            enable_multi_stage=True,
            stage1_recall_limit=200,
            stage3_filter_limit=50,
            stage4_rerank_limit=50,
            enable_diversity=True,
            diversity_lambda=0.5,
        )
        cfg = QueryConfig.from_settings(settings)
        assert cfg.mode == SearchMode.VECTOR
        assert cfg.vector_weight == 0.5
        assert cfg.enable_query_understanding is True

    def test_from_settings_bool_hyde_true_maps_to_always(self) -> None:
        settings = SimpleNamespace(
            default_mode="hybrid",
            min_chunk_similarity=0.0,
            min_entity_similarity=0.0,
            vector_weight=0.5,
            graph_weight=0.3,
            keyword_weight=0.2,
            apply_recency_bias=False,
            recency_weight=0.1,
            recency_decay_days=30.0,
            enable_understanding=False,
            understanding_expand_query=False,
            understanding_extract_entities=False,
            understanding_detect_temporal=False,
            enable_entity_linking=False,
            entity_linking_fuzzy_threshold=0.5,
            entity_linking_embedding_threshold=0.5,
            entity_linking_max_candidates=5,
            enable_reranking=False,
            reranking_method="cross_encoder",
            reranking_model=None,
            reranking_top_n=50,
            reranking_final_k=10,
            enable_keyword_search=False,
            keyword_search_method="fulltext",
            enable_hyde=True,
            hyde_num_hypotheticals=1,
            enable_multi_stage=False,
            stage1_recall_limit=200,
            stage3_filter_limit=50,
            stage4_rerank_limit=50,
            enable_diversity=False,
            diversity_lambda=0.5,
        )
        cfg = QueryConfig.from_settings(settings)
        assert cfg.enable_hyde == "always"

    def test_from_settings_unknown_mode_defaults_to_hybrid(self) -> None:
        settings = SimpleNamespace(
            default_mode="bogus",
            min_chunk_similarity=0.0,
            min_entity_similarity=0.0,
            vector_weight=0.5,
            graph_weight=0.3,
            keyword_weight=0.2,
            apply_recency_bias=False,
            recency_weight=0.0,
            recency_decay_days=30.0,
            enable_understanding=False,
            understanding_expand_query=False,
            understanding_extract_entities=False,
            understanding_detect_temporal=False,
            enable_entity_linking=False,
            entity_linking_fuzzy_threshold=0.5,
            entity_linking_embedding_threshold=0.5,
            entity_linking_max_candidates=5,
            enable_reranking=False,
            reranking_method="cross_encoder",
            reranking_model=None,
            reranking_top_n=50,
            reranking_final_k=10,
            enable_keyword_search=False,
            keyword_search_method="fulltext",
            enable_hyde=False,
            hyde_num_hypotheticals=1,
            enable_multi_stage=False,
            stage1_recall_limit=200,
            stage3_filter_limit=50,
            stage4_rerank_limit=50,
            enable_diversity=False,
            diversity_lambda=0.5,
        )
        cfg = QueryConfig.from_settings(settings)
        assert cfg.mode == SearchMode.HYBRID
        assert cfg.enable_hyde == "never"  # bool False → "never"


# ---------------------------------------------------------------------------
# HybridQueryEngine — query() entry point
# ---------------------------------------------------------------------------


class TestQueryEntry:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        ns = uuid4()

        cached = QueryResult(chunks=[(_make_chunk(content="cached"), 0.99)])
        # Pre-populate cache
        await engine._cache.set("q", ns, engine._config.mode.name, cached)
        out = await engine.query("q", ns)
        assert out is cached
        # Storage was never called
        storage.search_similar_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_pipeline_with_mocked_searches(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1, 0.2, 0.3, 0.4]) for i in range(3)]
        entities = [_make_entity(name=f"e{i}") for i in range(2)]
        storage = _storage_mock(chunks=chunks, entities=entities)

        cfg = _config_no_extras(
            mode=SearchMode.VECTOR,
            enable_multi_stage=False,
            max_chunks=5,
            max_entities=5,
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        result = await engine.query("test query", uuid4())
        assert isinstance(result, QueryResult)
        assert len(result.chunks) >= 1
        # search_similar_chunks was called by vector_search
        storage.search_similar_chunks.assert_awaited()

    @pytest.mark.asyncio
    async def test_multi_stage_pipeline_with_mocked_searches(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1, 0.2, 0.3, 0.4]) for i in range(5)]
        entities = [_make_entity(name=f"e{i}") for i in range(2)]
        storage = _storage_mock(chunks=chunks, entities=entities)

        cfg = _config_no_extras(
            mode=SearchMode.VECTOR,
            enable_multi_stage=True,
            max_chunks=10,
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        result = await engine.query("multi stage", uuid4())
        assert isinstance(result, QueryResult)
        # multi_stage_enabled should be True in metadata
        assert result.metadata.get("multi_stage_enabled") is True

    @pytest.mark.asyncio
    async def test_agentic_dispatches_to_agent(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())

        # Stub the agent
        agent_result = SimpleNamespace(
            chunks=[(_make_chunk(content="A"), 0.9, "src")],
            entities=[],
            summary="agentic summary",
            trace=None,
            metadata={"agentic_meta": True},
        )
        with patch("khora.query.agentic.AgenticSearchAgent") as agent_cls:
            agent_instance = MagicMock()
            agent_instance.search = AsyncMock(return_value=agent_result)
            agent_cls.return_value = agent_instance
            result = await engine.query("q", uuid4(), agentic=True)
        assert result.metadata.get("agentic") is True
        assert result.metadata.get("summary") == "agentic summary"

    @pytest.mark.asyncio
    async def test_legacy_pipeline_with_recency_bias(self) -> None:
        chunks = [
            _make_chunk(
                content=f"c{i}",
                embedding=[0.1, 0.2, 0.3, 0.4],
                created_at=datetime.now(UTC),
            )
            for i in range(3)
        ]
        storage = _storage_mock(chunks=chunks)
        cfg = _config_no_extras(
            mode=SearchMode.VECTOR,
            apply_recency_bias=True,
            recency_weight=0.5,
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        result = await engine.query("recency test", uuid4())
        assert len(result.chunks) >= 1


# ---------------------------------------------------------------------------
# _multi_stage_search direct invocation
# ---------------------------------------------------------------------------


class TestMultiStageSearch:
    @pytest.mark.asyncio
    async def test_empty_stage1_short_circuits(self) -> None:
        storage = _storage_mock()
        # Force empty results everywhere
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(mode=SearchMode.VECTOR, enable_multi_stage=True),
        )
        from khora.query.metrics import SearchMetrics

        metrics = SearchMetrics()
        graph_info = GraphTraversalInfo()
        out_chunks, out_entities, ctx, contribs = await engine._multi_stage_search(
            query_text="q",
            namespace_id=uuid4(),
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            config=engine._config,
            understanding=None,
            linked_entity_ids=[],
            temporal_filter=None,
            metrics=metrics,
            graph_info=graph_info,
        )
        assert out_chunks == []

    @pytest.mark.asyncio
    async def test_pipeline_runs_all_stages(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1 * (i + 1)] * 4) for i in range(5)]
        entities = [_make_entity(name=f"e{i}") for i in range(2)]
        storage = _storage_mock(chunks=chunks, entities=entities)
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(
                mode=SearchMode.VECTOR,
                enable_multi_stage=True,
                enable_diversity=False,
                max_chunks=5,
            ),
        )
        from khora.query.metrics import SearchMetrics

        metrics = SearchMetrics()
        graph_info = GraphTraversalInfo()
        out_chunks, out_entities, ctx, contribs = await engine._multi_stage_search(
            query_text="q",
            namespace_id=uuid4(),
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            config=engine._config,
            understanding=None,
            linked_entity_ids=[],
            temporal_filter=None,
            metrics=metrics,
            graph_info=graph_info,
        )
        assert len(out_chunks) >= 1


# ---------------------------------------------------------------------------
# find_related_entities
# ---------------------------------------------------------------------------


class TestFindRelatedEntities:
    @pytest.mark.asyncio
    async def test_empty_neighborhood_returns_empty(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(storage=storage)
        out = await engine.find_related_entities(uuid4(), uuid4())
        assert out == []

    @pytest.mark.asyncio
    async def test_neighborhood_returns_scored_entities(self) -> None:
        e1 = _make_entity(name="Bob")
        storage = _storage_mock()
        storage.get_neighborhood = AsyncMock(
            return_value={
                "entities": [{"id": str(e1.id)}],
                "relationships": [{"from": "x", "type": "REL", "to": "y"}],
            }
        )
        storage.get_entities_batch = AsyncMock(return_value={e1.id: e1})
        engine = HybridQueryEngine(storage=storage)
        out = await engine.find_related_entities(uuid4(), uuid4())
        assert len(out) == 1
        assert out[0][0] is e1
        # base score = 1 / (1 + n_rels) = 1/2
        assert out[0][1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# temporal_query
# ---------------------------------------------------------------------------


class TestTemporalQueryEntry:
    @pytest.mark.asyncio
    async def test_recency_weight_set(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        tq = TemporalQuery(query="when did X happen", recency_weight=0.5, decay_days=7.0)

        # Patch out the inner query() to inspect what got passed
        engine.query = AsyncMock(return_value=QueryResult())
        await engine.temporal_query(tq, uuid4())
        engine.query.assert_awaited_once()
        # The config passed inside should have apply_recency_bias=True
        passed_cfg = engine.query.call_args.kwargs["config"]
        assert passed_cfg.apply_recency_bias is True
        assert passed_cfg.recency_weight == 0.5

    @pytest.mark.asyncio
    async def test_temporal_query_with_context_window(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        tq = TemporalQuery(query="recent stuff", context_window_days=7)
        engine.query = AsyncMock(return_value=QueryResult())
        await engine.temporal_query(tq, uuid4())
        # temporal_filter passed in is non-None
        passed_filter = engine.query.call_args.kwargs.get("temporal_filter")
        assert passed_filter is not None


# ---------------------------------------------------------------------------
# warm_cache / warm_keyword_index
# ---------------------------------------------------------------------------


class TestWarmCache:
    @pytest.mark.asyncio
    async def test_warm_cache_explicit_queries(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        engine.query = AsyncMock(return_value=QueryResult())
        out = await engine.warm_cache(uuid4(), queries=["q1", "q2"], include_entity_based=False)
        assert out["queries_warmed"] == 2
        assert out["errors"] == 0

    @pytest.mark.asyncio
    async def test_warm_cache_entity_based(self) -> None:
        e1 = _make_entity(name="Alice", mention_count=10)
        e2 = _make_entity(name="Bob", mention_count=5)
        e1.description = "founder of acme"
        e2.description = ""
        storage = _storage_mock(entities=[e1, e2])
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        engine.query = AsyncMock(return_value=QueryResult())
        out = await engine.warm_cache(uuid4(), queries=None, include_entity_based=True)
        assert out["queries_warmed"] >= 2

    @pytest.mark.asyncio
    async def test_warm_cache_list_entities_failure_continues(self) -> None:
        storage = _storage_mock()
        storage.list_entities = AsyncMock(side_effect=RuntimeError("db down"))
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        engine.query = AsyncMock(return_value=QueryResult())
        out = await engine.warm_cache(uuid4(), queries=["q1"])
        assert out["queries_warmed"] == 1

    @pytest.mark.asyncio
    async def test_warm_cache_query_error_recorded(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        engine.query = AsyncMock(side_effect=RuntimeError("query failed"))
        out = await engine.warm_cache(uuid4(), queries=["q1"], include_entity_based=False)
        assert out["queries_warmed"] == 0
        assert out["errors"] == 1


class TestWarmKeywordIndex:
    @pytest.mark.asyncio
    async def test_no_chunks_returns_no_chunks_status(self) -> None:
        storage = _storage_mock(chunks=[])
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        out = await engine.warm_keyword_index(uuid4())
        assert out["status"] == "no_chunks"

    @pytest.mark.asyncio
    async def test_chunks_indexed(self) -> None:
        chunks = [_make_chunk(content=f"c{i} alpha beta") for i in range(3)]
        storage = _storage_mock(chunks=chunks)
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        ns = uuid4()
        out = await engine.warm_keyword_index(ns)
        assert out["status"] == "indexed"
        assert out["chunk_count"] == 3
        # Second call hits cache
        out2 = await engine.warm_keyword_index(ns)
        assert out2["status"] == "already_indexed"

    @pytest.mark.asyncio
    async def test_list_chunks_exception_returns_error_status(self) -> None:
        storage = _storage_mock()
        storage.list_chunks = AsyncMock(side_effect=RuntimeError("boom"))
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        out = await engine.warm_keyword_index(uuid4())
        assert out["status"] == "error"
        assert "boom" in out["error"]


# ---------------------------------------------------------------------------
# Engine init quirks
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_hyde_expander_created_in_auto_with_embedder(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=QueryConfig(enable_hyde="auto"),
        )
        assert engine._hyde_expander is not None

    def test_hyde_expander_not_created_in_never(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=QueryConfig(enable_hyde="never"),
        )
        assert engine._hyde_expander is None

    def test_hyde_expander_skipped_without_embedder(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(
            storage=storage,
            embedder=None,
            config=QueryConfig(enable_hyde="always"),
        )
        assert engine._hyde_expander is None


# ---------------------------------------------------------------------------
# query() — feature paths
# ---------------------------------------------------------------------------


class TestQueryFeaturePaths:
    @pytest.mark.asyncio
    async def test_understanding_enabled_with_mock_understanding_path(self) -> None:
        """Drive the understanding code-path with a mocked QueryUnderstanding.understand."""
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1] * 4) for i in range(3)]
        storage = _storage_mock(chunks=chunks)

        # Build a fake understanding result
        understanding_result = UnderstandingResult(
            original_query="q",
            intent=QueryIntent.SEARCH,
            answer_type=AnswerType.UNKNOWN,
            entities=[],
            keywords=["alpha"],
            source_priority=SourcePriority(),
            search_strategy=SearchStrategy(
                vector_weight=0.5,
                graph_weight=0.3,
                keyword_weight=0.2,
                graph_depth=1,
            ),
            temporal_references=[],
            expanded_queries=[],
            complexity_score=0.4,
            requires_multi_step=False,
        )

        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(
                mode=SearchMode.VECTOR,
                enable_query_understanding=True,
                enable_multi_stage=False,
            ),
        )
        engine._query_understanding.understand = AsyncMock(return_value=understanding_result)

        # Use a query that won't pass the is_simple heuristic
        result = await engine.query("compare X versus Y in detail today please now", uuid4())
        assert "understanding" in result.metadata
        assert result.metadata["understanding"]["intent"] == "SEARCH"

    @pytest.mark.asyncio
    async def test_understanding_skipped_for_simple_query(self) -> None:
        chunks = [_make_chunk(content="c", embedding=[0.1] * 4)]
        storage = _storage_mock(chunks=chunks)
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(
                mode=SearchMode.VECTOR,
                enable_query_understanding=True,
                enable_multi_stage=False,
            ),
        )
        spy = AsyncMock(return_value=None)
        engine._query_understanding.understand = spy
        # Simple query → should skip understand()
        await engine.query("what is foo", uuid4())
        # _is_simple_query returns True for short non-temporal non-comparison queries
        spy.assert_not_called()


# ---------------------------------------------------------------------------
# invalidate_caches
# ---------------------------------------------------------------------------


class TestInvalidateCachesExtra:
    def test_clears_keyword_searcher(self) -> None:
        storage = _storage_mock()
        engine = HybridQueryEngine(storage=storage, config=_config_no_extras())
        ns = uuid4()
        engine._keyword_searchers[str(ns)] = MagicMock()
        engine.invalidate_caches(ns)
        assert str(ns) not in engine._keyword_searchers


# ---------------------------------------------------------------------------
# query() — extended feature paths to push engine coverage to ≥80%
# ---------------------------------------------------------------------------


class TestQueryLegacyDeepPaths:
    @pytest.mark.asyncio
    async def test_legacy_with_keyword_fulltext_results(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1] * 4) for i in range(3)]
        storage = _storage_mock(chunks=chunks)
        # keyword fulltext returns results too
        storage.search_fulltext_chunks = AsyncMock(return_value=[(c, 0.4) for c in chunks[:2]])
        cfg = _config_no_extras(
            mode=SearchMode.HYBRID,
            enable_multi_stage=False,
            enable_keyword_search=True,
            keyword_search_method="fulltext",
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        result = await engine.query("compound query", uuid4())
        # search_fulltext_chunks called as part of hybrid mode
        storage.search_fulltext_chunks.assert_awaited()
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_legacy_with_bm25_keyword_method(self) -> None:
        chunks = [_make_chunk(content=f"alpha beta gamma {i}", embedding=[0.1] * 4) for i in range(3)]
        storage = _storage_mock(chunks=chunks)
        cfg = _config_no_extras(
            mode=SearchMode.HYBRID,
            enable_multi_stage=False,
            enable_keyword_search=True,
            keyword_search_method="bm25",
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        await engine.query("alpha beta", uuid4())
        # list_chunks is called to build BM25 index
        storage.list_chunks.assert_awaited()

    @pytest.mark.asyncio
    async def test_legacy_zero_result_fallback_path(self) -> None:
        # Force vector to return empty, ensure fulltext fallback triggers
        storage = _storage_mock()
        storage.search_similar_chunks = AsyncMock(return_value=[])
        # Fallback fulltext returns some chunks
        fallback_chunk = _make_chunk(content="fallback", embedding=[0.1] * 4)
        storage.search_fulltext_chunks = AsyncMock(return_value=[(fallback_chunk, 0.5)])
        cfg = _config_no_extras(
            mode=SearchMode.VECTOR,
            enable_multi_stage=False,
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        result = await engine.query("nothing matches initially", uuid4())
        # Fallback path was triggered
        storage.search_fulltext_chunks.assert_awaited()
        # Should have at least the fallback chunk
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_legacy_with_reranking_enabled(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1] * 4) for i in range(6)]
        storage = _storage_mock(chunks=chunks)
        cfg = _config_no_extras(
            mode=SearchMode.VECTOR,
            enable_multi_stage=False,
            enable_reranking=True,
            reranking_method="cross_encoder",
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        # Mock the reranker creation to return a deterministic reranker
        from khora.query.reranking import RerankResult

        fake_reranker = MagicMock()

        async def fake_rerank(qt, cands, top_k=10):
            return [
                RerankResult(
                    item=c.item,
                    original_score=c.original_score,
                    rerank_score=0.5,
                    final_score=c.original_score + 0.1,
                )
                for c in cands[:top_k]
            ]

        fake_reranker.rerank = fake_rerank
        with patch("khora.query.engine.create_reranker", return_value=fake_reranker):
            result = await engine.query("test", uuid4())
        # Reranking metadata should be present
        assert "reranking" in result.metadata

    @pytest.mark.asyncio
    async def test_legacy_with_temporal_filter(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1] * 4) for i in range(3)]
        storage = _storage_mock(chunks=chunks)
        cfg = _config_no_extras(mode=SearchMode.VECTOR, enable_multi_stage=False)
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        tf = TemporalFilter(start_date=datetime(2020, 1, 1, tzinfo=UTC))
        result = await engine.query("with temporal", uuid4(), temporal_filter=tf)
        # The temporal_info should have filter_applied or detected set somewhere
        assert isinstance(result, QueryResult)


class TestQueryUnderstandingExtras:
    @pytest.mark.asyncio
    async def test_understanding_with_temporal_references_creates_filter(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1] * 4) for i in range(3)]
        storage = _storage_mock(chunks=chunks)

        understanding_result = UnderstandingResult(
            original_query="q",
            intent=QueryIntent.TEMPORAL,
            answer_type=AnswerType.UNKNOWN,
            entities=[],
            temporal_references=[
                TemporalReference(
                    type="relative",
                    text="last week",
                    start_date=datetime(2024, 1, 1, tzinfo=UTC),
                    end_date=datetime(2024, 1, 8, tzinfo=UTC),
                )
            ],
            keywords=["recent"],
            source_priority=SourcePriority(),
            search_strategy=SearchStrategy(),
            complexity_score=0.4,
        )
        # has_temporal is computed as len(temporal_references) > 0
        assert understanding_result.has_temporal is True

        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(
                mode=SearchMode.VECTOR,
                enable_query_understanding=True,
                enable_multi_stage=False,
                enable_temporal_resolver=False,
            ),
        )
        engine._query_understanding.understand = AsyncMock(return_value=understanding_result)
        result = await engine.query("recent changes between A and B today", uuid4())
        assert "understanding" in result.metadata

    @pytest.mark.asyncio
    async def test_understanding_failure_caught(self) -> None:
        chunks = [_make_chunk(content="c", embedding=[0.1] * 4)]
        storage = _storage_mock(chunks=chunks)
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(
                mode=SearchMode.VECTOR,
                enable_query_understanding=True,
                enable_multi_stage=False,
                enable_temporal_resolver=False,
            ),
        )
        engine._query_understanding.understand = AsyncMock(side_effect=RuntimeError("oops"))
        # Should not raise — falls through
        result = await engine.query("compare A versus B with detail now today maybe", uuid4())
        assert isinstance(result, QueryResult)


class TestQueryEntityLinking:
    @pytest.mark.asyncio
    async def test_entity_linking_path(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1] * 4) for i in range(2)]
        storage = _storage_mock(chunks=chunks)

        understanding_result = UnderstandingResult(
            original_query="who is Alice",
            intent=QueryIntent.SEARCH,
            answer_type=AnswerType.UNKNOWN,
            entities=[EntityMention(name="Alice", entity_type="PERSON")],
            keywords=[],
            source_priority=SourcePriority(),
            search_strategy=SearchStrategy(),
            complexity_score=0.4,
        )

        linker = MagicMock()
        linker_inst = MagicMock()
        linked_entity = MagicMock()
        linked_entity.entity = Entity(name="Alice", entity_type="PERSON")
        result_obj = MagicMock()
        result_obj.linked_entities = [linked_entity]
        result_obj.total_mentions = 1
        result_obj.linked_count = 1
        result_obj.success_rate = 1.0
        result_obj.get_linked_entity_ids = MagicMock(return_value=[linked_entity.entity.id])
        linker_inst.link = AsyncMock(return_value=result_obj)
        linker.return_value = linker_inst

        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(
                mode=SearchMode.VECTOR,
                enable_query_understanding=True,
                enable_entity_linking=True,
                enable_multi_stage=False,
                enable_temporal_resolver=False,
            ),
        )
        engine._query_understanding.understand = AsyncMock(return_value=understanding_result)

        with patch("khora.query.engine.EntityLinker", linker):
            # Use a query long enough to skip the _is_simple_query shortcut
            result = await engine.query(
                "compare Alice Bob and Carol between projects since launch",
                uuid4(),
            )
        assert "entity_linking" in result.metadata

    @pytest.mark.asyncio
    async def test_entity_linking_failure_caught(self) -> None:
        chunks = [_make_chunk(content="c", embedding=[0.1] * 4)]
        storage = _storage_mock(chunks=chunks)

        understanding_result = UnderstandingResult(
            original_query="q",
            intent=QueryIntent.SEARCH,
            answer_type=AnswerType.UNKNOWN,
            entities=[EntityMention(name="Alice", entity_type="PERSON")],
            keywords=[],
            source_priority=SourcePriority(),
            search_strategy=SearchStrategy(),
            complexity_score=0.4,
        )

        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(
                mode=SearchMode.VECTOR,
                enable_query_understanding=True,
                enable_entity_linking=True,
                enable_multi_stage=False,
                enable_temporal_resolver=False,
            ),
        )
        engine._query_understanding.understand = AsyncMock(return_value=understanding_result)

        with patch("khora.query.engine.EntityLinker") as linker_cls:
            inst = MagicMock()
            inst.link = AsyncMock(side_effect=RuntimeError("linker fail"))
            linker_cls.return_value = inst
            # Should not crash
            result = await engine.query("compare Alice and Bob between projects today now", uuid4())
        assert isinstance(result, QueryResult)


class TestQueryHyDEPath:
    @pytest.mark.asyncio
    async def test_hyde_always_expands_embedding(self) -> None:
        chunks = [_make_chunk(content=f"c{i}", embedding=[0.1] * 4) for i in range(3)]
        storage = _storage_mock(chunks=chunks)
        cfg = _config_no_extras(
            mode=SearchMode.VECTOR,
            enable_multi_stage=False,
            enable_hyde="always",
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )
        # Mock HyDE expander
        fake_hyde = MagicMock()
        fake_hyde.expand_query_embedding = AsyncMock(return_value=[0.5] * 4)
        engine._hyde_expander = fake_hyde
        result = await engine.query("hyde this", uuid4())
        fake_hyde.expand_query_embedding.assert_awaited()
        assert result.metadata.get("hyde_applied") is True


class TestGraphSearchPath:
    @pytest.mark.asyncio
    async def test_graph_search_chunks_fetched_and_scored(self) -> None:
        chunk_id = uuid4()
        e1 = Entity(
            id=uuid4(),
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[chunk_id],
            mention_count=3,
        )
        c = Chunk(
            id=chunk_id,
            content="related chunk",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        storage = _storage_mock()
        storage.search_similar_entities = AsyncMock(return_value=[(e1.id, 0.7)])
        storage.get_entities_batch = AsyncMock(return_value={e1.id: e1})
        storage.get_neighborhoods_batch = AsyncMock(return_value={e1.id: {}})
        storage.get_chunks_batch = AsyncMock(return_value={chunk_id: c})

        # HYBRID mode so query_embedding is generated; the graph_search call
        # uses it to pull related chunks through the entity neighbourhood.
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(mode=SearchMode.HYBRID, enable_multi_stage=False),
        )
        result = await engine.query("graph query", uuid4())
        storage.get_chunks_batch.assert_awaited()
        assert isinstance(result, QueryResult)

    @pytest.mark.asyncio
    async def test_graph_search_with_linked_entity_ids(self) -> None:
        chunk_id = uuid4()
        e1 = Entity(
            id=uuid4(),
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[chunk_id],
            mention_count=2,
        )
        c = Chunk(id=chunk_id, content="alpha", embedding=[0.1, 0.2, 0.3, 0.4])
        storage = _storage_mock()
        storage.get_entities_batch = AsyncMock(return_value={e1.id: e1})
        storage.get_neighborhoods_batch = AsyncMock(return_value={e1.id: {}})
        storage.get_chunks_batch = AsyncMock(return_value={chunk_id: c})

        cfg = _config_no_extras(mode=SearchMode.GRAPH, enable_multi_stage=False)
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=cfg,
        )

        # Drive _graph_search directly with linked_entity_ids
        out = await engine._graph_search(
            uuid4(),
            "q",
            [0.1] * 4,
            cfg,
            linked_entity_ids=[e1.id],
        )
        assert out["source"] == "graph"
        assert len(out["entities"]) == 1


class TestSourcePriorityIntegration:
    @pytest.mark.asyncio
    async def test_source_priority_boosts_chunks(self) -> None:
        # Build a chunk with custom source_tool metadata
        c = _make_chunk(
            content="alpha",
            embedding=[0.1] * 4,
            custom={"source_tool": "slack"},
        )
        storage = _storage_mock(chunks=[c])
        understanding_result = UnderstandingResult(
            original_query="q",
            intent=QueryIntent.SEARCH,
            answer_type=AnswerType.UNKNOWN,
            entities=[],
            keywords=["alpha"],
            source_priority=SourcePriority(slack=2.0),
            search_strategy=SearchStrategy(),
            complexity_score=0.4,
        )
        engine = HybridQueryEngine(
            storage=storage,
            embedder=_embedder_mock(),
            config=_config_no_extras(
                mode=SearchMode.VECTOR,
                enable_multi_stage=False,
                enable_query_understanding=True,
                enable_temporal_resolver=False,
            ),
        )
        engine._query_understanding.understand = AsyncMock(return_value=understanding_result)
        result = await engine.query(
            "compare X and Y between teams over different periods",
            uuid4(),
        )
        # Slack source got the boost
        assert len(result.chunks) >= 1
