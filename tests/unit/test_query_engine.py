"""Unit tests for query/engine.py — HybridQueryEngine."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

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

# ---------------------------------------------------------------------------
# SearchMode enum
# ---------------------------------------------------------------------------


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_modes_exist(self) -> None:
        """All expected modes exist."""
        assert SearchMode.VECTOR is not None
        assert SearchMode.GRAPH is not None
        assert SearchMode.HYBRID is not None
        assert SearchMode.ALL is not None

    def test_modes_are_distinct(self) -> None:
        """Each mode has a unique value."""
        values = {SearchMode.VECTOR, SearchMode.GRAPH, SearchMode.HYBRID, SearchMode.ALL}
        assert len(values) == 4


# ---------------------------------------------------------------------------
# QueryConfig
# ---------------------------------------------------------------------------


class TestQueryConfig:
    """Tests for QueryConfig dataclass."""

    def test_defaults(self) -> None:
        """Default config values."""
        config = QueryConfig()
        assert config.mode == SearchMode.HYBRID
        assert config.max_chunks == 10
        assert config.max_entities == 10
        assert config.vector_weight == 0.5
        assert config.graph_weight == 0.3
        assert config.keyword_weight == 0.2
        assert config.enable_query_understanding is True
        assert config.enable_entity_linking is True

    def test_linked_entity_boost_default_and_custom(self) -> None:
        """linked_entity_boost defaults to 1.5 and is configurable."""
        assert QueryConfig().linked_entity_boost == 1.5
        assert QueryConfig(linked_entity_boost=3.0).linked_entity_boost == 3.0

    def test_custom_config(self) -> None:
        """Custom config values."""
        config = QueryConfig(
            mode=SearchMode.VECTOR,
            max_chunks=5,
            min_chunk_similarity=0.1,
            vector_weight=1.0,
        )
        assert config.mode == SearchMode.VECTOR
        assert config.max_chunks == 5
        assert config.min_chunk_similarity == 0.1

    def test_from_settings(self) -> None:
        """from_settings creates config from QuerySettings."""
        settings = MagicMock()
        settings.default_mode = "all"
        settings.min_chunk_similarity = 0.1
        settings.min_entity_similarity = 0.2
        settings.vector_weight = 0.6
        settings.graph_weight = 0.2
        settings.keyword_weight = 0.2
        settings.apply_recency_bias = True
        settings.recency_weight = 0.3
        settings.recency_decay_days = 14.0
        settings.enable_understanding = False
        settings.understanding_expand_query = False
        settings.understanding_extract_entities = True
        settings.understanding_detect_temporal = True
        settings.enable_entity_linking = True
        settings.entity_linking_fuzzy_threshold = 0.5
        settings.entity_linking_embedding_threshold = 0.3
        settings.entity_linking_max_candidates = 3
        settings.enable_reranking = False
        settings.reranking_method = "llm"
        settings.reranking_top_n = 30
        settings.reranking_final_k = 5
        settings.enable_keyword_search = True
        settings.keyword_search_method = "bm25"
        settings.enable_hyde = True
        settings.hyde_num_hypotheticals = 2
        # Multi-stage settings
        settings.enable_multi_stage = True
        settings.stage1_recall_limit = 150
        settings.stage3_filter_limit = 40
        settings.stage4_rerank_limit = 40
        settings.enable_diversity = True
        settings.diversity_lambda = 0.7
        settings.linked_entity_boost = 2.5

        config = QueryConfig.from_settings(settings)
        assert config.mode == SearchMode.ALL
        assert config.linked_entity_boost == 2.5
        assert config.vector_weight == 0.6
        assert config.apply_recency_bias is True
        assert config.enable_query_understanding is False
        assert config.reranking_method == "llm"
        assert config.enable_hyde == "always"
        # Multi-stage assertions
        assert config.enable_multi_stage is True
        assert config.stage1_recall_limit == 150
        assert config.stage3_filter_limit == 40
        assert config.stage4_rerank_limit == 40
        assert config.enable_diversity is True
        assert config.diversity_lambda == 0.7

    def test_from_settings_unknown_mode_defaults_to_hybrid(self) -> None:
        """Unknown mode string defaults to HYBRID."""
        settings = MagicMock()
        settings.default_mode = "nonexistent"
        settings.min_chunk_similarity = 0.05
        settings.min_entity_similarity = 0.05
        settings.vector_weight = 0.5
        settings.graph_weight = 0.3
        settings.keyword_weight = 0.2
        settings.apply_recency_bias = False
        settings.recency_weight = 0.2
        settings.recency_decay_days = 30.0
        settings.enable_understanding = True
        settings.understanding_expand_query = True
        settings.understanding_extract_entities = True
        settings.understanding_detect_temporal = True
        settings.enable_entity_linking = True
        settings.entity_linking_fuzzy_threshold = 0.6
        settings.entity_linking_embedding_threshold = 0.4
        settings.entity_linking_max_candidates = 5
        settings.enable_reranking = True
        settings.reranking_method = "cross_encoder"
        settings.reranking_top_n = 50
        settings.reranking_final_k = 10
        settings.enable_keyword_search = True
        settings.keyword_search_method = "fulltext"
        settings.enable_hyde = False
        settings.hyde_num_hypotheticals = 1
        # Multi-stage settings (defaults)
        settings.enable_multi_stage = True
        settings.stage1_recall_limit = 200
        settings.stage3_filter_limit = 50
        settings.stage4_rerank_limit = 50
        settings.enable_diversity = False
        settings.diversity_lambda = 0.5

        config = QueryConfig.from_settings(settings)
        assert config.mode == SearchMode.HYBRID


class TestMultiStageConfig:
    """Tests for multi-stage ranking pipeline configuration."""

    def test_defaults(self) -> None:
        """Test default multi-stage configuration values."""
        config = QueryConfig()
        assert config.enable_multi_stage is True
        assert config.stage1_recall_limit == 200
        assert config.stage3_filter_limit == 50
        assert config.stage4_rerank_limit == 50
        assert config.enable_diversity is True
        assert config.diversity_lambda == 0.5

    def test_custom_config(self) -> None:
        """Test custom multi-stage configuration."""
        config = QueryConfig(
            enable_multi_stage=False,
            stage1_recall_limit=100,
            stage3_filter_limit=30,
            stage4_rerank_limit=25,
            enable_diversity=True,
            diversity_lambda=0.8,
        )
        assert config.enable_multi_stage is False
        assert config.stage1_recall_limit == 100
        assert config.stage3_filter_limit == 30
        assert config.stage4_rerank_limit == 25
        assert config.enable_diversity is True
        assert config.diversity_lambda == 0.8


# ---------------------------------------------------------------------------
# SearchMethodStats
# ---------------------------------------------------------------------------


class TestSearchMethodStats:
    """Tests for SearchMethodStats."""

    def test_to_dict(self) -> None:
        """to_dict includes all fields."""
        stats = SearchMethodStats(
            chunk_count=5,
            entity_count=3,
            min_score=0.1,
            max_score=0.9,
            avg_score=0.5,
            latency_ms=100.0,
        )
        d = stats.to_dict()
        assert d["chunks"]["count"] == 5
        assert d["entities"]["count"] == 3
        assert d["scores"]["min"] == 0.1
        assert d["scores"]["max"] == 0.9
        assert d["latency_ms"] == 100.0

    def test_defaults(self) -> None:
        """Default values are zero."""
        stats = SearchMethodStats()
        assert stats.chunk_count == 0
        assert stats.entity_count == 0
        assert stats.latency_ms == 0.0


# ---------------------------------------------------------------------------
# SearchMethodContribution
# ---------------------------------------------------------------------------


class TestSearchMethodContribution:
    """Tests for SearchMethodContribution."""

    def test_compute_overlaps(self) -> None:
        """compute_overlaps correctly computes set operations."""
        contrib = SearchMethodContribution()
        contrib.vector.chunk_ids = ["a", "b", "c"]
        contrib.graph.chunk_ids = ["b", "c", "d"]
        contrib.keyword.chunk_ids = ["c", "e"]
        contrib.compute_overlaps()

        assert "a" in contrib.vector_only_chunks
        assert "d" in contrib.graph_only_chunks
        assert "e" in contrib.keyword_only_chunks
        assert "c" in contrib.all_methods_overlap
        assert "b" in contrib.vector_graph_overlap

    def test_compute_overlaps_empty(self) -> None:
        """compute_overlaps handles empty inputs."""
        contrib = SearchMethodContribution()
        contrib.compute_overlaps()
        assert contrib.all_methods_overlap == []
        assert contrib.vector_only_chunks == []

    def test_compute_entity_overlaps(self) -> None:
        """compute_overlaps computes entity set operations."""
        contrib = SearchMethodContribution()
        contrib.vector.entity_ids = ["e1", "e2"]
        contrib.graph.entity_ids = ["e2", "e3"]
        contrib.compute_overlaps()

        assert "e1" in contrib.vector_only_entities
        assert "e3" in contrib.graph_only_entities
        assert "e2" in contrib.vector_graph_entity_overlap

    def test_to_dict(self) -> None:
        """to_dict returns comprehensive statistics."""
        contrib = SearchMethodContribution()
        d = contrib.to_dict()
        assert "summary" in d
        assert "by_method" in d
        assert "chunk_overlap" in d
        assert "entity_overlap" in d

    def test_legacy_properties(self) -> None:
        """Legacy properties return chunk_ids."""
        contrib = SearchMethodContribution()
        contrib.vector.chunk_ids = ["a"]
        contrib.graph.chunk_ids = ["b"]
        contrib.keyword.chunk_ids = ["c"]
        assert contrib.vector_chunks == ["a"]
        assert contrib.graph_chunks == ["b"]
        assert contrib.keyword_chunks == ["c"]


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_top_chunks(self) -> None:
        """top_chunks strips scores."""
        chunk1 = MagicMock()
        chunk2 = MagicMock()
        result = QueryResult(chunks=[(chunk1, 0.9), (chunk2, 0.5)])
        assert result.top_chunks == [chunk1, chunk2]

    def test_top_entities(self) -> None:
        """top_entities strips scores."""
        entity1 = MagicMock()
        result = QueryResult(entities=[(entity1, 0.8)])
        assert result.top_entities == [entity1]

    def test_get_full_metadata(self) -> None:
        """get_full_metadata includes search contributions."""
        result = QueryResult(
            metadata={"query_id": "test"},
            search_contributions=SearchMethodContribution(),
        )
        meta = result.get_full_metadata()
        assert "query_id" in meta
        assert "search_methods" in meta

    def test_get_full_metadata_with_graph_and_temporal(self) -> None:
        """get_full_metadata includes graph and temporal info."""
        result = QueryResult(
            metadata={},
            graph_info=GraphTraversalInfo(entities_searched=["Alice"]),
            temporal_info=TemporalInfo(detected=True),
        )
        meta = result.get_full_metadata()
        assert "graph_traversal" in meta
        assert "temporal" in meta

    def test_get_full_metadata_without_extras(self) -> None:
        """get_full_metadata works without optional fields."""
        result = QueryResult(metadata={"foo": "bar"})
        meta = result.get_full_metadata()
        assert meta == {"foo": "bar"}


# ---------------------------------------------------------------------------
# GraphTraversalInfo
# ---------------------------------------------------------------------------


class TestGraphTraversalInfo:
    """Tests for GraphTraversalInfo."""

    def test_to_dict(self) -> None:
        """to_dict includes all fields."""
        info = GraphTraversalInfo(
            entities_searched=["Alice"],
            entities_linked=["Alice"],
            relationships_traversed=[("Alice", "WORKS_FOR", "Acme")],
            neighborhood_depth=2,
        )
        d = info.to_dict()
        assert d["entities_searched"] == ["Alice"]
        assert d["neighborhood_depth"] == 2
        assert len(d["relationships_traversed"]) == 1
        assert d["relationships_traversed"][0]["from"] == "Alice"

    def test_to_dict_truncates(self) -> None:
        """to_dict truncates long lists."""
        info = GraphTraversalInfo(
            entities_searched=[f"entity_{i}" for i in range(30)],
        )
        d = info.to_dict()
        assert len(d["entities_searched"]) == 20


# ---------------------------------------------------------------------------
# TemporalInfo
# ---------------------------------------------------------------------------


class TestTemporalInfo:
    """Tests for TemporalInfo in query engine."""

    def test_to_dict(self) -> None:
        """to_dict handles None times."""
        info = TemporalInfo(detected=True, filter_applied=False)
        d = info.to_dict()
        assert d["detected"] is True
        assert d["time_start"] is None

    def test_to_dict_with_times(self) -> None:
        """to_dict formats datetime values."""
        from datetime import datetime

        now = datetime.now(UTC)
        info = TemporalInfo(detected=True, filter_applied=True, time_start=now, time_end=now)
        d = info.to_dict()
        assert d["time_start"] == now.isoformat()
        assert d["time_end"] == now.isoformat()


# ---------------------------------------------------------------------------
# HybridQueryEngine._is_simple_query
# ---------------------------------------------------------------------------


class TestIsSimpleQuery:
    """Tests for _is_simple_query static method."""

    def test_short_query_is_simple(self) -> None:
        """Short query without special patterns is simple."""
        assert HybridQueryEngine._is_simple_query("hello world") is True

    def test_long_query_is_not_simple(self) -> None:
        """Query with more than 8 words is not simple."""
        assert HybridQueryEngine._is_simple_query("this is a very long query with many many words") is False

    def test_temporal_reference_not_simple(self) -> None:
        """Query with temporal references is not simple."""
        assert HybridQueryEngine._is_simple_query("events yesterday") is False
        assert HybridQueryEngine._is_simple_query("last week meetings") is False
        assert HybridQueryEngine._is_simple_query("changes since Monday") is False

    def test_quoted_phrase_not_simple(self) -> None:
        """Query with quotes is not simple."""
        assert HybridQueryEngine._is_simple_query('"exact phrase"') is False
        assert HybridQueryEngine._is_simple_query("'entity name'") is False

    def test_comparison_not_simple(self) -> None:
        """Query with comparison words is not simple."""
        assert HybridQueryEngine._is_simple_query("compare A vs B") is False
        assert HybridQueryEngine._is_simple_query("difference between X Y") is False

    def test_year_reference_not_simple(self) -> None:
        """Query with year is not simple."""
        assert HybridQueryEngine._is_simple_query("revenue in 2024") is False

    def test_month_reference_not_simple(self) -> None:
        """Query with month name is not simple."""
        assert HybridQueryEngine._is_simple_query("meetings in January") is False


# ---------------------------------------------------------------------------
# HybridQueryEngine._attribute_relevance_boost
# ---------------------------------------------------------------------------


class TestAttributeRelevanceBoost:
    """Tests for _attribute_relevance_boost static method."""

    def test_no_attributes(self) -> None:
        """Entity without attributes gets no boost."""
        entity = MagicMock(spec=[])  # No attributes attr
        assert HybridQueryEngine._attribute_relevance_boost(entity, ["urgent"]) == 0.0

    def test_matching_attribute(self) -> None:
        """Entity with matching attribute gets boost."""
        entity = MagicMock()
        entity.attributes = {"priority": "urgent", "status": "open"}
        boost = HybridQueryEngine._attribute_relevance_boost(entity, ["urgent"])
        assert boost == pytest.approx(0.1)

    def test_multiple_matches(self) -> None:
        """Multiple matches accumulate."""
        entity = MagicMock()
        entity.attributes = {"priority": "urgent", "assignee": "alice"}
        boost = HybridQueryEngine._attribute_relevance_boost(entity, ["urgent", "alice"])
        assert boost == pytest.approx(0.2)

    def test_boost_capped_at_03(self) -> None:
        """Boost is capped at 0.3."""
        entity = MagicMock()
        entity.attributes = {f"attr_{i}": f"keyword{i}" for i in range(10)}
        boost = HybridQueryEngine._attribute_relevance_boost(entity, [f"keyword{i}" for i in range(10)])
        assert boost == pytest.approx(0.3)

    def test_case_insensitive(self) -> None:
        """Matching is case-insensitive."""
        entity = MagicMock()
        entity.attributes = {"name": "Alice Smith"}
        boost = HybridQueryEngine._attribute_relevance_boost(entity, ["alice"])
        assert boost > 0.0

    def test_empty_attributes_dict(self) -> None:
        """Empty attributes dict returns no boost."""
        entity = MagicMock()
        entity.attributes = {}
        assert HybridQueryEngine._attribute_relevance_boost(entity, ["test"]) == 0.0

    def test_non_dict_attributes(self) -> None:
        """Non-dict attributes returns no boost."""
        entity = MagicMock()
        entity.attributes = "not a dict"
        assert HybridQueryEngine._attribute_relevance_boost(entity, ["test"]) == 0.0


# ---------------------------------------------------------------------------
# HybridQueryEngine.query (integration with mocks)
# ---------------------------------------------------------------------------


class TestHybridQueryEngineQuery:
    """Tests for HybridQueryEngine.query method."""

    def _make_engine(self) -> HybridQueryEngine:
        """Create engine with mocked dependencies."""
        storage = MagicMock()
        storage.search_similar_chunks = AsyncMock(return_value=[])
        storage.search_similar_entities = AsyncMock(return_value=[])
        storage.get_entities_batch = AsyncMock(return_value={})
        storage.get_neighborhood = AsyncMock(return_value={})
        storage.get_neighborhoods_batch = AsyncMock(return_value={})
        storage.list_chunks = AsyncMock(return_value=[])
        storage.search_fulltext_chunks = AsyncMock(return_value=[])
        storage.get_chunk = AsyncMock(return_value=None)

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        config = QueryConfig(
            enable_query_understanding=False,
            enable_entity_linking=False,
            enable_reranking=False,
            enable_keyword_search=False,
        )

        engine = HybridQueryEngine(storage=storage, embedder=embedder, config=config)
        return engine

    @pytest.mark.asyncio
    async def test_query_returns_query_result(self) -> None:
        """query() returns a QueryResult."""
        engine = self._make_engine()
        ns_id = uuid4()

        with patch("khora.telemetry.get_collector") as mock_tc:
            mock_tc.return_value = MagicMock()
            mock_tc.return_value.record_pipeline_stage = MagicMock()
            result = await engine.query("test query", ns_id)

        assert isinstance(result, QueryResult)
        assert result.metadata["mode"] == "HYBRID"

    @pytest.mark.asyncio
    async def test_query_vector_mode(self) -> None:
        """VECTOR mode only runs vector search."""
        engine = self._make_engine()
        ns_id = uuid4()
        config = QueryConfig(
            mode=SearchMode.VECTOR,
            enable_query_understanding=False,
            enable_entity_linking=False,
            enable_reranking=False,
            enable_keyword_search=False,
        )

        mock_chunk = MagicMock()
        mock_chunk.id = uuid4()
        mock_chunk.content = "test content"
        engine._storage.search_similar_chunks = AsyncMock(return_value=[(mock_chunk, 0.8)])

        with patch("khora.telemetry.get_collector") as mock_tc:
            mock_tc.return_value = MagicMock()
            mock_tc.return_value.record_pipeline_stage = MagicMock()
            result = await engine.query("test", ns_id, config=config)

        assert isinstance(result, QueryResult)

    @pytest.mark.asyncio
    async def test_query_caches_result(self) -> None:
        """query() caches the result for subsequent calls."""
        engine = self._make_engine()
        ns_id = uuid4()

        with patch("khora.telemetry.get_collector") as mock_tc:
            mock_tc.return_value = MagicMock()
            mock_tc.return_value.record_pipeline_stage = MagicMock()

            result1 = await engine.query("test query", ns_id)
            result2 = await engine.query("test query", ns_id)

        # Second call should return cached result
        assert result2 is result1

    @pytest.mark.asyncio
    async def test_query_agentic_mode(self) -> None:
        """Agentic mode delegates to AgenticSearchAgent."""
        engine = self._make_engine()
        ns_id = uuid4()

        mock_agentic_result = MagicMock()
        mock_agentic_result.chunks = []
        mock_agentic_result.entities = []
        mock_agentic_result.summary = "test summary"
        mock_agentic_result.trace = None
        mock_agentic_result.metadata = {}

        with patch("khora.query.agentic.AgenticSearchAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.search = AsyncMock(return_value=mock_agentic_result)
            mock_agent_cls.return_value = mock_agent

            result = await engine.query("complex query", ns_id, agentic=True)

        assert result.metadata.get("agentic") is True
        assert result.metadata.get("summary") == "test summary"


# ---------------------------------------------------------------------------
# HybridQueryEngine.find_related_entities
# ---------------------------------------------------------------------------


class TestFindRelatedEntities:
    """Tests for find_related_entities."""

    @pytest.mark.asyncio
    async def test_returns_scored_entities(self) -> None:
        """find_related_entities returns (entity, score) tuples using batch fetch."""
        storage = MagicMock()
        entity_id = uuid4()
        neighbor_id = uuid4()

        mock_entity = MagicMock()
        mock_entity.id = neighbor_id

        storage.get_neighborhood = AsyncMock(
            return_value={
                "entities": [{"id": str(neighbor_id)}],
                "relationships": [{"from": "a", "to": "b"}],
            }
        )
        # Use batch method instead of individual get_entity
        storage.get_entities_batch = AsyncMock(return_value={neighbor_id: mock_entity})

        config = QueryConfig(
            enable_query_understanding=False,
            enable_entity_linking=False,
            enable_reranking=False,
        )
        engine = HybridQueryEngine(storage=storage, config=config)

        results = await engine.find_related_entities(entity_id, uuid4(), max_depth=2)
        assert len(results) == 1
        assert results[0][0] is mock_entity
        assert 0.0 < results[0][1] <= 1.0

    @pytest.mark.asyncio
    async def test_empty_neighborhood(self) -> None:
        """find_related_entities returns empty for no neighbors."""
        storage = MagicMock()
        storage.get_neighborhood = AsyncMock(return_value={"entities": [], "relationships": []})

        config = QueryConfig(
            enable_query_understanding=False,
            enable_entity_linking=False,
            enable_reranking=False,
        )
        engine = HybridQueryEngine(storage=storage, config=config)

        results = await engine.find_related_entities(uuid4(), uuid4())
        assert results == []


# ---------------------------------------------------------------------------
# HybridQueryEngine.warm_cache
# ---------------------------------------------------------------------------


class TestWarmCache:
    """Tests for warm_cache method."""

    def _make_engine(self) -> HybridQueryEngine:
        """Create engine with mocked dependencies."""
        storage = MagicMock()
        storage.search_similar_chunks = AsyncMock(return_value=[])
        storage.search_similar_entities = AsyncMock(return_value=[])
        storage.get_entities_batch = AsyncMock(return_value={})
        storage.get_neighborhood = AsyncMock(return_value={})
        storage.get_neighborhoods_batch = AsyncMock(return_value={})
        storage.list_chunks = AsyncMock(return_value=[])
        storage.list_entities = AsyncMock(return_value=[])
        storage.search_fulltext_chunks = AsyncMock(return_value=[])

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        config = QueryConfig(
            enable_query_understanding=False,
            enable_entity_linking=False,
            enable_reranking=False,
            enable_keyword_search=False,
        )

        return HybridQueryEngine(storage=storage, embedder=embedder, config=config)

    @pytest.mark.asyncio
    async def test_warm_cache_with_explicit_queries(self) -> None:
        """warm_cache executes provided queries."""
        engine = self._make_engine()
        ns_id = uuid4()

        with patch("khora.telemetry.get_collector") as mock_tc:
            mock_tc.return_value = MagicMock()
            mock_tc.return_value.record_pipeline_stage = MagicMock()

            result = await engine.warm_cache(
                ns_id,
                queries=["query1", "query2"],
                include_entity_based=False,
            )

        assert result["queries_warmed"] == 2
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_warm_cache_generates_entity_queries(self) -> None:
        """warm_cache generates queries from top entities."""
        engine = self._make_engine()
        ns_id = uuid4()

        # Mock top entities
        mock_entity = MagicMock()
        mock_entity.name = "Alice"
        mock_entity.description = "A person"
        mock_entity.mention_count = 10
        engine._storage.list_entities = AsyncMock(return_value=[mock_entity])

        with patch("khora.telemetry.get_collector") as mock_tc:
            mock_tc.return_value = MagicMock()
            mock_tc.return_value.record_pipeline_stage = MagicMock()

            result = await engine.warm_cache(
                ns_id,
                queries=[],
                include_entity_based=True,
                max_entity_queries=5,
            )

        # Should have warmed entity-based queries
        assert result["queries_warmed"] >= 1

    @pytest.mark.asyncio
    async def test_warm_cache_handles_errors(self) -> None:
        """warm_cache tracks errors gracefully."""
        engine = self._make_engine()
        ns_id = uuid4()

        # Make the query method itself raise an exception (not just search)
        async def failing_query(*args, **kwargs):
            raise Exception("query completely failed")

        original_query = engine.query
        engine.query = failing_query

        result = await engine.warm_cache(
            ns_id,
            queries=["failing query"],
            include_entity_based=False,
        )

        assert result["queries_warmed"] == 0
        assert result["errors"] == 1

        # Restore original method
        engine.query = original_query

    @pytest.mark.asyncio
    async def test_warm_cache_returns_stats(self) -> None:
        """warm_cache returns cache statistics."""
        engine = self._make_engine()
        ns_id = uuid4()

        with patch("khora.telemetry.get_collector") as mock_tc:
            mock_tc.return_value = MagicMock()
            mock_tc.return_value.record_pipeline_stage = MagicMock()

            result = await engine.warm_cache(
                ns_id,
                queries=["test"],
                include_entity_based=False,
            )

        assert "cache_stats" in result
        assert "namespace_id" in result


# ---------------------------------------------------------------------------
# HybridQueryEngine.warm_keyword_index
# ---------------------------------------------------------------------------


class TestWarmKeywordIndex:
    """Tests for warm_keyword_index method."""

    def _make_engine(self) -> HybridQueryEngine:
        """Create engine with mocked dependencies."""
        storage = MagicMock()
        storage.list_chunks = AsyncMock(return_value=[])

        config = QueryConfig(
            enable_query_understanding=False,
            enable_entity_linking=False,
            enable_reranking=False,
        )

        return HybridQueryEngine(storage=storage, config=config)

    @pytest.mark.asyncio
    async def test_warm_keyword_index_builds_index(self) -> None:
        """warm_keyword_index builds BM25 index."""
        engine = self._make_engine()
        ns_id = uuid4()

        # Mock chunks
        mock_chunk = MagicMock()
        mock_chunk.id = uuid4()
        mock_chunk.content = "test content"
        engine._storage.list_chunks = AsyncMock(return_value=[mock_chunk])

        result = await engine.warm_keyword_index(ns_id)

        assert result["status"] == "indexed"
        assert result["chunk_count"] == 1
        assert str(ns_id) in engine._keyword_searchers

    @pytest.mark.asyncio
    async def test_warm_keyword_index_already_indexed(self) -> None:
        """warm_keyword_index returns early if already indexed."""
        engine = self._make_engine()
        ns_id = uuid4()

        # Pre-populate the index
        engine._keyword_searchers[str(ns_id)] = MagicMock()

        result = await engine.warm_keyword_index(ns_id)

        assert result["status"] == "already_indexed"
        engine._storage.list_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_warm_keyword_index_no_chunks(self) -> None:
        """warm_keyword_index handles empty namespace."""
        engine = self._make_engine()
        ns_id = uuid4()

        engine._storage.list_chunks = AsyncMock(return_value=[])

        result = await engine.warm_keyword_index(ns_id)

        assert result["status"] == "no_chunks"
        assert result["chunk_count"] == 0

    @pytest.mark.asyncio
    async def test_warm_keyword_index_handles_error(self) -> None:
        """warm_keyword_index handles errors gracefully."""
        engine = self._make_engine()
        ns_id = uuid4()

        engine._storage.list_chunks = AsyncMock(side_effect=Exception("db error"))

        result = await engine.warm_keyword_index(ns_id)

        assert result["status"] == "error"
        assert "error" in result
