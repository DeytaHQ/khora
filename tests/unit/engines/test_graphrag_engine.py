"""Unit tests for the GraphRAG engine recall path."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.memory_lake import RecallResult


def _mock_khora_config() -> MagicMock:
    """Create a mock KhoraConfig sufficient for GraphRAGEngine construction."""
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = None
    config.get_neo4j_user.return_value = None
    config.get_neo4j_password.return_value = None
    config.get_neo4j_database.return_value = None
    config.get_graph_config.return_value = None
    config.get_vector_config.return_value = None
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.embedding_dimension = 1536
    config.llm.model = "gpt-4o-mini"
    config.llm.embedding_model = "text-embedding-3-small"
    config.llm.embedding_dimension = 1536
    config.llm.timeout = 30
    config.llm.max_retries = 3
    config.llm.extraction_model = None
    config.llm.max_concurrent_llm_calls = 5
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.pipeline.extract_entities = True
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


@pytest.mark.unit
class TestGraphRAGEngineApiTemporalFilter:
    """DYT-3605: API-supplied temporal_filter synthesizes an EXPLICIT signal
    and propagates to recall metadata as temporal_category/temporal_confidence."""

    @pytest.mark.asyncio
    async def test_recall_with_api_temporal_filter_metadata_reports_explicit(self) -> None:
        """When recall() is invoked with a temporal_filter, the resulting
        RecallResult.metadata must report temporal_category="explicit" and
        temporal_confidence=1.0 — even though the engine never ran the
        dictionary/semantic detector."""
        from khora.engines.graphrag.engine import GraphRAGEngine
        from khora.engines.skeleton.backends import TemporalFilter
        from khora.query.engine import QueryResult

        engine = GraphRAGEngine(_mock_khora_config())
        engine._connected = True
        engine._storage = AsyncMock()
        engine._embedder = AsyncMock()

        query_engine = MagicMock()
        query_result = QueryResult(
            chunks=[],
            entities=[],
            metadata={"engine": "graphrag"},
        )
        query_engine.query = AsyncMock(return_value=query_result)
        engine._query_engine = query_engine

        # Bounds chosen so any in-window chunk would sit strictly in
        # [occurred_after, occurred_before) — occurred_before is exclusive.
        tf = TemporalFilter(
            occurred_after=datetime(2025, 1, 1, tzinfo=UTC),
            occurred_before=datetime(2025, 6, 1, tzinfo=UTC),
        )

        result = await engine.recall("anything", uuid4(), temporal_filter=tf)

        assert isinstance(result, RecallResult)
        assert result.metadata["temporal_category"] == "explicit"
        assert result.metadata["temporal_confidence"] == 1.0
