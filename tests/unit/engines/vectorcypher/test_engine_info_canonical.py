"""Canonical ``engine_info`` schema test for ``VectorCypherEngine.recall()``.

Pins the contract for the canonical keys emitted by the vectorcypher path:

    engine, mode, channels_used, rrf_k, temporal_signal, abstention_signals

Existing engine_info keys (``routing``, ``use_graph``, ``graph_depth``,
``raw_chunk_count``, ``validated_chunk_count``, ``temporal_category``,
``temporal_confidence``, ``is_temporal``, ``retrieval_mean_score``,
``retrieval_score_variance``, ``retrieval_top_score_gap``) MUST remain —
the canonical keys are added, not a replacement.

Mirrors the mock-at-the-retriever-boundary pattern from
``test_engine_coverage.py::TestRecallContextFormatting`` so no live
Neo4j / Postgres / Lance is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity, Relationship
from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.engines.vectorcypher.retriever import VectorCypherResult
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.query import SearchMode


def _make_config() -> MagicMock:
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = "bolt://localhost:7687"
    config.get_neo4j_user.return_value = "neo4j"
    config.get_neo4j_password.return_value = "password"
    config.get_neo4j_database.return_value = "neo4j"
    config.get_graph_config.return_value = MagicMock()
    config.get_vector_config.return_value = MagicMock()
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.embedding_dimension = 1536
    config.llm.model = "gpt-4o-mini"
    config.llm.timeout = 30
    config.pipeline.extract_entities = True
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _make_connected_engine() -> VectorCypherEngine:
    engine = VectorCypherEngine(_make_config())
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._retriever = AsyncMock()
    engine._router = MagicMock()
    engine._neo4j_driver = AsyncMock()
    return engine


def _make_populated_engine() -> VectorCypherEngine:
    """Engine with a mocked retriever that returns one chunk + one entity +
    one relationship — enough for ``result.chunks`` to be non-empty and for
    the canonical keys to be populated."""
    engine = _make_connected_engine()

    routing = RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=0.8,
        reasoning="test",
    )
    chunk = Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="Long enough content for the validation pass to keep this chunk in results",
    )
    entity = Entity(name="Alice", entity_type="PERSON", description="An engineer")
    relationship = Relationship(
        relationship_type="KNOWS",
        source_entity_name="Alice",
        target_entity_name="Bob",
        description="works with",
    )
    retriever_result = VectorCypherResult(
        chunks=[(chunk, 0.9)],
        entities=[(entity, 0.9)],
        relationships=[(relationship, 0.7)],
        routing_decision=routing,
        metadata={"vector_chunk_count": 1, "graph_chunk_count": 0, "bm25_chunk_count": 0},
    )
    engine._retriever.retrieve = AsyncMock(return_value=retriever_result)
    return engine


@pytest.mark.unit
class TestEngineInfoCanonicalKeys:
    """Canonical engine_info contract for vectorcypher recall."""

    @pytest.mark.asyncio
    async def test_all_six_canonical_keys_present(self) -> None:
        engine = _make_populated_engine()
        result = await engine.recall("q", uuid4(), mode=SearchMode.HYBRID)

        # 1. All 6 canonical keys must be present.
        for key in ("engine", "mode", "channels_used", "rrf_k", "temporal_signal", "abstention_signals"):
            assert key in result.engine_info, f"canonical key missing: {key!r}"

    @pytest.mark.asyncio
    async def test_engine_field_is_vectorcypher(self) -> None:
        engine = _make_populated_engine()
        result = await engine.recall("q", uuid4(), mode=SearchMode.HYBRID)
        # 2. engine_info["engine"] == "vectorcypher".
        assert result.engine_info["engine"] == "vectorcypher"

    @pytest.mark.asyncio
    async def test_mode_echoes_search_mode_value(self) -> None:
        engine = _make_populated_engine()
        result = await engine.recall("q", uuid4(), mode=SearchMode.HYBRID)
        # 3. engine_info["mode"] echoes the passed SearchMode as a lowercase string.
        assert result.engine_info["mode"] == "hybrid"

    @pytest.mark.asyncio
    async def test_temporal_signal_is_dict_with_category_and_source(self) -> None:
        engine = _make_populated_engine()
        result = await engine.recall("q", uuid4(), mode=SearchMode.HYBRID)
        # 4. engine_info["temporal_signal"] is a dict with both category (str)
        #    and source (str) keys.
        ts = result.engine_info["temporal_signal"]
        assert isinstance(ts, dict)
        assert "category" in ts
        assert "source" in ts
        assert isinstance(ts["category"], str)
        assert isinstance(ts["source"], str)

    @pytest.mark.asyncio
    async def test_abstention_signals_shape(self) -> None:
        engine = _make_populated_engine()
        result = await engine.recall("q", uuid4(), mode=SearchMode.HYBRID)
        # 5. engine_info["abstention_signals"] is a 6-key dict with the
        #    expected types.
        signals = result.engine_info["abstention_signals"]
        assert isinstance(signals, dict)
        expected_keys = {
            "entities_empty",
            "chunks_empty",
            "chunks_below_min",
            "top_score_low",
            "combined_score",
            "should_abstain",
        }
        assert set(signals.keys()) == expected_keys
        assert isinstance(signals["entities_empty"], bool)
        assert isinstance(signals["chunks_empty"], bool)
        assert isinstance(signals["chunks_below_min"], bool)
        assert isinstance(signals["top_score_low"], bool)
        assert isinstance(signals["combined_score"], float)
        assert isinstance(signals["should_abstain"], bool)

    @pytest.mark.asyncio
    async def test_channels_used_is_constrained_string_list(self) -> None:
        engine = _make_populated_engine()
        result = await engine.recall("q", uuid4(), mode=SearchMode.HYBRID)
        # 6. engine_info["channels_used"] is a list[str] of {vector, graph, bm25}.
        channels = result.engine_info["channels_used"]
        assert isinstance(channels, list)
        allowed = {"vector", "graph", "bm25"}
        for channel in channels:
            assert isinstance(channel, str)
            assert channel in allowed, f"unexpected channel: {channel!r}"

    @pytest.mark.asyncio
    async def test_rrf_k_matches_config(self) -> None:
        engine = _make_populated_engine()
        result = await engine.recall("q", uuid4(), mode=SearchMode.HYBRID)
        # 7. engine_info["rrf_k"] is an int matching engine._vc_config.fusion_rrf_k.
        rrf_k = result.engine_info["rrf_k"]
        assert isinstance(rrf_k, int)
        assert rrf_k == engine._vc_config.fusion_rrf_k

    @pytest.mark.asyncio
    async def test_existing_keys_still_present(self) -> None:
        """Canonical keys add to ``engine_info``; existing keys must remain."""
        engine = _make_populated_engine()
        result = await engine.recall("q", uuid4(), mode=SearchMode.HYBRID)

        # Existing keys per the ticket — the addition didn't replace them.
        existing_keys = [
            "routing",
            "use_graph",
            "graph_depth",
            "raw_chunk_count",
            "validated_chunk_count",
            "temporal_category",
            "temporal_confidence",
            "is_temporal",
            "retrieval_mean_score",
            "retrieval_score_variance",
            "retrieval_top_score_gap",
        ]
        for key in existing_keys:
            assert key in result.engine_info, f"existing engine_info key missing after canonical add: {key!r}"
