"""#833 mode contract tests for VectorCypherEngine.

VectorCypher implements all five modes honestly. The retriever respects
``mode`` by gating channels (vector / graph / BM25).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.engines.vectorcypher.retriever import VectorCypherResult
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.exceptions import EngineCapabilityError
from khora.query import SearchMode


def _make_config() -> MagicMock:
    config = MagicMock()
    config.storage.backend = "pgvector"
    config.llm.model = "gpt-4o-mini"
    config.llm.timeout = 30
    config.pipeline.extract_entities = True
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    # Abstention knobs (#1331) — the engine reads these off config.query now.
    config.query.abstention_min_chunks = 1
    config.query.abstention_min_top_score = 0.3
    config.query.abstention_combined_threshold = 0.5
    config.query.abstention_weight_entities_empty = 0.3
    config.query.abstention_weight_chunks_below_min = 0.4
    config.query.abstention_weight_top_score_low = 0.3
    config.query.abstention_mode = "cosine_floor"
    config.query.abstention_confidence_target_cosine = 0.5
    config.query.abstention_confidence_target_gap = 0.1
    return config


def _make_engine() -> VectorCypherEngine:
    engine = VectorCypherEngine(_make_config())
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._retriever = AsyncMock()
    engine._router = MagicMock()
    engine._neo4j_driver = AsyncMock()

    routing = RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=0.5,
        reasoning="t",
    )
    engine._retriever.retrieve = AsyncMock(
        return_value=VectorCypherResult(
            chunks=[],
            entities=[],
            relationships=[],
            routing_decision=routing,
            metadata={},
        )
    )
    return engine


def test_supported_modes_declaration() -> None:
    """VectorCypher honestly supports all five SearchMode values."""
    assert VectorCypherEngine.supported_modes == frozenset(
        {
            SearchMode.VECTOR,
            SearchMode.GRAPH,
            SearchMode.HYBRID,
            SearchMode.ALL,
            SearchMode.KEYWORD,
        }
    )


@pytest.mark.parametrize(
    "mode",
    [
        SearchMode.VECTOR,
        SearchMode.GRAPH,
        SearchMode.HYBRID,
        SearchMode.ALL,
        SearchMode.KEYWORD,
    ],
)
async def test_recall_forwards_mode_to_retriever(mode: SearchMode) -> None:
    """Every supported mode must be passed verbatim to retriever.retrieve()."""
    engine = _make_engine()
    await engine.recall("q", uuid4(), mode=mode)
    call = engine._retriever.retrieve.call_args
    assert call.kwargs["mode"] is mode


async def test_recall_emits_mode_in_engine_info() -> None:
    """The recall response's engine_info["mode"] is the requested mode (lower-cased)."""
    engine = _make_engine()
    result = await engine.recall("q", uuid4(), mode=SearchMode.VECTOR)
    assert result.engine_info["mode"] == "vector"


async def test_recall_unsupported_mode_path_blocked_at_engine() -> None:
    """When VectorCypher's supported_modes is artificially restricted (e.g. a
    subclass dropping GRAPH), the engine-level guard fires before retriever
    work. This protects the contract for downstream engines that swap the
    frozenset."""
    engine = _make_engine()
    # Simulate a more restrictive subclass by replacing supported_modes on
    # the instance. The guard checks ``self.supported_modes``.
    engine.supported_modes = frozenset({SearchMode.HYBRID})
    with pytest.raises(EngineCapabilityError) as excinfo:
        await engine.recall("q", uuid4(), mode=SearchMode.VECTOR)
    assert excinfo.value.engine_name == "vectorcypher"
    assert excinfo.value.mode is SearchMode.VECTOR
    engine._retriever.retrieve.assert_not_awaited()
