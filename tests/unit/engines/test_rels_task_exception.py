"""Unit tests for rels_task exception handling in VectorCypher retriever.

When `get_relationships_between()` raises during retrieval, the retriever
should catch the error, log a warning, and return results with empty
relationships — preserving entities and chunks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision


def _make_retriever(
    rels_side_effect: Exception | None = None,
) -> VectorCypherRetriever:
    """Build a VectorCypherRetriever with mocked dependencies.

    Args:
        rels_side_effect: If set, get_relationships_between raises this exception.
    """
    vector_store = AsyncMock()
    neo4j_driver = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    storage = AsyncMock()

    ns_id = uuid4()
    doc_id = uuid4()
    chunk_id = uuid4()
    entity_id_1 = uuid4()
    entity_id_2 = uuid4()

    # --- Vector store returns one chunk ---
    mock_result = MagicMock()
    mock_result.chunk = MagicMock()
    mock_result.chunk.id = chunk_id
    mock_result.chunk.namespace_id = ns_id
    mock_result.chunk.content = "test chunk"
    mock_result.chunk.document_id = doc_id
    mock_result.chunk.occurred_at = None
    mock_result.chunk.created_at = None
    mock_result.chunk.metadata = {}
    mock_result.combined_score = 0.85
    mock_result.similarity = 0.85
    vector_store.search = AsyncMock(return_value=[mock_result])

    # --- Storage returns entities ---
    entity_1 = Entity(
        id=entity_id_1,
        namespace_id=ns_id,
        name="Alice",
        entity_type="PERSON",
    )
    entity_2 = Entity(
        id=entity_id_2,
        namespace_id=ns_id,
        name="Bob",
        entity_type="PERSON",
    )
    storage.get_entities_batch = AsyncMock(return_value={entity_id_1: entity_1, entity_id_2: entity_2})

    config = RetrieverConfig(
        query_cache_ttl_seconds=0,
        coherence_weight=0.0,
    )

    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=neo4j_driver,
        embedder=embedder,
        config=config,
        storage=storage,
    )

    # Mock the router to return COMPLEX routing (triggers _vectorcypher_retrieve)
    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.COMPLEX,
            use_graph=True,
            graph_depth=2,
            confidence=0.9,
            reasoning="complex query",
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=2)

    # Mock internal methods to provide a working pipeline
    retriever._vector_search_entities = AsyncMock(return_value=[(entity_id_1, 0.9), (entity_id_2, 0.8)])
    retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    retriever._fetch_chunks_from_entities = AsyncMock(
        return_value=[
            (chunk_id, 0.7, Chunk(id=chunk_id, namespace_id=ns_id, document_id=doc_id, content="graph chunk"))
        ]
    )

    # Mock the dual_nodes manager — this is the key part
    if rels_side_effect:
        retriever._dual_nodes.get_relationships_between = AsyncMock(side_effect=rels_side_effect)
    else:
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])

    return retriever


@pytest.mark.unit
class TestRelsTaskExceptionHandling:
    """Tests for parallel rels_task exception handling."""

    @pytest.mark.asyncio
    async def test_rels_task_failure_returns_empty_relationships(self) -> None:
        """When get_relationships_between raises, VectorCypherResult has empty relationships."""
        retriever = _make_retriever(rels_side_effect=RuntimeError("Neo4j connection lost"))

        result = await retriever.retrieve("Tell me about Alice and Bob", uuid4())

        assert isinstance(result, VectorCypherResult)
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_rels_task_failure_preserves_entities_and_chunks(self) -> None:
        """Entities and chunks are still present despite relationship failure."""
        retriever = _make_retriever(rels_side_effect=RuntimeError("Neo4j connection lost"))

        result = await retriever.retrieve("Tell me about Alice and Bob", uuid4())

        assert isinstance(result, VectorCypherResult)
        # Should still have entities
        assert len(result.entities) > 0
        # Should still have chunks
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_rels_task_failure_logs_warning(self) -> None:
        """Verify a warning is logged when the relationship fetch fails."""
        retriever = _make_retriever(rels_side_effect=RuntimeError("Neo4j connection lost"))

        with patch("khora.engines.vectorcypher.retriever.logger") as mock_logger:
            await retriever.retrieve("Tell me about Alice and Bob", uuid4())

            mock_logger.warning.assert_called()
            # Verify the warning message mentions relationship failure
            warning_call = mock_logger.warning.call_args
            assert "Relationship fetch failed" in str(warning_call)
