"""Entity-vector-search and BM25 channel degradation observability (#1158).

ADR-001 (failure-observability contract): a channel that catches an
exception and returns a default MUST record a ``Degradation`` so the
silent fallback is observable on ``RecallResult.engine_info['degradations']``
and via the ``khora.{engine}.{component}.degraded_total`` counter.

Two VectorCypher recall channels previously swallowed failures with a
bare WARNING + ``return []``:

- ``_vector_search_entities``: entry-entity discovery. When it fails, the
  graph-expansion channel of GRAPH/HYBRID recall silently collapses to
  vector-only (no entry seeds -> ``_simple_retrieve`` fallback) with no
  machine-readable signal.
- ``_bm25_search_chunks``: the independent lexical channel. When it fails,
  the BM25 contribution silently disappears from RRF fusion.

These are pure-unit tests with a mocked storage coordinator / vector store
- no embedded stack, no LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.diagnostics import Degradation
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Channel-level unit tests (call the channel method directly)
# ---------------------------------------------------------------------------


async def test_vector_search_entities_records_degradation_on_failure() -> None:
    """When ``search_similar_entities`` raises, a Degradation is appended."""
    storage = MagicMock()
    storage.search_similar_entities = AsyncMock(side_effect=RuntimeError("pgvector down"))
    retriever = VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    degradations: list[Degradation] = []
    results = await retriever._vector_search_entities(
        query_embedding=[0.1] * 8,
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )

    # The channel degrades to empty rather than crashing ...
    assert results == []
    # ... and the silent fallback is now observable.
    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.entity_vector_search"
    assert deg["reason"] == "channel_exception"
    assert deg["exception"] == "RuntimeError"
    assert "pgvector down" in (deg.get("detail") or "")


async def test_vector_search_entities_no_degradation_when_sink_absent() -> None:
    """Without a ``degradations`` sink the channel still degrades cleanly."""
    storage = MagicMock()
    storage.search_similar_entities = AsyncMock(side_effect=RuntimeError("pgvector down"))
    retriever = VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    # ``degradations`` defaults to None - the guard must short-circuit.
    results = await retriever._vector_search_entities(
        query_embedding=[0.1] * 8,
        namespace_id=uuid4(),
        limit=10,
    )
    assert results == []


async def test_bm25_search_chunks_records_degradation_on_failure() -> None:
    """When the BM25 search raises, a Degradation is appended."""
    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("fulltext index missing"))
    vector_store = MagicMock()
    # No temporal-store fulltext method -> coordinator path is taken and raises.
    vector_store.search_fulltext = None
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="anything",
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )

    assert results == []
    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.bm25"
    assert deg["reason"] == "channel_exception"
    assert deg["exception"] == "RuntimeError"
    assert "fulltext index missing" in (deg.get("detail") or "")


async def test_bm25_search_chunks_no_degradation_when_sink_absent() -> None:
    """Without a ``degradations`` sink the BM25 channel still degrades cleanly."""
    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("fulltext index missing"))
    vector_store = MagicMock()
    vector_store.search_fulltext = None
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    results = await retriever._bm25_search_chunks(
        query="anything",
        namespace_id=uuid4(),
        limit=10,
    )
    assert results == []


# ---------------------------------------------------------------------------
# Empty-multitoken channel degradation (#1330): a >=2-token keyword query that
# returns ZERO BM25 rows is the OR-fix's residual failure mode. It does not
# raise, so it took its own reason value rather than channel_exception.
# ---------------------------------------------------------------------------


def _make_empty_bm25_retriever() -> VectorCypherRetriever:
    """A retriever whose BM25 channel returns 0 rows without raising."""
    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])
    vector_store = MagicMock()
    vector_store.search_fulltext = None
    return VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )


async def test_bm25_empty_multitoken_records_degradation() -> None:
    """A >=2-token query with 0 BM25 rows records an empty_multitoken_channel degradation."""
    retriever = _make_empty_bm25_retriever()
    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="status of MER-0001",
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )
    assert results == []
    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.bm25"
    assert deg["reason"] == "empty_multitoken_channel"


async def test_bm25_empty_singletoken_does_not_degrade() -> None:
    """A single-token (bare-ID) query with 0 rows is NOT a degradation.

    A bare ``MER-9999`` lookup that legitimately finds nothing is expected;
    only multi-token sentence queries that drop the whole lexical channel are
    the observable failure mode the #1330 fix targets.
    """
    retriever = _make_empty_bm25_retriever()
    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="MER-9999",
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )
    assert results == []
    assert degradations == []


async def test_bm25_empty_multitoken_under_filter_does_not_degrade() -> None:
    """Under a deterministic filter_ast, a 0-row multitoken result is a
    legitimate filtered miss (the predicate excluded every candidate), NOT a
    broken lexical channel. Flagging it would inflate the public counter with
    benign events (CodeRabbit on PR #1332)."""
    from khora.filter.ast import FilterNode, FilterOp

    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])
    vector_store = MagicMock()
    # Filtered path goes through the temporal store's search_fulltext, which
    # honors the predicate and legitimately returns 0 rows.
    vector_store.search_fulltext = AsyncMock(return_value=[])
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="status of MER-0001",
        namespace_id=uuid4(),
        limit=10,
        filter_ast=FilterNode(op=FilterOp.AND),
        degradations=degradations,
    )
    assert results == []
    assert degradations == []


# ---------------------------------------------------------------------------
# Full retrieve() tests (degradation surfaces on the result, recall survives)
# ---------------------------------------------------------------------------


def _make_retriever_for_retrieve() -> VectorCypherRetriever:
    """A retriever wired for the COMPLEX (graph-expansion) retrieve path.

    The vector chunk channel returns one chunk so recall produces a result;
    everything else is mocked.
    """
    vector_store = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    storage = AsyncMock()

    ns_id = uuid4()
    doc_id = uuid4()
    chunk_id = uuid4()

    mock_result = MagicMock()
    mock_result.chunk = MagicMock()
    mock_result.chunk.id = chunk_id
    mock_result.chunk.namespace_id = ns_id
    mock_result.chunk.content = "vector chunk"
    mock_result.chunk.document_id = doc_id
    mock_result.chunk.occurred_at = None
    mock_result.chunk.created_at = None
    mock_result.chunk.source_timestamp = None
    mock_result.chunk.metadata = {}
    mock_result.chunk.chunker_info = {}
    mock_result.combined_score = 0.85
    mock_result.similarity = 0.85
    vector_store.search = AsyncMock(return_value=[mock_result])

    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])

    config = RetrieverConfig(coherence_weight=0.0, enable_bm25_channel=False)

    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=embedder,
        config=config,
        storage=storage,
    )

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
    return retriever


async def test_retrieve_surfaces_entity_channel_degradation_and_degrades_to_vector() -> None:
    """An entity-vector-search failure surfaces a Degradation on the result.

    With no entry entities, recall degrades to the vector-only simple path
    instead of crashing, and the Degradation rides along on the result
    metadata (the engine spreads it onto engine_info['degradations']).
    """
    retriever = _make_retriever_for_retrieve()
    retriever._storage.search_similar_entities = AsyncMock(side_effect=RuntimeError("pgvector down"))

    result = await retriever.retrieve("Tell me about Alice", uuid4())

    assert isinstance(result, VectorCypherResult)
    # Recall survived and produced chunks via the vector-only fallback.
    assert len(result.chunks) > 0
    degradations = result.metadata.get("degradations") or []
    components = {d.get("component") for d in degradations}
    assert "vectorcypher.entity_vector_search" in components, f"degradations: {degradations!r}"
    entity_deg = next(d for d in degradations if d.get("component") == "vectorcypher.entity_vector_search")
    assert entity_deg["reason"] == "channel_exception"


async def test_retrieve_surfaces_bm25_channel_degradation() -> None:
    """A BM25 failure surfaces a Degradation while recall still returns chunks."""
    retriever = _make_retriever_for_retrieve()
    # Enable the BM25 channel and make it fail.
    retriever._config.enable_bm25_channel = True
    retriever._storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
    retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
    retriever._storage.get_entities_batch = AsyncMock(return_value={})
    retriever._dual_nodes = None
    # The temporal store exposes a fulltext method that raises.
    retriever._vector_store.search_fulltext = AsyncMock(side_effect=RuntimeError("fulltext down"))

    result = await retriever.retrieve("lexical query", uuid4())

    assert isinstance(result, VectorCypherResult)
    degradations = result.metadata.get("degradations") or []
    components = {d.get("component") for d in degradations}
    assert "vectorcypher.bm25" in components, f"degradations: {degradations!r}"
    bm25_deg = next(d for d in degradations if d.get("component") == "vectorcypher.bm25")
    assert bm25_deg["reason"] == "channel_exception"
