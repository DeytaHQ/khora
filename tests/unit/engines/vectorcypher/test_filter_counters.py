"""Hermetic spies for the two VectorCypher recall-filter service counters.

The recall-filter pushdown/enforcement contract is already pinned by the
pushdown-spy suite (``tests/unit/engines/test_vectorcypher_filter_pushdown.py``):
that the validated ``filter_ast`` reaches every retrieval channel unchanged, that
an unsupported Cypher compile propagates ``RecallFilterUnsupportedError`` instead
of degrading to vector-only, and that a transient Neo4j error degrades cleanly.
The one thing that suite does NOT assert is that the two *service-level filter
counters* actually fire on their declared conditions. That is this file's only
job:

* ``khora.recall.filter.graph_channel_empty`` — ``record_graph_channel_empty()``
  at ``retriever.py:1651``, fired when ``filter_ast is not None and graph_chunks``
  (the graph channel HELD candidates) and the full-AST metadata post-filter then
  empties it.
* ``khora.recall.filter.under_filled`` — ``record_under_filled()`` at
  ``engine.py:2195``, fired when ``filter_ast is not None and len(result.chunks) <
  limit`` (a caller filter narrowed the result below the requested k).

Both are driven against the REAL call site with a stubbed retriever/engine, so
no database, no embeddings model, and no ranking are involved — and each fires-
test is paired with a silent-control that proves the signal comes from the
declared condition, not merely from the presence of a filter. These run in the
main ``test-unit`` job (no Docker). The companion real-PG+Neo4j proof that a
genuine graph channel is emptied by a live metadata filter lives in
``tests/integration/test_vectorcypher_filter_counters.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

import khora.filter.telemetry as filter_telemetry
from khora.config import KhoraConfig
from khora.core.models import Chunk
from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.filter import RecallFilter, parse_to_ast
from khora.query import SearchMode

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Deterministic LLM stub — guarantees no real embedder is reached even if a
# call site touches one. The hermetic paths below stub the retriever/engine, so
# this is belt-and-suspenders for full isolation in the no-network unit job.
# ---------------------------------------------------------------------------


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [[1.0] + [0.0] * 1535 for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return [1.0] + [0.0] * 1535


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )


class _RecordingCounter:
    """Captures ``.add(value, attributes=...)`` calls for assertions."""

    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


# ===========================================================================
# graph_channel_empty — fires when a non-empty graph channel is emptied.
# ===========================================================================


def _graph_chunk(ns_id: UUID, *, tag: str) -> Chunk:
    """A chunk the graph channel returns, carrying a ``metadata.tag`` value."""
    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=uuid4(),
        content=f"graph chunk tagged {tag}",
        metadata={"tag": tag},
    )


def _make_retriever(ns_id: UUID) -> VectorCypherRetriever:
    """A retriever wired so BOTH the vector and BM25 chunk channels fire.

    Mirrors ``_make_retriever`` in
    ``tests/unit/engines/test_vectorcypher_filter_pushdown.py`` (the MODERATE /
    ``_vectorcypher_retrieve`` path): the vector channel returns one row and the
    BM25 channel returns one row, so the "vector/BM25 returned rows" arm of the
    graph-channel-empty condition genuinely holds. Graph helpers are stubbed so
    the cypher-expansion path completes without Neo4j; the caller overrides
    ``_fetch_chunks_from_entities`` to seed the graph channel.
    """
    vector_store = AsyncMock()
    neo4j_driver = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.model_name = "test-model"
    embedder.dimension = 1536

    vec_result = MagicMock()
    vec_result.chunk = MagicMock()
    vec_result.chunk.id = uuid4()
    vec_result.chunk.namespace_id = ns_id
    vec_result.chunk.document_id = uuid4()
    vec_result.chunk.content = "vector channel chunk"
    vec_result.chunk.occurred_at = None
    vec_result.chunk.created_at = None
    vec_result.chunk.source_timestamp = None
    vec_result.chunk.metadata = {}
    vec_result.chunk.chunker_info = {}
    vec_result.combined_score = 0.85
    vec_result.similarity = 0.85
    vector_store.search = AsyncMock(return_value=[vec_result])

    bm25_chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="bm25 channel chunk")
    vector_store.search_fulltext = AsyncMock(return_value=[(bm25_chunk, 1.0)])

    storage = AsyncMock()
    storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
    storage.get_entities_batch = AsyncMock(return_value={})
    storage.search_fulltext_chunks = AsyncMock(return_value=[(bm25_chunk, 1.0)])

    config = RetrieverConfig(enable_bm25_channel=True, enable_session_aware_search=False)
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=neo4j_driver,
        embedder=embedder,
        config=config,
        storage=storage,
    )

    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.MODERATE,
            use_graph=True,
            graph_depth=2,
            confidence=0.8,
            reasoning="moderate",
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=2)
    retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
    retriever._version_filter_entities = AsyncMock(return_value=[])
    return retriever


async def test_graph_channel_empty_counter_fires_for_emptied_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """``graph_channel_empty`` fires when the graph channel HELD candidates that
    the metadata post-filter then emptied — never vacuously.

    The graph channel is seeded with two chunks whose tags violate the filter, so
    the channel is genuinely NON-EMPTY before the full-AST in-memory post-filter
    runs; the post-filter then drops both, narrowing the graph channel to empty.
    Only THAT condition (``graph_chunks`` truthy -> post-filter -> empty) fires the
    service-level ``khora.recall.filter.graph_channel_empty`` counter. We spy on
    the lazily-built singleton (pre-seeding the module global makes the helper's
    getter return the fake) and assert it incremented exactly once with no labels.
    The negative control below proves it stays silent when a graph chunk survives.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_graph_channel_empty_counter", counter)

    ns_id = uuid4()
    retriever = _make_retriever(ns_id)
    # Both graph chunks violate the filter -> the channel is non-empty on fetch,
    # then the metadata post-filter empties it. This is the genuine "channel held
    # candidates and the predicate emptied it" trigger, not a vacuous empty.
    violators = [_graph_chunk(ns_id, tag="noise"), _graph_chunk(ns_id, tag="other")]
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(c.id, 0.9, c) for c in violators])

    ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent", "release"]}}))
    result = await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

    assert counter.adds == [(1, {})], (
        f"graph_channel_empty must fire exactly once (no labels) when a non-empty "
        f"graph channel is emptied under the filter; got {counter.adds}"
    )
    # Corroborate the RIGHT reason: the graph channel did hold candidates and was
    # narrowed to empty (the fused provenance carries zero graph chunks, and the
    # failure-observability degradation entry names the emptied channel).
    assert result.metadata["graph_chunk_count"] == 0
    degradations = result.metadata.get("degradations", [])
    assert any(
        d.get("component") == "vectorcypher.graph_channel" and d.get("reason") == "empty_under_filter"
        for d in degradations
    ), f"expected the graph-channel-empty degradation; got {degradations}"


async def test_graph_channel_empty_counter_silent_when_chunk_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control — when a graph chunk survives the post-filter, no counter fires.

    Proves the positive signal comes from the channel genuinely emptying, not from
    the mere presence of a filter: a single graph chunk whose tag satisfies the
    predicate survives, so ``graph_chunks`` stays non-empty after the post-filter
    and the counter must NOT increment.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_graph_channel_empty_counter", counter)

    ns_id = uuid4()
    retriever = _make_retriever(ns_id)
    survivor = _graph_chunk(ns_id, tag="urgent")  # satisfies the filter
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(survivor.id, 0.9, survivor)])

    ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent", "release"]}}))
    await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

    assert counter.adds == [], "graph_channel_empty must stay silent while a graph chunk survives the filter"


# ===========================================================================
# under_filled — fires when a filtered recall returns fewer than the limit.
# ===========================================================================


def _make_engine_with_stub_retriever(ns_id: UUID, *, chunks: list[tuple[Chunk, float]]) -> VectorCypherEngine:
    """A VectorCypher engine whose retriever is stubbed to return ``chunks``.

    The engine's ``recall`` is pure after the retriever returns (validation,
    abstention signals, and document projection read only the in-memory
    ``VectorCypherResult``), so a stubbed ``_get_retriever`` is enough to drive
    the real ``record_under_filled`` call site with no database. The stub
    retriever carries a real ``RetrieverConfig`` because ``recall`` saves/restores
    ``retriever._config.hybrid_alpha`` around the call.
    """
    engine = VectorCypherEngine(KhoraConfig())

    routing = RoutingDecision(
        complexity=QueryComplexity.MODERATE,
        use_graph=True,
        graph_depth=2,
        confidence=0.8,
        reasoning="moderate",
    )
    vc_result = VectorCypherResult(
        chunks=chunks,
        entities=[],
        routing_decision=routing,
        relationships=[],
        metadata={
            "max_raw_vector_score": 0.9,
            "vector_chunk_count": len(chunks),
            "graph_chunk_count": 0,
            "bm25_chunk_count": 0,
        },
    )

    stub_retriever = MagicMock()
    stub_retriever._config = RetrieverConfig()
    stub_retriever.retrieve = AsyncMock(return_value=vc_result)
    engine._get_retriever = MagicMock(return_value=stub_retriever)  # type: ignore[method-assign]
    return engine


async def test_under_filled_counter_fires_when_filtered_recall_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """``under_filled`` fires when a FILTERED recall returns fewer than the limit.

    The engine records ``khora.recall.filter.under_filled`` once per call when a
    caller filter is present AND the result has fewer chunks than the requested
    ``limit``. We stub the retriever to return a single chunk and request a much
    larger limit, with a filter supplied — the real engine call site must fire the
    counter exactly once with no labels. Spied via the lazily-built singleton.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_under_filled_counter", counter)

    ns_id = uuid4()
    chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="alpha bravo charlie content")
    engine = _make_engine_with_stub_retriever(ns_id, chunks=[(chunk, 0.9)])

    ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent"]}}))
    result = await engine.recall("alpha bravo charlie", ns_id, limit=10, mode=SearchMode.VECTOR, filter_ast=ast)

    assert len(result.chunks) < 10, "precondition: the filtered result is under the requested limit"
    assert counter.adds == [(1, {})], (
        f"under_filled must fire exactly once (no labels) when a filtered recall "
        f"returns fewer than the requested limit; got {counter.adds}"
    )


async def test_under_filled_counter_silent_without_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control — a short result with NO filter leaves ``under_filled`` silent.

    The under-filled counter is owned by the filter subsystem: it fires only when
    a caller filter narrowed the candidate set. A short result with no
    ``filter_ast`` is ordinary low recall, not filter-induced under-fill, so the
    counter must NOT increment — even though the same single-chunk result is below
    the requested limit.
    """
    counter = _RecordingCounter()
    monkeypatch.setattr(filter_telemetry, "_under_filled_counter", counter)

    ns_id = uuid4()
    chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="alpha bravo charlie content")
    engine = _make_engine_with_stub_retriever(ns_id, chunks=[(chunk, 0.9)])

    result = await engine.recall("alpha bravo charlie", ns_id, limit=10, mode=SearchMode.VECTOR)

    assert len(result.chunks) < 10, "precondition: the unfiltered result is under the requested limit"
    assert counter.adds == [], "under_filled must stay silent when no caller filter is supplied"
