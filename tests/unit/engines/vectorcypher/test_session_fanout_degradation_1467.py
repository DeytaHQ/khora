"""Session fan-out per-channel degradation observability (#1467).

ADR-001 (failure-observability contract): a channel that catches an
exception and drops its contribution MUST record a ``Degradation`` so the
silent fallback is observable on ``RecallResult.engine_info['degradations']``
and via the ``khora.vectorcypher.session_fanout.degraded_total`` counter.

The session-aware fan-out gathers one per-session vector search per discovered
channel plus one unscoped fallback with ``return_exceptions=True``. Previously a
per-session search that raised was logged at WARNING and its chunks were
dropped from the merge with no machine-readable signal - a recall that lost
several session channels looked healthy on ``RecallResult.engine_info``.

These are pure-unit tests with a mocked storage coordinator / vector store /
dual-node manager - no embedded stack, no LLM.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.query.temporal_detection import TemporalCategory, TemporalSignal

pytestmark = pytest.mark.unit


def _make_chunk(namespace_id):
    chunk = MagicMock()
    chunk.id = uuid4()
    chunk.namespace_id = namespace_id
    chunk.content = "session chunk"
    chunk.document_id = uuid4()
    chunk.occurred_at = None
    chunk.created_at = None
    chunk.source_timestamp = None
    chunk.metadata = {}
    chunk.chunker_info = {}
    # No embedding -> attach_relevance_scores reports 0.0 rather than trying to
    # cosine a MagicMock (which the accel layer rejects as dimension 0).
    chunk.embedding = None
    return chunk


def _make_session_fanout_retriever(ns_id):
    """A retriever wired so the session-aware fan-out activates.

    Two discovered channels + one entry entity + a temporal signal + the
    session-aware flag drive the ``len(fanout_channels) >= 2`` branch that
    issues per-session ``_vector_search_chunks`` calls.
    """
    vector_store = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    storage = AsyncMock()
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])

    config = RetrieverConfig(
        coherence_weight=0.0,
        enable_bm25_channel=False,
        enable_session_aware_search=True,
    )

    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=embedder,
        config=config,
        storage=storage,
    )

    # One entry entity so the fan-out precondition (>=1) holds.
    retriever._vector_search_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
    # Empty graph expansion keeps the pipeline on the vector/session channels.
    retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
    retriever._storage.get_entities_batch = AsyncMock(return_value={})

    # DualNodeManager: two discovered session channels activates the fan-out.
    dual_nodes = MagicMock()
    dual_nodes.get_entity_channels = AsyncMock(return_value=["session-a", "session-b"])
    dual_nodes.get_relationships_between = AsyncMock(return_value=[])
    retriever._dual_nodes = dual_nodes

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


async def test_session_fanout_failure_surfaces_degradation_and_still_returns_chunks() -> None:
    """A per-session channel failure records a Degradation; recall survives.

    ``_vector_search_chunks`` raises for the ``session-a`` channel and returns a
    chunk for every other call (the ``session-b`` channel and the unscoped
    fallback, which carries ``channel=None``). The failed channel is dropped
    from the merge but is now observable on the result's degradations, and the
    surviving channels still yield chunks.
    """
    ns_id = uuid4()
    retriever = _make_session_fanout_retriever(ns_id)

    async def fake_vector_search_chunks(*, temporal_filter=None, **kwargs):
        channel = getattr(temporal_filter, "channel", None)
        if channel == "session-a":
            raise RuntimeError("pgvector session partition down")
        chunk = _make_chunk(ns_id)
        return [(chunk.id, 0.8, chunk)]

    retriever._vector_search_chunks = AsyncMock(side_effect=fake_vector_search_chunks)

    signal = TemporalSignal(
        is_temporal=True,
        category=TemporalCategory.RECENCY,
        confidence=0.9,
        source="test",
    )

    result = await retriever.retrieve(
        "what changed recently",
        ns_id,
        temporal_signal=signal,
    )

    assert isinstance(result, VectorCypherResult)
    # Recall survived: the surviving session channel + fallback produced chunks.
    assert len(result.chunks) > 0

    degradations = result.metadata.get("degradations") or []
    fanout_degs = [d for d in degradations if d.get("component") == "vectorcypher.session_fanout"]
    assert fanout_degs, f"expected a session_fanout degradation, got: {degradations!r}"
    deg = fanout_degs[0]
    # The prompt/issue asks the Degradation.reason to carry the exception type.
    assert deg["reason"] == "RuntimeError"
    assert deg["exception"] == "RuntimeError"
    assert "session-a" in (deg.get("detail") or "")


async def test_session_fanout_all_channels_succeed_records_no_degradation() -> None:
    """When every fan-out channel succeeds, no session_fanout degradation lands."""
    ns_id = uuid4()
    retriever = _make_session_fanout_retriever(ns_id)

    async def fake_vector_search_chunks(*, temporal_filter=None, **kwargs):
        chunk = _make_chunk(ns_id)
        return [(chunk.id, 0.8, chunk)]

    retriever._vector_search_chunks = AsyncMock(side_effect=fake_vector_search_chunks)

    signal = TemporalSignal(
        is_temporal=True,
        category=TemporalCategory.RECENCY,
        confidence=0.9,
        source="test",
    )

    result = await retriever.retrieve(
        "what changed recently",
        ns_id,
        temporal_signal=signal,
    )

    assert isinstance(result, VectorCypherResult)
    degradations = result.metadata.get("degradations") or []
    fanout_degs = [d for d in degradations if d.get("component") == "vectorcypher.session_fanout"]
    assert fanout_degs == [], f"expected no session_fanout degradation, got: {fanout_degs!r}"


async def test_session_fanout_cancellederror_is_reraised_not_degraded() -> None:
    """A cancelled per-session task re-raises CancelledError, never degraded.

    ``gather(return_exceptions=True)`` surfaces a cancelled child task's
    ``CancelledError`` as a value; it is a ``BaseException`` (not ``Exception``),
    so swallowing it would break structured-concurrency / shutdown semantics.
    The merge loop must propagate it rather than record a Degradation.
    """
    ns_id = uuid4()
    retriever = _make_session_fanout_retriever(ns_id)

    async def fake_vector_search_chunks(*, temporal_filter=None, **kwargs):
        channel = getattr(temporal_filter, "channel", None)
        if channel == "session-a":
            raise asyncio.CancelledError
        chunk = _make_chunk(ns_id)
        return [(chunk.id, 0.8, chunk)]

    retriever._vector_search_chunks = AsyncMock(side_effect=fake_vector_search_chunks)

    signal = TemporalSignal(
        is_temporal=True,
        category=TemporalCategory.RECENCY,
        confidence=0.9,
        source="test",
    )

    with pytest.raises(asyncio.CancelledError):
        await retriever.retrieve("what changed recently", ns_id, temporal_signal=signal)
