"""Unit tests for the t0 query-embed overlap on VectorCypherEngine.recall (#1469).

The query embedding depends only on the query text, so recall() launches it as
an asyncio task before temporal-detect and hands the in-flight awaitable to the
retriever. These tests pin the correctness contract:

- the retriever receives a ``query_embedding_task`` whose awaited value is the
  embedder's output,
- exactly ONE embed happens per recall (no double-embed),
- the embed is genuinely in flight before the retriever awaits it (overlap).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.config.schema import KhoraConfig
from khora.engines.vectorcypher.engine import VectorCypherConfig, VectorCypherEngine
from khora.engines.vectorcypher.retriever import VectorCypherResult
from khora.query.router import QueryComplexity, RoutingDecision


def _routing() -> RoutingDecision:
    return RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=1.0,
        reasoning="test",
    )


def _make_engine(embedder: MagicMock, retriever: MagicMock) -> VectorCypherEngine:
    engine = VectorCypherEngine.__new__(VectorCypherEngine)
    engine._config = KhoraConfig()
    engine._vc_config = VectorCypherConfig()
    engine._embedder = embedder
    engine._retriever = retriever
    engine._storage = None
    engine._temporal_store = None
    engine._neo4j_driver = None
    engine._dual_nodes = None
    engine._router = None
    engine._connected = True
    # No entities -> _project_communities short-circuits without touching storage.
    return engine


@pytest.mark.unit
@pytest.mark.asyncio
class TestRecallEmbedOverlap:
    async def test_retriever_receives_embed_task_and_single_embed(self) -> None:
        """recall() hands a task to the retriever; the embedder is invoked once."""
        namespace_id = uuid4()
        embedding = [0.25] * 8

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=embedding)

        captured: dict[str, object] = {}

        async def fake_retrieve(*, query_embedding_task, **kwargs):
            # The engine must pass an awaitable that resolves to the embedding.
            captured["task"] = query_embedding_task
            captured["awaited"] = await query_embedding_task
            return VectorCypherResult(chunks=[], entities=[], routing_decision=_routing(), metadata={})

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=fake_retrieve)

        engine = _make_engine(embedder, retriever)

        result = await engine.recall("what happened", namespace_id, limit=5)

        assert captured["task"] is not None
        assert captured["awaited"] == embedding
        # Exactly one embed for the whole recall: the retriever awaited the task
        # rather than embedding inline (no double-embed).
        assert embedder.embed.await_count == 1
        assert result.namespace_id == namespace_id

    async def test_embed_is_in_flight_before_retriever_awaits(self) -> None:
        """The embed task is created and running before retrieve() awaits it.

        A gated embedder lets us assert the engine did NOT block on the embed
        before entering the retriever: the retrieve() coroutine starts while the
        embed is still pending, proving the work overlaps temporal-detect/route.
        """
        namespace_id = uuid4()
        embedding = [0.5] * 8
        release = asyncio.Event()

        async def gated_embed(_text: str) -> list[float]:
            await release.wait()
            return embedding

        embedder = MagicMock()
        embedder.embed = AsyncMock(side_effect=gated_embed)

        state: dict[str, object] = {}

        async def fake_retrieve(*, query_embedding_task, **kwargs):
            # We are inside retrieve() and the embed has not been released yet,
            # so the task must still be pending -> it was launched at t0, not
            # awaited serially before the retriever ran.
            state["pending_on_entry"] = not query_embedding_task.done()
            release.set()
            state["awaited"] = await query_embedding_task
            return VectorCypherResult(chunks=[], entities=[], routing_decision=_routing(), metadata={})

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(side_effect=fake_retrieve)

        engine = _make_engine(embedder, retriever)

        await engine.recall("state of the project", namespace_id, limit=5)

        assert state["pending_on_entry"] is True
        assert state["awaited"] == embedding
        assert embedder.embed.await_count == 1

    async def test_embed_task_not_orphaned_when_recall_raises(self, monkeypatch) -> None:
        """If temporal-detect raises before the retriever awaits, the embed task
        result is consumed by the done-callback (no orphaned-task warning)."""
        namespace_id = uuid4()
        embedding = [0.1] * 8

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=embedding)
        retriever = MagicMock()
        retriever.retrieve = AsyncMock()

        engine = _make_engine(embedder, retriever)

        boom = RuntimeError("temporal detect blew up")

        class _ExplodingDetector:
            def __init__(self, *a, **k) -> None:
                pass

            async def detect_async(self, *a, **k):
                raise boom

        monkeypatch.setattr(
            "khora.engines.vectorcypher.engine.TemporalDetector",
            _ExplodingDetector,
        )

        with pytest.raises(RuntimeError, match="temporal detect blew up"):
            await engine.recall("anything", namespace_id, limit=5)

        # Let the loop run the task's done-callback; it must consume the result
        # without raising or logging an "exception never retrieved" warning.
        await asyncio.sleep(0)
        retriever.retrieve.assert_not_awaited()
