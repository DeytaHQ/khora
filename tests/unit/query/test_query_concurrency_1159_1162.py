"""Regression tests for the two query-side concurrency bugs.

  * #1159 - ``HybridQueryEngine._cached_entity_search`` keyed its dedup cache
    by ``id(query_embedding)`` (a list).  Recycled CPython ids could make a
    later query await a different query's (and a different namespace's)
    cached task, returning silently-wrong, potentially cross-namespace
    results.  The cache key is now derived from ``namespace_id`` + the
    embedding *content*, entries are evicted when their task completes, and a
    cancelled/failed cached task is re-issued rather than re-awaited.

  * #1162 - ``CrossEncoderReranker.rerank`` called ``self._get_model()``
    (seconds of torch weight loading / HuggingFace download) directly on the
    event loop.  The load is now dispatched via ``asyncio.to_thread`` and
    single-flighted with an ``asyncio.Lock`` so concurrent first callers
    share one load.

Everything is mocked at the boundary - no live infrastructure required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.query.engine import HybridQueryEngine, QueryConfig
from khora.query.reranking import CrossEncoderReranker


def _make_engine() -> HybridQueryEngine:
    storage = MagicMock()
    storage.search_similar_entities = AsyncMock(return_value=[])
    config = QueryConfig(
        enable_query_understanding=False,
        enable_entity_linking=False,
        enable_reranking=False,
        enable_keyword_search=False,
    )
    return HybridQueryEngine(storage=storage, config=config)


# ---------------------------------------------------------------------------
# #1159 - entity-similarity dedup cache key
# ---------------------------------------------------------------------------


class TestEntitySearchCacheKey1159:
    @pytest.mark.asyncio
    async def test_distinct_embeddings_do_not_share_cache_entry(self) -> None:
        """Two different embeddings must hit storage separately, even if a
        previous list object's id() is recycled into the new one."""
        engine = _make_engine()
        ns = uuid4()
        a, b = uuid4(), uuid4()

        async def fake_search(namespace_id, embedding, *, limit, min_similarity):
            # Distinguish by the first element of the embedding.
            return [(a if embedding[0] < 0.5 else b, 0.9)]

        engine._storage.search_similar_entities = AsyncMock(side_effect=fake_search)

        out_low = await engine._cached_entity_search(ns, [0.1, 0.2], 5, 0.0)
        out_high = await engine._cached_entity_search(ns, [0.9, 0.8], 5, 0.0)

        assert out_low[0][0] == a
        assert out_high[0][0] == b
        assert engine._storage.search_similar_entities.await_count == 2

    @pytest.mark.asyncio
    async def test_same_embedding_different_namespace_does_not_alias(self) -> None:
        """Same embedding content but a different namespace must NOT reuse the
        first namespace's cached task (the original id()-keyed bug leaked
        results across namespaces)."""
        engine = _make_engine()
        ns_a, ns_b = uuid4(), uuid4()
        ent_a, ent_b = uuid4(), uuid4()
        embedding = [0.3, 0.4]

        async def fake_search(namespace_id, embedding, *, limit, min_similarity):
            return [(ent_a if namespace_id == ns_a else ent_b, 0.9)]

        engine._storage.search_similar_entities = AsyncMock(side_effect=fake_search)

        out_a = await engine._cached_entity_search(ns_a, embedding, 5, 0.0)
        out_b = await engine._cached_entity_search(ns_b, embedding, 5, 0.0)

        assert out_a[0][0] == ent_a
        assert out_b[0][0] == ent_b
        assert engine._storage.search_similar_entities.await_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_same_embedding_dedups_to_one_query(self) -> None:
        """Within one query, _vector_search and _graph_search hit this with the
        SAME embedding object concurrently - that must still collapse to a
        single storage call (the whole point of the cache)."""
        engine = _make_engine()
        ns = uuid4()
        eid = uuid4()
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_search(namespace_id, embedding, *, limit, min_similarity):
            started.set()
            await release.wait()
            return [(eid, 0.9)]

        engine._storage.search_similar_entities = AsyncMock(side_effect=slow_search)
        embedding = [0.5, 0.6]

        t1 = asyncio.create_task(engine._cached_entity_search(ns, embedding, 5, 0.0))
        await started.wait()
        t2 = asyncio.create_task(engine._cached_entity_search(ns, embedding, 5, 0.0))
        await asyncio.sleep(0)
        release.set()
        out1, out2 = await asyncio.gather(t1, t2)

        assert out1 == out2 == [(eid, 0.9)]
        assert engine._storage.search_similar_entities.await_count == 1

    @pytest.mark.asyncio
    async def test_cancelled_first_awaiter_does_not_break_later_call(self) -> None:
        """If the first awaiter is cancelled mid-flight, a later call with the
        same key must not trip over a permanently-cancelled cached task."""
        engine = _make_engine()
        ns = uuid4()
        eid = uuid4()
        first_started = asyncio.Event()
        calls = 0

        async def gated_search(namespace_id, embedding, *, limit, min_similarity):
            nonlocal calls
            calls += 1
            if calls == 1:
                # First in-flight task blocks until cancelled.
                first_started.set()
                await asyncio.sleep(3600)
            return [(eid, 0.9)]

        engine._storage.search_similar_entities = AsyncMock(side_effect=gated_search)
        embedding = [0.7, 0.1]

        first = asyncio.create_task(engine._cached_entity_search(ns, embedding, 5, 0.0))
        # Wait until the underlying storage call has actually started before
        # cancelling, so the first task is genuinely in-flight.
        await first_started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        # The second call must succeed by re-issuing rather than re-awaiting a
        # dead/cancelled cached task.
        out = await engine._cached_entity_search(ns, embedding, 5, 0.0)
        assert out == [(eid, 0.9)]


# ---------------------------------------------------------------------------
# #1162 - reranker model load must be off the event loop + single-flight
# ---------------------------------------------------------------------------


class TestRerankerModelLoadOffLoop1162:
    @pytest.mark.asyncio
    async def test_model_load_dispatched_via_to_thread(self, monkeypatch) -> None:
        """The synchronous model load must be offloaded with asyncio.to_thread,
        not invoked inline on the loop."""
        r = CrossEncoderReranker()

        to_thread_calls: list = []
        real_to_thread = asyncio.to_thread

        async def spy_to_thread(func, *args, **kwargs):
            to_thread_calls.append(func)
            return await real_to_thread(func, *args, **kwargs)

        # Make _get_model cheap + observable.
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.5]

        def fake_get_model():
            r._model = fake_model
            return fake_model

        monkeypatch.setattr(r, "_get_model", fake_get_model)
        monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

        from khora.query.reranking import RerankCandidate

        await r.rerank("q", [RerankCandidate(item="x", original_score=0.5, content="c")])

        # _get_model was sent through to_thread (and so was model.predict).
        assert fake_get_model in to_thread_calls

    @pytest.mark.asyncio
    async def test_concurrent_first_callers_trigger_single_load(self, monkeypatch) -> None:
        """Two concurrent first reranks must construct the model once."""
        r = CrossEncoderReranker()
        load_count = 0
        loading = asyncio.Event()

        def slow_get_model():
            nonlocal load_count
            load_count += 1
            fake = MagicMock()
            fake.predict.return_value = [0.5]
            r._model = fake
            return fake

        # Wrap to_thread so the first load yields control, letting the second
        # caller reach the guard while the load is still in flight.
        real_to_thread = asyncio.to_thread

        async def slow_to_thread(func, *args, **kwargs):
            if func is slow_get_model:
                loading.set()
                await asyncio.sleep(0.05)
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(r, "_get_model", slow_get_model)
        monkeypatch.setattr(asyncio, "to_thread", slow_to_thread)

        from khora.query.reranking import RerankCandidate

        cands = [RerankCandidate(item="x", original_score=0.5, content="c")]
        out1, out2 = await asyncio.gather(r.rerank("q", cands), r.rerank("q", cands))

        assert load_count == 1
        assert len(out1) == 1
        assert len(out2) == 1
