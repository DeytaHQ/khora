"""Unit tests for the epoch-invalidated recall result cache (#1469).

Two layers:

1. ``RecallResultCache`` in isolation - key sensitivity, TTL, LRU, and the
   epoch-invalidation contract (including the stale-set race guard).
2. ``VectorCypherEngine.recall`` end-to-end with a stubbed retriever - a repeat
   query hits the cache (identical result, retriever called once), a write to
   the namespace invalidates it (retriever runs again, no stale result), and a
   differing knob misses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.config.schema import KhoraConfig
from khora.core.models.recall import RecallResult
from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.engines.vectorcypher.recall_cache import RecallResultCache
from khora.engines.vectorcypher.retriever import VectorCypherResult
from khora.query.router import QueryComplexity, RoutingDecision


def _result(ns) -> RecallResult:
    return RecallResult(
        query="q",
        namespace_id=ns,
        documents=[],
        chunks=[],
        entities=[],
        relationships=[],
        communities=[],
        engine_info={"engine": "vectorcypher"},
    )


def _key_args(ns, **overrides):
    base = dict(
        query="what happened",
        namespace_id=ns,
        epoch=0,
        mode="hybrid",
        limit=10,
        min_similarity=0.0,
        graph_depth=None,
        hybrid_alpha=None,
        recency_bias=None,
        temporal_filter=None,
        filter_ast=None,
        config_fingerprint="cfg",
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# RecallResultCache in isolation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestRecallResultCache:
    def test_hit_returns_same_instance(self) -> None:
        ns = uuid4()
        cache = RecallResultCache()
        r = _result(ns)
        cache.set(result=r, **_key_args(ns))
        got = cache.get(**_key_args(ns))
        assert got is r

    def test_miss_on_differing_knob(self) -> None:
        ns = uuid4()
        cache = RecallResultCache()
        cache.set(result=_result(ns), **_key_args(ns, limit=10))
        assert cache.get(**_key_args(ns, limit=20)) is None
        assert cache.get(**_key_args(ns, min_similarity=0.3)) is None
        assert cache.get(**_key_args(ns, mode="vector")) is None
        assert cache.get(**_key_args(ns, query="different")) is None
        # ...but the original key still hits.
        assert cache.get(**_key_args(ns, limit=10)) is not None

    def test_miss_on_config_fingerprint_change(self) -> None:
        """A runtime config change (different fingerprint) is a distinct key."""
        ns = uuid4()
        cache = RecallResultCache()
        cache.set(result=_result(ns), **_key_args(ns, config_fingerprint="bm25=False"))
        assert cache.get(**_key_args(ns, config_fingerprint="bm25=False")) is not None
        assert cache.get(**_key_args(ns, config_fingerprint="bm25=True")) is None

    def test_epoch_bump_invalidates(self) -> None:
        ns = uuid4()
        cache = RecallResultCache()
        cache.set(result=_result(ns), **_key_args(ns, epoch=0))
        assert cache.get(**_key_args(ns, epoch=0)) is not None
        new_epoch = cache.bump_epoch(ns)
        # A get at the new epoch (what recall() would compute after the write) misses.
        assert cache.get(**_key_args(ns, epoch=new_epoch)) is None

    def test_set_refused_when_epoch_advanced_mid_recall(self) -> None:
        """Stale-set race guard: a write between get-miss and set must prevent
        the pre-write result from being stored under the new epoch."""
        ns = uuid4()
        cache = RecallResultCache()
        captured = cache.current_epoch(ns)  # 0
        assert cache.get(**_key_args(ns, epoch=captured)) is None  # miss
        # A concurrent write bumps the epoch...
        cache.bump_epoch(ns)
        # ...so storing the pre-write result under the captured (now stale) epoch
        # is refused.
        cache.set(result=_result(ns), **_key_args(ns, epoch=captured))
        # The post-write query recomputes rather than serving the stale result.
        assert cache.get(**_key_args(ns, epoch=cache.current_epoch(ns))) is None

    def test_ttl_expiry(self) -> None:
        ns = uuid4()
        cache = RecallResultCache(ttl_seconds=0)
        cache.set(result=_result(ns), **_key_args(ns))
        # ttl of 0 means every entry is immediately stale.
        assert cache.get(**_key_args(ns)) is None

    def test_lru_eviction(self) -> None:
        ns = uuid4()
        cache = RecallResultCache(max_size=2)
        cache.set(result=_result(ns), **_key_args(ns, query="a"))
        cache.set(result=_result(ns), **_key_args(ns, query="b"))
        cache.set(result=_result(ns), **_key_args(ns, query="c"))  # evicts "a"
        assert cache.get(**_key_args(ns, query="a")) is None
        assert cache.get(**_key_args(ns, query="b")) is not None
        assert cache.get(**_key_args(ns, query="c")) is not None

    def test_disabled_when_size_zero(self) -> None:
        ns = uuid4()
        cache = RecallResultCache(max_size=0)
        assert cache.enabled is False
        cache.set(result=_result(ns), **_key_args(ns))
        assert cache.get(**_key_args(ns)) is None

    def test_namespaces_isolated(self) -> None:
        ns_a, ns_b = uuid4(), uuid4()
        cache = RecallResultCache()
        cache.set(result=_result(ns_a), **_key_args(ns_a))
        cache.bump_epoch(ns_b)  # bumping B must not touch A
        assert cache.get(**_key_args(ns_a)) is not None

    def test_epochs_map_is_bounded(self) -> None:
        """The per-namespace epoch map is LRU-capped so it can't grow unbounded.

        An evicted namespace resets to epoch 0; since namespace_id is in the key
        digest, that can only cause an extra miss, never a stale hit.
        """
        cache = RecallResultCache(max_size=10)
        cache._epochs_cap = 3  # shrink for the test
        seen = [uuid4() for _ in range(5)]
        for ns in seen:
            cache.bump_epoch(ns)
        # Only the last 3 namespaces retain a bumped epoch; the first 2 evicted.
        assert cache.current_epoch(seen[0]) == 0
        assert cache.current_epoch(seen[1]) == 0
        assert cache.current_epoch(seen[-1]) == 1

    def test_bump_all_epochs(self) -> None:
        ns_a, ns_b = uuid4(), uuid4()
        cache = RecallResultCache()
        cache.set(result=_result(ns_a), **_key_args(ns_a))
        cache.set(result=_result(ns_b), **_key_args(ns_b))
        cache.bump_all_epochs()
        assert cache.get(**_key_args(ns_a, epoch=cache.current_epoch(ns_a))) is None
        assert cache.get(**_key_args(ns_b, epoch=cache.current_epoch(ns_b))) is None


# --------------------------------------------------------------------------- #
# VectorCypherEngine.recall end-to-end (stubbed retriever)
# --------------------------------------------------------------------------- #


def _routing() -> RoutingDecision:
    return RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=1.0,
        reasoning="test",
    )


def _engine(**query_overrides) -> tuple[VectorCypherEngine, MagicMock]:
    cfg = KhoraConfig()
    for k, v in query_overrides.items():
        setattr(cfg.query, k, v)

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=lambda **kwargs: VectorCypherResult(
            chunks=[], entities=[], routing_decision=_routing(), metadata={}
        )
    )

    engine = VectorCypherEngine.__new__(VectorCypherEngine)
    engine.__init__(cfg)  # runs the real __init__ to build the cache
    engine._embedder = embedder
    engine._retriever = retriever
    engine._storage = None
    engine._temporal_store = None
    engine._neo4j_driver = None
    engine._dual_nodes = None
    engine._router = None
    engine._connected = True
    return engine, retriever


@pytest.mark.unit
@pytest.mark.asyncio
class TestRecallCacheEndToEnd:
    async def test_repeat_query_hits_cache_identical_result(self) -> None:
        ns = uuid4()
        engine, retriever = _engine()

        first = await engine.recall("what happened", ns, limit=5)
        second = await engine.recall("what happened", ns, limit=5)

        assert second is first  # identical object served from cache
        retriever.retrieve.assert_awaited_once()  # retriever ran only once

    async def test_write_invalidates_cache_no_stale(self) -> None:
        ns = uuid4()
        engine, retriever = _engine()

        first = await engine.recall("what happened", ns, limit=5)
        # A write to the namespace bumps the epoch.
        engine.invalidate_recall_cache(ns)
        second = await engine.recall("what happened", ns, limit=5)

        assert second is not first  # recomputed, NOT the stale cached result
        assert retriever.retrieve.await_count == 2

    async def test_miss_on_differing_knob_end_to_end(self) -> None:
        ns = uuid4()
        engine, retriever = _engine()

        await engine.recall("what happened", ns, limit=5)
        await engine.recall("what happened", ns, limit=7)  # different limit -> miss

        assert retriever.retrieve.await_count == 2

    async def test_disabled_cache_never_hits(self) -> None:
        ns = uuid4()
        engine, retriever = _engine(enable_result_cache=False)

        await engine.recall("what happened", ns, limit=5)
        await engine.recall("what happened", ns, limit=5)

        assert retriever.retrieve.await_count == 2

    async def test_runtime_config_change_invalidates(self) -> None:
        """Toggling a retriever-config knob between two identical recalls misses
        the cache (the config fingerprint changed) - mirrors the integration
        filter-thread-through test that toggles enable_bm25_channel."""
        from khora.engines.vectorcypher.retriever import RetrieverConfig

        ns = uuid4()
        engine, retriever = _engine()
        retriever._config = RetrieverConfig(enable_bm25_channel=False)

        await engine.recall("what happened", ns, limit=5)
        # A caller flips a config knob at runtime, changing the result set.
        retriever._config.enable_bm25_channel = True
        await engine.recall("what happened", ns, limit=5)

        assert retriever.retrieve.await_count == 2
