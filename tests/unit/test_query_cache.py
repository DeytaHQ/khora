"""Unit tests for query/cache.py — QueryCache with TTL."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest

from khora.query.cache import QueryCache


class TestQueryCache:
    """Tests for QueryCache."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        cache = QueryCache()
        assert cache._max_size == 1000
        assert cache._ttl == timedelta(seconds=300)
        assert cache._hits == 0
        assert cache._misses == 0

    def test_init_custom(self) -> None:
        """Test custom initialization parameters."""
        cache = QueryCache(max_size=50, ttl_seconds=60)
        assert cache._max_size == 50
        assert cache._ttl == timedelta(seconds=60)

    def test_make_key_deterministic(self) -> None:
        """Same inputs produce same key."""
        ns = uuid4()
        key1 = QueryCache._make_key("hello", ns, "hybrid")
        key2 = QueryCache._make_key("hello", ns, "hybrid")
        assert key1 == key2

    def test_make_key_different_inputs(self) -> None:
        """Different inputs produce different keys."""
        ns = uuid4()
        key1 = QueryCache._make_key("hello", ns, "hybrid")
        key2 = QueryCache._make_key("world", ns, "hybrid")
        assert key1 != key2

    def test_make_key_different_extra(self) -> None:
        """Different per-call digests (temporal filter / config) produce different keys."""
        ns = uuid4()
        key1 = QueryCache._make_key("hello", ns, "hybrid", "filter-a")
        key2 = QueryCache._make_key("hello", ns, "hybrid", "filter-b")
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_extra_digest_isolates_entries(self) -> None:
        """Same query/namespace/mode but different extra digests don't collide.

        Reproduces #1129: a temporal-filtered (or limit-differing) recall must
        not serve the cached result of the same query text with a different
        filter/config.
        """
        cache = QueryCache()
        ns = uuid4()
        with patch.object(QueryCache, "_record_cache_event"):
            await cache.set("q", ns, "hybrid", "unfiltered", extra="")
            # Same text/ns/mode but a different temporal filter digest.
            hit = await cache.get("q", ns, "hybrid", extra="last_days_7")
            assert hit is None  # must not serve the unfiltered result
            await cache.set("q", ns, "hybrid", "filtered", extra="last_days_7")
            # Each digest resolves to its own entry.
            assert await cache.get("q", ns, "hybrid", extra="") == "unfiltered"
            assert await cache.get("q", ns, "hybrid", extra="last_days_7") == "filtered"

    @pytest.mark.asyncio
    async def test_get_miss(self) -> None:
        """Cache miss returns None."""
        cache = QueryCache()
        with patch.object(QueryCache, "_record_cache_event"):
            result = await cache.get("query", uuid4(), "hybrid")
        assert result is None
        assert cache._misses == 1

    @pytest.mark.asyncio
    async def test_set_and_get_roundtrip(self) -> None:
        """Set then get returns the cached value."""
        cache = QueryCache()
        ns = uuid4()
        with patch.object(QueryCache, "_record_cache_event"):
            await cache.set("query", ns, "hybrid", {"data": [1, 2, 3]})
            result = await cache.get("query", ns, "hybrid")
        assert result == {"data": [1, 2, 3]}
        assert cache._hits == 1

    @pytest.mark.asyncio
    async def test_get_expired(self) -> None:
        """Expired entries return None."""
        cache = QueryCache(ttl_seconds=1)
        ns = uuid4()
        # Manually insert an expired entry
        key = cache._make_key("query", ns, "hybrid")
        cache._cache[key] = (datetime.now() - timedelta(seconds=10), "old_result")

        with patch.object(QueryCache, "_record_cache_event"):
            result = await cache.get("query", ns, "hybrid")
        assert result is None
        assert key not in cache._cache  # Expired entry removed

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """Oldest entry is evicted when max_size is reached."""
        cache = QueryCache(max_size=2)
        ns = uuid4()

        with patch.object(QueryCache, "_record_cache_event"):
            await cache.set("q1", ns, "hybrid", "r1")
            await cache.set("q2", ns, "hybrid", "r2")
            # This should evict q1
            await cache.set("q3", ns, "hybrid", "r3")

        assert len(cache._cache) == 2
        # q1 should be evicted
        with patch.object(QueryCache, "_record_cache_event"):
            result = await cache.get("q1", ns, "hybrid")
        assert result is None

    def test_stats(self) -> None:
        """Stats return correct counters."""
        cache = QueryCache()
        cache._hits = 10
        cache._misses = 5
        cache._cache["key"] = (datetime.now(), "val")
        stats = cache.stats
        assert stats == {"size": 1, "hits": 10, "misses": 5}

    @pytest.mark.asyncio
    async def test_invalidate_all(self) -> None:
        """Invalidate with no namespace clears entire cache."""
        cache = QueryCache()
        ns = uuid4()
        with patch.object(QueryCache, "_record_cache_event"):
            await cache.set("q1", ns, "hybrid", "r1")
            await cache.set("q2", ns, "hybrid", "r2")
        count = await cache.invalidate()
        assert count == 2
        assert len(cache._cache) == 0

    @pytest.mark.asyncio
    async def test_invalidate_namespace(self) -> None:
        """Invalidate with namespace clears cache (safe fallback)."""
        cache = QueryCache()
        ns = uuid4()
        with patch.object(QueryCache, "_record_cache_event"):
            await cache.set("q1", ns, "hybrid", "r1")
        count = await cache.invalidate(namespace_id=ns)
        assert count == 1
        assert len(cache._cache) == 0

    def test_record_cache_event(self) -> None:
        """Telemetry recording doesn't raise."""
        with patch("khora.telemetry.get_collector") as mock_collector:
            mock_collector.return_value.record_llm_call = lambda **kwargs: None
            QueryCache._record_cache_event(True, uuid4())
            QueryCache._record_cache_event(False, uuid4())
