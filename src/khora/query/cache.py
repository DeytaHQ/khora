"""Query result cache with TTL for Khora Memory Lake.

Provides an in-memory LRU cache that avoids re-executing identical
queries within a configurable time-to-live window.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any
from uuid import UUID


class QueryCache:
    """LRU cache for query results with TTL."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum cached entries before eviction.
            ttl_seconds: Time-to-live for each entry in seconds.
        """
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(query: str, namespace_id: UUID, mode: str) -> str:
        return sha256(f"{query}:{namespace_id}:{mode}".encode()).hexdigest()

    async def get(self, query: str, namespace_id: UUID, mode: str) -> Any | None:
        """Look up a cached result.

        Returns None on miss or expiry.
        """
        key = self._make_key(query, namespace_id, mode)
        async with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                timestamp, result = entry
                if datetime.now() - timestamp < self._ttl:
                    self._hits += 1
                    self._record_cache_event(True, namespace_id)
                    return result
                del self._cache[key]
            self._misses += 1
            self._record_cache_event(False, namespace_id)
            return None

    @staticmethod
    def _record_cache_event(hit: bool, namespace_id: UUID) -> None:
        """Record a cache hit/miss to telemetry."""
        from khora.telemetry import get_collector

        get_collector().record_llm_call(
            operation="query_cache",
            cache_hit=hit,
            latency_ms=0.0,
            namespace_id=namespace_id,
            metadata={"cache_type": "query"},
        )

    async def set(self, query: str, namespace_id: UUID, mode: str, result: Any) -> None:
        """Store a result in the cache, evicting the oldest entry if full."""
        key = self._make_key(query, namespace_id, mode)
        async with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (datetime.now(), result)

    async def invalidate(self, namespace_id: UUID | None = None) -> int:
        """Remove entries, optionally filtered by namespace.

        Args:
            namespace_id: If provided, only remove entries whose key
                          was generated with this namespace. If None,
                          clears the entire cache.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            if namespace_id is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            # Namespace is hashed into the key, so we can't filter efficiently.
            # Clear everything as a safe fallback.
            count = len(self._cache)
            self._cache.clear()
            return count

    @property
    def stats(self) -> dict[str, int]:
        """Return hit/miss statistics."""
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
        }
