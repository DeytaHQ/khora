"""Epoch-invalidated recall result cache for the VectorCypher engine (#1469).

Agent and eval workloads re-ask identical queries constantly, but the default
``VectorCypherEngine.recall`` path recomputes everything per call (``QueryCache``
in ``khora.query.cache`` was only ever wired to the orphaned ``HybridQueryEngine``).

This module adds a bounded, TTL'd in-process cache at the ``recall`` boundary.

Correctness is the priority - a stale hit silently serves wrong results:

* **Conservative key.** The key digests EVERY input that changes the result set:
  namespace, normalized query text, mode, limit, min_similarity, graph_depth,
  hybrid_alpha, recency_bias, the temporal filter, and the recall-filter AST.
  Config knobs (fusion weights, reranking, HyDE, ...) are baked into the engine
  at construction, so the cache instance lifetime already scopes them - a fresh
  engine gets a fresh cache.
* **Epoch invalidation on ANY write.** Each namespace carries a monotonically
  increasing write-epoch, folded into the key. Every mutation to a namespace
  (remember / remember_batch / forget / dream-apply) bumps its epoch, so all
  prior entries for that namespace become unreachable at once - no per-key
  bookkeeping, no chance of serving a pre-write result.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from khora.core.models.recall import RecallResult
    from khora.core.temporal import ChunkTemporalFilter
    from khora.filter.ast import FilterNode


class RecallResultCache:
    """Bounded, TTL'd, epoch-invalidated cache of ``RecallResult`` objects."""

    def __init__(self, *, max_size: int = 1000, ttl_seconds: int = 300) -> None:
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        # Insertion-ordered so the oldest entry is evicted first (LRU on write;
        # a hit also moves the entry to the end so it is not evicted early).
        self._store: OrderedDict[str, tuple[datetime, RecallResult]] = OrderedDict()
        # Per-namespace write-epoch. A missing namespace is epoch 0.
        self._epochs: dict[UUID, int] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self._max_size > 0

    def bump_epoch(self, namespace_id: UUID) -> int:
        """Invalidate every cached result for ``namespace_id`` by bumping its epoch.

        Called on ANY write to the namespace. Prior entries keyed on the old
        epoch become unreachable (and age out via the LRU / TTL). Returns the new
        epoch. Cheap and always safe to call, even when the cache is disabled.
        """
        with self._lock:
            new_epoch = self._epochs.get(namespace_id, 0) + 1
            self._epochs[namespace_id] = new_epoch
            return new_epoch

    def bump_all_epochs(self) -> None:
        """Invalidate every namespace's cached results at once.

        For rare, broad mutations (dream-undo) where the affected namespace is
        not readily available at the call site. Bumps the epoch of every
        namespace the cache has ever seen and drops the store, so no prior entry
        can be served and any in-flight recall's ``set`` (guarded on its captured
        epoch) is refused.
        """
        with self._lock:
            for ns in list(self._epochs):
                self._epochs[ns] += 1
            self._store.clear()

    def current_epoch(self, namespace_id: UUID) -> int:
        """Snapshot the namespace's write-epoch at recall start.

        The caller captures this once and passes the SAME value to both ``get``
        and ``set`` for one recall. If a concurrent write bumps the epoch between
        the get-miss and the set, ``set`` sees the captured epoch is stale and
        refuses to store - so a result computed against pre-write state is never
        stored under (and later served for) the post-write epoch.
        """
        with self._lock:
            return self._epochs.get(namespace_id, 0)

    @staticmethod
    def _digest(
        *,
        query: str,
        namespace_id: UUID,
        epoch: int,
        mode: str,
        limit: int,
        min_similarity: float,
        graph_depth: int | None,
        hybrid_alpha: float | None,
        recency_bias: float | None,
        temporal_filter: ChunkTemporalFilter | None,
        filter_ast: FilterNode | None,
    ) -> str:
        # ``repr`` of the frozen filter AST and the small temporal dataclass is a
        # deterministic, order-stable string; ``asdict`` normalizes the temporal
        # filter's nested fields. Everything else is a scalar. The epoch is part
        # of the key, so a bump makes prior entries unreachable.
        tf = asdict(temporal_filter) if temporal_filter is not None else None
        parts = (
            query.strip().lower(),
            str(namespace_id),
            epoch,
            mode,
            limit,
            round(float(min_similarity), 6),
            graph_depth,
            None if hybrid_alpha is None else round(float(hybrid_alpha), 6),
            None if recency_bias is None else round(float(recency_bias), 6),
            repr(tf),
            repr(filter_ast),
        )
        return sha256(repr(parts).encode()).hexdigest()

    def get(
        self,
        *,
        query: str,
        namespace_id: UUID,
        epoch: int,
        mode: str,
        limit: int,
        min_similarity: float,
        graph_depth: int | None,
        hybrid_alpha: float | None,
        recency_bias: float | None,
        temporal_filter: ChunkTemporalFilter | None,
        filter_ast: FilterNode | None,
    ) -> RecallResult | None:
        """Return the cached result for these exact inputs, or ``None`` on miss/expiry."""
        if not self.enabled:
            return None
        with self._lock:
            key = self._digest(
                query=query,
                namespace_id=namespace_id,
                epoch=epoch,
                mode=mode,
                limit=limit,
                min_similarity=min_similarity,
                graph_depth=graph_depth,
                hybrid_alpha=hybrid_alpha,
                recency_bias=recency_bias,
                temporal_filter=temporal_filter,
                filter_ast=filter_ast,
            )
            entry = self._store.get(key)
            if entry is not None:
                timestamp, result = entry
                if datetime.now() - timestamp < self._ttl:
                    self._store.move_to_end(key)
                    self._hits += 1
                    return result
                # Expired.
                del self._store[key]
            self._misses += 1
            return None

    def set(
        self,
        *,
        query: str,
        namespace_id: UUID,
        epoch: int,
        mode: str,
        limit: int,
        min_similarity: float,
        graph_depth: int | None,
        hybrid_alpha: float | None,
        recency_bias: float | None,
        temporal_filter: ChunkTemporalFilter | None,
        filter_ast: FilterNode | None,
        result: RecallResult,
    ) -> None:
        """Store ``result`` under the digest of these inputs (LRU-evicting if full).

        ``epoch`` is the value captured at recall start. If a concurrent write
        bumped the namespace epoch since then, the result reflects pre-write
        state, so we refuse to store it - a later query reads the new epoch and
        recomputes rather than hitting this stale result.
        """
        if not self.enabled:
            return
        with self._lock:
            if self._epochs.get(namespace_id, 0) != epoch:
                return
            key = self._digest(
                query=query,
                namespace_id=namespace_id,
                epoch=epoch,
                mode=mode,
                limit=limit,
                min_similarity=min_similarity,
                graph_depth=graph_depth,
                hybrid_alpha=hybrid_alpha,
                recency_bias=recency_bias,
                temporal_filter=temporal_filter,
                filter_ast=filter_ast,
            )
            self._store[key] = (datetime.now(), result)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Drop every cached entry (epochs are retained; used on disconnect/tests)."""
        with self._lock:
            self._store.clear()

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"size": len(self._store), "hits": self._hits, "misses": self._misses}
