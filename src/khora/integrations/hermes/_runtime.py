"""Threading + cache + queue substrate for ``KhoraMemoryProvider``.

This module is the per-session plumbing the Hermes ``MemoryProvider``
sits on top of. It serializes write-side ``remember`` / ``remember_batch``
calls through a single-worker ``ThreadPoolExecutor`` (so ingestion order
matches conversation order), and it keeps a small TTL-bounded prefetch
cache for ``recall`` so the agent's "prefetch on user turn" pattern
doesn't trigger N concurrent recalls for the same question.

All async work routes through :func:`khora.integrations._sync.run_sync`
— this module does NOT own an event loop. That's the architect's
correction baked into the design: the bridge loop is process-wide and
shared with every other adapter; ours is only the FIFO submission
discipline plus the cache.

Private module (`_runtime`): this is implementation detail of the
Hermes adapter, not part of ``khora.integrations``' public API.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.integrations._sync import run_sync
from khora.telemetry import bounded_text_hash, metric_counter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Coroutine

    from khora.core.models.document import Document
    from khora.core.models.recall import RecallResult
    from khora.khora import Khora


# ---------------------------------------------------------------------------
# OTel metric instruments. Created once at module import; safe to call
# even when no MeterProvider is installed (OTel API returns no-ops).
# No namespace_id label on any of these — cardinality rule.
# ---------------------------------------------------------------------------

_REMEMBER_SUCCESS = metric_counter(
    "khora.hermes.remember.success_total",
    unit="1",
    description="remember/remember_batch calls that completed without raising.",
)
_REMEMBER_FAILED = metric_counter(
    "khora.hermes.remember.failed_total",
    unit="1",
    description="remember/remember_batch calls that raised.",
)
_QUEUE_SHED = metric_counter(
    "khora.hermes.queue.shed_total",
    unit="1",
    description="Pending writes dropped because the per-session queue was full.",
)


_MAX_ERROR_STR_LEN = 200
_DEFAULT_QUEUE_MAX = 256
_DEFAULT_PREFETCH_TTL_S = 30.0
_LOG_SINK_WARN_INTERVAL_S = 10.0


# Module-level flag so the loguru-sink warning fires at most once per
# process, no matter how many runtimes are constructed.
_loguru_sink_warning_emitted = False


class _CacheEntry:
    """One slot in the recall prefetch cache.

    Holds either a pending :class:`concurrent.futures.Future` (recall is
    still running) or a materialised :class:`RecallResult` (recall
    finished and the readers can use it directly). The ``inserted_at``
    monotonic timestamp drives TTL eviction; the entry is treated as a
    miss once ``time.monotonic() - inserted_at > ttl_s``.
    """

    __slots__ = ("future", "inserted_at", "result", "ttl_s")

    def __init__(
        self,
        *,
        future: concurrent.futures.Future[Any] | None,
        result: RecallResult | None,
        ttl_s: float,
    ) -> None:
        self.future = future
        self.result = result
        self.inserted_at = time.monotonic()
        self.ttl_s = ttl_s

    def is_expired(self) -> bool:
        return (time.monotonic() - self.inserted_at) > self.ttl_s


class _KhoraRuntime:
    """Per-session FIFO executor + prefetch cache + error bookkeeping.

    One instance per ``KhoraMemoryProvider`` (and in normal Hermes use,
    one ``KhoraMemoryProvider`` per session). The instance owns a
    ``ThreadPoolExecutor(max_workers=1)`` so writes serialise in
    submission order, plus a TTL-bounded prefetch cache keyed on
    ``(namespace_id, session_id, bounded_text_hash(query))``.
    """

    def __init__(
        self,
        *,
        queue_max_size: int = _DEFAULT_QUEUE_MAX,
        prefetch_cache_ttl_s: float = _DEFAULT_PREFETCH_TTL_S,
    ) -> None:
        if queue_max_size < 1:
            raise ValueError(f"queue_max_size must be >= 1, got {queue_max_size}")
        if prefetch_cache_ttl_s <= 0:
            raise ValueError(f"prefetch_cache_ttl_s must be > 0, got {prefetch_cache_ttl_s}")

        self._queue_max_size = queue_max_size
        self._prefetch_cache_ttl_s = prefetch_cache_ttl_s

        # Single worker = strict FIFO. Submission ordering on the
        # executor's internal queue matches conversation turn order.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="khora-hermes-write",
        )

        # Track pending+in-flight Futures so we can shed the oldest when
        # the queue is full. ``OrderedDict``-style FIFO via deque; mutated
        # under ``_pending_lock``.
        self._pending_lock = threading.Lock()
        self._pending: deque[concurrent.futures.Future[Any]] = deque()

        # Prefetch cache. Keyed by (namespace_id, session_id,
        # bounded_text_hash(query)). Readers and writers both take
        # ``_cache_lock``. Cache value is _CacheEntry; entry.future or
        # entry.result tells the reader which to wait on.
        self._cache_lock = threading.Lock()
        self._cache: dict[tuple[UUID, str, str], _CacheEntry] = {}

        # Failure bookkeeping for the provider's status reporting.
        self._counter_lock = threading.Lock()
        self._success_count = 0
        self._failure_count = 0
        # Bounded ring of last N exception strings. Each entry already
        # truncated to ``_MAX_ERROR_STR_LEN`` chars at insertion time.
        self._error_ring: deque[str] = deque(maxlen=16)

        # Shed-oldest log rate-limit. ``time.monotonic()`` timestamp of
        # the last WARN we emitted; bumped under ``_shed_log_lock``.
        self._shed_log_lock = threading.Lock()
        self._last_shed_warn_at: float = 0.0

        # Shutdown guard so a second shutdown() is a no-op.
        self._shutdown_lock = threading.Lock()
        self._shutdown_called = False

        self._warn_if_loguru_sink_is_sync()

    # ------------------------------------------------------------------
    # Public API consumed by KhoraMemoryProvider
    # ------------------------------------------------------------------

    def enqueue_remember(
        self,
        kb: Khora,
        namespace_id: UUID,
        document: Document,
    ) -> None:
        """Fire-and-forget write of one document via ``kb.remember``."""

        def _coro_factory() -> Coroutine[Any, Any, Any]:
            return kb.remember(
                document.content,
                namespace=namespace_id,
                title=document.title or "",
                source=document.source or "",
                source_type=document.source_type or "library",
                source_name=document.source_name,
                source_url=document.source_url,
                source_timestamp=document.source_timestamp,
                metadata=document.metadata or None,
                external_id=document.external_id,
                session_id=document.session_id,
                entity_types=[],
                relationship_types=[],
            )

        self._submit_write(_coro_factory, op="remember")

    def enqueue_remember_batch(
        self,
        kb: Khora,
        namespace_id: UUID,
        documents: list[Document],
    ) -> None:
        """Fire-and-forget batched write via ``kb.remember_batch``."""
        if not documents:
            return

        # Materialise the dict payload upfront. The worker thread only
        # has to call run_sync(); no Khora-shape construction inside the
        # critical section.
        payload: list[dict[str, Any]] = [
            {
                "content": doc.content,
                "title": doc.title or "",
                "source": doc.source or "",
                "source_type": doc.source_type or "library",
                "source_name": doc.source_name,
                "source_url": doc.source_url,
                "source_timestamp": doc.source_timestamp,
                "metadata": doc.metadata or None,
                "external_id": doc.external_id,
            }
            for doc in documents
        ]

        def _coro_factory() -> Coroutine[Any, Any, Any]:
            return kb.remember_batch(
                payload,
                namespace=namespace_id,
                entity_types=[],
                relationship_types=[],
            )

        self._submit_write(_coro_factory, op="remember_batch")

    def enqueue_recall(
        self,
        kb: Khora,
        namespace_id: UUID,
        session_id: str,
        query: str,
    ) -> None:
        """Fire-and-forget recall — stashes ``Future`` in the cache.

        TOCTOU fix: the Future is inserted into the cache BEFORE the
        worker submits the coroutine. A concurrent ``recall_sync`` for
        the same key will find this Future and wait on it instead of
        racing a second recall.
        """
        key = self._cache_key(namespace_id, session_id, query)

        def _coro_factory() -> Coroutine[Any, Any, Any]:
            return kb.recall(query, namespace=namespace_id)

        with self._cache_lock:
            existing = self._cache.get(key)
            if existing is not None and not existing.is_expired():
                # Another in-flight prefetch covers this exact query. Don't
                # double-submit. (Cheap idempotency for Hermes's "prefetch
                # on every turn" pattern.)
                return
            future = self._submit_write(_coro_factory, op="recall")
            entry = _CacheEntry(future=future, result=None, ttl_s=self._prefetch_cache_ttl_s)
            self._cache[key] = entry

            # Promote the Future's result onto the entry once it resolves
            # so subsequent reads return without waiting on the Future
            # again. Done callback runs on the worker thread; uses the
            # lock to publish.
            def _on_done(fut: concurrent.futures.Future[Any], entry_ref: _CacheEntry = entry) -> None:
                if fut.cancelled():
                    return
                if fut.exception() is not None:
                    # Failed recall: drop the cache slot so the next
                    # caller retries instead of returning a stale Future
                    # that will keep raising.
                    with self._cache_lock:
                        # Only drop if it's still ours — could've been
                        # replaced by a fresh enqueue_recall mid-flight.
                        if self._cache.get(key) is entry_ref:
                            self._cache.pop(key, None)
                    return
                with self._cache_lock:
                    entry_ref.result = fut.result()
                    entry_ref.future = None

            future.add_done_callback(_on_done)

    def recall_sync(
        self,
        kb: Khora,
        namespace_id: UUID,
        session_id: str,
        query: str,
        *,
        timeout: float,
    ) -> RecallResult | None:
        """Block until a recall result is available, or ``None`` on timeout.

        Resolution order:
          - cache hit (fresh RecallResult): return immediately.
          - cache hit (in-flight Future): wait on the Future with ``timeout``.
          - miss: enqueue a fresh recall, then wait on its Future.

        Returns ``None`` on timeout. The Future is **not** cancelled on
        timeout — other readers (later prefetches, post-turn reflection)
        may still want the result, and cancellation would force a redo.
        """
        key = self._cache_key(namespace_id, session_id, query)
        future: concurrent.futures.Future[Any] | None = None

        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is not None and entry.is_expired():
                # Expired — drop it and treat as miss.
                self._cache.pop(key, None)
                entry = None
            if entry is not None:
                if entry.result is not None:
                    return entry.result
                future = entry.future

        if future is None:
            # Miss. Enqueue a fresh recall; that path also inserts the
            # cache entry under the lock, so we can read it back.
            self.enqueue_recall(kb, namespace_id, session_id, query)
            with self._cache_lock:
                entry = self._cache.get(key)
                if entry is None:
                    # enqueue_recall short-circuited (unexpected — only
                    # happens on shed) — caller should retry later.
                    return None
                if entry.result is not None:
                    return entry.result
                future = entry.future

        if future is None:
            return None

        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None
        except Exception as exc:  # noqa: BLE001 - surface as None; counters already updated
            logger.warning("hermes.recall_sync future raised: {}", exc)
            return None

    def dispatch_sync(
        self,
        coro_fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Synchronously run an async function via ``run_sync``.

        Used by tool-call dispatch (where blocking the caller is fine —
        tool handlers are themselves the thing the agent is waiting on).
        Distinct from ``enqueue_*`` which is fire-and-forget and FIFO.
        """
        return run_sync(coro_fn(*args, **kwargs))

    def drain(self, *, timeout: float) -> int:
        """Wait up to ``timeout`` seconds for the queue to clear.

        Returns the count of items still pending after the timeout (lost
        on shutdown). Useful for the provider's ``cleanup`` path.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._pending_lock:
                snapshot = [f for f in self._pending if not f.done()]
            if not snapshot:
                return 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return len(snapshot)
            # Wait on the oldest in-flight Future, then loop. We poll
            # rather than wait(ALL) because the deque can grow while
            # we're blocked.
            try:
                snapshot[0].result(timeout=min(remaining, 0.5))
            except concurrent.futures.TimeoutError:
                continue
            except Exception:  # noqa: BLE001, S112 - failure is already counted in _record_failure
                continue

    def failure_rate_pct(self) -> float:
        """Percentage of remember/remember_batch tasks that raised.

        Returns ``0.0`` when no calls have been submitted yet so callers
        don't have to special-case the empty bucket.
        """
        with self._counter_lock:
            total = self._success_count + self._failure_count
            if total == 0:
                return 0.0
            return (self._failure_count / total) * 100.0

    def last_errors(self, n: int = 5) -> list[str]:
        """Return up to the most recent ``n`` exception strings.

        Thread-safe snapshot — the returned list is a fresh copy so the
        caller can iterate without holding the lock.
        """
        if n <= 0:
            return []
        with self._counter_lock:
            # Ring is bounded at 16; slice off the tail.
            return list(self._error_ring)[-n:]

    def shutdown(self) -> None:
        """Idempotent. Drain briefly, then shut the executor."""
        with self._shutdown_lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True
        # Brief drain so writes that are already running can finish; we
        # don't want to leave them mid-coroutine inside run_sync's
        # bridge loop.
        self.drain(timeout=2.0)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        namespace_id: UUID,
        session_id: str,
        query: str,
    ) -> tuple[UUID, str, str]:
        """Build the prefetch-cache key.

        The query is hashed via ``bounded_text_hash`` so we never hold
        raw user text in a long-lived dict — matches the cardinality
        rule already enforced on span attributes.
        """
        return (namespace_id, session_id, bounded_text_hash(query))

    def _submit_write(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        *,
        op: str,
    ) -> concurrent.futures.Future[Any]:
        """Submit a coroutine factory to the executor with shed-oldest.

        Holds ``_pending_lock`` for the duration so the shed-then-append
        sequence stays atomic relative to other submitters. Never
        acquires ``_cache_lock`` here — callers that want the cache
        coherent (``enqueue_recall``) acquire ``_cache_lock`` first and
        then call into us; the locks form a strict outer→inner order
        (cache → pending) which is never inverted, so there's no
        deadlock surface even though both paths can touch both locks.
        """
        with self._pending_lock:
            # Drop completed futures so the cap reflects only pending +
            # in-flight work, not historical wins.
            while self._pending and self._pending[0].done():
                self._pending.popleft()

            # Shed-oldest if at the cap.
            if len(self._pending) >= self._queue_max_size:
                shed = self._pending.popleft()
                shed.cancel()
                _QUEUE_SHED.add(1, attributes={"op": op})
                self._maybe_log_shed()

            future = self._executor.submit(self._run_task, coro_factory, op)
            self._pending.append(future)
            return future

    def _run_task(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        op: str,
    ) -> Any:
        """Worker-thread entry point.

        Builds the coroutine inside the worker so a queued task does not
        hold an unstarted coroutine across submission boundaries (which
        emits a 'coroutine was never awaited' warning on cancel). Then
        delegates to the process-wide sync bridge.
        """
        try:
            coro = coro_factory()
            result = run_sync(coro)
        except BaseException as exc:  # noqa: BLE001 - count and surface
            self._record_failure(exc, op=op)
            raise
        else:
            self._record_success(op=op)
            return result

    def _record_success(self, *, op: str) -> None:
        if op == "recall":
            # recall is not part of the remember-success counter; we only
            # track ingest failure rate via the provider status surface.
            return
        with self._counter_lock:
            self._success_count += 1
        _REMEMBER_SUCCESS.add(1, attributes={"op": op})

    def _record_failure(self, exc: BaseException, *, op: str) -> None:
        truncated = str(exc)[:_MAX_ERROR_STR_LEN]
        # recall failures don't move the remember failure-rate gauge,
        # but they still belong in the error ring so operators see the
        # last N problems regardless of source.
        with self._counter_lock:
            if op != "recall":
                self._failure_count += 1
            self._error_ring.append(f"[{op}] {truncated}")
        if op != "recall":
            _REMEMBER_FAILED.add(1, attributes={"op": op})
        logger.warning("hermes.{} task failed: {}", op, truncated)

    def _maybe_log_shed(self) -> None:
        """WARN about queue shedding, rate-limited to once per 10s."""
        now = time.monotonic()
        with self._shed_log_lock:
            if (now - self._last_shed_warn_at) < _LOG_SINK_WARN_INTERVAL_S:
                return
            self._last_shed_warn_at = now
        logger.warning(
            "khora hermes runtime sheds oldest write — queue at cap (max={}). "
            "Sustained shedding indicates ingest throughput cannot keep up; "
            "consider larger queue_max_size or fewer per-turn writes.",
            self._queue_max_size,
        )

    def _warn_if_loguru_sink_is_sync(self) -> None:
        """One-time WARN if no loguru sink has ``enqueue=True``.

        The runtime's worker thread calls ``logger.warning`` from inside
        ``run_sync`` — if every sink is sync (``enqueue=False``), each
        log call blocks the bridge's event loop on stderr write. Surface
        the misconfiguration once on init.
        """
        global _loguru_sink_warning_emitted
        if _loguru_sink_warning_emitted:
            return
        try:
            handlers = logger._core.handlers  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - loguru API drift
            return
        if not handlers:
            return
        any_enqueued = False
        for handler in handlers.values():
            if getattr(handler, "_enqueue", False):
                any_enqueued = True
                break
        if not any_enqueued:
            _loguru_sink_warning_emitted = True
            logger.warning(
                "khora.integrations.hermes runtime initialised but no loguru sink "
                "has enqueue=True. Async tasks routed through run_sync will block "
                "on every logger.* call. Call khora.logging_config.setup_logging() "
                "or pass enqueue=True to your own logger.add() sinks."
            )


__all__ = ["_KhoraRuntime"]
