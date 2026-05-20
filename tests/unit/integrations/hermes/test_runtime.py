"""Unit tests for ``khora.integrations.hermes._runtime``.

All tests use ``AsyncMock`` in place of a real :class:`khora.Khora`. The
runtime routes async work through ``khora.integrations._sync.run_sync``,
which runs the coroutine on the process-wide bridge loop — so an
``AsyncMock`` exercises the same submission discipline real khora does
without spinning up a database or vector index.

Tests are marked ``unit`` and avoid sleep-loops where possible by
blocking on Futures with bounded timeouts.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.integrations.hermes._runtime import _KhoraRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kb(*, recall_result: object | None = None, recall_raises: BaseException | None = None) -> MagicMock:
    """Build a Khora-shaped mock with async ``remember`` / ``recall``."""
    kb = MagicMock(name="Khora")
    kb.remember = AsyncMock(return_value=None)
    kb.remember_batch = AsyncMock(return_value=None)
    if recall_raises is not None:
        kb.recall = AsyncMock(side_effect=recall_raises)
    else:
        kb.recall = AsyncMock(return_value=recall_result)
    return kb


def _make_document(content: str = "hello") -> MagicMock:
    """Build a Document-shaped mock with the attributes the runtime reads."""
    doc = MagicMock(name="Document")
    doc.content = content
    doc.title = ""
    doc.source = ""
    doc.source_type = "library"
    doc.source_name = None
    doc.source_url = None
    doc.source_timestamp = None
    doc.metadata = {}
    doc.external_id = None
    doc.session_id = None
    return doc


def _wait_for_idle(runtime: _KhoraRuntime, *, timeout: float = 2.0) -> int:
    """Drain the runtime to a quiescent state. Fails the test on timeout."""
    pending = runtime.drain(timeout=timeout)
    if pending != 0:
        raise AssertionError(f"runtime still has {pending} pending tasks after {timeout}s drain")
    return pending


# ---------------------------------------------------------------------------
# remember / counters
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_enqueue_remember_increments_success_counter() -> None:
    rt = _KhoraRuntime()
    kb = _make_kb()
    ns = uuid4()
    try:
        rt.enqueue_remember(kb, ns, _make_document("alpha"))
        _wait_for_idle(rt)
        kb.remember.assert_awaited_once()
        assert rt.failure_rate_pct() == 0.0
        assert rt.last_errors() == []
    finally:
        rt.shutdown()


@pytest.mark.unit
def test_enqueue_remember_records_failure() -> None:
    rt = _KhoraRuntime()
    kb = _make_kb()
    kb.remember = AsyncMock(side_effect=RuntimeError("kaboom"))
    ns = uuid4()
    try:
        rt.enqueue_remember(kb, ns, _make_document("alpha"))
        _wait_for_idle(rt)
        kb.remember.assert_awaited_once()
        # 1 failure out of 1 attempt = 100%
        assert rt.failure_rate_pct() == 100.0
        errors = rt.last_errors()
        assert len(errors) == 1
        assert "kaboom" in errors[0]
        assert errors[0].startswith("[remember]")
    finally:
        rt.shutdown()


@pytest.mark.unit
def test_failure_rate_math() -> None:
    """3 failures out of 10 attempts → 30.0%."""
    rt = _KhoraRuntime()
    # Mix 7 successes + 3 failures by toggling the side effect mid-stream.
    kb = _make_kb()
    call_log: list[int] = []

    async def _maybe_raise(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        idx = len(call_log)
        call_log.append(idx)
        if idx in (2, 5, 8):
            raise ValueError(f"fail-{idx}")

    kb.remember = AsyncMock(side_effect=_maybe_raise)
    ns = uuid4()
    try:
        for _ in range(10):
            rt.enqueue_remember(kb, ns, _make_document())
        _wait_for_idle(rt)
        assert kb.remember.await_count == 10
        assert rt.failure_rate_pct() == pytest.approx(30.0)
    finally:
        rt.shutdown()


@pytest.mark.unit
def test_last_errors_truncates_and_limits() -> None:
    rt = _KhoraRuntime()
    long_msg = "x" * 500  # > 200 char truncation cap

    async def _raise(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError(long_msg)

    kb = _make_kb()
    kb.remember = AsyncMock(side_effect=_raise)
    ns = uuid4()
    try:
        for _ in range(4):
            rt.enqueue_remember(kb, ns, _make_document())
        _wait_for_idle(rt)
        recent_two = rt.last_errors(n=2)
        assert len(recent_two) == 2
        # Truncation: prefix "[remember] " (11) + up to 200 truncated chars
        for err in recent_two:
            assert err.startswith("[remember] ")
            # The exception str itself is truncated at 200 chars.
            payload = err.removeprefix("[remember] ")
            assert len(payload) <= 200
    finally:
        rt.shutdown()


# ---------------------------------------------------------------------------
# recall + prefetch cache
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recall_sync_hits_materialised_cache() -> None:
    """A second recall_sync within the TTL returns the cached RecallResult
    without re-awaiting ``kb.recall``."""
    rt = _KhoraRuntime(prefetch_cache_ttl_s=30.0)
    sentinel = object()
    kb = _make_kb(recall_result=sentinel)
    ns = uuid4()
    try:
        first = rt.recall_sync(kb, ns, "sess", "what did we discuss?", timeout=2.0)
        assert first is sentinel
        # Recall ran exactly once for the miss path.
        kb.recall.assert_awaited_once()

        second = rt.recall_sync(kb, ns, "sess", "what did we discuss?", timeout=0.5)
        assert second is sentinel
        # No second call: cache served the read.
        assert kb.recall.await_count == 1
    finally:
        rt.shutdown()


@pytest.mark.unit
def test_recall_sync_blocks_on_in_flight_future() -> None:
    """If the cache holds an in-flight Future (set by enqueue_recall), a
    parallel recall_sync waits on that Future, not a new submission."""
    import threading

    release = threading.Event()
    sentinel = object()

    async def _slow_recall(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Block the worker until the test releases it.
        release.wait(timeout=5.0)
        return sentinel

    rt = _KhoraRuntime()
    kb = _make_kb()
    kb.recall = AsyncMock(side_effect=_slow_recall)
    ns = uuid4()
    try:
        rt.enqueue_recall(kb, ns, "sess", "in-flight question")
        # Recall is queued/running but not done. The cache slot holds a
        # Future. recall_sync should bind to it, not enqueue a second
        # call.
        # Release the worker on a separate thread so the synchronous
        # recall_sync below can complete.
        threading.Timer(0.05, release.set).start()
        got = rt.recall_sync(kb, ns, "sess", "in-flight question", timeout=2.0)
        assert got is sentinel
        # Exactly one recall ran, even though enqueue + recall_sync both
        # asked for it.
        assert kb.recall.await_count == 1
    finally:
        release.set()
        rt.shutdown()


@pytest.mark.unit
def test_recall_sync_timeout_returns_none_and_keeps_future_live() -> None:
    """When the Future doesn't complete inside ``timeout``, recall_sync
    returns None and does NOT cancel the underlying Future (other
    readers may still want the result)."""
    import threading

    release = threading.Event()

    async def _hang(*args, **kwargs):  # type: ignore[no-untyped-def]
        release.wait(timeout=5.0)
        return object()

    rt = _KhoraRuntime()
    kb = _make_kb()
    kb.recall = AsyncMock(side_effect=_hang)
    ns = uuid4()
    try:
        out = rt.recall_sync(kb, ns, "sess", "hangs", timeout=0.01)
        assert out is None
        # The cache entry's Future should still be pending — not cancelled.
        key = rt._cache_key(ns, "sess", "hangs")
        entry = rt._cache.get(key)
        assert entry is not None, "cache slot should still hold the in-flight Future"
        assert entry.future is not None
        assert not entry.future.cancelled(), "timeout must not cancel the Future"
    finally:
        release.set()
        rt.shutdown()


@pytest.mark.unit
def test_enqueue_recall_is_idempotent_within_ttl() -> None:
    """Two enqueue_recall calls for the same key while one is in flight
    coalesce to a single Khora.recall invocation."""
    import threading

    release = threading.Event()

    async def _slow(*args, **kwargs):  # type: ignore[no-untyped-def]
        release.wait(timeout=5.0)
        return object()

    rt = _KhoraRuntime()
    kb = _make_kb()
    kb.recall = AsyncMock(side_effect=_slow)
    ns = uuid4()
    try:
        rt.enqueue_recall(kb, ns, "sess", "same query")
        rt.enqueue_recall(kb, ns, "sess", "same query")
        rt.enqueue_recall(kb, ns, "sess", "same query")
        release.set()
        _wait_for_idle(rt)
        assert kb.recall.await_count == 1
    finally:
        release.set()
        rt.shutdown()


# ---------------------------------------------------------------------------
# Queue overflow / shed-oldest
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_queue_overflow_sheds_oldest() -> None:
    """With queue_max_size=2 and the worker blocked, the 3rd submission
    shed the oldest pending Future."""
    import threading

    release = threading.Event()
    started = threading.Event()

    async def _block_first(*args, **kwargs):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(timeout=5.0)
        return None

    rt = _KhoraRuntime(queue_max_size=2)
    kb = _make_kb()
    kb.remember = AsyncMock(side_effect=_block_first)
    ns = uuid4()
    try:
        # First submission starts running and blocks. Subsequent two
        # queue up.
        rt.enqueue_remember(kb, ns, _make_document("a"))
        assert started.wait(timeout=2.0), "first task never started"
        rt.enqueue_remember(kb, ns, _make_document("b"))
        rt.enqueue_remember(kb, ns, _make_document("c"))
        # At this point the deque has the in-flight (a) + queued (b, c)
        # = 3 entries, exceeding cap=2. Pushing a 4th must shed the
        # oldest (the in-flight one OR b — implementation drops the
        # head of the deque).
        rt.enqueue_remember(kb, ns, _make_document("d"))

        # Verify the shed counter went up by inspecting failure-rate
        # bookkeeping is independent: we instead probe the deque length.
        # After the shed, queue should contain at most queue_max_size
        # pending Futures.
        with rt._pending_lock:
            # Drop the completed (none yet) — count only pending.
            pending_now = sum(1 for f in rt._pending if not f.done())
        assert pending_now <= 2, f"deque should not exceed cap, got {pending_now}"

        release.set()
        _wait_for_idle(rt)
        # We submitted 4 calls; at least one was shed. Worker awaited
        # the surviving ones.
        assert kb.remember.await_count <= 4
        assert kb.remember.await_count >= 2
    finally:
        release.set()
        rt.shutdown()


# ---------------------------------------------------------------------------
# drain / shutdown
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_drain_blocks_until_idle() -> None:
    rt = _KhoraRuntime()
    kb = _make_kb()
    ns = uuid4()
    try:
        for _ in range(5):
            rt.enqueue_remember(kb, ns, _make_document())
        t0 = time.monotonic()
        pending = rt.drain(timeout=2.0)
        elapsed = time.monotonic() - t0
        assert pending == 0
        assert elapsed < 2.0, f"drain returned 0 but took {elapsed}s — should be near-instant"
        assert kb.remember.await_count == 5
    finally:
        rt.shutdown()


@pytest.mark.unit
def test_shutdown_is_idempotent() -> None:
    rt = _KhoraRuntime()
    rt.shutdown()
    # Second call must not raise.
    rt.shutdown()


@pytest.mark.unit
def test_failure_rate_zero_when_no_calls() -> None:
    """Empty bucket: failure rate is 0.0, not a divide-by-zero."""
    rt = _KhoraRuntime()
    try:
        assert rt.failure_rate_pct() == 0.0
        assert rt.last_errors() == []
    finally:
        rt.shutdown()


@pytest.mark.unit
def test_last_errors_n_zero_returns_empty() -> None:
    rt = _KhoraRuntime()
    kb = _make_kb()
    kb.remember = AsyncMock(side_effect=RuntimeError("nope"))
    ns = uuid4()
    try:
        rt.enqueue_remember(kb, ns, _make_document())
        _wait_for_idle(rt)
        assert rt.last_errors(n=0) == []
    finally:
        rt.shutdown()


@pytest.mark.unit
def test_init_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        _KhoraRuntime(queue_max_size=0)
    with pytest.raises(ValueError):
        _KhoraRuntime(prefetch_cache_ttl_s=0)
