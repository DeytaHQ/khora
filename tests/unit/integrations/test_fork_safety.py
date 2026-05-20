"""Fork-safety tests for ``khora.integrations._sync`` and ``Khora.shared()``.

Issue #790. Validates that ``os.register_at_fork(after_in_child=...)``
handlers reset state in forked children so the parent's daemon thread
(_sync bridge) and the parent's cached Khora instances don't carry
ghost references into the child.

Requires Linux/macOS - fork is POSIX-only. The ``importorskip("posix")``
guard makes the test a no-op on Windows.
"""

from __future__ import annotations

import multiprocessing
import sys

import pytest

# POSIX-only: ``os.register_at_fork`` is not present on Windows, and
# the multiprocessing ``fork`` start method is also Linux/macOS only.
posix = pytest.importorskip("posix")
if sys.platform == "win32":  # pragma: no cover - belt-and-braces
    pytest.skip("fork() is POSIX-only", allow_module_level=True)


def _child_run_sync_after_fork(queue: multiprocessing.Queue) -> None:
    """Run inside a forked child.

    Calls ``run_sync(...)`` and pushes the result (or exception class
    name) onto the parent-readable queue. Used by the test below to
    verify the bridge re-initialises cleanly in the child.
    """
    try:
        from khora.integrations._sync import _loop, run_sync

        # After the at-fork handler fires, the parent's loop reference
        # must be gone. If it isn't, ``_ensure_loop()`` would return the
        # dead parent loop and ``run_coroutine_threadsafe`` would hang.
        if _loop is not None:
            queue.put(("FAIL", f"parent loop ref leaked into child: {_loop!r}"))
            return

        async def _trivial() -> int:
            return 42

        out = run_sync(_trivial())
        queue.put(("OK", out))
    except Exception as exc:  # pragma: no cover - covered by queue.put
        queue.put(("EXC", f"{type(exc).__name__}: {exc}"))


def _child_shared_cache_after_fork(queue: multiprocessing.Queue) -> None:
    """Run inside a forked child.

    Verifies the parent's ``_SHARED_INSTANCES`` cache was cleared by the
    at-fork handler so the child doesn't reuse the parent's asyncpg
    pool sockets.
    """
    try:
        from khora.khora import _SHARED_INSTANCES, _SHARED_LOCK

        if _SHARED_INSTANCES:
            queue.put(
                (
                    "FAIL",
                    f"parent shared-Khora cache leaked into child: {list(_SHARED_INSTANCES)}",
                )
            )
            return
        # The lock is reseated to a fresh asyncio.Lock; we can't easily
        # exercise it here (would need a running loop) but its identity
        # should differ from whatever the parent originally placed.
        queue.put(("OK", id(_SHARED_LOCK)))
    except Exception as exc:  # pragma: no cover
        queue.put(("EXC", f"{type(exc).__name__}: {exc}"))


@pytest.mark.unit
def test_run_sync_after_fork_does_not_hang() -> None:
    """A forked child can use run_sync without hanging on the parent's loop."""
    # Touch the bridge in the parent so the daemon loop is live before
    # we fork; otherwise the test is uninteresting (nothing to leak).
    from khora.integrations._sync import _ensure_loop

    _ensure_loop()

    ctx = multiprocessing.get_context("fork")
    queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(target=_child_run_sync_after_fork, args=(queue,))
    proc.start()
    proc.join(timeout=10.0)

    assert not proc.is_alive(), "child hung after fork - at-fork handler did not fire"
    assert proc.exitcode == 0, f"child exited non-zero: {proc.exitcode}"

    status, payload = queue.get(timeout=2.0)
    assert status == "OK", f"child reported {status}: {payload}"
    assert payload == 42


@pytest.mark.unit
def test_shared_cache_cleared_after_fork() -> None:
    """The Khora.shared() cache is cleared in the forked child."""
    # Seed the parent's cache with a fake entry to verify the handler
    # clears it. We don't need a real Khora() - just any value that
    # would be wrong to share into a child.
    from khora.khora import _SHARED_INSTANCES

    sentinel_key = "test-fork-sentinel"
    _SHARED_INSTANCES[sentinel_key] = "parent-only"  # type: ignore[assignment]
    try:
        ctx = multiprocessing.get_context("fork")
        queue: multiprocessing.Queue = ctx.Queue()
        proc = ctx.Process(target=_child_shared_cache_after_fork, args=(queue,))
        proc.start()
        proc.join(timeout=10.0)

        assert not proc.is_alive()
        assert proc.exitcode == 0

        status, payload = queue.get(timeout=2.0)
        assert status == "OK", f"child reported {status}: {payload}"
        # The parent's cache still has the sentinel - handler only fires
        # in the child, never in the parent.
        assert _SHARED_INSTANCES.get(sentinel_key) == "parent-only"
    finally:
        _SHARED_INSTANCES.pop(sentinel_key, None)
