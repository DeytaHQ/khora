"""Sync-bridge helper for adapters that wrap async khora in sync APIs.

CrewAI's ``StorageBackend``, LangGraph's ``BaseStore`` sync abstracts,
LlamaIndex's ``BaseChatStore`` — many frameworks expose sync methods
that must call into async khora. Without one shared bridge, every
adapter reinvents it, and one of them deadlocks.

This module owns one daemon-thread event loop. Sync callers hand it a
coroutine via :func:`run_sync`; the coroutine runs on that loop and the
caller blocks until it completes. Reentrancy is **rejected explicitly**:
calling :func:`run_sync` from inside a running event loop raises
:class:`RuntimeError`. That's the deadlock surface and we refuse to
paper over it.

Underscore-prefixed module: it's a contract surface for adapter authors,
not for end users. Adapters call ``run_sync(self.kb.recall(...))``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

# Lazy singleton daemon loop. Allocated on first call; reused thereafter.
# Daemon thread so it doesn't block process exit.
_loop_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Return the shared daemon-thread event loop, creating it if needed."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and not _loop.is_closed():
            return _loop

        ready = threading.Event()
        new_loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(new_loop)
            ready.set()
            try:
                new_loop.run_forever()
            finally:
                # On run_forever exit, close to release fds. Best-effort
                # — the loop may already be closed by _shutdown_for_tests
                # racing with this finally.
                try:
                    new_loop.close()
                except RuntimeError:
                    pass

        thread = threading.Thread(
            target=_runner,
            name="khora-integrations-sync-bridge",
            daemon=True,
        )
        thread.start()
        ready.wait()

        _loop = new_loop
        _loop_thread = thread
        return _loop


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync code and return its result.

    Adapters call this when their framework's sync entry point needs to
    invoke an async khora method::

        result = run_sync(self.kb.recall(query, namespace=self.namespace_id))

    Args:
        coro: The coroutine to run.

    Returns:
        The coroutine's return value.

    Raises:
        RuntimeError: If called from inside a running asyncio event loop.
            Spawning a thread to dispatch the coroutine and blocking the
            caller would deadlock on a single-loop application, so we
            refuse — the caller must restructure to ``await`` directly
            or hop to a worker thread first.
        Exception: Anything the coroutine itself raises is re-raised
            from this call (after the bridge releases its frame).
    """
    if not asyncio.iscoroutine(coro):
        raise TypeError(f"run_sync expects a coroutine, got {type(coro).__name__}")

    # Reentrancy check: if a loop is already running on THIS thread, we
    # can't block here without deadlocking.
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        # Close the coroutine to avoid the "coroutine was never awaited"
        # warning the user would otherwise get on top of the exception.
        coro.close()
        raise RuntimeError(
            "run_sync() cannot be called from inside a running event loop. "
            "Await the coroutine directly, or hop to a worker thread first."
        )

    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _shutdown_for_tests() -> None:
    """Stop and discard the bridge loop. Test-only.

    Production code never calls this — the daemon thread is meant to
    outlive every adapter. Tests use it to verify clean state between
    cases.
    """
    global _loop, _loop_thread
    with _loop_lock:
        loop = _loop
        thread = _loop_thread
        _loop = None
        _loop_thread = None
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5.0)
