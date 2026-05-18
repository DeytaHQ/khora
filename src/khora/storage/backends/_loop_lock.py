"""Per-event-loop ``asyncio.Lock`` cache for module-level serialization.

Module-level ``asyncio.Lock`` instances bind to whichever event loop
first acquires them. pytest-asyncio's default function-scoped loop
creates a fresh loop per test, and reusing a bound lock from a
different loop raises ``RuntimeError: Lock is bound to a different
event loop``.

``get_loop_lock(name)`` returns an ``asyncio.Lock`` keyed by the
currently running event loop. Within a single loop, all callers using
the same ``name`` share the same lock (the cross-instance serialization
the connection modules rely on). Across loops, each loop gets its own
independent lock, so tests on different loops don't collide.

The backing store is a :class:`weakref.WeakKeyDictionary`, so once a
test loop is garbage-collected its locks are released automatically.
"""

from __future__ import annotations

import asyncio
import weakref

_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = weakref.WeakKeyDictionary()


def get_loop_lock(name: str) -> asyncio.Lock:
    """Return the ``asyncio.Lock`` for ``name`` on the running event loop.

    Must be called from inside a running event loop.
    """
    loop = asyncio.get_running_loop()
    bucket = _locks.get(loop)
    if bucket is None:
        bucket = {}
        _locks[loop] = bucket
    lock = bucket.get(name)
    if lock is None:
        lock = asyncio.Lock()
        bucket[name] = lock
    return lock
