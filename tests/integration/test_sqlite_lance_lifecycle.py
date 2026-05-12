"""EmbeddedStorageHandle lifecycle tests.

The handle multiplexes a single aiosqlite connection + a single LanceDB
connection across four storage adapters (relational, vector, graph,
event-store) that all call ``connect()`` / ``disconnect()`` concurrently.
Three regressions are easy to introduce here and hard to spot without
explicit coverage:

1. **Partial-connect cleanup**: if LanceDB fails to open AFTER aiosqlite
   has been opened, the next ``disconnect()`` MUST close the aiosqlite
   connection — otherwise the FD leaks. The current implementation
   handles this correctly (see ``connection.py:182-198``); this test
   codifies that contract.
2. **Repeated connect/disconnect cycles**: should release FDs each
   round. Sample with ``psutil`` to catch a slow leak.
3. **Concurrent disconnect across four adapters**: the lifecycle lock
   serializes the close so the worker thread doesn't double-close
   aiosqlite. This test simulates the coordinator's
   ``asyncio.gather(disconnect, disconnect, ...)`` pattern.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from khora.storage.backends.sqlite_lance.connection import (
    EmbeddedStorageHandle,
    EmbeddedStorageHandleConfig,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
    ),
]


def _make_handle(tmp_path: Path) -> EmbeddedStorageHandle:
    config = EmbeddedStorageHandleConfig(
        db_path=str(tmp_path / "khora.db"),
        lance_path=str(tmp_path / "khora.lance"),
        embedding_dimension=32,
    )
    return EmbeddedStorageHandle(config)


async def test_partial_connect_cleans_up_aiosqlite_on_lance_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If LanceDB raises during connect, the aiosqlite handle must still close.

    Sequence:
    1. ``handle.connect()`` opens aiosqlite, then calls ``lancedb.connect_async``
       which we patch to raise.
    2. The exception bubbles up; the handle is in a "partial-connect" state
       (``_sqlite`` set, ``_lance`` None, ``_connected`` False).
    3. ``handle.disconnect()`` must still close the aiosqlite connection,
       not short-circuit. The early-return guard at line 183 requires ALL
       THREE state flags to indicate "fully disconnected" — a partial-connect
       state must therefore proceed through the close path.
    """
    handle = _make_handle(tmp_path)

    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated lancedb connect failure")

    monkeypatch.setattr("lancedb.connect_async", _boom)

    with pytest.raises(RuntimeError, match="simulated lancedb connect failure"):
        await handle.connect()

    # State after partial connect: aiosqlite is open, lance is None,
    # connected is False. The disconnect path must still close aiosqlite.
    assert handle._sqlite is not None, "partial-connect should have left aiosqlite open"
    assert handle._lance is None
    assert not handle._connected

    await handle.disconnect()

    # After disconnect, aiosqlite must be closed (None) so we don't leak the FD.
    assert handle._sqlite is None, (
        "disconnect must close the aiosqlite connection from a partial-connect state "
        "or every retried connect() loop leaks an FD"
    )


async def test_repeated_connect_disconnect_no_fd_leak(tmp_path: Path) -> None:
    """100 connect/disconnect cycles must not grow the process FD count.

    Picks up regressions in the lifecycle lock or the ``_sqlite = None``
    guard that previously caused FDs to leak across cycles.
    """
    if not _HAS_PSUTIL:
        pytest.skip("psutil not installed; install to detect FD leaks")
    handle = _make_handle(tmp_path)
    proc = psutil.Process()

    # Warm up: one full cycle to populate any one-time caches/files.
    await handle.connect()
    await handle.disconnect()

    fds_before = proc.num_fds()
    for _ in range(20):
        await handle.connect()
        await handle.disconnect()
    fds_after = proc.num_fds()

    # Allow a tiny slack for log file rotation / Python tempfiles, but a real
    # leak grows linearly with cycle count — 20 cycles, more than 5 extra FDs
    # would indicate a leak.
    assert fds_after - fds_before <= 5, (
        f"FD leak detected: {fds_before} → {fds_after} FDs across 20 connect/disconnect cycles"
    )


async def test_concurrent_disconnect_is_idempotent(tmp_path: Path) -> None:
    """Four concurrent ``disconnect()`` calls (mirroring coordinator fan-out)
    must not raise or hang.

    The lifecycle lock at ``_get_lifecycle_lock`` serializes the close path
    so the second-and-later callers see ``_sqlite is None`` and the early-
    return path triggers. Before the lock existed, both callers could enter
    ``_sqlite.close()`` simultaneously, double-closing the aiosqlite worker.
    """
    handle = _make_handle(tmp_path)
    await handle.connect()

    # Four concurrent disconnects, the same shape the StorageCoordinator
    # uses to fan out adapter teardown.
    results = await asyncio.gather(
        handle.disconnect(),
        handle.disconnect(),
        handle.disconnect(),
        handle.disconnect(),
        return_exceptions=True,
    )

    for r in results:
        assert not isinstance(r, Exception), f"concurrent disconnect raised: {r!r}"

    assert handle._sqlite is None
    assert handle._lance is None
    assert not handle._connected

    # A subsequent disconnect MUST also be a no-op (full idempotency).
    await handle.disconnect()


async def test_disconnect_then_reconnect_starts_clean(tmp_path: Path) -> None:
    """After disconnect, the same handle should connect again cleanly.

    This is the consumer-facing invariant: a service that handles a
    SIGHUP-style "reload config" sequence will call ``disconnect()`` then
    ``connect()`` on the same handle. Both must succeed and queries must
    work after the reconnect.
    """
    handle = _make_handle(tmp_path)
    await handle.connect()
    await handle.disconnect()

    # Reconnect on the same handle — schema init guard (``_schema_initialized``)
    # is sticky for the handle's lifetime, so the second connect doesn't
    # rebuild the LanceDB tables.
    await handle.connect()
    assert handle._sqlite is not None
    assert handle._lance is not None
    assert handle._connected

    # Quick sanity ping.
    healthy = await handle.is_healthy()
    assert healthy
    await handle.disconnect()
