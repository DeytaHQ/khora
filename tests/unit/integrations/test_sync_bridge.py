"""Tests for ``khora.integrations._sync.run_sync``.

Happy path, reentrancy refusal, exception propagation, and the
sync-bridge daemon-thread lifecycle.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from khora.integrations import _sync
from khora.integrations._sync import run_sync


def test_run_sync_returns_coroutine_result(shutdown_sync_bridge):
    async def coro() -> int:
        return 42

    assert run_sync(coro()) == 42


def test_run_sync_propagates_exceptions(shutdown_sync_bridge):
    class _Boom(RuntimeError):
        pass

    async def coro() -> None:
        raise _Boom("kaboom")

    with pytest.raises(_Boom, match="kaboom"):
        run_sync(coro())


def test_run_sync_rejects_non_coroutine(shutdown_sync_bridge):
    with pytest.raises(TypeError):
        run_sync("not a coroutine")  # type: ignore[arg-type]


def test_run_sync_reentrancy_raises_inside_running_loop(shutdown_sync_bridge):
    # Re-entering from inside an asyncio.run() loop is the deadlock
    # surface. run_sync must refuse rather than silently spawn.
    async def coro_inner() -> int:
        return 1

    async def driver() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            run_sync(coro_inner())

    asyncio.run(driver())


def test_run_sync_reuses_loop_across_calls(shutdown_sync_bridge):
    async def coro(x: int) -> int:
        return x * 2

    assert run_sync(coro(1)) == 2
    loop_after_first = _sync._loop
    assert run_sync(coro(5)) == 10
    assert _sync._loop is loop_after_first  # reused, not recreated


def test_sync_bridge_thread_is_daemon_and_named(shutdown_sync_bridge):
    async def coro() -> None:
        return None

    run_sync(coro())
    assert _sync._loop_thread is not None
    assert _sync._loop_thread.daemon is True
    assert _sync._loop_thread.name == "khora-integrations-sync-bridge"


def test_run_sync_works_from_worker_thread(shutdown_sync_bridge):
    # Many adapter call paths land here: a worker thread (FastAPI sync
    # handler, CrewAI tool) reaching into async khora.
    results: list[int] = []
    errors: list[BaseException] = []

    async def coro() -> int:
        return 7

    def worker() -> None:
        try:
            results.append(run_sync(coro()))
        except BaseException as exc:  # noqa: BLE001 — test recorder
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert errors == []
    assert results == [7, 7, 7, 7]


def test_shutdown_for_tests_releases_loop():
    async def coro() -> None:
        return None

    run_sync(coro())
    assert _sync._loop is not None
    _sync._shutdown_for_tests()
    assert _sync._loop is None
    assert _sync._loop_thread is None


def test_run_sync_recovers_after_shutdown():
    async def coro() -> int:
        return 99

    run_sync(coro())
    _sync._shutdown_for_tests()
    # A fresh call after shutdown rebuilds the loop transparently.
    assert run_sync(coro()) == 99
    _sync._shutdown_for_tests()


def test_run_sync_cleans_up_after_exception(shutdown_sync_bridge):
    # An exception inside the coroutine must not leave the bridge in a
    # broken state — subsequent calls succeed.
    async def boom() -> None:
        raise ValueError("fail")

    async def ok() -> int:
        return 1

    with pytest.raises(ValueError):
        run_sync(boom())
    assert run_sync(ok()) == 1
