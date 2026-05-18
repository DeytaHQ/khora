"""Regression test for the per-event-loop lock helper.

A plain module-level ``asyncio.Lock()`` binds to the first event loop
that touches it, so a second test on a fresh loop (e.g. pytest-asyncio's
default function-scoped loop) raises ``RuntimeError: Lock is bound to a
different event loop`` when it tries to acquire the same lock.

``get_loop_lock(name)`` must give each event loop an independent lock
keyed by the same ``name`` so cross-instance serialization still works
within a loop while different loops never collide.
"""

from __future__ import annotations

import asyncio

import pytest

from khora.storage.backends._loop_lock import get_loop_lock


@pytest.mark.unit
def test_same_loop_same_lock_for_same_name() -> None:
    """Two calls within the same running loop return the *same* lock."""

    async def _check() -> tuple[int, int]:
        a = get_loop_lock("x")
        b = get_loop_lock("x")
        return id(a), id(b)

    a_id, b_id = asyncio.run(_check())
    assert a_id == b_id


@pytest.mark.unit
def test_same_loop_distinct_names_distinct_locks() -> None:
    """Different ``name`` arguments are independent locks on the same loop."""

    async def _check() -> tuple[int, int]:
        a = get_loop_lock("x")
        b = get_loop_lock("y")
        return id(a), id(b)

    a_id, b_id = asyncio.run(_check())
    assert a_id != b_id


@pytest.mark.unit
def test_distinct_loops_distinct_locks_same_name() -> None:
    """Two ``asyncio.run`` invocations must each get their own lock for the
    same name — otherwise the second run hits ``RuntimeError: Lock is bound
    to a different event loop`` on acquire, which is the failure mode this
    helper exists to prevent.
    """

    async def _acquire_release() -> None:
        lock = get_loop_lock("shared")
        async with lock:
            pass

    asyncio.run(_acquire_release())
    # The second run uses a fresh loop. With a naive module-level
    # ``asyncio.Lock`` this raises RuntimeError on acquire. With the
    # per-loop helper it just works.
    asyncio.run(_acquire_release())


@pytest.mark.unit
def test_serializes_within_one_loop() -> None:
    """Cross-coroutine serialization still works within a single loop."""

    async def _run() -> list[str]:
        order: list[str] = []

        async def worker(name: str) -> None:
            async with get_loop_lock("serialize"):
                order.append(f"{name}-enter")
                await asyncio.sleep(0)
                order.append(f"{name}-exit")

        await asyncio.gather(worker("a"), worker("b"))
        return order

    order = asyncio.run(_run())
    # Each worker's enter must be followed by its own exit before the
    # other worker enters — otherwise the lock isn't serializing.
    assert order in (
        ["a-enter", "a-exit", "b-enter", "b-exit"],
        ["b-enter", "b-exit", "a-enter", "a-exit"],
    ), f"lock did not serialize: {order}"
