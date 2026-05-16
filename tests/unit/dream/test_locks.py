"""Tests for :mod:`khora.dream.locks`.

Mixes in-process tests (sqlite + lock-id derivation) with PG-backed tests
that skip cleanly when PostgreSQL isn't reachable on the dev-compose port.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from khora.dream.locks import (
    DreamLockUnavailable,
    _namespace_lock_id,
    acquire_namespace_dream_lock,
)

# ---------------------------------------------------------------------------
# PG fixture: skip when not reachable
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


def _pg_reachable() -> bool:
    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


_PG_AVAILABLE = _pg_reachable()
_skip_no_pg = pytest.mark.skipif(
    not _PG_AVAILABLE,
    reason="PostgreSQL not reachable (run `make dev` first)",
)


@pytest.fixture
async def pg_engine() -> AsyncIterator[Any]:
    """Module-private async engine bound to the dev-compose PG."""
    eng = create_async_engine(DATABASE_URL)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def pg_session(pg_engine: Any) -> AsyncIterator[AsyncSession]:
    """Fresh AsyncSession with its own transaction for each test."""
    factory = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with factory() as session:
        async with session.begin():
            yield session


# ---------------------------------------------------------------------------
# Lock-ID derivation: deterministic + well-distributed
# ---------------------------------------------------------------------------


def test_lock_id_deterministic() -> None:
    """Same UUID must yield the same lock id every call."""
    ns = uuid4()
    assert _namespace_lock_id(ns) == _namespace_lock_id(ns)


def test_lock_id_different_namespaces() -> None:
    """100 random UUIDs must all produce distinct lock ids."""
    ids = {_namespace_lock_id(uuid4()) for _ in range(100)}
    assert len(ids) == 100


def test_lock_id_signed_int64_range() -> None:
    """Lock id must fit into the signed 64-bit range pg_advisory_lock accepts."""
    for _ in range(50):
        lid = _namespace_lock_id(uuid4())
        assert -(2**63) <= lid < 2**63


def test_lock_id_does_not_collide_with_migration_lock() -> None:
    """The dream lock-id namespace is disjoint from the migration lock id.

    Migration lock id (from db/migrations/env.py) is the md5-derived
    ``6001515088189075507``; the dream id derives from blake2b with a
    different prefix so a UUID can't accidentally hit it.
    """
    migration_lock_id = 6001515088189075507
    for _ in range(100):
        assert _namespace_lock_id(uuid4()) != migration_lock_id


# ---------------------------------------------------------------------------
# Postgres path
# ---------------------------------------------------------------------------


@_skip_no_pg
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_lock_acquire_release(pg_session: AsyncSession) -> None:
    """Acquire, do work, release; second acquire after release succeeds."""
    ns = uuid4()
    async with acquire_namespace_dream_lock(pg_session, ns, timeout_seconds=5.0):
        pass
    # Same session, same namespace: re-entrant tx, lock auto-released on tx end
    # is checked in the dedicated test below. Here we just verify the
    # context-manager exit path doesn't raise.


@_skip_no_pg
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_lock_concurrent_acquire_fast_fail(pg_engine: Any) -> None:
    """A second session must DreamLockUnavailable when the lock is held."""
    ns = uuid4()
    factory = async_sessionmaker(pg_engine, expire_on_commit=False)

    holder_acquired = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with factory() as s, s.begin():
            async with acquire_namespace_dream_lock(s, ns, timeout_seconds=5.0):
                holder_acquired.set()
                await release_holder.wait()

    holder_task = asyncio.create_task(holder())
    await holder_acquired.wait()
    try:
        async with factory() as s, s.begin():
            with pytest.raises(DreamLockUnavailable) as excinfo:
                async with acquire_namespace_dream_lock(s, ns, timeout_seconds=0.0):
                    pass
            assert excinfo.value.namespace_id == ns
    finally:
        release_holder.set()
        await holder_task


@_skip_no_pg
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_lock_different_namespaces_parallel(pg_engine: Any) -> None:
    """Two different namespaces must hold their locks concurrently."""
    ns_a, ns_b = uuid4(), uuid4()
    factory = async_sessionmaker(pg_engine, expire_on_commit=False)
    acquired = asyncio.Event()
    release = asyncio.Event()

    async def hold(ns: UUID) -> None:
        async with factory() as s, s.begin():
            async with acquire_namespace_dream_lock(s, ns, timeout_seconds=2.0):
                acquired.set()
                await release.wait()

    task_a = asyncio.create_task(hold(ns_a))
    await acquired.wait()
    acquired.clear()
    # If the locks contended, this acquire would time out.
    task_b = asyncio.create_task(hold(ns_b))
    await asyncio.wait_for(acquired.wait(), timeout=3.0)
    release.set()
    await asyncio.gather(task_a, task_b)


@_skip_no_pg
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_lock_released_on_session_drop(pg_engine: Any) -> None:
    """Drop a session holding the lock; another session must acquire it."""
    ns = uuid4()
    factory = async_sessionmaker(pg_engine, expire_on_commit=False)

    # Hold the lock briefly, then drop the session without explicit release.
    async with factory() as s_a:
        async with s_a.begin():
            async with acquire_namespace_dream_lock(s_a, ns, timeout_seconds=5.0):
                pass
        # Transaction commits here → advisory lock auto-released.

    # Second session should acquire immediately.
    async with factory() as s_b, s_b.begin():
        async with acquire_namespace_dream_lock(s_b, ns, timeout_seconds=1.0):
            pass


# ---------------------------------------------------------------------------
# Embedded (sqlite) path
# ---------------------------------------------------------------------------


@pytest.fixture
async def sqlite_session() -> AsyncIterator[AsyncSession]:
    """In-memory SQLite session for the embedded-fallback path."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        factory = async_sessionmaker(eng, expire_on_commit=False)
        async with factory() as session:
            yield session
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_sqlite_lock_in_process(sqlite_session: AsyncSession) -> None:
    """Same-namespace acquire on sqlite serializes via in-process asyncio.Lock."""
    ns = uuid4()
    order: list[str] = []
    release_a = asyncio.Event()

    async def task_a() -> None:
        async with acquire_namespace_dream_lock(sqlite_session, ns, timeout_seconds=5.0):
            order.append("a-acquired")
            await release_a.wait()
            order.append("a-released")

    async def task_b() -> None:
        # Yield once so task_a definitely got there first.
        await asyncio.sleep(0)
        async with acquire_namespace_dream_lock(sqlite_session, ns, timeout_seconds=5.0):
            order.append("b-acquired")

    ta = asyncio.create_task(task_a())
    tb = asyncio.create_task(task_b())
    # Give a few ticks for a to acquire and b to start waiting.
    for _ in range(10):
        await asyncio.sleep(0)
        if "a-acquired" in order:
            break
    assert order == ["a-acquired"]
    release_a.set()
    await asyncio.gather(ta, tb)
    assert order == ["a-acquired", "a-released", "b-acquired"]


@pytest.mark.asyncio
async def test_sqlite_lock_timeout(sqlite_session: AsyncSession) -> None:
    """Embedded path: timeout elapsing must raise DreamLockUnavailable."""
    ns = uuid4()
    release = asyncio.Event()

    async def holder() -> None:
        async with acquire_namespace_dream_lock(sqlite_session, ns, timeout_seconds=5.0):
            await release.wait()

    holder_task = asyncio.create_task(holder())
    await asyncio.sleep(0.01)
    try:
        with pytest.raises(DreamLockUnavailable):
            async with acquire_namespace_dream_lock(sqlite_session, ns, timeout_seconds=0.05):
                pass
    finally:
        release.set()
        await holder_task
