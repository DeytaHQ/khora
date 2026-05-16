"""Per-namespace advisory locking for dream-phase orchestration.

The dream orchestrator acquires an exclusive PG advisory lock keyed on the
target ``namespace_id`` for the duration of a run. Concurrent dream runs
against the same namespace fast-fail (or queue, depending on timeout);
different namespaces dream in parallel.

On embedded backends (sqlite_lance, surrealdb embedded) we fall back to an
in-process ``asyncio.Lock`` keyed by ``namespace_id``. Cross-process safety
is explicitly **not** promised on embedded — operators running multi-process
workers against an embedded DB must serialize their dream calls themselves.

Lock-ID derivation::

    raw = blake2b(b"khora.dream" + namespace_id.bytes, digest_size=8).digest()
    lock_id = int.from_bytes(raw, "big", signed=True)

This yields a stable, well-distributed signed 64-bit integer that PG's
advisory-lock API accepts. The ``b"khora.dream"`` prefix domain-separates
us from the migration lock (id ``6001515088189075507``) so the two never
collide.

Pattern mirrors :mod:`khora.db.migrations.env` — transaction-scoped lock,
auto-released on commit / rollback / session drop.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from khora.exceptions import KhoraError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_LOCK_PREFIX = b"khora.dream"

# In-process fallback for embedded backends (sqlite, surrealdb-embedded).
# Keyed by namespace_id so two namespaces never serialize against each other.
# Cross-process safety is explicitly NOT promised on embedded — see module docstring.
_embedded_locks: dict[UUID, asyncio.Lock] = {}
_embedded_locks_guard = asyncio.Lock()


class DreamLockUnavailable(KhoraError):
    """Raised when a dream-phase advisory lock could not be acquired.

    Carries the ``namespace_id`` and the timeout that elapsed so callers
    can surface useful diagnostics to operators.
    """

    def __init__(self, namespace_id: UUID, timeout_seconds: float) -> None:
        self.namespace_id = namespace_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Could not acquire dream lock for namespace {namespace_id} "
            f"within {timeout_seconds}s — another dream run is in progress."
        )


def _namespace_lock_id(namespace_id: UUID) -> int:
    """Derive a stable signed-int64 lock id from a namespace UUID.

    Uses blake2b with the ``b"khora.dream"`` domain-separation prefix so the
    output cannot collide with the migration lock id even if the same UUID
    were ever used as a domain marker elsewhere. Always returns a value in
    the signed-int64 range that PG's ``pg_advisory_xact_lock`` accepts.
    """
    digest = hashlib.blake2b(
        _LOCK_PREFIX + namespace_id.bytes,
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big", signed=True)


async def _get_embedded_lock(namespace_id: UUID) -> asyncio.Lock:
    """Return the per-namespace in-process lock, creating on first use."""
    async with _embedded_locks_guard:
        lock = _embedded_locks.get(namespace_id)
        if lock is None:
            lock = asyncio.Lock()
            _embedded_locks[namespace_id] = lock
        return lock


@asynccontextmanager
async def acquire_namespace_dream_lock(
    session: AsyncSession,
    namespace_id: UUID,
    *,
    timeout_seconds: float = 60.0,
) -> AsyncIterator[None]:
    """Acquire an exclusive advisory lock for the namespace's dream run.

    On Postgres: tries ``pg_try_advisory_xact_lock`` repeatedly with a small
    sleep between attempts until the timeout elapses. The lock is
    transaction-scoped, so it auto-releases when the surrounding transaction
    commits, rolls back, or the session is dropped — no explicit release
    inside this context manager.

    On embedded backends (sqlite, surrealdb-embedded): falls back to an
    in-process ``asyncio.Lock`` keyed by ``namespace_id``. Cross-process
    safety is **not** promised.

    Args:
        session: An open SQLAlchemy ``AsyncSession``. On Postgres the lock
            is scoped to this session's current transaction.
        namespace_id: Stable namespace ID of the dream run.
        timeout_seconds: Maximum time to wait for the lock. ``0`` means
            "try once, fast-fail".

    Raises:
        DreamLockUnavailable: The lock is held by another dream run and
            the timeout elapsed without acquiring it.
    """
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be >= 0")

    dialect_name = session.bind.dialect.name if session.bind is not None else ""

    if dialect_name == "postgresql":
        lock_id = _namespace_lock_id(namespace_id)
        deadline = time.monotonic() + timeout_seconds
        while True:
            result = await session.execute(
                text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
                {"lock_id": lock_id},
            )
            if result.scalar():
                break
            if time.monotonic() >= deadline:
                raise DreamLockUnavailable(namespace_id, timeout_seconds)
            await asyncio.sleep(0.05)
        # Transaction-scoped — auto-release on tx end. Nothing to release here.
        yield
        return

    # Embedded fallback: in-process asyncio.Lock.
    lock = await _get_embedded_lock(namespace_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=timeout_seconds or None)
    except TimeoutError as exc:
        raise DreamLockUnavailable(namespace_id, timeout_seconds) from exc
    try:
        yield
    finally:
        lock.release()
