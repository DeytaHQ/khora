"""Throwaway Postgres databases for migration lifecycle tests.

A migration test that walks the chain down and back up must not do it against
the shared dev database. CI runs the integration job with
``--timeout-method=thread``, which kills the process outright — ``finally``
blocks do not run. A rewind-then-restore against the shared database would
therefore strand it at the previous revision on timeout, and every later test
in the serial job would fail against a stale schema with the real cause several
tests back. Owning the database removes the hazard rather than narrowing the
window: the worst a timeout can do is leak one uniquely named database.

This module is the single copy of that machinery. It was first written inline
in ``test_migration_054_documents_namespace_created_at_id.py``; a second
migration needing it is the point at which one copy beats two, since the skip
policy below is the kind of thing that must not be allowed to diverge between
files.

Skip policy
-----------
Exactly two conditions skip:

* the server is unreachable at the socket level (``pg_reachable``), and
* the role lacks ``CREATEDB`` (SQLSTATE ``42501``).

Both are environment limitations. **Everything else re-raises** — bad
credentials, an unparseable DSN, a driver fault, or a genuine migration failure
all mean the test could not run for a reason worth seeing. A skip reads as
"fine here", so suppressing a real failure into one turns a red lane green and
is strictly worse than no test at all.
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

__all__ = [
    "DATABASE_URL",
    "maintenance_url",
    "pg_reachable",
    "scratch_database",
    "skip_only_if_cannot_create_database",
    "sqlstates",
    "with_database",
]


def _normalize_async_dsn(url: str) -> str:
    """Force the asyncpg driver onto a bare Postgres DSN.

    ``create_async_engine`` needs an async driver in the scheme. A DSN carrying
    a *sync* driver (``postgresql+psycopg``, ``+psycopg2``, ``+pg8000``) is not
    merely missing the marker — it names a driver that cannot be used here, so
    rewriting the whole ``postgresql+<driver>`` prefix is the only handling
    that works. Leaving it would surface as ``InvalidRequestError: The asyncio
    extension requires an async driver`` well away from the DSN that caused it.
    An explicit ``+asyncpg`` is already correct and is left alone.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix) :]
    if url.startswith("postgresql+") and not url.startswith("postgresql+asyncpg://"):
        return "postgresql+asyncpg://" + url.split("://", 1)[1]
    return url


DATABASE_URL = _normalize_async_dsn(
    os.environ.get("KHORA_DATABASE_URL", "postgresql+asyncpg://khora:khora@localhost:5434/khora")
)

#: Postgres ``insufficient_privilege``. The only ``CREATE DATABASE`` failure
#: that justifies a skip: a role without CREATEDB is an environment limitation,
#: not a defect.
INSUFFICIENT_PRIVILEGE = "42501"


def pg_reachable(url: str = DATABASE_URL) -> bool:
    """True when something accepts TCP at the DSN's host:port.

    Deliberately only a socket probe — authentication and privilege failures
    must surface as failures, not as "Postgres isn't running".
    """
    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def sqlstates(exc: BaseException) -> set[str]:
    """Every SQLSTATE attached to *exc* or to anything it wraps.

    SQLAlchemy wraps the driver's exception (asyncpg carries ``sqlstate``) in a
    ``DBAPIError`` reachable via ``orig``, and re-raises can add ``__cause__`` /
    ``__context__`` links. Matching on message text instead would break the
    moment a driver reworded it, so the whole chain is searched for the code.
    """
    states: set[str] = set()
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        err = stack.pop()
        if err is None or id(err) in seen:
            continue
        seen.add(id(err))
        state = getattr(err, "sqlstate", None)
        if isinstance(state, str):
            states.add(state)
        stack.extend([getattr(err, "orig", None), err.__cause__, err.__context__])
    return states


def skip_only_if_cannot_create_database(exc: BaseException) -> None:
    """Re-raise unless *exc* is a missing-CREATEDB-privilege failure."""
    if INSUFFICIENT_PRIVILEGE not in sqlstates(exc):
        raise exc
    pytest.skip(f"role lacks CREATEDB, cannot create a throwaway database: {exc}")


def with_database(url: str, db_name: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def maintenance_url(url: str = DATABASE_URL) -> str:
    """The same server, pointed at the default maintenance database.

    ``CREATE DATABASE`` cannot run from inside the database being created, and
    cannot run inside a transaction either — hence the AUTOCOMMIT connections
    below.
    """
    return with_database(url, "postgres")


async def _create_database(maint_url: str, db_name: str) -> None:
    engine = create_async_engine(maint_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.exec_driver_sql(f'CREATE DATABASE "{db_name}"')
    finally:
        await engine.dispose()


async def _drop_database(maint_url: str, db_name: str) -> None:
    engine = create_async_engine(maint_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            # Alembic's own connections are closed by now, but a lingering
            # backend would make DROP DATABASE fail; evict any that remain.
            await conn.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :db AND pid <> pg_backend_pid()"
                ),
                {"db": db_name},
            )
            await conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        await engine.dispose()


@contextmanager
def scratch_database(tag: str, *, url: str = DATABASE_URL) -> Iterator[str]:
    """Create a uniquely named database, yield its DSN, drop it on exit.

    *tag* identifies the owning test in the database name so a leaked one is
    traceable — pass something like ``"mig055"``. Skips only when the role
    lacks CREATEDB; every other failure propagates.
    """
    maint = maintenance_url(url)
    db_name = f"khora_{tag}_{uuid.uuid4().hex[:8]}"

    try:
        asyncio.run(_create_database(maint, db_name))
    except Exception as exc:  # pragma: no cover - depends on role privileges
        skip_only_if_cannot_create_database(exc)

    try:
        yield with_database(url, db_name)
    finally:
        asyncio.run(_drop_database(maint, db_name))
