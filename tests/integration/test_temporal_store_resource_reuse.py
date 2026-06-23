"""Behavioral guard: ``StorageCoordinator.temporal_store`` reuses shared resources.

The temporal-store relocation keeps ``StorageCoordinator.temporal_store`` a thin
factory whose whole job is to hand the new store the coordinator's *existing*
connection so the two never fork a second handle / engine / connection. The unit
tests in ``tests/unit/test_coordinator_temporal_store.py`` pin the forwarded
kwargs with mocks; these tests prove the real thing on a live stack: the store
the coordinator returns holds the SAME underlying resource object the
coordinator already owns (``is``-identity), and two calls return DISTINCT store
objects that nonetheless share that one resource.

The embedded sqlite_lance path runs unconditionally (no external service). The
pgvector path is gated on Postgres reachability and the surrealdb path on its
optional SDK; both skip loudly when their backend is absent rather than passing
silently.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from khora.config import KhoraConfig
from tests.integration._sqlite_lance_fixtures import build_sqlite_lance_coordinator

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

try:
    import surrealdb  # noqa: F401

    _HAS_SURREALDB = True
except ImportError:
    _HAS_SURREALDB = False


_DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if _DATABASE_URL.startswith("postgresql://"):
    _DATABASE_URL = _DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _DATABASE_URL.startswith("postgres://"):
    _DATABASE_URL = _DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


def _pg_reachable() -> bool:
    parsed = urlparse(_DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _config() -> KhoraConfig:
    """A real KhoraConfig — opaque to the sqlite_lance store's connect()."""
    return KhoraConfig(app_name="khora-test", environment="test", debug=True)


# ===========================================================================
# sqlite_lance — embedded, always runs
# ===========================================================================


@pytest.mark.embedded
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed")
class TestSqliteLanceResourceReuse:
    """The embedded temporal store reuses the coordinator's EmbeddedStorageHandle."""

    async def test_store_shares_coordinator_handle(self, tmp_path: Path) -> None:
        """The returned store's ``_handle`` IS the coordinator's vector handle."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            coordinator_handle = coord._vector._handle
            assert coordinator_handle is not None

            store = await coord.temporal_store("sqlite_lance", _config())
            try:
                # The store must reuse the coordinator's single aiosqlite + LanceDB
                # pair — same object, not a freshly-opened second handle.
                assert store._handle is coordinator_handle
            finally:
                await store.disconnect()
        finally:
            await coord.disconnect()

    async def test_two_calls_distinct_stores_same_handle(self, tmp_path: Path) -> None:
        """Two calls yield DISTINCT stores that share the one underlying handle."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            coordinator_handle = coord._vector._handle

            first = await coord.temporal_store("sqlite_lance", _config())
            second = await coord.temporal_store("sqlite_lance", _config())
            try:
                # Factory semantics: not cached — each call is a fresh store.
                assert first is not second
                # ...but both bound to the coordinator's single shared handle.
                assert first._handle is coordinator_handle
                assert second._handle is coordinator_handle
                assert first._handle is second._handle
            finally:
                await first.disconnect()
                await second.disconnect()
        finally:
            await coord.disconnect()


# ===========================================================================
# pgvector — gated on a reachable Postgres
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not _pg_reachable(),
    reason="PostgreSQL not reachable (run `make dev` first)",
)
class TestPgvectorResourceReuse:
    """The pgvector temporal store reuses the coordinator's SQLAlchemy engine."""

    async def test_store_shares_coordinator_engine(self) -> None:
        """The returned store's ``_engine`` IS the coordinator's relational engine.

        Built the way the live-PG conformance leg builds it: one shared
        ``AsyncEngine`` injected into the relational backend, wired onto a bare
        ``StorageCoordinator``. ``temporal_store("pgvector")`` reads
        ``_vector._engine`` first then falls back to ``_relational._engine`` — with
        no vector adapter here it takes the relational engine, which is exactly
        the resource the store must reuse instead of opening its own pool.
        """
        from sqlalchemy.ext.asyncio import create_async_engine

        from khora.storage.backends.postgresql import PostgreSQLBackend
        from khora.storage.coordinator import StorageCoordinator

        engine = create_async_engine(_DATABASE_URL)
        relational = PostgreSQLBackend(_DATABASE_URL, engine=engine)
        coord = StorageCoordinator(relational=relational, vector=None)
        await coord.connect()
        try:
            assert coord._relational._engine is engine

            store = await coord.temporal_store("pgvector", _config())
            try:
                assert store._engine is engine
                # Shared engine: the store must NOT have opened its own pool.
                assert store._shared_engine is True
            finally:
                await store.disconnect()

            # Factory semantics: a second call is a distinct store on the same engine.
            second = await coord.temporal_store("pgvector", _config())
            try:
                assert second is not store
                assert second._engine is engine
            finally:
                await second.disconnect()
        finally:
            await coord.disconnect()
            await engine.dispose()


# ===========================================================================
# surrealdb — gated on the optional SDK
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_SURREALDB, reason="surrealdb SDK not installed")
class TestSurrealDBResourceReuse:
    """The surrealdb temporal store reuses the coordinator's SurrealDBConnection."""

    async def test_store_shares_coordinator_connection(self) -> None:
        """The returned store's ``_conn`` IS the coordinator's relational connection.

        Embedded in-memory SurrealDB (``mode="memory"``) — no docker. The shared
        connection matters most here: embedded ``surrealkv`` allows only one open
        handle per directory, so the store reusing the coordinator's connection
        (rather than opening a second) is the property the relocation must
        preserve. ``temporal_store("surrealdb")`` reads ``_relational._conn`` and
        ``config.storage.surrealdb``.
        """
        from khora.config.schema import SurrealDBConfig
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter
        from khora.storage.coordinator import StorageCoordinator

        conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="temporal_reuse")
        await conn.connect()
        relational = SurrealDBRelationalAdapter(conn)
        coord = StorageCoordinator(relational=relational, vector=None)
        try:
            assert coord._relational._conn is conn

            config = KhoraConfig(app_name="khora-test", environment="test", debug=True)
            config.storage.surrealdb = SurrealDBConfig(mode="memory")

            store = await coord.temporal_store("surrealdb", config)
            try:
                assert store._conn is conn
                # Shared connection: the store must not own (and later close) it.
                assert store._owns_connection is False
            finally:
                await store.disconnect()
        finally:
            await conn.disconnect()
