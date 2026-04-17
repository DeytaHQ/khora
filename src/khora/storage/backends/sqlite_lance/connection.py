"""Connection handle for the SQLite + LanceDB embedded unified backend.

``EmbeddedStorageHandle`` owns the aiosqlite connection and the LanceDB
async connection used by all four adapter roles (graph, vector,
relational, event store). The handle is created once and shared across
adapters by the factory — adapters must not open their own connections.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

from .schema import ensure_lance_tables

if TYPE_CHECKING:
    import aiosqlite
    from lancedb.db import AsyncConnection as LanceAsyncConnection


# Module-level lock to serialize schema initialization across adapters
# sharing the same embedded DB. The StorageFactory may instantiate
# multiple adapters concurrently; LanceDB's create_table is idempotent
# but we serialize to avoid interleaved writes to the Lance catalog.
_schema_init_lock = asyncio.Lock()


# SQLite pragmas tuned for concurrent reads with occasional writes.
# journal_mode=WAL allows readers and writers to coexist.
# busy_timeout handles SQLITE_BUSY by waiting up to 5s instead of erroring.
# mmap_size=256 MiB accelerates reads on warm pages.
# cache_size=-64000 => 64 MiB page cache (negative = KiB).
_SQLITE_PRAGMAS: tuple[tuple[str, str], ...] = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
    ("mmap_size", "268435456"),
    ("cache_size", "-64000"),
    ("temp_store", "MEMORY"),
    ("busy_timeout", "5000"),
)


@dataclass
class EmbeddedStorageHandleConfig:
    """Configuration for ``EmbeddedStorageHandle``.

    Mirrors the user-facing ``SQLiteLanceConfig`` but strips the
    discriminator and resolves defaults the connection actually needs.
    """

    db_path: str
    lance_path: str | None = None
    embedding_dimension: int = 1536
    use_halfvec: bool = False
    lance_index: Literal["auto", "ivf_pq", "hnsw", "brute"] = "auto"
    ivf_partitions: int | None = None
    hnsw_m: int = 16


class EmbeddedStorageHandle:
    """Shared connection handle for SQLite + LanceDB.

    One instance is created per (db_path, lance_path) pair and shared
    across the graph / vector / relational / event-store adapters.
    Opens both connections lazily on :meth:`connect`.
    """

    def __init__(self, config: EmbeddedStorageHandleConfig) -> None:
        self._config = config
        self._sqlite: aiosqlite.Connection | None = None
        self._lance: LanceAsyncConnection | None = None
        self._connected = False
        self._schema_initialized = False
        # Serializes connect / disconnect so concurrent adapter lifecycle
        # calls (via ``StorageCoordinator.connect()`` / ``disconnect()``
        # ``asyncio.gather``) don't double-open or double-close. Created
        # lazily on first use so the handle can be instantiated outside
        # an event loop.
        self._lifecycle_lock: asyncio.Lock | None = None

    def _get_lifecycle_lock(self) -> asyncio.Lock:
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()
        return self._lifecycle_lock

    @property
    def config(self) -> EmbeddedStorageHandleConfig:
        return self._config

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sqlite(self) -> aiosqlite.Connection:
        if self._sqlite is None:
            raise RuntimeError("EmbeddedStorageHandle is not connected (sqlite)")
        return self._sqlite

    @property
    def lance(self) -> LanceAsyncConnection:
        if self._lance is None:
            raise RuntimeError("EmbeddedStorageHandle is not connected (lance)")
        return self._lance

    def _resolve_lance_path(self) -> str:
        """Resolve the LanceDB directory path.

        When ``lance_path`` is None, derive a sibling ``.lance`` directory
        next to the SQLite file (e.g. ``./khora.db`` -> ``./khora.lance``).
        ``:memory:`` SQLite databases still require an on-disk LanceDB
        path since LanceDB has no in-memory mode.
        """
        if self._config.lance_path:
            return self._config.lance_path

        db_path = self._config.db_path
        if db_path == ":memory:" or db_path.startswith("file::memory:"):
            return "./khora.lance"

        p = Path(db_path)
        return str(p.with_suffix(".lance"))

    async def connect(self) -> None:
        """Open aiosqlite + LanceDB connections and initialize schema.

        Idempotent — subsequent calls are no-ops. Safe to call
        concurrently from multiple adapters; the lifecycle lock
        serializes the open.
        """
        async with self._get_lifecycle_lock():
            if self._connected:
                return

            import aiosqlite
            import lancedb

            sqlite_path = self._config.db_path
            lance_path = self._resolve_lance_path()
            logger.info(
                f"Opening embedded storage: sqlite={sqlite_path} lance={lance_path}",
            )

            self._sqlite = await aiosqlite.connect(sqlite_path)
            # Enable dict-like row access so adapters can use row["col"].
            self._sqlite.row_factory = aiosqlite.Row
            for pragma, value in _SQLITE_PRAGMAS:
                await self._sqlite.execute(f"PRAGMA {pragma}={value}")
            await self._sqlite.commit()

            self._lance = await lancedb.connect_async(lance_path)

            self._connected = True

            if not self._schema_initialized:
                async with _schema_init_lock:
                    if not self._schema_initialized:
                        await ensure_lance_tables(
                            self._lance,
                            self._config.embedding_dimension,
                            self._config.use_halfvec,
                        )
                        self._schema_initialized = True

            logger.info("Embedded storage connected")

    async def disconnect(self) -> None:
        """Close both connections. Safe to call concurrently and multiple times.

        When the coordinator disconnects all four adapters in parallel,
        several of them call ``handle.disconnect()`` at once. Without
        this lock, both callers can enter ``_sqlite.close()`` before
        either sets ``_sqlite = None``, which double-closes the aiosqlite
        connection and hangs its worker thread.
        """
        async with self._get_lifecycle_lock():
            if not self._connected and self._sqlite is None and self._lance is None:
                return

            if self._sqlite is not None:
                try:
                    await self._sqlite.close()
                except Exception:
                    logger.debug("Error closing aiosqlite connection (may already be closed)")
                self._sqlite = None

            # LanceDB AsyncConnection doesn't expose an explicit close in current
            # versions; dropping the reference releases native resources.
            self._lance = None

            self._connected = False
            logger.info("Embedded storage disconnected")

    async def is_healthy(self) -> bool:
        """Ping both backends. Returns True only if both respond."""
        if not self._connected or self._sqlite is None or self._lance is None:
            return False
        try:
            async with self._sqlite.execute("SELECT 1") as cursor:
                row = await cursor.fetchone()
                if row is None or row[0] != 1:
                    return False
            await self._lance.list_tables()
            return True
        except Exception as exc:
            logger.debug(f"Embedded storage health check failed: {exc}")
            return False
