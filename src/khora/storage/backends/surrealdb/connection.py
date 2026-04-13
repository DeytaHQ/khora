"""SurrealDB connection manager for Khora."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

# Module-level lock prevents concurrent schema initialization from
# multiple SurrealDBConnection instances (the StorageCoordinator
# connects 4 backends in parallel, all sharing the same embedded DB).
_schema_init_lock = asyncio.Lock()


class SurrealDBConnection:
    """Manages the lifecycle of a SurrealDB client connection.

    Supports three modes:
    - memory: In-process ephemeral database (for testing)
    - embedded: File-backed persistent database (zero infrastructure)
    - remote: WebSocket connection to a SurrealDB server
    """

    def __init__(
        self,
        *,
        mode: str = "memory",
        path: str | None = None,
        url: str | None = None,
        namespace: str = "khora",
        database: str = "default",
        user: str = "root",
        password: str = "root",  # noqa: S107
        sync_data: bool = True,
    ) -> None:
        self._mode = mode
        self._path = path
        self._url = url
        self._namespace = namespace
        self._database = database
        self._user = user
        self._password = password
        self._sync_data = sync_data
        self._client: Any = None
        self._connected = False
        self._schema_initialized = False

        # Write semaphore for embedded/memory modes: SurrealDB's surrealkv
        # engine can panic under high concurrent HNSW index mutations.
        # Semaphore(3) allows enough parallelism for the read-modify-write
        # entity upsert pattern while preventing thundering-herd writes.
        # Combined with retry-on-conflict, this balances throughput and safety.
        # Remote mode doesn't need this — the server serializes internally.
        self._write_sem: asyncio.Semaphore | None = asyncio.Semaphore(3) if mode in ("embedded", "memory") else None

        if not sync_data:
            logger.warning("SurrealDB running without sync_data — data may be lost on crash")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def client(self) -> Any:
        return self._client

    def _build_endpoint(self) -> str:
        if self._mode == "memory":
            return "memory://default"
        if self._mode == "embedded":
            if not self._path:
                raise ValueError("SurrealDB embedded mode requires 'path' to be set")
            return f"surrealkv://{self._path}"
        if self._mode == "remote":
            if not self._url:
                raise ValueError("SurrealDB remote mode requires 'url' to be set")
            return self._url
        raise ValueError(f"Unknown SurrealDB mode: {self._mode}")

    async def connect(self) -> None:
        if self._connected:
            return

        if self._sync_data:
            os.environ["SURREAL_SYNC_DATA"] = "true"

        import surrealdb

        endpoint = self._build_endpoint()
        logger.info(f"Connecting to SurrealDB ({self._mode}): {endpoint}")
        # Use getattr to create the client so ty does not infer the union
        # return type of AsyncSurreal() (whose HTTP variant .connect(url)
        # has a required url param that conflicts with the WS variant).
        factory = getattr(surrealdb, "AsyncSurreal")
        self._client = factory(endpoint)
        await self._client.connect()
        await self._client.use(self._namespace, self._database)
        # Embedded/memory modes don't have a root user — only sign in for remote
        if self._mode == "remote":
            await self._client.signin({"username": self._user, "password": self._password})
        self._connected = True
        logger.info(f"Connected to SurrealDB ({self._mode}), ns={self._namespace}, db={self._database}")

        # Auto-initialize schema (idempotent, skipped on reconnect).
        # The lock prevents concurrent connections from running DEFINE
        # statements in parallel, which causes SurrealDB transaction
        # conflicts on embedded mode.
        if not self._schema_initialized:
            async with _schema_init_lock:
                if not self._schema_initialized:
                    from .schema import initialize_schema

                    await initialize_schema(self)
                    self._schema_initialized = True

    async def disconnect(self) -> None:
        if self._client and self._connected:
            try:
                await self._client.close()
            except Exception:
                logger.debug("Error closing SurrealDB client (may already be closed)")
            self._connected = False
            self._client = None
            logger.info("Disconnected from SurrealDB")

    async def is_healthy(self) -> bool:
        if not self._connected or not self._client:
            return False
        try:
            result = await self._client.query("RETURN 1")
            return result is not None
        except Exception:
            return False

    _WRITE_CONFLICT_MSG = "read or write conflict"
    _MAX_CONFLICT_RETRIES = 5
    _CONFLICT_BASE_DELAY = 0.05  # 50ms, doubles each retry

    async def _execute_raw(self, sql: str, bindings: dict[str, Any] | None = None) -> Any:
        """Execute a SurrealQL statement with write serialization and retry.

        For embedded/memory modes, acquires a semaphore to prevent concurrent
        HNSW index mutations (which cause Rust panics in surrealkv).  Also
        retries on transaction conflicts with exponential backoff.
        """
        if not self._connected:
            raise RuntimeError("SurrealDB not connected")

        async def _do_query() -> Any:
            last_err: Exception | None = None
            for attempt in range(self._MAX_CONFLICT_RETRIES):
                try:
                    return await self._client.query(sql, bindings or {})
                except Exception as e:
                    if self._WRITE_CONFLICT_MSG in str(e) and attempt < self._MAX_CONFLICT_RETRIES - 1:
                        delay = self._CONFLICT_BASE_DELAY * (2**attempt)
                        logger.debug(f"SurrealDB write conflict (attempt {attempt + 1}), retrying in {delay:.3f}s")
                        await asyncio.sleep(delay)
                        last_err = e
                    else:
                        raise
            assert last_err is not None  # guaranteed by loop with retries > 0
            raise last_err

        if self._write_sem is not None:
            async with self._write_sem:
                return await _do_query()
        return await _do_query()

    async def query(self, sql: str, bindings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result = await self._execute_raw(sql, bindings)
        if isinstance(result, list):
            # SurrealDB returns list of statement results; flatten
            flat: list[dict[str, Any]] = []
            for item in result:
                if isinstance(item, dict):
                    flat.append(item)
                elif isinstance(item, list):
                    flat.extend(item)
            return flat
        if isinstance(result, dict):
            return [result]
        return []

    async def query_one(self, sql: str, bindings: dict[str, Any] | None = None) -> dict[str, Any] | None:
        results = await self.query(sql, bindings)
        return results[0] if results else None

    async def execute(self, sql: str, bindings: dict[str, Any] | None = None) -> Any:
        """Execute a SurrealQL statement, returning raw result."""
        return await self._execute_raw(sql, bindings)
