"""TelemetryCollector -- async buffered writer for telemetry events."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine

from .models import LLMEvent, PipelineEvent, StorageEvent
from .tables import SCHEMA_VERSION, llm_events, metadata, pipeline_events, storage_events


class TelemetryCollector:
    """Buffers telemetry events and periodically flushes them to PostgreSQL.

    All ``record_*`` methods are **synchronous** -- they append to an in-memory
    deque so callers never block on I/O.  A background ``asyncio.Task`` drains
    the buffer every *flush_interval* seconds or when the buffer exceeds
    *flush_threshold* events.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        service_name: str = "khora",
        flush_interval: float = 5.0,
        flush_threshold: int = 100,
    ) -> None:
        self._engine = engine
        self._service_name = service_name
        self._flush_interval = flush_interval
        self._flush_threshold = flush_threshold
        self._buffer: deque[tuple[str, dict[str, Any]]] = deque()
        self._flush_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create tables (if missing) and start the background flush loop.

        On startup, checks if the schema version has changed.  If so,
        drops the old tables and recreates them with the new schema.
        """
        try:
            await self._migrate_schema()
            logger.info("Telemetry tables ensured")
        except Exception as exc:
            logger.warning(f"Telemetry table creation failed (non-fatal): {exc}")

        self._flush_task = asyncio.create_task(self._flush_loop(), name="telemetry-flush")

    async def _migrate_schema(self) -> None:
        """Detect schema version and recreate tables if needed.

        Uses separate connections for probing and DDL to avoid aborted
        transaction issues with asyncpg.
        """
        import sqlalchemy as sa

        # Step 1: probe whether the current schema is up-to-date
        needs_recreate = False
        async with self._engine.connect() as conn:
            try:
                result = await conn.execute(sa.text("SELECT trace_id FROM llm_events LIMIT 0"))
                result.close()
            except Exception:
                needs_recreate = True
            finally:
                await conn.rollback()

        # Step 2: drop old tables if they exist but are outdated
        if needs_recreate:
            async with self._engine.begin() as conn:
                try:
                    await conn.execute(sa.text("SELECT 1 FROM llm_events LIMIT 0"))
                    # Old table exists without trace_id — drop all
                    logger.info(f"Telemetry schema v{SCHEMA_VERSION}: dropping old tables for migration")
                    await conn.run_sync(metadata.drop_all)
                except Exception:
                    # Tables don't exist at all — fine, create_all will handle it
                    await conn.rollback()

        # Step 3: create tables (no-op if already up-to-date)
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def shutdown(self) -> None:
        """Cancel the flush loop, do a final flush, and dispose the engine."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final drain
        await self._flush()

        await self._engine.dispose()
        logger.info("Telemetry collector shut down")

    # ------------------------------------------------------------------
    # Record helpers (sync -- safe to call from anywhere)
    # ------------------------------------------------------------------

    def _inject_trace_context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Auto-populate trace_id and parent_event_id from context vars."""
        from .context import get_parent_event_id, get_trace_id

        if "trace_id" not in kwargs or kwargs["trace_id"] is None:
            kwargs["trace_id"] = get_trace_id()
        if "parent_event_id" not in kwargs or kwargs["parent_event_id"] is None:
            kwargs["parent_event_id"] = get_parent_event_id()
        return kwargs

    def record_llm_call(self, **kwargs: Any) -> None:
        kwargs = self._inject_trace_context(kwargs)
        event = LLMEvent(service_name=self._service_name, **kwargs)
        self._buffer.append(("llm", event.model_dump()))

    def record_storage_op(self, **kwargs: Any) -> None:
        kwargs = self._inject_trace_context(kwargs)
        event = StorageEvent(service_name=self._service_name, **kwargs)
        self._buffer.append(("storage", event.model_dump()))

    def record_pipeline_stage(self, **kwargs: Any) -> None:
        kwargs = self._inject_trace_context(kwargs)
        event = PipelineEvent(service_name=self._service_name, **kwargs)
        self._buffer.append(("pipeline", event.model_dump()))

    # ------------------------------------------------------------------
    # Internal flush machinery
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Background loop that flushes periodically or when threshold hit."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                if self._buffer:
                    await self._flush()
        except asyncio.CancelledError:
            return

    async def _flush(self) -> None:
        """Batch-insert all buffered events.  Errors are logged, never raised."""
        if not self._buffer:
            return

        # Drain the buffer into local lists
        llm_rows: list[dict[str, Any]] = []
        storage_rows: list[dict[str, Any]] = []
        pipeline_rows: list[dict[str, Any]] = []

        while self._buffer:
            kind, data = self._buffer.popleft()
            row = dict(data)
            meta_value = row.pop("metadata", None)
            row["metadata"] = meta_value
            if kind == "llm":
                llm_rows.append(row)
            elif kind == "storage":
                storage_rows.append(row)
            elif kind == "pipeline":
                pipeline_rows.append(row)

        try:
            async with self._engine.begin() as conn:
                if llm_rows:
                    await conn.execute(llm_events.insert(), llm_rows)
                if storage_rows:
                    await conn.execute(storage_events.insert(), storage_rows)
                if pipeline_rows:
                    await conn.execute(pipeline_events.insert(), pipeline_rows)
            total = len(llm_rows) + len(storage_rows) + len(pipeline_rows)
            logger.debug(f"Telemetry flushed {total} events")
        except Exception as exc:
            logger.warning(f"Telemetry flush failed (events dropped): {exc}")
