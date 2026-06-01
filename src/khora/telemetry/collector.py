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
        max_buffer_size: int = 10_000,
    ) -> None:
        self._engine = engine
        self._service_name = service_name
        self._flush_interval = flush_interval
        self._flush_threshold = flush_threshold
        # Cap the in-memory buffer so a sustained DB-write failure (events are
        # re-enqueued on a failed flush, see _flush) can't grow it without
        # bound. On overflow we drop the oldest events with a WARN + counter.
        self._max_buffer_size = max_buffer_size
        self._buffer: deque[tuple[str, dict[str, Any]]] = deque()
        self._flush_task: asyncio.Task[None] | None = None
        self._threshold_flush_task: asyncio.Task[None] | None = None

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
            except Exception as e:
                logger.debug(f"Telemetry schema probe failed (will recreate): {e}")
                needs_recreate = True
            finally:
                await conn.rollback()

        # Step 2: drop old tables if they exist but are outdated
        if needs_recreate:
            try:
                async with self._engine.begin() as conn:
                    await conn.execute(sa.text("SELECT 1 FROM llm_events LIMIT 0"))
                    # Old table exists without trace_id — drop all
                    logger.info(f"Telemetry schema v{SCHEMA_VERSION}: dropping old tables for migration")
                    await conn.run_sync(metadata.drop_all)
            except Exception as e:
                # Tables don't exist at all — fine, create_all will handle it
                logger.debug(f"Old telemetry tables not found (will create fresh): {e}")

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

        # Let any in-flight threshold flush finish before the final drain.
        # _flush swallows its own errors, so the only thing to absorb here is
        # cancellation.
        if self._threshold_flush_task is not None and not self._threshold_flush_task.done():
            try:
                await self._threshold_flush_task
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

    def _maybe_schedule_flush(self) -> None:
        """Schedule an immediate flush when the buffer hits the threshold.

        ``record_*`` methods are synchronous, so we never block here: a flush is
        scheduled as a background task (the same primitive as the timer loop).
        No-op when no event loop is running (the next timer tick flushes
        instead) or when a threshold flush is already in flight.
        """
        if len(self._buffer) < self._flush_threshold:
            return
        if self._threshold_flush_task is not None and not self._threshold_flush_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._threshold_flush_task = loop.create_task(self._flush(), name="telemetry-threshold-flush")

    def record_llm_call(self, **kwargs: Any) -> None:
        # Capture cost_usd before LLMEvent strips unknown kwargs.
        cost_usd = kwargs.pop("cost_usd", None)
        kwargs = self._inject_trace_context(kwargs)
        event = LLMEvent(service_name=self._service_name, **kwargs)
        self._buffer.append(("llm", event.model_dump()))
        # Aggregate OTel metrics — fire on every call regardless of DB flush
        # so SLO dashboards work even when DB telemetry is disabled.
        from .aggregate_metrics import record_llm_call_metrics

        record_llm_call_metrics(
            model=event.model,
            operation=event.operation,
            status=event.status,
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
            cost_usd=cost_usd,
        )
        self._maybe_schedule_flush()

    def record_storage_op(self, **kwargs: Any) -> None:
        kwargs = self._inject_trace_context(kwargs)
        event = StorageEvent(service_name=self._service_name, **kwargs)
        self._buffer.append(("storage", event.model_dump()))
        self._maybe_schedule_flush()

    def record_pipeline_stage(self, **kwargs: Any) -> None:
        kwargs = self._inject_trace_context(kwargs)
        event = PipelineEvent(service_name=self._service_name, **kwargs)
        self._buffer.append(("pipeline", event.model_dump()))
        self._maybe_schedule_flush()

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
        """Batch-insert all buffered events.  Errors are logged, never raised.

        On a transient DB-write failure the drained events are re-enqueued at
        the front of the buffer so the next flush tick retries them instead of
        dropping them (issue #924).  The buffer is capped at *max_buffer_size*:
        on overflow the oldest events are dropped with a WARN + counter so a
        sustained failure can't grow memory without bound.
        """
        if not self._buffer:
            return

        # Drain the buffer, keeping the original ordered tuples so we can
        # re-enqueue them verbatim if the write fails.
        drained: list[tuple[str, dict[str, Any]]] = []
        while self._buffer:
            drained.append(self._buffer.popleft())

        llm_rows: list[dict[str, Any]] = []
        storage_rows: list[dict[str, Any]] = []
        pipeline_rows: list[dict[str, Any]] = []
        for kind, data in drained:
            row = dict(data)
            meta_value = row.pop("metadata", None)
            row["metadata"] = meta_value
            if kind == "llm":
                llm_rows.append(row)
            elif kind == "storage":
                storage_rows.append(row)
            elif kind == "pipeline":
                pipeline_rows.append(row)

        total = len(drained)
        try:
            async with self._engine.begin() as conn:
                if llm_rows:
                    await conn.execute(llm_events.insert(), llm_rows)
                if storage_rows:
                    await conn.execute(storage_events.insert(), storage_rows)
                if pipeline_rows:
                    await conn.execute(pipeline_events.insert(), pipeline_rows)
            logger.debug(f"Telemetry flushed {total} events")
        except Exception as exc:
            # Truncate error to avoid dumping huge SQL with embedding vectors
            err_str = str(exc)
            if len(err_str) > 300:
                err_str = err_str[:300] + "..."
            self._requeue(drained)
            logger.warning(f"Telemetry flush failed ({total} events re-queued for retry): {err_str}")

    def _requeue(self, drained: list[tuple[str, dict[str, Any]]]) -> None:
        """Put a failed batch back on the front of the buffer for retry.

        Records appended since the flush started are kept (they're newer); when
        re-enqueuing would exceed *max_buffer_size* the oldest events are
        dropped with a WARN + ``khora.telemetry.flush.dropped_total`` counter.
        """
        # Newest-first reconstruction: failed batch first (oldest), then
        # whatever was buffered while the flush was in flight.
        combined = drained + list(self._buffer)
        overflow = len(combined) - self._max_buffer_size
        if overflow > 0:
            combined = combined[overflow:]
            from .metrics import metric_counter

            metric_counter(
                "khora.telemetry.flush.dropped_total",
                unit="1",
                description="Telemetry events dropped because the buffer hit max_buffer_size after a flush failure.",
            ).add(overflow)
            logger.warning(
                f"Telemetry buffer full ({self._max_buffer_size}); dropped {overflow} oldest events after flush failure"
            )
        self._buffer.clear()
        self._buffer.extend(combined)
