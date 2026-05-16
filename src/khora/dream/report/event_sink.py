"""Event sink — bridges dream report events into :class:`HookDispatcher`.

Every :class:`DreamReportEvent` becomes a :class:`MemoryEvent` with a
matching :class:`EventType.DREAM_*` value. The existing dispatcher's
filter cascade (Level 0 type / Level 1 embedding / Level 2 LLM) and the
async semaphore / per-callback timeout protections are reused — no
parallel pub/sub is introduced.

Sync vs async delivery: by default the sink awaits
``dispatcher.dispatch`` directly (default async fan-out semantics from
the dispatcher). When ``delivery="sync"`` is selected by a subscription,
the dispatcher's per-callback timeout still applies; the only difference
is the sink does not return until every subscribed callback completes.

Bounded outbox: a per-sink :class:`asyncio.Queue` (default
``maxsize=10_000``) absorbs short bursts when the dispatcher is slow.
Overflow drops the oldest queued event (drop_oldest policy) and
increments :data:`_OVERFLOW_COUNTER` — operators get a metric without
the sink growing unbounded.
"""

from __future__ import annotations

import asyncio
from typing import Literal
from uuid import UUID, uuid4

from khora.core.models.event import EventType, MemoryEvent
from khora.dream.events import (
    DreamOperationEvent,
    DreamPhaseCompleted,
    DreamPhaseStarted,
    DreamReportEvent,
    DreamRunCompleted,
    DreamRunFailed,
    DreamRunStarted,
)
from khora.dream.report.base import ReportSink
from khora.hooks.dispatcher import HookDispatcher
from khora.telemetry.metrics import metric_counter

_OVERFLOW_COUNTER = metric_counter(
    "khora.dream.subscription.overflow_total",
    description="Dream event-sink outbox drops (queue full, oldest dropped).",
)


_EVENT_TYPE_BY_PAYLOAD: dict[type[DreamReportEvent], EventType] = {
    DreamRunStarted: EventType.DREAM_RUN_STARTED,
    DreamPhaseStarted: EventType.DREAM_PHASE_STARTED,
    DreamOperationEvent: EventType.DREAM_OP_DECIDED,
    DreamPhaseCompleted: EventType.DREAM_PHASE_COMPLETED,
    DreamRunCompleted: EventType.DREAM_RUN_COMPLETED,
    DreamRunFailed: EventType.DREAM_RUN_FAILED,
}


def _resource_id_for(event: DreamReportEvent) -> UUID:
    """Pick a stable resource id per payload kind.

    Op events use ``op_id``; everything else uses ``run_id`` so callbacks
    can group per-run events by ``resource_id``.
    """
    if isinstance(event, DreamOperationEvent):
        return event.op_id
    return event.run_id  # type: ignore[union-attr]


class DreamEventSink(ReportSink):
    """Bridge :class:`DreamReportEvent` → :class:`MemoryEvent` via dispatcher."""

    def __init__(
        self,
        dispatcher: HookDispatcher,
        *,
        delivery: Literal["sync", "async"] = "async",
        outbox_maxsize: int = 10_000,
        subscription_class: str = "dream",
    ) -> None:
        self._dispatcher = dispatcher
        self._delivery = delivery
        self._subscription_class = subscription_class
        self._outbox: asyncio.Queue[MemoryEvent] = asyncio.Queue(maxsize=outbox_maxsize)
        self._worker: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # ReportSink interface
    # ------------------------------------------------------------------

    async def emit(self, event: DreamReportEvent) -> None:
        mem_event = self._to_memory_event(event)
        if self._delivery == "sync":
            await self._dispatcher.dispatch(mem_event)
            return

        # async: enqueue with drop_oldest on full.
        if self._outbox.full():
            try:
                _ = self._outbox.get_nowait()
                # Balance the put → task_done accounting on Queue so that
                # ``flush()`` / ``close()`` don't wait for a phantom item.
                self._outbox.task_done()
            except asyncio.QueueEmpty:
                pass
            _OVERFLOW_COUNTER.add(
                1,
                attributes={"subscription_class": self._subscription_class},
            )
        await self._outbox.put(mem_event)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain())

    async def flush(self) -> None:
        if self._worker is not None and not self._worker.done():
            await self._outbox.join()

    async def close(self) -> None:
        await self.flush()
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                # Expected outcome of the cancel above.
                pass
        self._worker = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _drain(self) -> None:
        while True:
            try:
                mem_event = await self._outbox.get()
            except asyncio.CancelledError:
                return
            try:
                await self._dispatcher.dispatch(mem_event)
            finally:
                self._outbox.task_done()

    def _to_memory_event(self, event: DreamReportEvent) -> MemoryEvent:
        event_type = _EVENT_TYPE_BY_PAYLOAD[type(event)]
        data = event.model_dump(mode="json")
        return MemoryEvent(
            id=uuid4(),
            namespace_id=event.namespace_id,
            event_type=event_type,
            resource_type="dream",
            resource_id=_resource_id_for(event),
            data=data,
            correlation_id=event.run_id,  # type: ignore[union-attr]
        )


__all__ = ["DreamEventSink"]
