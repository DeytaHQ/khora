"""Tests for ``DreamEventSink`` — bridge into ``HookDispatcher`` (#666)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models.event import EventType, MemoryEvent
from khora.dream.events import (
    DreamOperationEvent,
    DreamRationale,
    DreamRunStarted,
)
from khora.dream.report.event_sink import DreamEventSink
from khora.hooks.dispatcher import HookDispatcher
from khora.hooks.models import SemanticFilter


def _now() -> datetime:
    return datetime.now(UTC)


def _op_event(*, op_type: str = "dedupe_entities", decision: str = "merge") -> DreamOperationEvent:
    return DreamOperationEvent(
        op_id=uuid4(),
        run_id=uuid4(),
        phase="audit",
        op_type=op_type,
        inputs={},
        outputs={},
        decision=decision,
        rationale=DreamRationale(strategy="cosine_above_threshold", rationale_hash="abcd1234"),
        started_at=_now(),
        duration_ms=5.0,
        namespace_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_sync_delivery_bridges_into_dispatcher() -> None:
    dispatcher = HookDispatcher()
    received: list[MemoryEvent] = []

    async def cb(event: MemoryEvent) -> None:
        received.append(event)

    dispatcher.subscribe(EventType.DREAM_OP_DECIDED, cb)
    sink = DreamEventSink(dispatcher, delivery="sync")

    ev = _op_event()
    await sink.emit(ev)

    assert len(received) == 1
    routed = received[0]
    assert routed.event_type == EventType.DREAM_OP_DECIDED
    assert routed.resource_type == "dream"
    assert routed.data["op_type"] == "dedupe_entities"
    assert routed.data["decision"] == "merge"


@pytest.mark.asyncio
async def test_dream_op_types_filter_narrows_subscribers() -> None:
    dispatcher = HookDispatcher()
    matched: list[MemoryEvent] = []

    async def cb(event: MemoryEvent) -> None:
        matched.append(event)

    dispatcher.subscribe(
        EventType.DREAM_OP_DECIDED,
        cb,
        filter=SemanticFilter(name="only_dedupe", dream_op_types=["dedupe_entities"]),
    )
    sink = DreamEventSink(dispatcher, delivery="sync")

    await sink.emit(_op_event(op_type="dedupe_entities"))
    await sink.emit(_op_event(op_type="prune_edges"))

    assert len(matched) == 1
    assert matched[0].data["op_type"] == "dedupe_entities"


@pytest.mark.asyncio
async def test_dream_decisions_filter_narrows_subscribers() -> None:
    dispatcher = HookDispatcher()
    matched: list[MemoryEvent] = []

    async def cb(event: MemoryEvent) -> None:
        matched.append(event)

    dispatcher.subscribe(
        EventType.DREAM_OP_DECIDED,
        cb,
        filter=SemanticFilter(name="only_merge", dream_decisions=["merge"]),
    )
    sink = DreamEventSink(dispatcher, delivery="sync")
    await sink.emit(_op_event(decision="merge"))
    await sink.emit(_op_event(decision="skip"))

    assert len(matched) == 1
    assert matched[0].data["decision"] == "merge"


@pytest.mark.asyncio
async def test_run_started_event_emitted_with_correct_type() -> None:
    dispatcher = HookDispatcher()
    received: list[MemoryEvent] = []

    async def cb(event: MemoryEvent) -> None:
        received.append(event)

    dispatcher.subscribe(EventType.DREAM_RUN_STARTED, cb)
    sink = DreamEventSink(dispatcher, delivery="sync")

    rid, ns = uuid4(), uuid4()
    await sink.emit(DreamRunStarted(run_id=rid, namespace_id=ns, mode="dry-run", trigger="manual", started_at=_now()))

    assert len(received) == 1
    assert received[0].event_type == EventType.DREAM_RUN_STARTED
    assert received[0].correlation_id == rid


@pytest.mark.asyncio
async def test_bounded_outbox_drops_oldest_on_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overflow on a full queue increments the counter and drops the oldest entry.

    We exercise the overflow branch by pre-filling the internal queue to
    capacity (no worker has been spawned yet because we never called
    ``emit``), then making a single ``emit`` call that has to overflow.
    The synchronous slow callback keeps the worker too busy to drain
    while the second emit lands.
    """
    dispatcher = HookDispatcher()
    sink = DreamEventSink(dispatcher, delivery="async", outbox_maxsize=1, subscription_class="test")

    overflow_calls: list[dict] = []

    from khora.dream.report import event_sink as ev_module

    class Spy:
        def add(self, n, *, attributes=None):  # noqa: ARG002
            overflow_calls.append(attributes or {})

    monkeypatch.setattr(ev_module, "_OVERFLOW_COUNTER", Spy())

    # Pre-fill the queue with a stale event. No worker is running yet
    # because emit() hasn't been called.
    stale = _stub_memory_event()
    sink._outbox.put_nowait(stale)

    # Now the queue is full. The next emit must trigger the overflow
    # branch (drop oldest + counter add) before enqueueing the new event.
    await sink.emit(_op_event())

    assert overflow_calls, "expected at least one overflow counter increment"
    assert overflow_calls[0].get("subscription_class") == "test"

    await sink.close()


def _stub_memory_event() -> MemoryEvent:
    """Helper: a MemoryEvent shaped like one the sink would produce."""
    return MemoryEvent(
        namespace_id=uuid4(),
        event_type=EventType.DREAM_OP_DECIDED,
        resource_type="dream",
        resource_id=uuid4(),
        data={"op_type": "dedupe_entities", "decision": "merge"},
    )
