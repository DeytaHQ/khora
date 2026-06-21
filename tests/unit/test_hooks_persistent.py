"""Unit tests for persistent hook subscriptions (#599, Phase 3).

Covers:
- flag-off: no subscription_store => register_persistent refuses, dispatch
  stays pure in-memory (no DB calls).
- round-trip + restart: persist a subscription, build a *fresh* dispatcher
  against the same store, load_persistent, replay an event, observe the
  delivery sink fire (the worker path).
- ADR-001 degrade: a store that raises on persist keeps the subscription in
  memory and records a Degradation rather than crashing or dropping it.
- filter (de)serialization round-trips the persisted fields.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from khora.core.models.event import EventType, MemoryEvent
from khora.hooks.dispatcher import HookDispatcher
from khora.hooks.models import SemanticFilter
from khora.hooks.subscription_store import (
    HookSubscriptionStore,
    PersistentSubscription,
    deserialize_filter,
    serialize_filter,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# In-memory SQLite store fixture (mirrors the embedded sqlite_lance shape)
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE khora_hook_subscriptions (
    id TEXT PRIMARY KEY,
    namespace_id TEXT,
    event_type TEXT NOT NULL,
    filter JSON,
    delivery JSON NOT NULL,
    created_at DATETIME NOT NULL,
    last_delivered_at DATETIME,
    delivery_failure_count INTEGER NOT NULL DEFAULT 0,
    paused_at DATETIME
)
"""


@pytest.fixture
async def session_factory():
    # A shared in-memory SQLite DB the store and a "restarted" dispatcher
    # both bind to. ``cache=shared`` keeps the DB alive across connections.
    engine = create_async_engine("sqlite+aiosqlite:///file:hooks?mode=memory&cache=shared&uri=true")
    async with engine.begin() as conn:
        await conn.execute(sa.text(_CREATE_TABLE))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _event(namespace_id=None):
    return MemoryEvent.entity_created(
        namespace_id=namespace_id or uuid4(),
        entity_id=uuid4(),
        data={"name": "Acme Corp", "entity_type": "ORGANIZATION"},
    )


# ---------------------------------------------------------------------------
# Filter serialization
# ---------------------------------------------------------------------------


class TestFilterSerialization:
    def test_round_trip(self) -> None:
        ns = uuid4()
        filt = SemanticFilter(
            name="competitors",
            description="Mentions of competitors",
            entity_types=["ORGANIZATION", "PRODUCT"],
            examples=["Acme released a widget"],
            similarity_threshold=0.7,
            namespace_id=ns,
        )
        blob = serialize_filter(filt)
        # Embeddings are intentionally dropped.
        assert "embedding" not in blob
        restored = deserialize_filter(blob, namespace_id=ns)
        assert restored.id == filt.id
        assert restored.name == "competitors"
        assert restored.entity_types == ["ORGANIZATION", "PRODUCT"]
        assert restored.examples == ["Acme released a widget"]
        assert restored.similarity_threshold == 0.7
        assert restored.namespace_id == ns


# ---------------------------------------------------------------------------
# Flag-off: pure in-memory, no DB
# ---------------------------------------------------------------------------


class TestFlagOff:
    async def test_register_persistent_requires_store(self) -> None:
        d = HookDispatcher()  # no subscription_store
        with pytest.raises(RuntimeError, match="subscription_store"):
            await d.register_persistent(EventType.ENTITY_CREATED, {"url": "https://x"})

    async def test_in_memory_dispatch_unaffected(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb)
        count = await d.dispatch(_event())
        assert count == 1
        cb.assert_awaited_once()
        assert d.persistent_count == 0


# ---------------------------------------------------------------------------
# Round-trip + restart simulation (the acceptance criterion)
# ---------------------------------------------------------------------------


class TestPersistAndRestart:
    async def test_subscription_durable_across_restart(self, session_factory) -> None:
        store = HookSubscriptionStore(session_factory)

        # Process #1: register a persistent subscription with a delivery target.
        d1 = HookDispatcher(subscription_store=store)
        sub_id = await d1.register_persistent(
            EventType.ENTITY_CREATED,
            {"type": "webhook", "url": "https://example.test/hook"},
        )
        assert d1.persistent_count == 1

        # --- simulate restart: a brand-new dispatcher, no in-memory state ---
        delivered: list = []

        async def sink(sub: PersistentSubscription, event: MemoryEvent) -> None:
            delivered.append((sub.id, event.resource_id))

        d2 = HookDispatcher(subscription_store=store, delivery_sink=sink)
        assert d2.persistent_count == 0  # nothing loaded yet
        loaded = await d2.load_persistent()
        assert loaded == 1
        assert d2.persistent_count == 1

        # An event delivered after the "restart" still finds the subscriber.
        event = _event()
        await d2.dispatch(event)
        assert delivered == [(sub_id, event.resource_id)]

    async def test_persistent_respects_filter(self, session_factory) -> None:
        store = HookSubscriptionStore(session_factory)
        d = HookDispatcher(subscription_store=store)
        # Filter only PRODUCT entities; our event is an ORGANIZATION.
        await d.register_persistent(
            EventType.ENTITY_CREATED,
            {"url": "https://x"},
            filter=SemanticFilter(name="products", entity_types=["PRODUCT"]),
        )

        delivered: list = []
        d2 = HookDispatcher(
            subscription_store=store,
            delivery_sink=lambda s, e: delivered.append(s.id),  # type: ignore[arg-type,return-value]
        )
        await d2.load_persistent()
        # ORGANIZATION event must not match the PRODUCT-only filter.
        await d2.dispatch(_event())
        assert delivered == []

    async def test_unregister_persistent_deletes_row(self, session_factory) -> None:
        store = HookSubscriptionStore(session_factory)
        d = HookDispatcher(subscription_store=store)
        sub_id = await d.register_persistent(EventType.ENTITY_CREATED, {"url": "https://x"})
        assert await d.unregister_persistent(sub_id) is True
        assert d.persistent_count == 0
        # A fresh dispatcher loads nothing.
        d2 = HookDispatcher(subscription_store=store)
        assert await d2.load_persistent() == 0


# ---------------------------------------------------------------------------
# ADR-001 degrade-on-persist-failure
# ---------------------------------------------------------------------------


class _RaisingStore:
    async def persist(self, sub) -> None:
        raise RuntimeError("db down")

    async def load_all(self):
        raise RuntimeError("db down")

    async def delete(self, sid) -> bool:
        raise RuntimeError("db down")


class TestDegradeOnFailure:
    async def test_persist_failure_keeps_in_memory_and_records_degradation(self) -> None:
        d = HookDispatcher(subscription_store=_RaisingStore())
        sub_id = await d.register_persistent(EventType.ENTITY_CREATED, {"url": "https://x"})
        # The subscription is NOT dropped — it lives in memory.
        assert d.persistent_count == 1
        assert sub_id in d._persistent_by_id
        # ADR-001: a structured Degradation is recorded.
        deg = d._last_persist_degradation
        assert deg is not None
        assert deg["component"] == "hooks.subscription_store"
        assert deg["reason"] == "persist_failed"

    async def test_load_failure_degrades_to_zero(self) -> None:
        d = HookDispatcher(subscription_store=_RaisingStore())
        loaded = await d.load_persistent()
        assert loaded == 0
        assert d._last_persist_degradation is not None
        assert d._last_persist_degradation["reason"] == "load_failed"
