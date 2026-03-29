"""Unit tests for semantic hooks and triggers."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.core.models.event import EventType, MemoryEvent
from khora.hooks.dispatcher import HookDispatcher
from khora.hooks.models import FilterMatch, SemanticFilter, SemanticHooksConfig

# ---------------------------------------------------------------------------
# SemanticHooksConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticHooksConfig:
    def test_defaults(self) -> None:
        config = SemanticHooksConfig()
        assert config.enabled is True
        assert config.filter_model == "gpt-4.1-nano"
        assert config.default_similarity_threshold == 0.5
        assert config.llm_batch_size == 10
        assert config.max_concurrent_callbacks == 10

    def test_custom_model(self) -> None:
        config = SemanticHooksConfig(filter_model="gpt-5-nano")
        assert config.filter_model == "gpt-5-nano"


# ---------------------------------------------------------------------------
# SemanticFilter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticFilter:
    def test_defaults(self) -> None:
        f = SemanticFilter(name="test", description="Test filter")
        assert f.name == "test"
        assert f.entity_types == []
        assert f.similarity_threshold == 0.5
        assert f.filter_model is None  # use config default
        assert f.namespace_id is None

    def test_with_types(self) -> None:
        f = SemanticFilter(
            name="competitor",
            description="Competitor mentions",
            entity_types=["ORGANIZATION", "PRODUCT"],
            filter_model="gpt-5-nano",
        )
        assert f.entity_types == ["ORGANIZATION", "PRODUCT"]
        assert f.filter_model == "gpt-5-nano"


# ---------------------------------------------------------------------------
# FilterMatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterMatch:
    def test_defaults(self) -> None:
        m = FilterMatch(filter_name="test")
        assert m.matched_at_level == 0
        assert m.similarity_score is None
        assert m.llm_confidence is None


# ---------------------------------------------------------------------------
# HookDispatcher — subscription management
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHookDispatcherSubscription:
    def test_subscribe_and_count(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        sub_id = d.subscribe(EventType.ENTITY_CREATED, cb)
        assert d.subscription_count == 1
        assert sub_id is not None

    def test_unsubscribe(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        sub_id = d.subscribe(EventType.ENTITY_CREATED, cb)
        assert d.unsubscribe(sub_id) is True
        assert d.subscription_count == 0

    def test_unsubscribe_nonexistent(self) -> None:
        d = HookDispatcher()
        assert d.unsubscribe(uuid4()) is False

    def test_clear(self) -> None:
        d = HookDispatcher()
        d.subscribe(EventType.ENTITY_CREATED, AsyncMock())
        d.subscribe(EventType.RELATIONSHIP_CREATED, AsyncMock())
        assert d.subscription_count == 2
        d.clear()
        assert d.subscription_count == 0

    def test_subscribe_with_string_event_type(self) -> None:
        d = HookDispatcher()
        d.subscribe("entity.created", AsyncMock())
        assert d.subscription_count == 1

    def test_multiple_subscribers_same_event(self) -> None:
        d = HookDispatcher()
        d.subscribe(EventType.ENTITY_CREATED, AsyncMock())
        d.subscribe(EventType.ENTITY_CREATED, AsyncMock())
        assert d.subscription_count == 2


# ---------------------------------------------------------------------------
# HookDispatcher — event dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHookDispatcherDispatch:
    async def test_dispatch_fires_callback(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb)

        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={"name": "Acme Corp", "entity_type": "ORGANIZATION"},
        )

        count = await d.dispatch(event)
        assert count == 1
        cb.assert_awaited_once_with(event)

    async def test_dispatch_no_subscribers(self) -> None:
        d = HookDispatcher()
        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={},
        )
        count = await d.dispatch(event)
        assert count == 0

    async def test_dispatch_wrong_event_type(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        d.subscribe(EventType.RELATIONSHIP_CREATED, cb)

        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={},
        )
        count = await d.dispatch(event)
        assert count == 0
        cb.assert_not_awaited()

    async def test_dispatch_multiple_callbacks(self) -> None:
        d = HookDispatcher()
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb1)
        d.subscribe(EventType.ENTITY_CREATED, cb2)

        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={},
        )
        count = await d.dispatch(event)
        assert count == 2
        cb1.assert_awaited_once()
        cb2.assert_awaited_once()

    async def test_dispatch_callback_failure_isolated(self) -> None:
        """A failing callback should not prevent other callbacks from running."""
        d = HookDispatcher()
        cb_fail = AsyncMock(side_effect=RuntimeError("boom"))
        cb_ok = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb_fail)
        d.subscribe(EventType.ENTITY_CREATED, cb_ok)

        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={},
        )
        count = await d.dispatch(event)
        assert count == 2
        cb_ok.assert_awaited_once()

    async def test_dispatch_disabled_subscription(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        sub_id = d.subscribe(EventType.ENTITY_CREATED, cb)
        # Disable the subscription
        d._sub_by_id[sub_id].enabled = False

        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={},
        )
        count = await d.dispatch(event)
        assert count == 0


# ---------------------------------------------------------------------------
# HookDispatcher — Level 0 type filtering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHookDispatcherTypeFilter:
    async def test_entity_type_filter_match(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        f = SemanticFilter(name="orgs", entity_types=["ORGANIZATION"])
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={"name": "Acme", "entity_type": "ORGANIZATION"},
        )
        count = await d.dispatch(event)
        assert count == 1
        cb.assert_awaited_once()

    async def test_entity_type_filter_reject(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        f = SemanticFilter(name="orgs", entity_types=["ORGANIZATION"])
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={"name": "Alice", "entity_type": "PERSON"},
        )
        count = await d.dispatch(event)
        assert count == 0
        cb.assert_not_awaited()

    async def test_empty_type_filter_matches_all(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        f = SemanticFilter(name="all", entity_types=[])  # empty = match all
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={"name": "Anything", "entity_type": "CONCEPT"},
        )
        count = await d.dispatch(event)
        assert count == 1

    async def test_namespace_scope_filter(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        target_ns = uuid4()
        other_ns = uuid4()
        f = SemanticFilter(name="scoped", namespace_id=target_ns)
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        # Event in target namespace — should match
        event_match = MemoryEvent.entity_created(
            namespace_id=target_ns,
            entity_id=uuid4(),
            data={},
        )
        count = await d.dispatch(event_match)
        assert count == 1

        cb.reset_mock()

        # Event in other namespace — should not match
        event_miss = MemoryEvent.entity_created(
            namespace_id=other_ns,
            entity_id=uuid4(),
            data={},
        )
        count = await d.dispatch(event_miss)
        assert count == 0

    async def test_relationship_type_filter(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()
        f = SemanticFilter(name="works_for", relationship_types=["WORKS_FOR"])
        d.subscribe(EventType.RELATIONSHIP_CREATED, cb, filter=f)

        # Matching relationship
        event = MemoryEvent.relationship_created(
            namespace_id=uuid4(),
            relationship_id=uuid4(),
            data={"relationship_type": "WORKS_FOR"},
        )
        count = await d.dispatch(event)
        assert count == 1

        cb.reset_mock()

        # Non-matching relationship
        event2 = MemoryEvent.relationship_created(
            namespace_id=uuid4(),
            relationship_id=uuid4(),
            data={"relationship_type": "KNOWS"},
        )
        count = await d.dispatch(event2)
        assert count == 0


# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPublicImports:
    def test_event_type_importable(self) -> None:
        from khora import EventType

        assert EventType.ENTITY_CREATED.value == "entity.created"

    def test_semantic_filter_importable(self) -> None:
        from khora import SemanticFilter

        f = SemanticFilter(name="test", description="Test")
        assert f.name == "test"
