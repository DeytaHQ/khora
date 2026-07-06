"""Unit tests for semantic hooks and triggers."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

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

    async def test_namespace_scope_matches_across_id_spaces(self) -> None:
        """#1399: a sub scoped by the STABLE namespace_id matches events emitted with the ROW id.

        ``remember()`` resolves the public stable namespace_id to the active
        version's row id before extraction, so ingest events carry the row id;
        ``subscribe(namespace_id=stable)`` stores the stable id. Without
        normalization the two never matched and every scoped subscription fired
        zero callbacks. The dispatcher now normalizes both sides through the
        (idempotent) resolver before comparing.
        """
        stable_ns = uuid4()
        row_ns = uuid4()  # what _resolve_namespace(stable_ns) returns at ingest

        async def resolver(ns: UUID) -> UUID:
            # Idempotent on row ids: stable -> row, row -> row (mirrors the real
            # coordinator.resolve_namespace, which matches on namespace_id OR id).
            return row_ns if ns == stable_ns else ns

        d = HookDispatcher(namespace_resolver=resolver)
        cb = AsyncMock()
        # Caller subscribes with the STABLE id (what create_namespace returns).
        d.subscribe(EventType.ENTITY_CREATED, cb, namespace_id=stable_ns)

        # Ingest emits with the RESOLVED ROW id.
        event = MemoryEvent.entity_created(namespace_id=row_ns, entity_id=uuid4(), data={})
        count = await d.dispatch(event)
        assert count == 1, "scoped subscription must fire when event carries the resolved row id"
        cb.assert_awaited_once()

        # A genuinely different namespace still must not match.
        cb.reset_mock()
        other = MemoryEvent.entity_created(namespace_id=uuid4(), entity_id=uuid4(), data={})
        assert await d.dispatch(other) == 0
        cb.assert_not_awaited()

    async def test_namespace_scope_survives_reversioning(self) -> None:
        """#1427: a stable-scoped sub keeps firing after ``create_namespace_version()``.

        Exact issue sequence: subscribe with the stable id, dispatch an event
        (which warms the stable→row cache), re-version the namespace (same
        stable id, NEW active row id), dispatch again with the new row id.
        Before the fix the cached stable→row_v1 mapping never invalidated, so
        the scope comparison false-negatived forever and the scoped
        subscription went silent. The dispatcher now re-resolves the suspect
        cached mapping once on a failed comparison and retries.
        """
        stable_ns = uuid4()
        row_v1 = uuid4()
        row_v2 = uuid4()
        active_row = row_v1
        resolve_calls: list[UUID] = []

        async def resolver(ns: UUID) -> UUID:
            # Mirrors coordinator.resolve_namespace: stable id -> ACTIVE row id;
            # idempotent on row ids.
            resolve_calls.append(ns)
            return active_row if ns == stable_ns else ns

        d = HookDispatcher(namespace_resolver=resolver)
        cb = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb, namespace_id=stable_ns)

        # Pre-reversion ingest fires (and warms the stable→row_v1 cache entry).
        event_v1 = MemoryEvent.entity_created(namespace_id=row_v1, entity_id=uuid4(), data={})
        assert await d.dispatch(event_v1) == 1
        cb.assert_awaited_once()

        # Re-version: same stable id, new active row id. The dispatcher gets
        # no signal - its cached stable→row_v1 mapping is now stale.
        active_row = row_v2

        # Post-reversion ingest emits with the NEW row id. Before #1427 this
        # fired 0 callbacks; the verification path must self-heal and fire.
        cb.reset_mock()
        event_v2 = MemoryEvent.entity_created(namespace_id=row_v2, entity_id=uuid4(), data={})
        assert await d.dispatch(event_v2) == 1, "scoped subscription must survive namespace re-versioning (#1427)"
        cb.assert_awaited_once()

        # A genuinely foreign event still must not match - and re-verifies the
        # scope mapping at most once (no unbounded resolver storm).
        cb.reset_mock()
        resolve_calls.clear()
        foreign = MemoryEvent.entity_created(namespace_id=uuid4(), entity_id=uuid4(), data={})
        assert await d.dispatch(foreign) == 0
        cb.assert_not_awaited()
        assert resolve_calls.count(stable_ns) <= 1, "foreign-event miss must re-resolve the scope at most once"

    async def test_namespace_scope_reproduces_1399_without_resolver(self) -> None:
        """Anti-vacuity: with NO resolver (pre-#1399 behavior) the mismatch drops the event."""
        stable_ns = uuid4()
        row_ns = uuid4()
        d = HookDispatcher()  # no namespace_resolver -> direct compare (old behavior)
        cb = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb, namespace_id=stable_ns)
        event = MemoryEvent.entity_created(namespace_id=row_ns, entity_id=uuid4(), data={})
        assert await d.dispatch(event) == 0  # the bug: zero callbacks

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
# Embedding filter (Phase 2, Level 1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingFilter:
    def test_to_binary_and_hamming(self) -> None:
        from khora.hooks.embedding_filter import _hamming_similarity, _to_binary

        a = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        b = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        ba, bb = _to_binary(a), _to_binary(b)
        assert _hamming_similarity(ba, bb, 8) == 1.0  # identical

    def test_hamming_different(self) -> None:
        from khora.hooks.embedding_filter import _hamming_similarity, _to_binary

        a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        b = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        ba, bb = _to_binary(a), _to_binary(b)
        assert _hamming_similarity(ba, bb, 8) == 0.0  # maximally different

    def test_cosine_similarity_identical(self) -> None:
        from khora.hooks.embedding_filter import cosine_similarity

        a = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        from khora.hooks.embedding_filter import cosine_similarity

        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cache_register_and_gate(self) -> None:
        from khora.hooks.embedding_filter import EmbeddingFilterCache

        cache = EmbeddingFilterCache(hamming_threshold=0.3)
        f = SemanticFilter(
            name="test",
            embedding=[1.0, 0.5, -0.3, 0.8, -0.1, 0.9, -0.5, 0.2],
            similarity_threshold=0.5,
        )
        cache.register_filter(f)

        # Similar embedding should pass
        similar = [0.9, 0.4, -0.2, 0.7, -0.05, 0.85, -0.4, 0.15]
        passes, score = cache.passes_embedding_gate(similar, f)
        assert passes is True
        assert score is not None
        assert score > 0.5

    def test_gate_rejects_dissimilar(self) -> None:
        from khora.hooks.embedding_filter import EmbeddingFilterCache

        cache = EmbeddingFilterCache(hamming_threshold=0.3)
        f = SemanticFilter(
            name="test",
            embedding=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            similarity_threshold=0.8,
        )
        cache.register_filter(f)

        # Very different embedding should fail
        different = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        passes, score = cache.passes_embedding_gate(different, f)
        assert passes is False

    def test_gate_skips_when_no_embedding(self) -> None:
        from khora.hooks.embedding_filter import EmbeddingFilterCache

        cache = EmbeddingFilterCache()
        f = SemanticFilter(name="no_emb", embedding=None)

        # No embedding = skip gate (don't reject)
        passes, score = cache.passes_embedding_gate([1.0, 2.0], f)
        assert passes is True
        assert score is None

    def test_gate_disabled_at_zero_threshold(self) -> None:
        from khora.hooks.embedding_filter import EmbeddingFilterCache

        cache = EmbeddingFilterCache()
        f = SemanticFilter(
            name="disabled",
            embedding=[1.0, 0.0],
            similarity_threshold=0.0,  # disabled
        )
        cache.register_filter(f)

        passes, score = cache.passes_embedding_gate([-1.0, 0.0], f)
        assert passes is True  # gate disabled

    def test_mixed_dim_filters_do_not_clobber_each_other(self) -> None:
        # Regression for #927: registering a second filter at a different
        # embedding dimension must not corrupt the first filter's pre-screen.
        from khora.hooks.embedding_filter import EmbeddingFilterCache

        cache = EmbeddingFilterCache(hamming_threshold=0.3)

        big = SemanticFilter(
            name="big",
            embedding=[1.0, 0.5, -0.3, 0.8, -0.1, 0.9, -0.5, 0.2],
            similarity_threshold=0.5,
        )
        small = SemanticFilter(
            name="small",
            embedding=[1.0, -1.0, 1.0, -1.0],
            similarity_threshold=0.5,
        )

        cache.register_filter(big)
        cache.register_filter(small)  # would clobber a shared bit count

        # A clearly-similar 8-dim entity must still pass against the 8-dim
        # filter, scored against the 8-dim filter's own bit count.
        similar = [0.9, 0.4, -0.2, 0.7, -0.05, 0.85, -0.4, 0.15]
        passes, score = cache.passes_embedding_gate(similar, big)
        assert passes is True
        assert score is not None
        assert score > 0.5

    def test_gate_rejects_dimension_mismatch(self) -> None:
        # An entity embedded at a different dimension than the filter cannot
        # be meaningfully compared; reject rather than produce a bogus score.
        from khora.hooks.embedding_filter import EmbeddingFilterCache

        cache = EmbeddingFilterCache(hamming_threshold=0.3)
        f = SemanticFilter(
            name="test",
            embedding=[1.0, 0.5, -0.3, 0.8, -0.1, 0.9, -0.5, 0.2],
            similarity_threshold=0.5,
        )
        cache.register_filter(f)

        entity_wrong_dim = [1.0, 0.5, -0.3, 0.8]  # 4 dims vs filter's 8
        passes, score = cache.passes_embedding_gate(entity_wrong_dim, f)
        assert passes is False
        assert score is None


@pytest.mark.unit
class TestDispatcherEmbeddingIntegration:
    async def test_embedding_filter_in_dispatch(self) -> None:
        """Embedding filter gates callbacks when entity has embedding in event data."""
        d = HookDispatcher()
        cb = AsyncMock()

        # Filter with an embedding
        f = SemanticFilter(
            name="similar_only",
            embedding=[1.0, 0.5, -0.3, 0.8],
            similarity_threshold=0.5,
        )
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        # Event with similar embedding → should pass
        event_similar = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={
                "name": "Test",
                "entity_type": "CONCEPT",
                "embedding": [0.9, 0.4, -0.2, 0.7],  # similar
            },
        )
        count = await d.dispatch(event_similar)
        assert count == 1
        cb.assert_awaited_once()

    async def test_embedding_filter_rejects_dissimilar(self) -> None:
        d = HookDispatcher()
        cb = AsyncMock()

        f = SemanticFilter(
            name="strict",
            embedding=[1.0, 1.0, 1.0, 1.0],
            similarity_threshold=0.9,
        )
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        # Event with very different embedding → should be rejected
        event_diff = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={
                "name": "Test",
                "entity_type": "CONCEPT",
                "embedding": [-1.0, -1.0, -1.0, -1.0],
            },
        )
        count = await d.dispatch(event_diff)
        assert count == 0
        cb.assert_not_awaited()

    async def test_no_embedding_in_event_skips_gate(self) -> None:
        """When event has no embedding, the gate is skipped (not rejected)."""
        d = HookDispatcher()
        cb = AsyncMock()

        f = SemanticFilter(
            name="with_emb",
            embedding=[1.0, 0.0],
            similarity_threshold=0.5,
        )
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        # Event without embedding → gate skipped, callback fires
        event = MemoryEvent.entity_created(
            namespace_id=uuid4(),
            entity_id=uuid4(),
            data={"name": "Test", "entity_type": "CONCEPT"},
        )
        count = await d.dispatch(event)
        assert count == 1


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
