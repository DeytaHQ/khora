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
# LLM evaluator (Phase 3, Level 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluator:
    def test_resolve_model_default(self) -> None:
        from khora.hooks.llm_evaluator import LLMFilterEvaluator
        from khora.hooks.models import SemanticHooksConfig

        config = SemanticHooksConfig(filter_model="gpt-5-nano")
        evaluator = LLMFilterEvaluator(config)

        f = SemanticFilter(name="test", description="test filter")
        assert evaluator._resolve_model(f) == "gpt-5-nano"

    def test_resolve_model_per_filter_override(self) -> None:
        from khora.hooks.llm_evaluator import LLMFilterEvaluator

        evaluator = LLMFilterEvaluator()

        f = SemanticFilter(name="test", description="test", filter_model="gemini-2.5-flash-lite")
        assert evaluator._resolve_model(f) == "gemini-2.5-flash-lite"

    def test_parse_json_direct(self) -> None:
        from khora.hooks.llm_evaluator import LLMFilterEvaluator

        result = LLMFilterEvaluator._parse_json('{"0": true, "1": false}')
        assert result == {"0": True, "1": False}

    def test_parse_json_markdown(self) -> None:
        from khora.hooks.llm_evaluator import LLMFilterEvaluator

        result = LLMFilterEvaluator._parse_json('Here is the result:\n```json\n{"0": true}\n```')
        assert result == {"0": True}

    def test_parse_json_invalid(self) -> None:
        from khora.hooks.llm_evaluator import LLMFilterEvaluator

        result = LLMFilterEvaluator._parse_json("This is not JSON")
        assert result == {}

    async def test_evaluate_batch_empty(self) -> None:
        from khora.hooks.llm_evaluator import LLMFilterEvaluator

        evaluator = LLMFilterEvaluator()
        f = SemanticFilter(name="test", description="test")
        results = await evaluator.evaluate_batch(f, [])
        assert results == []


@pytest.mark.unit
class TestDispatcherLLMIntegration:
    async def test_llm_verify_flag_triggers_evaluation(self) -> None:
        """When llm_verify=True, the dispatcher should call LLM evaluator."""
        from unittest.mock import patch

        from khora.hooks.llm_evaluator import LLMFilterResult

        d = HookDispatcher()
        cb = AsyncMock()

        f = SemanticFilter(
            name="llm_filter",
            description="Competitor companies",
            llm_verify=True,
        )
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        # Mock the LLM evaluator to return a match
        mock_results = [LLMFilterResult(entity_index=0, matches=True, model_used="gpt-4.1-nano")]
        with patch.object(d._llm_evaluator, "evaluate_batch", return_value=mock_results) as mock_eval:
            event = MemoryEvent.entity_created(
                namespace_id=uuid4(),
                entity_id=uuid4(),
                data={"name": "Acme Corp", "entity_type": "ORGANIZATION", "description": "A company"},
            )
            count = await d.dispatch(event)
            assert count == 1
            cb.assert_awaited_once()
            mock_eval.assert_awaited_once()

    async def test_llm_verify_rejects_non_match(self) -> None:
        """LLM says no match → callback should NOT fire."""
        from unittest.mock import patch

        from khora.hooks.llm_evaluator import LLMFilterResult

        d = HookDispatcher()
        cb = AsyncMock()

        f = SemanticFilter(
            name="strict_filter",
            description="Only healthcare companies",
            llm_verify=True,
        )
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        mock_results = [LLMFilterResult(entity_index=0, matches=False, model_used="gpt-4.1-nano")]
        with patch.object(d._llm_evaluator, "evaluate_batch", return_value=mock_results):
            event = MemoryEvent.entity_created(
                namespace_id=uuid4(),
                entity_id=uuid4(),
                data={"name": "Acme Corp", "entity_type": "ORGANIZATION", "description": "A tech company"},
            )
            count = await d.dispatch(event)
            assert count == 0
            cb.assert_not_awaited()

    async def test_non_llm_subs_unaffected(self) -> None:
        """Subscriptions without llm_verify=True should fire normally."""
        from unittest.mock import patch

        from khora.hooks.llm_evaluator import LLMFilterResult

        d = HookDispatcher()
        cb_simple = AsyncMock()
        cb_llm = AsyncMock()

        # Simple filter (no LLM)
        d.subscribe(EventType.ENTITY_CREATED, cb_simple)

        # LLM filter that rejects
        f = SemanticFilter(name="llm", description="test", llm_verify=True)
        d.subscribe(EventType.ENTITY_CREATED, cb_llm, filter=f)

        mock_results = [LLMFilterResult(entity_index=0, matches=False)]
        with patch.object(d._llm_evaluator, "evaluate_batch", return_value=mock_results):
            event = MemoryEvent.entity_created(
                namespace_id=uuid4(),
                entity_id=uuid4(),
                data={"name": "X"},
            )
            count = await d.dispatch(event)
            # Simple fires, LLM rejected
            assert count == 1
            cb_simple.assert_awaited_once()
            cb_llm.assert_not_awaited()

    async def test_llm_failure_defaults_to_no_match(self) -> None:
        """If LLM call fails, the subscription is skipped (not crashed)."""
        from unittest.mock import patch

        d = HookDispatcher()
        cb = AsyncMock()

        f = SemanticFilter(name="failing", description="test", llm_verify=True)
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=f)

        with patch.object(d._llm_evaluator, "evaluate_batch", side_effect=RuntimeError("LLM down")):
            event = MemoryEvent.entity_created(
                namespace_id=uuid4(),
                entity_id=uuid4(),
                data={"name": "X"},
            )
            count = await d.dispatch(event)
            assert count == 0  # Fail-safe: no match


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
