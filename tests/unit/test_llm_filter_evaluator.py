"""Unit tests for the Level 2 (LLM yes/no) hook filter evaluator.

Covers Issue #576 Phase 1, Item 7. Mocks
``khora.config.llm.acompletion`` — never hits a real LLM.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from khora.core.models.event import EventType, MemoryEvent
from khora.hooks.dispatcher import HookDispatcher
from khora.hooks.llm_evaluator import LLMFilterEvaluator
from khora.hooks.models import SemanticFilter, SemanticHooksConfig


def _make_event(name: str = "Acme Corp", etype: str = "ORGANIZATION") -> MemoryEvent:
    return MemoryEvent.entity_created(
        namespace_id=uuid4(),
        entity_id=uuid4(),
        data={"name": name, "entity_type": etype, "description": f"{name} is a company"},
    )


def _make_filter(*, examples: list[str] | None = None) -> SemanticFilter:
    return SemanticFilter(
        name="competitor_mention",
        description="Any mention of a competitor company",
        entity_types=["ORGANIZATION"],
        examples=examples if examples is not None else ["Acme Corp launched a widget"],
        anti_examples=["Internal employee"],
        llm_confidence_threshold=0.5,
    )


def _llm_response(*, match: bool, confidence: float = 0.9, i: int = 0) -> str:
    return json.dumps({"results": [{"i": i, "match": match, "confidence": confidence}]})


# ---------------------------------------------------------------------------
# LLMFilterEvaluator — single evaluation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluatorSingle:
    async def test_match_returns_true(self) -> None:
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        with patch(
            "khora.config.llm.acompletion",
            new=AsyncMock(return_value=_llm_response(match=True, confidence=0.9)),
        ) as mock_call:
            result = await evaluator.evaluate(event, filt)

        assert result is True
        mock_call.assert_awaited_once()

    async def test_no_match_returns_false(self) -> None:
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        with patch(
            "khora.config.llm.acompletion",
            new=AsyncMock(return_value=_llm_response(match=False, confidence=0.1)),
        ):
            result = await evaluator.evaluate(event, filt)

        assert result is False

    async def test_low_confidence_below_threshold_returns_false(self) -> None:
        """Even if match=True, confidence < threshold means rejection."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()
        filt.llm_confidence_threshold = 0.9

        with patch(
            "khora.config.llm.acompletion",
            new=AsyncMock(return_value=_llm_response(match=True, confidence=0.3)),
        ):
            result = await evaluator.evaluate(event, filt)

        assert result is False


# ---------------------------------------------------------------------------
# LLMFilterEvaluator — micro-batching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluatorBatching:
    async def test_two_evaluations_within_window_share_call(self) -> None:
        """Two concurrent evaluate() calls within the flush window
        produce ONE LLM call covering both pairs."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        # Long-ish window so the two awaits coalesce reliably.
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=200.0, batch_size=10)
        filt = _make_filter()
        event_a = _make_event(name="Acme Corp")
        event_b = _make_event(name="Initech Inc")
        # Force both events into the same namespace bucket so they
        # batch into one LLM call.
        event_b.namespace_id = event_a.namespace_id

        response = json.dumps(
            {
                "results": [
                    {"i": 0, "match": True, "confidence": 0.85},
                    {"i": 1, "match": False, "confidence": 0.20},
                ]
            }
        )
        mock_call = AsyncMock(return_value=response)
        with patch("khora.config.llm.acompletion", new=mock_call):
            results = await asyncio.gather(
                evaluator.evaluate(event_a, filt),
                evaluator.evaluate(event_b, filt),
            )

        assert results == [True, False]
        assert mock_call.await_count == 1, f"expected 1 batched LLM call, got {mock_call.await_count}"

    async def test_full_batch_flushes_immediately(self) -> None:
        """Reaching batch_size triggers immediate flush — does not wait
        for the flush timer."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10_000.0, batch_size=2)
        filt = _make_filter()
        ns = uuid4()
        event_a = _make_event(name="Acme")
        event_a.namespace_id = ns
        event_b = _make_event(name="Initech")
        event_b.namespace_id = ns

        response = json.dumps(
            {
                "results": [
                    {"i": 0, "match": True, "confidence": 0.9},
                    {"i": 1, "match": True, "confidence": 0.9},
                ]
            }
        )
        mock_call = AsyncMock(return_value=response)
        with patch("khora.config.llm.acompletion", new=mock_call):
            t0 = asyncio.get_event_loop().time()
            results = await asyncio.gather(
                evaluator.evaluate(event_a, filt),
                evaluator.evaluate(event_b, filt),
            )
            elapsed = asyncio.get_event_loop().time() - t0

        assert results == [True, True]
        # Should finish far below the 10s flush timeout — proves immediate flush.
        assert elapsed < 1.0, f"batch did not flush immediately: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# LLMFilterEvaluator — failure modes (fail-open)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluatorFailures:
    async def test_llm_raises_fails_open(self) -> None:
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        with patch(
            "khora.config.llm.acompletion",
            new=AsyncMock(side_effect=RuntimeError("nano tier down")),
        ):
            result = await evaluator.evaluate(event, filt)

        assert result is True  # fail-open

    async def test_unparseable_response_fails_open(self) -> None:
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        with patch(
            "khora.config.llm.acompletion",
            new=AsyncMock(return_value="garbage that is not json"),
        ):
            result = await evaluator.evaluate(event, filt)

        assert result is True  # fail-open on parse failure


# ---------------------------------------------------------------------------
# LLMFilterEvaluator — budget enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluatorBudget:
    async def test_budget_exceeded_fails_open_and_skips_call(self) -> None:
        """When budget is exhausted, evaluator MUST NOT call the LLM and
        must fail open."""
        # Cap of 1 token forces every realistic call to exceed.
        cfg = SemanticHooksConfig(
            llm_evaluation_enabled=True,
            llm_max_tokens_per_namespace_per_hour=1,
        )
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        mock_call = AsyncMock()
        with patch("khora.config.llm.acompletion", new=mock_call):
            result = await evaluator.evaluate(event, filt)

        assert result is True  # fail-open
        mock_call.assert_not_awaited()  # no LLM call charged

    async def test_budget_warning_emitted_once_per_window(self) -> None:
        """The 'budget exceeded' warning fires once per (namespace, hour),
        not on every refused request."""
        cfg = SemanticHooksConfig(
            llm_evaluation_enabled=True,
            llm_max_tokens_per_namespace_per_hour=1,
        )
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        filt = _make_filter()
        ns = uuid4()
        event_a = _make_event(name="Acme")
        event_a.namespace_id = ns
        event_b = _make_event(name="Initech")
        event_b.namespace_id = ns

        warnings: list[str] = []

        def capture(message: str, *args, **kwargs) -> None:
            warnings.append(message.format(*args) if args else message)

        with (
            patch("khora.hooks.llm_evaluator.logger.warning", side_effect=capture),
            patch("khora.config.llm.acompletion", new=AsyncMock()),
        ):
            await evaluator.evaluate(event_a, filt)
            await evaluator.evaluate(event_b, filt)

        budget_warnings = [w for w in warnings if "budget exceeded" in w]
        assert len(budget_warnings) == 1, (
            f"expected exactly one budget warning per window, got {len(budget_warnings)}: {warnings}"
        )

    async def test_budget_disabled_when_cap_is_zero(self) -> None:
        cfg = SemanticHooksConfig(
            llm_evaluation_enabled=True,
            llm_max_tokens_per_namespace_per_hour=0,
        )
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        mock_call = AsyncMock(return_value=_llm_response(match=True, confidence=0.9))
        with patch("khora.config.llm.acompletion", new=mock_call):
            result = await evaluator.evaluate(event, filt)

        assert result is True
        mock_call.assert_awaited_once()


# ---------------------------------------------------------------------------
# Batch-task lifecycle (Issue #1161): the full-batch flush fires _run_batch as
# a create_task. The handle must be strongly referenced (GC can't drop it) and
# a pre-resolve exception must not strand awaiters on `await future` forever.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluatorBatchTaskLifecycle:
    async def test_full_batch_task_is_strongly_referenced(self) -> None:
        """The fire-and-forget batch task is retained in a set, not dropped."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10_000.0, batch_size=2)
        filt = _make_filter()
        ns = uuid4()
        event_a = _make_event(name="Acme")
        event_a.namespace_id = ns
        event_b = _make_event(name="Initech")
        event_b.namespace_id = ns

        seen_tasks: list[int] = []

        async def slow_acompletion(prompt, config, **kwargs):  # noqa: ANN001
            # While the batch is in flight, the evaluator must hold a strong ref.
            assert len(evaluator._batch_tasks) >= 1
            seen_tasks.append(len(evaluator._batch_tasks))
            return json.dumps(
                {"results": [{"i": 0, "match": True, "confidence": 0.9}, {"i": 1, "match": True, "confidence": 0.9}]}
            )

        with patch("khora.config.llm.acompletion", new=slow_acompletion):
            results = await asyncio.gather(
                evaluator.evaluate(event_a, filt),
                evaluator.evaluate(event_b, filt),
            )

        assert results == [True, True]
        assert seen_tasks and seen_tasks[0] >= 1
        # Done-callback drains the set once the task completes.
        await asyncio.sleep(0)
        assert len(evaluator._batch_tasks) == 0

    async def test_pre_resolve_exception_does_not_strand_awaiters(self) -> None:
        """If the batch worker raises BEFORE resolving the per-item futures
        (e.g. in bucketing / token estimation), awaiters must still be released
        fail-open rather than blocking on `await future` forever (#1161)."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10_000.0, batch_size=2)
        filt = _make_filter()
        ns = uuid4()
        event_a = _make_event(name="Acme")
        event_a.namespace_id = ns
        event_b = _make_event(name="Initech")
        event_b.namespace_id = ns

        # Make the bucket evaluation explode before any future is resolved.
        async def _boom(_items):  # noqa: ANN001
            raise RuntimeError("pre-resolve explosion")

        with patch.object(evaluator, "_evaluate_bucket", side_effect=_boom):
            # Must not hang; both awaiters released fail-open.
            results = await asyncio.wait_for(
                asyncio.gather(
                    evaluator.evaluate(event_a, filt),
                    evaluator.evaluate(event_b, filt),
                ),
                timeout=5.0,
            )

        assert results == [True, True]  # fail-open, not stranded
        await asyncio.sleep(0)
        assert len(evaluator._batch_tasks) == 0


# ---------------------------------------------------------------------------
# Dispatcher integration — flag-gated, examples-required
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDispatcherLLMIntegration:
    async def test_disabled_flag_skips_level_2(self) -> None:
        """When llm_evaluation_enabled=False, the dispatcher never invokes
        the LLM evaluator even for filters with examples."""
        # Explicit False — default — exercises the gating branch.
        cfg = SemanticHooksConfig(llm_evaluation_enabled=False)
        d = HookDispatcher(config=cfg)

        # Filter with a description embedding so Level 1 has a fair shot.
        filt = SemanticFilter(
            name="competitor_mention",
            description="Any mention of a competitor company",
            examples=["Acme Corp launched a widget"],
            embedding=[1.0, 0.0, 0.0, 0.0],
            similarity_threshold=0.0,  # disable Level 1 gate for simplicity
        )
        cb = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=filt)

        event = _make_event()
        # Matching embedding so Level 1 would pass if it ran.
        event.data["embedding"] = [1.0, 0.0, 0.0, 0.0]

        mock_call = AsyncMock(return_value=_llm_response(match=False))
        with patch("khora.config.llm.acompletion", new=mock_call):
            count = await d.dispatch(event)

        assert count == 1
        cb.assert_awaited_once()
        mock_call.assert_not_awaited()
        # And the evaluator was never even built.
        assert d._llm_evaluator is None

    async def test_enabled_flag_with_examples_runs_level_2(self) -> None:
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        d = HookDispatcher(config=cfg)

        filt = SemanticFilter(
            name="competitor_mention",
            description="Any mention of a competitor company",
            examples=["Acme Corp launched a widget"],
            embedding=[1.0, 0.0, 0.0, 0.0],
            similarity_threshold=0.0,
        )
        cb = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=filt)

        event = _make_event()
        event.data["embedding"] = [1.0, 0.0, 0.0, 0.0]

        # LLM says no match — callback must NOT fire.
        mock_call = AsyncMock(return_value=_llm_response(match=False, confidence=0.1))
        with patch("khora.config.llm.acompletion", new=mock_call):
            count = await d.dispatch(event)

        assert count == 0, "Level 2 'no_match' should have dropped the subscription"
        cb.assert_not_awaited()
        mock_call.assert_awaited_once()

    async def test_enabled_but_no_examples_skips_level_2(self) -> None:
        """Filters without examples skip Level 2 even when the flag is on —
        the LLM has no calibration anchor."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True)
        d = HookDispatcher(config=cfg)

        filt = SemanticFilter(
            name="competitor_mention",
            description="Any mention of a competitor company",
            examples=[],  # empty — Level 2 must be skipped
            embedding=[1.0, 0.0, 0.0, 0.0],
            similarity_threshold=0.0,
        )
        cb = AsyncMock()
        d.subscribe(EventType.ENTITY_CREATED, cb, filter=filt)

        event = _make_event()
        event.data["embedding"] = [1.0, 0.0, 0.0, 0.0]

        mock_call = AsyncMock(return_value=_llm_response(match=False))
        with patch("khora.config.llm.acompletion", new=mock_call):
            count = await d.dispatch(event)

        assert count == 1
        cb.assert_awaited_once()
        mock_call.assert_not_awaited()


# ---------------------------------------------------------------------------
# Cache (Issue #601)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluatorCache:
    async def test_repeat_event_hits_cache_no_llm_call(self) -> None:
        """Same (filter, event_summary) twice → one LLM call, not two."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True, llm_cache_size=128)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        mock_call = AsyncMock(return_value=_llm_response(match=False, confidence=0.1))
        with patch("khora.config.llm.acompletion", new=mock_call):
            first = await evaluator.evaluate(event, filt)
            # Second identical evaluation (same name+type+description) — must be cached.
            second = await evaluator.evaluate(event, filt)

        assert first is False
        assert second is False
        mock_call.assert_awaited_once()

    async def test_cache_separates_by_filter_id(self) -> None:
        """Same event, different filters → separate cache entries."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True, llm_cache_size=128)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        f1 = _make_filter()
        f2 = _make_filter()  # different id even with same description

        mock_call = AsyncMock(return_value=_llm_response(match=True))
        with patch("khora.config.llm.acompletion", new=mock_call):
            await evaluator.evaluate(event, f1)
            await evaluator.evaluate(event, f2)

        # Two distinct (filter_id, hash) keys → two LLM calls.
        assert mock_call.await_count == 2

    async def test_cache_disabled_when_size_is_zero(self) -> None:
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True, llm_cache_size=0)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        mock_call = AsyncMock(return_value=_llm_response(match=False, confidence=0.1))
        with patch("khora.config.llm.acompletion", new=mock_call):
            await evaluator.evaluate(event, filt)
            await evaluator.evaluate(event, filt)

        # Cache disabled — every call hits the LLM.
        assert mock_call.await_count == 2

    async def test_cache_lru_evicts_oldest(self) -> None:
        """When size=1, a second distinct key evicts the first."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True, llm_cache_size=1)
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        e1 = _make_event(name="Acme Corp")
        e2 = _make_event(name="Globex Inc")
        filt = _make_filter()

        mock_call = AsyncMock(return_value=_llm_response(match=False, confidence=0.1))
        with patch("khora.config.llm.acompletion", new=mock_call):
            await evaluator.evaluate(e1, filt)  # populates cache
            await evaluator.evaluate(e2, filt)  # evicts e1
            await evaluator.evaluate(e1, filt)  # cache miss → LLM call again

        assert mock_call.await_count == 3

    async def test_fifty_events_one_filter_drops_to_one_call(self) -> None:
        """Acceptance bar from #601: 50 events × 1 subscription → ≤2 LLM calls
        when the events share the same (name, type, description). The first
        batch (batch_size=10) flushes one call; cache hits short-circuit the rest.
        """
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True, llm_cache_size=128)
        evaluator = LLMFilterEvaluator(cfg, batch_size=10, batch_flush_ms=10.0)
        filt = _make_filter()
        ns = uuid4()
        events = [
            MemoryEvent.entity_created(
                namespace_id=ns,
                entity_id=uuid4(),
                data={"name": "Acme Corp", "entity_type": "ORGANIZATION", "description": "x"},
            )
            for _ in range(50)
        ]

        mock_call = AsyncMock(return_value=_llm_response(match=True, confidence=0.9))
        with patch("khora.config.llm.acompletion", new=mock_call):
            # Run sequentially so the first batch completes (and populates cache)
            # before subsequent events arrive — matches the realistic stream pattern.
            results = []
            for ev in events:
                results.append(await evaluator.evaluate(ev, filt))

        assert all(results)
        # The acceptance bar is ≤2 LLM calls. In the sequential path we expect 1.
        assert mock_call.await_count <= 2


# ---------------------------------------------------------------------------
# Per-subscription budget (Issue #601)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluatorSubscriptionBudget:
    async def test_subscription_budget_fails_open_when_exceeded(self) -> None:
        cfg = SemanticHooksConfig(
            llm_evaluation_enabled=True,
            llm_cache_size=0,  # exercise the budget directly
            llm_max_tokens_per_subscription_per_hour=1,  # any nonzero estimate exceeds
        )
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        mock_call = AsyncMock(return_value=_llm_response(match=False))
        with patch("khora.config.llm.acompletion", new=mock_call):
            result = await evaluator.evaluate(event, filt)

        assert result is True  # fail open
        mock_call.assert_not_awaited()

    async def test_subscription_budget_disabled_when_zero(self) -> None:
        cfg = SemanticHooksConfig(
            llm_evaluation_enabled=True,
            llm_cache_size=0,
            llm_max_tokens_per_subscription_per_hour=0,
        )
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        event = _make_event()
        filt = _make_filter()

        mock_call = AsyncMock(return_value=_llm_response(match=True, confidence=0.9))
        with patch("khora.config.llm.acompletion", new=mock_call):
            result = await evaluator.evaluate(event, filt)

        assert result is True
        mock_call.assert_awaited_once()

    async def test_noisy_filter_does_not_starve_quiet_filter(self) -> None:
        """Per-subscription cap kicks in on the noisy filter; quiet filter unaffected.

        Cap is sized to fit one call per filter (~150 tokens estimated). The
        noisy filter consumes its bucket on call 1 and is throttled on call 2;
        the quiet filter's independent bucket is still full so it goes through.
        """
        cfg = SemanticHooksConfig(
            llm_evaluation_enabled=True,
            llm_cache_size=0,
            llm_max_tokens_per_namespace_per_hour=10_000,
            llm_max_tokens_per_subscription_per_hour=200,  # ~1 call per filter
        )
        evaluator = LLMFilterEvaluator(cfg, batch_flush_ms=10.0)
        noisy = _make_filter()
        quiet = _make_filter()

        with patch(
            "khora.config.llm.acompletion",
            new=AsyncMock(return_value=_llm_response(match=False)),
        ):
            first_noisy = await evaluator.evaluate(_make_event(name="N1"), noisy)
            second_noisy = await evaluator.evaluate(_make_event(name="N2"), noisy)
            quiet_result = await evaluator.evaluate(_make_event(name="Q1"), quiet)

        assert first_noisy is False
        assert second_noisy is True  # fail open from budget breach
        assert quiet_result is False


# ---------------------------------------------------------------------------
# Intra-batch coalescing (Issue #608)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMFilterEvaluatorIntraBatchCoalescing:
    async def test_ten_identical_events_in_one_batch_get_one_prompt_slot(self) -> None:
        """A full batch of identical (name, type, description) tuples should
        spend one LLM prompt slot, not ten — the decision is fanned out."""
        cfg = SemanticHooksConfig(
            llm_evaluation_enabled=True,
            llm_cache_size=0,  # disable cross-batch cache to isolate this path
        )
        evaluator = LLMFilterEvaluator(cfg, batch_size=10, batch_flush_ms=10.0)
        filt = _make_filter()
        ns = uuid4()

        captured_prompts: list[str] = []

        async def fake_acompletion(prompt, config, **kwargs):  # noqa: ANN001
            captured_prompts.append(prompt)
            return _llm_response(match=True, confidence=0.9)

        with patch("khora.config.llm.acompletion", new=fake_acompletion):
            # Fire 10 evaluate() coroutines concurrently so they land in one batch.
            futures = [
                evaluator.evaluate(
                    MemoryEvent.entity_created(
                        namespace_id=ns,
                        entity_id=uuid4(),
                        data={"name": "Acme", "entity_type": "ORGANIZATION", "description": "x"},
                    ),
                    filt,
                )
                for _ in range(10)
            ]
            results = await asyncio.gather(*futures)

        # All 10 futures resolve to the same fanned-out decision.
        assert all(results)
        # One LLM call total.
        assert len(captured_prompts) == 1
        # The prompt body contains exactly one ``[0] name=...`` slot — the rest
        # were coalesced.
        prompt = captured_prompts[0]
        assert prompt.count("[0] name=") == 1
        assert "[1] name=" not in prompt

    async def test_distinct_events_in_one_batch_remain_separate(self) -> None:
        """Sanity: when events differ, every one gets its own prompt slot."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True, llm_cache_size=0)
        evaluator = LLMFilterEvaluator(cfg, batch_size=3, batch_flush_ms=10.0)
        filt = _make_filter()
        ns = uuid4()

        captured_prompts: list[str] = []

        async def fake_acompletion(prompt, config, **kwargs):  # noqa: ANN001
            captured_prompts.append(prompt)
            # Return one result per item (we'll send 3 distinct events).
            return json.dumps(
                {
                    "results": [
                        {"i": 0, "match": True, "confidence": 0.9},
                        {"i": 1, "match": False, "confidence": 0.1},
                        {"i": 2, "match": True, "confidence": 0.8},
                    ]
                }
            )

        with patch("khora.config.llm.acompletion", new=fake_acompletion):
            futures = [
                evaluator.evaluate(
                    MemoryEvent.entity_created(
                        namespace_id=ns,
                        entity_id=uuid4(),
                        data={"name": name, "entity_type": "ORGANIZATION", "description": ""},
                    ),
                    filt,
                )
                for name in ("A", "B", "C")
            ]
            results = await asyncio.gather(*futures)

        assert results == [True, False, True]
        # One LLM call with all three distinct slots present.
        assert len(captured_prompts) == 1
        for idx in range(3):
            assert f"[{idx}] name=" in captured_prompts[0]

    async def test_mixed_batch_dedupes_only_the_duplicates(self) -> None:
        """5 events: A, A, B, A, C → 3 unique slots, 5 fanned-out decisions."""
        cfg = SemanticHooksConfig(llm_evaluation_enabled=True, llm_cache_size=0)
        evaluator = LLMFilterEvaluator(cfg, batch_size=5, batch_flush_ms=10.0)
        filt = _make_filter()
        ns = uuid4()

        captured_prompts: list[str] = []

        async def fake_acompletion(prompt, config, **kwargs):  # noqa: ANN001
            captured_prompts.append(prompt)
            # 3 unique → 3 result slots. A is index 0 (first seen), B is 1, C is 2.
            return json.dumps(
                {
                    "results": [
                        {"i": 0, "match": True, "confidence": 0.9},  # A
                        {"i": 1, "match": False, "confidence": 0.1},  # B
                        {"i": 2, "match": True, "confidence": 0.7},  # C
                    ]
                }
            )

        names = ["A", "A", "B", "A", "C"]
        with patch("khora.config.llm.acompletion", new=fake_acompletion):
            futures = [
                evaluator.evaluate(
                    MemoryEvent.entity_created(
                        namespace_id=ns,
                        entity_id=uuid4(),
                        data={"name": name, "entity_type": "ORGANIZATION", "description": ""},
                    ),
                    filt,
                )
                for name in names
            ]
            results = await asyncio.gather(*futures)

        # The three A's share one decision; B and C have their own.
        assert results == [True, True, False, True, True]
        # Exactly one LLM call with 3 deduped slots.
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "[0] name=" in prompt and "[1] name=" in prompt and "[2] name=" in prompt
        assert "[3]" not in prompt
