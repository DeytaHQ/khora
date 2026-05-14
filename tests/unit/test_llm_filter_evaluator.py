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
