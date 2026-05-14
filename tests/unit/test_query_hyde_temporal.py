"""Tests for the temporal-anchored HyDE prompt (#592 / Phase D1).

When the query is classified as RECENCY, STATE_QUERY, or CHANGE,
``HyDEExpander`` must select a system prompt that anchors the
hypothetical to today's date — so its surface tokens (ISO dates,
weekdays, relative markers) align with the recency-tagged chunks
the engine is trying to surface. Other categories (NONE, EXPLICIT,
ORDINAL, AGGREGATE) keep the original time-blind prompt.
"""

from __future__ import annotations

import pytest

from khora.query.hyde import _select_system_prompt
from khora.query.temporal_detection import TemporalCategory


@pytest.mark.unit
@pytest.mark.parametrize(
    "category",
    [TemporalCategory.RECENCY, TemporalCategory.STATE_QUERY, TemporalCategory.CHANGE],
)
def test_temporal_categories_use_anchored_prompt(category: TemporalCategory) -> None:
    today = "2026-05-14"
    prompt = _select_system_prompt(category.value, today)
    # The anchored prompt mentions the date so the LLM's hypothetical
    # contains it as a surface token — this is the entire point of D1.
    assert today in prompt
    # And it asks for "recent / current / recently-changed" framing.
    assert "recent" in prompt.lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    "category",
    [
        TemporalCategory.NONE,
        TemporalCategory.EXPLICIT,
        TemporalCategory.ORDINAL,
        TemporalCategory.AGGREGATE,
    ],
)
def test_non_temporal_categories_use_generic_prompt(category: TemporalCategory) -> None:
    today = "2026-05-14"
    prompt = _select_system_prompt(category.value, today)
    # Generic prompt: must NOT mention today's date or the temporal framing.
    assert today not in prompt
    assert "recent" not in prompt.lower()


@pytest.mark.unit
def test_none_category_uses_generic_prompt() -> None:
    """Passing None (no detector, no signal) falls back to the generic prompt."""
    today = "2026-05-14"
    prompt = _select_system_prompt(None, today)
    assert today not in prompt
    assert "recent" not in prompt.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_hypothetical_passes_category_into_prompt(monkeypatch) -> None:
    """End-to-end: generate_hypothetical() must forward the category-driven
    prompt to ``acompletion`` and pin ``today`` so the test is deterministic.
    """
    from khora.query import hyde

    captured: dict[str, object] = {}

    async def fake_acompletion(query, _llm_config, **kwargs):  # noqa: ANN001
        captured["query"] = query
        captured["system_prompt"] = kwargs.get("system_prompt")
        captured["telemetry_op"] = kwargs.get("_telemetry_op")
        return "stub hypothetical"

    monkeypatch.setattr("khora.config.llm.acompletion", fake_acompletion)

    expander = hyde.HyDEExpander(embedder=None, llm_config=None)  # type: ignore[arg-type]
    result = await expander.generate_hypothetical(
        "what did the team decide this week",
        temporal_category=TemporalCategory.RECENCY,
        today="2026-05-14",
    )

    assert result == "stub hypothetical"
    assert "2026-05-14" in captured["system_prompt"]  # type: ignore[operator]
    assert captured["telemetry_op"] == "hyde"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_hypothetical_internal_detection(monkeypatch) -> None:
    """When the caller doesn't pass a category, generate_hypothetical leaves
    the prompt selection to the caller of expand_query_embedding (which does
    the detection). generate_hypothetical itself is unopinionated — pass-through.
    """
    from khora.query import hyde

    captured: dict[str, object] = {}

    async def fake_acompletion(query, _llm_config, **kwargs):  # noqa: ANN001
        captured["system_prompt"] = kwargs.get("system_prompt")
        return "stub"

    monkeypatch.setattr("khora.config.llm.acompletion", fake_acompletion)
    expander = hyde.HyDEExpander(embedder=None, llm_config=None)  # type: ignore[arg-type]

    # No category, no today → generic prompt, no date pinned.
    await expander.generate_hypothetical("what is the capital of France")
    assert "recent" not in captured["system_prompt"].lower()  # type: ignore[union-attr]
