"""#1563 Part B: the doubled truncation-retry budget is clamped to the model cap.

Unclamped, khora's default max_tokens (12,288) doubled to 24,576 - above
gpt-4o-mini's 16,384 completion cap - so the retry drew an OpenAI 400,
which is retryable, so every tenacity attempt re-ran BOTH calls and burned
the full retry budget on a deterministic failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from khora.extraction.extractors.llm import LLMEntityExtractor, _model_output_cap


def _response(content: str, finish_reason: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    response.choices = [choice]
    response.model = "test-model"
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    return response


def test_model_output_cap_fallback_table() -> None:
    with patch("litellm.get_model_info", side_effect=Exception("unknown")):
        assert _model_output_cap("gpt-4o-mini-2024-07-18") == 16_384
        assert _model_output_cap("gpt-4o") == 16_384
        # Provider-qualified forms must hit the table too (CodeRabbit, #1567)
        assert _model_output_cap("openai/gpt-4o-mini") == 16_384
        assert _model_output_cap("azure/gpt-4o-2024-08-06") == 16_384
        assert _model_output_cap("some-exotic-model") is None


def test_model_output_cap_prefers_litellm_info() -> None:
    with patch("litellm.get_model_info", return_value={"max_output_tokens": 4_096}):
        assert _model_output_cap("anything") == 4_096


@pytest.mark.asyncio
async def test_retry_budget_clamped_to_cap(monkeypatch) -> None:
    """cfg 12,288 doubled must clamp to 16,384, not 24,576."""
    extractor = LLMEntityExtractor(model="gpt-4o-mini", max_tokens=12_288, max_retries=1)
    budgets: list[int] = []

    async def fake_acompletion(*args, **kwargs):
        budgets.append(kwargs.get("max_tokens"))
        if len(budgets) == 1:
            return _response("", "length")  # first call truncates
        return _response('{"entities": [], "relationships": []}', "stop")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    result = await extractor.extract("some text", entity_types=["PERSON"])
    assert budgets == [12_288, 16_384]  # clamped, not 24,576
    assert result.metadata.get("error") is None


@pytest.mark.asyncio
async def test_no_retry_when_already_at_cap(monkeypatch) -> None:
    """At the cap there is no headroom: a same-budget retry is a deterministic
    repeat - go straight to the loud persistent-truncation outcome."""
    extractor = LLMEntityExtractor(model="gpt-4o-mini", max_tokens=16_384, max_retries=1)
    calls = 0

    async def fake_acompletion(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _response("", "length")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    result = await extractor.extract("some text", entity_types=["PERSON"])
    assert calls == 1  # no wasted same-budget retry
    assert result.metadata.get("error") == "truncated_response"


@pytest.mark.asyncio
async def test_unknown_model_preserves_unclamped_double(monkeypatch) -> None:
    extractor = LLMEntityExtractor(model="mystery-llm", max_tokens=1_000, max_retries=1)
    budgets: list[int] = []

    async def fake_acompletion(*args, **kwargs):
        budgets.append(kwargs.get("max_tokens"))
        if len(budgets) == 1:
            return _response("", "length")
        return _response('{"entities": [], "relationships": []}', "stop")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    with patch("litellm.get_model_info", side_effect=Exception("unknown")):
        await extractor.extract("some text", entity_types=["PERSON"])
    assert budgets == [1_000, 2_000]  # today's behavior preserved
