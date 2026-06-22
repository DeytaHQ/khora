"""Unit tests for thinking-model truncation handling in single-doc extract().

Covers:
- ``finish_reason == "MAX_TOKENS"`` (Gemini/Vertex) recognized as truncation,
  not just OpenAI's ``"length"``.
- One automatic retry with a doubled token budget on truncation.
- A persistent truncation surfaces an ADR-001 ``Degradation`` (ERROR-level)
  instead of a silent zero-entity result.
- Known thinking models get a raised first-attempt token budget.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.extraction.extractors.llm import LLMEntityExtractor
from tests.test_helpers.diagnostics import assert_no_silent_degradation


def _truncated_response(finish_reason: str = "length") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = '{"entities": [{"name": "incomplet'
    resp.choices[0].finish_reason = finish_reason
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    resp.model = "test-model"
    return resp


def _good_response() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(
        {"entities": [{"name": "Alice", "entity_type": "PERSON"}], "relationships": []}
    )
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=200, total_tokens=300)
    resp.model = "test-model"
    return resp


class TestMaxTokensFinishReason:
    @pytest.mark.asyncio
    async def test_gemini_max_tokens_treated_as_truncation(self) -> None:
        """Gemini's MAX_TOKENS finish_reason must be handled like 'length'."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)

        # Always truncated, even after the doubled-budget retry.
        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_truncated_response("MAX_TOKENS"),
            ),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract("some text", entity_types=["PERSON"])

        assert len(result.entities) == 0
        assert result.metadata.get("error") == "truncated_response"
        assert result.metadata.get("finish_reason") == "MAX_TOKENS"

    @pytest.mark.asyncio
    async def test_batch_empty_content_with_max_tokens_is_truncation(self) -> None:
        """A thinking model can burn the whole budget and return empty content
        with finish_reason=MAX_TOKENS. The batch path must classify that as a
        truncation (drives bisection), not an empty response."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = ""  # all budget spent on "thinking"
        resp.choices[0].finish_reason = "MAX_TOKENS"
        resp.usage = MagicMock(prompt_tokens=100, completion_tokens=0, total_tokens=100)
        resp.model = "test-model"

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["text1", "text2"],
                batch_size=5,
                entity_types=["PERSON"],
                tiered_extraction=False,
            )

        # Both texts come back flagged as truncated, not empty_response.
        assert len(results) == 2
        for r in results:
            assert r.metadata.get("error") == "truncated_response"


class TestTruncationAutoRetry:
    @pytest.mark.asyncio
    async def test_retry_with_doubled_budget_recovers(self) -> None:
        """First attempt truncates; the retry with a doubled budget succeeds."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)

        responses = [_truncated_response("length"), _good_response()]
        seen_max_tokens: list[int] = []

        async def _fake_acompletion(*args: object, **kwargs: object) -> MagicMock:
            seen_max_tokens.append(int(kwargs["max_tokens"]))  # type: ignore[arg-type]
            return responses.pop(0)

        with (
            patch("litellm.acompletion", new=_fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract("some text", entity_types=["PERSON"])

        # Recovered: one entity, no error flag.
        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"
        assert "error" not in result.metadata
        assert_no_silent_degradation(result)
        # Second call used exactly the doubled budget.
        assert seen_max_tokens == [1000, 2000]

    @pytest.mark.asyncio
    async def test_persistent_truncation_surfaces_degradation(self) -> None:
        """When the retry also truncates, surface an ADR-001 Degradation."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=_truncated_response("length"),
            ),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract("some text", entity_types=["PERSON"])

        assert len(result.entities) == 0
        # Back-compat error flag preserved.
        assert result.metadata.get("error") == "truncated_response"
        # New: structured degradation the caller can read.
        degradations = result.metadata.get("degradations")
        assert degradations, "expected a non-empty degradations list"
        entry = degradations[0]
        assert entry["component"] == "extraction.llm"
        assert entry["reason"] == "truncated_response"


class TestThinkingModelBudgetFloor:
    @pytest.mark.asyncio
    async def test_gemini_25_first_attempt_uses_raised_budget(self) -> None:
        """A thinking model with a too-small configured budget is floored up."""
        # 12288 is the schema default; below the thinking-model floor.
        extractor = LLMEntityExtractor(model="gemini/gemini-2.5-flash", max_retries=1, max_tokens=12288)

        seen_max_tokens: list[int] = []

        async def _fake_acompletion(*args: object, **kwargs: object) -> MagicMock:
            seen_max_tokens.append(int(kwargs["max_tokens"]))  # type: ignore[arg-type]
            return _good_response()

        with (
            patch("litellm.acompletion", new=_fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract("some text", entity_types=["PERSON"])

        assert_no_silent_degradation(result)
        assert seen_max_tokens, "extract did not call the LLM"
        # First attempt budget raised above the configured 12288 for a thinking model.
        assert seen_max_tokens[0] > 12288

    @pytest.mark.asyncio
    async def test_non_thinking_model_uses_configured_budget(self) -> None:
        """A normal model uses exactly the configured budget on the first call."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1, max_tokens=12288)

        seen_max_tokens: list[int] = []

        async def _fake_acompletion(*args: object, **kwargs: object) -> MagicMock:
            seen_max_tokens.append(int(kwargs["max_tokens"]))  # type: ignore[arg-type]
            return _good_response()

        with (
            patch("litellm.acompletion", new=_fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract("some text", entity_types=["PERSON"])

        assert_no_silent_degradation(result)
        assert seen_max_tokens == [12288]
