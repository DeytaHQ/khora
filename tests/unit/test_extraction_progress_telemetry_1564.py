"""#1564: extraction progress telemetry (extractor side).

Covers the ``llm.py`` half of the progress-telemetry work:

- per-wave INFO in ``extract_multi`` fires even when every batch fails (the
  silent case that ran blind through three 25h benchmark ingests);
- the outcome/rescue/floor counters increment on their real paths;
- the batch truncation warning names the truncation class so genuine
  ``finish_reason=length`` output-cap events are greppable apart from other
  provider truncation reasons.

Hermetic: ``litellm.acompletion`` is mocked, so no network. The counter tests
monkeypatch the module-level instrument singletons with recording fakes
(mirroring ``tests/unit/filter/test_filter_telemetry.py``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from khora.extraction.extractors import llm as llm_mod
from khora.extraction.extractors.llm import LLMEntityExtractor

pytestmark = pytest.mark.unit


class _RecordingCounter:
    """Captures ``.add(value, attributes=...)`` calls for assertions."""

    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


class _CaptureLogs:
    """Collect loguru records at ``level`` for the duration of the block."""

    def __init__(self, level: str) -> None:
        self._level = level
        self.messages: list[str] = []

    def __enter__(self) -> _CaptureLogs:
        self._sink_id = logger.add(lambda m: self.messages.append(str(m)), level=self._level)
        return self

    def __exit__(self, *exc: object) -> None:
        logger.remove(self._sink_id)

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


def _good_response() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = '{"entities": [{"name": "Alice", "entity_type": "PERSON"}], "relationships": []}'
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=200, total_tokens=300)
    resp.model = "test-model"
    return resp


def _empty_response(finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = ""
    resp.choices[0].finish_reason = finish_reason
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=0, total_tokens=100)
    resp.model = "test-model"
    return resp


def _truncated_response(finish_reason: str = "length") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = '{"entities": [{"name": "incomplet'
    resp.choices[0].finish_reason = finish_reason
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    resp.model = "test-model"
    return resp


_TEXTS = [f"a substantive sentence number {i} about the shared alpha project" for i in range(5)]


class TestWaveProgressLogging:
    @pytest.mark.asyncio
    async def test_wave_info_emitted_on_failing_extraction(self) -> None:
        """The silent case: every batch fails, yet a wave line still surfaces."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)
        extractor._wave_size = 2  # 5 batches -> 3 waves

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=_empty_response("stop")),
            patch("khora.telemetry.get_collector") as mock_telem,
            _CaptureLogs("INFO") as logs,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                _TEXTS, batch_size=1, entity_types=["PERSON"], tiered_extraction=False
            )

        assert len(results) == 5
        assert all(r.metadata.get("error") == "empty_response" for r in results)
        assert "extraction wave" in logs.text
        assert "texts 5/5" in logs.text
        assert "errors=5" in logs.text

    @pytest.mark.asyncio
    async def test_wave_info_emitted_on_success(self) -> None:
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=_good_response()),
            patch("khora.telemetry.get_collector") as mock_telem,
            _CaptureLogs("INFO") as logs,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                _TEXTS, batch_size=1, entity_types=["PERSON"], tiered_extraction=False
            )

        assert len(results) == 5
        assert "extraction wave" in logs.text
        assert "texts 5/5" in logs.text
        assert "errors=0" in logs.text


class TestTruncationWarningSplit:
    @pytest.mark.asyncio
    async def test_length_truncation_named_output_cap(self) -> None:
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=_truncated_response("length")),
            patch("khora.telemetry.get_collector") as mock_telem,
            _CaptureLogs("WARNING") as logs,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            await extractor.extract_multi(
                ["t1 substantive text", "t2 substantive text"],
                batch_size=5,
                entity_types=["PERSON"],
                tiered_extraction=False,
            )

        assert "output-cap truncation" in logs.text
        assert "provider truncation" not in logs.text

    @pytest.mark.asyncio
    async def test_other_truncation_named_provider(self) -> None:
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=_truncated_response("MAX_TOKENS")),
            patch("khora.telemetry.get_collector") as mock_telem,
            _CaptureLogs("WARNING") as logs,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            await extractor.extract_multi(
                ["t1 substantive text", "t2 substantive text"],
                batch_size=5,
                entity_types=["PERSON"],
                tiered_extraction=False,
            )

        assert "provider truncation" in logs.text
        assert "output-cap truncation" not in logs.text


class TestExtractionCounters:
    def test_parse_rescued_counter_increments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extractor = LLMEntityExtractor(model="test-model")
        rec = _RecordingCounter()
        monkeypatch.setattr(llm_mod, "_EXTRACTION_PARSE_RESCUED_COUNTER", rec)

        result = extractor._extract_json_from_text(
            'noise before {"entities": [{"name": "X", "entity_type": "CONCEPT"}], "relationships": []} noise after'
        )

        assert len(result.entities) == 1
        assert len(rec.adds) == 1
        assert rec.adds[0][0] == 1

    def test_parse_rescued_not_incremented_when_unparseable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extractor = LLMEntityExtractor(model="test-model")
        rec = _RecordingCounter()
        monkeypatch.setattr(llm_mod, "_EXTRACTION_PARSE_RESCUED_COUNTER", rec)

        result = extractor._extract_json_from_text("no json object here at all")

        assert rec.adds == []
        assert result.metadata.get("error") == "unparseable_response"

    @pytest.mark.asyncio
    async def test_bisection_floor_counter_increments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Persistent truncation on a multi-text batch bisects to two floors."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)
        rec = _RecordingCounter()
        monkeypatch.setattr(llm_mod, "_EXTRACTION_BISECTION_FLOOR_COUNTER", rec)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=_truncated_response("length")),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            await extractor.extract_multi(
                ["t1 substantive text", "t2 substantive text"],
                batch_size=5,
                entity_types=["PERSON"],
                tiered_extraction=False,
            )

        # One floor per bisected single-text half.
        assert len(rec.adds) == 2

    @pytest.mark.asyncio
    async def test_chunk_outcome_taxonomy_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)
        rec = _RecordingCounter()
        monkeypatch.setattr(llm_mod, "_EXTRACTION_CHUNK_OUTCOME_COUNTER", rec)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=_good_response()),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            await extractor.extract_multi(_TEXTS, batch_size=1, entity_types=["PERSON"], tiered_extraction=False)

        outcomes = [attrs["outcome"] for _, attrs in rec.adds]
        assert outcomes == ["extracted"] * 5

    @pytest.mark.asyncio
    async def test_chunk_outcome_taxonomy_lost_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_tokens=1000)
        rec = _RecordingCounter()
        monkeypatch.setattr(llm_mod, "_EXTRACTION_CHUNK_OUTCOME_COUNTER", rec)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=_empty_response("stop")),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            await extractor.extract_multi(_TEXTS, batch_size=1, entity_types=["PERSON"], tiered_extraction=False)

        outcomes = [attrs["outcome"] for _, attrs in rec.adds]
        assert outcomes == ["lost_error"] * 5
