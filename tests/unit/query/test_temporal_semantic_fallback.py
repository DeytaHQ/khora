"""Tier-2 semantic (LLM) temporal-category fallback (#981).

The Aho-Corasick keyword tier (Tier-1) is English-only and phrase-literal:
German queries and paraphrased-English queries with real temporal intent
collapse to NONE. Tier-2 is an opt-in LLM classifier that fires only when
Tier-1 returns NONE and the detector was built with ``llm_enabled=True``.
It classifies into the correct 6-way TemporalCategory and degrades back to
the keyword result (NONE) on failure/timeout per ADR-001.

These tests mock the LLM backend deterministically so no network call is
made. With Tier-2 disabled (default), behavior is keyword-only and zero-cost.
"""

from __future__ import annotations

import pytest

from khora.query import temporal_detection as td
from khora.query.temporal_detection import (
    TemporalCategory,
    TemporalDetector,
    classify_temporal_category_llm,
)


@pytest.fixture(autouse=True)
def _clear_category_cache() -> None:
    td._TEMPORAL_CATEGORY_CACHE.clear()
    yield
    td._TEMPORAL_CATEGORY_CACHE.clear()


# ---------------------------------------------------------------------------
# Disabled by default — keyword-only, zero LLM cost.
# ---------------------------------------------------------------------------


class TestDisabledByDefault:
    def test_default_detector_is_keyword_only(self) -> None:
        detector = TemporalDetector()
        assert detector._llm_enabled is False

    @pytest.mark.asyncio
    async def test_german_collapses_to_none_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With Tier-2 off, German temporal intent still collapses to NONE and
        the LLM is never called."""

        async def _boom(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("LLM must not be called when Tier-2 is disabled")

        monkeypatch.setattr(td, "classify_temporal_category_llm", _boom)

        detector = TemporalDetector()  # llm_enabled defaults False
        signal = await detector.detect_async("Was ist der letzte Deploy?")
        assert signal.category == TemporalCategory.NONE
        assert signal.source == "none"

    @pytest.mark.asyncio
    async def test_english_keyword_hit_skips_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When Tier-1 fires, Tier-2 is never consulted even if enabled."""

        async def _boom(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("LLM must not be called when Tier-1 hits")

        monkeypatch.setattr(td, "classify_temporal_category_llm", _boom)

        detector = TemporalDetector(llm_enabled=True)
        signal = await detector.detect_async("What is the most recent update?")
        assert signal.category == TemporalCategory.RECENCY
        assert signal.source == "dictionary"


# ---------------------------------------------------------------------------
# Enabled — German + paraphrased-English classify into the correct category.
# ---------------------------------------------------------------------------


def _stub_llm(mapping: dict[str, TemporalCategory]):
    async def _classify(query: str, *, model=None, timeout: float = 3.0):
        cat = mapping.get(query, TemporalCategory.NONE)
        confidence = 1.0 if cat is not TemporalCategory.NONE else 0.0
        return cat, confidence

    return _classify


class TestEnabledMultilingualAndParaphrase:
    @pytest.mark.asyncio
    async def test_german_recency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        q = "Was ist der letzte Deploy?"
        monkeypatch.setattr(td, "classify_temporal_category_llm", _stub_llm({q: TemporalCategory.RECENCY}))
        detector = TemporalDetector(llm_enabled=True)
        degradations: list = []
        signal = await detector.detect_async(q, degradations=degradations)
        assert signal.category == TemporalCategory.RECENCY
        assert signal.source == "semantic"
        assert signal.is_temporal is True
        # Happy path: a successful Tier-2 classification records no degradation.
        # NOTE: the bare `degradations` list is checked directly, NOT via
        # assert_no_silent_degradation() - that helper only inspects a result
        # object's metadata/engine_info dict and is vacuous on a plain list.
        assert degradations == []

    @pytest.mark.asyncio
    async def test_german_state_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        q = "Wer leitet derzeit das Phoenix-Projekt?"
        monkeypatch.setattr(td, "classify_temporal_category_llm", _stub_llm({q: TemporalCategory.STATE_QUERY}))
        detector = TemporalDetector(llm_enabled=True)
        degradations: list = []
        signal = await detector.detect_async(q, degradations=degradations)
        assert signal.category == TemporalCategory.STATE_QUERY
        assert signal.source == "semantic"
        assert degradations == []

    @pytest.mark.asyncio
    async def test_paraphrase_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        q = "Walk me through how Alice's role evolved over time."
        # Sanity: Tier-1 misses this paraphrase.
        assert TemporalDetector().detect(q).category == TemporalCategory.NONE
        monkeypatch.setattr(td, "classify_temporal_category_llm", _stub_llm({q: TemporalCategory.CHANGE}))
        detector = TemporalDetector(llm_enabled=True)
        degradations: list = []
        signal = await detector.detect_async(q, degradations=degradations)
        assert signal.category == TemporalCategory.CHANGE
        assert signal.source == "semantic"
        assert degradations == []

    @pytest.mark.asyncio
    async def test_paraphrase_ordinal_timeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        q = "Give me a timeline of these milestones."
        assert TemporalDetector().detect(q).category == TemporalCategory.NONE
        monkeypatch.setattr(td, "classify_temporal_category_llm", _stub_llm({q: TemporalCategory.ORDINAL}))
        detector = TemporalDetector(llm_enabled=True)
        degradations: list = []
        signal = await detector.detect_async(q, degradations=degradations)
        assert signal.category == TemporalCategory.ORDINAL
        assert signal.source == "semantic"
        assert degradations == []

    @pytest.mark.asyncio
    async def test_llm_returns_none_stays_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        q = "What is the capital of France?"
        monkeypatch.setattr(td, "classify_temporal_category_llm", _stub_llm({q: TemporalCategory.NONE}))
        detector = TemporalDetector(llm_enabled=True)
        signal = await detector.detect_async(q)
        assert signal.category == TemporalCategory.NONE
        assert signal.source == "none"


# ---------------------------------------------------------------------------
# Degrade-to-keyword on failure/timeout (ADR-001).
# ---------------------------------------------------------------------------


class TestDegradeToKeyword:
    @pytest.mark.asyncio
    async def test_llm_exception_degrades_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(td, "classify_temporal_category_llm", _raise)
        detector = TemporalDetector(llm_enabled=True)
        degradations: list = []
        signal = await detector.detect_async("Was ist der letzte Deploy?", degradations=degradations)
        assert signal.category == TemporalCategory.NONE
        assert signal.source == "none"
        assert len(degradations) == 1
        assert degradations[0]["component"] == "vectorcypher.temporal_semantic_fallback"
        assert degradations[0]["reason"] == "llm_failed"

    @pytest.mark.asyncio
    async def test_no_degradation_recorded_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        q = "Was ist der letzte Deploy?"
        monkeypatch.setattr(td, "classify_temporal_category_llm", _stub_llm({q: TemporalCategory.RECENCY}))
        detector = TemporalDetector(llm_enabled=True)
        degradations: list = []
        await detector.detect_async(q, degradations=degradations)
        assert degradations == []


# ---------------------------------------------------------------------------
# classify_temporal_category_llm — parsing + caching.
# ---------------------------------------------------------------------------


class TestClassifyCategoryLLM:
    @pytest.mark.asyncio
    async def test_parses_each_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        words = {
            "EXPLICIT": TemporalCategory.EXPLICIT,
            "STATE_QUERY": TemporalCategory.STATE_QUERY,
            "ORDINAL": TemporalCategory.ORDINAL,
            "AGGREGATE": TemporalCategory.AGGREGATE,
            "RECENCY": TemporalCategory.RECENCY,
            "CHANGE": TemporalCategory.CHANGE,
            "NONE": TemporalCategory.NONE,
        }
        for word, expected in words.items():
            td._TEMPORAL_CATEGORY_CACHE.clear()

            async def _fake_acompletion(*args, _word=word, **kwargs):
                return f"{_word}\n"

            monkeypatch.setattr("khora.config.llm.acompletion", _fake_acompletion)
            cat, conf = await classify_temporal_category_llm(f"query for {word}")
            assert cat == expected
            assert conf == 1.0

    @pytest.mark.asyncio
    async def test_unparseable_response_is_none_zero_conf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_acompletion(*args, **kwargs):
            return "I'm not sure honestly"

        monkeypatch.setattr("khora.config.llm.acompletion", _fake_acompletion)
        cat, conf = await classify_temporal_category_llm("ambiguous query")
        assert cat == TemporalCategory.NONE
        assert conf == 0.0

    @pytest.mark.asyncio
    async def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        async def _fake_acompletion(*args, **kwargs):
            calls["n"] += 1
            return "RECENCY"

        monkeypatch.setattr("khora.config.llm.acompletion", _fake_acompletion)
        await classify_temporal_category_llm("repeated query")
        await classify_temporal_category_llm("repeated query")
        assert calls["n"] == 1
