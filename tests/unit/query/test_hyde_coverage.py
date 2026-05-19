"""Coverage-driven tests for ``khora.query.hyde``.

The module wraps an LLM call for hypothetical-document generation and an
embedder for averaging. Both boundaries are mocked. ``pipeline_stage``
is an async context manager from telemetry — we let it run as-is (it
no-ops without logfire) rather than mocking the telemetry layer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from khora.query.hyde import HyDEExpander, _detect_category, _select_system_prompt
from khora.query.temporal_detection import TemporalCategory


def _stub_embedder(vec: list[float]) -> MagicMock:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=vec)
    return embedder


@pytest.mark.unit
class TestSelectSystemPrompt:
    @pytest.mark.parametrize(
        "category",
        ["recency", "state_query", "change"],
    )
    def test_temporal_categories_use_anchored_prompt(self, category: str) -> None:
        out = _select_system_prompt(category, "2026-05-18")
        assert "2026-05-18" in out
        assert "authored today" in out

    @pytest.mark.parametrize(
        "category",
        [None, "none", "explicit", "ordinal", "aggregate"],
    )
    def test_non_temporal_categories_use_generic_prompt(self, category: str | None) -> None:
        out = _select_system_prompt(category, "2026-05-18")
        assert "2026-05-18" not in out
        assert "preamble" in out


@pytest.mark.unit
class TestDetectCategory:
    def test_returns_category_for_known_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("khora._accel.detect_temporal_category", lambda q: 5)
        assert _detect_category("latest") == TemporalCategory.RECENCY

    def test_returns_none_for_unknown_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("khora._accel.detect_temporal_category", lambda q: 99)
        # 99 is not in CATEGORY_MAP — dict.get returns None
        assert _detect_category("weird") is None

    def test_returns_none_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(q: str) -> int:
            raise RuntimeError("rust accelerator broken")

        monkeypatch.setattr("khora._accel.detect_temporal_category", boom)
        assert _detect_category("anything") is None


@pytest.mark.unit
class TestHyDEExpanderGenerateHypothetical:
    async def test_uses_temporal_prompt_when_category_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        async def fake_acomp(query, config, **kwargs) -> str:
            captured["system_prompt"] = kwargs.get("system_prompt", "")
            captured["query"] = query
            return "hypothetical content"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)

        expander = HyDEExpander(embedder=_stub_embedder([1.0]))
        out = await expander.generate_hypothetical(
            "what changed recently",
            temporal_category=TemporalCategory.RECENCY,
            today="2026-05-18",
        )
        assert out == "hypothetical content"
        assert "2026-05-18" in captured["system_prompt"]

    async def test_defaults_today_to_utc_now(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(query, config, **kwargs) -> str:
            return "result"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        expander = HyDEExpander(embedder=_stub_embedder([1.0]))
        # No exception → defaulting logic ran. Category=None → generic prompt.
        out = await expander.generate_hypothetical("plain query")
        assert out == "result"


@pytest.mark.unit
class TestHyDEExpanderExpandEmbedding:
    async def test_average_with_single_hypothetical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(query, config, **kwargs) -> str:
            return "hyde doc"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        embedder = _stub_embedder([0.0, 0.0, 2.0])
        expander = HyDEExpander(embedder=embedder, num_hypotheticals=1)

        out = await expander.expand_query_embedding(
            "any q",
            query_embedding=[2.0, 0.0, 0.0],
            temporal_category=TemporalCategory.NONE,
        )
        # Average of [2,0,0] and [0,0,2] → [1,0,1]
        assert out == [1.0, 0.0, 1.0]

    async def test_auto_detects_category_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        async def fake_acomp(query, config, **kwargs) -> str:
            captured["sp"] = kwargs.get("system_prompt", "")
            return "doc"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        # Force detector to return RECENCY (id=5)
        monkeypatch.setattr("khora._accel.detect_temporal_category", lambda q: 5)

        embedder = _stub_embedder([1.0, 1.0])
        expander = HyDEExpander(embedder=embedder)
        await expander.expand_query_embedding(
            "what changed",
            query_embedding=[0.0, 0.0],
            temporal_category=None,
        )
        # Auto-detected RECENCY → temporal prompt picked
        assert "authored today" in captured["sp"]

    async def test_failure_returns_original_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(query, config, **kwargs) -> str:
            raise RuntimeError("llm down")

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        embedder = _stub_embedder([1.0])
        expander = HyDEExpander(embedder=embedder)

        original = [9.9, 8.8]
        out = await expander.expand_query_embedding(
            "q", query_embedding=original, temporal_category=TemporalCategory.NONE
        )
        # Failure path → original embedding unchanged
        assert out == original

    async def test_multiple_hypotheticals_averaged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(query, config, **kwargs) -> str:
            return "doc"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        embedder = _stub_embedder([3.0, 3.0])
        expander = HyDEExpander(embedder=embedder, num_hypotheticals=2)
        # query=[0,0] + two hyde=[3,3] → mean = [2,2]
        out = await expander.expand_query_embedding(
            "q",
            query_embedding=[0.0, 0.0],
            temporal_category=TemporalCategory.NONE,
        )
        assert out == [2.0, 2.0]
