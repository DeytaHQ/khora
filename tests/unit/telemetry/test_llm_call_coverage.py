"""Coverage tests: LLM call sites that were previously not emitting telemetry.

Asserts that ``record_llm_call`` is invoked with the expected ``operation``
tag for each of the 5 newly-wired sites:

* ``hyde``                — ``query/hyde.py``
* ``llm_rerank``          — ``query/reranking.py`` (LLMReranker)
* ``listwise_rerank``     — ``query/reranking.py`` (llm_listwise_rerank)
* ``fact_extraction``     — ``engines/chronicle/compression.py``
* ``fact_reconcile``      — ``engines/chronicle/compression.py``
* ``event_extraction``    — ``engines/chronicle/events.py``
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.telemetry import NoOpCollector


class _RecordingCollector(NoOpCollector):
    """NoOp collector that records ``record_llm_call`` invocations."""

    def __init__(self) -> None:
        self.llm_calls: list[dict] = []

    def record_llm_call(self, **kwargs):  # type: ignore[override]
        self.llm_calls.append(kwargs)


@pytest.fixture
def recording_collector():
    collector = _RecordingCollector()
    with patch("khora.telemetry._collector", collector):
        yield collector


def _make_litellm_response(content: str = "ok") -> MagicMock:
    """Build a litellm-shaped response with usage."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.model = "gpt-4o-mini"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    return response


# ---------------------------------------------------------------------------
# query/hyde.py — HyDEExpander.generate_hypothetical
# ---------------------------------------------------------------------------


class TestHyDETelemetry:
    @pytest.mark.asyncio
    async def test_hyde_generation_records_operation(self, recording_collector):
        import litellm

        from khora.query.hyde import HyDEExpander

        embedder = MagicMock()
        expander = HyDEExpander(embedder=embedder)

        with patch.object(litellm, "acompletion", AsyncMock(return_value=_make_litellm_response("hypothetical"))):
            await expander.generate_hypothetical("what is khora?")

        ops = [c["operation"] for c in recording_collector.llm_calls]
        assert "hyde" in ops, f"expected 'hyde' op, got {ops}"


# ---------------------------------------------------------------------------
# query/reranking.py — LLMReranker.rerank
# ---------------------------------------------------------------------------


class TestRerankerTelemetry:
    @pytest.mark.asyncio
    async def test_llm_reranker_records_operation(self, recording_collector):
        import litellm

        from khora.core.models import Chunk
        from khora.query.reranking import LLMReranker, RerankCandidate

        chunk = Chunk(content="some passage content")
        candidates = [RerankCandidate(item=chunk, original_score=0.5, content=chunk.content, metadata={})]

        with patch.object(
            litellm,
            "acompletion",
            AsyncMock(return_value=_make_litellm_response('{"scores": [7]}')),
        ):
            await LLMReranker().rerank("query", candidates, top_k=1)

        ops = [c["operation"] for c in recording_collector.llm_calls]
        assert "llm_rerank" in ops, f"expected 'llm_rerank' op, got {ops}"

    @pytest.mark.asyncio
    async def test_listwise_rerank_records_operation(self, recording_collector, tmp_path, monkeypatch):
        import litellm

        from khora.core.models import Chunk
        from khora.query.reranking import llm_listwise_rerank

        # Force fresh cache dir so the LLM path is exercised, not a cache hit.
        monkeypatch.setenv("HOME", str(tmp_path))

        chunks = [
            (Chunk(content=f"passage {i}"), score)
            for i, score in enumerate([0.50, 0.49, 0.48])  # gap < 0.1 triggers rerank
        ]

        with patch.object(
            litellm,
            "acompletion",
            AsyncMock(return_value=_make_litellm_response("[1, 2, 3]")),
        ):
            await llm_listwise_rerank("query", chunks, top_n=3)

        ops = [c["operation"] for c in recording_collector.llm_calls]
        assert "listwise_rerank" in ops, f"expected 'listwise_rerank' op, got {ops}"


# ---------------------------------------------------------------------------
# engines/chronicle/compression.py — FactExtractor + MemoryCompressor
# ---------------------------------------------------------------------------


class TestChronicleCompressionTelemetry:
    @pytest.mark.asyncio
    async def test_extract_facts_records_operation(self, recording_collector):
        from khora.engines.chronicle.compression import FactExtractor

        extractor = FactExtractor(model="gpt-4o-mini")

        with patch("khora.engines.chronicle.compression.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(
                    '[{"subject":"X","predicate":"is","object":"Y","fact_text":"X is Y","confidence":0.9}]'
                )
            )
            await extractor.extract_facts("X is Y.")

        ops = [c["operation"] for c in recording_collector.llm_calls]
        assert "fact_extraction" in ops, f"expected 'fact_extraction' op, got {ops}"

    @pytest.mark.asyncio
    async def test_reconcile_fact_records_operation(self, recording_collector):
        from khora.engines.chronicle.compression import (
            MemoryCompressor,
            MemoryFact,
        )

        compressor = MemoryCompressor(model="gpt-4o-mini")
        existing = [MemoryFact(subject="A", predicate="is", object_="B", fact_text="A is B")]
        new_fact = MemoryFact(subject="A", predicate="is", object_="C", fact_text="A is C")

        with patch("khora.engines.chronicle.compression.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response('{"operation":"add","target_id":null,"reasoning":"new"}')
            )
            await compressor.reconcile_fact(existing, new_fact)

        ops = [c["operation"] for c in recording_collector.llm_calls]
        assert "fact_reconcile" in ops, f"expected 'fact_reconcile' op, got {ops}"


# ---------------------------------------------------------------------------
# engines/chronicle/events.py — EventExtractor.extract_events
# ---------------------------------------------------------------------------


class TestChronicleEventsTelemetry:
    @pytest.mark.asyncio
    async def test_extract_events_records_operation(self, recording_collector):
        from khora.engines.chronicle.events import EventExtractor

        extractor = EventExtractor(model="gpt-4o-mini")

        with patch("khora.engines.chronicle.events.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_make_litellm_response(
                    '[{"subject":"Alice","verb":"joined","object":"team","confidence":0.9}]'
                )
            )
            await extractor.extract_events("Alice joined the team.")

        ops = [c["operation"] for c in recording_collector.llm_calls]
        assert "event_extraction" in ops, f"expected 'event_extraction' op, got {ops}"
