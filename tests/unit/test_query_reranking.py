"""Unit tests for query/reranking.py — Neural reranking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.query.reranking import (
    CrossEncoderReranker,
    LLMReranker,
    RerankCandidate,
    RerankResult,
    create_reranker,
    rerank_chunks,
    rerank_entities,
)


def _make_candidate(content: str = "test", score: float = 0.5) -> RerankCandidate:
    """Create a RerankCandidate."""
    return RerankCandidate(item=content, original_score=score, content=content)


class TestRerankCandidate:
    """Tests for RerankCandidate dataclass."""

    def test_create(self) -> None:
        """Basic creation."""
        c = _make_candidate("doc text", 0.8)
        assert c.item == "doc text"
        assert c.original_score == 0.8
        assert c.content == "doc text"
        assert c.metadata == {}


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_create(self) -> None:
        """Basic creation."""
        r = RerankResult(item="doc", original_score=0.5, rerank_score=0.8, final_score=0.71)
        assert r.item == "doc"
        assert r.final_score == 0.71


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    @pytest.mark.asyncio
    async def test_empty_candidates(self) -> None:
        """Empty candidates returns empty results."""
        reranker = CrossEncoderReranker()
        results = await reranker.rerank("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_with_mock_model(self) -> None:
        """Rerank with mocked cross-encoder model."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3]
        reranker._model = mock_model

        candidates = [_make_candidate("relevant doc", 0.5), _make_candidate("irrelevant", 0.5)]
        results = await reranker.rerank("query", candidates, top_k=2)

        assert len(results) == 2
        # Higher rerank score should rank first
        assert results[0].rerank_score > results[1].rerank_score

    @pytest.mark.asyncio
    async def test_custom_blend_weight(self) -> None:
        """Blend weight parameterises rerank vs original score mix."""
        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        # Two candidates with identical original_score but different rerank scores.
        # With scores [0.9, 0.1], min-max normalises to [1.0, 0.0].
        mock_model.predict.return_value = [0.9, 0.1]
        reranker._model = mock_model

        candidates = [_make_candidate("a", 0.5), _make_candidate("b", 0.5)]

        # blend_weight=0.5 → final = 0.5 * normalized_rerank + 0.5 * original_score
        results = await reranker.rerank("query", candidates, top_k=2, blend_weight=0.5)
        # First result: 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert results[0].final_score == pytest.approx(0.75)
        # Second result: 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        assert results[1].final_score == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_top_k_limit(self) -> None:
        """Results are limited to top_k."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        reranker._model = mock_model

        candidates = [_make_candidate(f"doc{i}", 0.5) for i in range(3)]
        results = await reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_fallback_on_error(self) -> None:
        """Falls back to original ranking on error."""
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.side_effect = Exception("model error")

        candidates = [_make_candidate("a", 0.9), _make_candidate("b", 0.3)]
        results = await reranker.rerank("query", candidates)

        # Should fall back to original scores
        assert len(results) == 2
        assert results[0].final_score == 0.9


class TestLLMReranker:
    """Tests for LLMReranker."""

    @pytest.mark.asyncio
    async def test_empty_candidates(self) -> None:
        """Empty candidates returns empty results."""
        reranker = LLMReranker()
        results = await reranker.rerank("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_with_mock_llm(self) -> None:
        """LLM reranker with mocked response."""
        reranker = LLMReranker(batch_size=10)

        candidates = [_make_candidate("good doc", 0.5), _make_candidate("bad doc", 0.5)]

        with (
            patch(
                "khora.config.llm.acompletion",
                new_callable=AsyncMock,
                return_value='{"scores": [9.0, 2.0]}',
            ),
        ):
            results = await reranker.rerank("query", candidates, top_k=2)

        assert len(results) == 2
        # Higher LLM score should rank first
        assert results[0].rerank_score > results[1].rerank_score

    @pytest.mark.asyncio
    async def test_fallback_on_error(self) -> None:
        """Falls back to original ranking on outer error."""
        reranker = LLMReranker()

        candidates = [_make_candidate("a", 0.9), _make_candidate("b", 0.3)]

        with patch(
            "khora.config.llm.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            results = await reranker.rerank("query", candidates)

        assert len(results) == 2
        # On error, score_batch returns 5.0 (default), normalized to 0.5
        # final = 0.7 * 0.5 + 0.3 * 0.9 = 0.62
        assert results[0].final_score == pytest.approx(0.62)


class TestCreateReranker:
    """Tests for the create_reranker factory."""

    def test_cross_encoder(self) -> None:
        """Creates CrossEncoderReranker."""
        r = create_reranker("cross_encoder")
        assert isinstance(r, CrossEncoderReranker)

    def test_llm(self) -> None:
        """Creates LLMReranker."""
        r = create_reranker("llm")
        assert isinstance(r, LLMReranker)

    def test_unknown_method(self) -> None:
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown reranking method"):
            create_reranker("unknown")


class TestRerankChunks:
    """Tests for the rerank_chunks convenience function."""

    @pytest.mark.asyncio
    async def test_empty_chunks(self) -> None:
        """Empty chunks returns empty."""
        result = await rerank_chunks("query", [])
        assert result == []


class TestRerankEntities:
    """Tests for the rerank_entities convenience function."""

    @pytest.mark.asyncio
    async def test_empty_entities(self) -> None:
        """Empty entities returns empty."""
        result = await rerank_entities("query", [])
        assert result == []
