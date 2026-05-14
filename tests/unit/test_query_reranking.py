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


# ---------------------------------------------------------------------------
# D5: cross-encoder date-prefix experiment (#594)
# ---------------------------------------------------------------------------


class TestCrossEncoderDatePrefix:
    """Issue #594 — when ``include_date_prefix=True``, the cross-encoder receives
    each candidate's content prefixed with the source timestamp so off-the-shelf
    models can use date tokens as a recency signal."""

    @pytest.mark.asyncio
    async def test_default_off_does_not_inject_date(self) -> None:
        reranker = CrossEncoderReranker()  # default include_date_prefix=False
        captured: list = []

        def fake_predict(pairs, batch_size=32):
            captured.extend(pairs)
            return [0.5] * len(pairs)

        reranker._model = MagicMock()
        reranker._model.predict.side_effect = fake_predict

        candidate = RerankCandidate(
            item="x",
            original_score=0.5,
            content="meeting notes",
            metadata={"custom": {"occurred_at": "2026-05-14T10:00:00Z"}},
        )
        await reranker.rerank("query", [candidate], top_k=1)
        assert captured[0][1] == "meeting notes"

    @pytest.mark.asyncio
    async def test_flag_on_prepends_occurred_at_iso_date(self) -> None:
        reranker = CrossEncoderReranker(include_date_prefix=True)
        captured: list = []

        def fake_predict(pairs, batch_size=32):
            captured.extend(pairs)
            return [0.5] * len(pairs)

        reranker._model = MagicMock()
        reranker._model.predict.side_effect = fake_predict

        candidate = RerankCandidate(
            item="x",
            original_score=0.5,
            content="meeting notes",
            metadata={"custom": {"occurred_at": "2026-05-14T10:00:00Z"}},
        )
        await reranker.rerank("query", [candidate], top_k=1)
        # ISO timestamp truncated to YYYY-MM-DD.
        assert captured[0][1].startswith("[2026-05-14] ")
        assert captured[0][1].endswith("meeting notes")

    @pytest.mark.asyncio
    async def test_falls_back_to_sent_at_then_created_at(self) -> None:
        reranker = CrossEncoderReranker(include_date_prefix=True)
        captured: list = []

        def fake_predict(pairs, batch_size=32):
            captured.extend(pairs)
            return [0.5] * len(pairs)

        reranker._model = MagicMock()
        reranker._model.predict.side_effect = fake_predict

        # No occurred_at, but sent_at present.
        sent_only = RerankCandidate(
            item="a",
            original_score=0.5,
            content="email body",
            metadata={"custom": {"sent_at": "2026-04-01"}},
        )
        # No occurred_at or sent_at, fall back to metadata.created_at.
        from datetime import UTC, datetime

        created_only_meta = MagicMock()
        created_only_meta.custom = {}
        created_only_meta.created_at = datetime(2026, 3, 15, tzinfo=UTC)
        created_only = RerankCandidate(
            item="b",
            original_score=0.5,
            content="legacy doc",
            metadata=created_only_meta,
        )
        await reranker.rerank("query", [sent_only, created_only], top_k=2)
        assert captured[0][1].startswith("[2026-04-01] ")
        assert captured[1][1].startswith("[2026-03-15] ")

    @pytest.mark.asyncio
    async def test_no_timestamp_no_prefix_no_crash(self) -> None:
        reranker = CrossEncoderReranker(include_date_prefix=True)
        captured: list = []

        def fake_predict(pairs, batch_size=32):
            captured.extend(pairs)
            return [0.5] * len(pairs)

        reranker._model = MagicMock()
        reranker._model.predict.side_effect = fake_predict

        # Empty metadata — fall through silently to un-prefixed content.
        candidate = RerankCandidate(
            item="x",
            original_score=0.5,
            content="no metadata",
            metadata={},
        )
        await reranker.rerank("query", [candidate], top_k=1)
        assert captured[0][1] == "no metadata"

    @pytest.mark.asyncio
    async def test_date_prefix_with_title_keeps_both(self) -> None:
        """Title and date prefix coexist — date wraps the title."""
        reranker = CrossEncoderReranker(include_date_prefix=True)
        captured: list = []

        def fake_predict(pairs, batch_size=32):
            captured.extend(pairs)
            return [0.5] * len(pairs)

        reranker._model = MagicMock()
        reranker._model.predict.side_effect = fake_predict

        candidate = RerankCandidate(
            item="x",
            original_score=0.5,
            content="body text",
            metadata={"custom": {"occurred_at": "2026-05-14", "title": "Q2 review"}},
        )
        await reranker.rerank("query", [candidate], top_k=1)
        assert "[2026-05-14]" in captured[0][1]
        assert "[Q2 review]" in captured[0][1]
        assert "body text" in captured[0][1]

    def test_create_reranker_separates_cache_by_flag(self) -> None:
        """create_reranker() with and without the flag must yield distinct
        instances so they can coexist in the cache."""
        from khora.query.reranking import _reranker_cache, create_reranker

        _reranker_cache.clear()
        plain = create_reranker(method="cross_encoder")
        prefixed = create_reranker(method="cross_encoder", include_date_prefix=True)
        assert plain is not prefixed
        assert plain._include_date_prefix is False
        assert prefixed._include_date_prefix is True
