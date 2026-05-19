"""Coverage-driven tests for ``khora.query.reranking``.

Targets the three rerankers (cross-encoder, LLM, listwise) plus the
``_date_prefix_for`` helper and module-level ``create_reranker`` /
``rerank_chunks`` / ``rerank_entities`` convenience functions.

All outbound boundaries (sentence-transformers, litellm, disk cache)
are mocked. Pure-Python helpers (RerankCandidate / RerankResult, sort
order, blending math, batching) run unmocked so coverage is real.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.core.models.document import Chunk
from khora.core.models.entity import Entity
from khora.query.reranking import (
    CrossEncoderReranker,
    LLMReranker,
    RerankCandidate,
    Reranker,
    RerankResult,
    _date_prefix_for,
    _reranker_cache,
    create_reranker,
    llm_listwise_rerank,
    rerank_chunks,
    rerank_entities,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _date_prefix_for
# ---------------------------------------------------------------------------


class TestDatePrefixFor:
    def test_returns_empty_when_no_metadata(self) -> None:
        assert _date_prefix_for(None, None) == ""

    def test_prefers_occurred_at_over_sent_at(self) -> None:
        custom = {
            "occurred_at": datetime(2024, 5, 12, 9, 0, tzinfo=UTC),
            "sent_at": datetime(2024, 6, 12, 9, 0, tzinfo=UTC),
        }
        assert _date_prefix_for(None, custom) == "2024-05-12"

    def test_falls_back_to_sent_at(self) -> None:
        custom = {"sent_at": datetime(2024, 7, 3, 8, 30, tzinfo=UTC)}
        assert _date_prefix_for(None, custom) == "2024-07-03"

    def test_handles_date_object(self) -> None:
        custom = {"occurred_at": date(2023, 1, 15)}
        assert _date_prefix_for(None, custom) == "2023-01-15"

    def test_handles_iso_string(self) -> None:
        custom = {"occurred_at": "2024-09-21T12:34:56Z"}
        assert _date_prefix_for(None, custom) == "2024-09-21"

    def test_rejects_malformed_string(self) -> None:
        custom = {"occurred_at": "not a date"}
        # falls through to created_at lookup which is None
        assert _date_prefix_for(None, custom) == ""

    def test_short_string_rejected(self) -> None:
        custom = {"occurred_at": "20"}
        assert _date_prefix_for(None, custom) == ""

    def test_unknown_type_rejected(self) -> None:
        custom = {"occurred_at": 12345}
        assert _date_prefix_for(None, custom) == ""

    def test_falls_back_to_metadata_created_at_attr(self) -> None:
        meta = MagicMock()
        meta.created_at = datetime(2022, 4, 5, tzinfo=UTC)
        # custom missing both keys
        assert _date_prefix_for(meta, {}) == "2022-04-05"

    def test_falls_back_to_metadata_dict_created_at(self) -> None:
        meta = {"created_at": "2021-12-31T00:00:00"}
        assert _date_prefix_for(meta, None) == "2021-12-31"

    def test_custom_not_a_dict_still_checks_metadata(self) -> None:
        # custom is not a dict -> skipped, then checks metadata
        meta = MagicMock()
        meta.created_at = datetime(2025, 6, 1, tzinfo=UTC)
        assert _date_prefix_for(meta, "not a dict") == "2025-06-01"


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


def _candidate(content: str = "passage", score: float = 0.5, **meta_kwargs) -> RerankCandidate:
    # Post-#748: Chunk.metadata is a flat dict; the old ``ChunkMetadata.custom``
    # nesting was flattened into the dict itself.
    meta = dict(meta_kwargs.get("custom", {}))
    return RerankCandidate(
        item=content,
        original_score=score,
        content=content,
        metadata=meta,
    )


class TestCrossEncoderReranker:
    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self) -> None:
        r = CrossEncoderReranker()
        assert await r.rerank("q", []) == []

    @pytest.mark.asyncio
    async def test_predict_called_and_scores_normalized(self) -> None:
        r = CrossEncoderReranker()
        fake_model = MagicMock()
        # raw cross-encoder logits, will be normalized to [0,1]
        fake_model.predict.return_value = [0.1, 0.9, 0.5]
        r._model = fake_model

        cands = [
            _candidate("a", 0.4),
            _candidate("b", 0.6),
            _candidate("c", 0.2),
        ]
        out = await r.rerank("query", cands, top_k=3, blend_weight=0.7)
        assert len(out) == 3
        # b had highest rerank score; should sort top
        assert out[0].item == "b"
        # normalized into [0,1] — max maps to 1.0, min to 0.0
        max_r = max(o.rerank_score for o in out)
        min_r = min(o.rerank_score for o in out)
        assert max_r == pytest.approx(1.0)
        assert min_r == pytest.approx(0.0)
        # results sorted by final_score descending
        scores = [o.final_score for o in out]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_all_same_score_normalizes_to_05(self) -> None:
        r = CrossEncoderReranker()
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.5, 0.5, 0.5]
        r._model = fake_model

        cands = [_candidate(f"c{i}", 0.5) for i in range(3)]
        out = await r.rerank("q", cands)
        for o in out:
            assert o.rerank_score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_top_k_truncates(self) -> None:
        r = CrossEncoderReranker()
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        r._model = fake_model

        cands = [_candidate(f"c{i}", 0.5) for i in range(5)]
        out = await r.rerank("q", cands, top_k=2)
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_doc_title_included_in_pair(self) -> None:
        r = CrossEncoderReranker()
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.5, 0.5]
        r._model = fake_model

        # Post-#748: Chunk.metadata is a flat dict, so the title key lives
        # at the top level of the metadata dict (no longer behind .custom).
        cand1 = RerankCandidate(
            item="x",
            original_score=0.5,
            content="hello",
            metadata={"title": "TitleA"},
        )
        cand2 = RerankCandidate(
            item="y",
            original_score=0.5,
            content="world",
            metadata={"title": "TitleB"},
        )
        await r.rerank("q", [cand1, cand2])
        pairs = fake_model.predict.call_args[0][0]
        assert pairs[0] == ("q", "[TitleA] hello")
        assert pairs[1] == ("q", "[TitleB] world")

    @pytest.mark.asyncio
    async def test_include_date_prefix(self) -> None:
        r = CrossEncoderReranker(include_date_prefix=True)
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.5]
        r._model = fake_model

        cand = RerankCandidate(
            item="x",
            original_score=0.5,
            content="hello",
            metadata={"occurred_at": "2024-05-12T00:00:00", "title": "T"},
        )
        await r.rerank("q", [cand])
        pair = fake_model.predict.call_args[0][0][0]
        # Date prefix is prepended to "[T] hello"
        assert pair[1].startswith("[2024-05-12] [T] hello")

    @pytest.mark.asyncio
    async def test_predict_exception_falls_back_to_original(self) -> None:
        r = CrossEncoderReranker()
        fake_model = MagicMock()
        fake_model.predict.side_effect = RuntimeError("boom")
        r._model = fake_model

        cands = [
            _candidate("a", 0.4),
            _candidate("b", 0.9),
            _candidate("c", 0.2),
        ]
        out = await r.rerank("q", cands, top_k=2)
        assert len(out) == 2
        # falls back to original ordering by original_score desc
        assert out[0].item == "b"
        assert out[1].item == "a"
        assert all(o.rerank_score == o.original_score for o in out)

    @pytest.mark.asyncio
    async def test_lazy_model_load_raises_if_import_fails(self) -> None:
        r = CrossEncoderReranker()
        # Simulate sentence_transformers not installed by patching the module
        # in sys.modules so the import inside _get_model fails
        import sys

        saved = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((RuntimeError, ImportError, TypeError)):
                r._get_model()
        finally:
            if saved is None:
                del sys.modules["sentence_transformers"]
            else:
                sys.modules["sentence_transformers"] = saved


# ---------------------------------------------------------------------------
# LLMReranker
# ---------------------------------------------------------------------------


class TestLLMReranker:
    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self) -> None:
        r = LLMReranker()
        assert await r.rerank("q", []) == []

    @pytest.mark.asyncio
    async def test_scores_parsed_and_normalized(self) -> None:
        r = LLMReranker(batch_size=5)
        fake_response = json.dumps({"scores": [10, 5, 0]})

        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value=fake_response)):
            cands = [
                _candidate("a", 0.1),
                _candidate("b", 0.2),
                _candidate("c", 0.3),
            ]
            out = await r.rerank("q", cands, top_k=3, blend_weight=0.7)

        assert len(out) == 3
        # raw score 10 → normalized 1.0; raw 0 → 0.0
        # top item should be the one with raw=10
        top = out[0]
        assert top.item == "a"
        # final = 0.7 * 1.0 + 0.3 * 0.1 = 0.73
        assert top.final_score == pytest.approx(0.73)

    @pytest.mark.asyncio
    async def test_invalid_json_response_returns_default_5(self) -> None:
        r = LLMReranker()
        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="not json")):
            cands = [_candidate("a", 0.5), _candidate("b", 0.5)]
            out = await r.rerank("q", cands)
        # All assigned 5.0 → normalized 0.5
        for o in out:
            assert o.rerank_score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_padding_when_fewer_scores_than_batch(self) -> None:
        r = LLMReranker()
        fake_response = json.dumps({"scores": [8]})  # only one score for 3 candidates
        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value=fake_response)):
            cands = [
                _candidate("a", 0.1),
                _candidate("b", 0.2),
                _candidate("c", 0.3),
            ]
            out = await r.rerank("q", cands)
        assert len(out) == 3
        # First gets 8/10=0.8, others get 5/10=0.5
        rerank_by_item = {o.item: o.rerank_score for o in out}
        assert rerank_by_item["a"] == pytest.approx(0.8)
        assert rerank_by_item["b"] == pytest.approx(0.5)
        assert rerank_by_item["c"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_clamps_scores_to_range(self) -> None:
        r = LLMReranker()
        fake_response = json.dumps({"scores": [-5, 99, 7]})
        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value=fake_response)):
            cands = [_candidate(f"c{i}", 0.5) for i in range(3)]
            out = await r.rerank("q", cands)
        rerank_by_item = {o.item: o.rerank_score for o in out}
        assert rerank_by_item["c0"] == pytest.approx(0.0)  # clamped from -5
        assert rerank_by_item["c1"] == pytest.approx(1.0)  # clamped from 99
        assert rerank_by_item["c2"] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_doc_title_in_prompt(self) -> None:
        r = LLMReranker()
        ac = AsyncMock(return_value=json.dumps({"scores": [5]}))
        with patch("khora.config.llm.acompletion", new=ac):
            cand = RerankCandidate(
                item="x",
                original_score=0.5,
                content="hello",
                metadata={"title": "MyTitle"},
            )
            await r.rerank("hello query", [cand])
        prompt = ac.call_args[0][0]
        assert "[MyTitle]" in prompt
        assert "hello query" in prompt

    @pytest.mark.asyncio
    async def test_batches_split(self) -> None:
        r = LLMReranker(batch_size=2)
        ac = AsyncMock(
            side_effect=[
                json.dumps({"scores": [8, 6]}),
                json.dumps({"scores": [4]}),
            ]
        )
        with patch("khora.config.llm.acompletion", new=ac):
            cands = [_candidate(f"c{i}", 0.5) for i in range(3)]
            out = await r.rerank("q", cands)
        assert ac.call_count == 2
        assert len(out) == 3

    @pytest.mark.asyncio
    async def test_gather_exception_falls_back(self) -> None:
        # Force the outer gather to raise by patching asyncio.gather
        r = LLMReranker()
        ac = AsyncMock(return_value=json.dumps({"scores": [5]}))
        import asyncio as _asyncio

        async def boom(*a, **k):
            raise RuntimeError("gather fail")

        with patch("khora.config.llm.acompletion", new=ac), patch.object(_asyncio, "gather", new=boom):
            cands = [
                _candidate("a", 0.4),
                _candidate("b", 0.7),
                _candidate("c", 0.1),
            ]
            out = await r.rerank("q", cands, top_k=2)
        # Falls back to original_score ordering
        assert out[0].item == "b"
        assert out[1].item == "a"


# ---------------------------------------------------------------------------
# create_reranker
# ---------------------------------------------------------------------------


class TestCreateReranker:
    def setup_method(self) -> None:
        _reranker_cache.clear()

    def test_cross_encoder_default(self) -> None:
        r = create_reranker()
        assert isinstance(r, CrossEncoderReranker)
        # Cached on second call
        r2 = create_reranker()
        assert r is r2

    def test_cross_encoder_custom_model(self) -> None:
        r = create_reranker(method="cross_encoder", model="other-model")
        assert isinstance(r, CrossEncoderReranker)
        assert r._model_name == "other-model"

    def test_cross_encoder_date_prefix_variants_cached_separately(self) -> None:
        r1 = create_reranker(include_date_prefix=False)
        r2 = create_reranker(include_date_prefix=True)
        assert r1 is not r2
        assert r1._include_date_prefix is False
        assert r2._include_date_prefix is True

    def test_llm_method_returns_fresh_each_time(self) -> None:
        r1 = create_reranker(method="llm")
        r2 = create_reranker(method="llm")
        assert isinstance(r1, LLMReranker)
        assert isinstance(r2, LLMReranker)
        assert r1 is not r2

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reranking method"):
            create_reranker(method="random")


# ---------------------------------------------------------------------------
# rerank_chunks / rerank_entities convenience
# ---------------------------------------------------------------------------


class TestRerankConvenience:
    @pytest.mark.asyncio
    async def test_rerank_chunks_empty(self) -> None:
        assert await rerank_chunks("q", []) == []

    @pytest.mark.asyncio
    async def test_rerank_chunks_uses_reranker(self) -> None:
        chunk = Chunk(content="payload")
        # Make the cross-encoder reranker yield a deterministic order
        fake_reranker = MagicMock(spec=Reranker)
        fake_reranker.rerank = AsyncMock(
            return_value=[RerankResult(item=chunk, original_score=0.8, rerank_score=0.9, final_score=0.95)]
        )
        with patch("khora.query.reranking.create_reranker", return_value=fake_reranker):
            out = await rerank_chunks("q", [(chunk, 0.8)], top_k=5)
        assert out == [(chunk, 0.95)]
        fake_reranker.rerank.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rerank_entities_empty(self) -> None:
        assert await rerank_entities("q", []) == []

    @pytest.mark.asyncio
    async def test_rerank_entities_constructs_content_string(self) -> None:
        e = Entity(name="Alice", description="founder", entity_type="PERSON")
        fake_reranker = MagicMock(spec=Reranker)
        fake_reranker.rerank = AsyncMock(
            return_value=[RerankResult(item=e, original_score=0.3, rerank_score=0.6, final_score=0.5)]
        )
        with patch("khora.query.reranking.create_reranker", return_value=fake_reranker):
            out = await rerank_entities("q", [(e, 0.3)])
        # The reranker should receive a candidate whose content includes name and type
        cands = fake_reranker.rerank.call_args.args[1]
        assert cands[0].content == "Alice: founder (PERSON)"
        assert out == [(e, 0.5)]


# ---------------------------------------------------------------------------
# llm_listwise_rerank
# ---------------------------------------------------------------------------


class TestLLMListwiseRerank:
    @pytest.mark.asyncio
    async def test_returns_input_if_less_than_two(self) -> None:
        chunks = [(Chunk(content="x"), 0.5)]
        out = await llm_listwise_rerank("q", chunks)
        assert out == chunks

    @pytest.mark.asyncio
    async def test_skips_when_gap_above_threshold(self) -> None:
        # gap = 0.9 - 0.2 = 0.7 >= confidence_threshold 0.1 → no rerank
        chunks = [
            (Chunk(content="a"), 0.9),
            (Chunk(content="b"), 0.2),
        ]
        out = await llm_listwise_rerank("q", chunks, confidence_threshold=0.1)
        assert out == chunks

    @pytest.mark.asyncio
    async def test_reranks_when_gap_below_threshold(self, tmp_path, monkeypatch) -> None:
        # Patch the cache dir so the test isn't polluted by user home
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        c1 = Chunk(content="alpha")
        c2 = Chunk(content="beta")
        c3 = Chunk(content="gamma")
        chunks = [(c1, 0.85), (c2, 0.84), (c3, 0.10)]  # gap = 0.01 < 0.1

        # LLM picks order [2, 1, 3]
        ac = AsyncMock(return_value="[2, 1, 3]")
        with patch("khora.config.llm.acompletion", new=ac):
            out = await llm_listwise_rerank("q", chunks, confidence_threshold=0.1, top_n=3)
        ids = [c.id for c, _ in out]
        # c2 (index 1) becomes first, c1 second, c3 third
        assert ids[0] == c2.id
        assert ids[1] == c1.id
        ac.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uses_disk_cache(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        c1 = Chunk(content="alpha")
        c2 = Chunk(content="beta")
        chunks = [(c1, 0.5), (c2, 0.49)]  # gap = 0.01

        # Function converts ranks to 0-indexed by subtracting 1. "[2, 1]" → [1, 0]
        ac = AsyncMock(return_value="[2, 1]")
        with patch("khora.config.llm.acompletion", new=ac):
            out1 = await llm_listwise_rerank("q", chunks, confidence_threshold=0.1, top_n=2)
            out2 = await llm_listwise_rerank("q", chunks, confidence_threshold=0.1, top_n=2)
        # LLM only called once — second call hits cache
        assert ac.await_count == 1
        # Both produce a result with c2 first because LLM picked rank 2 (c2 is at index 1 → c2 first)
        assert out1[0][0].id == c2.id
        assert out2[0][0].id == c2.id

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_input(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        c1 = Chunk(content="alpha")
        c2 = Chunk(content="beta")
        chunks = [(c1, 0.5), (c2, 0.49)]

        ac = AsyncMock(side_effect=RuntimeError("api down"))
        with patch("khora.config.llm.acompletion", new=ac):
            out = await llm_listwise_rerank("q", chunks, confidence_threshold=0.1)
        # Returns original on failure
        assert out == chunks

    @pytest.mark.asyncio
    async def test_corrupt_cache_falls_through(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        c1 = Chunk(content="alpha")
        c2 = Chunk(content="beta")
        chunks = [(c1, 0.5), (c2, 0.49)]

        # Pre-write a corrupt cache file by running once with a known LLM
        # response, then corrupting the resulting cache file.
        ac = AsyncMock(return_value="[1, 0]")
        with patch("khora.config.llm.acompletion", new=ac):
            await llm_listwise_rerank("q", chunks, confidence_threshold=0.1, top_n=2)
        cache_dir = tmp_path / ".cache" / "khora" / "llm_reranker"
        for f in cache_dir.glob("*.json"):
            f.write_text("not json{[")

        # Next call: cache is corrupt, falls through to LLM again
        ac2 = AsyncMock(return_value="[0, 1]")
        with patch("khora.config.llm.acompletion", new=ac2):
            out = await llm_listwise_rerank("q", chunks, confidence_threshold=0.1, top_n=2)
        ac2.assert_awaited_once()
        assert out[0][0].id == c1.id
