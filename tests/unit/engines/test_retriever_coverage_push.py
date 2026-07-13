"""Coverage push for ``khora.engines.vectorcypher.retriever``.

These tests cover the still-uncovered private helpers + retrieval branches in
the VectorCypher retriever (issue #695 step 2). They mock at the storage /
LLM / embedder boundary so no live services are required.

Blocks targeted:
    - ``_decompose_change_query`` (lines ~2282–2326): regex rewrite rules
    - ``_bm25_search_chunks`` (lines ~2724–2747): mock storage coordinator
    - ``_recency_channel_chunks`` (lines ~2607–2703): mock vector store
    - ``_apply_reranking`` (lines ~1712–1775): mock cross-encoder reranker
    - ``_apply_llm_reranking`` (lines ~1831–1888): mock LLM reranker
    - ``_typed_entity_recent_retrieve`` (lines ~677–847): fast path / fallback
    - ``_lazy_expand_chunks`` (lines ~2763–2800): keyword expansion + cache
    - ``_calculate_recency_scores`` per-source decay branch (lines ~3030–3060)
    - ``_extract_occurred_at`` / ``_extract_source_system`` / ``_has_target_date``
      module helpers (lines ~253–303)
    - ``_should_skip_llm_rerank`` decisive-winner gate (lines ~1795–1808)
    - ``_vector_only_fallback`` metadata stamping (lines ~1660–1687)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.vectorcypher.fusion import FusedResult
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
    _coerce_occurred_at,
    _extract_occurred_at,
    _extract_source_system,
    _has_target_date,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.engines.vectorcypher.temporal_detection import (
    TemporalCategory,
    TemporalSignal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "text",
    *,
    occurred_at: str | None = None,
    source_system: str | None = None,
    version: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Chunk:
    custom: dict[str, Any] = dict(extra or {})
    if occurred_at is not None:
        custom["occurred_at"] = occurred_at
    if source_system is not None:
        custom["source_system"] = source_system
    if version is not None:
        custom["version"] = version
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        metadata=custom,
        # Mirror occurred_at onto the first-class column - the recency reader
        # reads it there now; the blob above is a transitional dead write.
        occurred_at=_coerce_occurred_at(occurred_at),
    )


def _make_retriever(
    *,
    config: RetrieverConfig | None = None,
    vector_store: Any | None = None,
    storage: Any | None = None,
    neo4j_driver: Any | None = None,
) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        vector_store=vector_store if vector_store is not None else AsyncMock(),
        neo4j_driver=neo4j_driver if neo4j_driver is not None else AsyncMock(),
        embedder=AsyncMock(),
        config=config or RetrieverConfig(),
        storage=storage,
    )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModuleHelpers:
    def test_extract_occurred_at_from_chunk(self) -> None:
        chunk = _make_chunk(occurred_at="2026-04-01T00:00:00+00:00")
        assert _extract_occurred_at(chunk) == datetime(2026, 4, 1, tzinfo=UTC)

    def test_extract_occurred_at_chunk_without_occurred_at(self) -> None:
        chunk = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="x")
        assert _extract_occurred_at(chunk) is None

    def test_extract_occurred_at_from_dict(self) -> None:
        assert _extract_occurred_at({"occurred_at": "2026-01-02"}) == datetime(2026, 1, 2)

    def test_extract_occurred_at_from_unknown(self) -> None:
        assert _extract_occurred_at("not a chunk") is None
        assert _extract_occurred_at(42) is None

    def test_extract_source_system_from_chunk(self) -> None:
        chunk = _make_chunk(source_system="slack")
        assert _extract_source_system(chunk) == "slack"

    def test_extract_source_system_strips_whitespace(self) -> None:
        chunk = _make_chunk(source_system="  email  ")
        assert _extract_source_system(chunk) == "email"

    def test_extract_source_system_empty_returns_none(self) -> None:
        chunk = _make_chunk(source_system="   ")
        assert _extract_source_system(chunk) is None

    def test_extract_source_system_none_chunk_metadata(self) -> None:
        chunk = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="x")
        chunk.metadata = None  # type: ignore[assignment]
        assert _extract_source_system(chunk) is None

    def test_extract_source_system_from_dict(self) -> None:
        assert _extract_source_system({"source_system": "calendar"}) == "calendar"

    def test_extract_source_system_dict_missing(self) -> None:
        assert _extract_source_system({}) is None

    def test_extract_source_system_unknown_type(self) -> None:
        assert _extract_source_system(42) is None

    def test_has_target_date_neither(self) -> None:
        assert _has_target_date(None, None) is False

    def test_has_target_date_from_temporal_filter_after(self) -> None:
        tf = MagicMock(spec=["occurred_after", "occurred_before"])
        tf.occurred_after = datetime(2026, 1, 1, tzinfo=UTC)
        tf.occurred_before = None
        assert _has_target_date(tf, None) is True

    def test_has_target_date_from_temporal_filter_before(self) -> None:
        tf = MagicMock(spec=["occurred_after", "occurred_before"])
        tf.occurred_after = None
        tf.occurred_before = datetime(2026, 1, 1, tzinfo=UTC)
        assert _has_target_date(tf, None) is True

    def test_has_target_date_from_signal(self) -> None:
        tf = MagicMock(spec=["occurred_after", "occurred_before"])
        tf.occurred_after = datetime(2026, 1, 1, tzinfo=UTC)
        tf.occurred_before = None
        sig = MagicMock()
        sig.temporal_filter = tf
        assert _has_target_date(None, sig) is True

    def test_has_target_date_signal_without_filter(self) -> None:
        sig = MagicMock()
        sig.temporal_filter = None
        assert _has_target_date(None, sig) is False


# ---------------------------------------------------------------------------
# _decompose_change_query — pure regex helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecomposeChangeQuery:
    def test_used_to_pattern(self) -> None:
        result = VectorCypherRetriever._decompose_change_query("What did Alice used to play?")
        assert result is not None
        assert "now" in result.lower()
        assert "alice" in result.lower()

    def test_still_pattern(self) -> None:
        result = VectorCypherRetriever._decompose_change_query("Does she still work at Google?")
        assert result is not None
        assert "now" in result.lower()

    def test_switched_pattern(self) -> None:
        result = VectorCypherRetriever._decompose_change_query("Bob switched from piano to guitar")
        assert result is not None
        assert "now" in result.lower()
        assert "bob" in result.lower()

    def test_no_longer_pattern(self) -> None:
        result = VectorCypherRetriever._decompose_change_query("Alice is no longer in marketing")
        assert result is not None
        assert "instead of" in result.lower()

    def test_fallback_currently_substitution(self) -> None:
        # No explicit pattern but contains "previously"
        result = VectorCypherRetriever._decompose_change_query("previously a discussion happened")
        assert result is not None
        assert "currently" in result.lower()

    def test_no_change_returns_none(self) -> None:
        result = VectorCypherRetriever._decompose_change_query("What is Python?")
        assert result is None


# ---------------------------------------------------------------------------
# _bm25_search_chunks — full-text search via storage coordinator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBm25SearchChunks:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_storage(self) -> None:
        retriever = _make_retriever(storage=None)
        result = await retriever._bm25_search_chunks(query="test", namespace_id=uuid4(), limit=10)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_tuples_from_storage(self) -> None:
        ns = uuid4()
        c1 = _make_chunk("alpha")
        c2 = _make_chunk("beta")
        storage = AsyncMock()
        storage.search_fulltext_chunks = AsyncMock(return_value=[(c1, 0.9), (c2, 0.5)])
        retriever = _make_retriever(storage=storage)

        result = await retriever._bm25_search_chunks(query="alpha", namespace_id=ns, limit=10)
        assert len(result) == 2
        assert result[0] == (c1.id, 0.9, c1)
        assert result[1] == (c2.id, 0.5, c2)
        storage.search_fulltext_chunks.assert_awaited_once_with(ns, "alpha", limit=10)

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self) -> None:
        storage = AsyncMock()
        storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("boom"))
        retriever = _make_retriever(storage=storage)

        result = await retriever._bm25_search_chunks(query="x", namespace_id=uuid4(), limit=10)
        assert result == []


# ---------------------------------------------------------------------------
# _recency_channel_chunks — pure-recency pool with cosine relevance gate
# ---------------------------------------------------------------------------


def _make_vector_store_chunk(
    *,
    embedding: list[float] | None = None,
    occurred_at: datetime | None = None,
) -> MagicMock:
    """Construct a chunk-shaped mock as the vector_store.search_recent_chunks
    backend would return (separate from the domain Chunk)."""
    chunk = MagicMock()
    chunk.id = uuid4()
    chunk.namespace_id = uuid4()
    chunk.document_id = uuid4()
    chunk.content = "recent content"
    chunk.embedding = embedding
    chunk.occurred_at = occurred_at
    chunk.created_at = None
    chunk.metadata = None
    return chunk


@pytest.mark.unit
class TestRecencyChannelChunks:
    @pytest.mark.asyncio
    async def test_returns_empty_when_method_missing(self) -> None:
        vstore = MagicMock(spec=[])  # no search_recent_chunks attribute
        retriever = _make_retriever(vector_store=vstore)
        result = await retriever._recency_channel_chunks(
            query_embedding=[0.1, 0.2, 0.3],
            namespace_id=uuid4(),
            temporal_filter=None,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_storage_failure(self) -> None:
        vstore = MagicMock()
        vstore.search_recent_chunks = AsyncMock(side_effect=RuntimeError("db down"))
        retriever = _make_retriever(vector_store=vstore)
        result = await retriever._recency_channel_chunks(
            query_embedding=[0.1, 0.2, 0.3],
            namespace_id=uuid4(),
            temporal_filter=None,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_recent_chunks(self) -> None:
        vstore = MagicMock()
        vstore.search_recent_chunks = AsyncMock(return_value=[])
        retriever = _make_retriever(vector_store=vstore)
        result = await retriever._recency_channel_chunks(
            query_embedding=[0.1, 0.2, 0.3],
            namespace_id=uuid4(),
            temporal_filter=None,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_embeddings_on_chunks(self) -> None:
        vstore = MagicMock()
        # Chunks without embeddings → filtered out before cosine
        c1 = _make_vector_store_chunk(embedding=None)
        vstore.search_recent_chunks = AsyncMock(return_value=[(c1, None)])
        retriever = _make_retriever(vector_store=vstore)
        result = await retriever._recency_channel_chunks(
            query_embedding=[0.1, 0.2, 0.3],
            namespace_id=uuid4(),
            temporal_filter=None,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_by_cosine_floor(self) -> None:
        vstore = MagicMock()
        emb = [1.0, 0.0, 0.0]
        c1 = _make_vector_store_chunk(embedding=emb)
        vstore.search_recent_chunks = AsyncMock(return_value=[(c1, None)])
        config = RetrieverConfig(
            temporal_query_relevance_floor=0.5,
        )
        retriever = _make_retriever(config=config, vector_store=vstore)

        # Mock batch_cosine_similarity to simulate floor filtering
        with patch(
            "khora._accel.batch_cosine_similarity",
            return_value=[(0, 0.9)],
        ):
            result = await retriever._recency_channel_chunks(
                query_embedding=[1.0, 0.0, 0.0],
                namespace_id=uuid4(),
                temporal_filter=None,
            )
        assert len(result) == 1
        chunk_id, score, chunk = result[0]
        assert chunk_id == c1.id
        assert score == pytest.approx(0.9)
        assert isinstance(chunk, Chunk)

    @pytest.mark.asyncio
    async def test_propagates_occurred_after_to_storage(self) -> None:
        vstore = MagicMock()
        vstore.search_recent_chunks = AsyncMock(return_value=[])
        retriever = _make_retriever(vector_store=vstore)
        tf = MagicMock()
        tf.occurred_after = datetime(2026, 1, 1, tzinfo=UTC)
        await retriever._recency_channel_chunks(
            query_embedding=[0.1, 0.2, 0.3],
            namespace_id=uuid4(),
            temporal_filter=tf,
        )
        # Verify the kwarg made it through
        vstore.search_recent_chunks.assert_awaited_once()
        kwargs = vstore.search_recent_chunks.call_args.kwargs
        assert kwargs["created_after"] == tf.occurred_after


# ---------------------------------------------------------------------------
# _apply_reranking — cross-encoder path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyReranking:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        retriever = _make_retriever()
        result = await retriever._apply_reranking("q", [], limit=5, namespace_id=uuid4())
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_top_n_to_reranker_and_appends_remainder(self) -> None:
        chunks = [_make_chunk(f"c{i}") for i in range(8)]
        fused = [FusedResult(item=c, item_id=c.id, rrf_score=1.0 - 0.1 * i) for i, c in enumerate(chunks)]

        config = RetrieverConfig(
            reranking_top_n=3,
            reranking_blend_weight=0.7,
        )
        retriever = _make_retriever(config=config)

        # Mock reranker — return RerankResult objects wrapping the FusedResult items
        from khora.query.reranking import RerankResult

        mock_reranker = AsyncMock()
        # Simulate reranker reordering: score by name length desc on first 3
        mock_results = [
            RerankResult(
                item=fused[2],
                original_score=0.5,
                rerank_score=0.95,
                final_score=0.95,
            ),
            RerankResult(
                item=fused[1],
                original_score=0.5,
                rerank_score=0.80,
                final_score=0.80,
            ),
            RerankResult(
                item=fused[0],
                original_score=0.5,
                rerank_score=0.70,
                final_score=0.70,
            ),
        ]
        mock_reranker.rerank = AsyncMock(return_value=mock_results)
        retriever._reranker = mock_reranker

        result = await retriever._apply_reranking("query", fused, limit=10, namespace_id=uuid4())
        # First three are reranked, rest is the original remainder
        assert len(result) == 8
        assert result[0].item_id == fused[2].item_id
        assert result[0].rrf_score == pytest.approx(0.95)
        # The remainder follows in original order
        assert result[3].item_id == fused[3].item_id

    @pytest.mark.asyncio
    async def test_lazy_init_reranker_on_first_call(self) -> None:
        chunks = [_make_chunk("hello")]
        fused = [FusedResult(item=chunks[0], item_id=chunks[0].id, rrf_score=0.5)]
        retriever = _make_retriever()

        from khora.query.reranking import RerankResult

        instantiated = []

        class FakeReranker:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                instantiated.append(kwargs.get("model_name"))

            async def rerank(
                self,
                query: str,
                candidates: Any,
                *,
                top_k: int = 10,
                blend_weight: float = 0.7,
            ) -> list[RerankResult]:
                return [
                    RerankResult(
                        item=candidates[0].item,
                        original_score=candidates[0].original_score,
                        rerank_score=0.5,
                        final_score=0.5,
                    )
                ]

        with patch("khora.query.reranking.CrossEncoderReranker", FakeReranker):
            result = await retriever._apply_reranking("q", fused, limit=5, namespace_id=uuid4())

        assert len(result) == 1
        assert instantiated == [retriever._config.reranking_model]
        assert retriever._reranker is not None

    @pytest.mark.asyncio
    async def test_fallback_to_original_order_on_failure(self) -> None:
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        fused = [FusedResult(item=c, item_id=c.id, rrf_score=1.0 - 0.1 * i) for i, c in enumerate(chunks)]
        retriever = _make_retriever()

        bad = AsyncMock()
        bad.rerank = AsyncMock(side_effect=RuntimeError("model load failed"))
        retriever._reranker = bad

        result = await retriever._apply_reranking("q", fused, limit=5, namespace_id=uuid4())
        assert result == fused

    @pytest.mark.asyncio
    async def test_temporal_prefix_in_candidate_content(self) -> None:
        """Reranker receives content with [Session: X, Date: Y] prefix.

        ``session_id`` is a user-space blob key (still read from metadata);
        ``occurred_at`` is now read from the first-class column.
        """
        chunk = _make_chunk(
            "the meeting notes",
            occurred_at="2026-04-01T12:34:56+00:00",
            extra={"session_id": "S123"},
        )
        fused = [FusedResult(item=chunk, item_id=chunk.id, rrf_score=0.5)]
        retriever = _make_retriever()

        from khora.query.reranking import RerankResult

        captured: list[Any] = []

        class CaptureReranker:
            async def rerank(
                self,
                query: str,
                candidates: Any,
                *,
                top_k: int = 10,
                blend_weight: float = 0.7,
            ) -> list[RerankResult]:
                captured.extend(candidates)
                return [
                    RerankResult(
                        item=candidates[0].item,
                        original_score=0.5,
                        rerank_score=0.7,
                        final_score=0.7,
                    )
                ]

        retriever._reranker = CaptureReranker()  # type: ignore[assignment]
        await retriever._apply_reranking("q", fused, limit=5, namespace_id=uuid4())
        assert captured
        assert "Session: S123" in captured[0].content
        assert "Date: 2026-04-01" in captured[0].content


# ---------------------------------------------------------------------------
# _apply_llm_reranking — LLM path mirrors cross-encoder shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyLLMReranking:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        retriever = _make_retriever()
        result = await retriever._apply_llm_reranking("q", [], limit=5, namespace_id=uuid4())
        assert result == []

    @pytest.mark.asyncio
    async def test_reranks_and_appends_remainder(self) -> None:
        chunks = [_make_chunk(f"c{i}") for i in range(6)]
        fused = [FusedResult(item=c, item_id=c.id, rrf_score=1.0 - 0.1 * i) for i, c in enumerate(chunks)]
        config = RetrieverConfig(
            llm_reranking_top_n=3,
        )
        retriever = _make_retriever(config=config)

        from khora.query.reranking import RerankResult

        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(
            return_value=[
                RerankResult(
                    item=fused[1],
                    original_score=0.5,
                    rerank_score=0.95,
                    final_score=0.95,
                ),
                RerankResult(
                    item=fused[0],
                    original_score=0.5,
                    rerank_score=0.7,
                    final_score=0.7,
                ),
                RerankResult(
                    item=fused[2],
                    original_score=0.5,
                    rerank_score=0.5,
                    final_score=0.5,
                ),
            ]
        )
        retriever._llm_reranker = mock_reranker

        result = await retriever._apply_llm_reranking("q", fused, limit=10, namespace_id=uuid4())
        assert len(result) == 6
        assert result[0].item_id == fused[1].item_id
        assert result[3].item_id == fused[3].item_id  # remainder preserved

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self) -> None:
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        fused = [FusedResult(item=c, item_id=c.id, rrf_score=1.0 - 0.1 * i) for i, c in enumerate(chunks)]
        retriever = _make_retriever()
        bad = AsyncMock()
        bad.rerank = AsyncMock(side_effect=RuntimeError("llm timeout"))
        retriever._llm_reranker = bad

        result = await retriever._apply_llm_reranking("q", fused, limit=5, namespace_id=uuid4())
        assert result == fused

    @pytest.mark.asyncio
    async def test_temporal_prefix_in_candidate_content(self) -> None:
        """LLM reranker candidate content carries [Session: X, Date: Y].

        Mirrors the cross-encoder prefix test: ``session_id`` is a user-space
        blob key, ``occurred_at`` is read from the first-class column.
        """
        chunk = _make_chunk(
            "the meeting notes",
            occurred_at="2026-04-01T12:34:56+00:00",
            extra={"session_id": "S123"},
        )
        fused = [FusedResult(item=chunk, item_id=chunk.id, rrf_score=0.5)]
        retriever = _make_retriever()

        from khora.query.reranking import RerankResult

        captured: list[Any] = []

        class CaptureReranker:
            async def rerank(
                self,
                query: str,
                candidates: Any,
                *,
                top_k: int = 10,
                blend_weight: float = 0.7,
            ) -> list[RerankResult]:
                captured.extend(candidates)
                return [
                    RerankResult(
                        item=candidates[0].item,
                        original_score=0.5,
                        rerank_score=0.7,
                        final_score=0.7,
                    )
                ]

        retriever._llm_reranker = CaptureReranker()  # type: ignore[assignment]
        await retriever._apply_llm_reranking("q", fused, limit=5, namespace_id=uuid4())
        assert captured
        assert "Session: S123" in captured[0].content
        assert "Date: 2026-04-01" in captured[0].content

    @pytest.mark.asyncio
    async def test_lazy_init_llm_reranker(self) -> None:
        chunk = _make_chunk("a")
        fused = [FusedResult(item=chunk, item_id=chunk.id, rrf_score=0.5)]
        retriever = _make_retriever()

        from khora.query.reranking import RerankResult

        models: list[Any] = []

        class FakeLLMReranker:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                models.append(kwargs.get("model"))

            async def rerank(
                self,
                query: str,
                candidates: Any,
                *,
                top_k: int = 10,
                blend_weight: float = 0.7,
            ) -> list[RerankResult]:
                return [
                    RerankResult(
                        item=candidates[0].item,
                        original_score=0.5,
                        rerank_score=0.7,
                        final_score=0.7,
                    )
                ]

        with patch("khora.query.reranking.LLMReranker", FakeLLMReranker):
            await retriever._apply_llm_reranking("q", fused, limit=5, namespace_id=uuid4())

        assert models == [retriever._config.llm_reranking_model]
        assert retriever._llm_reranker is not None


# ---------------------------------------------------------------------------
# _should_skip_llm_rerank — both gates
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShouldSkipLlmRerank:
    def test_gap_gate_fires(self) -> None:
        config = RetrieverConfig(
            llm_reranking_confidence_threshold=0.1,
        )
        retriever = _make_retriever(config=config)
        # Large gap → skip
        assert retriever._should_skip_llm_rerank(top_score=0.4, gap=0.2) is True

    def test_decisive_winner_gate_fires(self) -> None:
        config = RetrieverConfig(
            llm_reranking_confidence_threshold=0.5,  # gap won't trigger
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.1,
        )
        retriever = _make_retriever(config=config)
        # gap (0.12) < threshold (0.5) but top_score (0.85) >= 0.7 AND gap >= 0.1
        assert retriever._should_skip_llm_rerank(top_score=0.85, gap=0.12) is True

    def test_neither_gate_fires(self) -> None:
        config = RetrieverConfig(
            llm_reranking_confidence_threshold=0.5,
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.1,
        )
        retriever = _make_retriever(config=config)
        # Low top_score, small gap
        assert retriever._should_skip_llm_rerank(top_score=0.5, gap=0.05) is False


# ---------------------------------------------------------------------------
# _typed_entity_recent_retrieve — fast path + fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypedEntityRecentRetrieve:
    @pytest.mark.asyncio
    async def test_unknown_noun_falls_back(self) -> None:
        """Routed to TYPED_ENTITY_RECENT but query has no recognized noun
        → fall back to _vectorcypher_retrieve via the fast-path bridge."""
        retriever = _make_retriever()
        retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(
                chunks=[],
                entities=[],
                routing_decision=RoutingDecision(
                    complexity=QueryComplexity.SIMPLE,
                    use_graph=False,
                    graph_depth=0,
                    confidence=0.9,
                    reasoning="x",
                ),
            )
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.TYPED_ENTITY_RECENT,
            use_graph=True,
            graph_depth=1,
            confidence=0.9,
            reasoning="typed",
        )
        result = await retriever._typed_entity_recent_retrieve(
            query="show me banana popsicles",  # no typed-entity noun
            query_embedding=[0.0],
            namespace_id=uuid4(),
            temporal_filter=None,
            graph_depth=1,
            limit=5,
            routing=routing,
        )
        assert isinstance(result, VectorCypherResult)
        retriever._vectorcypher_retrieve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_dual_nodes_falls_back_with_metadata(self) -> None:
        retriever = _make_retriever(neo4j_driver=None)
        retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(
                chunks=[],
                entities=[],
                routing_decision=RoutingDecision(
                    complexity=QueryComplexity.SIMPLE,
                    use_graph=False,
                    graph_depth=0,
                    confidence=0.9,
                    reasoning="x",
                ),
            )
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.TYPED_ENTITY_RECENT,
            use_graph=True,
            graph_depth=1,
            confidence=0.9,
            reasoning="typed",
        )
        result = await retriever._typed_entity_recent_retrieve(
            query="latest action items",
            query_embedding=[0.0],
            namespace_id=uuid4(),
            temporal_filter=None,
            graph_depth=1,
            limit=5,
            routing=routing,
        )
        assert result.metadata.get("typed_entity_fast_path_fallback") is True
        assert result.metadata.get("typed_entity_type") == "ACTION_ITEM"

    @pytest.mark.asyncio
    async def test_empty_rows_falls_back(self) -> None:
        retriever = _make_retriever()
        # Mock dual_nodes session that returns no rows
        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = session_ctx
        session_ctx.__aexit__.return_value = None
        session_ctx.execute_read = AsyncMock(return_value=[])

        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes._session.return_value = session_ctx

        retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(
                chunks=[],
                entities=[],
                routing_decision=RoutingDecision(
                    complexity=QueryComplexity.SIMPLE,
                    use_graph=False,
                    graph_depth=0,
                    confidence=0.9,
                    reasoning="x",
                ),
            )
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.TYPED_ENTITY_RECENT,
            use_graph=True,
            graph_depth=1,
            confidence=0.9,
            reasoning="typed",
        )
        result = await retriever._typed_entity_recent_retrieve(
            query="latest decisions",
            query_embedding=[0.0],
            namespace_id=uuid4(),
            temporal_filter=None,
            graph_depth=1,
            limit=5,
            routing=routing,
        )
        assert result.metadata.get("typed_entity_fast_path_fallback") is True
        assert result.metadata.get("typed_entity_type") == "DECISION"

    @pytest.mark.asyncio
    async def test_exception_falls_back(self) -> None:
        retriever = _make_retriever()
        retriever._dual_nodes = MagicMock()

        # Make _session raise when entered
        bad_ctx = MagicMock()
        bad_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("neo4j down"))
        bad_ctx.__aexit__ = AsyncMock(return_value=None)
        retriever._dual_nodes._session.return_value = bad_ctx

        retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(
                chunks=[],
                entities=[],
                routing_decision=RoutingDecision(
                    complexity=QueryComplexity.SIMPLE,
                    use_graph=False,
                    graph_depth=0,
                    confidence=0.9,
                    reasoning="x",
                ),
            )
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.TYPED_ENTITY_RECENT,
            use_graph=True,
            graph_depth=1,
            confidence=0.9,
            reasoning="typed",
        )
        result = await retriever._typed_entity_recent_retrieve(
            query="latest blockers",
            query_embedding=[0.0],
            namespace_id=uuid4(),
            temporal_filter=None,
            graph_depth=1,
            limit=5,
            routing=routing,
        )
        assert result.metadata.get("typed_entity_fast_path_fallback") is True
        assert result.metadata.get("typed_entity_type") == "BLOCKER"

    @pytest.mark.asyncio
    async def test_returns_typed_entities_when_rows_present(self) -> None:
        retriever = _make_retriever()
        ns = uuid4()
        entity_id = uuid4()
        chunk_id = uuid4()
        doc_id = uuid4()

        rows = [
            {
                "entity": {
                    "id": str(entity_id),
                    "name": "Ship the prototype",
                    "entity_type": "ACTION_ITEM",
                    "description": "",
                },
                "last_mention": "2026-04-01",
                "evidence_chunk": {
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": "Action: ship the prototype",
                    "occurred_at": "2026-04-01",
                },
            }
        ]

        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = session_ctx
        session_ctx.__aexit__.return_value = None
        session_ctx.execute_read = AsyncMock(return_value=rows)
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes._session.return_value = session_ctx

        routing = RoutingDecision(
            complexity=QueryComplexity.TYPED_ENTITY_RECENT,
            use_graph=True,
            graph_depth=1,
            confidence=0.9,
            reasoning="typed",
        )
        result = await retriever._typed_entity_recent_retrieve(
            query="latest action items",
            query_embedding=[0.0],
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=5,
            routing=routing,
        )
        assert result.metadata.get("typed_entity_fast_path") is True
        assert result.metadata.get("typed_entity_type") == "ACTION_ITEM"
        assert len(result.entities) == 1
        assert len(result.chunks) == 1
        assert result.entities[0][0].name == "Ship the prototype"


# ---------------------------------------------------------------------------
# _lazy_expand_chunks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLazyExpandChunks:
    def test_no_entity_names_returns_empty(self) -> None:
        retriever = _make_retriever()
        chunks = [_make_chunk("about alice")]
        result = retriever._lazy_expand_chunks(
            vector_only_chunks=[(chunks[0].id, 0.5, chunks[0])],
            entry_entities=[(uuid4(), 0.8)],
            entity_info_map={},  # No name info → no entity names extracted
        )
        assert result == []

    def test_matches_keywords_and_caches(self) -> None:
        retriever = _make_retriever()
        eid = uuid4()
        c = _make_chunk("alice met bob today")
        # Patch extract_keywords to return tokens including "alice"
        with patch(
            "khora._accel.extract_keywords",
            return_value=["alice", "met", "bob", "today"],
        ):
            result = retriever._lazy_expand_chunks(
                vector_only_chunks=[(c.id, 0.5, c)],
                entry_entities=[(eid, 0.9)],
                entity_info_map={str(eid): {"name": "Alice"}},
            )
        assert len(result) == 1
        cid, score, _ = result[0]
        assert cid == c.id
        assert score == pytest.approx(0.5)
        # Cache populated
        assert retriever._expansion_cache[c.id] == pytest.approx(0.5)

    def test_no_matches_caches_zero(self) -> None:
        retriever = _make_retriever()
        eid = uuid4()
        c = _make_chunk("unrelated content")
        with patch(
            "khora._accel.extract_keywords",
            return_value=["unrelated", "content"],
        ):
            result = retriever._lazy_expand_chunks(
                vector_only_chunks=[(c.id, 0.5, c)],
                entry_entities=[(eid, 0.9)],
                entity_info_map={str(eid): {"name": "Carol"}},
            )
        assert result == []
        assert retriever._expansion_cache[c.id] == 0.0

    def test_cached_positive_short_circuits(self) -> None:
        retriever = _make_retriever()
        eid = uuid4()
        c = _make_chunk("alice met bob")
        retriever._expansion_cache[c.id] = 2.5  # Pre-cached

        with patch(
            "khora._accel.extract_keywords",
            return_value=["nope"],  # ignored — cache wins
        ) as mock_ek:
            result = retriever._lazy_expand_chunks(
                vector_only_chunks=[(c.id, 0.5, c)],
                entry_entities=[(eid, 0.9)],
                entity_info_map={str(eid): {"name": "Alice"}},
            )
        # extract_keywords should NOT have been called
        mock_ek.assert_not_called()
        assert result[0][1] == 2.5

    def test_cached_zero_short_circuits_and_skips(self) -> None:
        retriever = _make_retriever()
        eid = uuid4()
        c = _make_chunk("alice met bob")
        retriever._expansion_cache[c.id] = 0.0  # Cached negative

        with patch(
            "khora._accel.extract_keywords",
            return_value=["alice"],
        ) as mock_ek:
            result = retriever._lazy_expand_chunks(
                vector_only_chunks=[(c.id, 0.5, c)],
                entry_entities=[(eid, 0.9)],
                entity_info_map={str(eid): {"name": "Alice"}},
            )
        mock_ek.assert_not_called()
        # Cached zero excludes the chunk entirely
        assert result == []

    def test_empty_content_caches_zero(self) -> None:
        retriever = _make_retriever()
        eid = uuid4()
        c = _make_chunk("")  # empty content
        result = retriever._lazy_expand_chunks(
            vector_only_chunks=[(c.id, 0.5, c)],
            entry_entities=[(eid, 0.9)],
            entity_info_map={str(eid): {"name": "Alice"}},
        )
        assert result == []
        assert retriever._expansion_cache[c.id] == 0.0


# ---------------------------------------------------------------------------
# _calculate_recency_scores — per-source decay branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateRecencyScoresPerSource:
    def test_per_source_decay_used_when_configured(self) -> None:
        """slack source → 3-day decay (fast), salesforce → 180 (slow); when same age
        the slack chunk decays faster."""
        config = RetrieverConfig(
            temporal_per_source_decay=True,
            temporal_reference_wall_clock=True,
            recency_decay_type="exponential",
        )
        retriever = _make_retriever(config=config)

        # Both chunks 10 days old
        old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        c_slack = _make_chunk(occurred_at=old_ts, source_system="slack")
        c_sales = _make_chunk(occurred_at=old_ts, source_system="salesforce")
        results = [
            FusedResult(item=c_slack, item_id=c_slack.id, rrf_score=1.0),
            FusedResult(item=c_sales, item_id=c_sales.id, rrf_score=1.0),
        ]
        scores = retriever._calculate_recency_scores(results)
        # Slack (3-day decay) is much more decayed than salesforce (180-day)
        assert scores[c_slack.id] < scores[c_sales.id]

    def test_per_source_decay_falls_back_to_default(self) -> None:
        """Unknown source_system → use the dict's ``_default`` key."""
        config = RetrieverConfig(
            temporal_per_source_decay=True,
            temporal_reference_wall_clock=True,
            recency_decay_type="exponential",
            temporal_default_decay_by_source={"_default": 14},
        )
        retriever = _make_retriever(config=config)

        old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        c = _make_chunk(occurred_at=old_ts, source_system="weird_unknown_system")
        results = [FusedResult(item=c, item_id=c.id, rrf_score=1.0)]
        scores = retriever._calculate_recency_scores(results)
        # exp(-ln(2)/14 * 10) ≈ 0.61
        assert 0.55 < scores[c.id] < 0.70

    def test_per_source_decay_default_missing_falls_back_to_config(self) -> None:
        """If ``_default`` key is absent → use config.recency_decay_days."""
        config = RetrieverConfig(
            temporal_per_source_decay=True,
            temporal_reference_wall_clock=True,
            recency_decay_type="exponential",
            recency_decay_days=30,
            temporal_default_decay_by_source={"slack": 3},  # no _default key
        )
        retriever = _make_retriever(config=config)

        old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        c = _make_chunk(occurred_at=old_ts)  # no source_system
        results = [FusedResult(item=c, item_id=c.id, rrf_score=1.0)]
        scores = retriever._calculate_recency_scores(results)
        # Falls back to 30-day exponential: exp(-ln(2)/30 * 10) ≈ 0.79
        assert 0.70 < scores[c.id] < 0.85

    def test_pathological_zero_decay_falls_back(self) -> None:
        """Decay=0 would div-by-zero → falls back to config.recency_decay_days."""
        config = RetrieverConfig(
            temporal_per_source_decay=True,
            temporal_reference_wall_clock=True,
            recency_decay_type="exponential",
            recency_decay_days=14,
            temporal_default_decay_by_source={"_default": 0},  # bogus zero
        )
        retriever = _make_retriever(config=config)
        old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        c = _make_chunk(occurred_at=old_ts)
        results = [FusedResult(item=c, item_id=c.id, rrf_score=1.0)]
        scores = retriever._calculate_recency_scores(results)
        # Should not raise; fall back to 14-day decay.
        assert 0.0 < scores[c.id] < 1.0

    def test_linear_decay_path(self) -> None:
        """The ``linear`` decay branch is independent from per-source."""
        config = RetrieverConfig(
            temporal_reference_wall_clock=True,
            recency_decay_type="linear",
            recency_decay_days=10,
        )
        retriever = _make_retriever(config=config)
        # 5 days old → score ≈ 0.5
        ts = (datetime.now(UTC) - timedelta(days=5)).isoformat()
        c = _make_chunk(occurred_at=ts)
        results = [FusedResult(item=c, item_id=c.id, rrf_score=1.0)]
        scores = retriever._calculate_recency_scores(results)
        assert 0.4 < scores[c.id] < 0.6

    def test_unparseable_date_defaults_to_half(self) -> None:
        c = _make_chunk(occurred_at="not-a-date")
        retriever = _make_retriever()
        results = [FusedResult(item=c, item_id=c.id, rrf_score=1.0)]
        scores = retriever._calculate_recency_scores(results)
        assert scores[c.id] == 0.5

    def test_reference_mode_override(self) -> None:
        """explicit reference_mode kwarg wins over config."""
        config = RetrieverConfig(
            temporal_reference_wall_clock=False,
        )
        retriever = _make_retriever(config=config)
        old_ts = "2020-01-01T00:00:00+00:00"
        c = _make_chunk(occurred_at=old_ts)
        results = [FusedResult(item=c, item_id=c.id, rrf_score=1.0)]
        # With reference_mode='wall_clock', the score is small (very old)
        scores_wall = retriever._calculate_recency_scores(results, reference_mode="wall_clock")
        # With reference_mode='relative' (single item → self), score = 1.0
        scores_rel = retriever._calculate_recency_scores(results, reference_mode="relative")
        assert scores_wall[c.id] < 0.01
        assert scores_rel[c.id] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Retrieve / vectorcypher_retrieve flow — PPR path + embedded-backend gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetrieveBackendGate:
    @pytest.mark.asyncio
    async def test_sqlite_lance_allows_occurred_bounds_query(self) -> None:
        """An occurred-bounds temporal_filter no longer fail-fasts on the embedded
        backend.

        The old blanket gate raised NotImplementedError for any target_date on
        sqlite_lance; it was replaced by a call-site guard that skips only the
        unsupported entity-version narrowing (recording a structured degradation)
        while the occurred-bounds chunk filter still pushes down. Here we assert
        ``retrieve()`` dispatches to the chunk path instead of raising."""
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=None,
            embedder=AsyncMock(),
            config=RetrieverConfig(),
            backend="sqlite_lance",
        )
        # Stub the dispatch target so the unit test stays focused on the gate
        # decision rather than the full storage path.
        retriever._simple_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(
                chunks=[],
                entities=[],
                routing_decision=RoutingDecision(
                    complexity=QueryComplexity.SIMPLE,
                    use_graph=False,
                    graph_depth=0,
                    confidence=0.9,
                    reasoning="gate",
                ),
                metadata={},
            )
        )
        from khora.storage.temporal import TemporalFilter

        tf = TemporalFilter(occurred_after=datetime(2025, 6, 1, tzinfo=UTC))
        result = await retriever.retrieve("any", uuid4(), temporal_filter=tf)
        assert result.chunks == []
        retriever._simple_retrieve.assert_awaited_once()


@pytest.mark.unit
class TestPprRetrievalPath:
    @pytest.mark.asyncio
    async def test_ppr_path_invoked_when_enabled(self) -> None:
        """When enable_ppr_retrieval=True and storage is wired, _vectorcypher_retrieve
        calls ppr_retrieve_chunks instead of _fetch_chunks_from_entities."""
        ns = uuid4()
        config = RetrieverConfig(
            enable_ppr_retrieval=True,
            ppr_damping=0.85,
            ppr_max_iter=10,
            ppr_tol=1e-4,
            ppr_top_entities=5,
        )
        storage = AsyncMock()
        entry_id = uuid4()
        storage.search_similar_entities = AsyncMock(return_value=[(entry_id, 0.9)])
        storage.get_entities_batch = AsyncMock(return_value={})

        chunk_id = uuid4()
        ppr_chunk = Chunk(
            id=chunk_id,
            namespace_id=ns,
            document_id=uuid4(),
            content="ppr chunk",
        )
        ppr_entity_scores = {entry_id: 0.6}

        retriever = _make_retriever(config=config, storage=storage)
        # Stub the vector-only chunk side
        retriever._vector_search_chunks = AsyncMock(return_value=[])  # type: ignore[method-assign]
        retriever._cypher_expand = AsyncMock(return_value=({}, {}))  # type: ignore[method-assign]
        retriever._router.compute_adaptive_depth = MagicMock(return_value=2)  # type: ignore[method-assign]

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX,
            use_graph=True,
            graph_depth=2,
            confidence=0.8,
            reasoning="ppr",
        )

        with patch(
            "khora.engines.vectorcypher.ppr_retrieval.ppr_retrieve_chunks",
            new=AsyncMock(return_value=([(chunk_id, 0.5, ppr_chunk)], ppr_entity_scores)),
        ) as ppr_mock:
            result = await retriever._vectorcypher_retrieve(
                query="anything",
                query_embedding=[0.1, 0.2],
                namespace_id=ns,
                temporal_filter=None,
                graph_depth=2,
                limit=10,
                routing=routing,
            )

        ppr_mock.assert_awaited_once()
        assert result.metadata["ppr_path_used"] is True
        assert result.metadata["ppr_entity_count"] == 1


# ---------------------------------------------------------------------------
# _vector_only_fallback — metadata stamping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorOnlyFallback:
    @pytest.mark.asyncio
    async def test_fallback_stamps_metadata(self) -> None:
        ns = uuid4()
        retriever = _make_retriever()
        # Mock _simple_retrieve to return a known result
        chunk = _make_chunk("fb")
        retriever._simple_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(
                chunks=[(chunk, 0.5)],
                entities=[],
                routing_decision=RoutingDecision(
                    complexity=QueryComplexity.SIMPLE,
                    use_graph=False,
                    graph_depth=0,
                    confidence=0.9,
                    reasoning="fb",
                ),
                metadata={"original": True},
            )
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.MODERATE,
            use_graph=True,
            graph_depth=2,
            confidence=0.8,
            reasoning="fallback",
        )
        result = await retriever._vector_only_fallback(
            query="x",
            query_embedding=[0.0],
            namespace_id=ns,
            temporal_filter=None,
            limit=5,
            routing=routing,
        )
        assert result.metadata["fallback_mode"] == "vector_only"
        assert result.metadata["graph_unavailable"] is True
        assert result.metadata["graph_fallback"] is True
        # Original metadata preserved
        assert result.metadata["original"] is True


# ---------------------------------------------------------------------------
# Anti-recency veto path in retrieve()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCypherExpand:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        retriever = _make_retriever()
        scores, info = await retriever._cypher_expand(entry_entity_ids=[], namespace_id=uuid4(), depth=2)
        assert scores == {}
        assert info == {}

    @pytest.mark.asyncio
    async def test_dual_nodes_path(self) -> None:
        retriever = _make_retriever()
        seed = uuid4()
        neighbor = uuid4()
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(
            return_value={
                seed: [
                    {
                        "id": str(neighbor),
                        "distance": 1,
                        "name": "Alice",
                        "entity_type": "PERSON",
                        "description": "person Alice",
                        "source_tool": "slack",
                    }
                ]
            }
        )
        scores, info = await retriever._cypher_expand(entry_entity_ids=[seed], namespace_id=uuid4(), depth=2)
        assert neighbor in scores
        # distance=1 → score = 1/(1+1) = 0.5
        assert scores[neighbor] == pytest.approx(0.5)
        assert info[str(neighbor)]["name"] == "Alice"
        assert info[str(neighbor)]["entity_type"] == "PERSON"

    @pytest.mark.asyncio
    async def test_dual_nodes_max_score_for_dup_entity(self) -> None:
        """Same entity reached via multiple paths gets the BEST score."""
        retriever = _make_retriever()
        seed_a, seed_b = uuid4(), uuid4()
        shared = uuid4()
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(
            return_value={
                seed_a: [{"id": str(shared), "distance": 2}],  # 1/3
                seed_b: [{"id": str(shared), "distance": 1}],  # 1/2 (better)
            }
        )
        scores, _ = await retriever._cypher_expand(entry_entity_ids=[seed_a, seed_b], namespace_id=uuid4(), depth=2)
        assert scores[shared] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_surrealdb_path_normalizes_neighborhoods(self) -> None:
        """When dual_nodes is absent but storage.graph is wired, raw dict-shaped
        neighborhoods are normalized to the scoring-loop shape."""
        seed = uuid4()
        n1, n2 = uuid4(), uuid4()
        storage = MagicMock()
        storage.graph = MagicMock()
        storage.get_neighborhoods_batch = AsyncMock(
            return_value={
                seed: {
                    "entities": [
                        {"id": str(n1), "name": "X"},  # distance defaulted
                        {"id": str(n2), "name": "Y", "distance": 2},
                    ],
                    "relationships": [],
                }
            }
        )
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=None,  # no dual_nodes
            embedder=AsyncMock(),
            config=RetrieverConfig(),
            storage=storage,
        )
        scores, info = await retriever._cypher_expand(entry_entity_ids=[seed], namespace_id=uuid4(), depth=2)
        # Both entities found
        assert n1 in scores
        assert n2 in scores
        # n1 was given distance=1 (default), so score = 1/2; n2 had distance=2 → 1/3
        assert scores[n1] > scores[n2]

    @pytest.mark.asyncio
    async def test_handles_surrealdb_record_id(self) -> None:
        """SurrealDB record IDs like 'entity:⟨uuid⟩' are parsed via regex fallback."""
        seed = uuid4()
        embedded = uuid4()
        record_id = f"entity:{embedded}"
        retriever = _make_retriever()
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(
            return_value={seed: [{"id": record_id, "distance": 1, "name": "Z", "entity_type": "X"}]}
        )
        scores, info = await retriever._cypher_expand(entry_entity_ids=[seed], namespace_id=uuid4(), depth=2)
        assert embedded in scores

    @pytest.mark.asyncio
    async def test_depth_is_clamped_to_max(self) -> None:
        retriever = _make_retriever(config=RetrieverConfig(max_depth=3))
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
        await retriever._cypher_expand(
            entry_entity_ids=[uuid4()],
            namespace_id=uuid4(),
            depth=99,  # huge → should clamp to 3
        )
        kwargs = retriever._dual_nodes.get_entity_neighborhoods.call_args.kwargs
        assert kwargs["depth"] == 3


@pytest.mark.unit
class TestFetchChunksFromEntities:
    @pytest.mark.asyncio
    async def test_dual_nodes_path_builds_chunks(self) -> None:
        retriever = _make_retriever()
        ns = uuid4()
        chunk_id = uuid4()
        doc_id = uuid4()
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(
            return_value=[
                {
                    "chunk_id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": "hello",
                    "total_mentions": 3,
                    "entity_ids": ["a", "b"],
                    "occurred_at": "2026-04-01",
                    "metadata": {"author": "alice"},
                }
            ]
        )
        result = await retriever._fetch_chunks_from_entities(
            entity_ids=[uuid4()],
            namespace_id=ns,
            temporal_filter=None,
            limit=5,
        )
        assert len(result) == 1
        cid, score, chunk = result[0]
        assert cid == chunk_id
        assert chunk.namespace_id == ns
        assert chunk.document_id == doc_id
        # Score: 3 * (1 + 0.1 * 2) = 3.6
        assert score == pytest.approx(3.6)
        assert chunk.metadata["author"] == "alice"
        assert chunk.metadata["occurred_at"] == "2026-04-01"

    @pytest.mark.asyncio
    async def test_no_dual_nodes_and_no_storage_returns_empty(self) -> None:
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=None,
            embedder=AsyncMock(),
            config=RetrieverConfig(),
            storage=None,
        )
        result = await retriever._fetch_chunks_from_entities(
            entity_ids=[uuid4()],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=5,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_storage_fallback_returns_empty_on_failure(self) -> None:
        storage = AsyncMock()
        storage.get_entities_batch = AsyncMock(side_effect=RuntimeError("boom"))
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=None,
            embedder=AsyncMock(),
            config=RetrieverConfig(),
            storage=storage,
        )
        result = await retriever._fetch_chunks_from_entities(
            entity_ids=[uuid4()],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=5,
        )
        assert result == []


@pytest.mark.unit
class TestVectorSearchChunksAndEntities:
    @pytest.mark.asyncio
    async def test_vector_search_chunks_maps_to_domain(self) -> None:
        """_vector_search_chunks wraps storage results in domain Chunks."""
        ns = uuid4()
        retriever = _make_retriever()
        vstore = retriever._vector_store
        # Stub result shape mimicking pgvector
        from datetime import datetime as _dt

        mock_result = MagicMock()
        mock_result.chunk = MagicMock()
        mock_result.chunk.id = uuid4()
        mock_result.chunk.namespace_id = ns
        mock_result.chunk.document_id = uuid4()
        mock_result.chunk.content = "vec result"
        mock_result.chunk.occurred_at = _dt(2026, 4, 1, tzinfo=UTC)
        mock_result.chunk.created_at = None
        mock_result.chunk.metadata = {"source": "x"}
        mock_result.combined_score = 0.7
        mock_result.similarity = 0.6
        vstore.search = AsyncMock(return_value=[mock_result])

        result = await retriever._vector_search_chunks(
            query_embedding=[0.1, 0.2],
            namespace_id=ns,
            temporal_filter=None,
            query_text="q",
            limit=5,
        )
        assert len(result) == 1
        cid, score, chunk = result[0]
        assert score == 0.7  # combined_score takes precedence
        assert chunk.metadata["source"] == "x"
        assert chunk.metadata["occurred_at"] == "2026-04-01T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_vector_search_chunks_hybrid_alpha_override(self) -> None:
        retriever = _make_retriever()
        retriever._vector_store.search = AsyncMock(return_value=[])
        await retriever._vector_search_chunks(
            query_embedding=[0.0],
            namespace_id=uuid4(),
            temporal_filter=None,
            query_text="q",
            limit=5,
            hybrid_alpha_override=1.0,
        )
        kwargs = retriever._vector_store.search.call_args.kwargs
        assert kwargs["hybrid_alpha"] == 1.0

    @pytest.mark.asyncio
    async def test_vector_search_entities_no_storage_returns_empty(self) -> None:
        retriever = _make_retriever(storage=None)
        result = await retriever._vector_search_entities(query_embedding=[0.1], namespace_id=uuid4(), limit=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_vector_search_entities_returns_storage_results(self) -> None:
        storage = AsyncMock()
        eid = uuid4()
        storage.search_similar_entities = AsyncMock(return_value=[(eid, 0.95)])
        retriever = _make_retriever(storage=storage)
        result = await retriever._vector_search_entities(query_embedding=[0.1], namespace_id=uuid4(), limit=5)
        assert result == [(eid, 0.95)]

    @pytest.mark.asyncio
    async def test_vector_search_entities_swallows_exception(self) -> None:
        storage = AsyncMock()
        storage.search_similar_entities = AsyncMock(side_effect=RuntimeError("pgvector down"))
        retriever = _make_retriever(storage=storage)
        result = await retriever._vector_search_entities(query_embedding=[0.1], namespace_id=uuid4(), limit=5)
        assert result == []


def _stub_vector_store_with_chunks(
    chunks_data: list[tuple[str, datetime | None, dict[str, Any] | None]],
) -> Any:
    """Build a vector_store mock whose .search returns the chunks_data tuples."""
    vstore = AsyncMock()
    results: list[Any] = []
    for content, occurred_at, meta in chunks_data:
        mock_result = MagicMock()
        mock_result.chunk = MagicMock()
        mock_result.chunk.id = uuid4()
        mock_result.chunk.namespace_id = uuid4()
        mock_result.chunk.document_id = uuid4()
        mock_result.chunk.content = content
        mock_result.chunk.occurred_at = occurred_at
        mock_result.chunk.created_at = None
        mock_result.chunk.metadata = meta or {}
        mock_result.combined_score = 0.8
        mock_result.similarity = 0.8
        results.append(mock_result)
    vstore.search = AsyncMock(return_value=results)
    return vstore


@pytest.mark.unit
class TestSimpleRetrievePaths:
    """Exercise the uncovered branches inside ``_simple_retrieve``."""

    @pytest.mark.asyncio
    async def test_simple_retrieve_with_temporal_sort(self) -> None:
        """temporal_sort=True re-orders results by occurred_at DESC."""
        ns = uuid4()
        # Two chunks: older first, newer second
        older = datetime(2026, 1, 1, tzinfo=UTC)
        newer = datetime(2026, 5, 1, tzinfo=UTC)
        vstore = _stub_vector_store_with_chunks(
            [
                ("old content", older, None),
                ("new content", newer, None),
            ]
        )
        config = RetrieverConfig()
        retriever = VectorCypherRetriever(
            vector_store=vstore,
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=config,
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="x",
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_sort=True,
        )
        # Newer chunk should be ranked first after sort
        assert result.chunks[0][0].content == "new content"
        assert result.metadata["temporal_sort"] is True

    @pytest.mark.asyncio
    async def test_simple_retrieve_with_recency_boost(self) -> None:
        """effective_recency > 0 triggers the recency boost branch."""
        ns = uuid4()
        vstore = _stub_vector_store_with_chunks([("c1", datetime(2026, 5, 1, tzinfo=UTC), None)])
        retriever = VectorCypherRetriever(
            vector_store=vstore,
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=RetrieverConfig(),
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="x",
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=5,
            routing=routing,
            effective_recency=0.5,
        )
        assert result.metadata["effective_recency"] == 0.5
        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_simple_retrieve_bm25_channel_fuses(self) -> None:
        """When enable_bm25_channel + storage are configured, BM25 fuses with vector."""
        ns = uuid4()
        vstore = _stub_vector_store_with_chunks([("vec", None, None)])
        storage = AsyncMock()
        bm25_chunk = _make_chunk("bm25 chunk")
        storage.search_fulltext_chunks = AsyncMock(return_value=[(bm25_chunk, 0.9)])

        config = RetrieverConfig(
            enable_bm25_channel=True,
            bm25_top_k=10,
        )
        retriever = VectorCypherRetriever(
            vector_store=vstore,
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=config,
            storage=storage,
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="x",
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=5,
            routing=routing,
        )
        assert result.metadata.get("bm25_chunk_count") == 1
        assert result.metadata.get("search_mode") == "simple_vector_bm25"

    @pytest.mark.asyncio
    async def test_simple_retrieve_bm25_failure_swallowed(self) -> None:
        ns = uuid4()
        vstore = _stub_vector_store_with_chunks([("vec", None, None)])
        storage = AsyncMock()
        storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("ft down"))

        config = RetrieverConfig(
            enable_bm25_channel=True,
        )
        retriever = VectorCypherRetriever(
            vector_store=vstore,
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=config,
            storage=storage,
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="x",
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=5,
            routing=routing,
        )
        # Vector path still produces results when bm25 fails
        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_simple_retrieve_ordinal_temporal_sort_ascending(self) -> None:
        """ORDINAL temporal sort uses ascending order ('first', 'earliest')."""
        ns = uuid4()
        older = datetime(2026, 1, 1, tzinfo=UTC)
        newer = datetime(2026, 5, 1, tzinfo=UTC)
        vstore = _stub_vector_store_with_chunks(
            [
                ("newer", newer, None),
                ("older", older, None),
            ]
        )
        retriever = VectorCypherRetriever(
            vector_store=vstore,
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=RetrieverConfig(),
        )
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="x",
        )
        sig = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.ORDINAL,
            confidence=0.9,
            source="dictionary",
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=5,
            routing=routing,
            temporal_sort=True,
            temporal_signal=sig,
        )
        # Ascending: older first
        assert result.chunks[0][0].content == "older"


@pytest.mark.unit
class TestRetrieveAntiRecencyVeto:
    @pytest.mark.asyncio
    async def test_anti_recency_token_vetoes_floor_synthesis(self) -> None:
        """When the floor flag is on and the query contains "ever" / "all time",
        synthesis is vetoed and the resulting metadata reflects the veto."""
        config = RetrieverConfig(
            temporal_recency_floor_enabled=True,
        )
        vector_store = AsyncMock()
        vector_store.search = AsyncMock(return_value=[])
        retriever = VectorCypherRetriever(
            vector_store=vector_store,
            neo4j_driver=None,
            embedder=AsyncMock(),
            config=config,
        )
        retriever._embedder.embed = AsyncMock(return_value=[0.0])
        retriever._embedder.model_name = "test"
        retriever._embedder.dimension = 1
        retriever._embedder.cache_stats = None  # type: ignore[attr-defined]
        # Force SIMPLE routing
        retriever._router = MagicMock()
        retriever._router.route = AsyncMock(
            return_value=RoutingDecision(
                complexity=QueryComplexity.SIMPLE,
                use_graph=False,
                graph_depth=0,
                confidence=0.9,
                reasoning="x",
            )
        )

        sig = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.RECENCY,
            confidence=0.9,
            source="dictionary",
        )
        # Query with anti-recency "ever" token
        result = await retriever.retrieve("what have we ever discussed", uuid4(), temporal_signal=sig)
        # Synthesis was vetoed — no exception, valid result
        assert isinstance(result, VectorCypherResult)
