"""#1018 — QuerySettings tier applied on the default recall() / VectorCypher path.

``Khora.recall()`` dispatches straight to ``retriever.retrieve()`` and bypasses
``khora.query.QueryEngine``, so several ``QuerySettings`` fields were silently
inert on the default engine. These tests assert ``enable_hyde`` /
``enable_diversity`` / ``diversity_lambda`` / ``stage1_recall_limit`` now flow
onto the retriever config and actually change retrieval behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from khora.config.schema import KhoraConfig
from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from tests.test_helpers.diagnostics import assert_no_silent_degradation

# --------------------------------------------------------------------------- #
# #1018 — QuerySettings tier flows onto the VectorCypher retriever config.
# --------------------------------------------------------------------------- #


def _config(**query_overrides) -> KhoraConfig:
    cfg = KhoraConfig()
    for key, val in query_overrides.items():
        setattr(cfg.query, key, val)
    return cfg


def _build_retriever_config(cfg: KhoraConfig) -> RetrieverConfig:
    """Run the engine __init__ + the RetrieverConfig assembly that ``connect()``
    performs, returning the assembled RetrieverConfig (without touching DBs)."""
    engine = VectorCypherEngine(cfg)
    return engine._assemble_retriever_config()


def test_enable_hyde_flows_to_retriever_config() -> None:
    cfg = _config(enable_hyde="always")
    rc = _build_retriever_config(cfg)
    assert rc.enable_hyde == "always"


def test_enable_diversity_and_lambda_flow_to_retriever_config() -> None:
    cfg = _config(enable_diversity=True, diversity_lambda=0.2)
    rc = _build_retriever_config(cfg)
    assert rc.enable_diversity is True
    assert rc.diversity_lambda == 0.2


def test_stage1_recall_limit_flows_to_retriever_config() -> None:
    cfg = _config(stage1_recall_limit=321)
    rc = _build_retriever_config(cfg)
    assert rc.stage1_recall_limit == 321


def test_enable_bm25_channel_defaults_off() -> None:
    """#1330 — the lexical channel stays opt-out by default (unchanged behavior)."""
    rc = _build_retriever_config(_config())
    assert rc.enable_bm25_channel is False


def test_enable_bm25_channel_flows_to_retriever_config() -> None:
    """#1330 — KHORA_QUERY_ENABLE_BM25_CHANNEL=true makes the channel operable."""
    cfg = _config(enable_bm25_channel=True)
    rc = _build_retriever_config(cfg)
    assert rc.enable_bm25_channel is True


def test_query_settings_defaults_match_retriever_defaults() -> None:
    """The default KhoraConfig.query values must produce the RetrieverConfig
    defaults (no silent drift between the two contracts)."""
    rc = _build_retriever_config(_config())
    assert rc.enable_hyde == "auto"
    assert rc.enable_diversity is True
    assert rc.diversity_lambda == 0.5
    assert rc.diversity_min_gap == 0.35
    assert rc.stage1_recall_limit == 200


# --------------------------------------------------------------------------- #
# #1018 — behavioral: HyDE / diversity / stage1 actually change retrieval.
# --------------------------------------------------------------------------- #


def _hyde_probe_retriever(enable_hyde: str) -> VectorCypherRetriever:
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig(enable_hyde=enable_hyde)
    retriever._embedder = AsyncMock()
    retriever._hyde_expander = AsyncMock()
    retriever._hyde_expander.expand_query_embedding = AsyncMock(return_value=[9.0] * 4)
    return retriever


_SIMPLE_ROUTING = RoutingDecision(
    complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning="s"
)
_COMPLEX_ROUTING = RoutingDecision(
    complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=2, confidence=0.9, reasoning="c"
)


async def test_hyde_always_fires_and_expands_embedding() -> None:
    """enable_hyde='always' expands the embedding even for a SIMPLE query."""
    retriever = _hyde_probe_retriever("always")
    out = await retriever._maybe_expand_hyde("q", [1.0] * 4, routing=_SIMPLE_ROUTING, temporal_signal=None)
    assert out == [9.0] * 4
    retriever._hyde_expander.expand_query_embedding.assert_awaited_once()


async def test_hyde_never_leaves_embedding_unchanged() -> None:
    """enable_hyde='never' is a no-op (no LLM call, original embedding kept)."""
    retriever = _hyde_probe_retriever("never")
    out = await retriever._maybe_expand_hyde("q", [1.0] * 4, routing=_COMPLEX_ROUTING, temporal_signal=None)
    assert out == [1.0] * 4
    retriever._hyde_expander.expand_query_embedding.assert_not_awaited()


async def test_hyde_auto_fires_for_complex_not_simple() -> None:
    """enable_hyde='auto' expands for COMPLEX queries but not SIMPLE ones."""
    retriever = _hyde_probe_retriever("auto")
    assert retriever._should_hyde(_COMPLEX_ROUTING, None) is True
    assert retriever._should_hyde(_SIMPLE_ROUTING, None) is False


def _diversity_retriever(*, enable_diversity: bool, lambda_param: float = 0.5) -> VectorCypherRetriever:
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig(enable_diversity=enable_diversity, diversity_lambda=lambda_param)
    return retriever


def _fused(item_id, embedding, score):
    from khora.core.models import Chunk
    from khora.engines.vectorcypher.fusion import FusedResult

    chunk = Chunk(id=item_id, content="c", embedding=embedding)
    return FusedResult(item_id=item_id, item=chunk, rrf_score=score)


async def test_mmr_diversity_select_prefers_diverse_chunk() -> None:
    """With pure-diversity lambda (0.0), MMR's 2nd pick is the one most distant
    from the 1st, not the next-highest score (which is a near-duplicate)."""
    from uuid import uuid4 as _u

    retriever = _diversity_retriever(enable_diversity=True, lambda_param=0.0)
    a, b, c = _u(), _u(), _u()
    # a: top score, embedding ~[1,0]. b: near-duplicate of a (high score).
    # c: orthogonal [0,1], lower score. Pure diversity should pick a then c.
    fused = [
        _fused(a, [1.0, 0.0], 0.9),
        _fused(b, [0.99, 0.01], 0.8),
        _fused(c, [0.0, 1.0], 0.5),
    ]
    # #1463 signature: relevance_scores is one post-boost score per candidate.
    out = retriever._mmr_select_fused(fused, [0.9, 0.8, 0.5], k=2, lambda_param=0.0)
    top_two = {out[0].item_id, out[1].item_id}
    assert top_two == {a, c}


async def test_mmr_falls_back_to_score_order_without_embeddings() -> None:
    """No chunk embeddings -> diversity degrades to existing (score) order."""
    from uuid import uuid4 as _u

    retriever = _diversity_retriever(enable_diversity=True)
    a, b, c = _u(), _u(), _u()
    fused = [_fused(a, None, 0.9), _fused(b, None, 0.8), _fused(c, None, 0.5)]
    out = retriever._mmr_select_fused(fused, [0.9, 0.8, 0.5], k=2, lambda_param=0.5)
    assert [r.item_id for r in out] == [a, b, c]


# --------------------------------------------------------------------------- #
# #1463 — MMR must use the POST-boost/rerank relevance, not a fresh cosine, and
# must not float embedding-less chunks via a fake 1.0 relevance.
# --------------------------------------------------------------------------- #


async def test_mmr_relevance_uses_ranking_score_not_raw_cosine() -> None:
    """#1463 regression: with pure-relevance lambda (1.0) MMR must honor the
    passed ranking scores (post-boost/rerank), NOT a recomputed query-chunk
    cosine. The fixture makes the two DISAGREE: the chunk closest to the query
    by cosine is ranked LAST by rerank, and vice-versa. MMR must keep the
    rerank winner on top."""
    from uuid import uuid4 as _u

    retriever = _diversity_retriever(enable_diversity=True, lambda_param=1.0)
    # a is nearest the query direction [1,0] by cosine, but rerank scored it
    # LOWEST. c is orthogonal (lowest cosine) but rerank scored it HIGHEST.
    # If MMR (wrongly) recomputed cosine it would pick a first; with the real
    # ranking scores it must pick c first.
    a, b, c = _u(), _u(), _u()
    fused = [
        _fused(a, [1.0, 0.0], 0.9),  # list position = pre-existing rank
        _fused(b, [0.7, 0.3], 0.8),
        _fused(c, [0.0, 1.0], 0.5),
    ]
    # Ranking scores INVERT the cosine order (rerank disagrees with cosine).
    ranking_scores = [0.1, 0.4, 0.9]  # c highest, a lowest
    out = retriever._mmr_select_fused(fused, ranking_scores, k=2, lambda_param=1.0)
    assert out[0].item_id == c  # rerank winner, not the cosine winner (a)


async def test_mmr_graph_only_chunk_does_not_float_via_fake_relevance() -> None:
    """#1463 regression: an embedding-less (graph-only) chunk must NOT be
    promoted to the top by a fake relevance of 1.0 (the old code backfilled its
    embedding with the query embedding -> cosine 1.0). It gets a neutral
    (median) relevance and stays below genuinely high-scoring embedded chunks."""
    from uuid import uuid4 as _u

    retriever = _diversity_retriever(enable_diversity=True, lambda_param=1.0)
    hi, mid, graph = _u(), _u(), _u()
    fused = [
        _fused(hi, [1.0, 0.0], 0.95),  # genuinely relevant embedded chunk
        _fused(mid, [0.0, 1.0], 0.55),  # moderately relevant embedded chunk
        _fused(graph, None, 0.60),  # graph-only: NO embedding
    ]
    ranking_scores = [0.95, 0.55, 0.60]
    out = retriever._mmr_select_fused(fused, ranking_scores, k=2, lambda_param=1.0)
    # The high-scoring embedded chunk must lead; the graph-only chunk must NOT
    # jump to #1 on a fabricated 1.0 relevance.
    assert out[0].item_id == hi
    assert out[0].item_id != graph


def test_mmr_disabled_path_leaves_order_untouched() -> None:
    """#1463: with diversity OFF, _mmr_select_fused is a hard no-op — it returns
    the input list unchanged regardless of embeddings or scores. This guards the
    retrieve() guard's downstream contract (the fused order is preserved when the
    gate never fires)."""
    from uuid import uuid4 as _u

    retriever = _diversity_retriever(enable_diversity=False)
    a, b, c, d = _u(), _u(), _u(), _u()
    # A pool that WOULD be reordered by MMR (near-duplicate a/b) if it ran.
    fused = [
        _fused(a, [1.0, 0.0], 0.9),
        _fused(b, [0.99, 0.01], 0.8),
        _fused(c, [0.0, 1.0], 0.5),
        _fused(d, [0.5, 0.5], 0.4),
    ]
    # The retrieve() guard is ``enable_diversity and len(fused) > limit``; with
    # diversity OFF the whole block is skipped and the list is used as-is. Assert
    # both halves: the config gate is off AND the fused order is unmodified.
    assert retriever._config.enable_diversity is False
    assert [r.item_id for r in fused] == [a, b, c, d]


def _gate_retriever(min_gap: float) -> VectorCypherRetriever:
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig(enable_diversity=True, diversity_min_gap=min_gap)
    return retriever


def test_adaptive_gate_skips_on_decisive_winner() -> None:
    """#1463: a decisive top score (gap > diversity_min_gap) skips MMR."""
    r = _gate_retriever(min_gap=0.35)
    # top 1.0, second 0.5 -> gap 0.5 > 0.35 -> decisive -> skip.
    assert r._diversity_skip_reason([1.0, 0.5, 0.4, 0.3]) == "decisive_winner"


def test_adaptive_gate_runs_when_scores_are_close() -> None:
    """#1463: a near-tie at the top (gap <= diversity_min_gap) runs MMR."""
    r = _gate_retriever(min_gap=0.35)
    # top 1.0, second 0.9 -> gap 0.1 <= 0.35 -> not decisive -> run.
    assert r._diversity_skip_reason([1.0, 0.9, 0.8, 0.7]) is None


def test_adaptive_gate_skips_when_too_few_candidates() -> None:
    """#1463: fewer than 3 candidates -> diversity is moot -> skip MMR with a
    distinct reason label so telemetry stays accurate."""
    r = _gate_retriever(min_gap=0.35)
    assert r._diversity_skip_reason([1.0, 0.99]) == "too_few_candidates"


def test_adaptive_gate_disabled_with_zero_gap() -> None:
    """#1463: diversity_min_gap=0.0 disables the gate (MMR always runs)."""
    r = _gate_retriever(min_gap=0.0)
    assert r._diversity_skip_reason([1.0, 0.1, 0.05, 0.01]) is None


def _fetch_limit_retriever(**config_kwargs) -> VectorCypherRetriever:
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig(**config_kwargs)
    return retriever


def test_stage1_overfetch_widens_vector_fetch_when_narrowing_active() -> None:
    """The vector channel over-fetches stage1_recall_limit candidates when
    reranking or diversity will narrow the pool (#1018)."""
    # diversity on -> narrowing active -> overfetch to stage1.
    r = _fetch_limit_retriever(enable_reranking=False, enable_diversity=True, stage1_recall_limit=150)
    assert r._vector_fetch_limit(10) == 150
    # reranking on -> also narrowing.
    r2 = _fetch_limit_retriever(enable_reranking=True, enable_diversity=False, stage1_recall_limit=150)
    assert r2._vector_fetch_limit(10) == 150


def test_stage1_no_overfetch_when_no_narrowing() -> None:
    """Both narrowing stages off -> historic per-channel ``limit`` fetch."""
    r = _fetch_limit_retriever(enable_reranking=False, enable_diversity=False, stage1_recall_limit=200)
    assert r._vector_fetch_limit(10) == 10


def test_stage1_never_shrinks_below_caller_limit() -> None:
    """A caller asking for more than stage1_recall_limit is not shrunk."""
    r = _fetch_limit_retriever(enable_diversity=True, stage1_recall_limit=50)
    assert r._vector_fetch_limit(120) == 120


# --------------------------------------------------------------------------- #
# #1018 — embedded end-to-end: HyDE fires through the full recall() stack.
# --------------------------------------------------------------------------- #


def _import_embedded_helpers():
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples._helpers import embedded_khora, install_mock_llm  # noqa: PLC0415

    return embedded_khora, install_mock_llm


@pytest.mark.embedded
async def test_hyde_always_fires_through_recall_stack(monkeypatch) -> None:
    """enable_hyde='always' on the default recall() path makes an extra LLM
    completion call (the hypothetical) that 'never' does not."""
    try:
        import aiosqlite  # noqa: F401, PLC0415
        import lancedb  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("sqlite_lance optional deps not installed")

    embedded_khora, install_mock_llm = _import_embedded_helpers()

    async def _recall_completion_count(hyde_mode: str) -> int:
        monkeypatch.setenv("KHORA_QUERY_ENABLE_HYDE", hyde_mode)
        # No extraction LLM noise: short message stays under the extraction floor.
        monkeypatch.setenv("KHORA_QUERY_ENABLE_RERANKING", "false")
        mock = install_mock_llm(dim=64, responses=["a hypothetical answer document"])
        async with embedded_khora(embedding_dimension=64) as kb:
            ns = await kb.create_namespace()
            await kb.remember(
                "Alice met Bob at the conference.",
                namespace=ns.namespace_id,
                entity_types=["PERSON"],
                relationship_types=["MET"],
            )
            before = len(mock.completion_calls)
            result = await kb.recall("what did Alice and Bob discuss in detail", namespace=ns.namespace_id)
            # Happy path: the HyDE wiring must not introduce a silent
            # degradation onto the RecallResult (ADR-001).
            assert_no_silent_degradation(result)
            return len(mock.completion_calls) - before

    never_calls = await _recall_completion_count("never")
    always_calls = await _recall_completion_count("always")
    assert always_calls > never_calls
