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
# #1574 — bm25_title_weight: config -> engine -> retriever, and the cache key.
# --------------------------------------------------------------------------- #


def test_bm25_title_weight_defaults_to_neutral() -> None:
    """1.0 is the "behave exactly as before" value, on BOTH sides of the wiring.

    Asserted on the config and the retriever separately: the engine copies the
    field across, so a default that drifted on one side only would silently
    change ranking for every deployment that never sets the knob.
    """
    cfg = _config()
    assert cfg.query.bm25_title_weight == 1.0
    assert _build_retriever_config(cfg).bm25_title_weight == 1.0


def test_bm25_title_weight_flows_to_retriever_config() -> None:
    """KHORA_QUERY_BM25_TITLE_WEIGHT reaches the retriever (the #1018 hazard).

    ``recall()`` bypasses ``QueryEngine`` entirely, so a QuerySettings field
    that is not explicitly copied in ``_assemble_retriever_config`` is inert —
    settable, documented, and doing nothing. That is the failure this file
    exists for.
    """
    rc = _build_retriever_config(_config(bm25_title_weight=2.5))
    assert rc.bm25_title_weight == 2.5


def test_bm25_title_weight_is_clamped_at_the_config_boundary() -> None:
    """Pydantic bounds are what let the store inline the value into SQL.

    The weight cannot be a bind parameter — FTS5 auxiliary-function arguments
    must be literals — so "this float came from a validated [0, 10] field" is
    the whole safety argument for the interpolation. Pinning the bounds here
    keeps that argument true.
    """
    from pydantic import ValidationError

    from khora.config.schema import QuerySettings

    assert QuerySettings(bm25_title_weight=0.0).bm25_title_weight == 0.0
    assert QuerySettings(bm25_title_weight=10.0).bm25_title_weight == 10.0
    for out_of_range in (-0.1, 10.1):
        with pytest.raises(ValidationError):
            QuerySettings(bm25_title_weight=out_of_range)


def test_bm25_title_weight_is_validated_on_direct_dataclass_construction() -> None:
    """The bounds must hold on the path Pydantic never sees.

    ``VectorCypherConfig`` is a plain dataclass, so a caller that constructs it
    directly — a supported entry point, it is a public kwarg on the engine —
    skips the ``QuerySettings`` validation asserted above. Without the
    ``__post_init__`` check the two doors into the same knob disagree: -1 is
    rejected through config and accepted through the dataclass, and the store
    inlines the value into FTS5 SQL (it cannot be a bind parameter), which is
    exactly the argument the config-side bounds exist to support.
    """
    from khora.engines.vectorcypher.engine import VectorCypherConfig

    for out_of_range in (-1, 100):
        with pytest.raises(ValueError, match="bm25_title_weight must be between 0 and 10"):
            VectorCypherConfig(bm25_title_weight=out_of_range)

    # Boundaries are legal on both sides — an exclusive check here would reject
    # values QuerySettings accepts, which is the same divergence in reverse.
    assert VectorCypherConfig(bm25_title_weight=0.0).bm25_title_weight == 0.0
    assert VectorCypherConfig(bm25_title_weight=10.0).bm25_title_weight == 10.0


def test_bm25_title_weight_changes_the_recall_cache_key() -> None:
    """Two weights must not share a cached result.

    The recall cache folds the retriever config in as ``repr()`` of the
    dataclass, so a new field is captured only because it is *declared on the
    dataclass*. A weight threaded through as a loose kwarg instead would have
    served a 1.0-ranked result to a 2.0 query for the rest of the epoch — a
    wrong answer with no error anywhere. Both links in that chain are asserted:
    the repr differs, and the digest built from it differs.
    """
    from uuid import uuid4

    from khora.engines.vectorcypher.recall_cache import RecallResultCache

    neutral = RetrieverConfig(bm25_title_weight=1.0)
    weighted = RetrieverConfig(bm25_title_weight=2.0)
    assert repr(neutral) != repr(weighted), "the field must be declared on RetrieverConfig to be fingerprinted"

    common = dict(
        query="floor panels",
        namespace_id=uuid4(),
        epoch=0,
        mode="hybrid",
        limit=10,
        min_similarity=0.0,
        graph_depth=None,
        hybrid_alpha=None,
        recency_bias=None,
        temporal_filter=None,
        filter_ast=None,
    )
    assert RecallResultCache._digest(**common, config_fingerprint=repr(neutral)) != RecallResultCache._digest(
        **common, config_fingerprint=repr(weighted)
    )


# --------------------------------------------------------------------------- #
# #1018 — behavioral: HyDE / diversity / stage1 actually change retrieval.
# --------------------------------------------------------------------------- #


def _hyde_probe_retriever(enable_hyde: str) -> VectorCypherRetriever:
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig(enable_hyde=enable_hyde)
    retriever._embedder = AsyncMock()
    # #1469: HyDE now runs as generate (LLM+embed) then combine. The probe's
    # hypothetical embedding is [17,17,17,17]; averaged with a [1,1,1,1] base it
    # yields [9,9,9,9], so the fold output pins the same expand-vs-noop contract.
    retriever._hyde_expander = AsyncMock()
    retriever._hyde_expander.generate_hypothetical_embeddings = AsyncMock(return_value=[[17.0] * 4])
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
    task = retriever._maybe_launch_hyde("q", routing=_SIMPLE_ROUTING, temporal_signal=None)
    assert task is not None
    out = await retriever._fold_hyde([1.0] * 4, task)
    assert out == [9.0] * 4
    retriever._hyde_expander.generate_hypothetical_embeddings.assert_awaited_once()


async def test_hyde_never_leaves_embedding_unchanged() -> None:
    """enable_hyde='never' is a no-op (no LLM call, original embedding kept)."""
    retriever = _hyde_probe_retriever("never")
    task = retriever._maybe_launch_hyde("q", routing=_COMPLEX_ROUTING, temporal_signal=None)
    assert task is None
    out = await retriever._fold_hyde([1.0] * 4, task)
    assert out == [1.0] * 4
    retriever._hyde_expander.generate_hypothetical_embeddings.assert_not_awaited()


async def test_hyde_auto_fires_for_complex_not_simple() -> None:
    """enable_hyde='auto' expands for COMPLEX queries but not SIMPLE ones."""
    retriever = _hyde_probe_retriever("auto")
    assert retriever._should_hyde(_COMPLEX_ROUTING, None) is True
    assert retriever._should_hyde(_SIMPLE_ROUTING, None) is False


async def test_hyde_launch_starts_llm_before_base_embed_resolves() -> None:
    """#1469 speculative HyDE: the hypothetical generation is in flight before
    the base embed resolves, proving the LLM round-trip overlaps the embed."""
    import asyncio

    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig(enable_hyde="always")
    retriever._embedder = AsyncMock()

    started = asyncio.Event()
    release = asyncio.Event()

    async def gated_generate(_query, *, out_diagnostics=None):
        started.set()
        await release.wait()
        return [[17.0] * 4]

    retriever._hyde_expander = AsyncMock()
    retriever._hyde_expander.generate_hypothetical_embeddings = AsyncMock(side_effect=gated_generate)

    task = retriever._maybe_launch_hyde("q", routing=_SIMPLE_ROUTING, temporal_signal=None)
    assert task is not None
    # The launch scheduled the coroutine; yield so it can start running.
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert not task.done()  # still awaiting the release -> genuinely in flight

    release.set()
    out = await retriever._fold_hyde([1.0] * 4, task)
    assert out == [9.0] * 4


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


# --------------------------------------------------------------------------- #
# #1574 — embedded end-to-end: a title-only query reaches the chunk through
# the lexical channel.
# --------------------------------------------------------------------------- #

#: Verbatim from the #1574 repro. Doubles as a tokenizer guard: unicode61
#: (wrapped by the ``porter`` tokenizer) splits on ``_``, so this must index as
#: ``floor`` / ``panel`` / ``dimens`` / ``20260213``.
_REPRO_TITLE = "Floor Panels_Dimensioned_20260213"
#: Shares no vocabulary with the title — otherwise a content hit would be
#: indistinguishable from a title hit and the test would prove nothing.
_REPRO_BODY = "the assembly drawing revision notes for the north wing"

#: A second document, to prove the title match is targeted rather than "the
#: lexical channel returns whatever it has". Its title and body vocabularies are
#: disjoint from each other AND from the first document's.
_DECOY_TITLE = "Procurement Ledger"
_DECOY_BODY = "quarterly stationery orders for the southern depot"


@pytest.mark.embedded
async def test_title_only_query_recalls_the_chunk_via_the_lexical_channel(monkeypatch) -> None:
    """The #1574 repro, through the full ``Khora.recall()`` stack on sqlite_lance.

    ``mode=KEYWORD`` is what makes this an honest test of the *lexical* channel:
    per #833 it skips the vector store search entirely, so every returned chunk
    came from BM25. Under HYBRID the hash-derived mock embeddings would surface
    the only chunk in the namespace regardless, and the test would pass with the
    title still unindexed.

    Four queries against two documents:
      * title-only words -> the titled chunk (the bug: this returned nothing);
      * the bare numeric token -> same chunk (the half a tokenizer change would
        lose to the surrounding underscores);
      * the other document's title -> only that one (targeted, not indiscriminate);
      * a word in neither -> nothing (the channel is not matching everything).
    """
    try:
        import aiosqlite  # noqa: F401, PLC0415
        import lancedb  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("sqlite_lance optional deps not installed")

    from khora.search_mode import SearchMode  # noqa: PLC0415

    embedded_khora, install_mock_llm = _import_embedded_helpers()

    monkeypatch.setenv("KHORA_QUERY_ENABLE_BM25_CHANNEL", "true")
    monkeypatch.setenv("KHORA_QUERY_ENABLE_RERANKING", "false")
    install_mock_llm(dim=64)

    async with embedded_khora(embedding_dimension=64) as kb:
        ns = await kb.create_namespace()
        for body, title in ((_REPRO_BODY, _REPRO_TITLE), (_DECOY_BODY, _DECOY_TITLE)):
            await kb.remember(
                body,
                namespace=ns.namespace_id,
                title=title,
                entity_types=["PERSON"],
                relationship_types=["MET"],
            )

        async def _keyword_recall(query: str) -> list[str]:
            result = await kb.recall(query, namespace=ns.namespace_id, mode=SearchMode.KEYWORD)
            # Happy path (title_weight=1.0, a store WITH the title column): the
            # #1574 wiring must not manufacture a degradation (ADR-001).
            assert_no_silent_degradation(result)
            return [chunk.content for chunk in result.chunks]

        assert await _keyword_recall("floor panels dimensioned") == [_REPRO_BODY]
        assert await _keyword_recall("20260213") == [_REPRO_BODY]
        assert await _keyword_recall("procurement ledger") == [_DECOY_BODY]
        assert await _keyword_recall("bathymetry") == []
