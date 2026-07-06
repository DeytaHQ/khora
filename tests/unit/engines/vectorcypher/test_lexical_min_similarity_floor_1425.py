"""Regression tests for #1425: ``min_similarity`` must bound the lexical channel.

The engine-level residual of #1404: that fix floored the keyword backfill and
hybrid fusion inside the temporal stores, but the VectorCypher retriever's OWN
lexical channel (``_lexical_search_chunks`` -> BM25 / keyword-PPR) ran and
returned its results without ever consulting the floor:

1. **Simple-path fusion** (``_simple_retrieve``, HYBRID/ALL with the channel
   enabled) - the fused set was the union of vector and lexical ids, so
   lexical-only chunks entered the output unfloored.
2. **mode=KEYWORD** (``_simple_retrieve``) - BM25 was the sole chunk source
   and the floor was never applied at all.
3. **Main-path fusion** (``_vectorcypher_retrieve`` -> ``_fuse_results``) -
   same union hole as (1), with the graph channel alongside.

The fix mirrors #1404's semantics: with an explicit floor (resolved > 0),
lexical-only chunks are excluded from fusion output while lexical rank
evidence still boosts floor-passing chunks; KEYWORD mode gates its hits
against a floored pure-vector search; ``0.0`` keeps today's behaviour exactly.
The keyword_ppr lexical-channel variant (#1391) fills the same fusion slot,
so it is covered by the same gates.

These tests drive ``_simple_retrieve`` / ``_fuse_results`` with the storage
seams mocked - no DB, no LLM (mirrors
``tests/unit/storage/test_temporal_min_similarity_floor.py``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.query import SearchMode
from khora.storage.temporal import TemporalChunk, TemporalSearchResult

pytestmark = pytest.mark.unit

_NS = uuid4()


def _chunk(name: str) -> Chunk:
    return Chunk(id=uuid4(), namespace_id=_NS, document_id=uuid4(), content=name)


def _vector_result(chunk: Chunk, similarity: float) -> TemporalSearchResult:
    """A floor-passing vector-channel row for the given chunk id."""
    tc = TemporalChunk(
        id=chunk.id,
        namespace_id=_NS,
        document_id=chunk.document_id,
        content=chunk.content,
    )
    return TemporalSearchResult(chunk=tc, similarity=similarity, combined_score=similarity)


def _make_retriever(
    *,
    vector: list[TemporalSearchResult],
    bm25: list[tuple[Chunk, float]],
    config: RetrieverConfig | None = None,
) -> VectorCypherRetriever:
    """Graph-less retriever whose vector + BM25 seams return canned rows."""
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=vector)
    # _bm25_search_chunks prefers the temporal store's search_fulltext.
    vector_store.search_fulltext = AsyncMock(return_value=bm25)

    # Truthy storage so the lexical channel launches; #857 entity projection
    # is exercised on every recall, keep it empty.
    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])

    return VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=config or RetrieverConfig(enable_reranking=False, enable_bm25_channel=True),
        storage=storage,
    )


def _routing() -> RoutingDecision:
    return RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=0.5,
        reasoning="",
    )


async def _simple(retriever: VectorCypherRetriever, *, min_similarity: float, mode: SearchMode):
    return await retriever._simple_retrieve(
        query="optical networks",
        query_embedding=[0.1] * 4,
        namespace_id=_NS,
        temporal_filter=None,
        limit=5,
        routing=_routing(),
        min_similarity=min_similarity,
        mode=mode,
    )


def _chunk_ids(result) -> set[UUID]:
    return {c.id for c, _score in result.chunks}


# ---------------------------------------------------------------------------
# Simple path, HYBRID fusion (enable_bm25_channel=True)
# ---------------------------------------------------------------------------


async def test_floor_excludes_bm25_only_from_simple_fusion() -> None:
    """Hybrid fusion with an explicit floor drops BM25-only chunks.

    The chunk that passed the vector floor stays (it also appears in the BM25
    list, so its rank evidence is merged); the BM25-only chunks are excluded.
    """
    passed = _chunk("passed-floor")
    bm25_only = [_chunk(f"kw{i}") for i in range(3)]

    retriever = _make_retriever(
        vector=[_vector_result(passed, 0.8)],
        bm25=[(passed, 2.0), *((c, 1.0) for c in bm25_only)],
    )

    result = await _simple(retriever, min_similarity=0.5, mode=SearchMode.HYBRID)

    assert _chunk_ids(result) == {passed.id}


async def test_floor_with_empty_vector_channel_returns_no_chunks() -> None:
    """Floor above every chunk's cosine => zero chunks, even with BM25 hits.

    This is the issue's repro shape: ``min_similarity=0.99`` on a corpus whose
    ceiling is 0.11 previously returned a full limit of BM25 chunks.
    """
    retriever = _make_retriever(
        vector=[],
        bm25=[(_chunk(f"kw{i}"), 1.0) for i in range(3)],
    )

    result = await _simple(retriever, min_similarity=0.99, mode=SearchMode.HYBRID)

    assert result.chunks == []


async def test_no_floor_keeps_union_in_simple_fusion() -> None:
    """Default ``min_similarity=0.0`` keeps the historical union semantics."""
    passed = _chunk("vec")
    bm25_only = [_chunk(f"kw{i}") for i in range(3)]

    retriever = _make_retriever(
        vector=[_vector_result(passed, 0.8)],
        bm25=[(c, 1.0) for c in bm25_only],
    )

    result = await _simple(retriever, min_similarity=0.0, mode=SearchMode.HYBRID)

    assert _chunk_ids(result) == {passed.id, *(c.id for c in bm25_only)}


# ---------------------------------------------------------------------------
# Simple path, mode=KEYWORD
# ---------------------------------------------------------------------------


async def test_keyword_mode_floor_gates_on_vector_floor() -> None:
    """KEYWORD with a floor keeps only BM25 hits that pass a floored vector gate."""
    passing = _chunk("passing")
    failing = _chunk("failing")

    retriever = _make_retriever(
        # The gate search returns the floor-passing id set.
        vector=[_vector_result(passing, 0.8)],
        bm25=[(failing, 2.0), (passing, 1.5)],
    )

    result = await _simple(retriever, min_similarity=0.5, mode=SearchMode.KEYWORD)

    assert _chunk_ids(result) == {passing.id}
    # The gate is the ONLY vector search KEYWORD mode runs, and it carries the floor.
    retriever._vector_store.search.assert_awaited_once()
    gate_kwargs = retriever._vector_store.search.await_args.kwargs
    assert gate_kwargs["min_similarity"] == 0.5
    assert gate_kwargs["hybrid_alpha"] is None


async def test_keyword_mode_floor_above_ceiling_returns_no_chunks() -> None:
    """KEYWORD with a floor no chunk passes => zero chunks (was: full limit)."""
    retriever = _make_retriever(
        vector=[],  # nothing passes the floored gate
        bm25=[(_chunk(f"kw{i}"), 1.0) for i in range(3)],
    )

    result = await _simple(retriever, min_similarity=0.99, mode=SearchMode.KEYWORD)

    assert result.chunks == []


async def test_keyword_mode_no_floor_stays_pure_bm25() -> None:
    """Without a floor, KEYWORD mode is unchanged: pure BM25, no vector search."""
    hits = [_chunk(f"kw{i}") for i in range(3)]
    retriever = _make_retriever(
        vector=[],
        bm25=[(c, float(3 - i)) for i, c in enumerate(hits)],
    )

    result = await _simple(retriever, min_similarity=0.0, mode=SearchMode.KEYWORD)

    assert [c.id for c, _ in result.chunks] == [c.id for c in hits]
    retriever._vector_store.search.assert_not_awaited()


# ---------------------------------------------------------------------------
# keyword_ppr lexical-channel variant (#1391) - same fusion slot, same hole
# ---------------------------------------------------------------------------


async def test_floor_excludes_keyword_ppr_only_from_simple_fusion() -> None:
    """The keyword_ppr channel fills the BM25 fusion slot; the floor applies."""
    passed = _chunk("passed-floor")
    ppr_only = [_chunk(f"ppr{i}") for i in range(3)]

    retriever = _make_retriever(
        vector=[_vector_result(passed, 0.8)],
        bm25=[],  # not used - the keyword_ppr branch is dispatched instead
        config=RetrieverConfig(enable_reranking=False, lexical_channel="keyword_ppr"),
    )
    retriever._keyword_ppr_search_chunks = AsyncMock(
        return_value=[(passed.id, 2.0, passed), *((c.id, 1.0, c) for c in ppr_only)]
    )

    result = await _simple(retriever, min_similarity=0.5, mode=SearchMode.HYBRID)

    assert _chunk_ids(result) == {passed.id}
    retriever._keyword_ppr_search_chunks.assert_awaited_once()


# ---------------------------------------------------------------------------
# Main graph path fusion (_fuse_results, threaded from _vectorcypher_retrieve)
# ---------------------------------------------------------------------------


def _triples(chunks: list[Chunk], *, start_score: float = 0.9) -> list[tuple[UUID, float, Chunk]]:
    return [(c.id, start_score - i * 0.1, c) for i, c in enumerate(chunks)]


async def test_fuse_results_exclude_bm25_only_drops_lexical_only_chunks() -> None:
    """``exclude_bm25_only=True`` restricts the fused set to vector/graph chunks.

    BM25 rank evidence still boosts the floor-passing chunk: its fused score
    is strictly higher than in an identical fusion where BM25 did not rank it.
    """
    vec_a, vec_b = _chunk("vec-a"), _chunk("vec-b")
    graph_g = _chunk("graph-g")
    bm25_only = [_chunk(f"kw{i}") for i in range(2)]

    retriever = _make_retriever(vector=[], bm25=[])

    fused = retriever._fuse_results(
        vector_chunks=_triples([vec_a, vec_b]),
        graph_chunks=_triples([graph_g]),
        bm25_chunks=[(vec_b.id, 3.0, vec_b), *((c.id, 1.0, c) for c in bm25_only)],
        exclude_bm25_only=True,
    )

    assert {r.item_id for r in fused} == {vec_a.id, vec_b.id, graph_g.id}

    # Same fusion without BM25 evidence for vec_b: its score must be lower.
    fused_without_b = retriever._fuse_results(
        vector_chunks=_triples([vec_a, vec_b]),
        graph_chunks=_triples([graph_g]),
        bm25_chunks=[(c.id, 1.0, c) for c in bm25_only],
        exclude_bm25_only=True,
    )
    score_with = next(r.rrf_score for r in fused if r.item_id == vec_b.id)
    score_without = next(r.rrf_score for r in fused_without_b if r.item_id == vec_b.id)
    assert score_with > score_without, "BM25 evidence no longer boosts floor-passing chunks"


async def test_fuse_results_default_keeps_union() -> None:
    """Without the flag (no explicit floor), fusion keeps the historical union."""
    vec_a = _chunk("vec-a")
    graph_g = _chunk("graph-g")
    bm25_only = [_chunk(f"kw{i}") for i in range(2)]

    retriever = _make_retriever(vector=[], bm25=[])

    fused = retriever._fuse_results(
        vector_chunks=_triples([vec_a]),
        graph_chunks=_triples([graph_g]),
        bm25_chunks=[(c.id, 1.0, c) for c in bm25_only],
    )

    assert {r.item_id for r in fused} == {vec_a.id, graph_g.id, *(c.id for c in bm25_only)}


async def test_main_path_threads_floor_into_fusion() -> None:
    """``_vectorcypher_retrieve`` passes ``exclude_bm25_only`` iff a floor is set.

    Driven through ``retrieve()`` on the MODERATE (graph) path with the graph
    helpers stubbed, mirroring ``test_vectorcypher_filter_pushdown``'s harness:
    with a floor the BM25-only chunk is excluded from the result; without one
    it survives (union).
    """
    vec_chunk = _chunk("vector-hit")
    bm25_only = _chunk("bm25-only")

    for floor, expect_bm25_only in ((0.5, False), (0.0, True)):
        retriever = _make_retriever(
            vector=[_vector_result(vec_chunk, 0.8)],
            bm25=[(vec_chunk, 2.0), (bm25_only, 1.0)],
            config=RetrieverConfig(
                enable_reranking=False,
                enable_bm25_channel=True,
                enable_session_aware_search=False,
            ),
        )
        retriever._embedder.embed = AsyncMock(return_value=[0.1] * 4)
        # Entry entity so the graph path does not fall back to _simple_retrieve.
        retriever._storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
        retriever._storage.get_entities_batch = AsyncMock(return_value={})
        retriever._router = MagicMock()
        retriever._router.route = AsyncMock(
            return_value=RoutingDecision(
                complexity=QueryComplexity.MODERATE,
                use_graph=True,
                graph_depth=2,
                confidence=0.8,
                reasoning="moderate",
            )
        )
        retriever._router.compute_adaptive_depth = MagicMock(return_value=2)
        # Stub graph helpers so the cypher-expansion path completes without Neo4j.
        retriever._cypher_expand = AsyncMock(return_value=({}, {}))
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])

        result = await retriever.retrieve(
            "optical networks",
            _NS,
            limit=5,
            min_similarity=floor,
        )

        returned = {c.id for c, _ in result.chunks}
        assert vec_chunk.id in returned, f"floor={floor}: vector chunk must survive"
        assert (bm25_only.id in returned) is expect_bm25_only, (
            f"floor={floor}: BM25-only chunk membership should be {expect_bm25_only}"
        )
