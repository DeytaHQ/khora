"""Regression tests for #1404: ``min_similarity`` must bound keyword results.

``recall(min_similarity=X)`` promises only chunks with similarity >= X. The
floor was wired into the vector search by #837 (v0.17.1), but two later steps
in every temporal store re-introduced chunks that never passed it:

1. **Keyword fallback** — when the (floored) vector search returns fewer than
   ``limit`` rows, the store backfilled from BM25 with no floor check.
2. **Hybrid RRF fusion** — the fused id set was the union of vector and BM25
   ids, so BM25-only chunks entered the output unfloored.

The fix: the fallback is skipped when an explicit floor is set, and hybrid
fusion excludes BM25-only chunks when an explicit floor is set (BM25 ranks
still contribute to the scores of vector-passing chunks).

These tests drive ``_search_inner`` on all three affected stores (pgvector,
sqlite_lance, surrealdb) with the vector/BM25 internals stubbed — no DB.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.temporal import TemporalChunk, TemporalSearchResult

_NS = uuid4()
_QUERY_EMBEDDING = [0.0] * 4


def _result(name: str, similarity: float = 0.0, bm25_score: float | None = 1.0) -> TemporalSearchResult:
    return TemporalSearchResult(
        chunk=TemporalChunk(
            id=uuid4(),
            namespace_id=_NS,
            document_id=uuid4(),
            content=name,
        ),
        similarity=similarity,
        bm25_score=bm25_score,
        combined_score=None,
    )


def _make_pgvector(vector: list, bm25: list):
    from khora.storage.temporal.pgvector import PgVectorTemporalStore

    store = PgVectorTemporalStore.__new__(PgVectorTemporalStore)

    @asynccontextmanager
    async def _session():
        yield MagicMock()

    store._get_session = _session
    store._vector_search = AsyncMock(return_value=vector)
    store._bm25_search = AsyncMock(return_value=bm25)
    return store


def _make_sqlite_lance(vector: list, bm25: list):
    from khora.storage.temporal.sqlite_lance import SQLiteLanceTemporalStore

    store = SQLiteLanceTemporalStore.__new__(SQLiteLanceTemporalStore)
    store._ast_post_filter = MagicMock(return_value=None)
    store._vector_search = AsyncMock(return_value=vector)
    store._bm25_search = AsyncMock(return_value=bm25)
    return store


def _make_surrealdb(vector: list, bm25: list):
    from khora.storage.temporal.surrealdb import SurrealDBTemporalStore

    store = SurrealDBTemporalStore.__new__(SurrealDBTemporalStore)
    store._build_filter_clauses = MagicMock(return_value=([], {}))
    store._vector_search = AsyncMock(return_value=vector)
    store._bm25_search = AsyncMock(return_value=bm25)
    return store


_FACTORIES = [
    pytest.param(_make_pgvector, id="pgvector"),
    pytest.param(_make_sqlite_lance, id="sqlite_lance"),
    pytest.param(_make_surrealdb, id="surrealdb"),
]


async def _search(store, *, min_similarity: float, hybrid_alpha: float | None):
    return await store._search_inner(
        _NS,
        _QUERY_EMBEDDING,
        limit=5,
        min_similarity=min_similarity,
        temporal_filter=None,
        hybrid_alpha=hybrid_alpha,
        query_text="optical networks",
    )


@pytest.mark.parametrize("factory", _FACTORIES)
@pytest.mark.asyncio
async def test_floor_skips_keyword_fallback(factory) -> None:
    """Vector search floored to empty + explicit floor => zero results.

    Previously the BM25 fallback backfilled a full ``limit`` of chunks that
    never passed the floor (#1404).
    """
    store = factory(vector=[], bm25=[_result(f"kw{i}") for i in range(3)])

    results = await _search(store, min_similarity=0.99, hybrid_alpha=None)

    assert results == []
    store._bm25_search.assert_not_awaited()


@pytest.mark.parametrize("factory", _FACTORIES)
@pytest.mark.asyncio
async def test_no_floor_keeps_keyword_fallback(factory) -> None:
    """Default ``min_similarity=0.0`` preserves the keyword-fallback recall aid."""
    store = factory(vector=[], bm25=[_result(f"kw{i}") for i in range(3)])

    results = await _search(store, min_similarity=0.0, hybrid_alpha=None)

    assert len(results) == 3
    store._bm25_search.assert_awaited_once()


@pytest.mark.parametrize("factory", _FACTORIES)
@pytest.mark.asyncio
async def test_floor_excludes_bm25_only_from_hybrid_fusion(factory) -> None:
    """Hybrid fusion with an explicit floor drops BM25-only chunks.

    The chunk that passed the vector floor stays and still receives its BM25
    rank contribution; the BM25-only chunks are excluded (#1404).
    """
    passed = _result("passed-floor", similarity=0.8, bm25_score=None)
    bm25_dup = _result("passed-floor", bm25_score=2.0)
    bm25_dup.chunk.id = passed.chunk.id  # same chunk seen by both channels
    bm25_only = [_result(f"kw{i}") for i in range(3)]

    store = factory(vector=[passed], bm25=[bm25_dup, *bm25_only])

    results = await _search(store, min_similarity=0.5, hybrid_alpha=0.5)

    assert [r.chunk.id for r in results] == [passed.chunk.id]
    # BM25 evidence for the surviving chunk is still merged.
    assert results[0].bm25_score == 2.0


@pytest.mark.parametrize("factory", _FACTORIES)
@pytest.mark.asyncio
async def test_no_floor_hybrid_fusion_keeps_union(factory) -> None:
    """Default ``min_similarity=0.0`` keeps the historical union semantics."""
    passed = _result("vec", similarity=0.8, bm25_score=None)
    bm25_only = [_result(f"kw{i}") for i in range(3)]

    store = factory(vector=[passed], bm25=list(bm25_only))

    results = await _search(store, min_similarity=0.0, hybrid_alpha=0.5)

    returned_ids = {r.chunk.id for r in results}
    assert passed.chunk.id in returned_ids
    assert returned_ids == {passed.chunk.id, *(r.chunk.id for r in bm25_only)}
