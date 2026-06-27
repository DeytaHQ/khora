"""Unit tests for the keyword_ppr query channel (#1391).

Stub storage returns a known keyword->chunk edge set; assert the channel ranks
the chunk containing the query keyword highest and degrades to [] on empty /
no-overlap inputs.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from khora.extraction.tokenize import tokenize_multilingual
from khora.query.keyword_ppr import keyword_ppr_retrieve_chunks

pytestmark = pytest.mark.unit


class _StubStorage:
    """Storage stub exposing only get_keyword_chunk_edges."""

    def __init__(self, edges: list[tuple[str, UUID, float]]) -> None:
        self._edges = edges
        self.last_limit: int | None = None

    async def get_keyword_chunk_edges(self, namespace_id: UUID, *, limit: int) -> list[tuple[str, UUID, float]]:
        self.last_limit = limit
        return self._edges[:limit]


def _kwargs():
    return {
        "tokenizer": tokenize_multilingual,
        "damping": 0.85,
        "max_iter": 50,
        "tol": 1e-6,
        "limit": 10,
        "max_edges": 50_000,
    }


async def test_ranks_chunk_with_query_keyword_highest() -> None:
    target = uuid4()
    other = uuid4()
    isolated = uuid4()
    edges = [
        ("alpha", target, 2.0),
        ("beta", target, 1.5),
        ("beta", other, 1.5),
        ("gamma", other, 1.0),
        ("delta", isolated, 3.0),
    ]
    storage = _StubStorage(edges)

    results = await keyword_ppr_retrieve_chunks(storage, uuid4(), "alpha topic please", **_kwargs())

    assert results, "channel returned no chunks for a matching keyword"
    assert results[0][0] == target, "chunk containing the query keyword should rank highest"
    # The isolated chunk (no query-keyword overlap, no shared keyword with the
    # seeded chunks) must not outrank the seeded one.
    scores = {cid: score for cid, score in results}
    assert scores[target] >= scores.get(isolated, 0.0)


async def test_empty_when_no_query_keywords() -> None:
    storage = _StubStorage([("alpha", uuid4(), 2.0)])
    # "a an the" are stopwords / sub-3-char -> tokenizer yields nothing.
    results = await keyword_ppr_retrieve_chunks(storage, uuid4(), "a an the", **_kwargs())
    assert results == []


async def test_empty_when_no_edges() -> None:
    storage = _StubStorage([])
    results = await keyword_ppr_retrieve_chunks(storage, uuid4(), "alpha", **_kwargs())
    assert results == []


async def test_empty_when_no_keyword_overlap() -> None:
    edges = [("alpha", uuid4(), 2.0), ("beta", uuid4(), 1.0)]
    storage = _StubStorage(edges)
    # Query keyword "zzzword" is not in the bipartite -> no seed -> [].
    results = await keyword_ppr_retrieve_chunks(storage, uuid4(), "zzzword unrelated", **_kwargs())
    assert results == []


async def test_respects_max_edges_cap() -> None:
    edges = [("alpha", uuid4(), 2.0) for _ in range(100)]
    storage = _StubStorage(edges)
    kwargs = _kwargs()
    kwargs["max_edges"] = 5
    await keyword_ppr_retrieve_chunks(storage, uuid4(), "alpha", **kwargs)
    assert storage.last_limit == 5


async def test_respects_limit() -> None:
    # 4 distinct chunks all sharing the query keyword -> all seeded, but limit=2.
    chunks = [uuid4() for _ in range(4)]
    edges = [("alpha", c, 2.0) for c in chunks]
    storage = _StubStorage(edges)
    kwargs = _kwargs()
    kwargs["limit"] = 2
    results = await keyword_ppr_retrieve_chunks(storage, uuid4(), "alpha", **kwargs)
    assert len(results) == 2
