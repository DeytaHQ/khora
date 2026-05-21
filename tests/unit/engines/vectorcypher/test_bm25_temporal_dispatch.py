"""Coverage: VectorCypher's BM25 dispatch path (GitHub issue #813).

Before the fix, ``_bm25_search_chunks`` only called
``StorageCoordinator.search_fulltext_chunks``, which reads the
relational ``chunks`` table. The batch ingest path writes to
``khora_chunks`` (the temporal-store table), so BM25 silently returned
zero rows whenever the engine used the streaming-ingest pipeline.

These tests pin the new behaviour:

1. When the temporal store exposes ``search_fulltext``, it is preferred.
2. When the temporal store has no such method (or returns empty), the
   coordinator's ``search_fulltext_chunks`` is used as fallback.
3. When BOTH return empty, a one-shot WARNING is emitted (not
   per-query — operators need a single, loud signal).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)


def _chunk(content: str = "hello") -> Chunk:
    return Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content=content)


def _make_retriever(
    *,
    vector_store: object | None = None,
    storage: object | None = None,
) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        vector_store=vector_store if vector_store is not None else AsyncMock(),
        neo4j_driver=AsyncMock(),
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_prefers_temporal_store_when_available() -> None:
    ns = uuid4()
    chunk = _chunk("temporal-store-hit")

    vstore = MagicMock()
    vstore.search_fulltext = AsyncMock(return_value=[(chunk, 0.85)])

    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])

    retriever = _make_retriever(vector_store=vstore, storage=storage)

    out = await retriever._bm25_search_chunks("hello", ns, limit=10)

    assert len(out) == 1
    assert out[0][2] is chunk
    assert vstore.search_fulltext.await_count == 1
    storage.search_fulltext_chunks.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_falls_back_to_coordinator_when_temporal_empty() -> None:
    ns = uuid4()
    chunk = _chunk("coordinator-hit")

    vstore = MagicMock()
    vstore.search_fulltext = AsyncMock(return_value=[])

    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[(chunk, 0.5)])

    retriever = _make_retriever(vector_store=vstore, storage=storage)
    out = await retriever._bm25_search_chunks("hello", ns, limit=10)

    assert len(out) == 1
    assert out[0][2] is chunk
    storage.search_fulltext_chunks.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_falls_back_when_temporal_store_lacks_method() -> None:
    """Backends without ``search_fulltext`` (e.g. weaviate today) fall through."""
    ns = uuid4()
    chunk = _chunk("coord-hit")

    # vstore is a plain object with no ``search_fulltext`` attribute.
    class _NoFulltext:
        pass

    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[(chunk, 0.4)])

    retriever = _make_retriever(vector_store=_NoFulltext(), storage=storage)
    out = await retriever._bm25_search_chunks("hello", ns, limit=10)

    assert len(out) == 1
    storage.search_fulltext_chunks.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_warns_once_per_namespace_on_empty() -> None:
    ns = uuid4()

    vstore = MagicMock()
    vstore.search_fulltext = AsyncMock(return_value=[])

    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])

    retriever = _make_retriever(vector_store=vstore, storage=storage)
    # First call adds the namespace to the warned set; subsequent calls
    # MUST NOT re-warn (this would spam logs at one-per-query).
    await retriever._bm25_search_chunks("q1", ns, limit=10)
    await retriever._bm25_search_chunks("q2", ns, limit=10)
    await retriever._bm25_search_chunks("q3", ns, limit=10)
    assert str(ns) in retriever._bm25_empty_warned_ns
    assert len(retriever._bm25_empty_warned_ns) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bm25_handles_exceptions_gracefully() -> None:
    """A backend error must not crash the retriever — return [] and continue."""
    ns = uuid4()

    vstore = MagicMock()
    vstore.search_fulltext = AsyncMock(side_effect=RuntimeError("boom"))

    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])

    retriever = _make_retriever(vector_store=vstore, storage=storage)
    out = await retriever._bm25_search_chunks("q", ns, limit=10)
    assert out == []
