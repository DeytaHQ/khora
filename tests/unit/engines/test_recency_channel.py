"""Unit tests for the parallel recency channel (Issue #567 A3).

Devil's-Advocate demand #3: a chunk with cosine similarity below the
relevance floor must NOT enter the merged pool, even if it's
today-dated. This test pins the gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.skeleton.backends import TemporalVectorStore
from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever
from khora.filter import FilterNode, RecallFilter, parse_to_ast


def _make_chunk(content: str, occurred_at: datetime | None, embedding: list[float]) -> Chunk:
    """Build a Chunk shaped like what pgvector.search_recent_chunks returns."""
    chunk = MagicMock(spec=Chunk)
    chunk.id = uuid4()
    chunk.namespace_id = uuid4()
    chunk.document_id = uuid4()
    chunk.content = content
    chunk.embedding = embedding
    chunk.occurred_at = occurred_at
    chunk.created_at = occurred_at
    chunk.metadata = None
    return chunk


@pytest.fixture
def retriever_with_mocked_store():
    """A VectorCypherRetriever stub wired with a mock vector_store.

    We bypass the real engine plumbing — only ``_recency_channel_chunks``
    is under test, and it depends solely on ``self._config`` and
    ``self._vector_store.search_recent_chunks``.
    """
    cfg = RetrieverConfig(
        temporal_recency_channel_enabled=True,
        temporal_query_relevance_floor=0.30,
        temporal_recency_channel_limit=10,
    )
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = cfg
    retriever._vector_store = MagicMock()
    return retriever


@pytest.mark.asyncio
async def test_below_floor_chunk_excluded(retriever_with_mocked_store) -> None:
    """A today-dated chunk with cosine=0.20 must NOT enter the pool when
    the floor is 0.30 (Devil's-Advocate demand #3, exact scenario)."""
    retriever = retriever_with_mocked_store

    # Two chunks: one above floor (0.5), one below (0.2). Both today-dated.
    query_emb = [1.0, 0.0, 0.0]
    above_floor_chunk = _make_chunk("recent and relevant", datetime.now(UTC), [0.5, 0.866, 0.0])
    below_floor_chunk = _make_chunk("recent and irrelevant", datetime.now(UTC), [0.2, 0.0, 0.979])

    retriever._vector_store.search_recent_chunks = AsyncMock(
        return_value=[(above_floor_chunk, None), (below_floor_chunk, None)]
    )

    result = await retriever._recency_channel_chunks(
        query_embedding=query_emb,
        namespace_id=uuid4(),
        temporal_filter=None,
    )

    # Only the above-floor chunk survives.
    assert len(result) == 1
    assert result[0][0] == above_floor_chunk.id


@pytest.mark.asyncio
async def test_empty_store_returns_empty(retriever_with_mocked_store) -> None:
    """No candidates from SQL → empty result, no crash."""
    retriever = retriever_with_mocked_store
    retriever._vector_store.search_recent_chunks = AsyncMock(return_value=[])

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
    )
    assert result == []


@pytest.mark.asyncio
async def test_sql_failure_returns_empty_does_not_raise(retriever_with_mocked_store) -> None:
    """If search_recent_chunks throws (missing index, etc.), the channel
    must degrade silently — the caller's retrieve() must not fail."""
    retriever = retriever_with_mocked_store
    retriever._vector_store.search_recent_chunks = AsyncMock(side_effect=RuntimeError("boom"))

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
    )
    assert result == []


@pytest.mark.asyncio
async def test_no_search_recent_chunks_method_returns_empty() -> None:
    """If the vector store doesn't implement search_recent_chunks (e.g.
    SurrealDB path), the channel returns [] without raising."""
    cfg = RetrieverConfig(
        temporal_recency_channel_enabled=True,
        temporal_query_relevance_floor=0.30,
    )
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = cfg

    # A vector_store stub WITHOUT search_recent_chunks attribute.
    # Use a real object (not MagicMock — which auto-creates attrs).
    class _NoRecency:
        pass

    retriever._vector_store = _NoRecency()

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
    )
    assert result == []


@pytest.mark.asyncio
async def test_chunks_without_embeddings_skipped(retriever_with_mocked_store) -> None:
    """A chunk that came back from SQL with embedding=None can't be
    cosine-filtered, so the gate must skip it rather than treating it as
    above-floor."""
    retriever = retriever_with_mocked_store
    no_emb = _make_chunk("today no embedding", datetime.now(UTC), [])
    no_emb.embedding = None

    retriever._vector_store.search_recent_chunks = AsyncMock(return_value=[(no_emb, None)])

    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
    )
    assert result == []


class _ProtocolDefaultStore(TemporalVectorStore):
    """A concrete TemporalVectorStore that relies on the Protocol's DEFAULT
    ``search_recent_chunks`` (which returns ``[]``).

    Unlike SurrealDB or pgvector this store does NOT override the recency
    method — it inherits the no-op default. The only abstract members it
    implements are the bare minimum to instantiate; none are exercised by
    the recency channel under test, so they raise if mis-called.
    """

    async def connect(self) -> None:  # pragma: no cover - not exercised
        raise NotImplementedError

    async def disconnect(self) -> None:  # pragma: no cover - not exercised
        raise NotImplementedError

    async def create_chunk(self, chunk: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def create_chunks_batch(self, chunks: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:  # pragma: no cover
        raise NotImplementedError

    async def search(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def health_check(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


def _filter_ast() -> FilterNode:
    """A real recall-filter AST so the plan-recording branch is reachable."""
    return parse_to_ast(RecallFilter.model_validate({"metadata.tier": "gold"}))


@pytest.mark.asyncio
async def test_protocol_default_store_returns_empty_and_records_no_plan() -> None:
    """A real ``TemporalVectorStore`` that inherits the Protocol DEFAULT
    ``search_recent_chunks`` (returns ``[]``) contributes nothing: the
    channel returns ``[]`` and records NO ChannelPlan, even when a caller
    filter is present (the early-return on empty SQL rows precedes the
    plan-recording site). A channel that never produced post-filtered
    chunks must never be credited with a disposition in the report."""
    cfg = RetrieverConfig(
        temporal_recency_channel_enabled=True,
        temporal_query_relevance_floor=0.30,
    )
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = cfg
    retriever._vector_store = _ProtocolDefaultStore()

    filter_channel_plans: dict[str, Any] = {}
    result = await retriever._recency_channel_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        namespace_id=uuid4(),
        temporal_filter=None,
        filter_ast=_filter_ast(),
        filter_channel_plans=filter_channel_plans,
    )

    assert result == []
    # The recency channel honestly never appears in the report.
    assert "recency" not in filter_channel_plans
    assert filter_channel_plans == {}
