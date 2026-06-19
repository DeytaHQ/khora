"""Regression coverage for issue #1115.

The temporal sort in ``VectorCypherRetriever._simple_retrieve`` re-orders
chunks by ``occurred_at`` for RECENCY / STATE_QUERY style queries. The sort
key ``_ts`` parsed ``metadata['occurred_at']`` via ``fromisoformat`` with no
tzinfo normalization, fell back to ``created_at`` (tz-aware from the DB), and
finally to naive ``datetime.min``. Sorting a list that mixes naive and aware
datetimes raises ``TypeError: can't compare offset-naive and offset-aware
datetimes`` and crashes the whole recall. The metadata dict is built as
``{'occurred_at': <column iso>, **(chunk.metadata or {})}``, so a
user-supplied naive ISO string overrides the column value - reachable by
default since ``enable_reranking`` defaults to ``False``.

Same tz-boundary class as fb6ede76; ``_calculate_recency_scores`` in the
same file is the reference normalization pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.storage.temporal import TemporalChunk, TemporalSearchResult


def _make_retriever() -> VectorCypherRetriever:
    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        # No Neo4j driver -> simple path is the only path.
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(enable_reranking=False, enable_bm25_channel=False),
        storage=storage,
    )


def _make_search_result(
    content: str,
    *,
    occurred_at: datetime | None = None,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_id: UUID | None = None,
) -> TemporalSearchResult:
    tc = TemporalChunk(
        id=chunk_id or uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        embedding=None,
        occurred_at=occurred_at,
        created_at=created_at,
        metadata=metadata or {},
    )
    return TemporalSearchResult(chunk=tc, similarity=0.9, combined_score=0.9)


def _routing() -> RoutingDecision:
    return RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=0.5,
        reasoning="",
    )


@pytest.mark.unit
class TestTemporalSortTimezoneNormalization1115:
    """#1115: temporal sort must not crash on mixed naive/aware datetimes."""

    @pytest.mark.asyncio
    async def test_mixed_naive_and_aware_timestamps_sort_without_typeerror(self) -> None:
        """Naive metadata occurred_at + aware created_at + datetime.min sentinel.

        Pre-fix this raised ``TypeError: can't compare offset-naive and
        offset-aware datetimes`` mid-recall. Post-fix all keys are normalized
        to UTC and the descending order is (aware created_at 2024-05-01) >
        (naive occurred_at 2024-03-01, treated as UTC) > (sentinel).
        """
        # (a) user-supplied NAIVE ISO string in chunk metadata overrides the
        # occurred_at column value in the metadata dict.
        naive_meta = _make_search_result(
            "naive-meta",
            occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            metadata={"occurred_at": "2024-03-01T12:00:00"},
        )
        # (b) no occurred_at -> falls back to tz-AWARE created_at.
        aware_created = _make_search_result(
            "aware-created",
            created_at=datetime(2024, 5, 1, tzinfo=UTC),
        )
        # (c) neither occurred_at nor created_at -> datetime.min sentinel.
        sentinel = _make_search_result("sentinel")

        retriever = _make_retriever()
        retriever._vector_store.search = AsyncMock(return_value=[naive_meta, aware_created, sentinel])

        result = await retriever._simple_retrieve(
            query="what happened recently?",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=_routing(),
            temporal_sort=True,
        )

        contents = [c.content for c, _score in result.chunks]
        assert contents == ["aware-created", "naive-meta", "sentinel"]

    @pytest.mark.asyncio
    async def test_z_suffix_occurred_at_parses_and_sorts(self) -> None:
        """'Z'-suffixed ISO strings normalize like _calculate_recency_scores."""
        z_suffix = _make_search_result(
            "z-suffix",
            metadata={"occurred_at": "2024-06-01T00:00:00Z"},
        )
        naive = _make_search_result(
            "naive",
            metadata={"occurred_at": "2024-02-01T00:00:00"},
        )

        retriever = _make_retriever()
        retriever._vector_store.search = AsyncMock(return_value=[naive, z_suffix])

        result = await retriever._simple_retrieve(
            query="latest status?",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=_routing(),
            temporal_sort=True,
        )

        contents = [c.content for c, _score in result.chunks]
        assert contents == ["z-suffix", "naive"]
