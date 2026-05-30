"""``VectorCypherRetriever`` surfaces persisted ``occurred_at`` on recall (#859).

Three ``Chunk(...)`` construction sites in ``retriever.py`` used to drop
the persisted ``occurred_at`` column when building the in-memory ``Chunk``,
even though they put it in ``metadata``. The downstream recall projection
in ``engine._build_recall_result`` reads ``chunk.source_timestamp`` (default
``None`` on ``Chunk``), so the final ``RecallChunk.occurred_at`` was
silently ``None``.

The three sites:

* ``_typed_entity_recent_retrieve`` (Neo4j fast path) - ``c_data`` is a
  Neo4j Chunk node dict, ``occurred_at`` is an ISO string.
* ``_simple_retrieve`` (graph-less vector path) - ``r.chunk`` is a
  ``TemporalChunk`` from the vector store, ``occurred_at`` is a datetime.
* ``_fetch_chunks_from_entities`` (SurrealDB-only fallback / Neo4j
  read-path) - ``record["occurred_at"]`` may be either a datetime
  (SurrealDB shim) or an ISO string (Neo4j).

The fix passes ``source_timestamp=`` through to the ``Chunk`` constructor
at each site, coercing strings to datetimes via ``_coerce_occurred_at``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity
from khora.engines.skeleton.backends import TemporalChunk, TemporalSearchResult
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
    _coerce_occurred_at,
)

T_SOURCE = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)


@pytest.mark.unit
class TestCoerceOccurredAtHelper:
    """The shared coercion helper used by the three Chunk-construction sites."""

    def test_datetime_passthrough(self) -> None:
        assert _coerce_occurred_at(T_SOURCE) == T_SOURCE

    def test_iso_string_is_parsed(self) -> None:
        assert _coerce_occurred_at("2024-06-15T12:30:00+00:00") == T_SOURCE

    def test_none_passthrough(self) -> None:
        assert _coerce_occurred_at(None) is None

    def test_garbage_returns_none(self) -> None:
        assert _coerce_occurred_at("not-a-date") is None

    def test_int_returns_none(self) -> None:
        assert _coerce_occurred_at(12345) is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestSimpleRetrieveSourceTimestamp:
    """Slice B - vector-only path (line ~2185 in ``_simple_retrieve``)."""

    async def test_chunk_carries_source_timestamp_from_temporal_chunk(self) -> None:
        """The vector store returns ``TemporalChunk.occurred_at``; it must
        round-trip onto the constructed ``Chunk.source_timestamp``."""
        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._config = RetrieverConfig(enable_reranking=False, enable_bm25_channel=False)
        retriever._dual_nodes = None
        retriever._neo4j_driver = None
        retriever._storage = None
        retriever._bm25_empty_warned_ns = set()
        retriever._reranker = None
        retriever._expansion_cache = {}

        namespace_id = uuid4()
        tc = TemporalChunk(
            id=uuid4(),
            namespace_id=namespace_id,
            document_id=uuid4(),
            content="hit",
            embedding=None,
            occurred_at=T_SOURCE,
        )
        result = TemporalSearchResult(chunk=tc, similarity=0.9, combined_score=0.9)

        retriever._vector_store = MagicMock()
        retriever._vector_store.search = AsyncMock(return_value=[result])

        # Stub BM25 to skip the parallel branch entirely.
        retriever._bm25_search_chunks = AsyncMock(return_value=[])  # type: ignore[method-assign]

        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.5,
            reasoning="",
        )

        vc_result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1] * 8,
            namespace_id=namespace_id,
            temporal_filter=None,
            limit=5,
            routing=routing,
        )

        assert len(vc_result.chunks) == 1
        chunk, _score = vc_result.chunks[0]
        # Before the fix this was None - the bug Damir reported.
        assert chunk.source_timestamp == T_SOURCE


@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchChunksFromEntitiesSourceTimestamp:
    """Slice B - SurrealDB fallback path (line ~2832 in ``_fetch_chunks_from_entities``)."""

    async def test_chunk_carries_source_timestamp_from_persisted_chunk(self) -> None:
        """The SurrealDB fallback reads ``chunk.source_timestamp`` from the
        unified backend; that value must end up on the returned ``Chunk``."""
        namespace_id = uuid4()
        document_id = uuid4()
        persisted = Chunk(
            id=uuid4(),
            namespace_id=namespace_id,
            document_id=document_id,
            content="persisted",
            source_timestamp=T_SOURCE,
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=namespace_id,
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[persisted.id],
        )

        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._config = RetrieverConfig()
        retriever._dual_nodes = None
        retriever._vector_store = MagicMock()
        retriever._neo4j_driver = None
        storage = MagicMock()
        storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})
        storage.get_chunks_batch = AsyncMock(return_value={persisted.id: persisted})
        retriever._storage = storage

        results = await retriever._fetch_chunks_from_entities(
            entity_ids=[entity.id],
            namespace_id=namespace_id,
            temporal_filter=None,
            limit=10,
        )

        assert len(results) == 1
        _, _, returned_chunk = results[0]
        # Before the fix this was None - the Chunk constructor never received
        # ``source_timestamp`` and the field defaults to None.
        assert returned_chunk.source_timestamp == T_SOURCE

    async def test_chunk_with_no_persisted_source_timestamp_stays_none(self) -> None:
        """When the persisted chunk has no ``source_timestamp``, the result
        chunk also has ``None`` (no false fabrication)."""
        namespace_id = uuid4()
        document_id = uuid4()
        persisted = Chunk(
            id=uuid4(),
            namespace_id=namespace_id,
            document_id=document_id,
            content="persisted-no-ts",
            source_timestamp=None,
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=namespace_id,
            name="Bob",
            entity_type="PERSON",
            source_chunk_ids=[persisted.id],
        )

        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._config = RetrieverConfig()
        retriever._dual_nodes = None
        retriever._vector_store = MagicMock()
        retriever._neo4j_driver = None
        storage = MagicMock()
        storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})
        storage.get_chunks_batch = AsyncMock(return_value={persisted.id: persisted})
        retriever._storage = storage

        results = await retriever._fetch_chunks_from_entities(
            entity_ids=[entity.id],
            namespace_id=namespace_id,
            temporal_filter=None,
            limit=10,
        )

        assert len(results) == 1
        _, _, returned_chunk = results[0]
        assert returned_chunk.source_timestamp is None
