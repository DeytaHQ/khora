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
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
    _coerce_occurred_at,
)
from khora.storage.temporal import TemporalChunk, TemporalSearchResult

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
        round-trip onto the constructed ``Chunk.occurred_at`` (the chunk
        event-time), distinct from the producer ``source_timestamp``."""
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
        # The persisted chunk event-time round-trips onto Chunk.occurred_at.
        # No producer value was supplied, so source_timestamp stays None;
        # the recall projection applies the event-time-then-producer fallback.
        assert chunk.occurred_at == T_SOURCE
        assert chunk.source_timestamp is None


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


# A producer time distinct from the chunk event-time — used to prove the two
# fields are read from their own authoritative sources, never mirrored.
T_PRODUCER = datetime(2024, 6, 10, 8, 0, 0, tzinfo=UTC)


def _bare_retriever() -> VectorCypherRetriever:
    """A retriever instance with the attributes the recency channel touches,
    bypassing ``__init__`` (matches ``TestSimpleRetrieveSourceTimestamp``)."""
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig()
    retriever._dual_nodes = None
    retriever._neo4j_driver = None
    retriever._storage = None
    return retriever


@pytest.mark.unit
@pytest.mark.asyncio
class TestRecencyChannelFirstClassOccurredAt:
    """Recency channel (``_recency_channel_chunks``) now populates the
    first-class ``Chunk.occurred_at`` from the source chunk's own
    ``occurred_at``, not just the metadata blob."""

    async def test_first_class_occurred_at_surfaces(self) -> None:
        """A source chunk carrying ``occurred_at`` produces a Chunk whose
        first-class ``occurred_at`` is that value (not None)."""
        retriever = _bare_retriever()
        namespace_id = uuid4()

        # The temporal store returns a chunk-shaped object with an embedding
        # (required to survive the cosine gate) and its own occurred_at.
        src = MagicMock()
        src.id = uuid4()
        src.namespace_id = namespace_id
        src.document_id = uuid4()
        src.content = "recent"
        src.embedding = [1.0, 0.0, 0.0]
        src.occurred_at = T_SOURCE
        src.source_timestamp = None
        src.created_at = None
        src.metadata = None
        src.chunker_info = None

        retriever._vector_store = MagicMock()
        retriever._vector_store.search_recent_chunks = AsyncMock(return_value=[(src, None)])

        with patch("khora._accel.batch_cosine_similarity", return_value=[(0, 0.9)]):
            result = await retriever._recency_channel_chunks(
                query_embedding=[1.0, 0.0, 0.0],
                namespace_id=namespace_id,
                temporal_filter=None,
            )

        assert len(result) == 1
        _, _, chunk = result[0]
        assert chunk.occurred_at == T_SOURCE

    async def test_occurred_at_column_set_but_source_timestamp_null(self) -> None:
        """Regression: a chunk with the ``occurred_at`` column set but
        ``source_timestamp`` NULL now surfaces ``occurred_at`` on the recall
        chunk. Before the fix the recency channel never passed ``occurred_at``
        to the constructor, so the field was silently None."""
        retriever = _bare_retriever()
        namespace_id = uuid4()

        src = MagicMock()
        src.id = uuid4()
        src.namespace_id = namespace_id
        src.document_id = uuid4()
        src.content = "recent-no-producer"
        src.embedding = [1.0, 0.0, 0.0]
        src.occurred_at = T_SOURCE
        src.source_timestamp = None
        src.created_at = None
        src.metadata = None
        src.chunker_info = None

        retriever._vector_store = MagicMock()
        retriever._vector_store.search_recent_chunks = AsyncMock(return_value=[(src, None)])

        with patch("khora._accel.batch_cosine_similarity", return_value=[(0, 0.9)]):
            result = await retriever._recency_channel_chunks(
                query_embedding=[1.0, 0.0, 0.0],
                namespace_id=namespace_id,
                temporal_filter=None,
            )

        assert len(result) == 1
        _, _, chunk = result[0]
        assert chunk.occurred_at == T_SOURCE
        assert chunk.source_timestamp is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestPPRRewrapTemporalFields:
    """The PPR channel re-wraps hydrated chunks (``ppr_retrieve_chunks``);
    the re-wrap now carries BOTH first-class ``occurred_at`` and
    ``source_timestamp`` from the hydrated source chunk."""

    def _seed_storage(self, chunk: Chunk, entity: Entity) -> MagicMock:
        """A storage coordinator that resolves the single-entity PPR graph and
        hydrates the one source chunk."""
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[entity])
        storage.list_relationships = AsyncMock(return_value=[])
        storage.get_chunks_batch = AsyncMock(return_value={chunk.id: chunk})
        return storage

    async def test_rewrap_carries_both_occurred_at_and_source_timestamp(self) -> None:
        """The hydrated source chunk carries distinct ``occurred_at`` and
        ``source_timestamp``; the re-wrapped Chunk must carry both verbatim
        (source_timestamp is NOT a mirror of occurred_at)."""
        from khora.engines.vectorcypher.ppr_retrieval import ppr_retrieve_chunks

        namespace_id = uuid4()
        hydrated = Chunk(
            id=uuid4(),
            namespace_id=namespace_id,
            document_id=uuid4(),
            content="ppr hit",
            occurred_at=T_SOURCE,
            source_timestamp=T_PRODUCER,
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=namespace_id,
            name="Carol",
            entity_type="PERSON",
            source_chunk_ids=[hydrated.id],
        )
        storage = self._seed_storage(hydrated, entity)

        results, _scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=namespace_id,
            entry_entities=[(entity.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-6,
            top_entities=10,
            limit=10,
        )

        assert len(results) == 1
        _, _, chunk = results[0]
        assert chunk.occurred_at == T_SOURCE
        assert chunk.source_timestamp == T_PRODUCER

    async def test_rewrap_occurred_at_set_source_timestamp_null(self) -> None:
        """Regression: hydrated chunk has the ``occurred_at`` column set but
        ``source_timestamp`` NULL. The re-wrapped Chunk surfaces ``occurred_at``
        (was None before the fix — the re-wrap constructor never received it)."""
        from khora.engines.vectorcypher.ppr_retrieval import ppr_retrieve_chunks

        namespace_id = uuid4()
        hydrated = Chunk(
            id=uuid4(),
            namespace_id=namespace_id,
            document_id=uuid4(),
            content="ppr hit no producer",
            occurred_at=T_SOURCE,
            source_timestamp=None,
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=namespace_id,
            name="Dave",
            entity_type="PERSON",
            source_chunk_ids=[hydrated.id],
        )
        storage = self._seed_storage(hydrated, entity)

        results, _scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=namespace_id,
            entry_entities=[(entity.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-6,
            top_entities=10,
            limit=10,
        )

        assert len(results) == 1
        _, _, chunk = results[0]
        assert chunk.occurred_at == T_SOURCE
        assert chunk.source_timestamp is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestTypedEntityFastPathTemporalFields:
    """Typed-entity fast path (``_typed_entity_recent_retrieve``): the graph
    node's serialized user metadata blob is preserved (no injected
    ``occurred_at`` key that erases user keys) and the real ``source_timestamp``
    node property is read (not mirrored from ``occurred_at``)."""

    def _fast_path_retriever(self, rows: list) -> VectorCypherRetriever:
        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._config = RetrieverConfig()
        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = session_ctx
        session_ctx.__aexit__.return_value = None
        session_ctx.execute_read = AsyncMock(return_value=rows)
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes._session.return_value = session_ctx
        return retriever

    async def test_preserves_user_metadata_and_reads_real_source_timestamp(self) -> None:
        """``c_data.metadata`` is the serialized user blob and the node's
        ``source_timestamp`` differs from ``occurred_at``. The produced Chunk
        must keep the deserialized user keys (no ``occurred_at`` key injected
        that would erase them) and read the real ``source_timestamp``."""
        from khora.storage.backends.mixins import serialize_dict

        namespace_id = uuid4()
        entity_id = uuid4()
        chunk_id = uuid4()
        doc_id = uuid4()

        user_blob = {"author": "erin", "topic": "roadmap"}
        rows = [
            {
                "entity": {
                    "id": str(entity_id),
                    "name": "Ship the prototype",
                    "entity_type": "ACTION_ITEM",
                    "description": "",
                },
                "last_mention": T_SOURCE.isoformat(),
                "evidence_chunk": {
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": "Action: ship the prototype",
                    # The :Chunk node stores the user metadata as a JSON string.
                    "metadata": serialize_dict(user_blob),
                    "occurred_at": T_SOURCE.isoformat(),
                    "source_timestamp": T_PRODUCER.isoformat(),
                },
            }
        ]
        retriever = self._fast_path_retriever(rows)

        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

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
            namespace_id=namespace_id,
            temporal_filter=None,
            graph_depth=1,
            limit=5,
            routing=routing,
        )

        assert len(result.chunks) == 1
        chunk, _score = result.chunks[0]
        # User keys survive verbatim; no injected occurred_at key erased them.
        assert chunk.metadata == user_blob
        assert "occurred_at" not in chunk.metadata
        # Real node property read, distinct from occurred_at (not a mirror).
        assert chunk.occurred_at == T_SOURCE
        assert chunk.source_timestamp == T_PRODUCER


@pytest.mark.unit
@pytest.mark.asyncio
class TestNeo4jGraphBranchSourceTimestamp:
    """Neo4j graph channel (``_fetch_chunks_from_entities`` shared result
    loop): the ``source_timestamp`` record field is read verbatim, NOT
    mirrored from ``occurred_at``."""

    async def test_reads_real_source_timestamp_not_mirror(self) -> None:
        """The Neo4j ``get_chunks_by_entities`` record carries distinct
        ``occurred_at`` and ``source_timestamp`` ISO strings. The produced
        Chunk must read the real ``source_timestamp`` (not occurred_at)."""
        namespace_id = uuid4()
        chunk_id = uuid4()
        doc_id = uuid4()

        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._config = RetrieverConfig()
        retriever._storage = None
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(
            return_value=[
                {
                    "chunk_id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": "graph hit",
                    "total_mentions": 1,
                    "entity_ids": ["a"],
                    "occurred_at": T_SOURCE.isoformat(),
                    "source_timestamp": T_PRODUCER.isoformat(),
                    "metadata": {"author": "frank"},
                }
            ]
        )

        results = await retriever._fetch_chunks_from_entities(
            entity_ids=[uuid4()],
            namespace_id=namespace_id,
            temporal_filter=None,
            limit=10,
        )

        assert len(results) == 1
        _, _, chunk = results[0]
        assert chunk.occurred_at == T_SOURCE
        # The real source_timestamp field, not a mirror of occurred_at.
        assert chunk.source_timestamp == T_PRODUCER
        assert chunk.source_timestamp != chunk.occurred_at
