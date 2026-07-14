"""Stored-blob metadata pass-through, per recall channel.

Every recall channel that produces chunks rebuilds a public ``Chunk`` from what
its store returned. The contract these tests pin is a single invariant applied
uniformly across the channels: the rebuilt ``chunk.metadata`` is the STORED blob
verbatim — the channel injects NO read-time keys into it. In particular the
first-class event-time lives on the ``occurred_at`` COLUMN, never folded back
into the blob, and the graph/PPR bookkeeping (connected entities, the PPR mass)
stays out of it too.

The three keys a channel must never smuggle into the blob:

* ``occurred_at``     — the chunk event-time is a first-class ``Chunk`` column;
* ``connected_entities`` — graph adjacency bookkeeping;
* ``ppr_score``       — the PageRank mass a chunk earned.

Each test seeds a channel's store boundary with a CLEAN blob plus a populated
``occurred_at`` column and asserts the rebuilt chunk keeps the blob intact and
surfaces the event-time on the column.

The contract is symmetric, so ``TestCallerStoredKeysReturnedVerbatim`` covers the
other direction: when the caller legitimately stores those same key NAMES
(``occurred_at`` / ``connected_entities`` / ``ppr_score``) as user data, recall
returns them verbatim — the pass-through must not SCRUB user keys either.

These are complementary to ``test_retriever_coverage_push.py`` (which already
pins the ``_vector_search_chunks`` and the ``_fetch_chunks_from_entities``
dual-nodes channels): the channels here are the remaining producers — the simple
vector path, BM25, the storage graph fallback, recency, PPR, and the typed-entity
fast path. The skeleton engine's channel and the end-to-end cross-channel filter
behaviour are pinned in the matrix integration lane
(``tests/integration/matrix/test_stored_blob_filter_invariant.py``), where a real
recall exercises them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity, Relationship
from khora.engines.vectorcypher.ppr_retrieval import ppr_retrieve_chunks
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.query import SearchMode

pytestmark = pytest.mark.unit

# The event-time carried on the first-class column throughout this module.
_OCCURRED_AT = datetime(2026, 4, 1, tzinfo=UTC)

# The keys no channel may inject into the stored blob at read time.
_INJECTION_KEYS = ("occurred_at", "connected_entities", "ppr_score")

# A representative clean stored blob: user-owned keys only, none of which is an
# injection key. Frozen per-call via ``dict(...)`` so a channel that mutated it
# in place would be caught by the identity-independent equality assertion.
_CLEAN_BLOB: dict[str, Any] = {"author": "alice", "tier": "gold", "score": 3}

# The adversarial mirror: a stored blob whose USER keys are named exactly like
# the read-time bookkeeping (``occurred_at`` / ``connected_entities`` /
# ``ppr_score``) but hold caller-owned values. The blob is user data, so recall
# must return these verbatim — never scrub or overwrite them with the first-class
# column or graph/PPR bookkeeping. The values are deliberately un-column-like (a
# plain string event-time, a list, a float) so a channel that folded the column
# in would be caught by the exact-equality check.
_CALLER_BLOB: dict[str, Any] = {
    "author": "bob",
    "occurred_at": "caller-owned-event-string",
    "connected_entities": ["caller-entity-1", "caller-entity-2"],
    "ppr_score": 0.123,
}


def _make_retriever(
    *,
    config: RetrieverConfig | None = None,
    vector_store: Any | None = None,
    storage: Any | None = None,
) -> VectorCypherRetriever:
    """A retriever with mocked boundaries (mirrors the coverage-push helper)."""
    return VectorCypherRetriever(
        vector_store=vector_store if vector_store is not None else AsyncMock(),
        neo4j_driver=AsyncMock(),
        embedder=AsyncMock(),
        config=config or RetrieverConfig(),
        storage=storage,
    )


def _assert_stored_blob(chunk: Chunk) -> None:
    """The rebuilt chunk carries the clean blob verbatim, no injected keys."""
    assert chunk.metadata == _CLEAN_BLOB
    for key in _INJECTION_KEYS:
        assert key not in chunk.metadata, f"channel injected {key!r} into the stored blob"
    # The event-time is on the first-class column, not the blob.
    assert chunk.occurred_at == _OCCURRED_AT


def _assert_caller_blob_verbatim(chunk: Chunk) -> None:
    """The rebuilt chunk returns the caller's blob verbatim, keys and values intact.

    The blob legitimately carries user keys named ``occurred_at`` /
    ``connected_entities`` / ``ppr_score``. Recall must NOT scrub them (an
    over-correction) nor overwrite the ``occurred_at`` key with the first-class
    column: the caller string stays in the blob and the column is surfaced
    separately on ``chunk.occurred_at``.
    """
    assert chunk.metadata == _CALLER_BLOB
    for key in _INJECTION_KEYS:
        assert key in chunk.metadata, f"channel scrubbed caller-owned key {key!r} from the stored blob"
    assert chunk.metadata["occurred_at"] == "caller-owned-event-string"
    # The caller's blob value is untouched; the real event-time is on the column.
    assert chunk.occurred_at == _OCCURRED_AT


# --------------------------------------------------------------------------- #
# vector "simple" channel — _simple_retrieve
# --------------------------------------------------------------------------- #


def _vector_store_result() -> MagicMock:
    """A pgvector-shaped search result carrying a clean blob + occurred_at column."""
    result = MagicMock()
    result.chunk = MagicMock()
    result.chunk.id = uuid4()
    result.chunk.namespace_id = uuid4()
    result.chunk.document_id = uuid4()
    result.chunk.content = "simple vector hit"
    result.chunk.metadata = dict(_CLEAN_BLOB)
    result.chunk.occurred_at = _OCCURRED_AT
    result.chunk.created_at = None
    result.chunk.source_timestamp = None
    result.chunk.chunker_info = {}
    result.combined_score = 0.7
    result.similarity = 0.6
    return result


class TestSimpleRetrieveChannel:
    @pytest.mark.asyncio
    async def test_simple_retrieve_preserves_stored_blob(self) -> None:
        retriever = _make_retriever()
        retriever._vector_store.search = AsyncMock(return_value=[_vector_store_result()])

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="simple",
        )
        # VECTOR mode: pure vector store search, no BM25 fan-out or graph channel.
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1, 0.2],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=5,
            routing=routing,
            mode=SearchMode.VECTOR,
        )
        assert len(result.chunks) == 1
        chunk, _score = result.chunks[0]
        _assert_stored_blob(chunk)


# --------------------------------------------------------------------------- #
# BM25 channel — _bm25_search_chunks (coordinator fallback pass-through)
# --------------------------------------------------------------------------- #


class TestBm25Channel:
    @pytest.mark.asyncio
    async def test_bm25_passes_stored_blob_through(self) -> None:
        ns = uuid4()
        stored = Chunk(
            id=uuid4(),
            namespace_id=ns,
            document_id=uuid4(),
            content="bm25 hit",
            metadata=dict(_CLEAN_BLOB),
            occurred_at=_OCCURRED_AT,
        )
        storage = AsyncMock()
        storage.search_fulltext_chunks = AsyncMock(return_value=[(stored, 0.9)])
        # No temporal-store fulltext method → the coordinator fallback runs.
        retriever = _make_retriever(storage=storage, vector_store=MagicMock(spec=[]))

        result = await retriever._bm25_search_chunks(query="alpha beta", namespace_id=ns, limit=10)
        assert len(result) == 1
        _cid, _score, chunk = result[0]
        _assert_stored_blob(chunk)


# --------------------------------------------------------------------------- #
# graph SurrealDB-fallback channel — _fetch_chunks_from_entities (no dual_nodes)
# --------------------------------------------------------------------------- #


class TestGraphStorageFallbackChannel:
    @pytest.mark.asyncio
    async def test_storage_fallback_preserves_stored_blob(self) -> None:
        ns = uuid4()
        stored = Chunk(
            id=uuid4(),
            namespace_id=ns,
            document_id=uuid4(),
            content="graph fallback hit",
            metadata=dict(_CLEAN_BLOB),
            occurred_at=_OCCURRED_AT,
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=ns,
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[stored.id],
        )
        # SurrealDB-only shape: no graph backend (dual_nodes None), storage wired.
        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._config = RetrieverConfig()
        retriever._dual_nodes = None
        retriever._vector_store = MagicMock()
        retriever._neo4j_driver = None
        storage = MagicMock()
        storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})
        storage.get_chunks_batch = AsyncMock(return_value={stored.id: stored})
        retriever._storage = storage

        results = await retriever._fetch_chunks_from_entities(
            entity_ids=[entity.id],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
        )
        assert len(results) == 1
        _cid, _score, chunk = results[0]
        _assert_stored_blob(chunk)


# --------------------------------------------------------------------------- #
# recency channel — _recency_channel_chunks
# --------------------------------------------------------------------------- #


class TestRecencyChannel:
    @pytest.mark.asyncio
    async def test_recency_preserves_stored_blob(self) -> None:
        # A recency-store chunk carrying an embedding (so it clears the cosine
        # gate), a clean blob, and a populated occurred_at column.
        store_chunk = MagicMock()
        store_chunk.id = uuid4()
        store_chunk.namespace_id = uuid4()
        store_chunk.document_id = uuid4()
        store_chunk.content = "recent hit"
        store_chunk.embedding = [1.0, 0.0, 0.0]
        store_chunk.metadata = dict(_CLEAN_BLOB)
        store_chunk.occurred_at = _OCCURRED_AT
        store_chunk.created_at = None
        store_chunk.source_timestamp = None
        store_chunk.chunker_info = {}

        vstore = MagicMock()
        vstore.search_recent_chunks = AsyncMock(return_value=[(store_chunk, None)])
        retriever = _make_retriever(
            config=RetrieverConfig(temporal_query_relevance_floor=0.5),
            vector_store=vstore,
        )

        with patch("khora._accel.batch_cosine_similarity", return_value=[(0, 0.9)]):
            result = await retriever._recency_channel_chunks(
                query_embedding=[1.0, 0.0, 0.0],
                namespace_id=uuid4(),
                temporal_filter=None,
            )
        assert len(result) == 1
        _cid, _score, chunk = result[0]
        _assert_stored_blob(chunk)


# --------------------------------------------------------------------------- #
# PPR channel — ppr_retrieve_chunks
# --------------------------------------------------------------------------- #


class TestPprChannel:
    @pytest.mark.asyncio
    async def test_ppr_rebuild_omits_ppr_score_and_keeps_blob(self) -> None:
        ns = uuid4()
        # Two chunks so the PPR ranking is non-degenerate; the seed's chunk is the
        # one whose rebuilt blob we assert on.
        chunk_a = Chunk(
            id=uuid4(),
            namespace_id=ns,
            document_id=uuid4(),
            content="Alice met Bob",
            metadata=dict(_CLEAN_BLOB),
            occurred_at=_OCCURRED_AT,
        )
        chunk_b = Chunk(
            id=uuid4(),
            namespace_id=ns,
            document_id=uuid4(),
            content="Bob met Carol",
            metadata={"author": "carol"},
            occurred_at=_OCCURRED_AT,
        )
        e_alice = Entity(id=uuid4(), namespace_id=ns, name="Alice", source_chunk_ids=[chunk_a.id])
        e_bob = Entity(id=uuid4(), namespace_id=ns, name="Bob", source_chunk_ids=[chunk_a.id, chunk_b.id])
        e_carol = Entity(id=uuid4(), namespace_id=ns, name="Carol", source_chunk_ids=[chunk_b.id])
        rels = [
            Relationship(
                id=uuid4(),
                namespace_id=ns,
                source_entity_id=e_alice.id,
                target_entity_id=e_bob.id,
                relationship_type="RELATES_TO",
                weight=1.0,
            ),
            Relationship(
                id=uuid4(),
                namespace_id=ns,
                source_entity_id=e_bob.id,
                target_entity_id=e_carol.id,
                relationship_type="RELATES_TO",
                weight=1.0,
            ),
        ]
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[e_alice, e_bob, e_carol])
        storage.list_relationships = AsyncMock(return_value=rels)
        storage.get_chunks_batch = AsyncMock(return_value={chunk_a.id: chunk_a, chunk_b.id: chunk_b})

        results, _entity_scores = await ppr_retrieve_chunks(
            storage=storage,
            namespace_id=ns,
            entry_entities=[(e_alice.id, 1.0)],
            damping=0.85,
            max_iter=50,
            tol=1e-5,
            top_entities=10,
            limit=10,
        )
        assert results
        rebuilt = {cid: chunk for cid, _score, chunk in results}
        assert chunk_a.id in rebuilt
        _assert_stored_blob(rebuilt[chunk_a.id])


# --------------------------------------------------------------------------- #
# typed-entity fast path — _typed_entity_recent_retrieve
# --------------------------------------------------------------------------- #


class TestTypedEntityFastPathChannel:
    @pytest.mark.asyncio
    async def test_fast_path_deserializes_stored_blob_without_injection(self) -> None:
        retriever = _make_retriever()
        ns = uuid4()
        entity_id = uuid4()
        chunk_id = uuid4()
        doc_id = uuid4()

        # The :Chunk node stores the user blob serialized as JSON; the fast path
        # deserializes it. occurred_at is a separate node property.
        rows = [
            {
                "entity": {
                    "id": str(entity_id),
                    "name": "Ship the prototype",
                    "entity_type": "ACTION_ITEM",
                    "description": "",
                },
                "last_mention": _OCCURRED_AT.isoformat(),
                "evidence_chunk": {
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": "Action: ship the prototype",
                    "metadata": '{"author": "alice", "tier": "gold", "score": 3}',
                    "occurred_at": _OCCURRED_AT.isoformat(),
                },
            }
        ]

        session_ctx = AsyncMock()
        session_ctx.__aenter__.return_value = session_ctx
        session_ctx.__aexit__.return_value = None
        session_ctx.execute_read = AsyncMock(return_value=rows)
        retriever._dual_nodes = MagicMock()
        retriever._dual_nodes._session.return_value = session_ctx

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
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=5,
            routing=routing,
        )
        assert len(result.chunks) == 1
        chunk, _score = result.chunks[0]
        _assert_stored_blob(chunk)


# --------------------------------------------------------------------------- #
# graph channel in-memory post-filter agrees with the SQL-pushdown channels
# --------------------------------------------------------------------------- #


class TestGraphChannelPostFilterAgreement:
    """A graph-produced chunk clears the same filters the SQL pushdown matches.

    The graph channel reads chunks from the graph store (metadata is a serialized
    property, not push-downable), so the engine re-checks the WHOLE filter AST in
    memory against each graph chunk with
    ``compile_python(filter_ast, build_compile_context("Chunk", "split"))`` before
    fusion (retriever.py graph post-filter site). Because the graph channel now
    rebuilds ``chunk.metadata`` as the clean stored blob, that in-memory re-check
    reaches the SAME verdict the vector/BM25 SQL pushdown reaches on the same blob
    — the cross-channel agreement this suite exists to pin. A read-time injection
    of ``occurred_at`` into the blob would flip ``metadata.occurred_at $exists``
    on the graph side only, silently dropping the chunk from the graph channel
    while the SQL channels kept it.
    """

    @staticmethod
    def _graph_post_filter(filter_dict: dict[str, Any]):  # noqa: ANN205 - predicate callable
        """Build the exact in-memory post-filter the graph channel applies."""
        from khora.filter import RecallFilter
        from khora.filter.ast import parse_to_ast
        from khora.filter.compilers.python import compile_python
        from khora.filter.execute import build_compile_context

        ast = parse_to_ast(RecallFilter.model_validate(filter_dict))
        return compile_python(ast, build_compile_context("Chunk", on_unsupported="split")).predicate

    async def _graph_chunk(self) -> Chunk:
        """The chunk the storage graph-fallback channel produces from a clean blob."""
        ns = uuid4()
        stored = Chunk(
            id=uuid4(),
            namespace_id=ns,
            document_id=uuid4(),
            content="graph hit",
            metadata=dict(_CLEAN_BLOB),
            occurred_at=_OCCURRED_AT,
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=ns,
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[stored.id],
        )
        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._config = RetrieverConfig()
        retriever._dual_nodes = None
        retriever._vector_store = MagicMock()
        retriever._neo4j_driver = None
        storage = MagicMock()
        storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})
        storage.get_chunks_batch = AsyncMock(return_value={stored.id: stored})
        retriever._storage = storage

        results = await retriever._fetch_chunks_from_entities(
            entity_ids=[entity.id],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
        )
        assert len(results) == 1
        return results[0][2]

    @pytest.mark.asyncio
    async def test_absent_occurred_at_key_survives_graph_post_filter(self) -> None:
        """``metadata.occurred_at $exists false`` keeps the clean-blob graph chunk."""
        chunk = await self._graph_chunk()
        keep = self._graph_post_filter({"metadata.occurred_at": {"$exists": False}})
        drop = self._graph_post_filter({"metadata.occurred_at": {"$exists": True}})
        assert keep(chunk) is True
        assert drop(chunk) is False

    @pytest.mark.asyncio
    async def test_whole_blob_eq_survives_graph_post_filter(self) -> None:
        """A whole-blob ``$eq`` keeps the graph chunk whose stored blob matches."""
        chunk = await self._graph_chunk()
        match = self._graph_post_filter({"metadata": dict(_CLEAN_BLOB)})
        miss = self._graph_post_filter({"metadata": {"author": "someone-else"}})
        assert match(chunk) is True
        assert miss(chunk) is False


# --------------------------------------------------------------------------- #
# caller-stored keys are user data — returned verbatim, never scrubbed
# --------------------------------------------------------------------------- #


class TestCallerStoredKeysReturnedVerbatim:
    """A caller-stored ``occurred_at`` / ``connected_entities`` / ``ppr_score`` in
    the blob is USER data and comes back verbatim.

    The rest of this suite asserts these keys are ABSENT — but that only guards
    against read-time INJECTION. The stored-blob contract is symmetric: recall
    must also not SCRUB a key the caller legitimately stored under one of those
    names, nor overwrite the caller's ``occurred_at`` value with the first-class
    column. An over-correction that stripped the bookkeeping names from user blobs
    would pass every absent-key test but break this one. Covers the two changed
    constructor sites where the risk is real: the graph fetch (where
    ``connected_entities`` used to be injected) and the vector search.
    """

    @pytest.mark.asyncio
    async def test_graph_fetch_returns_caller_keys_verbatim(self) -> None:
        ns = uuid4()
        stored = Chunk(
            id=uuid4(),
            namespace_id=ns,
            document_id=uuid4(),
            content="graph hit",
            metadata=dict(_CALLER_BLOB),
            occurred_at=_OCCURRED_AT,
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=ns,
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[stored.id],
        )
        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._config = RetrieverConfig()
        retriever._dual_nodes = None
        retriever._vector_store = MagicMock()
        retriever._neo4j_driver = None
        storage = MagicMock()
        storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})
        storage.get_chunks_batch = AsyncMock(return_value={stored.id: stored})
        retriever._storage = storage

        results = await retriever._fetch_chunks_from_entities(
            entity_ids=[entity.id],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
        )
        assert len(results) == 1
        _cid, _score, chunk = results[0]
        _assert_caller_blob_verbatim(chunk)

    @pytest.mark.asyncio
    async def test_vector_search_returns_caller_keys_verbatim(self) -> None:
        ns = uuid4()
        result = _vector_store_result()
        result.chunk.namespace_id = ns
        result.chunk.metadata = dict(_CALLER_BLOB)
        retriever = _make_retriever()
        retriever._vector_store.search = AsyncMock(return_value=[result])

        results = await retriever._vector_search_chunks(
            query_embedding=[0.1, 0.2],
            namespace_id=ns,
            temporal_filter=None,
            query_text="q",
            limit=5,
        )
        assert len(results) == 1
        _cid, _score, chunk = results[0]
        _assert_caller_blob_verbatim(chunk)
