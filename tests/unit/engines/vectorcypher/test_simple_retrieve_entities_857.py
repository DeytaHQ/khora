"""Regression coverage for issue #857.

``VectorCypherRetriever._simple_retrieve`` is the path taken by backends
without a Neo4j driver (sqlite_lance, surrealdb, postgres-only). Prior to
the #857 fix it returned ``entities=[]`` and ``relationships=[]``
unconditionally, so downstream ``Khora.recall()`` callers received empty
entity / relationship lists even when the graph was populated.

The tests below pin the new behaviour: when chunks are recalled, the
return value projects entities (filtered by ``source_chunk_ids`` overlap)
and relationships (whose endpoints are both in the recalled entity set)
from the storage coordinator.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.storage.temporal import TemporalChunk, TemporalSearchResult


def _make_retriever(*, storage: Any | None = None) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        # No Neo4j driver -> simple path is the only path.
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(enable_reranking=False),
        storage=storage,
    )


def _make_search_result(content: str, *, chunk_id: UUID | None = None) -> TemporalSearchResult:
    tc = TemporalChunk(
        id=chunk_id or uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        embedding=None,
        occurred_at=datetime.now(UTC),
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
class TestSimpleRetrieveEntityProjection857:
    """#857: entities + relationships are populated from storage in the
    simple (graph-less) retrieve path."""

    @pytest.mark.asyncio
    async def test_entities_populated_from_storage_filtered_by_chunk_overlap(self) -> None:
        """Entity whose ``source_chunk_ids`` overlaps a recalled chunk is included.

        Entity whose ``source_chunk_ids`` does NOT overlap is excluded.
        """
        chunk_id = uuid4()
        sr = _make_search_result("hit", chunk_id=chunk_id)

        ns = uuid4()
        hit_entity = Entity(
            id=uuid4(),
            namespace_id=ns,
            name="HitEntity",
            entity_type="PERSON",
            source_chunk_ids=[chunk_id],
        )
        miss_entity = Entity(
            id=uuid4(),
            namespace_id=ns,
            name="MissEntity",
            entity_type="PERSON",
            source_chunk_ids=[uuid4()],  # different chunk
        )

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[hit_entity, miss_entity])
        storage.list_relationships = AsyncMock(return_value=[])

        retriever = _make_retriever(storage=storage)
        retriever._vector_store.search = AsyncMock(return_value=[sr])

        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
            routing=_routing(),
        )

        assert len(result.chunks) == 1
        # Only the entity whose source_chunk_ids intersects recalled chunks.
        entity_names = {e.name for e, _score in result.entities}
        assert entity_names == {"HitEntity"}
        # list_entities was called against the coordinator with a sensible cap.
        storage.list_entities.assert_awaited_once()
        assert storage.list_entities.await_args.kwargs.get("limit") == 1000

    @pytest.mark.asyncio
    async def test_list_entities_called_with_recalled_chunk_ids(self) -> None:
        """#1448: the entity projection pushes the recalled chunk ids down to
        ``list_entities(source_chunk_ids=...)`` rather than scanning the whole
        namespace and filtering in Python."""
        c1, c2 = uuid4(), uuid4()
        sr1 = _make_search_result("c1", chunk_id=c1)
        sr2 = _make_search_result("c2", chunk_id=c2)

        ns = uuid4()
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])
        storage.list_relationships = AsyncMock(return_value=[])

        retriever = _make_retriever(storage=storage)
        retriever._vector_store.search = AsyncMock(return_value=[sr1, sr2])

        await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
            routing=_routing(),
        )

        storage.list_entities.assert_awaited_once()
        pushed = storage.list_entities.await_args.kwargs.get("source_chunk_ids")
        assert pushed is not None, "recalled chunk ids must be pushed down to list_entities"
        assert set(pushed) == {c1, c2}

    @pytest.mark.asyncio
    async def test_list_relationships_called_with_recalled_entity_ids(self) -> None:
        """#1451: the relationship projection pushes the recalled entity ids
        down to ``list_relationships(between_entity_ids=...)`` — the
        BOTH-endpoints filter now runs in the backend rather than a
        namespace-wide scan + Python filter (the analog of the #1448 entity-side
        chunk-id pushdown)."""
        chunk_id = uuid4()
        sr = _make_search_result("hit", chunk_id=chunk_id)

        ns = uuid4()
        ent_a = Entity(id=uuid4(), namespace_id=ns, name="A", source_chunk_ids=[chunk_id])
        ent_b = Entity(id=uuid4(), namespace_id=ns, name="B", source_chunk_ids=[chunk_id])

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[ent_a, ent_b])
        storage.list_relationships = AsyncMock(return_value=[])

        retriever = _make_retriever(storage=storage)
        retriever._vector_store.search = AsyncMock(return_value=[sr])

        await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
            routing=_routing(),
        )

        storage.list_relationships.assert_awaited_once()
        pushed = storage.list_relationships.await_args.kwargs.get("between_entity_ids")
        assert pushed is not None, "recalled entity ids must be pushed down to list_relationships"
        assert set(pushed) == {ent_a.id, ent_b.id}

    @pytest.mark.asyncio
    async def test_relationships_from_storage_surface_verbatim(self) -> None:
        """The endpoint filter is now pushed down to the backend (#1451): the
        retriever passes ``between_entity_ids`` and trusts ``list_relationships``
        to return only edges whose BOTH endpoints are in the recalled set. The
        retriever no longer re-filters in Python — every edge the backend
        returns must surface verbatim."""
        chunk_id = uuid4()
        sr = _make_search_result("hit", chunk_id=chunk_id)

        ns = uuid4()
        ent_a = Entity(id=uuid4(), namespace_id=ns, name="A", source_chunk_ids=[chunk_id])
        ent_b = Entity(id=uuid4(), namespace_id=ns, name="B", source_chunk_ids=[chunk_id])

        # Backend has already applied the between_entity_ids filter, so only the
        # in-set edge A→B comes back.
        rel_ab = Relationship(
            id=uuid4(),
            namespace_id=ns,
            source_entity_id=ent_a.id,
            target_entity_id=ent_b.id,
            relationship_type="KNOWS",
        )

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[ent_a, ent_b])
        storage.list_relationships = AsyncMock(return_value=[rel_ab])

        retriever = _make_retriever(storage=storage)
        retriever._vector_store.search = AsyncMock(return_value=[sr])

        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
            routing=_routing(),
        )

        assert {e.name for e, _ in result.entities} == {"A", "B"}
        rel_ids = {r.id for r, _ in result.relationships}
        assert rel_ids == {rel_ab.id}, "every edge the backend returns must surface verbatim"

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty_entities_no_storage_call(self) -> None:
        """When no chunks are recalled we should not query storage at all
        and the result must carry empty entity / relationship lists."""
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])
        storage.list_relationships = AsyncMock(return_value=[])

        retriever = _make_retriever(storage=storage)
        retriever._vector_store.search = AsyncMock(return_value=[])

        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=_routing(),
        )

        assert result.chunks == []
        assert result.entities == []
        assert result.relationships == []
        # No need to scan the namespace when there are no chunks to filter by.
        storage.list_entities.assert_not_awaited()
        storage.list_relationships.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_storage_coordinator_returns_empty_entities(self) -> None:
        """Retriever with ``storage=None`` (legacy / test path) must still
        return chunks and just leave entities / relationships empty rather
        than crash."""
        retriever = _make_retriever(storage=None)
        retriever._vector_store.search = AsyncMock(return_value=[_make_search_result("hit")])

        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=_routing(),
        )

        assert len(result.chunks) == 1
        assert result.entities == []
        assert result.relationships == []

    @pytest.mark.asyncio
    async def test_storage_failure_degrades_to_empty_entities(self) -> None:
        """A storage error must not crash the recall - empty entities /
        relationships is the documented degradation, matching the rest of
        the retriever's defensive style."""
        storage = MagicMock()
        storage.list_entities = AsyncMock(side_effect=RuntimeError("simulated DB blip"))
        storage.list_relationships = AsyncMock(return_value=[])

        retriever = _make_retriever(storage=storage)
        retriever._vector_store.search = AsyncMock(return_value=[_make_search_result("hit")])

        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=_routing(),
        )

        assert len(result.chunks) == 1
        assert result.entities == []
        # No relationships fetch is attempted when entity fetch failed.
        storage.list_relationships.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_entity_score_reflects_mention_overlap_fraction(self) -> None:
        """An entity mentioned in many recalled chunks should outrank one
        mentioned in just one, all else equal. Concretely, score is
        overlap / len(source_chunk_ids)."""
        c1, c2, c3 = uuid4(), uuid4(), uuid4()
        sr1 = _make_search_result("c1", chunk_id=c1)
        sr2 = _make_search_result("c2", chunk_id=c2)
        sr3 = _make_search_result("c3", chunk_id=c3)

        ns = uuid4()
        wide = Entity(id=uuid4(), namespace_id=ns, name="Wide", source_chunk_ids=[c1, c2, c3])
        narrow = Entity(id=uuid4(), namespace_id=ns, name="Narrow", source_chunk_ids=[c1, uuid4(), uuid4()])

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[wide, narrow])
        storage.list_relationships = AsyncMock(return_value=[])

        retriever = _make_retriever(storage=storage)
        retriever._vector_store.search = AsyncMock(return_value=[sr1, sr2, sr3])

        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
            routing=_routing(),
        )

        scores_by_name = {e.name: s for e, s in result.entities}
        # 3/3 vs 1/3
        assert scores_by_name["Wide"] == pytest.approx(1.0)
        assert scores_by_name["Narrow"] == pytest.approx(1 / 3)
