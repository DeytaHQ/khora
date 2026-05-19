"""Regression test for #754.

``VectorCypherRetriever._fetch_chunks_from_entities`` falls back to
``self._storage`` (the unified backend) when no graph backend (Neo4j /
``self._dual_nodes``) is wired. The fallback writes a dict per chunk that
the downstream result-building loop consumes — and it must include
``document_id`` because the loop unconditionally does
``UUID(record["document_id"])``. Forgetting that key crashes the recall
path on SurrealDB-only deployments.

This test mirrors that exact shape: configure the retriever with
``_dual_nodes=None`` and a stub ``_storage`` that returns one entity with
one source chunk, then call ``_fetch_chunks_from_entities`` and assert
the returned chunk preserves the ``document_id`` from the storage layer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity
from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever


def _retriever_with_surrealdb_only_storage(entity: Entity, chunk: Chunk) -> VectorCypherRetriever:
    """Build a retriever that mirrors the SurrealDB-only deployment shape.

    No graph backend → ``_dual_nodes`` is None. ``_storage`` is mocked at
    the boundary; we don't need the rest of the retriever's surface here.
    """
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig()
    retriever._dual_nodes = None
    retriever._vector_store = MagicMock()
    retriever._neo4j_driver = None

    storage = MagicMock()
    storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})
    storage.get_chunks_batch = AsyncMock(return_value={chunk.id: chunk})
    retriever._storage = storage
    return retriever


@pytest.mark.unit
@pytest.mark.asyncio
async def test_surrealdb_fallback_preserves_document_id() -> None:
    """The fallback must populate ``document_id`` so the result-builder
    loop's ``UUID(record["document_id"])`` doesn't KeyError out."""
    namespace_id = uuid4()
    document_id = uuid4()
    chunk = Chunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=document_id,
        content="repro chunk",
    )
    entity = Entity(
        id=uuid4(),
        namespace_id=namespace_id,
        name="Alice",
        entity_type="PERSON",
        source_chunk_ids=[chunk.id],
    )

    retriever = _retriever_with_surrealdb_only_storage(entity, chunk)

    results = await retriever._fetch_chunks_from_entities(
        entity_ids=[entity.id],
        namespace_id=namespace_id,
        temporal_filter=None,
        limit=10,
    )

    assert len(results) == 1
    _, _, returned_chunk = results[0]
    # Before the fix this assertion was unreachable — the function raised
    # KeyError on the missing key before returning.
    assert returned_chunk.document_id == document_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_surrealdb_fallback_empty_entity_chunks_returns_empty() -> None:
    """When an entity has no source chunks the fallback short-circuits
    cleanly — this path was already correct but is worth pinning."""
    namespace_id = uuid4()
    entity = Entity(
        id=uuid4(),
        namespace_id=namespace_id,
        name="Bob",
        entity_type="PERSON",
        source_chunk_ids=[],
    )

    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig()
    retriever._dual_nodes = None
    retriever._vector_store = MagicMock()
    retriever._neo4j_driver = None
    storage = MagicMock()
    storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})
    storage.get_chunks_batch = AsyncMock(return_value={})
    retriever._storage = storage

    results = await retriever._fetch_chunks_from_entities(
        entity_ids=[entity.id],
        namespace_id=namespace_id,
        temporal_filter=None,
        limit=10,
    )

    assert results == []
