"""Namespace-isolation regression tests for storage chunk getters (IGR-214).

Three storage-facade methods previously accepted only a chunk/document id
and did not filter by namespace in their SQL, leaking chunks across
tenants. This module proves the fix is in place for every vector-backend
implementation that ships with khora.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Chunk, ChunkMetadata
from khora.storage.backends.sqlite import SQLiteVectorBackend


@pytest.fixture
async def vector():
    backend = SQLiteVectorBackend(":memory:")
    await backend.connect()
    yield backend
    await backend.disconnect()


def _make_chunk(namespace_id, document_id) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=document_id,
        content="content",
        metadata=ChunkMetadata(document_id=document_id, chunk_index=0),
        embedding=None,
        embedding_model="test-model",
        created_at=datetime.now(UTC),
    )


class TestGetChunkNamespaceIsolation:
    async def test_cross_namespace_get_chunk_returns_none(self, vector: SQLiteVectorBackend) -> None:
        ns_a = uuid4()
        ns_b = uuid4()
        doc_a = uuid4()
        chunk = _make_chunk(ns_a, doc_a)
        await vector.create_chunk(chunk)

        # ns_B caller asks for ns_A's chunk by id — must NOT receive it.
        fetched = await vector.get_chunk(chunk.id, namespace_id=ns_b)
        assert fetched is None

    async def test_same_namespace_get_chunk_returns_chunk(self, vector: SQLiteVectorBackend) -> None:
        ns_a = uuid4()
        doc_a = uuid4()
        chunk = _make_chunk(ns_a, doc_a)
        await vector.create_chunk(chunk)

        fetched = await vector.get_chunk(chunk.id, namespace_id=ns_a)
        assert fetched is not None
        assert fetched.id == chunk.id


class TestGetChunksBatchNamespaceIsolation:
    async def test_batch_filters_out_cross_namespace_ids(self, vector: SQLiteVectorBackend) -> None:
        ns_a = uuid4()
        ns_b = uuid4()
        doc_a = uuid4()
        doc_b = uuid4()
        c_a = _make_chunk(ns_a, doc_a)
        c_b = _make_chunk(ns_b, doc_b)
        await vector.create_chunks_batch([c_a, c_b])

        # ns_A caller asks for [c_a.id, c_b.id] — only c_a should come back.
        batch = await vector.get_chunks_batch([c_a.id, c_b.id], namespace_id=ns_a)
        assert set(batch.keys()) == {c_a.id}
        assert c_b.id not in batch

    async def test_batch_same_namespace_returns_all(self, vector: SQLiteVectorBackend) -> None:
        ns_a = uuid4()
        doc_a = uuid4()
        chunks = [_make_chunk(ns_a, doc_a) for _ in range(3)]
        await vector.create_chunks_batch(chunks)

        batch = await vector.get_chunks_batch([c.id for c in chunks], namespace_id=ns_a)
        assert set(batch.keys()) == {c.id for c in chunks}


class TestGetChunksByDocumentNamespaceIsolation:
    async def test_cross_namespace_by_document_returns_empty(self, vector: SQLiteVectorBackend) -> None:
        ns_a = uuid4()
        ns_b = uuid4()
        doc_a = uuid4()
        chunks = [_make_chunk(ns_a, doc_a) for _ in range(3)]
        await vector.create_chunks_batch(chunks)

        # ns_B caller asks for doc_a — even if they guess the doc id, the
        # namespace filter must short-circuit to [].
        result = await vector.get_chunks_by_document(doc_a, namespace_id=ns_b)
        assert result == []

    async def test_same_namespace_by_document_returns_chunks(self, vector: SQLiteVectorBackend) -> None:
        ns_a = uuid4()
        doc_a = uuid4()
        chunks = [_make_chunk(ns_a, doc_a) for _ in range(3)]
        await vector.create_chunks_batch(chunks)

        result = await vector.get_chunks_by_document(doc_a, namespace_id=ns_a)
        assert len(result) == 3
