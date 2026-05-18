"""Tests for SQLite embedded backend (relational + vector)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Chunk, ChunkMetadata, Document, DocumentMetadata, MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.core.models.entity import Entity
from khora.core.models.tenancy import TenancyMode
from khora.storage.backends.sqlite import SQLiteRelationalBackend, SQLiteVectorBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def relational():
    backend = SQLiteRelationalBackend(":memory:")
    await backend.connect()
    yield backend
    await backend.disconnect()


@pytest.fixture
async def vector():
    backend = SQLiteVectorBackend(":memory:")
    await backend.connect()
    yield backend
    await backend.disconnect()


def _make_namespace(**kwargs) -> MemoryNamespace:
    defaults = dict(
        id=uuid4(),
        namespace_id=uuid4(),
        tenancy_mode=TenancyMode.SHARED,
        version=1,
        is_active=True,
        config_overrides={},
        sync_checkpoints={},
        metadata={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    defaults.update(kwargs)
    return MemoryNamespace(**defaults)


def _make_document(namespace_id, **kwargs) -> Document:
    defaults = dict(
        id=uuid4(),
        namespace_id=namespace_id,
        content="Hello world",
        status=DocumentStatus.PENDING,
        metadata=DocumentMetadata(
            source="test.txt",
            source_type="file",
            content_type="text/plain",
            title="Test Document",
            author="tester",
            checksum="abc123",
        ),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    defaults.update(kwargs)
    return Document(**defaults)


def _make_chunk(namespace_id, document_id, embedding=None, **kwargs) -> Chunk:
    defaults = dict(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=document_id,
        content="Test chunk content",
        metadata=ChunkMetadata(document_id=document_id, chunk_index=0),
        embedding=embedding,
        embedding_model="test-model",
        created_at=datetime.now(UTC),
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def _make_entity(namespace_id, embedding=None, **kwargs) -> Entity:
    defaults = dict(
        id=uuid4(),
        namespace_id=namespace_id,
        name="Test Entity",
        entity_type="CONCEPT",
        description="A test entity",
        embedding=embedding,
        embedding_model="test-model",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    defaults.update(kwargs)
    return Entity(**defaults)


def _unit_embedding(dim: int, index: int = 0) -> list[float]:
    """Create a unit vector with a 1.0 at the given index."""
    vec = [0.0] * dim
    vec[index % dim] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    async def test_relational_connect_creates_tables(self, relational: SQLiteRelationalBackend):
        assert await relational.is_healthy()

    async def test_vector_connect_creates_tables(self, vector: SQLiteVectorBackend):
        assert await vector.is_healthy()

    async def test_disconnect_marks_unhealthy(self):
        backend = SQLiteRelationalBackend(":memory:")
        await backend.connect()
        assert await backend.is_healthy()
        await backend.disconnect()
        assert not await backend.is_healthy()


# ---------------------------------------------------------------------------
# Namespace CRUD
# ---------------------------------------------------------------------------


class TestNamespaceCRUD:
    async def test_create_and_get(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        created = await relational.create_namespace(ns)
        assert created.id == ns.id

        fetched = await relational.get_namespace(ns.id)
        assert fetched is not None
        assert fetched.namespace_id == ns.namespace_id

    async def test_resolve_namespace_by_namespace_id(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)
        resolved = await relational.resolve_namespace(ns.namespace_id)
        assert resolved == ns.id

    async def test_resolve_namespace_by_id(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)
        resolved = await relational.resolve_namespace(ns.id)
        assert resolved == ns.id

    async def test_resolve_missing_raises(self, relational: SQLiteRelationalBackend):
        with pytest.raises(ValueError, match="No active namespace"):
            await relational.resolve_namespace(uuid4())

    async def test_list_namespaces(self, relational: SQLiteRelationalBackend):
        ns1 = _make_namespace()
        ns2 = _make_namespace()
        await relational.create_namespace(ns1)
        await relational.create_namespace(ns2)

        result = await relational.list_namespaces()
        assert result.total == 2
        assert len(result.items) == 2

    async def test_update_namespace(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        ns.config_overrides = {"key": "value"}
        updated = await relational.update_namespace(ns)
        assert updated.config_overrides == {"key": "value"}

    async def test_create_namespace_version(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        new_ns = await relational.create_namespace_version(previous_version=ns)
        assert new_ns.version == 2
        assert new_ns.namespace_id == ns.namespace_id
        assert new_ns.id != ns.id

    async def test_deactivate_namespace(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        await relational.deactivate_namespace(ns.id)

        # resolve should now fail
        with pytest.raises(ValueError, match="No active namespace"):
            await relational.resolve_namespace(ns.namespace_id)


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------


class TestDocumentCRUD:
    async def test_create_and_get(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        created = await relational.create_document(doc)
        assert created.id == doc.id

        fetched = await relational.get_document(doc.id)
        assert fetched is not None
        assert fetched.metadata.title == "Test Document"

    async def test_list_documents(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc1 = _make_document(ns.id)
        doc2 = _make_document(ns.id)
        doc2.metadata.checksum = "def456"
        await relational.create_document(doc1)
        await relational.create_document(doc2)

        docs = await relational.list_documents(ns.id)
        assert len(docs) == 2

    async def test_update_document(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        await relational.create_document(doc)

        doc.content = "Updated content"
        doc.status = DocumentStatus.COMPLETED
        updated = await relational.update_document(doc)
        assert updated.content == "Updated content"

        fetched = await relational.get_document(doc.id)
        assert fetched.status == DocumentStatus.COMPLETED

    async def test_delete_document(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        await relational.create_document(doc)

        result = await relational.delete_document(doc.id)
        assert result is True

        fetched = await relational.get_document(doc.id)
        assert fetched is None

    async def test_count_documents(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        await relational.create_document(doc)

        count = await relational.count_documents(ns.id)
        assert count == 1

    async def test_update_document_external_id(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        await relational.create_document(doc)
        assert doc.external_id is None

        doc.external_id = "ext-456"
        await relational.update_document(doc)

        fetched = await relational.get_document(doc.id)
        assert fetched is not None
        assert fetched.external_id == "ext-456"

    async def test_create_document_with_external_id(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id, external_id="ext-123")
        created = await relational.create_document(doc)
        assert created.external_id == "ext-123"

        fetched = await relational.get_document(doc.id)
        assert fetched is not None
        assert fetched.external_id == "ext-123"

    async def test_create_document_without_external_id_is_none(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        created = await relational.create_document(doc)
        assert created.external_id is None

        fetched = await relational.get_document(doc.id)
        assert fetched is not None
        assert fetched.external_id is None

    async def test_get_last_activity_at(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        await relational.create_document(doc)

        last = await relational.get_last_activity_at(ns.id)
        assert last is not None

    async def test_get_document_by_checksum(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        await relational.create_document(doc)

        fetched = await relational.get_document_by_checksum(ns.id, "abc123")
        assert fetched is not None
        assert fetched.id == doc.id

    async def test_get_document_by_checksum_excludes_failed(self, relational: SQLiteRelationalBackend):
        """FAILED documents should not be returned by checksum lookup."""
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id, status=DocumentStatus.FAILED)
        await relational.create_document(doc)

        fetched = await relational.get_document_by_checksum(ns.id, "abc123")
        assert fetched is None

    async def test_get_document_by_checksum_returns_completed(self, relational: SQLiteRelationalBackend):
        """COMPLETED documents should still be returned by checksum lookup."""
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id, status=DocumentStatus.COMPLETED)
        await relational.create_document(doc)

        fetched = await relational.get_document_by_checksum(ns.id, "abc123")
        assert fetched is not None
        assert fetched.id == doc.id

    async def test_get_document_stats(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        await relational.create_document(doc)

        count, last = await relational.get_document_stats(ns.id)
        assert count == 1
        assert last is not None

    async def test_get_documents_batch(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc1 = _make_document(ns.id)
        doc2 = _make_document(ns.id)
        doc2.metadata.checksum = "xyz789"
        await relational.create_document(doc1)
        await relational.create_document(doc2)

        batch = await relational.get_documents_batch([doc1.id, doc2.id])
        assert len(batch) == 2

    async def test_get_document_sources_batch(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        doc = _make_document(ns.id)
        await relational.create_document(doc)

        sources = await relational.get_document_sources_batch([doc.id])
        assert len(sources) == 1
        assert sources[doc.id].title == "Test Document"


# ---------------------------------------------------------------------------
# Sync checkpoints
# ---------------------------------------------------------------------------


class TestSyncCheckpoints:
    async def test_set_and_get(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        await relational.set_sync_checkpoint(ns.id, "slack", "2024-01-01")
        checkpoint = await relational.get_sync_checkpoint(ns.id, "slack")
        assert checkpoint == "2024-01-01"

    async def test_get_missing_returns_none(self, relational: SQLiteRelationalBackend):
        checkpoint = await relational.get_sync_checkpoint(uuid4(), "slack")
        assert checkpoint is None

    async def test_upsert_checkpoint(self, relational: SQLiteRelationalBackend):
        ns = _make_namespace()
        await relational.create_namespace(ns)

        await relational.set_sync_checkpoint(ns.id, "slack", "v1")
        await relational.set_sync_checkpoint(ns.id, "slack", "v2")
        checkpoint = await relational.get_sync_checkpoint(ns.id, "slack")
        assert checkpoint == "v2"


# ---------------------------------------------------------------------------
# Chunk operations
# ---------------------------------------------------------------------------


class TestChunkOperations:
    async def test_create_and_get_chunk(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()
        embedding = _unit_embedding(8)
        chunk = _make_chunk(ns_id, doc_id, embedding=embedding)

        created = await vector.create_chunk(chunk)
        assert created.id == chunk.id

        fetched = await vector.get_chunk(chunk.id, namespace_id=ns_id)
        assert fetched is not None
        assert fetched.content == "Test chunk content"
        assert fetched.embedding is not None

    async def test_create_chunks_batch(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()
        chunks = [_make_chunk(ns_id, doc_id, embedding=_unit_embedding(8, i)) for i in range(5)]

        result = await vector.create_chunks_batch(chunks)
        assert len(result) == 5

    async def test_get_chunks_batch(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()
        chunks = [_make_chunk(ns_id, doc_id) for _ in range(3)]
        await vector.create_chunks_batch(chunks)

        batch = await vector.get_chunks_batch([c.id for c in chunks], namespace_id=ns_id)
        assert len(batch) == 3

    async def test_get_chunks_by_document(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()
        chunks = [
            _make_chunk(ns_id, doc_id, metadata=ChunkMetadata(document_id=doc_id, chunk_index=i)) for i in range(3)
        ]
        await vector.create_chunks_batch(chunks)

        result = await vector.get_chunks_by_document(doc_id, namespace_id=ns_id)
        assert len(result) == 3
        # Should be ordered by chunk_index
        assert [c.metadata.chunk_index for c in result] == [0, 1, 2]

    async def test_delete_chunks_by_document(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()
        chunks = [_make_chunk(ns_id, doc_id) for _ in range(3)]
        await vector.create_chunks_batch(chunks)

        deleted = await vector.delete_chunks_by_document(doc_id)
        assert deleted == 3

        result = await vector.get_chunks_by_document(doc_id, namespace_id=ns_id)
        assert len(result) == 0

    async def test_count_chunks(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()
        chunks = [_make_chunk(ns_id, doc_id) for _ in range(4)]
        await vector.create_chunks_batch(chunks)

        count = await vector.count_chunks(ns_id)
        assert count == 4


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    async def test_search_similar(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()

        # Create chunks with different embeddings
        chunks = []
        for i in range(5):
            emb = _unit_embedding(16, i)
            chunks.append(_make_chunk(ns_id, doc_id, embedding=emb))
        await vector.create_chunks_batch(chunks)

        # Search with the first unit vector — should match chunk 0 perfectly
        query = _unit_embedding(16, 0)
        results = await vector.search_similar(ns_id, query, limit=3)
        assert len(results) > 0
        best_chunk, best_score = results[0]
        assert best_score > 0.99  # near-perfect match

    async def test_search_similar_with_min_similarity(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()

        # One matching, one orthogonal
        chunks = [
            _make_chunk(ns_id, doc_id, embedding=_unit_embedding(16, 0)),
            _make_chunk(ns_id, doc_id, embedding=_unit_embedding(16, 1)),
        ]
        await vector.create_chunks_batch(chunks)

        query = _unit_embedding(16, 0)
        results = await vector.search_similar(ns_id, query, min_similarity=0.5)
        # Only the matching one should pass the threshold
        assert len(results) == 1

    async def test_search_similar_with_filter(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_a = uuid4()
        doc_b = uuid4()

        await vector.create_chunk(_make_chunk(ns_id, doc_a, embedding=_unit_embedding(8, 0)))
        await vector.create_chunk(_make_chunk(ns_id, doc_b, embedding=_unit_embedding(8, 0)))

        query = _unit_embedding(8, 0)
        results = await vector.search_similar(ns_id, query, filter_document_ids=[doc_a])
        assert len(results) == 1

    async def test_search_empty_namespace(self, vector: SQLiteVectorBackend):
        results = await vector.search_similar(uuid4(), _unit_embedding(8))
        assert results == []


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


class TestEntityOperations:
    async def test_create_and_exists(self, vector: SQLiteVectorBackend):
        entity = _make_entity(uuid4(), embedding=_unit_embedding(8))
        await vector.create_entity(entity)

        assert await vector.entity_exists(entity.id)
        assert not await vector.entity_exists(uuid4())

    async def test_update_entity(self, vector: SQLiteVectorBackend):
        entity = _make_entity(uuid4(), embedding=_unit_embedding(8))
        await vector.create_entity(entity)

        entity.description = "Updated description"
        await vector.update_entity(entity)

        # Verify by checking it still exists (no get_entity on vector protocol)
        assert await vector.entity_exists(entity.id)

    async def test_update_entity_embedding(self, vector: SQLiteVectorBackend):
        entity = _make_entity(uuid4())
        await vector.create_entity(entity)

        new_emb = _unit_embedding(8, 3)
        await vector.update_entity_embedding(entity.id, new_emb, "updated-model")
        assert await vector.entity_exists(entity.id)

    async def test_update_entity_embeddings_batch(self, vector: SQLiteVectorBackend):
        entities = [_make_entity(uuid4()) for _ in range(3)]
        for e in entities:
            await vector.create_entity(e)

        updates = [(e.id, _unit_embedding(8, i), "batch-model") for i, e in enumerate(entities)]
        count = await vector.update_entity_embeddings_batch(updates)
        assert count == 3

    async def test_search_similar_entities(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        entities = []
        for i in range(5):
            e = _make_entity(ns_id, embedding=_unit_embedding(16, i), name=f"Entity {i}")
            entities.append(e)
            await vector.create_entity(e)

        query = _unit_embedding(16, 0)
        results = await vector.search_similar_entities(ns_id, query, limit=3)
        assert len(results) > 0
        best_id, best_score = results[0]
        assert best_score > 0.99
        assert best_id == entities[0].id


# ---------------------------------------------------------------------------
# Full-text search
# ---------------------------------------------------------------------------


class TestFullTextSearch:
    async def test_fts_search(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        doc_id = uuid4()

        c1 = _make_chunk(ns_id, doc_id, content="Python programming language guide")
        c2 = _make_chunk(ns_id, doc_id, content="Rust systems programming")
        c3 = _make_chunk(ns_id, doc_id, content="Cooking recipes for dinner")
        await vector.create_chunks_batch([c1, c2, c3])

        results = await vector.search_fulltext(ns_id, "programming")
        assert len(results) >= 1
        contents = [chunk.content for chunk, _ in results]
        assert any("programming" in c.lower() for c in contents)

    async def test_fts_empty_result(self, vector: SQLiteVectorBackend):
        ns_id = uuid4()
        results = await vector.search_fulltext(ns_id, "nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# from_config()
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_relational_from_config_memory(self):
        class FakeConfig:
            url = "sqlite:///:memory:"
            embedding_dimension = 1536

        backend = SQLiteRelationalBackend.from_config(FakeConfig())
        assert backend._database_path == ":memory:"

    def test_relational_from_config_file(self):
        class FakeConfig:
            url = "sqlite:///tmp/test.db"
            embedding_dimension = 1536

        backend = SQLiteRelationalBackend.from_config(FakeConfig())
        assert backend._database_path == "tmp/test.db"

    def test_vector_from_config(self):
        class FakeConfig:
            url = "sqlite:///:memory:"
            embedding_dimension = 768

        backend = SQLiteVectorBackend.from_config(FakeConfig())
        assert backend._database_path == ":memory:"
        assert backend._embedding_dim == 768

    def test_relational_from_config_none_url(self):
        class FakeConfig:
            url = None
            embedding_dimension = 1536

        backend = SQLiteRelationalBackend.from_config(FakeConfig())
        assert backend._database_path == ":memory:"


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_sqlite_vector_config(self):
        from khora.config.schema import SQLiteVectorConfig

        cfg = SQLiteVectorConfig(url="sqlite:///test.db")
        assert cfg.backend == "sqlite"
        assert cfg.embedding_dimension == 1536

    def test_vector_config_discriminator(self):
        from khora.config.schema import SQLiteVectorConfig

        cfg = SQLiteVectorConfig.model_validate({"backend": "sqlite", "url": "sqlite:///test.db"})
        assert cfg.backend == "sqlite"
