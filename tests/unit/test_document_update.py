"""Unit tests for document update pipeline (DYT-209).

Tests the source-based update detection in:
- GraphRAGEngine.remember()
- stage_document() / stage_documents_batch()
- StorageCoordinator.cleanup_document_references()
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Document, DocumentMetadata
from khora.core.models.document import DocumentStatus
from khora.memory_lake import BatchResult, RememberResult
from khora.storage.coordinator import StorageCoordinator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_document(
    *,
    namespace_id=None,
    source: str = "",
    content: str = "test content",
    checksum: str = "",
) -> Document:
    """Create a Document with sensible defaults."""
    ns_id = namespace_id or uuid4()
    cs = checksum or hashlib.sha256(content.encode("utf-8")).hexdigest()
    return Document(
        id=uuid4(),
        namespace_id=ns_id,
        content=content,
        status=DocumentStatus.COMPLETED,
        metadata=DocumentMetadata(
            source=source,
            source_type="api",
            checksum=cs,
            size_bytes=len(content.encode("utf-8")),
        ),
        chunk_count=3,
        entity_count=5,
    )


def _make_storage_mock(
    *,
    checksum_doc: Document | None = None,
    source_doc: Document | None = None,
) -> MagicMock:
    """Create a StorageCoordinator mock with common methods."""
    storage = MagicMock(spec=StorageCoordinator)
    storage.get_document_by_checksum = AsyncMock(return_value=checksum_doc)
    storage.get_document_by_source = AsyncMock(return_value=source_doc)
    storage.get_documents_by_sources = AsyncMock(return_value={})
    storage.get_documents_by_checksums = AsyncMock(return_value={})
    storage.create_document = AsyncMock(side_effect=lambda d: d)
    storage.update_document = AsyncMock(side_effect=lambda d: d)
    storage.cleanup_document_references = AsyncMock(
        return_value={
            "chunks_deleted": 3,
            "entities_updated": 2,
            "entities_deleted": 1,
            "relationships_updated": 1,
            "relationships_deleted": 0,
        }
    )
    return storage


# ===========================================================================
# GraphRAGEngine.remember() — update detection
# ===========================================================================


class TestRememberUpdateDetection:
    """Tests for update detection in GraphRAGEngine.remember()."""

    @pytest.mark.asyncio
    async def test_remember_new_doc_no_source(self) -> None:
        """Document without source always creates new (current behavior)."""
        ns_id = uuid4()
        storage = _make_storage_mock()

        with patch("khora.pipelines.flows.ingest.process_document", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"chunks": 2, "entities": 1, "relationships": 0}

            from khora.engines.graphrag.engine import GraphRAGEngine

            engine = GraphRAGEngine.__new__(GraphRAGEngine)
            engine._storage = storage
            engine._config = MagicMock()
            engine._config.llm.embedding_model = "test-model"
            engine._config.llm.extraction_model = "test-model"
            engine._config.llm.model = "test-model"
            engine._query_engine = MagicMock()
            engine._query_engine.invalidate_caches = MagicMock()
            engine._connected = True

            result = await engine.remember("new content", ns_id, source="")

            storage.create_document.assert_awaited_once()
            storage.get_document_by_source.assert_not_awaited()
            assert result.updated is False

    @pytest.mark.asyncio
    async def test_remember_duplicate_checksum_skips(self) -> None:
        """Identical content (same checksum) is always skipped."""
        ns_id = uuid4()
        existing = _make_document(namespace_id=ns_id, content="same content")
        storage = _make_storage_mock(checksum_doc=existing)

        from khora.engines.graphrag.engine import GraphRAGEngine

        engine = GraphRAGEngine.__new__(GraphRAGEngine)
        engine._storage = storage
        engine._config = MagicMock()
        engine._query_engine = MagicMock()
        engine._connected = True

        result = await engine.remember("same content", ns_id)

        assert result.document_id == existing.id
        assert result.metadata.get("duplicate") is True
        assert result.updated is False
        storage.create_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_remember_same_source_different_content_updates(self) -> None:
        """Same source + different content triggers update."""
        ns_id = uuid4()
        old_doc = _make_document(
            namespace_id=ns_id,
            source="https://example.com/doc",
            content="old content",
        )
        storage = _make_storage_mock(source_doc=old_doc)

        with patch("khora.pipelines.flows.ingest.process_document", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"chunks": 4, "entities": 2, "relationships": 1}

            from khora.engines.graphrag.engine import GraphRAGEngine

            engine = GraphRAGEngine.__new__(GraphRAGEngine)
            engine._storage = storage
            engine._config = MagicMock()
            engine._config.llm.embedding_model = "test-model"
            engine._config.llm.extraction_model = "test-model"
            engine._config.llm.model = "test-model"
            engine._query_engine = MagicMock()
            engine._query_engine.invalidate_caches = MagicMock()
            engine._connected = True

            result = await engine.remember(
                "new content",
                ns_id,
                source="https://example.com/doc",
            )

            # Should have cleaned up old doc
            storage.cleanup_document_references.assert_awaited_once_with(old_doc.id, ns_id)
            # Should have updated (not created) the document
            storage.update_document.assert_awaited_once()
            storage.create_document.assert_not_awaited()
            assert result.updated is True

    @pytest.mark.asyncio
    async def test_remember_update_reuses_document_id(self) -> None:
        """Updated document preserves the original UUID."""
        ns_id = uuid4()
        old_doc = _make_document(
            namespace_id=ns_id,
            source="file:///data/report.txt",
            content="version 1",
        )
        storage = _make_storage_mock(source_doc=old_doc)

        with patch("khora.pipelines.flows.ingest.process_document", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"chunks": 1, "entities": 0, "relationships": 0}

            from khora.engines.graphrag.engine import GraphRAGEngine

            engine = GraphRAGEngine.__new__(GraphRAGEngine)
            engine._storage = storage
            engine._config = MagicMock()
            engine._config.llm.embedding_model = "test-model"
            engine._config.llm.extraction_model = "test-model"
            engine._config.llm.model = "test-model"
            engine._query_engine = MagicMock()
            engine._query_engine.invalidate_caches = MagicMock()
            engine._connected = True

            result = await engine.remember(
                "version 2",
                ns_id,
                source="file:///data/report.txt",
            )

            # Document ID should be reused
            assert result.document_id == old_doc.id

    @pytest.mark.asyncio
    async def test_remember_allow_update_false(self) -> None:
        """When allow_update=False, always create new even with matching source."""
        ns_id = uuid4()
        storage = _make_storage_mock()

        with patch("khora.pipelines.flows.ingest.process_document", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"chunks": 1, "entities": 0, "relationships": 0}

            from khora.engines.graphrag.engine import GraphRAGEngine

            engine = GraphRAGEngine.__new__(GraphRAGEngine)
            engine._storage = storage
            engine._config = MagicMock()
            engine._config.llm.embedding_model = "test-model"
            engine._config.llm.extraction_model = "test-model"
            engine._config.llm.model = "test-model"
            engine._query_engine = MagicMock()
            engine._query_engine.invalidate_caches = MagicMock()
            engine._connected = True

            result = await engine.remember(
                "content",
                ns_id,
                source="https://example.com/doc",
                allow_update=False,
            )

            # Should NOT check for source match
            storage.get_document_by_source.assert_not_awaited()
            # Should create new document
            storage.create_document.assert_awaited_once()
            assert result.updated is False


# ===========================================================================
# stage_document() — update detection
# ===========================================================================


class TestStageDocumentUpdate:
    """Tests for source-based update detection in stage_document()."""

    @pytest.mark.asyncio
    async def test_stage_new_document(self) -> None:
        """New document (no checksum or source match) is created."""
        ns_id = uuid4()
        storage = _make_storage_mock()

        from khora.pipelines.flows.ingest import stage_document

        doc = await stage_document(
            {"content": "hello world", "source": ""},
            ns_id,
            storage,
        )

        assert doc is not None
        storage.create_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stage_unchanged_skips(self) -> None:
        """Document with matching checksum is skipped."""
        ns_id = uuid4()
        existing = _make_document(namespace_id=ns_id, content="same")
        storage = _make_storage_mock(checksum_doc=existing)

        from khora.pipelines.flows.ingest import stage_document

        doc = await stage_document(
            {"content": "same", "source": "https://example.com"},
            ns_id,
            storage,
        )

        assert doc is None

    @pytest.mark.asyncio
    async def test_stage_source_update(self) -> None:
        """Same source with different content triggers update."""
        ns_id = uuid4()
        old_doc = _make_document(
            namespace_id=ns_id,
            source="https://example.com/api",
            content="v1",
        )
        storage = _make_storage_mock(source_doc=old_doc)

        from khora.pipelines.flows.ingest import stage_document

        doc = await stage_document(
            {"content": "v2", "source": "https://example.com/api"},
            ns_id,
            storage,
        )

        assert doc is not None
        storage.cleanup_document_references.assert_awaited_once()
        storage.update_document.assert_awaited_once()
        storage.create_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stage_allow_update_false(self) -> None:
        """allow_update=False skips source check, creates new."""
        ns_id = uuid4()
        storage = _make_storage_mock()

        from khora.pipelines.flows.ingest import stage_document

        doc = await stage_document(
            {"content": "content", "source": "https://example.com"},
            ns_id,
            storage,
            allow_update=False,
        )

        assert doc is not None
        storage.get_document_by_source.assert_not_awaited()
        storage.create_document.assert_awaited_once()


# ===========================================================================
# stage_documents_batch() — batch update detection
# ===========================================================================


class TestStageDocumentsBatchUpdate:
    """Tests for batch source-based update detection."""

    @pytest.mark.asyncio
    async def test_batch_with_updates(self) -> None:
        """Batch detects source-based updates for non-deduped docs."""
        ns_id = uuid4()
        old_doc = _make_document(
            namespace_id=ns_id,
            source="https://example.com/api",
            content="old",
        )
        storage = _make_storage_mock()
        storage.get_documents_by_sources = AsyncMock(return_value={"https://example.com/api": old_doc})

        from khora.pipelines.flows.ingest import stage_documents_batch

        docs = [
            {"content": "new content", "source": "https://example.com/api"},
            {"content": "brand new", "source": ""},
        ]
        results = await stage_documents_batch(docs, ns_id, storage)

        assert len(results) == 2
        # First doc should be updated (not None)
        assert results[0] is not None
        storage.cleanup_document_references.assert_awaited_once()
        # Second doc should be created
        assert results[1] is not None


# ===========================================================================
# StorageCoordinator.cleanup_document_references()
# ===========================================================================


class TestCoordinatorCleanup:
    """Tests for StorageCoordinator.cleanup_document_references()."""

    @pytest.mark.asyncio
    async def test_cleanup_orchestrates_all_backends(self) -> None:
        """cleanup_document_references calls all cleanup methods."""
        doc_id = uuid4()
        ns_id = uuid4()

        vec = MagicMock()
        vec.delete_chunks_by_document = AsyncMock(return_value=5)
        vec.remove_document_from_entity_sources = AsyncMock(return_value=(3, 1))
        vec.remove_document_from_relationship_sources = AsyncMock(return_value=(2, 0))

        graph = MagicMock()
        graph.remove_document_from_entities = AsyncMock(return_value=([uuid4(), uuid4()], [uuid4()]))
        graph.remove_document_from_relationships = AsyncMock(return_value=(1, 1))

        coord = StorageCoordinator(vector=vec, graph=graph)
        stats = await coord.cleanup_document_references(doc_id, ns_id)

        vec.delete_chunks_by_document.assert_awaited_once_with(doc_id)
        graph.remove_document_from_entities.assert_awaited_once_with(doc_id, ns_id)
        graph.remove_document_from_relationships.assert_awaited_once_with(doc_id, ns_id)
        vec.remove_document_from_entity_sources.assert_awaited_once_with(doc_id)
        vec.remove_document_from_relationship_sources.assert_awaited_once_with(doc_id)

        assert stats["chunks_deleted"] == 5
        assert stats["entities_updated"] >= 2
        assert stats["entities_deleted"] >= 1

    @pytest.mark.asyncio
    async def test_cleanup_without_graph(self) -> None:
        """Cleanup works without graph backend (vector only)."""
        doc_id = uuid4()
        ns_id = uuid4()

        vec = MagicMock()
        vec.delete_chunks_by_document = AsyncMock(return_value=3)
        vec.remove_document_from_entity_sources = AsyncMock(return_value=(2, 1))
        vec.remove_document_from_relationship_sources = AsyncMock(return_value=(1, 0))

        coord = StorageCoordinator(vector=vec)
        stats = await coord.cleanup_document_references(doc_id, ns_id)

        assert stats["chunks_deleted"] == 3
        assert stats["entities_updated"] == 2
        assert stats["entities_deleted"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_without_any_backends(self) -> None:
        """Cleanup is a no-op without backends."""
        coord = StorageCoordinator()
        stats = await coord.cleanup_document_references(uuid4(), uuid4())

        assert stats["chunks_deleted"] == 0
        assert stats["entities_updated"] == 0


# ===========================================================================
# StorageCoordinator delegation tests
# ===========================================================================


class TestCoordinatorSourceLookup:
    """Tests for source-based lookup delegation in coordinator."""

    @pytest.mark.asyncio
    async def test_get_document_by_source(self) -> None:
        """get_document_by_source delegates to relational backend."""
        ns_id = uuid4()
        rel = MagicMock()
        rel.get_document_by_source = AsyncMock(return_value=None)
        coord = StorageCoordinator(relational=rel)

        await coord.get_document_by_source(ns_id, "https://example.com")
        rel.get_document_by_source.assert_awaited_once_with(ns_id, "https://example.com")

    @pytest.mark.asyncio
    async def test_get_documents_by_sources(self) -> None:
        """get_documents_by_sources delegates to relational backend."""
        ns_id = uuid4()
        rel = MagicMock()
        rel.get_documents_by_sources = AsyncMock(return_value={})
        coord = StorageCoordinator(relational=rel)

        await coord.get_documents_by_sources(ns_id, ["src1", "src2"])
        rel.get_documents_by_sources.assert_awaited_once_with(ns_id, ["src1", "src2"])

    @pytest.mark.asyncio
    async def test_get_document_by_source_missing_relational(self) -> None:
        """get_document_by_source raises without relational backend."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await coord.get_document_by_source(uuid4(), "src")


# ===========================================================================
# RememberResult / BatchResult — updated field
# ===========================================================================


class TestResultTypes:
    """Tests for updated field on result types."""

    def test_remember_result_updated_default_false(self) -> None:
        """RememberResult.updated defaults to False."""
        result = RememberResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        assert result.updated is False

    def test_remember_result_updated_true(self) -> None:
        """RememberResult.updated can be set to True."""
        result = RememberResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
            updated=True,
        )
        assert result.updated is True

    def test_batch_result_updated_default_zero(self) -> None:
        """BatchResult.updated defaults to 0."""
        result = BatchResult(
            total=5,
            processed=5,
            skipped=0,
            failed=0,
            chunks=10,
            entities=3,
            relationships=1,
        )
        assert result.updated == 0

    def test_batch_result_updated_count(self) -> None:
        """BatchResult.updated tracks number of updated docs."""
        result = BatchResult(
            total=5,
            processed=5,
            skipped=0,
            failed=0,
            chunks=10,
            entities=3,
            relationships=1,
            updated=2,
        )
        assert result.updated == 2
