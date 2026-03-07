"""Unit tests for temporal chunk writing during ingest_documents().

Verifies that ingest_documents() writes TemporalChunk records to a
temporal_store (PgVectorTemporalStore) when one is provided. This is
the fix for the bug where VectorCypher/Skeleton engines read from
khora_chunks but ingest only writes to the standard chunks table.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.engines.skeleton.backends import TemporalChunk


def _make_storage_mock() -> MagicMock:
    """Create a mock StorageCoordinator with the methods used by ingest_documents."""
    storage = MagicMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    storage.update_document = AsyncMock()
    storage.create_chunks_batch = AsyncMock()
    storage.upsert_entities_batch = AsyncMock(return_value=[])
    storage.update_entity_embeddings_batch = AsyncMock()
    storage.create_relationships_batch = AsyncMock(return_value=0)
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])
    return storage


def _make_chunk(ns_id, doc_id, content="test content", embedding=None):
    """Create a Chunk instance for testing."""
    from khora.core.models import Chunk, ChunkMetadata

    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=doc_id,
        content=content,
        metadata=ChunkMetadata(),
        embedding=embedding or [0.1, 0.2, 0.3],
        created_at=datetime.now(UTC),
    )


def _make_document_mock(doc_id, ns_id, content, metadata_custom=None):
    """Create a mock Document with the attributes process_document needs."""
    doc = MagicMock()
    doc.id = doc_id
    doc.namespace_id = ns_id
    doc.content = content
    doc.metadata = MagicMock(custom=metadata_custom or {}, title="")
    doc.created_at = datetime.now(UTC)
    doc.mark_processing = MagicMock()
    doc.mark_completed = MagicMock()
    doc.mark_failed = MagicMock()
    doc.status = "pending"
    return doc


@pytest.mark.unit
class TestIngestTemporalChunks:
    """Tests that process_document writes to temporal_store when provided."""

    @pytest.mark.asyncio
    async def test_process_document_writes_temporal_chunks_when_store_provided(self) -> None:
        """When temporal_store is passed, process_document should call
        temporal_store.create_chunks_batch with TemporalChunk objects
        containing the correct temporal metadata."""
        from khora.pipelines.flows.ingest import process_document

        ns_id = uuid4()
        doc_id = uuid4()

        # Create a document mock with temporal metadata
        metadata_custom = {
            "source_system": "slack",
            "author": "alice",
            "channel": "#general",
            "tags": ["meeting", "conference"],
            "occurred_at": "2024-06-15T10:30:00+00:00",
        }
        document = _make_document_mock(doc_id, ns_id, "Alice met Bob at the conference.", metadata_custom)

        storage = _make_storage_mock()

        # Create chunks that chunk_document will return
        chunks = [_make_chunk(ns_id, doc_id, content="Alice met Bob at the conference.")]

        # Mock temporal_store
        temporal_store = MagicMock()
        temporal_store.create_chunks_batch = AsyncMock(return_value=[])

        # Patch only the low-level pipeline tasks
        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=chunks)),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=chunks)),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([], []))),
        ):
            await process_document(
                document,
                storage,
                temporal_store=temporal_store,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        # The key assertion: temporal_store.create_chunks_batch must have been called
        temporal_store.create_chunks_batch.assert_awaited_once()

        # Verify the TemporalChunk objects passed contain correct fields
        call_args = temporal_store.create_chunks_batch.call_args
        temporal_chunks = call_args[0][0]  # first positional arg

        assert len(temporal_chunks) == 1
        tc = temporal_chunks[0]
        assert isinstance(tc, TemporalChunk)
        assert tc.namespace_id == ns_id
        assert tc.document_id == doc_id
        assert tc.source_system == "slack"
        assert tc.author == "alice"
        assert tc.channel == "#general"
        assert tc.tags == ["meeting", "conference"]
        assert tc.content == "Alice met Bob at the conference."
        assert isinstance(tc.metadata, dict)
        # Verify metadata is JSON-serializable (no UUID objects)
        import json

        json.dumps(tc.metadata)  # raises TypeError if UUIDs leak through

    @pytest.mark.asyncio
    async def test_process_document_without_temporal_store_unchanged(self) -> None:
        """When temporal_store is NOT passed (None), process_document should
        behave exactly as before -- no temporal writes happen."""
        from khora.pipelines.flows.ingest import process_document

        ns_id = uuid4()
        doc_id = uuid4()

        document = _make_document_mock(doc_id, ns_id, "Some content here.")
        storage = _make_storage_mock()
        chunks = [_make_chunk(ns_id, doc_id, content="Some content here.")]

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=chunks)),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=chunks)),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([], []))),
        ):
            result = await process_document(
                document,
                storage,
                # No temporal_store -- default None
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        assert result["chunks"] == 1
        # storage.create_chunks_batch should be called (standard path)
        storage.create_chunks_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_temporal_chunks_have_occurred_at_from_metadata(self) -> None:
        """TemporalChunks should have occurred_at populated from the document's
        custom metadata when available."""
        from khora.pipelines.flows.ingest import process_document

        ns_id = uuid4()
        doc_id = uuid4()

        metadata_custom = {
            "occurred_at": "2024-03-20T14:00:00+00:00",
            "source_system": "google_calendar",
            "author": "bob",
        }
        document = _make_document_mock(doc_id, ns_id, "Important event happened today.", metadata_custom)

        storage = _make_storage_mock()
        chunks = [_make_chunk(ns_id, doc_id, content="Important event happened today.")]

        temporal_store = MagicMock()
        temporal_store.create_chunks_batch = AsyncMock(return_value=[])

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=chunks)),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=chunks)),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([], []))),
        ):
            await process_document(
                document,
                storage,
                temporal_store=temporal_store,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        temporal_store.create_chunks_batch.assert_awaited_once()
        call_args = temporal_store.create_chunks_batch.call_args
        temporal_chunks = call_args[0][0]

        assert len(temporal_chunks) == 1
        tc = temporal_chunks[0]
        assert isinstance(tc, TemporalChunk)
        # occurred_at should be parsed from the metadata
        assert tc.occurred_at is not None
        assert tc.occurred_at.year == 2024
        assert tc.occurred_at.month == 3
        assert tc.occurred_at.day == 20
        assert tc.source_system == "google_calendar"
        assert tc.author == "bob"

    @pytest.mark.asyncio
    async def test_temporal_store_receives_chunks_from_multi_chunk_doc(self) -> None:
        """When a document produces multiple chunks, temporal_store should
        receive all of them."""
        from khora.pipelines.flows.ingest import process_document

        ns_id = uuid4()
        doc_id = uuid4()

        metadata_custom = {
            "source_system": "api",
            "author": "author_0",
        }
        document = _make_document_mock(doc_id, ns_id, "Long document content.", metadata_custom)

        storage = _make_storage_mock()

        # Create multiple chunks for the document
        chunks = [
            _make_chunk(ns_id, doc_id, content="Chunk 1 content."),
            _make_chunk(ns_id, doc_id, content="Chunk 2 content."),
            _make_chunk(ns_id, doc_id, content="Chunk 3 content."),
        ]

        temporal_store = MagicMock()
        temporal_store.create_chunks_batch = AsyncMock(return_value=[])

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=chunks)),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=chunks)),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([], []))),
        ):
            await process_document(
                document,
                storage,
                temporal_store=temporal_store,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        temporal_store.create_chunks_batch.assert_awaited_once()
        call_args = temporal_store.create_chunks_batch.call_args
        temporal_chunks = call_args[0][0]

        assert len(temporal_chunks) == 3
        for i, tc in enumerate(temporal_chunks):
            assert isinstance(tc, TemporalChunk)
            assert tc.namespace_id == ns_id
            assert tc.document_id == doc_id
            assert tc.source_system == "api"
            assert tc.content == f"Chunk {i + 1} content."
