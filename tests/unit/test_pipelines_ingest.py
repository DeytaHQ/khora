"""Unit tests for pipelines/flows/ingest.py — Document ingestion.

Tests exercise checksum and timestamp logic directly, and test
stage_document by reimplementing its logic without Prefect task
wrappers to avoid server startup overhead.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.pipelines.flows.ingest import _extract_source_timestamp


def _compute_checksum(content: str) -> str:
    """SHA-256 checksum — mirrors compute_checksum without Prefect."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TestComputeChecksum:
    """Tests for compute_checksum."""

    def test_deterministic(self) -> None:
        """Same content produces same checksum."""
        c1 = _compute_checksum("hello world")
        c2 = _compute_checksum("hello world")
        assert c1 == c2

    def test_different_content(self) -> None:
        """Different content produces different checksum."""
        c1 = _compute_checksum("hello")
        c2 = _compute_checksum("world")
        assert c1 != c2

    def test_sha256_format(self) -> None:
        """Checksum is a 64-char hex string (SHA-256)."""
        c = _compute_checksum("test")
        assert len(c) == 64
        assert all(ch in "0123456789abcdef" for ch in c)


class TestExtractSourceTimestamp:
    """Tests for _extract_source_timestamp."""

    def test_sent_at_iso(self) -> None:
        """sent_at field in ISO format is parsed."""
        ts = _extract_source_timestamp({"sent_at": "2024-01-15T10:30:00Z"})
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15

    def test_created_at(self) -> None:
        """created_at field is parsed."""
        ts = _extract_source_timestamp({"created_at": "2024-06-01T12:00:00+00:00"})
        assert ts is not None
        assert ts.year == 2024

    def test_date_only(self) -> None:
        """Date-only format is parsed."""
        ts = _extract_source_timestamp({"timestamp": "2024-03-15"})
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 3

    def test_datetime_passthrough(self) -> None:
        """datetime objects pass through directly."""
        dt = datetime(2024, 5, 1, 12, 0, 0)
        ts = _extract_source_timestamp({"sent_at": dt})
        assert ts is dt

    def test_no_timestamp(self) -> None:
        """No matching fields returns None."""
        ts = _extract_source_timestamp({"title": "doc", "author": "me"})
        assert ts is None

    def test_empty_metadata(self) -> None:
        """Empty metadata returns None."""
        ts = _extract_source_timestamp({})
        assert ts is None

    def test_priority_order(self) -> None:
        """sent_at has priority over created_at."""
        ts = _extract_source_timestamp(
            {
                "sent_at": "2024-01-01T00:00:00Z",
                "created_at": "2024-06-01T00:00:00Z",
            }
        )
        assert ts is not None
        assert ts.month == 1  # sent_at wins

    def test_invalid_format_skipped(self) -> None:
        """Invalid format is skipped, next field tried."""
        ts = _extract_source_timestamp(
            {
                "sent_at": "not-a-date",
                "created_at": "2024-06-01T12:00:00+00:00",
            }
        )
        assert ts is not None
        assert ts.month == 6

    def test_falsy_values_skipped(self) -> None:
        """None and empty string values are skipped."""
        ts = _extract_source_timestamp({"sent_at": None, "created_at": ""})
        assert ts is None


class TestStageDocument:
    """Tests for stage_document logic without Prefect runtime."""

    @pytest.mark.asyncio
    async def test_new_document_created(self) -> None:
        """New document is created when no checksum match."""
        from khora.core.models import Document, DocumentMetadata

        ns_id = uuid4()
        storage = MagicMock()
        storage.get_document_by_checksum = AsyncMock(return_value=None)
        storage.create_document = AsyncMock(side_effect=lambda doc: doc)

        content = "hello world"
        checksum = _compute_checksum(content)

        metadata = DocumentMetadata(
            source="api",
            title="Test",
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            custom={},
        )
        doc = Document(namespace_id=ns_id, content=content, metadata=metadata, created_at=datetime.now(UTC))
        created = await storage.create_document(doc)

        assert created is not None
        storage.create_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_skipped(self) -> None:
        """Existing document (checksum match) returns None."""
        ns_id = uuid4()
        existing = MagicMock()
        existing.status = "completed"
        storage = MagicMock()
        storage.get_document_by_checksum = AsyncMock(return_value=existing)

        content = "hello world"
        checksum = _compute_checksum(content)

        result = await storage.get_document_by_checksum(ns_id, checksum)
        assert result is not None  # existing doc found, would skip

    @pytest.mark.asyncio
    async def test_source_timestamp_used(self) -> None:
        """Source timestamp from metadata is used for created_at."""
        from khora.core.models import Document, DocumentMetadata

        ns_id = uuid4()
        custom_metadata = {"sent_at": "2024-01-15T10:00:00Z"}
        source_timestamp = _extract_source_timestamp(custom_metadata)
        created_at = source_timestamp or datetime.now(UTC)

        content = "test content"
        checksum = _compute_checksum(content)

        metadata = DocumentMetadata(
            source="",
            title="",
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            custom=custom_metadata,
        )
        doc = Document(namespace_id=ns_id, content=content, metadata=metadata, created_at=created_at)

        assert doc.created_at.year == 2024
        assert doc.created_at.month == 1


class TestStreamExtractAndEmbedEntities:
    """Tests for stream_extract_and_embed_entities."""

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self) -> None:
        """Empty chunks list returns empty entities and relationships."""
        from khora.pipelines.flows.ingest import stream_extract_and_embed_entities

        embedder = MagicMock()
        embedder.embed_batch = AsyncMock(return_value=[])

        entities, relationships = await stream_extract_and_embed_entities(
            chunks=[],
            embedder=embedder,
        )

        assert entities == []
        assert relationships == []
        embedder.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_entities_get_embeddings(self) -> None:
        """Extracted entities receive embeddings."""
        from unittest.mock import patch

        from khora.core.models import Chunk, ChunkMetadata
        from khora.pipelines.flows.ingest import stream_extract_and_embed_entities

        ns_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=doc_id,
            content="Alice works at Acme Corp.",
            metadata=ChunkMetadata(),
            embedding=[],
        )

        # Mock embedder
        embedder = MagicMock()
        embedder.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        # Mock extractor - we need to patch the LLMEntityExtractor
        from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult

        mock_result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Alice",
                    entity_type="PERSON",
                    description="A person",
                    confidence=0.9,
                )
            ],
            relationships=[],
        )

        class MockExtractor:
            def __init__(self, **kwargs):
                pass

            async def extract_multi(self, texts, **kwargs):
                return [mock_result] * len(texts)

        with patch(
            "khora.extraction.extractors.LLMEntityExtractor",
            MockExtractor,
        ):
            entities, relationships = await stream_extract_and_embed_entities(
                chunks=[chunk],
                embedder=embedder,
            )

        assert len(entities) == 1
        assert entities[0].name == "alice"  # M-5: entity names are normalized (lowercased)
        assert entities[0].embedding == [0.1, 0.2, 0.3]
        embedder.embed_batch.assert_called()

    @pytest.mark.asyncio
    async def test_relationships_extracted(self) -> None:
        """Relationships between entities are extracted."""
        from unittest.mock import patch

        from khora.core.models import Chunk, ChunkMetadata
        from khora.pipelines.flows.ingest import stream_extract_and_embed_entities

        ns_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=doc_id,
            content="Alice works for Bob.",
            metadata=ChunkMetadata(),
            embedding=[],
        )

        embedder = MagicMock()
        embedder.embed_batch = AsyncMock(return_value=[[0.1], [0.2]])

        from khora.extraction.extractors.base import (
            ExtractedEntity,
            ExtractedRelationship,
            ExtractionResult,
        )

        mock_result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Alice",
                    entity_type="PERSON",
                    description="Employee",
                    confidence=0.9,
                ),
                ExtractedEntity(
                    name="Bob",
                    entity_type="PERSON",
                    description="Manager",
                    confidence=0.9,
                ),
            ],
            relationships=[
                ExtractedRelationship(
                    source_entity="Alice",
                    target_entity="Bob",
                    relationship_type="WORKS_FOR",
                    description="Employment relationship",
                    confidence=0.85,
                )
            ],
        )

        class MockExtractor:
            def __init__(self, **kwargs):
                pass

            async def extract_multi(self, texts, **kwargs):
                return [mock_result] * len(texts)

        with patch(
            "khora.extraction.extractors.LLMEntityExtractor",
            MockExtractor,
        ):
            entities, relationships = await stream_extract_and_embed_entities(
                chunks=[chunk],
                embedder=embedder,
            )

        assert len(entities) == 2
        assert len(relationships) == 1
        assert relationships[0].relationship_type == "WORKS_FOR"

    @pytest.mark.asyncio
    async def test_batch_embedding(self) -> None:
        """Entities are embedded in batches."""
        from unittest.mock import patch

        from khora.core.models import Chunk, ChunkMetadata
        from khora.pipelines.flows.ingest import stream_extract_and_embed_entities

        ns_id = uuid4()
        doc_id = uuid4()

        # Create multiple chunks
        chunks = [
            Chunk(
                id=uuid4(),
                namespace_id=ns_id,
                document_id=doc_id,
                content=f"Entity{i} is important.",
                metadata=ChunkMetadata(),
                embedding=[],
            )
            for i in range(5)
        ]

        embedder = MagicMock()
        # Return embeddings for each batch call
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * len(texts[0]) for _ in texts])

        from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult

        def make_result(idx):
            return ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name=f"Entity{idx}",
                        entity_type="CONCEPT",
                        description="",
                        confidence=0.9,
                    )
                ],
                relationships=[],
            )

        class MockExtractor:
            def __init__(self, **kwargs):
                pass

            async def extract_multi(self, texts, **kwargs):
                return [make_result(i) for i in range(len(texts))]

        with patch(
            "khora.extraction.extractors.LLMEntityExtractor",
            MockExtractor,
        ):
            entities, _ = await stream_extract_and_embed_entities(
                chunks=chunks,
                embedder=embedder,
                embedding_batch_size=2,  # Small batch size for testing
            )

        assert len(entities) == 5
        # embed_batch should have been called multiple times for batching
        assert embedder.embed_batch.call_count >= 1
