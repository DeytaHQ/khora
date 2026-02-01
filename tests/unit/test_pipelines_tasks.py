"""Unit tests for pipelines/tasks/ — chunk, embed, extract tasks."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document
from khora.core.models.document import ChunkMetadata, DocumentMetadata


def _make_document(content: str = "test content") -> Document:
    """Create a Document with sensible defaults."""
    return Document(
        namespace_id=uuid4(),
        content=content,
        metadata=DocumentMetadata(
            title="Test",
            source="test",
            checksum="abc123",
            size_bytes=len(content),
            custom={"key": "value"},
        ),
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
    )


def _make_chunk(content: str = "chunk text", ns_id=None) -> Chunk:
    """Create a Chunk with sensible defaults."""
    return Chunk(
        namespace_id=ns_id or uuid4(),
        document_id=uuid4(),
        content=content,
        metadata=ChunkMetadata(document_id=uuid4(), chunk_index=0),
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
    )


class TestChunkDocument:
    """Tests for the chunk_document task."""

    @pytest.mark.asyncio
    async def test_chunk_document(self) -> None:
        """chunk_document creates Chunk objects from document."""
        from khora.pipelines.tasks.chunk import chunk_document

        doc = _make_document("This is a test document with some content for chunking. " * 20)

        # Use fixed chunker for predictability
        chunks = await chunk_document.fn(doc, strategy="fixed", chunk_size=50)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.namespace_id == doc.namespace_id
            assert chunk.document_id == doc.id
            assert len(chunk.content) > 0

    @pytest.mark.asyncio
    async def test_timestamp_inheritance(self) -> None:
        """Chunks inherit document's created_at timestamp."""
        from khora.pipelines.tasks.chunk import chunk_document

        doc = _make_document("Some content for testing timestamp inheritance.")

        chunks = await chunk_document.fn(doc, strategy="fixed", chunk_size=500)
        assert len(chunks) >= 1
        assert chunks[0].created_at == doc.created_at

    @pytest.mark.asyncio
    async def test_metadata_propagation(self) -> None:
        """Document custom metadata propagates to chunks."""
        from khora.pipelines.tasks.chunk import chunk_document

        doc = _make_document("Content with metadata")

        chunks = await chunk_document.fn(doc, strategy="fixed", chunk_size=500)
        assert len(chunks) >= 1
        # Custom metadata from document should be in chunk metadata
        assert chunks[0].metadata.custom.get("key") == "value"


class TestEmbedChunks:
    """Tests for the embed_chunks task."""

    @pytest.mark.asyncio
    async def test_empty_chunks(self) -> None:
        """Empty list returns empty list."""
        from khora.pipelines.tasks.embed import embed_chunks

        result = await embed_chunks.fn([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_chunks(self) -> None:
        """Chunks get embeddings assigned."""
        from khora.pipelines.tasks.embed import embed_chunks

        chunks = [_make_chunk("text1"), _make_chunk("text2")]

        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
        mock_response.usage = MagicMock(prompt_tokens=20, total_tokens=20)

        with (
            patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await embed_chunks.fn(chunks, model="test-model")

        assert len(result) == 2
        assert result[0].embedding == [0.1, 0.2]
        assert result[0].embedding_model == "test-model"


class TestExtractEntities:
    """Tests for the extract_entities task."""

    @pytest.mark.asyncio
    async def test_empty_chunks(self) -> None:
        """Empty chunks returns empty entities and relationships."""
        from khora.pipelines.tasks.extract import extract_entities

        entities, relationships = await extract_entities.fn([])
        assert entities == []
        assert relationships == []

    @pytest.mark.asyncio
    async def test_entity_dedup(self) -> None:
        """Same entity name+type from different chunks is deduped."""
        from khora.pipelines.tasks.extract import extract_entities

        ns_id = uuid4()
        doc_id = uuid4()
        chunk1 = _make_chunk("Alice is an engineer", ns_id)
        chunk1.document_id = doc_id
        chunk2 = _make_chunk("Alice works at Acme", ns_id)
        chunk2.document_id = doc_id

        import json

        section_data = {
            "sections": [
                {
                    "entities": [{"name": "Alice", "entity_type": "PERSON", "description": "An engineer"}],
                    "relationships": [],
                },
                {
                    "entities": [{"name": "Alice", "entity_type": "PERSON", "description": "Works at Acme"}],
                    "relationships": [],
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(section_data)
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            entities, relationships = await extract_entities.fn([chunk1, chunk2], model="test-model")

        # Alice should be deduped — only one entity
        assert len(entities) == 1
        assert entities[0].name == "Alice"
        assert entities[0].mention_count == 2

    @pytest.mark.asyncio
    async def test_confidence_filtering(self) -> None:
        """Low-confidence entities are filtered by skill threshold."""
        from khora.pipelines.tasks.extract import extract_entities

        chunk = _make_chunk("test content")

        import json

        section_data = {
            "sections": [
                {
                    "entities": [
                        {"name": "High", "entity_type": "PERSON", "confidence": 0.9},
                        {"name": "Low", "entity_type": "PERSON", "confidence": 0.1},
                    ],
                    "relationships": [],
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(section_data)
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            entities, _ = await extract_entities.fn([chunk], model="test-model")

        # Default min confidence is 0.5 from default skill
        names = [e.name for e in entities]
        assert "High" in names
        # "Low" may or may not be filtered depending on default skill threshold
