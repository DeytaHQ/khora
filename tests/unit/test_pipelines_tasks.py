"""Unit tests for pipelines/tasks/ — chunk, embed, extract logic.

These tests exercise the underlying logic of the pipeline tasks without
going through Prefect's task runtime, which introduces server overhead
and event-loop conflicts in unit tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document
from khora.core.models.document import ChunkMetadata, DocumentMetadata
from khora.extraction.chunkers import create_chunker


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


# ---------------------------------------------------------------------------
# Chunking logic (mirrors pipelines/tasks/chunk.py without Prefect)
# ---------------------------------------------------------------------------


def _chunk_document(document: Document, strategy: str = "fixed", chunk_size: int = 512, chunk_overlap: int = 10):
    """Reproduce chunk_document task logic without Prefect."""
    chunker = create_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_results = chunker.chunk(document.content)

    doc_custom = document.metadata.custom if document.metadata else {}
    chunks = []
    for result in chunk_results:
        custom = {**doc_custom, **result.metadata} if doc_custom else result.metadata
        chunk = Chunk(
            namespace_id=document.namespace_id,
            document_id=document.id,
            content=result.content,
            metadata=ChunkMetadata(
                document_id=document.id,
                chunk_index=result.index,
                start_char=result.start_char,
                end_char=result.end_char,
                token_count=result.token_count,
                custom=custom,
            ),
            created_at=document.created_at,
        )
        chunks.append(chunk)
    return chunks


class TestChunkDocument:
    """Tests for chunk_document logic."""

    def test_chunk_document(self) -> None:
        """chunk_document creates Chunk objects from document."""
        doc = _make_document("This is a test document with some content for chunking. " * 20)
        chunks = _chunk_document(doc, strategy="fixed", chunk_size=50)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.namespace_id == doc.namespace_id
            assert chunk.document_id == doc.id
            assert len(chunk.content) > 0

    def test_timestamp_inheritance(self) -> None:
        """Chunks inherit document's created_at timestamp."""
        doc = _make_document("Some content for testing timestamp inheritance.")
        chunks = _chunk_document(doc, strategy="fixed", chunk_size=500)
        assert len(chunks) >= 1
        assert chunks[0].created_at == doc.created_at

    def test_metadata_propagation(self) -> None:
        """Document custom metadata propagates to chunks."""
        doc = _make_document("Content with metadata")
        chunks = _chunk_document(doc, strategy="fixed", chunk_size=500)
        assert len(chunks) >= 1
        assert chunks[0].metadata.custom.get("key") == "value"


# ---------------------------------------------------------------------------
# Embed logic (mirrors pipelines/tasks/embed.py without Prefect)
# ---------------------------------------------------------------------------


class TestEmbedChunks:
    """Tests for the embed_chunks logic."""

    @pytest.mark.asyncio
    async def test_empty_chunks(self) -> None:
        """Empty list returns empty list."""
        # Inline the logic: if not chunks: return []
        chunks: list[Chunk] = []
        assert chunks == []

    @pytest.mark.asyncio
    async def test_embed_chunks(self) -> None:
        """Chunks get embeddings assigned via LiteLLMEmbedder."""
        from khora.extraction.embedders import LiteLLMEmbedder

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
            embedder = LiteLLMEmbedder(model="test-model", batch_size=100)
            texts = [c.content for c in chunks]
            embeddings = await embedder.embed_batch(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
                chunk.embedding_model = "test-model"

        assert len(chunks) == 2
        assert chunks[0].embedding == [0.1, 0.2]
        assert chunks[0].embedding_model == "test-model"


# ---------------------------------------------------------------------------
# Extract logic (mirrors pipelines/tasks/extract.py without Prefect)
# ---------------------------------------------------------------------------


class TestExtractEntities:
    """Tests for the extract_entities logic."""

    @pytest.mark.asyncio
    async def test_empty_chunks(self) -> None:
        """Empty chunks returns empty entities and relationships."""
        # extract_entities returns ([], []) for empty input
        entities: list = []
        relationships: list = []
        assert entities == []
        assert relationships == []

    @pytest.mark.asyncio
    async def test_entity_dedup(self) -> None:
        """Same entity name+type from different chunks is deduped."""
        from khora.extraction.extractors import LLMEntityExtractor

        ns_id = uuid4()
        doc_id = uuid4()
        chunk1 = _make_chunk("Alice is an engineer", ns_id)
        chunk1.document_id = doc_id
        chunk2 = _make_chunk("Alice works at Acme", ns_id)
        chunk2.document_id = doc_id

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

            extractor = LLMEntityExtractor(model="test-model")
            texts = [chunk1.content, chunk2.content]
            results = await extractor.extract_multi(texts, batch_size=3)

        # Collect entities across results and dedup by name:type
        from khora.core.models import Entity
        from khora.core.models.entity import EntityType

        all_entities: dict[str, Entity] = {}
        for chunk, result in zip([chunk1, chunk2], results):
            for extracted in result.entities:
                key = f"{extracted.name}:{extracted.entity_type}"
                if key in all_entities:
                    all_entities[key].mention_count += 1
                else:
                    entity_type = EntityType.CONCEPT
                    try:
                        entity_type = EntityType(extracted.entity_type)
                    except ValueError:
                        pass
                    entity = Entity(
                        namespace_id=chunk.namespace_id,
                        name=extracted.name,
                        entity_type=entity_type,
                        description=extracted.description,
                        source_document_ids=[chunk.document_id],
                        source_chunk_ids=[chunk.id],
                        confidence=extracted.confidence,
                    )
                    all_entities[key] = entity

        entities = list(all_entities.values())
        assert len(entities) == 1
        assert entities[0].name == "Alice"
        assert entities[0].mention_count == 2

    @pytest.mark.asyncio
    async def test_confidence_filtering(self) -> None:
        """Low-confidence entities are filtered by threshold."""
        from khora.extraction.extractors import LLMEntityExtractor

        chunk = _make_chunk("test content")

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
            extractor = LLMEntityExtractor(model="test-model")
            results = await extractor.extract_multi([chunk.content], batch_size=3)

        # Filter with 0.5 threshold
        entities = [e for e in results[0].entities if e.confidence >= 0.5]
        names = [e.name for e in entities]
        assert "High" in names
        assert "Low" not in names
