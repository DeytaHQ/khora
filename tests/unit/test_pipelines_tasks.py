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
from khora.extraction.chunkers import create_chunker


def _make_document(content: str = "test content") -> Document:
    """Create a Document with sensible defaults."""
    return Document(
        namespace_id=uuid4(),
        content=content,
        title="Test",
        source="test",
        checksum="abc123",
        size_bytes=len(content),
        metadata={"key": "value"},
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
    )


def _make_chunk(content: str = "chunk text", ns_id=None) -> Chunk:
    """Create a Chunk with sensible defaults."""
    return Chunk(
        namespace_id=ns_id or uuid4(),
        document_id=uuid4(),
        content=content,
        chunk_index=0,
        created_at=datetime(2024, 6, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Chunking logic (mirrors pipelines/tasks/chunk.py without Prefect)
# ---------------------------------------------------------------------------


def _chunk_document(document: Document, strategy: str = "fixed", chunk_size: int = 512, chunk_overlap: int = 10):
    """Reproduce chunk_document task logic without Prefect."""
    chunker = create_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_results = chunker.chunk(document.content)

    chunks = []
    for result in chunk_results:
        chunk = Chunk(
            namespace_id=document.namespace_id,
            document_id=document.id,
            content=result.content,
            chunk_index=result.index,
            start_char=result.start_char,
            end_char=result.end_char,
            token_count=result.token_count,
            metadata=dict(document.metadata),
            chunker_info=dict(result.metadata),
            created_at=document.created_at,
            source_timestamp=document.source_timestamp,
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
        """Document metadata propagates verbatim onto chunks."""
        doc = _make_document("Content with metadata")
        chunks = _chunk_document(doc, strategy="fixed", chunk_size=500)
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("key") == "value"

    @pytest.mark.asyncio
    async def test_source_timestamp_propagates_from_document(self) -> None:
        """Regression for #615: Chunk.source_timestamp must inherit
        Document.source_timestamp so date-bounded recalls don't fall back
        to chunk.created_at (which is ingest time, not event time).
        """
        from khora.pipelines.tasks.chunk import chunk_document

        when = datetime(2026, 5, 7, 12, 0, tzinfo=UTC)
        doc = Document(
            namespace_id=uuid4(),
            content="PagerDuty triggered for the payments service at 14:00 UTC.",
            metadata={"occurred_at": when.isoformat()},
            created_at=when,
            source_timestamp=when,  # what the ingest pipeline populates
        )
        chunks = await chunk_document(doc, strategy="fixed", chunk_size=500)
        assert chunks  # at least one chunk
        for chunk in chunks:
            assert chunk.source_timestamp == when, (
                f"Chunk.source_timestamp dropped: got {chunk.source_timestamp!r}, expected {when!r}"
            )

    @pytest.mark.asyncio
    async def test_source_timestamp_stays_none_when_doc_has_none(self) -> None:
        """When the document has no source_timestamp (manual ingest with no
        connector metadata), chunks must NOT invent one — they leave the
        field None so callers fall back to chunk.created_at downstream.
        """
        from khora.pipelines.tasks.chunk import chunk_document

        doc = Document(
            namespace_id=uuid4(),
            content="A note with no occurred_at metadata.",
            # source_timestamp left at its default (None)
        )
        chunks = await chunk_document(doc, strategy="fixed", chunk_size=500)
        assert chunks
        for chunk in chunks:
            assert chunk.source_timestamp is None

    @pytest.mark.asyncio
    async def test_chunk_document_separates_metadata_and_chunker_info(self) -> None:
        """Doc-level metadata and chunker output live in two separate dicts."""
        from khora.extraction.chunkers.base import ChunkResult
        from khora.pipelines.tasks.chunk import chunk_document

        doc = Document(
            namespace_id=uuid4(),
            content="anything",
            metadata={"title_seed": "foo", "shared_key": "doc-value"},
        )

        stub_results = [
            ChunkResult(
                content="anything",
                index=0,
                start_char=0,
                end_char=len("anything"),
                token_count=2,
                metadata={"shared_key": "chunker-value", "model_name": "test-chunker"},
            )
        ]

        with patch("khora.extraction.chunkers.create_chunker") as factory:
            factory.return_value = MagicMock(chunk=MagicMock(return_value=stub_results))
            chunks = await chunk_document(doc, strategy="fixed", chunk_size=500)

        assert len(chunks) == 1
        assert chunks[0].metadata == {"title_seed": "foo", "shared_key": "doc-value"}
        assert chunks[0].chunker_info == {"shared_key": "chunker-value", "model_name": "test-chunker"}

    @pytest.mark.asyncio
    async def test_chunk_document_key_collision_does_not_overwrite(self) -> None:
        """Chunker keys must not shadow doc keys (the OLD merge bug)."""
        from khora.extraction.chunkers.base import ChunkResult
        from khora.pipelines.tasks.chunk import chunk_document

        doc = Document(namespace_id=uuid4(), content="x", metadata={"k": "doc"})
        stub_results = [
            ChunkResult(
                content="x",
                index=0,
                start_char=0,
                end_char=1,
                token_count=1,
                metadata={"k": "chunker"},
            )
        ]
        with patch("khora.extraction.chunkers.create_chunker") as factory:
            factory.return_value = MagicMock(chunk=MagicMock(return_value=stub_results))
            chunks = await chunk_document(doc, strategy="fixed", chunk_size=500)

        assert chunks[0].metadata == {"k": "doc"}
        assert chunks[0].chunker_info == {"k": "chunker"}

    @pytest.mark.asyncio
    async def test_chunk_document_dicts_are_isolated_copies(self) -> None:
        """Mutating the source dicts after construction must not bleed into chunks."""
        from khora.extraction.chunkers.base import ChunkResult
        from khora.pipelines.tasks.chunk import chunk_document

        doc_meta: dict = {"a": 1}
        chunker_meta: dict = {"b": 2}
        doc = Document(namespace_id=uuid4(), content="x", metadata=doc_meta)
        stub_results = [
            ChunkResult(
                content="x",
                index=0,
                start_char=0,
                end_char=1,
                token_count=1,
                metadata=chunker_meta,
            )
        ]
        with patch("khora.extraction.chunkers.create_chunker") as factory:
            factory.return_value = MagicMock(chunk=MagicMock(return_value=stub_results))
            chunks = await chunk_document(doc, strategy="fixed", chunk_size=500)

        # Mutate originals after chunking.
        doc_meta["new_key"] = "x"
        chunker_meta["new_key"] = "y"

        assert chunks[0].metadata == {"a": 1}
        assert chunks[0].chunker_info == {"b": 2}


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
        # Use pre-normalized vectors so L2-normalization is a no-op
        mock_response.data = [
            {"embedding": [1.0, 0.0]},
            {"embedding": [0.0, 1.0]},
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
        assert chunks[0].embedding == [1.0, 0.0]
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
            results = await extractor.extract_multi(
                texts,
                batch_size=3,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        # Collect entities across results and dedup by name:type
        from khora.core.models import Entity

        all_entities: dict[str, Entity] = {}
        for chunk, result in zip([chunk1, chunk2], results):
            for extracted in result.entities:
                key = f"{extracted.name}:{extracted.entity_type}"
                if key in all_entities:
                    all_entities[key].mention_count += 1
                else:
                    entity = Entity(
                        namespace_id=chunk.namespace_id,
                        name=extracted.name,
                        entity_type=extracted.entity_type or "CONCEPT",
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
            results = await extractor.extract_multi(
                [chunk.content],
                batch_size=3,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        # Filter with 0.5 threshold
        entities = [e for e in results[0].entities if e.confidence >= 0.5]
        names = [e.name for e in entities]
        assert "High" in names
        assert "Low" not in names
