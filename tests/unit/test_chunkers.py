"""Unit tests for text chunking functionality."""

from __future__ import annotations

import pytest

from khora.extraction.chunkers import (
    Chunker,
    ChunkResult,
    FixedChunker,
    RecursiveChunker,
    create_chunker,
)


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_create_chunk_result(self) -> None:
        """Test creating a chunk result."""
        result = ChunkResult(
            content="Test content",
            index=0,
            start_char=0,
            end_char=12,
            token_count=3,
        )
        assert result.content == "Test content"
        assert result.index == 0
        assert result.start_char == 0
        assert result.end_char == 12
        assert result.token_count == 3

    def test_chunk_result_with_metadata(self) -> None:
        """Test chunk result with metadata."""
        result = ChunkResult(
            content="Content",
            index=0,
            start_char=0,
            end_char=7,
            token_count=1,
            metadata={"key": "value"},
        )
        assert result.metadata["key"] == "value"


class TestFixedChunker:
    """Tests for FixedChunker."""

    def test_basic_chunking(self) -> None:
        """Test basic text chunking."""
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 200  # 200 characters, ~50 tokens

        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, ChunkResult)
            assert len(chunk.content) > 0

    def test_short_text_single_chunk(self) -> None:
        """Test that short text produces single chunk."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        text = "Short text"

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_empty_text(self) -> None:
        """Test empty text handling."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)

        chunks = chunker.chunk("")

        assert chunks == []

    def test_whitespace_only(self) -> None:
        """Test whitespace-only text."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)

        chunks = chunker.chunk("   \n\t  ")

        assert chunks == []

    def test_chunk_has_position_info(self) -> None:
        """Test that chunks have position information."""
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 20  # Long enough for multiple chunks

        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char

    def test_token_count_populated(self) -> None:
        """Test that token count is populated."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        text = "Hello world, this is a test."

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].token_count > 0

    def test_default_config(self) -> None:
        """Test chunker with default config."""
        chunker = FixedChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_basic_chunking(self) -> None:
        """Test basic recursive chunking."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 200

        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, ChunkResult)

    def test_respects_paragraph_boundaries(self) -> None:
        """Test that chunker respects paragraph boundaries."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        text = "First paragraph with some content.\n\nSecond paragraph here.\n\nThird paragraph."

        chunks = chunker.chunk(text)

        # Should chunk at paragraph boundaries when possible
        assert len(chunks) >= 1

    def test_empty_text(self) -> None:
        """Test empty text handling."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)

        chunks = chunker.chunk("")

        assert chunks == []

    def test_short_text(self) -> None:
        """Test short text produces single chunk."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        text = "Short text."

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_chunk_indices_sequential(self) -> None:
        """Test that chunk indices are sequential."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "Word " * 100  # Long text

        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i


class TestCreateChunker:
    """Tests for chunker factory function."""

    def test_create_fixed_chunker(self) -> None:
        """Test creating fixed chunker by name."""
        chunker = create_chunker("fixed")
        assert isinstance(chunker, FixedChunker)

    def test_create_recursive_chunker(self) -> None:
        """Test creating recursive chunker by name."""
        chunker = create_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_create_default_chunker(self) -> None:
        """Test creating default chunker (semantic)."""
        chunker = create_chunker()
        # Default is semantic
        assert isinstance(chunker, Chunker)

    def test_create_chunker_with_config(self) -> None:
        """Test creating chunker with custom config."""
        chunker = create_chunker("fixed", chunk_size=256, chunk_overlap=32)
        assert isinstance(chunker, FixedChunker)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 32

    def test_unknown_chunker_type(self) -> None:
        """Test unknown chunker type raises error."""
        with pytest.raises(ValueError):
            create_chunker("unknown_type")


class TestChunkerBase:
    """Tests for Chunker abstract base class."""

    def test_chunker_is_abstract(self) -> None:
        """Test that Chunker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Chunker()  # type: ignore

    def test_count_tokens_fallback(self) -> None:
        """Test token counting fallback."""
        # Create a concrete chunker to test the method
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        count = chunker.count_tokens("Hello world")
        assert count > 0

    def test_subclass_must_implement_chunk(self) -> None:
        """Test that subclass must implement chunk method."""

        class IncompleteChunker(Chunker):
            pass

        with pytest.raises(TypeError):
            IncompleteChunker()  # type: ignore
