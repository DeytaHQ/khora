"""Base chunker protocol and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tiktoken as _tiktoken

# Module-level cached encoding to avoid repeated initialization
_TIKTOKEN_ENCODING: _tiktoken.Encoding | None = None

# Minimum characters for a valid chunk — chunks shorter than this are
# likely tokenizer artifacts (whitespace, partial tokens) and are filtered.
MIN_CHUNK_CHARS = 10


def _get_tiktoken_encoding() -> _tiktoken.Encoding | None:
    """Get or create the cached tiktoken encoding."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        try:
            import tiktoken

            _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass
    return _TIKTOKEN_ENCODING


@dataclass
class ChunkResult:
    """Result of chunking a document.

    Every chunker MUST stamp ``metadata["chunker"]`` with its registered
    strategy name (``"fixed"``, ``"recursive"``, ``"semantic"``, ``"conversation"``).
    """

    content: str
    index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Cache encoding reference for this instance (avoids repeated global lookups)
        self._encoding = _get_tiktoken_encoding()

    @abstractmethod
    def chunk(self, text: str) -> list[ChunkResult]:
        """Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of ChunkResult objects
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Uses tiktoken for accurate token counting.

        Args:
            text: Text to count tokens in

        Returns:
            Token count
        """
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        # Fallback: estimate ~4 chars per token
        return len(text) // 4

    def filter_empty_chunks(self, chunks: list[ChunkResult]) -> list[ChunkResult]:
        """Remove empty or extremely short chunks and re-index."""
        filtered = [c for c in chunks if c.content.strip() and len(c.content.strip()) >= MIN_CHUNK_CHARS]
        for i, c in enumerate(filtered):
            c.index = i
        return filtered
