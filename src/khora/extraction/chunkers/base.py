"""Base chunker protocol and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkResult:
    """Result of chunking a document."""

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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: estimate ~4 chars per token
            return len(text) // 4
