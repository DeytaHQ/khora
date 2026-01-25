"""Text chunking strategies for Khora Memory Lake."""

from __future__ import annotations

from .base import Chunker, ChunkResult
from .fixed import FixedChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker


def create_chunker(
    strategy: str = "semantic",
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> Chunker:
    """Create a chunker based on the specified strategy.

    Args:
        strategy: Chunking strategy (fixed, semantic, recursive)
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        **kwargs: Additional strategy-specific arguments

    Returns:
        Configured Chunker instance
    """
    if strategy == "fixed":
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "semantic":
        return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    elif strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


__all__ = [
    "Chunker",
    "ChunkResult",
    "FixedChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "create_chunker",
]
