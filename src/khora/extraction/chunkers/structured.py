"""Structure-aware chunker.

Respects pre-defined chunk boundaries from document metadata.
Only sub-divides segments that exceed chunk_size.
"""

from __future__ import annotations

from .base import Chunker, ChunkResult
from .semantic import SemanticChunker


class StructuredChunker(Chunker):
    """Chunker that respects caller-defined structural boundaries.

    If the document provides boundary markers (e.g., section headers),
    the chunker splits at those boundaries first. Segments that exceed
    chunk_size are sub-divided using semantic splitting (paragraph/sentence).
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        boundary_pattern: str = r"\n\n",
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._boundary_pattern = boundary_pattern
        # Fallback chunker for oversized segments
        self._fallback = SemanticChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, text: str, *, boundaries: list[tuple[int, int]] | None = None) -> list[ChunkResult]:
        """Split text respecting structural boundaries.

        Args:
            text: Text to chunk.
            boundaries: Optional list of (start_char, end_char) tuples
                        defining pre-determined segment boundaries.
                        If None, splits on boundary_pattern.

        Returns:
            List of ChunkResult with correct offsets.
        """
        if not text or not text.strip():
            return []

        # Get segments from boundaries or pattern
        if boundaries:
            segments = [
                (start, end, text[start:end])
                for start, end in boundaries
                if start < len(text)
            ]
        else:
            segments = self._split_by_pattern(text)

        results: list[ChunkResult] = []
        chunk_index = 0

        for seg_start, seg_end, seg_text in segments:
            seg_text = seg_text.strip()
            if not seg_text:
                continue

            token_count = self.count_tokens(seg_text)

            if token_count <= self.chunk_size:
                # Fits in one chunk
                results.append(
                    ChunkResult(
                        content=seg_text,
                        index=chunk_index,
                        start_char=seg_start,
                        end_char=seg_end,
                        token_count=token_count,
                    )
                )
                chunk_index += 1
            else:
                # Too large — sub-divide with semantic chunker
                sub_chunks = self._fallback.chunk(seg_text)
                for sc in sub_chunks:
                    results.append(
                        ChunkResult(
                            content=sc.content,
                            index=chunk_index,
                            start_char=seg_start + sc.start_char,
                            end_char=seg_start + sc.end_char,
                            token_count=sc.token_count,
                        )
                    )
                    chunk_index += 1

        return self.filter_empty_chunks(results)

    def _split_by_pattern(self, text: str) -> list[tuple[int, int, str]]:
        """Split text by double-newline boundaries."""
        import re

        parts = re.split(self._boundary_pattern, text)
        segments: list[tuple[int, int, str]] = []
        pos = 0
        for part in parts:
            start = text.find(part, pos)
            if start == -1:
                start = pos
            end = start + len(part)
            segments.append((start, end, part))
            pos = end
        return segments
