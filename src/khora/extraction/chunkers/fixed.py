"""Fixed-size text chunker."""

from __future__ import annotations

from .base import Chunker, ChunkResult


class FixedChunker(Chunker):
    """Fixed-size chunker that splits text at token boundaries.

    Simple chunking strategy that creates chunks of approximately
    equal size with configurable overlap.
    """

    def chunk(self, text: str) -> list[ChunkResult]:
        """Split text into fixed-size chunks.

        Args:
            text: Text to chunk

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        if self._encoding is None:
            # Fallback: character-based chunking
            return self._chunk_by_chars(text)

        tokens = self._encoding.encode(text)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            # Get the chunk tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoding.decode(chunk_tokens).strip()

            # Calculate character positions
            # This is approximate for token-based chunking
            if chunk_index == 0:
                start_char = 0
            else:
                start_char = text.find(chunk_text[:50])
                if start_char == -1:
                    start_char = 0

            end_char = start_char + len(chunk_text)

            chunks.append(
                ChunkResult(
                    content=chunk_text,
                    index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=len(chunk_tokens),
                    metadata={"chunker": "fixed"},
                )
            )

            # Move start with overlap
            start = end - self.chunk_overlap if end < len(tokens) else end
            chunk_index += 1

        return self.filter_empty_chunks(chunks)

    def _chunk_by_chars(self, text: str) -> list[ChunkResult]:
        """Fallback character-based chunking."""
        # Estimate chars per token (~4)
        chars_per_chunk = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chars_per_chunk, len(text))

            # Try to break at word boundary
            if end < len(text):
                space_pos = text.rfind(" ", start + chars_per_chunk // 2, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    ChunkResult(
                        content=chunk_text,
                        index=chunk_index,
                        start_char=start,
                        end_char=end,
                        token_count=len(chunk_text) // 4,
                        metadata={"chunker": "fixed"},
                    )
                )
                chunk_index += 1

            start = end - overlap_chars if end < len(text) else end

        return chunks
