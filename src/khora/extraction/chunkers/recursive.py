"""Recursive character text chunker."""

from __future__ import annotations

from .base import Chunker, ChunkResult


class RecursiveChunker(Chunker):
    """Recursive chunker that tries multiple separators.

    Attempts to split on increasingly smaller separators until
    chunks are within the target size. Based on LangChain's
    RecursiveCharacterTextSplitter approach.
    """

    # Default separators in order of preference
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",  # Lines
        ". ",  # Sentences
        ", ",  # Clauses
        " ",  # Words
        "",  # Characters (last resort)
    ]

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the recursive chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            separators: Custom list of separators to try
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str) -> list[ChunkResult]:
        """Split text recursively using separators.

        Args:
            text: Text to chunk

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # Split recursively
        splits = self._split_text(text, self.separators)

        # Merge small splits and create final chunks
        chunks = self._merge_splits(splits, text)

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator list."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on current separator
        if separator:
            splits = text.split(separator)
            # Re-add the separator to maintain text
            splits = [s + separator if i < len(splits) - 1 else s for i, s in enumerate(splits)]
        else:
            # Character-level split (last resort)
            return list(text)

        final_splits = []
        for split in splits:
            if not split.strip():
                continue

            split_tokens = self.count_tokens(split)

            if split_tokens <= self.chunk_size:
                final_splits.append(split)
            elif remaining_separators:
                # Recursively split further
                final_splits.extend(self._split_text(split, remaining_separators))
            else:
                # Can't split further, keep as is
                final_splits.append(split)

        return final_splits

    def _merge_splits(self, splits: list[str], original_text: str) -> list[ChunkResult]:
        """Merge small splits into chunks of target size."""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0

        for split in splits:
            split_tokens = self.count_tokens(split)

            # If adding this split keeps us under the limit
            if current_tokens + split_tokens <= self.chunk_size:
                current_chunk += split
                current_tokens += split_tokens
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(self._create_chunk_result(current_chunk.strip(), chunk_index, original_text))
                    chunk_index += 1

                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + split
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = split
                    current_tokens = split_tokens

        # Add final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk_result(current_chunk.strip(), chunk_index, original_text))

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap portion from the end of text."""
        if self._encoding is not None:
            tokens = self._encoding.encode(text)

            if len(tokens) <= self.chunk_overlap:
                return text

            overlap_tokens = tokens[-self.chunk_overlap :]
            return self._encoding.decode(overlap_tokens)

        # Character-based fallback
        overlap_chars = self.chunk_overlap * 4
        return text[-overlap_chars:] if len(text) > overlap_chars else text

    def _create_chunk_result(self, content: str, index: int, original_text: str) -> ChunkResult:
        """Create a ChunkResult with character positions."""
        # Find position in original text
        start_char = original_text.find(content)
        if start_char == -1:
            # Try with first 50 chars for partial match
            start_char = original_text.find(content[:50]) if len(content) > 50 else 0

        end_char = start_char + len(content) if start_char >= 0 else len(content)

        return ChunkResult(
            content=content,
            index=index,
            start_char=max(0, start_char),
            end_char=end_char,
            token_count=self.count_tokens(content),
        )
