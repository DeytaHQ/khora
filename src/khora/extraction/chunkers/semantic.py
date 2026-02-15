"""Semantic text chunker that preserves meaning."""

from __future__ import annotations

import re

from .base import Chunker, ChunkResult

# Precompiled regex patterns for performance
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

try:
    import spacy

    _nlp = spacy.blank("en")
    _nlp.add_pipe("sentencizer")
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False


class SemanticChunker(Chunker):
    """Semantic chunker that respects document structure.

    Attempts to split text at natural boundaries (paragraphs, sentences)
    while staying within token limits. Preserves semantic coherence.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        respect_sentences: bool = True,
        respect_paragraphs: bool = True,
    ) -> None:
        """Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            respect_sentences: Try to break at sentence boundaries
            respect_paragraphs: Try to break at paragraph boundaries
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs

    def chunk(self, text: str) -> list[ChunkResult]:
        """Split text into semantic chunks.

        Args:
            text: Text to chunk

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # First split into paragraphs
        if self.respect_paragraphs:
            paragraphs = self._split_paragraphs(text)
        else:
            paragraphs = [text]

        # Build chunks from paragraphs
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        current_start = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If paragraph fits in current chunk
            if current_tokens + para_tokens <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(self._create_chunk_result(current_chunk, chunk_index, current_start, text))
                    chunk_index += 1

                # Handle large paragraphs
                if para_tokens > self.chunk_size:
                    # Split large paragraph into sentences
                    sub_chunks = self._split_large_paragraph(para)
                    for sub_chunk in sub_chunks:
                        chunks.append(self._create_chunk_result(sub_chunk, chunk_index, current_start, text))
                        chunk_index += 1
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = para
                    current_tokens = para_tokens

                current_start = text.find(current_chunk) if current_chunk else current_start

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk_result(current_chunk, chunk_index, current_start, text))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines or multiple newlines
        paragraphs = _PARAGRAPH_SPLIT.split(text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_large_paragraph(self, para: str) -> list[str]:
        """Split a large paragraph that exceeds chunk size."""
        if not self.respect_sentences:
            # Fall back to fixed chunking
            return self._fixed_split(para)

        # Split into sentences
        sentences = self._split_sentences(para)

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens <= self.chunk_size:
                current_chunk = (current_chunk + " " + sentence).strip()
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # Handle sentences larger than chunk_size
                if sentence_tokens > self.chunk_size:
                    chunks.extend(self._fixed_split(sentence))
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Uses spaCy's sentencizer when available for better handling of
        sentence boundaries. Falls back to regex splitting when spaCy
        is not installed.
        """
        if _HAS_SPACY:
            return self._split_sentences_spacy(text)
        # Regex fallback
        sentences = _SENTENCE_ENDINGS.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_sentences_spacy(self, text: str) -> list[str]:
        """Split text into sentences using spaCy's sentencizer."""
        doc = _nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _fixed_split(self, text: str) -> list[str]:
        """Fixed-size split for text that can't be split semantically."""
        if self._encoding is not None:
            tokens = self._encoding.encode(text)

            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunks.append(self._encoding.decode(chunk_tokens))
                start = end - self.chunk_overlap if end < len(tokens) else end

            return chunks

        # Character-based fallback
        chars_per_chunk = self.chunk_size * 4
        return [text[i : i + chars_per_chunk] for i in range(0, len(text), chars_per_chunk - self.chunk_overlap * 4)]

    def _create_chunk_result(self, content: str, index: int, hint_start: int, full_text: str) -> ChunkResult:
        """Create a ChunkResult with proper character positions."""
        # Find actual position in text
        start_char = full_text.find(content, max(0, hint_start - 100))
        if start_char == -1:
            start_char = hint_start

        end_char = start_char + len(content)

        return ChunkResult(
            content=content,
            index=index,
            start_char=start_char,
            end_char=end_char,
            token_count=self.count_tokens(content),
        )
