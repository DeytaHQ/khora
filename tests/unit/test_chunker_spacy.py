"""Tests for spaCy-enhanced sentence splitting in SemanticChunker."""

from __future__ import annotations

from unittest.mock import patch

from khora.extraction.chunkers.semantic import SemanticChunker


class TestSentenceSplittingWithSpacy:
    """Tests for sentence splitting when spaCy is available."""

    def test_basic_sentences(self) -> None:
        """Basic sentence splitting works with spaCy."""
        chunker = SemanticChunker()
        sentences = chunker._split_sentences("Hello world. How are you? I am fine!")
        assert len(sentences) >= 3

    def test_abbreviations(self) -> None:
        """Abbreviations like Dr. are handled by spaCy sentencizer."""
        chunker = SemanticChunker()
        sentences = chunker._split_sentences("Dr. Smith went to Washington D.C. He met the president.")
        # With spaCy sentencizer, "Dr." should NOT be a sentence break
        assert any("Dr." in s and "Smith" in s for s in sentences)

    def test_decimal_numbers(self) -> None:
        """Decimal numbers like 3.14 are not treated as sentence breaks."""
        chunker = SemanticChunker()
        sentences = chunker._split_sentences("The value is 3.14. That is pi.")
        # "3.14" should stay in one sentence
        assert any("3.14" in s for s in sentences)

    def test_urls_preserved(self) -> None:
        """URLs with dots are not split incorrectly."""
        chunker = SemanticChunker()
        text = "Visit https://example.com for more info. It has great content."
        sentences = chunker._split_sentences(text)
        assert any("https://example.com" in s for s in sentences)

    def test_multiple_sentence_endings(self) -> None:
        """Multiple sentence-ending punctuation types are handled."""
        chunker = SemanticChunker()
        sentences = chunker._split_sentences("First sentence. Second one! Third one?")
        assert len(sentences) == 3


class TestSentenceSplittingFallback:
    """Tests for regex fallback when spaCy is not available."""

    def test_fallback_basic_sentences(self) -> None:
        """Regex fallback handles basic sentences."""
        chunker = SemanticChunker()
        with patch("khora.extraction.chunkers.semantic._HAS_SPACY", False):
            sentences = chunker._split_sentences("Hello world. How are you? Fine!")
        assert len(sentences) >= 3

    def test_fallback_splits_on_punctuation(self) -> None:
        """Regex fallback splits on sentence-ending punctuation."""
        chunker = SemanticChunker()
        with patch("khora.extraction.chunkers.semantic._HAS_SPACY", False):
            sentences = chunker._split_sentences("First sentence. Second sentence! Third?")
        assert len(sentences) == 3


class TestSpacyAvailabilityFlag:
    """Tests for the _HAS_SPACY module-level flag."""

    def test_has_spacy_is_bool(self) -> None:
        """_HAS_SPACY is a boolean."""
        import khora.extraction.chunkers.semantic as mod

        assert isinstance(mod._HAS_SPACY, bool)

    def test_spacy_method_exists(self) -> None:
        """SemanticChunker has _split_sentences_spacy method."""
        chunker = SemanticChunker()
        assert hasattr(chunker, "_split_sentences_spacy")


class TestChunkerIntegration:
    """Integration tests for chunking with spaCy-enhanced splitting."""

    def test_chunk_large_paragraph(self) -> None:
        """Large paragraphs produce valid chunks."""
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
        text = (
            "Dr. Smith published a paper on quantum computing. "
            "The paper discussed the value 3.14 in quantum mechanics. "
            "He presented it at the conference in Washington D.C. "
            "The audience was impressed by the results."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Content should be preserved
        full_content = " ".join(c.content for c in chunks)
        assert "Dr. Smith" in full_content
