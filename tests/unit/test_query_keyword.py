"""Unit tests for query/keyword.py — BM25 keyword search."""

from __future__ import annotations

import re
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from khora.query.keyword import (
    BM25Index,
    KeywordSearcher,
    _basic_stem,
    build_keyword_index,
    normalize_bm25_score,
    tokenize,
)


class TestTokenize:
    """Tests for the tokenize function."""

    def test_basic(self) -> None:
        """Basic tokenization splits and lowercases."""
        tokens = tokenize("Hello World Programming", use_stemming=False, remove_stopwords=False)
        assert "hello" in tokens
        assert "world" in tokens

    def test_stopwords_removed(self) -> None:
        """Stopwords are removed when enabled."""
        tokens = tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens

    def test_stopwords_kept(self) -> None:
        """Stopwords kept when disabled."""
        tokens = tokenize("the cat is on the mat", remove_stopwords=False, use_stemming=False)
        assert "the" in tokens

    def test_stemming_applied(self) -> None:
        """Stemming reduces words to stems."""
        tokens = tokenize("running played creation", use_stemming=True, remove_stopwords=False)
        # "running" → "runn" (strip "ing"), "played" → "play" (strip "ed")
        assert "runn" in tokens or "run" in tokens
        assert "play" in tokens

    def test_short_tokens_filtered(self) -> None:
        """Tokens with 2 or fewer characters are filtered."""
        tokens = tokenize("I am at it go do", remove_stopwords=False, use_stemming=False)
        for token in tokens:
            assert len(token) > 2

    def test_numbers_included(self) -> None:
        """Numbers are tokenized."""
        tokens = tokenize("version 3000 release", remove_stopwords=False, use_stemming=False)
        assert "3000" in tokens

    def test_ascii_tokenization_unchanged(self) -> None:
        """The Unicode tokenizer is byte-identical to the old ASCII pattern on ASCII text.

        This is the safety guarantee for the multilingual switch: English ranking
        must not move. Digits and word order are preserved exactly.
        """
        text = "Marie Curie discovered radium element 1898"
        tokens = tokenize(text, use_stemming=False, remove_stopwords=False)
        assert tokens == ["marie", "curie", "discovered", "radium", "element", "1898"]
        # ...and that matches what the previous ASCII-only pattern produced.
        legacy = [t for t in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) if len(t) > 2]
        assert tokens == legacy

    def test_cyrillic_now_tokenizes(self) -> None:
        """Cyrillic text tokenizes (the old ASCII pattern returned zero tokens → zero recall)."""
        text = "Марија Кири открила радијум"
        tokens = tokenize(text, use_stemming=False, remove_stopwords=False)
        assert tokens == ["марија", "кири", "открила", "радијум"]
        # Anti-vacuity: the old ASCII-only pattern dropped all of this to nothing.
        assert re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) == []

    def test_accented_latin_kept_whole(self) -> None:
        """Accented Latin words are kept whole, not split at the accent."""
        tokens = tokenize("café Zürich naïve", use_stemming=False, remove_stopwords=False)
        assert "café" in tokens
        assert "zürich" in tokens
        # The old pattern would have split "café" into "caf" (dropping the é).
        assert "caf" not in tokens

    def test_cjk_collapses_to_single_run(self) -> None:
        """Known limitation: a CJK run has no inter-word spaces, so it stays one token.

        Still strictly better than the old ASCII pattern (which returned nothing);
        real CJK segmentation is a separate, heavier follow-up.
        """
        tokens = tokenize("玛丽居里发现了镭元素", use_stemming=False, remove_stopwords=False)
        assert tokens == ["玛丽居里发现了镭元素"]
        assert re.findall(r"\b[a-zA-Z0-9]+\b", "玛丽居里发现了镭元素") == []


class TestBasicStem:
    """Tests for the _basic_stem function."""

    def test_ing_suffix(self) -> None:
        """Strips -ing suffix."""
        assert _basic_stem("running") == "runn"

    def test_ed_suffix(self) -> None:
        """Strips -ed suffix."""
        assert _basic_stem("played") == "play"

    def test_tion_suffix(self) -> None:
        """Strips -tion suffix."""
        assert _basic_stem("creation") == "crea"

    def test_short_word_protection(self) -> None:
        """Short words are not stemmed (would become too short)."""
        # "sing" has "ing" but len("sing") = 4, suffix "ing" len 3,
        # 4 > 3+2=5 is False, so "sing" should be unchanged
        assert _basic_stem("sing") == "sing"
        assert _basic_stem("red") == "red"

    def test_no_matching_suffix(self) -> None:
        """Words without matching suffixes return unchanged."""
        assert _basic_stem("python") == "python"


class TestBM25Index:
    """Tests for BM25Index."""

    def test_add_document(self) -> None:
        """Adding a document updates index state."""
        idx = BM25Index()
        idx.add_document("doc1", "the quick brown fox")
        assert idx.total_docs == 1
        assert "doc1" in idx.doc_lengths
        assert "doc1" in idx.doc_freqs

    def test_add_documents_batch(self) -> None:
        """Batch add inserts multiple documents."""
        idx = BM25Index()
        idx.add_documents([("d1", "hello world"), ("d2", "foo bar baz")])
        assert idx.total_docs == 2

    def test_idf(self) -> None:
        """IDF scores are positive for terms that appear in some docs."""
        idx = BM25Index()
        idx.add_documents([("d1", "alpha beta"), ("d2", "gamma delta")])
        # "alpha" appears in 1 of 2 docs
        tokens = tokenize("alpha", idx.use_stemming, idx.remove_stopwords)
        if tokens:
            idf = idx._idf(tokens[0])
            assert idf > 0

    def test_idf_unknown_term(self) -> None:
        """IDF for unknown term is 0."""
        idx = BM25Index()
        idx.add_document("d1", "hello world")
        assert idx._idf("nonexistent_xyzzy") == 0.0

    def test_score(self) -> None:
        """Scoring a matching doc returns positive score."""
        idx = BM25Index()
        idx.add_document("d1", "machine learning algorithms neural networks")
        score = idx.score("machine learning", "d1")
        assert score > 0

    def test_score_unknown_doc(self) -> None:
        """Scoring an unknown doc returns 0."""
        idx = BM25Index()
        assert idx.score("query", "nonexistent") == 0.0

    def test_search_top_k(self) -> None:
        """Search returns at most k results."""
        idx = BM25Index()
        for i in range(20):
            idx.add_document(f"d{i}", f"document number {i} about machine learning")
        results = idx.search("machine learning", limit=5)
        assert len(results) <= 5

    def test_search_min_score(self) -> None:
        """Search respects min_score threshold."""
        idx = BM25Index()
        idx.add_document("d1", "machine learning algorithms")
        idx.add_document("d2", "unrelated content about cooking recipes")
        results = idx.search("machine learning", min_score=0.0)
        doc_ids = [doc_id for doc_id, _ in results]
        assert "d1" in doc_ids

    def test_search_ranking(self) -> None:
        """Results are ranked by score descending."""
        idx = BM25Index()
        idx.add_document("d1", "machine learning deep learning neural networks")
        idx.add_document("d2", "machine")
        results = idx.search("machine learning neural")
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_avg_doc_length(self) -> None:
        """Average document length is computed correctly."""
        idx = BM25Index(use_stemming=False, remove_stopwords=False)
        idx.add_document("d1", "aaa bbb ccc")  # 3 tokens
        idx.add_document("d2", "ddd eee")  # 2 tokens (after filtering short tokens)
        # Exact values depend on tokenization but avg should be positive
        assert idx.avg_doc_length > 0


class TestKeywordSearcher:
    """Tests for KeywordSearcher."""

    def test_index_chunks(self) -> None:
        """Indexing chunks populates internal state."""
        searcher = KeywordSearcher()
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.content = "machine learning algorithms"
        searcher.index_chunks([chunk])
        assert str(chunk.id) in searcher._chunks

    def test_search(self) -> None:
        """Searching indexed chunks returns results."""
        searcher = KeywordSearcher()
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.content = "machine learning algorithms neural networks deep learning"
        searcher.index_chunks([chunk])
        results = searcher.search("machine learning")
        assert len(results) >= 1
        assert results[0][0] is chunk

    def test_search_empty_index(self) -> None:
        """Searching empty index returns empty results."""
        searcher = KeywordSearcher()
        results = searcher.search("anything")
        assert results == []

    def test_search_with_keywords(self) -> None:
        """search_with_keywords joins keywords into a query."""
        searcher = KeywordSearcher()
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.content = "machine learning algorithms"
        searcher.index_chunks([chunk])
        results = searcher.search_with_keywords(["machine", "learning"])
        assert len(results) >= 1


class TestNormalizeBM25Score:
    """Tests for normalize_bm25_score."""

    def test_zero_score(self) -> None:
        """Zero score returns 0."""
        assert normalize_bm25_score(0.0) == 0.0

    def test_negative_score(self) -> None:
        """Negative score returns 0."""
        assert normalize_bm25_score(-1.0) == 0.0

    def test_partial_score(self) -> None:
        """Partial score is normalized correctly."""
        result = normalize_bm25_score(5.0, max_score=10.0)
        assert result == 0.5

    def test_max_cap(self) -> None:
        """Score capped at 1.0."""
        result = normalize_bm25_score(20.0, max_score=10.0)
        assert result == 1.0

    def test_default_max(self) -> None:
        """Default max_score is 10.0."""
        result = normalize_bm25_score(10.0)
        assert result == 1.0


class TestBuildKeywordIndex:
    """Tests for the async build_keyword_index helper."""

    @pytest.mark.asyncio
    async def test_build_and_search(self) -> None:
        """Build index from chunks and search."""
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.content = "knowledge graph entities relationships"
        searcher = await build_keyword_index([chunk])
        results = searcher.search("knowledge graph")
        assert len(results) >= 1
