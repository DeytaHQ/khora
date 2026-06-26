"""Keyword search module for Khora.

Provides BM25-based keyword search for improved recall alongside
vector search.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from khora.core.models import Chunk


# Common English stopwords
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "the",
    "this",
    "but",
    "they",
    "have",
    "had",
    "what",
    "when",
    "where",
    "who",
    "which",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "can",
    "should",
    "now",
    "i",
    "you",
    "we",
    "our",
    "your",
    "my",
    "me",
    "him",
    "her",
    "them",
    "their",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "could",
    "if",
    "then",
    "else",
    "or",
    "because",
    "until",
    "while",
    "am",
}


def tokenize(text: str, use_stemming: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Tokenize text for keyword search.

    Args:
        text: Text to tokenize
        use_stemming: Apply basic stemming
        remove_stopwords: Remove common stopwords

    Returns:
        List of tokens
    """
    # Lowercase and split into word runs. ``[^\W_]+`` is the Unicode-aware
    # equivalent of ``[a-zA-Z0-9]+``: it matches runs of letters/digits in ANY
    # script (Cyrillic, accented Latin, Greek, ...) while excluding underscore.
    # On pure-ASCII text it yields byte-identical tokens to the old pattern, so
    # English ranking is unchanged; non-Latin text — which the old ASCII-only
    # pattern dropped to zero tokens (and thus zero keyword recall) — now
    # tokenizes. CJK scripts have no inter-word spaces, so a Han/Kana run still
    # collapses to one token here; real CJK segmentation (jieba/MeCab or
    # character n-grams) is a separate, heavier follow-up.
    tokens = re.findall(r"[^\W_]+", text.lower())

    # Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    # Basic stemming (simple suffix removal)
    if use_stemming:
        tokens = [_basic_stem(t) for t in tokens]

    # Filter short tokens
    tokens = [t for t in tokens if len(t) > 2]

    return tokens


def _basic_stem(word: str) -> str:
    """Basic Porter-like stemming.

    Args:
        word: Word to stem

    Returns:
        Stemmed word
    """
    # Simple suffix removal
    suffixes = ["ing", "ed", "tion", "ness", "ment", "able", "ible", "ful", "less", "ly", "er", "est", "es", "s"]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]

    return word


@dataclass
class BM25Index:
    """BM25 index for keyword search.

    Implements the BM25 ranking function for full-text search.
    """

    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75

    # Index data
    doc_lengths: dict[str, int] = field(default_factory=dict)
    doc_freqs: dict[str, Counter] = field(default_factory=dict)  # doc_id -> term -> count
    term_doc_freqs: Counter = field(default_factory=Counter)  # term -> num_docs
    avg_doc_length: float = 0.0
    total_docs: int = 0

    # Inverted index: term -> set of doc_ids containing that term
    _inverted_index: dict[str, set[str]] = field(default_factory=dict)

    # Running sum for O(1) average length updates
    _total_length: int = 0

    # Stemming and stopwords
    use_stemming: bool = True
    remove_stopwords: bool = True

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index.

        Args:
            doc_id: Document identifier
            text: Document text
        """
        tokens = tokenize(text, self.use_stemming, self.remove_stopwords)

        self.doc_lengths[doc_id] = len(tokens)
        self.doc_freqs[doc_id] = Counter(tokens)

        # Update term document frequencies and inverted index
        for term in set(tokens):
            self.term_doc_freqs[term] += 1
            if term not in self._inverted_index:
                self._inverted_index[term] = set()
            self._inverted_index[term].add(doc_id)

        self.total_docs += 1
        self._total_length += len(tokens)
        self._update_avg_length()

    def add_documents(self, documents: list[tuple[str, str]]) -> None:
        """Add multiple documents to the index.

        Args:
            documents: List of (doc_id, text) tuples
        """
        for doc_id, text in documents:
            tokens = tokenize(text, self.use_stemming, self.remove_stopwords)
            self.doc_lengths[doc_id] = len(tokens)
            self.doc_freqs[doc_id] = Counter(tokens)
            for term in set(tokens):
                self.term_doc_freqs[term] += 1
                if term not in self._inverted_index:
                    self._inverted_index[term] = set()
                self._inverted_index[term].add(doc_id)
            self.total_docs += 1
            self._total_length += len(tokens)

        self._update_avg_length()

    def _update_avg_length(self) -> None:
        """Update average document length using running sum (O(1))."""
        if self.total_docs > 0:
            self.avg_doc_length = self._total_length / self.total_docs

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term.

        Args:
            term: Term to calculate IDF for

        Returns:
            IDF score
        """
        n = self.total_docs
        df = self.term_doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_id: str) -> float:
        """Calculate BM25 score for a query-document pair.

        Args:
            query: Query text
            doc_id: Document identifier

        Returns:
            BM25 score
        """
        if doc_id not in self.doc_freqs:
            return 0.0

        query_tokens = tokenize(query, self.use_stemming, self.remove_stopwords)
        doc_freq = self.doc_freqs[doc_id]
        doc_len = self.doc_lengths[doc_id]

        score = 0.0
        for term in query_tokens:
            if term not in doc_freq:
                continue

            tf = doc_freq[term]
            idf = self._idf(term)

            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1))
            score += idf * numerator / denominator

        return score

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search the index for relevant documents.

        Uses the inverted index to only score documents containing at
        least one query term, reducing complexity from O(D*Q) to
        O(postings*Q).

        Args:
            query: Query text
            limit: Maximum results to return
            min_score: Minimum score threshold

        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        query_tokens = tokenize(query, self.use_stemming, self.remove_stopwords)
        if not query_tokens:
            return []

        # Gather candidate doc IDs via inverted index
        candidate_doc_ids: set[str] = set()
        for term in query_tokens:
            candidate_doc_ids.update(self._inverted_index.get(term, set()))

        results = []
        for doc_id in candidate_doc_ids:
            s = self.score(query, doc_id)
            if s > min_score:
                results.append((doc_id, s))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class KeywordSearcher:
    """Keyword search for chunks.

    Provides BM25-based search over chunk content.
    """

    def __init__(
        self,
        use_stemming: bool = True,
        remove_stopwords: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize the keyword searcher.

        Args:
            use_stemming: Apply stemming
            remove_stopwords: Remove stopwords
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self._index = BM25Index(
            k1=k1,
            b=b,
            use_stemming=use_stemming,
            remove_stopwords=remove_stopwords,
        )
        self._chunks: dict[str, Chunk] = {}  # doc_id -> Chunk

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks for keyword search.

        Args:
            chunks: Chunks to index
        """
        for chunk in chunks:
            doc_id = str(chunk.id)
            self._chunks[doc_id] = chunk
            self._index.add_document(doc_id, chunk.content)

        logger.debug(f"Indexed {len(chunks)} chunks for keyword search")

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Search for chunks matching the query.

        Args:
            query: Query text
            limit: Maximum results
            min_score: Minimum BM25 score

        Returns:
            List of (chunk, score) tuples
        """
        results = self._index.search(query, limit=limit, min_score=min_score)

        chunk_results = []
        for doc_id, score in results:
            if doc_id in self._chunks:
                chunk_results.append((self._chunks[doc_id], score))

        return chunk_results

    def search_with_keywords(
        self,
        keywords: list[str],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Search using pre-extracted keywords.

        Args:
            keywords: Keywords to search for
            limit: Maximum results
            min_score: Minimum BM25 score

        Returns:
            List of (chunk, score) tuples
        """
        # Join keywords into query
        query = " ".join(keywords)
        return self.search(query, limit=limit, min_score=min_score)


async def build_keyword_index(
    chunks: list[Chunk],
    use_stemming: bool = True,
    remove_stopwords: bool = True,
) -> KeywordSearcher:
    """Build a keyword search index from chunks.

    Args:
        chunks: Chunks to index
        use_stemming: Apply stemming
        remove_stopwords: Remove stopwords

    Returns:
        KeywordSearcher instance
    """
    searcher = KeywordSearcher(
        use_stemming=use_stemming,
        remove_stopwords=remove_stopwords,
    )
    searcher.index_chunks(chunks)
    return searcher


def normalize_bm25_score(score: float, max_score: float = 10.0) -> float:
    """Normalize BM25 score to 0-1 range.

    Uses sigmoid-like normalization.

    Args:
        score: Raw BM25 score
        max_score: Expected max score for normalization

    Returns:
        Normalized score between 0 and 1
    """
    if score <= 0:
        return 0.0
    return min(1.0, score / max_score)
