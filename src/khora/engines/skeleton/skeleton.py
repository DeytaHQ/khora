"""Skeleton-based indexing for cost optimization (KET-RAG inspired).

This module implements cost-efficient indexing:
- PageRank-based core chunk selection
- Skeleton graph builder (full KG only for core chunks)
- Keyword-chunk bipartite graph for non-core chunks
- Lazy entity expansion on retrieval

Expected cost reduction: 5-10x fewer LLM extraction calls compared to full KG.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

if TYPE_CHECKING:
    from khora.engines.skeleton.backends import TemporalChunk


@dataclass
class ChunkNode:
    """Node in the chunk graph for PageRank calculation."""

    chunk_id: UUID
    content: str
    keywords: set[str] = field(default_factory=set)
    pagerank_score: float = 0.0
    is_core: bool = False


@dataclass
class KeywordNode:
    """Node representing a keyword in the bipartite graph."""

    keyword: str
    chunk_ids: set[UUID] = field(default_factory=set)
    idf_score: float = 0.0  # Inverse document frequency


class SkeletonIndexer:
    """Builds skeleton indexes for cost-efficient retrieval.

    The skeleton approach (inspired by KET-RAG) works as follows:
    1. Extract keywords from all chunks (fast, no LLM)
    2. Build keyword-chunk bipartite graph
    3. Calculate PageRank to identify core chunks
    4. Only run LLM extraction on core chunks (5-10% of total)
    5. For non-core chunks, use keyword matching for retrieval
    """

    def __init__(
        self,
        core_ratio: float = 0.1,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
    ):
        """Initialize the indexer.

        Args:
            core_ratio: Fraction of chunks to mark as core (default: 10%)
            damping_factor: PageRank damping factor
            max_iterations: Max PageRank iterations
            convergence_threshold: PageRank convergence threshold
        """
        self._core_ratio = core_ratio
        self._damping_factor = damping_factor
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold

        # Index structures
        self._chunks: dict[UUID, ChunkNode] = {}
        self._keywords: dict[str, KeywordNode] = {}

    def add_chunk(self, chunk: TemporalChunk) -> None:
        """Add a chunk to the index.

        Args:
            chunk: Temporal chunk to add
        """
        # Extract keywords using simple TF-based approach
        keywords = self._extract_keywords(chunk.content)

        chunk_node = ChunkNode(
            chunk_id=chunk.id,
            content=chunk.content,
            keywords=keywords,
        )
        self._chunks[chunk.id] = chunk_node

        # Update keyword-chunk links
        for keyword in keywords:
            if keyword not in self._keywords:
                self._keywords[keyword] = KeywordNode(keyword=keyword)
            self._keywords[keyword].chunk_ids.add(chunk.id)

    def add_chunks_batch(self, chunks: list[TemporalChunk]) -> None:
        """Add multiple chunks to the index.

        Args:
            chunks: Temporal chunks to add
        """
        for chunk in chunks:
            self.add_chunk(chunk)

    def build_skeleton(self) -> list[UUID]:
        """Build the skeleton index and return core chunk IDs.

        Returns:
            List of core chunk IDs that should have full KG extraction
        """
        if not self._chunks:
            return []

        # Calculate IDF scores for keywords
        self._calculate_idf_scores()

        # Build chunk-to-chunk edges via shared keywords
        edges = self._build_chunk_edges()

        # Run PageRank
        self._calculate_pagerank(edges)

        # Select core chunks
        core_ids = self._select_core_chunks()

        logger.info(
            f"Skeleton built: {len(core_ids)}/{len(self._chunks)} core chunks "
            f"({len(core_ids) / len(self._chunks) * 100:.1f}%)"
        )

        return core_ids

    def get_core_chunks(self) -> list[UUID]:
        """Get list of core chunk IDs.

        Returns:
            List of chunk IDs marked as core
        """
        return [cid for cid, node in self._chunks.items() if node.is_core]

    def get_chunks_by_keyword(self, keyword: str) -> list[UUID]:
        """Get chunks containing a keyword.

        Args:
            keyword: Keyword to search

        Returns:
            List of chunk IDs containing the keyword
        """
        keyword_lower = keyword.lower()
        if keyword_lower in self._keywords:
            return list(self._keywords[keyword_lower].chunk_ids)
        return []

    def get_related_chunks(
        self,
        chunk_id: UUID,
        *,
        limit: int = 10,
    ) -> list[tuple[UUID, float]]:
        """Get chunks related to a given chunk via keyword overlap.

        Args:
            chunk_id: Source chunk
            limit: Maximum number of related chunks

        Returns:
            List of (chunk_id, relevance_score) tuples
        """
        if chunk_id not in self._chunks:
            return []

        source_node = self._chunks[chunk_id]
        if not source_node.keywords:
            return []

        # Calculate Jaccard-like similarity with IDF weighting
        scores: dict[UUID, float] = defaultdict(float)

        for keyword in source_node.keywords:
            if keyword not in self._keywords:
                continue

            keyword_node = self._keywords[keyword]
            idf = keyword_node.idf_score

            for other_id in keyword_node.chunk_ids:
                if other_id != chunk_id:
                    scores[other_id] += idf

        # Normalize and sort
        if not scores:
            return []

        max_score = max(scores.values())
        normalized = [(cid, score / max_score) for cid, score in scores.items()]
        normalized.sort(key=lambda x: x[1], reverse=True)

        return normalized[:limit]

    def search_by_keywords(
        self,
        keywords: list[str],
        *,
        limit: int = 10,
    ) -> list[tuple[UUID, float]]:
        """Search chunks by multiple keywords.

        Args:
            keywords: Keywords to search
            limit: Maximum results

        Returns:
            List of (chunk_id, relevance_score) tuples
        """
        scores: dict[UUID, float] = defaultdict(float)

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self._keywords:
                continue

            keyword_node = self._keywords[keyword_lower]
            idf = keyword_node.idf_score

            for chunk_id in keyword_node.chunk_ids:
                scores[chunk_id] += idf

        if not scores:
            return []

        # Sort by score
        results = [(cid, score) for cid, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def get_pagerank_score(self, chunk_id: UUID) -> float:
        """Get PageRank score for a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            PageRank score (0.0 if not found)
        """
        if chunk_id in self._chunks:
            return self._chunks[chunk_id].pagerank_score
        return 0.0

    def is_core_chunk(self, chunk_id: UUID) -> bool:
        """Check if a chunk is marked as core.

        Args:
            chunk_id: Chunk ID

        Returns:
            True if chunk is core
        """
        if chunk_id in self._chunks:
            return self._chunks[chunk_id].is_core
        return False

    # =========================================================================
    # Private methods
    # =========================================================================

    def _extract_keywords(self, content: str) -> set[str]:
        """Extract keywords from content using simple TF approach.

        Uses stopword filtering and basic normalization.
        """
        # Simple stopword list (expand as needed)
        stopwords = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "he",
            "she",
            "they",
            "them",
            "his",
            "her",
            "their",
            "we",
            "our",
            "you",
            "your",
            "i",
            "me",
            "my",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
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
        }

        # Tokenize and normalize
        import re

        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
        keywords = {w for w in words if w not in stopwords}

        return keywords

    def _calculate_idf_scores(self) -> None:
        """Calculate IDF scores for all keywords."""
        n_docs = len(self._chunks)
        if n_docs == 0:
            return

        for keyword_node in self._keywords.values():
            df = len(keyword_node.chunk_ids)
            keyword_node.idf_score = math.log(n_docs / (1 + df)) + 1

    def _build_chunk_edges(self) -> dict[UUID, list[tuple[UUID, float]]]:
        """Build weighted edges between chunks via shared keywords.

        Returns:
            Dict mapping chunk_id -> list of (neighbor_id, weight)
        """
        edges: dict[UUID, list[tuple[UUID, float]]] = defaultdict(list)

        # For each keyword, connect all chunks that share it
        for keyword_node in self._keywords.values():
            chunk_list = list(keyword_node.chunk_ids)
            weight = keyword_node.idf_score

            for i, cid1 in enumerate(chunk_list):
                for cid2 in chunk_list[i + 1 :]:
                    edges[cid1].append((cid2, weight))
                    edges[cid2].append((cid1, weight))

        return edges

    def _calculate_pagerank(
        self,
        edges: dict[UUID, list[tuple[UUID, float]]],
    ) -> None:
        """Calculate PageRank scores for all chunks."""
        if not self._chunks:
            return

        n = len(self._chunks)
        chunk_ids = list(self._chunks.keys())

        # Initialize scores
        scores = {cid: 1.0 / n for cid in chunk_ids}

        # Calculate out-degree (sum of weights)
        out_degree: dict[UUID, float] = {}
        for cid in chunk_ids:
            if cid in edges:
                out_degree[cid] = sum(w for _, w in edges[cid])
            else:
                out_degree[cid] = 0.0

        # Iterative PageRank
        d = self._damping_factor
        base = (1 - d) / n

        for iteration in range(self._max_iterations):
            new_scores: dict[UUID, float] = {}
            diff = 0.0

            for cid in chunk_ids:
                # Sum contributions from neighbors
                contrib = 0.0
                if cid in edges:
                    for neighbor_id, weight in edges[cid]:
                        if out_degree[neighbor_id] > 0:
                            contrib += scores[neighbor_id] * weight / out_degree[neighbor_id]

                new_score = base + d * contrib
                diff += abs(new_score - scores[cid])
                new_scores[cid] = new_score

            scores = new_scores

            if diff < self._convergence_threshold:
                logger.debug(f"PageRank converged after {iteration + 1} iterations")
                break

        # Store scores
        for cid, score in scores.items():
            self._chunks[cid].pagerank_score = score

    def _select_core_chunks(self) -> list[UUID]:
        """Select core chunks based on PageRank scores.

        Returns:
            List of core chunk IDs
        """
        if not self._chunks:
            return []

        # Sort by PageRank score
        sorted_chunks = sorted(
            self._chunks.items(),
            key=lambda x: x[1].pagerank_score,
            reverse=True,
        )

        # Select top N%
        n_core = max(1, int(len(sorted_chunks) * self._core_ratio))
        core_ids = []

        for i, (cid, node) in enumerate(sorted_chunks):
            if i < n_core:
                node.is_core = True
                core_ids.append(cid)
            else:
                node.is_core = False

        return core_ids


class LazyEntityExpander:
    """Lazy entity expansion for non-core chunks.

    Instead of running LLM extraction on all chunks upfront, this class
    provides on-demand extraction when a non-core chunk is retrieved.
    """

    def __init__(
        self,
        skeleton_indexer: SkeletonIndexer,
        extraction_model: str = "gpt-4o-mini",
    ):
        """Initialize the expander.

        Args:
            skeleton_indexer: Skeleton indexer for core/non-core classification
            extraction_model: LLM model for entity extraction
        """
        self._skeleton = skeleton_indexer
        self._extraction_model = extraction_model
        self._expanded_chunks: set[UUID] = set()

    async def maybe_expand(
        self,
        chunk_id: UUID,
        chunk_content: str,
    ) -> dict[str, Any] | None:
        """Maybe expand a chunk with entity extraction.

        Only extracts if the chunk is not already expanded.

        Args:
            chunk_id: Chunk ID
            chunk_content: Chunk content

        Returns:
            Extraction result or None if already expanded/core
        """
        # Skip if already expanded
        if chunk_id in self._expanded_chunks:
            return None

        # Skip if core (should already have extraction)
        if self._skeleton.is_core_chunk(chunk_id):
            return None

        # Run extraction (simplified for now)
        # In production, this would call the full extraction pipeline
        logger.debug(f"Lazy expanding non-core chunk {chunk_id}")
        self._expanded_chunks.add(chunk_id)

        # Placeholder - return keywords as pseudo-entities
        keywords = self._skeleton._extract_keywords(chunk_content)
        return {
            "entities": [{"name": kw, "type": "KEYWORD"} for kw in list(keywords)[:10]],
            "relationships": [],
        }


__all__ = ["ChunkNode", "KeywordNode", "LazyEntityExpander", "SkeletonIndexer"]
