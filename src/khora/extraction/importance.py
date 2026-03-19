"""Chunk importance scoring for selective extraction (KET-RAG style).

Scores chunks by importance to decide which ones warrant full LLM extraction
vs. lightweight rule-based edge creation. This can reduce LLM extraction cost
by 50%+ while maintaining graph connectivity via co-occurrence edges.
"""

from __future__ import annotations

from itertools import combinations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from khora.core.models import Chunk

# Pattern for capitalized multi-word phrases (potential entities)
# Matches sequences like "John Smith", "United Nations", "New York City"
_CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+(?:of|the|and|for|in|de|van|von)\s+)?){1,5}[A-Z][a-z]+\b")

# Pattern for single capitalized words (proper nouns)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


class ChunkImportanceScorer:
    """Score chunks by extraction importance using lightweight heuristics.

    Signals used:
    - Entity density: count of capitalized multi-word phrases (potential entities)
    - Information density: unique word ratio (type-token ratio)
    - Position: first and last chunks in a document are typically more important
    - Length: very short chunks are low-value; medium-length chunks are ideal
    """

    def __init__(
        self,
        *,
        entity_density_weight: float = 0.35,
        information_density_weight: float = 0.25,
        position_weight: float = 0.20,
        length_weight: float = 0.20,
    ) -> None:
        self.entity_density_weight = entity_density_weight
        self.information_density_weight = information_density_weight
        self.position_weight = position_weight
        self.length_weight = length_weight

    def score_chunks(
        self,
        chunks: list[Chunk],
        *,
        document_position: bool = True,
    ) -> list[tuple[Chunk, float]]:
        """Score chunks by extraction importance.

        Returns (chunk, score) pairs sorted by score descending.
        """
        if not chunks:
            return []

        scored: list[tuple[Chunk, float]] = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            score = self._score_single(chunk, index=i, total=total_chunks, use_position=document_position)
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def select_for_extraction(
        self,
        chunks: list[Chunk],
        *,
        ratio: float = 0.7,
        min_score: float = 0.2,
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Split chunks into (extract_with_llm, lightweight_only) groups.

        Args:
            chunks: Chunks to classify.
            ratio: Fraction of chunks to send to LLM extraction (top-K by score).
            min_score: Always include chunks scoring above this threshold.

        Returns:
            Tuple of (llm_chunks, lightweight_chunks).
        """
        if not chunks:
            return [], []

        scored = self.score_chunks(chunks)

        # Determine the cutoff: take top ratio% or all above min_score
        k = max(1, int(len(scored) * ratio))

        llm_chunks: list[Chunk] = []
        lightweight_chunks: list[Chunk] = []

        for idx, (chunk, score) in enumerate(scored):
            if idx < k or score >= min_score:
                llm_chunks.append(chunk)
            else:
                lightweight_chunks.append(chunk)

        return llm_chunks, lightweight_chunks

    def _score_single(
        self,
        chunk: Chunk,
        *,
        index: int,
        total: int,
        use_position: bool,
    ) -> float:
        """Compute importance score for a single chunk (0-1 scale)."""
        text = chunk.content

        # --- Entity density signal ---
        phrases = _CAPITALIZED_PHRASE_RE.findall(text)
        proper_nouns = _PROPER_NOUN_RE.findall(text)
        entity_count = len(phrases) + len(proper_nouns) * 0.5
        # Normalize: 0 entities -> 0, 10+ entities -> 1.0
        entity_density = min(entity_count / 10.0, 1.0)

        # --- Information density (type-token ratio) ---
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
        else:
            unique_ratio = 0.0
        # TTR is typically 0.3-0.8; normalize to 0-1
        information_density = min(max((unique_ratio - 0.2) / 0.6, 0.0), 1.0)

        # --- Position signal ---
        if use_position and total > 1:
            # First and last chunks get highest position score
            if index == 0 or index == total - 1:
                position_score = 1.0
            elif index == 1 or index == total - 2:
                position_score = 0.6
            else:
                position_score = 0.3
        else:
            position_score = 0.5  # Neutral when position is disabled or single chunk

        # --- Length signal ---
        word_count = len(words)
        if word_count < 20:
            # Very short — likely low value
            length_score = 0.1
        elif word_count < 50:
            length_score = 0.4
        elif word_count <= 300:
            # Sweet spot
            length_score = 1.0
        elif word_count <= 500:
            length_score = 0.7
        else:
            # Very long — still valuable but may be noisy
            length_score = 0.5

        # Weighted combination
        score = (
            self.entity_density_weight * entity_density
            + self.information_density_weight * information_density
            + self.position_weight * position_score
            + self.length_weight * length_score
        )

        return round(min(max(score, 0.0), 1.0), 4)


def extract_capitalized_phrases(text: str) -> list[str]:
    """Extract capitalized phrases from text as entity candidates.

    Returns deduplicated list of potential entity names.
    """
    multi_word = _CAPITALIZED_PHRASE_RE.findall(text)
    single_word = _PROPER_NOUN_RE.findall(text)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for phrase in multi_word + single_word:
        normalized = phrase.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def extract_lightweight_edges(chunk: Chunk) -> list[tuple[str, str, str]]:
    """Extract (entity1, relationship_type, entity2) triples using rules only.

    Creates CO_OCCURS_WITH edges between capitalized phrases found in the
    same chunk. This provides basic graph connectivity without LLM calls.
    """
    entities = extract_capitalized_phrases(chunk.content)

    if len(entities) < 2:
        return []

    # Create co-occurrence edges for all entity pairs in the chunk
    edges: list[tuple[str, str, str]] = []
    for e1, e2 in combinations(entities, 2):
        edges.append((e1, "CO_OCCURS_WITH", e2))

    return edges
