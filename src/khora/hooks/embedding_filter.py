"""Embedding-based pre-filter for semantic hooks (Level 1).

Uses binary quantized embeddings for sub-microsecond pre-screening
before any LLM call. The binary gate reduces candidate filters by
~95% at a ~4% false negative rate (acceptable since this is an
optimization, not a correctness gate).

Two-stage approach:
1. Binary Hamming distance (POPCNT, ~2 CPU cycles per comparison)
2. Full cosine similarity on survivors (precise but more expensive)

Users who need zero false negatives can set similarity_threshold=0.0
to disable the embedding gate entirely.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from .models import SemanticFilter


def _to_binary(embedding: list[float] | Any) -> bytes:
    """Quantize a float embedding to a binary code via sign threshold.

    Each dimension becomes 1 bit: positive → 1, non-positive → 0.
    Result is a compact bytes object for Hamming distance via XOR.

    For 1536-dim embeddings: 1536 bits = 192 bytes (vs 6144 bytes float32).
    """
    arr = np.asarray(embedding, dtype=np.float32)
    bits = np.packbits((arr > 0).astype(np.uint8))
    return bits.tobytes()


def _hamming_distance(a: bytes, b: bytes) -> int:
    """Compute Hamming distance between two binary codes.

    Uses XOR + popcount for maximum efficiency.
    """
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def _hamming_similarity(a: bytes, b: bytes, n_bits: int) -> float:
    """Convert Hamming distance to a similarity score in [0, 1].

    1.0 = identical, 0.0 = maximally different.
    """
    if n_bits == 0:
        return 0.0
    dist = _hamming_distance(a, b)
    return 1.0 - (dist / n_bits)


def cosine_similarity(a: list[float] | Any, b: list[float] | Any) -> float:
    """Compute cosine similarity between two embeddings.

    For L2-normalized vectors (Khora's default), this equals the dot product.
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingFilterCache:
    """Manages binary-quantized filter embeddings for fast pre-screening.

    Stores binary codes for each registered filter and provides
    two-stage matching: binary Hamming gate → full cosine verification.
    """

    def __init__(self, hamming_threshold: float = 0.4) -> None:
        """Initialize the cache.

        Args:
            hamming_threshold: Minimum binary similarity to pass the
                Hamming gate. Lower = more permissive (fewer false negatives).
                Default 0.4 passes ~95% of true matches.
        """
        self._filter_embeddings: dict[str, list[float]] = {}  # filter_id → embedding
        self._filter_binary: dict[str, bytes] = {}  # filter_id → binary code
        self._filter_n_bits: dict[str, int] = {}  # filter_id → embedding dimension
        self._hamming_threshold = hamming_threshold

    def register_filter(self, filter: SemanticFilter) -> None:
        """Register a filter's embedding for fast lookup.

        Call this after embedding the filter description.
        """
        if filter.embedding is None:
            return

        fid = str(filter.id)
        n_bits = len(filter.embedding)
        self._filter_embeddings[fid] = filter.embedding
        self._filter_binary[fid] = _to_binary(filter.embedding)
        self._filter_n_bits[fid] = n_bits

        logger.debug("Registered embedding filter: {} ({} dims)", filter.name, n_bits)

    def unregister_filter(self, filter_id: str) -> None:
        """Remove a filter from the cache."""
        self._filter_embeddings.pop(filter_id, None)
        self._filter_binary.pop(filter_id, None)
        self._filter_n_bits.pop(filter_id, None)

    def passes_embedding_gate(
        self,
        entity_embedding: list[float] | Any,
        filter: SemanticFilter,
    ) -> tuple[bool, float | None]:
        """Check if an entity's embedding passes a filter's similarity gate.

        Two-stage:
        1. Binary Hamming distance (sub-µs) — fast reject
        2. Full cosine similarity — precise score for survivors

        Args:
            entity_embedding: The entity's embedding vector.
            filter: The semantic filter to check against.

        Returns:
            Tuple of (passes: bool, similarity_score: float | None).
            Score is None if rejected at the binary stage.
        """
        if entity_embedding is None or filter.embedding is None:
            return True, None  # No embedding = skip gate (don't reject)

        fid = str(filter.id)
        threshold = filter.similarity_threshold

        if threshold <= 0.0:
            return True, None  # Gate disabled

        # Dimension mismatch between the entity and this filter is not a
        # meaningful comparison (different embedding models / re-embed). Reject
        # rather than score over a misaligned prefix or wrong bit count.
        entity_n_bits = len(entity_embedding)
        filter_n_bits = self._filter_n_bits.get(fid, len(filter.embedding))
        if entity_n_bits != filter_n_bits:
            logger.warning(
                "Embedding dimension mismatch for filter {} "
                "(entity {} dims vs filter {} dims); rejecting at pre-screen",
                filter.name,
                entity_n_bits,
                filter_n_bits,
            )
            return False, None

        # Stage 1: Binary Hamming gate
        entity_binary = _to_binary(entity_embedding)
        filter_binary = self._filter_binary.get(fid)

        if filter_binary is not None and filter_n_bits > 0:
            ham_sim = _hamming_similarity(entity_binary, filter_binary, filter_n_bits)
            if ham_sim < self._hamming_threshold:
                return False, None  # Fast reject

        # Stage 2: Full cosine similarity
        filter_emb = self._filter_embeddings.get(fid)
        if filter_emb is None:
            return True, None  # No cached embedding = skip

        sim = cosine_similarity(entity_embedding, filter_emb)
        return sim >= threshold, sim
