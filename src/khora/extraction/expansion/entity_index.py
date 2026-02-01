"""Fast entity lookup and candidate blocking for entity resolution.

Provides an in-memory index that grows during ingestion, enabling O(1)
exact dedup and O(k) fuzzy/embedding candidate retrieval via token blocking.

No external dependencies beyond the standard library (uses numpy only if
available for faster cosine similarity).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from khora.core.models import Entity


def _normalize_name(name: str) -> str:
    """Normalize an entity name for exact matching."""
    return name.lower().strip()


def _entity_type_str(entity: Entity) -> str:
    """Get entity type as a plain string."""
    et = entity.entity_type
    return et.value if hasattr(et, "value") else str(et)


def _tokenize(name: str) -> set[str]:
    """Split a name into tokens for blocking.

    Produces lowercase alphanumeric tokens of length >= 2.
    """
    normalized = _normalize_name(name)
    tokens: set[str] = set()
    for token in normalized.split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if len(cleaned) >= 2:
            tokens.add(cleaned)
    return tokens


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def _levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1.0 = identical)."""
    a, b = s1.lower(), s2.lower()
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0

    # Single-row DP for memory efficiency
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr

    distance = prev[lb]
    return 1.0 - (distance / max(la, lb))


class EntityIndex:
    """Fast entity lookup and candidate blocking for entity resolution.

    Maintains several indices for O(1) exact dedup and O(k) fuzzy/embedding
    candidate retrieval during ingestion.

    Usage::

        index = EntityIndex()
        for entity in extracted_entities:
            existing = index.add(entity)
            if existing is not None:
                existing.merge_with(entity)

    After ingestion, use ``find_fuzzy_candidates`` and
    ``find_embedding_candidates`` for cross-document resolution.
    """

    def __init__(self) -> None:
        # (normalized_name, type_str) -> entity  —  O(1) exact dedup
        self._exact: dict[tuple[str, str], Entity] = {}

        # name_token -> set of entity UUIDs  —  token blocking
        self._token: dict[str, set[UUID]] = {}

        # type_str -> list of entities
        self._type: dict[str, list[Entity]] = {}

        # entity UUID -> entity  —  master lookup
        self._by_id: dict[UUID, Entity] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, entity: Entity) -> Entity | None:
        """Add an entity to the index.

        If an entity with the same (normalized_name, type) already exists,
        returns the existing entity so the caller can merge.  Otherwise
        inserts the new entity and returns ``None``.

        Complexity: O(1) amortized.
        """
        key = (_normalize_name(entity.name), _entity_type_str(entity))

        existing = self._exact.get(key)
        if existing is not None:
            return existing

        # Insert into all indices
        self._exact[key] = entity
        self._by_id[entity.id] = entity

        type_str = _entity_type_str(entity)
        self._type.setdefault(type_str, []).append(entity)

        for token in _tokenize(entity.name):
            self._token.setdefault(token, set()).add(entity.id)

        return None

    def get(self, entity_id: UUID) -> Entity | None:
        """Look up an entity by ID."""
        return self._by_id.get(entity_id)

    def get_by_name(self, name: str, entity_type: str) -> Entity | None:
        """Look up an entity by (name, type) — exact match."""
        return self._exact.get((_normalize_name(name), entity_type))

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, entity_id: UUID) -> bool:
        return entity_id in self._by_id

    # ------------------------------------------------------------------
    # Candidate retrieval (for post-ingestion resolution)
    # ------------------------------------------------------------------

    def find_fuzzy_candidates(
        self,
        entity: Entity,
        threshold: float = 0.85,
    ) -> list[tuple[Entity, float]]:
        """Find entities sharing name tokens AND same type, ranked by Levenshtein.

        Only computes Levenshtein on the *blocked* candidate set (entities
        sharing at least one token), giving O(k) instead of O(n).

        Args:
            entity: Query entity.
            threshold: Minimum similarity to return.

        Returns:
            List of (candidate, similarity) pairs, highest first.
        """
        type_str = _entity_type_str(entity)
        tokens = _tokenize(entity.name)
        if not tokens:
            return []

        # Gather candidate IDs via token blocking
        candidate_ids: set[UUID] = set()
        for token in tokens:
            candidate_ids |= self._token.get(token, set())

        # Remove self
        candidate_ids.discard(entity.id)

        results: list[tuple[Entity, float]] = []
        normalized = _normalize_name(entity.name)

        for cid in candidate_ids:
            candidate = self._by_id.get(cid)
            if candidate is None:
                continue
            if _entity_type_str(candidate) != type_str:
                continue
            # Skip exact matches (already handled by add())
            if _normalize_name(candidate.name) == normalized:
                continue
            sim = _levenshtein_similarity(entity.name, candidate.name)
            if sim >= threshold:
                results.append((candidate, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_embedding_candidates(
        self,
        entity: Entity,
        threshold: float = 0.85,
    ) -> list[tuple[Entity, float]]:
        """Find entities sharing name tokens AND same type, ranked by cosine similarity.

        Uses token blocking to reduce the candidate set, then computes
        cosine similarity only within that set.

        Args:
            entity: Query entity (must have a non-None embedding).
            threshold: Minimum cosine similarity to return.

        Returns:
            List of (candidate, similarity) pairs, highest first.
        """
        if not entity.embedding:
            return []

        type_str = _entity_type_str(entity)
        tokens = _tokenize(entity.name)

        # Gather candidate IDs via token blocking
        candidate_ids: set[UUID] = set()
        for token in tokens:
            candidate_ids |= self._token.get(token, set())

        # Also include all same-type entities (embedding similarity
        # can catch entities with completely different names)
        for e in self._type.get(type_str, []):
            candidate_ids.add(e.id)

        candidate_ids.discard(entity.id)

        results: list[tuple[Entity, float]] = []
        for cid in candidate_ids:
            candidate = self._by_id.get(cid)
            if candidate is None or not candidate.embedding:
                continue
            if _entity_type_str(candidate) != type_str:
                continue
            sim = _cosine_similarity(entity.embedding, candidate.embedding)
            if sim >= threshold:
                results.append((candidate, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def get_all_entities(self) -> list[Entity]:
        """Return all indexed entities (unordered)."""
        return list(self._by_id.values())

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Return all entities of a given type."""
        return list(self._type.get(entity_type, []))

    def stats(self) -> dict[str, int]:
        """Return basic index statistics."""
        return {
            "total_entities": len(self._by_id),
            "exact_keys": len(self._exact),
            "token_keys": len(self._token),
            "type_groups": len(self._type),
        }
