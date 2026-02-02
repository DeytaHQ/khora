"""Fast entity lookup and candidate blocking for entity resolution.

Provides an in-memory index that grows during ingestion, enabling O(1)
exact dedup and O(k) fuzzy/embedding candidate retrieval via token blocking.

Uses numpy (cosine similarity) and rapidfuzz (Levenshtein) when available,
with pure-Python fallbacks via ``khora._accel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from khora._accel import batch_cosine_similarity, levenshtein_similarity

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
            sim = levenshtein_similarity(entity.name, candidate.name)
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

        candidate_ids.discard(entity.id)

        # Collect valid candidates and their embeddings for batch comparison
        valid_candidates: list[Entity] = []
        candidate_embeddings: list[list[float]] = []
        for cid in candidate_ids:
            candidate = self._by_id.get(cid)
            if candidate is None or not candidate.embedding:
                continue
            if _entity_type_str(candidate) != type_str:
                continue
            valid_candidates.append(candidate)
            candidate_embeddings.append(candidate.embedding)

        if not valid_candidates:
            return []

        # Batch cosine similarity (uses numpy when available)
        scored = batch_cosine_similarity(entity.embedding, candidate_embeddings, threshold=threshold)
        results: list[tuple[Entity, float]] = [(valid_candidates[idx], sim) for idx, sim in scored]

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
