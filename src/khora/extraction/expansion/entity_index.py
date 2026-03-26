"""Fast entity lookup and candidate blocking for entity resolution.

Provides an in-memory index that grows during ingestion, enabling O(1)
exact dedup and O(k) fuzzy/embedding candidate retrieval via token blocking.

Uses numpy (cosine similarity) and rapidfuzz (Levenshtein) when available,
with pure-Python fallbacks via ``khora._accel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from khora._accel import batch_dot_product, batch_levenshtein, normalize_entity_name

if TYPE_CHECKING:
    from khora.core.models import Entity


def _tokenize(name: str) -> set[str]:
    """Split a name into tokens for blocking.

    Produces lowercase alphanumeric tokens of length >= 2.
    """
    normalized = normalize_entity_name(name)
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
        key = (normalize_entity_name(entity.name), entity.entity_type)

        existing = self._exact.get(key)
        if existing is not None:
            return existing

        # Insert into all indices
        self._exact[key] = entity
        self._by_id[entity.id] = entity

        type_str = entity.entity_type
        self._type.setdefault(type_str, []).append(entity)

        for token in _tokenize(entity.name):
            self._token.setdefault(token, set()).add(entity.id)

        return None

    def get(self, entity_id: UUID) -> Entity | None:
        """Look up an entity by ID."""
        return self._by_id.get(entity_id)

    def get_by_name(self, name: str, entity_type: str) -> Entity | None:
        """Look up an entity by (name, type) — exact match."""
        return self._exact.get((normalize_entity_name(name), entity_type))

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, entity_id: UUID) -> bool:
        return entity_id in self._by_id

    # ------------------------------------------------------------------
    # Candidate retrieval (for post-ingestion resolution)
    # ------------------------------------------------------------------

    def find_candidates_fast(
        self,
        entity: Entity,
        max_candidates: int = 20,
    ) -> list[Entity]:
        """Find merge candidates using multi-level blocking.

        Uses a hierarchical blocking strategy to reduce candidate sets from
        O(n^2) to O(n + k*log(c)) where c is the average cluster size:

        1. Same entity_type (required) - Primary partition
        2. Same first letter of normalized name (optional speedup) - Secondary partition
        3. Token overlap (existing logic) - Tertiary filter

        This approach creates smaller candidate pools by leveraging the natural
        clustering of entities by type and name prefix, significantly reducing
        the number of expensive similarity comparisons needed.

        Args:
            entity: Query entity to find candidates for.
            max_candidates: Maximum number of candidates to return.

        Returns:
            List of candidate entities, prioritized by blocking level match.
        """
        type_str = entity.entity_type
        normalized_name = normalize_entity_name(entity.name)

        # Fast path: no entities of this type
        if type_str not in self._type:
            return []

        type_entities = self._type[type_str]

        # Extract first letter for secondary blocking
        first_letter = normalized_name[0] if normalized_name else ""

        # Level 1 + 2: Same type AND same first letter (highest priority)
        first_letter_matches: list[Entity] = []

        # Level 1 only: Same type but different first letter
        type_only_matches: list[Entity] = []

        for candidate in type_entities:
            if candidate.id == entity.id:
                continue
            candidate_normalized = normalize_entity_name(candidate.name)
            # Skip exact matches (already handled by add())
            if candidate_normalized == normalized_name:
                continue

            if first_letter and candidate_normalized and candidate_normalized[0] == first_letter:
                first_letter_matches.append(candidate)
            else:
                type_only_matches.append(candidate)

        # Level 3: Apply token overlap filter to prioritize within each group
        tokens = _tokenize(entity.name)

        def token_overlap_score(candidate: Entity) -> int:
            """Count shared tokens with the query entity."""
            candidate_tokens = _tokenize(candidate.name)
            return len(tokens & candidate_tokens)

        # Sort each group by token overlap (higher is better)
        first_letter_matches.sort(key=token_overlap_score, reverse=True)
        type_only_matches.sort(key=token_overlap_score, reverse=True)

        # Combine results: first letter matches first, then type-only matches
        candidates = first_letter_matches + type_only_matches

        return candidates[:max_candidates]

    def find_fuzzy_candidates(
        self,
        entity: Entity,
        threshold: float = 0.85,
        *,
        max_candidates: int = 100,
        min_shared_tokens: int = 1,
        skip_ids: set[UUID] | None = None,
    ) -> list[tuple[Entity, float]]:
        """Find entities sharing name tokens AND same type, ranked by Levenshtein.

        Uses token blocking to generate candidates, then scores the entire
        batch in a single ``batch_levenshtein`` call (Rust/rayon parallelism).

        Args:
            entity: Query entity.
            threshold: Minimum similarity to return.
            max_candidates: Cap on candidates before scoring (by token overlap).
            min_shared_tokens: Minimum shared tokens to be a candidate.
            skip_ids: Entity IDs to exclude (e.g., already-processed).

        Returns:
            List of (candidate, similarity) pairs, highest first.
        """
        type_str = entity.entity_type
        tokens = _tokenize(entity.name)
        if not tokens:
            return []

        # Gather candidate IDs via token blocking with overlap counting
        candidate_overlap: dict[UUID, int] = {}
        for token in tokens:
            posting = self._token.get(token)
            if posting is None:
                continue
            # Skip high-frequency tokens (stopword-like, defeat blocking)
            if len(posting) > 500:
                continue
            for cid in posting:
                candidate_overlap[cid] = candidate_overlap.get(cid, 0) + 1

        # Remove self and skip_ids
        candidate_overlap.pop(entity.id, None)
        if skip_ids:
            for sid in skip_ids:
                candidate_overlap.pop(sid, None)

        # Filter by minimum shared tokens
        if min_shared_tokens > 1:
            candidate_overlap = {cid: count for cid, count in candidate_overlap.items() if count >= min_shared_tokens}

        if not candidate_overlap:
            return []

        # Filter to same type, skip exact matches, cap candidates
        normalized = normalize_entity_name(entity.name)
        valid_candidates: list[Entity] = []
        valid_names: list[str] = []

        # Sort by overlap count descending, take top max_candidates
        sorted_ids = sorted(candidate_overlap, key=candidate_overlap.__getitem__, reverse=True)
        for cid in sorted_ids:
            if len(valid_candidates) >= max_candidates:
                break
            candidate = self._by_id.get(cid)
            if candidate is None or candidate.entity_type != type_str:
                continue
            if normalize_entity_name(candidate.name) == normalized:
                continue
            valid_candidates.append(candidate)
            valid_names.append(candidate.name)

        if not valid_candidates:
            return []

        # Batch Levenshtein — single Rust call with rayon parallelism
        scored = batch_levenshtein(entity.name, valid_names, threshold)
        results: list[tuple[Entity, float]] = [(valid_candidates[idx], sim) for idx, sim in scored]

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_embedding_candidates(
        self,
        entity: Entity,
        threshold: float = 0.85,
        *,
        max_candidates: int = 100,
        skip_ids: set[UUID] | None = None,
    ) -> list[tuple[Entity, float]]:
        """Find entities sharing name tokens AND same type, ranked by cosine similarity.

        Uses token blocking to reduce the candidate set, then computes
        cosine similarity only within that set.

        Args:
            entity: Query entity (must have a non-None embedding).
            threshold: Minimum cosine similarity to return.
            max_candidates: Cap on candidates before scoring (by token overlap).
            skip_ids: Entity IDs to exclude (e.g., already-processed).

        Returns:
            List of (candidate, similarity) pairs, highest first.
        """
        if not entity.embedding:
            return []

        type_str = entity.entity_type
        tokens = _tokenize(entity.name)

        # Gather candidate IDs via token blocking with overlap counting
        candidate_overlap: dict[UUID, int] = {}
        for token in tokens:
            posting = self._token.get(token)
            if posting is None:
                continue
            if len(posting) > 500:
                continue
            for cid in posting:
                candidate_overlap[cid] = candidate_overlap.get(cid, 0) + 1

        candidate_overlap.pop(entity.id, None)
        if skip_ids:
            for sid in skip_ids:
                candidate_overlap.pop(sid, None)

        if not candidate_overlap:
            return []

        # Collect valid candidates and their embeddings, capped by token overlap
        valid_candidates: list[Entity] = []
        candidate_embeddings: list[list[float]] = []

        sorted_ids = sorted(candidate_overlap, key=candidate_overlap.__getitem__, reverse=True)
        for cid in sorted_ids:
            if len(valid_candidates) >= max_candidates:
                break
            candidate = self._by_id.get(cid)
            if candidate is None or not candidate.embedding:
                continue
            if candidate.entity_type != type_str:
                continue
            valid_candidates.append(candidate)
            candidate_embeddings.append(candidate.embedding)

        if not valid_candidates:
            return []

        # Batch dot product (embeddings are pre-normalized at ingest, so dot == cosine)
        scored = batch_dot_product(entity.embedding, candidate_embeddings, threshold=threshold)
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
