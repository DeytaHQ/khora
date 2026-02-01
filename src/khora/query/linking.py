"""Entity linking module for Khora Memory Lake.

Links entity mentions from queries to existing entities in the knowledge graph.
Uses multiple strategies:
- Exact name matching
- Fuzzy string matching
- Embedding similarity
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from .understanding import EntityMention

if TYPE_CHECKING:
    from khora.core.models import Entity
    from khora.extraction.embedders import Embedder
    from khora.storage import StorageCoordinator


@dataclass
class LinkedEntity:
    """An entity mention linked to a stored entity."""

    mention: EntityMention
    entity: Entity | None = None
    match_method: str = ""  # exact, fuzzy, embedding
    match_score: float = 0.0
    candidates: list[tuple[Entity, float]] = field(default_factory=list)

    @property
    def is_linked(self) -> bool:
        """Check if the mention was successfully linked."""
        return self.entity is not None


@dataclass
class LinkingResult:
    """Result of entity linking."""

    linked_entities: list[LinkedEntity] = field(default_factory=list)
    unlinked_count: int = 0
    total_mentions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def linked_count(self) -> int:
        """Get count of successfully linked entities."""
        return sum(1 for le in self.linked_entities if le.is_linked)

    @property
    def success_rate(self) -> float:
        """Get linking success rate."""
        if self.total_mentions == 0:
            return 0.0
        return self.linked_count / self.total_mentions

    def get_linked_entity_ids(self) -> list[UUID]:
        """Get IDs of all linked entities."""
        return [le.entity.id for le in self.linked_entities if le.entity is not None]


class EntityLinker:
    """Links query entity mentions to stored entities.

    Uses multiple strategies to find matches:
    1. Exact name match (fastest, highest precision)
    2. Fuzzy string matching (handles typos, variations)
    3. Embedding similarity (semantic matching)
    """

    def __init__(
        self,
        storage: StorageCoordinator,
        embedder: Embedder | None = None,
        *,
        exact_match: bool = True,
        fuzzy_match: bool = True,
        embedding_match: bool = True,
        fuzzy_threshold: float = 0.8,
        embedding_threshold: float = 0.7,
        max_candidates: int = 5,
    ) -> None:
        """Initialize the entity linker.

        Args:
            storage: Storage coordinator for entity access
            embedder: Embedder for semantic matching (optional)
            exact_match: Enable exact name matching
            fuzzy_match: Enable fuzzy string matching
            embedding_match: Enable embedding-based matching
            fuzzy_threshold: Minimum fuzzy match ratio (0-1)
            embedding_threshold: Minimum embedding similarity (0-1)
            max_candidates: Maximum candidates to return per mention
        """
        self._storage = storage
        self._embedder = embedder
        self._exact_match = exact_match
        self._fuzzy_match = fuzzy_match
        self._embedding_match = embedding_match
        self._fuzzy_threshold = fuzzy_threshold
        self._embedding_threshold = embedding_threshold
        self._max_candidates = max_candidates

    async def link(
        self,
        mentions: list[EntityMention],
        namespace_id: UUID,
    ) -> LinkingResult:
        """Link entity mentions to stored entities.

        Pre-computes embeddings for all mentions in a single batch call,
        then runs linking in parallel using the cached embeddings.

        Args:
            mentions: List of entity mentions to link
            namespace_id: Namespace to search in

        Returns:
            LinkingResult with linked and unlinked entities
        """
        if not mentions:
            return LinkingResult(total_mentions=0)

        # Pre-compute all mention embeddings in a single batch call
        # This avoids N individual embed() calls (each ~418ms) in favor of
        # one batch call (~500ms total for typical mention counts).
        mention_embeddings: dict[str, list[float]] = {}
        if self._embedding_match and self._embedder:
            mention_texts = [f"{m.entity_type}: {m.name}" for m in mentions]
            try:
                embeddings = await self._embedder.embed_batch(mention_texts)
                for text, embedding in zip(mention_texts, embeddings):
                    mention_embeddings[text] = embedding
            except Exception as e:
                logger.warning(f"Batch mention embedding failed, falling back to per-mention: {e}")

        # Parallelize linking across all mentions
        linked_entities = await asyncio.gather(
            *[self._link_single(m, namespace_id, mention_embeddings) for m in mentions]
        )
        unlinked_count = sum(1 for le in linked_entities if not le.is_linked)

        return LinkingResult(
            linked_entities=list(linked_entities),
            unlinked_count=unlinked_count,
            total_mentions=len(mentions),
        )

    async def _link_single(
        self,
        mention: EntityMention,
        namespace_id: UUID,
        precomputed_embeddings: dict[str, list[float]] | None = None,
    ) -> LinkedEntity:
        """Link a single entity mention.

        Args:
            mention: Entity mention to link
            namespace_id: Namespace to search in
            precomputed_embeddings: Pre-computed embeddings keyed by mention text

        Returns:
            LinkedEntity with match result
        """
        candidates: list[tuple[Entity, float, str]] = []  # (entity, score, method)

        # 1. Try exact match first (fastest)
        if self._exact_match:
            exact = await self._exact_name_match(mention, namespace_id)
            if exact:
                return LinkedEntity(
                    mention=mention,
                    entity=exact,
                    match_method="exact",
                    match_score=1.0,
                )

        # 2+3. Run fuzzy and embedding matching in parallel
        tasks: dict[str, Any] = {}
        if self._fuzzy_match:
            tasks["fuzzy"] = self._fuzzy_name_match(mention, namespace_id)
        if self._embedding_match and self._embedder:
            tasks["embedding"] = self._embedding_match_entities(mention, namespace_id, precomputed_embeddings)

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for method, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.debug(f"{method} match failed: {result}")
                    continue
                candidates.extend([(e, s, method) for e, s in result])

        # Select best match
        if candidates:
            # Sort by score descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_entity, best_score, best_method = candidates[0]

            # Return with candidates for context
            return LinkedEntity(
                mention=mention,
                entity=best_entity,
                match_method=best_method,
                match_score=best_score,
                candidates=[(e, s) for e, s, _ in candidates[: self._max_candidates]],
            )

        # No matches found
        return LinkedEntity(
            mention=mention,
            entity=None,
            match_method="",
            match_score=0.0,
        )

    async def _exact_name_match(
        self,
        mention: EntityMention,
        namespace_id: UUID,
    ) -> Entity | None:
        """Try exact name match.

        Args:
            mention: Entity mention
            namespace_id: Namespace to search

        Returns:
            Matched entity or None
        """
        try:
            # Try exact match with specified type
            entity = await self._storage.get_entity_by_name(
                namespace_id,
                mention.name,
                mention.entity_type,
            )
            if entity:
                return entity

            # Try case-insensitive match by listing and filtering
            entities = await self._storage.list_entities(
                namespace_id,
                entity_type=mention.entity_type,
                limit=1000,
            )
            name_lower = mention.name.lower()
            for entity in entities:
                if entity.name.lower() == name_lower:
                    return entity

        except Exception as e:
            logger.debug(f"Exact match failed: {e}")

        return None

    async def _fuzzy_name_match(
        self,
        mention: EntityMention,
        namespace_id: UUID,
    ) -> list[tuple[Entity, float]]:
        """Find entities with fuzzy name matching.

        Args:
            mention: Entity mention
            namespace_id: Namespace to search

        Returns:
            List of (entity, score) tuples
        """
        matches = []

        try:
            # Get all entities — search broadly to catch cross-type matches
            # that we can then penalize appropriately
            entities = await self._storage.list_entities(
                namespace_id,
                entity_type=mention.entity_type,
                limit=1000,
            )

            mention_name_lower = mention.name.lower()

            for entity in entities:
                # Calculate fuzzy similarity
                ratio = SequenceMatcher(
                    None,
                    mention_name_lower,
                    entity.name.lower(),
                ).ratio()

                if ratio >= self._fuzzy_threshold:
                    # Apply type penalty so wrong-type matches score lower
                    penalty = self._type_penalty(mention.entity_type, entity.entity_type.value)
                    matches.append((entity, ratio * penalty))

            # Also try partial matching (for handling first/last name, abbreviations)
            for entity in entities:
                if entity in [m[0] for m in matches]:
                    continue

                # Check if mention is contained in entity name or vice versa
                entity_name_lower = entity.name.lower()
                penalty = self._type_penalty(mention.entity_type, entity.entity_type.value)
                if mention_name_lower in entity_name_lower:
                    ratio = len(mention_name_lower) / len(entity_name_lower)
                    if ratio >= 0.5:  # At least 50% match
                        matches.append((entity, ratio * self._fuzzy_threshold * penalty))
                elif entity_name_lower in mention_name_lower:
                    ratio = len(entity_name_lower) / len(mention_name_lower)
                    if ratio >= 0.5:
                        matches.append((entity, ratio * self._fuzzy_threshold * penalty))

        except Exception as e:
            logger.debug(f"Fuzzy match failed: {e}")

        # Sort by score and limit
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[: self._max_candidates]

    async def _embedding_match_entities(
        self,
        mention: EntityMention,
        namespace_id: UUID,
        precomputed_embeddings: dict[str, list[float]] | None = None,
    ) -> list[tuple[Entity, float]]:
        """Find entities using embedding similarity.

        Args:
            mention: Entity mention
            namespace_id: Namespace to search
            precomputed_embeddings: Pre-computed embeddings keyed by mention text

        Returns:
            List of (entity, score) tuples
        """
        if not self._embedder:
            return []

        matches = []

        try:
            # Use pre-computed embedding if available, otherwise compute on the fly
            mention_text = f"{mention.entity_type}: {mention.name}"
            if precomputed_embeddings and mention_text in precomputed_embeddings:
                embedding = precomputed_embeddings[mention_text]
            else:
                embedding = await self._embedder.embed(mention_text)

            # Search for similar entities
            results = await self._storage.search_similar_entities(
                namespace_id,
                embedding,
                limit=self._max_candidates * 2,
                min_similarity=self._embedding_threshold,
            )

            # Fetch full entities
            for entity_id, score in results:
                entity = await self._storage.get_entity(entity_id)
                if entity:
                    # Only include entities of compatible types
                    if self._types_compatible(mention.entity_type, entity.entity_type.value):
                        # Apply type penalty so exact type matches rank higher
                        penalty = self._type_penalty(mention.entity_type, entity.entity_type.value)
                        matches.append((entity, score * penalty))

        except Exception as e:
            logger.debug(f"Embedding match failed: {e}")

        return matches[: self._max_candidates]

    def _types_compatible(self, mention_type: str, entity_type: str) -> bool:
        """Check if entity types are compatible for linking.

        Args:
            mention_type: Type from query mention
            entity_type: Type from stored entity

        Returns:
            True if types are compatible
        """
        # Exact match
        if mention_type.upper() == entity_type.upper():
            return True

        # CONCEPT is compatible with most types
        if mention_type.upper() == "CONCEPT" or entity_type.upper() == "CONCEPT":
            return True

        # CUSTOM is a wildcard
        if mention_type.upper() == "CUSTOM" or entity_type.upper() == "CUSTOM":
            return True

        return False

    def _type_penalty(self, mention_type: str | None, entity_type: str) -> float:
        """Compute a score penalty based on entity type compatibility.

        When the query understanding provides an entity type hint,
        penalize matches of the wrong type. For example, a query about
        "the company Linear" should penalize matching a PERSON named "Linear".

        Args:
            mention_type: Expected type from query mention (None = no hint)
            entity_type: Type of the candidate entity

        Returns:
            Multiplier between 0.0 and 1.0 (1.0 = no penalty)
        """
        if not mention_type:
            return 1.0  # No type info, no penalty

        mention_upper = mention_type.upper()
        entity_upper = entity_type.upper()

        # Exact type match — no penalty
        if mention_upper == entity_upper:
            return 1.0

        # Wildcards — minimal penalty
        if mention_upper in ("CONCEPT", "CUSTOM") or entity_upper in ("CONCEPT", "CUSTOM"):
            return 0.9

        # Wrong type — heavy penalty
        return 0.3


async def link_query_entities(
    mentions: list[EntityMention],
    namespace_id: UUID,
    storage: StorageCoordinator,
    embedder: Embedder | None = None,
    *,
    fuzzy_threshold: float = 0.8,
    embedding_threshold: float = 0.7,
    max_candidates: int = 5,
) -> LinkingResult:
    """Convenience function to link query entities.

    Args:
        mentions: Entity mentions from query understanding
        namespace_id: Namespace to search in
        storage: Storage coordinator
        embedder: Optional embedder for semantic matching
        fuzzy_threshold: Fuzzy match threshold
        embedding_threshold: Embedding similarity threshold
        max_candidates: Max candidates per mention

    Returns:
        LinkingResult with linked entities
    """
    linker = EntityLinker(
        storage,
        embedder,
        fuzzy_threshold=fuzzy_threshold,
        embedding_threshold=embedding_threshold,
        max_candidates=max_candidates,
    )
    return await linker.link(mentions, namespace_id)
