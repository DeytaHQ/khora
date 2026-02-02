"""Entity resolution for deduplication during extraction.

Resolves and merges entities to avoid duplicates in the knowledge graph.
Uses multiple strategies:
- Exact name match
- Alias matching
- Embedding similarity
- Fuzzy name matching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora._accel import sequence_match_ratio
from khora.core.models.entity import entity_type_str

if TYPE_CHECKING:
    from khora.core.models import Entity
    from khora.extraction.embedders import Embedder
    from khora.storage import StorageCoordinator


@dataclass
class ResolutionCandidate:
    """A candidate entity for resolution."""

    entity: Entity
    match_type: str  # exact, alias, fuzzy, embedding
    score: float


@dataclass
class ResolutionResult:
    """Result of entity resolution."""

    is_duplicate: bool
    existing_entity: Entity | None = None
    match_type: str = ""
    match_score: float = 0.0
    should_merge: bool = False


class EntityResolver:
    """Resolves new entities against existing ones.

    Before creating a new entity, use this resolver to check
    if an equivalent entity already exists.
    """

    def __init__(
        self,
        storage: StorageCoordinator,
        embedder: Embedder | None = None,
        *,
        exact_match: bool = True,
        alias_match: bool = True,
        fuzzy_match: bool = True,
        embedding_match: bool = True,
        fuzzy_threshold: float = 0.85,
        embedding_threshold: float = 0.85,
    ) -> None:
        """Initialize the entity resolver.

        Args:
            storage: Storage coordinator for entity access
            embedder: Embedder for semantic matching
            exact_match: Enable exact name matching
            alias_match: Enable alias matching
            fuzzy_match: Enable fuzzy string matching
            embedding_match: Enable embedding-based matching
            fuzzy_threshold: Minimum fuzzy match ratio
            embedding_threshold: Minimum embedding similarity
        """
        self._storage = storage
        self._embedder = embedder
        self._exact_match = exact_match
        self._alias_match = alias_match
        self._fuzzy_match = fuzzy_match
        self._embedding_match = embedding_match
        self._fuzzy_threshold = fuzzy_threshold
        self._embedding_threshold = embedding_threshold

        # Cache for performance
        self._entity_cache: dict[str, list[Entity]] = {}

    async def resolve(
        self,
        name: str,
        entity_type: str,
        namespace_id: UUID,
        *,
        description: str = "",
        aliases: list[str] | None = None,
    ) -> ResolutionResult:
        """Resolve a potential entity against existing entities.

        Args:
            name: Entity name
            entity_type: Entity type (PERSON, ORGANIZATION, etc.)
            namespace_id: Namespace to check
            description: Optional description for semantic matching
            aliases: Optional aliases for the entity

        Returns:
            ResolutionResult indicating if entity is a duplicate
        """
        aliases = aliases or []
        candidates: list[ResolutionCandidate] = []

        # Load entities of same type from cache or storage
        cache_key = f"{namespace_id}:{entity_type}"
        if cache_key not in self._entity_cache:
            try:
                entities = await self._storage.list_entities(
                    namespace_id,
                    entity_type=entity_type,
                    limit=1000,
                )
                self._entity_cache[cache_key] = entities
            except Exception as e:
                logger.debug(f"Failed to load entities for resolution: {e}")
                self._entity_cache[cache_key] = []

        existing_entities = self._entity_cache.get(cache_key, [])

        # 1. Exact name match
        if self._exact_match:
            for entity in existing_entities:
                if entity.name.lower() == name.lower():
                    return ResolutionResult(
                        is_duplicate=True,
                        existing_entity=entity,
                        match_type="exact",
                        match_score=1.0,
                        should_merge=True,
                    )

        # 2. Alias match
        if self._alias_match:
            name_lower = name.lower()
            aliases_lower = [a.lower() for a in aliases]

            for entity in existing_entities:
                # Check if name matches any existing alias
                entity_aliases = entity.metadata.get("aliases", [])
                entity_aliases_lower = [a.lower() for a in entity_aliases]

                if name_lower in entity_aliases_lower:
                    candidates.append(ResolutionCandidate(entity, "alias", 0.95))
                    continue

                # Check if any new alias matches existing name or alias
                if entity.name.lower() in aliases_lower:
                    candidates.append(ResolutionCandidate(entity, "alias", 0.95))
                    continue

                for alias in aliases_lower:
                    if alias in entity_aliases_lower:
                        candidates.append(ResolutionCandidate(entity, "alias", 0.9))
                        break

        # 3. Fuzzy name match
        if self._fuzzy_match:
            name_lower = name.lower()
            for entity in existing_entities:
                # Skip if already matched
                if entity in [c.entity for c in candidates]:
                    continue

                ratio = sequence_match_ratio(name_lower, entity.name.lower())
                if ratio >= self._fuzzy_threshold:
                    candidates.append(ResolutionCandidate(entity, "fuzzy", ratio))

        # 4. Embedding similarity match
        if self._embedding_match and self._embedder and description:
            try:
                # Create text for embedding
                search_text = f"{entity_type}: {name}. {description}"
                embedding = await self._embedder.embed(search_text)

                # Search for similar entities
                results = await self._storage.search_similar_entities(
                    namespace_id,
                    embedding,
                    limit=5,
                    min_similarity=self._embedding_threshold,
                )

                for entity_id, score in results:
                    entity = await self._storage.get_entity(entity_id)
                    if entity and entity_type_str(entity.entity_type) == entity_type:
                        # Skip if already matched
                        if entity in [c.entity for c in candidates]:
                            continue
                        candidates.append(ResolutionCandidate(entity, "embedding", score))

            except Exception as e:
                logger.debug(f"Embedding match failed: {e}")

        # Select best match
        if candidates:
            candidates.sort(key=lambda c: c.score, reverse=True)
            best = candidates[0]

            return ResolutionResult(
                is_duplicate=True,
                existing_entity=best.entity,
                match_type=best.match_type,
                match_score=best.score,
                should_merge=best.score >= 0.85,
            )

        # No match found - this is a new entity
        return ResolutionResult(is_duplicate=False)

    def invalidate_cache(self, namespace_id: UUID | None = None) -> None:
        """Invalidate the entity cache.

        Args:
            namespace_id: Optional namespace to invalidate.
                         If None, invalidates all.
        """
        if namespace_id is None:
            self._entity_cache.clear()
        else:
            keys_to_remove = [k for k in self._entity_cache if k.startswith(str(namespace_id))]
            for key in keys_to_remove:
                del self._entity_cache[key]


async def resolve_and_merge_entity(
    name: str,
    entity_type: str,
    namespace_id: UUID,
    storage: StorageCoordinator,
    embedder: Embedder | None = None,
    *,
    description: str = "",
    aliases: list[str] | None = None,
    attributes: dict[str, Any] | None = None,
    source_document_id: UUID | None = None,
    source_chunk_id: UUID | None = None,
) -> tuple[Entity, bool]:
    """Resolve and optionally merge an entity.

    Convenience function that resolves an entity and either returns
    the existing entity (merged) or indicates a new entity should be created.

    Args:
        name: Entity name
        entity_type: Entity type
        namespace_id: Namespace ID
        storage: Storage coordinator
        embedder: Optional embedder
        description: Entity description
        aliases: Entity aliases
        attributes: Entity attributes
        source_document_id: Source document ID
        source_chunk_id: Source chunk ID

    Returns:
        Tuple of (entity, is_new) where is_new indicates if a new entity
        should be created (False means use the returned existing entity)
    """
    from khora.core.models import Entity, EntityType

    resolver = EntityResolver(storage, embedder)
    result = await resolver.resolve(
        name,
        entity_type,
        namespace_id,
        description=description,
        aliases=aliases or [],
    )

    if result.is_duplicate and result.existing_entity and result.should_merge:
        # Merge into existing entity
        existing = result.existing_entity

        # Update source tracking
        if source_document_id and source_document_id not in existing.source_document_ids:
            existing.source_document_ids.append(source_document_id)
        if source_chunk_id and source_chunk_id not in existing.source_chunk_ids:
            existing.source_chunk_ids.append(source_chunk_id)

        # Increment mention count
        existing.mention_count += 1

        # Merge attributes
        if attributes:
            for key, value in attributes.items():
                if key not in existing.attributes:
                    existing.attributes[key] = value

        # Merge aliases
        existing_aliases = existing.metadata.get("aliases", [])
        if aliases:
            for alias in aliases:
                if alias not in existing_aliases:
                    existing_aliases.append(alias)
            existing.metadata["aliases"] = existing_aliases

        # Update description if empty
        if not existing.description and description:
            existing.description = description

        logger.debug(
            f"Merged entity '{name}' with existing '{existing.name}' "
            f"(match: {result.match_type}, score: {result.match_score:.2f})"
        )

        return existing, False

    # Create new entity
    try:
        etype: EntityType | str = EntityType(entity_type.upper())
    except ValueError:
        etype = entity_type.upper() or "CONCEPT"

    new_entity = Entity(
        namespace_id=namespace_id,
        name=name,
        entity_type=etype,
        description=description,
        attributes=attributes or {},
        source_document_ids=[source_document_id] if source_document_id else [],
        source_chunk_ids=[source_chunk_id] if source_chunk_id else [],
        mention_count=1,
        metadata={"aliases": aliases or []},
    )

    return new_entity, True
