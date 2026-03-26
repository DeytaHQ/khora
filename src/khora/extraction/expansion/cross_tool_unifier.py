"""Cross-tool entity unification.

Unifies entities from different tools/sources that represent the same
real-world concept using correlation rules from expertise configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora._accel import cosine_similarity, levenshtein_similarity

from .rule_engine import RuleEngine, RuleEvaluationContext

if TYPE_CHECKING:
    from khora.core.models import Entity, Relationship
    from khora.extraction.skills import ExpertiseConfig

    from .entity_index import EntityIndex


@dataclass
class UnificationResult:
    """Result of cross-tool entity unification."""

    # Entities after unification (merged duplicates)
    unified_entities: list[Entity] = field(default_factory=list)

    # Mapping from original entity IDs to unified entity IDs
    entity_mapping: dict[UUID, UUID] = field(default_factory=dict)

    # Relationships updated with unified entity references
    updated_relationships: list[Relationship] = field(default_factory=list)

    # New relationships created by correlation rules
    new_relationships: list[Relationship] = field(default_factory=list)

    # Statistics
    entities_merged: int = 0
    merge_groups: list[list[UUID]] = field(default_factory=list)  # Groups of merged entity IDs


class CrossToolUnifier:
    """Unifies entities across different tools and sources.

    Uses correlation rules from expertise configuration to identify
    entities that should be merged:
    - Email matching for people
    - Domain matching for companies
    - Pattern matching for references (e.g., JIRA-123)
    - Embedding similarity for fuzzy matching
    """

    def __init__(
        self,
        expertise: ExpertiseConfig | None = None,
        *,
        embedding_threshold: float = 0.85,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        """Initialize the cross-tool unifier.

        Args:
            expertise: ExpertiseConfig with correlation rules
            embedding_threshold: Similarity threshold for embedding matching
            fuzzy_threshold: Threshold for fuzzy string matching
        """
        self._expertise = expertise
        self._embedding_threshold = embedding_threshold
        self._fuzzy_threshold = fuzzy_threshold
        self._rule_engine = RuleEngine(expertise)

    def unify(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        *,
        use_embeddings: bool = True,
        use_fuzzy: bool = True,
        entity_index: EntityIndex | None = None,
    ) -> UnificationResult:
        """Unify entities and update relationships.

        Args:
            entities: Entities to unify
            relationships: Relationships between entities
            use_embeddings: Whether to use embedding similarity
            use_fuzzy: Whether to use fuzzy string matching

        Returns:
            UnificationResult with unified entities and updated relationships
        """
        result = UnificationResult()

        if not entities:
            return result

        # Build evaluation context
        context = RuleEvaluationContext.from_data(entities, relationships)

        # Find entity groups that should be merged
        merge_groups = self._find_merge_groups(entities, context, use_embeddings, use_fuzzy, entity_index=entity_index)

        if not merge_groups:
            # No merging needed
            result.unified_entities = entities.copy()
            result.updated_relationships = relationships.copy()
            return result

        # Merge entities in each group
        entity_mapping: dict[UUID, UUID] = {}
        unified_entities: list[Entity] = []
        processed_ids: set[UUID] = set()

        for group in merge_groups:
            if len(group) < 2:
                continue

            # Merge all entities in the group
            merged_entity = self._merge_entity_group([e for e in entities if e.id in group])
            unified_entities.append(merged_entity)

            # Map all original IDs to the merged entity's ID
            for entity_id in group:
                entity_mapping[entity_id] = merged_entity.id
                processed_ids.add(entity_id)

            result.entities_merged += len(group) - 1
            result.merge_groups.append(list(group))

        # Add entities that weren't merged
        for entity in entities:
            if entity.id not in processed_ids:
                unified_entities.append(entity)
                entity_mapping[entity.id] = entity.id

        result.unified_entities = unified_entities
        result.entity_mapping = entity_mapping

        # Update relationships with new entity IDs
        result.updated_relationships = self._update_relationships(relationships, entity_mapping)

        # Find new relationships from correlation rules
        result.new_relationships = self._find_new_relationships(merge_groups, entities)

        logger.debug(
            f"Unified {len(entities)} entities into {len(unified_entities)} "
            f"({result.entities_merged} merged in {len(merge_groups)} groups)"
        )

        return result

    def _find_merge_groups(
        self,
        entities: list[Entity],
        context: RuleEvaluationContext,
        use_embeddings: bool,
        use_fuzzy: bool,
        *,
        entity_index: EntityIndex | None = None,
    ) -> list[set[UUID]]:
        """Find groups of entities that should be merged."""
        # Use Union-Find to track merge groups
        parent: dict[UUID, UUID] = {e.id: e.id for e in entities}

        def find(x: UUID) -> UUID:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: UUID, y: UUID) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Apply correlation rules
        if self._expertise and self._expertise.correlation_rules:
            for rule in self._expertise.correlation_rules:
                if rule.match_fields:
                    matches = self._find_field_match_pairs(entities, rule.match_fields, rule.entity_types)
                    for e1_id, e2_id in matches:
                        union(e1_id, e2_id)

        # Exact name matching within same type
        self._find_exact_name_matches(entities, union)

        # Optional: Embedding similarity matching
        if use_embeddings:
            embedding_matches = self._find_embedding_matches(entities, entity_index=entity_index)
            for e1_id, e2_id in embedding_matches:
                union(e1_id, e2_id)

        # Optional: Fuzzy string matching
        if use_fuzzy:
            fuzzy_matches = self._find_fuzzy_matches(entities, entity_index=entity_index)
            for e1_id, e2_id in fuzzy_matches:
                union(e1_id, e2_id)

        # Build groups from union-find
        groups: dict[UUID, set[UUID]] = {}
        for entity in entities:
            root = find(entity.id)
            if root not in groups:
                groups[root] = set()
            groups[root].add(entity.id)

        # Return only groups with multiple entities
        return [group for group in groups.values() if len(group) > 1]

    def _find_field_match_pairs(
        self,
        entities: list[Entity],
        match_fields: list[str],
        entity_types: list[str],
    ) -> list[tuple[UUID, UUID]]:
        """Find entity pairs that match on specified fields."""
        pairs = []

        # Filter by entity types if specified
        candidates = entities
        if entity_types:
            candidates = [e for e in entities if str(e.entity_type) in entity_types]

        # Group by field values
        for field_name in match_fields:
            field_groups: dict[Any, list[Entity]] = {}
            for entity in candidates:
                field_value = entity.attributes.get(field_name)
                if field_value:
                    # Normalize strings
                    if isinstance(field_value, str):
                        field_value = field_value.lower().strip()
                    if field_value not in field_groups:
                        field_groups[field_value] = []
                    field_groups[field_value].append(entity)

            # Create pairs from groups
            for group_entities in field_groups.values():
                if len(group_entities) > 1:
                    # Add pairs for all combinations in group
                    for i, e1 in enumerate(group_entities):
                        for e2 in group_entities[i + 1 :]:
                            pairs.append((e1.id, e2.id))

        return pairs

    def _find_exact_name_matches(
        self,
        entities: list[Entity],
        union: Any,
    ) -> None:
        """Find entities with exact name matches (same type)."""
        # Group by (normalized_name, type)
        name_type_groups: dict[tuple[str, str], list[Entity]] = {}
        for entity in entities:
            key = (
                entity.name.lower().strip(),
                str(entity.entity_type),
            )
            if key not in name_type_groups:
                name_type_groups[key] = []
            name_type_groups[key].append(entity)

        # Union entities in each group
        for group in name_type_groups.values():
            if len(group) > 1:
                first = group[0]
                for entity in group[1:]:
                    union(first.id, entity.id)

    def _find_embedding_matches(
        self,
        entities: list[Entity],
        *,
        entity_index: EntityIndex | None = None,
    ) -> list[tuple[UUID, UUID]]:
        """Find entity pairs with similar embeddings.

        When *entity_index* is provided, uses token blocking to reduce the
        candidate set from O(n^2) to O(n*k) where k is ~10-20 candidates
        per entity.
        """
        pairs = []

        # Filter to entities with embeddings
        with_embeddings = [e for e in entities if e.embedding]
        if len(with_embeddings) < 2:
            return pairs

        if entity_index is not None:
            # Blocked matching: O(n*k) via entity_index.
            # Pass processed_ids to skip reverse comparisons (A→B and B→A).
            seen: set[tuple[UUID, UUID]] = set()
            processed_ids: set[UUID] = set()
            for entity in with_embeddings:
                for candidate, similarity in entity_index.find_embedding_candidates(
                    entity, threshold=self._embedding_threshold, skip_ids=processed_ids
                ):
                    pair = (min(entity.id, candidate.id), max(entity.id, candidate.id))
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append((entity.id, candidate.id))
                processed_ids.add(entity.id)
            return pairs

        # Fallback: O(n^2) pairwise comparison
        for i, e1 in enumerate(with_embeddings):
            for e2 in with_embeddings[i + 1 :]:
                # Only compare same type
                if e1.entity_type != e2.entity_type:
                    continue

                similarity = cosine_similarity(e1.embedding, e2.embedding)
                if similarity >= self._embedding_threshold:
                    pairs.append((e1.id, e2.id))

        return pairs

    def _find_fuzzy_matches(
        self,
        entities: list[Entity],
        *,
        entity_index: EntityIndex | None = None,
    ) -> list[tuple[UUID, UUID]]:
        """Find entity pairs with similar names (fuzzy matching).

        When *entity_index* is provided, uses token blocking to reduce the
        candidate set from O(n^2) to O(n*k).
        """
        pairs = []

        if entity_index is not None:
            # Blocked matching: O(n*k) with batch_levenshtein.
            # Pass processed_ids to skip reverse comparisons.
            seen: set[tuple[UUID, UUID]] = set()
            processed_ids: set[UUID] = set()
            for entity in entities:
                for candidate, similarity in entity_index.find_fuzzy_candidates(
                    entity, threshold=self._fuzzy_threshold, skip_ids=processed_ids
                ):
                    pair = (min(entity.id, candidate.id), max(entity.id, candidate.id))
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append((entity.id, candidate.id))
                processed_ids.add(entity.id)
            return pairs

        # Fallback: O(n^2) pairwise within type groups
        type_groups: dict[str, list[Entity]] = {}
        for entity in entities:
            type_key = str(entity.entity_type)
            if type_key not in type_groups:
                type_groups[type_key] = []
            type_groups[type_key].append(entity)

        for group in type_groups.values():
            if len(group) < 2:
                continue

            for i, e1 in enumerate(group):
                for e2 in group[i + 1 :]:
                    similarity = levenshtein_similarity(e1.name, e2.name)
                    if similarity >= self._fuzzy_threshold:
                        pairs.append((e1.id, e2.id))

        return pairs

    def _merge_entity_group(self, entities: list[Entity]) -> Entity:
        """Merge a group of entities into one.

        Strategy:
        - Use the entity with highest confidence as base
        - Merge attributes (non-empty values preferred)
        - Combine source document/chunk IDs
        - Sum mention counts
        - Keep earliest created_at
        """
        if not entities:
            raise ValueError("Cannot merge empty entity list")

        if len(entities) == 1:
            return entities[0]

        # Sort by confidence (highest first)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        base = sorted_entities[0]

        # Merge in remaining entities
        for entity in sorted_entities[1:]:
            base.merge_with(entity)

        return base

    def _update_relationships(
        self,
        relationships: list[Relationship],
        entity_mapping: dict[UUID, UUID],
    ) -> list[Relationship]:
        """Update relationships to use unified entity IDs."""
        updated = []
        for rel in relationships:
            # Create a copy with updated IDs
            new_source = entity_mapping.get(rel.source_entity_id, rel.source_entity_id)
            new_target = entity_mapping.get(rel.target_entity_id, rel.target_entity_id)

            # Skip self-referential relationships that emerged from merging
            if new_source == new_target:
                continue

            # Update the relationship IDs
            rel.source_entity_id = new_source
            rel.target_entity_id = new_target
            updated.append(rel)

        return updated

    def _find_new_relationships(
        self,
        merge_groups: list[set[UUID]],
        entities: list[Entity],
    ) -> list[Relationship]:
        """Create CROSS_REFERENCED relationships when entities from different documents merge."""
        from datetime import UTC, datetime
        from uuid import uuid4

        from khora.core.models import Relationship

        entity_lookup = {e.id: e for e in entities}
        new_rels: list[Relationship] = []

        for group in merge_groups:
            group_entities = [entity_lookup[eid] for eid in group if eid in entity_lookup]
            if len(group_entities) < 2:
                continue

            # Check if entities span multiple documents
            all_doc_ids: set[UUID] = set()
            for e in group_entities:
                all_doc_ids.update(e.source_document_ids)

            if len(all_doc_ids) < 2:
                continue

            # Create CROSS_REFERENCED between the merged entity pairs
            base = group_entities[0]
            for other in group_entities[1:]:
                # Only create if they come from different documents
                base_docs = set(base.source_document_ids)
                other_docs = set(other.source_document_ids)
                if base_docs & other_docs:
                    continue

                new_rels.append(
                    Relationship(
                        id=uuid4(),
                        namespace_id=base.namespace_id,
                        source_entity_id=base.id,
                        target_entity_id=other.id,
                        relationship_type="CROSS_REFERENCED",
                        description=f"Entity '{base.name}' referenced across documents",
                        properties={"inferred": True},
                        source_document_ids=list(all_doc_ids),
                        source_chunk_ids=[],
                        confidence=0.7,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )
                )

        if new_rels:
            logger.debug(
                f"Created {len(new_rels)} CROSS_REFERENCED relationships from {len(merge_groups)} merge groups"
            )

        return new_rels
