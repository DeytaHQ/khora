"""Relationship inference from existing graph patterns.

Infers new relationships based on configurable inference rules
defined in expertise configuration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger

from .rule_engine import RuleEngine, RuleEvaluationContext, RuleMatch

if TYPE_CHECKING:
    from khora.core.models import Entity, Relationship
    from khora.extraction.skills import ExpertiseConfig


@dataclass
class InferredRelationship:
    """An inferred relationship from rule evaluation."""

    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    description: str = ""
    confidence: float = 0.5
    rule_name: str = ""  # Name of inference rule that created this
    evidence: list[UUID] = field(default_factory=list)  # IDs of relationships used as evidence


class RelationshipTypeIndex:
    """Index for O(1) relationship lookups by type.

    Provides efficient access to relationships organized by their type,
    reducing iteration complexity from O(n) to O(1) for type-based queries.
    This is critical for inference rules that match on relationship types.
    """

    def __init__(self) -> None:
        """Initialize empty type index."""
        # relationship_type -> list of relationships
        self._by_type: dict[str, list[Relationship]] = {}
        # source_entity_id -> list of relationships
        self._by_source: dict[UUID, list[Relationship]] = {}
        # target_entity_id -> list of relationships
        self._by_target: dict[UUID, list[Relationship]] = {}
        # (source_id, target_id, type) -> relationship for dedup checks
        self._lookup: dict[tuple[UUID, UUID, str], Relationship] = {}

    def build(self, relationships: list[Relationship]) -> None:
        """Build the index from a list of relationships.

        Args:
            relationships: List of relationships to index
        """
        self._by_type.clear()
        self._by_source.clear()
        self._by_target.clear()
        self._lookup.clear()

        for rel in relationships:
            rel_type = self._get_type_str(rel)

            # Index by type
            if rel_type not in self._by_type:
                self._by_type[rel_type] = []
            self._by_type[rel_type].append(rel)

            # Index by source
            if rel.source_entity_id not in self._by_source:
                self._by_source[rel.source_entity_id] = []
            self._by_source[rel.source_entity_id].append(rel)

            # Index by target
            if rel.target_entity_id not in self._by_target:
                self._by_target[rel.target_entity_id] = []
            self._by_target[rel.target_entity_id].append(rel)

            # Dedup lookup
            key = (rel.source_entity_id, rel.target_entity_id, rel_type)
            self._lookup[key] = rel

    def get_by_type(self, rel_type: str) -> list[Relationship]:
        """Get all relationships of a given type in O(1).

        Args:
            rel_type: Relationship type string

        Returns:
            List of relationships (empty if type not found)
        """
        return self._by_type.get(rel_type, [])

    def get_by_source(self, source_id: UUID) -> list[Relationship]:
        """Get all relationships from a source entity in O(1).

        Args:
            source_id: Source entity UUID

        Returns:
            List of outgoing relationships
        """
        return self._by_source.get(source_id, [])

    def get_by_target(self, target_id: UUID) -> list[Relationship]:
        """Get all relationships to a target entity in O(1).

        Args:
            target_id: Target entity UUID

        Returns:
            List of incoming relationships
        """
        return self._by_target.get(target_id, [])

    def get_by_source_and_type(self, source_id: UUID, rel_type: str) -> list[Relationship]:
        """Get relationships from a source with a specific type.

        Efficient for chained inference patterns like:
        "A -[MANAGES]-> B -[WORKS_ON]-> C"

        Args:
            source_id: Source entity UUID
            rel_type: Relationship type string

        Returns:
            Filtered list of relationships
        """
        source_rels = self._by_source.get(source_id, [])
        return [r for r in source_rels if self._get_type_str(r) == rel_type]

    def get_by_target_and_type(self, target_id: UUID, rel_type: str) -> list[Relationship]:
        """Get relationships to a target with a specific type.

        Args:
            target_id: Target entity UUID
            rel_type: Relationship type string

        Returns:
            Filtered list of relationships
        """
        target_rels = self._by_target.get(target_id, [])
        return [r for r in target_rels if self._get_type_str(r) == rel_type]

    def exists(self, source_id: UUID, target_id: UUID, rel_type: str) -> bool:
        """Check if a relationship exists in O(1).

        Args:
            source_id: Source entity UUID
            target_id: Target entity UUID
            rel_type: Relationship type string

        Returns:
            True if relationship exists
        """
        return (source_id, target_id, rel_type) in self._lookup

    def get_types(self) -> set[str]:
        """Get all relationship types in the index.

        Returns:
            Set of relationship type strings
        """
        return set(self._by_type.keys())

    def stats(self) -> dict[str, int]:
        """Get index statistics.

        Returns:
            Dict with counts for types, sources, targets, and total relationships
        """
        return {
            "type_groups": len(self._by_type),
            "source_groups": len(self._by_source),
            "target_groups": len(self._by_target),
            "total_relationships": len(self._lookup),
        }

    @staticmethod
    def _get_type_str(rel: Relationship) -> str:
        """Extract relationship type as string."""
        rt = rel.relationship_type
        return str(rt.value) if hasattr(rt, "value") else str(rt)


class RelationshipInferrer:
    """Infers new relationships based on existing graph patterns.

    Uses inference rules from expertise configuration to deduce
    relationships that aren't explicitly stated but can be inferred
    from existing relationships.

    Includes RelationshipTypeIndex for O(1) lookups by relationship type,
    reducing inference complexity from O(n) per rule to O(1) type lookup
    plus O(k) candidate iteration where k << n.

    Example inference rules:
    - If A manages B and B works on project C, A is stakeholder of C
    - If person P is mentioned in channel for project X, P is involved in X
    - If PR author and reviewer work on same PR, they collaborate
    """

    def __init__(
        self,
        expertise: ExpertiseConfig | None = None,
        *,
        min_confidence: float = 0.3,
        max_inferences_per_rule: int = 500,
    ) -> None:
        """Initialize the relationship inferrer.

        Args:
            expertise: ExpertiseConfig with inference rules
            min_confidence: Minimum confidence for inferred relationships
            max_inferences_per_rule: Maximum inferences per rule to prevent explosion
        """
        self._expertise = expertise
        self._min_confidence = min_confidence
        self._max_inferences_per_rule = max_inferences_per_rule
        self._rule_engine = RuleEngine(expertise)
        self._rel_index: RelationshipTypeIndex | None = None

    def infer(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        *,
        depth: int = 1,
    ) -> list[InferredRelationship]:
        """Infer new relationships from existing graph.

        Uses RelationshipTypeIndex for O(1) lookups by relationship type,
        significantly improving performance for graphs with many relationships.

        Args:
            entities: Existing entities
            relationships: Existing relationships
            depth: Number of inference passes (for transitive inference)

        Returns:
            List of inferred relationships
        """
        if not self._expertise or not self._expertise.inference_rules:
            logger.debug("No expertise or inference rules configured, skipping inference")
            return []

        # Build relationship type index for O(1) lookups
        self._rel_index = RelationshipTypeIndex()
        self._rel_index.build(relationships)
        logger.debug(f"Built relationship index: {self._rel_index.stats()}")

        # Diagnostic logging
        entity_types = Counter(
            e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type) for e in entities
        )
        logger.debug(f"Inference input: {len(entities)} entities, types: {dict(entity_types)}")

        rel_types = Counter(
            r.relationship_type.value if hasattr(r.relationship_type, "value") else str(r.relationship_type)
            for r in relationships
        )
        logger.debug(f"Inference input: {len(relationships)} relationships, types: {dict(rel_types)}")

        # Detailed pattern diagnostics (debug-only to avoid overhead)
        if logger._core.min_level <= 10:  # type: ignore[unresolved-attribute]  # DEBUG level
            entity_type_lookup = {
                e.id: (e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)) for e in entities
            }
            rel_type_patterns: dict[str, Counter] = {}
            for r in relationships:
                rt = (
                    str(r.relationship_type.value)
                    if hasattr(r.relationship_type, "value")
                    else str(r.relationship_type)
                )
                source_type = entity_type_lookup.get(r.source_entity_id, "UNKNOWN")
                target_type = entity_type_lookup.get(r.target_entity_id, "UNKNOWN")
                if rt not in rel_type_patterns:
                    rel_type_patterns[rt] = Counter()
                rel_type_patterns[rt][f"{source_type}->{target_type}"] += 1

            for rel_type, patterns in rel_type_patterns.items():
                logger.debug(f"  {rel_type} patterns: {dict(patterns.most_common(5))}")

        # Check rule compatibility (log mismatches as warnings)
        expected_rels = set()
        expected_entities = set()
        for rule in self._expertise.inference_rules:
            for cond in rule.when:
                if hasattr(cond, "relationship"):
                    expected_rels.add(cond.relationship)
                if hasattr(cond, "source_type"):
                    expected_entities.add(cond.source_type)
                if hasattr(cond, "target_type"):
                    expected_entities.add(cond.target_type)

        if expected_rels or expected_entities:
            actual_rel_types = {
                r.relationship_type.value if hasattr(r.relationship_type, "value") else str(r.relationship_type)
                for r in relationships
            }
            actual_entity_types = {
                e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type) for e in entities
            }
            if not (actual_rel_types & expected_rels):
                logger.debug(
                    f"No relationship type overlap: rules expect {sorted(expected_rels)}, "
                    f"graph has {sorted(actual_rel_types)}"
                )
            if not (actual_entity_types & expected_entities):
                logger.debug(
                    f"No entity type overlap: rules expect {sorted(expected_entities)}, "
                    f"graph has {sorted(actual_entity_types)}"
                )

        all_inferred: list[InferredRelationship] = []
        current_relationships = list(relationships)

        for pass_num in range(depth):
            # Build context with current state
            context = RuleEvaluationContext.from_data(entities, current_relationships)

            # Evaluate inference rules
            matches = self._rule_engine.evaluate_inference_rules(context)
            logger.debug(f"Pass {pass_num + 1}: Rule engine returned {len(matches)} matches")

            # Log details of first few matches (debug only, avoids list() overhead)
            if logger._core.min_level <= 10:  # type: ignore[unresolved-attribute]
                for i, match in enumerate(matches[:5]):
                    logger.debug(
                        f"  Match {i + 1}: rule={match.rule_name}, "
                        f"confidence={match.confidence:.2f}, "
                        f"metadata_keys={list(match.metadata.keys())}"
                    )

            # Convert matches to inferred relationships
            pass_inferred = self._matches_to_relationships(matches, context)
            logger.debug(f"Pass {pass_num + 1}: Converted to {len(pass_inferred)} inferred relationships")

            if not pass_inferred:
                logger.debug(f"Pass {pass_num + 1}: No new inferences, stopping early")
                break

            # Filter out duplicates and already existing relationships
            new_inferred = self._filter_duplicates(pass_inferred, current_relationships, all_inferred)

            if not new_inferred:
                break

            all_inferred.extend(new_inferred)

            # Add inferred to current for next pass (as mock relationships)
            current_relationships.extend(self._to_mock_relationships(new_inferred, entities))

            logger.debug(f"Inference pass {pass_num + 1}: {len(new_inferred)} new relationships")

        logger.debug(f"Inference complete: {len(all_inferred)} total relationships inferred")
        return all_inferred

    def infer_co_occurrences(
        self,
        entities: list[Entity],
        *,
        min_co_occurrences: int = 2,
    ) -> list[InferredRelationship]:
        """Infer CO_OCCURS_WITH relationships for entities sharing source chunks.

        Entities extracted from the same chunk but without an explicit relationship
        get a CO_OCCURS_WITH edge. Filters by entity type pair validity to avoid
        noise from low-value pairings (e.g., DATE + URL).

        Args:
            entities: Entities with source_chunk_ids populated
            min_co_occurrences: Minimum shared chunks to create a relationship

        Returns:
            List of inferred CO_OCCURS_WITH relationships
        """
        # Build chunk_id -> entities mapping
        chunk_to_entities: dict[UUID, list[Entity]] = {}
        for entity in entities:
            for chunk_id in entity.source_chunk_ids:
                chunk_to_entities.setdefault(chunk_id, []).append(entity)

        # Count co-occurrences for each entity pair
        pair_counts: Counter[tuple[UUID, UUID]] = Counter()
        for chunk_entities in chunk_to_entities.values():
            for i, e1 in enumerate(chunk_entities):
                for e2 in chunk_entities[i + 1 :]:
                    pair = (min(e1.id, e2.id), max(e1.id, e2.id))
                    pair_counts[pair] += 1

        # Entity type pairs that produce noisy co-occurrence edges
        INVALID_PAIRS = {
            frozenset({"DATE", "DATE"}),
            frozenset({"URL", "URL"}),
            frozenset({"EMAIL", "EMAIL"}),
            frozenset({"DATE", "URL"}),
            frozenset({"DATE", "EMAIL"}),
            frozenset({"URL", "EMAIL"}),
        }

        entity_lookup = {e.id: e for e in entities}
        inferred: list[InferredRelationship] = []

        for (id1, id2), count in pair_counts.items():
            if count < min_co_occurrences:
                continue
            e1 = entity_lookup.get(id1)
            e2 = entity_lookup.get(id2)
            if not e1 or not e2:
                continue

            et1 = e1.entity_type.value if hasattr(e1.entity_type, "value") else str(e1.entity_type)
            et2 = e2.entity_type.value if hasattr(e2.entity_type, "value") else str(e2.entity_type)
            if frozenset({et1, et2}) in INVALID_PAIRS:
                continue

            confidence = min(0.7, 0.3 + 0.1 * count)
            inferred.append(
                InferredRelationship(
                    source_entity_id=id1,
                    target_entity_id=id2,
                    relationship_type="CO_OCCURS_WITH",
                    description=f"Co-occurs in {count} chunks",
                    confidence=confidence,
                    rule_name="co_occurrence",
                )
            )

        logger.debug(f"Co-occurrence inference: {len(inferred)} relationships from {len(pair_counts)} pairs")
        return inferred

    def infer_from_pattern(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        pattern: str,
    ) -> list[InferredRelationship]:
        """Infer relationships matching a specific pattern.

        Args:
            entities: Existing entities
            relationships: Existing relationships
            pattern: Pattern to match (e.g., "A -> WORKS_FOR -> B, B -> OWNS -> C")

        Returns:
            List of inferred relationships
        """
        # Parse pattern and find matches
        # For now, delegate to rule engine
        context = RuleEvaluationContext.from_data(entities, relationships)
        matches = self._rule_engine.evaluate_inference_rules(context)
        return self._matches_to_relationships(matches, context)

    def find_chains_indexed(
        self,
        first_type: str,
        second_type: str,
        entities: list[Entity],
    ) -> list[tuple[Relationship, Relationship]]:
        """Find relationship chains A-[first_type]->B-[second_type]->C using index.

        Uses RelationshipTypeIndex for O(1) type lookups followed by O(k) iteration
        over matching relationships, where k is the number of relationships of
        that type. This is significantly faster than O(n^2) nested iteration.

        Example: Find "MANAGES -> WORKS_ON" chains to infer STAKEHOLDER_OF.

        Args:
            first_type: Relationship type for first hop (e.g., "MANAGES")
            second_type: Relationship type for second hop (e.g., "WORKS_ON")
            entities: List of entities (for entity lookup by ID)

        Returns:
            List of (first_rel, second_rel) tuples forming valid chains
        """
        if not self._rel_index:
            logger.warning("Relationship index not built, returning empty chains")
            return []

        # Build entity lookup for validation
        entity_lookup: dict[UUID, Entity] = {e.id: e for e in entities}

        chains: list[tuple[Relationship, Relationship]] = []

        # O(1) lookup for first type
        first_rels = self._rel_index.get_by_type(first_type)

        for first_rel in first_rels:
            # The intermediate entity is the target of the first relationship
            intermediate_id = first_rel.target_entity_id

            # O(1) lookup for second type from the intermediate entity
            second_rels = self._rel_index.get_by_source_and_type(intermediate_id, second_type)

            for second_rel in second_rels:
                # Validate that all entities exist
                if (
                    first_rel.source_entity_id in entity_lookup
                    and intermediate_id in entity_lookup
                    and second_rel.target_entity_id in entity_lookup
                ):
                    chains.append((first_rel, second_rel))

        logger.debug(
            f"Found {len(chains)} chains for {first_type} -> {second_type} "
            f"(checked {len(first_rels)} first-hop relationships)"
        )

        return chains

    def get_relationship_index(self) -> RelationshipTypeIndex | None:
        """Get the current relationship type index.

        Returns the index built during the last infer() call, or None if
        inference hasn't been run yet.

        Returns:
            RelationshipTypeIndex or None
        """
        return self._rel_index

    def _matches_to_relationships(
        self,
        matches: list[RuleMatch],
        context: RuleEvaluationContext,
    ) -> list[InferredRelationship]:
        """Convert rule matches to inferred relationships."""
        inferred = []
        rule_counts: dict[str, int] = {}

        # Diagnostic counters
        filtered_rule_limit = 0
        filtered_confidence = 0
        filtered_entity_resolution = 0
        filtered_self_ref = 0

        for match in matches:
            # Check rule limit
            rule_counts[match.rule_name] = rule_counts.get(match.rule_name, 0) + 1
            if rule_counts[match.rule_name] > self._max_inferences_per_rule:
                filtered_rule_limit += 1
                continue

            # Check confidence threshold
            if match.confidence < self._min_confidence:
                filtered_confidence += 1
                continue

            # Resolve source and target from metadata
            source_entity, target_entity = self._resolve_inference_entities(match, context)

            if not source_entity or not target_entity:
                filtered_entity_resolution += 1
                continue

            # Skip self-referential
            if source_entity.id == target_entity.id:
                filtered_self_ref += 1
                continue

            relationship_type = match.metadata.get("then_relationship", "RELATES_TO")

            inferred.append(
                InferredRelationship(
                    source_entity_id=source_entity.id,
                    target_entity_id=target_entity.id,
                    relationship_type=relationship_type,
                    description=f"Inferred by rule: {match.rule_name}",
                    confidence=match.confidence,
                    rule_name=match.rule_name,
                    evidence=[r.id for r in match.matched_relationships],
                )
            )

        # Log filtering breakdown
        total_filtered = filtered_rule_limit + filtered_confidence + filtered_entity_resolution + filtered_self_ref
        if total_filtered > 0 or len(matches) > 0:
            logger.debug(
                f"Match filtering: {len(matches)} input -> {len(inferred)} output | "
                f"rule_limit={filtered_rule_limit}, confidence={filtered_confidence}, "
                f"entity_resolution={filtered_entity_resolution}, self_ref={filtered_self_ref}"
            )

        return inferred

    def _resolve_inference_entities(
        self,
        match: RuleMatch,
        context: RuleEvaluationContext,
    ) -> tuple[Entity | None, Entity | None]:
        """Resolve source and target entities from match metadata.

        The then_source and then_target specify which entity to use:
        - "first.source": Source entity of first matched relationship
        - "first.target": Target entity of first matched relationship
        - "second.source": Source entity of second matched relationship
        - "second.target": Target entity of second matched relationship
        """
        then_source = match.metadata.get("then_source", "first.source")
        then_target = match.metadata.get("then_target", "second.target")

        first = match.metadata.get("first", {})
        second = match.metadata.get("second", {})

        def resolve_ref(ref: str) -> Entity | None:
            parts = ref.split(".")
            if len(parts) != 2:
                return None

            group, position = parts
            if group == "first":
                data = first
            elif group == "second":
                data = second
            else:
                logger.warning(f"Rule '{match.rule_name}': invalid reference '{ref}' (only 'first'/'second' supported)")
                return None

            return data.get(position)

        source = resolve_ref(then_source)
        target = resolve_ref(then_target)

        return source, target

    def _filter_duplicates(
        self,
        new_inferred: list[InferredRelationship],
        existing_relationships: list[Relationship],
        already_inferred: list[InferredRelationship],
    ) -> list[InferredRelationship]:
        """Filter out duplicate inferences."""
        # Build set of existing relationship keys
        existing_keys: set[tuple[UUID, UUID, str]] = set()

        for rel in existing_relationships:
            rel_type = str(
                rel.relationship_type.value if hasattr(rel.relationship_type, "value") else rel.relationship_type
            )
            existing_keys.add((rel.source_entity_id, rel.target_entity_id, rel_type))

        for inf in already_inferred:
            existing_keys.add((inf.source_entity_id, inf.target_entity_id, inf.relationship_type))

        # Filter new inferences
        filtered = []
        for inf in new_inferred:
            key = (inf.source_entity_id, inf.target_entity_id, inf.relationship_type)
            if key not in existing_keys:
                filtered.append(inf)
                existing_keys.add(key)

        return filtered

    def _to_mock_relationships(
        self,
        inferred: list[InferredRelationship],
        entities: list[Entity],
    ) -> list[Relationship]:
        """Convert inferred relationships to mock Relationship objects for next pass."""
        from khora.core.models import Relationship
        from khora.core.models.entity import RelationshipType

        mock_rels = []
        # Get namespace from first entity if available
        namespace_id = entities[0].namespace_id if entities else uuid4()

        for inf in inferred:
            # Preserve original type string for domain-specific types
            try:
                rel_type: RelationshipType | str = RelationshipType[inf.relationship_type]
            except (KeyError, AttributeError):
                rel_type = inf.relationship_type

            mock_rels.append(
                Relationship(
                    id=uuid4(),
                    namespace_id=namespace_id,
                    source_entity_id=inf.source_entity_id,
                    target_entity_id=inf.target_entity_id,
                    relationship_type=rel_type,
                    description=inf.description,
                    properties={},
                    source_document_ids=[],
                    source_chunk_ids=[],
                    confidence=inf.confidence,
                    metadata={"inferred": True, "rule": inf.rule_name},
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )

        return mock_rels


def to_relationship(
    inferred: InferredRelationship,
    namespace_id: UUID,
) -> Relationship:
    """Convert an InferredRelationship to a domain Relationship model.

    Args:
        inferred: The inferred relationship to convert
        namespace_id: Namespace ID for the relationship

    Returns:
        Domain Relationship model
    """
    from khora.core.models import Relationship
    from khora.core.models.entity import RelationshipType

    # Preserve original type string for domain-specific types
    try:
        rel_type: RelationshipType | str = RelationshipType[inferred.relationship_type]
    except (KeyError, AttributeError):
        rel_type = inferred.relationship_type

    return Relationship(
        id=uuid4(),
        namespace_id=namespace_id,
        source_entity_id=inferred.source_entity_id,
        target_entity_id=inferred.target_entity_id,
        relationship_type=rel_type,
        description=inferred.description,
        properties={
            "inferred": True,
            "rule_name": inferred.rule_name,
            "evidence": [str(e) for e in inferred.evidence],
        },
        source_document_ids=[],
        source_chunk_ids=[],
        confidence=inferred.confidence,
        metadata={"inferred": True},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
