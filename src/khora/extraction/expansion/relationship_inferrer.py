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


class RelationshipInferrer:
    """Infers new relationships based on existing graph patterns.

    Uses inference rules from expertise configuration to deduce
    relationships that aren't explicitly stated but can be inferred
    from existing relationships.

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
        max_inferences_per_rule: int = 100,
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

    def infer(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        *,
        depth: int = 1,
    ) -> list[InferredRelationship]:
        """Infer new relationships from existing graph.

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

        # Diagnostic: Log entity types in the graph
        entity_types = Counter(
            e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type) for e in entities
        )
        logger.info(f"Inference input: {len(entities)} entities, types: {dict(entity_types)}")

        # Diagnostic: Log relationship types in the graph
        rel_types = Counter(
            r.relationship_type.value if hasattr(r.relationship_type, "value") else str(r.relationship_type)
            for r in relationships
        )
        logger.info(f"Inference input: {len(relationships)} relationships, types: {dict(rel_types)}")

        # Diagnostic: Log source/target entity types for each relationship type
        # Build entity ID to type lookup
        entity_type_lookup = {
            e.id: (e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)) for e in entities
        }

        # For each relationship type, count source->target type combinations
        rel_type_patterns: dict[str, Counter] = {}
        for r in relationships:
            rel_type = r.relationship_type.value if hasattr(r.relationship_type, "value") else str(r.relationship_type)
            source_type = entity_type_lookup.get(r.source_entity_id, "UNKNOWN")
            target_type = entity_type_lookup.get(r.target_entity_id, "UNKNOWN")
            pattern = f"{source_type}->{target_type}"

            if rel_type not in rel_type_patterns:
                rel_type_patterns[rel_type] = Counter()
            rel_type_patterns[rel_type][pattern] += 1

        # Log top patterns for relationship types that rules expect
        for rel_type, patterns in rel_type_patterns.items():
            top_patterns = patterns.most_common(5)
            logger.debug(f"  {rel_type} patterns: {dict(top_patterns)}")

        # Diagnostic: Log what inference rules expect
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
        logger.info(f"Inference rules expect relationships: {expected_rels}")
        logger.info(f"Inference rules expect entity types: {expected_entities}")

        # Diagnostic: Check for matches between actual and expected
        actual_rel_types = set(rel_types.keys())
        actual_entity_types = set(entity_types.keys())
        matching_rels = actual_rel_types & expected_rels
        matching_entities = actual_entity_types & expected_entities
        logger.info(f"Matching relationship types: {matching_rels or 'NONE'}")
        logger.info(f"Matching entity types: {matching_entities or 'NONE'}")

        if not matching_rels:
            logger.warning(
                f"No relationship type matches! Rules expect {expected_rels} " f"but graph has {actual_rel_types}"
            )
        if not matching_entities:
            logger.warning(
                f"No entity type matches! Rules expect {expected_entities} " f"but graph has {actual_entity_types}"
            )

        all_inferred: list[InferredRelationship] = []
        current_relationships = list(relationships)

        for pass_num in range(depth):
            # Build context with current state
            context = RuleEvaluationContext.from_data(entities, current_relationships)

            # Evaluate inference rules
            matches = self._rule_engine.evaluate_inference_rules(context)
            logger.info(f"Pass {pass_num + 1}: Rule engine returned {len(matches)} matches")

            # Log details of first few matches for debugging
            for i, match in enumerate(matches[:5]):
                logger.info(
                    f"  Match {i + 1}: rule={match.rule_name}, "
                    f"confidence={match.confidence:.2f}, "
                    f"metadata_keys={list(match.metadata.keys())}"
                )

            # Convert matches to inferred relationships
            pass_inferred = self._matches_to_relationships(matches, context)
            logger.info(f"Pass {pass_num + 1}: Converted to {len(pass_inferred)} inferred relationships")

            if not pass_inferred:
                # No new inferences, stop early
                logger.info(f"Pass {pass_num + 1}: No new inferences, stopping early")
                break

            # Filter out duplicates and already existing relationships
            new_inferred = self._filter_duplicates(pass_inferred, current_relationships, all_inferred)

            if not new_inferred:
                break

            all_inferred.extend(new_inferred)

            # Add inferred to current for next pass (as mock relationships)
            current_relationships.extend(self._to_mock_relationships(new_inferred, entities))

            logger.info(f"Inference pass {pass_num + 1}: {len(new_inferred)} new relationships")

        logger.info(f"Inference complete: {len(all_inferred)} total relationships inferred")
        return all_inferred

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

    def _matches_to_relationships(
        self,
        matches: list[RuleMatch],
        context: RuleEvaluationContext,
    ) -> list[InferredRelationship]:
        """Convert rule matches to inferred relationships."""
        inferred = []
        rule_counts: dict[str, int] = {}

        for match in matches:
            # Check rule limit
            rule_counts[match.rule_name] = rule_counts.get(match.rule_name, 0) + 1
            if rule_counts[match.rule_name] > self._max_inferences_per_rule:
                continue

            # Check confidence threshold
            if match.confidence < self._min_confidence:
                continue

            # Resolve source and target from metadata
            source_entity, target_entity = self._resolve_inference_entities(match, context)

            if not source_entity or not target_entity:
                continue

            # Skip self-referential
            if source_entity.id == target_entity.id:
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
            data = first if group == "first" else second

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
