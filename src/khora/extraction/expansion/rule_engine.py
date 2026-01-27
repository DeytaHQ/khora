"""Configurable rule evaluation engine.

Provides a generic rule evaluation system for both correlation rules
and inference rules defined in expertise configurations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from khora.core.models import Entity, Relationship
    from khora.extraction.skills import CorrelationRule, ExpertiseConfig, InferenceRule


@dataclass
class RuleMatch:
    """Result of a rule match."""

    rule_name: str
    matched_value: str | None = None
    matched_entities: list[Entity] = field(default_factory=list)
    matched_relationships: list[Relationship] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleEvaluationContext:
    """Context for rule evaluation containing available data."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    entity_index: dict[str, list[Entity]] = field(default_factory=dict)  # name -> entities
    type_index: dict[str, list[Entity]] = field(default_factory=dict)  # type -> entities
    relationship_index: dict[str, list[Relationship]] = field(default_factory=dict)  # type -> relationships
    # New: Index relationships by entity for fast chain lookups
    rels_by_source: dict[str, list[Relationship]] = field(default_factory=dict)  # entity_id -> outgoing rels
    rels_by_target: dict[str, list[Relationship]] = field(default_factory=dict)  # entity_id -> incoming rels
    entity_by_id: dict[str, Entity] = field(default_factory=dict)  # entity_id -> entity

    @classmethod
    def from_data(
        cls,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> RuleEvaluationContext:
        """Create context with built indices."""
        ctx = cls(entities=entities, relationships=relationships)

        # Build entity indices
        for entity in entities:
            # Index by name (lowercase for matching)
            name_key = entity.name.lower()
            if name_key not in ctx.entity_index:
                ctx.entity_index[name_key] = []
            ctx.entity_index[name_key].append(entity)

            # Index by type
            type_key = str(entity.entity_type.value if hasattr(entity.entity_type, "value") else entity.entity_type)
            if type_key not in ctx.type_index:
                ctx.type_index[type_key] = []
            ctx.type_index[type_key].append(entity)

            # Index by ID for fast lookup
            ctx.entity_by_id[str(entity.id)] = entity

        # Build relationship indices
        for rel in relationships:
            rel_type = str(
                rel.relationship_type.value if hasattr(rel.relationship_type, "value") else rel.relationship_type
            )
            if rel_type not in ctx.relationship_index:
                ctx.relationship_index[rel_type] = []
            ctx.relationship_index[rel_type].append(rel)

            # Index by source entity for fast chain lookups
            source_key = str(rel.source_entity_id)
            if source_key not in ctx.rels_by_source:
                ctx.rels_by_source[source_key] = []
            ctx.rels_by_source[source_key].append(rel)

            # Index by target entity
            target_key = str(rel.target_entity_id)
            if target_key not in ctx.rels_by_target:
                ctx.rels_by_target[target_key] = []
            ctx.rels_by_target[target_key].append(rel)

        return ctx


# Type hierarchy for flexible matching
# Child types can match parent types in inference rules
# e.g., EMPLOYEE matches rules expecting PERSON
TYPE_HIERARCHY: dict[str, list[str]] = {
    "EMPLOYEE": ["PERSON"],
    "EXTERNAL_PERSON": ["PERSON"],
    "COMPANY": ["ORGANIZATION"],
    "DEPARTMENT": ["ORGANIZATION"],
    "TEAM": ["ORGANIZATION"],
    "MESSAGE": ["CONTENT"],
    "PAGE": ["CONTENT", "DOCUMENT"],
    "ISSUE": ["WORK_ITEM", "CONTENT"],
    "SALES_NOTE": ["CONTENT", "NOTE"],
    "CALL": ["EVENT", "MEETING"],
    "DEAL": ["OPPORTUNITY"],
    "PROJECT": ["WORK_ITEM"],
    "CYCLE": ["WORK_ITEM", "SPRINT"],
}


def types_match(actual_type: str, expected_type: str) -> bool:
    """Check if actual type matches expected, considering type hierarchy.

    Args:
        actual_type: The actual entity type from the graph
        expected_type: The expected type from the inference rule

    Returns:
        True if types match directly or via hierarchy
    """
    # Direct match
    if actual_type == expected_type:
        return True

    # Check if actual is a subtype of expected (e.g., EMPLOYEE matches PERSON)
    parents = TYPE_HIERARCHY.get(actual_type, [])
    if expected_type in parents:
        return True

    # Check reverse: if expected is a subtype of actual (e.g., PERSON matches EMPLOYEE)
    # This allows rules written for specific types to match more generic extractions
    parents_of_expected = TYPE_HIERARCHY.get(expected_type, [])
    if actual_type in parents_of_expected:
        return True

    return False


class RuleEngine:
    """Engine for evaluating correlation and inference rules.

    Supports:
    - Pattern-based matching (regex)
    - Field-based matching
    - Multi-condition inference rules
    - Confidence scoring
    - Type hierarchy matching (EMPLOYEE matches rules expecting PERSON)
    """

    def __init__(self, expertise: ExpertiseConfig | None = None) -> None:
        """Initialize the rule engine.

        Args:
            expertise: ExpertiseConfig with rules to evaluate
        """
        self._expertise = expertise
        self._compiled_patterns: dict[str, re.Pattern] = {}

    def evaluate_correlation_rules(
        self,
        text: str,
        context: RuleEvaluationContext,
    ) -> list[RuleMatch]:
        """Evaluate correlation rules against text and context.

        Args:
            text: Text to search for patterns
            context: Evaluation context with entities and relationships

        Returns:
            List of rule matches
        """
        if not self._expertise:
            return []

        matches = []
        for rule in self._expertise.correlation_rules:
            rule_matches = self._evaluate_correlation_rule(rule, text, context)
            matches.extend(rule_matches)

        return matches

    def evaluate_inference_rules(
        self,
        context: RuleEvaluationContext,
    ) -> list[RuleMatch]:
        """Evaluate inference rules against the context.

        Inference rules create new relationships based on existing patterns.

        Args:
            context: Evaluation context with entities and relationships

        Returns:
            List of rule matches that would create new relationships
        """
        if not self._expertise:
            return []

        matches = []
        for rule in self._expertise.inference_rules:
            rule_matches = self._evaluate_inference_rule(rule, context)
            matches.extend(rule_matches)

        return matches

    def find_pattern_matches(
        self,
        pattern: str,
        text: str,
    ) -> list[tuple[str, int, int]]:
        """Find all pattern matches in text.

        Args:
            pattern: Regex pattern to match
            text: Text to search

        Returns:
            List of (matched_value, start, end) tuples
        """
        compiled = self._get_compiled_pattern(pattern)
        if not compiled:
            return []

        matches = []
        for match in compiled.finditer(text):
            matches.append((match.group(), match.start(), match.end()))

        return matches

    def match_entities_by_field(
        self,
        entities: list[Entity],
        field_name: str,
        field_value: Any,
    ) -> list[Entity]:
        """Find entities matching a field value.

        Args:
            entities: Entities to search
            field_name: Attribute field name
            field_value: Value to match

        Returns:
            Matching entities
        """
        matches = []
        for entity in entities:
            entity_value = entity.attributes.get(field_name)
            if entity_value is None:
                continue

            # Normalize for comparison
            if isinstance(entity_value, str) and isinstance(field_value, str):
                if entity_value.lower() == field_value.lower():
                    matches.append(entity)
            elif entity_value == field_value:
                matches.append(entity)

        return matches

    def _evaluate_correlation_rule(
        self,
        rule: CorrelationRule,
        text: str,
        context: RuleEvaluationContext,
    ) -> list[RuleMatch]:
        """Evaluate a single correlation rule."""
        matches = []

        # Pattern-based matching
        if rule.pattern:
            pattern_matches = self.find_pattern_matches(rule.pattern, text)
            for matched_value, start, end in pattern_matches:
                # Find entities that might match this value
                matched_entities = self._find_entities_for_pattern_match(matched_value, rule.entity_types, context)

                matches.append(
                    RuleMatch(
                        rule_name=rule.name,
                        matched_value=matched_value,
                        matched_entities=matched_entities,
                        confidence=rule.confidence,
                        metadata={
                            "start": start,
                            "end": end,
                            "creates_relationship": rule.creates_relationship,
                        },
                    )
                )

        # Field-based matching
        if rule.match_fields:
            # This requires comparing entities against each other
            field_matches = self._find_field_matches(rule, context)
            matches.extend(field_matches)

        return matches

    def _evaluate_inference_rule(
        self,
        rule: InferenceRule,
        context: RuleEvaluationContext,
        *,
        max_matches: int = 1000,
    ) -> list[RuleMatch]:
        """Evaluate a single inference rule.

        Uses indexed lookups for O(1) entity access and O(k) chain lookups
        where k is the number of relationships per entity (typically small).
        """
        if not rule.when or len(rule.when) < 1:
            return []

        matches = []

        # Get relationships matching the first condition
        first_condition = rule.when[0]
        first_rels = context.relationship_index.get(first_condition.relationship, [])

        # Diagnostic: Log how many relationships match before filtering
        before_filter_count = len(first_rels)

        # Filter by source/target type if specified
        first_rels = self._filter_relationships_by_types(
            first_rels, first_condition.source_type, first_condition.target_type, context
        )

        # Diagnostic: Log filtering results for debugging
        if before_filter_count > 0 and len(first_rels) == 0:
            logger.debug(
                f"Rule '{rule.name}': {before_filter_count} {first_condition.relationship} rels "
                f"all filtered out by type filter (need {first_condition.source_type}->{first_condition.target_type})"
            )

        if len(rule.when) == 1:
            # Single condition rule
            for rel in first_rels:
                if len(matches) >= max_matches:
                    break
                source_entity = self._find_entity_by_id(rel.source_entity_id, context)
                target_entity = self._find_entity_by_id(rel.target_entity_id, context)

                if source_entity and target_entity:
                    matches.append(
                        RuleMatch(
                            rule_name=rule.name,
                            matched_relationships=[rel],
                            matched_entities=[source_entity, target_entity],
                            confidence=rule.confidence,
                            metadata={
                                "then_relationship": rule.then_relationship,
                                "then_source": rule.then_source,
                                "then_target": rule.then_target,
                                "first": {"source": source_entity, "target": target_entity},
                            },
                        )
                    )
        else:
            # Multi-condition rule - use indexed lookups for O(k) instead of O(n)
            second_condition = rule.when[1]
            second_rel_type = second_condition.relationship

            for rel1 in first_rels:
                if len(matches) >= max_matches:
                    break

                # Get candidate second relationships connected to rel1's entities
                # Instead of iterating ALL second_rels, only check those connected to rel1
                candidate_rels: list[Relationship] = []

                # Chain pattern: rel1.target -> rel2.source
                target_key = str(rel1.target_entity_id)
                for rel2 in context.rels_by_source.get(target_key, []):
                    rel2_type = str(
                        rel2.relationship_type.value
                        if hasattr(rel2.relationship_type, "value")
                        else rel2.relationship_type
                    )
                    if rel2_type == second_rel_type:
                        candidate_rels.append(rel2)

                # Shared source pattern: rel1.source == rel2.source
                source_key = str(rel1.source_entity_id)
                for rel2 in context.rels_by_source.get(source_key, []):
                    rel2_type = str(
                        rel2.relationship_type.value
                        if hasattr(rel2.relationship_type, "value")
                        else rel2.relationship_type
                    )
                    if rel2_type == second_rel_type and rel2 not in candidate_rels:
                        candidate_rels.append(rel2)

                # Shared target pattern: rel1.target == rel2.target
                for rel2 in context.rels_by_target.get(target_key, []):
                    rel2_type = str(
                        rel2.relationship_type.value
                        if hasattr(rel2.relationship_type, "value")
                        else rel2.relationship_type
                    )
                    if rel2_type == second_rel_type and rel2 not in candidate_rels:
                        candidate_rels.append(rel2)

                # Filter candidates by type
                candidate_rels = self._filter_relationships_by_types(
                    candidate_rels, second_condition.source_type, second_condition.target_type, context
                )

                # Create matches from candidates
                for rel2 in candidate_rels:
                    if len(matches) >= max_matches:
                        break
                    source1 = self._find_entity_by_id(rel1.source_entity_id, context)
                    target1 = self._find_entity_by_id(rel1.target_entity_id, context)
                    source2 = self._find_entity_by_id(rel2.source_entity_id, context)
                    target2 = self._find_entity_by_id(rel2.target_entity_id, context)

                    if all([source1, target1, source2, target2]):
                        matches.append(
                            RuleMatch(
                                rule_name=rule.name,
                                matched_relationships=[rel1, rel2],
                                matched_entities=[source1, target1, source2, target2],
                                confidence=rule.confidence,
                                metadata={
                                    "then_relationship": rule.then_relationship,
                                    "then_source": rule.then_source,
                                    "then_target": rule.then_target,
                                    "first": {"source": source1, "target": target1},
                                    "second": {"source": source2, "target": target2},
                                },
                            )
                        )

        return matches

    def _find_entities_for_pattern_match(
        self,
        matched_value: str,
        entity_types: list[str],
        context: RuleEvaluationContext,
    ) -> list[Entity]:
        """Find entities that might match a pattern value.

        Uses type hierarchy matching so EMPLOYEE matches expected type PERSON.
        """
        candidates = []

        # Check entity names
        name_key = matched_value.lower()
        if name_key in context.entity_index:
            candidates.extend(context.entity_index[name_key])

        # Check entity attributes for the matched value
        for entity in context.entities:
            for attr_value in entity.attributes.values():
                if isinstance(attr_value, str) and matched_value in attr_value:
                    if entity not in candidates:
                        candidates.append(entity)
                    break

        # Filter by entity types if specified, using hierarchy matching
        if entity_types:
            filtered = []
            for e in candidates:
                actual_type = str(e.entity_type.value if hasattr(e.entity_type, "value") else e.entity_type)
                # Check if actual type matches any of the expected types via hierarchy
                if any(types_match(actual_type, expected_type) for expected_type in entity_types):
                    filtered.append(e)
            candidates = filtered

        return candidates

    def _find_field_matches(
        self,
        rule: CorrelationRule,
        context: RuleEvaluationContext,
    ) -> list[RuleMatch]:
        """Find entities that match on specified fields."""
        matches = []

        # Filter entities by type
        candidates = []
        if rule.entity_types:
            for entity_type in rule.entity_types:
                candidates.extend(context.type_index.get(entity_type, []))
        else:
            candidates = context.entities

        # Group entities by field values
        for field_name in rule.match_fields:
            field_groups: dict[Any, list[Entity]] = {}
            for entity in candidates:
                field_value = entity.attributes.get(field_name)
                if field_value:
                    # Normalize string values
                    if isinstance(field_value, str):
                        field_value = field_value.lower()
                    if field_value not in field_groups:
                        field_groups[field_value] = []
                    field_groups[field_value].append(entity)

            # Create matches for groups with multiple entities
            for field_value, entities in field_groups.items():
                if len(entities) > 1:
                    matches.append(
                        RuleMatch(
                            rule_name=rule.name,
                            matched_value=str(field_value),
                            matched_entities=entities,
                            confidence=rule.confidence,
                            metadata={
                                "match_field": field_name,
                                "creates_relationship": rule.creates_relationship,
                            },
                        )
                    )

        return matches

    def _filter_relationships_by_types(
        self,
        relationships: list[Relationship],
        source_type: str | None,
        target_type: str | None,
        context: RuleEvaluationContext,
    ) -> list[Relationship]:
        """Filter relationships by source and target entity types.

        Uses type hierarchy matching so EMPLOYEE matches rules expecting PERSON,
        and vice versa.
        """
        if not source_type and not target_type:
            return relationships

        filtered = []
        for rel in relationships:
            source_entity = self._find_entity_by_id(rel.source_entity_id, context)
            target_entity = self._find_entity_by_id(rel.target_entity_id, context)

            if not source_entity or not target_entity:
                continue

            # Get actual entity types
            actual_source_type = str(
                source_entity.entity_type.value
                if hasattr(source_entity.entity_type, "value")
                else source_entity.entity_type
            )
            actual_target_type = str(
                target_entity.entity_type.value
                if hasattr(target_entity.entity_type, "value")
                else target_entity.entity_type
            )

            # Use hierarchy-aware type matching
            source_matches = not source_type or types_match(actual_source_type, source_type)
            target_matches = not target_type or types_match(actual_target_type, target_type)

            if source_matches and target_matches:
                filtered.append(rel)

        return filtered

    def _find_entity_by_id(
        self,
        entity_id: Any,
        context: RuleEvaluationContext,
    ) -> Entity | None:
        """Find entity by ID in context using indexed lookup."""
        return context.entity_by_id.get(str(entity_id))

    def _relationships_connect(
        self,
        rel1: Relationship,
        rel2: Relationship,
        context: RuleEvaluationContext,
    ) -> bool:
        """Check if two relationships connect (share an entity)."""
        # Check if rel1's target is rel2's source (chain pattern)
        if rel1.target_entity_id == rel2.source_entity_id:
            return True
        # Check if they share source or target
        if rel1.source_entity_id == rel2.source_entity_id:
            return True
        if rel1.target_entity_id == rel2.target_entity_id:
            return True
        return False

    def _get_compiled_pattern(self, pattern: str) -> re.Pattern | None:
        """Get or compile a regex pattern."""
        if pattern not in self._compiled_patterns:
            try:
                self._compiled_patterns[pattern] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                return None

        return self._compiled_patterns[pattern]
