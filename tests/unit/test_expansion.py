"""Unit tests for semantic expansion components."""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.extraction.expansion import (
    CrossToolUnifier,
    RelationshipInferrer,
    SemanticExpander,
)
from khora.extraction.expansion.rule_engine import (
    RuleEngine,
    RuleEvaluationContext,
)
from khora.extraction.skills import (
    CorrelationRule,
    ExpertiseConfig,
    InferenceCondition,
    InferenceRule,
)


class TestRuleEvaluationContext:
    """Tests for RuleEvaluationContext."""

    def test_empty_context(self) -> None:
        """Test creating empty context."""
        ctx = RuleEvaluationContext()
        assert ctx.entities == []
        assert ctx.relationships == []
        assert ctx.entity_index == {}
        assert ctx.type_index == {}

    def test_from_data_builds_indices(self) -> None:
        """Test context builds indices from data."""
        namespace_id = uuid4()
        entities = [
            Entity(
                id=uuid4(),
                name="John Smith",
                entity_type="PERSON",
                namespace_id=namespace_id,
            ),
            Entity(
                id=uuid4(),
                name="Acme Corp",
                entity_type="ORGANIZATION",
                namespace_id=namespace_id,
            ),
            Entity(
                id=uuid4(),
                name="Jane Doe",
                entity_type="PERSON",
                namespace_id=namespace_id,
            ),
        ]

        ctx = RuleEvaluationContext.from_data(entities, [])

        # Check entity index (by name, lowercase)
        assert "john smith" in ctx.entity_index
        assert "acme corp" in ctx.entity_index
        assert len(ctx.entity_index["john smith"]) == 1

        # Check type index
        assert "PERSON" in ctx.type_index
        assert "ORGANIZATION" in ctx.type_index
        assert len(ctx.type_index["PERSON"]) == 2
        assert len(ctx.type_index["ORGANIZATION"]) == 1

    def test_from_data_builds_relationship_index(self) -> None:
        """Test context builds relationship index."""
        namespace_id = uuid4()
        e1_id = uuid4()
        e2_id = uuid4()

        relationships = [
            Relationship(
                id=uuid4(),
                source_entity_id=e1_id,
                target_entity_id=e2_id,
                relationship_type="WORKS_FOR",
                namespace_id=namespace_id,
            ),
            Relationship(
                id=uuid4(),
                source_entity_id=e2_id,
                target_entity_id=e1_id,
                relationship_type="EMPLOYS",
                namespace_id=namespace_id,
            ),
        ]

        ctx = RuleEvaluationContext.from_data([], relationships)

        assert "WORKS_FOR" in ctx.relationship_index
        assert "EMPLOYS" in ctx.relationship_index
        assert len(ctx.relationship_index["WORKS_FOR"]) == 1


class TestRuleEngine:
    """Tests for RuleEngine."""

    def test_engine_without_expertise(self) -> None:
        """Test engine without expertise returns empty matches."""
        engine = RuleEngine()
        ctx = RuleEvaluationContext()

        correlation_matches = engine.evaluate_correlation_rules("test text", ctx)
        inference_matches = engine.evaluate_inference_rules(ctx)

        assert correlation_matches == []
        assert inference_matches == []

    def test_find_pattern_matches(self) -> None:
        """Test finding regex pattern matches."""
        engine = RuleEngine()

        text = "Working on PROJ-123 and TEAM-456 today"
        matches = engine.find_pattern_matches(r"[A-Z]+-\d+", text)

        assert len(matches) == 2
        assert matches[0][0] == "PROJ-123"
        assert matches[1][0] == "TEAM-456"

    def test_find_pattern_matches_with_positions(self) -> None:
        """Test pattern matches include positions."""
        engine = RuleEngine()

        text = "Issue ABC-1"
        matches = engine.find_pattern_matches(r"[A-Z]+-\d+", text)

        assert len(matches) == 1
        matched_value, start, end = matches[0]
        assert matched_value == "ABC-1"
        assert text[start:end] == "ABC-1"

    def test_find_pattern_matches_invalid_regex(self) -> None:
        """Test invalid regex returns empty list."""
        engine = RuleEngine()

        matches = engine.find_pattern_matches(r"[invalid", "test")
        assert matches == []

    def test_match_entities_by_field(self) -> None:
        """Test matching entities by field value."""
        namespace_id = uuid4()
        entities = [
            Entity(
                id=uuid4(),
                name="John",
                entity_type="PERSON",
                namespace_id=namespace_id,
                attributes={"email": "john@example.com"},
            ),
            Entity(
                id=uuid4(),
                name="Jane",
                entity_type="PERSON",
                namespace_id=namespace_id,
                attributes={"email": "jane@example.com"},
            ),
            Entity(
                id=uuid4(),
                name="John Copy",
                entity_type="PERSON",
                namespace_id=namespace_id,
                attributes={"email": "john@example.com"},
            ),
        ]

        engine = RuleEngine()
        matches = engine.match_entities_by_field(entities, "email", "john@example.com")

        assert len(matches) == 2

    def test_match_entities_case_insensitive(self) -> None:
        """Test field matching is case insensitive for strings."""
        namespace_id = uuid4()
        entities = [
            Entity(
                id=uuid4(),
                name="Test",
                entity_type="PERSON",
                namespace_id=namespace_id,
                attributes={"domain": "EXAMPLE.COM"},
            ),
        ]

        engine = RuleEngine()
        matches = engine.match_entities_by_field(entities, "domain", "example.com")

        assert len(matches) == 1

    def test_evaluate_correlation_rules_pattern(self) -> None:
        """Test evaluating correlation rules with patterns."""
        expertise = ExpertiseConfig(
            name="test",
            correlation_rules=[
                CorrelationRule(
                    name="issue_ref",
                    pattern=r"[A-Z]+-\d+",
                    creates_relationship="REFERENCES",
                    confidence=0.9,
                ),
            ],
        )

        engine = RuleEngine(expertise=expertise)
        ctx = RuleEvaluationContext()

        matches = engine.evaluate_correlation_rules("See PROJ-123 for details", ctx)

        assert len(matches) == 1
        assert matches[0].rule_name == "issue_ref"
        assert matches[0].matched_value == "PROJ-123"
        assert matches[0].confidence == 0.9

    def test_evaluate_correlation_rules_field_matching(self) -> None:
        """Test evaluating correlation rules with field matching."""
        namespace_id = uuid4()
        entities = [
            Entity(
                id=uuid4(),
                name="Person A",
                entity_type="PERSON",
                namespace_id=namespace_id,
                attributes={"email": "shared@example.com"},
            ),
            Entity(
                id=uuid4(),
                name="Person B",
                entity_type="PERSON",
                namespace_id=namespace_id,
                attributes={"email": "shared@example.com"},
            ),
        ]

        expertise = ExpertiseConfig(
            name="test",
            correlation_rules=[
                CorrelationRule(
                    name="email_match",
                    match_fields=["email"],
                    entity_types=["PERSON"],
                    creates_relationship="SAME_AS",
                ),
            ],
        )

        engine = RuleEngine(expertise=expertise)
        ctx = RuleEvaluationContext.from_data(entities, [])

        matches = engine.evaluate_correlation_rules("", ctx)

        assert len(matches) == 1
        assert matches[0].rule_name == "email_match"
        assert len(matches[0].matched_entities) == 2


class TestCrossToolUnifier:
    """Tests for CrossToolUnifier."""

    async def test_unifier_without_expertise(self) -> None:
        """Test unifier works without expertise."""
        unifier = CrossToolUnifier()
        namespace_id = uuid4()

        entities = [
            Entity(
                id=uuid4(),
                name="Test",
                entity_type="PERSON",
                namespace_id=namespace_id,
            ),
        ]

        result = await unifier.unify(entities, [], use_embeddings=False, use_fuzzy=False)

        assert len(result.unified_entities) == 1
        assert result.entities_merged == 0

    async def test_unify_by_email(self) -> None:
        """Test unifying entities by email with correlation rule."""
        # Expertise with email matching rule
        expertise = ExpertiseConfig(
            name="test",
            correlation_rules=[
                CorrelationRule(
                    name="email_match",
                    match_fields=["email"],
                    entity_types=["PERSON"],
                ),
            ],
        )
        unifier = CrossToolUnifier(expertise=expertise)
        namespace_id = uuid4()

        e1 = Entity(
            id=uuid4(),
            name="John Smith",
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"email": "john@example.com", "source": "slack"},
        )
        e2 = Entity(
            id=uuid4(),
            name="J. Smith",
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"email": "john@example.com", "source": "jira"},
        )
        e3 = Entity(
            id=uuid4(),
            name="Jane Doe",
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"email": "jane@example.com"},
        )

        result = await unifier.unify([e1, e2, e3], [], use_embeddings=False, use_fuzzy=False)

        # e1 and e2 should be merged, e3 separate
        assert len(result.unified_entities) == 2
        assert result.entities_merged == 1
        assert len(result.entity_mapping) == 3

    async def test_unify_by_domain(self) -> None:
        """Test unifying entities by domain with correlation rule."""
        # Expertise with domain matching rule
        expertise = ExpertiseConfig(
            name="test",
            correlation_rules=[
                CorrelationRule(
                    name="domain_match",
                    match_fields=["domain"],
                    entity_types=["CUSTOMER"],
                ),
            ],
        )
        unifier = CrossToolUnifier(expertise=expertise)
        namespace_id = uuid4()

        e1 = Entity(
            id=uuid4(),
            name="Acme Corp",
            entity_type="CUSTOMER",
            namespace_id=namespace_id,
            attributes={"domain": "acme.com"},
        )
        e2 = Entity(
            id=uuid4(),
            name="Acme Corporation",
            entity_type="CUSTOMER",
            namespace_id=namespace_id,
            attributes={"domain": "acme.com"},
        )

        result = await unifier.unify([e1, e2], [], use_embeddings=False, use_fuzzy=False)

        assert len(result.unified_entities) == 1
        assert result.entities_merged == 1

    async def test_unify_updates_relationships(self) -> None:
        """Test that unification updates relationship entity IDs."""
        # Expertise with email matching rule
        expertise = ExpertiseConfig(
            name="test",
            correlation_rules=[
                CorrelationRule(
                    name="email_match",
                    match_fields=["email"],
                    entity_types=["PERSON"],
                ),
            ],
        )
        unifier = CrossToolUnifier(expertise=expertise)
        namespace_id = uuid4()

        e1 = Entity(
            id=uuid4(),
            name="John",
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"email": "john@example.com"},
        )
        e2 = Entity(
            id=uuid4(),
            name="John Smith",
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"email": "john@example.com"},
        )
        e3 = Entity(
            id=uuid4(),
            name="Acme",
            entity_type="ORGANIZATION",
            namespace_id=namespace_id,
        )

        # Relationship from e2 to e3
        rel = Relationship(
            id=uuid4(),
            source_entity_id=e2.id,
            target_entity_id=e3.id,
            relationship_type="WORKS_FOR",
            namespace_id=namespace_id,
        )

        result = await unifier.unify([e1, e2, e3], [rel], use_embeddings=False, use_fuzzy=False)

        # e1 and e2 merged, relationship should be updated
        assert len(result.unified_entities) == 2
        assert len(result.updated_relationships) == 1

        # Relationship source should now point to canonical entity
        updated_rel = result.updated_relationships[0]
        canonical_id = result.entity_mapping[e2.id]
        assert updated_rel.source_entity_id == canonical_id

    async def test_unify_with_fuzzy_matching(self) -> None:
        """Test unifying with fuzzy string matching."""
        unifier = CrossToolUnifier(fuzzy_threshold=0.8)
        namespace_id = uuid4()

        e1 = Entity(
            id=uuid4(),
            name="John Smith",
            entity_type="PERSON",
            namespace_id=namespace_id,
        )
        e2 = Entity(
            id=uuid4(),
            name="Jon Smith",  # Typo
            entity_type="PERSON",
            namespace_id=namespace_id,
        )

        result = await unifier.unify([e1, e2], [], use_embeddings=False, use_fuzzy=True)

        # Names are similar enough to merge
        assert len(result.unified_entities) == 1
        assert result.entities_merged == 1


class TestCrossToolUnifierMergeEvents:
    """Tests for ENTITY_MERGED hook event emission (Issue #576, Phase 1, Item 5)."""

    async def test_merge_dispatches_single_entity_merged_event(self) -> None:
        """A merge of 2 entities dispatches exactly one entity.merged event with merged_from length 2."""
        from unittest.mock import AsyncMock

        from khora.core.models.event import EventType

        unifier = CrossToolUnifier()
        namespace_id = uuid4()

        e1 = Entity(
            id=uuid4(),
            name="John Smith",
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"source_tool": "slack"},
        )
        e2 = Entity(
            id=uuid4(),
            name="John Smith",  # Exact name + type match
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"source_tool": "salesforce"},
        )

        storage = AsyncMock()

        result = await unifier.unify([e1, e2], [], use_embeddings=False, use_fuzzy=False, storage=storage)

        assert result.entities_merged == 1
        assert storage.dispatch_hook.await_count == 1

        event = storage.dispatch_hook.await_args.args[0]
        assert event.event_type == EventType.ENTITY_MERGED
        assert event.resource_type == "entity"
        assert len(event.data["merged_from"]) == 2
        assert set(event.data["merged_from"]) == {str(e1.id), str(e2.id)}
        assert set(event.data["source_tools"]) == {"slack", "salesforce"}
        assert event.data["strategy"] == "name_match"
        assert event.data["namespace_id"] == str(namespace_id)

    async def test_merge_event_surviving_id_matches_canonical_entity(self) -> None:
        """The event's surviving_id matches the canonical (post-merge) entity id."""
        from unittest.mock import AsyncMock

        unifier = CrossToolUnifier()
        namespace_id = uuid4()

        e1 = Entity(
            id=uuid4(),
            name="Acme Corp",
            entity_type="ORGANIZATION",
            namespace_id=namespace_id,
            confidence=0.7,
        )
        e2 = Entity(
            id=uuid4(),
            name="Acme Corp",
            entity_type="ORGANIZATION",
            namespace_id=namespace_id,
            confidence=0.9,  # Higher confidence -> becomes base
        )

        storage = AsyncMock()

        result = await unifier.unify([e1, e2], [], use_embeddings=False, use_fuzzy=False, storage=storage)

        assert storage.dispatch_hook.await_count == 1
        event = storage.dispatch_hook.await_args.args[0]
        # The surviving entity is the one in the unified list
        canonical = result.unified_entities[0]
        assert event.resource_id == canonical.id
        assert event.data["surviving_id"] == str(canonical.id)

    async def test_no_merge_does_not_dispatch_event(self) -> None:
        """When no merge happens, no entity.merged event is dispatched."""
        from unittest.mock import AsyncMock

        unifier = CrossToolUnifier()
        namespace_id = uuid4()

        e1 = Entity(
            id=uuid4(),
            name="John Smith",
            entity_type="PERSON",
            namespace_id=namespace_id,
        )
        e2 = Entity(
            id=uuid4(),
            name="Jane Doe",  # Distinct name -> no merge
            entity_type="PERSON",
            namespace_id=namespace_id,
        )

        storage = AsyncMock()

        result = await unifier.unify([e1, e2], [], use_embeddings=False, use_fuzzy=False, storage=storage)

        assert result.entities_merged == 0
        storage.dispatch_hook.assert_not_awaited()


class TestRelationshipInferrer:
    """Tests for RelationshipInferrer."""

    def test_inferrer_without_expertise(self) -> None:
        """Test inferrer without expertise returns empty."""
        inferrer = RelationshipInferrer()
        result = inferrer.infer([], [])
        assert result == []

    def test_inferrer_with_empty_rules(self) -> None:
        """Test inferrer with expertise but no rules."""
        expertise = ExpertiseConfig(name="test")
        inferrer = RelationshipInferrer(expertise=expertise)
        result = inferrer.infer([], [])
        assert result == []

    def test_single_condition_inference(self) -> None:
        """Test inference with single condition rule."""
        namespace_id = uuid4()

        person = Entity(
            id=uuid4(),
            name="John",
            entity_type="PERSON",
            namespace_id=namespace_id,
        )
        project = Entity(
            id=uuid4(),
            name="Project Alpha",
            entity_type="PROJECT",
            namespace_id=namespace_id,
        )

        owns_rel = Relationship(
            id=uuid4(),
            source_entity_id=person.id,
            target_entity_id=project.id,
            relationship_type="OWNS",
            namespace_id=namespace_id,
        )

        expertise = ExpertiseConfig(
            name="test",
            inference_rules=[
                InferenceRule(
                    name="owner_is_stakeholder",
                    when=[
                        InferenceCondition(
                            relationship="OWNS",
                            source_type="PERSON",
                            target_type="PROJECT",
                        ),
                    ],
                    then_relationship="STAKEHOLDER_OF",
                    then_source="first.source",
                    then_target="first.target",
                    confidence=0.8,
                ),
            ],
        )

        inferrer = RelationshipInferrer(expertise=expertise, min_confidence=0.5)
        inferred = inferrer.infer([person, project], [owns_rel], depth=1)

        assert len(inferred) == 1
        assert inferred[0].relationship_type == "STAKEHOLDER_OF"
        assert inferred[0].source_entity_id == person.id
        assert inferred[0].target_entity_id == project.id
        assert inferred[0].confidence == 0.8

    def test_confidence_filtering(self) -> None:
        """Test that low confidence inferences are filtered."""
        namespace_id = uuid4()

        e1 = Entity(id=uuid4(), name="A", entity_type="TYPE", namespace_id=namespace_id)
        e2 = Entity(id=uuid4(), name="B", entity_type="TYPE", namespace_id=namespace_id)

        rel = Relationship(
            id=uuid4(),
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relationship_type="REL",
            namespace_id=namespace_id,
        )

        expertise = ExpertiseConfig(
            name="test",
            inference_rules=[
                InferenceRule(
                    name="low_confidence",
                    when=[InferenceCondition(relationship="REL")],
                    then_relationship="INFERRED",
                    confidence=0.2,  # Below threshold
                ),
            ],
        )

        inferrer = RelationshipInferrer(expertise=expertise, min_confidence=0.5)
        inferred = inferrer.infer([e1, e2], [rel])

        assert len(inferred) == 0

    def test_no_duplicate_inference(self) -> None:
        """Test that existing relationships aren't re-inferred."""
        namespace_id = uuid4()

        e1 = Entity(id=uuid4(), name="A", entity_type="TYPE", namespace_id=namespace_id)
        e2 = Entity(id=uuid4(), name="B", entity_type="TYPE", namespace_id=namespace_id)

        # Original relationship
        rel = Relationship(
            id=uuid4(),
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relationship_type="OWNS",
            namespace_id=namespace_id,
        )

        # Relationship that would be inferred (already exists)
        existing = Relationship(
            id=uuid4(),
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relationship_type="STAKEHOLDER_OF",
            namespace_id=namespace_id,
        )

        expertise = ExpertiseConfig(
            name="test",
            inference_rules=[
                InferenceRule(
                    name="test",
                    when=[InferenceCondition(relationship="OWNS")],
                    then_relationship="STAKEHOLDER_OF",
                    confidence=0.8,
                ),
            ],
        )

        inferrer = RelationshipInferrer(expertise=expertise)
        inferred = inferrer.infer([e1, e2], [rel, existing])

        # Should not create duplicate
        assert len(inferred) == 0


class TestSemanticExpander:
    """Tests for SemanticExpander."""

    @pytest.mark.asyncio
    async def test_expander_no_entities(self) -> None:
        """Test expander with no entities."""
        expander = SemanticExpander()
        result = await expander.expand([], [])

        assert result.total_entities == 0
        assert result.total_relationships == 0
        assert result.original_entity_count == 0

    @pytest.mark.asyncio
    async def test_expander_passthrough(self) -> None:
        """Test expander passes through entities when disabled."""
        namespace_id = uuid4()
        entities = [
            Entity(id=uuid4(), name="Test", entity_type="PERSON", namespace_id=namespace_id),
        ]

        expander = SemanticExpander(
            enable_unification=False,
            enable_inference=False,
        )
        result = await expander.expand(entities, [])

        assert result.total_entities == 1
        assert result.original_entity_count == 1
        assert result.merged_entity_count == 0

    @pytest.mark.asyncio
    async def test_expander_with_unification(self) -> None:
        """Test expander performs unification with expertise."""
        namespace_id = uuid4()

        e1 = Entity(
            id=uuid4(),
            name="John",
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"email": "john@test.com"},
        )
        e2 = Entity(
            id=uuid4(),
            name="John Smith",
            entity_type="PERSON",
            namespace_id=namespace_id,
            attributes={"email": "john@test.com"},
        )

        # Need expertise with correlation rules for email matching
        expertise = ExpertiseConfig(
            name="test",
            correlation_rules=[
                CorrelationRule(name="email_match", match_fields=["email"], entity_types=["PERSON"]),
            ],
        )

        expander = SemanticExpander(
            expertise=expertise,
            enable_unification=True,
            enable_inference=False,
        )
        result = await expander.expand([e1, e2], [])

        assert result.original_entity_count == 2
        assert result.total_entities == 1
        assert result.merged_entity_count == 1

    @pytest.mark.asyncio
    async def test_expander_with_expertise(self) -> None:
        """Test expander uses expertise configuration."""
        from khora.extraction.skills import ExpansionConfig

        expertise = ExpertiseConfig(
            name="test",
            expansion=ExpansionConfig(
                enabled=True,
                cross_tool_unification=True,
                relationship_inference=False,
                depth=1,
            ),
        )

        namespace_id = uuid4()
        entities = [
            Entity(id=uuid4(), name="Test", entity_type="PERSON", namespace_id=namespace_id),
        ]

        expander = SemanticExpander(expertise=expertise)
        result = await expander.expand(entities, [])

        assert result.total_entities == 1

    @pytest.mark.asyncio
    async def test_expander_sync_alternative(self) -> None:
        """Test expand method directly instead of deprecated sync version."""
        namespace_id = uuid4()
        entities = [
            Entity(id=uuid4(), name="Test", entity_type="PERSON", namespace_id=namespace_id),
        ]

        expander = SemanticExpander(
            enable_unification=False,
            enable_inference=False,
        )
        result = await expander.expand(entities, [])

        assert result.total_entities == 1

    def test_from_expertise(self) -> None:
        """Test creating expander from expertise config."""
        from khora.extraction.skills import ExpansionConfig

        expertise = ExpertiseConfig(
            name="test",
            expansion=ExpansionConfig(
                enabled=True,
                depth=3,
                cross_tool_unification=True,
                relationship_inference=True,
            ),
        )

        expander = SemanticExpander.from_expertise(expertise)

        assert expander._enable_unification is True
        assert expander._enable_inference is True
        assert expander._inference_depth == 3

    def test_from_expertise_name(self) -> None:
        """Test creating expander from expertise name."""
        expander = SemanticExpander.from_expertise_name("general")
        assert expander._expertise is not None
        assert expander._expertise.name == "general"

    @pytest.mark.asyncio
    async def test_expansion_result_properties(self) -> None:
        """Test ExpansionResult computed properties."""
        namespace_id = uuid4()

        e1 = Entity(id=uuid4(), name="A", entity_type="PERSON", namespace_id=namespace_id)
        e2 = Entity(id=uuid4(), name="B", entity_type="PERSON", namespace_id=namespace_id)

        rel = Relationship(
            id=uuid4(),
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relationship_type="KNOWS",
            namespace_id=namespace_id,
        )

        expander = SemanticExpander(
            enable_unification=False,
            enable_inference=False,
        )
        result = await expander.expand([e1, e2], [rel])

        assert result.total_entities == 2
        assert result.total_relationships == 1
        assert len(result.all_relationships) == 1
