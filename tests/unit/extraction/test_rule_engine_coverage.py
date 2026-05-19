"""Coverage tests for ``khora.extraction.expansion.rule_engine``.

Targets uncovered branches:
- ``RuleEvaluationContext.update`` incremental adds (105-138)
- ``types_match`` hierarchy paths (176-186)
- ``find_pattern_matches`` with invalid regex (301, 698-705)
- ``match_entities_by_field`` non-string and missing-attr (307-308)
- ``_evaluate_inference_rule`` empty `when` (363)
- ``_evaluate_inference_rule`` multi-condition chain walking (424-515)
- ``_resolve_entity_ref`` malformed / missing (671)
- ``_relationships_connect`` various patterns (688-695)
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.extraction.expansion.rule_engine import (
    RuleEngine,
    RuleEvaluationContext,
    types_match,
)
from khora.extraction.skills import (
    CorrelationRule,
    ExpertiseConfig,
    InferenceCondition,
    InferenceRule,
)


def _make_entity(name: str, etype: str = "PERSON", ns_id=None, **kw) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns_id or uuid4(),
        name=name,
        entity_type=etype,
        attributes=kw.get("attributes", {}),
    )


def _make_rel(source: Entity, target: Entity, rel_type: str = "WORKS_FOR") -> Relationship:
    return Relationship(
        id=uuid4(),
        namespace_id=source.namespace_id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        relationship_type=rel_type,
    )


# ---------------------------------------------------------------------------
# types_match — hierarchy paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypesMatch:
    def test_exact_match(self) -> None:
        assert types_match("PERSON", "PERSON")

    def test_child_to_parent(self) -> None:
        # EMPLOYEE has parent PERSON
        assert types_match("EMPLOYEE", "PERSON")

    def test_parent_to_child_reverse(self) -> None:
        # Reverse: rule says EMPLOYEE but actual is PERSON
        assert types_match("PERSON", "EMPLOYEE")

    def test_no_match(self) -> None:
        assert not types_match("PERSON", "LOCATION")

    def test_unknown_types_no_match(self) -> None:
        assert not types_match("FOO", "BAR")


# ---------------------------------------------------------------------------
# RuleEvaluationContext.update
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContextUpdate:
    def test_update_adds_new_entities(self) -> None:
        e1 = _make_entity("Alice")
        ctx = RuleEvaluationContext.from_data([e1], [])

        e2 = _make_entity("Bob")
        ctx.update([e2], [])

        assert e2 in ctx.entities
        assert "bob" in ctx.entity_index
        assert e2 in ctx.type_index["PERSON"]
        assert str(e2.id) in ctx.entity_by_id

    def test_update_skips_duplicates(self) -> None:
        e1 = _make_entity("Alice")
        ctx = RuleEvaluationContext.from_data([e1], [])
        initial_count = len(ctx.entities)
        ctx.update([e1], [])
        assert len(ctx.entities) == initial_count

    def test_update_adds_relationships(self) -> None:
        e1 = _make_entity("Alice")
        e2 = _make_entity("Acme", "ORGANIZATION")
        ctx = RuleEvaluationContext.from_data([e1, e2], [])
        r = _make_rel(e1, e2)
        ctx.update([], [r])

        assert r in ctx.relationships
        assert "WORKS_FOR" in ctx.relationship_index
        assert r in ctx.rels_by_source[str(e1.id)]
        assert r in ctx.rels_by_target[str(e2.id)]


# ---------------------------------------------------------------------------
# RuleEngine.find_pattern_matches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPatternMatching:
    def test_find_pattern_matches_basic(self) -> None:
        engine = RuleEngine()
        matches = engine.find_pattern_matches(r"\bfoo\b", "foo bar foo")
        assert len(matches) == 2

    def test_find_pattern_matches_invalid_regex(self) -> None:
        engine = RuleEngine()
        # Unbalanced parens — re.error logged, returns []
        result = engine.find_pattern_matches("[invalid", "anything")
        assert result == []

    def test_get_compiled_pattern_caches(self) -> None:
        engine = RuleEngine()
        engine.find_pattern_matches(r"\d+", "abc 123")
        # Cache populated
        assert r"\d+" in engine._compiled_patterns
        # Second call hits cache
        engine.find_pattern_matches(r"\d+", "abc 456")


# ---------------------------------------------------------------------------
# RuleEngine.match_entities_by_field
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMatchEntitiesByField:
    def test_string_match_case_insensitive(self) -> None:
        engine = RuleEngine()
        e1 = _make_entity("Alice", attributes={"email": "alice@x.com"})
        e2 = _make_entity("Bob", attributes={"email": "BOB@X.COM"})
        e3 = _make_entity("Carol", attributes={})
        matches = engine.match_entities_by_field([e1, e2, e3], "email", "bob@x.com")
        assert matches == [e2]

    def test_non_string_equality(self) -> None:
        engine = RuleEngine()
        e1 = _make_entity("X", attributes={"count": 5})
        e2 = _make_entity("Y", attributes={"count": 3})
        matches = engine.match_entities_by_field([e1, e2], "count", 5)
        assert matches == [e1]

    def test_missing_field_skipped(self) -> None:
        engine = RuleEngine()
        e1 = _make_entity("X", attributes={})
        matches = engine.match_entities_by_field([e1], "email", "any@x.com")
        assert matches == []


# ---------------------------------------------------------------------------
# RuleEngine without expertise / empty when
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluateNoExpertise:
    def test_no_expertise_returns_empty(self) -> None:
        engine = RuleEngine()
        ctx = RuleEvaluationContext()
        assert engine.evaluate_correlation_rules("text", ctx) == []
        assert engine.evaluate_inference_rules(ctx) == []

    def test_empty_when_returns_empty(self) -> None:
        rule = InferenceRule(
            name="empty",
            when=[],
            then_relationship="X",
            then_source="first.source",
            then_target="first.target",
        )
        expertise = ExpertiseConfig(name="t", inference_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext()
        assert engine.evaluate_inference_rules(ctx) == []


# ---------------------------------------------------------------------------
# Single-condition inference rule
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleConditionInference:
    def test_single_condition_match(self) -> None:
        ns = uuid4()
        alice = _make_entity("Alice", "PERSON", ns_id=ns)
        acme = _make_entity("Acme", "ORGANIZATION", ns_id=ns)
        rel = _make_rel(alice, acme, "WORKS_FOR")

        rule = InferenceRule(
            name="works-implies-knows",
            when=[InferenceCondition(relationship="WORKS_FOR")],
            then_relationship="ASSOCIATED_WITH",
            then_source="first.source",
            then_target="first.target",
            confidence=0.7,
        )
        expertise = ExpertiseConfig(name="t", inference_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([alice, acme], [rel])

        matches = engine.evaluate_inference_rules(ctx)
        assert len(matches) == 1
        assert matches[0].confidence == 0.7
        assert matches[0].rule_name == "works-implies-knows"

    def test_single_condition_self_reference_skipped(self) -> None:
        ns = uuid4()
        alice = _make_entity("Alice", "PERSON", ns_id=ns)
        # Self-relationship
        rel = Relationship(
            id=uuid4(),
            namespace_id=ns,
            source_entity_id=alice.id,
            target_entity_id=alice.id,
            relationship_type="LIKES",
        )

        rule = InferenceRule(
            name="self",
            when=[InferenceCondition(relationship="LIKES")],
            then_relationship="LOVES",
            then_source="first.source",
            then_target="first.target",
        )
        expertise = ExpertiseConfig(name="t", inference_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([alice], [rel])

        matches = engine.evaluate_inference_rules(ctx)
        # Self-referential match is skipped
        assert matches == []


# ---------------------------------------------------------------------------
# Multi-condition (chain) inference rule
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiConditionInference:
    def test_two_condition_chain(self) -> None:
        ns = uuid4()
        alice = _make_entity("Alice", "PERSON", ns_id=ns)
        acme = _make_entity("Acme", "ORGANIZATION", ns_id=ns)
        seattle = _make_entity("Seattle", "LOCATION", ns_id=ns)

        works_for = _make_rel(alice, acme, "WORKS_FOR")
        located_in = _make_rel(acme, seattle, "LOCATED_IN")

        # Chain: Alice -WORKS_FOR-> Acme -LOCATED_IN-> Seattle
        # Infer: Alice -BASED_IN-> Seattle
        rule = InferenceRule(
            name="employee-location",
            when=[
                InferenceCondition(relationship="WORKS_FOR"),
                InferenceCondition(relationship="LOCATED_IN"),
            ],
            then_relationship="BASED_IN",
            then_source="first.source",
            then_target="second.target",
            confidence=0.6,
        )
        expertise = ExpertiseConfig(name="t", inference_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([alice, acme, seattle], [works_for, located_in])

        matches = engine.evaluate_inference_rules(ctx)
        assert len(matches) >= 1
        match = matches[0]
        assert match.rule_name == "employee-location"
        assert match.confidence == 0.6
        assert len(match.matched_relationships) == 2

    def test_two_condition_no_chain(self) -> None:
        """No matching second relationship — no inference."""
        ns = uuid4()
        alice = _make_entity("Alice", "PERSON", ns_id=ns)
        acme = _make_entity("Acme", "ORGANIZATION", ns_id=ns)
        works_for = _make_rel(alice, acme, "WORKS_FOR")

        rule = InferenceRule(
            name="x",
            when=[
                InferenceCondition(relationship="WORKS_FOR"),
                InferenceCondition(relationship="LOCATED_IN"),
            ],
            then_relationship="Y",
            then_source="first.source",
            then_target="second.target",
        )
        expertise = ExpertiseConfig(name="t", inference_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([alice, acme], [works_for])

        assert engine.evaluate_inference_rules(ctx) == []

    def test_with_source_type_filter(self) -> None:
        """source_type filter must match."""
        ns = uuid4()
        alice = _make_entity("Alice", "PERSON", ns_id=ns)
        acme = _make_entity("Acme", "ORGANIZATION", ns_id=ns)
        rel = _make_rel(alice, acme, "WORKS_FOR")

        rule = InferenceRule(
            name="x",
            when=[
                InferenceCondition(
                    relationship="WORKS_FOR",
                    source_type="PERSON",
                    target_type="ORGANIZATION",
                ),
            ],
            then_relationship="Y",
            then_source="first.source",
            then_target="first.target",
        )
        expertise = ExpertiseConfig(name="t", inference_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([alice, acme], [rel])

        assert len(engine.evaluate_inference_rules(ctx)) == 1

    def test_source_type_mismatch_filters_out(self) -> None:
        ns = uuid4()
        alice = _make_entity("Alice", "PERSON", ns_id=ns)
        acme = _make_entity("Acme", "ORGANIZATION", ns_id=ns)
        rel = _make_rel(alice, acme, "WORKS_FOR")

        rule = InferenceRule(
            name="x",
            when=[
                InferenceCondition(
                    relationship="WORKS_FOR",
                    source_type="LOCATION",
                ),
            ],
            then_relationship="Y",
            then_source="first.source",
            then_target="first.target",
        )
        expertise = ExpertiseConfig(name="t", inference_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([alice, acme], [rel])

        assert engine.evaluate_inference_rules(ctx) == []


# ---------------------------------------------------------------------------
# Correlation rules: pattern + field matching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorrelationRules:
    def test_pattern_rule_finds_emails(self) -> None:
        rule = CorrelationRule(
            name="emails",
            pattern=r"\b[A-Za-z]+@[A-Za-z]+\.[A-Za-z]+\b",
            entity_types=[],
            confidence=0.9,
        )
        expertise = ExpertiseConfig(name="t", correlation_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([], [])

        matches = engine.evaluate_correlation_rules(
            "Contact alice@x.com or bob@y.com today.",
            ctx,
        )
        assert len(matches) == 2

    def test_field_rule_groups_entities(self) -> None:
        e1 = _make_entity("Alice", attributes={"email": "shared@x.com"})
        e2 = _make_entity("Bob", attributes={"email": "shared@x.com"})
        e3 = _make_entity("Carol", attributes={"email": "lone@x.com"})

        rule = CorrelationRule(
            name="email-group",
            match_fields=["email"],
            entity_types=["PERSON"],
        )
        expertise = ExpertiseConfig(name="t", correlation_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([e1, e2, e3], [])
        matches = engine.evaluate_correlation_rules("", ctx)

        # One group with 2 entities (Carol's email is unique → skipped)
        assert len(matches) == 1
        assert len(matches[0].matched_entities) == 2

    def test_field_rule_no_entity_types(self) -> None:
        e1 = _make_entity("Alice", "PERSON", attributes={"email": "shared@x.com"})
        e2 = _make_entity("Acme", "ORGANIZATION", attributes={"email": "shared@x.com"})

        rule = CorrelationRule(
            name="email-group",
            match_fields=["email"],
        )
        expertise = ExpertiseConfig(name="t", correlation_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([e1, e2], [])
        matches = engine.evaluate_correlation_rules("", ctx)
        assert len(matches) == 1
        assert len(matches[0].matched_entities) == 2

    def test_pattern_finds_entities_via_attributes(self) -> None:
        """Pattern value found inside entity attribute string."""
        e1 = _make_entity(
            "Alice",
            attributes={"bio": "Senior engineer at Acme Corp"},
        )
        rule = CorrelationRule(
            name="acme-refs",
            pattern=r"Acme",
            entity_types=[],
        )
        expertise = ExpertiseConfig(name="t", correlation_rules=[rule])
        engine = RuleEngine(expertise)
        ctx = RuleEvaluationContext.from_data([e1], [])
        matches = engine.evaluate_correlation_rules("Acme is great", ctx)
        # Alice's bio contains Acme → she's a candidate
        assert len(matches) == 1
        assert e1 in matches[0].matched_entities


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInternalHelpers:
    def test_resolve_entity_ref_invalid_format(self) -> None:
        engine = RuleEngine()
        result = engine._resolve_entity_ref("bad_format", {})
        assert result is None

    def test_resolve_entity_ref_unknown_group(self) -> None:
        engine = RuleEngine()
        result = engine._resolve_entity_ref("third.source", {"first": {}})
        assert result is None

    def test_resolve_entity_ref_unknown_position(self) -> None:
        engine = RuleEngine()
        e = _make_entity("X")
        result = engine._resolve_entity_ref("first.middle", {"first": {"source": e}})
        assert result is None

    def test_resolve_entity_ref_valid(self) -> None:
        engine = RuleEngine()
        e = _make_entity("X")
        result = engine._resolve_entity_ref("first.source", {"first": {"source": e}})
        assert result is e

    def test_relationships_connect_chain(self) -> None:
        ns = uuid4()
        a = _make_entity("A", ns_id=ns)
        b = _make_entity("B", ns_id=ns)
        c = _make_entity("C", ns_id=ns)
        r1 = _make_rel(a, b)
        r2 = _make_rel(b, c)
        engine = RuleEngine()
        ctx = RuleEvaluationContext.from_data([a, b, c], [r1, r2])
        assert engine._relationships_connect(r1, r2, ctx)

    def test_relationships_connect_shared_source(self) -> None:
        ns = uuid4()
        a = _make_entity("A", ns_id=ns)
        b = _make_entity("B", ns_id=ns)
        c = _make_entity("C", ns_id=ns)
        r1 = _make_rel(a, b)
        r2 = _make_rel(a, c)
        engine = RuleEngine()
        ctx = RuleEvaluationContext.from_data([a, b, c], [r1, r2])
        assert engine._relationships_connect(r1, r2, ctx)

    def test_relationships_connect_shared_target(self) -> None:
        ns = uuid4()
        a = _make_entity("A", ns_id=ns)
        b = _make_entity("B", ns_id=ns)
        c = _make_entity("C", ns_id=ns)
        r1 = _make_rel(a, c)
        r2 = _make_rel(b, c)
        engine = RuleEngine()
        ctx = RuleEvaluationContext.from_data([a, b, c], [r1, r2])
        assert engine._relationships_connect(r1, r2, ctx)

    def test_relationships_dont_connect(self) -> None:
        ns = uuid4()
        a = _make_entity("A", ns_id=ns)
        b = _make_entity("B", ns_id=ns)
        c = _make_entity("C", ns_id=ns)
        d = _make_entity("D", ns_id=ns)
        r1 = _make_rel(a, b)
        r2 = _make_rel(c, d)
        engine = RuleEngine()
        ctx = RuleEvaluationContext.from_data([a, b, c, d], [r1, r2])
        assert not engine._relationships_connect(r1, r2, ctx)

    def test_find_entity_by_id(self) -> None:
        engine = RuleEngine()
        e = _make_entity("X")
        ctx = RuleEvaluationContext.from_data([e], [])
        assert engine._find_entity_by_id(e.id, ctx) is e
        assert engine._find_entity_by_id(uuid4(), ctx) is None
