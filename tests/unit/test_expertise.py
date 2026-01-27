"""Unit tests for expertise configuration system."""

from __future__ import annotations

import pytest

from khora.extraction.skills import (
    ConfidenceConfig,
    CorrelationRule,
    EntityTypeConfig,
    ExpansionConfig,
    ExpertiseConfig,
    ExpertiseLoader,
    InferenceCondition,
    InferenceRule,
    RelationshipTypeConfig,
    get_default_loader,
)


class TestEntityTypeConfig:
    """Tests for EntityTypeConfig dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic EntityTypeConfig creation."""
        entity_type = EntityTypeConfig(
            name="PERSON",
            description="A human individual",
        )
        assert entity_type.name == "PERSON"
        assert entity_type.description == "A human individual"
        assert entity_type.attributes == {}
        assert entity_type.identifiers == []
        assert entity_type.aliases == []

    def test_with_attributes(self) -> None:
        """Test EntityTypeConfig with attributes."""
        entity_type = EntityTypeConfig(
            name="TICKET",
            description="Issue tracker ticket",
            attributes={"required": ["key", "status"], "optional": ["assignee"]},
            identifiers=["key"],
            aliases=["issue", "bug", "story"],
        )
        assert entity_type.attributes["required"] == ["key", "status"]
        assert entity_type.identifiers == ["key"]
        assert "issue" in entity_type.aliases

    def test_to_dict(self) -> None:
        """Test EntityTypeConfig serialization."""
        entity_type = EntityTypeConfig(
            name="CUSTOMER",
            description="Customer account",
            identifiers=["domain"],
        )
        data = entity_type.to_dict()
        assert data["name"] == "CUSTOMER"
        assert data["description"] == "Customer account"
        assert data["identifiers"] == ["domain"]

    def test_from_dict(self) -> None:
        """Test EntityTypeConfig deserialization."""
        data = {
            "name": "PROJECT",
            "description": "A project",
            "attributes": {"required": ["name"]},
        }
        entity_type = EntityTypeConfig.from_dict(data)
        assert entity_type.name == "PROJECT"
        assert entity_type.attributes["required"] == ["name"]


class TestRelationshipTypeConfig:
    """Tests for RelationshipTypeConfig dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic RelationshipTypeConfig creation."""
        rel_type = RelationshipTypeConfig(
            name="WORKS_FOR",
            description="Employment relationship",
        )
        assert rel_type.name == "WORKS_FOR"
        assert rel_type.source_types == []
        assert rel_type.target_types == []

    def test_with_constraints(self) -> None:
        """Test RelationshipTypeConfig with type constraints."""
        rel_type = RelationshipTypeConfig(
            name="ASSIGNED_TO",
            description="Task assignment",
            source_types=["TICKET", "TASK"],
            target_types=["PERSON", "TEAM"],
        )
        assert "TICKET" in rel_type.source_types
        assert "PERSON" in rel_type.target_types

    def test_serialization(self) -> None:
        """Test RelationshipTypeConfig round-trip serialization."""
        original = RelationshipTypeConfig(
            name="OWNS",
            description="Ownership",
            source_types=["PERSON"],
            target_types=["PROJECT"],
        )
        data = original.to_dict()
        restored = RelationshipTypeConfig.from_dict(data)
        assert restored.name == original.name
        assert restored.source_types == original.source_types


class TestCorrelationRule:
    """Tests for CorrelationRule dataclass."""

    def test_pattern_rule(self) -> None:
        """Test correlation rule with regex pattern."""
        rule = CorrelationRule(
            name="issue_reference",
            description="Match issue keys",
            pattern=r"[A-Z]+-\d+",
            creates_relationship="REFERENCES",
        )
        assert rule.pattern == r"[A-Z]+-\d+"
        assert rule.creates_relationship == "REFERENCES"
        assert rule.confidence == 0.9  # default

    def test_field_matching_rule(self) -> None:
        """Test correlation rule with field matching."""
        rule = CorrelationRule(
            name="email_match",
            description="Match by email",
            match_fields=["email"],
            entity_types=["PERSON", "CONTACT"],
            confidence=0.95,
        )
        assert rule.match_fields == ["email"]
        assert rule.entity_types == ["PERSON", "CONTACT"]
        assert rule.confidence == 0.95

    def test_serialization(self) -> None:
        """Test CorrelationRule round-trip serialization."""
        original = CorrelationRule(
            name="test",
            pattern=r"\d+",
            confidence=0.7,
        )
        data = original.to_dict()
        restored = CorrelationRule.from_dict(data)
        assert restored.name == original.name
        assert restored.pattern == original.pattern
        assert restored.confidence == original.confidence


class TestInferenceRule:
    """Tests for InferenceRule dataclass."""

    def test_basic_rule(self) -> None:
        """Test basic inference rule creation."""
        rule = InferenceRule(
            name="project_stakeholder",
            description="Infer stakeholder relationship",
            when=[
                InferenceCondition(relationship="OWNS", source_type="PERSON", target_type="PROJECT"),
            ],
            then_relationship="STAKEHOLDER_OF",
            then_source="first.source",
            then_target="first.target",
            confidence=0.7,
        )
        assert rule.name == "project_stakeholder"
        assert len(rule.when) == 1
        assert rule.then_relationship == "STAKEHOLDER_OF"
        assert rule.confidence == 0.7

    def test_multi_condition_rule(self) -> None:
        """Test inference rule with multiple conditions."""
        rule = InferenceRule(
            name="transitive_membership",
            when=[
                InferenceCondition(relationship="MEMBER_OF", source_type="PERSON", target_type="TEAM"),
                InferenceCondition(relationship="PART_OF", source_type="TEAM", target_type="DEPARTMENT"),
            ],
            then_relationship="BELONGS_TO",
            confidence=0.6,
        )
        assert len(rule.when) == 2

    def test_serialization(self) -> None:
        """Test InferenceRule round-trip serialization."""
        original = InferenceRule(
            name="test_rule",
            when=[InferenceCondition(relationship="KNOWS")],
            then_relationship="CONNECTED_TO",
        )
        data = original.to_dict()
        restored = InferenceRule.from_dict(data)
        assert restored.name == original.name
        assert restored.then_relationship == original.then_relationship


class TestConfidenceConfig:
    """Tests for ConfidenceConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default confidence thresholds."""
        config = ConfidenceConfig()
        assert config.min_entity == 0.5
        assert config.min_relationship == 0.5
        assert config.min_inferred == 0.3

    def test_custom_thresholds(self) -> None:
        """Test custom confidence thresholds."""
        config = ConfidenceConfig(
            min_entity=0.7,
            min_relationship=0.6,
            min_inferred=0.4,
        )
        assert config.min_entity == 0.7
        assert config.min_relationship == 0.6
        assert config.min_inferred == 0.4


class TestExpansionConfig:
    """Tests for ExpansionConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default expansion settings."""
        config = ExpansionConfig()
        assert config.enabled is True
        assert config.depth == 2
        assert config.cross_tool_unification is True
        assert config.relationship_inference is True

    def test_disabled(self) -> None:
        """Test disabled expansion."""
        config = ExpansionConfig(
            enabled=False,
            cross_tool_unification=False,
            relationship_inference=False,
        )
        assert config.enabled is False


class TestExpertiseConfig:
    """Tests for ExpertiseConfig dataclass."""

    def test_minimal_config(self) -> None:
        """Test minimal expertise configuration."""
        config = ExpertiseConfig(name="test")
        assert config.name == "test"
        assert config.version == "1.0.0"
        assert config.entity_types == []
        assert config.relationship_types == []

    def test_full_config(self) -> None:
        """Test full expertise configuration."""
        config = ExpertiseConfig(
            name="saas_expert",
            version="2.0.0",
            description="SaaS tools expertise",
            extends=["general"],
            system_prompt="You are an expert...",
            extraction_prompt="Extract entities from: {{ text }}",
            entity_types=[
                EntityTypeConfig(name="TICKET", description="Issue ticket"),
                EntityTypeConfig(name="CUSTOMER", description="Customer account"),
            ],
            relationship_types=[
                RelationshipTypeConfig(name="ASSIGNED_TO", description="Assignment"),
            ],
            correlation_rules=[
                CorrelationRule(name="email_match", match_fields=["email"]),
            ],
            inference_rules=[
                InferenceRule(
                    name="test",
                    when=[InferenceCondition(relationship="OWNS")],
                    then_relationship="STAKEHOLDER_OF",
                ),
            ],
            confidence=ConfidenceConfig(min_entity=0.6),
            expansion=ExpansionConfig(depth=3),
        )
        assert config.name == "saas_expert"
        assert len(config.entity_types) == 2
        assert len(config.relationship_types) == 1
        assert len(config.correlation_rules) == 1
        assert len(config.inference_rules) == 1
        assert config.confidence.min_entity == 0.6
        assert config.expansion.depth == 3

    def test_to_dict(self) -> None:
        """Test ExpertiseConfig serialization."""
        config = ExpertiseConfig(
            name="test",
            entity_types=[EntityTypeConfig(name="PERSON", description="A person")],
        )
        data = config.to_dict()
        assert data["name"] == "test"
        assert len(data["entity_types"]) == 1
        assert data["entity_types"][0]["name"] == "PERSON"

    def test_from_dict(self) -> None:
        """Test ExpertiseConfig deserialization."""
        data = {
            "name": "restored",
            "version": "1.0.0",
            "entity_types": [{"name": "ORG", "description": "Organization"}],
            "relationship_types": [{"name": "OWNS", "description": "Ownership"}],
        }
        config = ExpertiseConfig.from_dict(data)
        assert config.name == "restored"
        assert len(config.entity_types) == 1
        assert config.entity_types[0].name == "ORG"

    def test_round_trip_serialization(self) -> None:
        """Test full round-trip serialization."""
        original = ExpertiseConfig(
            name="roundtrip_test",
            version="1.2.3",
            description="Test config",
            entity_types=[
                EntityTypeConfig(name="A", description="Type A", identifiers=["id"]),
            ],
            relationship_types=[
                RelationshipTypeConfig(name="R", source_types=["A"], target_types=["A"]),
            ],
            correlation_rules=[
                CorrelationRule(name="c1", pattern=r"\d+"),
            ],
            inference_rules=[
                InferenceRule(
                    name="i1",
                    when=[InferenceCondition(relationship="R")],
                    then_relationship="R2",
                ),
            ],
        )
        data = original.to_dict()
        restored = ExpertiseConfig.from_dict(data)

        assert restored.name == original.name
        assert restored.version == original.version
        assert len(restored.entity_types) == len(original.entity_types)
        assert len(restored.relationship_types) == len(original.relationship_types)
        assert len(restored.correlation_rules) == len(original.correlation_rules)
        assert len(restored.inference_rules) == len(original.inference_rules)

    def test_to_extraction_skill(self) -> None:
        """Test conversion to ExtractionSkill."""
        config = ExpertiseConfig(
            name="skill_test",
            description="Test skill",
            entity_types=[
                EntityTypeConfig(name="PERSON", description="A person"),
                EntityTypeConfig(name="ORG", description="An organization"),
            ],
            relationship_types=[
                RelationshipTypeConfig(name="WORKS_FOR", description="Employment"),
            ],
        )
        skill = config.to_extraction_skill()

        assert skill.name == "skill_test"
        assert skill.description == "Test skill"
        assert "PERSON" in skill.entity_types
        assert "ORG" in skill.entity_types
        assert "WORKS_FOR" in skill.relationship_types


class TestExpertiseLoader:
    """Tests for ExpertiseLoader."""

    def test_get_default_loader(self) -> None:
        """Test getting the default loader singleton."""
        loader1 = get_default_loader()
        loader2 = get_default_loader()
        assert loader1 is loader2

    def test_load_builtin_general(self) -> None:
        """Test loading built-in general expertise."""
        loader = ExpertiseLoader()
        config = loader.load_builtin("general")

        assert config.name == "general"
        assert len(config.entity_types) > 0
        assert len(config.relationship_types) > 0

        # Check for expected entity types
        entity_names = [e.name for e in config.entity_types]
        assert "PERSON" in entity_names
        assert "ORGANIZATION" in entity_names

    def test_load_source_builtin_prefix(self) -> None:
        """Test loading with builtin: prefix."""
        loader = ExpertiseLoader()
        config = loader.load_source("builtin:general")
        assert config.name == "general"

    def test_load_file(self) -> None:
        """Test loading from YAML file."""
        loader = ExpertiseLoader()
        config = loader.load_file("examples/config/expertise/saas_expert.yaml")

        assert config.name == "saas_expert"
        assert config.version == "1.0.0"
        assert len(config.entity_types) > 20  # SaaS has many entity types
        assert len(config.correlation_rules) > 0
        assert len(config.inference_rules) > 0

    def test_load_source_file_prefix(self) -> None:
        """Test loading with file: prefix."""
        loader = ExpertiseLoader()
        config = loader.load_source("file:examples/config/expertise/technical_docs.yaml")
        assert config.name == "technical_docs"

    def test_load_source_direct_path(self) -> None:
        """Test loading with direct file path."""
        loader = ExpertiseLoader()
        config = loader.load_source("examples/config/expertise/business_intel.yaml")
        assert config.name == "business_intel"

    def test_cache_behavior(self) -> None:
        """Test that caching works correctly."""
        loader = ExpertiseLoader()

        # First load
        config1 = loader.load_builtin("general", use_cache=True)

        # Second load should return cached
        config2 = loader.load_builtin("general", use_cache=True)
        assert config1 is config2

        # Clear cache and reload
        loader.clear_cache()
        config3 = loader.load_builtin("general", use_cache=True)
        assert config3 is not config1

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file raises error."""
        from khora.extraction.skills.loader import ExpertiseLoadError

        loader = ExpertiseLoader()
        with pytest.raises(ExpertiseLoadError):
            loader.load_file("/nonexistent/path.yaml")

    def test_load_nonexistent_builtin(self) -> None:
        """Test loading nonexistent builtin raises error."""
        from khora.extraction.skills.loader import ExpertiseLoadError

        loader = ExpertiseLoader()
        with pytest.raises(ExpertiseLoadError):
            loader.load_builtin("nonexistent_builtin_xyz")

    def test_load_merged(self) -> None:
        """Test merging multiple expertise configs."""
        loader = ExpertiseLoader()
        merged = loader.load_merged(
            [
                "builtin:general",
                "file:examples/config/expertise/technical_docs.yaml",
            ]
        )

        # Should have entity types from both
        entity_names = [e.name for e in merged.entity_types]
        assert "PERSON" in entity_names  # from general
        assert "API" in entity_names  # from technical_docs

    def test_resolve_extends(self) -> None:
        """Test resolving extends inheritance."""
        loader = ExpertiseLoader()

        child = ExpertiseConfig(
            name="child",
            extends=["builtin:general"],
            entity_types=[
                EntityTypeConfig(name="CUSTOM", description="Custom type"),
            ],
        )

        resolved = loader.resolve_extends(child)

        # Should have entities from both parent and child
        entity_names = [e.name for e in resolved.entity_types]
        assert "PERSON" in entity_names  # from parent
        assert "CUSTOM" in entity_names  # from child
