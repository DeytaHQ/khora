"""Unit tests for skill registry and composer."""

from __future__ import annotations

import pytest

from khora.extraction.skills import (
    EntityTypeConfig,
    ExpertiseConfig,
    ExtractionSkill,
    RelationshipTypeConfig,
    SkillRegistry,
    get_default_registry,
)
from khora.extraction.skills.composer import ExpertiseComposer
from khora.extraction.skills.loader import ExpertiseLoader


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_create_registry(self) -> None:
        """Test creating a new registry."""
        registry = SkillRegistry()
        # Should have built-in skills
        skills = registry.list_skills()
        assert "general_entities" in skills

    def test_register_skill(self) -> None:
        """Test registering a custom skill."""
        registry = SkillRegistry()
        skill = ExtractionSkill(
            name="custom_skill",
            description="A custom skill",
            entity_types=["CUSTOM_TYPE"],
            relationship_types=["CUSTOM_REL"],
        )
        registry.register(skill)

        assert "custom_skill" in registry.list_skills()
        retrieved = registry.get("custom_skill")
        assert retrieved is not None
        assert retrieved.name == "custom_skill"

    def test_register_expertise_config(self) -> None:
        """Test registering an ExpertiseConfig."""
        registry = SkillRegistry()
        expertise = ExpertiseConfig(
            name="test_expertise",
            description="Test expertise",
            entity_types=[
                EntityTypeConfig(name="TEST", description="Test entity"),
            ],
        )
        registry.register(expertise)

        # Should be available as both expertise and skill
        assert "test_expertise" in registry.list_expertise()
        assert "test_expertise" in registry.list_skills()

    def test_get_nonexistent_skill(self) -> None:
        """Test getting a skill that doesn't exist."""
        registry = SkillRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_get_or_default(self) -> None:
        """Test get_or_default falls back to general_entities."""
        registry = SkillRegistry()
        result = registry.get_or_default("nonexistent")
        assert result.name == "general_entities"

    def test_unregister_skill(self) -> None:
        """Test unregistering a skill."""
        registry = SkillRegistry()
        skill = ExtractionSkill(name="to_remove", description="Will be removed")
        registry.register(skill)

        assert "to_remove" in registry.list_skills()

        removed = registry.unregister("to_remove")
        assert removed is True
        assert "to_remove" not in registry.list_skills()

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent skill returns False."""
        registry = SkillRegistry()
        removed = registry.unregister("nonexistent")
        assert removed is False

    def test_all_skills(self) -> None:
        """Test getting all skills."""
        registry = SkillRegistry()
        skills = registry.all_skills()
        assert len(skills) >= 4  # Built-in skills
        assert all(isinstance(s, ExtractionSkill) for s in skills)

    def test_get_expertise(self) -> None:
        """Test getting expertise by name."""
        registry = SkillRegistry()
        expertise = ExpertiseConfig(name="test_exp", description="Test")
        registry.register(expertise)

        retrieved = registry.get_expertise("test_exp")
        assert retrieved is not None
        assert retrieved.name == "test_exp"

    def test_get_expertise_nonexistent(self) -> None:
        """Test getting nonexistent expertise."""
        registry = SkillRegistry()
        result = registry.get_expertise("nonexistent")
        assert result is None

    def test_get_expertise_or_default(self) -> None:
        """Test get_expertise_or_default falls back to general."""
        registry = SkillRegistry()
        result = registry.get_expertise_or_default("nonexistent")
        # Should return some default expertise
        assert result is not None
        assert result.name in ["general", "nonexistent"]

    def test_all_expertise(self) -> None:
        """Test getting all expertise configs."""
        registry = SkillRegistry()
        expertise = ExpertiseConfig(name="exp1", description="First")
        registry.register(expertise)

        all_exp = registry.all_expertise()
        assert len(all_exp) >= 1
        assert all(isinstance(e, ExpertiseConfig) for e in all_exp)

    def test_register_from_config(self) -> None:
        """Test registering skills from config dictionaries."""
        registry = SkillRegistry()
        config = [
            {
                "name": "from_config",
                "description": "Loaded from config",
                "entity_types": ["TYPE_A"],
                "relationship_types": ["REL_A"],
            },
        ]
        registry.register_from_config(config)

        assert "from_config" in registry.list_skills()

    def test_to_dict(self) -> None:
        """Test exporting registry to dictionary."""
        registry = SkillRegistry()
        data = registry.to_dict()

        assert "general_entities" in data
        assert isinstance(data["general_entities"], dict)


class TestDefaultRegistry:
    """Tests for the default global registry."""

    def test_get_default_registry_singleton(self) -> None:
        """Test that get_default_registry returns singleton."""
        reg1 = get_default_registry()
        reg2 = get_default_registry()
        assert reg1 is reg2

    def test_default_registry_has_builtins(self) -> None:
        """Test default registry has built-in skills."""
        registry = get_default_registry()
        skills = registry.list_skills()

        assert "general_entities" in skills
        assert "technical_docs" in skills
        assert "business_intel" in skills
        assert "research_papers" in skills


class TestExtractionSkill:
    """Tests for ExtractionSkill dataclass."""

    def test_create_skill(self) -> None:
        """Test creating an extraction skill."""
        skill = ExtractionSkill(
            name="test_skill",
            description="A test skill",
            entity_types=["PERSON", "ORG"],
            relationship_types=["WORKS_FOR"],
        )
        assert skill.name == "test_skill"
        assert "PERSON" in skill.entity_types
        assert "WORKS_FOR" in skill.relationship_types

    def test_skill_to_dict(self) -> None:
        """Test skill serialization."""
        skill = ExtractionSkill(
            name="test",
            description="Test skill",
            entity_types=["A", "B"],
            relationship_types=["R"],
        )
        data = skill.to_dict()

        assert data["name"] == "test"
        assert data["description"] == "Test skill"
        assert "A" in data["entity_types"]

    def test_skill_from_dict(self) -> None:
        """Test skill deserialization."""
        data = {
            "name": "loaded",
            "description": "Loaded skill",
            "entity_types": ["X", "Y"],
            "relationship_types": ["Z"],
        }
        skill = ExtractionSkill.from_dict(data)

        assert skill.name == "loaded"
        assert "X" in skill.entity_types
        assert "Z" in skill.relationship_types

    def test_builtin_general_entities(self) -> None:
        """Test built-in general_entities skill."""
        skill = ExtractionSkill.general_entities()
        assert skill.name == "general_entities"
        assert len(skill.entity_types) > 0

    def test_builtin_technical_docs(self) -> None:
        """Test built-in technical_docs skill."""
        skill = ExtractionSkill.technical_docs()
        assert skill.name == "technical_docs"

    def test_builtin_business_intel(self) -> None:
        """Test built-in business_intel skill."""
        skill = ExtractionSkill.business_intel()
        assert skill.name == "business_intel"

    def test_builtin_research_papers(self) -> None:
        """Test built-in research_papers skill."""
        skill = ExtractionSkill.research_papers()
        assert skill.name == "research_papers"


class TestExpertiseComposer:
    """Tests for ExpertiseComposer."""

    def test_create_composer(self) -> None:
        """Test creating a composer."""
        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)
        assert composer is not None

    def test_merge_single_config(self) -> None:
        """Test merging a single config returns it unchanged."""
        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)

        config = ExpertiseConfig(
            name="single",
            entity_types=[EntityTypeConfig(name="A", description="Type A")],
        )

        merged = composer.merge([config])
        assert merged.name == "single"
        assert len(merged.entity_types) == 1

    def test_merge_multiple_configs(self) -> None:
        """Test merging multiple configs."""
        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)

        config1 = ExpertiseConfig(
            name="first",
            entity_types=[EntityTypeConfig(name="A", description="Type A")],
            relationship_types=[RelationshipTypeConfig(name="R1", description="Rel 1")],
        )

        config2 = ExpertiseConfig(
            name="second",
            entity_types=[EntityTypeConfig(name="B", description="Type B")],
            relationship_types=[RelationshipTypeConfig(name="R2", description="Rel 2")],
        )

        merged = composer.merge([config1, config2])

        # Should have entity types from both
        entity_names = [e.name for e in merged.entity_types]
        assert "A" in entity_names
        assert "B" in entity_names

        # Should have relationship types from both
        rel_names = [r.name for r in merged.relationship_types]
        assert "R1" in rel_names
        assert "R2" in rel_names

    def test_merge_overwrites_same_name(self) -> None:
        """Test that later config overwrites same-named items."""
        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)

        config1 = ExpertiseConfig(
            name="first",
            entity_types=[EntityTypeConfig(name="A", description="Original A")],
        )

        config2 = ExpertiseConfig(
            name="second",
            entity_types=[EntityTypeConfig(name="A", description="Updated A")],
        )

        merged = composer.merge([config1, config2])

        # Should have updated description
        a_type = next(e for e in merged.entity_types if e.name == "A")
        assert a_type.description == "Updated A"

    def test_merge_system_prompts(self) -> None:
        """Test merging system prompts."""
        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)

        config1 = ExpertiseConfig(
            name="first",
            system_prompt="First prompt",
        )

        config2 = ExpertiseConfig(
            name="second",
            system_prompt="Second prompt",
        )

        merged = composer.merge([config1, config2])

        # Later prompt should win
        assert merged.system_prompt == "Second prompt"

    def test_merge_empty_list(self) -> None:
        """Test merging empty list raises error."""
        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)

        with pytest.raises(ValueError):
            composer.merge([])

    def test_merge_preserves_confidence_config(self) -> None:
        """Test that merge preserves confidence config from last."""
        from khora.extraction.skills import ConfidenceConfig

        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)

        config1 = ExpertiseConfig(
            name="first",
            confidence=ConfidenceConfig(min_entity=0.5),
        )

        config2 = ExpertiseConfig(
            name="second",
            confidence=ConfidenceConfig(min_entity=0.7),
        )

        merged = composer.merge([config1, config2])

        assert merged.confidence.min_entity == 0.7

    def test_merge_preserves_parent_disabled_expansion(self) -> None:
        """A parent that disabled expansion is not re-enabled by a child that
        leaves expansion at its dataclass defaults (#1126)."""
        from khora.extraction.skills import ExpansionConfig

        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)

        parent = ExpertiseConfig(
            name="parent",
            expansion=ExpansionConfig(
                enabled=False,
                cross_tool_unification=False,
                relationship_inference=False,
            ),
        )
        # Child does not mention expansion -> all-default (True) ExpansionConfig.
        child = ExpertiseConfig(name="child")

        merged = composer.merge([parent, child])

        assert merged.expansion.enabled is False
        assert merged.expansion.cross_tool_unification is False
        assert merged.expansion.relationship_inference is False

    def test_merge_propagates_expansion_extra_fields(self) -> None:
        """Non-default inference_mode / preload_existing / batch_storage_size on
        either side flow through the merge instead of being dropped (#1126).

        Like the numeric fields, these use a value-differs-from-default
        heuristic: a non-default overlay value wins, otherwise the base value
        survives.
        """
        from khora.extraction.skills import ExpansionConfig

        loader = ExpertiseLoader()
        composer = ExpertiseComposer(loader)

        # Parent sets a non-default inference_mode; child only overrides the
        # other two extra fields (leaving inference_mode at its default).
        parent = ExpertiseConfig(
            name="parent",
            expansion=ExpansionConfig(inference_mode="batch"),
        )
        child = ExpertiseConfig(
            name="child",
            expansion=ExpansionConfig(preload_existing=False, batch_storage_size=25),
        )

        merged = composer.merge([parent, child])

        # Parent's non-default inference_mode is not clobbered by the child's
        # default "smart".
        assert merged.expansion.inference_mode == "batch"
        # Child's explicit (non-default) overrides win.
        assert merged.expansion.preload_existing is False
        assert merged.expansion.batch_storage_size == 25
