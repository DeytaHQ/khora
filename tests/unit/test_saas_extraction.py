"""Tests for tool-schema extraction, source boosting, and type-aware linking."""

from khora.core.models.entity import Entity
from khora.extraction.extractors.base import ExtractedEntity
from khora.extraction.extractors.llm import LLMEntityExtractor
from khora.extraction.skills.base import ExpertiseConfig
from khora.query.linking import EntityLinker


class TestToolSchemaContext:
    """Test tool-schema-aware extraction prompt building."""

    def test_build_tool_context_with_schema(self):
        """_build_tool_context should produce context when tool_schemas match."""
        extractor = LLMEntityExtractor(model="test")
        expertise = ExpertiseConfig(
            name="linear",
            tool_schemas={
                "linear": {
                    "issue": {
                        "fields": ["identifier", "title", "status"],
                        "status_values": ["backlog", "todo", "done"],
                    }
                }
            },
        )
        context = {"source_tool": "linear"}

        result = extractor._build_tool_context(expertise, context)
        assert "linear" in result
        assert "identifier" in result
        assert "status_values" in result

    def test_build_tool_context_no_match(self):
        """_build_tool_context returns empty when source_tool doesn't match."""
        extractor = LLMEntityExtractor(model="test")
        expertise = ExpertiseConfig(
            name="jira",
            tool_schemas={"jira": {"issue": {"fields": ["key"]}}},
        )
        context = {"source_tool": "slack"}

        result = extractor._build_tool_context(expertise, context)
        assert result == ""

    def test_build_tool_context_no_expertise(self):
        """_build_tool_context returns empty without expertise."""
        extractor = LLMEntityExtractor(model="test")
        result = extractor._build_tool_context(None, {"source_tool": "linear"})
        assert result == ""

    def test_build_tool_context_no_context(self):
        """_build_tool_context returns empty without context."""
        extractor = LLMEntityExtractor(model="test")
        expertise = ExpertiseConfig(name="linear", tool_schemas={"linear": {}})
        result = extractor._build_tool_context(expertise, None)
        assert result == ""

    def test_build_tool_context_omits_attribute_hints(self):
        """_build_tool_context no longer emits the EXPECTED ENTITY ATTRIBUTES block.

        Per-type attribute keys moved to _build_attribute_schema_block (ungated,
        injected directly into the prompt) so they are no longer double-listed
        here. The SOURCE CONTEXT tool-field output must stay intact.
        """
        from khora.extraction.skills.base import EntityTypeConfig

        extractor = LLMEntityExtractor(model="test")
        expertise = ExpertiseConfig(
            name="linear",
            entity_types=[
                EntityTypeConfig(
                    name="TICKET",
                    attributes={"required": ["identifier", "title", "status"], "optional": ["priority"]},
                )
            ],
            tool_schemas={"linear": {"issue": {"fields": ["identifier", "title"]}}},
        )
        context = {"source_tool": "linear"}

        result = extractor._build_tool_context(expertise, context)
        # The attribute-schema block is gone from tool context.
        assert "EXPECTED ENTITY ATTRIBUTES" not in result
        # SOURCE CONTEXT tool-field output is unchanged.
        assert "SOURCE CONTEXT" in result
        assert "linear" in result
        assert "issue fields: identifier, title" in result


class TestExtractedEntitySourceTool:
    """Test source_tool field on ExtractedEntity."""

    def test_source_tool_field_exists(self):
        """ExtractedEntity should have source_tool field."""
        entity = ExtractedEntity(name="Test", entity_type="PERSON", source_tool="slack")
        assert entity.source_tool == "slack"

    def test_source_tool_default_empty(self):
        """ExtractedEntity source_tool should default to empty."""
        entity = ExtractedEntity(name="Test", entity_type="PERSON")
        assert entity.source_tool == ""


class TestTypePenalty:
    """Test entity-type-aware linking penalty."""

    def test_exact_type_match_no_penalty(self):
        """Exact type match should have no penalty."""
        linker = EntityLinker(storage=None, embedder=None)
        assert linker._type_penalty("PERSON", "PERSON") == 1.0
        assert linker._type_penalty("ORGANIZATION", "ORGANIZATION") == 1.0

    def test_no_type_info_no_penalty(self):
        """No type info (None) should have no penalty."""
        linker = EntityLinker(storage=None, embedder=None)
        assert linker._type_penalty(None, "PERSON") == 1.0

    def test_wrong_type_heavy_penalty(self):
        """Wrong type should get heavy penalty."""
        linker = EntityLinker(storage=None, embedder=None)
        penalty = linker._type_penalty("PERSON", "PRODUCT")
        assert penalty == 0.3

    def test_concept_wildcard_minimal_penalty(self):
        """CONCEPT type should have minimal penalty (wildcard)."""
        linker = EntityLinker(storage=None, embedder=None)
        assert linker._type_penalty("CONCEPT", "PERSON") == 0.9
        assert linker._type_penalty("PERSON", "CONCEPT") == 0.9

    def test_custom_wildcard_minimal_penalty(self):
        """CUSTOM type should have minimal penalty (wildcard)."""
        linker = EntityLinker(storage=None, embedder=None)
        assert linker._type_penalty("CUSTOM", "PERSON") == 0.9

    def test_case_insensitive(self):
        """Type comparison should be case-insensitive."""
        linker = EntityLinker(storage=None, embedder=None)
        assert linker._type_penalty("person", "PERSON") == 1.0


class TestAttributeRelevanceBoost:
    """Test attribute-aware search scoring."""

    def test_matching_attributes_boost(self):
        """Entity with matching attribute values should get a boost."""
        from khora.query.engine import HybridQueryEngine

        entity = Entity(
            name="ENG-123",
            entity_type="CONCEPT",
            attributes={"priority": "urgent", "assignee": "Alice"},
        )
        boost = HybridQueryEngine._attribute_relevance_boost(entity, ["urgent", "alice"])
        assert boost > 0
        assert boost <= 0.3

    def test_no_matching_attributes(self):
        """Entity with no matching attributes should get zero boost."""
        from khora.query.engine import HybridQueryEngine

        entity = Entity(
            name="ENG-456",
            entity_type="CONCEPT",
            attributes={"priority": "low", "assignee": "Bob"},
        )
        boost = HybridQueryEngine._attribute_relevance_boost(entity, ["urgent", "alice"])
        assert boost == 0.0

    def test_boost_capped_at_0_3(self):
        """Boost should be capped at 0.3 even with many matches."""
        from khora.query.engine import HybridQueryEngine

        entity = Entity(
            name="Test",
            entity_type="CONCEPT",
            attributes={"a": "foo", "b": "foo", "c": "foo", "d": "foo", "e": "foo"},
        )
        boost = HybridQueryEngine._attribute_relevance_boost(entity, ["foo"])
        assert boost == 0.3

    def test_no_attributes(self):
        """Entity with no attributes should get zero boost."""
        from khora.query.engine import HybridQueryEngine

        entity = Entity(name="Test", entity_type="CONCEPT", attributes={})
        boost = HybridQueryEngine._attribute_relevance_boost(entity, ["urgent"])
        assert boost == 0.0
