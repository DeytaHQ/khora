"""Tests for entity attribute schema validation."""

from pydantic import BaseModel

from khora.core.models.entity import Entity
from khora.core.models.schemas import (
    ATTRIBUTE_SCHEMAS,
    register_attribute_schema,
    validate_attributes,
)


class TestAttributeSchemas:
    """Test the attribute schema registry and validation."""

    def test_registry_has_standard_types(self):
        """All standard entity types should have schemas."""
        expected = {"PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "TECHNOLOGY", "PRODUCT", "DATE"}
        assert expected.issubset(set(ATTRIBUTE_SCHEMAS.keys()))

    def test_register_attribute_schema(self):
        """Downstream projects can register custom schemas."""

        class CustomAttributes(BaseModel):
            name: str
            custom_field: str | None = None

        register_attribute_schema("CUSTOM_TYPE", CustomAttributes, aliases=["CUSTOM_ALIAS"])
        assert ATTRIBUTE_SCHEMAS["CUSTOM_TYPE"] is CustomAttributes
        assert ATTRIBUTE_SCHEMAS["CUSTOM_ALIAS"] is CustomAttributes

        # Validate works with the registered schema
        result = validate_attributes("CUSTOM_TYPE", {"name": "test", "custom_field": "val"})
        assert result["name"] == "test"
        assert result["custom_field"] == "val"

        # Clean up
        del ATTRIBUTE_SCHEMAS["CUSTOM_TYPE"]
        del ATTRIBUTE_SCHEMAS["CUSTOM_ALIAS"]


class TestValidateAttributes:
    """Test the validate_attributes function."""

    def test_valid_person_attributes(self):
        """Valid person attributes pass validation."""
        attrs = {"name": "Alice Smith", "title": "Engineer", "email": "alice@example.com"}
        result = validate_attributes("PERSON", attrs)
        assert result["name"] == "Alice Smith"
        assert result["title"] == "Engineer"
        assert result["email"] == "alice@example.com"

    def test_person_attributes_strips_none(self):
        """None values are excluded from validated output."""
        attrs = {"name": "Bob", "title": None, "role": None}
        result = validate_attributes("PERSON", attrs)
        assert result == {"name": "Bob"}

    def test_unknown_entity_type_passthrough(self):
        """Unknown entity types pass through without validation."""
        attrs = {"foo": "bar", "baz": 42}
        result = validate_attributes("UNKNOWN_TYPE", attrs)
        assert result == attrs

    def test_extra_fields_ignored(self):
        """Extra fields not in schema are silently dropped."""
        attrs = {"name": "Alice", "unknown_field": "value"}
        result = validate_attributes("PERSON", attrs)
        assert "name" in result
        # Pydantic by default ignores extra fields
        assert "unknown_field" not in result

    def test_case_insensitive_type_lookup(self):
        """Entity type lookup should be case-insensitive."""
        attrs = {"name": "Test Person"}
        result = validate_attributes("person", attrs)
        assert result["name"] == "Test Person"

    def test_empty_attributes_with_required_fields(self):
        """Empty attributes for a type with required fields degrades gracefully."""
        result = validate_attributes("PERSON", {})
        assert result == {}  # Falls back to original


class TestEntityValidation:
    """Test Entity.validate() method."""

    def test_entity_validate_cleans_attributes(self):
        """Entity.validate() should clean attributes via schema."""
        entity = Entity(
            name="Alice",
            entity_type="PERSON",
            attributes={"name": "Alice", "title": "CTO", "garbage": None},
        )
        entity.validate()
        assert entity.attributes["name"] == "Alice"
        assert entity.attributes["title"] == "CTO"

    def test_entity_validate_unknown_type(self):
        """Entity.validate() should pass through for CUSTOM type."""
        entity = Entity(
            name="Something",
            entity_type="CUSTOM",
            attributes={"foo": "bar"},
        )
        entity.validate()
        assert entity.attributes == {"foo": "bar"}

    def test_entity_source_tool_field(self):
        """Entity should have source_tool field."""
        entity = Entity(name="Test", source_tool="linear")
        assert entity.source_tool == "linear"

    def test_entity_source_tool_default_empty(self):
        """Entity source_tool should default to empty string."""
        entity = Entity(name="Test")
        assert entity.source_tool == ""
