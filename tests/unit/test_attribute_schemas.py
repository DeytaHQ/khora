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

    def test_extra_fields_preserved(self):
        """Extra fields not in the schema are preserved (additive-safe)."""
        attrs = {"name": "Alice", "unknown_field": "value"}
        result = validate_attributes("PERSON", attrs)
        assert result["name"] == "Alice"
        # Unknown/ontology keys survive validation rather than being dropped.
        assert result["unknown_field"] == "value"

    def test_unknown_keys_preserved_additive(self):
        """Unknown ontology keys survive alongside known coerced fields."""
        attrs = {"name": "X", "slack_user_id": "U1", "timezone": "CET"}
        result = validate_attributes("PERSON", attrs)
        assert result["name"] == "X"
        assert result["slack_user_id"] == "U1"
        assert result["timezone"] == "CET"

    def test_none_known_field_stripped_while_unknown_key_preserved(self):
        """Known None fields are stripped even as unknown ontology keys survive."""
        attrs = {"name": "X", "title": None, "slack_user_id": "U1"}
        result = validate_attributes("PERSON", attrs)
        assert result == {"name": "X", "slack_user_id": "U1"}

    def test_none_unknown_key_dropped_on_validated_path(self):
        """A None-valued unknown key is excluded on the validated path (base-dict filter)."""
        result = validate_attributes("PERSON", {"name": "X", "foo": None})
        assert result == {"name": "X"}

    def test_unregistered_type_returned_unchanged(self):
        """An unregistered entity type is returned unchanged."""
        attrs = {"identifier": "ENG-123", "title": "Fix bug", "state": "open"}
        result = validate_attributes("LINEAR_ISSUE", attrs)
        assert result == attrs

    def test_known_fields_coerced_for_registered_type(self):
        """Registered-schema fields are still coerced (coercion wins over raw input)."""
        attrs = {"name": "AI", "related_concepts": ("ML", "DL")}
        result = validate_attributes("CONCEPT", attrs)
        assert isinstance(result["related_concepts"], list)
        assert result["related_concepts"] == ["ML", "DL"]

    def test_validation_error_returns_original_attributes(self):
        """A ValidationError degrades gracefully, returning original attributes unchanged."""
        attrs = {"title": "Engineer", "slack_user_id": "U1"}  # missing required name
        result = validate_attributes("PERSON", attrs)
        assert result == attrs

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
