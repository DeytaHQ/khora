"""Regression tests for PostgreSQL-incompatible NUL bytes in stored models."""

from khora.core.models import Chunk, Document, Entity, Relationship


def test_storage_models_strip_nul_bytes_from_text_and_json_fields() -> None:
    document = Document(content="before\x00after", metadata={"nested\x00key": ["value\x00"]})
    chunk = Chunk(content="chunk\x00text", metadata={"value": "nul\x00"})
    entity = Entity(
        name="Acme\x00 Corp",
        description="entity\x00description",
        attributes={"nested": {"value": "nul\x00"}},
    )
    relationship = Relationship(
        relationship_type="WORKS\x00_FOR",
        description="relationship\x00description",
        properties={"nested": ["nul\x00"]},
    )

    assert document.content == "beforeafter"
    assert document.metadata == {"nestedkey": ["value"]}
    assert chunk.content == "chunktext"
    assert chunk.metadata == {"value": "nul"}
    assert entity.name == "Acme Corp"
    assert entity.description == "entitydescription"
    assert entity.attributes == {"nested": {"value": "nul"}}
    assert relationship.relationship_type == "WORKS_FOR"
    assert relationship.description == "relationshipdescription"
    assert relationship.properties == {"nested": ["nul"]}
