"""Unit tests for Neo4j serialization helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.storage.backends.neo4j import (
    _coerce_neo4j_datetime,
    _entity_to_cypher_params,
    _relationship_to_cypher_params,
)


class TestEntityToCypherParams:
    """Tests for _entity_to_cypher_params helper."""

    def test_basic_entity(self) -> None:
        """Converts a basic entity to Cypher params dict."""
        doc_id = uuid4()
        chunk_id = uuid4()
        now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        entity = Entity(
            namespace_id=uuid4(),
            name="Alice",
            entity_type="PERSON",
            description="A person",
            attributes={"role": "engineer"},
            source_document_ids=[doc_id],
            source_chunk_ids=[chunk_id],
            mention_count=3,
            confidence=0.95,
            created_at=now,
            updated_at=now,
        )

        params = _entity_to_cypher_params(entity)

        assert params["id"] == str(entity.id)
        assert params["namespace_id"] == str(entity.namespace_id)
        assert params["name"] == "Alice"
        assert params["entity_type"] == "PERSON"
        assert params["description"] == "A person"
        assert params["source_document_ids"] == [str(doc_id)]
        assert params["source_chunk_ids"] == [str(chunk_id)]
        assert params["mention_count"] == 3
        assert params["confidence"] == 0.95
        assert params["created_at"] == now.isoformat()
        assert params["updated_at"] == now.isoformat()
        assert params["valid_from"] is None
        assert params["valid_until"] is None

    def test_string_entity_type(self) -> None:
        """String entity types pass through without .value."""
        entity = Entity(
            namespace_id=uuid4(),
            name="Widget",
            entity_type="CUSTOM_TYPE",
            description="",
        )
        params = _entity_to_cypher_params(entity)
        assert params["entity_type"] == "CUSTOM_TYPE"

    def test_temporal_fields(self) -> None:
        """valid_from and valid_until pass through as native datetime objects (#1472).

        The neo4j driver serializes a Python ``datetime`` to a native ZONED
        DATETIME property, so the temporal filter is a cast-free indexed range.
        """
        vf = datetime(2024, 1, 1, tzinfo=UTC)
        vu = datetime(2024, 12, 31, tzinfo=UTC)
        entity = Entity(
            namespace_id=uuid4(),
            name="Test",
            entity_type="CONCEPT",
            valid_from=vf,
            valid_until=vu,
        )
        params = _entity_to_cypher_params(entity)
        assert params["valid_from"] == vf
        assert params["valid_until"] == vu
        assert isinstance(params["valid_from"], datetime)
        assert isinstance(params["valid_until"], datetime)

    def test_attributes_serialized(self) -> None:
        """Attributes dict is JSON-serialized."""
        entity = Entity(
            namespace_id=uuid4(),
            name="Test",
            entity_type="CONCEPT",
            attributes={"key": "value"},
        )
        params = _entity_to_cypher_params(entity)
        # _serialize_dict returns a JSON string
        assert isinstance(params["attributes"], str)
        assert "key" in params["attributes"]


class TestRelationshipToCypherParams:
    """Tests for _relationship_to_cypher_params helper."""

    def test_basic_relationship(self) -> None:
        """Converts a basic relationship to Cypher params dict."""
        src = uuid4()
        tgt = uuid4()
        now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=src,
            target_entity_id=tgt,
            relationship_type="WORKS_FOR",
            description="works for",
            properties={"since": "2020"},
            confidence=0.9,
            weight=1.5,
            created_at=now,
            updated_at=now,
        )

        params = _relationship_to_cypher_params(rel)

        assert params["id"] == str(rel.id)
        assert params["namespace_id"] == str(rel.namespace_id)
        assert params["source_id"] == str(src)
        assert params["target_id"] == str(tgt)
        assert params["description"] == "works for"
        assert params["confidence"] == 0.9
        assert params["weight"] == 1.5
        assert params["created_at"] == now.isoformat()
        assert params["valid_from"] is None

    def test_string_relationship_type(self) -> None:
        """String relationship types are handled (no .value call)."""
        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="CUSTOM_REL",
        )
        # _relationship_to_cypher_params doesn't include relationship_type
        # in the dict (it's used for the Cypher label), but all other
        # fields should be present
        params = _relationship_to_cypher_params(rel)
        assert "id" in params
        assert "source_id" in params
        assert "target_id" in params

    def test_properties_serialized(self) -> None:
        """Properties dict is JSON-serialized."""
        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="RELATES_TO",
            properties={"key": "value"},
        )
        params = _relationship_to_cypher_params(rel)
        assert isinstance(params["properties"], str)
        assert "key" in params["properties"]


class TestCoerceNeo4jDatetime:
    """Dual-read coercion for native ZONED DATETIME + legacy ISO strings (#1472)."""

    def test_none_passthrough(self) -> None:
        assert _coerce_neo4j_datetime(None) is None

    def test_legacy_iso_string(self) -> None:
        """Old graphs (and not-yet-backfilled nodes) still carry ISO strings."""
        out = _coerce_neo4j_datetime("2024-12-31T00:00:00+00:00")
        assert out == datetime(2024, 12, 31, tzinfo=UTC)

    def test_python_datetime_passthrough(self) -> None:
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        assert _coerce_neo4j_datetime(dt) is dt

    def test_native_neo4j_datetime(self) -> None:
        """A neo4j ``DateTime`` (the new native write form) coerces via to_native()."""
        from neo4j.time import DateTime

        native = DateTime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
        out = _coerce_neo4j_datetime(native)
        assert isinstance(out, datetime)
        assert out == datetime(2024, 6, 1, tzinfo=UTC)

    def test_unexpected_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _coerce_neo4j_datetime(12345)
