"""Unit tests for the PostgreSQL AGE graph backend.

Tests cover parsing helpers, escape utilities, domain model conversion,
config parsing, and protocol conformance -- all without requiring a
running PostgreSQL instance.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest

from khora.config.schema import AGEConfig
from khora.storage.backends.age import AGEBackend

# ---------------------------------------------------------------------------
# _escape()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEscape:
    """AGEBackend._escape() handles Cypher string injection vectors."""

    def test_empty_string(self):
        assert AGEBackend._escape("") == ""

    def test_plain_text(self):
        assert AGEBackend._escape("hello world") == "hello world"

    def test_single_quotes(self):
        assert AGEBackend._escape("it's a test") == "it\\'s a test"

    def test_backslashes(self):
        assert AGEBackend._escape("back\\slash") == "back\\\\slash"

    def test_newlines(self):
        assert AGEBackend._escape("line1\nline2") == "line1\\nline2"

    def test_carriage_returns(self):
        assert AGEBackend._escape("line1\rline2") == "line1\\rline2"

    def test_combined(self):
        result = AGEBackend._escape("it's a\nnew\\line\r")
        assert result == "it\\'s a\\nnew\\\\line\\r"


# ---------------------------------------------------------------------------
# _serialize_dict_literal() — Cypher injection regression coverage
# ---------------------------------------------------------------------------


import re  # noqa: E402 — kept local to the regression suite for readability


def _count_unescaped_quotes(s: str) -> int:
    """Count single quotes NOT preceded by a backslash (literal delimiters)."""
    return len(re.findall(r"(?<!\\)'", s))


@pytest.mark.unit
class TestSerializeDictLiteral:
    """``_serialize_dict_literal`` is the defense against caller-controlled
    Cypher injection via ``Entity.attributes``, ``Entity.metadata``,
    ``Relationship.properties / metadata``, and ``Episode.metadata``. Each
    of those fields lands inside ``f"...: '{serialize_dict(value)}'"`` in
    the Cypher template — without this helper, a single quote inside the
    serialised JSON closes the Cypher literal early and the rest is
    executed as Cypher.
    """

    def test_none_returns_empty_object_literal(self):
        assert AGEBackend._serialize_dict_literal(None) == "{}"

    def test_empty_dict_returns_empty_object_literal(self):
        assert AGEBackend._serialize_dict_literal({}) == "{}"

    def test_plain_dict_round_trips(self):
        out = AGEBackend._serialize_dict_literal({"k": "v"})
        # JSON-shaped output, no escape needed for plain text.
        assert out == '{"k": "v"}'

    def test_single_quote_in_value_is_escaped(self):
        # The canonical injection payload from the bug report. Without
        # escape, the embedded `'` closes the Cypher literal early.
        payload = {"note": "x'; MATCH (n) DETACH DELETE n; //"}
        out = AGEBackend._serialize_dict_literal(payload)
        assert "\\'" in out
        # Wrapped in the Cypher template, the unescaped-quote count must
        # be exactly 2 (open + close), not 3 (open + payload-borne + close).
        fragment = f"attributes: '{out}',"
        assert _count_unescaped_quotes(fragment) == 2

    def test_backslash_in_value_is_escaped(self):
        out = AGEBackend._serialize_dict_literal({"path": "C:\\Users"})
        # JSON encodes the backslash as `\\`; our escape then doubles each
        # backslash, so the four-character JSON `"\\"` becomes the
        # eight-character Cypher fragment `"\\\\"`.
        assert "\\\\\\\\" in out

    def test_dollar_dollar_payload_does_not_break_sql_wrapping(self):
        # ``$$`` would close the legacy SQL dollar-quote wrapping. The
        # per-literal escape doesn't have to neutralise it (the value
        # round-trips into the graph literally) — the SQL-level defense
        # is the uniquely-tagged ``$khora_age$`` wrap in ``_cypher``.
        # Here we only assert that the literal stays balanced.
        payload = {"note": "x'; $$; DROP TABLE chunks; $$;"}
        out = AGEBackend._serialize_dict_literal(payload)
        fragment = f"attributes: '{out}',"
        assert _count_unescaped_quotes(fragment) == 2

    def test_newline_in_value_is_escaped(self):
        out = AGEBackend._serialize_dict_literal({"k": "a\nb"})
        # JSON encodes the newline as the two-char sequence `\n`; the
        # Cypher escape then turns the backslash into `\\`.
        assert "\\\\n" in out

    def test_assembled_cypher_fragment_is_balanced(self):
        # End-to-end check on the assembled fragment shape: the exact
        # template used inside create_entity / update_entity et al.
        for payload in [
            {"note": "single ' quote"},
            {"note": 'double " quote'},
            {"note": "$$-dollar-quote"},
            {"key": "x'); DROP TABLE chunks; //"},
            {"k": "v", "nested": {"inner": "with ' quote"}},
        ]:
            out = AGEBackend._serialize_dict_literal(payload)
            fragment = f"attributes: '{out}',"
            assert _count_unescaped_quotes(fragment) == 2, (
                f"payload {payload!r} produced fragment with "
                f"{_count_unescaped_quotes(fragment)} unescaped quotes "
                f"(want 2): {fragment!r}"
            )


# ---------------------------------------------------------------------------
# _sanitize_label()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSanitizeLabel:
    """AGEBackend._sanitize_label() delegates to ``sanitize_cypher_label`` so
    AGE produces the same UPPER_SNAKE_CASE labels as Neo4j / Memgraph /
    sqlite_lance (issue #749).  Prior to #749 AGE preserved the user's case
    and silently produced an invalid empty-label Cypher fragment on empty
    input — both fixed by the shared helper."""

    def test_alphanumeric_passthrough(self):
        assert AGEBackend._sanitize_label("RELATES_TO") == "RELATES_TO"

    def test_special_chars_replaced(self):
        assert AGEBackend._sanitize_label("has space!") == "HAS_SPACE_"

    def test_empty(self):
        assert AGEBackend._sanitize_label("") == "RELATES_TO"


# ---------------------------------------------------------------------------
# _parse_agtype()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseAgtype:
    """AGEBackend._parse_agtype() converts AGE result values to Python."""

    def test_none(self):
        assert AGEBackend._parse_agtype(None) is None

    def test_int(self):
        assert AGEBackend._parse_agtype(42) == 42

    def test_float(self):
        assert AGEBackend._parse_agtype(3.14) == 3.14

    def test_bool(self):
        assert AGEBackend._parse_agtype(True) is True

    def test_dict_passthrough(self):
        d = {"key": "value"}
        assert AGEBackend._parse_agtype(d) == d

    def test_json_string_node(self):
        node_json = json.dumps(
            {
                "id": 123,
                "label": "Entity",
                "properties": {"id": str(uuid4()), "name": "Test"},
            }
        )
        result = AGEBackend._parse_agtype(node_json)
        assert isinstance(result, dict)
        assert result["label"] == "Entity"

    def test_json_string_edge(self):
        edge_json = json.dumps(
            {
                "id": 456,
                "label": "RELATES_TO",
                "start_id": 1,
                "end_id": 2,
                "properties": {"weight": 1.0},
            }
        )
        result = AGEBackend._parse_agtype(edge_json)
        assert isinstance(result, dict)
        assert result["start_id"] == 1

    def test_plain_string(self):
        assert AGEBackend._parse_agtype("just a string") == "just a string"

    def test_agtype_suffix_stripped(self):
        """The ``::agtype`` suffix that AGE appends should be stripped."""
        raw = '"hello"::agtype'
        result = AGEBackend._parse_agtype(raw)
        assert result == "hello"

    def test_non_string_non_primitive(self):
        """Unknown types are stringified."""
        result = AGEBackend._parse_agtype(object())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _entity_from_agtype()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityFromAgtype:
    """AGEBackend._entity_from_agtype() converts parsed agtype to Entity."""

    def _backend(self) -> AGEBackend:
        return AGEBackend(database_url="postgresql://localhost/test")

    def test_valid_node(self):
        entity_id = str(uuid4())
        ns_id = str(uuid4())
        data = {
            "properties": {
                "id": entity_id,
                "namespace_id": ns_id,
                "name": "Alice",
                "entity_type": "PERSON",
                "description": "A person",
                "attributes": "{}",
                "source_document_ids": [],
                "source_chunk_ids": [],
                "mention_count": 3,
                "confidence": 0.95,
                "metadata": "{}",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
        entity = self._backend()._entity_from_agtype(data)
        assert entity is not None
        assert entity.id == UUID(entity_id)
        assert entity.namespace_id == UUID(ns_id)
        assert entity.name == "Alice"
        assert entity.entity_type == "PERSON"
        assert entity.mention_count == 3
        assert entity.confidence == 0.95

    def test_flat_dict(self):
        """When properties are flat (no 'properties' key)."""
        entity_id = str(uuid4())
        ns_id = str(uuid4())
        data = {
            "id": entity_id,
            "namespace_id": ns_id,
            "name": "Bob",
            "entity_type": "CONCEPT",
        }
        entity = self._backend()._entity_from_agtype(data)
        assert entity is not None
        assert entity.name == "Bob"

    def test_missing_id_returns_none(self):
        data = {"properties": {"name": "NoId"}}
        assert self._backend()._entity_from_agtype(data) is None


# ---------------------------------------------------------------------------
# _relationship_from_agtype()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRelationshipFromAgtype:
    """AGEBackend._relationship_from_agtype() converts parsed agtype to Relationship."""

    def _backend(self) -> AGEBackend:
        return AGEBackend(database_url="postgresql://localhost/test")

    def test_valid_edge(self):
        rel_id = str(uuid4())
        ns_id = str(uuid4())
        src_id = str(uuid4())
        tgt_id = str(uuid4())
        data = {
            "properties": {
                "id": rel_id,
                "namespace_id": ns_id,
                "description": "works at",
                "properties": "{}",
                "source_document_ids": [],
                "source_chunk_ids": [],
                "confidence": 0.8,
                "weight": 2.0,
                "metadata": "{}",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
        rel = self._backend()._relationship_from_agtype(data, src_id, tgt_id, "WORKS_AT")
        assert rel.id == UUID(rel_id)
        assert rel.source_entity_id == UUID(src_id)
        assert rel.target_entity_id == UUID(tgt_id)
        assert rel.relationship_type == "WORKS_AT"
        assert rel.confidence == 0.8
        assert rel.weight == 2.0


# ---------------------------------------------------------------------------
# _episode_from_agtype()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEpisodeFromAgtype:
    """AGEBackend._episode_from_agtype() converts parsed agtype to Episode."""

    def _backend(self) -> AGEBackend:
        return AGEBackend(database_url="postgresql://localhost/test")

    def test_valid_episode(self):
        ep_id = str(uuid4())
        ns_id = str(uuid4())
        data = {
            "properties": {
                "id": ep_id,
                "namespace_id": ns_id,
                "name": "Meeting",
                "description": "Team standup",
                "occurred_at": "2026-01-15T10:00:00+00:00",
                "duration_seconds": 900,
                "entity_ids": [],
                "source_document_ids": [],
                "source_chunk_ids": [],
                "metadata": "{}",
                "created_at": "2026-01-15T10:00:00+00:00",
                "updated_at": "2026-01-15T10:00:00+00:00",
            }
        }
        episode = self._backend()._episode_from_agtype(data)
        assert episode.id == UUID(ep_id)
        assert episode.name == "Meeting"
        assert episode.duration_seconds == 900


# ---------------------------------------------------------------------------
# AGEConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAGEConfig:
    """AGEConfig Pydantic model parses correctly."""

    def test_defaults(self):
        config = AGEConfig()
        assert config.backend == "age"
        assert config.url is None
        assert config.graph_name == "khora_graph"
        assert config.pool_size == 10
        assert config.max_overflow == 20

    def test_custom_values(self):
        config = AGEConfig(
            url="postgresql://localhost:5432/khora",
            graph_name="my_graph",
            pool_size=5,
            max_overflow=10,
        )
        assert config.url.get_secret_value() == "postgresql://localhost:5432/khora"
        assert config.graph_name == "my_graph"
        assert config.pool_size == 5
        assert config.max_overflow == 10


# ---------------------------------------------------------------------------
# from_config()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromConfig:
    """AGEBackend.from_config() maps AGEConfig fields correctly."""

    def test_from_config_mapping(self):
        config = AGEConfig(
            url="postgresql://localhost:5432/test",
            graph_name="test_graph",
            pool_size=5,
            max_overflow=15,
        )
        backend = AGEBackend.from_config(config)
        assert backend._database_url == "postgresql://localhost:5432/test"
        assert backend._graph_name == "test_graph"
        assert backend._pool_size == 5
        assert backend._max_overflow == 15

    def test_from_config_defaults(self):
        config = AGEConfig()
        backend = AGEBackend.from_config(config)
        assert backend._database_url == ""
        assert backend._graph_name == "khora_graph"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolConformance:
    """AGEBackend implements all GraphBackendProtocol methods."""

    def test_all_required_methods_exist(self):
        backend = AGEBackend(database_url="postgresql://localhost/test")
        required_methods = [
            "connect",
            "disconnect",
            "is_healthy",
            "create_entity",
            "get_entity",
            "get_entity_by_name",
            "update_entity",
            "delete_entity",
            "list_entities",
            "create_relationship",
            "get_relationship",
            "delete_relationship",
            "get_entity_relationships",
            "list_relationships",
            "create_episode",
            "get_episode",
            "list_episodes",
            "find_paths",
            "get_neighborhood",
            "search_entities_by_attribute",
            # Batch/aggregate from GraphBackendBase
            "get_entities_batch",
            "get_neighborhoods_batch",
            "count_entities",
        ]
        for method in required_methods:
            assert hasattr(backend, method), f"Missing method: {method}"
            assert callable(getattr(backend, method)), f"Not callable: {method}"


# ---------------------------------------------------------------------------
# _cypher_str_list()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCypherStrList:
    """AGEBackend._cypher_str_list() formats Python lists as Cypher literals."""

    def test_empty_list(self):
        assert AGEBackend._cypher_str_list([]) == "[]"

    def test_single_item(self):
        result = AGEBackend._cypher_str_list(["abc"])
        assert result == "['abc']"

    def test_multiple_items(self):
        result = AGEBackend._cypher_str_list(["a", "b", "c"])
        assert result == "['a', 'b', 'c']"

    def test_escapes_quotes(self):
        result = AGEBackend._cypher_str_list(["it's"])
        assert result == "['it\\'s']"
