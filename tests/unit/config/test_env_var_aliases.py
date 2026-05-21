"""Env-var alias coverage for Issue #789.

Locks in that every consolidated nested-config field accepts both the
canonical single-underscore form (e.g. ``KHORA_STORAGE_GRAPH_URL``) and
the legacy double-underscore form (``KHORA_STORAGE__GRAPH__URL``), and
that ``KhoraConfig`` raises when the two aliases disagree.

The single-underscore form is what gets exercised going forward; the
double-underscore form is preserved indefinitely for back-compat with
existing operator ``.env`` files.

These tests depend on PE#3's schema changes (per-field ``AliasChoices``
plus a conflict-detection model validator). If those changes haven't
landed yet, the alias / conflict assertions will fail — see the test
docstrings for which side of the alias they exercise.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from khora.config.schema import (
    AGEConfig,
    KhoraConfig,
    Neo4jConfig,
    NeptuneConfig,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# storage.graph (Neo4j default)
# ---------------------------------------------------------------------------


def test_storage_graph_url_via_single_underscore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical single-underscore form must populate ``storage.graph.url``."""
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "bolt://test:7687")

    config = KhoraConfig()
    assert isinstance(config.storage.graph, Neo4jConfig)
    assert config.storage.graph.url.get_secret_value() == "bolt://test:7687"


def test_storage_graph_url_via_double_underscore_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy double-underscore form must keep working."""
    monkeypatch.setenv("KHORA_STORAGE__GRAPH__BACKEND", "neo4j")
    monkeypatch.setenv("KHORA_STORAGE__GRAPH__URL", "bolt://legacy:7687")

    config = KhoraConfig()
    assert isinstance(config.storage.graph, Neo4jConfig)
    assert config.storage.graph.url.get_secret_value() == "bolt://legacy:7687"


@pytest.mark.parametrize(
    ("env_var", "expected"),
    [
        ("KHORA_STORAGE_GRAPH_QUERY_TIMEOUT", 7.5),
        ("KHORA_STORAGE__GRAPH__QUERY_TIMEOUT", 7.5),
    ],
)
def test_storage_graph_query_timeout_both_forms(monkeypatch: pytest.MonkeyPatch, env_var: str, expected: float) -> None:
    """Both alias spellings populate ``Neo4jConfig.query_timeout``."""
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "bolt://localhost:7687")
    monkeypatch.setenv(env_var, str(expected))

    config = KhoraConfig()
    assert isinstance(config.storage.graph, Neo4jConfig)
    assert config.storage.graph.query_timeout == expected


# ---------------------------------------------------------------------------
# storage.sqlite_lance
# ---------------------------------------------------------------------------


def test_storage_sqlite_lance_db_path_single_underscore(monkeypatch: pytest.MonkeyPatch) -> None:
    """``KHORA_STORAGE_SQLITE_LANCE_DB_PATH`` must populate the embedded backend."""
    monkeypatch.setenv("KHORA_STORAGE_BACKEND", "sqlite_lance")
    monkeypatch.setenv("KHORA_STORAGE_SQLITE_LANCE_DB_PATH", "/tmp/test.db")

    config = KhoraConfig()
    assert config.storage.sqlite_lance is not None
    assert config.storage.sqlite_lance.db_path == "/tmp/test.db"


def test_storage_sqlite_lance_db_path_double_underscore_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy double-underscore spelling must keep working for sqlite_lance."""
    monkeypatch.setenv("KHORA_STORAGE__BACKEND", "sqlite_lance")
    monkeypatch.setenv("KHORA_STORAGE__SQLITE_LANCE__DB_PATH", "/tmp/legacy.db")

    config = KhoraConfig()
    assert config.storage.sqlite_lance is not None
    assert config.storage.sqlite_lance.db_path == "/tmp/legacy.db"


# ---------------------------------------------------------------------------
# storage.vector
# ---------------------------------------------------------------------------


def test_storage_vector_embedding_dimension_single_underscore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KHORA_STORAGE_VECTOR_EMBEDDING_DIMENSION`` must reach
    ``storage.vector.embedding_dimension``.
    """
    monkeypatch.setenv("KHORA_STORAGE_VECTOR_EMBEDDING_DIMENSION", "768")

    config = KhoraConfig()
    assert config.storage.vector.embedding_dimension == 768


def test_storage_vector_embedding_dimension_double_underscore_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy double-underscore spelling must keep working for vector embed dim."""
    monkeypatch.setenv("KHORA_STORAGE__VECTOR__EMBEDDING_DIMENSION", "1024")

    config = KhoraConfig()
    assert config.storage.vector.embedding_dimension == 1024


# ---------------------------------------------------------------------------
# Neo4j *_max provenance fields (PE#1's audit gap)
# ---------------------------------------------------------------------------


def test_neo4j_relationship_source_document_ids_max_single_underscore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX`` populates the
    Neo4j-specific provenance cap. PE#1's audit flagged this field (and its
    three siblings) as missing from the documented env-var surface.
    """
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "bolt://localhost:7687")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX", "500")

    config = KhoraConfig()
    assert isinstance(config.storage.graph, Neo4jConfig)
    assert config.storage.graph.relationship_source_document_ids_max == 500


@pytest.mark.parametrize(
    ("field_name", "env_var", "value"),
    [
        (
            "relationship_source_chunk_ids_max",
            "KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_CHUNK_IDS_MAX",
            1000,
        ),
        (
            "entity_source_document_ids_max",
            "KHORA_STORAGE_GRAPH_ENTITY_SOURCE_DOCUMENT_IDS_MAX",
            300,
        ),
        (
            "entity_source_chunk_ids_max",
            "KHORA_STORAGE_GRAPH_ENTITY_SOURCE_CHUNK_IDS_MAX",
            777,
        ),
    ],
)
def test_neo4j_provenance_max_fields_single_underscore(
    monkeypatch: pytest.MonkeyPatch, field_name: str, env_var: str, value: int
) -> None:
    """The other three ``*_max`` provenance fields flagged by PE#1's audit."""
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "bolt://localhost:7687")
    monkeypatch.setenv(env_var, str(value))

    config = KhoraConfig()
    assert isinstance(config.storage.graph, Neo4jConfig)
    assert getattr(config.storage.graph, field_name) == value


# ---------------------------------------------------------------------------
# dream.ops nested
# ---------------------------------------------------------------------------


def test_dream_ops_dedupe_via_single_underscore(monkeypatch: pytest.MonkeyPatch) -> None:
    """``KHORA_DREAM_OPS_DEDUPE_ENTITIES`` must flip ``dream.ops.dedupe_entities``."""
    monkeypatch.setenv("KHORA_DREAM_OPS_DEDUPE_ENTITIES", "true")

    config = KhoraConfig()
    assert config.dream.ops.dedupe_entities is True


def test_dream_ops_dedupe_via_double_underscore_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy double-underscore form keeps working for dream.ops."""
    monkeypatch.setenv("KHORA_DREAM_OPS__DEDUPE_ENTITIES", "true")

    config = KhoraConfig()
    assert config.dream.ops.dedupe_entities is True


# ---------------------------------------------------------------------------
# Discriminated union — backend-specific fields
# ---------------------------------------------------------------------------


def test_age_backend_resolves_with_graph_name_via_single_underscore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``backend=age`` + ``graph_name`` selects ``AGEConfig`` at the discriminator."""
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_BACKEND", "age")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "postgresql://localhost:5432/khora")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_GRAPH_NAME", "test_graph")

    config = KhoraConfig()
    assert isinstance(config.storage.graph, AGEConfig)
    assert config.storage.graph.graph_name == "test_graph"


def test_neptune_backend_resolves_with_iam_and_region_via_single_underscore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``backend=neptune`` + ``iam_auth`` + ``aws_region`` selects ``NeptuneConfig``."""
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_BACKEND", "neptune")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "bolt://cluster:8182")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_IAM_AUTH", "true")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_AWS_REGION", "us-west-2")

    config = KhoraConfig()
    assert isinstance(config.storage.graph, NeptuneConfig)
    assert config.storage.graph.iam_auth is True
    assert config.storage.graph.aws_region == "us-west-2"


# ---------------------------------------------------------------------------
# Conflict-detection model validator
# ---------------------------------------------------------------------------


def test_alias_conflict_raises_when_values_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting both the new and legacy spellings to *different* values must raise.

    The conflict-detection validator must mention both env-var names so the
    operator can see exactly which spellings disagree.
    """
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "bolt://new:7687")
    monkeypatch.setenv("KHORA_STORAGE__GRAPH__URL", "bolt://old:7687")

    with pytest.raises(ValueError) as excinfo:
        KhoraConfig()
    message = str(excinfo.value)
    assert "KHORA_STORAGE_GRAPH_URL" in message
    assert "KHORA_STORAGE__GRAPH__URL" in message


def test_alias_no_conflict_when_values_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting both spellings to the *same* value succeeds — no conflict."""
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "bolt://both:7687")
    monkeypatch.setenv("KHORA_STORAGE__GRAPH__URL", "bolt://both:7687")

    config = KhoraConfig()
    assert isinstance(config.storage.graph, Neo4jConfig)
    assert config.storage.graph.url.get_secret_value() == "bolt://both:7687"


def test_dream_ops_alias_conflict_raises_when_values_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conflict detection must also fire for ``dream.ops.*`` aliases."""
    monkeypatch.setenv("KHORA_DREAM_OPS_DEDUPE_ENTITIES", "true")
    monkeypatch.setenv("KHORA_DREAM_OPS__DEDUPE_ENTITIES", "false")

    with pytest.raises(ValueError) as excinfo:
        KhoraConfig()
    message = str(excinfo.value)
    assert "KHORA_DREAM_OPS_DEDUPE_ENTITIES" in message
    assert "KHORA_DREAM_OPS__DEDUPE_ENTITIES" in message


# ---------------------------------------------------------------------------
# extra=forbid on discriminated-union members
# ---------------------------------------------------------------------------


def test_neptune_field_on_neo4j_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting a Neptune-only field on ``backend=neo4j`` must raise ``ValidationError``.

    Previously (pre-#789) such fields were silently dropped because the
    discriminated union members defaulted to ``extra='ignore'``. With
    ``extra='forbid'`` on every union member, the misconfiguration is now
    surfaced loudly at config-load time.
    """
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_URL", "bolt://localhost:7687")
    monkeypatch.setenv("KHORA_STORAGE_GRAPH_IAM_AUTH", "true")

    with pytest.raises(ValidationError):
        KhoraConfig()
