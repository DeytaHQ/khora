"""F-EXISTS primitive / routing assertions — compiler output, NO DB.

The F-EXISTS conformance family asserts row-sets across the 8-state ``$exists``
truth table (absent / present-JSON-null / present, scalar and nested). Those
row-sets are checked by the catalog test through the Python oracle. *This* module
guards the layer below the row-sets: the **primitive each backend compiler emits**
for a metadata ``$exists``. The make-or-break states are s4/s7 (a metadata key
present but explicitly JSON-``null``): a backend that reaches the value instead of
testing key-presence collapses "absent" and "present-but-null" into one answer and
silently mis-classifies those two states.

So each test here compiles a *literal* ``$exists`` filter through the REAL
backend compiler and asserts on ``compiled.predicate`` (and ``compiled.params``):

* **SQLite / Lance** must use ``json_type`` — NOT ``json_extract``. ``json_extract``
  returns SQL ``NULL`` for BOTH an absent path AND a present JSON-``null`` value, so
  it cannot tell s3 (absent) from s4 (present-null). ``json_type`` returns ``NULL``
  only for an absent path and the string ``'null'`` for a present JSON-null, so
  ``json_type(...) IS NOT NULL`` is the presence test that keeps s4/s7.
* **Postgres** must test key-presence, not value-extraction: a single segment uses
  the GIN-friendly ``?`` has-key operator; a nested path uses ``#> ... IS NOT NULL``
  (``#>`` is SQL ``NULL`` only when the path is missing, ``'null'::jsonb`` for a
  present JSON-null). Never a brittle ``->>`` text compare (which is ``NULL`` for a
  present-but-null value, collapsing s4/s7).
* **SurrealDB** must use its NONE-boolean presence test: ``IS NOT NONE`` /
  ``IS NONE`` (``NONE`` = absent path, ``NULL`` = explicit json null — distinct, so
  presence keeps the present-JSON-null row).

These assert the *primitive*, not just the rows — the s4/s7 distinction is the
thing that is easy to get wrong and invisible at the row-set layer on a seed that
omits a present-JSON-null record.
"""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.elements import ColumnElement

from khora.filter import RecallFilter
from khora.filter.ast import FilterNode, parse_to_ast
from khora.filter.compilers import compile_lance, compile_postgres, compile_surrealdb
from khora.filter.context import SchemaCapabilities
from khora.filter.execute import build_compile_context

pytestmark = pytest.mark.unit


# JSON1 is available in the embedded backend runtime; the default lance context
# advertises it (matching how the sqlite_lance backend builds its own context).
_JSON1 = SchemaCapabilities(sqlite_json1=True)


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter through the REAL validator + lower to the AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _pg_sql(wire: dict) -> str:
    """Compile a filter via the real Postgres compiler; render predicate as SQL.

    Rendering with ``literal_binds`` makes the emitted operators (``?`` has-key,
    ``#>`` path, null-guards) visible as a single lowercased string.
    """
    ctx = build_compile_context("khora_chunks")
    compiled = compile_postgres(_ast(wire), ctx)
    predicate = compiled.predicate
    assert isinstance(predicate, ColumnElement), f"predicate is not a ColumnElement: {type(predicate)!r}"
    sql = str(predicate.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    return " ".join(sql.split()).lower()


def _lance_predicate(wire: dict) -> str:
    """Compile a filter via the real Lance compiler; return the SQL predicate string."""
    ctx = build_compile_context("khora_chunks", schema_capabilities=_JSON1)
    return " ".join(compile_lance(_ast(wire), ctx).predicate.split()).lower()


def _surql_predicate(wire: dict) -> str:
    """Compile a filter via the real SurrealDB compiler; return the SurrealQL string."""
    ctx = build_compile_context("temporal_chunk", field_mapping={"metadata": "metadata_"})
    return " ".join(compile_surrealdb(_ast(wire), ctx).predicate.split()).lower()


# ===========================================================================
# SQLite / Lance — json_type, NEVER json_extract (the s4/s7 landmine).
# ===========================================================================


def test_lance_metadata_exists_uses_json_type_not_json_extract() -> None:
    # The make-or-break primitive: json_extract returns NULL for BOTH an absent
    # path (s3) and a present JSON-null value (s4), so it cannot distinguish them.
    # json_type returns NULL only for absent, 'null' (a non-NULL string) for a
    # present JSON-null — so "json_type(...) IS NOT NULL" keeps the s4 row.
    sql = _lance_predicate({"metadata.m": {"$exists": True}})
    assert "json_type" in sql
    assert "json_extract" not in sql
    assert "is not null" in sql


def test_lance_metadata_exists_false_negates_json_type() -> None:
    sql = _lance_predicate({"metadata.m": {"$exists": False}})
    assert "json_type" in sql
    assert "json_extract" not in sql
    # $exists:false is the negation of the presence test (NOT (... IS NOT NULL)).
    assert "not" in sql


def test_lance_nested_metadata_exists_uses_json_type() -> None:
    # The nested-path state (s6/s7) addresses through the JSON document; presence
    # is still json_type (a per-segment json_type IS NOT NULL), never json_extract.
    sql = _lance_predicate({"metadata.a.b": {"$exists": True}})
    assert "json_type" in sql
    assert "json_extract" not in sql


# ===========================================================================
# Postgres — key-presence (? / #> IS NOT NULL), NEVER ->> value extraction.
# ===========================================================================


def test_postgres_single_segment_exists_uses_has_key_operator() -> None:
    # A single metadata segment uses the GIN-friendly ``?`` has-key operator — a
    # KEY-PRESENCE test that is TRUE for a present JSON-null value (s4), never a
    # ->> text extraction (which is NULL for a present-but-null value, collapsing
    # s4 into the absent case).
    sql = _pg_sql({"metadata.m": {"$exists": True}})
    assert "?" in sql
    assert "->>" not in sql


def test_postgres_nested_segment_exists_uses_path_is_not_null() -> None:
    # A nested path uses ``#> ... IS NOT NULL`` — ``#>`` returns SQL NULL only when
    # the path is missing ('null'::jsonb for a present JSON-null), so IS NOT NULL is
    # the presence test that keeps s7. Still never a ->> text extraction.
    sql = _pg_sql({"metadata.a.b": {"$exists": True}})
    assert "#>" in sql
    assert "is not null" in sql
    assert "->>" not in sql


def test_postgres_exists_false_negates_presence() -> None:
    sql = _pg_sql({"metadata.m": {"$exists": False}})
    # Negation of the has-key presence test; still key-presence, not ->> extraction.
    assert "?" in sql
    assert "->>" not in sql
    assert ("not" in sql) or ("is null" in sql)


# ===========================================================================
# SurrealDB — NONE-boolean presence (IS NOT NONE / IS NONE), distinct from NULL.
# ===========================================================================


def test_surrealdb_metadata_exists_uses_is_not_none() -> None:
    # SurrealQL distinguishes NONE (absent path) from NULL (explicit json null), so
    # presence is the NONE test: ``IS NOT NONE`` is TRUE for a present JSON-null
    # value (s4), keeping it distinct from the absent state.
    surql = _surql_predicate({"metadata.m": {"$exists": True}})
    assert "is not none" in surql


def test_surrealdb_metadata_exists_false_uses_is_none() -> None:
    surql = _surql_predicate({"metadata.m": {"$exists": False}})
    assert "is none" in surql
    assert "is not none" not in surql


def test_surrealdb_nested_metadata_exists_uses_is_not_none() -> None:
    # The nested path descends natively (metadata_.a.b) and still tests NONE-presence.
    surql = _surql_predicate({"metadata.a.b": {"$exists": True}})
    assert "is not none" in surql
    # Native nested descent — the segments are addressed, not collapsed to a token.
    assert "a.b" in surql
