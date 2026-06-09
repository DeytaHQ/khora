"""Unit tests for the Postgres recall-filter compiler (Layer 4) — ``@internal``.

``compile_postgres(ast, ctx)`` lowers a canonical :class:`~khora.filter.ast.FilterNode`
into a SQLAlchemy boolean predicate over the ``khora_chunks`` temporal store. These
tests pin the §4 *Field-match contract* at the SQL level: each test builds an
AST (by validating a :class:`~khora.filter.RecallFilter` and lowering it with
:func:`~khora.filter.ast.parse_to_ast`), compiles it, and asserts on the rendered SQL
(``literal_binds`` so the structure is visible inline) — never re-implementing the
compiler, only inspecting what it emits.

The contract these tests lock (§4):

* Every operator ``$eq``/``$ne``/``$gt``/``$gte``/``$lt``/``$lte``/``$in``/``$nin``/``$exists``.
* System key (typed column) vs metadata (JSONB) emission paths, incl. nested paths.
* ``$and``/``$or``/``$not`` composition; the empty AST → match-everything (``true``).
* ``$date``: a system date key compiles to a direct column compare; a metadata
  ``$date`` literal goes through the guarded ``khora_try_timestamptz`` cast.
* The four field-match rules:
  1. metadata numeric range emits a ``jsonb_typeof`` gate (no bare cast);
  2. ``$ne``/``$nin`` emit ``col IS NULL OR ...`` for system cols and include-absent
     for metadata;
  3. a list operand against a scalar system column → constant ``false``;
  4. ``$exists`` / null use ``?`` / ``#>``-vs-``null``, not ``->>`` value extraction.
* Array containment: scalar ``$eq`` on metadata → OR of the scalar-doc ``@>``
  and the array-wrapped ``@>`` (so it matches both a scalar field and an
  array-valued field containing the value); ``$in`` → OR of those containments;
  array operand ``$eq`` → exact ``#>`` = jsonb.

If the compiler module is not yet on the branch (implementation lands separately),
the whole module skips cleanly so the suite stays green.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.elements import ColumnElement

from khora.filter import RecallFilter
from khora.filter.ast import FilterClause, FilterNode, parse_to_ast
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.context import CompileContext
from khora.filter.model import Op

# Hard import (NOT importorskip): the compiler is on the branch, so an import
# failure here must be a LOUD test error — never a silent module skip that would
# pass CI green with zero coverage.

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

_CTX = CompileContext(backend_target="khora_chunks")


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _sql(node: FilterNode | FilterClause, ctx: CompileContext = _CTX) -> str:
    """Compile an AST and render the predicate as inline-literal Postgres SQL.

    The predicate is a SQLAlchemy ``ColumnElement[bool]``; rendering with
    ``literal_binds`` makes the emitted structure (operators, JSONB paths, casts,
    null-guards) visible as a single string the tests can assert substrings on.
    """
    compiled = compile_postgres(node, ctx)
    predicate = compiled.predicate
    assert isinstance(predicate, ColumnElement), f"predicate is not a SQLAlchemy ColumnElement: {type(predicate)!r}"
    return str(
        predicate.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def _norm(sql: str) -> str:
    """Collapse whitespace and lowercase for resilient substring matching."""
    return " ".join(sql.split()).lower()


def _sql_norm(node: FilterNode | FilterClause, ctx: CompileContext = _CTX) -> str:
    return _norm(_sql(node, ctx))


# ===========================================================================
# Empty AST → match-everything.
# ===========================================================================


def test_empty_filter_compiles_to_true() -> None:
    # A bare RecallFilter() (no predicates) lowers to the empty match-everything
    # AND. The compiler must emit a tautology so the engine applies no constraint.
    sql = _sql_norm(_ast({}))
    assert sql == "true"


def test_empty_and_node_compiles_to_true() -> None:
    # Construct the empty AND directly — same match-everything contract.
    sql = _sql_norm(FilterNode(op=Op.AND, children=()))
    assert sql == "true"


# ===========================================================================
# Scalar operators on a SYSTEM (typed-column) key.
# ===========================================================================


def test_system_eq_is_column_equals() -> None:
    sql = _sql_norm(_ast({"source_name": "linear"}))
    assert "source_name" in sql
    assert "= 'linear'" in sql
    # System key is a typed column, NOT a JSONB extraction.
    assert "->>" not in sql
    assert "#>>" not in sql


def test_system_gt_is_column_compare() -> None:
    # A system DATE key takes a plain ISO-8601 string operand (DateOps parses it
    # to a datetime) — the `{"$date": ...}` typed literal is a METADATA-only form.
    sql = _sql_norm(_ast({"created_at": {"$gt": "2026-01-01T00:00:00Z"}}))
    assert "created_at >" in sql
    # Direct column compare on a TIMESTAMPTZ column — no try-cast wrapper.
    assert "khora_try_timestamptz" not in sql


@pytest.mark.parametrize(
    ("wire_op", "sql_op"),
    [
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ],
)
def test_system_date_range_operators(wire_op: str, sql_op: str) -> None:
    sql = _sql_norm(_ast({"occurred_at": {wire_op: "2026-03-04T05:06:07Z"}}))
    assert f"occurred_at {sql_op}" in sql


def test_system_eq_on_string_key_with_ops_form() -> None:
    sql = _sql_norm(_ast({"source_type": {"$eq": "slack"}}))
    assert "source_type" in sql
    assert "= 'slack'" in sql


# ===========================================================================
# Rule 2 — $ne / $nin include NULL / absent.
# ===========================================================================


def test_system_ne_includes_null_rows() -> None:
    # Rule 2: $ne on a system column must match rows where the column IS NULL
    # too (a missing value is "not equal"). Emitted as `col IS NULL OR col != x`.
    sql = _sql_norm(_ast({"source_name": {"$ne": "linear"}}))
    assert "source_name is null" in sql
    assert "!=" in sql or "<>" in sql
    assert " or " in sql


def test_system_nin_includes_null_rows() -> None:
    # $nin is the list form of $ne — NULL rows are included.
    sql = _sql_norm(_ast({"source_name": {"$nin": ["a", "b"]}}))
    assert "source_name is null" in sql
    assert "not in" in sql
    assert " or " in sql


def test_metadata_ne_includes_absent_key() -> None:
    # Rule 2 for metadata: $ne must also match rows where the key is ABSENT.
    # The compiler negates the total `@>` containment form — `NOT (... @> ...)`
    # is TRUE for an absent key (containment is FALSE there, never NULL), so the
    # absent row is admitted.
    sql = _sql_norm(_ast({"metadata.tier": {"$ne": "gold"}}))
    assert "not (" in sql
    assert "@>" in sql


# ===========================================================================
# Rule 3 — list operand vs scalar SYSTEM column → constant false.
# ===========================================================================


def test_system_eq_array_operand_is_constant_false() -> None:
    # A bare-list on a scalar system column lowers to $eq EXACT-ARRAY. A scalar
    # TIMESTAMPTZ/VARCHAR column can never equal an array → the compiler folds it
    # to a constant false (no row can match) rather than emitting invalid SQL.
    sql = _sql_norm(_ast({"source_name": ["a", "b"]}))
    assert sql == "false"


def test_system_date_in_is_membership_list_not_false() -> None:
    sql = _sql_norm(_ast({"occurred_at": {"$in": ["2026-01-01T00:00:00Z"]}}))
    # $in on a system DATE column is membership, NOT exact-array, so this is a
    # real IN list — distinct from the exact-array false case above.
    assert sql != "false"
    assert " in " in sql


# ===========================================================================
# $in / $nin on a SYSTEM column → IN / NOT IN list.
# ===========================================================================


def test_system_in_is_in_list() -> None:
    sql = _sql_norm(_ast({"source_name": {"$in": ["a", "b", "c"]}}))
    assert "source_name in (" in sql
    assert "'a'" in sql and "'b'" in sql and "'c'" in sql


def test_system_nin_is_not_in_list_with_null_guard() -> None:
    sql = _sql_norm(_ast({"source_type": {"$nin": ["spam", "junk"]}}))
    assert "not in" in sql
    assert "source_type is null" in sql


# ===========================================================================
# $exists on a SYSTEM column → constant (the column is structurally always
# present on the denormalized chunk row, so presence is not value-dependent).
# ===========================================================================


def test_system_exists_true_is_constant_true() -> None:
    # A system column always exists on the row — $exists:true is a tautology,
    # NOT a value-dependent IS NOT NULL check (postgres.py:182-184, deliberate).
    sql = _sql_norm(_ast({"source_name": {"$exists": True}}))
    assert sql == "true"


def test_system_exists_false_is_constant_false() -> None:
    # ...and $exists:false matches nothing — a present column is never "absent".
    # NULL-ness on a system column is reached via {"key": null} ($eq null), not
    # via $exists:false.
    sql = _sql_norm(_ast({"source_name": {"$exists": False}}))
    assert sql == "false"


# ===========================================================================
# Null operand ($eq None) — active null-or-missing match.
# ===========================================================================


def test_system_null_operand_is_is_null() -> None:
    # An explicit null is an active null-or-missing match → IS NULL.
    sql = _sql_norm(_ast({"source_name": None}))
    assert "source_name is null" in sql


# ===========================================================================
# Metadata (JSONB) paths.
# ===========================================================================


def test_metadata_eq_scalar_uses_containment() -> None:
    # Array-containment rule: a scalar $eq on a metadata field is emitted as the
    # OR of two JSONB containments (@>) — the scalar-doc form AND an
    # array-wrapped form — so it matches both a scalar value AND membership in an
    # array-valued field (JSONB @> does not treat {"tier":"gold"} as contained by
    # {"tier":["gold"]}, so the array-wrapped form is required). Never a brittle
    # ->> text compare.
    sql = _sql_norm(_ast({"metadata.tier": "gold"}))
    assert "@>" in sql
    # Both the scalar-doc and the array-wrapped containment docs are emitted.
    assert '{"tier": "gold"}' in sql
    assert '{"tier": ["gold"]}' in sql


def test_metadata_nested_path_extraction() -> None:
    # A nested metadata path addresses through the JSONB document with #> / #>>.
    sql = _sql_norm(_ast({"metadata.labels.tier": {"$gte": 5}}))
    assert "#>" in sql  # covers both #> and #>>
    # The path segments below the metadata root appear in the extraction.
    assert "labels" in sql and "tier" in sql


def test_metadata_numeric_range_has_jsonb_typeof_gate() -> None:
    # Rule 1: a metadata numeric range must gate on jsonb_typeof(...) = 'number'
    # BEFORE casting, so a string / array / object value is excluded instead of
    # erroring the statement on a bad cast.
    sql = _sql_norm(_ast({"metadata.score": {"$gte": 10}}))
    assert "jsonb_typeof" in sql
    assert "'number'" in sql


@pytest.mark.parametrize("wire_op", ["$gt", "$gte", "$lt", "$lte"])
def test_metadata_all_ranges_gate_on_typeof(wire_op: str) -> None:
    sql = _sql_norm(_ast({"metadata.score": {wire_op: 5}}))
    assert "jsonb_typeof" in sql


def test_metadata_in_is_or_of_containments_or_overlap() -> None:
    # $in on a metadata field is membership across scalar and array fields: an OR
    # of @> containments — each $in value contributing BOTH a scalar-doc and an
    # array-wrapped containment so membership spans scalar- and array-valued
    # fields.
    sql = _sql_norm(_ast({"metadata.tier": {"$in": ["gold", "silver"]}}))
    assert "@>" in sql
    # Each operand value emits its scalar-doc and array-wrapped containment.
    assert '{"tier": "gold"}' in sql
    assert '{"tier": ["gold"]}' in sql
    assert '{"tier": "silver"}' in sql
    assert '{"tier": ["silver"]}' in sql


def test_metadata_empty_in_is_constant_false() -> None:
    # An empty $in operand list is a valid filter (the validator accepts it) with
    # a defined row-set: a positive membership over ∅ matches nothing. It must
    # render as a constant FALSE — never a vanishing sa.or_() that the enclosing
    # AND would drop (which would wrongly match every row).
    sql = _sql_norm(_ast({"metadata.tier": {"$in": []}}))
    assert sql == "false"


def test_metadata_empty_nin_is_constant_true() -> None:
    # An empty $nin operand list matches everything (negation over ∅). It must
    # render as a constant TRUE — never sa.not_() of an empty OR chain, which is
    # invalid SQL that would error at execute time.
    sql = _sql_norm(_ast({"metadata.tier": {"$nin": []}}))
    assert sql == "true"


def test_metadata_array_operand_eq_is_exact_jsonb_match() -> None:
    # A bare list on a metadata path is $eq EXACT-ARRAY: the field must equal the
    # whole JSON array, emitted as #> = <jsonb array> (NOT containment, NOT $in).
    sql = _sql_norm(_ast({"metadata.tags": ["x", "y"]}))
    assert "#>" in sql
    # Exact array equality renders the array as a jsonb literal on the RHS.
    assert "jsonb" in sql or "::json" in sql or "[" in sql


def test_metadata_dict_operand_eq_is_exact_object_match_not_containment() -> None:
    # A dict on a metadata path is $eq object_equal: the extracted node must EXACTLY
    # equal the subdocument, emitted as #> = <jsonb object> (NOT @> containment, so a
    # stored object with extra keys does NOT match). Keys render sorted, so the
    # compare is key-order-insensitive (PG JSONB `=` is structural).
    sql = _sql_norm(_ast({"metadata.labels": {"tier": "gold", "team": "x"}}))
    assert "#>" in sql
    # Exact equality, never @> containment, for the dict operand.
    assert "@>" not in sql
    # Keys are sorted in the jsonb literal (order-insensitive structural compare).
    assert '{"team": "x", "tier": "gold"}' in sql


def test_metadata_dict_operand_ne_negates_exact_object_match() -> None:
    # $ne on a dict operand negates the total exact-object form. The positive form
    # is coalesced to FALSE, so NOT flips absent / wrong-type rows to TRUE (Rule 2
    # polarity — a row missing the key satisfies $ne).
    sql = _sql_norm(_ast({"metadata.labels": {"$ne": {"team": "x"}}}))
    assert "not" in sql
    assert "#>" in sql
    assert "@>" not in sql


def test_metadata_nested_dict_operand_eq_is_exact_object_match() -> None:
    # A nested metadata path with a dict operand addresses through #> and compares
    # the extracted node exactly against the subdocument.
    sql = _sql_norm(_ast({"metadata.outer.inner": {"team": "x"}}))
    assert "#>" in sql
    assert "outer" in sql and "inner" in sql
    assert "@>" not in sql


# ===========================================================================
# Rule 4 — $exists / null on metadata use key-presence, not value extraction.
# ===========================================================================


def test_metadata_exists_true_uses_has_key_operator() -> None:
    # Rule 4: $exists on metadata is a KEY-PRESENCE test (?), never a ->> value
    # extraction (which would be NULL for a present-but-null value).
    sql = _sql_norm(_ast({"metadata.tier": {"$exists": True}}))
    assert "?" in sql
    assert "->>" not in sql


def test_metadata_exists_false_negates_has_key() -> None:
    sql = _sql_norm(_ast({"metadata.tier": {"$exists": False}}))
    assert "?" in sql
    assert ("not" in sql) or ("is null" in sql)


def test_metadata_null_operand_uses_jsonb_null_not_text_extraction() -> None:
    # Rule 4: an active null match on metadata compares the JSONB value to
    # 'null'::jsonb via #> (json-null), NOT ->> (which conflates absent + null).
    sql = _sql_norm(_ast({"metadata.tier": None}))
    assert "#>" in sql
    assert "null" in sql
    assert "->>" not in sql


# ===========================================================================
# $date typed literal — system vs metadata path.
# ===========================================================================


def test_system_date_is_direct_column_compare() -> None:
    # A system date key takes a plain ISO string (the `$date` typed literal is a
    # metadata-only form). It compiles to a direct TIMESTAMPTZ column equality —
    # no guarded khora_try_timestamptz cast.
    sql = _sql_norm(_ast({"source_timestamp": "2026-02-03T00:00:00Z"}))
    assert "source_timestamp =" in sql
    assert "khora_try_timestamptz" not in sql


def test_metadata_date_literal_uses_guarded_cast() -> None:
    # A $date on a metadata path must go through khora_try_timestamptz so a
    # malformed stored string yields NULL (non-match) instead of erroring.
    sql = _sql_norm(_ast({"metadata.sent_at": {"$date": "2026-02-03T00:00:00Z"}}))
    assert "khora_try_timestamptz" in sql


def test_metadata_date_range_uses_guarded_cast() -> None:
    sql = _sql_norm(_ast({"metadata.sent_at": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}))
    assert "khora_try_timestamptz" in sql


# ===========================================================================
# Logical composition — AND / OR / NOT.
# ===========================================================================


def test_and_node_joins_with_and() -> None:
    sql = _sql_norm(_ast({"source_name": "linear", "source_type": "slack"}))
    assert " and " in sql
    assert "source_name" in sql and "source_type" in sql


def test_or_node_joins_with_or() -> None:
    sql = _sql_norm(_ast({"$or": [{"source_name": "linear"}, {"source_type": "slack"}]}))
    assert " or " in sql
    assert "source_name" in sql and "source_type" in sql


def test_not_node_negates_with_null_inclusive_total_child() -> None:
    # SEMANTICS over tokens: NOT compiles via sa.not_(child), and the child
    # equality is built as a TOTAL boolean (`coalesce(col = v, false)`) precisely
    # so the negation is NULL-INCLUSIVE — `NOT coalesce(NULL = v, false)` =
    # `NOT false` = TRUE flips a NULL/absent row IN (Rule 2), making $not($eq)
    # behave like $ne rather than dropping the NULL row. We assert the total-child
    # SHAPE here (the `coalesce(..., false)` wrap is the load-bearing guarantee);
    # the actual NULL-inclusion ROW-SET behavior is proven against real Postgres
    # in test_compile_postgres_rowset.py::test_not_eq_admits_null_row_like_ne.
    sql = _sql_norm(_ast({"$not": {"source_name": "linear"}}))
    assert sql.startswith("not ")
    # The total-child guard (coalesce → false) is what makes the negation
    # NULL-inclusive — assert it is present, not merely the != token.
    assert "coalesce(" in sql
    assert "false" in sql
    assert "source_name" in sql and "= 'linear'" in sql


def test_nested_and_or_structure() -> None:
    sql = _sql_norm(
        _ast(
            {
                "source_name": "linear",
                "$or": [{"source_type": "slack"}, {"content_type": "text/plain"}],
            }
        )
    )
    assert " and " in sql
    assert " or " in sql


# ===========================================================================
# CompiledFilter envelope — consumed_keys / canonical_hash / params.
# ===========================================================================


def test_compiled_filter_carries_canonical_hash() -> None:
    node = _ast({"source_name": "linear"})
    compiled = compile_postgres(node, _CTX)
    from khora.filter.ast import canonical_hash

    assert compiled.canonical_hash == canonical_hash(node)


def test_compiled_filter_consumed_keys_is_frozenset() -> None:
    compiled = compile_postgres(_ast({"source_name": "linear"}), _CTX)
    assert isinstance(compiled.consumed_keys, frozenset)


def test_empty_filter_consumes_nothing() -> None:
    compiled = compile_postgres(_ast({}), _CTX)
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# field_mapping — one compiler, different schemas, no engine branching.
# ===========================================================================


def test_field_mapping_remaps_system_column_name() -> None:
    # CompileContext.field_mapping lets the same compiler serve a different schema
    # (a column under a different name) without any per-engine `if engine == ...`
    # branching. The mapped name must resolve to a real column on the resolved
    # table — here we remap source_name → the existing `source` column.
    ctx = CompileContext(
        backend_target="khora_chunks",
        field_mapping={"source_name": "source"},
    )
    sql = _sql_norm(_ast({"source_name": "linear"}), ctx)
    assert "source = 'linear'" in sql


# ===========================================================================
# Datetime construction directly on the AST (no wire round-trip).
# ===========================================================================


def test_direct_datetime_clause_compiles() -> None:
    # Build a leaf clause directly with a tz-aware datetime operand — the AST is a
    # valid compiler input independent of the validator path.
    clause = FilterClause(
        path=("occurred_at",),
        op=Op.GTE,
        operand=datetime(2026, 1, 1, tzinfo=UTC),
    )
    node = FilterNode(op=Op.AND, children=(clause,))
    sql = _sql_norm(node)
    assert "occurred_at >=" in sql


# ===========================================================================
# Bare metadata blob — whole-document $eq equality.
# ===========================================================================


def test_bare_metadata_blob_eq_is_jsonb_document_equality() -> None:
    # A bare ``metadata`` dict is whole-document JSONB equality (PG `=` is
    # structural-normalized: key order / whitespace insensitive). Not a per-field
    # containment — the entire blob must equal the operand object.
    sql = _sql_norm(_ast({"metadata": {"a": 1, "b": 2}}))
    assert "metadata" in sql
    assert "=" in sql
    assert "jsonb" in sql
    # Whole-blob equality, NOT array containment / extraction.
    assert "@>" not in sql
    assert "#>" not in sql


# ===========================================================================
# $ne null on a metadata path — present AND not a JSON null value.
# ===========================================================================


def test_metadata_ne_null_is_present_and_not_json_null() -> None:
    # {"metadata.k": {"$ne": null}} → NOT(active-null-or-missing): the key is
    # present AND its value is not an explicit JSON null. Emitted as the negation
    # of the null-or-missing form.
    sql = _sql_norm(_ast({"metadata.tier": {"$ne": None}}))
    assert sql.startswith("not (")
    assert "null" in sql
    assert "?" in sql  # the presence half of the negated null-or-missing form


# ===========================================================================
# $in with $date elements — containment doc carries the ISO string.
# ===========================================================================


def test_metadata_in_with_date_literals_uses_iso_containment() -> None:
    # A $date element of a metadata $in becomes an ISO-8601 string inside the @>
    # containment doc (a datetime is not directly JSON-serializable).
    sql = _sql_norm(_ast({"metadata.day": {"$in": [{"$date": "2026-01-01T00:00:00Z"}]}}))
    assert "@>" in sql
    assert "2026-01-01" in sql


# ===========================================================================
# on_unsupported policy — "raise" vs "split".
# ===========================================================================


def _unknown_key_node() -> FilterNode:
    # A leaf whose path is neither a system key nor a metadata path — the only
    # clause this backend cannot express (should not occur post-validation).
    return FilterNode(op=Op.AND, children=(FilterClause(path=("not_a_key",), op=Op.EQ, operand="x"),))


def test_unsupported_clause_raises_in_raise_mode() -> None:
    from khora.filter import RecallFilterUnsupportedError

    ctx = CompileContext(backend_target="khora_chunks", on_unsupported="raise")
    with pytest.raises(RecallFilterUnsupportedError):
        compile_postgres(_unknown_key_node(), ctx)


def test_unsupported_clause_splits_to_nonconstraining_true() -> None:
    # In "split" mode the engine post-filters what the backend cannot express, so
    # the compiler emits a non-constraining predicate and leaves the key OUT of
    # consumed_keys (it does not narrow the result set).
    ctx = CompileContext(backend_target="khora_chunks", on_unsupported="split")
    compiled = compile_postgres(_unknown_key_node(), ctx)
    sql = _norm(str(compiled.predicate.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})))
    assert sql == "true"
    assert "not_a_key" not in compiled.consumed_keys


# ===========================================================================
# table_alias — columns render under the alias.
# ===========================================================================


def test_table_alias_qualifies_columns() -> None:
    ctx = CompileContext(backend_target="khora_chunks", table_alias="kc")
    sql = _sql_norm(_ast({"source_name": "linear"}), ctx)
    assert "kc.source_name" in sql
