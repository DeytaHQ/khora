"""Unit + conformance tests for the SQLite ``compile_lance`` compiler (Layer 4).

``@internal``. ``compile_lance(ast, ctx)`` lowers a canonical
:class:`~khora.filter.ast.FilterNode` into a SQLite ``WHERE``-fragment *string*
against ``khora_chunks`` (denormalized columns + a JSON-TEXT ``metadata`` column),
paired with ``params == {"args": [...]}`` — an ordered positional bind list, one
entry per ``?`` placeholder in depth-first emit order. It is the embedded-stack
sibling of :func:`~khora.filter.compilers.postgres.compile_postgres` /
:func:`~khora.filter.compilers.cypher.compile_cypher`.

This module has two halves:

* **EMIT** — build an AST (validate a :class:`RecallFilter`, lower with
  :func:`parse_to_ast`), compile it, and assert on the emitted predicate string +
  ``args`` for every operator, the system/metadata key kinds, the four §4 rules,
  ``$and``/``$or``/``$not``, the empty AST, the :class:`CompiledFilter` envelope,
  and the ``on_unsupported`` raise/split policy. Never re-implements the compiler,
  only inspects what it emits.
* **CONFORMANCE** — materialize a curated record corpus into an in-memory SQLite
  ``khora_chunks``-shaped table, run the ``compile_lance`` SQL, and assert the
  returned id-set equals the :func:`~khora.filter.compilers.python.compile_python`
  oracle row-set (run over the SAME records) for each curated filter. The
  in-memory ``compile_python`` predicate is the reference every backend compiler
  must agree with — its own per-rule behavior is pinned in
  ``tests/recall/test_compile_python.py``; here it is the row-set oracle, proving
  ``compile_lance``'s SQL partitions a real corpus identically. The F-EXISTS s4/s7
  presence cases (ABSENT vs PRESENT-NULL vs PRESENT) are exercised on the LANCE
  side explicitly — they ride the ``json_type`` path, which is the new code.

The SQLite specifics this locks:

* SQLite has no boolean type — the totality sentinel is the integer ``0`` (vs
  Postgres ``false()`` / Cypher ``false``); the empty AST is the truthy ``1``.
* Dates bind as ``.isoformat()`` strings (lexicographic compare).
* Metadata pushdown is gated on ``ctx.schema_capabilities.sqlite_json1``; metadata
  predicates use ``json_extract`` / ``json_type`` / ``json_each`` and bind the
  JSONPath as a ``?`` param. Three metadata cases (bare-blob ``$eq``,
  ``object_equal`` dict operand, ``$date`` compare) are unsupported even with
  JSON1 — they emit the non-constraining ``1`` and are left to the post-filter.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pytest

from khora.filter import RecallFilter, RecallFilterUnsupportedError
from khora.filter.ast import FilterClause, FilterNode, parse_to_ast
from khora.filter.compilers.lance import compile_lance
from khora.filter.compilers.python import compile_python
from khora.filter.context import CompileContext, SchemaCapabilities
from khora.filter.model import Op

# Hard import (NOT importorskip): the compiler is on the branch, so an import
# failure here must be a LOUD test error — never a silent module skip that would
# pass CI green with zero coverage.

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

# JSON1 is available in the embedded backend's runtime, so the default test
# context advertises it (matching how the sqlite_lance backend builds its ctx).
_JSON1 = SchemaCapabilities(sqlite_json1=True)
_CTX = CompileContext(backend_target="khora_chunks", schema_capabilities=_JSON1)
# A split-mode context for the deferred-metadata cases (the backend drives
# "split" so metadata it cannot express falls to the compile_python post-filter).
_CTX_SPLIT = CompileContext(backend_target="khora_chunks", schema_capabilities=_JSON1, on_unsupported="split")


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _norm(sql: str) -> str:
    """Collapse whitespace and lowercase for resilient substring matching."""
    return " ".join(sql.split()).lower()


def _lance(wire: dict, ctx: CompileContext = _CTX) -> Any:
    """Compile a wire filter and return its :class:`CompiledFilter`."""
    return compile_lance(_ast(wire), ctx)


def _lance_norm(wire: dict, ctx: CompileContext = _CTX) -> str:
    """Compile and return the predicate string, whitespace-collapsed + lowercased."""
    return _norm(_lance(wire, ctx).predicate)


# ===========================================================================
# Empty AST → match-everything (the SQLite truthy literal "1").
# ===========================================================================


def test_empty_filter_compiles_to_one() -> None:
    # A bare RecallFilter() lowers to the empty match-everything AND — SQLite has
    # no boolean, so the tautology is the integer literal "1".
    assert _lance_norm({}) == "1"


def test_empty_and_node_compiles_to_one() -> None:
    compiled = compile_lance(FilterNode(op=Op.AND, children=()), _CTX)
    assert compiled.predicate == "1"


def test_empty_filter_binds_nothing() -> None:
    assert _lance({}).params == {"args": []}


# ===========================================================================
# Scalar operators on a SYSTEM (typed-column) key.
# ===========================================================================


def test_system_eq_is_coalesced_column_equals() -> None:
    # $eq is a total boolean: coalesce(col = ?, 0) so an absent/NULL column reads
    # 0 (excludes) and a wrapping $not flips it in. The operand binds positionally.
    compiled = _lance({"source_name": "linear"})
    sql = _norm(compiled.predicate)
    assert "coalesce(khora_chunks.source_name = ?, 0)" in sql
    assert compiled.params == {"args": ["linear"]}
    # System key is a typed column, NOT a JSON extraction.
    assert "json_extract" not in sql


def test_system_eq_on_string_key_with_ops_form() -> None:
    compiled = _lance({"source_type": {"$eq": "slack"}})
    assert "coalesce(khora_chunks.source_type = ?, 0)" in _norm(compiled.predicate)
    assert compiled.params == {"args": ["slack"]}


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
    compiled = _lance({"occurred_at": {wire_op: "2026-03-04T05:06:07Z"}})
    sql = _norm(compiled.predicate)
    assert f"coalesce(khora_chunks.occurred_at {sql_op} ?, 0)" in sql
    # Dates bind as the UTC-normalized .isoformat() string (lexicographic compare).
    assert compiled.params["args"][0].startswith("2026-03-04")


def test_system_date_binds_iso_string() -> None:
    compiled = _lance({"source_timestamp": "2026-02-03T00:00:00Z"})
    assert "coalesce(khora_chunks.source_timestamp = ?, 0)" in _norm(compiled.predicate)
    assert compiled.params["args"][0].startswith("2026-02-03")


# ===========================================================================
# Rule 2 — $ne / $nin include NULL / absent (null-inclusive, no coalesce).
# ===========================================================================


def test_system_ne_includes_null_rows() -> None:
    compiled = _lance({"source_name": {"$ne": "linear"}})
    sql = _norm(compiled.predicate)
    assert "khora_chunks.source_name is null or khora_chunks.source_name <> ?" in sql
    assert compiled.params == {"args": ["linear"]}


def test_system_nin_includes_null_rows() -> None:
    compiled = _lance({"source_name": {"$nin": ["a", "b"]}})
    sql = _norm(compiled.predicate)
    assert "khora_chunks.source_name is null or not khora_chunks.source_name in (?, ?)" in sql
    assert compiled.params == {"args": ["a", "b"]}


# ===========================================================================
# $in / $nin on a SYSTEM column → coalesced IN / null-guarded NOT IN.
# ===========================================================================


def test_system_in_is_coalesced_in_list() -> None:
    compiled = _lance({"source_name": {"$in": ["a", "b", "c"]}})
    assert "coalesce(khora_chunks.source_name in (?, ?, ?), 0)" in _norm(compiled.predicate)
    assert compiled.params == {"args": ["a", "b", "c"]}


def test_system_empty_in_is_constant_zero() -> None:
    # A positive membership over ∅ matches nothing → the constant 0 (wrapped "(0)"
    # for the single-clause AND), binding nothing.
    compiled = _lance({"source_name": {"$in": []}})
    assert _norm(compiled.predicate) == "(0)"
    assert compiled.params == {"args": []}


def test_system_empty_nin_is_constant_one() -> None:
    # Negation over ∅ matches everything → the constant 1, binding nothing.
    compiled = _lance({"source_name": {"$nin": []}})
    assert _norm(compiled.predicate) == "(1)"
    assert compiled.params == {"args": []}


# ===========================================================================
# Rule 3 — bare-list (exact-array) operand vs scalar SYSTEM column → const 0.
# ===========================================================================


def test_system_eq_array_operand_is_constant_zero() -> None:
    # A bare list on a scalar system column lowers to $eq EXACT-ARRAY; a scalar
    # column never equals a list → unsatisfiable, folds to the constant 0.
    compiled = _lance({"source_name": ["a", "b"]})
    assert _norm(compiled.predicate) == "(0)"
    assert compiled.params == {"args": []}


def test_system_in_is_membership_not_exact_array() -> None:
    compiled = _lance({"occurred_at": {"$in": ["2026-01-01T00:00:00Z"]}})
    sql = _norm(compiled.predicate)
    assert "coalesce(khora_chunks.occurred_at in (?), 0)" in sql
    assert "<>" not in sql


# ===========================================================================
# Rule 4 (system) — $exists is constant; {k: null} / $ne null are IS [NOT] NULL.
# ===========================================================================


def test_system_exists_true_is_constant_one() -> None:
    # A system column is structurally always present → $exists True is constant 1.
    compiled = _lance({"source_name": {"$exists": True}})
    assert _norm(compiled.predicate) == "(1)"
    assert compiled.params == {"args": []}


def test_system_exists_false_is_constant_zero() -> None:
    compiled = _lance({"source_name": {"$exists": False}})
    assert _norm(compiled.predicate) == "(0)"
    assert compiled.params == {"args": []}


def test_system_null_operand_is_is_null() -> None:
    compiled = _lance({"source_name": None})
    assert "khora_chunks.source_name is null" in _norm(compiled.predicate)
    assert compiled.params == {"args": []}


def test_system_ne_null_is_is_not_null() -> None:
    compiled = _lance({"source_name": {"$ne": None}})
    assert "khora_chunks.source_name is not null" in _norm(compiled.predicate)
    assert compiled.params == {"args": []}


def test_direct_datetime_clause_binds_iso_string() -> None:
    clause = FilterClause(path=("occurred_at",), op=Op.GTE, operand=datetime(2026, 1, 1, tzinfo=UTC))
    node = FilterNode(op=Op.AND, children=(clause,))
    compiled = compile_lance(node, _CTX)
    assert "coalesce(khora_chunks.occurred_at >= ?, 0)" in _norm(compiled.predicate)
    assert compiled.params["args"][0] == datetime(2026, 1, 1, tzinfo=UTC).isoformat()


# ===========================================================================
# Logical composition — AND / OR / NOT.
# ===========================================================================


def test_and_node_joins_with_and() -> None:
    sql = _lance_norm({"source_name": "linear", "source_type": "slack"})
    assert " and " in sql
    assert "khora_chunks.source_name = ?" in sql
    assert "khora_chunks.source_type = ?" in sql


def test_or_node_joins_with_or() -> None:
    sql = _lance_norm({"$or": [{"source_name": "linear"}, {"source_type": "slack"}]})
    assert " or " in sql
    assert "khora_chunks.source_name = ?" in sql
    assert "khora_chunks.source_type = ?" in sql


def test_not_node_negates_with_total_child() -> None:
    # NOT compiles via (NOT (<child>)), and the child equality is the TOTAL
    # coalesce(col = ?, 0) so the negation is NULL-INCLUSIVE — NOT coalesce(NULL=?,
    # 0) = NOT 0 = true flips a NULL/absent row IN (Rule 2), making $not($eq)
    # behave like $ne. The coalesce(..., 0) wrap is the load-bearing guarantee.
    sql = _lance_norm({"$not": {"source_name": "linear"}})
    assert sql.startswith("(not (")
    assert "coalesce(khora_chunks.source_name = ?, 0)" in sql


def test_composition_binds_are_ordered_depth_first() -> None:
    # The positional args list follows depth-first emit order — three keys produce
    # three ordered binds, one per ? in the predicate.
    compiled = _lance(
        {
            "source_name": "linear",
            "$or": [{"source_type": "slack"}, {"content_type": "text/plain"}],
        }
    )
    args = compiled.params["args"]
    assert len(args) == 3
    assert set(args) == {"linear", "slack", "text/plain"}
    assert compiled.predicate.count("?") == 3


# ===========================================================================
# Metadata (JSON-TEXT) — JSON1-gated pushdown.
# ===========================================================================


def test_metadata_eq_scalar_uses_json_type_gated_containment() -> None:
    # A scalar $eq on a metadata field is array-aware containment via json_each:
    # EXISTS over json_each(metadata, ?) matching an element whose value equals the
    # operand AND whose json_type is in the operand's gate (a string => 'text'),
    # coalesced to total. json_each iterates the single scalar OR each array element,
    # so one form covers both. The JSONPath + value bind as ? params (the path never
    # enters the SQL text), preserving the bool-vs-number type distinction.
    compiled = _lance({"metadata.tier": "gold"})
    sql = _norm(compiled.predicate)
    assert "json_each(khora_chunks.metadata, ?)" in sql
    assert "je.value = ?" in sql
    assert "je.type in ('text')" in sql
    assert "coalesce(" in sql
    # The bound JSONPath addresses the quoted "tier" key; the operand binds too.
    assert '$."tier"' in compiled.params["args"]
    assert "gold" in compiled.params["args"]


def test_metadata_numeric_range_gates_on_json_type_number() -> None:
    # Rule 1: a metadata numeric range gates on json_type IN ('integer','real')
    # BEFORE comparing, so a string/bool/absent value reads 0 instead of erroring.
    sql = _lance_norm({"metadata.score": {"$gte": 5}})
    assert "json_type(khora_chunks.metadata, ?) in ('integer', 'real')" in sql
    assert ">= ?" in sql
    assert "coalesce(" in sql


def test_metadata_bool_eq_gates_on_json_type_token() -> None:
    # A bool $eq gates the json_each element's type on ('true','false') so a stored
    # JSON bool matches but a stored number 1 does NOT — preserving the oracle's
    # bool-vs-number distinction even though SQLite collapses a JSON bool to 0/1.
    compiled = _lance({"metadata.flag": True})
    sql = _norm(compiled.predicate)
    assert "json_each(khora_chunks.metadata, ?)" in sql
    assert "je.type in ('true', 'false')" in sql
    # The type gate is what bites; a bare numeric compare that would match True == 1
    # must not be the sole guard.
    assert "je.value = ?" in sql


def test_metadata_exists_true_uses_json_type_not_null() -> None:
    # Rule 4: $exists is presence via json_type IS NOT NULL — json_type is NULL for
    # an absent path and non-NULL (incl. 'null') for a present value. NOT json_extract
    # (which is NULL for both absent AND a stored JSON null).
    compiled = _lance({"metadata.tier": {"$exists": True}})
    sql = _norm(compiled.predicate)
    assert "json_type(khora_chunks.metadata, ?) is not null" in sql
    assert "json_extract" not in sql


def test_metadata_exists_false_negates_json_type_presence() -> None:
    sql = _lance_norm({"metadata.tier": {"$exists": False}})
    assert "not (json_type(khora_chunks.metadata, ?) is not null)" in sql


def test_metadata_null_match_uses_json_type_null_or_missing() -> None:
    # Rule 4: {k: null} is an active null-OR-missing match — json_type IS NULL
    # (absent path) OR json_type == 'null' (stored JSON null). Mirrors the oracle's
    # MISSING-or-None.
    sql = _lance_norm({"metadata.tier": None})
    assert "json_type(khora_chunks.metadata, ?) is null" in sql
    assert "= 'null'" in sql


def test_metadata_ne_includes_absent_via_negated_containment() -> None:
    # Rule 2 for metadata: $ne negates the total containment, so an absent / wrong-
    # type row (containment = 0) is admitted by the negation.
    sql = _lance_norm({"metadata.tier": {"$ne": "gold"}})
    assert sql.startswith("((not")
    assert "coalesce(" in sql


def test_metadata_array_operand_eq_is_exact_json_array_match() -> None:
    # A bare list on a metadata path is $eq EXACT-ARRAY: the node must be a JSON
    # array equal to the operand (json()-normalized, order-significant).
    compiled = _lance({"metadata.tags": ["x", "y"]})
    sql = _norm(compiled.predicate)
    assert "json_type(khora_chunks.metadata, ?) = 'array'" in sql
    assert "json(json_extract(khora_chunks.metadata, ?)) = json(?)" in sql
    assert json.dumps(["x", "y"]) in compiled.params["args"]


def test_metadata_in_is_or_of_containments() -> None:
    sql = _lance_norm({"metadata.tag": {"$in": ["gold", "silver"]}})
    assert " or " in sql
    assert "json_each(khora_chunks.metadata, ?)" in sql


def test_metadata_empty_in_is_constant_zero() -> None:
    compiled = _lance({"metadata.tag": {"$in": []}})
    assert _norm(compiled.predicate) == "(0)"


def test_metadata_empty_nin_is_constant_one() -> None:
    compiled = _lance({"metadata.tag": {"$nin": []}})
    assert _norm(compiled.predicate) == "(1)"


# ===========================================================================
# Metadata pushdown gated on JSON1 — absent → unsupported.
# ===========================================================================


def test_metadata_raises_when_json1_unavailable_in_raise_mode() -> None:
    # Without JSON1 the SQLite compiler cannot express any metadata leaf; the
    # default "raise" mode surfaces the public unsupported error.
    ctx = CompileContext(backend_target="khora_chunks", schema_capabilities=SchemaCapabilities(sqlite_json1=False))
    with pytest.raises(RecallFilterUnsupportedError):
        compile_lance(_ast({"metadata.tier": "gold"}), ctx)


def test_metadata_splits_to_nonconstraining_one_when_json1_unavailable() -> None:
    # In "split" mode (the backend's mode) a metadata leaf with no JSON1 falls to
    # the compile_python post-filter: emit the non-constraining "1", bind nothing
    # constraining, and leave the key OUT of consumed_keys.
    ctx = CompileContext(
        backend_target="khora_chunks",
        schema_capabilities=SchemaCapabilities(sqlite_json1=False),
        on_unsupported="split",
    )
    compiled = compile_lance(_ast({"metadata.tier": "gold"}), ctx)
    assert _norm(compiled.predicate) == "(1)"
    assert "metadata.tier" not in compiled.consumed_keys


# ===========================================================================
# Metadata cases unsupported in SQL even WITH JSON1 (deferred to post-filter).
# ===========================================================================
#
# Bare-blob $eq (SQLite json() is key-order-sensitive, the oracle is not),
# object_equal dict operand (same), a $date metadata compare (ISO parse-or-
# exclude SQLite cannot replicate to match the oracle), and a $in / $nin carrying
# an element json_each containment cannot match — a dict (object_equal-per-element,
# bound as a JSON-text scalar) or a None (a JSON-null member, which the type gate
# excludes). In raise mode they raise; in split mode they emit "1" and stay out of
# consumed_keys (the compile_python post-filter re-checks them).


@pytest.mark.parametrize(
    "wire",
    [
        {"metadata": {"a": 1}},  # bare-blob $eq
        {"metadata.labels": {"team": "x"}},  # object_equal dict operand
        {"metadata.due": {"$date": "2026-01-01T00:00:00Z"}},  # $date metadata compare
        {"metadata.labels": {"$in": [{"team": "x"}]}},  # dict $in element (object_equal)
        {"metadata.labels": {"$nin": [{"team": "x"}]}},  # dict $nin element (object_equal)
        {"metadata.labels": {"$in": [{"team": "x"}, "scalar"]}},  # mixed dict+scalar $in
        {"metadata.x": {"$in": [None, "v"]}},  # None $in element (JSON-null member)
        {"metadata.x": {"$nin": [None]}},  # None $nin element (JSON-null member)
    ],
)
def test_metadata_sql_unsupported_cases_raise_in_raise_mode(wire: dict) -> None:
    with pytest.raises(RecallFilterUnsupportedError):
        compile_lance(_ast(wire), _CTX)


@pytest.mark.parametrize(
    "wire",
    [
        {"metadata": {"a": 1}},
        {"metadata.labels": {"team": "x"}},
        {"metadata.due": {"$date": "2026-01-01T00:00:00Z"}},
        {"metadata.labels": {"$in": [{"team": "x"}]}},
        {"metadata.labels": {"$nin": [{"team": "x"}]}},
        {"metadata.labels": {"$in": [{"team": "x"}, "scalar"]}},
        {"metadata.x": {"$in": [None, "v"]}},
        {"metadata.x": {"$nin": [None]}},
    ],
)
def test_metadata_sql_unsupported_cases_split_to_nonconstraining_one(wire: dict) -> None:
    compiled = compile_lance(_ast(wire), _CTX_SPLIT)
    assert _norm(compiled.predicate) == "(1)"
    # The key is left to the compile_python post-filter — not consumed here.
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# on_unsupported policy on a structurally-unknown key (neither system nor metadata).
# ===========================================================================


def _unknown_key_node() -> FilterNode:
    return FilterNode(op=Op.AND, children=(FilterClause(path=("not_a_key",), op=Op.EQ, operand="x"),))


def test_unsupported_clause_raises_in_raise_mode() -> None:
    with pytest.raises(RecallFilterUnsupportedError):
        compile_lance(_unknown_key_node(), _CTX)


def test_unsupported_clause_splits_to_nonconstraining_one() -> None:
    compiled = compile_lance(_unknown_key_node(), _CTX_SPLIT)
    assert _norm(compiled.predicate) == "(1)"
    assert "not_a_key" not in compiled.consumed_keys


# ===========================================================================
# All-or-nothing OR / NOT deferral (split mode) — the false-exclude guard.
# ===========================================================================
#
# When ANY descendant leaf of an $or / $not cannot be pushed to SQL, the WHOLE
# node must defer to the non-constraining "1" (consuming nothing) — pushing only
# the consumable disjuncts would make the OR match-all here while a wrapping NOT
# would then FALSE-EXCLUDE. An $and is independent: a non-consumable child becomes
# "1" and the consumable siblings still narrow. The compile_python post-filter
# re-checks the deferred remainder (the exactness guarantee). A $date metadata
# compare and an object_equal dict operand are the canonical unconsumable leaves.


def test_or_with_unconsumable_disjunct_defers_whole_node() -> None:
    # $or mixing a pushable system key with an unconsumable $date metadata leaf:
    # the whole OR defers to "1" and consumes nothing (all-or-nothing) — the
    # post-filter narrows. Pushing only source_name would make the OR match-all.
    compiled = compile_lance(
        _ast({"$or": [{"source_name": "linear"}, {"metadata.due": {"$date": "2026-01-01T00:00:00Z"}}]}),
        _CTX_SPLIT,
    )
    assert compiled.predicate == "1"
    assert compiled.consumed_keys == frozenset()


def test_not_over_unconsumable_child_defers_whole_node() -> None:
    # $not over an object_equal dict operand (unconsumable) defers wholesale — a
    # partial push under a NOT is the false-exclude trap the all-or-nothing gate
    # closes.
    compiled = compile_lance(_ast({"$not": {"metadata.labels": {"team": "x"}}}), _CTX_SPLIT)
    assert compiled.predicate == "1"
    assert compiled.consumed_keys == frozenset()


def test_and_pushes_consumable_sibling_defers_unconsumable() -> None:
    # $and is independent: the pushable source_name narrows; the $date metadata
    # leaf becomes "1" and is left to the post-filter. consumed_keys carries only
    # the pushed key.
    compiled = compile_lance(
        _ast({"source_name": "linear", "metadata.due": {"$date": "2026-01-01T00:00:00Z"}}),
        _CTX_SPLIT,
    )
    sql = _norm(compiled.predicate)
    assert "coalesce(khora_chunks.source_name = ?, 0)" in sql
    assert " and 1" in sql  # the deferred $date leaf
    assert compiled.consumed_keys == frozenset({"source_name"})


# ===========================================================================
# CompiledFilter envelope — predicate type / params / consumed_keys / hash.
# ===========================================================================


def test_compiled_predicate_is_str_and_params_carry_args_list() -> None:
    compiled = _lance({"source_name": "linear"})
    assert isinstance(compiled.predicate, str)
    # params is the positional-bind carrier: {"args": [...]}.
    assert set(compiled.params) == {"args"}
    assert isinstance(compiled.params["args"], list)


def test_compiled_filter_carries_canonical_hash() -> None:
    from khora.filter.ast import canonical_hash

    node = _ast({"source_name": "linear"})
    compiled = compile_lance(node, _CTX)
    assert compiled.canonical_hash == canonical_hash(node)


def test_compiled_filter_consumed_keys_is_frozenset() -> None:
    compiled = _lance({"source_name": "linear"})
    assert isinstance(compiled.consumed_keys, frozenset)
    assert "source_name" in compiled.consumed_keys


def test_empty_filter_consumes_nothing() -> None:
    compiled = _lance({})
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# field_mapping / table_alias — one compiler, many schemas, no engine branching.
# ===========================================================================


def test_table_alias_qualifies_columns() -> None:
    # The qualifier is table_alias when set (the aliased FTS-join path); else
    # backend_target (the unaliased vector post-fetch path).
    ctx = CompileContext(backend_target="khora_chunks", schema_capabilities=_JSON1, table_alias="c")
    assert "coalesce(c.source_name = ?, 0)" in _lance_norm({"source_name": "linear"}, ctx)


def test_field_mapping_remaps_system_column_name() -> None:
    ctx = CompileContext(
        backend_target="khora_chunks",
        schema_capabilities=_JSON1,
        field_mapping={"source_name": "src"},
    )
    assert "coalesce(khora_chunks.src = ?, 0)" in _lance_norm({"source_name": "linear"}, ctx)


# ===========================================================================
# CONFORMANCE — compile_lance SQL row-set == compile_python oracle row-set.
# ===========================================================================
#
# Materialize a curated record corpus into an in-memory SQLite ``khora_chunks``-
# shaped table (denormalized columns + a JSON-TEXT ``metadata`` column), run the
# compile_lance WHERE fragment with its positional binds, and assert the returned
# id-set equals the compile_python oracle row-set over the SAME records. This is
# the cross-compiler parity check: a row the SQL returns must be exactly a row the
# in-memory oracle accepts. (The oracle's own per-rule behavior is pinned in
# tests/recall/test_compile_python.py; here we use it as the row-set reference.)
#
# Uses stdlib sqlite3 only (no aiosqlite / lancedb), so it runs in the unit suite
# without the embedded extras. SQLite's JSON1 is compiled in by default on modern
# builds; the conftest-free skip below guards the rare build without it.


# Curated corpus spanning ABSENT / PRESENT-NULL / PRESENT for source_name and
# metadata.tier (the F-EXISTS s4/s7 presence states), plus number/string/bool/
# array stored shapes and a date string — so the lance ``json_type`` presence path
# is exercised across the full §4 contract.
#
# ``occurred_at`` is a SYSTEM DATE column carrying a tz-aware ``datetime``. It
# guards the date-precision edge: the backend stores it as an ``.isoformat()``
# string (lexicographic SQL compare) but decodes it back to a ``datetime`` before
# the post-filter, and the oracle compares datetimes — so the two views must agree
# even when stored values carry MICROSECONDS and the operand does not. r1 carries
# microseconds (``...00.500000+00:00``), r2 does not (``...00:00+00:00``); the
# ``.`` (0x2E) > ``+`` (0x2B) byte ordering makes the lexicographic string compare
# agree with chronological order for these UTC-normalized values, which is exactly
# the invariant the system-date pushdown relies on.
_MICRO_TS = datetime(2026, 6, 1, 12, 0, 0, 500000, tzinfo=UTC)  # has microseconds
_PLAIN_TS = datetime(2026, 1, 15, 8, 30, 0, tzinfo=UTC)  # no microseconds
_OLD_TS = datetime(2025, 3, 1, 0, 0, 0, 999999, tzinfo=UTC)  # before the bound, microseconds

_CORPUS: dict[str, dict[str, Any]] = {
    "r1": {
        "source_name": "linear",
        "source_type": "issue",
        "occurred_at": _MICRO_TS,
        "metadata": {"tier": "gold", "score": 9, "tags": ["urgent", "p1"], "flag": True},
    },
    "r2": {
        "source_name": "linear",
        "source_type": "doc",
        "occurred_at": _PLAIN_TS,
        "metadata": {"tier": "silver", "score": 3, "tags": ["later"], "flag": False},
    },
    "r3": {
        "source_name": "slack",
        "source_type": "msg",
        "occurred_at": _OLD_TS,
        "metadata": {"tier": "bronze", "score": "n/a", "tags": ["p1"]},
    },
    "r4": {
        "source_name": None,
        "source_type": "issue",
        "occurred_at": _MICRO_TS,
        "metadata": {"tier": None, "score": 7},
    },
    "r5": {
        "source_type": "issue",
        # occurred_at ABSENT (NULL column) — a positive date compare must exclude it.
        "metadata": {"score": 5, "other": "x"},
    },
    "r6": {
        "source_name": "linear",
        "source_type": "doc",
        "occurred_at": _PLAIN_TS,
    },
    "r7": {
        "source_name": "github",
        "source_type": "pr",
        "occurred_at": _MICRO_TS,
        "metadata": {"tier": "gold", "score": 5, "flag": 1, "due": "2026-06-01T00:00:00Z"},
    },
}

# The string system-key columns the conformance table materializes. ``occurred_at``
# is materialized separately (as an ISO-string TEXT column) because its oracle
# value is a ``datetime`` while its stored value is the ``.isoformat()`` string —
# the production split (SQL compares text, post-filter compares the decoded dt).
_SYSTEM_COLS = ("source_name", "source_type")


def _json1_available() -> bool:
    """True iff this stdlib SQLite build has the JSON1 functions."""
    con = sqlite3.connect(":memory:")
    try:
        con.execute("SELECT json_extract('{}', '$.x')")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        con.close()


def _materialize(con: sqlite3.Connection) -> None:
    """Create a khora_chunks-shaped table and insert the curated corpus.

    ``metadata`` is stored as JSON-TEXT (the embedded backend's column shape) and
    ``occurred_at`` as an ISO-string TEXT column (``.isoformat()``, mirroring how
    the backend persists a datetime) — so the SQL compares the stored string
    lexicographically while the oracle compares the decoded ``datetime``. An absent
    system key (string or date) is inserted as SQL NULL; an absent metadata blob is
    the column DEFAULT ``'{}'`` (coalesced-empty, matching the oracle's read).
    """
    cols = ", ".join(f"{c} TEXT" for c in _SYSTEM_COLS)
    con.execute(
        f"CREATE TABLE khora_chunks "
        f"(id TEXT PRIMARY KEY, {cols}, occurred_at TEXT, metadata TEXT NOT NULL DEFAULT '{{}}')"
    )
    col_list = ", ".join(("id", *_SYSTEM_COLS, "occurred_at", "metadata"))
    placeholders = ", ".join(["?"] * (3 + len(_SYSTEM_COLS)))
    for rid, rec in _CORPUS.items():
        occurred = rec.get("occurred_at")
        occurred_text = occurred.isoformat() if occurred is not None else None
        values = [rid, *(rec.get(c) for c in _SYSTEM_COLS), occurred_text, json.dumps(rec.get("metadata", {}))]
        con.execute(f"INSERT INTO khora_chunks ({col_list}) VALUES ({placeholders})", values)  # noqa: S608 — controlled tokens
    con.commit()


def _oracle_ids(wire: dict) -> set[str]:
    """The compile_python oracle row-set for ``wire`` over the curated corpus.

    The oracle reads each record dict directly, so ``occurred_at`` is the
    tz-aware ``datetime`` (the decoded-chunk view the production post-filter sees),
    NOT the ISO string the SQLite table stores.
    """
    pred = compile_python(_ast(wire), _CTX).predicate
    return {rid for rid, rec in _CORPUS.items() if pred(rec)}


def _lance_ids(con: sqlite3.Connection, wire: dict) -> set[str]:
    """The compile_lance SQL row-set for ``wire`` over the materialized table.

    Runs in split mode (the backend's mode) so a deferred-metadata leaf emits the
    non-constraining ``1`` rather than raising; the conformance corpus filters
    below are all fully pushable, so the SQL alone is the row-set under test.
    """
    compiled = compile_lance(_ast(wire), _CTX_SPLIT)
    sql = f"SELECT id FROM khora_chunks WHERE {compiled.predicate}"  # noqa: S608 — predicate is compiler output, binds are positional
    rows = con.execute(sql, compiled.params["args"]).fetchall()
    return {row[0] for row in rows}


# Curated filter wires that are FULLY pushable to SQLite (every leaf is a system
# key or a JSON1-expressible metadata predicate) — so the SQL row-set must equal
# the oracle row-set exactly. The deferred-metadata cases (bare-blob, object_equal,
# $date) are intentionally excluded: they fall to the post-filter, so the SQL
# alone is not expected to match the oracle and the parity is asserted at the
# engine layer (the integration suite), not here.
_CONFORMANCE_WIRES: list[dict] = [
    {},  # match-everything
    {"source_name": "linear"},  # $eq, excludes present-null + absent
    {"source_name": {"$ne": "linear"}},  # $ne includes present-null + absent
    {"source_name": {"$in": ["linear", "github"]}},
    {"source_name": {"$nin": ["linear", "github"]}},
    {"source_name": {"$in": []}},  # empty $in → nothing
    {"source_name": {"$nin": []}},  # empty $nin → everything
    {"source_name": None},  # {k: null} system: present-null + absent
    {"source_name": {"$ne": None}},  # $ne null: present concrete
    {"source_name": {"$exists": True}},  # trivially all
    {"source_name": {"$exists": False}},  # trivially none
    {"source_name": ["linear"]},  # exact-array vs scalar → nothing
    {"metadata.tier": "gold"},  # metadata scalar containment
    {"metadata.tier": {"$ne": "gold"}},  # metadata $ne includes absent/null/wrong
    {"metadata.tier": {"$exists": True}},  # present incl present-null
    {"metadata.tier": {"$exists": False}},  # absent-only
    {"metadata.tier": None},  # null-or-missing
    {"metadata.tier": {"$ne": None}},  # present concrete only
    {"metadata.score": {"$gte": 5}},  # numeric range, type-gated
    {"metadata.score": {"$gt": 5}},  # strict boundary
    {"metadata.flag": {"$gte": 1}},  # bool-vs-number gate (only the real int 1)
    {"metadata.tags": "p1"},  # array-aware containment
    {"metadata.tags": ["later"]},  # exact-array equality
    {"metadata.tags": {"$in": ["urgent", "later"]}},  # contains-any over the stored tags array
    {"metadata.tier": {"$in": []}},  # empty $in
    {"metadata.tier": {"$nin": []}},  # empty $nin
    # System-date pushdown — DATE-PRECISION edge. The operand has NO microseconds;
    # stored values mix microsecond (r1/r4/r7) and non-microsecond (r2/r6) ISO
    # strings. The lexicographic SQL compare must agree with the oracle's datetime
    # compare across the fractional-seconds boundary (consumed → exact parity).
    {"occurred_at": {"$gte": "2026-01-15T08:30:00Z"}},  # bound == r2/r6 instant exactly
    {"occurred_at": {"$gt": "2026-01-15T08:30:00Z"}},  # strict: excludes the at-bound r2/r6
    {"occurred_at": {"$gte": "2026-06-01T12:00:00Z"}},  # bound below the .500000 micro rows
    {"occurred_at": {"$lt": "2026-06-01T12:00:00Z"}},  # upper bound straddling the micro rows
    {"occurred_at": "2026-06-01T12:00:00Z"},  # $eq vs a stored .500000-micro value (must NOT match)
    # Composition.
    {"source_name": "linear", "metadata.tier": "gold"},  # AND intersection
    {"$or": [{"source_name": "slack"}, {"metadata.tier": "gold"}]},  # OR union
    {"$not": {"source_name": "linear"}},  # null-inclusive complement
    {"$nor": [{"source_name": "linear"}, {"metadata.tier": "gold"}]},
    {"source_name": "linear", "metadata.tier": "gold", "metadata.score": {"$gte": 5}},  # 3-predicate
    # System date + metadata composed (both consumed): the date pushdown and the
    # metadata containment must intersect to the oracle row-set exactly.
    {"occurred_at": {"$gte": "2026-06-01T00:00:00Z"}, "metadata.tier": "gold"},
]


@pytest.mark.skipif(not _json1_available(), reason="stdlib SQLite build lacks JSON1 functions")
@pytest.mark.parametrize("wire", _CONFORMANCE_WIRES, ids=[json.dumps(w, sort_keys=True) for w in _CONFORMANCE_WIRES])
def test_compile_lance_sql_matches_compile_python_oracle(wire: dict) -> None:
    # The compile_lance SQL, run against the materialized corpus, must return the
    # EXACT id-set the compile_python oracle selects for the same filter. This is
    # the cross-compiler parity gate for every fully-pushable §4 predicate.
    con = sqlite3.connect(":memory:")
    try:
        _materialize(con)
        sql_ids = _lance_ids(con, wire)
        oracle_ids = _oracle_ids(wire)
        assert sql_ids == oracle_ids, (
            f"compile_lance / compile_python row-set divergence for {wire!r}: "
            f"sql={sorted(sql_ids)} oracle={sorted(oracle_ids)}"
        )
    finally:
        con.close()


# Mixed-deferral wires: at least one leaf is unconsumable in SQL ($date metadata,
# object_equal dict, or such a leaf under an $or / $not so the all-or-nothing gate
# defers the whole node). The SQL alone CANNOT match the oracle (it deliberately
# emits "1" for the deferred part) — but it must be SUPERSET-SAFE: the SQL row-set
# ⊇ the oracle row-set, never a false-exclude, because the compile_python
# post-filter only narrows what the SQL returned. The engine layer (integration
# suite) then composes SQL ∩ post-filter == oracle; here we pin the invariant the
# SQL half must uphold so a regression that over-narrows is caught at the unit level.
_SUPERSET_SAFE_WIRES: list[dict] = [
    {"metadata.due": {"$date": "2026-06-01T00:00:00Z"}},  # $date metadata — deferred leaf
    {"metadata.labels": {"team": "x"}},  # object_equal dict — deferred leaf
    {"metadata": {"tier": "gold"}},  # bare-blob $eq — deferred leaf
    # $date leaf under $or → whole OR defers to "1" (all-or-nothing); SQL = all rows.
    {"$or": [{"source_name": "linear"}, {"metadata.due": {"$date": "2026-06-01T00:00:00Z"}}]},
    # object_equal under $not → whole NOT defers; SQL = all rows.
    {"$not": {"metadata.labels": {"team": "x"}}},
    # AND with a pushable sibling + a deferred $date leaf: SQL pushes source_name,
    # defers the date — SQL ⊇ oracle (the date is enforced by the post-filter).
    {"source_name": "linear", "metadata.due": {"$date": "2026-06-01T00:00:00Z"}},
]


def _compose_ids(sql_ids: set[str], wire: dict) -> set[str]:
    """Apply the compile_python post-filter to the SQL row-set (the engine path).

    Mirrors the backend: the SQL fragment narrows to ``sql_ids``, then
    ``compile_python`` re-checks the full AST against each surviving decoded record
    (the corpus record, with ``occurred_at`` as a ``datetime``). The composition
    SQL ∩ post-filter is what the engine actually returns.
    """
    pred = compile_python(_ast(wire), _CTX).predicate
    return {rid for rid in sql_ids if pred(_CORPUS[rid])}


@pytest.mark.skipif(not _json1_available(), reason="stdlib SQLite build lacks JSON1 functions")
@pytest.mark.parametrize(
    "wire", _SUPERSET_SAFE_WIRES, ids=[json.dumps(w, sort_keys=True) for w in _SUPERSET_SAFE_WIRES]
)
def test_compile_lance_sql_is_superset_safe_for_deferred_leaves(wire: dict) -> None:
    # For a filter with an SQL-unsupported leaf, the compile_lance SQL must NEVER
    # wrongly exclude an oracle-matching row — the post-filter only narrows. So the
    # SQL row-set is a (possibly strict) SUPERSET of the oracle row-set (a), AND the
    # engine composition SQL ∩ compile_python post-filter recovers the oracle row-set
    # EXACTLY (b). (a) guards against over-narrowing; (b) proves the deferred leaf is
    # still enforced — together they pin the split-mode contract end-to-end at the
    # unit level (the integration suite re-checks it over a real store).
    con = sqlite3.connect(":memory:")
    try:
        _materialize(con)
        sql_ids = _lance_ids(con, wire)
        oracle_ids = _oracle_ids(wire)
        # (a) superset-safety — no false-exclude.
        assert oracle_ids <= sql_ids, (
            f"compile_lance SQL FALSE-EXCLUDED an oracle-matching row for {wire!r}: "
            f"sql={sorted(sql_ids)} oracle={sorted(oracle_ids)} missing={sorted(oracle_ids - sql_ids)}"
        )
        # (b) composition — SQL pushdown ∩ python post-filter == the oracle.
        composed = _compose_ids(sql_ids, wire)
        assert composed == oracle_ids, (
            f"engine composition (SQL ∩ post-filter) diverged from the oracle for {wire!r}: "
            f"composed={sorted(composed)} oracle={sorted(oracle_ids)}"
        )
    finally:
        con.close()
