"""Unit tests for the in-memory ``compile_python`` recall-filter compiler (Layer 4).

``@internal``. ``compile_python(ast, ctx)`` lowers a canonical
:class:`~khora.filter.ast.FilterNode` into an in-memory ``callable(record) -> bool``
predicate carried on :attr:`CompiledFilter.predicate`. It is the engine-side
post-filter half of the partial-pushdown split: a backend pushes down what it can
(e.g. the Chronicle compiler pushes the indexed date keys), and the engine
evaluates the rest of the §4 *Field-match contract* against each candidate record
in Python.

These tests pin the §4 Field-match contract at the Python level: each test builds
an AST (by validating a :class:`~khora.filter.RecallFilter` and lowering it with
:func:`~khora.filter.ast.parse_to_ast`), compiles it to a predicate, and asserts
on the *boolean result* of calling that predicate against crafted records — never
re-implementing the compiler, only inspecting whether a given record matches.

The contract these tests lock (§4), mirroring ``compile_postgres`` / ``compile_cypher``:

* Type-gate: a positive op ($eq/$gt/$gte/$lt/$lte/$in) EXCLUDES a wrong-typed or
  absent value; ``number`` means ``int``/``float`` and NOT ``bool``.
* Polarity: $ne / $nin INCLUDE a missing key AND a wrong-typed value (Mongo-faithful);
  never drop an absent/null row on a negation.
* Metadata array containment: a scalar operand matches a stored-array element; a
  bare list operand is exact-array equality (NOT membership); $in = contains-any.
* Scalar range ($gt/$gte/$lt/$lte) type-gating: a comparison against a wrong-typed
  stored value is False (the gate yields exclude, never an error).
* ``$date`` parse-or-exclude: a malformed stored date string is excluded under a
  positive ``$date`` compare.
* ``$exists`` and ``{k: null}`` distinguish ABSENT from PRESENT-NULL; a system-key
  ``$exists`` is trivially all/none (system columns are always present), distinct
  from a metadata ``$exists`` which tests key presence in the metadata dict.
* AND / OR / NOT nesting; the empty filter matches everything.

RECORD-SHAPE DEFENSIVENESS (interface point A): the predicate must accept BOTH a
dict-like record and an attribute-style record (chunk/event dataclasses). Each
semantic is asserted against both shapes via :func:`_both`. A key is ABSENT when
neither attribute nor mapping access yields it; PRESENT-NULL when access yields
Python ``None``. ``metadata`` is resolved the same way and defaults to ``{}``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from khora.filter import RecallFilter
from khora.filter.ast import FilterClause, FilterNode, parse_to_ast
from khora.filter.compilers.python import compile_python
from khora.filter.context import CompileContext
from khora.filter.model import Op

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

_CTX = CompileContext(backend_target="chronicle")


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _pred(wire: dict, ctx: CompileContext = _CTX) -> Any:
    """Compile a wire filter and return its in-memory ``callable(record) -> bool``."""
    compiled = compile_python(_ast(wire), ctx)
    return compiled.predicate


def _attr_record(d: dict[str, Any]) -> SimpleNamespace:
    """An attribute-style record (chunk/event dataclass shape) from a dict.

    Mirrors a chunk/event dataclass: system keys are attributes and ``metadata``
    is an attribute holding the dict. Absent keys are simply not set as
    attributes, so attribute access raises ``AttributeError`` (the ABSENT signal).
    """
    return SimpleNamespace(**d)


def _both(wire: dict, record: dict[str, Any]) -> bool:
    """Evaluate the predicate against BOTH a dict and an attr record; assert agreement.

    The §4 contract is record-shape-agnostic (interface point A): a dict-like and
    an attribute-style record describing the same fields MUST yield the same
    match. Asserting agreement in one place lets every semantic test below call
    ``_both`` once and trust both shapes are covered.
    """
    pred = _pred(wire)
    via_dict = pred(dict(record))
    via_attr = pred(_attr_record(record))
    assert via_dict == via_attr, (
        f"dict-record and attr-record disagree for filter={wire!r} record={record!r}: dict={via_dict} attr={via_attr}"
    )
    return via_dict


# ===========================================================================
# Empty filter → match-everything.
# ===========================================================================


def test_empty_filter_matches_everything() -> None:
    # A bare RecallFilter() (no predicates) lowers to the empty match-everything
    # AND; the predicate must return True for any record, including {}.
    pred = _pred({})
    assert pred({}) is True
    assert pred({"source_name": "anything"}) is True
    assert pred(_attr_record({"metadata": {"x": 1}})) is True


def test_empty_and_node_matches_everything() -> None:
    # Construct the empty AND directly — same match-everything contract.
    pred = compile_python(FilterNode(op=Op.AND, children=()), _CTX).predicate
    assert pred({}) is True


# ===========================================================================
# §4 type-gate — positive ops EXCLUDE wrong-type / absent.
# ===========================================================================


def test_system_eq_matches_equal_value() -> None:
    assert _both({"source_name": "linear"}, {"source_name": "linear"}) is True


def test_system_eq_excludes_different_value() -> None:
    assert _both({"source_name": "linear"}, {"source_name": "slack"}) is False


def test_system_eq_excludes_absent_key() -> None:
    # $eq is a positive op: an absent system value cannot equal the operand.
    assert _both({"source_name": "linear"}, {}) is False


def test_metadata_numeric_gt_excludes_string_stored_value() -> None:
    # Number type-gate: $gt 5 against a STRING stored value is a wrong-type compare
    # → excluded (the gate yields exclude, never a TypeError).
    assert _both({"metadata.score": {"$gt": 5}}, {"metadata": {"score": "high"}}) is False


def test_metadata_numeric_gt_matches_number_stored_value() -> None:
    assert _both({"metadata.score": {"$gt": 5}}, {"metadata": {"score": 9}}) is True
    assert _both({"metadata.score": {"$gt": 5}}, {"metadata": {"score": 3}}) is False


def test_metadata_numeric_gate_rejects_bool_as_number() -> None:
    # number type-gate = isinstance int/float AND NOT bool. A stored True must NOT
    # satisfy a numeric range even though `True == 1` in Python.
    assert _both({"metadata.score": {"$gt": 0}}, {"metadata": {"score": True}}) is False


def test_metadata_eq_bool_matches_only_bool() -> None:
    # $eq True matches a stored bool True, and a stored 1 must NOT match True
    # (bool is type-distinct from number on the equality path too).
    assert _both({"metadata.flag": True}, {"metadata": {"flag": True}}) is True
    assert _both({"metadata.flag": True}, {"metadata": {"flag": 1}}) is False


# ===========================================================================
# §4 polarity — $ne / $nin INCLUDE missing & wrong-type.
# ===========================================================================


def test_system_ne_matches_different_value() -> None:
    assert _both({"source_name": {"$ne": "linear"}}, {"source_name": "slack"}) is True


def test_system_ne_excludes_equal_value() -> None:
    assert _both({"source_name": {"$ne": "linear"}}, {"source_name": "linear"}) is False


def test_system_ne_includes_absent_key() -> None:
    # Rule 2 polarity: an absent system value is "not equal" → $ne matches it.
    assert _both({"source_name": {"$ne": "linear"}}, {}) is True


def test_system_ne_includes_present_null() -> None:
    # A present-but-None system value also satisfies $ne <non-null>.
    assert _both({"source_name": {"$ne": "linear"}}, {"source_name": None}) is True


def test_metadata_ne_includes_missing_key() -> None:
    # $ne on a metadata path includes a row missing the key entirely (Mongo-faithful).
    assert _both({"metadata.tier": {"$ne": "gold"}}, {"metadata": {}}) is True


def test_metadata_ne_includes_wrong_type() -> None:
    # A wrong-typed stored value is "not equal" to the operand → $ne includes it.
    assert _both({"metadata.tier": {"$ne": "gold"}}, {"metadata": {"tier": 7}}) is True


def test_metadata_ne_excludes_equal_value() -> None:
    assert _both({"metadata.tier": {"$ne": "gold"}}, {"metadata": {"tier": "gold"}}) is False


def test_system_nin_includes_absent_key() -> None:
    assert _both({"source_name": {"$nin": ["a", "b"]}}, {}) is True


def test_system_nin_excludes_member() -> None:
    assert _both({"source_name": {"$nin": ["a", "b"]}}, {"source_name": "a"}) is False


def test_system_nin_matches_non_member() -> None:
    assert _both({"source_name": {"$nin": ["a", "b"]}}, {"source_name": "c"}) is True


def test_system_empty_in_matches_nothing() -> None:
    # $in over ∅ on a system key matches nothing (positive membership over empty).
    assert _both({"source_name": {"$in": []}}, {"source_name": "a"}) is False


def test_system_empty_nin_matches_everything() -> None:
    # $nin over ∅ matches everything (negation over empty), present or absent.
    assert _both({"source_name": {"$nin": []}}, {"source_name": "a"}) is True
    assert _both({"source_name": {"$nin": []}}, {}) is True


def test_metadata_nin_includes_missing_key() -> None:
    assert _both({"metadata.tag": {"$nin": ["x", "y"]}}, {"metadata": {}}) is True


# ===========================================================================
# §4 metadata array containment.
# ===========================================================================


def test_metadata_scalar_eq_matches_stored_array_element() -> None:
    # A scalar operand matches a stored ARRAY that contains it (array-aware
    # containment, mirroring Postgres `@>`).
    assert _both({"metadata.tags": "urgent"}, {"metadata": {"tags": ["urgent", "p1"]}}) is True


def test_metadata_scalar_eq_matches_stored_scalar() -> None:
    # The same scalar operand also matches a stored scalar equal to it.
    assert _both({"metadata.tags": "urgent"}, {"metadata": {"tags": "urgent"}}) is True


def test_metadata_scalar_eq_excludes_array_without_element() -> None:
    assert _both({"metadata.tags": "urgent"}, {"metadata": {"tags": ["p1", "later"]}}) is False


def test_metadata_bare_list_is_exact_array_equality() -> None:
    # A bare list operand is EXACT-ARRAY equality (NOT membership): it matches only
    # a stored array equal element-for-element, in order.
    assert _both({"metadata.tags": ["a", "b"]}, {"metadata": {"tags": ["a", "b"]}}) is True


def test_metadata_bare_list_excludes_superset_array() -> None:
    # Exact-array, so a stored superset does NOT match (distinguishes it from $in).
    assert _both({"metadata.tags": ["a", "b"]}, {"metadata": {"tags": ["a", "b", "c"]}}) is False


def test_metadata_bare_list_excludes_reordered_array() -> None:
    # Order is significant for exact-array equality.
    assert _both({"metadata.tags": ["a", "b"]}, {"metadata": {"tags": ["b", "a"]}}) is False


def test_metadata_bare_list_excludes_scalar_store() -> None:
    # An exact-array operand never equals a stored scalar.
    assert _both({"metadata.tags": ["a", "b"]}, {"metadata": {"tags": "a"}}) is False


def test_metadata_in_is_contains_any() -> None:
    # $in is contains-any: matches if the stored value (scalar) is one of the set,
    # OR if the stored array shares ANY element with the set.
    assert _both({"metadata.tag": {"$in": ["x", "y"]}}, {"metadata": {"tag": "y"}}) is True
    assert _both({"metadata.tag": {"$in": ["x", "y"]}}, {"metadata": {"tag": ["y", "z"]}}) is True
    assert _both({"metadata.tag": {"$in": ["x", "y"]}}, {"metadata": {"tag": "z"}}) is False


def test_metadata_in_excludes_missing_key() -> None:
    # Positive membership over an absent key → excluded.
    assert _both({"metadata.tag": {"$in": ["x", "y"]}}, {"metadata": {}}) is False


def test_metadata_empty_in_matches_nothing() -> None:
    # $in over an empty set is a valid filter (validator accepts it): positive
    # membership over ∅ matches nothing — even a present value (mirrors the
    # cypher/postgres compilers' empty-$in => constant-false).
    assert _both({"metadata.tag": {"$in": []}}, {"metadata": {"tag": "x"}}) is False


def test_metadata_empty_nin_matches_everything() -> None:
    # $nin over ∅ is the polarity mirror — matches every row, including a present
    # value (negation over ∅), so it never excludes.
    assert _both({"metadata.tag": {"$nin": []}}, {"metadata": {"tag": "x"}}) is True
    # ...and a missing key too (negation includes absent).
    assert _both({"metadata.tag": {"$nin": []}}, {"metadata": {}}) is True


# ===========================================================================
# §4 scalar range ($gt/$gte/$lt/$lte) type-gating.
# ===========================================================================


@pytest.mark.parametrize(
    ("op", "stored", "expected"),
    [
        ("$gt", 6, True),
        ("$gt", 5, False),
        ("$gte", 5, True),
        ("$gte", 4, False),
        ("$lt", 4, True),
        ("$lt", 5, False),
        ("$lte", 5, True),
        ("$lte", 6, False),
    ],
)
def test_metadata_numeric_range_boundaries(op: str, stored: int, expected: bool) -> None:
    assert _both({"metadata.score": {op: 5}}, {"metadata": {"score": stored}}) is expected


def test_metadata_range_excludes_absent_key() -> None:
    # A range op over an absent key is a positive op → excluded.
    assert _both({"metadata.score": {"$gte": 5}}, {"metadata": {}}) is False


def test_metadata_range_excludes_wrong_type() -> None:
    # A numeric range against a non-numeric stored value is type-gated out.
    assert _both({"metadata.score": {"$lt": 100}}, {"metadata": {"score": "n/a"}}) is False


# ===========================================================================
# §4 $date parse-or-exclude.
# ===========================================================================


def test_metadata_date_compare_matches_valid_iso() -> None:
    # A $date literal compared against a valid ISO-8601 stored string.
    wire = {"metadata.due": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}
    assert _both(wire, {"metadata": {"due": "2026-06-01T00:00:00Z"}}) is True
    assert _both(wire, {"metadata": {"due": "2025-12-01T00:00:00Z"}}) is False


def test_metadata_date_compare_excludes_malformed_stored_value() -> None:
    # parse-or-exclude: a malformed stored date string under a positive $date
    # compare is EXCLUDED (it cannot be parsed, so it cannot satisfy the bound).
    wire = {"metadata.due": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}
    assert _both(wire, {"metadata": {"due": "not-a-date"}}) is False


def test_metadata_date_compare_excludes_absent_key() -> None:
    wire = {"metadata.due": {"$lte": {"$date": "2026-01-01T00:00:00Z"}}}
    assert _both(wire, {"metadata": {}}) is False


# ===========================================================================
# §4 $exists and {k: null} — absent vs present-null.
# ===========================================================================


def test_metadata_exists_true_matches_present_key() -> None:
    assert _both({"metadata.tier": {"$exists": True}}, {"metadata": {"tier": "gold"}}) is True


def test_metadata_exists_true_matches_present_null() -> None:
    # PRESENT-NULL is still PRESENT: $exists True matches a key explicitly set to null.
    assert _both({"metadata.tier": {"$exists": True}}, {"metadata": {"tier": None}}) is True


def test_metadata_exists_true_excludes_absent_key() -> None:
    assert _both({"metadata.tier": {"$exists": True}}, {"metadata": {}}) is False


def test_metadata_exists_false_matches_absent_key() -> None:
    assert _both({"metadata.tier": {"$exists": False}}, {"metadata": {}}) is True


def test_metadata_exists_false_excludes_present_key() -> None:
    assert _both({"metadata.tier": {"$exists": False}}, {"metadata": {"tier": "gold"}}) is False


def test_metadata_exists_false_excludes_present_null() -> None:
    # A present-null key is PRESENT, so $exists False excludes it.
    assert _both({"metadata.tier": {"$exists": False}}, {"metadata": {"tier": None}}) is False


def test_metadata_null_match_matches_present_null() -> None:
    # {k: null} is an active null-or-missing match: a present-null value matches.
    assert _both({"metadata.tier": None}, {"metadata": {"tier": None}}) is True


def test_metadata_null_match_matches_absent_key() -> None:
    # ...and an absent key also matches {k: null} (null-OR-missing).
    assert _both({"metadata.tier": None}, {"metadata": {}}) is True


def test_metadata_null_match_excludes_present_value() -> None:
    assert _both({"metadata.tier": None}, {"metadata": {"tier": "gold"}}) is False


# ----- system-key $exists is trivially all/none -----------------------------


def test_system_exists_true_is_trivially_all() -> None:
    # System columns are always present in a row, so $exists True is a tautology —
    # it matches regardless of whether the record carries a value (mirrors
    # compile_postgres returning a constant true).
    assert _both({"source_name": {"$exists": True}}, {"source_name": "linear"}) is True
    assert _both({"source_name": {"$exists": True}}, {}) is True


def test_system_exists_false_is_trivially_none() -> None:
    assert _both({"source_name": {"$exists": False}}, {"source_name": "linear"}) is False
    assert _both({"source_name": {"$exists": False}}, {}) is False


def test_system_null_match_uses_value_null_not_presence() -> None:
    # Distinct from $exists: a system {k: null} match is about the VALUE being
    # null-or-missing (a None-valued or absent system value matches), NOT the
    # trivial column-presence test.
    assert _both({"source_name": None}, {"source_name": None}) is True
    assert _both({"source_name": None}, {}) is True
    assert _both({"source_name": None}, {"source_name": "linear"}) is False


# ===========================================================================
# §4 logical composition — AND / OR / NOT nesting.
# ===========================================================================


def test_and_requires_all_predicates() -> None:
    wire = {"source_name": "linear", "metadata.tier": "gold"}
    assert _both(wire, {"source_name": "linear", "metadata": {"tier": "gold"}}) is True
    # One predicate failing fails the AND.
    assert _both(wire, {"source_name": "linear", "metadata": {"tier": "silver"}}) is False
    assert _both(wire, {"source_name": "slack", "metadata": {"tier": "gold"}}) is False


def test_or_requires_any_predicate() -> None:
    wire = {"$or": [{"source_name": "linear"}, {"source_type": "slack"}]}
    assert _both(wire, {"source_name": "linear", "source_type": "x"}) is True
    assert _both(wire, {"source_name": "x", "source_type": "slack"}) is True
    assert _both(wire, {"source_name": "x", "source_type": "y"}) is False


def test_not_negates_with_null_inclusion() -> None:
    # $not($eq) behaves like $ne: it flips an absent/wrong row IN (Rule 2 totality).
    wire = {"$not": {"source_name": "linear"}}
    assert _both(wire, {"source_name": "linear"}) is False
    assert _both(wire, {"source_name": "slack"}) is True
    assert _both(wire, {}) is True  # absent → NOT(excluded) → included


def test_nested_and_or() -> None:
    wire = {
        "source_name": "linear",
        "$or": [{"source_type": "slack"}, {"content_type": "text/plain"}],
    }
    assert _both(wire, {"source_name": "linear", "source_type": "slack", "content_type": "x"}) is True
    assert _both(wire, {"source_name": "linear", "source_type": "x", "content_type": "x"}) is False
    assert _both(wire, {"source_name": "other", "source_type": "slack", "content_type": "x"}) is False


def test_nor_excludes_all_branches() -> None:
    # $nor desugars to $not($or(...)): matches only when NONE of the branches match.
    wire = {"$nor": [{"source_name": "linear"}, {"source_type": "slack"}]}
    assert _both(wire, {"source_name": "other", "source_type": "other"}) is True
    assert _both(wire, {"source_name": "linear", "source_type": "other"}) is False


# ===========================================================================
# Bare-blob metadata equality.
# ===========================================================================


def test_bare_metadata_blob_equality() -> None:
    # A bare metadata dict is whole-blob $eq equality (key-order-insensitive object
    # equality, mirroring Mongo embedded-doc equality / Postgres JSONB `=`).
    wire = {"metadata": {"a": 1, "b": 2}}
    assert _both(wire, {"metadata": {"b": 2, "a": 1}}) is True
    assert _both(wire, {"metadata": {"a": 1}}) is False
    assert _both(wire, {"metadata": {"a": 1, "b": 2, "c": 3}}) is False


def test_bare_metadata_blob_against_absent_metadata() -> None:
    # An absent / None metadata defaults to {} — only an empty-blob operand matches.
    assert _both({"metadata": {"a": 1}}, {}) is False
    assert _both({"metadata": {}}, {}) is True


# ===========================================================================
# Nested-subdocument equality is EXACT (whole-object ==), NOT containment.
# ===========================================================================
#
# A dict operand on a metadata SUB-PATH (e.g. {"metadata.labels": {"team": "x"}})
# is whole-subdocument equality: the stored object must equal the operand EXACTLY
# (order-insensitive), so a stored SUPERSET does NOT match. This is the ADR
# object_equal / Python "dict ==" rule.
#
# NOTE: compile_postgres currently uses @> (containment) on this case — a separate
# pre-existing bug, NOT this slice's target. So these tests pin compile_python's
# OWN exact-equality behavior and deliberately do NOT assert python == postgres
# here. (Everywhere else compile_python mirrors compile_postgres; this one cell of
# the matrix is the documented exception.)


def test_nested_subdocument_equality_is_exact_match() -> None:
    wire = {"metadata.labels": {"team": "x"}}
    assert _both(wire, {"metadata": {"labels": {"team": "x"}}}) is True


def test_nested_subdocument_equality_rejects_superset() -> None:
    # EXACT, not containment: a stored superset must NOT match (this is the line
    # that separates object-equality from @> containment).
    wire = {"metadata.labels": {"team": "x"}}
    assert _both(wire, {"metadata": {"labels": {"team": "x", "y": 2}}}) is False


def test_nested_subdocument_equality_is_order_insensitive() -> None:
    # Object equality ignores key order (normalized dict ==), so a reordered stored
    # object with the same key-set + values still matches.
    wire = {"metadata.labels": {"a": 1, "b": 2}}
    assert _both(wire, {"metadata": {"labels": {"b": 2, "a": 1}}}) is True


def test_nested_subdocument_equality_rejects_subset_and_value_diff() -> None:
    wire = {"metadata.labels": {"team": "x", "tier": 1}}
    assert _both(wire, {"metadata": {"labels": {"team": "x"}}}) is False  # subset
    assert _both(wire, {"metadata": {"labels": {"team": "y", "tier": 1}}}) is False  # value diff


def test_nested_subdocument_equality_excludes_absent_path() -> None:
    # A positive $eq over an absent sub-path excludes (Rule 1: missing → no match).
    assert _both({"metadata.labels": {"team": "x"}}, {"metadata": {}}) is False


# ===========================================================================
# Direct-clause construction (AST is a valid input independent of the validator).
# ===========================================================================


def test_direct_datetime_clause() -> None:
    # Build a leaf clause directly with a tz-aware datetime operand on a system
    # date key — the AST is a valid compiler input independent of the wire path.
    clause = FilterClause(path=("occurred_at",), op=Op.GTE, operand=datetime(2026, 1, 1, tzinfo=UTC))
    node = FilterNode(op=Op.AND, children=(clause,))
    pred = compile_python(node, _CTX).predicate
    assert pred({"occurred_at": datetime(2026, 6, 1, tzinfo=UTC)}) is True
    assert pred({"occurred_at": datetime(2025, 1, 1, tzinfo=UTC)}) is False


# ===========================================================================
# System date-key tz alignment — naive/aware datetime pairs compare AND order
# without raising (a naive stored value is read as UTC at the compare boundary).
#
# Some backends return tz-NAIVE datetimes for the system date columns (e.g. the
# embedded sqlite store, whose SQLAlchemy ``DateTime`` column is tz-naive). The
# §4 type-gate treats datetime-vs-datetime as comparable regardless of tz, so the
# shared ``_system_eq``/``_ne``/``_in``/``_nin``/``_range`` path must NOT raise
# ``TypeError: can't compare offset-naive and offset-aware datetimes`` — it
# normalizes both sides to UTC (Rule 1: never abort the query). A genuinely
# non-datetime cross-family pair must still stay non-comparable (exclude, no raise).
# ===========================================================================


def _system_clause(op: Op, operand: Any, *, key: str = "occurred_at") -> Any:
    """Compile a single system date-key leaf clause to its in-memory predicate."""
    node = FilterNode(op=Op.AND, children=(FilterClause(path=(key,), op=op, operand=operand),))
    return compile_python(node, _CTX).predicate


@pytest.mark.parametrize("key", ["occurred_at", "created_at", "source_timestamp"])
def test_system_range_naive_stored_vs_aware_operand_orders(key: str) -> None:
    # naive STORED (sqlite shape) vs aware OPERAND: GTE/LTE order correctly with
    # the naive value read as UTC — never a TypeError.
    bound = datetime(2026, 1, 1, tzinfo=UTC)
    gte = _system_clause(Op.GTE, bound, key=key)
    assert gte({key: datetime(2026, 6, 1)}) is True  # naive, after bound
    assert gte({key: datetime(2025, 6, 1)}) is False  # naive, before bound
    lte = _system_clause(Op.LTE, bound, key=key)
    assert lte({key: datetime(2025, 6, 1)}) is True
    assert lte({key: datetime(2026, 6, 1)}) is False


def test_system_range_aware_stored_vs_naive_operand_orders() -> None:
    # Inverse mix: aware STORED vs naive OPERAND also orders without raising.
    naive_bound = datetime(2026, 1, 1)  # read as UTC
    gte = _system_clause(Op.GTE, naive_bound)
    assert gte({"occurred_at": datetime(2026, 6, 1, tzinfo=UTC)}) is True
    assert gte({"occurred_at": datetime(2025, 6, 1, tzinfo=UTC)}) is False


def test_system_range_both_naive_orders() -> None:
    # Both-naive (no aware side at all) still orders — alignment is a no-op here.
    gte = _system_clause(Op.GTE, datetime(2026, 1, 1))
    assert gte({"occurred_at": datetime(2026, 6, 1)}) is True
    assert gte({"occurred_at": datetime(2025, 6, 1)}) is False


def test_system_range_both_aware_orders() -> None:
    # Both-aware is the original happy path — still correct after the fix.
    gte = _system_clause(Op.GTE, datetime(2026, 1, 1, tzinfo=UTC))
    assert gte({"occurred_at": datetime(2026, 6, 1, tzinfo=UTC)}) is True
    assert gte({"occurred_at": datetime(2025, 6, 1, tzinfo=UTC)}) is False


def test_system_eq_naive_stored_matches_same_instant_aware_operand() -> None:
    # Equality path ($eq / $ne): a naive stored value read as UTC equals an aware
    # operand denoting the same instant — including an operand in a non-UTC zone
    # whose instant maps onto the naive-as-UTC value.
    instant = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    eq = _system_clause(Op.EQ, instant)
    assert eq({"occurred_at": datetime(2026, 1, 1, 12, 0)}) is True  # naive == same UTC instant
    assert eq({"occurred_at": datetime(2026, 1, 1, 13, 0)}) is False  # different instant
    # Operand offset +02:00 → same absolute instant as the naive 12:00 UTC stored.
    plus_two = datetime(2026, 1, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    eq_offset = _system_clause(Op.EQ, plus_two)
    assert eq_offset({"occurred_at": datetime(2026, 1, 1, 12, 0)}) is True
    ne_offset = _system_clause(Op.NE, plus_two)
    assert ne_offset({"occurred_at": datetime(2026, 1, 1, 12, 0)}) is False


def test_system_in_naive_stored_matches_same_instant_aware_operand() -> None:
    # Membership path ($in / $nin) aligns each member: a naive stored value matches
    # an aware member denoting the same instant.
    members = [datetime(2026, 1, 1, 12, 0, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)]
    in_pred = _system_clause(Op.IN, members)
    assert in_pred({"occurred_at": datetime(2026, 1, 1, 12, 0)}) is True
    assert in_pred({"occurred_at": datetime(2028, 1, 1)}) is False
    nin_pred = _system_clause(Op.NIN, members)
    assert nin_pred({"occurred_at": datetime(2026, 1, 1, 12, 0)}) is False


@pytest.mark.parametrize("stored", ["2026-01-01T00:00:00Z", 1735689600])
def test_system_range_non_datetime_cross_family_excludes_without_raising(stored: Any) -> None:
    # A datetime OPERAND vs a non-datetime stored value (str / int) is a genuine
    # cross-family pair: the §4 gate excludes it (Rule 1) and never raises — the
    # tz alignment is scoped to datetime-vs-datetime only.
    bound = datetime(2026, 1, 1, tzinfo=UTC)
    assert _system_clause(Op.GTE, bound)({"occurred_at": stored}) is False
    assert _system_clause(Op.EQ, bound)({"occurred_at": stored}) is False
    # Negation polarity ($ne / $nin) INCLUDES a wrong-typed value — also no raise.
    assert _system_clause(Op.NE, bound)({"occurred_at": stored}) is True
    assert _system_clause(Op.NIN, [bound])({"occurred_at": stored}) is True


# ===========================================================================
# CompiledFilter envelope — predicate is callable, canonical_hash present.
# ===========================================================================


def test_compiled_filter_predicate_is_callable() -> None:
    compiled = compile_python(_ast({"source_name": "linear"}), _CTX)
    assert callable(compiled.predicate)


def test_compiled_filter_carries_canonical_hash() -> None:
    from khora.filter.ast import canonical_hash

    node = _ast({"source_name": "linear"})
    compiled = compile_python(node, _CTX)
    assert compiled.canonical_hash == canonical_hash(node)


# ===========================================================================
# unindexed_metadata telemetry — emitted PER METADATA LEAF at COMPILE time.
# ===========================================================================
#
# Contract (confirmed against the binding precedent, tests/unit/filter/
# test_filter_telemetry.py): the in-memory post-filter compiler emits
# ``record_unindexed_metadata(op=...)`` ONCE PER METADATA LEAF while building the
# predicate (the AST walk), mirroring ``compile_postgres`` at postgres.py:187 —
# NOT inside the returned callable, and NOT per candidate record. So the count is
# stable regardless of how many records the predicate is later evaluated against,
# and a system-key-only filter emits nothing. The fixture monkeypatches the same
# three module-level singletons in ``khora.filter.telemetry`` the postgres
# telemetry tests use.


class _RecordingCounter:
    """Captures ``.add(value, attributes=...)`` calls for assertions."""

    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


@pytest.fixture
def recording_counters(monkeypatch: pytest.MonkeyPatch) -> dict[str, _RecordingCounter]:
    """Replace the three filter-telemetry counter singletons with recording fakes.

    The ``_get_*`` helpers return the singleton if already set, so pre-seeding each
    module global makes ``record_unindexed_metadata`` (and any ``.add()`` site)
    land on the fake. Identical shape to test_filter_telemetry.py::recording_counters.
    """
    from khora.filter import telemetry as filter_telemetry

    counters = {
        "unindexed_metadata": _RecordingCounter(),
        "under_filled": _RecordingCounter(),
        "graph_channel_empty": _RecordingCounter(),
    }
    monkeypatch.setattr(filter_telemetry, "_unindexed_metadata_counter", counters["unindexed_metadata"])
    monkeypatch.setattr(filter_telemetry, "_under_filled_counter", counters["under_filled"])
    monkeypatch.setattr(filter_telemetry, "_graph_channel_empty_counter", counters["graph_channel_empty"])
    return counters


def test_unindexed_metadata_fires_once_per_metadata_leaf_at_compile(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    # Compiling (not evaluating) a single metadata predicate emits exactly one
    # observation, with the leaf's op as the bounded attribute.
    compile_python(_ast({"metadata.tier": "gold"}), _CTX)

    adds = recording_counters["unindexed_metadata"].adds
    assert len(adds) == 1
    assert adds[0] == (1, {"op": "$eq"})


def test_unindexed_metadata_counts_each_metadata_leaf(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    # N metadata leaves → N observations; a sibling system key does NOT add.
    wire = {
        "source_name": "linear",  # system key — no emit
        "metadata.tier": "gold",  # $eq metadata leaf
        "metadata.score": {"$gt": 5},  # $gt metadata leaf
    }
    compile_python(_ast(wire), _CTX)

    adds = recording_counters["unindexed_metadata"].adds
    assert len(adds) == 2
    assert sorted(a[1]["op"] for a in adds) == ["$eq", "$gt"]
    assert all(a[0] == 1 for a in adds)


def test_unindexed_metadata_silent_for_system_key_only(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    # A system-key-only filter post-filters typed columns — no JSONB/metadata access.
    compile_python(_ast({"source_name": "linear"}), _CTX)
    assert recording_counters["unindexed_metadata"].adds == []


def test_unindexed_metadata_not_emitted_per_record_evaluation(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    # The load-bearing contract: emission is at COMPILE time, not per evaluation.
    # Build the predicate once, then call it on several records — the count must
    # NOT grow (emission happened during the compile AST walk, not in the callable).
    pred = compile_python(_ast({"metadata.tier": "gold"}), _CTX).predicate
    baseline = len(recording_counters["unindexed_metadata"].adds)
    assert baseline == 1

    for tier in ("gold", "silver", "bronze", "gold", "gold"):
        pred({"metadata": {"tier": tier}})

    assert len(recording_counters["unindexed_metadata"].adds) == baseline, (
        "the counter must not fire per candidate record — emission is compile-time only"
    )


def test_v1_only_counters_stay_quiet_on_compile(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    # The two V1-only counters have no call site on the compile path here.
    compile_python(_ast({"metadata.tier": "gold"}), _CTX)
    assert recording_counters["under_filled"].adds == []
    assert recording_counters["graph_channel_empty"].adds == []
