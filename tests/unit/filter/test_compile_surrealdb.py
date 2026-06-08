"""Unit tests for the SurrealDB SurrealQL recall-filter compiler (Layer 4) — ``@internal``.

``compile_surrealdb(ast, ctx)`` lowers a canonical :class:`~khora.filter.ast.FilterNode`
into a SurrealQL ``WHERE``-fragment *string* (plus an out-of-band ``params`` bind dict)
over the recall ``temporal_chunk`` table. These tests pin the §4 *Field-match contract*
at the SurrealQL level: each test builds an AST (by validating a
:class:`~khora.filter.RecallFilter` and lowering it with
:func:`~khora.filter.ast.parse_to_ast`), compiles it, and asserts on the emitted
predicate string and bind dict — never re-implementing the compiler, only inspecting
what it emits.

Like the Cypher compiler (and unlike Postgres, which inlines literals into a SQLAlchemy
expression), the SurrealDB compiler returns a plain string with ``$name`` placeholders
and carries the values out-of-band in ``params``. So the assertions match against the
string directly and check ``params`` for the bound values.

The contract these tests lock (§4), and the one deliberate divergence from Cypher:

* Every operator ``$eq``/``$ne``/``$gt``/``$gte``/``$lt``/``$lte``/``$in``/``$nin``/``$exists``.
* System key (typed column) emission; the empty AST → match-everything (``true``).
* A *nested* metadata path descends natively (``metadata.a.b.c`` → ``metadata_.a.b.c``),
  never collapsed/mangled into a single token.
* The four field-match rules, as SurrealQL expresses them via its NONE-boolean algebra
  (**no coalesce / totality wrapper** — SurrealQL comparisons against an absent path are
  already total, so a negation flips an absent row in correctly):
  1. never-abort: a metadata range op is type-gated (``type::is::<t>(node) AND ...``) so a
     wrong-typed / absent value never participates in the compare;
  2. polarity: ``$ne``/``$nin``/``$not`` admit absent rows (``!=`` and ``!(... INSIDE ...)``
     are already true for an absent path);
  3. a bare-list (exact-array) operand binds as a list against a scalar column — false at
     query time, no special-cased constant;
  4. presence/null: ``$exists`` → ``IS [NOT] NONE``; ``{k: null}`` → ``(= NULL OR IS NONE)``
     (NONE = absent path, NULL = explicit json null — distinct).
* Dates: a system datetime column binds the real :class:`~datetime.datetime` (the SDK
  encodes it as a SurrealQL datetime); a metadata ``$date`` operand is gated as a string
  and binds its ``.isoformat()`` form (lexicographic compare on the FLEXIBLE object column).
* The :class:`CompiledFilter` envelope: ``params`` binds, ``consumed_keys`` frozenset,
  ``canonical_hash``.
* ``field_mapping`` / ``param_namespace`` honored — one compiler, many schemas, no
  per-engine branching.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khora.filter import RecallFilter
from khora.filter.ast import FilterClause, FilterNode, canonical_hash, parse_to_ast
from khora.filter.compilers.surrealdb import compile_surrealdb
from khora.filter.context import CompileContext
from khora.filter.model import Op

# Hard import (NOT importorskip): the compiler is pure-Python string emission with
# no SurrealDB SDK dependency, so an import failure here must be a LOUD test error,
# never a silent module skip that would pass CI green with zero coverage.

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

# The live recall path's context: the ``metadata`` root maps to the physical
# ``metadata_`` column; system keys map identity (bare columns on the table).
_CTX = CompileContext(backend_target="temporal_chunk", field_mapping={"metadata": "metadata_"})


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _norm(surql: str) -> str:
    """Collapse whitespace and lowercase for resilient substring matching."""
    return " ".join(surql.split()).lower()


def _surql_norm(node: FilterNode | FilterClause, ctx: CompileContext = _CTX) -> str:
    """Compile an AST and return the predicate string, whitespace-collapsed.

    The predicate is already a SurrealQL boolean string (the bind values live in
    the sibling ``params`` dict), so there is no compile step — normalize and
    assert substrings directly.
    """
    return _norm(compile_surrealdb(node, ctx).predicate)


# ===========================================================================
# Empty AST → match-everything.
# ===========================================================================


def test_empty_filter_compiles_to_true() -> None:
    # A bare RecallFilter() (no predicates) lowers to the empty match-everything
    # AND. The compiler emits the literal tautology so the engine applies no
    # constraint (the bare "true", NOT a wrapped "(true)").
    assert _surql_norm(_ast({})) == "true"


def test_empty_and_node_compiles_to_true() -> None:
    # Construct the empty AND directly — same match-everything contract.
    assert _surql_norm(FilterNode(op=Op.AND, children=())) == "true"


def test_empty_filter_binds_nothing() -> None:
    assert compile_surrealdb(_ast({}), _CTX).params == {}


# ===========================================================================
# Scalar operators on a SYSTEM (typed column) key.
# ===========================================================================


def test_system_eq_is_bare_property_equals() -> None:
    compiled = compile_surrealdb(_ast({"source_name": "linear"}), _CTX)
    sql = _norm(compiled.predicate)
    # $eq is a total boolean in SurrealQL: a plain ``=`` compare (no coalesce — an
    # absent column compares false, and a wrapping $not flips it). The operand is
    # bound, not inlined. The system key is a bare physical column, not metadata.
    assert "(source_name = $f_0)" in sql
    assert "coalesce" not in sql
    assert "metadata" not in sql
    assert compiled.params == {"f_0": "linear"}


def test_system_eq_with_ops_form() -> None:
    compiled = compile_surrealdb(_ast({"source_type": {"$eq": "slack"}}), _CTX)
    assert "(source_type = $f_0)" in _norm(compiled.predicate)
    assert compiled.params == {"f_0": "slack"}


def test_system_ne_is_bare_not_equals() -> None:
    # Rule 2 (polarity): ``!=`` against an absent column already returns true in
    # SurrealQL, so $ne admits absent rows without an extra ``OR IS NONE`` — no
    # null guard needed (the divergence from Cypher's ``IS NULL OR ...``).
    compiled = compile_surrealdb(_ast({"source_name": {"$ne": "linear"}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(source_name != $f_0)" in sql
    assert "is none" not in sql
    assert compiled.params == {"f_0": "linear"}


def test_system_gt_is_property_compare() -> None:
    # A system DATE key takes a plain ISO-8601 string operand (DateOps parses it to
    # a datetime); the operand binds as a real datetime (the SDK encodes it as a
    # SurrealQL datetime — the system-column divergence from the metadata path).
    compiled = compile_surrealdb(_ast({"created_at": {"$gt": "2026-01-01T00:00:00Z"}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(created_at > $f_0)" in sql
    assert isinstance(compiled.params["f_0"], datetime)


@pytest.mark.parametrize(
    ("wire_op", "surql_op"),
    [
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ],
)
def test_system_date_range_operators(wire_op: str, surql_op: str) -> None:
    # System datetime columns are typed, so their range ops are UNGATED (no
    # ``type::is::*`` prefix — the gate is metadata-only).
    sql = _surql_norm(_ast({"occurred_at": {wire_op: "2026-03-04T05:06:07Z"}}))
    assert f"(occurred_at {surql_op} $f_0)" in sql
    assert "type::is::" not in sql


# ===========================================================================
# All ten system keys lower to a bare field ref (incl. the 8 always-absent ones).
# ===========================================================================
#
# Eight of the ten system keys (source_timestamp / source_type / source_name /
# source_url / external_id / content_type / source / title) are not written onto
# the temporal_chunk table, so they read NONE at query time — but the compiler
# still emits a bare field ref for each (an undefined-field ref does not error in
# SurrealQL). The two date keys (occurred_at / created_at) are real columns. This
# pins that EVERY system key pushes down to a ``<key> = $bind`` compare and lands
# in consumed_keys, independent of whether the column is physically present.
_ALWAYS_ABSENT_SYSTEM_KEYS = (
    "source_type",
    "source_name",
    "source_url",
    "external_id",
    "content_type",
    "source",
    "title",
)


@pytest.mark.parametrize("key", _ALWAYS_ABSENT_SYSTEM_KEYS)
def test_always_absent_string_key_emits_bare_field_ref(key: str) -> None:
    compiled = compile_surrealdb(_ast({key: "v"}), _CTX)
    sql = _norm(compiled.predicate)
    assert f"({key} = $f_0)" in sql
    assert compiled.params == {"f_0": "v"}
    assert key in compiled.consumed_keys
    # A system key is a bare column ref, never a serialized-metadata lookup.
    assert "metadata" not in sql


def test_source_timestamp_emits_bare_field_ref() -> None:
    # source_timestamp is the one always-absent datetime-typed system key. It
    # pushes down as a bare ``=`` compare and binds a real datetime (system-column
    # binding), completing the 8-key always-absent set with the string keys above.
    compiled = compile_surrealdb(_ast({"source_timestamp": "2026-02-03T00:00:00Z"}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(source_timestamp = $f_0)" in sql
    assert isinstance(compiled.params["f_0"], datetime)
    assert "source_timestamp" in compiled.consumed_keys


# ===========================================================================
# $in / $nin on a SYSTEM key → INSIDE / !(INSIDE) membership.
# ===========================================================================


def test_system_in_is_inside_list() -> None:
    compiled = compile_surrealdb(_ast({"source_name": {"$in": ["a", "b", "c"]}}), _CTX)
    assert "(source_name inside $f_0)" in _norm(compiled.predicate)
    assert compiled.params == {"f_0": ["a", "b", "c"]}


def test_system_nin_is_negated_inside() -> None:
    # Rule 2 (polarity): $nin is the negated INSIDE. An absent column is not INSIDE
    # the set, so the negation is true — absent rows are admitted, total without an
    # extra null guard (the divergence from Cypher's ``IS NULL OR NOT ... IN``).
    compiled = compile_surrealdb(_ast({"source_type": {"$nin": ["spam", "junk"]}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "!(source_type inside $f_0)" in sql
    assert "is none" not in sql
    assert compiled.params == {"f_0": ["spam", "junk"]}


def test_system_empty_in_is_constant_false() -> None:
    # An empty $in operand list is a valid filter with a defined row-set: positive
    # membership over ∅ matches nothing. The compiler emits the constant explicitly
    # (the single-clause filter normalizes to AND([leaf]), rendering as the wrapped
    # "(false)") and binds nothing.
    compiled = compile_surrealdb(_ast({"source_name": {"$in": []}}), _CTX)
    assert _norm(compiled.predicate) == "(false)"
    assert compiled.params == {}


def test_system_empty_nin_is_constant_true() -> None:
    # An empty $nin operand list matches everything (negation over ∅) — the polarity
    # mirror of the empty-$in case. Emitted as the wrapped constant "(true)".
    compiled = compile_surrealdb(_ast({"source_name": {"$nin": []}}), _CTX)
    assert _norm(compiled.predicate) == "(true)"
    assert compiled.params == {}


# ===========================================================================
# Rule 3 — bare-list (exact-array) operand vs scalar SYSTEM column.
# ===========================================================================


def test_system_eq_array_operand_binds_as_list() -> None:
    # A bare list on a scalar system key lowers to $eq EXACT-ARRAY. The compiler
    # binds the operand as a SurrealQL list and emits the same plain ``=`` compare;
    # a scalar column never equals a list, so it yields false at query time — no
    # special-cased constant, the list binds like any other operand. (The AST
    # carries a tuple; it is bound as a driver list, not a tuple.)
    compiled = compile_surrealdb(_ast({"source_name": ["a", "b"]}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(source_name = $f_0)" in sql
    assert "inside" not in sql  # exact-array $eq, NOT membership
    assert compiled.params == {"f_0": ["a", "b"]}


def test_system_in_is_membership_not_exact_array() -> None:
    # $in on a system key is membership (INSIDE), NOT exact-array — distinct from
    # the bare-list $eq case above.
    compiled = compile_surrealdb(_ast({"occurred_at": {"$in": ["2026-01-01T00:00:00Z"]}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(occurred_at inside $f_0)" in sql
    assert "!=" not in sql


# ===========================================================================
# Rule 4 — $exists / null resolve to IS [NOT] NONE / (= NULL OR IS NONE).
# ===========================================================================


def test_system_exists_true_is_is_not_none() -> None:
    # An unwritten column reads NONE, so $exists is a presence test (IS NOT NONE),
    # binding nothing.
    compiled = compile_surrealdb(_ast({"source_name": {"$exists": True}}), _CTX)
    assert "(source_name is not none)" in _norm(compiled.predicate)
    assert compiled.params == {}


def test_system_exists_false_is_is_none() -> None:
    compiled = compile_surrealdb(_ast({"source_name": {"$exists": False}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(source_name is none)" in sql
    assert "is not none" not in sql
    assert compiled.params == {}


def test_system_null_operand_is_null_or_none() -> None:
    # An explicit null is an active null-or-missing match. NONE (absent path) and
    # NULL (explicit json null) are distinct in SurrealQL, so the match covers both.
    compiled = compile_surrealdb(_ast({"source_name": None}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(source_name = null or source_name is none)" in sql
    assert compiled.params == {}


def test_system_ne_null_is_present_and_not_null() -> None:
    # $ne null → present (IS NOT NONE) and not explicit-null (!= NULL).
    compiled = compile_surrealdb(_ast({"source_name": {"$ne": None}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(source_name is not none and source_name != null)" in sql
    assert compiled.params == {}


# ===========================================================================
# Dates — system column binds a datetime; metadata gates+binds an ISO string.
# ===========================================================================


def test_system_date_binds_real_datetime() -> None:
    # A system date key takes a plain ISO string (the `$date` typed literal is a
    # metadata-only form). It compiles to a plain ``=`` compare; the operand binds
    # as a real, UTC-normalized datetime (the SDK encodes it as a SurrealQL
    # datetime — NOT an .isoformat() string, unlike Cypher and unlike the metadata
    # path).
    compiled = compile_surrealdb(_ast({"source_timestamp": "2026-02-03T00:00:00Z"}), _CTX)
    assert "(source_timestamp = $f_0)" in _norm(compiled.predicate)
    bound = compiled.params["f_0"]
    assert isinstance(bound, datetime)
    assert bound == datetime(2026, 2, 3, tzinfo=UTC)


def test_direct_datetime_clause_binds_real_datetime() -> None:
    # Build a leaf clause directly with a tz-aware datetime operand — a valid
    # compiler input independent of the validator path. A system datetime column
    # binds the datetime object verbatim.
    clause = FilterClause(path=("occurred_at",), op=Op.GTE, operand=datetime(2026, 1, 1, tzinfo=UTC))
    node = FilterNode(op=Op.AND, children=(clause,))
    compiled = compile_surrealdb(node, _CTX)
    assert "(occurred_at >= $f_0)" in _norm(compiled.predicate)
    assert compiled.params["f_0"] == datetime(2026, 1, 1, tzinfo=UTC)


# ===========================================================================
# Nested metadata paths — native dot-descent (NOT collapsed to one token).
# ===========================================================================


def test_metadata_nested_path_descends_natively() -> None:
    # metadata.a.b.c → metadata_.a.b.c: the path DESCENDS through the remapped
    # object root, never collapsed or mangled into a single token (the old
    # single-token sanitizer would have treated "a.b.c" as one key). Assert the
    # full descended substring is present.
    compiled = compile_surrealdb(_ast({"metadata.a.b.c": "v"}), _CTX)
    sql = _norm(compiled.predicate)
    assert "metadata_.a.b.c" in sql
    assert "(metadata_.a.b.c = $f_0)" in sql
    assert compiled.params == {"f_0": "v"}
    assert "metadata.a.b.c" in compiled.consumed_keys


def test_metadata_deep_path_not_mangled_into_single_token() -> None:
    # A path with several segments the old single-token sanitizer would have
    # treated as one key. Each segment must appear as its own dotted descent
    # component — assert the descent reaches the leaf and the parent prefixes are
    # all present as substrings (no underscore-mangled single token).
    compiled = compile_surrealdb(_ast({"metadata.labels.region.code": {"$exists": True}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "metadata_.labels.region.code" in sql
    # The collapsed/mangled forms the bad sanitizer would have produced must NOT
    # appear.
    assert "metadata_labels" not in sql
    assert "labels_region" not in sql


def test_metadata_eq_single_segment_descends() -> None:
    compiled = compile_surrealdb(_ast({"metadata.tier": "gold"}), _CTX)
    assert "(metadata_.tier = $f_0)" in _norm(compiled.predicate)
    assert compiled.params == {"f_0": "gold"}


def test_bare_metadata_blob_eq() -> None:
    # The bare ``metadata`` blob is whole-object $eq (structural, key-order
    # insensitive). It binds against the remapped ``metadata_`` root directly.
    compiled = compile_surrealdb(_ast({"metadata": {"a": 1}}), _CTX)
    assert "(metadata_ = $f_0)" in _norm(compiled.predicate)
    assert compiled.params == {"f_0": {"a": 1}}


# ===========================================================================
# Metadata range type-gate — every range leaf emits the type::is::<t> prefix.
# ===========================================================================


@pytest.mark.parametrize(
    ("wire_op", "surql_op"),
    [
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ],
)
def test_metadata_numeric_range_is_number_gated(wire_op: str, surql_op: str) -> None:
    # Rule 1 (never-abort): a numeric metadata range op gates on type::is::number;
    # the AND short-circuits so a wrong-typed / absent value never reaches the
    # compare. The whole leaf is the gated pair — no coalesce wrapper.
    compiled = compile_surrealdb(_ast({"metadata.score": {wire_op: 5}}), _CTX)
    sql = _norm(compiled.predicate)
    assert f"(type::is::number(metadata_.score) and metadata_.score {surql_op} $f_0)" in sql
    assert compiled.params == {"f_0": 5}


def test_metadata_bool_range_picks_bool_gate() -> None:
    # A bool operand picks type::is::bool, NOT number — a bool is an int subclass
    # in Python, and SurrealDB agrees a bool is not a number, so the gate must be
    # ``bool`` (checked before ``number``).
    compiled = compile_surrealdb(_ast({"metadata.flag": {"$gt": True}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "type::is::bool(metadata_.flag)" in sql
    assert "type::is::number" not in sql
    assert compiled.params == {"f_0": True}


def test_metadata_string_range_is_string_gated() -> None:
    # A string operand gates on type::is::string (lexicographic text compare).
    compiled = compile_surrealdb(_ast({"metadata.name": {"$gte": "m"}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(type::is::string(metadata_.name) and metadata_.name >= $f_0)" in sql
    assert compiled.params == {"f_0": "m"}


def test_metadata_date_range_is_string_gated_and_binds_isoformat() -> None:
    # A metadata $date / datetime operand is gated as a STRING (metadata datetimes
    # round-trip through the FLEXIBLE object column as ISO strings) and binds its
    # .isoformat() form — the documented divergence from the system-column path
    # (which binds a real datetime).
    compiled = compile_surrealdb(_ast({"metadata.ts": {"$gt": {"$date": "2026-01-01T00:00:00Z"}}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "type::is::string(metadata_.ts)" in sql
    assert "metadata_.ts > $f_0" in sql
    bound = compiled.params["f_0"]
    assert isinstance(bound, str)
    assert bound == datetime(2026, 1, 1, tzinfo=UTC).isoformat()


def test_metadata_direct_datetime_range_is_string_gated() -> None:
    # A raw datetime operand (built directly on the AST) on a metadata range is
    # also string-gated and bound as its .isoformat() string.
    clause = FilterClause(path=("metadata", "ts"), op=Op.LT, operand=datetime(2026, 5, 6, tzinfo=UTC))
    node = FilterNode(op=Op.AND, children=(clause,))
    compiled = compile_surrealdb(node, _CTX)
    sql = _norm(compiled.predicate)
    assert "type::is::string(metadata_.ts)" in sql
    assert compiled.params["f_0"] == datetime(2026, 5, 6, tzinfo=UTC).isoformat()


# ===========================================================================
# Metadata field-match rules — polarity / presence / null on a metadata leaf.
# ===========================================================================


def test_metadata_ne_is_bare_not_equals() -> None:
    # Rule 2 (polarity): a metadata $ne is a plain ``!=`` — already true for an
    # absent / wrong-type path, so absent rows are admitted without a null guard.
    compiled = compile_surrealdb(_ast({"metadata.tier": {"$ne": "gold"}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(metadata_.tier != $f_0)" in sql
    assert "is none" not in sql
    assert compiled.params == {"f_0": "gold"}


def test_metadata_nin_is_negated_inside() -> None:
    # Rule 2: metadata $nin is the negated INSIDE — absent paths are not INSIDE,
    # so the negation is true (admits absent rows).
    compiled = compile_surrealdb(_ast({"metadata.tier": {"$nin": ["a", "b"]}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "!(metadata_.tier inside $f_0)" in sql
    assert compiled.params == {"f_0": ["a", "b"]}


def test_metadata_exists_true_is_is_not_none() -> None:
    # Rule 4: presence on a metadata path → IS NOT NONE.
    compiled = compile_surrealdb(_ast({"metadata.tier": {"$exists": True}}), _CTX)
    assert "(metadata_.tier is not none)" in _norm(compiled.predicate)
    assert compiled.params == {}


def test_metadata_exists_false_is_is_none() -> None:
    compiled = compile_surrealdb(_ast({"metadata.tier": {"$exists": False}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(metadata_.tier is none)" in sql
    assert "is not none" not in sql


def test_metadata_null_operand_is_null_or_none() -> None:
    # Rule 4: {k: null} → explicit json NULL OR absent (NONE) path.
    compiled = compile_surrealdb(_ast({"metadata.tier": None}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(metadata_.tier = null or metadata_.tier is none)" in sql
    assert compiled.params == {}


def test_metadata_ne_null_is_present_and_not_null() -> None:
    compiled = compile_surrealdb(_ast({"metadata.tier": {"$ne": None}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(metadata_.tier is not none and metadata_.tier != null)" in sql
    assert compiled.params == {}


def test_metadata_eq_array_operand_binds_as_list() -> None:
    # Rule 3: a bare list on a metadata path is $eq EXACT-ARRAY, not membership.
    compiled = compile_surrealdb(_ast({"metadata.tags": ["a", "b"]}), _CTX)
    sql = _norm(compiled.predicate)
    assert "(metadata_.tags = $f_0)" in sql
    assert "inside" not in sql
    assert compiled.params == {"f_0": ["a", "b"]}


def test_metadata_empty_in_is_constant_false() -> None:
    compiled = compile_surrealdb(_ast({"metadata.tier": {"$in": []}}), _CTX)
    assert _norm(compiled.predicate) == "(false)"
    assert compiled.params == {}


def test_metadata_empty_nin_is_constant_true() -> None:
    compiled = compile_surrealdb(_ast({"metadata.tier": {"$nin": []}}), _CTX)
    assert _norm(compiled.predicate) == "(true)"
    assert compiled.params == {}


# ===========================================================================
# Logical composition — AND / OR / NOT + totality.
# ===========================================================================


def test_and_node_joins_with_and() -> None:
    # Two sibling keys compose with " AND ". Child order is normalized (the system
    # keys lower in a fixed order), not authored order — so do NOT assert order,
    # only that both leaves and the AND join are present.
    sql = _surql_norm(_ast({"source_name": "linear", "source_type": "slack"}))
    assert " and " in sql
    assert "source_name = $" in sql
    assert "source_type = $" in sql


def test_or_node_joins_with_or() -> None:
    sql = _surql_norm(_ast({"$or": [{"source_name": "linear"}, {"source_type": "slack"}]}))
    assert " or " in sql
    assert "source_name = $" in sql
    assert "source_type = $" in sql


def test_not_node_negates_with_bang_and_plain_child() -> None:
    # SEMANTICS over tokens: NOT compiles via ``!(<child>)``. SurrealQL's
    # NONE-boolean algebra makes every leaf total on its own, so — unlike Cypher —
    # the inner leaf is the PLAIN (un-coalesced) ``=`` compare, and the ``!(...)``
    # flips an absent/null row IN correctly (Rule 2) without any coalesce wrapper.
    # We assert the bang-negation SHAPE and that the inner leaf carries NO coalesce.
    sql = _surql_norm(_ast({"$not": {"source_name": "linear"}}))
    assert sql.startswith("!(")
    assert "(source_name = $f_0)" in sql
    assert "coalesce" not in sql


def test_not_over_metadata_range_keeps_plain_gated_leaf() -> None:
    # $not over a metadata range: the negation wraps the plain type-gated leaf
    # (no coalesce). The gated pair is the inner total expression that ``!(...)``
    # flips — so an absent / wrong-typed / non-matching row is kept by the negation.
    sql = _surql_norm(_ast({"$not": {"metadata.score": {"$gt": 5}}}))
    assert sql.startswith("!(")
    assert "type::is::number(metadata_.score) and metadata_.score > $f_0" in sql
    assert "coalesce" not in sql


def test_nested_and_or_structure() -> None:
    sql = _surql_norm(
        _ast(
            {
                "source_name": "linear",
                "$or": [{"source_type": "slack"}, {"content_type": "text/plain"}],
            }
        )
    )
    assert " and " in sql
    assert " or " in sql


def test_composition_binds_are_distinct_per_clause() -> None:
    # Each leaf allocates a fresh monotonic bind name, so two clauses on different
    # keys never collide — three keys → f_0, f_1, f_2.
    compiled = compile_surrealdb(
        _ast(
            {
                "source_name": "linear",
                "$or": [{"source_type": "slack"}, {"content_type": "text/plain"}],
            }
        ),
        _CTX,
    )
    assert set(compiled.params) == {"f_0", "f_1", "f_2"}
    assert set(compiled.params.values()) == {"linear", "slack", "text/plain"}


# ===========================================================================
# CompiledFilter envelope — consumed_keys / canonical_hash / params.
# ===========================================================================


def test_compiled_filter_carries_canonical_hash() -> None:
    node = _ast({"source_name": "linear"})
    compiled = compile_surrealdb(node, _CTX)
    assert compiled.canonical_hash == canonical_hash(node)


def test_compiled_filter_consumed_keys_is_frozenset() -> None:
    compiled = compile_surrealdb(_ast({"source_name": "linear"}), _CTX)
    assert isinstance(compiled.consumed_keys, frozenset)
    assert "source_name" in compiled.consumed_keys


def test_consumed_keys_collects_every_leaf() -> None:
    # A mixed system + nested-metadata filter consumes the dotted metadata key
    # (its full dot-path string) alongside the system key.
    compiled = compile_surrealdb(
        _ast({"source_name": "linear", "metadata.labels.tier": "gold"}),
        _CTX,
    )
    assert compiled.consumed_keys == frozenset({"source_name", "metadata.labels.tier"})


def test_empty_filter_consumes_nothing() -> None:
    compiled = compile_surrealdb(_ast({}), _CTX)
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# field_mapping / param_namespace — one compiler, many schemas.
# ===========================================================================


def test_field_mapping_remaps_metadata_root() -> None:
    # The live recall path remaps the ``metadata`` root to the physical
    # ``metadata_`` column — the same compiler serves the schema without any
    # per-engine branching.
    ctx = CompileContext(backend_target="temporal_chunk", field_mapping={"metadata": "metadata_"})
    assert "(metadata_.tier = $f_0)" in _surql_norm(_ast({"metadata.tier": "gold"}), ctx)


def test_field_mapping_default_metadata_root_unremapped() -> None:
    # With no field_mapping the ``metadata`` root is the identity ``metadata`` — the
    # remap is a context choice, not hardcoded into the compiler.
    ctx = CompileContext(backend_target="temporal_chunk")
    assert "(metadata.tier = $f_0)" in _surql_norm(_ast({"metadata.tier": "gold"}), ctx)


def test_field_mapping_remaps_system_key_name() -> None:
    # A system-key remap renames the bare column ref (e.g. the legacy schema's
    # ``source_system`` column behind the ``source`` logical key).
    ctx = CompileContext(
        backend_target="temporal_chunk",
        field_mapping={"source": "source_system"},
    )
    assert "(source_system = $f_0)" in _surql_norm(_ast({"source": "slack"}), ctx)


def test_param_namespace_prefixes_bind_names() -> None:
    # param_namespace prefixes every bind name so compiled params cannot collide
    # with the engine's own query parameters.
    ctx = CompileContext(backend_target="temporal_chunk", param_namespace="p")
    compiled = compile_surrealdb(_ast({"source_name": "linear"}), ctx)
    assert "$p_0" in compiled.predicate
    assert compiled.params == {"p_0": "linear"}
