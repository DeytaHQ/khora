"""Unit tests for the Neo4j Cypher recall-filter compiler (Layer 4) — ``@internal``.

``compile_cypher(ast, ctx)`` lowers a canonical :class:`~khora.filter.ast.FilterNode`
into a Cypher ``WHERE``-fragment *string* (plus a ``params`` bind dict) over a recall
``Chunk`` node. These tests pin the §4 *Field-match contract* at the Cypher level: each
test builds an AST (by validating a :class:`~khora.filter.RecallFilter` and lowering it
with :func:`~khora.filter.ast.parse_to_ast`), compiles it, and asserts on the emitted
predicate string and bind dict — never re-implementing the compiler, only inspecting
what it emits.

Unlike the Postgres compiler (which inlines literals into a SQLAlchemy expression), the
Cypher compiler returns a plain string with ``$name`` placeholders and carries the
values out-of-band in ``params``. So the assertions match against the string directly
(no SQLAlchemy ``compile()`` step) and check ``params`` for the bound values.

The contract these tests lock (§4):

* Every operator ``$eq``/``$ne``/``$gt``/``$gte``/``$lt``/``$lte``/``$in``/``$nin``/``$exists``.
* System key (typed node property) emission; the empty AST → match-everything (``true``).
* ``$and``/``$or``/``$not`` composition.
* The four field-match rules, as Cypher expresses them:
  1. range / ``$eq`` compares are wrapped in ``coalesce(..., false)`` so an absent or
     wrong-typed property yields ``false`` (never aborts);
  2. ``$ne``/``$nin`` emit ``IS NULL OR ...`` so a null/absent row is admitted;
  3. a bare-list (exact-array) operand binds as a list against a scalar property — the
     compare yields ``false`` at query time, no special-cased constant;
  4. ``$exists`` / null resolve to ``IS NOT NULL`` / ``IS NULL`` (Neo4j has no stored
     null — an absent property *is* null).
* Dates bind as ``.isoformat()`` strings (lexicographic compare).
* Metadata predicates are NOT pushed down (Neo4j stores metadata as a serialized JSON
  string property): ``on_unsupported="raise"`` raises; ``"split"`` emits a
  non-constraining ``true`` and consumes nothing.
* The :class:`CompiledFilter` envelope: ``params`` binds, ``consumed_keys`` frozenset,
  ``canonical_hash``.
* ``field_mapping`` / ``table_alias`` / ``param_namespace`` honored — one compiler, many
  schemas, no per-engine branching.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khora.filter import RecallFilter
from khora.filter.ast import FilterClause, FilterNode, parse_to_ast
from khora.filter.compilers.cypher import compile_cypher
from khora.filter.context import CompileContext
from khora.filter.model import Op

# Hard import (NOT importorskip): the compiler is on the branch, so an import
# failure here must be a LOUD test error — never a silent module skip that would
# pass CI green with zero coverage.

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

_CTX = CompileContext(backend_target="Chunk")


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _norm(cypher: str) -> str:
    """Collapse whitespace and lowercase for resilient substring matching."""
    return " ".join(cypher.split()).lower()


def _cypher_norm(node: FilterNode | FilterClause, ctx: CompileContext = _CTX) -> str:
    """Compile an AST and return the predicate string, whitespace-collapsed.

    The predicate is already a Cypher boolean string (the bind values live in the
    sibling ``params`` dict), so there is no compile step — normalize and assert
    substrings directly.
    """
    return _norm(compile_cypher(node, ctx).predicate)


# ===========================================================================
# Empty AST → match-everything.
# ===========================================================================


def test_empty_filter_compiles_to_true() -> None:
    # A bare RecallFilter() (no predicates) lowers to the empty match-everything
    # AND. The compiler must emit a tautology so the engine applies no constraint.
    assert _cypher_norm(_ast({})) == "true"


def test_empty_and_node_compiles_to_true() -> None:
    # Construct the empty AND directly — same match-everything contract.
    assert _cypher_norm(FilterNode(op=Op.AND, children=())) == "true"


def test_empty_filter_binds_nothing() -> None:
    assert compile_cypher(_ast({}), _CTX).params == {}


# ===========================================================================
# Scalar operators on a SYSTEM (typed node property) key.
# ===========================================================================


def test_system_eq_is_coalesced_property_equals() -> None:
    compiled = compile_cypher(_ast({"source_name": "linear"}), _CTX)
    sql = _norm(compiled.predicate)
    # $eq is a total boolean: coalesce(prop = $bind, false) so an absent property
    # excludes (and a wrapping $not flips it in). The operand is bound, not inlined.
    assert "coalesce(c.source_name = $f_0, false)" in sql
    assert compiled.params == {"f_0": "linear"}
    # System key is a typed node property, NOT a metadata extraction.
    assert "metadata" not in sql


def test_system_eq_on_string_key_with_ops_form() -> None:
    compiled = compile_cypher(_ast({"source_type": {"$eq": "slack"}}), _CTX)
    assert "coalesce(c.source_type = $f_0, false)" in _norm(compiled.predicate)
    assert compiled.params == {"f_0": "slack"}


# ===========================================================================
# Pushdown guard — all 8 denormalized document keys lower to Cypher.
# ===========================================================================
#
# The eight document-grained keys are projected onto the recall Chunk
# node so a recall filter on any of them is pushed down to Cypher (not left to a
# post-filter). SYSTEM_KEYS already contains all ten filterable keys, and the
# compiler handles them generically via the system-key path — so this guard pins
# the *contract* (each denorm key pushes down to a `c.<key>` property predicate
# AND lands in consumed_keys), independent of which individual keys other tests
# happen to exercise. It catches a regression that drops one of the eight from
# SYSTEM_KEYS (it would then fall through to `_unsupported` and stop pushing
# down) and locks the source_timestamp datetime → .isoformat() string binding.
_DENORM_STRING_KEYS = (
    "source_type",
    "source_name",
    "source_url",
    "external_id",
    "content_type",
    "source",
    "title",
)


@pytest.mark.parametrize("key", _DENORM_STRING_KEYS)
def test_denorm_string_key_pushes_down(key: str) -> None:
    # Each of the seven string-valued denorm keys compiles to a coalesced `=`
    # property compare on the chunk node, binds the operand, and is reported in
    # consumed_keys (engine applies no post-filter for a pushed-down key).
    compiled = compile_cypher(_ast({key: "v"}), _CTX)
    sql = _norm(compiled.predicate)
    assert f"coalesce(c.{key} = $f_0, false)" in sql
    assert compiled.params == {"f_0": "v"}
    assert key in compiled.consumed_keys
    # A denorm key is a typed node property, never a serialized-metadata lookup.
    assert "metadata" not in sql


def test_denorm_source_timestamp_pushes_down_as_iso_string() -> None:
    # source_timestamp is the one datetime-typed denorm key. It pushes down like
    # the string keys but binds its operand as the UTC-normalized .isoformat()
    # string (lexicographic compare), matching how dual_nodes serializes it at
    # write time. Completes the 8-key set with the string-key guard above.
    compiled = compile_cypher(_ast({"source_timestamp": "2026-02-03T00:00:00Z"}), _CTX)
    sql = _norm(compiled.predicate)
    assert "coalesce(c.source_timestamp = $f_0, false)" in sql
    assert compiled.params["f_0"].startswith("2026-02-03")
    assert "source_timestamp" in compiled.consumed_keys


def test_system_gt_is_property_compare() -> None:
    # A system DATE key takes a plain ISO-8601 string operand (DateOps parses it to
    # a datetime); the operand binds as its .isoformat() string.
    compiled = compile_cypher(_ast({"created_at": {"$gt": "2026-01-01T00:00:00Z"}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "coalesce(c.created_at > $f_0, false)" in sql
    assert compiled.params["f_0"].startswith("2026-01-01")


@pytest.mark.parametrize(
    ("wire_op", "cypher_op"),
    [
        ("$gt", ">"),
        ("$gte", ">="),
        ("$lt", "<"),
        ("$lte", "<="),
    ],
)
def test_system_date_range_operators(wire_op: str, cypher_op: str) -> None:
    sql = _cypher_norm(_ast({"occurred_at": {wire_op: "2026-03-04T05:06:07Z"}}))
    assert f"coalesce(c.occurred_at {cypher_op} $f_0, false)" in sql


# ===========================================================================
# Rule 2 — $ne / $nin include NULL / absent.
# ===========================================================================


def test_system_ne_includes_null_rows() -> None:
    # Rule 2: $ne on a system property must match rows where the property IS NULL
    # too (an absent property is "not equal"). Emitted as the null-inclusive OR.
    compiled = compile_cypher(_ast({"source_name": {"$ne": "linear"}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "c.source_name is null or c.source_name <> $f_0" in sql
    assert compiled.params == {"f_0": "linear"}


def test_system_nin_includes_null_rows() -> None:
    # $nin is the list form of $ne — NULL rows are included via the leading IS NULL.
    compiled = compile_cypher(_ast({"source_name": {"$nin": ["a", "b"]}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "c.source_name is null or not c.source_name in $f_0" in sql
    assert compiled.params == {"f_0": ["a", "b"]}


# ===========================================================================
# Rule 3 — bare-list (exact-array) operand vs scalar SYSTEM property.
# ===========================================================================


def test_system_eq_array_operand_binds_as_list() -> None:
    # A bare list on a scalar system key lowers to $eq EXACT-ARRAY. The compiler
    # binds the operand as a Cypher list and emits the same coalesced `=` compare;
    # a scalar property never equals a list, so it yields false at query time —
    # no special-cased constant, the list binds like any other operand. (The AST
    # carries a tuple; it is bound as a driver list, not a tuple.)
    compiled = compile_cypher(_ast({"source_name": ["a", "b"]}), _CTX)
    assert "coalesce(c.source_name = $f_0, false)" in _norm(compiled.predicate)
    assert compiled.params == {"f_0": ["a", "b"]}


def test_system_in_is_membership_not_exact_array() -> None:
    # $in on a system key is membership (coalesced IN), NOT exact-array — distinct
    # from the bare-list $eq case above.
    compiled = compile_cypher(_ast({"occurred_at": {"$in": ["2026-01-01T00:00:00Z"]}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "coalesce(c.occurred_at in $f_0, false)" in sql
    assert "<>" not in sql


# ===========================================================================
# $in / $nin on a SYSTEM key → IN / NOT IN membership.
# ===========================================================================


def test_system_in_is_coalesced_in_list() -> None:
    compiled = compile_cypher(_ast({"source_name": {"$in": ["a", "b", "c"]}}), _CTX)
    assert "coalesce(c.source_name in $f_0, false)" in _norm(compiled.predicate)
    assert compiled.params == {"f_0": ["a", "b", "c"]}


def test_system_nin_is_not_in_with_null_guard() -> None:
    compiled = compile_cypher(_ast({"source_type": {"$nin": ["spam", "junk"]}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "not c.source_type in $f_0" in sql
    assert "c.source_type is null" in sql
    assert compiled.params == {"f_0": ["spam", "junk"]}


def test_system_empty_in_is_constant_false() -> None:
    # An empty $in operand list is a valid filter (the validator accepts it) with a
    # defined row-set: a positive membership over ∅ matches nothing. The compiler
    # emits the constant explicitly (the single-clause filter normalizes to
    # AND([leaf]), so it renders as the wrapped "(false)") and binds nothing.
    compiled = compile_cypher(_ast({"source_name": {"$in": []}}), _CTX)
    assert _norm(compiled.predicate) == "(false)"
    assert compiled.params == {}


def test_system_empty_nin_is_constant_true() -> None:
    # An empty $nin operand list matches everything (negation over ∅). Emitted as
    # the constant "true" (wrapped "(true)" for the single-clause filter), binding
    # nothing — the polarity mirror of the empty-$in case above.
    compiled = compile_cypher(_ast({"source_name": {"$nin": []}}), _CTX)
    assert _norm(compiled.predicate) == "(true)"
    assert compiled.params == {}


# ===========================================================================
# Rule 4 — $exists on a system key is a CONSTANT (the always-present axiom); a
# null operand resolves to IS [NOT] NULL.
# ===========================================================================


def test_system_exists_true_is_constant_true() -> None:
    # A system key is treated as ALWAYS PRESENT (the oracle's axiom), so $exists:true
    # is a CONSTANT ``true`` — NOT a presence test (``IS NOT NULL``), which would
    # exclude rows where an unwritten denormalized doc key is genuinely null. Matches
    # compile_python / compile_postgres / compile_lance. Binds nothing.
    compiled = compile_cypher(_ast({"source_name": {"$exists": True}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "true" in sql
    assert "is not null" not in sql and "is null" not in sql
    assert compiled.params == {}


def test_system_exists_false_is_constant_false() -> None:
    compiled = compile_cypher(_ast({"source_name": {"$exists": False}}), _CTX)
    sql = _norm(compiled.predicate)
    assert "false" in sql
    assert "is null" not in sql
    assert compiled.params == {}


def test_system_null_operand_is_is_null() -> None:
    # An explicit null is an active null-or-missing match → IS NULL.
    compiled = compile_cypher(_ast({"source_name": None}), _CTX)
    assert "c.source_name is null" in _norm(compiled.predicate)
    assert compiled.params == {}


def test_system_ne_null_is_is_not_null() -> None:
    # $ne null → present (IS NOT NULL).
    compiled = compile_cypher(_ast({"source_name": {"$ne": None}}), _CTX)
    assert "c.source_name is not null" in _norm(compiled.predicate)
    assert compiled.params == {}


# ===========================================================================
# Dates bind as ISO strings.
# ===========================================================================


def test_system_date_binds_iso_string() -> None:
    # A system date key takes a plain ISO string (the `$date` typed literal is a
    # metadata-only form). It compiles to a coalesced `=` compare; the operand
    # binds as its UTC-normalized .isoformat() string.
    compiled = compile_cypher(_ast({"source_timestamp": "2026-02-03T00:00:00Z"}), _CTX)
    assert "coalesce(c.source_timestamp = $f_0, false)" in _norm(compiled.predicate)
    assert compiled.params["f_0"].startswith("2026-02-03")


def test_direct_datetime_clause_binds_iso_string() -> None:
    # Build a leaf clause directly with a tz-aware datetime operand — the AST is a
    # valid compiler input independent of the validator path, and the datetime
    # binds as its .isoformat() string.
    clause = FilterClause(path=("occurred_at",), op=Op.GTE, operand=datetime(2026, 1, 1, tzinfo=UTC))
    node = FilterNode(op=Op.AND, children=(clause,))
    compiled = compile_cypher(node, _CTX)
    assert "coalesce(c.occurred_at >= $f_0, false)" in _norm(compiled.predicate)
    assert compiled.params["f_0"] == datetime(2026, 1, 1, tzinfo=UTC).isoformat()


# ===========================================================================
# Logical composition — AND / OR / NOT.
# ===========================================================================


def test_and_node_joins_with_and() -> None:
    # Two sibling keys compose with " AND ". Child order is normalized (the system
    # keys lower in a fixed order), not authored order — so do NOT assert order,
    # only that both leaves and the AND join are present.
    sql = _cypher_norm(_ast({"source_name": "linear", "source_type": "slack"}))
    assert " and " in sql
    assert "c.source_name = $" in sql
    assert "c.source_type = $" in sql


def test_or_node_joins_with_or() -> None:
    sql = _cypher_norm(_ast({"$or": [{"source_name": "linear"}, {"source_type": "slack"}]}))
    assert " or " in sql
    assert "c.source_name = $" in sql
    assert "c.source_type = $" in sql


def test_not_node_negates_with_total_child() -> None:
    # SEMANTICS over tokens: NOT compiles via `(NOT (<child>))`, and the child
    # equality is built as a TOTAL boolean (`coalesce(prop = $v, false)`) precisely
    # so the negation is NULL-INCLUSIVE — `NOT coalesce(NULL = v, false)` =
    # `NOT false` = TRUE flips a NULL/absent row IN (Rule 2), making $not($eq)
    # behave like $ne rather than dropping the NULL row. We assert the total-child
    # SHAPE here (the `coalesce(..., false)` wrap is the load-bearing guarantee).
    sql = _cypher_norm(_ast({"$not": {"source_name": "linear"}}))
    assert sql.startswith("(not (")
    assert "coalesce(c.source_name = $f_0, false)" in sql


def test_nested_and_or_structure() -> None:
    sql = _cypher_norm(
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
    # keys never collide — three keys → $f_0, $f_1, $f_2.
    compiled = compile_cypher(
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
# Metadata predicates — NOT pushed down (serialized-JSON-string property).
# ===========================================================================


def test_metadata_leaf_raises_in_raise_mode() -> None:
    # Neo4j stores metadata as a serialized JSON string, not a nested map, so a
    # metadata sub-path cannot be expressed in Cypher. In the default "raise" mode
    # the compiler raises the public unsupported error.
    from khora.filter import RecallFilterUnsupportedError

    with pytest.raises(RecallFilterUnsupportedError):
        compile_cypher(_ast({"metadata.tier": "gold"}), _CTX)


def test_metadata_leaf_splits_to_nonconstraining_true() -> None:
    # In "split" mode the engine post-filters what the backend cannot express, so
    # the compiler emits a non-constraining predicate, binds nothing, and leaves
    # the key OUT of consumed_keys. A lone metadata leaf normalizes to AND([leaf]),
    # so the single-clause filter renders as the wrapped "(true)".
    ctx = CompileContext(backend_target="Chunk", on_unsupported="split")
    compiled = compile_cypher(_ast({"metadata.tier": "gold"}), ctx)
    assert _norm(compiled.predicate) == "(true)"
    assert compiled.params == {}
    assert "metadata.tier" not in compiled.consumed_keys


def test_bare_metadata_blob_splits_to_nonconstraining_true() -> None:
    # The bare metadata blob ($eq whole-document) is equally not pushdownable.
    ctx = CompileContext(backend_target="Chunk", on_unsupported="split")
    compiled = compile_cypher(_ast({"metadata": {"a": 1}}), ctx)
    assert _norm(compiled.predicate) == "(true)"
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# on_unsupported policy on a structurally-unknown key (neither system nor metadata).
# ===========================================================================


def _unknown_key_node() -> FilterNode:
    # A leaf whose path is neither a system key nor a metadata path — the only
    # clause this backend cannot express (should not occur post-validation).
    return FilterNode(op=Op.AND, children=(FilterClause(path=("not_a_key",), op=Op.EQ, operand="x"),))


def test_unsupported_clause_raises_in_raise_mode() -> None:
    from khora.filter import RecallFilterUnsupportedError

    ctx = CompileContext(backend_target="Chunk", on_unsupported="raise")
    with pytest.raises(RecallFilterUnsupportedError):
        compile_cypher(_unknown_key_node(), ctx)


def test_unsupported_clause_splits_to_nonconstraining_true() -> None:
    ctx = CompileContext(backend_target="Chunk", on_unsupported="split")
    compiled = compile_cypher(_unknown_key_node(), ctx)
    assert _norm(compiled.predicate) == "(true)"
    assert "not_a_key" not in compiled.consumed_keys


# ===========================================================================
# All-or-nothing OR / NOT deferral (split mode) — the false-exclude guard.
# ===========================================================================
#
# Cypher pushes ONLY system keys; a metadata leaf is unpushable and emits the
# non-constraining "true" under split. That placeholder is superset-safe in
# positive position (A AND true ≡ A), but NOT (A OR true) ≡ NOT true ≡ false would
# FALSE-EXCLUDE every row — and the compile_python post-filter only narrows, so it
# could not recover the wrongly-dropped rows. So when ANY descendant leaf of an
# $or / $not is an unpushable metadata leaf, the WHOLE node defers to "true"
# (consuming nothing). An $and stays independent: the metadata child becomes
# "true" and the pushable system-key sibling still narrows. Mirrors the same guard
# in compile_lance. These four shapes are exactly the cypher conformance leg's
# divergences ($not over a metadata range / $exists).

_CTX_SPLIT = CompileContext(backend_target="Chunk", on_unsupported="split")


def test_not_over_metadata_exists_defers_to_nonconstraining_true() -> None:
    # {metadata.k: {$not: {$exists: true}}} — the F-LOGIC-not-exists divergence.
    # The metadata $exists leaf is unpushable; compiling the NOT naively would emit
    # (NOT (true)) ≡ false and exclude every row. The all-or-nothing guard defers
    # the whole NOT to the non-constraining "true" and consumes nothing — the
    # post-filter then applies the real $not($exists).
    compiled = compile_cypher(_ast({"metadata.k": {"$not": {"$exists": True}}}), _CTX_SPLIT)
    sql = _norm(compiled.predicate)
    assert "true" in sql
    assert "not (" not in sql and "false" not in sql
    assert "metadata.k" not in compiled.consumed_keys


@pytest.mark.parametrize("wire_op", ["$lt", "$gt"])
def test_not_range_over_metadata_defers_to_nonconstraining_true(wire_op: str) -> None:
    # {metadata.num: {$not: {$lt|$gt: N}}} — the F-POLARITY-num-not-lt/gt
    # divergences. A $not-range over an unpushable metadata leaf must defer the
    # whole NOT to "true" (not constrain via (NOT (...)) / false) and consume
    # nothing, so the prefilter stays a superset.
    compiled = compile_cypher(_ast({"metadata.num": {"$not": {wire_op: 5}}}), _CTX_SPLIT)
    sql = _norm(compiled.predicate)
    assert "true" in sql
    assert "not (" not in sql and "false" not in sql
    assert "metadata.num" not in compiled.consumed_keys
    assert compiled.params == {}


def test_or_with_unpushable_metadata_disjunct_defers_whole_node() -> None:
    # $or mixing a pushable system key with an unpushable metadata leaf: the whole
    # OR defers to "true" and consumes nothing (all-or-nothing). Pushing only
    # source_name would make the OR match-all here while a wrapping NOT would
    # false-exclude — the trap the guard closes.
    compiled = compile_cypher(
        _ast({"$or": [{"source_name": "linear"}, {"metadata.tier": "gold"}]}),
        _CTX_SPLIT,
    )
    assert compiled.predicate == "true"
    assert compiled.consumed_keys == frozenset()


def test_and_pushes_system_key_defers_metadata_clause() -> None:
    # $and distribution stays intact: the pushable source_name narrows; the
    # unpushable metadata leaf becomes "true" and is left to the post-filter.
    # consumed_keys carries only the pushed system key.
    compiled = compile_cypher(
        _ast({"source_name": "linear", "metadata.tier": "gold"}),
        _CTX_SPLIT,
    )
    sql = _norm(compiled.predicate)
    assert "coalesce(c.source_name = $f_0, false)" in sql
    assert " and true" in sql  # the deferred metadata leaf
    assert compiled.consumed_keys == frozenset({"source_name"})


def test_and_with_deferred_or_child_still_pushes_system_sibling() -> None:
    # AND distribution holds even when an AND child is itself a deferred OR: the
    # OR-over-(system + metadata) defers to "true", and the sibling system key still
    # narrows. Only the sibling lands in consumed_keys.
    compiled = compile_cypher(
        _ast(
            {
                "source_type": "slack",
                "$or": [{"source_name": "linear"}, {"metadata.tier": "gold"}],
            }
        ),
        _CTX_SPLIT,
    )
    sql = _norm(compiled.predicate)
    assert "coalesce(c.source_type = $f_0, false)" in sql
    assert " and true" in sql  # the deferred OR
    assert compiled.consumed_keys == frozenset({"source_type"})


def test_nested_not_not_over_metadata_defers_whole_node() -> None:
    # NOT(NOT(metadata-leaf)) — a nested negation whose innermost leaf is unpushable
    # metadata. The all-or-nothing guard fires at the outer NOT (the whole subtree
    # is non-consumable), deferring to "true" and consuming nothing — no (NOT (NOT
    # (...))) constraint reaches the prefilter.
    inner = FilterClause(path=("metadata", "k"), op=Op.EQ, operand="v")
    node = FilterNode(
        op=Op.NOT,
        children=(FilterNode(op=Op.NOT, children=(inner,)),),
    )
    compiled = compile_cypher(node, _CTX_SPLIT)
    assert compiled.predicate == "true"
    assert compiled.consumed_keys == frozenset()


def test_not_over_system_key_still_pushes_in_split_mode() -> None:
    # A $not over a fully-consumable system-key leaf is unaffected by the guard: it
    # still compiles to the constraining (NOT (...)) with a total child and consumes
    # the key. Only unpushable subtrees defer.
    compiled = compile_cypher(_ast({"$not": {"source_name": "linear"}}), _CTX_SPLIT)
    sql = _norm(compiled.predicate)
    assert sql.startswith("(not (")
    assert "coalesce(c.source_name = $f_0, false)" in sql
    assert "source_name" in compiled.consumed_keys


# ===========================================================================
# CompiledFilter envelope — consumed_keys / canonical_hash / params.
# ===========================================================================


def test_compiled_filter_carries_canonical_hash() -> None:
    node = _ast({"source_name": "linear"})
    compiled = compile_cypher(node, _CTX)
    from khora.filter.ast import canonical_hash

    assert compiled.canonical_hash == canonical_hash(node)


def test_compiled_filter_consumed_keys_is_frozenset() -> None:
    compiled = compile_cypher(_ast({"source_name": "linear"}), _CTX)
    assert isinstance(compiled.consumed_keys, frozenset)
    assert "source_name" in compiled.consumed_keys


def test_empty_filter_consumes_nothing() -> None:
    compiled = compile_cypher(_ast({}), _CTX)
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# field_mapping / table_alias / param_namespace — one compiler, many schemas.
# ===========================================================================


def test_field_mapping_remaps_system_property_name() -> None:
    # CompileContext.field_mapping lets the same compiler serve a different schema
    # (a property under a different name) without any per-engine `if engine == ...`
    # branching.
    ctx = CompileContext(
        backend_target="Chunk",
        field_mapping={"source_name": "src"},
    )
    assert "coalesce(c.src = $f_0, false)" in _cypher_norm(_ast({"source_name": "linear"}), ctx)


def test_table_alias_qualifies_node_variable() -> None:
    # The node variable defaults to "c"; table_alias overrides it.
    ctx = CompileContext(backend_target="Chunk", table_alias="n")
    assert "coalesce(n.source_name = $f_0, false)" in _cypher_norm(_ast({"source_name": "linear"}), ctx)


def test_table_alias_and_field_mapping_compose() -> None:
    ctx = CompileContext(
        backend_target="Chunk",
        table_alias="ch",
        field_mapping={"source_name": "src"},
    )
    assert "coalesce(ch.src = $f_0, false)" in _cypher_norm(_ast({"source_name": "x"}), ctx)


def test_param_namespace_prefixes_bind_names() -> None:
    # param_namespace prefixes every bind name so compiled params cannot collide
    # with the engine's own query parameters.
    ctx = CompileContext(backend_target="Chunk", param_namespace="p")
    compiled = compile_cypher(_ast({"source_name": "linear"}), ctx)
    assert "$p_0" in compiled.predicate
    assert compiled.params == {"p_0": "linear"}
