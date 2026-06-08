"""Unit tests for the Weaviate recall-filter compiler (Layer 4) — ``@internal``.

``compile_weaviate(ast, ctx)`` lowers a canonical :class:`~khora.filter.ast.FilterNode`
into a Weaviate v4 ``Filter`` combinator tree (or ``None`` for match-all) over a recall
chunk collection. These tests pin the *superset-safe pushdown* contract at the Weaviate
level: each test builds an AST (by validating a :class:`~khora.filter.RecallFilter` and
lowering it with :func:`~khora.filter.ast.parse_to_ast`), compiles it, and asserts on the
emitted ``Filter`` object's structure (``.target`` / ``.operator`` / ``.value`` /
``.filters``) and on ``consumed_keys`` — never re-implementing the compiler.

The contract these tests lock:

* **Operator translations** (declared property only): ``$eq`` → ``equal``;
  ``$gt`` / ``$gte`` / ``$lt`` / ``$lte`` → ``greater_than`` / ``greater_or_equal`` /
  ``less_than`` / ``less_or_equal``; ``$in`` → ``any_of([equal(x), ...])`` (an OR
  of scalar equals — exact membership on a scalar property, NOT ``contains_any``).
  Dates bind as ``.isoformat()`` strings.
* **Superset-safe routing** — the load-bearing correctness rule. Only
  monotone-narrowing predicates on a declared property push down and land in
  ``consumed_keys``; ``$ne`` / ``$nin`` / ``$not``, ``$exists`` (both polarities),
  ``{k: null}``, every metadata path, and every undeclared key are NOT consumed
  (the engine's ``compile_python`` post-filter re-checks them). ``$exists`` / null
  are not pushed because the oracle treats a system key as always-present, so a
  server-side ``is_none`` push would diverge (a ``$exists:true`` push would
  false-exclude null-property rows). An ``$and`` pushes its pushable conjuncts and
  drops the rest; an ``$or`` is all-or-nothing.
* The :class:`CompiledFilter` envelope: ``predicate`` (a ``_Filters`` or ``None``),
  empty ``params``, ``consumed_keys`` frozenset, ``canonical_hash``.

The weaviate v4 ``Filter`` objects are private dataclasses (``_FilterValue`` /
``_FilterAnd`` / ``_FilterOr``); the tests read their stable instance attributes
(``target`` / ``operator`` / ``value`` / ``filters``) and compare the ``operator``
enum by its ``.value`` string so no private symbol is imported.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from khora.filter import RecallFilter, RecallFilterUnsupportedError
from khora.filter.ast import FilterClause, FilterNode, canonical_hash, parse_to_ast
from khora.filter.compilers.weaviate import compile_weaviate
from khora.filter.context import CompileContext
from khora.filter.model import Op

# Hard skip (not a silent module skip): the compiler module imports without the
# weaviate extra, but these tests build real ``Filter`` objects, so they need the
# extra installed. Skip the module if it is absent rather than fail CI red on a
# stack that does not install the optional weaviate-client.
pytest.importorskip("weaviate", reason="weaviate-client extra not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

# The two date keys are the declared+pushable property set for these tests (the
# keys of field_mapping ARE the declared set, identity-mapped). Everything else —
# the 8 doc-grained system keys and every metadata path — is undeclared here, so
# it is never pushed.
_FIELD_MAPPING = {"occurred_at": "occurred_at", "created_at": "created_at"}
# This backend runs in "split" mode: an unpushable clause is silently dropped to
# the engine's post-filter (not raised). The dedicated on_unsupported tests below
# construct their own "raise"-mode context.
_CTX = CompileContext(backend_target="KhoraChunk", field_mapping=_FIELD_MAPPING, on_unsupported="split")


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _compile(wire: dict, ctx: CompileContext = _CTX) -> Any:
    """Compile a wire-form filter and return the CompiledFilter."""
    return compile_weaviate(_ast(wire), ctx)


def _op(node: Any) -> str:
    """Return a Filter node's operator as its wire string (e.g. ``"Equal"``)."""
    # ``_Operator`` is a str-enum; ``.value`` is the wire literal. Read it without
    # importing the private enum type.
    return node.operator.value


# ===========================================================================
# Empty AST → match-all (None predicate).
# ===========================================================================


def test_empty_filter_compiles_to_none() -> None:
    # A bare RecallFilter() (no predicates) lowers to the empty match-everything
    # AND. With nothing to push, the predicate is None (engine applies no
    # server-side filter) and nothing is consumed.
    compiled = _compile({})
    assert compiled.predicate is None
    assert compiled.params == {}
    assert compiled.consumed_keys == frozenset()


def test_empty_and_node_compiles_to_none() -> None:
    compiled = compile_weaviate(FilterNode(op=Op.AND, children=()), _CTX)
    assert compiled.predicate is None


# ===========================================================================
# Operator translations on a DECLARED property → pushed + consumed.
# ===========================================================================


def test_eq_translates_to_equal() -> None:
    compiled = _compile({"occurred_at": "2026-02-03T00:00:00Z"})
    pred = compiled.predicate
    assert pred.target == "occurred_at"
    assert _op(pred) == "Equal"
    # Date binds as its .isoformat() string (lexicographic compare on the stored
    # ISO string property).
    assert isinstance(pred.value, str) and pred.value.startswith("2026-02-03")
    assert "occurred_at" in compiled.consumed_keys


@pytest.mark.parametrize(
    ("wire_op", "weaviate_operator"),
    [
        ("$gt", "GreaterThan"),
        ("$gte", "GreaterThanEqual"),
        ("$lt", "LessThan"),
        ("$lte", "LessThanEqual"),
    ],
)
def test_range_operators_translate(wire_op: str, weaviate_operator: str) -> None:
    compiled = _compile({"occurred_at": {wire_op: "2026-03-04T05:06:07Z"}})
    pred = compiled.predicate
    assert pred.target == "occurred_at"
    assert _op(pred) == weaviate_operator
    assert pred.value.startswith("2026-03-04")
    assert "occurred_at" in compiled.consumed_keys


def test_in_translates_to_any_of_equal() -> None:
    # $in is membership → any_of([equal(x) for x in v]) — an OR of scalar equals,
    # which is exact $in semantics on a scalar property. NOT contains_any (that is
    # an array-membership operator whose behavior on a scalar DATE prop is not
    # guaranteed). _FilterOr carries a .filters list of the per-value Equal leaves.
    compiled = _compile({"created_at": {"$in": ["2026-01-01T00:00:00Z", "2026-02-02T00:00:00Z"]}})
    pred = compiled.predicate
    assert _op(pred) == "Or"
    assert len(pred.filters) == 2
    assert all(leaf.target == "created_at" for leaf in pred.filters)
    assert all(_op(leaf) == "Equal" for leaf in pred.filters)
    assert {leaf.value[:10] for leaf in pred.filters} == {"2026-01-01", "2026-02-02"}
    assert "created_at" in compiled.consumed_keys


def test_in_single_value_collapses_to_bare_equal() -> None:
    # A single-element $in needs no Or wrapper — it collapses to the bare equal
    # (the OR-of-one IS just the one membership check).
    compiled = _compile({"created_at": {"$in": ["2026-01-01T00:00:00Z"]}})
    pred = compiled.predicate
    assert _op(pred) == "Equal"
    assert pred.target == "created_at"
    assert pred.value[:10] == "2026-01-01"
    assert "created_at" in compiled.consumed_keys


def test_direct_datetime_clause_binds_iso_string() -> None:
    # A leaf built directly with a tz-aware datetime operand binds as its
    # .isoformat() string (the AST is a valid compiler input independent of the
    # validator path).
    clause = FilterClause(path=("occurred_at",), op=Op.GTE, operand=datetime(2026, 1, 1, tzinfo=UTC))
    node = FilterNode(op=Op.AND, children=(clause,))
    compiled = compile_weaviate(node, _CTX)
    pred = compiled.predicate
    assert _op(pred) == "GreaterThanEqual"
    assert pred.value == datetime(2026, 1, 1, tzinfo=UTC).isoformat()


# ===========================================================================
# Superset-safe routing — negations / null are NOT pushed (not consumed).
# ===========================================================================


def test_ne_on_date_is_not_pushed() -> None:
    # $ne would drop null/absent rows server-side (false-exclude). Unpushable: no
    # predicate, nothing consumed — the engine post-filter applies it.
    compiled = _compile({"occurred_at": {"$ne": "2026-02-03T00:00:00Z"}})
    assert compiled.predicate is None
    assert "occurred_at" not in compiled.consumed_keys


def test_nin_on_date_is_not_pushed() -> None:
    compiled = _compile({"created_at": {"$nin": ["2026-01-01T00:00:00Z"]}})
    assert compiled.predicate is None
    assert "created_at" not in compiled.consumed_keys


def test_not_node_is_not_pushed() -> None:
    # A $not wrapping a (otherwise-pushable) date predicate is a negation — not
    # monotone-narrowing, so the whole node is unpushable.
    compiled = _compile({"$not": {"occurred_at": "2026-02-03T00:00:00Z"}})
    assert compiled.predicate is None
    assert "occurred_at" not in compiled.consumed_keys


def test_ne_null_is_not_pushed() -> None:
    compiled = _compile({"occurred_at": {"$ne": None}})
    assert compiled.predicate is None
    assert "occurred_at" not in compiled.consumed_keys


def test_exists_true_is_not_pushed() -> None:
    # $exists is NOT pushed on a system key. The post-filter oracle (compile_python)
    # treats a system key as always-present, so $exists:true is a CONSTANT TRUE
    # there — a record with a null property still passes. A server-side
    # is_none(False) push would EXCLUDE those null-property rows → false-exclusion.
    # Left entirely to the post-filter. Built as a direct clause (the validator
    # forbids $exists on a date key).
    clause = FilterClause(path=("occurred_at",), op=Op.EXISTS, operand=True)
    compiled = compile_weaviate(FilterNode(op=Op.AND, children=(clause,)), _CTX)
    assert compiled.predicate is None
    assert "occurred_at" not in compiled.consumed_keys


def test_exists_false_is_not_pushed() -> None:
    # $exists:false on a system key is a constant FALSE in the oracle — likewise not
    # pushed (parity with $exists:true; nothing to gain server-side).
    clause = FilterClause(path=("occurred_at",), op=Op.EXISTS, operand=False)
    compiled = compile_weaviate(FilterNode(op=Op.AND, children=(clause,)), _CTX)
    assert compiled.predicate is None
    assert "occurred_at" not in compiled.consumed_keys


def test_null_match_is_not_pushed() -> None:
    # {k: null} ($eq None) hinges on null/absent resolution; the oracle treats a
    # system key as always-present, so this is left to the post-filter rather than
    # approximated by a server-side is_none() that could diverge.
    compiled = _compile({"occurred_at": None})
    assert compiled.predicate is None
    assert "occurred_at" not in compiled.consumed_keys


def test_eq_exact_array_is_not_pushed() -> None:
    # An $eq EXACT-ARRAY operand (a bare list lowers to a tuple-operand $eq).
    # Pushing equal([...]) would match nothing server-side (a scalar property never
    # equals an array) — wrong direction, could false-exclude. Left to the
    # post-filter. Built as a direct clause (a bare list on a date key fails
    # validation; the tuple-operand exact-array form is the string-key sugar).
    clause = FilterClause(path=("occurred_at",), op=Op.EQ, operand=("a", "b"))
    compiled = compile_weaviate(FilterNode(op=Op.AND, children=(clause,)), _CTX)
    assert compiled.predicate is None
    assert "occurred_at" not in compiled.consumed_keys


def test_empty_in_is_not_pushed() -> None:
    # $in over ∅ matches nothing. There is no match-nothing Filter primitive;
    # pushing nothing and letting the post-filter exclude everything is the
    # superset-safe choice (None predicate, not consumed).
    compiled = _compile({"occurred_at": {"$in": []}})
    assert compiled.predicate is None
    assert "occurred_at" not in compiled.consumed_keys


# ===========================================================================
# Undeclared keys & metadata → never pushed.
# ===========================================================================

# The 8 doc-grained system keys are NOT in this test's field_mapping, so they are
# undeclared and never pushed (a real engine declares only the props its
# collection actually has). The 7 string-valued keys take a bare string; the one
# datetime key (source_timestamp) takes an ISO string the validator parses.
_UNDECLARED_STRING_KEYS = (
    "source_type",
    "source_name",
    "source_url",
    "external_id",
    "content_type",
    "source",
    "title",
)


@pytest.mark.parametrize("key", _UNDECLARED_STRING_KEYS)
def test_undeclared_system_key_is_not_pushed(key: str) -> None:
    # A system key absent from field_mapping is undeclared → not pushable, even with
    # a monotone-narrowing op. No predicate, not consumed.
    compiled = _compile({key: "v"})
    assert compiled.predicate is None
    assert key not in compiled.consumed_keys


def test_undeclared_date_key_is_not_pushed() -> None:
    # source_timestamp is a date key absent from field_mapping — undeclared, so a
    # narrowing op on it still does not push.
    compiled = _compile({"source_timestamp": {"$gte": "2026-01-01T00:00:00Z"}})
    assert compiled.predicate is None
    assert "source_timestamp" not in compiled.consumed_keys


def test_metadata_path_is_not_pushed() -> None:
    # A metadata sub-path is never pushed (serialized-JSON-string property; and a
    # metadata negation has the same null-drop hazard regardless). Post-filter only.
    compiled = _compile({"metadata.tier": "gold"})
    assert compiled.predicate is None
    assert "metadata.tier" not in compiled.consumed_keys


def test_bare_metadata_blob_is_not_pushed() -> None:
    compiled = _compile({"metadata": {"a": 1}})
    assert compiled.predicate is None
    assert compiled.consumed_keys == frozenset()


def test_declared_key_with_positive_op_pushes() -> None:
    # Mirror of the undeclared guard: a declared date key WITH a narrowing op DOES
    # push and land in consumed_keys — proves the gate is the field_mapping
    # membership, not the op alone.
    compiled = _compile({"created_at": {"$gte": "2026-01-01T00:00:00Z"}})
    assert compiled.predicate is not None
    assert "created_at" in compiled.consumed_keys


# ===========================================================================
# $and — push the pushable conjuncts, DROP the unpushable ones (still a superset).
# ===========================================================================


def test_and_pushes_date_conjunct_drops_metadata() -> None:
    # An $and may push the pushable date conjunct and DROP the metadata one —
    # AND-ing fewer constraints only widens the candidate set, so the push stays a
    # superset. Only the pushed leaf is consumed; the dropped one is post-filtered.
    compiled = _compile(
        {
            "occurred_at": {"$gte": "2026-01-01T00:00:00Z"},
            "metadata.tier": "gold",
        }
    )
    pred = compiled.predicate
    # The single surviving conjunct collapses to the bare date leaf (no AND wrap).
    assert pred.target == "occurred_at"
    assert _op(pred) == "GreaterThanEqual"
    assert "occurred_at" in compiled.consumed_keys
    assert "metadata.tier" not in compiled.consumed_keys


def test_and_with_two_pushable_conjuncts_builds_all_of() -> None:
    # Two pushable date conjuncts compose into an all_of (_FilterAnd). Both consumed.
    compiled = _compile(
        {
            "occurred_at": {"$gte": "2026-01-01T00:00:00Z"},
            "created_at": {"$lt": "2026-12-31T00:00:00Z"},
        }
    )
    pred = compiled.predicate
    assert _op(pred) == "And"
    assert len(pred.filters) == 2
    targets = {leaf.target for leaf in pred.filters}
    assert targets == {"occurred_at", "created_at"}
    assert compiled.consumed_keys == frozenset({"occurred_at", "created_at"})


def test_and_all_unpushable_compiles_to_none() -> None:
    # If every conjunct is unpushable (a $ne and a metadata path), the $and pushes
    # nothing — None predicate, nothing consumed.
    compiled = _compile(
        {
            "occurred_at": {"$ne": "2026-01-01T00:00:00Z"},
            "metadata.tier": "gold",
        }
    )
    assert compiled.predicate is None
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# $or — ALL-OR-NOTHING (dropping a disjunct would narrow the union).
# ===========================================================================


def test_or_all_pushable_builds_any_of() -> None:
    # Both disjuncts are pushable date predicates → push the whole $or as any_of.
    compiled = _compile(
        {
            "$or": [
                {"occurred_at": "2026-01-01T00:00:00Z"},
                {"created_at": "2026-02-02T00:00:00Z"},
            ]
        }
    )
    pred = compiled.predicate
    assert _op(pred) == "Or"
    assert len(pred.filters) == 2
    assert compiled.consumed_keys == frozenset({"occurred_at", "created_at"})


def test_or_with_metadata_child_is_wholly_unpushable() -> None:
    # One disjunct is a metadata path (unpushable). Dropping it would NARROW the
    # union and could false-exclude rows, so the WHOLE $or is unpushable — None
    # predicate, and the speculatively-consumed date leaf is rolled back (nothing
    # consumed).
    compiled = _compile(
        {
            "$or": [
                {"occurred_at": "2026-01-01T00:00:00Z"},
                {"metadata.tier": "gold"},
            ]
        }
    )
    assert compiled.predicate is None
    assert compiled.consumed_keys == frozenset()


def test_or_with_undeclared_child_is_wholly_unpushable() -> None:
    # An undeclared system key as one disjunct makes the whole $or unpushable too.
    compiled = _compile(
        {
            "$or": [
                {"occurred_at": "2026-01-01T00:00:00Z"},
                {"source_name": "linear"},
            ]
        }
    )
    assert compiled.predicate is None
    assert compiled.consumed_keys == frozenset()


def test_single_pushable_or_child_collapses_to_bare_leaf() -> None:
    # A one-branch $or normalizes (in the AST) to AND([leaf]), so a single pushable
    # disjunct compiles to the bare leaf predicate (no Or/And wrapper) and is
    # consumed. Pins the single-child collapse the reviewer flagged as untested.
    compiled = _compile({"$or": [{"occurred_at": "2026-01-01T00:00:00Z"}]})
    pred = compiled.predicate
    assert pred.target == "occurred_at"
    assert _op(pred) == "Equal"
    assert compiled.consumed_keys == frozenset({"occurred_at"})


def test_deep_and_of_unpushable_or_and_pushable_date() -> None:
    # Deep AND(OR(pushable, unpushable), pushable): the inner $or is wholly
    # unpushable (one metadata disjunct) so it is dropped, while the outer $and
    # keeps its pushable date conjunct. The surviving conjunct collapses to the
    # bare date leaf; only that key is consumed. Pins the deep-nesting case the
    # reviewer flagged.
    compiled = _compile(
        {
            "$and": [
                {"$or": [{"occurred_at": "2026-01-01T00:00:00Z"}, {"metadata.tier": "gold"}]},
                {"created_at": {"$gte": "2026-01-01T00:00:00Z"}},
            ]
        }
    )
    pred = compiled.predicate
    assert pred.target == "created_at"
    assert _op(pred) == "GreaterThanEqual"
    assert compiled.consumed_keys == frozenset({"created_at"})


# ===========================================================================
# on_unsupported policy.
# ===========================================================================


def test_unsupported_clause_raises_in_raise_mode() -> None:
    # In raise mode, an unpushable clause raises the public error rather than
    # silently dropping to the post-filter.
    ctx = CompileContext(backend_target="KhoraChunk", field_mapping=_FIELD_MAPPING, on_unsupported="raise")
    with pytest.raises(RecallFilterUnsupportedError):
        compile_weaviate(_ast({"metadata.tier": "gold"}), ctx)


def test_unsupported_negation_raises_via_clause_in_raise_mode() -> None:
    # A $ne on a declared key reaches _compile_system_clause, returns None (not
    # narrowing), then routes through _unsupported → raises in raise mode.
    ctx = CompileContext(backend_target="KhoraChunk", field_mapping=_FIELD_MAPPING, on_unsupported="raise")
    with pytest.raises(RecallFilterUnsupportedError):
        compile_weaviate(_ast({"occurred_at": {"$ne": "2026-01-01T00:00:00Z"}}), ctx)


def test_split_mode_drops_unsupported_silently() -> None:
    # Split mode (the mode this backend runs in) drops the unpushable clause
    # without raising: None predicate, nothing consumed.
    ctx = CompileContext(backend_target="KhoraChunk", field_mapping=_FIELD_MAPPING, on_unsupported="split")
    compiled = compile_weaviate(_ast({"metadata.tier": "gold"}), ctx)
    assert compiled.predicate is None
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# field_mapping remaps the physical property name.
# ===========================================================================


def test_field_mapping_remaps_property_name() -> None:
    # field_mapping maps the logical key to the physical property name the
    # collection actually declares — the pushed Filter targets the physical name.
    ctx = CompileContext(backend_target="KhoraChunk", field_mapping={"occurred_at": "event_ts"})
    compiled = compile_weaviate(_ast({"occurred_at": "2026-02-03T00:00:00Z"}), ctx)
    pred = compiled.predicate
    assert pred.target == "event_ts"
    # consumed_keys is the LOGICAL path, not the physical name.
    assert "occurred_at" in compiled.consumed_keys


def test_no_field_mapping_pushes_nothing() -> None:
    # With no field_mapping, NO property is declared, so even a narrowing op on a
    # system key is not pushed (split mode drops it to the post-filter).
    ctx = CompileContext(backend_target="KhoraChunk", on_unsupported="split")
    compiled = compile_weaviate(_ast({"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}}), ctx)
    assert compiled.predicate is None
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# CompiledFilter envelope.
# ===========================================================================


def test_compiled_filter_carries_canonical_hash() -> None:
    node = _ast({"occurred_at": "2026-02-03T00:00:00Z"})
    compiled = compile_weaviate(node, _CTX)
    assert compiled.canonical_hash == canonical_hash(node)


def test_params_always_empty() -> None:
    # Weaviate binds operands inline in the Filter object — params is always empty.
    compiled = _compile({"occurred_at": "2026-02-03T00:00:00Z"})
    assert compiled.params == {}


def test_consumed_keys_is_frozenset() -> None:
    compiled = _compile({"occurred_at": "2026-02-03T00:00:00Z"})
    assert isinstance(compiled.consumed_keys, frozenset)


# ===========================================================================
# Module imports without the weaviate extra (the lazy-import contract).
# ===========================================================================


def test_module_imports_without_weaviate_symbol_at_top_level() -> None:
    # The compiler module must import even when weaviate-client is absent — the
    # ``Filter`` symbol is imported lazily inside the builder, never at module
    # scope. Assert there is no module-level ``weaviate`` / ``Filter`` binding.
    import khora.filter.compilers.weaviate as mod

    assert not hasattr(mod, "Filter")
    assert "weaviate" not in vars(mod)
