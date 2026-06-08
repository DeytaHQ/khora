"""Unit tests for the Chronicle recall-filter compiler pushdown split (Layer 4).

``@internal``. The Chronicle compiler lowers a canonical
:class:`~khora.filter.ast.FilterNode` into a narrowing **date-bound** the engine
intersects with its recency window. Chronicle has no general predicate-pushdown
surface, but every retrieval channel already honors a ``created_after`` /
``created_before`` window on the indexed event timestamp — so the one thing this
compiler pushes down is exactly that bound. Everything else (the eight
denormalized document keys + all metadata) is left UNCONSUMED for the engine to
post-filter with :func:`~khora.filter.compilers.python.compile_python`.

These tests pin the SPLIT CONTRACT (task part b), against the actual implemented
shape:

* The compiler pushes ONLY the recency window's PRIMARY dimension —
  ``source_timestamp`` — and only a *conjunctive* (top-level ``AND``) range /
  ``$eq`` clause on it. The window narrows on ``COALESCE(source_timestamp,
  created_at)``, so a bound on ``created_at`` (the fallback axis) or ``occurred_at``
  (a different dimension the window never references) is cross-dimension and would
  false-exclude — both are left unconsumed and enforced by the post-filter instead.
* The unconsumed system keys (the 7 string denorm doc keys + the post-filtered
  date keys ``occurred_at`` / ``created_at``) and every metadata predicate are NOT
  pushed down — absent from ``consumed_keys``.
* The exact pushed set is pinned in the single :data:`_PUSHED_DATE_KEYS` constant
  (one edit point if Backend's final DTO confirmation shifts it).
* ``predicate`` is a :class:`ChronicleDateBound(created_after, created_before)`
  frozen dataclass (NOT a query string). ``$gt``/``$gte`` tighten the lower bound,
  ``$lt``/``$lte`` the upper, ``$eq`` pins a point window; ``max``/``min`` combine
  multiple clauses. Boundary strictness is intentionally dropped (the window is a
  safe over-approximation; ``compile_python`` re-checks strictness on survivors).
* A date clause under ``$or`` / ``$not``, and ``$ne``/``$in``/``$nin``/``{k:null}``
  on a date key, are NOT a single contiguous window → left unconsumed.
* The :class:`CompiledFilter` envelope: ``params`` empty, ``consumed_keys`` a
  frozenset, ``canonical_hash`` present.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from khora.filter import RecallFilter
from khora.filter.ast import FilterNode, parse_to_ast
from khora.filter.compilers.chronicle import ChronicleDateBound, compile_chronicle
from khora.filter.context import CompileContext
from khora.filter.model import SYSTEM_KEYS

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Key-set constants — THE SINGLE EDIT POINT for the pushdown date-key set.
# ---------------------------------------------------------------------------
#
# Pinned to the implemented contract: the recency
# window narrows on COALESCE(source_timestamp, created_at), so a bound on
# source_timestamp is the only same-axis-safe pushdown — for every row a
# source_timestamp filter keeps, the window value equals source_timestamp. The
# event-time key occurred_at and the fallback key created_at are NOT the window
# axis, so they are post-filtered (cross-dimension), never folded into the window.
# This one frozenset is the single edit point; the parametrized tests below derive
# from it (with _DATE_TYPED) so a pushdown-set change stays self-consistent.
_PUSHED_DATE_KEYS = frozenset({"source_timestamp"})

# The three date-typed system keys (the candidate pushdown universe). The
# un-pushed date keys are exactly this set minus _PUSHED_DATE_KEYS, so date-key
# tests derive from these two constants and stay correct across a pushdown flip.
_DATE_TYPED = frozenset({"occurred_at", "created_at", "source_timestamp"})

# Everything NOT pushed down: the 7 string (denormalized document) system keys
# PLUS the date keys that are post-filtered = the complement of _PUSHED_DATE_KEYS.
_UNCONSUMED_SYSTEM_KEYS = SYSTEM_KEYS - _PUSHED_DATE_KEYS


def test_unconsumed_system_key_set_complements_pushed() -> None:
    # The unconsumed set is exactly SYSTEM_KEYS minus the pushed date key(s): the
    # 7 string keys + the post-filtered (non-pushed) date keys.
    assert _UNCONSUMED_SYSTEM_KEYS == SYSTEM_KEYS - _PUSHED_DATE_KEYS
    assert len(_UNCONSUMED_SYSTEM_KEYS) == len(SYSTEM_KEYS) - len(_PUSHED_DATE_KEYS)
    # The date-typed keys NOT pushed are post-filtered (cross-dimension), not folded.
    assert (_DATE_TYPED - _PUSHED_DATE_KEYS) <= _UNCONSUMED_SYSTEM_KEYS


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

# The Chronicle engine registers + drives the compiler with the "chunks" storage
# target and on_unsupported="split" so the unconsumed remainder is post-filtered
# rather than raised (CompilerRegistry.register("chronicle", "chunks", ...)).
# compile_chronicle does not read backend_target, but matching the engine's value
# keeps the unit-test context faithful to production usage.
_CTX = CompileContext(backend_target="chunks", on_unsupported="split")
_RAISE_CTX = CompileContext(backend_target="chunks", on_unsupported="raise")


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _compiled(wire: dict, ctx: CompileContext = _CTX) -> Any:
    return compile_chronicle(_ast(wire), ctx)


_LB = "2026-01-01T00:00:00Z"  # a lower-bound instant
_UB = "2026-12-31T00:00:00Z"  # an upper-bound instant
_LB_DT = datetime(2026, 1, 1, tzinfo=UTC)
_UB_DT = datetime(2026, 12, 31, tzinfo=UTC)

# A representative pushable date key for single-key tests (driven by the constant
# so a change to _PUSHED_DATE_KEYS flows through without editing every test).
_PUSHED_KEY = sorted(_PUSHED_DATE_KEYS)[0]


# ===========================================================================
# Pushed date keys → consumed + folded into the bound.
# ===========================================================================


@pytest.mark.parametrize("key", sorted(_PUSHED_DATE_KEYS))
def test_pushed_date_key_lower_bound_is_consumed(key: str) -> None:
    compiled = _compiled({key: {"$gte": _LB}})
    assert key in compiled.consumed_keys
    assert isinstance(compiled.predicate, ChronicleDateBound)
    assert compiled.predicate.created_after == _LB_DT
    assert compiled.predicate.created_before is None


@pytest.mark.parametrize("key", sorted(_PUSHED_DATE_KEYS))
def test_pushed_date_key_upper_bound_is_consumed(key: str) -> None:
    compiled = _compiled({key: {"$lte": _UB}})
    assert key in compiled.consumed_keys
    assert compiled.predicate.created_before == _UB_DT
    assert compiled.predicate.created_after is None


@pytest.mark.parametrize("op", ["$gt", "$gte"])
def test_lower_ops_tighten_lower_bound(op: str) -> None:
    # Boundary strictness ($gt vs $gte) is intentionally NOT carried — both fold to
    # the same lower bound instant (the post-filter re-checks the strict compare).
    compiled = _compiled({_PUSHED_KEY: {op: _LB}})
    assert compiled.predicate.created_after == _LB_DT
    assert compiled.predicate.created_before is None


@pytest.mark.parametrize("op", ["$lt", "$lte"])
def test_upper_ops_tighten_upper_bound(op: str) -> None:
    compiled = _compiled({_PUSHED_KEY: {op: _UB}})
    assert compiled.predicate.created_before == _UB_DT
    assert compiled.predicate.created_after is None


def test_eq_pins_point_window() -> None:
    # $eq on a pushed date key pins BOTH bounds to the same instant.
    compiled = _compiled({_PUSHED_KEY: _LB})
    assert compiled.predicate.created_after == _LB_DT
    assert compiled.predicate.created_before == _LB_DT
    assert _PUSHED_KEY in compiled.consumed_keys


def test_two_sided_range_on_one_key_folds_both_bounds() -> None:
    compiled = _compiled({_PUSHED_KEY: {"$gte": _LB, "$lte": _UB}})
    assert compiled.predicate.created_after == _LB_DT
    assert compiled.predicate.created_before == _UB_DT
    assert _PUSHED_KEY in compiled.consumed_keys


def test_multiple_lower_bounds_on_pushed_key_intersect_via_max() -> None:
    # Two conjunctive lower bounds on the same pushed key → the LATER (max) wins.
    early_lb = "2026-01-01T00:00:00Z"
    late_lb = "2026-06-01T00:00:00Z"
    compiled = _compiled({_PUSHED_KEY: {"$gt": early_lb, "$gte": late_lb}})
    assert compiled.predicate.created_after == datetime(2026, 6, 1, tzinfo=UTC)
    assert compiled.consumed_keys == frozenset({_PUSHED_KEY})


def test_date_literal_operand_folds() -> None:
    # The compiler accepts a DateLiteral operand directly off the AST (the AST is a
    # valid compiler input independent of the wire validator — the {"$date": ...}
    # wire form is metadata-only, so a date-key DateLiteral is reached by building
    # the clause directly, as the postgres/cypher unit tests also do).
    from khora.filter.ast import DateLiteral, FilterClause
    from khora.filter.model import Op

    clause = FilterClause(path=(_PUSHED_KEY,), op=Op.GTE, operand=DateLiteral(value=_LB_DT))
    node = FilterNode(op=Op.AND, children=(clause,))
    compiled = compile_chronicle(node, _CTX)
    assert compiled.predicate.created_after == _LB_DT
    assert _PUSHED_KEY in compiled.consumed_keys


# ===========================================================================
# Date keys that DON'T fold into a contiguous window → unconsumed.
# ===========================================================================


def test_cross_dimension_date_keys_are_not_pushed() -> None:
    # The date-typed keys that are NOT the pushed window axis are cross-dimension —
    # pushing them would false-exclude against COALESCE(source_timestamp,
    # created_at) — so they are post-filtered, never folded into the bound. Driven
    # by the _PUSHED_DATE_KEYS constant so it stays correct across a pushdown-set
    # change (the un-pushed date keys are exactly the date-typed keys minus the
    # pushed one).
    for key in sorted(_DATE_TYPED - _PUSHED_DATE_KEYS):
        compiled = _compiled({key: {"$gte": _LB}})
        assert key not in compiled.consumed_keys, f"{key} must be post-filtered, not pushed"
        assert compiled.predicate == ChronicleDateBound()


def test_pushed_date_key_ne_is_not_pushed() -> None:
    # $ne is not a single contiguous window — left unconsumed even on the pushed key.
    compiled = _compiled({_PUSHED_KEY: {"$ne": _LB}})
    assert _PUSHED_KEY not in compiled.consumed_keys
    assert compiled.predicate == ChronicleDateBound()


def test_pushed_date_key_in_is_not_pushed() -> None:
    compiled = _compiled({_PUSHED_KEY: {"$in": [_LB, _UB]}})
    assert _PUSHED_KEY not in compiled.consumed_keys
    assert compiled.predicate == ChronicleDateBound()


def test_pushed_date_key_null_match_is_not_pushed() -> None:
    compiled = _compiled({_PUSHED_KEY: None})
    assert _PUSHED_KEY not in compiled.consumed_keys


def test_pushed_date_key_under_or_is_not_pushed() -> None:
    # A pushed date clause inside an $or is NOT conjunctive — folding a bound out of
    # a disjunction would wrongly narrow the other branch, so it is left unconsumed.
    compiled = _compiled({"$or": [{_PUSHED_KEY: {"$gte": _LB}}, {"source_name": "linear"}]})
    assert compiled.consumed_keys == frozenset()
    assert compiled.predicate == ChronicleDateBound()


def test_pushed_date_key_under_not_is_not_pushed() -> None:
    compiled = _compiled({"$not": {_PUSHED_KEY: {"$gte": _LB}}})
    assert compiled.consumed_keys == frozenset()
    assert compiled.predicate == ChronicleDateBound()


# ===========================================================================
# Denorm doc keys + metadata → NEVER pushed.
# ===========================================================================


@pytest.mark.parametrize("key", sorted(_UNCONSUMED_SYSTEM_KEYS))
def test_unconsumed_system_key_is_not_pushed(key: str) -> None:
    # Each unconsumed system key (the 7 string keys + the post-filtered date keys)
    # is left for the post-filter, never folded into the date bound. A date-typed
    # unconsumed key takes a range operand; a string key takes a scalar.
    wire: dict[str, Any] = {key: {"$gte": _LB}} if key in _DATE_TYPED else {key: "x"}
    compiled = _compiled(wire)
    assert key not in compiled.consumed_keys


def test_metadata_path_is_not_pushed() -> None:
    compiled = _compiled({"metadata.tier": "gold"})
    assert not compiled.consumed_keys
    assert compiled.predicate == ChronicleDateBound()


def test_bare_metadata_blob_is_not_pushed() -> None:
    compiled = _compiled({"metadata": {"a": 1}})
    assert compiled.consumed_keys == frozenset()


# ===========================================================================
# Mixed filter — only the conjunctive date key(s) are consumed.
# ===========================================================================


def test_mixed_filter_consumes_only_pushed_date_key() -> None:
    # The pushed date key folds; source_name + metadata.tier stay unconsumed.
    wire = {
        _PUSHED_KEY: {"$gte": _LB},
        "source_name": "linear",
        "metadata.tier": "gold",
    }
    compiled = _compiled(wire)
    assert compiled.consumed_keys == frozenset({_PUSHED_KEY})
    assert compiled.predicate.created_after == _LB_DT


def test_consumed_keys_subset_of_pushed_date_keys() -> None:
    # STRUCTURAL invariant: the compiler never consumes a non-pushable key, even
    # when the filter spans all three date-typed keys + string keys + metadata.
    wire = {
        "created_at": {"$gte": _LB},
        "occurred_at": {"$lte": _UB},
        "source_timestamp": {"$gte": _LB},
        "source_name": "linear",
        "title": "x",
        "metadata.tier": "gold",
    }
    compiled = _compiled(wire)
    assert compiled.consumed_keys <= _PUSHED_DATE_KEYS
    assert not (compiled.consumed_keys & _UNCONSUMED_SYSTEM_KEYS)
    assert not any(str(k).startswith("metadata") for k in compiled.consumed_keys)


# ===========================================================================
# on_unsupported policy.
# ===========================================================================


def test_split_mode_does_not_raise_on_unconsumed() -> None:
    # The engine's mode: unconsumed content is silently left for the post-filter.
    compiled = _compiled({"source_name": "linear", "metadata.tier": "gold"})
    assert compiled.consumed_keys == frozenset()


def test_raise_mode_raises_on_unpushable_clause() -> None:
    from khora.filter import RecallFilterUnsupportedError

    with pytest.raises(RecallFilterUnsupportedError):
        _compiled({"source_name": "linear"}, _RAISE_CTX)


def test_raise_mode_allows_pure_pushed_date_filter() -> None:
    # A filter that is entirely consumable (pushed-key) date bounds does not raise
    # even in "raise" mode (there is no unconsumed remainder).
    compiled = _compiled({_PUSHED_KEY: {"$gte": _LB}}, _RAISE_CTX)
    assert compiled.consumed_keys == frozenset({_PUSHED_KEY})


# ===========================================================================
# CompiledFilter envelope.
# ===========================================================================


def test_empty_filter_consumes_nothing_and_is_unbounded() -> None:
    compiled = _compiled({})
    assert compiled.consumed_keys == frozenset()
    assert compiled.predicate == ChronicleDateBound()


def test_params_is_empty_dict() -> None:
    # The bound carries datetimes directly; no out-of-band binds.
    compiled = _compiled({_PUSHED_KEY: {"$gte": _LB}})
    assert compiled.params == {}


def test_consumed_keys_is_frozenset() -> None:
    compiled = _compiled({_PUSHED_KEY: {"$gte": _LB}})
    assert isinstance(compiled.consumed_keys, frozenset)


def test_carries_canonical_hash() -> None:
    from khora.filter.ast import canonical_hash

    node = _ast({_PUSHED_KEY: {"$gte": _LB}})
    compiled = compile_chronicle(node, _CTX)
    assert compiled.canonical_hash == canonical_hash(node)
