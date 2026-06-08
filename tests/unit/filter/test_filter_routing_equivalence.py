"""Routing-equivalence parity tests for the Weaviate recall-filter path — ``@internal``.

The Weaviate backend honors a deterministic recall filter in two halves
(``engines/skeleton/backends/weaviate.py::search``):

1. **Push-down.** ``compile_weaviate(ast, ctx)`` lowers the *superset-safe* slice
   of the AST to a native Weaviate v4 ``Filter`` (passed as ``filters=``). Only
   the two declared DATE properties (``occurred_at`` / ``created_at``) push down;
   every other predicate is left unconsumed.
2. **Post-filter.** ``compile_python(WHOLE ast, ctx)`` compiles the *entire* AST
   to an in-memory ``callable(record) -> bool`` re-applied to every candidate the
   server returned. ``compile_python`` is the ORACLE — its accept set IS the
   correct §4 result set (it is the oracle the SQL/Cypher compilers are checked
   against, see ``compilers/python.py``).

This module proves the ticket AC: *the Weaviate path (pushdown + python
post-filter) returns the SAME row-set as the pure ``compile_python`` oracle*. The
two properties, established over a corpus of in-memory records and a battery of
filters spanning the full operator set, are:

* **Superset-safety** (the part that can actually go wrong): the records the
  ``compile_weaviate`` pushdown KEEPS must be a SUPERSET of the records the oracle
  ``compile_python`` accepts. The pushdown must NEVER false-exclude a record the
  oracle keeps — over-returning is safe (the post-filter narrows), under-returning
  is a correctness bug. We evaluate the emitted Weaviate ``Filter`` against each
  in-memory record with a small interpreter for the v4 ``_Filters`` tree
  (``_weaviate_keeps``), so a real false-exclusion in the pushdown surfaces here.
* **Post-filter parity** (by construction given superset-safety, but asserted
  end-to-end so a regression in the pushdown OR the wiring is caught): applying
  the ``compile_python`` post-filter to the kept candidates yields EXACTLY the
  oracle's accept set over the whole corpus — i.e. the final result == oracle.

The Weaviate ``Filter`` objects are private dataclasses
(``_FilterValue`` / ``_FilterAnd`` / ``_FilterOr``); the interpreter reads their
stable instance attributes (``target`` / ``operator`` / ``value`` / ``filters``)
and compares the operator enum by its ``.value`` string, importing no private
symbol — the same convention as ``test_compile_weaviate.py``.

Gated on the weaviate extra with ``importorskip`` (the interpreter builds real
``Filter`` objects, so it needs the client). The file collects cleanly WITHOUT
the extra — the module-level skip fires before any ``Filter`` is touched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from khora.filter import RecallFilter
from khora.filter.ast import FilterClause, FilterNode, parse_to_ast
from khora.filter.compilers.python import compile_python
from khora.filter.compilers.weaviate import compile_weaviate
from khora.filter.context import CompileContext
from khora.filter.model import Op

# Hard skip if the weaviate extra is absent: the interpreter introspects real
# ``Filter`` objects, so it needs the client. Skip rather than fail CI red on a
# stack that does not install the optional weaviate-client.
pytest.importorskip("weaviate", reason="weaviate-client extra not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Compile contexts — mirror the engine wiring (weaviate.py::search).
# ---------------------------------------------------------------------------

# The push-down declares exactly the two DATE properties the KhoraChunk
# collection stores as queryable Weaviate properties (identity-mapped). This IS
# the field_mapping the engine passes; everything else is undeclared → unpushed.
_FIELD_MAPPING = {"occurred_at": "occurred_at", "created_at": "created_at"}
_PUSH_CTX = CompileContext(
    backend_target="KhoraChunk",
    field_mapping=_FIELD_MAPPING,
    on_unsupported="split",
)
# The post-filter compiles the WHOLE AST with no field_mapping (it reads record
# attributes directly) — exactly as the engine constructs it.
_POST_CTX = CompileContext(backend_target="KhoraChunk", on_unsupported="split")


# ---------------------------------------------------------------------------
# Record corpus.
#
# Each record is a plain dict (compile_python is record-shape-defensive — it
# resolves a system key by attribute then mapping access, so a dict event row is
# a valid record). The two declared date keys carry real datetimes (or are
# absent, or None) to exercise the range / contains pushdown — and the
# UNPUSHED null/$exists path — against present / null / missing values. The
# metadata blob carries arrays, missing keys, wrong-type values, an explicit
# JSON null, and ISO date strings.
# ---------------------------------------------------------------------------


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=UTC)


CORPUS: list[dict[str, Any]] = [
    # 0: both dates present, rich metadata with arrays + nested object.
    {
        "occurred_at": _dt(2026, 1, 10),
        "created_at": _dt(2026, 1, 11),
        "source_name": "linear",
        "source_type": "ticket",
        "metadata": {
            "tier": "gold",
            "tags": ["urgent", "backend"],
            "priority": 5,
            "labels": {"team": "core"},
            "score": 4.5,
            "active": True,
            "deadline": "2026-03-01T00:00:00Z",
            "nullable": None,
        },
    },
    # 1: occurred_at later, created_at earlier; different scalar metadata.
    {
        "occurred_at": _dt(2026, 6, 1),
        "created_at": _dt(2026, 1, 1),
        "source_name": "github",
        "source_type": "pr",
        "metadata": {
            "tier": "silver",
            "tags": ["backend"],
            "priority": 2,
            "score": 9.0,
            "active": False,
            "deadline": "2026-02-01T00:00:00Z",
        },
    },
    # 2: occurred_at present, created_at absent (missing key) → resolves to None.
    {
        "occurred_at": _dt(2026, 3, 15),
        "source_name": "linear",
        "source_type": "ticket",
        "metadata": {
            "tier": "gold",
            "tags": [],  # empty array
            "priority": "high",  # wrong-type vs a numeric range
            "labels": {"team": "platform"},
        },
    },
    # 3: occurred_at explicitly None, created_at present.
    {
        "occurred_at": None,
        "created_at": _dt(2026, 5, 5),
        "source_name": "slack",
        "metadata": {
            "tier": "bronze",
            "tags": ["frontend", "urgent"],
            "priority": 0,
            "nullable": None,  # explicit JSON null
        },
    },
    # 4: both dates absent (no keys at all) → both resolve to None. No metadata.
    {
        "source_name": "notion",
        "source_type": "doc",
    },
    # 5: both dates present and EQUAL to a probe value; metadata with a list of
    #    objects + a numeric string deadline.
    {
        "occurred_at": _dt(2026, 2, 3),
        "created_at": _dt(2026, 2, 3),
        "source_name": "linear",
        "metadata": {
            "tier": "gold",
            "tags": ["backend", "infra"],
            "priority": 7,
            "deadline": "not-a-date",  # unparseable → $date excludes
            "score": 1,  # int (not float)
        },
    },
    # 6: occurred_at present, metadata present but EMPTY dict.
    {
        "occurred_at": _dt(2026, 4, 20),
        "created_at": _dt(2026, 4, 20),
        "source_name": "github",
        "metadata": {},
    },
    # 7: metadata is None (coalesces to {} in compile_python); dates present.
    {
        "occurred_at": _dt(2026, 7, 7),
        "created_at": _dt(2026, 7, 7),
        "source_name": "linear",
        "metadata": None,
    },
]


# ---------------------------------------------------------------------------
# Weaviate ``_Filters`` interpreter.
#
# A small, faithful evaluator of the v4 Filter combinator tree against one
# in-memory record. Reads only the stable instance attributes the compiler
# emits — ``.filters`` on _FilterAnd/_FilterOr, ``.target`` / ``.operator`` /
# ``.value`` on _FilterValue — and dispatches on ``operator.value`` (the wire
# literal). The final compiler emits exactly seven operators, exhaustively
# enumerated against ``compilers/weaviate.py``: the And / Or combinators plus the
# five leaf comparisons Equal / GreaterThan / GreaterThanEqual / LessThan /
# LessThanEqual. It deliberately does NOT push ``is_none`` for $exists / null
# (that would diverge from the always-present post-filter oracle), and a ``$in``
# on a declared key lowers to an ``any_of([equal(x)…])`` Or-of-Equals tree (a
# single-member $in collapses to a bare ``Equal`` leaf) — both handled by the
# And / Or walk + the Equal leaf, so no ``ContainsAny`` / ``IsNull`` branch is
# needed. A single-child ``all_of`` / ``any_of`` does not wrap (it returns the
# bare child leaf); the ``_weaviate_keeps`` recursion treats any node that is not
# an And / Or as a leaf, so that collapse is handled.
#
# Date binding model: the pushdown binds each datetime operand as its
# ``.isoformat()`` string (the chunk stores DATE props as ISO strings and
# Weaviate compares lexicographically over the UTC-normalized values). So the
# interpreter renders the record's stored datetime to its isoformat string and
# compares string-to-string — exactly what the server does on disk.
# ---------------------------------------------------------------------------


def _record_iso(record: dict[str, Any], target: str) -> str | None:
    """Render the record's value at ``target`` as the ISO string Weaviate stores.

    Returns ``None`` when the property is absent or explicitly None (the
    null/absent case a value comparator implicitly drops). The two declared
    targets are datetimes in the corpus; any other stored type passes through
    ``str`` defensively.
    """
    value = record.get(target)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _eval_leaf(leaf: Any, record: dict[str, Any]) -> bool:
    """Evaluate one ``_FilterValue`` leaf against a record (Weaviate semantics).

    The compiler emits exactly five leaf operators (exhaustively enumerated
    against the final ``compile_weaviate``): ``Equal`` and the four range ops. It
    does NOT emit ``IsNull`` ($exists / null are left to the post-filter — the
    oracle treats a system key as always-present, so an ``is_none`` push would
    diverge) nor ``ContainsAny`` ($in lowers to an ``any_of([equal(x)…])``
    Or-of-Equals tree the And/Or walk handles). Any other ``operator.value`` is a
    compiler regression and trips the ``AssertionError`` guard below.
    """
    op = leaf.operator.value
    target = leaf.target
    stored = _record_iso(record, target)

    if stored is None:
        # All five emittable leaf ops are value comparisons; Weaviate's value
        # comparators do NOT match a null/absent property — they implicitly drop
        # it. This is precisely the null-drop hazard the compiler avoids by never
        # pushing a negation: a positive op never keeps a null row.
        return False

    if op == "Equal":
        return stored == leaf.value
    if op == "GreaterThan":
        return stored > leaf.value
    if op == "GreaterThanEqual":
        return stored >= leaf.value
    if op == "LessThan":
        return stored < leaf.value
    if op == "LessThanEqual":
        return stored <= leaf.value
    raise AssertionError(f"interpreter does not model Weaviate operator {op!r}")


def _weaviate_keeps(predicate: Any, record: dict[str, Any]) -> bool:
    """Evaluate a compiled Weaviate ``Filter`` tree against a record.

    ``None`` predicate (nothing pushable) keeps EVERY record — the over-fetch
    keeps all candidates and the post-filter does all the narrowing. A combinator
    node dispatches on ``operator.value`` ("And" / "Or"); anything else (including
    a single-child ``all_of`` / ``any_of`` that collapsed to a bare leaf) is a
    leaf, evaluated via :func:`_eval_leaf`.
    """
    if predicate is None:
        return True
    op = getattr(predicate, "operator", None)
    if op is not None and op.value == "And":
        return all(_weaviate_keeps(child, record) for child in predicate.filters)
    if op is not None and op.value == "Or":
        return any(_weaviate_keeps(child, record) for child in predicate.filters)
    return _eval_leaf(predicate, record)


# ---------------------------------------------------------------------------
# Routing helpers — mirror weaviate.py::search exactly.
# ---------------------------------------------------------------------------


def _route(ast: FilterNode) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the two-stage Weaviate routing over the corpus.

    Returns ``(kept_candidates, final)``:

    * ``kept_candidates`` — records the push-down KEEPS (the over-fetch
      candidate pool: a ``None`` predicate keeps everything; otherwise the
      ``_weaviate_keeps`` interpreter decides).
    * ``final`` — the post-filter (``compile_python`` over the WHOLE AST) applied
      to the kept candidates. This is the row-set the engine actually returns.
    """
    pushed = compile_weaviate(ast, _PUSH_CTX).predicate
    post = compile_python(ast, _POST_CTX).predicate
    kept = [r for r in CORPUS if _weaviate_keeps(pushed, r)]
    final = [r for r in kept if post(r)]
    return kept, final


def _oracle(ast: FilterNode) -> list[dict[str, Any]]:
    """The pure ``compile_python`` oracle accept-set over the whole corpus."""
    predicate = compile_python(ast, _POST_CTX).predicate
    return [r for r in CORPUS if predicate(r)]


def _ids(records: list[dict[str, Any]]) -> set[int]:
    """Identity set of records (by position in CORPUS) for set comparison."""
    return {CORPUS.index(r) for r in records}


# ---------------------------------------------------------------------------
# The filter battery.
#
# Each entry is a (label, wire-filter-dict) pair spanning the full operator set
# against the declared date keys, the undeclared system keys, and metadata
# (arrays, missing keys, wrong-type, JSON null, ISO date strings, exact-array vs
# $in, mixed declared+metadata). Built via the public RecallFilter so they are
# real, validated filters lowered through parse_to_ast.
# ---------------------------------------------------------------------------

_ISO = "2026-02-03T00:00:00Z"

_WIRE_FILTERS: list[tuple[str, dict[str, Any]]] = [
    ("empty", {}),
    # --- declared date keys, pushable ops ---
    ("date_eq", {"occurred_at": _ISO}),
    ("date_gt", {"occurred_at": {"$gt": "2026-03-01T00:00:00Z"}}),
    ("date_gte", {"occurred_at": {"$gte": "2026-03-01T00:00:00Z"}}),
    ("date_lt", {"created_at": {"$lt": "2026-03-01T00:00:00Z"}}),
    ("date_lte", {"created_at": {"$lte": "2026-02-03T00:00:00Z"}}),
    ("date_in", {"created_at": {"$in": ["2026-01-01T00:00:00Z", "2026-02-03T00:00:00Z"]}}),
    ("date_in_single", {"created_at": {"$in": ["2026-02-03T00:00:00Z"]}}),
    # occurred_at $in where a candidate's date EQUALS an operand and is kept —
    # exercises the date-$in membership pushdown (an any_of([equal(x)…]) Or-of-Equals
    # tree) with a real surviving row, so the superset-safety assertion is
    # non-vacuous.
    ("date_in_occurred", {"occurred_at": {"$in": ["2026-01-10T00:00:00Z", "2026-06-01T00:00:00Z"]}}),
    # --- declared date keys, UNPUSHABLE (negation / null) ops ---
    ("date_ne", {"occurred_at": {"$ne": _ISO}}),
    ("date_ne_null", {"occurred_at": {"$ne": None}}),
    ("date_nin", {"created_at": {"$nin": ["2026-01-01T00:00:00Z"]}}),
    ("date_null", {"occurred_at": None}),
    # --- undeclared system keys (never pushed) ---
    ("undeclared_eq", {"source_name": "linear"}),
    ("undeclared_in", {"source_name": {"$in": ["linear", "github"]}}),
    ("undeclared_ne", {"source_name": {"$ne": "linear"}}),
    ("undeclared_exists", {"source_type": {"$exists": True}}),
    ("undeclared_missing", {"source_type": {"$exists": False}}),
    # --- metadata: never pushed; full operator set ---
    ("md_eq_scalar", {"metadata.tier": "gold"}),
    ("md_array_containment", {"metadata.tags": "urgent"}),
    ("md_in", {"metadata.tier": {"$in": ["gold", "silver"]}}),
    ("md_nin", {"metadata.tier": {"$nin": ["bronze"]}}),
    ("md_range", {"metadata.priority": {"$gte": 5}}),
    ("md_range_wrong_type", {"metadata.priority": {"$gt": 1}}),  # record 2 has str priority
    ("md_ne", {"metadata.tier": {"$ne": "gold"}}),
    ("md_exists_true", {"metadata.labels": {"$exists": True}}),
    ("md_exists_false", {"metadata.labels": {"$exists": False}}),
    ("md_null_match", {"metadata.nullable": None}),
    ("md_ne_null", {"metadata.nullable": {"$ne": None}}),
    ("md_exact_array", {"metadata.tags": ["urgent", "backend"]}),  # exact-array $eq
    ("md_exact_array_vs_in", {"metadata.tags": {"$in": ["urgent", "backend"]}}),  # membership
    ("md_empty_in", {"metadata.tags": {"$in": []}}),  # $in over ∅ → ∅
    ("md_empty_nin", {"metadata.tags": {"$nin": []}}),  # $nin over ∅ → all
    ("md_date_str", {"metadata.deadline": {"$date": "2026-03-01T00:00:00Z"}}),
    ("md_date_range", {"metadata.deadline": {"$gt": {"$date": "2026-01-01T00:00:00Z"}}}),
    ("md_object_eq", {"metadata.labels": {"team": "core"}}),  # subdocument object_equal
    ("md_bool", {"metadata.active": True}),
    # --- mixed declared (pushable) + metadata (post-only) ---
    ("mixed_and", {"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}, "metadata.tier": "gold"}),
    (
        "mixed_and_two_dates_plus_md",
        {
            "occurred_at": {"$gte": "2026-01-01T00:00:00Z"},
            "created_at": {"$lt": "2026-12-31T00:00:00Z"},
            "metadata.priority": {"$gte": 5},
        },
    ),
    ("mixed_all_unpushable", {"occurred_at": {"$ne": _ISO}, "metadata.tier": "gold"}),
    # --- explicit logical operators ---
    (
        "or_both_pushable",
        {"$or": [{"occurred_at": "2026-01-10T00:00:00Z"}, {"created_at": "2026-02-03T00:00:00Z"}]},
    ),
    (
        "or_one_metadata",  # one disjunct unpushable → whole $or unpushable
        {"$or": [{"occurred_at": "2026-01-10T00:00:00Z"}, {"metadata.tier": "gold"}]},
    ),
    (
        "or_one_undeclared",
        {"$or": [{"occurred_at": "2026-01-10T00:00:00Z"}, {"source_name": "linear"}]},
    ),
    (
        "and_of_or",
        {
            "$and": [
                {"$or": [{"occurred_at": "2026-01-10T00:00:00Z"}, {"created_at": "2026-02-03T00:00:00Z"}]},
                {"metadata.tier": "gold"},
            ]
        },
    ),
    (
        "nor",  # $nor desugars to $not($or(...))
        {"$nor": [{"metadata.tier": "bronze"}, {"source_name": "slack"}]},
    ),
    (
        "not_date",  # $not over a pushable date → unpushable as a whole
        {"$not": {"occurred_at": "2026-01-10T00:00:00Z"}},
    ),
    (
        "not_metadata",
        {"$not": {"metadata.tier": "gold"}},
    ),
    (
        "deeply_nested_mixed",
        {
            "$and": [
                {"created_at": {"$gte": "2026-01-01T00:00:00Z"}},
                {
                    "$or": [
                        {"metadata.tier": "gold"},
                        {"$not": {"metadata.tier": "silver"}},
                    ]
                },
                {"occurred_at": {"$lt": "2026-12-31T00:00:00Z"}},
            ]
        },
    ),
]


def _ast(wire: dict[str, Any]) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


# Direct-AST filters that the validator path cannot produce (a date key has no
# $exists field, and a bare-list / exact-array on a date key fails validation) —
# but which are valid compiler inputs and exercise the $exists / exact-array
# routing branches (all of which the compiler leaves UNPUSHED). Each is a
# (label, FilterNode) pair.
def _clause_node(path: tuple[str, ...], op: Op, operand: Any) -> FilterNode:
    return FilterNode(op=Op.AND, children=(FilterClause(path=path, op=op, operand=operand),))


_DIRECT_AST_FILTERS: list[tuple[str, FilterNode]] = [
    ("direct_exists_true", _clause_node(("occurred_at",), Op.EXISTS, True)),
    ("direct_exists_false", _clause_node(("occurred_at",), Op.EXISTS, False)),
    ("direct_eq_exact_array", _clause_node(("occurred_at",), Op.EQ, ("a", "b"))),
    ("direct_ne_exact_array", _clause_node(("occurred_at",), Op.NE, ("a", "b"))),
    (
        "direct_and_exists_plus_range",
        FilterNode(
            op=Op.AND,
            children=(
                FilterClause(path=("created_at",), op=Op.EXISTS, operand=True),
                FilterClause(path=("occurred_at",), op=Op.GTE, operand=datetime(2026, 1, 1, tzinfo=UTC)),
            ),
        ),
    ),
]


def _all_cases() -> list[tuple[str, FilterNode]]:
    """The full battery: validated wire filters + direct-AST filters."""
    cases = [(label, _ast(wire)) for label, wire in _WIRE_FILTERS]
    cases.extend(_DIRECT_AST_FILTERS)
    return cases


_CASES = _all_cases()
_CASE_IDS = [label for label, _ in _CASES]


# ===========================================================================
# Property 1 — SUPERSET-SAFETY: pushdown keeps ⊇ oracle accepts.
# ===========================================================================


@pytest.mark.parametrize(("label", "ast"), _CASES, ids=_CASE_IDS)
def test_pushdown_is_superset_of_oracle(label: str, ast: FilterNode) -> None:
    """The push-down candidate set must be a SUPERSET of the oracle accept set.

    The load-bearing correctness rule: the server-side push-down must never
    false-exclude a record the oracle (``compile_python``) keeps. A record the
    oracle accepts but the push-down drops would be lost before the post-filter
    ever sees it — an under-return bug. (Over-returning is always safe: the
    post-filter narrows.)
    """
    kept, _ = _route(ast)
    oracle = _oracle(ast)
    missing = _ids(oracle) - _ids(kept)
    assert not missing, (
        f"[{label}] push-down FALSE-EXCLUDED records the oracle keeps: "
        f"corpus indices {sorted(missing)} (kept={sorted(_ids(kept))}, oracle={sorted(_ids(oracle))})"
    )


# ===========================================================================
# Property 2 — POST-FILTER PARITY: final result == oracle, end-to-end.
# ===========================================================================


@pytest.mark.parametrize(("label", "ast"), _CASES, ids=_CASE_IDS)
def test_routed_result_equals_oracle(label: str, ast: FilterNode) -> None:
    """The two-stage routed result must EXACTLY equal the oracle accept set.

    By construction this follows from superset-safety (the post-filter re-checks
    the whole AST against the kept candidates, so it recovers exactly the oracle
    set as long as no candidate was false-excluded). Asserting it end-to-end
    catches a regression in EITHER the push-down (a dropped row) OR the wiring
    (a post-filter applied wrong / over-fetch too small in spirit).
    """
    _, final = _route(ast)
    oracle = _oracle(ast)
    assert _ids(final) == _ids(oracle), (
        f"[{label}] routed result {sorted(_ids(final))} != oracle {sorted(_ids(oracle))}"
    )


# ===========================================================================
# Structural superset-safety guard — every CONSUMED leaf is a superset-safe op
# on a DECLARED key. A second, independent line of defense behind the empirical
# interpreter: if the compiler ever consumes a $ne / $nin / undeclared / metadata
# leaf, this fails even if the interpreter happened to agree.
# ===========================================================================


@pytest.mark.parametrize(("label", "ast"), _CASES, ids=_CASE_IDS)
def test_consumed_keys_are_declared_only(label: str, ast: FilterNode) -> None:
    """Every consumed key is one of the two DECLARED date keys — never metadata
    or an undeclared system key."""
    consumed = compile_weaviate(ast, _PUSH_CTX).consumed_keys
    assert consumed <= frozenset(_FIELD_MAPPING), (
        f"[{label}] consumed an undeclared/metadata key: {sorted(consumed - frozenset(_FIELD_MAPPING))}"
    )


# ===========================================================================
# Cross-cutting edge cases the per-compiler tests do not pin at the routing
# level. Each asserts the concrete oracle row-set so the empirical corpus
# semantics are locked, not just the superset relation.
# ===========================================================================


def test_empty_in_matches_nothing() -> None:
    # $in over ∅ matches nothing; the routed result and oracle are both empty.
    ast = _ast({"metadata.tags": {"$in": []}})
    _, final = _route(ast)
    assert _ids(final) == set()
    assert _ids(_oracle(ast)) == set()


def test_empty_nin_matches_everything() -> None:
    # $nin over ∅ matches every record (including those without the metadata key).
    ast = _ast({"metadata.tags": {"$nin": []}})
    _, final = _route(ast)
    assert _ids(final) == set(range(len(CORPUS)))


def test_exact_array_distinct_from_in() -> None:
    # An exact-array $eq (bare list) matches ONLY a record whose tags array equals
    # the list exactly; the $in membership form matches any record sharing an
    # element. The two must produce different row-sets here (record 0's tags are
    # exactly ["urgent","backend"]; records 1/3/5 share an element).
    exact = _oracle(_ast({"metadata.tags": ["urgent", "backend"]}))
    member = _oracle(_ast({"metadata.tags": {"$in": ["urgent", "backend"]}}))
    assert _ids(exact) == {0}
    assert _ids(exact) < _ids(member)  # exact is a strict subset of membership


def test_nor_desugars_to_not_or() -> None:
    # $nor([A, B]) ≡ $not($or(A, B)) — a record matches iff NEITHER A nor B holds.
    # Negations are unpushable, so the whole filter post-filters; routed == oracle.
    ast = _ast({"$nor": [{"metadata.tier": "bronze"}, {"source_name": "slack"}]})
    kept, final = _route(ast)
    # Nothing pushable → the candidate pool is the whole corpus.
    assert _ids(kept) == set(range(len(CORPUS)))
    assert _ids(final) == _ids(_oracle(ast))


def test_deeply_nested_and_or_not_mixing_pushable_and_unpushable() -> None:
    # A deep $and over (a pushable range, an $or mixing metadata + a $not, another
    # pushable range). The $or is wholly unpushable (a metadata disjunct), so only
    # the two date ranges push; the post-filter recovers the exact oracle set.
    ast = _ast(
        {
            "$and": [
                {"created_at": {"$gte": "2026-01-01T00:00:00Z"}},
                {"$or": [{"metadata.tier": "gold"}, {"$not": {"metadata.tier": "silver"}}]},
                {"occurred_at": {"$lt": "2026-12-31T00:00:00Z"}},
            ]
        }
    )
    consumed = compile_weaviate(ast, _PUSH_CTX).consumed_keys
    # Both date ranges push; the $or's metadata leaf does not.
    assert consumed == frozenset({"created_at", "occurred_at"})
    _, final = _route(ast)
    assert _ids(final) == _ids(_oracle(ast))


def test_or_with_unpushable_disjunct_keeps_whole_corpus() -> None:
    # An $or with one metadata disjunct is ALL-OR-NOTHING unpushable: the push-down
    # emits no constraint (keeps everything) and the post-filter does all the work.
    # This is the case where a NAIVE pushdown (pushing only the date disjunct) would
    # NARROW the union and false-exclude a row matching only the metadata disjunct —
    # the superset-safety test above guards exactly this, but pin the candidate pool.
    ast = _ast({"$or": [{"occurred_at": "2026-01-10T00:00:00Z"}, {"metadata.tier": "bronze"}]})
    pushed = compile_weaviate(ast, _PUSH_CTX).predicate
    assert pushed is None  # wholly unpushable
    kept, final = _route(ast)
    assert _ids(kept) == set(range(len(CORPUS)))
    assert _ids(final) == _ids(_oracle(ast))


def test_null_and_exists_are_not_pushed_and_round_trip() -> None:
    # {k: null}, $exists:true, and $exists:false hinge on null/absent resolution.
    # The post-filter oracle treats a system key as ALWAYS present (an absent key
    # resolves to None, still "present-with-null" for $exists), so any server-side
    # is_none() push would DIVERGE (a $exists:true → is_none(False) push would
    # false-exclude the null-property rows the oracle keeps). The compiler therefore
    # leaves them UNPUSHED (predicate None, nothing consumed) and the post-filter
    # alone honors them — routed result still equals the oracle.
    for label, ast in (
        ("null", _ast({"occurred_at": None})),
        ("exists_true", _clause_node(("occurred_at",), Op.EXISTS, True)),
        ("exists_false", _clause_node(("occurred_at",), Op.EXISTS, False)),
    ):
        compiled = compile_weaviate(ast, _PUSH_CTX)
        assert compiled.predicate is None, f"[{label}] expected unpushable (None predicate)"
        assert compiled.consumed_keys == frozenset(), f"[{label}] expected nothing consumed"
        _, final = _route(ast)
        assert _ids(final) == _ids(_oracle(ast)), f"[{label}] routed != oracle"


def test_date_in_pushdown_is_pushed_and_keeps_matching_row() -> None:
    # A $in on a declared date key IS pushable membership — it lowers to an
    # any_of([equal(x)…]) Or-of-Equals tree (exact scalar $in semantics). Assert
    # the path is non-vacuous: the date key is CONSUMED (it pushed), the kept set is a
    # non-empty SUPERSET of the oracle that includes the record whose date equals
    # an operand, and the routed result equals the oracle. Record 0
    # (occurred_at=2026-01-10) and record 1 (occurred_at=2026-06-01) each equal an
    # operand and must survive.
    ast = _ast({"occurred_at": {"$in": ["2026-01-10T00:00:00Z", "2026-06-01T00:00:00Z"]}})
    compiled = compile_weaviate(ast, _PUSH_CTX)
    assert compiled.predicate is not None, "date $in must push a predicate"
    assert "occurred_at" in compiled.consumed_keys, "date $in must consume the date key"
    kept, final = _route(ast)
    oracle = _oracle(ast)
    assert oracle, "fixture sanity: the date $in must match at least one record"
    assert _ids(oracle) <= _ids(kept), "pushdown false-excluded a date-$in match"
    assert {0, 1} <= _ids(kept), "expected the operand-equal records to be kept candidates"
    assert _ids(final) == _ids(oracle)
