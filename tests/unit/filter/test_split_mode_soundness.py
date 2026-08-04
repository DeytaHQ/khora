"""Unit tests for the split-mode pushdown gate the SQL-ish compilers share.

``@internal``. :mod:`khora.filter.compilers._split` is the one place the
postgres / lance / surrealdb compilers agree on what ``on_unsupported="split"``
means, and its three rules are what every caller's post-filter contract rests
on:

1. **AND distributes; OR / NOT are all-or-nothing.** The match-all placeholder
   an unconsumable leaf emits is superset-safe only in positive position — under
   a negation it inverts to match-NOTHING, dropping rows a post-filter can never
   add back. So an ``OR`` / ``NOT`` is pushed only when its ENTIRE subtree is
   consumable.
2. **``consumed_keys`` is per-occurrence honest.** A dotted path pushed in one
   branch and deferred in another is NOT reported consumed, or a caller
   differencing ``leaf_keys - consumed_keys`` would skip the occurrence that
   never reached the backend.
3. **``field_mapping is None`` is the identity mapping, not an empty
   whitelist.** Every chunk-tier context passes ``None``; reading it as "nothing
   is declared" would defer every system-key pushdown on the hot recall path.

**What the conformance corpus does NOT cover, and why the hand-written cases
below are load-bearing.** Measured over all 209 corpus ASTs against the real
chunk-tier contexts: the OR/NOT gate changes the outcome for 1 of them, and the
key-in-both-positions shape of rule 2 has ZERO corpus coverage. The corpus is
excellent at operator semantics and useless at these two shapes, so
:func:`test_negated_or_over_an_unconsumable_leaf_defers_the_whole_node` and
:func:`test_key_pushed_in_one_branch_and_deferred_in_another_is_not_consumed`
are explicit and must not be traded for corpus-driven assertions.

**The corpus IS the right tool for the two structural invariants**, which is
what :func:`test_clause_consumable_mirrors_what_emission_did` (the per-leaf
mirror) and :func:`test_raise_mode_raises_exactly_when_a_leaf_is_unconsumable`
(the raise-mode biconditional) use it for. Those encode relationships rather
than expected outputs, so a new corpus case strengthens them without edits.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.elements import ColumnElement

from khora.filter import RecallFilter
from khora.filter.ast import FilterClause, FilterNode, canonical_hash, parse_to_ast
from khora.filter.compilers import lance as lance_module
from khora.filter.compilers import postgres as postgres_module
from khora.filter.compilers import surrealdb as surrealdb_module
from khora.filter.compilers._split import consumed_subtree
from khora.filter.compilers.lance import compile_lance
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.compilers.surrealdb import compile_surrealdb
from khora.filter.conformance import (
    _resolve_ast,
    f_array_cases,
    f_coerce_cases,
    f_dates_cases,
    f_dotkey_cases,
    f_exists_cases,
    f_impossible_cases,
    f_logic_cases,
    f_nullval_cases,
    f_objeq_cases,
    f_op_cases,
    f_polarity_cases,
    f_sel_cases,
    f_sugar_cases,
    f_unsup_cases,
)
from khora.filter.context import CompileContext, CompileError, SchemaCapabilities
from khora.filter.execute import build_compile_context, filter_leaf_keys
from khora.filter.model import SYSTEM_KEYS, Op, RecallFilterUnsupportedError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Corpus + context fixtures.
# ---------------------------------------------------------------------------

# The same 14 family generators the conformance catalog enumerates, listed here
# rather than imported as a collection so a new family shows up as an explicit
# edit in both places.
_FAMILY_GENERATORS = (
    f_op_cases,
    f_sugar_cases,
    f_impossible_cases,
    f_exists_cases,
    f_coerce_cases,
    f_polarity_cases,
    f_objeq_cases,
    f_dotkey_cases,
    f_array_cases,
    f_logic_cases,
    f_dates_cases,
    f_nullval_cases,
    f_sel_cases,
    f_unsup_cases,
)

_D1 = "2026-01-01T00:00:00+00:00"
_D2 = "2026-02-01T00:00:00+00:00"
_D3 = "2026-03-01T00:00:00+00:00"

_JSON1_ON = SchemaCapabilities(sqlite_json1=True)
_JSON1_OFF = SchemaCapabilities(sqlite_json1=False)

# The nine system keys a ``documents`` row backs, restated independently of the
# store modules (a test that imported the constant under test asserts nothing).
_DOCUMENT_KEYS = frozenset(SYSTEM_KEYS) - {"occurred_at"}

# The SurrealDB chunk-tier whitelist, mirroring ``storage/temporal/surrealdb.py``.
_SURREAL_CHUNK_MAPPING = {"occurred_at": "occurred_at", "created_at": "created_at", "metadata": "metadata_"}

# The date-typed system keys. The validator gives them a range grammar and no
# ``$exists``, so operator-shape helpers have to branch on this.
_DATE_KEYS = frozenset({"created_at", "source_timestamp", "occurred_at"})


def _corpus() -> list[tuple[str, FilterNode]]:
    """Every conformance case, lowered through the real validator + ``parse_to_ast``."""
    cases: list[tuple[str, FilterNode]] = []
    for generator in _FAMILY_GENERATORS:
        for index, case in enumerate(generator()):
            cases.append((f"{generator.__name__}[{index}]", _resolve_ast(case.filter)))
    return cases


def _distinct_corpus_leaves() -> list[FilterClause]:
    """Every structurally distinct leaf clause anywhere in the corpus."""
    leaves: list[FilterClause] = []
    seen: set[tuple[Any, ...]] = set()
    for _name, ast in _corpus():
        for leaf in _iter_leaves(ast):
            key = (leaf.path, leaf.op, repr(leaf.operand))
            if key not in seen:
                seen.add(key)
                leaves.append(leaf)
    return leaves


def _iter_leaves(node: FilterNode | FilterClause) -> Iterator[FilterClause]:
    """Yield every leaf OCCURRENCE, duplicates preserved."""
    if isinstance(node, FilterClause):
        yield node
        return
    for child in node.children:
        yield from _iter_leaves(child)


def _ast(wire: dict) -> FilterNode:
    return parse_to_ast(RecallFilter.model_validate(wire))


def _norm(sql: str) -> str:
    return " ".join(sql.split())


def _render(predicate: Any) -> str:
    """The emitted fragment as a string, rendering SQLAlchemy elements inline.

    ``literal_binds`` is load-bearing, not cosmetic: SQLAlchemy folds
    ``or_(x, true())`` to ``true`` during compilation, so whether the gate fired
    is only observable after rendering. ``str(element)`` shows the
    un-short-circuited tree and would hide it.
    """
    if isinstance(predicate, ColumnElement):
        return _norm(str(predicate.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})))
    return _norm(str(predicate))


# ``(id, compile fn, builder module, context)`` for every compiler/context pair
# the gate runs under. Both tiers are present on purpose: the chunk tier is where
# the gate is LIVE on the embedded read path, the documents tier is where the
# whitelist actually withholds keys.
def _split_targets() -> tuple[tuple[str, Callable, Any, CompileContext], ...]:
    return (
        (
            "lance/chunk",
            compile_lance,
            lance_module._Builder,
            CompileContext(backend_target="khora_chunks", on_unsupported="split", schema_capabilities=_JSON1_ON),
        ),
        (
            "lance/documents",
            compile_lance,
            lance_module._Builder,
            CompileContext(
                backend_target="documents",
                field_mapping={k: k for k in _DOCUMENT_KEYS - {"created_at", "source_timestamp"}}
                | {"metadata": "metadata"},
                on_unsupported="split",
                schema_capabilities=_JSON1_ON,
            ),
        ),
        (
            "postgres/chunk",
            compile_postgres,
            postgres_module._Builder,
            CompileContext(backend_target="khora_chunks", on_unsupported="split"),
        ),
        (
            "postgres/documents",
            compile_postgres,
            postgres_module._Builder,
            CompileContext(
                backend_target="documents",
                field_mapping={k: k for k in _DOCUMENT_KEYS} | {"metadata": "metadata"},
                on_unsupported="split",
            ),
        ),
        (
            "surreal/chunk",
            compile_surrealdb,
            surrealdb_module._Builder,
            CompileContext(
                backend_target="temporal_chunk", field_mapping=_SURREAL_CHUNK_MAPPING, on_unsupported="split"
            ),
        ),
        (
            "surreal/documents",
            compile_surrealdb,
            surrealdb_module._Builder,
            CompileContext(
                backend_target="document",
                field_mapping={k: k for k in _DOCUMENT_KEYS} | {"metadata": "metadata_"},
                on_unsupported="split",
            ),
        ),
    )


# The two chunk-tier contexts that ship ``on_unsupported="raise"``, copied from
# their production call sites (``storage/temporal/pgvector.py`` and
# ``storage/temporal/surrealdb.py``). The remaining chunk-tier stores —
# sqlite_lance and weaviate — are SPLIT and live.
def _raise_targets() -> tuple[tuple[str, Callable, Any, CompileContext], ...]:
    return (
        (
            "postgres/chunk",
            compile_postgres,
            postgres_module._Builder,
            build_compile_context("khora_chunks", on_unsupported="raise"),
        ),
        (
            "surreal/chunk",
            compile_surrealdb,
            surrealdb_module._Builder,
            CompileContext(
                backend_target="temporal_chunk", field_mapping=_SURREAL_CHUNK_MAPPING, on_unsupported="raise"
            ),
        ),
    )


_SPLIT_IDS = [target[0] for target in _split_targets()]
_RAISE_IDS = [target[0] for target in _raise_targets()]

# The targets whose gate cases below can be driven from an ORDINARY wire form —
# a system key the context withholds, or a metadata shape the backend cannot
# express. ``postgres/chunk`` is absent because neither exists there:
# ``field_mapping=None`` declares every system key, every dotted metadata path is
# pushable, and every bare-``metadata`` wire form folds to ``$eq``
# (:func:`test_every_bare_metadata_wire_form_folds_to_eq`).
#
# That is NOT the same as "no unconsumable leaf exists" for that context. One
# does — any path that is neither a system key nor ``metadata.*`` — and it is
# reachable, so the guard rejecting it is live rather than dead code. It is
# covered by :func:`test_an_undeclared_path_never_reaches_a_column_token` and by
# the raise-mode biconditional's non-vacuity branch, both of which run against
# the shipped ``postgres/chunk`` context.
_GATED_TARGETS = tuple(target for target in _split_targets() if target[0] != "postgres/chunk")
_GATED_IDS = [target[0] for target in _GATED_TARGETS]


# ===========================================================================
# Rule 1 — AND distributes; OR / NOT are all-or-nothing.
# ===========================================================================


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _GATED_TARGETS, ids=_GATED_IDS)
def test_negated_or_over_an_unconsumable_leaf_defers_the_whole_node(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """``$not`` over an ``$or`` holding an unconsumable leaf defers the WHOLE node.

    The shape the corpus does not cover and the reason the gate exists. Without
    it the unconsumable disjunct emits a match-all, the enclosing ``$or`` becomes
    match-all, and the ``$not`` inverts it to match-NOTHING — the query returns
    ZERO rows while ``consumed_keys`` still names the consumable sibling, so the
    caller post-filters a set the backend already emptied. A post-filter only
    narrows, so those rows are gone.

    Asserted on the RENDERED fragment (see :func:`_render`) and stated as "the
    bare match-all", not "does not contain ``false``": an emitted ``NOT (...)``
    over a placeholder is equally wrong even where it does not literally spell
    ``false``.
    """
    consumable_leaf, unconsumable_leaf = _leaf_pair(name)
    compiled = compiler(_ast({"$not": {"$or": [consumable_leaf, unconsumable_leaf]}}), ctx)
    sql = _render(compiled.predicate)

    assert sql.strip() in {"true", "1"}, f"expected the bare match-all placeholder, got: {sql}"
    assert compiled.consumed_keys == frozenset()

    # CONTROLS — without these the assertion above is satisfied by a compiler
    # that pushes nothing at all on this context.
    consumable_key = next(iter(filter_leaf_keys(_ast(consumable_leaf))))
    assert compiler(_ast(consumable_leaf), ctx).consumed_keys == frozenset({consumable_key})
    assert compiler(_ast(unconsumable_leaf), ctx).consumed_keys == frozenset()


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _GATED_TARGETS, ids=_GATED_IDS)
def test_bare_or_over_an_unconsumable_leaf_also_defers_whole(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """An ``$or`` in POSITIVE position defers too — the readable half of a real invariant.

    Worth its own case because ``consumed_keys`` cannot tell the two apart: the
    deferred-path subtraction reports the same empty set whether or not the OR
    branch of the gate ran, since it derives from ``consumed_subtree`` rather
    than from emission. Only the emitted fragment differs — with the gate it is
    the bare placeholder, without it the compilers emit
    ``<pushed leaf> OR <placeholder>``.

    This is NOT merely emission hygiene. ``A OR <match-all>`` is indeed
    equivalent to ``<match-all>``, so no rows move — but the pruned slice says
    NOTHING was pushed while emission pushed a real leaf, which breaks the
    ``canonical_hash`` contract. That consequence is pinned separately and
    formatting-independently in
    :func:`test_equal_hash_implies_equal_emitted_predicate`; this case exists for
    the readable failure message, and that one for the property that will not rot
    when the emitted SQL is reformatted.
    """
    consumable_leaf, unconsumable_leaf = _leaf_pair(name)
    compiled = compiler(_ast({"$or": [consumable_leaf, unconsumable_leaf]}), ctx)
    sql = _render(compiled.predicate)

    assert sql.strip() in {"true", "1"}, f"expected the bare placeholder, got: {sql}"
    assert " or " not in sql.lower()
    assert compiled.consumed_keys == frozenset()


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _GATED_TARGETS, ids=_GATED_IDS)
def test_bare_not_over_an_unconsumable_leaf_defers(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """A ``$not`` directly over an unconsumable LEAF — the NOT branch on its own.

    Separated from the ``$not($or)`` case deliberately. ``OR`` and ``NOT`` are
    different branches of :meth:`compile_node`, and ``$not($or)`` exercises both
    at once, so a bug confined to the plain ``NOT``-over-leaf path can hide
    behind the OR branch handling it first. This is the shape with no ``$or``
    anywhere in it.
    """
    _consumable_leaf, unconsumable_leaf = _leaf_pair(name)
    compiled = compiler(_ast({"$not": unconsumable_leaf}), ctx)
    sql = _render(compiled.predicate)

    assert sql.strip() in {"true", "1"}, f"expected the bare match-all placeholder, got: {sql}"
    assert "not" not in sql.lower()
    assert "!(" not in sql
    assert compiled.consumed_keys == frozenset()


# ``(id, shape builder)`` — the three logical shapes the gate can defer, built
# from two leaves. Used for the over-deferral controls below.
_LOGICAL_SHAPES: tuple[tuple[str, Callable[[dict, dict], dict]], ...] = (
    ("or", lambda a, b: {"$or": [a, b]}),
    ("not", lambda a, _b: {"$not": a}),
    ("not_or", lambda a, b: {"$not": {"$or": [a, b]}}),
)


# NOTE this one runs over ALL split targets, not just the gated ones: it needs
# no unconsumable leaf, only two consumable ones. ``postgres/chunk`` is excluded
# elsewhere for lack of an ordinary unconsumable leaf, but it is precisely where
# an over-deferral regression would hurt most — it is the pgvector recall path.
@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _split_targets(), ids=_SPLIT_IDS)
@pytest.mark.parametrize(("shape", "build"), _LOGICAL_SHAPES, ids=[case[0] for case in _LOGICAL_SHAPES])
def test_a_fully_consumable_or_not_is_still_pushed(
    shape: str,
    build: Callable[[dict, dict], dict],
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """OVER-DEFERRAL CONTROL — the gate must defer correctly, not defer everything.

    Every other gate case in this module asserts that something IS deferred, so
    all of them pass against a compiler that simply never pushes an ``$or`` /
    ``$not`` at all. Measured, not assumed: forcing both gate branches to defer
    unconditionally leaves every dedicated gate test in this file GREEN, and is
    caught only incidentally by
    :func:`test_the_two_accounting_invariants_over_the_corpus`, whose failure
    message says nothing about over-deferral.

    So this is the other direction, stated per shape: with EVERY leaf consumable,
    the node is pushed and both keys are consumed. Losing this costs no
    correctness — an under-pushed filter still returns the right rows — which is
    exactly why nothing else would notice it, and why a silent collapse of all
    disjunctive pushdown needs its own guard.
    """
    first, second = _consumable_pair(name)
    keys = filter_leaf_keys(_ast(first)) | filter_leaf_keys(_ast(second))

    compiled = compiler(_ast(build(first, second)), ctx)
    sql = _render(compiled.predicate)

    assert sql.strip() not in {"true", "1"}, f"{shape} was deferred despite being fully consumable: {sql}"
    expected = filter_leaf_keys(_ast(first)) if shape == "not" else keys
    assert compiled.consumed_keys == expected
    # The negation / disjunction really is in the emitted fragment, rather than
    # the leaves having been flattened into a bare conjunction.
    assert ("!(" in sql or "not" in sql.lower()) if shape.startswith("not") else (" or " in sql.lower())


def _consumable_pair(name: str) -> tuple[dict, dict]:
    """Two leaves on DIFFERENT keys that the given target can both push."""
    if name == "surreal/chunk":
        # Only occurred_at / created_at / metadata are declared here.
        return {"created_at": {"$gte": _D1}}, {"occurred_at": {"$gte": _D1}}
    if name == "lance/chunk":
        return {"source_type": "email"}, {"source_name": "linear"}
    # Both documents tiers declare these two.
    return {"title": "x"}, {"source_type": "email"}


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _GATED_TARGETS, ids=_GATED_IDS)
def test_and_still_distributes_over_an_unconsumable_sibling(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """``AND`` is NOT all-or-nothing — the consumable sibling still narrows.

    The other half of rule 1, and the one that would silently disappear if the
    gate were applied uniformly to every logical node. Losing it costs no
    correctness (an under-pushed filter is still a superset) but it would move
    every conjunctive pushdown to the post-filter, which is why it is pinned
    rather than left implied by the OR/NOT case above.
    """
    consumable_leaf, unconsumable_leaf = _leaf_pair(name)
    consumable_key = next(iter(filter_leaf_keys(_ast(consumable_leaf))))

    compiled = compiler(_ast({"$and": [consumable_leaf, unconsumable_leaf]}), ctx)
    assert compiled.consumed_keys == frozenset({consumable_key})
    assert consumable_key in _render(compiled.predicate)


def _leaf_pair(name: str) -> tuple[dict, dict]:
    """A ``(consumable, unconsumable)`` leaf pair for the given target.

    Picked per target because "unconsumable" has a different cause on each: an
    undeclared system key where a whitelist withholds one, and a ``$date``
    metadata compare on the chunk-tier contexts that declare everything.
    """
    if name == "lance/chunk":
        # ``field_mapping=None`` declares every system key, so the unconsumable
        # leaf has to be a shape SQLite cannot express.
        return {"source_type": "email"}, {"metadata.when": {"$date": _D3}}
    if name == "surreal/chunk":
        # Only occurred_at / created_at / metadata are declared here.
        return {"created_at": {"$gte": _D1}}, {"source_type": "email"}
    # Both documents tiers withhold ``occurred_at``.
    return {"title": "x"}, {"occurred_at": {"$gt": _D1}}


# ===========================================================================
# Rule 2 — ``consumed_keys`` is per-occurrence honest.
# ===========================================================================


def test_key_pushed_in_one_branch_and_deferred_in_another_is_not_consumed() -> None:
    """The exact repro, on the LIVE embedded chunk path.

    ``storage/temporal/sqlite_lance.py`` compiles with ``on_unsupported="split"``,
    so this shape was reachable in production, not merely theoretical. Before the
    fix this AST compiled to ``(coalesce(khora_chunks.created_at >= ?, 0) AND 1)``
    with ``consumed_keys == {"created_at"}`` — the ``$not`` half deferred to the
    placeholder while the report claimed the key had been fully pushed, so a
    caller differencing ``leaf_keys - consumed_keys`` never re-checked it and the
    query returned rows the filter excludes.

    The emitted SQL is UNCHANGED by the fix and is asserted here to say so: the
    defect was in the report, not the predicate. That is also why a test that
    only checked ``consumed_keys`` would be weak — it would pass against a
    compiler that had simply stopped pushing the conjunctive leaf.
    """
    ctx = CompileContext(backend_target="khora_chunks", on_unsupported="split", schema_capabilities=_JSON1_ON)
    ast = _ast(
        {
            "$and": [
                {"created_at": {"$gte": _D1}},
                {"$not": {"$or": [{"created_at": {"$lt": _D2}}, {"metadata.when": {"$date": _D3}}]}},
            ]
        }
    )
    compiled = compile_lance(ast, ctx)

    # The conjunctive occurrence still pushes; the ``$not`` is the placeholder.
    assert _render(compiled.predicate) == "(coalesce(khora_chunks.created_at >= ?, 0) AND 1)"
    # ...and ``created_at`` is nonetheless a residual, so the caller re-checks it.
    assert compiled.consumed_keys == frozenset()
    assert "created_at" in filter_leaf_keys(ast) - compiled.consumed_keys

    # CONTROL — the same key WITHOUT the deferred second occurrence is consumed,
    # so the exclusion above is per-occurrence accounting, not the key being
    # unpushable on this context.
    assert compile_lance(_ast({"created_at": {"$gte": _D1}}), ctx).consumed_keys == frozenset({"created_at"})


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _GATED_TARGETS, ids=_GATED_IDS)
def test_repeated_path_accounting_holds_on_every_compiler(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """The same per-occurrence rule, across all three compilers and both tiers.

    The case above pins the production repro verbatim; this one pins that the
    rule is shared rather than a property of ``compile_lance``.
    """
    consumable_leaf, unconsumable_leaf = _leaf_pair(name)
    key = next(iter(filter_leaf_keys(_ast(consumable_leaf))))

    compiled = compiler(
        _ast({"$and": [consumable_leaf, {"$not": {"$or": [consumable_leaf, unconsumable_leaf]}}]}),
        ctx,
    )
    assert key not in compiled.consumed_keys
    assert compiler(_ast(consumable_leaf), ctx).consumed_keys == frozenset({key})


# ===========================================================================
# The two accounting invariants — DELIBERATELY DIFFERENT ARTIFACTS.
# ===========================================================================


def _oracle_split(
    node: FilterNode | FilterClause,
    consumable: Callable[[FilterClause], bool],
) -> tuple[list[FilterClause], list[FilterNode | FilterClause]]:
    """An independent re-derivation of the gate: ``(kept leaves, deferred nodes)``.

    Written straight from the three rules rather than by calling into
    :mod:`khora.filter.compilers._split`, so the invariants below compare the
    shipped walk against a second implementation instead of against itself.
    """
    if isinstance(node, FilterClause):
        return ([node], []) if consumable(node) else ([], [node])
    if node.op == Op.AND:
        kept: list[FilterClause] = []
        deferred: list[FilterNode | FilterClause] = []
        for child in node.children:
            child_kept, child_deferred = _oracle_split(child, consumable)
            kept.extend(child_kept)
            deferred.extend(child_deferred)
        return kept, deferred
    # OR / NOT — all-or-nothing, never recursed into.
    if all(consumable(leaf) for leaf in _iter_leaves(node)):
        return list(_iter_leaves(node)), []
    return [], [node]


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _split_targets(), ids=_SPLIT_IDS)
def test_the_two_accounting_invariants_over_the_corpus(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """Two SEPARATE invariants over every corpus AST. They are not the same set.

    (a) The RAW emission accumulator — what ``compile_clause`` recorded, before
        the deferred-path subtraction — equals the dotted paths of the leaves in
        the pruned slice. This is the per-occurrence, structural artifact, and it
        is what the cache key hashes.

    (b) ``consumed_keys`` equals that set MINUS the paths under every deferred
        node. This is the per-path CONSERVATIVE artifact, and it is what the
        caller's post-filter contract reads.

    **The tempting single assertion ``leaf_keys(slice) == consumed_keys`` is
    wrong and must not replace these.** It is SATISFIED BY the very defect this
    change fixes — a key pushed in one branch and deferred in another appears in
    the slice AND was reported consumed, so the equality held while the key was
    only half-enforced — and it is FAILED BY the correct behaviour. The two sets
    are different on purpose; collapsing them re-opens the gap.
    """
    for case, ast in _corpus():
        builder = builder_cls(ctx=ctx, consumed=set())
        try:
            compiled = compiler(ast, ctx)
        except CompileError:
            # Injection guard, orthogonal to consumability — see the raise-mode
            # biconditional below for why these are excluded rather than accepted.
            continue

        kept, deferred = _oracle_split(ast, builder._clause_consumable)

        # (a) the raw accumulator == the pruned slice's leaf paths, per occurrence.
        raw = _replay_raw_consumed(compiler, builder_cls, ctx, ast)
        assert raw == {_dotted(leaf) for leaf in kept}, f"{name} {case}: raw accumulator diverged from the slice"
        assert raw == set(filter_leaf_keys(consumed_subtree(ast, builder._clause_consumable))), (
            f"{name} {case}: shipped slice diverged from the oracle slice"
        )

        # (b) consumed_keys == (a) minus every path under a deferred node.
        deferred_paths = {_dotted(leaf) for node in deferred for leaf in _iter_leaves(node)}
        assert compiled.consumed_keys == frozenset(raw - deferred_paths), (
            f"{name} {case}: consumed_keys is not the slice minus the deferred paths"
        )


def _replay_raw_consumed(
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
    ast: FilterNode,
) -> set[str]:
    """The emission accumulator BEFORE the deferred-path subtraction.

    ``CompiledFilter`` only exposes the post-subtraction set, so the raw one is
    read by driving the builder the way ``compiler`` does. That is white-box, and
    intentionally so: invariant (a) is a statement about the emission walk, which
    is not otherwise observable.
    """
    consumed: set[str] = set()
    builder_cls(ctx=ctx, consumed=consumed).compile_node(ast)
    return consumed


def _dotted(leaf: FilterClause) -> str:
    return ".".join(leaf.path)


def test_the_slice_and_consumed_keys_are_deliberately_different_sets() -> None:
    """A worked case where the two artifacts DISAGREE — the point of keeping both.

    Without this, "the slice and ``consumed_keys`` are different artifacts" is
    only a claim in a docstring. Here ``created_at`` IS in the consumed slice
    (its conjunctive occurrence really was pushed, and the cache key must reflect
    that) and is NOT in ``consumed_keys`` (its other occurrence was deferred, so
    the post-filter must still enforce it).
    """
    ctx = CompileContext(backend_target="khora_chunks", on_unsupported="split", schema_capabilities=_JSON1_ON)
    ast = _ast(
        {
            "$and": [
                {"created_at": {"$gte": _D1}},
                {"$not": {"$or": [{"created_at": {"$lt": _D2}}, {"metadata.when": {"$date": _D3}}]}},
            ]
        }
    )
    builder = lance_module._Builder(ctx=ctx, consumed=set())
    slice_keys = filter_leaf_keys(consumed_subtree(ast, builder._clause_consumable))

    assert "created_at" in slice_keys
    assert "created_at" not in compile_lance(ast, ctx).consumed_keys


# ===========================================================================
# Rule 3 — ``field_mapping is None`` is identity, NOT an empty whitelist.
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "compiler", "ctx"),
    (
        ("postgres", compile_postgres, CompileContext(backend_target="khora_chunks", on_unsupported="split")),
        (
            "lance",
            compile_lance,
            CompileContext(backend_target="khora_chunks", on_unsupported="split", schema_capabilities=_JSON1_ON),
        ),
    ),
    ids=["postgres", "lance"],
)
def test_identity_field_mapping_still_pushes_every_system_key(
    name: str,
    compiler: Callable,
    ctx: CompileContext,
) -> None:
    """THE OUTAGE GUARD for the whitelist gate.

    Every chunk-tier postgres / lance context passes ``field_mapping=None``. If
    the gate had been written as a plain membership test, ``None`` would fold to
    "nothing is declared" and EVERY system-key pushdown on the default recall
    path would move to the post-filter — a silent, total pushdown outage that no
    correctness test would catch, because an under-pushed filter still returns
    the right rows (just after fetching far more of them).

    So: with the identity mapping, all ten system keys must still push and still
    be consumed. Parametrized over the operator too, since the gate sits on the
    leaf dispatch and a shape-specific regression would slip past a single ``$eq``.
    """
    assert ctx.field_mapping is None, "the whole point of this case is the identity mapping"

    for key in sorted(SYSTEM_KEYS):
        for wire in _system_key_wires(key):
            compiled = compiler(_ast(wire), ctx)
            assert compiled.consumed_keys == frozenset({key}), f"{name}: {key} not pushed for {wire}"
            # The column must appear in the fragment too, so "consumed" is not
            # satisfied by a match-all that consumed the key without narrowing.
            # ``$exists`` is the documented exception: every system key is a real
            # NOT NULL column, so existence is statically true and BOTH compilers
            # fold it to the placeholder while still (correctly) consuming the key.
            assert key in _render(compiled.predicate), f"{name}: {key} absent from the fragment for {wire}"

    # ``$exists`` is only in the validator's grammar for the string-typed keys.
    for key in sorted(SYSTEM_KEYS - _DATE_KEYS):
        exists = compiler(_ast({key: {"$exists": True}}), ctx)
        assert exists.consumed_keys == frozenset({key}), f"{name}: {key} $exists not consumed"
        assert _render(exists.predicate).strip() in {"true", "(1)"}


def _system_key_wires(key: str) -> tuple[dict, ...]:
    """Narrowing operator shapes per system key, with operands the validator accepts.

    ``$exists`` is handled separately by the caller — it is statically true on a
    NOT NULL column and folds to the placeholder, so it cannot carry the
    "column appears in the fragment" half of the assertion.
    """
    if key in _DATE_KEYS:
        return ({key: {"$gte": _D1}}, {key: {"$lt": _D2}})
    return ({key: "x"}, {key: {"$in": ["x", "y"]}})


def test_a_declared_whitelist_does_withhold_an_undeclared_key() -> None:
    """The other side of rule 3 — a NON-``None`` mapping really is a whitelist.

    Paired with the case above so neither reading can be satisfied vacuously: the
    identity mapping must whitelist nothing away, and a real mapping must.
    """
    ctx = CompileContext(
        backend_target="documents",
        field_mapping={"title": "title", "metadata": "metadata"},
        on_unsupported="split",
    )
    assert compile_postgres(_ast({"title": "x"}), ctx).consumed_keys == frozenset({"title"})
    assert compile_postgres(_ast({"source_type": "email"}), ctx).consumed_keys == frozenset()


def test_surrealdb_reads_an_absent_mapping_as_declaring_nothing() -> None:
    """SurrealDB is DELIBERATELY asymmetric with postgres / lance here.

    SQL fails loud on a missing column — the statement errors. On a SCHEMAFULL
    SurrealDB table a missing field reads ``NONE`` and SurrealQL's total-false
    absent-compare silently drops every row, so this compiler must fail CLOSED:
    ``field_mapping=None`` declares nothing rather than everything. Pinned so the
    asymmetry is not "unified away" as an inconsistency.
    """
    ctx = CompileContext(backend_target="temporal_chunk", on_unsupported="split")
    assert ctx.field_mapping is None
    compiled = compile_surrealdb(_ast({"source_type": "email"}), ctx)
    assert compiled.consumed_keys == frozenset()
    assert compiled.predicate.strip() == "(true)"


# ===========================================================================
# The cache key — ``canonical_hash`` over the consumed slice.
# ===========================================================================


def test_two_different_splits_of_one_filter_hash_differently() -> None:
    """The same AST that splits differently must NOT share a plan-identity hash.

    Two contexts differing ONLY in ``sqlite_json1``: with JSON1 the metadata leaf
    is pushed, without it the leaf is deferred and the backend receives a
    match-all. Those are different queries, so the hash identifying the emitted
    plan must distinguish them. Hashing the whole AST — what the compilers did
    before — gave them the same value.
    """
    ast = _ast({"metadata.tier": "gold"})
    on = CompileContext(backend_target="khora_chunks", on_unsupported="split", schema_capabilities=_JSON1_ON)
    off = dataclasses.replace(on, schema_capabilities=_JSON1_OFF)

    pushed = compile_lance(ast, on)
    deferred = compile_lance(ast, off)

    assert pushed.consumed_keys == frozenset({"metadata.tier"})
    assert deferred.consumed_keys == frozenset()
    assert pushed.consumed_slice_hash != deferred.consumed_slice_hash

    # The pushed side consumed everything, so its slice IS the whole AST.
    assert pushed.consumed_slice_hash == canonical_hash(ast)
    # The deferred side pushed nothing, so its slice is the match-everything AST
    # — and any other fully-deferred filter shares that hash, correctly, because
    # they send the backend the identical query. That collision is also exactly
    # why this value is NOT a result-cache key: these two keep different rows.
    assert deferred.consumed_slice_hash == canonical_hash(FilterNode(op=Op.AND, children=()))
    assert deferred.consumed_slice_hash == compile_lance(_ast({"metadata.other": "x"}), off).consumed_slice_hash


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _GATED_TARGETS, ids=_GATED_IDS)
def test_equal_hash_implies_equal_emitted_predicate(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """The plan-identity contract: equal hash ⟹ equal emitted predicate.

    ``CompiledFilter.consumed_slice_hash`` identifies *the predicate this compiler
    emitted*, so anything keying a prepared-statement or plan cache on it needs
    this implication to hold. A *result* cache must NOT key on it — the deferred
    remainder still narrows the rows; see the field's docstring. It is what the OR branch of the gate
    actually buys, and it is why that branch is correctness rather than tidiness.

    Two filters differing ONLY in a bind inside a deferred ``$or`` both prune to
    the empty AND — ``consumed_subtree`` applies the all-or-nothing OR rule
    regardless of the gate — so they share a hash by construction. Without the OR
    branch, emission would still push the consumable disjunct, and the two would
    emit DIFFERENT SQL under one key.

    Preferred over an exact-fragment assertion because it survives any harmless
    reformatting of the emitted SQL while still failing if the gate is removed.
    """
    consumable_leaf, unconsumable_leaf = _leaf_pair(name)
    key = next(iter(filter_leaf_keys(_ast(consumable_leaf))))
    other = _second_operand(name, key)

    first = compiler(_ast({"$or": [consumable_leaf, unconsumable_leaf]}), ctx)
    second = compiler(_ast({"$or": [other, unconsumable_leaf]}), ctx)

    # Non-vacuity: the two filters really are different, and really do collide.
    assert consumable_leaf != other
    assert first.consumed_slice_hash == second.consumed_slice_hash
    assert _render(first.predicate) == _render(second.predicate)


def _second_operand(name: str, key: str) -> dict:
    """A second leaf on the same key with a DIFFERENT operand."""
    if key in _DATE_KEYS:
        return {key: {"$gte": _D2}}
    return {key: "a-different-value"}


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _raise_targets(), ids=_RAISE_IDS)
def test_raise_mode_canonical_hash_is_unchanged_over_the_corpus(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """In ``"raise"`` mode the hash still equals ``canonical_hash(ast)``.

    Chunk-tier plan-identity hashes must not move. In raise mode every leaf that reaches
    emission is consumable (an unconsumable one raises instead), so the consumed
    slice IS the whole tree and hashing the slice is hashing the AST. Asserted
    over the whole corpus rather than a sample, because "the keys did not move"
    is the kind of claim a handful of cases can pass while a real one shifts.
    """
    for case, ast in _corpus():
        try:
            compiled = compiler(ast, ctx)
        except (RecallFilterUnsupportedError, CompileError):
            continue
        assert compiled.consumed_slice_hash == canonical_hash(ast), f"{name} {case}: plan hash moved"
        assert compiled.consumed_keys == filter_leaf_keys(ast), f"{name} {case}: raise mode left a residual"


# ===========================================================================
# Raise-mode inertness — the gate must contribute NOTHING when it cannot fire.
# ===========================================================================

# Emissions captured from the compilers as they stood BEFORE the split gate
# landed (commit dc313e60), for the two chunk-tier contexts that ship
# ``on_unsupported="raise"``. Measured, not hand-written. The full measurement
# covered all 209 corpus ASTs across six contexts (1254 records) and found every
# predicate and every bind byte-identical; this table is the standing guard that
# keeps a representative slice of that result checked on every run.
#
# The shapes are chosen for what the gate COULD have touched — ``$or``, ``$not``,
# and a ``$not`` over an ``$or`` with a mixed subtree — plus ordinary leaves as
# controls. If a legitimate change to the emitted SQL lands, update these strings
# and say so; do not delete the table.
_RAISES = object()

_PG_RAISE_PINS: tuple[tuple[str, dict, Any], ...] = (
    ("bare", {}, "true"),
    (
        "sys_range",
        {"created_at": {"$gte": _D1}},
        "coalesce(khora_chunks.created_at >= '2026-01-01 00:00:00+00:00', false)",
    ),
    ("sys_exists", {"source_url": {"$exists": True}}, "true"),
    (
        "or_sys",
        {"$or": [{"occurred_at": {"$gte": _D1}}, {"created_at": {"$lt": _D2}}]},
        "coalesce(khora_chunks.occurred_at >= '2026-01-01 00:00:00+00:00', false) "
        "OR coalesce(khora_chunks.created_at < '2026-02-01 00:00:00+00:00', false)",
    ),
    (
        "not_sys",
        {"$not": {"created_at": {"$gte": _D1}}},
        "NOT coalesce(khora_chunks.created_at >= '2026-01-01 00:00:00+00:00', false)",
    ),
    (
        "not_or_mixed",
        {"$not": {"$or": [{"metadata.tier": "gold"}, {"created_at": {"$lt": _D2}}]}},
        "NOT ((coalesce(khora_chunks.metadata, CAST('{}' AS JSONB)) @> CAST('{\"tier\": \"gold\"}' AS JSONB)) "
        "OR (coalesce(khora_chunks.metadata, CAST('{}' AS JSONB)) @> CAST('{\"tier\": [\"gold\"]}' AS JSONB)) "
        "OR coalesce(khora_chunks.created_at < '2026-02-01 00:00:00+00:00', false))",
    ),
    (
        "mixed_and",
        {"$and": [{"created_at": {"$gte": _D1}}, {"metadata.tier": "gold"}]},
        "coalesce(khora_chunks.created_at >= '2026-01-01 00:00:00+00:00', false) "
        "AND ((coalesce(khora_chunks.metadata, CAST('{}' AS JSONB)) @> CAST('{\"tier\": \"gold\"}' AS JSONB)) "
        "OR (coalesce(khora_chunks.metadata, CAST('{}' AS JSONB)) @> CAST('{\"tier\": [\"gold\"]}' AS JSONB)))",
    ),
)

# The bind values the ``$f_N`` placeholders below refer to, as real objects —
# ``compile_surrealdb`` carries datetimes through verbatim rather than
# stringifying them, which is itself part of what these pins protect.
_DT1 = datetime(2026, 1, 1, tzinfo=UTC)
_DT2 = datetime(2026, 2, 1, tzinfo=UTC)

_SURREAL_RAISE_PINS: tuple[tuple[str, dict, Any, dict], ...] = (
    ("bare", {}, "true", {}),
    (
        "sys_range",
        {"created_at": {"$gte": _D1}},
        "((created_at IS NOT NONE AND created_at >= $f_0))",
        {"f_0": _DT1},
    ),
    # Undeclared on this whitelist — raising is the pre-existing behaviour and
    # the gate must not have widened it into a placeholder.
    ("undeclared_sys", {"source_type": "email"}, _RAISES, {}),
    ("undeclared_exists", {"source_url": {"$exists": True}}, _RAISES, {}),
    (
        "or_sys",
        {"$or": [{"occurred_at": {"$gte": _D1}}, {"created_at": {"$lt": _D2}}]},
        "(((occurred_at IS NOT NONE AND occurred_at >= $f_0)) OR ((created_at IS NOT NONE AND created_at < $f_1)))",
        {"f_0": _DT1, "f_1": _DT2},
    ),
    (
        "not_sys",
        {"$not": {"created_at": {"$gte": _D1}}},
        "!(((created_at IS NOT NONE AND created_at >= $f_0)))",
        {"f_0": _DT1},
    ),
    (
        "not_or_mixed",
        {"$not": {"$or": [{"metadata.tier": "gold"}, {"created_at": {"$lt": _D2}}]}},
        "!((((metadata_.tier = $f_0 OR (type::is::array(metadata_.tier) AND metadata_.tier CONTAINS $f_0))) "
        "OR ((created_at IS NOT NONE AND created_at < $f_1))))",
        {"f_0": "gold", "f_1": _DT2},
    ),
    (
        "mixed_and",
        {"$and": [{"created_at": {"$gte": _D1}}, {"metadata.tier": "gold"}]},
        "((created_at IS NOT NONE AND created_at >= $f_0) "
        "AND (metadata_.tier = $f_1 OR (type::is::array(metadata_.tier) AND metadata_.tier CONTAINS $f_1)))",
        {"f_0": _DT1, "f_1": "gold"},
    ),
)


@pytest.mark.parametrize(("case", "wire", "expected"), _PG_RAISE_PINS, ids=[pin[0] for pin in _PG_RAISE_PINS])
def test_postgres_raise_mode_emission_is_byte_identical(case: str, wire: dict, expected: Any) -> None:
    """The pgvector chunk-tier context emits exactly what it emitted before.

    ``storage/temporal/pgvector.py`` is the hottest compile path in the library
    and it runs ``on_unsupported="raise"``, where the gate is unreachable by
    construction: an unconsumable leaf raises before any OR/NOT node can defer.
    "Unreachable by construction" is an argument, though, and this is the
    measurement.
    """
    ctx = build_compile_context("khora_chunks", on_unsupported="raise")
    if expected is _RAISES:
        with pytest.raises(RecallFilterUnsupportedError):
            compile_postgres(_ast(wire), ctx)
        return
    assert _render(compile_postgres(_ast(wire), ctx).predicate) == expected


@pytest.mark.parametrize(
    ("case", "wire", "expected", "expected_params"),
    _SURREAL_RAISE_PINS,
    ids=[pin[0] for pin in _SURREAL_RAISE_PINS],
)
def test_surrealdb_raise_mode_emission_is_byte_identical(
    case: str, wire: dict, expected: Any, expected_params: dict
) -> None:
    """The SurrealDB chunk-tier context emits exactly what it emitted before.

    Both halves are pinned: the predicate string AND the bind values it refers
    to. ``compile_surrealdb`` emits NAMED placeholders (``$f_0``), so — unlike
    the postgres table, where ``literal_binds`` inlines operands into the string
    — the predicate alone would leave every operand unasserted, and "predicate
    and binds unchanged" would only be half-checked.
    """
    ctx = CompileContext(backend_target="temporal_chunk", field_mapping=_SURREAL_CHUNK_MAPPING, on_unsupported="raise")
    if expected is _RAISES:
        with pytest.raises(RecallFilterUnsupportedError):
            compile_surrealdb(_ast(wire), ctx)
        return
    compiled = compile_surrealdb(_ast(wire), ctx)
    assert _render(compiled.predicate) == expected
    assert compiled.params == expected_params


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _raise_targets(), ids=_RAISE_IDS)
def test_raise_mode_raises_exactly_when_a_leaf_is_unconsumable(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """BICONDITIONAL over the corpus: raises ⇔ some leaf is not consumable.

    The structural reason raise-mode emission cannot have moved: the gate is
    consulted only under ``"split"``, so in ``"raise"`` mode the ONLY thing
    ``_clause_consumable`` can correspond to is whether the compile raised. Both
    directions matter — a compile that raised on a fully-consumable AST would
    mean the gate had started rejecting things, and one that succeeded with an
    unconsumable leaf would mean an unsupported leaf had slipped through as a
    placeholder.

    The left-hand side is specifically ``RecallFilterUnsupportedError``, NOT "any
    exception". :class:`CompileError` (an unsafe metadata identifier — one corpus
    case, ``f_dotkey_cases`` ``{'metadata.$ref': 'x'}``) is an injection guard,
    orthogonal to consumability: the predicate legitimately stays ``True`` while
    the compile raises. Those cases are EXCLUDED. Accepting "either error" would
    make the test vacuously pass on exactly the cases it should ignore.
    """
    excluded = 0
    for case, ast in _corpus():
        builder = builder_cls(ctx=ctx, consumed=set())
        predicted = any(not builder._clause_consumable(leaf) for leaf in _iter_leaves(ast))
        try:
            compiler(ast, ctx)
            actual = False
        except RecallFilterUnsupportedError:
            actual = True
        except CompileError:
            excluded += 1
            continue
        assert predicted == actual, f"{name} {case}: predicted_raise={predicted} actually_raised={actual}"

    assert excluded <= 1, f"{name}: unexpected number of CompileError cases ({excluded})"

    # Non-vacuity. A biconditional is trivially true when one side is constant,
    # so both branches have to be reachable SOMEWHERE — but the honest statement
    # differs per target, and flattening the two would hide which is which.
    assert compiler(_ast({"created_at": {"$gte": _D1}}), ctx).consumed_keys == frozenset({"created_at"})
    if name == "postgres/chunk":
        # No CORPUS case raises here: ``field_mapping=None`` declares every
        # system key and every dotted metadata path is pushable, so every leaf
        # the corpus contains is consumable. That is a fact about the corpus, NOT
        # about the context — an unconsumable leaf does exist for this context
        # (a path that is neither a system key nor ``metadata.*``), and the
        # raising branch is exercised with one below on the SHIPPED context
        # rather than on a whitelisted variant of it.
        assert all(not _raises(compile_postgres, ast, ctx) for _case, ast in _corpus())
        with pytest.raises(RecallFilterUnsupportedError):
            compile_postgres(FilterNode(op=Op.AND, children=(FilterClause(path=("a",), op=Op.EQ, operand=1),)), ctx)
        return

    # SurrealDB's chunk whitelist withholds most system keys, so the corpus
    # itself exercises both branches.
    outcomes = {_raises(compiler, ast, ctx) for _case, ast in _corpus()} - {None}
    assert outcomes == {True, False}, f"{name}: corpus no longer exercises both branches ({outcomes})"


def _raises(compiler: Callable, ast: FilterNode, ctx: CompileContext) -> bool | None:
    """``True``/``False`` for raised-or-not; ``None`` for an excluded CompileError."""
    try:
        compiler(ast, ctx)
    except RecallFilterUnsupportedError:
        return True
    except CompileError:
        return None
    return False


@pytest.mark.parametrize(
    "wire",
    (
        {"metadata": {"a": 1}},
        {"metadata": {"$ne": {"a": 1}}},
        {"metadata": {"$gt": {"a": 1}}},
        {"metadata": {"$gte": {"a": 1}}},
        {"metadata": {"$lt": {"a": 1}}},
        {"metadata": {"$lte": {"a": 1}}},
        {"metadata": {"$in": [{"a": 1}]}},
        {"metadata": {"$nin": [{"a": 1}]}},
        {"metadata": {"$exists": True}},
        {"metadata": {}},
    ),
    ids=lambda w: str(next(iter(w.values()))),
)
def test_every_bare_metadata_wire_form_folds_to_eq(wire: dict) -> None:
    """The bare ``metadata`` blob always lowers to ``$eq`` over a literal operand.

    This is the genuinely validator-guaranteed property, and it is what makes
    ``_clause_consumable``'s bare-metadata branch (``len(path) > 1 or op ==
    Op.EQ``) agree with emission in practice: the ``op != Op.EQ`` half is not
    reachable through a ``metadata`` wire form, because every operator-looking
    key is read as part of the literal blob rather than as an operator.

    Pinned per form rather than as prose because the whole bare-metadata
    consumability branch rests on it — if the validator ever grows real operator
    parsing on the bare blob, these are the cases that say so.
    """
    leaves = list(_iter_leaves(_ast(wire)))
    assert [leaf.path for leaf in leaves] == [("metadata",)]
    assert [leaf.op for leaf in leaves] == [Op.EQ]


# ``(id, compiler, AND-wrapped placeholder, bare placeholder)``. The two spellings
# differ on ``compile_lance`` and are NOT interchangeable: an AND root wraps its
# single child, so a lone deferred leaf renders ``(1)``, whereas a deferred
# ``$not`` returns the placeholder from the NOT node itself with no wrapper and
# renders a bare ``1``. ``compile_postgres`` collapses both to ``true`` because
# SQLAlchemy folds the wrapper away. Asserting the exact spelling per position
# rather than "contains a placeholder" is what keeps the two positions distinct.
_PLACEHOLDER_SHAPES: tuple[tuple[str, Callable, str, str], ...] = (
    ("postgres", compile_postgres, "true", "true"),
    ("lance", compile_lance, "(1)", "1"),
)


@pytest.mark.parametrize(
    ("name", "compiler", "placeholder", "bare_placeholder"),
    _PLACEHOLDER_SHAPES,
    ids=[case[0] for case in _PLACEHOLDER_SHAPES],
)
def test_an_undeclared_path_never_reaches_a_column_token(
    name: str,
    compiler: Callable,
    placeholder: str,
    bare_placeholder: str,
) -> None:
    """A path that is neither a system key nor ``metadata.*`` is DEFERRED, never emitted.

    This pins ``compile_clause``'s final ``else`` branch as LIVE. That branch is
    the only thing between an arbitrary path and ``_col``, which builds its
    column token with ``sa.literal_column(f"{qualifier}.{physical}")`` — raw SQL
    text, not a bind — and on every chunk-tier context ``field_mapping is None``,
    so ``physical`` would be the path segment verbatim.

    ``_col``'s own docstring says ``logical_name`` "is always a controlled token
    … never free user text". That is true only BECAUSE this branch rejects
    everything else first: it is a property maintained upstream, not a property
    of the input. So without this test, the branch reads as dead code and the
    docstring reads as a standing guarantee — two mutually reinforcing reasons to
    delete a live guard.

    The clause is constructed DIRECTLY rather than through
    :meth:`RecallFilter.model_validate`, deliberately. This asserts compiler
    behaviour, which is what the guard is; routing it through the validator would
    couple the test to whichever wire forms happen to reach a compiler today, and
    it would start erroring rather than passing if that ever narrows.
    """
    ctx = CompileContext(backend_target="khora_chunks", on_unsupported="split", schema_capabilities=_JSON1_ON)
    payload = 'evil" OR 1=1--'
    ast = FilterNode(op=Op.AND, children=(FilterClause(path=(payload,), op=Op.EQ, operand=1),))

    compiled = compiler(ast, ctx)
    assert _render(compiled.predicate).strip() == placeholder
    assert compiled.consumed_keys == frozenset()
    # THE ASSERTION THAT ACTUALLY PROVES CONTAINMENT. "the leaf is unconsumable"
    # would still pass if a future refactor routed the path into the column token
    # by some other route; the absence of the payload from the rendered SQL is
    # what says it did not. Rendering is via ``_render``, i.e. ``literal_binds``
    # for SQLAlchemy, so this sees the real statement text rather than ``:param_1``.
    assert payload not in _render(compiled.predicate)

    # UNDER A NEGATION — the gate must refuse to push it there too. Emitting a
    # placeholder inside a ``$not`` would invert it to match-nothing, which is
    # the unrecoverable direction, and it is a separate branch of the gate from
    # the positive case above. The expected shape is ``bare_placeholder``, not
    # ``placeholder``: the NOT branch returns the gate's placeholder directly,
    # whereas the positive case above reaches it through the AND branch, which
    # parenthesizes its joined children. On postgres the two coincide; on lance
    # they are ``1`` and ``(1)``, so asserting the AND-wrapped form here would
    # pin a shape the NOT branch never emits.
    negated = FilterNode(op=Op.NOT, children=(ast,))
    negated_compiled = compiler(negated, ctx)
    assert _render(negated_compiled.predicate).strip() == bare_placeholder
    assert negated_compiled.consumed_keys == frozenset()
    assert payload not in _render(negated_compiled.predicate)

    # Same leaf under "raise" is a structured error, not a silent placeholder —
    # in both positions.
    raising = dataclasses.replace(ctx, on_unsupported="raise")
    with pytest.raises(RecallFilterUnsupportedError):
        compiler(ast, raising)
    with pytest.raises(RecallFilterUnsupportedError):
        compiler(negated, raising)


# ===========================================================================
# THE MIRROR — the only check standing between the gate predicate and a silent
# row-drop. Rank it above everything else in this module.
# ===========================================================================


@pytest.mark.parametrize(("name", "compiler", "builder_cls", "ctx"), _split_targets(), ids=_SPLIT_IDS)
def test_clause_consumable_mirrors_what_emission_did(
    name: str,
    compiler: Callable,
    builder_cls: Any,
    ctx: CompileContext,
) -> None:
    """``_clause_consumable`` must agree with what ``compile_clause`` ACTUALLY did.

    **This is the load-bearing test for the DANGEROUS direction.** The gate
    defers an ``OR`` / ``NOT`` based on ``_clause_consumable``, and the shipped
    implementation RE-DERIVES the deferred set from that same predicate rather
    than recording what emission chose. So if the predicate ever OVER-claims — a
    new unsupported metadata shape handled in ``compile_clause`` but not mirrored
    in ``_clause_consumable`` — the gate would judge a subtree consumable, push
    it, and let a match-all placeholder invert under a negation. That is a silent
    row-drop, and the deferred-path subtraction cannot catch it: the subtraction
    guards the REPORT, only this guards the PREDICATE.

    **It is one-directional, and deliberately not the whole story.** It reads
    ``actual = bool(compiled.consumed_keys)``, and ``consumed_keys`` is itself
    derived from ``_clause_consumable`` via ``deferred_paths`` — so when the
    predicate says *unconsumable*, the subtraction forces ``consumed_keys`` empty
    no matter what emission did, and the assertion becomes self-satisfying.
    Measured: making the postgres metadata branch return ``False`` while emission
    still pushes those leaves leaves this test GREEN on all six targets, and is
    caught only by invariant (a) in
    :func:`test_the_two_accounting_invariants_over_the_corpus`, which compares
    the raw emission accumulator against an independently-derived slice.

    That UNDER-claim direction costs pushdown rather than rows, so it is the
    cheaper failure — but do not read an all-green mirror as proof that predicate
    and emission agree in both directions. Invariant (a) carries that half.

    **One known, documented under-claim exists today.** ``compile_lance``'s
    ``_clause_unconsumable`` tests ``isinstance(operand, (DateLiteral, dict))``
    *before* dispatching on the operator, while ``_compile_metadata_clause`` only
    consults the operand type for ``$eq`` / ``$ne`` / range. So for ``$exists``,
    and for range / ``$in`` / ``$nin`` carrying a dict operand, the predicate
    reports unconsumable while emission pushes the leaf. It is validator-reachable
    (``{"metadata.tier": {"$gt": {"a": 1}}}`` validates), and it is in the SAFE
    direction — the leaf stays post-filtered, so no row is dropped. Its one visible
    consequence is that ``consumed_slice_hash`` prunes a leaf the WHERE still
    contains, which makes the equal-hash-implies-equal-predicate property
    (:func:`test_equal_hash_implies_equal_emitted_predicate`) hold with this single
    documented exception. The line pre-dates the split machinery; narrowing it to
    mirror the emit path's op dispatch is a tracked follow-up, deliberately not
    done here because it changes pre-existing pushdown behaviour and needs its own
    corpus case.

    Each leaf runs in ISOLATION (``FilterNode(AND, (leaf,))``), which is the
    right shape: ``_clause_consumable`` is per-leaf by definition, and
    composition is the gate's job — covered by the biconditional and the
    accounting invariants above.

    :class:`CompileError` cases are excluded (see the biconditional's docstring).
    """
    checked = 0
    for leaf in _distinct_corpus_leaves():
        builder = builder_cls(ctx=ctx, consumed=set())
        predicted = builder._clause_consumable(leaf)
        try:
            compiled = compiler(FilterNode(op=Op.AND, children=(leaf,)), ctx)
        except CompileError:
            continue
        actual = bool(compiled.consumed_keys)
        assert predicted == actual, (
            f"{name}: {_dotted(leaf)} {leaf.op} — _clause_consumable said {predicted}, emission consumed {actual}"
        )
        checked += 1
    assert checked > 100, f"{name}: only {checked} leaves reached the mirror; the corpus shrank"


# The whitelist branch must actually rule some leaves unconsumable, or the mirror
# above is green for a compiler that never exercises it. Measured counts, of the
# 182 distinct corpus leaves. If a corpus change drops one of these to zero the
# mirror has gone vacuous on that target and the test must be FIXED, not deleted.
_WHITELIST_REJECTION_COUNTS: tuple[tuple[str, int, frozenset[str]], ...] = (
    ("postgres/documents", 16, frozenset({"occurred_at"})),
    ("lance/documents", 37, frozenset({"created_at", "occurred_at", "source_timestamp"})),
    (
        "surreal/chunk",
        77,
        frozenset(
            {
                "content_type",
                "external_id",
                "source",
                "source_name",
                "source_timestamp",
                "source_type",
                "source_url",
                "title",
            }
        ),
    ),
)


@pytest.mark.parametrize(
    ("target", "expected_count", "expected_keys"),
    _WHITELIST_REJECTION_COUNTS,
    ids=[case[0] for case in _WHITELIST_REJECTION_COUNTS],
)
def test_the_mirror_is_not_vacuous_on_the_whitelist_branch(
    target: str,
    expected_count: int,
    expected_keys: frozenset[str],
) -> None:
    """The whitelist branch is REACHED — without this, an all-green mirror proves nothing.

    A mirror test passes trivially if every leaf is consumable on every context:
    ``_clause_consumable`` returning a constant ``True`` would satisfy it. These
    counts are the proof that the interesting branch runs, and the key sets say
    WHICH keys each context withholds, so a mapping change that silently stops
    withholding one shows up here rather than as a quiet loss of coverage.
    """
    by_name = {name: (builder_cls, ctx) for name, _fn, builder_cls, ctx in _split_targets()}
    builder_cls, ctx = by_name[target]
    builder = builder_cls(ctx=ctx, consumed=set())

    system_leaves = [leaf for leaf in _distinct_corpus_leaves() if len(leaf.path) == 1 and leaf.path[0] in SYSTEM_KEYS]
    rejected = [leaf for leaf in system_leaves if not builder._clause_consumable(leaf)]

    assert rejected, f"{target}: the whitelist branch rules NOTHING unconsumable — the mirror is vacuous here"
    assert {leaf.path[0] for leaf in rejected} == expected_keys
    assert len(rejected) == expected_count


# ===========================================================================
# The metadata-identifier guard.
# ===========================================================================


def test_safe_segment_pattern_rejects_a_trailing_newline() -> None:
    """``"tier\\n"`` must NOT be accepted as a metadata path segment.

    A WRONG-FIELD bug before an injection one. The pattern was anchored with
    ``$``, which in Python also matches just before a trailing newline, so
    ``"tier\\n"`` passed the guard and was interpolated verbatim into the
    predicate string — addressing ``metadata_.tier\\n``, a field name no document
    has, which on a SCHEMAFULL table reads ``NONE`` and silently matches no rows.
    ``\\Z`` is the absolute end of the string.
    """
    for pattern in (surrealdb_module._SAFE_SEGMENT_RE, _conformance_segment_re()):
        assert pattern.match("tier") is not None
        assert pattern.match("tier\n") is None
        # The anchor is the fix; a segment containing an interior newline was
        # already rejected by the character class and proves nothing on its own.
        assert pattern.match("tier\nother") is None
        assert re.search(r"\\Z", pattern.pattern), f"{pattern.pattern!r} is not \\Z-anchored"


def _conformance_segment_re() -> re.Pattern[str]:
    """The conformance harness's mirror of the compiler's pattern.

    Imported through the module rather than at file scope to keep the two copies
    visibly separate — they are duplicated on purpose (the harness must decide
    what SurrealDB will reject without importing the compiler), and a fix to one
    that misses the other is exactly what this test exists to catch.
    """
    from khora.filter import conformance

    return conformance._SURREAL_SAFE_SEGMENT_RE


def test_the_two_segment_patterns_have_not_drifted() -> None:
    # The mirror is only a mirror while the patterns agree; the previous
    # revision's bug was present in both and had to be fixed in both.
    assert surrealdb_module._SAFE_SEGMENT_RE.pattern == _conformance_segment_re().pattern


@pytest.mark.parametrize("mode", ("split", "raise"))
def test_an_unsafe_metadata_segment_raises_in_both_modes(mode: str) -> None:
    """KNOWN GAP, pinned as it behaves — the guard is not a split-mode capability gap.

    ``compile_surrealdb`` raises :class:`CompileError` on a non-identifier
    metadata segment under BOTH modes, rather than routing it to the
    unsupported-leaf path where ``"split"`` would defer it. That is deliberate
    today: it is an injection guard, and ``_clause_consumable`` therefore does
    NOT consider segment safety — reporting such a leaf unconsumable would let
    the enclosing subtree defer, the guard would never run, and the error would
    vanish rather than surface.

    Pinned as current behaviour, not endorsed. If the guard is ever turned into a
    real capability gap, ``_clause_consumable`` must become mode-aware in the
    same change or the mirror above will start failing.
    """
    ctx = CompileContext(
        backend_target="document",
        field_mapping={"title": "title", "metadata": "metadata_"},
        on_unsupported=mode,
    )
    with pytest.raises(CompileError):
        compile_surrealdb(_ast({"metadata.$ref": "x"}), ctx)


# ===========================================================================
# Scope note — the compilers deliberately NOT fixed here.
# ===========================================================================


def test_the_split_helpers_are_only_wired_into_the_three_sql_compilers() -> None:
    """``compile_cypher`` / ``compile_weaviate`` / ``compile_chronicle`` are OUT of scope.

    They share the same over-claiming ``consumed_keys`` shape and are
    deliberately left alone in this change, so their cache-key pins stay green.
    Asserted as an import fact rather than as behaviour: if one of them is fixed
    later, this is the line that says "update the scope note too".
    """
    from khora.filter.compilers import _split, chronicle, cypher, weaviate

    def _uses_split(module: Any) -> bool:
        """Whether ``module`` bound any name exported by :mod:`._split`."""
        return any(getattr(module, name, None) is getattr(_split, name) for name in _split.__all__)

    candidates = (lance_module, postgres_module, surrealdb_module, cypher, weaviate, chronicle)
    wired = {module.__name__.rsplit(".", 1)[-1] for module in candidates if _uses_split(module)}
    assert wired == {"lance", "postgres", "surrealdb"}
