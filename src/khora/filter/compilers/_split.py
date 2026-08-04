"""Split-mode pushdown soundness — the shared gate every SQL-ish compiler uses.

``@internal``. One home for the three rules a compiler running under
``CompileContext.on_unsupported == "split"`` must obey, so the postgres / lance /
surrealdb compilers cannot drift from each other (or from the invariants their
callers' post-filters rely on).

**1. AND distributes; OR / NOT are all-or-nothing.** The match-all placeholder an
unconsumable leaf emits under ``"split"`` (``sa.true()`` / ``"1"`` / ``"true"``)
is superset-safe only in *positive* position: ``A AND <match-all>`` ≡ ``A`` (still
narrows correctly), but ``NOT (A OR <match-all>)`` ≡ ``NOT <match-all>`` ≡
match-nothing — which would *drop every row the filter keeps*, breaking the
superset invariant (a ``compile_python`` post-filter only narrows; it cannot add
a wrongly-excluded row back). So an ``OR`` / ``NOT`` node is pushed down only when
its **entire** subtree is consumable (:func:`node_consumable`); otherwise the whole
node emits the placeholder and consumes nothing, deferring it wholesale to the
post-filter. An ``AND`` still handles each child independently.

**2. The reported consumed slice is the pruned tree, not the emission trace.**
:func:`consumed_subtree` reconstructs exactly the sub-AST that reached the
backend. Hashing it (:func:`consumed_slice_hash`) yields a **plan-identity hash**:
two filters that differ only in a deferred subtree compile to the same predicate
and therefore share it. That is emphatically **not** a result-cache key — the
deferred remainder still narrows the result set through the caller's post-filter,
so two filters sharing this hash routinely return different rows. Key a result
cache on ``canonical_hash(filter_ast)`` over the *whole* AST instead. The slice is
also what makes ``consumed_keys`` honest per *occurrence*: a dotted path is only
fully pushed when **every** one of its occurrences landed in the slice, so a key
pushed in one branch and deferred in another is reported via
:func:`deferred_paths` and stays post-filtered.

**3. ``field_mapping is None`` means identity, NOT an empty whitelist.** A
compiler MAY read a *non-``None``* ``field_mapping``'s key set as the backend's
declared+pushable system-key whitelist (:func:`system_key_declared`), but ``None``
is the documented identity mapping — every chunk-tier postgres / lance context
passes ``None``, so folding it into "nothing is declared" would defer every
system-key pushdown on the hot recall path. Only a backend that must fail *closed*
reads an absent mapping as an empty whitelist — surrealdb (a missing field on a
SCHEMAFULL table reads ``NONE`` and total-false drops every row) and weaviate —
and both do it with their own ``dict(ctx.field_mapping or {})`` rather than this
helper.

These helpers are pure: no bind allocation, no ``consumed`` mutation, no
telemetry. A compiler supplies a ``ClauseConsumable`` predicate mirroring its own
leaf dispatch, and the gate does the rest.

**Memoization.** The gate is consulted once per ``OR`` / ``NOT`` node during the
emission walk and again while pruning, so without memoization a leaf is re-tested
once per enclosing level — O(depth x leaves). Every entry point therefore accepts
a ``memo`` dict that a compiler creates once per compile pass and threads through,
making the whole gate O(n). The memo is keyed on ``id(node)``, which is sound only
because the AST is held alive by the caller for the pass's duration; it MUST NOT
outlive the compile.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from typing import Literal

from khora.filter import CompileContext
from khora.filter.ast import (
    FilterClause,
    FilterNode,
    _normalize,
    canonical_hash,
)
from khora.filter.model import Op

__all__ = [
    "consumed_slice_hash",
    "consumed_subtree",
    "deferred_paths",
    "node_consumable",
    "split_report",
    "system_key_declared",
]


# A compiler's own leaf-dispatch mirror: ``True`` iff this leaf compiles to a real
# backend predicate (rather than the match-all placeholder). It must be a pure
# predicate over the clause, and it must NOT consider guards that raise in both
# modes (e.g. surrealdb's unsafe-identifier check) — those are injection guards,
# not capability gaps, and swallowing them here would silently drop the error.
ClauseConsumable = Callable[[FilterClause], bool]

# Per-compile-pass memo for :func:`node_consumable`, keyed on ``id(node)``.
ConsumableMemo = dict[int, bool]


def system_key_declared(ctx: CompileContext, key: str) -> bool:
    """Whether ``ctx`` declares a real backend column for system ``key``.

    ``field_mapping is None`` is the **identity mapping** — no whitelist, so every
    system key is declared (the chunk-tier contract; see the module docstring).
    A non-``None`` mapping IS the declared+pushable whitelist: a system key it
    omits is not backed by a column this compiler may push to, so the leaf is
    unconsumable and falls to the caller's post-filter.
    """
    return ctx.field_mapping is None or key in ctx.field_mapping


def node_consumable(
    node: FilterNode | FilterClause,
    ok: ClauseConsumable,
    memo: ConsumableMemo | None = None,
) -> bool:
    """True iff ``node``'s whole subtree compiles (nothing falls to the placeholder).

    A logical node is consumable iff every child is; a leaf is consumable iff
    ``ok`` says so. This is the all-or-nothing gate for ``OR`` / ``NOT`` (rule 1).

    Pass the compile pass's ``memo`` to make repeated consultations O(1) per node
    (see the module docstring); omitting it is correct but re-walks the subtree.
    """
    if memo is None:
        memo = {}
    cached = memo.get(id(node))
    if cached is not None:
        return cached
    if isinstance(node, FilterClause):
        result = ok(node)
    else:
        result = all(node_consumable(child, ok, memo) for child in node.children)
    memo[id(node)] = result
    return result


def consumed_subtree(
    node: FilterNode | FilterClause,
    ok: ClauseConsumable,
    memo: ConsumableMemo | None = None,
) -> FilterNode:
    """Reconstruct the sub-AST that actually reaches the backend under ``"split"``.

    Mirrors the emission walk exactly: an ``AND`` keeps its consumable children
    (possibly none — the empty ``AND`` is match-everything, the placeholder's
    meaning); an ``OR`` / ``NOT`` is kept whole iff :func:`node_consumable`, else
    dropped entirely. The pruned tree is run back through the AST's own
    normalization so it is in the same canonical form
    :func:`~khora.filter.ast.parse_to_ast` produces — otherwise a pruned
    single-child wrapper would hash differently from the identical filter written
    without the deferred sibling.
    """
    pruned = _prune(node, ok, memo if memo is not None else {})
    if pruned is None:
        # Everything deferred — the compiler emitted the match-all placeholder,
        # whose AST equivalent is the empty (match-everything) AND.
        return FilterNode(op=Op.AND, children=())
    if isinstance(pruned, FilterClause):
        # Keep the root a FilterNode (the compiler-boundary contract).
        pruned = FilterNode(op=Op.AND, children=(pruned,))
    normalized = _normalize(pruned)
    assert isinstance(normalized, FilterNode)  # noqa: S101 - root-shape invariant (ast._normalize rule 2)
    return normalized


def consumed_slice_hash(
    node: FilterNode | FilterClause,
    ok: ClauseConsumable,
    memo: ConsumableMemo | None = None,
) -> str:
    """Plan-identity hash over the consumed slice — **never** a result-cache key.

    Identifies the predicate the backend actually received, so two filters
    differing only in a deferred subtree share it. Their *result sets* still
    differ, because the deferred remainder is enforced by the caller's
    post-filter — see rule 2 in the module docstring.

    In ``on_unsupported="raise"`` mode every leaf that reaches emission is
    consumable (an unconsumable one raises instead of returning), so the slice is
    the whole tree and the hash is identical to ``canonical_hash(node)``.
    """
    return canonical_hash(consumed_subtree(node, ok, memo))


def deferred_paths(
    node: FilterNode | FilterClause,
    ok: ClauseConsumable,
    memo: ConsumableMemo | None = None,
) -> frozenset[str]:
    """Dotted paths with at least one leaf occurrence NOT in the consumed slice.

    ``consumed_keys`` is a set of dotted paths, but a path can occur many times in
    one AST — pushed in a conjunctive leaf and deferred inside an ``$or`` / ``$not``
    the gate defers wholesale. Reporting such a path consumed would tell a caller
    differencing ``leaf_keys - consumed_keys`` that the deferred occurrence is
    already enforced, so it would never be post-filtered and the query would
    return rows the filter excludes. Subtracting this set from the emission
    accumulator leaves exactly the paths **every** occurrence of which was pushed.
    """
    return _deferred_from_slice(node, consumed_subtree(node, ok, memo))


def split_report(
    node: FilterNode,
    ok: ClauseConsumable,
    *,
    on_unsupported: Literal["raise", "split"],
    memo: ConsumableMemo | None = None,
) -> tuple[frozenset[str], str]:
    """``(deferred_paths, consumed_slice_hash)`` from a single pruned-tree build.

    The two outputs share one :func:`consumed_subtree` reconstruction — computing
    them separately prunes and re-normalizes the whole AST twice per compile.

    Under ``"raise"`` the result is fixed by construction and skips the walk
    entirely: if compilation returned at all then no leaf hit the unsupported
    path, so nothing was deferred and the slice IS the whole tree. This is the
    live recall path for the raise-mode chunk-tier contexts, where the pruning
    work would be pure waste.
    """
    if on_unsupported == "raise":
        return frozenset(), canonical_hash(node)
    slice_ = consumed_subtree(node, ok, memo)
    return _deferred_from_slice(node, slice_), canonical_hash(slice_)


def _deferred_from_slice(node: FilterNode | FilterClause, slice_: FilterNode) -> frozenset[str]:
    """Paths occurring more often in ``node`` than in its consumed ``slice_``."""
    kept = Counter(_leaf_paths(slice_))
    return frozenset(path for path, count in Counter(_leaf_paths(node)).items() if count > kept[path])


def _prune(
    node: FilterNode | FilterClause,
    ok: ClauseConsumable,
    memo: ConsumableMemo,
) -> FilterNode | FilterClause | None:
    """The consumed-slice walk: ``None`` for a subtree the compiler defers whole."""
    if isinstance(node, FilterClause):
        return node if node_consumable(node, ok, memo) else None
    if node.op == Op.AND:
        kept = tuple(p for p in (_prune(child, ok, memo) for child in node.children) if p is not None)
        return FilterNode(op=Op.AND, children=kept)
    # OR / NOT — all-or-nothing (rule 1): never recursed into, so a leaf under a
    # deferred node is deferred with it.
    return node if node_consumable(node, ok, memo) else None


def _leaf_paths(node: FilterNode | FilterClause) -> Iterator[str]:
    """Yield the dotted key of every leaf occurrence (duplicates preserved).

    Same traversal as ``khora.filter.execute.iter_leaf_clauses`` / the compilers'
    ``_path_str``; inlined so this module stays a dependency-free leaf of the
    filter package.
    """
    if isinstance(node, FilterClause):
        yield ".".join(node.path)
        return
    for child in node.children:
        yield from _leaf_paths(child)
