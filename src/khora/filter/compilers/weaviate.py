"""Weaviate recall-filter compiler — ``@internal``.

Lowers a canonical :class:`~khora.filter.ast.FilterNode` to a Weaviate v4
``Filter`` combinator tree (``Filter.all_of`` / ``Filter.any_of`` /
``Filter.by_property(p).equal(...)`` ...) against a recall chunk collection. The
output is a :class:`~khora.filter.registry.CompiledFilter` whose ``predicate`` is
a weaviate ``_Filters`` object the engine hands to its query call, or ``None`` for
a match-all (no constraint).

Unlike the Postgres / Cypher compilers — which are *total* (every leaf lowers to
a boolean expression, and a wrapping ``NOT`` flips absent rows in via a
coalesced total boolean) — this compiler is a **superset-safe partial pushdown**.
Weaviate has no SQL-style three-valued logic the compiler can wrap to make a
negation null-inclusive: a server-side negation (``$ne`` / ``$nin`` /
``$exists:false`` / ``$not``) silently drops rows where the property is
null/absent, which would FALSE-EXCLUDE rows the filter should keep. So the
compiler pushes down ONLY *monotone-narrowing* predicates — ones that, when
applied server-side, can only ever return a SUPERSET of the true result set — and
leaves everything else to the engine's post-filter (``compile_python`` re-checks
the whole AST against each candidate). Over-returning is always safe (the
post-filter narrows); under-returning (dropping a valid row server-side) is a
correctness bug, so the routing is deliberately conservative.

The superset-safe routing rules:

* **Pushable leaf** (declared property + monotone-narrowing op): ``$eq``,
  ``$gt`` / ``$gte`` / ``$lt`` / ``$lte`` (range), ``$in`` (→ ``any_of`` of scalar
  ``equal`` checks). Each pushes a ``Filter.by_property`` constraint and is added
  to ``consumed_keys``.
* **Unpushable leaf** (``$ne`` / ``$nin`` / a ``$not`` over a leaf, ``$exists`` /
  ``{k: null}``, ANY metadata path, or any undeclared key): contributes NO
  constraint (``None``) and is NOT added to ``consumed_keys`` — the engine
  post-filters it. The negations would drop null/absent rows server-side,
  false-excluding rows the filter must keep; ``$exists`` / null are left to the
  post-filter because the oracle treats a system key as always-present (so a
  server-side ``is_none`` push would diverge — see ``_compile_system_clause``).
* **``$and``** MAY push its pushable conjuncts and drop the unpushable ones:
  AND-ing fewer constraints only WIDENS the candidate set, so it stays a
  superset (the dropped conjuncts are still re-checked by the post-filter).
* **``$or`` / ``$not``** are ALL-OR-NOTHING: an ``$or`` is pushed only if every
  child is pushable (dropping one disjunct would NARROW the union and could drop
  valid rows); a ``$not`` is unpushable as a whole (negation is not
  monotone-narrowing here).

**Declared, pushable properties** are exactly the keys of ``ctx.field_mapping``
(identity-mapped to physical property names). A clause whose key is not in
``field_mapping`` is treated as undeclared and is not pushed.

**Dates** bind as their ``.isoformat()`` string — the chunk's date properties are
stored ISO-8601 (lexicographic compare agrees with chronological order for the
UTC-normalized values the validator produces). The v4 ``Filter`` API also accepts
``datetime`` directly, but the recall chunk stores the value as a string property,
so the string form matches what is on disk.

``@internal``. Reachable as ``khora.filter.compilers.weaviate.compile_weaviate``
for khora's own engines; not re-exported from :mod:`khora.__init__`.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from khora.filter import (
    CompileContext,
    CompiledFilter,
    RecallFilterUnsupportedError,
)
from khora.filter.ast import (
    DateLiteral,
    FilterClause,
    FilterNode,
    canonical_hash,
)
from khora.filter.model import SYSTEM_KEYS, Op

if TYPE_CHECKING:
    # Only for type hints — the real import is lazy (weaviate-client is an
    # optional extra; this module must import without it).
    from weaviate.collections.classes.filters import _Filters

__all__ = ["compile_weaviate"]


# The monotone-narrowing range ops, keyed by AST op to the ``_FilterByProperty``
# method name. Applying any of these server-side can only return a superset of the
# true result (a row that passes the real predicate also passes the pushed one), so
# they are safe to push down on a declared property.
_RANGE_METHOD = {
    Op.GT: "greater_than",
    Op.GTE: "greater_or_equal",
    Op.LT: "less_than",
    Op.LTE: "less_or_equal",
}


def compile_weaviate(ast: FilterNode, ctx: CompileContext) -> CompiledFilter[Any]:
    """Compile a canonical AST to a superset-safe Weaviate ``Filter`` tree.

    ``ast`` is always a :class:`FilterNode` (the ``parse_to_ast`` root invariant).
    Returns a :class:`CompiledFilter` whose ``predicate`` is a weaviate v4
    ``_Filters`` object, or ``None`` when nothing is pushable (an empty
    match-everything ``AND``, or a tree all of whose leaves are unpushable) — the
    engine treats ``None`` as "apply no server-side filter, post-filter everything".

    ``params`` is always empty (weaviate binds operands inline inside the ``Filter``
    object). ``consumed_keys`` is the set of dotted leaf paths actually pushed
    down; the engine's ``compile_python`` post-filter re-checks the rest (and, in
    fact, re-checks the whole AST regardless, which is what makes over-returning
    safe). Property names are derived from ``ctx.field_mapping`` alone (its keys
    are the declared+pushable set, identity-mapped to physical names), so the
    compiler embeds no engine schema.

    Honors ``ctx.on_unsupported``: in ``"raise"`` mode a clause this backend cannot
    push raises :class:`RecallFilterUnsupportedError`; in ``"split"`` mode (the
    mode this backend is used in) it is silently dropped from the pushed-down
    filter and left to the post-filter. Note that *most* unpushable clauses are a
    correctness requirement here (a pushed negation would false-exclude rows), not
    a backend gap — so the typical engine wiring uses ``"split"``.
    """
    consumed: set[str] = set()
    builder = _Builder(ctx=ctx, consumed=consumed)
    predicate = builder.compile_node(ast)
    return CompiledFilter(
        predicate=predicate,
        params={},
        consumed_keys=frozenset(consumed),
        # canonical_hash over the whole AST — the engine re-checks the whole AST in
        # its post-filter (the pushed-down filter is only a superset prefilter), so
        # the cache key is keyed on the full predicate, not the pushed slice.
        canonical_hash=canonical_hash(ast),
    )


def _path_str(path: tuple[str, ...]) -> str:
    """Render an AST path as the dotted key string used for diagnostics/consumed."""
    return ".".join(path)


class _Builder:
    """Per-compile-pass state: the :class:`CompileContext` and consumed set.

    A compile pass threads the ``ctx`` (declared property names are derived from
    its ``field_mapping``) and the ``consumed`` accumulator. The weaviate
    ``Filter`` symbol is imported lazily once per pass (the module must import
    without the optional ``weaviate-client`` extra installed) and cached on the
    instance.
    """

    def __init__(self, *, ctx: CompileContext, consumed: set[str]) -> None:
        self._ctx = ctx
        self._consumed = consumed
        # Declared+pushable property set = the keys of field_mapping, mapped to
        # physical property names (identity when a key maps to itself). An empty /
        # absent mapping means nothing is declared, so nothing is pushable.
        self._declared: dict[str, str] = dict(ctx.field_mapping or {})
        self._Filter = _import_filter()

    # ----- logical node walk ---------------------------------------------- #

    def compile_node(self, node: FilterNode | FilterClause) -> _Filters | None:
        """Compile a logical node or leaf to a ``_Filters`` or ``None``.

        ``None`` means "no pushable constraint" — the caller widens around it (an
        ``$and`` drops it, an ``$or`` / ``$not`` becomes wholly unpushable).
        """
        if isinstance(node, FilterClause):
            return self.compile_clause(node)
        if node.op == Op.AND:
            # AND-ing FEWER constraints only widens the candidate set, so an $and
            # may push the pushable conjuncts and DROP the unpushable ones — the
            # result stays a superset (the post-filter re-narrows). Drop the None
            # children (unpushable conjuncts).
            pushed = [c for c in (self.compile_node(child) for child in node.children) if c is not None]
            if not pushed:
                # Empty match-everything AND, or every conjunct was unpushable.
                return None
            if len(pushed) == 1:
                return pushed[0]
            return self._Filter.all_of(pushed)
        if node.op == Op.OR:
            # ALL-OR-NOTHING: dropping a disjunct would NARROW the union and could
            # false-exclude valid rows. Push the $or only if EVERY child is
            # pushable; otherwise the whole node is unpushable (post-filter handles
            # it) and we roll back any leaf this branch speculatively consumed.
            before = set(self._consumed)
            children = [self.compile_node(child) for child in node.children]
            if not children or any(c is None for c in children):
                self._consumed.intersection_update(before)  # roll back partial consumes
                return None
            pushed = [c for c in children if c is not None]  # all non-None by the guard
            if len(pushed) == 1:
                return pushed[0]
            return self._Filter.any_of(pushed)
        # Op.NOT — a negation is not monotone-narrowing on this backend (a
        # server-side NOT drops null/absent rows), so it is ALWAYS unpushable as a
        # whole. Consume nothing under it; the engine post-filters the negation.
        return None

    # ----- leaf key-kind split -------------------------------------------- #

    def compile_clause(self, clause: FilterClause) -> _Filters | None:
        """Dispatch a leaf: push it iff it is a declared property + narrowing op."""
        path = clause.path
        # Only a single-segment system key that is ALSO declared in field_mapping
        # is a candidate for pushdown. A metadata path (multi-segment, "metadata"
        # root) is never pushed — the chunk stores metadata as a serialized JSON
        # string property, and even on a native-map backend a metadata negation
        # has the same null-drop hazard, so it is left to the post-filter.
        if len(path) != 1 or path[0] not in SYSTEM_KEYS or path[0] not in self._declared:
            return self._unsupported(clause, "key is not a declared, pushable property")
        expr = self._compile_system_clause(clause)
        if expr is None:
            # The op is not monotone-narrowing ($ne / $nin / $exists:false / null).
            return self._unsupported(clause, "operator is not monotone-narrowing — not superset-safe to push")
        self._consumed.add(_path_str(path))
        return expr

    # ----- system-key leaves (declared properties) ----------------------- #

    def _compile_system_clause(self, clause: FilterClause) -> _Filters | None:
        """Build the ``Filter`` for a declared system key, or ``None`` if not pushable.

        Returns ``None`` for any op that is NOT a superset-safe push — a ``$ne`` /
        ``$nin`` / ``$not`` negation drops null/absent rows server-side, which would
        false-exclude rows the filter must keep. The caller routes a ``None`` to the
        post-filter.

        ``$exists`` and ``{k: null}`` are deliberately NOT pushed, even though the
        v4 API has ``is_none(...)`` for them. The post-filter oracle
        (``compile_python``) treats a system key as *always present* — so on a
        system key ``$exists:true`` is a constant TRUE (every record, including one
        with a null property, passes). Pushing ``is_none(False)`` would EXCLUDE the
        null-property rows the oracle keeps → a false-exclusion. Leaving ``$exists``
        / null to the post-filter is the only superset-safe choice and keeps this
        compiler exactly aligned with the routing-equivalence oracle.
        """
        prop = self._declared[clause.path[0]]
        by = self._Filter.by_property(prop)
        op = clause.op
        operand = clause.operand

        if op == Op.EXISTS:
            # See the method docstring: $exists on a system key is constant in the
            # post-filter oracle, so any server-side is_none() push would diverge
            # ($exists:true would false-exclude null-property rows). Not pushed.
            return None

        if op in (Op.IN, Op.NIN):
            if op == Op.NIN:
                # $nin is a negation — drops null/absent rows server-side. Unpushable.
                return None
            if not operand:
                # $in over an empty list matches NOTHING. There is no
                # match-nothing Filter primitive, and returning None would WIDEN to
                # match-all (wrong direction — could false-INCLUDE, but the
                # post-filter narrows so over-returning stays correct). Pushing
                # nothing and letting the post-filter (which knows $in [] == ∅)
                # exclude everything is the superset-safe choice.
                return None
            # $in → any_of([equal(x) for x in v]) — membership as an OR of scalar
            # equals ("prop == x1 OR prop == x2 ..."), which is exact $in semantics
            # on a SCALAR property. Deliberately NOT contains_any: that is a
            # Weaviate ARRAY / text-membership operator whose behavior on a scalar
            # DATE property (the only keys compile_weaviate ever pushes $in on) is
            # not guaranteed — it could under-return or raise at query time, which
            # the in-memory parity test cannot catch. The OR-of-equals is
            # unambiguously superset-exact. (The existing backend's contains_any is
            # for the `tags` TEXT_ARRAY property — an array case never pushed here.)
            members = [by.equal(_system_value(v)) for v in operand]
            if len(members) == 1:
                # A single-member $in needs no any_of wrapper — the bare equal IS
                # the membership test.
                return members[0]
            return self._Filter.any_of(members)

        if operand is None:
            # {k: null} ($eq None) / {$ne: None} hinge on null/absent. The oracle
            # treats a system key as always-present, so these are best left to the
            # post-filter rather than approximated by a server-side is_none() that
            # could diverge from the oracle's resolution. Not pushed.
            return None

        if op == Op.NE:
            # $ne drops null/absent rows server-side (Weaviate not_equal excludes
            # them). Unpushable — the post-filter applies it null-inclusively.
            return None

        if op == Op.EQ:
            # A bare-list (exact-array) $eq operand is carried as a tuple. A scalar
            # chunk property never equals an array, so pushing equal([...]) would
            # match nothing server-side (could false-EXCLUDE the rows the
            # post-filter would keep via list semantics). Leave it to the
            # post-filter rather than push a wrong-direction constraint.
            if isinstance(operand, tuple):
                return None
            return by.equal(_system_value(operand))

        # Range ops ($gt/$gte/$lt/$lte) — all monotone-narrowing.
        method = getattr(by, _RANGE_METHOD[op])
        return method(_system_value(operand))

    # ----- unsupported ---------------------------------------------------- #

    def _unsupported(self, clause: FilterClause, reason: str) -> _Filters | None:
        """Handle a clause this backend cannot (safely) push per ``ctx.on_unsupported``.

        ``"raise"`` raises the public :class:`RecallFilterUnsupportedError`;
        ``"split"`` (the mode this backend runs in) leaves the clause out of
        ``consumed_keys`` and contributes ``None`` — no server-side constraint, so
        the engine's post-filter re-checks it. A ``None`` never narrows the pushed
        filter, keeping it a superset of the true result.
        """
        if self._ctx.on_unsupported == "raise":
            raise RecallFilterUnsupportedError(_path_str(clause.path), reason)
        return None


# --------------------------------------------------------------------------- #
# Module-level helpers (no per-pass state).
# --------------------------------------------------------------------------- #


def _import_filter() -> Any:
    """Import the weaviate v4 ``Filter`` builder lazily.

    Kept out of module scope so the compiler module imports without the optional
    ``weaviate-client`` extra. Raises a clear :class:`RecallFilterUnsupportedError`
    (rather than a bare ``ImportError``) if a caller reaches the compiler without
    the extra installed — but in normal use only the weaviate engine (which
    depends on the extra) ever calls this.
    """
    try:
        from weaviate.classes.query import Filter
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RecallFilterUnsupportedError(
            "<weaviate>",
            "weaviate-client is not installed; install the 'weaviate' extra to push filters down",
        ) from exc
    return Filter


def _system_value(value: Any) -> Any:
    """Coerce a system-key operand to a weaviate-bindable value.

    A chunk stores datetime properties as ISO-8601 strings, so a
    :class:`DateLiteral` / :class:`~datetime.datetime` binds as its
    ``.isoformat()`` string (lexicographic compare). Other scalars pass through.
    Tuples (bare-list ``$eq`` exact-array operands) never reach here — they are
    routed to the post-filter in :meth:`_Builder._compile_system_clause`.
    """
    if isinstance(value, DateLiteral):
        return value.value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value
