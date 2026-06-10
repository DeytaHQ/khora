"""Neo4j Cypher recall-filter compiler — ``@internal``.

Lowers a canonical :class:`~khora.filter.ast.FilterNode` to a single Cypher
``WHERE`` fragment against a recall ``Chunk`` node (no relationship traversal —
the filterable document keys are projected onto the chunk node). The output is a
:class:`~khora.filter.registry.CompiledFilter` whose ``predicate`` is a Cypher
boolean *string* the engine splices into its own ``WHERE``, paired with a
``params`` dict of ``$name`` binds the engine passes through to the driver.

The compiler is the Layer-4 half of the §4/§7 filter contract. It speaks
the four emission rules:

1. **Never abort.** Cypher comparisons against a property of the wrong type or
   an absent property evaluate to ``null`` (not an error), so a numeric compare
   on a string-valued property never blows up — it yields ``null`` (then
   ``false`` via ``coalesce``) instead of erroring.
2. **Polarity.** Negations include absent/null rows: a system ``$ne`` is
   ``(var.key IS NULL OR var.key <> $v)``; a ``$nin`` is ``(var.key IS NULL OR
   NOT var.key IN $vs)``. Never drop a NULL row on a negation.
3. **Impossible pairs (narrow).** ONLY an ``$eq`` exact-array (tuple) operand
   against a scalar node property is compiled as a normal ``=`` against a list
   bind — a scalar property never equals a list, so Cypher yields ``false``; no
   special-cased constant is emitted. ``$in`` / ``$nin`` are normal membership.
4. **Presence/null.** ``$exists`` on a system key is a CONSTANT (``true`` /
   ``false``) — a system key is treated as always-present (the oracle's axiom), so a
   presence test would diverge whenever the property is genuinely unset on the node.
   A ``{k: null}`` match resolves to ``var.key IS NULL`` (Neo4j has no stored-null
   property — an absent property *is* null).

**Totality (the rule that makes negation uniform).** ``NOT`` is compiled as
``(NOT (<child>))``; Cypher ``NOT null`` is ``null`` (drops the row), which would
violate Rule 2. So every leaf that *can* produce ``null`` (an absent property in
an ``$eq`` / range / ``$in`` compare) is wrapped in ``coalesce(<expr>, false)``
— a total boolean. The positive use is unchanged (``false`` excludes exactly as
``null`` would), and a wrapping ``NOT`` then flips absent rows to ``true``
correctly. ``IS NULL`` / ``IS NOT NULL`` are already total.

**Split-mode soundness — "AND distributes; OR/NOT are all-or-nothing."** Cypher
pushes ONLY system keys; a metadata leaf is deferred (it emits the
non-constraining ``"true"`` under ``on_unsupported="split"``). That placeholder is
superset-safe in positive position (``A AND true`` ≡ ``A`` still narrows), but
``NOT (A OR true)`` ≡ ``NOT true`` ≡ ``false`` would *drop every row the filter
keeps* — the ``compile_python`` post-filter only narrows, so it could not add the
wrongly-excluded rows back. So an ``OR`` / ``NOT`` node is pushed only when its
*entire* subtree is consumable; otherwise the whole node defers to ``"true"`` and
consumes nothing. ``AND`` still distributes (a non-consumable child becomes
``"true"`` and the consumable siblings narrow). Matches the same guard in
:func:`~khora.filter.compilers.lance.compile_lance`.

**Dates compare as ISO strings.** A chunk stores every datetime property as an
ISO-8601 string (``.isoformat()``), so a :class:`~datetime.datetime` /
:class:`DateLiteral` operand binds as its ``.isoformat()`` string and compares
lexicographically — ISO-8601's fixed-width, big-endian layout makes string order
agree with chronological order for the UTC-normalized values the validator
produces.

``@internal``. Reachable as ``khora.filter.compilers.cypher.compile_cypher`` for
khora's own engines; not re-exported from :mod:`khora.__init__`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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

__all__ = ["compile_cypher"]


# Cypher comparison operator strings for the range ops, keyed by AST op.
_RANGE_OP = {
    Op.GT: ">",
    Op.GTE: ">=",
    Op.LT: "<",
    Op.LTE: "<=",
}


def compile_cypher(ast: FilterNode, ctx: CompileContext) -> CompiledFilter[str]:
    """Compile a canonical AST to a Cypher ``WHERE`` fragment + bind dict.

    ``ast`` is always a :class:`FilterNode` (the ``parse_to_ast`` root invariant).
    An empty ``AND`` (the match-everything root of a bare filter) compiles to the
    literal ``"true"``. Binds are returned out-of-band in ``params`` as a
    ``{name: value}`` dict; each ``$name`` placeholder in the predicate string has
    a matching entry. Bind names are generated as ``{param_namespace}_{n}`` so
    they cannot collide with the engine's own query parameters.

    Property references are derived from ``ctx`` alone: the node variable is
    ``ctx.table_alias`` if set else ``"c"``, and ``field_mapping`` remaps a logical
    key to its physical property name (identity when ``None``), so the compiler
    embeds no engine schema. Honors ``ctx.on_unsupported``: on a clause this
    backend cannot express, ``"raise"`` raises :class:`RecallFilterUnsupportedError`;
    ``"split"`` omits it from ``consumed_keys`` and emits a non-constraining
    ``"true"`` placeholder.
    """
    consumed: set[str] = set()
    builder = _Builder(ctx=ctx, consumed=consumed)
    predicate = builder.compile_node(ast)
    return CompiledFilter(
        predicate=predicate,
        params=builder.params,
        consumed_keys=frozenset(consumed),
        # canonical_hash over the whole AST. In on_unsupported="raise" mode (the
        # only mode used today) every leaf is consumed, so the whole tree == the
        # consumed slice. When split-mode is implemented, hash the reconstructed
        # consumed sub-AST instead, not the whole tree.
        canonical_hash=canonical_hash(ast),
    )


def _path_str(path: tuple[str, ...]) -> str:
    """Render an AST path as the dotted key string used for diagnostics/consumed."""
    return ".".join(path)


class _Builder:
    """Per-compile-pass state: the :class:`CompileContext`, consumed set, binds.

    A compile pass threads the ``ctx`` (the node variable and property names are
    derived from it), the ``consumed`` accumulator, and the ``params`` bind dict
    every leaf appends to. Keeping them on a small object avoids passing the
    triple through every recursion frame. A monotonic counter names each bind
    ``{param_namespace}_{n}`` so two clauses on the same key never collide.
    """

    def __init__(self, *, ctx: CompileContext, consumed: set[str]) -> None:
        self._ctx = ctx
        self._consumed = consumed
        self._var = ctx.table_alias or "c"
        self.params: dict[str, Any] = {}
        self._counter = 0

    # ----- bind allocation ------------------------------------------------ #

    def _bind(self, value: Any) -> str:
        """Allocate a fresh ``$name`` placeholder bound to ``value``."""
        name = f"{self._ctx.param_namespace}_{self._counter}"
        self._counter += 1
        self.params[name] = value
        return f"${name}"

    def _prop(self, key: str) -> str:
        """Render a node-property reference ``var.key`` for a logical key.

        ``field_mapping`` remaps a logical key to its physical property name
        (identity when ``None``). ``key`` is always a controlled token (a system
        key from the closed ``SYSTEM_KEYS`` whitelist) — never free user text, so
        the property access is not an injection surface. Metadata leaves never
        reach this method; they are handled separately.
        """
        physical = (self._ctx.field_mapping or {}).get(key, key)
        return f"{self._var}.{physical}"

    # ----- logical node walk ---------------------------------------------- #

    def compile_node(self, node: FilterNode | FilterClause) -> str:
        """Compile a logical node or leaf to a Cypher boolean string.

        **Split-mode soundness — "AND distributes; OR/NOT are all-or-nothing."**
        The match-all placeholder ``"true"`` an unsupported metadata leaf emits
        under ``on_unsupported="split"`` is superset-safe only in *positive*
        position: ``A AND true`` ≡ ``A`` (still narrows correctly), but
        ``NOT (A OR true)`` ≡ ``NOT true`` ≡ ``false`` — which would *drop every
        row the filter keeps*, breaking the superset invariant (the
        ``compile_python`` post-filter only narrows; it cannot add a
        wrongly-excluded row back). So an ``OR`` / ``NOT`` node is pushed down only
        when its **entire** subtree is consumable; otherwise the whole node emits
        ``"true"`` and consumes nothing, deferring it wholesale to the post-filter.
        An ``AND`` still handles each child independently — a non-consumable child
        becomes ``"true"`` and the consumable siblings still narrow.
        """
        if isinstance(node, FilterClause):
            return self.compile_clause(node)
        if node.op == Op.AND:
            if not node.children:
                # The empty match-everything root — no constraint.
                return "true"
            return "(" + " AND ".join(self.compile_node(c) for c in node.children) + ")"
        if node.op == Op.OR:
            if not node.children:
                # The validator forbids an empty $or; guard defensively.
                return "false"
            if self._ctx.on_unsupported == "split" and not self._consumable(node):
                # A non-consumable disjunct would compile to "true", making the
                # whole OR match-all here while the post-filter still narrows —
                # safe in positive position but it under-pushes silently AND
                # becomes a false-exclude if a parent NOT wraps it. Defer the whole
                # OR. In "raise" mode we instead descend so the offending leaf
                # raises.
                return "true"
            return "(" + " OR ".join(self.compile_node(c) for c in node.children) + ")"
        # Op.NOT — exactly one child per the AST contract. Pushed only when the
        # child is fully consumable (then its Cypher is exact + total in both
        # polarities, so the negation is sound); otherwise defer the whole NOT
        # (in "split" mode). In "raise" mode we descend so the offending leaf
        # raises rather than being silently swallowed by the all-or-nothing gate.
        if self._ctx.on_unsupported == "split" and not self._consumable(node):
            return "true"
        # Leaves are built total (never null) so this negation flips absent rows
        # correctly.
        return f"(NOT ({self.compile_node(node.children[0])}))"

    def _consumable(self, node: FilterNode | FilterClause) -> bool:
        """True iff ``node``'s whole subtree compiles to Cypher (no ``"true"`` defer).

        A pure predicate — no bind allocation, no ``consumed`` mutation. A logical
        node is consumable iff every child is; a leaf is consumable iff it is a
        system key (a metadata path leaf hits ``_unsupported``). Used to keep an
        ``OR`` / ``NOT`` all-or-nothing: a node is pushed only when nothing inside
        it would fall to the ``"true"`` placeholder. Mirrors the same helper in
        :mod:`khora.filter.compilers.lance`.
        """
        if isinstance(node, FilterClause):
            return not self._clause_unconsumable(node)
        return all(self._consumable(c) for c in node.children)

    def _clause_unconsumable(self, clause: FilterClause) -> bool:
        """True iff this leaf cannot be pushed to Cypher (would emit ``"true"``).

        Mirrors the leaf dispatch in :meth:`compile_clause` without side effects: a
        lone system key pushes down; everything else (a metadata path, or any
        structurally-unknown path) is unsupported and would fall to the
        ``"true"`` placeholder under split.
        """
        path = clause.path
        return not (len(path) == 1 and path[0] in SYSTEM_KEYS)

    # ----- leaf key-kind split -------------------------------------------- #

    def compile_clause(self, clause: FilterClause) -> str:
        """Dispatch a leaf on key-kind (system key vs metadata path)."""
        path = clause.path
        if len(path) == 1 and path[0] in SYSTEM_KEYS:
            expr = self._compile_system_clause(clause)
        elif path and path[0] == "metadata":
            # Neo4j stores metadata as a serialized JSON string property, not a
            # nested map, so a metadata sub-path is not pushdownable here. Defer to
            # the engine's post-filter per ``ctx.on_unsupported``.
            return self._unsupported(clause, "metadata predicates are not pushed down to Cypher")
        else:
            return self._unsupported(clause, "path is neither a system key nor a metadata path")
        self._consumed.add(_path_str(path))
        return expr

    # ----- system-key leaves (typed node properties) ---------------------- #

    def _compile_system_clause(self, clause: FilterClause) -> str:
        key = clause.path[0]
        prop = self._prop(key)
        op = clause.op
        operand = clause.operand

        if op == Op.EXISTS:
            # A system key is treated as ALWAYS PRESENT (the oracle's axiom: an
            # absent system value resolves to None, which still counts as
            # present-with-null for $exists). So $exists on a system key is a
            # CONSTANT — true / false — matching compile_python / compile_postgres /
            # compile_lance, NOT a presence test (``IS NOT NULL``). A presence test
            # would diverge whenever the property is genuinely unset on the node
            # (e.g. an unwritten denormalized doc key), which the oracle keeps under
            # $exists:true. The chunk row itself always exists.
            return "true" if operand else "false"

        if op in (Op.IN, Op.NIN):
            if not operand:
                # An empty operand list is a valid filter with a defined row-set
                # (the validator accepts it). Positive $in over ∅ matches nothing;
                # $nin over ∅ matches everything. Cypher's IN already yields this
                # (``x IN []`` is false for every x, including null), but emit the
                # constant explicitly — it mirrors the Postgres compiler and makes
                # the contract self-evident without relying on the empty-list rule.
                return "false" if op == Op.IN else "true"
            values_bind = self._bind([_system_value(v) for v in operand])
            if op == Op.IN:
                # Made total so a wrapping field-level $not includes NULL rows.
                return f"coalesce({prop} IN {values_bind}, false)"
            # $nin includes NULL rows (Rule 2 polarity). Already a total boolean.
            return f"({prop} IS NULL OR NOT {prop} IN {values_bind})"

        # Rule 3 (NARROW): a bare list on a system key lowers to EQ-with-tuple. A
        # scalar property can never equal a list, so the compare yields false at
        # query time — no special-cased constant is needed (the list binds like
        # any other operand). The $ne complement is the polarity mirror and stays
        # uniform with the scalar $ne form below. $in / $nin are NOT impossible —
        # they were already handled above as normal membership.
        if operand is None:
            # {k: null} → active null-or-missing match. An absent property IS
            # null on a chunk node. $ne null → IS NOT NULL. Both are total.
            if op == Op.NE:
                return f"({prop} IS NOT NULL)"
            return f"({prop} IS NULL)"

        value_bind = self._bind(_system_value(operand))
        if op == Op.NE:
            # Include NULL rows (Rule 2): a row whose property is null satisfies
            # $ne. Already a total boolean (no coalesce needed).
            return f"({prop} IS NULL OR {prop} <> {value_bind})"
        # $eq and the range ops compare directly. Wrap in coalesce(..., false) so
        # the leaf is a total boolean: a null property yields false (same
        # exclusion as the bare null comparison on the positive side), and a
        # wrapping field-level $not then flips a null-property row to true — which
        # is what $not($eq) / $not($gt) must do (parity with the explicit $ne).
        symbol = "=" if op == Op.EQ else _RANGE_OP[op]
        return f"coalesce({prop} {symbol} {value_bind}, false)"

    # ----- unsupported ---------------------------------------------------- #

    def _unsupported(self, clause: FilterClause, reason: str) -> str:
        """Handle a clause this backend cannot express per ``ctx.on_unsupported``.

        ``"raise"`` raises the public :class:`RecallFilterUnsupportedError`;
        ``"split"`` leaves the clause out of ``consumed_keys`` (the engine
        post-filters it) and emits a non-constraining ``"true"`` so it does not
        narrow the result set.
        """
        path = _path_str(clause.path)
        if self._ctx.on_unsupported == "raise":
            raise RecallFilterUnsupportedError(path, reason)
        return "true"


# --------------------------------------------------------------------------- #
# Module-level helpers (no per-pass state).
# --------------------------------------------------------------------------- #


def _system_value(value: Any) -> Any:
    """Coerce a system-key operand to a driver-bindable value.

    A chunk stores datetime properties as ISO-8601 strings, so a
    :class:`DateLiteral` / :class:`~datetime.datetime` binds as its
    ``.isoformat()`` string (lexicographic compare). An exact-array ``tuple`` (a
    bare-list ``$eq`` operand) binds as a ``list`` — the driver's list value type
    — with each element coerced the same way. Other scalars pass through.
    """
    if isinstance(value, DateLiteral):
        return value.value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_system_value(item) for item in value]
    return value
