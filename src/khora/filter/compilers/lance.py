"""SQLite recall-filter compiler for the sqlite_lance backend — ``@internal``.

Lowers a canonical :class:`~khora.filter.ast.FilterNode` to a single SQLite
``WHERE`` fragment against ``khora_chunks`` (no documents join — the filterable
document columns are denormalized onto the chunk row). The output is a
:class:`~khora.filter.registry.CompiledFilter` whose ``predicate`` is a SQLite
boolean *string* the backend ``AND``-s into its own ``WHERE``, paired with
``params == {"args": [...]}`` — an ordered list of positional binds, one per
``?`` placeholder in emit (depth-first) order. The host query is positional, so
binds cannot be named (``:name``) without mixing the two styles.

The compiler is the Layer-4 SQLite sibling of
:func:`~khora.filter.compilers.postgres.compile_postgres` and is checked for
row-set parity against :func:`~khora.filter.compilers.python.compile_python` (the
oracle). It speaks the four emission rules:

1. **Never abort.** A comparison against a wrong-typed or absent value yields
   SQL ``NULL``; every such leaf is wrapped in ``coalesce(<expr>, 0)`` so it
   reads ``0`` (false) instead — SQLite has no boolean type, so the totality
   sentinel is the integer ``0`` (cf. ``false()`` in Postgres / ``false`` in
   Cypher). A numeric compare on a string value never errors: the
   ``json_type`` gate excludes it first.
2. **Polarity.** Negations include absent/null/wrong-type rows: a system ``$ne``
   is ``(col IS NULL OR col <> ?)``; a ``$nin`` is ``(col IS NULL OR NOT col IN
   (...))``. Never drop a NULL row on a negation.
3. **Impossible pairs (narrow).** An ``$eq`` exact-array (tuple) operand against
   a scalar system column never matches (a scalar column is not a list), so it
   emits the constant ``0``; its ``$ne`` mirror emits ``1``. ``$in`` / ``$nin``
   are normal membership.
4. **Presence/null.** ``$exists`` and a ``{k: null}`` match resolve to constants
   on a system column (a column is always present in the row) and to
   ``json_type``-based tests on metadata: ``json_extract`` returns SQL ``NULL``
   for BOTH an absent path AND a stored JSON ``null``, so the absent-vs-null
   distinction is read off ``json_type`` (``NULL`` for absent, the string
   ``'null'`` for a JSON null value) — mirroring the oracle's ``MISSING`` vs
   ``None``.

**Totality (the rule that makes negation uniform).** ``NOT`` is compiled as
``(NOT (<child>))``; SQLite ``NOT NULL`` is ``NULL`` (drops the row), which would
violate Rule 2. So every leaf that *can* produce ``NULL`` (an absent / wrong-type
compare) is wrapped in ``coalesce(<expr>, 0)`` — a total boolean. The positive
use is unchanged (``0`` excludes exactly as ``NULL`` would), and a wrapping
``NOT`` then flips absent rows to true correctly. ``json_type ... IS NULL`` /
``EXISTS(...)`` are already total.

**Dates compare as ISO strings.** A chunk stores every datetime column as an
ISO-8601 string (``.isoformat()``), so a :class:`~datetime.datetime` /
:class:`DateLiteral` operand binds as its ``.isoformat()`` string and compares
lexicographically — ISO-8601's fixed-width, big-endian layout makes string order
agree with chronological order for the UTC-normalized values the validator
produces (same contract as :func:`compile_cypher`).

**Bool-vs-number gate.** SQLite's ``json_extract`` collapses a JSON ``true`` to
the integer ``1`` and ``false`` to ``0``, so a raw extracted-value compare would
wrongly match ``True == 1``. The oracle treats a bool and a number as unequal
(JSONB semantics). Every metadata scalar/range compare therefore type-gates on
``json_type`` (``'true'``/``'false'`` for a bool operand, ``'integer'``/``'real'``
for a number, ``'text'`` for a string) before comparing the value — mirroring
``compile_postgres``'s ``jsonb_typeof`` gate and the oracle's ``_comparable``.

**Metadata pushdown is gated on JSON1.** Metadata predicates require the SQLite
JSON1 functions (``json_extract`` / ``json_type`` / ``json_each``). When
``ctx.schema_capabilities.sqlite_json1`` is ``False``, every metadata leaf is
treated as unsupported (handled per ``on_unsupported`` — the backend drives
``"split"``, leaving metadata to the ``compile_python`` post-filter while system
keys still push down). Three metadata cases are *always* unsupported even with
JSON1, because they cannot match the oracle's row-set in pure SQL: the bare-blob
``$eq`` (whole-document equality), a sub-path ``object_equal`` (dict-operand)
``$eq`` — SQLite ``json()`` is key-order-sensitive but the oracle is not — and a
``$date`` metadata compare. They emit the non-constraining ``"1"`` and are left
out of ``consumed_keys`` so ``compile_python`` re-checks them.

**Superset-safety invariant.** ``compile_lance`` must NEVER wrongly *exclude* a
row — the ``compile_python`` post-filter only narrows what the pushdown returned.
Every NULL-producing leaf coalesces to ``0`` (excludes only on a true mismatch),
and any op whose SQL semantics risk over-narrowing is left unconsumed (emits
``"1"``) so the post-filter decides.

``@internal``. Reachable as ``khora.filter.compilers.lance.compile_lance`` for
khora's own engines; not re-exported from :mod:`khora.__init__`. Registration
against the ``("skeleton.sqlite_lance", "khora_chunks")`` key lives at the bottom
of the backend module (``engines/skeleton/backends/sqlite_lance.py``), keeping
this package registry-import-free.
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

__all__ = ["compile_lance"]


# SQLite comparison operator strings for the range ops, keyed by AST op.
_RANGE_OP = {
    Op.GT: ">",
    Op.GTE: ">=",
    Op.LT: "<",
    Op.LTE: "<=",
}

# json_type gate-token sets by operand kind. SQLite reports a JSON bool as
# 'true'/'false' and a number as 'integer'/'real'; a string is 'text'. The gate
# preserves the oracle's bool-vs-number distinction (a JSON true is NOT a number).
_NUMBER_TYPES = "('integer', 'real')"
_STRING_TYPES = "('text')"


def compile_lance(ast: FilterNode, ctx: CompileContext) -> CompiledFilter[str]:
    """Compile a canonical AST to a SQLite ``WHERE`` fragment + positional binds.

    ``ast`` is always a :class:`FilterNode` (the ``parse_to_ast`` root invariant).
    An empty ``AND`` (the match-everything root of a bare filter) compiles to the
    literal ``"1"`` (SQLite truthy). Binds are returned out-of-band in
    ``params == {"args": [...]}`` as an ordered list, one entry per ``?`` in
    depth-first emit order; the backend splices the fragment into its positional
    SELECT and extends its own bind list with ``args``.

    Column references are derived from ``ctx`` alone: the qualifier is
    ``ctx.table_alias`` if set else ``ctx.backend_target`` (so ``c.occurred_at`` on
    the aliased FTS-join path, ``khora_chunks.occurred_at`` on the unaliased
    vector post-fetch path), and ``field_mapping`` remaps a logical key to its
    physical column name (identity when ``None``), so the compiler embeds no
    engine schema.

    Honors ``ctx.on_unsupported``: on a clause this backend cannot express,
    ``"raise"`` raises :class:`RecallFilterUnsupportedError`; ``"split"`` omits it
    from ``consumed_keys`` and emits the non-constraining ``"1"`` so the engine
    post-filters it with ``compile_python``. Metadata predicates are unsupported
    when ``ctx.schema_capabilities.sqlite_json1`` is ``False`` (no JSON1
    functions), plus three cases that cannot match the oracle in SQL even with
    JSON1 (bare-blob ``$eq``, ``object_equal`` dict operands, ``$date`` compares).

    """
    consumed: set[str] = set()
    builder = _Builder(ctx=ctx, consumed=consumed)
    predicate = builder.compile_node(ast)
    return CompiledFilter(
        predicate=predicate,
        params={"args": builder.args},
        consumed_keys=frozenset(consumed),
        # canonical_hash over the whole AST. In on_unsupported="raise" mode every
        # leaf is consumed, so the whole tree == the consumed slice. When
        # split-mode is implemented end-to-end, hash the reconstructed consumed
        # sub-AST instead, not the whole tree.
        canonical_hash=canonical_hash(ast),
    )


def _path_str(path: tuple[str, ...]) -> str:
    """Render an AST path as the dotted key string used for diagnostics/consumed."""
    return ".".join(path)


def _jsonpath(segs: tuple[str, ...]) -> str:
    """Build a quoted SQLite JSONPath string (``$."a"."b"``) for ``segs``.

    Each segment is wrapped in double quotes with embedded ``"`` and ``\\``
    escaped, so a segment containing ``.``, ``$``, ``[``, whitespace, or a quote
    addresses the right key (an unquoted ``$.a.b`` would mis-parse such keys). The
    returned string is **always bound as a ``?`` parameter** to ``json_extract`` /
    ``json_type`` / ``json_each`` — it never enters the SQL text, so a
    user-supplied metadata key is not an injection surface (the framing here is
    only so the bound JSONPath literal is well-formed).
    """
    parts = []
    for seg in segs:
        escaped = seg.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'"{escaped}"')
    return "$." + ".".join(parts)


class _Builder:
    """Per-compile-pass state: the :class:`CompileContext`, consumed set, binds.

    Mirrors the cypher compiler's ``_Builder`` — a small object threading the
    ``ctx`` (column refs are derived from it), the ``consumed`` accumulator, and
    the positional ``args`` bind list every leaf appends to. Keeping them on one
    object avoids passing the triple through every recursion frame. Each
    ``compile_*`` method returns a SQLite boolean *string*; the logical-node walk
    composes them with ``AND`` / ``OR`` / ``NOT``.
    """

    def __init__(self, *, ctx: CompileContext, consumed: set[str]) -> None:
        self._ctx = ctx
        self._consumed = consumed
        self._qualifier = ctx.table_alias or ctx.backend_target
        self.args: list[Any] = []

    # ----- bind allocation ------------------------------------------------ #

    def _bind(self, value: Any) -> str:
        """Append ``value`` to the positional bind list and return a ``?`` token."""
        self.args.append(value)
        return "?"

    def _col(self, key: str) -> str:
        """Render a qualified column reference ``qualifier.col`` for a logical key.

        ``field_mapping`` remaps a logical key to its physical column name
        (identity when ``None``). ``key`` is always a controlled token (a system
        key from the closed ``SYSTEM_KEYS`` whitelist, or the literal
        ``"metadata"``) — never free user text, so the column token is not an
        injection surface. Metadata path segments only ever reach the bound
        JSONPath literal, never this token.
        """
        physical = (self._ctx.field_mapping or {}).get(key, key)
        return f"{self._qualifier}.{physical}"

    # ----- logical node walk ---------------------------------------------- #

    def compile_node(self, node: FilterNode | FilterClause) -> str:
        """Compile a logical node or leaf to a SQLite boolean string.

        **Split-mode soundness — "AND distributes; OR/NOT are all-or-nothing."**
        The match-all placeholder ``"1"`` an unconsumable leaf emits under
        ``on_unsupported="split"`` is superset-safe only in *positive* position:
        ``A AND 1`` ≡ ``A`` (still narrows correctly), but ``NOT (A OR 1)`` ≡
        ``NOT 1`` ≡ ``0`` — which would *drop every row the filter keeps*, breaking
        the superset invariant (the ``compile_python`` post-filter only narrows;
        it cannot add a wrongly-excluded row back). So an ``OR`` / ``NOT`` node is
        pushed down only when its **entire** subtree is consumable; otherwise the
        whole node emits ``"1"`` and consumes nothing, deferring it wholesale to
        the post-filter. An ``AND`` still handles each child independently — a
        non-consumable child becomes ``"1"`` and the consumable siblings still
        narrow.
        """
        if isinstance(node, FilterClause):
            return self.compile_clause(node)
        if node.op == Op.AND:
            if not node.children:
                # The empty match-everything root — no constraint.
                return "1"
            return "(" + " AND ".join(self.compile_node(c) for c in node.children) + ")"
        if node.op == Op.OR:
            if not node.children:
                # The validator forbids an empty $or; guard defensively.
                return "0"
            if self._ctx.on_unsupported == "split" and not self._consumable(node):
                # A non-consumable disjunct would compile to "1", making the whole
                # OR match-all here while the post-filter still narrows — safe in
                # positive position but it under-pushes silently AND becomes a
                # false-exclude if a parent NOT wraps it. Defer the whole OR. In
                # "raise" mode we instead descend so the offending leaf raises.
                return "1"
            return "(" + " OR ".join(self.compile_node(c) for c in node.children) + ")"
        # Op.NOT — exactly one child per the AST contract. Pushed only when the
        # child is fully consumable (then its SQL is exact + total in both
        # polarities, so the negation is sound); otherwise defer the whole NOT
        # (in "split" mode). In "raise" mode we descend so the offending leaf
        # raises rather than being silently swallowed by the all-or-nothing gate.
        if self._ctx.on_unsupported == "split" and not self._consumable(node):
            return "1"
        return f"(NOT ({self.compile_node(node.children[0])}))"

    def _consumable(self, node: FilterNode | FilterClause) -> bool:
        """True iff ``node``'s whole subtree compiles to SQL (no ``"1"`` deferral).

        A pure predicate — no bind allocation, no ``consumed`` mutation, no
        telemetry. A logical node is consumable iff every child is; a leaf is
        consumable iff it is a system key or a pushdownable metadata predicate
        (mirrors :meth:`_clause_unconsumable`). Used to keep an ``OR`` / ``NOT``
        all-or-nothing: a node is pushed only when nothing inside it would fall to
        the ``"1"`` placeholder.
        """
        if isinstance(node, FilterClause):
            return not self._clause_unconsumable(node)
        return all(self._consumable(c) for c in node.children)

    def _clause_unconsumable(self, clause: FilterClause) -> bool:
        """True iff this leaf cannot be pushed to SQL (would emit ``"1"``).

        Mirrors the leaf dispatch in :meth:`compile_clause` /
        :meth:`_compile_metadata_clause` without side effects: a system key always
        pushes; a metadata leaf is unconsumable when JSON1 is unavailable, or when
        it is one of the shapes that cannot match the oracle in SQL — the bare-blob
        ``$eq``, an ``object_equal`` (dict-operand) ``$eq`` / ``$ne``, a ``$date``
        compare, or a ``$in`` / ``$nin`` carrying a dict (object_equal) or ``None``
        (JSON-null member) element. Any non-system, non-metadata path is unconsumable.
        """
        path = clause.path
        if len(path) == 1 and path[0] in SYSTEM_KEYS:
            return False
        if not (path and path[0] == "metadata"):
            return True
        if not self._ctx.schema_capabilities.sqlite_json1:
            return True
        if len(path) == 1:
            # Bare-blob metadata $eq — never pushdownable.
            return True
        operand = clause.operand
        if isinstance(operand, (DateLiteral, dict)):
            # $date (any op) and object_equal dict ($eq / $ne) are deferred.
            return True
        if clause.op in (Op.IN, Op.NIN) and _has_unpushable_in_element(operand):
            # A dict or None $in / $nin element cannot be matched by the json_each
            # containment SQLite emits (a dict is object_equal-per-element; a None
            # must match a stored JSON null, but the type gate excludes nulls). Defer
            # the whole leaf to the compile_python post-filter (Rule 2: a genuine
            # capability gap, not a silent drop).
            return True
        return False

    # ----- leaf key-kind split -------------------------------------------- #

    def compile_clause(self, clause: FilterClause) -> str:
        """Dispatch a leaf on key-kind (system key vs metadata path)."""
        path = clause.path
        if len(path) == 1 and path[0] in SYSTEM_KEYS:
            expr = self._compile_system_clause(clause)
        elif path and path[0] == "metadata":
            if not self._ctx.schema_capabilities.sqlite_json1:
                # No JSON1 functions — every metadata leaf falls to the
                # compile_python post-filter (the backend drives "split").
                return self._unsupported(clause, "SQLite JSON1 functions are unavailable")
            expr = self._compile_metadata_clause(clause)
            if expr is None:
                # A metadata case that cannot match the oracle in SQL even with
                # JSON1 (bare-blob $eq, object_equal dict operand, $date compare).
                # Already reported as unsupported inside the metadata compiler.
                return self._unsupported(clause, "metadata predicate is not pushed down to SQLite")
        else:
            return self._unsupported(clause, "path is neither a system key nor a metadata path")
        self._consumed.add(_path_str(path))
        return expr

    # ----- system-key leaves (typed columns) ------------------------------ #

    def _compile_system_clause(self, clause: FilterClause) -> str:
        key = clause.path[0]
        col = self._col(key)
        op = clause.op
        operand = clause.operand

        if op == Op.EXISTS:
            # A system column is always present in the row — $exists is constant
            # (parity with compile_python / compile_postgres).
            return "1" if operand else "0"

        if op in (Op.IN, Op.NIN):
            if not operand:
                # Empty $in matches nothing; empty $nin matches everything.
                return "0" if op == Op.IN else "1"
            placeholders = ", ".join(self._bind(_system_value(v)) for v in operand)
            if op == Op.IN:
                # Made total so a wrapping field-level $not includes NULL rows.
                return f"coalesce({col} IN ({placeholders}), 0)"
            # $nin includes NULL rows (Rule 2 polarity). Already a total boolean.
            return f"({col} IS NULL OR NOT {col} IN ({placeholders}))"

        # Rule 3 (NARROW): an $eq EXACT-ARRAY (tuple) operand against a scalar
        # column never matches; its $ne mirror always matches.
        if op == Op.EQ and isinstance(operand, tuple):
            return "0"
        if op == Op.NE and isinstance(operand, tuple):
            return "1"

        if operand is None:
            # {k: null} → active null-or-missing match. For a system column,
            # "missing" is NULL. $ne null → IS NOT NULL. Both are total.
            if op == Op.NE:
                return f"({col} IS NOT NULL)"
            return f"({col} IS NULL)"

        value_bind = self._bind(_system_value(operand))
        if op == Op.NE:
            # Include NULL rows (Rule 2): a row whose column is NULL satisfies $ne.
            # Already a total boolean (no coalesce needed).
            return f"({col} IS NULL OR {col} <> {value_bind})"
        # $eq and the range ops compare directly. Wrap in coalesce(..., 0) so the
        # leaf is total: a NULL column reads 0 (same exclusion as the bare NULL
        # comparison), and a wrapping $not then flips a NULL-column row to true.
        symbol = "=" if op == Op.EQ else _RANGE_OP[op]
        return f"coalesce({col} {symbol} {value_bind}, 0)"

    # ----- metadata leaves (JSON-TEXT column) ----------------------------- #

    def _compile_metadata_clause(self, clause: FilterClause) -> str | None:
        """Compile a metadata leaf, or ``None`` if it is unsupported in SQL.

        Returns ``None`` (not a SQL string) for the three cases that cannot match
        the ``compile_python`` oracle's row-set in pure SQLite even with JSON1 —
        the caller turns ``None`` into the ``on_unsupported`` handling.
        """
        path = clause.path
        op = clause.op
        operand = clause.operand

        # Bare-blob metadata $eq (whole-document equality). SQLite json() is
        # key-order-sensitive whereas the oracle's whole-blob compare is
        # structural/order-insensitive — cannot match in SQL, so defer.
        if len(path) == 1:
            return None

        segs = path[1:]  # drop the leading "metadata" root
        json_col = self._col("metadata")

        if op == Op.EXISTS:
            # json_type is NULL for an absent path, non-NULL (incl. 'null') for a
            # present value — so presence is "json_type IS NOT NULL", matching the
            # oracle's "the path resolves" (an explicit JSON null counts present).
            present = f"json_type({json_col}, {self._jpath_bind(segs)}) IS NOT NULL"
            return f"({present})" if operand else f"(NOT ({present}))"

        if op in (Op.IN, Op.NIN):
            if not operand:
                # $in over ∅ matches nothing; $nin over ∅ matches everything.
                return "0" if op == Op.IN else "1"
            if _has_unpushable_in_element(operand):
                # A dict or None element cannot be matched by json_each containment
                # (a dict is object_equal-per-element; a None must match a stored JSON
                # null but the type gate excludes nulls). Defer the whole leaf to the
                # post-filter — None routes to ``on_unsupported`` (Rule 2: a capability
                # gap, not a silent drop). Mirrors :meth:`_clause_unconsumable`.
                return None
            chain = " OR ".join(self._md_contains(segs, v) for v in operand)
            if op == Op.IN:
                return f"({chain})"
            # $nin: NOT in the set. Containment is total (0 on absent/mismatch),
            # so the negation includes absent / wrong-type rows (Rule 2).
            return f"(NOT ({chain}))"

        if op == Op.EQ:
            if operand is None:
                return self._md_null(segs)
            if isinstance(operand, DateLiteral):
                # A $date metadata compare needs ISO-string parse-or-exclude that
                # SQLite cannot replicate to match the oracle — defer.
                return None
            if isinstance(operand, dict):
                # object_equal: SQLite json() is key-order-sensitive, the oracle
                # is not — defer to the post-filter.
                return None
            if isinstance(operand, tuple):
                # Exact-array equality. Arrays are ordered in both the oracle and
                # SQLite, so json()-normalized text equality matches.
                return self._md_exact_array(segs, operand)
            # Array-aware scalar containment (a scalar matches the node or a list
            # element). Mirrors the oracle's _md_contains.
            return self._md_contains(segs, operand)
        if op == Op.NE:
            if operand is None:
                # $ne null → present AND not a JSON null value.
                return f"(NOT {self._md_null(segs)})"
            if isinstance(operand, DateLiteral):
                return None
            if isinstance(operand, dict):
                return None
            if isinstance(operand, tuple):
                return f"(NOT {self._md_exact_array(segs, operand)})"
            # Negate the (total) positive containment so absent / wrong-type rows
            # are included (Rule 2).
            return f"(NOT {self._md_contains(segs, operand)})"

        # Range ops ($gt/$gte/$lt/$lte). A $date operand cannot be replicated to
        # match the oracle — defer; otherwise type-gated scalar compare.
        if isinstance(operand, DateLiteral):
            return None
        return self._md_range(segs, op, operand)

    def _jpath_bind(self, segs: tuple[str, ...]) -> str:
        """Bind the JSONPath for ``segs`` as a ``?`` param and return the token."""
        return self._bind(_jsonpath(segs))

    def _md_contains(self, segs: tuple[str, ...], value: Any) -> str:
        """Array-aware, type-strict containment — total boolean (0 on absent).

        Matches the oracle's ``_md_contains``: a scalar matches the node directly
        OR an element of a stored list node. ``json_each`` over the path yields one
        row for a scalar node, N rows for an array node, and zero rows for an absent
        path — so a single ``EXISTS`` subquery covers scalar-eq, array-contains, and
        the absent case at once. Its per-element ``type`` column is gated against
        the operand's Python type, which preserves the oracle's bool-vs-number
        distinction (a JSON ``true`` and the number ``1`` both surface ``value`` 1,
        but their ``type`` differs — ``'true'`` vs ``'integer'``), even for an
        element nested inside an array. The wrapping ``coalesce(..., 0)`` keeps the
        leaf total so a wrapping ``NOT`` includes absent / wrong-type rows.
        """
        json_col = self._col("metadata")
        gate, is_bool = _operand_type_gate(value)
        # A JSON bool surfaces in ``json_each.value`` as the integer 1 / 0, so bind
        # that (not a "true"/"false" string); the ``type`` gate is what enforces
        # the bool-vs-number distinction. Other scalars bind via _jsonable_scalar.
        bound_value = (1 if value else 0) if is_bool else _jsonable_scalar(value)
        contains = (
            # json_col is a controlled token; gate is a module constant; the path +
            # value are bound ``?`` params (no user text reaches the SQL text).
            f"EXISTS(SELECT 1 FROM json_each({json_col}, {self._jpath_bind(segs)}) je "  # noqa: S608
            f"WHERE je.value = {self._bind(bound_value)} AND je.type IN {gate})"
        )
        return f"coalesce(({contains}), 0)"

    def _md_exact_array(self, segs: tuple[str, ...], operand: tuple[Any, ...]) -> str:
        """Exact-array equality (bare-list ``$eq``) — total boolean.

        ``json(json_extract(...))`` re-renders the stored node in canonical form
        and compares to the ``json()``-normalized operand array. Arrays are
        ordered in both the oracle and SQLite, so this matches. The node must be a
        JSON array (gated on ``json_type``) and the path must resolve; otherwise
        ``coalesce`` reads ``0``.
        """
        json_col = self._col("metadata")
        import json as _json

        operand_json = _json.dumps([_jsonable_scalar(item) for item in operand])
        gate = (
            f"CASE WHEN json_type({json_col}, {self._jpath_bind(segs)}) = 'array' "
            f"THEN json(json_extract({json_col}, {self._jpath_bind(segs)})) = json({self._bind(operand_json)}) "
            f"ELSE 0 END"
        )
        return f"coalesce(({gate}), 0)"

    def _md_range(self, segs: tuple[str, ...], op: Op, operand: Any) -> str:
        """A metadata range op — ``json_type``-gated scalar compare, total.

        The stored node must match the operand's type-gate (number vs number,
        string vs string) before comparing; a mismatch or absent node reads ``0``
        via ``coalesce`` (Rule 1). Mirrors ``compile_postgres._md_range`` and the
        oracle's ``_md_range_compare``. A bool operand is not a range operand (the
        validator never produces one), so only the number / string gates apply.
        """
        json_col = self._col("metadata")
        gate, _is_bool = _operand_type_gate(operand)
        symbol = _RANGE_OP[op]
        expr = (
            f"json_type({json_col}, {self._jpath_bind(segs)}) IN {gate} "
            f"AND json_extract({json_col}, {self._jpath_bind(segs)}) {symbol} {self._bind(_jsonable_scalar(operand))}"
        )
        return f"coalesce(({expr}), 0)"

    def _md_null(self, segs: tuple[str, ...]) -> str:
        """Active null-or-missing match — total boolean.

        ``json_type`` is SQL ``NULL`` for an absent path and the string ``'null'``
        for a stored JSON null. ``{k: null}`` matches both — mirroring the oracle's
        ``_md_is_null`` (``MISSING`` or ``None``).
        """
        json_col = self._col("metadata")
        # Two json_type occurrences → two ``?`` placeholders → bind the path twice
        # (each ``?`` needs its own positional arg, in emit order).
        absent = f"json_type({json_col}, {self._jpath_bind(segs)}) IS NULL"
        is_null = f"json_type({json_col}, {self._jpath_bind(segs)}) = 'null'"
        return f"({absent} OR {is_null})"

    # ----- unsupported ---------------------------------------------------- #

    def _unsupported(self, clause: FilterClause, reason: str) -> str:
        """Handle a clause this backend cannot express per ``ctx.on_unsupported``.

        ``"raise"`` raises the public :class:`RecallFilterUnsupportedError`;
        ``"split"`` leaves the clause out of ``consumed_keys`` (the engine
        post-filters it with ``compile_python``) and emits the non-constraining
        ``"1"`` so it does not narrow the result set.
        """
        path = _path_str(clause.path)
        if self._ctx.on_unsupported == "raise":
            raise RecallFilterUnsupportedError(path, reason)
        return "1"


# --------------------------------------------------------------------------- #
# Module-level helpers (no per-pass state).
# --------------------------------------------------------------------------- #


def _has_unpushable_in_element(operand: Any) -> bool:
    """True iff a ``$in`` / ``$nin`` operand sequence has an element SQLite can't push.

    Two element kinds cannot be matched by the ``json_each`` containment the SQLite
    compiler emits, so the whole leaf is deferred to the ``compile_python``
    post-filter (Rule 2: a documented capability gap, not a silent drop):

    * a **dict** element is per-element ``object_equal`` (the oracle matches an array
      element OR the scalar node EXACTLY equal to the dict); ``json_each`` binds the
      element as a JSON-text scalar, so it can neither represent nor match it;
    * a **``None``** (JSON-null) element must match a stored JSON ``null``, but the
      containment gates ``je.type`` to the operand's *non-null* type (``'text'`` /
      number / bool), so a ``null`` member is never matched array-aware. The oracle's
      ``_md_contains(node, None)`` keeps a stored ``null``, so pushing this would
      under-return and — because the leaf would be marked consumed — never be
      re-checked.

    ``operand`` is the normalized ``$in`` / ``$nin`` sequence (a tuple / list); a
    non-sequence (defensive) yields ``False``.
    """
    if not isinstance(operand, (tuple, list)):
        return False
    return any(item is None or isinstance(item, dict) for item in operand)


def _system_value(value: Any) -> Any:
    """Coerce a system-key operand to a SQLite-bindable value.

    A chunk stores datetime columns as ISO-8601 strings, so a :class:`DateLiteral`
    / :class:`~datetime.datetime` binds as its ``.isoformat()`` string
    (lexicographic compare). Other scalars pass through.
    """
    if isinstance(value, DateLiteral):
        return value.value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _is_number(value: Any) -> bool:
    """True for an int / float operand, excluding bool (a number type-gate).

    ``bool`` is an ``int`` subclass; the §4 number gate treats a bool as a
    boolean, never a number, so it is excluded here (matching the oracle's
    ``_is_number`` and ``compile_postgres._operand_json_type``).
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _operand_type_gate(operand: Any) -> tuple[str, bool]:
    """Map an operand to its ``json_type`` gate-token set + a bool flag.

    Returns ``(gate, is_bool)``: ``is_bool`` is ``True`` for a bool operand (the
    caller binds the collapsed ``json_each`` value — 1 / 0 — and relies on this
    ``('true', 'false')`` gate to distinguish a JSON bool from the number 1 / 0);
    otherwise the gate is the number or string ``json_type`` set. ``bool`` is
    checked before ``int`` (``bool`` is an ``int`` subclass).
    """
    if isinstance(operand, bool):
        return "('true', 'false')", True
    if _is_number(operand):
        return _NUMBER_TYPES, False
    return _STRING_TYPES, False


def _jsonable_scalar(value: Any) -> Any:
    """Coerce a scalar comparison operand to a SQLite-bindable value.

    A :class:`DateLiteral` / :class:`~datetime.datetime` renders as its ISO-8601
    string (matching how a metadata date is stored); other scalars pass through.
    Used for the value side of ``json_extract`` / ``json_each`` compares.
    """
    if isinstance(value, DateLiteral):
        return value.value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value
