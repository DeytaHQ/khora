"""PostgreSQL recall-filter compiler — ``@internal``.

Lowers a canonical :class:`~khora.filter.ast.FilterNode` to a single-table
SQLAlchemy ``WHERE`` predicate against ``khora_chunks`` (no documents join — the
filterable document columns are denormalized onto the chunk row). The output is a
:class:`~khora.filter.registry.CompiledFilter` whose ``predicate`` is a SQLAlchemy
``ColumnElement[bool]`` the engine ``AND``-s into its own conditions list.

The compiler is the Layer-4 half of the §4/§7 filter contract. It speaks
the four emission rules:

1. **Never abort.** A metadata range op type-gates its cast through
   ``jsonb_typeof`` so a string/bool value never blows up a numeric compare — the
   gate yields ``NULL`` (then ``FALSE`` via ``coalesce``) instead of erroring.
2. **Polarity.** Negations include absent/null/wrong-type rows: metadata ``$ne``
   matches a row missing the key; a system ``$ne`` is ``col IS NULL OR col <> v``.
   Never drop a NULL row on a negation.
3. **Impossible pairs (narrow).** ONLY an ``$eq`` exact-array (tuple) operand
   against a scalar system column emits ``sqlalchemy.false()`` (unsatisfiable —
   a scalar column never equals an array). ``$in`` / ``$nin`` are normal
   membership, not impossible.
4. **Presence/null.** ``$exists`` and a ``{k: null}`` match resolve to JSONB
   presence (``?`` / ``#>``) on metadata and to a constant / ``IS NULL`` on a
   system column.

**Totality (the rule that makes negation uniform).** ``NOT`` is compiled as
``sqlalchemy.not_(child)``; SQL ``NOT NULL`` is ``NULL`` (drops the row), which
would violate Rule 2. So every metadata leaf that *can* produce ``NULL`` (the
scalar-range gate, the ``$date`` gate, an exact-array equality whose path is
absent) is wrapped in ``coalesce(<expr>, false())`` — a total boolean. The
positive use is unchanged (``FALSE`` excludes exactly as ``NULL`` would), and a
wrapping ``NOT`` then flips absent rows to ``TRUE`` correctly. Containment /
presence (``@>`` / ``?``) are already total.

``@internal``. Reachable as ``khora.filter.compilers.postgres.compile_postgres``
for khora's own engines; not re-exported from :mod:`khora.__init__`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import ColumnElement
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

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
from khora.filter.telemetry import record_unindexed_metadata

__all__ = ["compile_postgres"]


def _col(ctx: CompileContext, logical_name: str) -> ColumnElement[Any]:
    """Resolve a logical key to a SQLAlchemy column reference from ``ctx`` alone.

    The compiler never imports the engine's ``Table`` — it receives schema *shape*
    through ``ctx`` and emits a raw column token, so one ``compile_postgres`` can
    serve any single-table schema (e.g. the legacy ``chunks``/``documents`` tables)
    just by varying ``field_mapping`` / ``table_alias``.

    ``field_mapping`` remaps a logical key to its physical column name (identity
    when ``None``); the qualifier is ``table_alias`` if set else ``backend_target``.
    ``logical_name`` is always a controlled token (a system key from the closed
    ``SYSTEM_KEYS`` whitelist, or the literal ``"metadata"``) — never free user
    text, so the f-string column token is not an injection surface. (User-supplied
    metadata path segments only ever reach the bound ``#>`` path literal, never
    this column token.)
    """
    physical = (ctx.field_mapping or {}).get(logical_name, logical_name)
    qualifier = ctx.table_alias or ctx.backend_target
    return sa.literal_column(f"{qualifier}.{physical}")


# Python operator callables for the range ops, keyed by AST op.
_RANGE_FN = {
    Op.GT: lambda a, b: a > b,
    Op.GTE: lambda a, b: a >= b,
    Op.LT: lambda a, b: a < b,
    Op.LTE: lambda a, b: a <= b,
}


def compile_postgres(ast: FilterNode, ctx: CompileContext) -> CompiledFilter[ColumnElement[bool]]:
    """Compile a canonical AST to a ``khora_chunks`` SQLAlchemy ``WHERE`` predicate.

    ``ast`` is always a :class:`FilterNode` (the ``parse_to_ast`` root invariant).
    An empty ``AND`` (the match-everything root of a bare filter) compiles to
    ``sqlalchemy.true()``. Binds are carried inline inside the expression
    (asyncpg renders them as ``$N`` at execute time), so ``params`` is always
    empty — the field exists for backends that bind out-of-band.

    Column references are derived from ``ctx`` alone (``backend_target`` /
    ``table_alias`` qualify, ``field_mapping`` remaps) via :func:`_col`, so the
    compiler embeds no engine schema. Honors ``ctx.on_unsupported``: on a clause
    this backend cannot express (should not happen post-validation), ``"raise"``
    raises :class:`RecallFilterUnsupportedError`; ``"split"`` omits it from
    ``consumed_keys`` and emits a non-constraining placeholder.
    """
    consumed: set[str] = set()
    builder = _Builder(ctx=ctx, consumed=consumed)
    predicate = builder.compile_node(ast)
    return CompiledFilter(
        predicate=predicate,
        params={},
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
    """Per-compile-pass state: the :class:`CompileContext` and consumed set.

    A compile pass threads two things every helper needs — the ``ctx`` (column
    refs are derived from it via :func:`_col`) and the ``consumed`` accumulator.
    Keeping them on a small object avoids passing the pair through every recursion
    frame.
    """

    def __init__(self, *, ctx: CompileContext, consumed: set[str]) -> None:
        self._ctx = ctx
        self._consumed = consumed
        # The metadata column is nullable. Coalesce a NULL blob to an empty JSONB
        # object once here so every downstream operator (`?`, `@>`, `#>`, `=`) is
        # total against a NULL row: `'{}' ? k` / `'{}' @> x` return FALSE (not
        # NULL) and `'{}' #> path` returns NULL for a missing path — exactly the
        # absent-key semantics. Without this, `NOT (metadata ? k)` on a NULL blob
        # would be `NOT NULL = NULL` and wrongly drop the row from a negation.
        self._md = sa.func.coalesce(_col(ctx, "metadata"), sa.cast(sa.literal("{}"), JSONB))

    # ----- logical node walk ---------------------------------------------- #

    def compile_node(self, node: FilterNode | FilterClause) -> ColumnElement[bool]:
        """Compile a logical node or leaf to a boolean expression."""
        if isinstance(node, FilterClause):
            return self.compile_clause(node)
        if node.op == Op.AND:
            if not node.children:
                # The empty match-everything root — no constraint.
                return sa.true()
            return sa.and_(*(self.compile_node(c) for c in node.children))
        if node.op == Op.OR:
            if not node.children:
                # The validator forbids an empty $or; guard defensively.
                return sa.false()
            return sa.or_(*(self.compile_node(c) for c in node.children))
        # Op.NOT — exactly one child per the AST contract. Leaves are built total
        # (never NULL) so this negation flips absent rows correctly.
        return sa.not_(self.compile_node(node.children[0]))

    # ----- leaf key-kind split -------------------------------------------- #

    def compile_clause(self, clause: FilterClause) -> ColumnElement[bool]:
        """Dispatch a leaf on key-kind (system key vs metadata path)."""
        path = clause.path
        if len(path) == 1 and path[0] in SYSTEM_KEYS:
            expr = self._compile_system_clause(clause)
        elif path and path[0] == "metadata":
            expr = self._compile_metadata_clause(clause)
            # A metadata leaf compiles to an unindexed JSONB column access.
            # Emit once per metadata leaf (a filter with N metadata predicates
            # emits N times) — each leaf is a separate unindexed-column access.
            # compile_postgres runs once per recall (the compiled predicate is
            # reused across the vector + bm25 channels), so this does not
            # double-count on the hybrid path.
            record_unindexed_metadata(op=clause.op.value)
        else:
            return self._unsupported(clause, "path is neither a system key nor a metadata path")
        self._consumed.add(_path_str(path))
        return expr

    # ----- system-key leaves (typed columns) ------------------------------ #

    def _system_column(self, key: str) -> ColumnElement[Any]:
        """Resolve a system key to its column ref from ``ctx`` (via :func:`_col`)."""
        return _col(self._ctx, key)

    def _compile_system_clause(self, clause: FilterClause) -> ColumnElement[bool]:
        key = clause.path[0]
        col = self._system_column(key)
        op = clause.op
        operand = clause.operand

        if op == Op.EXISTS:
            # A system column is always present in the row — $exists is constant.
            return sa.true() if operand else sa.false()

        if op in (Op.IN, Op.NIN):
            values = [sa.literal(_system_value(v)) for v in operand]
            if op == Op.IN:
                # Made total so a wrapping field-level $not includes NULL rows.
                return sa.func.coalesce(col.in_(values), sa.false())
            # $nin includes NULL rows (Rule 2 polarity). Already a total boolean.
            return sa.or_(col.is_(None), col.notin_(values))

        # Rule 3 (NARROW): the ONLY system-key clause that short-circuits to a
        # constant is an $eq EXACT-ARRAY (tuple) operand against a scalar column —
        # a bare list on a system key lowers to EQ-with-tuple, and a scalar column
        # can never equal an array, so it is unsatisfiable → sa.false(). The $ne
        # complement is unreachable (StringOps.ne is str-only, so the validator
        # never produces a tuple here); it is the polarity-mirror of the $eq case
        # → sa.true(). $in / $nin are NOT impossible — they were already handled
        # above as normal membership.
        if op == Op.EQ and isinstance(operand, tuple):
            return sa.false()
        if op == Op.NE and isinstance(operand, tuple):
            return sa.true()

        if operand is None:
            # {k: null} → active null-or-missing match. For a system column,
            # "missing" is NULL. $ne null → IS NOT NULL. Both are total booleans.
            if op == Op.NE:
                return col.isnot(None)
            return col.is_(None)

        value = sa.literal(_system_value(operand))
        if op == Op.NE:
            # Include NULL rows (Rule 2): a row whose column is NULL satisfies $ne.
            # Already a total boolean (no coalesce needed).
            return sa.or_(col.is_(None), col != value)
        # $eq and the range ops compare directly. Wrap in coalesce(..., false())
        # so the leaf is a total boolean: a NULL column yields FALSE (same
        # exclusion as the bare NULL comparison on the positive side), and a
        # wrapping field-level $not then flips a NULL-column row to TRUE — which
        # is what $not($eq) / $not($gt) must do (parity with the explicit $ne).
        positive = (col == value) if op == Op.EQ else _RANGE_FN[op](col, value)
        return sa.func.coalesce(positive, sa.false())

    # ----- metadata leaves (JSONB) ---------------------------------------- #

    def _compile_metadata_clause(self, clause: FilterClause) -> ColumnElement[bool]:
        path = clause.path
        op = clause.op
        operand = clause.operand

        # Bare-blob metadata $eq (whole-document equality). JSONB `=` is
        # structural-normalized in PG (key order / whitespace insensitive). The
        # NULL-coalesced ``self._md`` keeps this total: a NULL blob compares as
        # the empty object, so a wrapping $not includes it.
        if len(path) == 1:
            if op != Op.EQ:
                return self._unsupported(clause, "only $eq is defined on the bare metadata blob")
            return self._md == _jsonb_literal(operand)

        segs = path[1:]  # drop the leading "metadata" root

        if op == Op.EXISTS:
            present = self._md_exists(segs)
            return present if operand else sa.not_(present)

        if op in (Op.IN, Op.NIN):
            if not operand:
                # An empty operand list is a valid filter with a defined row-set
                # (the validator accepts it). Positive $in over ∅ matches nothing;
                # $nin over ∅ matches everything. Guard before building the OR
                # chain: sa.or_() with no clauses vanishes from the enclosing AND
                # (so $in would wrongly match all), and sa.not_() of that empty
                # chain is invalid SQL (so $nin would error at execute time).
                return sa.false() if op == Op.IN else sa.true()
            chain = sa.or_(*(self._md_containment(segs, v) for v in operand))
            return chain if op == Op.IN else sa.not_(chain)

        if op == Op.EQ:
            if operand is None:
                # Active null-or-missing match: an explicit JSON null value OR an
                # absent key. `@>` containment of `null` would only catch the
                # former, so emit the explicit OR (Rule 4).
                return self._md_null(segs)
            return self._md_eq(segs, operand)
        if op == Op.NE:
            if operand is None:
                # $ne null → present AND not a JSON null value.
                return sa.not_(self._md_null(segs))
            # Negate the positive equality form. Each positive form is a total
            # boolean (containment, or coalesced exact-array), so NOT flips
            # absent rows to TRUE (Rule 2).
            return sa.not_(self._md_eq(segs, operand))

        # Range ops ($gt/$gte/$lt/$lte) — type-gated cast, made total.
        return self._md_range(segs, op, operand)

    def _md_eq(self, segs: tuple[str, ...], operand: Any) -> ColumnElement[bool]:
        """Positive metadata equality — always a total boolean.

        A ``$date`` literal compares through the guarded timestamptz gate; a tuple
        is an exact-array equality on the extracted node; any other scalar is
        array-aware ``@>`` containment.
        """
        if isinstance(operand, DateLiteral):
            return self._md_date_compare(segs, Op.EQ, operand)
        if isinstance(operand, tuple):
            # Exact-array equality (bare list → $eq exact-array). `#>` is NULL when
            # the path is absent, so coalesce the comparison to FALSE for totality.
            node = self._md_json(segs)
            return sa.func.coalesce(node == _jsonb_literal(list(operand)), sa.false())
        return self._md_containment(segs, operand)

    def _md_range(self, segs: tuple[str, ...], op: Op, operand: Any) -> ColumnElement[bool]:
        """A metadata range op — jsonb_typeof-gated cast, coalesced to total."""
        if isinstance(operand, DateLiteral):
            return self._md_date_compare(segs, op, operand)
        json_type, sa_type = _operand_json_type(operand)
        node = self._md_json(segs)
        if sa_type is None:
            # String range: lexicographic compare on the text extraction, gated.
            gated = sa.case((sa.func.jsonb_typeof(node) == json_type, self._md_text(segs)), else_=sa.null())
        else:
            gated = sa.case(
                (sa.func.jsonb_typeof(node) == json_type, sa.cast(self._md_text(segs), sa_type)),
                else_=sa.null(),
            )
        return sa.func.coalesce(_RANGE_FN[op](gated, sa.literal(operand)), sa.false())

    def _md_date_compare(self, segs: tuple[str, ...], op: Op, operand: DateLiteral) -> ColumnElement[bool]:
        """Guarded ``$date`` compare via ``khora_try_timestamptz`` — total.

        The compare relies on the session ``TimeZone='UTC'`` (set in the backend's
        ``create_async_engine`` ``server_settings``): a zoneless stored date string
        is cast to ``timestamptz`` under that session zone, matching the
        UTC-normalized ``DateLiteral.value`` operand. Deliberately NO per-expression
        ``AT TIME ZONE 'UTC'`` — the session-TZ contract is the documented
        mechanism (§7). **Caveat:** when ``StorageCoordinator`` injects a
        *shared* engine, the backend does not control its session TZ, so a zoneless
        ``$date`` metadata string is interpreted in the pool's default zone. Known,
        documented limitation — not defended against in SQL.
        """
        text = self._md_text(segs)
        gate = sa.case(
            (
                sa.and_(
                    sa.func.jsonb_typeof(self._md_json(segs)) == "string",
                    text.op("~")(sa.literal(r"^\d{4}-\d\d-\d\d")),
                ),
                sa.func.khora_try_timestamptz(text),
            ),
            else_=sa.null(),
        )
        cmp = sa.cast(sa.literal(operand.value), sa.DateTime(timezone=True))
        return sa.func.coalesce(_RANGE_FN[op](gate, cmp) if op in _RANGE_FN else gate == cmp, sa.false())

    def _md_containment(self, segs: tuple[str, ...], value: Any) -> ColumnElement[bool]:
        """Array-aware ``@>`` containment rooted at the metadata column.

        JSONB ``@>`` does NOT treat ``'{"tags": "x"}'`` as contained by
        ``'{"tags": ["x"]}'`` — the array-contains-primitive exception applies
        only when the operand itself is an array, not at a nested object key. So
        a single ``metadata @> '{"tags": v}'`` misses an array-valued field. To
        match BOTH a scalar field equal to ``v`` and an array field containing
        ``v``, OR the scalar-doc form with an array-wrapped form whose leaf value
        is a one-element list:

            ``metadata @> '{"tags": v}'  OR  metadata @> '{"tags": [v]}'``

        For a nested path the one-key doc is wrapped at each segment; the array
        form wraps the LEAF value (``{"a": {"tags": v}}`` / ``{"a": {"tags":
        [v]}}``). Each ``@>`` is a total boolean (``FALSE`` on absent/mismatch,
        never ``NULL``), so the OR of two totals is total and negation-safe.
        """
        leaf = _date_to_jsonable(value)
        scalar_doc: Any = leaf
        array_doc: Any = [leaf]
        for seg in reversed(segs):
            scalar_doc = {seg: scalar_doc}
            array_doc = {seg: array_doc}
        # Both arms are `@>` probes, so a GIN index on `metadata` still serves
        # this OR (the planner bitmap-ORs the two index scans) — do not collapse
        # it into a non-`@>` form that loses the index. ``value is None`` only
        # reaches here from `{k: {$in: [null, ...]}}` (a `{k: null}` $eq is routed
        # to `_md_null` before this), where the array arm just adds `@> {k:[null]}`.
        return sa.or_(
            self._md.op("@>")(_jsonb_literal(scalar_doc)),
            self._md.op("@>")(_jsonb_literal(array_doc)),
        )

    def _md_exists(self, segs: tuple[str, ...]) -> ColumnElement[bool]:
        """Presence test for a metadata path — total boolean.

        Single segment uses the GIN-friendly ``?`` key-existence operator. A
        nested path uses ``#>``-is-not-null: ``#>`` returns SQL NULL only when the
        path is missing (an explicit JSON ``null`` value yields ``'null'::jsonb``),
        so ``IS NOT NULL`` is TRUE iff the path resolves — matching Mongo
        ``$exists``.
        """
        if len(segs) == 1:
            return self._md.op("?")(sa.literal(segs[0]))
        return self._md_json(segs).isnot(None)

    def _md_null(self, segs: tuple[str, ...]) -> ColumnElement[bool]:
        """Active null-or-missing match — total boolean.

        ``(metadata #> '{path}') = 'null'::jsonb OR NOT <present>``: the value is
        an explicit JSON ``null`` OR the key is absent. Both count as "null" for
        the ``{k: null}`` filter, matching Mongo semantics.
        """
        is_json_null = self._md_json(segs) == _jsonb_literal(None)
        return sa.or_(is_json_null, sa.not_(self._md_exists(segs)))

    # ----- JSONB extraction primitives ------------------------------------ #

    def _md_json(self, segs: tuple[str, ...]) -> ColumnElement[Any]:
        """``metadata #> '{a,b}'`` — extract the JSONB node at ``segs``."""
        return self._md.op("#>")(_jpath(segs))

    def _md_text(self, segs: tuple[str, ...]) -> ColumnElement[Any]:
        """``metadata #>> '{a,b}'`` — extract the text at ``segs``."""
        return self._md.op("#>>")(_jpath(segs))

    # ----- unsupported ---------------------------------------------------- #

    def _unsupported(self, clause: FilterClause, reason: str) -> ColumnElement[bool]:
        """Handle a clause this backend cannot express per ``ctx.on_unsupported``.

        ``"raise"`` raises the public :class:`RecallFilterUnsupportedError`;
        ``"split"`` leaves the clause out of ``consumed_keys`` (the engine
        post-filters it) and emits a non-constraining ``sqlalchemy.true()`` so it
        does not narrow the result set.
        """
        path = _path_str(clause.path)
        if self._ctx.on_unsupported == "raise":
            raise RecallFilterUnsupportedError(path, reason)
        return sa.true()


# --------------------------------------------------------------------------- #
# Module-level helpers (no per-pass state).
# --------------------------------------------------------------------------- #


def _jpath(segments: tuple[str, ...]) -> ColumnElement[Any]:
    """Build the Postgres ``text[]`` path operand for ``#>`` / ``#>>``.

    The ``#>`` / ``#>>`` operators require a ``text[]`` right operand. Binding the
    segments as a Python list typed ``ARRAY(Text())`` makes asyncpg transmit a
    native ``text[]`` array over the protocol — no manual array-literal string,
    so a segment containing ``,`` ``{`` ``}`` ``"`` ``\\`` or whitespace round-trips
    verbatim (the protocol-level encoder handles framing, not us).

    Prod impact of getting this wrong: only the ``#>`` / ``#>>`` path forms were
    affected — nested ``metadata.a.b`` paths, nested-path ``$exists``, ``$date``
    gates, exact-array ``$eq``, and nested range ops. The single-level ``@>``
    containment and ``?`` key-presence forms never route through ``_jpath`` and
    were unaffected.
    """
    return sa.literal(list(segments), type_=ARRAY(sa.Text()))


def _jsonb_literal(value: Any) -> ColumnElement[Any]:
    """A ``CAST('<json>' AS JSONB)`` literal for a Python value (sorted keys)."""
    return sa.cast(sa.literal(json.dumps(value, sort_keys=True)), JSONB)


def _operand_json_type(operand: Any) -> tuple[str, Any]:
    """Map a range operand's Python type to its ``jsonb_typeof`` gate + cast type.

    ``int``/``float`` → ``("number", Numeric)``; ``bool`` → ``("boolean",
    Boolean)``; everything else → ``("string", None)`` (lexicographic text
    compare, no cast). ``bool`` is checked before ``int`` (``bool`` is an
    ``int`` subclass).
    """
    if isinstance(operand, bool):
        return "boolean", sa.Boolean()
    if isinstance(operand, (int, float)):
        return "number", sa.Numeric()
    return "string", None


def _system_value(value: Any) -> Any:
    """Unwrap a :class:`DateLiteral` to its datetime; pass other values through."""
    if isinstance(value, DateLiteral):
        return value.value
    return value


def _date_to_jsonable(value: Any) -> Any:
    """Coerce a containment operand to a JSON-serializable scalar.

    A :class:`DateLiteral` / :class:`~datetime.datetime` is rendered as its
    ISO-8601 string so the ``@>`` doc is valid JSON; other scalars pass through.
    """
    if isinstance(value, DateLiteral):
        return value.value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value
