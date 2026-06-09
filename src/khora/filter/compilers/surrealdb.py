"""SurrealDB SurrealQL recall-filter compiler — ``@internal``.

Lowers a canonical :class:`~khora.filter.ast.FilterNode` to a single SurrealQL
boolean *string* the engine splices into its own ``WHERE``, paired with a
``params`` dict of ``$name`` binds the connection passes through to the SDK. The
output is a :class:`~khora.filter.registry.CompiledFilter` whose ``predicate`` is
that string.

The compiler is the Layer-4 half of the §4/§7 filter contract, mirroring
:func:`~khora.filter.compilers.cypher.compile_cypher` structurally — a string
predicate plus an out-of-band ``params`` dict named ``{param_namespace}_{n}``.
It speaks the four emission rules (never abort, polarity, narrow impossible
pairs, presence/null), but with **one deliberate divergence from cypher and
postgres: there is NO ``coalesce`` / totality wrapper.**

**Why no totality wrapper (verified on embedded SurrealDB).** SurrealQL
comparisons against an absent (``NONE``) path return a *total* boolean rather
than a propagating ``NULL``: ``absent = x`` → ``false``, ``absent != x`` →
``true``, ``absent > 5`` → ``false``, and ``!(...)`` flips an absent row
correctly. Cypher and Postgres wrap every leaf that *could* produce ``NULL`` in
``coalesce(expr, false)`` precisely because ``NOT NULL`` is ``NULL`` (which would
drop a row from a negation); SurrealQL has no such hazard, so the wrapper is
omitted by design. A reviewer expecting the cypher/postgres coalesce should read
this as intentional, not an oversight — the totality the wrapper buys elsewhere
is already a property of SurrealQL's NONE-boolean algebra.

**NONE vs NULL are distinct.** ``NONE`` is an absent path; ``NULL`` is an
explicit JSON null value. A ``{k: null}`` match resolves to ``(p = NULL OR p IS
NONE)``; ``$exists: true`` is ``(p IS NOT NONE)`` and ``$exists: false`` is
``(p IS NONE)``.

**Metadata equality / membership is array-aware.** A scalar ``$eq`` operand
matches **both** a scalar field equal to it **and** an array field that contains
it — ``(path = $b OR (type::is::array(path) AND path CONTAINS $b))`` — mirroring
the Postgres ``@>`` two-arm form and the ``compile_python`` oracle's
``_md_contains``. ``$in`` is contains-any (``path INSIDE $list OR
(type::is::array(path) AND path CONTAINSANY $list)``); ``$ne`` / ``$nin`` are the
negations (which therefore admit absent / wrong-type rows). The
``type::is::array`` guard is load-bearing: SurrealQL ``CONTAINS`` / ``CONTAINSANY``
do *substring* matching on a string left operand, so gating them to real arrays
keeps element matching exact. A bare-list ``$eq`` (exact-array) and a sub-document
dict ``$eq`` keep **exact** structural equality (``=``), not containment — this is
the canonical ``tags: list[str]`` case, so array-awareness is the common path, not
an edge case.

**Range ops on metadata are type-gated.** A range op on a metadata path emits
``(type::is::<t>(path) AND path <op> $b)`` — the ``AND`` short-circuits so a
wrong-typed / absent value never participates in the compare (``type::is::*``
returns ``false``, not an error, for a missing / wrong-type / whole-metadata-NONE
path). ``type::is::bool`` is checked *before* ``type::is::number`` because a bool
is an int subclass in Python and SurrealDB agrees a bool is not a number. System
datetime columns (``occurred_at`` / ``created_at``) are typed columns, so their
range ops are **ungated**.

**Dates.** A system datetime column binds the real Python :class:`~datetime.datetime`
directly — the SDK encodes it as a SurrealQL datetime (unlike cypher, which binds
``.isoformat()``). A metadata ``$date`` / datetime operand is gated with
``type::is::string`` and bound as its ``.isoformat()`` string: metadata datetimes
round-trip through the FLEXIBLE ``object`` column as ISO strings, and ISO-8601's
fixed-width big-endian layout makes a lexicographic string compare agree with
chronological order for the UTC-normalized values the validator produces.

**Injection guard.** User-supplied metadata path segments are interpolated into
the predicate string (SurrealQL cannot bind a field name as a parameter), so each
segment is validated against a strict identifier regex before interpolation; an
unsafe segment raises :class:`~khora.filter.context.CompileError`. System keys
come from the closed :data:`~khora.filter.model.SYSTEM_KEYS` whitelist and are
safe. User *values* always bind via ``$params``, never interpolate.

``@internal``. Reachable as ``khora.filter.compilers.surrealdb.compile_surrealdb``
for khora's own engines; not re-exported from :mod:`khora.__init__`.
"""

from __future__ import annotations

import re
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
from khora.filter.context import CompileError
from khora.filter.model import SYSTEM_KEYS, Op
from khora.filter.telemetry import record_unindexed_metadata

__all__ = ["compile_surrealdb"]


# SurrealQL comparison operator strings for the range ops, keyed by AST op.
_RANGE_OP = {
    Op.GT: ">",
    Op.GTE: ">=",
    Op.LT: "<",
    Op.LTE: "<=",
}

# A safe SurrealQL identifier for a metadata path segment: a leading letter or
# underscore, then alphanumerics / underscores. Stricter than the storage-layer
# ``_sanitize_field_name`` (no dots — each AST path segment is already a single
# field name) and kept self-contained so the compiler embeds no storage import.
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def compile_surrealdb(ast: FilterNode, ctx: CompileContext) -> CompiledFilter[str]:
    """Compile a canonical AST to a SurrealQL ``WHERE`` fragment + bind dict.

    ``ast`` is always a :class:`FilterNode` (the ``parse_to_ast`` root invariant).
    An empty ``AND`` (the match-everything root of a bare filter) compiles to the
    literal ``"true"``. Binds are returned out-of-band in ``params`` as a
    ``{name: value}`` dict; each ``$name`` placeholder in the predicate string has
    a matching entry. Bind names are generated as ``{param_namespace}_{n}`` so
    they cannot collide with the engine's own query parameters.

    Field references are derived from ``ctx`` alone: a system key is the bare
    physical name (``ctx.field_mapping`` remaps, identity when ``None``), prefixed
    with ``ctx.table_alias`` only when set; a metadata path descends natively
    through the remapped ``metadata`` root (``metadata`` → ``metadata_`` on the
    live recall path). Honors ``ctx.on_unsupported``: on a clause this backend
    cannot express, ``"raise"`` raises :class:`RecallFilterUnsupportedError`;
    ``"split"`` omits it from ``consumed_keys`` and emits a non-constraining
    ``"true"`` placeholder. A metadata path segment that is not a safe SurrealQL
    identifier raises :class:`CompileError` regardless of mode (an injection
    guard, not a capability gap).
    """
    consumed: set[str] = set()
    builder = _Builder(ctx=ctx, consumed=consumed)
    predicate = builder.compile_node(ast)
    return CompiledFilter(
        predicate=predicate,
        params=builder.params,
        consumed_keys=frozenset(consumed),
        # canonical_hash over the whole AST. In on_unsupported="raise" mode (the
        # only mode the skeleton engine uses) every leaf is consumed, so the whole
        # tree == the consumed slice.
        canonical_hash=canonical_hash(ast),
    )


def _path_str(path: tuple[str, ...]) -> str:
    """Render an AST path as the dotted key string used for diagnostics/consumed."""
    return ".".join(path)


class _Builder:
    """Per-compile-pass state: the :class:`CompileContext`, consumed set, binds.

    A compile pass threads the ``ctx`` (field references are derived from it), the
    ``consumed`` accumulator, and the ``params`` bind dict every leaf appends to.
    Keeping them on a small object avoids passing the triple through every
    recursion frame. A monotonic counter names each bind ``{param_namespace}_{n}``
    so two clauses on the same key never collide.
    """

    def __init__(self, *, ctx: CompileContext, consumed: set[str]) -> None:
        self._ctx = ctx
        self._consumed = consumed
        self._alias = ctx.table_alias
        self.params: dict[str, Any] = {}
        self._counter = 0

    # ----- bind allocation ------------------------------------------------ #

    def _bind(self, value: Any) -> str:
        """Allocate a fresh ``$name`` placeholder bound to ``value``."""
        name = f"{self._ctx.param_namespace}_{self._counter}"
        self._counter += 1
        self.params[name] = value
        return f"${name}"

    def _system_field(self, key: str) -> str:
        """Render a system-key field reference for a logical key.

        ``field_mapping`` remaps a logical key to its physical field name
        (identity when ``None``); ``table_alias`` prefixes it only when set (the
        live recall path queries the base table unaliased, matching
        ``_build_filter_clauses``' bare ``source_system`` / ``occurred_at``).
        ``key`` is always a controlled token (a system key from the closed
        ``SYSTEM_KEYS`` whitelist) — never free user text, so the field access is
        not an injection surface.
        """
        physical = (self._ctx.field_mapping or {}).get(key, key)
        return f"{self._alias}.{physical}" if self._alias else physical

    def _metadata_path(self, segs: tuple[str, ...]) -> str:
        """Render a native dot-descent metadata path, validating each segment.

        The ``metadata`` root is remapped via ``field_mapping`` (``metadata`` →
        ``metadata_`` on the live recall path); each sub-segment is user-supplied,
        so it is validated against :data:`_SAFE_SEGMENT_RE` and interpolated into
        the path string — an unsafe segment raises :class:`CompileError` (the
        injection guard). SurrealQL has no field-name bind, so interpolation is
        unavoidable; the regex is the defense.
        """
        root = (self._ctx.field_mapping or {}).get("metadata", "metadata")
        parts = [root]
        for seg in segs:
            if not _SAFE_SEGMENT_RE.match(seg):
                raise CompileError(f"unsafe metadata path segment {seg!r} (not a SurrealQL identifier)")
            parts.append(seg)
        return ".".join(parts)

    # ----- logical node walk ---------------------------------------------- #

    def compile_node(self, node: FilterNode | FilterClause) -> str:
        """Compile a logical node or leaf to a SurrealQL boolean string."""
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
            return "(" + " OR ".join(self.compile_node(c) for c in node.children) + ")"
        # Op.NOT — exactly one child per the AST contract. SurrealQL's NONE-boolean
        # algebra makes every leaf total, so this negation flips absent rows
        # correctly without a coalesce wrapper.
        return f"!({self.compile_node(node.children[0])})"

    # ----- leaf key-kind split -------------------------------------------- #

    def compile_clause(self, clause: FilterClause) -> str:
        """Dispatch a leaf on key-kind (system key vs metadata path)."""
        path = clause.path
        if len(path) == 1 and path[0] in SYSTEM_KEYS:
            expr = self._compile_system_clause(clause)
        elif path and path[0] == "metadata":
            expr = self._compile_metadata_clause(clause)
            # A metadata leaf reads the FLEXIBLE ``object`` column (no index).
            # Emit once per metadata leaf (a filter with N metadata predicates
            # emits N times) — mirrors compile_postgres / compile_python.
            record_unindexed_metadata(op=clause.op.value)
        else:
            return self._unsupported(clause, "path is neither a system key nor a metadata path")
        self._consumed.add(_path_str(path))
        return expr

    # ----- system-key leaves (typed columns) ------------------------------ #

    def _compile_system_clause(self, clause: FilterClause) -> str:
        field = self._system_field(clause.path[0])
        op = clause.op
        operand = clause.operand

        if op == Op.EXISTS:
            # The 8 unwritten document keys read NONE on the SCHEMAFULL table, so
            # presence is IS NOT NONE (true) / IS NONE (false).
            return f"({field} IS NOT NONE)" if operand else f"({field} IS NONE)"

        if op in (Op.IN, Op.NIN):
            if not operand:
                # $in over ∅ matches nothing; $nin over ∅ matches everything.
                # Emit the constant explicitly (INSIDE [] already yields this, but
                # the explicit form mirrors the cypher/postgres compilers).
                return "false" if op == Op.IN else "true"
            values_bind = self._bind([_system_value(v) for v in operand])
            if op == Op.IN:
                return f"({field} INSIDE {values_bind})"
            # $nin includes absent rows (Rule 2 polarity): an absent field is not
            # INSIDE the set, so the negation is true — already total.
            return f"!({field} INSIDE {values_bind})"

        # Rule 3 (NARROW): a bare list on a system key lowers to EQ-with-tuple. A
        # scalar column never equals a list, so the compare is false at query time
        # — no special-cased constant needed (the list binds like any operand).
        if operand is None:
            # {k: null} → active null-or-missing match. NONE (absent) and NULL
            # (explicit json null) are distinct, so cover both. $ne null → present
            # and not null.
            if op == Op.NE:
                return f"({field} IS NOT NONE AND {field} != NULL)"
            return f"({field} = NULL OR {field} IS NONE)"

        value_bind = self._bind(_system_value(operand))
        if op == Op.NE:
            # ``!=`` against an absent field already returns true in SurrealQL, so
            # this admits absent rows without an extra ``OR IS NONE`` (Rule 2).
            return f"({field} != {value_bind})"
        # $eq and the range ops compare directly. An absent / wrong-type field
        # yields a total false (no coalesce needed), and a wrapping $not flips it.
        symbol = "=" if op == Op.EQ else _RANGE_OP[op]
        return f"({field} {symbol} {value_bind})"

    # ----- metadata leaves (native object descent) ------------------------ #

    def _compile_metadata_clause(self, clause: FilterClause) -> str:
        path = clause.path
        op = clause.op
        operand = clause.operand

        # Bare-blob metadata $eq (whole-document equality). SurrealQL object
        # equality is structural / key-order-insensitive.
        if len(path) == 1:
            if op != Op.EQ:
                return self._unsupported(clause, "only $eq is defined on the bare metadata blob")
            root = (self._ctx.field_mapping or {}).get("metadata", "metadata")
            return f"({root} = {self._bind(operand)})"

        segs = path[1:]  # drop the leading "metadata" root
        node = self._metadata_path(segs)

        if op == Op.EXISTS:
            return f"({node} IS NOT NONE)" if operand else f"({node} IS NONE)"

        if op in (Op.IN, Op.NIN):
            if not operand:
                return "false" if op == Op.IN else "true"
            values_bind = self._bind([_metadata_value(v) for v in operand])
            inner = self._md_contains_any(node, values_bind)
            # $nin admits absent / wrong-type rows: the inner is total, so the
            # negation is true for them (Rule 2 polarity).
            return inner if op == Op.IN else f"!{inner}"

        if operand is None:
            # {k: null} → explicit json null OR absent path. $ne null → present
            # and not null.
            if op == Op.NE:
                return f"({node} IS NOT NONE AND {node} != NULL)"
            return f"({node} = NULL OR {node} IS NONE)"

        if op == Op.EQ:
            return self._md_eq(node, operand)
        if op == Op.NE:
            # Negate the (total) positive equality/containment form so an absent /
            # wrong-type path is admitted (Rule 2) — same polarity as the oracle.
            return f"!{self._md_eq(node, operand)}"

        # Range ops ($gt/$gte/$lt/$lte) — type-gated. The AND short-circuits so a
        # wrong-typed / absent value never reaches the compare.
        return self._md_range(node, op, operand)

    def _md_eq(self, node: str, operand: Any) -> str:
        """Positive metadata equality / containment for a non-null ``$eq`` operand.

        Array-aware for a plain scalar (mirrors the oracle's ``_md_contains`` and
        the Postgres ``@>`` two-arm form): a scalar operand matches **both** a
        scalar field equal to it **and** an array field that contains it —
        ``(node = $b OR (type::is::array(node) AND node CONTAINS $b))``. The
        ``type::is::array`` guard is load-bearing: SurrealQL ``CONTAINS`` does
        *substring* matching when its left operand is a string, so without the
        guard a scalar field ``"xyz"`` would wrongly match operand ``"x"``;
        gating to real arrays keeps element matching exact (verified on the
        embedded engine). Both arms reuse a single bind.

        A ``$date`` / ``datetime``, an exact-array ``tuple`` (bare-list ``$eq``),
        and a sub-document ``dict`` keep **exact** equality — SurrealQL object /
        array ``=`` is structural — matching the oracle's date-compare /
        ``_exact_array_eq`` / ``_md_object_eq`` modes (NOT containment).
        """
        if isinstance(operand, (DateLiteral, datetime, tuple, dict)):
            return f"({node} = {self._bind(_metadata_value(operand))})"
        bind = self._bind(_metadata_value(operand))
        return f"({node} = {bind} OR (type::is::array({node}) AND {node} CONTAINS {bind}))"

    def _md_contains_any(self, node: str, values_bind: str) -> str:
        """Array-aware contains-any for ``$in`` (mirrors ``any(_md_contains(node, v))``).

        A scalar field that is a member of the operand list (``node INSIDE
        $list`` — exact membership, since the right operand is a list) **or** an
        array field sharing any element with it (``node CONTAINSANY $list``,
        guarded to real arrays so a scalar string's substring ``CONTAINSANY``
        never fires). Total against absent / wrong-type paths, so the ``$nin``
        negation admits them.
        """
        return f"({node} INSIDE {values_bind} OR (type::is::array({node}) AND {node} CONTAINSANY {values_bind}))"

    def _md_range(self, node: str, op: Op, operand: Any) -> str:
        """A type-gated metadata range op: ``(type::is::<t>(node) AND node <op> $b)``.

        The gate function is chosen by the operand's Python type. A ``$date`` /
        datetime operand is gated as a string and bound as its ISO-8601 form
        (metadata datetimes round-trip as ISO strings; lexicographic compare
        agrees with chronological order for UTC-normalized values). ``bool`` is
        gated before ``number`` (a bool is an int subclass, and SurrealDB agrees a
        bool is not a number).
        """
        if isinstance(operand, (DateLiteral, datetime)):
            gate = "string"
            bound = self._bind(_metadata_value(operand))
        else:
            gate = _range_gate(operand)
            bound = self._bind(operand)
        symbol = _RANGE_OP[op]
        return f"(type::is::{gate}({node}) AND {node} {symbol} {bound})"

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
    """Coerce a system-key operand to an SDK-bindable value.

    A typed datetime column binds a real :class:`~datetime.datetime` — the SDK
    encodes it as a SurrealQL datetime, so a :class:`DateLiteral` is unwrapped to
    its ``.value`` rather than stringified (unlike cypher). An exact-array
    ``tuple`` (a bare-list ``$eq`` operand) binds as a ``list``, each element
    coerced the same way. Other scalars pass through.
    """
    if isinstance(value, DateLiteral):
        return value.value
    if isinstance(value, tuple):
        return [_system_value(item) for item in value]
    return value


def _metadata_value(value: Any) -> Any:
    """Coerce a metadata operand to an SDK-bindable value.

    A metadata datetime round-trips through the FLEXIBLE ``object`` column as an
    ISO-8601 *string*, so a :class:`DateLiteral` / :class:`~datetime.datetime`
    binds as its ``.isoformat()`` string (the gated string compare relies on
    lexicographic order). A ``tuple`` (exact-array / ``$in`` list) binds as a
    ``list`` with each element coerced the same way. Other scalars pass through.
    """
    if isinstance(value, DateLiteral):
        return value.value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_metadata_value(item) for item in value]
    return value


def _range_gate(operand: Any) -> str:
    """Map a range operand's Python type to its ``type::is::*`` gate function.

    ``bool`` → ``bool`` (checked first — a bool is an ``int`` subclass);
    ``int`` / ``float`` → ``number``; everything else → ``string`` (lexicographic
    text compare). Mirrors ``compile_postgres._operand_json_type`` / the python
    compiler's ``_comparable`` type-gate.
    """
    if isinstance(operand, bool):
        return "bool"
    if isinstance(operand, (int, float)):
        return "number"
    return "string"
