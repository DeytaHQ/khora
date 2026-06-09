"""In-memory Python recall-filter predicate compiler — ``@internal``.

Lowers a canonical :class:`~khora.filter.ast.FilterNode` to a pure
``Callable[[Any], bool]`` that evaluates the *whole* filter against one
in-memory record. The output is a :class:`~khora.filter.registry.CompiledFilter`
whose ``predicate`` is that callable; the engine applies it to each retrieved
candidate as a post-filter (the half of a deterministic recall filter a backend
could not push down).

It is the Python sibling of :func:`~khora.filter.compilers.postgres.compile_postgres`
and is the **oracle** the SQL/Cypher compilers are checked against: it speaks the
same four §4 emission rules, so a record that the predicate accepts is exactly a
row the SQL ``WHERE`` would have returned.

1. **Never abort.** A comparison against a wrong-typed or absent value evaluates
   to ``False`` (a positive op) — never an exception. A ``$date`` operand that
   fails to parse excludes the record rather than raising.
2. **Polarity.** Negations include absent/null/wrong-type records: ``$ne`` /
   ``$nin`` match a record missing the key or holding a wrong-typed value; a
   ``$ne`` against a present, equal value is the only exclusion.
3. **Impossible pairs (narrow).** An ``$eq`` exact-array (tuple) operand against
   a scalar system value never matches (a scalar is not a list), so it excludes;
   its ``$ne`` mirror includes. ``$in`` / ``$nin`` are normal membership.
4. **Presence/null.** ``$exists`` and a ``{k: null}`` match distinguish absent
   from present-null on a metadata path (``seg in d`` vs ``d.get(seg) is None``,
   per segment). System keys are treated as always-present (their ``$exists`` is
   trivially all/none), and a system value of ``None`` is the null-or-missing
   case for ``{k: null}`` / ``$ne null``.

**Record shape.** The predicate is deliberately defensive about the record it is
handed, so one callable serves both a chronicle ``Chunk`` dataclass and a plain
``dict`` event row. A system key is resolved by trying the record's own
attribute, then its ``source_document`` projection's attribute, then mapping
access — the first hit wins, and a key found nowhere is treated as a ``None``
system value (the always-present-but-null case). The ``metadata`` blob is read
the same way and coalesced to ``{}`` when absent or ``None``.

**Per-key support matrix** (Python evaluates *every* key — there is no pushdown
split here; the Chronicle compiler decides what it pushes down, and this
predicate re-checks the whole AST):

* ``occurred_at`` / ``created_at`` / ``source_timestamp`` — system datetime
  compare (``$eq``/``$ne``/range/``$in``/``$nin``/``{k:null}``).
* the eight denormalized document keys (``source_type`` / ``source_name`` /
  ``source_url`` / ``external_id`` / ``content_type`` / ``source`` / ``title``)
  — system scalar compare, resolved defensively (missing ⇒ ``None``).
* ``metadata.<path>`` — scalar array-aware containment / exact-array (bare list)
  / EXACT ``object_equal`` (sub-path dict operand) / range / presence. Mirrors
  the ADR §4 match-mode taxonomy (the sub-path dict case is exact ``=``, NOT
  ``@>`` containment).

``@internal``. Reachable as ``khora.filter.compilers.python.compile_python`` for
khora's own engines; not re-exported from :mod:`khora.__init__`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
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
from khora.filter.telemetry import record_unindexed_metadata

__all__ = ["compile_python"]


# A sentinel distinct from ``None``: ``None`` is a legitimate stored value
# (an explicit JSON null / a NULL system column), so "absent" needs its own
# marker. Used for metadata path extraction where absent vs present-null is
# load-bearing (Rule 4).
class _Missing:
    """Singleton marker for an absent value (distinct from a present ``None``)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "<MISSING>"


MISSING = _Missing()


# Python operator callables for the range ops, keyed by AST op.
_RANGE_FN: dict[Op, Callable[[Any, Any], bool]] = {
    Op.GT: lambda a, b: a > b,
    Op.GTE: lambda a, b: a >= b,
    Op.LT: lambda a, b: a < b,
    Op.LTE: lambda a, b: a <= b,
}


def compile_python(ast: FilterNode, ctx: CompileContext) -> CompiledFilter[Callable[[Any], bool]]:
    """Compile a canonical AST to an in-memory ``callable(record) -> bool``.

    ``ast`` is always a :class:`FilterNode` (the ``parse_to_ast`` root invariant).
    An empty ``AND`` (the match-everything root of a bare filter) compiles to a
    predicate that accepts every record. ``params`` is always empty (the
    predicate closes over the operands directly). ``consumed_keys`` is every leaf
    in the AST — this compiler evaluates the whole filter, so it consumes
    everything (it is the post-filter, not a pushdown that splits).

    ``ctx.field_mapping`` remaps a logical system key to the record attribute /
    mapping key to read (identity when ``None``). ``ctx.on_unsupported`` is
    honored for the (post-validation unreachable) clause this compiler cannot
    express: ``"raise"`` raises :class:`RecallFilterUnsupportedError`; ``"split"``
    drops the clause from ``consumed_keys`` and treats it as match-all so it does
    not narrow the record set.

    The ``khora.recall.filter.unindexed_metadata`` counter fires once per metadata
    leaf **at compile time** (during this AST walk), mirroring
    :func:`compile_postgres` — never inside the returned per-record callable.
    """
    consumed: set[str] = set()
    builder = _Builder(ctx=ctx, consumed=consumed)
    predicate = builder.compile_node(ast)
    return CompiledFilter(
        predicate=predicate,
        params={},
        consumed_keys=frozenset(consumed),
        canonical_hash=canonical_hash(ast),
    )


def _path_str(path: tuple[str, ...]) -> str:
    """Render an AST path as the dotted key string used for diagnostics/consumed."""
    return ".".join(path)


class _Builder:
    """Per-compile-pass state: the :class:`CompileContext` and consumed set.

    Mirrors the postgres compiler's ``_Builder`` — a small object that threads
    the ``ctx`` (field remapping) and the ``consumed`` accumulator through the
    recursion so each leaf-compile method need not pass the pair down. Each
    ``compile_*`` method returns a ``callable(record) -> bool``; the logical-node
    walk composes them with ``all`` / ``any`` / negation.
    """

    def __init__(self, *, ctx: CompileContext, consumed: set[str]) -> None:
        self._ctx = ctx
        self._consumed = consumed

    # ----- logical node walk ---------------------------------------------- #

    def compile_node(self, node: FilterNode | FilterClause) -> Callable[[Any], bool]:
        """Compile a logical node or leaf to a ``callable(record) -> bool``."""
        if isinstance(node, FilterClause):
            return self.compile_clause(node)
        if node.op == Op.AND:
            if not node.children:
                # The empty match-everything root — no constraint.
                return lambda _record: True
            child_fns = [self.compile_node(c) for c in node.children]
            return lambda record: all(fn(record) for fn in child_fns)
        if node.op == Op.OR:
            if not node.children:
                # The validator forbids an empty $or; guard defensively.
                return lambda _record: False
            child_fns = [self.compile_node(c) for c in node.children]
            return lambda record: any(fn(record) for fn in child_fns)
        # Op.NOT — exactly one child per the AST contract. Leaves are built total
        # (every callable returns a real bool, never raises) so this negation
        # flips absent/wrong-type records correctly (Rule 2).
        child_fn = self.compile_node(node.children[0])
        return lambda record: not child_fn(record)

    # ----- leaf key-kind split -------------------------------------------- #

    def compile_clause(self, clause: FilterClause) -> Callable[[Any], bool]:
        """Dispatch a leaf on key-kind (system key vs metadata path)."""
        path = clause.path
        if len(path) == 1 and path[0] in SYSTEM_KEYS:
            fn = self._compile_system_clause(clause)
        elif path and path[0] == "metadata":
            fn = self._compile_metadata_clause(clause)
            # A metadata leaf reads an unindexed blob. Emit once per metadata
            # leaf at compile time (a filter with N metadata predicates emits N
            # times) — matching compile_postgres. NOT inside ``fn`` (that would
            # fire per record).
            record_unindexed_metadata(op=clause.op.value)
        else:
            return self._unsupported(clause, "path is neither a system key nor a metadata path")
        self._consumed.add(_path_str(path))
        return fn

    # ----- value resolution ----------------------------------------------- #

    def _resolve_system(self, record: Any, key: str) -> Any:
        """Resolve a system key on a record to its value (``None`` if absent).

        ``field_mapping`` remaps the logical key to the physical attribute /
        mapping key (identity when ``None``). Resolution is defensive so one
        predicate serves both a chunk dataclass and a dict event row: try the
        record's own attribute, then its ``source_document`` projection (where
        the denormalized document keys live on a chunk), then mapping access.
        A key found nowhere resolves to ``None`` — the always-present-but-null
        case for a system key (Rule 4), never an exception.
        """
        physical = (self._ctx.field_mapping or {}).get(key, key)
        value = _getattr_or_missing(record, physical)
        if value is not MISSING:
            return value
        source_doc = _getattr_or_missing(record, "source_document")
        if source_doc is not MISSING and source_doc is not None:
            value = _getattr_or_missing(source_doc, physical)
            if value is not MISSING:
                return value
        value = _getitem_or_missing(record, physical)
        if value is not MISSING:
            return value
        return None

    def _resolve_metadata(self, record: Any) -> Mapping[str, Any]:
        """Resolve the ``metadata`` blob on a record, coalescing absent/None to ``{}``.

        Reads ``record.metadata`` then ``record["metadata"]``; a non-mapping or
        absent/``None`` blob coalesces to an empty dict so every metadata
        operator is total against a record with no metadata.
        """
        value = _getattr_or_missing(record, "metadata")
        if value is MISSING:
            value = _getitem_or_missing(record, "metadata")
        if isinstance(value, Mapping):
            return value
        return {}

    # ----- system-key leaves ---------------------------------------------- #

    def _compile_system_clause(self, clause: FilterClause) -> Callable[[Any], bool]:
        key = clause.path[0]
        op = clause.op
        operand = clause.operand

        if op == Op.EXISTS:
            # A system key is always present on the record (an absent one
            # resolves to None, which still counts as present-with-null for
            # $exists). $exists is therefore constant — true / false.
            return (lambda _record: True) if operand else (lambda _record: False)

        if op in (Op.IN, Op.NIN):
            values = [_system_value(v) for v in operand]
            if op == Op.IN:
                return lambda record: _system_in(self._resolve_system(record, key), values)
            # $nin includes a record whose value is None or wrong-type / not in
            # the set (Rule 2 polarity).
            return lambda record: _system_nin(self._resolve_system(record, key), values)

        # Rule 3 (NARROW): an $eq EXACT-ARRAY (tuple) operand against a scalar
        # system value never matches; its $ne mirror always matches. (A bare
        # list on a system key lowers to EQ-with-tuple; the $ne complement is
        # unreachable via the validator but kept uniform.)
        if op == Op.EQ and isinstance(operand, tuple):
            return lambda _record: False
        if op == Op.NE and isinstance(operand, tuple):
            return lambda _record: True

        if operand is None:
            # {k: null} → active null-or-missing match. A system value of None is
            # "null"; $ne null → value is not None.
            if op == Op.NE:
                return lambda record: self._resolve_system(record, key) is not None
            return lambda record: self._resolve_system(record, key) is None

        value = _system_value(operand)
        if op == Op.NE:
            # Include None / wrong-type records (Rule 2): a record whose value is
            # None, or whose value cannot be compared, satisfies $ne.
            return lambda record: _system_ne(self._resolve_system(record, key), value)
        if op == Op.EQ:
            return lambda record: _system_eq(self._resolve_system(record, key), value)
        # Range ops — type-safe scalar compare; None / uncomparable excludes.
        range_fn = _RANGE_FN[op]
        return lambda record: _system_range(self._resolve_system(record, key), value, range_fn)

    # ----- metadata leaves ------------------------------------------------ #

    def _compile_metadata_clause(self, clause: FilterClause) -> Callable[[Any], bool]:
        path = clause.path
        op = clause.op
        operand = clause.operand

        # Bare-blob metadata $eq (whole-document equality). Normalized JSON
        # equality (dict compare is structural in Python). Coalesced metadata
        # ({} for a record with none) keeps this total.
        if len(path) == 1:
            if op != Op.EQ:
                return self._unsupported(clause, "only $eq is defined on the bare metadata blob")
            expected = _jsonable(operand)
            return lambda record: _json_eq(_jsonable(dict(self._resolve_metadata(record))), expected)

        segs = path[1:]  # drop the leading "metadata" root

        if op == Op.EXISTS:
            if operand:
                return lambda record: _md_present(self._resolve_metadata(record), segs)
            return lambda record: not _md_present(self._resolve_metadata(record), segs)

        if op in (Op.IN, Op.NIN):
            if not operand:
                # $in over ∅ matches nothing; $nin over ∅ matches everything.
                return (lambda _record: False) if op == Op.IN else (lambda _record: True)
            values = list(operand)
            if op == Op.IN:
                return lambda record: any(_md_contains(self._md_extract(record, segs), v) for v in values)
            # $nin: NOT in the set. Containment is total (False on absent/
            # mismatch), so the negation includes absent / wrong-type (Rule 2).
            return lambda record: not any(_md_contains(self._md_extract(record, segs), v) for v in values)

        if op == Op.EQ:
            if operand is None:
                return lambda record: _md_is_null(self._resolve_metadata(record), segs)
            return self._md_eq_fn(segs, operand)
        if op == Op.NE:
            if operand is None:
                # $ne null → present AND not a JSON null value.
                return lambda record: not _md_is_null(self._resolve_metadata(record), segs)
            eq_fn = self._md_eq_fn(segs, operand)
            # Negate the (total) positive equality form so absent / wrong-type
            # records are included (Rule 2).
            return lambda record: not eq_fn(record)

        # Range ops ($gt/$gte/$lt/$lte) — type-gated scalar compare.
        return self._md_range_fn(segs, op, operand)

    def _md_extract(self, record: Any, segs: tuple[str, ...]) -> Any:
        """Extract the metadata node at ``segs`` (``MISSING`` if the path is absent)."""
        return _md_dig(self._resolve_metadata(record), segs)

    def _md_eq_fn(self, segs: tuple[str, ...], operand: Any) -> Callable[[Any], bool]:
        """Positive metadata equality — total boolean.

        Match-mode by operand kind (ADR §4 taxonomy):

        * a ``$date`` literal compares the stored value as a UTC datetime;
        * a ``tuple`` (bare-list ``$eq``) is exact-array equality on the node;
        * a ``dict`` (sub-path subdocument operand) is ``object_equal`` — EXACT
          structural equality on the extracted node (mirrors the ADR's
          ``metadata #> '{path}' = '{...}'::jsonb``), order-insensitive on keys
          via :func:`_json_eq`. NOT ``@>`` containment;
        * any other scalar is array-aware containment (a scalar matches the node
          or an element of a stored list).
        """
        if isinstance(operand, DateLiteral):
            return lambda record: _md_date_compare(self._md_extract(record, segs), Op.EQ, operand)
        if isinstance(operand, tuple):
            expected = [_jsonable(item) for item in operand]
            return lambda record: _exact_array_eq(self._md_extract(record, segs), expected)
        if isinstance(operand, Mapping):
            # Subdocument equality: EXACT object_equal, not containment.
            expected_obj = _jsonable(operand)
            return lambda record: _md_object_eq(self._md_extract(record, segs), expected_obj)
        return lambda record: _md_contains(self._md_extract(record, segs), operand)

    def _md_range_fn(self, segs: tuple[str, ...], op: Op, operand: Any) -> Callable[[Any], bool]:
        """A metadata range op — type-gated scalar compare, total.

        A ``$date`` operand compares via UTC-datetime parse-or-exclude. Otherwise
        the stored node must match the operand's type-gate (number vs number,
        bool vs bool, str vs str) before comparing — a mismatch or absent node
        excludes (Rule 1), so the positive op never matches a wrong-typed value.
        """
        range_fn = _RANGE_FN[op]
        if isinstance(operand, DateLiteral):
            return lambda record: _md_date_compare(self._md_extract(record, segs), op, operand)
        return lambda record: _md_range_compare(self._md_extract(record, segs), operand, range_fn)

    # ----- unsupported ---------------------------------------------------- #

    def _unsupported(self, clause: FilterClause, reason: str) -> Callable[[Any], bool]:
        """Handle a clause this compiler cannot express per ``ctx.on_unsupported``.

        ``"raise"`` raises the public :class:`RecallFilterUnsupportedError`;
        ``"split"`` leaves the clause out of ``consumed_keys`` and emits a
        non-constraining match-all so it does not narrow the record set.
        """
        path = _path_str(clause.path)
        if self._ctx.on_unsupported == "raise":
            raise RecallFilterUnsupportedError(path, reason)
        return lambda _record: True


# --------------------------------------------------------------------------- #
# Module-level helpers (no per-pass state).
# --------------------------------------------------------------------------- #


def _getattr_or_missing(obj: Any, name: str) -> Any:
    """Return ``getattr(obj, name)`` or :data:`MISSING` if absent / unreadable."""
    try:
        return getattr(obj, name)
    except AttributeError:
        return MISSING


def _getitem_or_missing(obj: Any, key: str) -> Any:
    """Return ``obj[key]`` (mapping access) or :data:`MISSING` if absent / not subscriptable."""
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
        return MISSING
    return MISSING


def _system_value(value: Any) -> Any:
    """Unwrap a :class:`DateLiteral` to its datetime; pass other values through."""
    if isinstance(value, DateLiteral):
        return value.value
    return value


def _is_number(value: Any) -> bool:
    """True for an int / float operand, excluding bool (a number type-gate).

    ``bool`` is an ``int`` subclass; the §4 number gate treats a bool as a
    boolean, never a number, so it is excluded here (matching ``compile_postgres``
    ``_operand_json_type``).
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _comparable(stored: Any, operand: Any) -> bool:
    """True when ``stored`` and ``operand`` are order-comparable under §4's gate.

    Numbers compare with numbers (int/float, bool excluded), bools with bools,
    strings with strings, datetimes with datetimes. Any cross-family pair is
    NOT comparable — the positive op excludes it (Rule 1: never abort, exclude
    instead of raising a ``TypeError`` on ``str < int``).

    Two datetimes are comparable regardless of tz-awareness: a naive stored
    value (some backends — e.g. the embedded sqlite store, whose SQLAlchemy
    ``DateTime`` column is tz-naive — return naive datetimes) is normalized to
    UTC at the comparison boundary by :func:`_align_dt`, so the pair both
    compares and orders without a ``TypeError``. The gate stays a same-family
    test; the tz alignment is applied to the *values* before ``==`` / ``<`` /
    membership, never here.
    """
    if _is_number(operand):
        return _is_number(stored)
    if isinstance(operand, bool):
        return isinstance(stored, bool)
    if isinstance(operand, datetime):
        return isinstance(stored, datetime)
    if isinstance(operand, str):
        return isinstance(stored, str)
    # Any other operand type (shouldn't reach range/scalar compare) — only an
    # exact same-type pair is comparable.
    return type(stored) is type(operand)


def _to_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC-aware — naive is read as UTC (mirrors :func:`_try_parse_utc`)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _align_dt(stored: Any, value: Any) -> tuple[Any, Any]:
    """Align a comparable datetime pair to UTC-aware so ``==`` / ``<`` never raise.

    Only rewrites when BOTH operands are datetimes (the sole family where a
    naive/aware mix raises ``TypeError`` on compare). A naive operand on either
    side — stored (e.g. the embedded sqlite store) or the filter value — is read
    as UTC, mirroring :func:`_try_parse_utc` / :func:`_md_date_compare`. Any
    other pair (already-handled by :func:`_comparable`) passes through unchanged.
    """
    if isinstance(stored, datetime) and isinstance(value, datetime):
        return _to_utc(stored), _to_utc(value)
    return stored, value


# ----- system-key value comparisons (None / wrong-type aware) -------------- #


def _system_eq(stored: Any, value: Any) -> bool:
    """System ``$eq``: present, comparable, and equal. Excludes None / wrong-type."""
    if stored is None:
        return False
    if not _comparable(stored, value):
        return False
    stored, value = _align_dt(stored, value)
    return stored == value


def _system_ne(stored: Any, value: Any) -> bool:
    """System ``$ne``: include None / wrong-type (Rule 2); exclude only equal."""
    if stored is None:
        return True
    if not _comparable(stored, value):
        return True
    stored, value = _align_dt(stored, value)
    return stored != value


def _system_in(stored: Any, values: list[Any]) -> bool:
    """System ``$in``: present, comparable to, and equal to some member."""
    if stored is None:
        return False
    return any(_comparable(stored, v) and _eq_aligned(stored, v) for v in values)


def _system_nin(stored: Any, values: list[Any]) -> bool:
    """System ``$nin``: include None / wrong-type; exclude only an in-set match."""
    if stored is None:
        return True
    return not any(_comparable(stored, v) and _eq_aligned(stored, v) for v in values)


def _eq_aligned(stored: Any, value: Any) -> bool:
    """``stored == value`` with a naive/aware datetime pair aligned to UTC first."""
    stored, value = _align_dt(stored, value)
    return stored == value


def _system_range(stored: Any, value: Any, range_fn: Callable[[Any, Any], bool]) -> bool:
    """System range op: present, comparable, and in range. Excludes None / wrong-type."""
    if stored is None:
        return False
    if not _comparable(stored, value):
        return False
    stored, value = _align_dt(stored, value)
    return range_fn(stored, value)


# ----- metadata extraction + comparisons ----------------------------------- #


def _md_dig(blob: Mapping[str, Any], segs: tuple[str, ...]) -> Any:
    """Extract the node at ``segs`` from a metadata blob, or :data:`MISSING`.

    Descends one mapping per segment. A path that runs off a non-mapping or an
    absent key yields :data:`MISSING` (the absent case, distinct from a stored
    ``None``).
    """
    node: Any = blob
    for seg in segs:
        if isinstance(node, Mapping) and seg in node:
            node = node[seg]
        else:
            return MISSING
    return node


def _md_present(blob: Mapping[str, Any], segs: tuple[str, ...]) -> bool:
    """True iff the metadata path resolves (an explicit ``None`` value counts as present).

    Mirrors ``compile_postgres._md_exists``: ``#>`` is NULL only when the path is
    missing; an explicit JSON ``null`` is present. So presence is "the path
    resolves to a value", whether that value is ``None`` or not.
    """
    return _md_dig(blob, segs) is not MISSING


def _md_is_null(blob: Mapping[str, Any], segs: tuple[str, ...]) -> bool:
    """Active null-or-missing match: an explicit ``None`` value OR an absent path.

    Mirrors ``compile_postgres._md_null``. ``{k: null}`` matches both.
    """
    node = _md_dig(blob, segs)
    return node is MISSING or node is None


def _md_contains(node: Any, value: Any) -> bool:
    """Array-aware containment: a scalar matches the node, or an element of a node list.

    Mirrors the Postgres ``@>`` semantics: ``metadata @> '{"k": v}'`` matches a
    scalar field equal to ``v`` AND a list field containing ``v``. An absent node
    (:data:`MISSING`) never matches (total — False, not an error).
    """
    if node is MISSING:
        return False
    target = _jsonable(value)
    if isinstance(node, list):
        return any(_json_eq(_jsonable(item), target) for item in node)
    return _json_eq(_jsonable(node), target)


def _exact_array_eq(node: Any, expected: list[Any]) -> bool:
    """Exact-array equality: the stored node is a list equal to ``expected`` (ordered).

    Mirrors ``compile_postgres._md_eq`` tuple branch (``#> = '[...]'::jsonb``). An
    absent node or a non-list node never matches.
    """
    if node is MISSING or not isinstance(node, list):
        return False
    return _json_eq([_jsonable(item) for item in node], expected)


def _md_object_eq(node: Any, expected: Any) -> bool:
    """Subdocument ``object_equal``: the stored node EXACTLY equals ``expected``.

    The ADR §4 ``object_equal`` match-mode for a sub-path dict operand —
    ``metadata #> '{path}' = '{...}'::jsonb``, exact structural equality (NOT
    ``@>`` containment), order-insensitive on keys via :func:`_json_eq`. An absent
    node (:data:`MISSING`) or a non-mapping node never matches. ``expected`` is
    already :func:`_jsonable`-normalized.
    """
    if node is MISSING or not isinstance(node, Mapping):
        return False
    return _json_eq(_jsonable(node), expected)


def _md_range_compare(node: Any, operand: Any, range_fn: Callable[[Any, Any], bool]) -> bool:
    """Type-gated metadata range compare — total (excludes absent / wrong-type).

    Mirrors ``compile_postgres._md_range``: the stored node must match the
    operand's type-gate (number/bool/string) before comparing. A list operand is
    not a range operand (the validator never produces one here), so it is
    excluded by the comparability gate.
    """
    if node is MISSING:
        return False
    if not _comparable(node, operand):
        return False
    return range_fn(node, operand)


def _md_date_compare(node: Any, op: Op, operand: DateLiteral) -> bool:
    """Guarded ``$date`` compare on a metadata node — parse-or-exclude, total.

    Mirrors ``compile_postgres._md_date_compare``: the stored value is parsed as
    an ISO-8601 datetime (normalized to UTC) and compared to the operand's
    UTC-normalized datetime. A node that is absent, non-string, or unparseable
    excludes the record (Rule 1) — never raises.
    """
    if node is MISSING or not isinstance(node, str):
        return False
    parsed = _try_parse_utc(node)
    if parsed is None:
        return False
    cmp = operand.value
    if op == Op.EQ:
        return parsed == cmp
    return _RANGE_FN[op](parsed, cmp)


def _try_parse_utc(text: str) -> datetime | None:
    """Parse an ISO-8601 string to a UTC-aware datetime, or ``None`` on failure."""
    from datetime import UTC

    try:
        parsed = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _jsonable(value: Any) -> Any:
    """Coerce a value to a JSON-comparable form for containment / equality.

    A :class:`DateLiteral` / :class:`~datetime.datetime` renders as its ISO-8601
    string (matching ``compile_postgres._date_to_jsonable``); a mapping / list is
    coerced recursively; other scalars pass through. This keeps a metadata
    ``$date`` operand comparable to a stored ISO string and a dict / list operand
    comparable structurally.
    """
    if isinstance(value, DateLiteral):
        return value.value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_eq(a: Any, b: Any) -> bool:
    """Type-strict JSON equality, mirroring Postgres JSONB ``@>`` / ``=`` semantics.

    Both operands are already :func:`_jsonable`-normalized. JSONB distinguishes a
    boolean from a number (``true`` is not ``1``), so a Python ``True == 1`` must
    NOT match here — exactly one operand being a ``bool`` is unequal. Otherwise
    the comparison is structural: dicts compare key-sets + per-key values, lists
    compare ordered + element-wise, scalars compare with ``==``.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        # JSONB bool/number distinction: equal only if BOTH are bools and equal.
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if a.keys() != b.keys():
            return False
        return all(_json_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_eq(x, y) for x, y in zip(a, b))
    # A list / dict never equals a scalar (and vice versa); == handles that too.
    return a == b
