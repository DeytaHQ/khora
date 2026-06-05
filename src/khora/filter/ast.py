"""Canonical filter AST (Layer 3) — ``@internal``.

The AST is the desugared, normalized intermediate form that backend compilers
(Layer 4) consume. It sits between the public :class:`~khora.filter.RecallFilter`
pydantic model (Layer 2, the validated wire contract) and the per-backend
compilers (Layer 4). :func:`parse_to_ast` lowers a *validated* ``RecallFilter``
into this form; :func:`canonical_hash` derives a stable cache key from it.

``@internal``. Nothing here is exported from :mod:`khora.__init__`. The AST is
``__all__``'d under :mod:`khora.filter` only and is free to evolve in minor
releases (a new node kind for ``$regex`` later does not break the public
surface). Backend compilers reach it through the (internal) ``CompileContext``;
callers never see it.

Two lowering invariants the rest of the subsystem depends on — **compilers
never see dot-strings or sugar**:

* **Paths are always segment tuples, never dot-strings.** A system key lowers to
  a single-segment path (``("source_name",)``); a folded ``metadata.<path>``
  predicate splits on ``.`` into a multi-segment path (``("metadata", "a", "b")``).
* **All bare-value / logical sugar is desugared.** A bare scalar becomes an
  ``$eq`` clause; a bare list becomes an ``$eq`` *exact-array* clause (NOT ``$in``
  — membership is the explicit ``$in`` form); ``$nor`` becomes ``$not($or(...))``;
  implicit sibling AND (several keys on one filter document) becomes an explicit
  AND node.

Comparison operands stay **opaque** — they are carried verbatim, never recursed
into as nested clauses (a ``$or`` nested inside an ``$eq`` operand is matched as
a literal object, mirroring ``model._check_literal_operand``). The one
recognized typed literal is ``{"$date": "<ISO-8601>"}``, which lowers to a
:class:`DateLiteral` operand at the leaf.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from khora.filter.model import (
    DateOps,
    Op,
    RecallFilter,
    StringOps,
)

__all__ = [
    "DateLiteral",
    "FilterClause",
    "FilterNode",
    "FilterOp",
    "canonical_hash",
    "parse_to_ast",
]


# ``FilterOp`` is the AST's operator vocabulary. It is the same closed set the
# wire model speaks, so it is a direct alias of ``model.Op`` rather than a
# parallel enum that could drift out of sync.
FilterOp = Op


# The logical operators that compose child nodes (as opposed to leaf
# comparison operators). ``$nor`` is desugared away in the lowering step, so it
# never reaches the AST as a node kind.
_LOGICAL_OPS: frozenset[FilterOp] = frozenset({Op.AND, Op.OR, Op.NOT})

# The comparison operators whose operand is an ordered list whose element order
# is semantically significant (so ``canonical_hash`` must preserve it).
_LIST_OPS: frozenset[FilterOp] = frozenset({Op.IN, Op.NIN})


# --------------------------------------------------------------------------- #
# Leaf operand: the typed ``$date`` literal.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DateLiteral:
    """A normalized ``{"$date": "<ISO-8601>"}`` typed-literal operand.

    ``@internal``. A ``$date`` literal appearing in operand position on a
    metadata predicate is lowered to this carrier so compilers receive an
    unambiguous datetime instead of re-parsing a string. The validator already
    proved the string is ISO-8601 and normalized it to UTC (``model._parse_date_literal``
    / ``DateOps._utc_*``); this just carries the resulting ``datetime``.

    ``value`` is a tz-aware UTC :class:`~datetime.datetime`.
    """

    value: datetime


# --------------------------------------------------------------------------- #
# Leaf predicate.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FilterClause:
    """A single leaf predicate: ``path <op> operand``.

    ``@internal``. ``path`` is the segment tuple addressing the field — a
    single-segment tuple for a system key (``("source_name",)``) or a
    multi-segment tuple for a metadata sub-path (``("metadata", "labels", "tier")``).
    Never a dot-string.

    ``op`` is the comparison operator (never a logical operator — those are
    :class:`FilterNode` kinds). ``operand`` is the opaque comparison value:

    * a scalar / :class:`~datetime.datetime` / :class:`DateLiteral` for the
      scalar comparison ops (``$eq``/``$ne``/``$gt``/``$gte``/``$lt``/``$lte``),
    * an ordered ``tuple`` for ``$in``/``$nin`` (element order significant),
    * a ``tuple`` for an ``$eq`` *exact-array* operand (a bare list desugars to
      ``$eq`` exact-array — NOT ``$in``; element order significant),
    * a ``bool`` for ``$exists``,
    * ``None`` for an active null-or-missing match (``$eq None``).

    Operands are carried verbatim and never recursed into as clauses.
    """

    path: tuple[str, ...]
    op: FilterOp
    operand: Any = None


# --------------------------------------------------------------------------- #
# Logical composition node.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FilterNode:
    """A logical-composition node: ``AND`` / ``OR`` / ``NOT`` over children.

    ``@internal``. The :func:`parse_to_ast` normalization pass guarantees the root
    is **always** a ``FilterNode`` (never a bare :class:`FilterClause`): a
    single-predicate filter is ``AND([clause])``, an implicit-AND or multi-predicate
    filter is an ``AND`` of its predicates, a bare ``$or`` / ``$not`` filter has
    that logical node as root, and an empty filter is the empty match-everything
    ``AND``. Same-operator nesting is flattened (no normalized ``AND`` directly
    contains an ``AND`` child, nor ``OR`` an ``OR``), and a single-child
    ``AND``/``OR`` whose child is a *logical* node is collapsed away — but a
    single *leaf* child is kept as ``AND([leaf])`` so the node never degenerates
    to a bare clause.

    ``op`` is one of ``Op.AND`` / ``Op.OR`` / ``Op.NOT``. ``children`` is the
    ordered tuple of operands — each a :class:`FilterClause` or a nested
    :class:`FilterNode`. ``Op.NOT`` carries exactly one child; a normalized
    ``AND`` / ``OR`` carries zero (the empty match-everything ``AND``), one (only
    when that child is a single leaf clause, i.e. ``AND([leaf])``), or
    two-or-more children.

    Child order is preserved as authored. ``canonical_hash`` is what imposes the
    order-insensitivity of the commutative ``AND`` / ``OR`` operators — the node
    itself keeps authored order so a round-trip / debug dump is faithful.
    """

    op: FilterOp
    children: tuple[FilterNode | FilterClause, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Lowering: RecallFilter -> FilterNode.
# --------------------------------------------------------------------------- #


def parse_to_ast(filter_: RecallFilter) -> FilterNode:
    """Lower a *validated* :class:`RecallFilter` into the canonical AST.

    ``@internal``. The input must already have passed
    :meth:`RecallFilter.model_validate` (or kwarg construction) — this function
    does not re-validate; it desugars and normalizes:

    * system-key fields → single-segment :class:`FilterClause` leaves,
    * folded ``metadata.<path>`` predicates → multi-segment leaves,
    * the bare ``metadata`` blob → whole-blob ``$eq`` leaf,
    * bare scalar → ``$eq``; bare list → ``$eq`` exact-array (NOT ``$in``),
    * ``$nor`` → ``$not($or(...))``,
    * field-level and document-level ``$not`` → ``NOT`` nodes,
    * several sibling predicates on one document → an explicit ``AND`` node.

    A final :func:`_normalize` pass canonicalizes the logical structure so that
    semantically-equal filters share a structure (and therefore a
    :func:`canonical_hash`):

    * **Same-operator nesting is flattened** — an ``AND`` directly containing an
      ``AND`` (or ``OR``-of-``OR``) splices the grandchildren up (associativity).
      Flattening never crosses a ``NOT`` boundary and never merges ``AND`` into
      ``OR``.
    * **A single-child ``AND``/``OR`` whose child is a *logical* node collapses to
      that child** — a lone ``$and``/``$or`` wrapper around another logical node
      disappears. A single-child node whose child is a *leaf clause* is rewritten
      to ``AND([leaf])`` (the op is normalized to ``AND``) — it is **not**
      collapsed to a bare clause, so the root is always a logical node.

    **The root is always a** :class:`FilterNode` — never a bare
    :class:`FilterClause`. This is the engine boundary contract: a compiler is a
    ``Callable[[FilterNode, CompileContext], CompiledFilter]``, so a bare-clause
    root would not be a valid compiler input. A single-predicate filter therefore
    normalizes to ``AND([clause])``.

    Result shape after normalization:

    * a single-predicate filter (``{a}``) → ``AND([a])``;
    * an implicit-AND filter (``{a, b}``) and the explicit ``{$and:[{a},{b}]}``
      both → ``AND([a, b])`` — identical structure, identical hash;
    * a bare ``$or`` / ``$not`` filter has that logical node as the root (no
      spurious ``AND`` wrap);
    * a bare ``RecallFilter()`` (no predicates) is the empty ``AND`` —
      match-everything; compilers treat it as "no constraint".
    """
    children = _lower_filter(filter_)
    # Always wrap the document's siblings in an AND, then normalize. ``_normalize``
    # collapses the redundant wrapper when the sole child is itself a logical node
    # (e.g. a bare ``$or``/``$not`` document), but keeps an ``AND([clause])`` when
    # the sole child is a leaf — so normalizing a FilterNode always yields a
    # FilterNode (never a bare clause). The isinstance assert both documents and
    # type-narrows that root invariant.
    root = _normalize(FilterNode(op=Op.AND, children=tuple(children)))
    assert isinstance(root, FilterNode)  # noqa: S101 - root-shape invariant (see _normalize rule 2)
    return root


def _lower_filter(filter_: RecallFilter) -> list[FilterNode | FilterClause]:
    """Lower one filter document into its list of sibling AST operands.

    Each system-key field, folded metadata predicate, bare-metadata blob, and
    logical operator on this document contributes one sibling. Sibling
    predicates are implicitly AND-ed (the caller wraps them in an ``AND`` node).
    """
    out: list[FilterNode | FilterClause] = []
    set_fields = filter_.model_fields_set

    # --- system keys -------------------------------------------------------- #
    # ``model_fields_set`` distinguishes an explicit ``null`` (active
    # null-or-missing match) from an unset field (no filter). Only
    # fields the caller actually provided contribute a clause.
    for key in _SYSTEM_KEY_FIELDS:
        if key not in set_fields:
            continue
        value = getattr(filter_, key)
        out.extend(_lower_system_key((key,), value))

    # --- bare metadata blob ------------------------------------------------- #
    # A bare ``metadata`` dict is whole-blob ``$eq`` equality (Mongo embedded-doc
    # equality). Folded ``metadata.<path>`` predicates live on a separate carrier
    # and are handled below, so this is only the bare-blob form.
    if "metadata" in set_fields and filter_.metadata is not None:
        out.append(FilterClause(path=("metadata",), op=Op.EQ, operand=filter_.metadata))

    # --- folded metadata.<path> predicates ---------------------------------- #
    for dotted_key, predicate in filter_.folded_predicates_.items():
        # ``metadata.a.b`` -> ("metadata", "a", "b"). The leading "metadata."
        # prefix is part of the key; split it whole so the path keeps the
        # ``metadata`` root segment.
        path = tuple(dotted_key.split("."))
        out.extend(_lower_metadata_predicate(path, predicate))

    # --- logical operators -------------------------------------------------- #
    if filter_.and_ is not None:
        out.append(FilterNode(op=Op.AND, children=_lower_branches(filter_.and_)))
    if filter_.or_ is not None:
        out.append(FilterNode(op=Op.OR, children=_lower_branches(filter_.or_)))
    if filter_.nor_ is not None:
        # ``$nor`` desugars to ``$not($or(...))`` — no distinct AST node.
        or_node = FilterNode(op=Op.OR, children=_lower_branches(filter_.nor_))
        out.append(FilterNode(op=Op.NOT, children=(or_node,)))
    if filter_.not_ is not None:
        # Document-form ``$not``: negate the whole inner filter.
        out.append(FilterNode(op=Op.NOT, children=(parse_to_ast(filter_.not_),)))

    return out


def _lower_branches(branches: list[RecallFilter]) -> tuple[FilterNode | FilterClause, ...]:
    """Lower each branch of a logical array to a child AST node."""
    return tuple(parse_to_ast(branch) for branch in branches)


# --------------------------------------------------------------------------- #
# Logical-structure normalization (canonical form for stable hashing).
# --------------------------------------------------------------------------- #


def _normalize(node: FilterNode | FilterClause) -> FilterNode | FilterClause:
    """Canonicalize logical structure so semantically-equal filters match.

    Recurses bottom-up. A leaf :class:`FilterClause` is returned unchanged. A
    ``NOT`` normalizes its single operand and re-wraps it (``NOT`` is an opaque
    boundary — its operand is never spliced/collapsed into a parent). An
    ``AND`` / ``OR`` node ``N`` is rewritten by:

    1. **Flatten same-operator children.** An ``AND`` child of an ``AND`` (or
       ``OR`` of ``OR``) is spliced in place of itself, lifting its grandchildren
       up one level (associativity). Per-operator only — an ``OR`` child of an
       ``AND`` is left intact, and a ``NOT`` is never spliced across.
    2. **Single-child handling.** After flattening, if ``N`` has exactly one
       child:

       * child is a **logical node** (``AND``/``OR``/``NOT``) → **collapse**:
         replace ``N`` with that child (the wrapper is redundant);
       * child is a **leaf clause** → rewrite to ``AND([leaf])``. The operator
         is normalized to ``AND`` **even when ``N`` was an ``OR``** — a lone
         ``OR`` and a lone ``AND`` around a single clause are semantically
         identical, so the original ``OR`` is intentionally dropped. **Do not**
         collapse to a bare clause.

       An *empty* ``AND``/``OR`` is preserved (the empty ``AND`` is the
       match-everything root).

    Rule (2) keeps the AST root a :class:`FilterNode` — a single-predicate filter
    becomes ``AND([clause])``, never a bare clause — which is the engine-boundary
    contract (a compiler takes a ``FilterNode``). The rewrites still converge the
    locked equivalences: ``{a, b}`` and ``{$and:[{a},{b}]}`` → ``AND([a, b])``;
    ``{$and:[{a}]}`` → ``AND([AND([a])])`` → flatten → ``AND([a])``;
    ``{$or:[{a}]}`` → ``OR([AND([a])])`` → collapse single logical child →
    ``AND([a])``.
    """
    if isinstance(node, FilterClause):
        return node

    # Normalize children first (bottom-up), so a child AND/OR is already
    # flattened/collapsed before we decide whether to splice it.
    norm_children = [_normalize(child) for child in node.children]

    if node.op == Op.NOT:
        # NOT is an opaque boundary — normalize its operand but never splice
        # across it. A NOT always has exactly one child (may be a clause).
        return FilterNode(op=Op.NOT, children=tuple(norm_children))

    # AND / OR: flatten same-operator children (associativity), never merging a
    # different operator or crossing a NOT.
    flattened: list[FilterNode | FilterClause] = []
    for child in norm_children:
        if isinstance(child, FilterNode) and child.op == node.op:
            flattened.extend(child.children)
        else:
            flattened.append(child)

    if len(flattened) == 1:
        only = flattened[0]
        if isinstance(only, FilterNode):
            # Single logical child -> collapse the redundant wrapper.
            return only
        # Single leaf child -> normalize the op to AND and keep AND([leaf]) so
        # the result is a FilterNode, never a bare clause.
        return FilterNode(op=Op.AND, children=(only,))
    return FilterNode(op=node.op, children=tuple(flattened))


# --------------------------------------------------------------------------- #
# System-key lowering.
# --------------------------------------------------------------------------- #


def _lower_system_key(
    path: tuple[str, ...],
    value: datetime | DateOps | StringOps | str | list[Any] | None,
) -> list[FilterNode | FilterClause]:
    """Lower one system-key field value to its AST leaves.

    Bare scalar -> ``$eq``; bare list -> ``$eq`` *exact-array* (NOT ``$in``);
    an ``*Ops`` submodel -> one clause per set operator (a field-level ``$not``
    becomes a ``NOT`` node wrapping its inner operator-expression).
    """
    if value is None:
        # Explicit null -> active null-or-missing $eq match.
        return [FilterClause(path=path, op=Op.EQ, operand=None)]
    if isinstance(value, (DateOps, StringOps)):
        return _lower_ops_submodel(path, value)
    if isinstance(value, list):
        # Bare list ⇒ $eq EXACT-ARRAY equality. This is NOT $in: a bare list is
        # the uniform "bare value ⇒ $eq" sugar. Membership ("one of") is never
        # synthesized from a bare list — it is the explicit StringOps(in_=[...])
        # form.
        return [FilterClause(path=path, op=Op.EQ, operand=tuple(value))]
    # Bare scalar (str / datetime) ⇒ $eq sugar.
    return [FilterClause(path=path, op=Op.EQ, operand=value)]


def _lower_ops_submodel(
    path: tuple[str, ...],
    ops: DateOps | StringOps,
) -> list[FilterNode | FilterClause]:
    """Lower a ``StringOps`` / ``DateOps`` submodel to per-operator leaves.

    Each set operator field becomes one clause. A field-level ``$not`` negates
    the inner operator-expression: it lowers to a ``NOT`` node wrapping the AST
    of the inner ``*Ops``. ``$in``/``$nin`` operands become order-significant
    tuples.
    """
    out: list[FilterNode | FilterClause] = []
    set_fields = ops.model_fields_set
    for attr, op in _OPS_FIELD_TO_OP:
        if attr not in set_fields:
            continue
        operand = getattr(ops, attr)
        if op == Op.NOT:
            # Field-level $not: negate the inner operator-expression. ``operand``
            # is itself an ``*Ops`` submodel.
            inner = _lower_ops_submodel(path, operand)
            out.append(FilterNode(op=Op.NOT, children=tuple(inner)))
            continue
        if op in _LIST_OPS:
            out.append(FilterClause(path=path, op=op, operand=tuple(operand)))
            continue
        out.append(FilterClause(path=path, op=op, operand=operand))
    return out


# --------------------------------------------------------------------------- #
# Metadata-predicate lowering (free-form, validator-checked grammar).
# --------------------------------------------------------------------------- #


def _lower_metadata_predicate(
    path: tuple[str, ...],
    predicate: Any,
) -> list[FilterNode | FilterClause]:
    """Lower one (already-validated) metadata-field predicate to AST leaves.

    Mirrors the validator's ``_walk_predicate`` shape (the predicate is known
    well-formed here):

    * a non-dict (scalar / list) -> ``$eq`` sugar (a list is an exact-array
      operand, NOT ``$in``),
    * an empty / all-non-``$`` dict -> whole-subdocument ``$eq`` equality,
    * a sole-key ``{"$date": ...}`` -> ``$eq`` :class:`DateLiteral`,
    * an operator-expression -> one clause per operator (field ``$not`` -> NOT).
    """
    if not isinstance(predicate, dict):
        if isinstance(predicate, list):
            # Bare list ⇒ $eq exact-array (NOT $in), same uniform rule as above.
            return [FilterClause(path=path, op=Op.EQ, operand=tuple(predicate))]
        return [FilterClause(path=path, op=Op.EQ, operand=predicate)]

    if not predicate:
        # Empty {} ⇒ whole-subdocument equality (opaque operand).
        return [FilterClause(path=path, op=Op.EQ, operand={})]

    dollar_keys = [k for k in predicate if isinstance(k, str) and k.startswith("$")]
    if not dollar_keys:
        # All non-$ keys ⇒ whole-subdocument equality. Opaque operand, not
        # recursed (the validator already rejected nested operators here).
        return [FilterClause(path=path, op=Op.EQ, operand=predicate)]

    # A sole-key {"$date": ...} in value position is a typed literal == $eq.
    if set(predicate.keys()) == {Op.DATE.value}:
        return [FilterClause(path=path, op=Op.EQ, operand=_make_date_literal(predicate[Op.DATE.value]))]

    # Operator expression: one leaf per operator.
    out: list[FilterNode | FilterClause] = []
    for op_key, operand in predicate.items():
        op = Op(op_key)
        if op == Op.NOT:
            # Field-position $not negates an inner operator-expression (a dict).
            inner = _lower_metadata_operator_expr(path, operand)
            out.append(FilterNode(op=Op.NOT, children=tuple(inner)))
            continue
        if op in _LIST_OPS:
            out.append(FilterClause(path=path, op=op, operand=_lower_operand_list(operand)))
            continue
        if op == Op.EXISTS:
            out.append(FilterClause(path=path, op=op, operand=operand))
            continue
        # Scalar comparison ($eq/$ne/$gt/$gte/$lt/$lte): opaque operand, with the
        # one recognized {"$date": ...} typed-literal lowered to a DateLiteral.
        out.append(FilterClause(path=path, op=op, operand=_lower_scalar_operand(operand)))
    return out


def _lower_metadata_operator_expr(
    path: tuple[str, ...],
    expr: dict[str, Any],
) -> list[FilterNode | FilterClause]:
    """Lower the inner operator-expression of a field-position ``$not``.

    The operand is a pure operator-expression (validator-guaranteed all-``$``).
    Each inner operator becomes a clause; the caller wraps the list in a ``NOT``
    node. A nested ``$not`` recurses into another ``NOT`` node — the validator
    permits ``$not`` inside a negated operator-expression (``_walk_not_operand``
    re-validates each inner op, including ``$not``), so the lowering must handle
    it rather than emit a malformed ``FilterClause(op=NOT, operand=<raw dict>)``.
    """
    out: list[FilterNode | FilterClause] = []
    for op_key, operand in expr.items():
        op = Op(op_key)
        if op == Op.NOT:
            inner = _lower_metadata_operator_expr(path, operand)
            out.append(FilterNode(op=Op.NOT, children=tuple(inner)))
        elif op in _LIST_OPS:
            out.append(FilterClause(path=path, op=op, operand=_lower_operand_list(operand)))
        elif op == Op.EXISTS:
            out.append(FilterClause(path=path, op=op, operand=operand))
        else:
            out.append(FilterClause(path=path, op=op, operand=_lower_scalar_operand(operand)))
    return out


def _lower_operand_list(operand: list[Any]) -> tuple[Any, ...]:
    """Lower an ``$in``/``$nin`` operand list (each item may be a ``$date``)."""
    return tuple(_lower_scalar_operand(item) for item in operand)


def _lower_scalar_operand(operand: Any) -> Any:
    """Lower an opaque comparison operand, recognizing the ``$date`` literal.

    A sole-key ``{"$date": "<ISO-8601>"}`` becomes a :class:`DateLiteral`; any
    other value is carried verbatim (opaque — never recursed into as a clause).
    """
    if isinstance(operand, dict) and set(operand.keys()) == {Op.DATE.value}:
        return _make_date_literal(operand[Op.DATE.value])
    return operand


def _make_date_literal(raw: Any) -> DateLiteral:
    """Build a normalized :class:`DateLiteral` from a validated ``$date`` value.

    The validator already proved ``raw`` parses as ISO-8601; normalize to UTC
    the same way ``DateOps`` does so the AST carries a uniform tz-aware value.
    """
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return DateLiteral(value=parsed)


# Ordered (field-name, Op) pairs for the ``*Ops`` submodels. Order is fixed so
# lowering is deterministic; ``canonical_hash`` re-sorts commutative siblings,
# but a stable emit order keeps debug dumps readable.
_OPS_FIELD_TO_OP: tuple[tuple[str, FilterOp], ...] = (
    ("eq", Op.EQ),
    ("ne", Op.NE),
    ("gt", Op.GT),
    ("gte", Op.GTE),
    ("lt", Op.LT),
    ("lte", Op.LTE),
    ("in_", Op.IN),
    ("nin", Op.NIN),
    ("exists", Op.EXISTS),
    ("not_", Op.NOT),
)

# The system-key field names on ``RecallFilter`` in a fixed order (the ten
# system keys). ``metadata`` and the logical operators are handled separately.
_SYSTEM_KEY_FIELDS: tuple[str, ...] = (
    "occurred_at",
    "created_at",
    "source_timestamp",
    "source_type",
    "source_name",
    "source_url",
    "external_id",
    "content_type",
    "source",
    "title",
)


# --------------------------------------------------------------------------- #
# Canonical hashing.
# --------------------------------------------------------------------------- #


def canonical_hash(node: FilterNode | FilterClause) -> str:
    """Return a stable SHA-256 hex digest of an AST.

    ``@internal``. The hash is the cache-key source (the engine assembles the
    full recall cache key from it). Stability rules:

    * **Commutative** ``$and`` / ``$or`` siblings are sorted, so reordering
      sibling predicates does not change the hash (``{a AND b}`` == ``{b AND a}``).
    * **Order is preserved** for a ``$not`` operand, for ``$in``/``$nin`` lists,
      and for ``$eq`` exact-array operands — those are order-significant.
    * **Dict operands** (whole-subdocument / whole-blob equality) have their keys
      sorted so semantically equal objects hash equally.
    * Two ASTs that differ only in **semantics** (different op, path, or operand)
      hash differently.

    Implementation: build a canonical, JSON-serializable representation, then
    SHA-256 over its compact JSON encoding. All ordering normalization happens
    while building that representation, in two distinct places — commutative
    ``$and`` / ``$or`` children are sorted by their own canonical serialization
    in :func:`_canonicalize`, and dict-operand keys are sorted recursively in
    :func:`_canonicalize_operand`. The final :func:`json.dumps` deliberately
    uses ``sort_keys=False``: the representation's field order is already fixed
    and the meaningful sorting is baked into the structure, so re-sorting the
    encoding keys would be redundant (and must not be relied on for stability).
    """
    canonical = _canonicalize(node)
    encoded = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonicalize(node: FilterNode | FilterClause) -> Any:
    """Build the stable, JSON-serializable representation of an AST subtree."""
    if isinstance(node, FilterClause):
        return {
            "k": "clause",
            "path": list(node.path),
            "op": node.op.value,
            "operand": _canonicalize_operand(node.operand),
        }
    # FilterNode (logical).
    children = [_canonicalize(child) for child in node.children]
    if node.op in (Op.AND, Op.OR):
        # Commutative: sort siblings by their stable serialization so order is
        # irrelevant. The serialization is deterministic, so sorting on it is a
        # total, stable order.
        children = sorted(children, key=lambda c: json.dumps(c, separators=(",", ":"), sort_keys=False))
    # NOT preserves its single operand's position (no sort).
    return {"k": "node", "op": node.op.value, "children": children}


def _canonicalize_operand(operand: Any) -> Any:
    """Canonicalize a leaf operand into JSON-serializable form.

    Tuples (``$in``/``$nin`` lists, ``$eq`` exact-arrays) keep their order. Dict
    operands (subdocument / blob equality) sort their keys recursively so
    key-order is not significant. :class:`DateLiteral` and :class:`~datetime.datetime`
    are tagged + ISO-encoded so they never collide with a plain string.
    """
    if isinstance(operand, DateLiteral):
        return {"$date": operand.value.isoformat()}
    if isinstance(operand, datetime):
        return {"$dt": operand.isoformat()}
    if isinstance(operand, tuple):
        # Order-significant — preserve it.
        return [_canonicalize_operand(item) for item in operand]
    if isinstance(operand, list):
        return [_canonicalize_operand(item) for item in operand]
    if isinstance(operand, dict):
        # Order-insensitive object equality — sort keys recursively.
        return {k: _canonicalize_operand(operand[k]) for k in sorted(operand, key=str)}
    return operand
