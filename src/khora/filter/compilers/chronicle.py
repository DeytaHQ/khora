"""Chronicle recall-filter compiler — ``@internal``.

Lowers a canonical :class:`~khora.filter.ast.FilterNode` to a narrowing
**date-bound** the Chronicle engine intersects with its recency window. Chronicle
has no general predicate-pushdown surface (its retrieval fans out across four
channels through the storage coordinator), but every channel already honors a
``created_after`` / ``created_before`` window that narrows on
``COALESCE(source_timestamp, created_at)`` (pgvector + sqlite_lance alike) — the
**source_timestamp axis** with a ``created_at`` fallback. The one key this
compiler can push down same-axis-safely is a bound on ``source_timestamp``.
Everything else is left **unconsumed** for the engine's
:func:`~khora.filter.compilers.python.compile_python` post-filter (the full-AST
safety net), so no constraint is ever silently dropped.

The output is a :class:`~khora.filter.registry.CompiledFilter` whose
``predicate`` is a :class:`ChronicleDateBound` (a small frozen ``(created_after,
created_before)`` pair, either side ``None`` for unbounded) and whose
``consumed_keys`` are exactly the date keys folded into that bound.

**Why only ``source_timestamp`` pushes down (the same-axis rule).** The recency
window column is ``COALESCE(source_timestamp, created_at)``. Pushing a bound on a
different axis would apply it to the wrong timestamp and silently drop rows:

* ``source_timestamp`` is the window's primary axis — for every row a
  ``source_timestamp >= A`` filter keeps, ``source_timestamp`` is non-null and
  equals the window value, so the pushdown agrees with the filter; a null-
  ``source_timestamp`` row is admitted by the window's ``created_at`` fallback and
  then dropped by the post-filter (superset-safe — the pushed set ⊇ the
  post-filtered set, never the reverse). **Pushed down.**
* ``occurred_at`` is the EVENT-time axis — ``record.occurred_at`` is
  ``COALESCE(occurred_at, source_timestamp)`` (the chunk's real ``occurred_at``
  when carried, else ``source_timestamp``; see the engine's ``_chunk_to_record``
  and the matching ``RecallChunk.occurred_at`` surface derivation). That differs
  from the window's ``COALESCE(source_timestamp, created_at)`` axis, so pushing it
  would false-exclude. **Not pushed**; enforced by the post-filter against the
  event-time value.
* ``created_at`` == the literal ingest column. The window uses ``created_at`` only
  as a *fallback*, so a ``created_at`` bound is cross-axis. **Not pushed**;
  post-filtered against the literal ``created_at``.

**Per-key support matrix (Chronicle chunk read path).**

* ``source_timestamp`` — a *conjunctive* range / ``$eq`` clause folds into the
  date-bound and is **consumed** (``$gt`` / ``$gte`` tighten ``created_after``;
  ``$lt`` / ``$lte`` tighten ``created_before``; ``$eq`` pins both to a point
  window). "Conjunctive" means the clause sits in the top-level ``AND``; under an
  ``$or`` / ``$not`` it is **not** consumed (folding a bound out of a disjunction
  / negation would wrongly narrow the other branch). ``$ne`` / ``$in`` / ``$nin``
  / ``{k: null}`` are not a single contiguous window, so they are also left
  unconsumed. In every unconsumed case the post-filter still enforces the
  predicate against the literal ``source_timestamp``.
* ``occurred_at`` — NOT pushed (event-time axis, distinct from the window);
  post-filtered against ``record.occurred_at`` = ``COALESCE(occurred_at,
  source_timestamp)`` (honors the real event time; matches ``RecallChunk``).
* ``created_at`` — NOT pushed (cross-axis with the coalesced window); post-filtered
  against the literal ``created_at`` field (carried on both backends → correct).
* the eight denormalized document keys (``source_type`` / ``source_name`` /
  ``source_url`` / ``external_id`` / ``content_type`` / ``source`` / ``title``)
  — NOT pushed; left for the post-filter. Carried on the ``khora_chunks`` schema
  but NOT on the legacy pgvector ``chunks`` DTO, so positive predicates on them
  return empty on the legacy path (documented limitation — the ADR
  population-caveat / no ``chunks`` migration in V3-a).
* ``metadata.<path>`` — NOT pushed; post-filtered against the carried
  ``chunk.metadata`` dict (carried on every backend → correct).

This compiler raises :class:`RecallFilterUnsupportedError` only when
``ctx.on_unsupported == "raise"`` and a clause is not a consumable date bound —
the Chronicle engine always drives it with ``"split"``, so in practice nothing
raises and the unconsumed remainder is handed to the post-filter.

``@internal``. Reachable as ``khora.filter.compilers.chronicle.compile_chronicle``
for khora's own engines; not re-exported from :mod:`khora.__init__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
from khora.filter.model import Op

__all__ = ["ChronicleDateBound", "compile_chronicle"]


# The ONLY date key that pushes down. The recency window narrows on
# ``COALESCE(source_timestamp, created_at)`` — the source_timestamp axis with a
# created_at fallback — so only a ``source_timestamp`` bound is same-axis-safe
# (for every row a ``source_timestamp >= A`` filter keeps, source_timestamp is
# non-null and equals the window value; null-source_timestamp rows are admitted
# via the created_at fallback and then dropped by the post-filter — superset-safe,
# never a false-exclude). ``occurred_at`` is the EVENT-time axis
# (COALESCE(occurred_at, source_timestamp); see the engine's _chunk_to_record /
# RecallChunk.occurred_at), a different dimension than the window, so it is NOT
# pushed — it is enforced by the post-filter. ``created_at`` (the window's
# fallback, not its primary axis) is likewise post-filter-only.
_PUSHDOWN_DATE_KEYS: frozenset[str] = frozenset({"source_timestamp"})

# Range ops that tighten the lower / upper bound of the window.
_LOWER_OPS: frozenset[Op] = frozenset({Op.GT, Op.GTE})
_UPPER_OPS: frozenset[Op] = frozenset({Op.LT, Op.LTE})


@dataclass(frozen=True, slots=True)
class ChronicleDateBound:
    """A narrowing date window distilled from the consumed date-key clauses.

    ``@internal``. ``created_after`` is the inclusive/exclusive lower bound and
    ``created_before`` the upper bound (either ``None`` for unbounded on that
    side). Both are tz-aware UTC datetimes (the AST's ``DateLiteral`` / system
    datetime operands are UTC-normalized by the validator).

    The engine intersects this with its existing recency window by taking the
    ``max`` of the two lower bounds and the ``min`` of the two upper bounds
    (narrow only — the filter can shrink the window but never widen it). The
    bound is distilled ONLY from ``source_timestamp`` clauses, the window's
    primary axis (``COALESCE(source_timestamp, created_at)``), so the pushdown is
    same-axis and superset-safe: the pushed candidate set ⊇ the post-filtered set
    (never the reverse) — no false-exclude. A null-``source_timestamp`` row is
    admitted via the window's ``created_at`` fallback and then dropped by the
    post-filter.

    The boundary strictness of the range op (``$gt`` vs ``$gte``) is intentionally
    NOT carried: the window narrows to the bound instant, and
    :func:`compile_python` re-checks the exact strict/non-strict comparison on
    every surviving candidate, so an off-by-one-instant row at the boundary is
    dropped by the post-filter. This keeps the pushed-down window a safe
    over-approximation (it never drops a row the filter would keep).
    """

    created_after: datetime | None = None
    created_before: datetime | None = None


def compile_chronicle(ast: FilterNode, ctx: CompileContext) -> CompiledFilter[ChronicleDateBound]:
    """Compile a canonical AST to a Chronicle :class:`ChronicleDateBound`.

    ``ast`` is always a :class:`FilterNode` (the ``parse_to_ast`` root invariant).
    Only conjunctive ``source_timestamp`` range / ``$eq`` clauses (the direct
    children of the top-level ``AND``) fold into the bound and are added to
    ``consumed_keys`` — ``source_timestamp`` is the recency window's primary axis
    (``COALESCE(source_timestamp, created_at)``), so its pushdown is same-axis and
    superset-safe. Every other key — ``occurred_at`` (event-time axis) /
    ``created_at`` (the window's fallback), a ``source_timestamp`` clause under an
    ``$or`` / ``$not``, the eight denormalized document keys, and all metadata —
    is left unconsumed for the engine's :func:`compile_python` post-filter.
    ``params`` is always empty (the bound carries the datetimes directly).

    ``ctx.on_unsupported`` governs the unconsumed remainder: ``"split"`` (the
    Chronicle engine's mode) omits it silently; ``"raise"`` raises
    :class:`RecallFilterUnsupportedError` for the first non-consumable clause.
    """
    consumed: set[str] = set()
    lower, upper = _fold_top_level_dates(ast, ctx, consumed)
    return CompiledFilter(
        predicate=ChronicleDateBound(created_after=lower, created_before=upper),
        params={},
        consumed_keys=frozenset(consumed),
        canonical_hash=canonical_hash(ast),
    )


def _fold_top_level_dates(
    ast: FilterNode,
    ctx: CompileContext,
    consumed: set[str],
) -> tuple[datetime | None, datetime | None]:
    """Fold the conjunctive date-key clauses into a ``(lower, upper)`` window.

    Only the top-level ``AND`` conjunction is safe to push down (see
    :class:`ChronicleDateBound`). A non-``AND`` root (a bare ``$or`` / ``$not``)
    pushes nothing down; under ``"raise"`` mode it surfaces its first clause as
    unsupported.
    """
    lower: datetime | None = None
    upper: datetime | None = None

    if ast.op != Op.AND:
        # A bare $or / $not root is not a conjunction — nothing folds into the
        # window. Honor on_unsupported for the (non-date or non-conjunctive)
        # content the engine must post-filter.
        _reject_if_raise(ast, ctx)
        return lower, upper

    for child in ast.children:
        bound = _consumable_date_clause(child)
        if bound is None:
            # Not a consumable conjunctive date clause — leave it unconsumed for
            # the post-filter (or raise under "raise" mode).
            _reject_if_raise(child, ctx)
            continue
        key, op, value = bound
        if op == Op.EQ:
            lower = _max_lower(lower, value)
            upper = _min_upper(upper, value)
        elif op in _LOWER_OPS:
            lower = _max_lower(lower, value)
        else:  # op in _UPPER_OPS
            upper = _min_upper(upper, value)
        consumed.add(key)

    return lower, upper


def _consumable_date_clause(
    node: FilterNode | FilterClause,
) -> tuple[str, Op, datetime] | None:
    """Return ``(key, op, datetime)`` if ``node`` is a pushdownable date clause, else ``None``.

    A pushdownable clause is a leaf on ``source_timestamp`` (the window's primary axis,
    ``COALESCE(source_timestamp, created_at)``) with a range op (``$gt``/``$gte``/
    ``$lt``/``$lte``) or ``$eq`` and a datetime / :class:`DateLiteral` operand.
    ``occurred_at`` (event-time axis) and ``created_at`` (window's fallback, not
    primary axis) are cross-dimensional and post-filtered instead. ``$ne`` / ``$in``
    / ``$nin`` / ``{k: null}`` on any date key, and any logical sub-node, return
    ``None`` (not a single contiguous window).
    """
    if not isinstance(node, FilterClause):
        return None
    if len(node.path) != 1 or node.path[0] not in _PUSHDOWN_DATE_KEYS:
        return None
    if node.op != Op.EQ and node.op not in _LOWER_OPS and node.op not in _UPPER_OPS:
        return None
    value = _as_datetime(node.operand)
    if value is None:
        return None
    return node.path[0], node.op, value


def _as_datetime(operand: object) -> datetime | None:
    """Coerce a date-clause operand to a datetime, or ``None`` if it is not one.

    A bare ``$eq`` exact-array (tuple) operand, a ``None`` (``{k: null}``), or any
    non-datetime value is not a window bound.
    """
    if isinstance(operand, DateLiteral):
        return operand.value
    if isinstance(operand, datetime):
        return operand
    return None


def _max_lower(current: datetime | None, candidate: datetime) -> datetime:
    """Tighten a lower bound: the later of the two (``None`` = unbounded below)."""
    if current is None:
        return candidate
    return max(current, candidate)


def _min_upper(current: datetime | None, candidate: datetime) -> datetime:
    """Tighten an upper bound: the earlier of the two (``None`` = unbounded above)."""
    if current is None:
        return candidate
    return min(current, candidate)


def _reject_if_raise(node: FilterNode | FilterClause, ctx: CompileContext) -> None:
    """Raise :class:`RecallFilterUnsupportedError` for unconsumed content under ``"raise"``.

    A no-op under ``"split"`` (the Chronicle engine's mode) — the engine
    post-filters the unconsumed remainder. Under ``"raise"`` it surfaces a
    representative path so a caller that demanded full pushdown sees the gap.
    """
    if ctx.on_unsupported != "raise":
        return
    path = _representative_path(node)
    raise RecallFilterUnsupportedError(
        path,
        "Chronicle pushes down only conjunctive source_timestamp date bounds; this clause must be post-filtered",
    )


def _representative_path(node: FilterNode | FilterClause) -> str:
    """A human-readable path for an unsupported-error message (best-effort)."""
    if isinstance(node, FilterClause):
        return ".".join(node.path)
    return f"${node.op.value.lstrip('$')}"
