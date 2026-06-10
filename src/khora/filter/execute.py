"""Production filter compile/execute seams — ``@internal``.

Two thin helpers that pull the engines' inline filter-compile plumbing into a
single named place so the same construction can be exercised outside an engine:

* :func:`build_compile_context` — the one production
  :class:`~khora.filter.context.CompileContext` builder. Its keyword defaults
  mirror ``CompileContext``'s own field defaults exactly, so a call with only a
  ``backend_target`` (plus the engine's ``on_unsupported``) yields a dataclass
  field-for-field identical to the previous inline construction.
* :func:`plan_chronicle_filter` / :func:`run_chronicle_filter` — the Chronicle
  engine's two-compile composition (a pushdown date-bound + a full-AST in-memory
  post-filter) captured as a reusable plan. Both compiles run under
  ``on_unsupported="split"``, so neither raises.

``@internal``. Reachable as ``khora.filter.execute`` for khora's own engines; not
in any public ``__all__`` and not re-exported from :mod:`khora.__init__`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from khora.filter.ast import FilterClause, FilterNode
from khora.filter.compilers.chronicle import ChronicleDateBound, compile_chronicle
from khora.filter.compilers.python import compile_python
from khora.filter.context import CompileContext, SchemaCapabilities
from khora.filter.model import Op


def build_compile_context(
    backend_target: str,
    *,
    table_alias: str | None = None,
    param_namespace: str = "f",
    field_mapping: Mapping[str, str] | None = None,
    schema_capabilities: SchemaCapabilities = SchemaCapabilities.DEFAULTS,
    on_unsupported: Literal["raise", "split"] = "raise",
) -> CompileContext:
    """Build the per-compile-pass :class:`CompileContext` for a backend target.

    The keyword defaults mirror ``CompileContext``'s own field defaults exactly,
    so ``build_compile_context("khora_chunks", on_unsupported="raise")`` is
    identical to constructing ``CompileContext`` with the same two arguments.
    """
    return CompileContext(
        backend_target=backend_target,
        table_alias=table_alias,
        param_namespace=param_namespace,
        field_mapping=field_mapping,
        schema_capabilities=schema_capabilities,
        on_unsupported=on_unsupported,
    )


@dataclass(frozen=True, slots=True)
class ChronicleFilterPlan:
    """The two compiled halves of a Chronicle filter pass.

    ``@internal``. Chronicle is partial-pushdown by design: only a conjunctive
    ``source_timestamp`` bound folds into the recency window; the full filter is
    always enforced in memory. This pairs the pushdown half with the post-filter
    half so the engine narrows its candidate reads and still re-checks every
    predicate.

    * ``date_bound`` — the narrowing window distilled from the consumed
      ``source_timestamp`` clauses (either side ``None`` for unbounded).
    * ``pushed_keys`` — the AST keys folded into ``date_bound`` (the
      ``source_timestamp`` subset), for the engine's honest pushdown report.
    * ``post_filter`` — an in-memory ``callable(record) -> bool`` over the FULL
      AST, the safety net that enforces every predicate against the field each
      candidate record carries.
    """

    date_bound: ChronicleDateBound
    pushed_keys: frozenset[str]
    post_filter: Callable[[Any], bool]


def plan_chronicle_filter(filter_ast: FilterNode) -> ChronicleFilterPlan:
    """Compose the Chronicle pushdown bound and full-AST post-filter for ``filter_ast``.

    Runs the same two compiles the engine drives inline, both under
    ``on_unsupported="split"`` (so neither raises): ``compile_chronicle`` for the
    ``source_timestamp`` date-bound + its consumed keys, and ``compile_python``
    over the whole AST for the in-memory post-filter.
    """
    ctx = build_compile_context("chunks", on_unsupported="split")
    compiled_bound = compile_chronicle(filter_ast, ctx)
    post_filter = compile_python(filter_ast, ctx).predicate
    return ChronicleFilterPlan(
        date_bound=compiled_bound.predicate,
        pushed_keys=compiled_bound.consumed_keys,
        post_filter=post_filter,
    )


def run_chronicle_filter(
    filter_ast: FilterNode,
    records: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Apply the full-AST post-filter to in-memory candidate ``records``.

    The pushdown ``date_bound`` is a recency-window narrowing that the engine
    applies to its channel reads; over already-materialized candidate records it
    is a no-op, because the full-AST post-filter re-checks ``source_timestamp``
    (and every other predicate) directly. Returns the surviving records in order.

    **Conformance note.** The recall-filter conformance harness drives this seam
    through ``ChronicleExecutor`` to prove the pushdown + post-filter *composition*
    is faithful. Because the post-filter half is :func:`compile_python` — the same
    callable the harness uses as its oracle — a chronicle conformance pass is
    oracle-equivalent **by construction** and is NOT independent backend coverage
    (it cannot diverge from the oracle the way the postgres / surrealdb / cypher /
    weaviate / sqlite_lance compilers can). The independent evidence comes from
    those DB backends; the chronicle leg is the execution-seam check.
    """
    post_filter = plan_chronicle_filter(filter_ast).post_filter
    return [record for record in records if post_filter(record)]


# --------------------------------------------------------------------------- #
# AST leaf inspection — one walk, several thin detectors.
#
# An engine composing a caller filter across adaptive sub-searches needs a few
# cheap structural questions answered about the AST: which keys it constrains,
# whether any leaf falls outside a backend's pushdown set, whether it touches a
# date key, and whether it pins a metadata channel at the top level. The
# key-set detectors (which-keys, residual-pushdown) are thin consumers of
# :func:`iter_leaf_clauses` — the single canonical leaf walk. The date-key and
# channel detectors instead inspect ONLY the root ``AND``'s direct children,
# because a predicate buried under ``$or`` / ``$not`` is not a hard conjunctive
# constraint and must not drive ranking-mode decisions.
# --------------------------------------------------------------------------- #


def iter_leaf_clauses(node: FilterNode | FilterClause) -> Iterator[FilterClause]:
    """Yield every :class:`FilterClause` leaf in an AST subtree.

    The one canonical walk: ``AND`` / ``OR`` / ``NOT`` recurse through
    ``FilterNode.children``; a :class:`FilterClause` is a leaf. The detectors
    below are thin consumers of this generator so the traversal is defined once.
    """
    if isinstance(node, FilterClause):
        yield node
        return
    for child in node.children:
        yield from iter_leaf_clauses(child)


def filter_leaf_keys(node: FilterNode | FilterClause) -> frozenset[str]:
    """Return the set of dotted keys every leaf in the AST constrains.

    ``".".join(clause.path)`` for each leaf — identical to how the compilers
    build :attr:`CompiledFilter.consumed_keys` (their ``_path_str`` joins the
    same segment tuple), so this set can be differenced against ``consumed_keys``
    to find residual (un-pushed) leaves.
    """
    return frozenset(".".join(clause.path) for clause in iter_leaf_clauses(node))


def has_residual_metadata(node: FilterNode | FilterClause, consumed_keys: frozenset[str]) -> bool:
    """Return whether any AST leaf was not pushed down (needs a post-filter).

    ``True`` when at least one leaf key is absent from ``consumed_keys`` — i.e.
    the backend compiler (run in ``"split"`` mode) could not express it, so the
    engine must apply an in-memory post-filter and may want to over-fetch first.
    """
    return bool(filter_leaf_keys(node) - consumed_keys)


def filter_constrains_date_key(node: FilterNode) -> bool:
    """Return whether a top-level conjunctive leaf constrains a date system key.

    Inspects ONLY the root ``AND``'s direct children — mirroring
    :func:`caller_channel_constraint`. A date predicate buried inside an
    ``$or`` / ``$not`` is not a hard conjunctive constraint, so it does not flag
    (it is still enforced on every channel via the threaded filter; this gate
    only governs EXPLICIT recency synthesis). Keys on the leaf *path*, not its
    operator, so it covers every form an ``occurred_at`` / ``created_at``
    predicate can take — a bare ``$eq``, a range (``$gte`` / ``$lte``), ``$in``,
    or ``$exists``. A bare single predicate parses to ``AND`` with one child, so
    the common ``{"occurred_at": {"$gte": ...}}`` case still flags.
    """
    if node.op != Op.AND:
        return False
    return any(
        isinstance(child, FilterClause) and child.path in (("occurred_at",), ("created_at",)) for child in node.children
    )


def caller_channel_constraint(node: FilterNode) -> frozenset[str] | None:
    """Return the channels a top-level ``metadata.channel`` predicate pins, or ``None``.

    Inspects ONLY the root ``AND``'s direct children. A ``metadata.channel``
    constraint buried inside an ``$or`` / ``$not`` is not a hard conjunctive
    constraint, so it yields ``None`` (never over-restrict). For a direct child
    that is a ``metadata.channel`` leaf: ``$eq`` of a string contributes that
    value; ``$in`` contributes its string members. Any other shape (or no such
    leaf) yields ``None``. Conservative on purpose — equality / membership only.
    """
    if node.op != Op.AND:
        return None
    channels: set[str] = set()
    for child in node.children:
        if not isinstance(child, FilterClause) or child.path != ("metadata", "channel"):
            continue
        if child.op == Op.EQ and isinstance(child.operand, str):
            channels.add(child.operand)
        elif child.op == Op.IN:
            channels.update(item for item in child.operand if isinstance(item, str))
        else:
            return None
    return frozenset(channels) if channels else None
