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

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from khora.filter.ast import FilterNode
from khora.filter.compilers.chronicle import ChronicleDateBound, compile_chronicle
from khora.filter.compilers.python import compile_python
from khora.filter.context import CompileContext, SchemaCapabilities


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
    """
    post_filter = plan_chronicle_filter(filter_ast).post_filter
    return [record for record in records if post_filter(record)]
