"""Engine-agnostic ∃-over-provenance item filter — ``@internal``.

GitHub #1457. A recall filter narrows the chunk surface, but a graph-derived
result surface (entities, relationships) is assembled from traversal the chunk
filter never touched. :func:`filter_items_by_provenance` re-applies the SAME
``"Chunk"`` predicate the chunk channels use to each item's *provenance* chunks
and keeps an item iff at least one of its provenance chunks passes (the
existential / ∃ rule).

The function is deliberately engine-agnostic — it takes anything exposing a
``.source_chunk_ids`` sequence, compiles the predicate itself from the filter
AST, and fetches provenance itself in bounded pages. The VectorCypher retriever
is the first caller; the Chronicle follow-up (#1458) reuses it verbatim.

Fail-closed (ADR-001): if a provenance-chunk fetch page raises, every item not
yet proven to have a passing chunk is DROPPED (never returned unverified — that
would re-introduce the #1457 leak), exactly ONE
:class:`~khora.core.diagnostics.Degradation` is appended (WARNING + ``exc_info``),
and the verified survivors are returned. The call never re-raises. Because every
returned item is verified, the surface is legitimately enforced even on the
degraded path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from loguru import logger

from khora.core.diagnostics import Degradation
from khora.filter.ast import FilterNode
from khora.filter.compilers.python import compile_python
from khora.filter.execute import build_compile_context
from khora.telemetry.metrics import metric_counter

# Page size for the provenance-chunk fetch. Bounds the backend IN-list per round
# trip (Chronicle can pass many items) and, with the early-exit below, caps the
# cost at one page per 500 undecided provenance chunks.
_PAGE_SIZE = 500

# Provenance-filter degradation counter (GitHub #1457, ADR-001). The metric name
# is engine-agnostic to match the module: this counter is emitted from within the
# reusable helper, so it must NOT hard-code an engine (Chronicle #1458 reuses the
# same module). The ``component`` label carries the caller's per-surface identity
# (``vectorcypher.entity_filter`` / ``vectorcypher.relationship_filter`` / future
# ``chronicle.*``), a low-cardinality enum. NO namespace_id label.
_PROVENANCE_FILTER_DEGRADED_COUNTER = metric_counter(
    "khora.filter.provenance.degraded_total",
    unit="1",
    description=(
        "GitHub #1457. Engine-agnostic ∃-over-provenance item-filter fallbacks "
        "(khora.filter.provenance.filter_items_by_provenance). Incremented when a "
        "provenance-chunk fetch page raises; items with no filter-passing chunk in "
        "the pages fetched so far are dropped (fail-closed) and the verified "
        "survivors returned. Labels: component (the caller's per-surface value, "
        "e.g. vectorcypher.entity_filter), reason (provenance_fetch_failed). "
        "NO namespace_id label - cardinality rule."
    ),
)


@runtime_checkable
class _HasProvenance(Protocol):
    """Any item carrying a provenance chunk-id sequence."""

    @property
    def source_chunk_ids(self) -> Sequence[UUID]: ...


@runtime_checkable
class _ChunkStore(Protocol):
    """The storage seam the filter needs: a batched chunk fetch by id."""

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Any]: ...


async def filter_items_by_provenance[T: _HasProvenance](
    items: Sequence[T],
    filter_ast: FilterNode,
    *,
    namespace_id: UUID,
    storage: _ChunkStore,
    component: str,
    degradations: list[Degradation],
) -> list[T]:
    """Keep the items whose provenance satisfies ``filter_ast`` (∃ rule).

    An item survives iff at least one of its ``source_chunk_ids`` chunks passes
    the compiled ``"Chunk"`` predicate. Items with no provenance can never
    satisfy the existential and are dropped. Input order is preserved.

    The provenance chunks are fetched in pages of :data:`_PAGE_SIZE` (the storage
    seam issues a single ``IN (...)`` per call with no internal paging, so an
    unbounded union would blow the backend variable limit once a caller feeds it
    many items). Each page's filter-passing chunk ids accumulate into a shared
    ``surviving`` set; an item is kept iff any of its ids landed in that set.

    Fail-closed (ADR-001): on a fetch-page exception the accumulated set is taken
    as final — items with a hit in the pages fetched SO FAR survive, the rest are
    dropped (their provenance could not be verified). Exactly one
    :class:`Degradation` (``reason="provenance_fetch_failed"``, ``component`` as
    passed in) is appended, logged at WARNING with ``exc_info=True``. Never
    re-raises.
    """
    if not items:
        return []

    predicate = compile_python(filter_ast, build_compile_context("Chunk", on_unsupported="split")).predicate

    # Unique, order-stable provenance chunk ids across every item. Items with no
    # provenance contribute nothing and are dropped by the ∃ test at the end.
    ordered_ids = list(dict.fromkeys(cid for item in items for cid in item.source_chunk_ids))

    # Ids whose chunk passes the predicate, accumulated across pages. A chunk id
    # that did not resolve is silently absent → treated as predicate False.
    surviving: set[UUID] = set()
    for start in range(0, len(ordered_ids), _PAGE_SIZE):
        page = ordered_ids[start : start + _PAGE_SIZE]
        try:
            chunks_map = await storage.get_chunks_batch(page, namespace_id=namespace_id)
        except Exception as exc:
            # ADR-001 fail-closed: stop fetching and take the set-so-far as final;
            # items with no hit yet drop (unverified), record ONE degradation.
            logger.warning(
                "Provenance chunk fetch failed ({component}); items unverified after this page drop",
                component=component,
                exc_info=True,
            )
            degradations.append(
                Degradation(
                    component=component,
                    reason="provenance_fetch_failed",
                    detail=str(exc)[:200] or None,
                    exception=type(exc).__name__,
                )
            )
            _PROVENANCE_FILTER_DEGRADED_COUNTER.add(
                1, attributes={"component": component, "reason": "provenance_fetch_failed"}
            )
            break

        surviving.update(cid for cid, ch in chunks_map.items() if predicate(ch))

    return [item for item in items if any(cid in surviving for cid in item.source_chunk_ids)]
