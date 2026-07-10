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

Document-key hydration (GitHub #1494): fetched provenance chunks structurally
lack the seven denormalized document keys (``source``, ``source_name``,
``source_type``, ``source_url``, ``external_id``, ``content_type``, ``title``) —
those live on the parent document. A filter whose leaves reference any of them
would otherwise over-drop every item, because the predicate reads the key as
absent on the bare chunk. When (and only when) the filter AST references a
document key, the helper hydrates the per-document ``DocumentProjection`` via
``get_document_projections_batch`` alongside the chunk pages and folds the doc
keys onto the record the predicate evaluates. Filters with no doc-key leaf skip
the projection fetch entirely (zero extra query, byte-identical behavior).

Fail-closed (ADR-001): if a provenance-chunk fetch page (or, on the doc-key
path, a ``get_document_projections_batch`` page) raises, every item not yet
proven to have a passing chunk is DROPPED (never returned unverified — that
would re-introduce the #1457 leak), exactly ONE
:class:`~khora.core.diagnostics.Degradation` is appended (WARNING + ``exc_info``,
``reason="provenance_fetch_failed"`` or ``"document_fetch_failed"``), and the
verified survivors are returned. The call never re-raises. Because every
returned item is verified, the surface is legitimately enforced even on the
degraded path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from loguru import logger

from khora.core.diagnostics import Degradation
from khora.core.models.recall import DocumentProjection
from khora.filter.ast import FilterNode
from khora.filter.compilers.python import compile_python
from khora.filter.execute import build_compile_context, filter_leaf_keys
from khora.telemetry.metrics import metric_counter

# Page size for the provenance-chunk fetch AND the doc-key hydration fetch. Bounds
# the backend IN-list per round trip (Chronicle can pass many items); chunk cost is
# bounded by the number of unique provenance chunk ids across the items, and
# doc-key cost by the number of unique parent document ids, each fetched in pages
# of this size.
_PAGE_SIZE = 500

# The seven denormalized document keys a chunk does NOT carry structurally — they
# live on the parent document, hydrated via ``get_document_projections_batch`` and
# exposed as attributes of the same names on ``DocumentProjection``. A filter that
# references any of these needs doc hydration or it over-drops every item (#1494).
# ``source_timestamp`` is intentionally EXCLUDED: it stays chunk-carried.
_DOC_KEYS: frozenset[str] = frozenset(
    {"source_type", "source_name", "source_url", "external_id", "content_type", "source", "title"}
)

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
        "provenance-chunk fetch page raises (reason=provenance_fetch_failed) or a "
        "doc-key hydration page raises (reason=document_fetch_failed, #1494); items "
        "with no filter-passing chunk in the pages fetched so far are dropped "
        "(fail-closed) and the verified survivors returned. Labels: component (the "
        "caller's per-surface value, e.g. vectorcypher.entity_filter), reason "
        "(provenance_fetch_failed | document_fetch_failed). "
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
    """The storage seam the filter needs: a batched chunk fetch by id plus a
    batched per-document projection fetch for doc-key hydration (#1494)."""

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Any]: ...

    async def get_document_projections_batch(
        self, document_ids: list[UUID], *, namespace_id: UUID
    ) -> dict[UUID, DocumentProjection]: ...


async def filter_items_by_provenance[T: _HasProvenance](
    items: Sequence[T],
    filter_ast: FilterNode,
    *,
    namespace_id: UUID,
    storage: _ChunkStore,
    component: str,
    degradations: list[Degradation],
    chunk_record_adapter: Callable[[Any, Any | None], Any] | None = None,
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

    ``chunk_record_adapter`` maps each ``(chunk, doc)`` pair to the record the
    predicate is evaluated against, so the entity/relationship surface enforces
    the filter with the SAME field semantics its chunk channel uses. Chronicle
    passes its ``_chunk_to_record`` (which resolves ``occurred_at`` as
    ``COALESCE(occurred_at, source_timestamp)`` and folds ``doc``'s denormalized
    document keys, matching its chunk post-filter); without it a raw chunk
    carrying its event time only in ``source_timestamp`` would false-drop the
    entity even though the chunk channel kept the chunk. ``doc`` is the hydrated
    :class:`DocumentProjection` for the chunk's parent document when the filter
    needs doc keys, else ``None``. When ``chunk_record_adapter`` is ``None``
    (VectorCypher, whose chunks carry ``occurred_at`` natively) the helper builds
    the record itself: the raw chunk when no doc key is referenced (byte-identical
    to pre-#1494 behavior), else a dict carrying the raw ``occurred_at`` /
    ``created_at`` / ``source_timestamp`` / ``metadata`` PLUS the hydrated doc
    keys. ``occurred_at`` stays the RAW chunk value on the no-adapter path — it is
    NOT coalesced, so the ∃ pass agrees exactly with VectorCypher's graph chunk
    channel (which evaluates the same predicate over the raw chunk). The no-adapter
    path therefore assumes the ``Chunk`` carries ``occurred_at`` natively; an engine
    whose chunk channel coalesces ``occurred_at`` MUST pass a ``chunk_record_adapter``
    so the ∃ record matches its chunk semantics.

    Doc-key hydration (#1494): the fetched provenance chunks do not carry the
    seven :data:`_DOC_KEYS` document keys. When the filter AST references any of
    them, the helper batch-fetches the parent-document
    :class:`DocumentProjection`\\ s via ``get_document_projections_batch`` (in
    :data:`_PAGE_SIZE` pages, unique document ids only) and threads the projection
    into the record. When no doc-key leaf is present the projection fetcher is
    never called and the raw chunk flows through unchanged.

    Fail-closed (ADR-001): on a chunk OR document fetch-page exception the
    accumulated set is taken as final — items with a hit in the pages fetched SO
    FAR survive, the rest are dropped (their provenance could not be verified).
    Exactly one :class:`Degradation` (``reason="provenance_fetch_failed"`` for a
    chunk-page failure or ``"document_fetch_failed"`` for a projection-page
    failure, ``component`` as passed in) is appended, logged at WARNING with
    ``exc_info=True``. Never re-raises. Note the deliberate asymmetry with
    Chronicle's chunk channel, which degrades *open* on a hydration failure
    (evaluates the post-filter with the doc keys absent): this ∃ pass degrades
    *closed* (drops unverified items), because an existence claim over provenance
    must be verified, not guessed. A transient projection-fetch failure can thus
    return chunks while the entity/relationship surface narrows — surfaced via the
    ``document_fetch_failed`` degradation.
    """
    if not items:
        return []

    predicate = compile_python(filter_ast, build_compile_context("Chunk", on_unsupported="split")).predicate

    # Whether the filter references any denormalized document key. Only then do we
    # pay the projection hydration fetch (#1494); otherwise the raw chunk flows
    # through unchanged and the fetcher is never called.
    needs_docs = bool(filter_leaf_keys(filter_ast) & _DOC_KEYS)

    # Unique, order-stable provenance chunk ids across every item. Items with no
    # provenance contribute nothing and are dropped by the ∃ test at the end.
    ordered_ids = list(dict.fromkeys(cid for item in items for cid in item.source_chunk_ids))

    # Hydrated per-document projections, accumulated across pages (only when
    # ``needs_docs``). Keyed by document id; missing ids are simply absent.
    projections: dict[UUID, DocumentProjection] = {}

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

        # Doc-key hydration: fetch the parent-document projections for this page's
        # chunks (unique, not-yet-hydrated document ids) so the doc keys resolve on
        # the record. Same fail-closed contract as the chunk fetch — on failure
        # stop, drop unverified items, record ONE degradation.
        if needs_docs:
            doc_ids = list(
                dict.fromkeys(
                    ch.document_id
                    for ch in chunks_map.values()
                    if ch.document_id is not None and ch.document_id not in projections
                )
            )
            try:
                for doc_start in range(0, len(doc_ids), _PAGE_SIZE):
                    doc_page = doc_ids[doc_start : doc_start + _PAGE_SIZE]
                    projections.update(
                        await storage.get_document_projections_batch(doc_page, namespace_id=namespace_id)
                    )
            except Exception as exc:
                logger.warning(
                    "Provenance document hydration failed ({component}); items unverified after this page drop",
                    component=component,
                    exc_info=True,
                )
                degradations.append(
                    Degradation(
                        component=component,
                        reason="document_fetch_failed",
                        detail=str(exc)[:200] or None,
                        exception=type(exc).__name__,
                    )
                )
                _PROVENANCE_FILTER_DEGRADED_COUNTER.add(
                    1, attributes={"component": component, "reason": "document_fetch_failed"}
                )
                break

        surviving.update(
            cid
            for cid, ch in chunks_map.items()
            if predicate(_record_for(ch, needs_docs, projections, chunk_record_adapter))
        )

    return [item for item in items if any(cid in surviving for cid in item.source_chunk_ids)]


def _record_for(
    ch: Any,
    needs_docs: bool,
    projections: dict[UUID, DocumentProjection],
    chunk_record_adapter: Callable[[Any, Any | None], Any] | None,
) -> Any:
    """Build the record the ``"Chunk"`` predicate is evaluated against.

    Adapter path (Chronicle): always call the adapter so it keeps its field
    semantics; ``doc`` is the hydrated projection when ``needs_docs`` else ``None``.

    No-adapter path (VectorCypher): return the RAW chunk when no doc key is
    referenced (byte-identical to pre-#1494 behavior); else a dict carrying the
    raw ``occurred_at`` (NOT coalesced — must agree with the graph chunk channel),
    ``created_at``, ``source_timestamp``, and ``metadata``, plus each hydrated doc
    key whose projection value is not ``None`` (a missing key stays ABSENT so
    compile_python's missing-key semantics apply).
    """
    doc = projections.get(ch.document_id) if needs_docs else None
    if chunk_record_adapter is not None:
        return chunk_record_adapter(ch, doc)
    if not needs_docs:
        return ch
    record: dict[str, Any] = {
        "occurred_at": ch.occurred_at,
        "created_at": ch.created_at,
        "source_timestamp": ch.source_timestamp,
        "metadata": ch.metadata or {},
    }
    if doc is not None:
        for key in _DOC_KEYS:
            value = getattr(doc, key, None)
            if value is not None:
                record[key] = value
    return record
