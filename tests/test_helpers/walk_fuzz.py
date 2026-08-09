"""Corpus, oracle, strategies and walk driver for the document-enumeration walk fuzzer.

The keyset enumeration surface (``StorageCoordinator.scan_documents_page``, the
engine ``list_documents`` passthrough, and the ``Khora.list_documents`` facade
above it) is pinned by hand-authored tests per store: the bounded scan primitive,
the cursor serialization, the pushdown split. Those are precise but finite — they
check the shapes someone thought to enumerate, one page at a time.

This module is the shared half of the complementary force: a *walk* fuzzer. It
generates a filter + ``status`` + ``updated_before`` triple, drives a whole
multi-page walk to exhaustion, and checks the concatenation against a pure-Python
oracle. What that catches is the class of defect a single-page test structurally
cannot see — a cursor that skips a tie-mate on the page boundary, a page that
reports ``exhausted`` while rows remain, a walk whose answer depends on how the
scan budget happened to slice it.

Five properties ride on this module (the test legs assert them; the helpers here
make them expressible):

1. **Exactly-once / completeness** — the concatenation of every page is the
   oracle's survivor set, no repeats, nothing missing.
2. **Cursor stitching** — the same walk at ``scan_bound=1`` (a cursor boundary
   between every raw row) and unbounded (one page, no cursor at all) agree.
3. **Order** — the concatenation is strictly descending on ``(created_at, id)``.
4. **Limit invariance** — the answer does not depend on ``limit``.
5. **Honest termination** — ``next_after is None`` iff ``exhausted``, and only
   the terminal page carries either.

Two shapes are deliberately NOT in the corpus, and both are worth stating so a
future reader does not "fix" them:

* **No ``occurred_at``.** The facade rejects it as non-enumerable (a document row
  has no event-time column), so :func:`walk_filter` draws its date channel from
  ``created_at`` / ``source_timestamp`` only.
* **No NULL ``updated_at``.** ``updated_before`` compiles to a half-open
  ``updated_at < :bound``, which is NULL — and therefore excluding — for a row
  whose ``updated_at`` is NULL. That exclusion is real and documented on
  ``build_documents_scan_query``, but it is *unreachable through the production
  write API*: ``DocumentModel.updated_at`` carries a Python-side ``default``, so
  an explicit ``None`` at INSERT is replaced by ``now()``, and both
  ``update_document`` and the partial-update path stamp the column themselves
  (``updated_at`` is not in the partial-update allowlist). Seeding a NULL would
  require raw SQL against a state production cannot produce, and this corpus
  seeds only through ``create_document``. The oracle below still implements the
  NULL branch, so it stays correct if a write path ever gains one.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from hypothesis import strategies as st

from khora.core.models import Document
from khora.core.models.document import DocumentCursor, DocumentPage, DocumentStatus
from khora.filter import RecallFilter, RecallFilterValidationError
from khora.filter.ast import FilterNode, parse_to_ast
from khora.filter.compilers.python import compile_python
from khora.filter.execute import build_compile_context
from tests.test_helpers.document_order import id_ladder, seed_order
from tests.test_helpers.document_scan import WHOLE_SECOND, as_utc

# --------------------------------------------------------------------------- #
# Value pools.
# --------------------------------------------------------------------------- #
#
# The seven string keys are exactly the ones the embedded store declares
# pushable (``_PUSHABLE_SYSTEM_KEYS``), so a drawn predicate exercises the SQL
# half; the two date keys are the ones it deliberately withholds, so a drawn date
# predicate exercises the in-memory post-filter half, so the walk properties hold
# over both physical plans. That is NOT a cross-check of the two halves against
# each other: the post-filter re-checks the whole AST with the same
# ``compile_python`` predicate the oracle uses, so it silently repairs a
# too-permissive pushdown. See the leg modules' docstrings.

STRING_POOLS: dict[str, list[str]] = {
    "source_type": ["library", "connection", "direct"],
    "source_name": ["linear", "slack"],
    "content_type": ["text/markdown", "application/pdf"],
    # Unique per row in the corpus — ``(namespace_id, external_id)`` is a UNIQUE
    # index — so these operands select single rows plus one guaranteed miss.
    "external_id": ["ext-01", "ext-05", "ext-13", "ext-missing"],
    "source": ["api", "ingest"],
    "title": ["alpha", "beta", "gamma", "delta", "epsilon"],
    "source_url": ["https://example.test/a", "https://example.test/o"],
}

METADATA_PATHS: tuple[str, ...] = (
    "metadata.tier",
    "metadata.score",
    "metadata.tags",
    "metadata.a.b",
    "metadata.mk",
)

META_SCALAR_POOL: list[Any] = ["gold", "silver", "urgent", "release", "okrs", 0, 3, 7, 10, "v", "w"]

# Six blobs cycled across the corpus: scalars, an array, the EMPTY array, a
# nested sub-document, a present JSON null, and a bare ``{}``.
_META_BLOBS: tuple[dict[str, Any], ...] = (
    {"tier": "gold", "score": 10, "tags": ["urgent", "release"]},
    {"tier": "silver", "score": 0, "tags": ["okrs"]},
    {"tier": "gold", "score": 3, "tags": []},
    {"score": 7, "a": {"b": "v"}},
    {"tier": "silver", "score": 7, "mk": None, "a": {"b": "w"}},
    {},
)

# ``created_at`` blocks, newest first. Two blocks of one row are single rows; the
# 3- and 4-row blocks are the tie blocks the ``id DESC`` leg has to resolve, and
# the walk puts a cursor boundary INSIDE both of them at ``scan_bound=1``.
_TIE_PLAN: tuple[int, ...] = (3, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 2)

CORPUS_SIZE: int = sum(_TIE_PLAN)

# ``updated_at`` spreads over distinct whole seconds on distinct days, so an
# ``updated_before`` bound cuts the corpus at a known, non-trivial fraction.
_UPDATED_BASE = datetime(2026, 2, 1, 9, 0, 0, tzinfo=UTC)

# ``source_timestamp`` operands (and the NULL some rows carry) — the second date
# channel, which unlike ``created_at`` is nullable.
_TS_POOL: tuple[datetime, ...] = (
    datetime(2026, 6, 1, tzinfo=UTC),
    datetime(2026, 3, 15, tzinfo=UTC),
    datetime(2026, 1, 1, tzinfo=UTC),
)
_TS_MISS = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class WalkCorpus:
    """A seeded document corpus plus everything an assertion needs to be non-vacuous."""

    namespace_id: UUID
    """The namespace every :attr:`documents` row is written to."""

    documents: tuple[Document, ...]
    """The rows, in the deliberately non-monotonic order they must be WRITTEN in."""

    expected: tuple[UUID, ...]
    """The one correct unfiltered enumeration, ``(created_at DESC, id DESC)``."""

    tie_blocks: tuple[tuple[UUID, ...], ...]
    """Ids sharing one ``created_at`` (blocks of >= 2 only), already in expected order."""

    def by_id(self) -> dict[UUID, Document]:
        """The corpus keyed by document id."""
        return {doc.id: doc for doc in self.documents}


def build_walk_corpus(namespace_id: UUID) -> WalkCorpus:
    """Build the ~20-document corpus the read-only walk properties share.

    Three things are load-bearing and none of them are incidental.

    **The ladder.** Ids come from
    :func:`~tests.test_helpers.document_order.id_ladder`, so they ascend by
    construction and the expected enumeration is known up front rather than
    sampled. With random ids a scan that dropped the ``id DESC`` tie-break could
    still match by luck inside a two-row block.

    **The tie blocks.** Four blocks share a ``created_at`` to the microsecond (one
    of three rows, one of four, two of two), so at ``scan_bound=1`` the walk puts
    a cursor boundary strictly INSIDE a tie block — the mid-tie resume that only
    lands correctly when the cursor's ``id`` half is compared as a UUID and its
    ``created_at`` half is bound in the store's own serialization. Every stamp is
    a whole second derived from
    :data:`~tests.test_helpers.document_scan.WHOLE_SECOND`; see that module for
    why the microsecond field is pinned rather than sampled from the clock.

    **The attribute spread.** Every filterable surface carries distinct, repeated
    AND absent values, so a generated predicate lands a STRICT subset rather than
    all-or-nothing. Attributes are assigned by *ladder* index with deliberately
    co-prime strides, which is neither the write order nor the enumeration order —
    so no expectation anywhere can accidentally be derived from a loop counter.

    ``created_at`` and ``id`` are in conflict across blocks: the newest block
    holds the LOWEST ids. An implementation that sorted ``id`` first would park
    those rows at exactly the wrong end and cannot reproduce :attr:`expected`.
    """
    ladder = id_ladder(CORPUS_SIZE)

    # Assign each ladder index its block's created_at; blocks descend in time as
    # the ladder ascends, so the two sort keys disagree.
    stamps: dict[UUID, datetime] = {}
    blocks: list[tuple[UUID, ...]] = []
    cursor = 0
    for block_index, size in enumerate(_TIE_PLAN):
        block = tuple(ladder[cursor : cursor + size])
        instant = WHOLE_SECOND + timedelta(seconds=2 * (len(_TIE_PLAN) - block_index))
        for doc_id in block:
            stamps[doc_id] = instant
        blocks.append(block)
        cursor += size

    expected: list[UUID] = []
    for block in blocks:
        expected.extend(reversed(block))

    documents = tuple(
        _build_document(namespace_id, doc_id, ladder.index(doc_id), stamps[doc_id]) for doc_id in seed_order(ladder)
    )

    return WalkCorpus(
        namespace_id=namespace_id,
        documents=documents,
        expected=tuple(expected),
        tie_blocks=tuple(tuple(reversed(b)) for b in blocks if len(b) > 1),
    )


def _build_document(namespace_id: UUID, doc_id: UUID, index: int, created_at: datetime) -> Document:
    """One corpus row: attributes derived from the LADDER index with co-prime strides.

    The strides (3 / 5 / 7 / 6 / 8 / 2) are chosen so no two string keys are
    perfectly correlated — a conjunction over two of them narrows further than
    either alone, which is what makes a composed filter discriminate.
    """
    return Document(
        id=doc_id,
        namespace_id=namespace_id,
        content=f"walk corpus row {index}",
        checksum=f"walk-{doc_id.hex}",
        created_at=created_at,
        updated_at=_UPDATED_BASE + timedelta(days=index, seconds=index),
        status=_STATUS_CYCLE[index % len(_STATUS_CYCLE)],
        source_type=STRING_POOLS["source_type"][index % 3],
        source_name=None if index % 5 == 0 else STRING_POOLS["source_name"][index % 2],
        content_type=None if index % 7 == 0 else STRING_POOLS["content_type"][(index // 2) % 2],
        external_id=None if index % 6 == 0 else f"ext-{index:02d}",
        source=None if index % 8 == 0 else STRING_POOLS["source"][(index // 3) % 2],
        title=STRING_POOLS["title"][index % 5],
        source_url=None if index % 3 == 0 else STRING_POOLS["source_url"][(index // 3) % 2],
        source_timestamp=None if index % 4 == 3 else _TS_POOL[index % 3],
        metadata=copy.deepcopy(_META_BLOBS[index % len(_META_BLOBS)]),
    )


# Four of the five statuses, cycled — enough that a ``status`` narrowing keeps a
# strict subset on every value in the pool.
_STATUS_CYCLE: tuple[DocumentStatus, ...] = (
    DocumentStatus.PENDING,
    DocumentStatus.COMPLETED,
    DocumentStatus.FAILED,
    DocumentStatus.ARCHIVED,
)

STATUS_POOL: tuple[str | None, ...] = (None, *(s.value for s in _STATUS_CYCLE))
"""``status`` operands the strategies draw from — ``None`` (unfiltered) plus every seeded value."""

UPDATED_BEFORE_POOL: tuple[datetime | None, ...] = (
    None,
    # Exactly one row's ``updated_at``. Half-open means that row is EXCLUDED and
    # only the one below it survives — the boundary case an inclusive ``<=`` would
    # get wrong, and a near-empty walk without being a vacuously empty one.
    _UPDATED_BASE + timedelta(days=1, seconds=1),
    _UPDATED_BASE + timedelta(days=7),
    _UPDATED_BASE + timedelta(days=14),
    _UPDATED_BASE + timedelta(days=CORPUS_SIZE + 1),  # after every row -> full corpus
)
"""``updated_before`` operands: unfiltered, the exact half-open boundary, two cuts, and saturation.

Deliberately no operand strictly below every row: the empty-result walk (a whole
scanned window rejected, page after page) is already the single most common shape
the generated *filters* produce, so spending a fifth of every draw's bound
reproducing it would only dilute the strict-subset draws the properties actually
discriminate on.
"""


# --------------------------------------------------------------------------- #
# The oracle.
# --------------------------------------------------------------------------- #


def walk_oracle(
    documents: Sequence[Document],
    *,
    filter_ast: FilterNode | None = None,
    status: str | None = None,
    updated_before: datetime | None = None,
) -> list[UUID]:
    """The ids a correct walk must enumerate, in the one correct order.

    Pure Python over the in-memory corpus — no store, no cursor, no paging. The
    three narrowings are applied in the same semantics the read path uses:

    * ``status`` is compared against the enum's stored VALUE string
      (``DocumentModel.status`` is an ``Enum`` with ``values_callable``, so the
      column holds ``'completed'``, not ``'DocumentStatus.COMPLETED'``).
    * ``updated_before`` is half-open (``updated_at < bound``) and a NULL
      ``updated_at`` is EXCLUDED — the SQL ``NULL < :bound`` outcome, preserved
      here even though the corpus cannot seed one (see the module docstring).
    * ``filter_ast`` is compiled through the SAME call shape the coordinator uses
      (``compile_python`` over ``build_compile_context("documents",
      on_unsupported="split")``), so the oracle is the coordinator's own
      post-filter applied to a corpus the coordinator never touched.

    Survivors are then sorted on ``(created_at, id)`` DESCENDING — the total
    enumeration order — and returned as ids. ``as_utc`` normalizes the naive
    stamps the embedded store reads back, so a corpus and a read-back row sort
    identically.
    """
    predicate: Callable[[Any], bool] | None = None
    if filter_ast is not None:
        predicate = compile_python(filter_ast, build_compile_context("documents", on_unsupported="split")).predicate

    survivors: list[Document] = []
    for doc in documents:
        if status is not None and doc.status.value != status:
            continue
        if updated_before is not None and (doc.updated_at is None or as_utc(doc.updated_at) >= as_utc(updated_before)):
            continue
        if predicate is not None and not predicate(doc):
            continue
        survivors.append(doc)

    survivors.sort(key=lambda d: (as_utc(d.created_at), d.id), reverse=True)
    return [d.id for d in survivors]


# --------------------------------------------------------------------------- #
# Filter strategy.
# --------------------------------------------------------------------------- #
#
# Per-key operator rules mirror the validator's typed submodels so nearly every
# draw validates; the rare rejection is discarded by ``validated_walk_ast``.
# Operands come from the corpus's own value pools so a predicate separates a
# known subset rather than matching everything or nothing.

_DATE_KEYS: tuple[str, ...] = ("created_at", "source_timestamp")

# ``source_timestamp`` operands come from its own pool (plus a guaranteed miss);
# the four ``WHOLE_SECOND`` offsets land INSIDE the ``created_at`` block ladder,
# which spans ``+2s`` to ``+26s``. Without them a ``created_at`` predicate would
# be all-or-nothing — every row shares one minute — and the date channel would
# stop discriminating.
_DATE_OPERANDS: tuple[datetime, ...] = (
    *_TS_POOL,
    _TS_MISS,
    WHOLE_SECOND + timedelta(seconds=6),
    WHOLE_SECOND + timedelta(seconds=14),
    WHOLE_SECOND + timedelta(seconds=20),
    WHOLE_SECOND + timedelta(seconds=25),  # strictly BETWEEN two blocks
)


def _date_iso(value: datetime) -> str:
    """The system-key ``DateOps`` operand form (a plain ISO-8601 string, not ``$date``)."""
    return value.isoformat().replace("+00:00", "Z")


@st.composite
def _date_predicate(draw: st.DrawFn) -> Any:
    """A date-key predicate: a bare ISO scalar or a ``DateOps`` operator-expression."""
    if draw(st.booleans()):
        return _date_iso(draw(st.sampled_from(_DATE_OPERANDS)))
    op = draw(st.sampled_from(["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]))
    if op in ("$in", "$nin"):
        values = draw(st.lists(st.sampled_from(_DATE_OPERANDS), min_size=0, max_size=3))
        return {op: [_date_iso(v) for v in values]}
    return {op: _date_iso(draw(st.sampled_from(_DATE_OPERANDS)))}


@st.composite
def _string_predicate(draw: st.DrawFn, key: str) -> Any:
    """A string-key predicate: a bare scalar, an exact-array, or a ``StringOps`` expression."""
    pool = STRING_POOLS[key]
    kind = draw(st.sampled_from(["bare", "eq", "ne", "in", "nin", "exists", "bare_list"]))
    if kind == "bare":
        return draw(st.sampled_from(pool))
    if kind == "bare_list":  # bare list => $eq exact-array (NOT $in)
        return draw(st.lists(st.sampled_from(pool), min_size=1, max_size=2))
    if kind == "exists":
        return {"$exists": draw(st.booleans())}
    if kind in ("eq", "ne"):
        return {f"${kind}": draw(st.sampled_from(pool))}
    return {f"${kind}": draw(st.lists(st.sampled_from(pool), min_size=0, max_size=3))}


@st.composite
def _metadata_predicate(draw: st.DrawFn) -> tuple[str, Any]:
    """A ``metadata.<path>`` key + predicate over the corpus's metadata surface."""
    key = draw(st.sampled_from(METADATA_PATHS))
    kind = draw(
        st.sampled_from(["bare", "bare_list", "eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "exists", "date"])
    )
    if kind == "bare":
        return key, draw(st.sampled_from(META_SCALAR_POOL))
    if kind == "bare_list":  # bare list => $eq exact-array
        return key, draw(st.lists(st.sampled_from(["urgent", "release", "okrs"]), min_size=0, max_size=3))
    if kind == "exists":
        return key, {"$exists": draw(st.booleans())}
    if kind == "date":
        return key, {"$eq": {"$date": _date_iso(draw(st.sampled_from(_DATE_OPERANDS)))}}
    if kind in ("in", "nin"):
        return key, {f"${kind}": draw(st.lists(st.sampled_from(META_SCALAR_POOL), min_size=0, max_size=3))}
    return key, {f"${kind}": draw(st.sampled_from(META_SCALAR_POOL))}


@st.composite
def _single_predicate(draw: st.DrawFn) -> dict[str, Any]:
    """One single-field predicate (a date key, a string key, or a metadata path).

    ``occurred_at`` is absent by construction: the enumeration facade rejects it
    with ``key_not_enumerable``, so drawing it would generate filters no caller
    can submit rather than filters that stress the walk.
    """
    channel = draw(st.sampled_from(["date", "string", "metadata"]))
    if channel == "date":
        return {draw(st.sampled_from(_DATE_KEYS)): draw(_date_predicate())}
    if channel == "string":
        key = draw(st.sampled_from(list(STRING_POOLS)))
        return {key: draw(_string_predicate(key))}
    meta_key, predicate = draw(_metadata_predicate())
    return {meta_key: predicate}


def walk_filter(max_depth: int = 3) -> st.SearchStrategy[dict[str, Any]]:
    """A recursively-composed enumerable filter dict (bounded depth/breadth).

    A leaf is a single-field predicate; an internal node composes children with a
    logical operator. Depth- and breadth-bounded so a single draw stays cheap —
    each draw drives a whole multi-page walk, so the budget per example is far
    larger here than in a single-query fuzzer.
    """

    def extend(children: st.SearchStrategy[dict[str, Any]]) -> st.SearchStrategy[dict[str, Any]]:
        branch = st.lists(children, min_size=1, max_size=3)
        return st.one_of(
            children,
            branch.map(lambda cs: {"$and": cs}),
            branch.map(lambda cs: {"$or": cs}),
            branch.map(lambda cs: {"$nor": cs}),
            children.map(lambda c: {"$not": c}),
            # A bag of sibling single-field predicates (implicit AND).
            st.lists(_single_predicate(), min_size=2, max_size=3).map(
                lambda preds: {k: v for p in preds for k, v in p.items()}
            ),
        )

    return st.recursive(_single_predicate(), extend, max_leaves=max_depth * 3)


def walk_status() -> st.SearchStrategy[str | None]:
    """A ``status`` narrowing drawn from the seeded values (or ``None``)."""
    return st.sampled_from(STATUS_POOL)


def walk_updated_before() -> st.SearchStrategy[datetime | None]:
    """An ``updated_before`` bound: unfiltered, both saturating ends, or a real cut."""
    return st.sampled_from(UPDATED_BEFORE_POOL)


def validated_walk_ast(filter_dict: dict[str, Any]) -> FilterNode | None:
    """Validate + lower a generated dict, or ``None`` if it fails validation.

    A draw the validator rejects (a malformed shape the per-key rules did not
    fully constrain) is discarded by the caller — the strategy is biased so this
    is rare, not the common path.
    """
    try:
        model = RecallFilter.model_validate(filter_dict)
    except RecallFilterValidationError:
        return None
    return parse_to_ast(model)


def to_walk_ast(wire: dict[str, Any]) -> FilterNode:
    """Lower a known-valid wire filter to the canonical AST (deterministic tests)."""
    return parse_to_ast(RecallFilter.model_validate(wire))


# --------------------------------------------------------------------------- #
# The walk driver.
# --------------------------------------------------------------------------- #


ScanPage = Callable[..., Awaitable[DocumentPage]]


async def drive_walk(
    scan_page: ScanPage,
    namespace_id: UUID,
    *,
    limit: int,
    scan_bound: int,
    max_pages: int = 200,
    **kwargs: Any,
) -> list[DocumentPage]:
    """Walk a namespace to exhaustion, chaining ``next_after``; return every page.

    The caller asserts on the pages' concatenation (order, exactly-once,
    completeness) and on their individual shape (``next_after`` / ``exhausted``
    polarity, ``len <= limit``). This only drives.

    ``scan_bound`` is passed on EVERY call, so the whole walk runs under one
    budget regime — that is what makes the differential property (a
    ``scan_bound=1`` walk against an unbounded one) compare like for like.

    Two guardrails, because both failures a walk fuzzer exists to catch are
    non-terminating rather than wrong-answer. A cursor that fails to advance past
    its own row — the symptom of a hand-formatted timestamp bind, whose ISO
    ``'T'`` separator sorts above every stored value — would otherwise spin
    forever, so the driver raises the moment ``next_after`` repeats.
    ``max_pages`` is the backstop for every other form of non-termination. A
    guardrail must fail, not hang.

    A non-terminal page with ``next_after is None`` also raises here rather than
    silently ending the walk: it is a contract violation
    (``next_after is None`` iff ``exhausted``), and continuing would restart the
    walk from the newest row and loop forever.
    """
    pages: list[DocumentPage] = []
    after: DocumentCursor | None = None
    seen_positions: set[tuple[datetime, UUID]] = set()
    while len(pages) < max_pages:
        page = await scan_page(
            namespace_id,
            limit=limit,
            after=None if after is None else (after.created_at, after.id),
            scan_bound=scan_bound,
            **kwargs,
        )
        pages.append(page)
        if page.exhausted:
            return pages
        if page.next_after is None:
            raise AssertionError(
                f"page {len(pages) - 1} reported next_after=None without exhausted; "
                "the walk has no sound position to resume from"
            )
        # Any repeat, not only an immediate one: a longer cycle (A -> B -> A)
        # cannot terminate either, and would otherwise burn the whole max_pages
        # backstop instead of naming the offending position.
        position = (page.next_after.created_at, page.next_after.id)
        if position in seen_positions:
            raise AssertionError(
                f"walk cursor repeated position {page.next_after!r} at page {len(pages) - 1}; "
                "the walk revisits an earlier cursor and can never terminate"
            )
        seen_positions.add(position)
        after = page.next_after
    raise AssertionError(f"walk did not report exhausted within {max_pages} pages of limit={limit}")


def walked_ids(pages: Sequence[DocumentPage]) -> list[UUID]:
    """The concatenated document ids across every page, in walk order."""
    return [doc.id for page in pages for doc in page]


def walked_keys(pages: Sequence[DocumentPage]) -> list[tuple[datetime, UUID]]:
    """The concatenated ``(created_at, id)`` sort keys, UTC-normalized, in walk order."""
    return [(as_utc(doc.created_at), doc.id) for page in pages for doc in page]


# --------------------------------------------------------------------------- #
# Anti-vacuous collectors.
# --------------------------------------------------------------------------- #
#
# A walk property that always compared the FULL corpus against the FULL corpus
# would agree trivially and prove nothing, and so would one whose every walk fit
# in a single page. Both legs record what their draws actually produced and a
# later plain test judges the distribution. A module-level collector is robust and
# simple (no dependence on Hypothesis statistics plumbing).


@dataclass
class WalkCollectors:
    """Per-module records of what the generated walks actually exercised."""

    page_counts: list[int] = field(default_factory=list)
    """Page count per ``scan_bound=1`` walk — judged for a multi-page fraction."""


def assert_multipage_fraction(page_counts: Sequence[int], *, minimum: float = 0.50) -> None:
    """At least ``minimum`` of recorded walks produced >= 2 NON-TERMINAL pages.

    The anti-vacuous guard for the WALK half specifically. A suite whose every
    walk terminated on its first page would check the page contract and never the
    cursor — the stitching across a page boundary is the whole point.

    This is a FLOOR, not a claim about match distribution. Under the budget the
    properties actually run at (``scan_bound=1``) it is satisfied by construction,
    because a page scans one raw row whether or not that row matches — so it
    cannot detect "every walk's matches happened to land on one page". It exists
    to catch a future budget or corpus change that quietly collapsed the walks to
    a single page. The complementary check — that matches are genuinely SPLIT
    across pages with rejected pages between them — is a deterministic test in the
    leg modules, where it can assert a concrete filter's page shape instead of a
    fraction that would sit one standard deviation from its own threshold.
    """
    assert page_counts, "no walk page counts recorded"
    # The terminal page is the empty exhausted one; a walk that "crossed a
    # boundary" needs two pages BEFORE it.
    multi = sum(1 for count in page_counts if count - 1 >= 2)
    fraction = multi / len(page_counts)
    assert fraction >= minimum, (
        f"only {fraction:.0%} of {len(page_counts)} walks produced >=2 non-terminal pages "
        f"(need >={minimum:.0%}); the walks are not crossing page boundaries"
    )


__all__ = [
    "CORPUS_SIZE",
    "METADATA_PATHS",
    "META_SCALAR_POOL",
    "STATUS_POOL",
    "STRING_POOLS",
    "UPDATED_BEFORE_POOL",
    "WalkCollectors",
    "WalkCorpus",
    "assert_multipage_fraction",
    "build_walk_corpus",
    "drive_walk",
    "to_walk_ast",
    "validated_walk_ast",
    "walk_filter",
    "walk_oracle",
    "walk_status",
    "walk_updated_before",
    "walked_ids",
    "walked_keys",
]
