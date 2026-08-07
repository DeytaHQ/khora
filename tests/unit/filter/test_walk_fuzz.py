"""Property-based walk fuzzer for the document-enumeration surface (embedded leg).

``StorageCoordinator.scan_documents_page`` is pinned per store by hand-authored
tests: the bounded scan primitive, the cursor serialization, the pushdown split.
Those are precise but finite, and — more to the point — they are *page* tests.
They check one ``SELECT`` at a time.

This module checks the WALK. It generates a ``(filter, status,
updated_before)`` triple with Hypothesis, drives the whole multi-page
enumeration to exhaustion under an injected scan budget, and compares the
concatenation against a pure-Python oracle over the same corpus. The defects
that live only there are the ones a page test structurally cannot see: a cursor
that skips a tie-mate exactly on a page boundary, a page that reports
``exhausted`` while rows remain, a walk whose answer depends on how the budget
happened to slice it.

What it does NOT check is filter semantics: the coordinator's post-filter and the
oracle share the same ``compile_python`` predicate, so a bug inside it corrupts
both sides identically — that surface belongs to the recall filter fuzzer and the
SQL-compiler superset checks. On this store the withheld date keys are
post-filtered on BOTH sides, so their filtering in particular is not
independently cross-checked here (the PostgreSQL leg, which pushes them into SQL,
is where that comparison happens).

Five properties, folded into three walks so the SELECT count stays bounded:

* **A — exactly-once, order, honest termination** (``TestWalkEnumeration``).
  One walk per draw at ``limit=3`` and a deliberately tiny ``scan_bound``, so
  most pages return one match and the walk is nearly all cursor boundaries. The
  concatenation must equal the oracle *as a sequence* (completeness + no repeats
  + total order), every page's ``(created_at, id)`` must strictly descend across
  the seam, and ``next_after is None`` must hold exactly on the terminal page.
* **B — cursor stitching** (``TestCursorStitchDifferential``, load-bearing). The
  same draw walked at ``scan_bound=1`` (a cursor boundary between every raw row)
  and unbounded (one page, no cursor at all) must agree with each other AND with
  the oracle. The unbounded walk never serializes a cursor, so it is the control:
  a divergence is the cursor, not the filter.
* **C — limit invariance** (``TestLimitInvariance``). ``limit`` is a page-size
  knob, not a semantic one; three limits over one large budget must produce one
  answer.

Plus two deterministic companions that a generated corpus cannot express:
``TestMutationDuringWalk`` (no snapshot spans pages — what a concurrent insert or
delete does to a walk in flight) and ``TestCursorCodec`` (the ``DocumentCursor``
round trip at both microsecond polarities, and the typed-bind contrast the whole
keyset predicate rests on).

The store is the real embedded ``sqlite_lance`` relational adapter behind a real
``StorageCoordinator``, seeded once per process through ``create_document`` — the
production write API. Seeding by raw SQL would compare the walk against a corpus
production could never produce, which is exactly the comparison the cursor tests
must not make.
"""

from __future__ import annotations

import tempfile
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from hypothesis import HealthCheck, assume, given, seed, settings
from hypothesis import strategies as st

from khora.core.models import Document, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentCursor, DocumentPage
from khora.db.session import run_migrations
from khora.storage.backends._documents_scan import build_documents_scan_query
from khora.storage.backends.sqlite_lance import SQLiteLanceRelationalAdapter
from khora.storage.backends.sqlite_lance._helpers import uuid_to_text
from khora.storage.backends.sqlite_lance.connection import (
    EmbeddedStorageHandle,
    EmbeddedStorageHandleConfig,
)
from khora.storage.coordinator import StorageCoordinator
from tests.integration._sqlite_lance_fixtures import EMBED_DIM
from tests.integration.matrix._conformance_lance import _run_async
from tests.test_helpers.document_order import id_ladder
from tests.test_helpers.document_scan import WHOLE_SECOND, ScanSeed, as_utc, scan_seed
from tests.test_helpers.walk_fuzz import (
    CORPUS_SIZE,
    WalkCollectors,
    assert_multipage_fraction,
    build_walk_corpus,
    drive_walk,
    to_walk_ast,
    validated_walk_ast,
    walk_filter,
    walk_oracle,
    walk_status,
    walk_updated_before,
    walked_ids,
    walked_keys,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# The fixed corpus + the seeded embedded store.
# --------------------------------------------------------------------------- #
#
# RE-ENTRANCY: the ``@lru_cache`` singleton is resolved on the CALLER (test)
# thread; only the read/write coroutines are submitted to the loop thread that
# owns the aiosqlite connection (``_run_async``). An aiosqlite handle is bound to
# the loop it was opened on, so every later call MUST go through that same loop —
# and resolving the cache from inside a coroutine already running there would
# deadlock (the loop would block on a future only it can complete). Module-level
# singletons (not a function-scoped fixture) also keep Hypothesis's
# ``function_scoped_fixture`` health check from firing on the @given tests.

CORPUS_NAMESPACE_ID = uuid4()
CORPUS = build_walk_corpus(CORPUS_NAMESPACE_ID)

# Anything at or above the corpus size is "unbounded" for these walks: the budget
# counts RAW rows scanned per page, so a bound past the corpus can never cut a
# page short and the whole namespace is one page. Deliberately +1 so a corpus
# that grew by one row would still be covered by the name.
UNBOUNDED = CORPUS_SIZE + 1

COLLECTORS = WalkCollectors()


def _budget(max_examples: int) -> settings:
    """The shared Hypothesis profile: no deadline (a walk is many round trips).

    ``function_scoped_fixture`` is suppressed because these tests take none — the
    store is a module-level singleton — and ``too_slow`` because a single example
    drives a whole multi-page enumeration, which is inherently slower than the
    single-query draws Hypothesis's default budget assumes.
    """
    return settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )


class _SeededWalkStore:
    """A connected relational-only coordinator holding the frozen walk corpus."""

    def __init__(self, coord: StorageCoordinator, handle: EmbeddedStorageHandle) -> None:
        self.coord = coord
        self.handle = handle


async def _build_seeded_store() -> _SeededWalkStore:
    """Migrate a tmp SQLite file, wire the relational adapter, seed the corpus."""
    tmp_path = Path(tempfile.mkdtemp(prefix="khora-walk-fuzz-"))
    db_path = str(tmp_path / "khora.db")
    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    if not result.success:
        raise RuntimeError(f"migration failed: {result.error}")

    handle = EmbeddedStorageHandle(
        EmbeddedStorageHandleConfig(
            db_path=db_path,
            lance_path=str(tmp_path / "khora.lance"),
            embedding_dimension=EMBED_DIM,
        )
    )
    await handle.connect()
    # ``vector=None``: enumeration is a relational-only read path, and a
    # ``delete_document`` with no vector backend skips the chunk purge, which is
    # what the mutation companion wants — it mutates the document table only.
    coord = StorageCoordinator(relational=SQLiteLanceRelationalAdapter(handle), vector=None)
    await coord.connect()

    await _create_namespace(coord, CORPUS_NAMESPACE_ID)
    for document in CORPUS.documents:
        await coord.create_document(document)

    # Materialization guard (fail loud, at build time). If the seeder silently
    # dropped rows, BOTH the oracle and the walk would later agree on a too-small
    # corpus — the empty-store false-green. Counted through the scan surface at a
    # budget past the corpus, so the guard also proves a one-page full read works
    # before any property depends on it.
    page = await coord.scan_documents_page(CORPUS_NAMESPACE_ID, limit=CORPUS_SIZE + 5, scan_bound=UNBOUNDED)
    seeded = [doc.id for doc in page]
    if seeded != list(CORPUS.expected):
        raise RuntimeError(
            f"corpus did not materialize as expected: seeded {len(seeded)} rows, "
            f"expected {len(CORPUS.expected)} in a pinned order\n"
            f"  missing = {sorted(set(CORPUS.expected) - set(seeded))}\n"
            f"  extra   = {sorted(set(seeded) - set(CORPUS.expected))}\n"
            f"  seeded  = {seeded}"
        )
    return _SeededWalkStore(coord, handle)


@lru_cache(maxsize=1)
def _seeded_store() -> _SeededWalkStore:
    """The process-wide seeded embedded store (built + seeded exactly once).

    Resolved on the CALLER thread, never inside a coroutine already running on the
    loop thread. Like the conformance helper's loop-thread coordinator, the store
    is intentionally process-lived and left to interpreter shutdown rather than
    closed via an ``atexit`` hook that would log into a torn-down loguru sink.
    """
    return _run_async(_build_seeded_store())


async def _create_namespace(coord: StorageCoordinator, namespace_id: UUID) -> None:
    """Create a namespace whose row id and stable ``namespace_id`` agree.

    Documents are scoped by the row-level ``id`` here (that is what
    ``DocumentModel.namespace_id`` references), so pinning both to one UUID keeps
    every call site in this module unambiguous.
    """
    await coord.create_namespace(
        MemoryNamespace(id=namespace_id, namespace_id=namespace_id, tenancy_mode=TenancyMode.SHARED)
    )


def _walk(*, limit: int, scan_bound: int, namespace_id: UUID | None = None, **kwargs: Any) -> list[DocumentPage]:
    """Drive a whole walk on the loop thread and return its pages.

    ``scan_bound`` is passed straight to ``scan_documents_page``. The facade
    (``Khora.list_documents``) derives it from
    ``query.document_scan_overfetch_multiplier`` / ``document_scan_min_bound``
    instead; that path is deliberately BYPASSED here, because the fuzzer's whole
    leverage is choosing budgets small enough to force a page boundary between
    almost every raw row — which no reachable config produces.
    """
    store = _seeded_store()  # caller-thread resolution — never inside the loop coroutine
    return _run_async(
        drive_walk(
            store.coord.scan_documents_page,
            CORPUS_NAMESPACE_ID if namespace_id is None else namespace_id,
            limit=limit,
            scan_bound=scan_bound,
            **kwargs,
        )
    )


def assert_page_contract(pages: list[DocumentPage], *, limit: int) -> None:
    """Every page's shape invariants — the ``next_after``/``exhausted`` polarity.

    ``next_after is None`` **iff** ``exhausted`` is the contract
    ``DocumentPage`` states, and it is the one a caller's loop condition rests
    on: a page that returned neither a resume position nor exhaustion silently
    truncates the walk, and one that returned both invites an extra round trip
    that re-reads the tail. Only the terminal page may be exhausted — an interior
    ``exhausted=True`` would have ended the walk early.
    """
    for index, page in enumerate(pages):
        is_last = index == len(pages) - 1
        assert len(page) <= limit, f"page {index} returned {len(page)} matches over limit={limit}"
        assert (page.next_after is None) is page.exhausted, (
            f"page {index} broke the next_after/exhausted contract: "
            f"next_after={page.next_after!r}, exhausted={page.exhausted}"
        )
        assert page.exhausted is is_last, f"page {index} exhausted={page.exhausted} but is_last={is_last}"
    assert pages[-1].exhausted is True, "the walk ended without an exhausted page"


def assert_strictly_descending(pages: list[DocumentPage]) -> None:
    """The concatenated ``(created_at, id)`` keys strictly descend across page seams.

    Strict (not merely non-increasing) is the assertion that matters: an equal
    adjacent pair is a row served twice, and the tie blocks in the corpus mean a
    tie-break bug shows up here as an equality rather than as an inversion.
    """
    keys = walked_keys(pages)
    for previous, current in zip(keys, keys[1:], strict=False):
        assert current < previous, f"walk order is not strictly descending: {previous!r} then {current!r}"


# --------------------------------------------------------------------------- #
# Property A — exactly-once + completeness + order + honest termination.
# --------------------------------------------------------------------------- #


class TestWalkEnumeration:
    """One generated walk must enumerate exactly the oracle, once each, in order.

    Combined into a single walk per draw on purpose: each of the four properties
    it asserts is cheap to check and expensive to produce (a ``scan_bound=1``
    walk over the corpus is ~21 round trips), so splitting them across four
    ``@given`` tests would quadruple the SELECT count for no additional coverage.
    """

    @given(
        filter_dict=walk_filter(),
        status=walk_status(),
        updated_before=walk_updated_before(),
        scan_bound=st.sampled_from([1, 2, 3, 5]),
    )
    @_budget(100)
    def test_walk_matches_the_oracle_exactly_once_and_in_order(
        self,
        filter_dict: dict[str, Any],
        status: str | None,
        updated_before: datetime | None,
        scan_bound: int,
    ) -> None:
        ast = validated_walk_ast(filter_dict)
        assume(ast is not None)  # invalid draw — rare; the strategy is biased to validate
        assert ast is not None  # narrow for the type checker after the assume

        expected = walk_oracle(CORPUS.documents, filter_ast=ast, status=status, updated_before=updated_before)
        pages = _walk(
            limit=3,
            scan_bound=scan_bound,
            filter_ast=ast,
            status=status,
            updated_before=updated_before,
        )
        seen = walked_ids(pages)

        if scan_bound == 1:
            COLLECTORS.page_counts.append(len(pages))

        assert len(seen) == len(set(seen)), f"a document was served twice: {filter_dict!r}"
        assert set(seen) == set(expected), (
            "walk/oracle set divergence:\n"
            f"  filter = {filter_dict!r} status={status!r} updated_before={updated_before!r}\n"
            f"  missing = {sorted(set(expected) - set(seen))}\n"
            f"  extra   = {sorted(set(seen) - set(expected))}"
        )
        assert seen == expected, f"walk/oracle ORDER divergence at scan_bound={scan_bound}: {filter_dict!r}"
        assert_strictly_descending(pages)
        assert_page_contract(pages, limit=3)


# --------------------------------------------------------------------------- #
# Property B — cursor stitching (the load-bearing differential).
# --------------------------------------------------------------------------- #


class TestCursorStitchDifferential:
    """A walk stitched from many cursors equals one that never serializes a cursor.

    The load-bearing property. At ``scan_bound=1`` every raw row is a page
    boundary, so the whole answer is reassembled from ~21 round-tripped
    ``DocumentCursor`` positions — several of them strictly inside a tie block,
    where only a correctly-bound ``(created_at, id)`` lands on the right row. The
    unbounded walk reads the same namespace in ONE page with ``after=None``, so
    it exercises no cursor at all and is the control: if the two disagree, the
    cursor is what broke, not the filter (the oracle third leg then says which
    side moved).
    """

    @given(filter_dict=walk_filter(), status=walk_status(), updated_before=walk_updated_before())
    @_budget(60)
    def test_stitched_walk_equals_unstitched_walk_and_oracle(
        self, filter_dict: dict[str, Any], status: str | None, updated_before: datetime | None
    ) -> None:
        ast = validated_walk_ast(filter_dict)
        assume(ast is not None)
        assert ast is not None

        narrowing: dict[str, Any] = {"filter_ast": ast, "status": status, "updated_before": updated_before}
        stitched = walked_ids(_walk(limit=3, scan_bound=1, **narrowing))
        unstitched = walked_ids(_walk(limit=CORPUS_SIZE + 5, scan_bound=UNBOUNDED, **narrowing))
        expected = walk_oracle(CORPUS.documents, **narrowing)

        assert stitched == unstitched, (
            "cursor-stitched and single-page walks disagree — the cursor is the only "
            f"difference between them:\n  filter = {filter_dict!r}\n"
            f"  stitched   = {stitched}\n  unstitched = {unstitched}"
        )
        assert stitched == expected, f"both walks agree but diverge from the oracle: {filter_dict!r}"


class TestWalkSeedSweep:
    """Property B re-run under eight distinct Hypothesis seeds — stability sweep.

    A single PRNG stream could miss a discriminating draw by luck; eight fixed
    seeds (few examples each, so total CI time stays bounded) confirm the
    stitching agreement holds across independent streams, not one.
    """

    @pytest.mark.parametrize("hypothesis_seed", range(8))
    def test_stitch_holds_under_seed(self, hypothesis_seed: int) -> None:
        @seed(hypothesis_seed)
        @given(filter_dict=walk_filter(), status=walk_status(), updated_before=walk_updated_before())
        @_budget(12)
        def _check(filter_dict: dict[str, Any], status: str | None, updated_before: datetime | None) -> None:
            ast = validated_walk_ast(filter_dict)
            assume(ast is not None)
            assert ast is not None

            narrowing: dict[str, Any] = {"filter_ast": ast, "status": status, "updated_before": updated_before}
            stitched = walked_ids(_walk(limit=3, scan_bound=1, **narrowing))
            expected = walk_oracle(CORPUS.documents, **narrowing)
            assert stitched == expected, (
                f"seed={hypothesis_seed} stitched walk diverged from the oracle:\n  filter = {filter_dict!r}"
            )

        _check()


# --------------------------------------------------------------------------- #
# Property C — limit invariance.
# --------------------------------------------------------------------------- #


class TestLimitInvariance:
    """``limit`` slices the answer into pages; it must not change the answer.

    Three limits (one below, one at, one above the natural page shape) over one
    budget large enough that the scan bound never cuts a page short — so the ONLY
    thing varying is where the page boundaries fall. A ``limit``-dependent answer
    would mean the page-fill loop is dropping or duplicating rows at the seam
    (the coordinator sizes each scan step to the match shortfall, so a
    mis-computed shortfall shows up here and nowhere else).
    """

    @given(filter_dict=walk_filter(), status=walk_status(), updated_before=walk_updated_before())
    @_budget(50)
    def test_answer_is_independent_of_limit(
        self, filter_dict: dict[str, Any], status: str | None, updated_before: datetime | None
    ) -> None:
        ast = validated_walk_ast(filter_dict)
        assume(ast is not None)
        assert ast is not None

        narrowing: dict[str, Any] = {"filter_ast": ast, "status": status, "updated_before": updated_before}
        answers = {limit: walked_ids(_walk(limit=limit, scan_bound=UNBOUNDED, **narrowing)) for limit in (1, 3, 7)}
        expected = walk_oracle(CORPUS.documents, **narrowing)

        assert answers[1] == answers[3] == answers[7], f"the answer depends on limit: {filter_dict!r}\n  {answers}"
        assert answers[1] == expected, f"every limit agrees but diverges from the oracle: {filter_dict!r}"


# --------------------------------------------------------------------------- #
# Determinism — a repeated walk over a frozen namespace reproduces itself.
# --------------------------------------------------------------------------- #


def page_shapes(pages: list[DocumentPage]) -> list[tuple[Any, ...]]:
    """Everything a caller can observe about each page, flattened for comparison.

    The cursor is compared as its ``(created_at, id)`` pair rather than as the
    dataclass, so a divergence in either half is named by the failure output.
    """
    return [
        (
            tuple(doc.id for doc in page),
            None if page.next_after is None else (page.next_after.created_at, page.next_after.id),
            page.exhausted,
            page.post_filtered_keys,
        )
        for page in pages
    ]


class TestWalkDeterminism:
    """The same walk, run twice over an unchanged namespace, must be the same walk.

    Every generated property above compares a walk to an oracle, which would still
    pass if the store returned a *different but equally correct* slicing on each
    run — a page-boundary that moved, a cursor that landed one row earlier. A
    caller resuming a paged enumeration depends on more than the concatenation:
    the cursor it stored must still mean what it meant, page for page. This pins
    the whole page sequence, not just its concatenation.
    """

    def test_repeating_a_walk_reproduces_every_page_exactly(self) -> None:
        """Two identical walks agree on ids, page boundaries, cursors and flags.

        ``source_type = library`` keeps a strict, multi-row subset, and
        ``limit=3, scan_bound=1`` forces the answer to be stitched from one raw
        row per page — so the walk genuinely crosses boundaries rather than
        collapsing into a single page where determinism would be trivial.
        """
        ast = to_walk_ast({"source_type": "library"})
        first = _walk(limit=3, scan_bound=1, filter_ast=ast)
        second = _walk(limit=3, scan_bound=1, filter_ast=ast)

        expected = walk_oracle(CORPUS.documents, filter_ast=ast)
        assert 0 < len(expected) < CORPUS_SIZE, expected
        # Non-vacuity: a single-page walk would make the comparison below trivial.
        assert len([page for page in first if len(page) > 0]) >= 2, [len(p) for p in first]

        assert page_shapes(first) == page_shapes(second), (
            "a repeated walk over a frozen namespace produced a different page sequence:\n"
            f"  first  = {page_shapes(first)}\n  second = {page_shapes(second)}"
        )
        assert walked_ids(first) == expected


# --------------------------------------------------------------------------- #
# Anti-vacuous guards.
# --------------------------------------------------------------------------- #


class TestWalkDiscriminates:
    """The corpus, the strategy and the budgets actually exercise what they claim.

    Two false-green shapes are ruled out here without leaning on any generated
    draw. A corpus every filter keeps whole (or drops whole) would make the oracle
    comparison trivial — ``test_a_partial_filter_keeps_a_strict_subset`` rules it
    out deterministically. A budget regime where every walk fits in one page would
    make the *walk* half of a walk fuzzer inert — the multipage floor below plus
    the hand-written reachability check rule that out.

    There is deliberately no strict-subset *fraction* guard. A ``>=30%`` floor over
    this strategy is not sound: its true strict-subset rate hugs the floor (a
    100-draw oracle sample has been observed at 26%), and under ``-n auto`` the
    process-local collector holds only the draws that landed on this worker, so a
    fraction read off it is a partial sample of an already-marginal rate. The
    deterministic partial-filter test is the discrimination proof instead. The
    multipage floor survives because it is not marginal: at ``scan_bound=1`` every
    walk crosses boundaries by construction, and it falls back to one concrete
    full-corpus walk when the collector is empty.
    """

    def test_most_generated_walks_cross_a_page_boundary(self) -> None:
        """At least half the ``scan_bound=1`` walks needed two or more pages.

        Falls back to one concrete unfiltered walk when the collector is empty.
        The bound makes multi-page walks the norm by construction; the assertion
        exists so a future budget change that quietly collapsed every walk to a
        single page cannot pass silently.
        """
        counts = COLLECTORS.page_counts or [len(_walk(limit=3, scan_bound=1))]
        assert_multipage_fraction(counts)

    def test_matches_are_split_across_pages_with_rejected_pages_between(self) -> None:
        """A filtered walk really does stitch matches across a rejected gap.

        The complementary half of the page-count floor above, which by
        construction cannot see this: at ``scan_bound=1`` a page scans one raw
        row whether or not it matches, so a walk can be long and still have every
        match land on one page. What must be reachable is the shape the
        coordinator's "resume from the last RAW row, not the last MATCH" design
        exists for — two match-bearing pages with at least one page that scanned a
        row and returned nothing in between. A walk that resumed from the last
        matching row instead could not advance past a rejected gap at all.

        The filter is a ``source_timestamp`` leaf on purpose. This store WITHHOLDS
        both date system keys from pushdown (their stored TEXT format does not
        order against the compiler's ISO binds), so the leaf is enforced only in
        memory and the raw window is the whole namespace — which is what produces
        scanned-but-rejected pages. A pushable leaf such as ``source_type`` would
        narrow the window in SQL and every scanned row would match, so this test
        would prove nothing. The ``post_filtered_keys`` assertion pins that
        premise rather than trusting it.
        """
        ast = to_walk_ast({"source_timestamp": {"$eq": "2026-06-01T00:00:00Z"}})
        pages = _walk(limit=3, scan_bound=1, filter_ast=ast)

        assert pages[0].post_filtered_keys == ("source_timestamp",), pages[0].post_filtered_keys
        bearing = [index for index, page in enumerate(pages) if len(page) > 0]
        assert len(bearing) >= 2, f"the filter did not split its matches across pages: {bearing}"
        gaps = [
            index
            for index in range(bearing[0] + 1, bearing[-1])
            if len(pages[index]) == 0 and not pages[index].exhausted
        ]
        assert gaps, "no rejected page sits between two match-bearing pages"
        assert walked_ids(pages) == walk_oracle(CORPUS.documents, filter_ast=ast)

    def test_a_partial_filter_keeps_a_strict_subset(self) -> None:
        """Explicit (non-Hypothesis) proof that the corpus discriminates.

        A concrete partial filter must keep a strict, non-empty subset on both
        the oracle and the real walk. If this fails, every generated agreement
        above was an agreement about the whole corpus.
        """
        ast = to_walk_ast({"source_type": "library"})
        expected = walk_oracle(CORPUS.documents, filter_ast=ast)
        assert 0 < len(expected) < CORPUS_SIZE, expected
        assert walked_ids(_walk(limit=3, scan_bound=UNBOUNDED, filter_ast=ast)) == expected

    def test_unfiltered_walk_at_scan_bound_one_pages_the_whole_corpus(self) -> None:
        """The reachability check the two fraction guards lean on.

        A ``scan_bound=1`` walk over the unfiltered corpus scans one raw row per
        page, so it takes exactly ``CORPUS_SIZE + 1`` pages — the corpus, plus the
        empty tail page that is the only sound termination signal — and its
        concatenation is the pinned ``(created_at DESC, id DESC)`` enumeration.
        """
        pages = _walk(limit=3, scan_bound=1)

        assert len(pages) == CORPUS_SIZE + 1, [len(p) for p in pages]
        assert walked_ids(pages) == list(CORPUS.expected)
        assert [len(p) for p in pages[:-1]] == [1] * CORPUS_SIZE
        assert len(pages[-1]) == 0
        assert_page_contract(pages, limit=3)


# --------------------------------------------------------------------------- #
# Mutation during a walk (deterministic — no snapshot spans pages).
# --------------------------------------------------------------------------- #
#
# ``scan_documents_page`` explicitly claims no consistent snapshot: each page is
# its own SELECT in its own session. That is a real, caller-visible property, and
# it is *not* "anything can happen" — a keyset walk moving monotonically down
# ``(created_at DESC, id DESC)`` has an exact, checkable answer for each of the
# four mutation shapes. Pinning them is what stops a future "improvement" from
# quietly changing which rows a concurrent writer's document lands in.


async def _seed_gap_ladder(coord: StorageCoordinator, namespace_id: UUID, ids: list[UUID], base: datetime) -> None:
    """Seed ``ids`` with strictly DESCENDING ``created_at``, ten seconds apart.

    No ties and a deliberate gap between neighbours, so a mutation test can place
    a new row strictly between two existing ones and name where it must land. The
    enumeration order is therefore exactly ``ids`` — ladder order — which keeps
    these scenarios readable. (The tie-break itself is covered exhaustively by the
    generated properties above; repeating it here would only obscure the
    mutation semantics.)
    """
    for offset, doc_id in enumerate(ids):
        await coord.create_document(
            Document(
                id=doc_id,
                namespace_id=namespace_id,
                content=f"mutation row {offset}",
                checksum=f"mut-{doc_id.hex}",
                created_at=base - timedelta(seconds=10 * offset),
                updated_at=base,
            )
        )


async def _insert(coord: StorageCoordinator, namespace_id: UUID, doc_id: UUID, created_at: datetime) -> None:
    await coord.create_document(
        Document(
            id=doc_id,
            namespace_id=namespace_id,
            content="inserted mid-walk",
            checksum=f"mut-{doc_id.hex}",
            created_at=created_at,
            updated_at=created_at,
        )
    )


async def _walk_with_mutations(
    coord: StorageCoordinator,
    namespace_id: UUID,
    mutations: dict[int, Callable[[], Awaitable[None]]],
    *,
    max_pages: int = 60,
) -> list[UUID]:
    """Walk one row at a time, running ``mutations[i]`` after page ``i`` returns.

    ``limit=1, scan_bound=1`` puts a mutation window between every pair of rows,
    which is the only way to place a write at a *known* point in the walk. Returns
    the concatenated ids the walk served.
    """
    seen: list[UUID] = []
    after: DocumentCursor | None = None
    index = 0
    while index < max_pages:
        page = await coord.scan_documents_page(
            namespace_id,
            limit=1,
            after=None if after is None else (after.created_at, after.id),
            scan_bound=1,
        )
        seen.extend(doc.id for doc in page)
        mutate = mutations.get(index)
        if mutate is not None:
            await mutate()
        if page.exhausted:
            return seen
        assert page.next_after is not None
        after = page.next_after
        index += 1
    raise AssertionError(f"mutating walk did not terminate within {max_pages} pages")


class TestMutationDuringWalk:
    """What a concurrent insert or delete does to a walk already in flight.

    Each scenario runs in its own fresh namespace inside the shared store, so a
    mutation can never leak into the read-only corpus the generated properties
    depend on. Deliberately NOT a Hypothesis test: these are four exact,
    hand-placed answers, and generating them would only obscure which one broke.
    """

    def test_insert_above_the_cursor_is_not_seen(self) -> None:
        """A row newer than the cursor is already behind the walk — never served.

        A keyset walk moves monotonically down ``(created_at DESC, id DESC)`` and
        never looks back, so a document inserted with a ``created_at`` above the
        current position is invisible to THIS walk. That is not a bug to fix at
        the walk layer — it is the price of resumability, and a caller that needs
        the new row starts a new walk.
        """
        result = _run_async(self._insert_above())
        assert result["above_id"] not in result["seen"]
        assert result["seen"] == result["ladder"]

    async def _insert_above(self) -> dict[str, Any]:
        coord = _seeded_store().coord
        namespace_id, ladder, base = await _fresh_ladder(coord, 5)
        above_id = uuid4()

        async def mutate() -> None:
            await _insert(coord, namespace_id, above_id, base + timedelta(seconds=30))

        seen = await _walk_with_mutations(coord, namespace_id, {1: mutate})
        return {"seen": seen, "ladder": ladder, "above_id": above_id}

    def test_insert_below_the_cursor_is_seen(self) -> None:
        """A row older than the cursor is still ahead of the walk — served in order.

        The mirror of the case above, and the half that is easy to get wrong: the
        insert lands strictly between two existing rows, so it must appear at that
        exact position in the concatenation, not appended at the end.
        """
        result = _run_async(self._insert_below())
        ladder = result["ladder"]
        # Placed between ladder[2] and ladder[3] by construction.
        assert result["seen"] == [*ladder[:3], result["below_id"], *ladder[3:]]

    async def _insert_below(self) -> dict[str, Any]:
        coord = _seeded_store().coord
        namespace_id, ladder, base = await _fresh_ladder(coord, 5)
        below_id = uuid4()

        async def mutate() -> None:
            # ladder[2] is at base-20s and ladder[3] at base-30s; land between.
            await _insert(coord, namespace_id, below_id, base - timedelta(seconds=25))

        seen = await _walk_with_mutations(coord, namespace_id, {1: mutate})
        return {"seen": seen, "ladder": ladder, "below_id": below_id}

    def test_delete_of_an_unscanned_row_below_the_cursor_excludes_it(self) -> None:
        """A row deleted before the walk reaches it is simply never served.

        The walk must also not stall on the hole: the rows after the deleted one
        still come back, in order.
        """
        result = _run_async(self._delete_ahead())
        ladder = result["ladder"]
        assert result["seen"] == [d for d in ladder if d != ladder[4]]
        assert ladder[4] not in result["seen"]

    async def _delete_ahead(self) -> dict[str, Any]:
        coord = _seeded_store().coord
        namespace_id, ladder, _base = await _fresh_ladder(coord, 6)

        async def mutate() -> None:
            await coord.delete_document(ladder[4], namespace_id=namespace_id)

        seen = await _walk_with_mutations(coord, namespace_id, {1: mutate})
        return {"seen": seen, "ladder": ladder}

    def test_delete_of_an_already_returned_row_does_not_retract_it(self) -> None:
        """A row deleted AFTER it was served stays in the caller's hands.

        Nothing can unsend a page, and — the part worth pinning — the walk also
        does not stumble on resuming from a cursor whose own row no longer
        exists. The keyset predicate is a comparison, not a lookup, so a deleted
        anchor is fine; a walk that re-read its anchor row would break here.
        """
        result = _run_async(self._delete_behind())
        assert result["seen"] == result["ladder"]

    async def _delete_behind(self) -> dict[str, Any]:
        coord = _seeded_store().coord
        namespace_id, ladder, _base = await _fresh_ladder(coord, 5)

        async def mutate() -> None:
            # Delete the row page 1 just served — the walk's current anchor.
            await coord.delete_document(ladder[1], namespace_id=namespace_id)

        seen = await _walk_with_mutations(coord, namespace_id, {1: mutate})
        return {"seen": seen, "ladder": ladder}

    def test_all_four_mutations_in_one_walk(self) -> None:
        """The umbrella: every shape at once, in a single walk.

        ``seen`` must be exactly ``(initial ∪ inserted-below) − deleted-before-
        scanned``, each id once, with the row deleted after it was served still
        present and the row inserted above the cursor still absent. Running the
        four together is not redundant with running them apart — it is the case
        where a fix for one could regress another.
        """
        result = _run_async(self._umbrella())
        seen = result["seen"]
        ladder = result["ladder"]

        assert len(seen) == len(set(seen)), "a document was served twice across the mutating walk"
        assert set(seen) == (set(ladder) | {result["below_id"]}) - {ladder[4]}
        assert result["above_id"] not in seen
        assert ladder[0] in seen, "a row deleted after it was served must stay served"
        assert seen == [*ladder[:3], result["below_id"], ladder[3], ladder[5]]

    async def _umbrella(self) -> dict[str, Any]:
        coord = _seeded_store().coord
        namespace_id, ladder, base = await _fresh_ladder(coord, 6)
        above_id, below_id = uuid4(), uuid4()

        async def mutate() -> None:
            await _insert(coord, namespace_id, above_id, base + timedelta(seconds=30))
            await _insert(coord, namespace_id, below_id, base - timedelta(seconds=25))
            await coord.delete_document(ladder[4], namespace_id=namespace_id)  # unscanned, ahead
            await coord.delete_document(ladder[0], namespace_id=namespace_id)  # already served

        seen = await _walk_with_mutations(coord, namespace_id, {1: mutate})
        return {"seen": seen, "ladder": ladder, "above_id": above_id, "below_id": below_id}


async def _fresh_ladder(coord: StorageCoordinator, total: int) -> tuple[UUID, list[UUID], datetime]:
    """A brand-new namespace seeded with ``total`` gap-separated rows.

    Returns ``(namespace_id, ids-in-enumeration-order, base_instant)``.
    """
    namespace_id = uuid4()
    await _create_namespace(coord, namespace_id)
    base = WHOLE_SECOND
    ids = id_ladder(total)
    await _seed_gap_ladder(coord, namespace_id, ids, base)
    return namespace_id, ids, base


# --------------------------------------------------------------------------- #
# Cursor codec — the DocumentCursor round trip at the public layer.
# --------------------------------------------------------------------------- #


class TestCursorCodec:
    """A ``DocumentCursor`` fed back in resumes at the exact next row.

    The adapter-level scan tests already pin that the *builder* binds its operands
    through the ORM column types. What is checked here is one level up: that a
    :class:`DocumentCursor` handed to a caller by ``scan_documents_page`` and
    handed straight back resumes correctly — including from the middle of a tie
    block, at BOTH microsecond polarities, which is where this store's TEXT
    ``created_at`` column is most fragile.

    Both polarities are seeded because neither one alone is conclusive on this
    store. At ``.000000`` a hand-formatted ``str(naive_datetime)`` omits the
    microsecond field entirely and diverges from the stored six-digit form; at a
    non-zero microsecond it is byte-identical to it, so a corpus seeded only from
    ``datetime.now(UTC)`` would silently agree with a broken bind. See
    :mod:`tests.test_helpers.document_scan`.
    """

    def test_whole_second_tie_block_resume(self) -> None:
        """microsecond=0: resume from mid-tie excludes the anchor, keeps its tie-mate."""
        self._assert_mid_tie_resume(WHOLE_SECOND)

    def test_sub_second_tie_block_resume(self) -> None:
        """microsecond=.123456: the opposite polarity, same contract."""
        self._assert_mid_tie_resume(WHOLE_SECOND.replace(microsecond=123456))

    def _assert_mid_tie_resume(self, instant: datetime) -> None:
        result = _run_async(self._mid_tie_resume(instant))
        anchor, tie_mate, resumed, expected_tail = (
            result["anchor"],
            result["tie_mate"],
            result["resumed"],
            result["expected_tail"],
        )
        assert anchor not in resumed, "the cursor's own row came back — a resumed walk would never advance"
        assert tie_mate in resumed, "the cursor's tie-mate was skipped — a resumed walk would lose rows"
        assert resumed == expected_tail

    async def _mid_tie_resume(self, instant: datetime) -> dict[str, Any]:
        coord = _seeded_store().coord
        namespace_id, seed_plan = await _fresh_tie_corpus(coord, instant)

        full = await coord.scan_documents_page(namespace_id, limit=20, scan_bound=50)
        assert [doc.id for doc in full] == seed_plan.expected

        anchor_doc = next(doc for doc in full if doc.id == seed_plan.tied_ids[0])
        cursor = DocumentCursor(created_at=anchor_doc.created_at, id=anchor_doc.id)
        resumed_page = await coord.scan_documents_page(
            namespace_id, limit=20, after=(cursor.created_at, cursor.id), scan_bound=50
        )
        return {
            "anchor": anchor_doc.id,
            "tie_mate": seed_plan.tied_ids[1],
            "resumed": [doc.id for doc in resumed_page],
            "expected_tail": seed_plan.expected[seed_plan.expected.index(anchor_doc.id) + 1 :],
        }

    def test_mid_tie_resume_at_limit_one_lands_on_the_exact_next_row(self) -> None:
        """A one-row page resumed from mid-tie returns exactly the next id.

        The sharpest form of the tie-break assertion: with ``limit=1`` there is
        nowhere for a wrong answer to hide behind a set comparison. The rows on
        either side of the anchor share its ``created_at`` to the microsecond, so
        this lands correctly only if the cursor's ``id`` half is compared as a
        UUID rather than as some rendering of one.
        """
        result = _run_async(self._next_row_after_mid_tie())
        assert result["got"] == [result["want"]]

    async def _next_row_after_mid_tie(self) -> dict[str, Any]:
        coord = _seeded_store().coord
        namespace_id, seed_plan = await _fresh_tie_corpus(coord, WHOLE_SECOND)

        full = await coord.scan_documents_page(namespace_id, limit=20, scan_bound=50)
        anchor_doc = next(doc for doc in full if doc.id == seed_plan.tied_ids[0])
        page = await coord.scan_documents_page(
            namespace_id, limit=1, after=(anchor_doc.created_at, anchor_doc.id), scan_bound=50
        )
        want = seed_plan.expected[seed_plan.expected.index(anchor_doc.id) + 1]
        return {"got": [doc.id for doc in page], "want": want}

    def test_typed_cursor_bind_excludes_its_own_row_where_a_naive_one_does_not(self) -> None:
        """The typed bind is what makes the cursor exclude its anchor. Contrast, not assertion.

        ``build_documents_scan_query`` binds each cursor operand through the
        column's own type (``sa.literal(value, DocumentModel.created_at.type)``,
        see ``storage/backends/_documents_scan.py``), which serializes
        ``created_at`` in the space-separated, six-digit-microsecond form this
        store actually holds. Hand-formatting the same instant with
        ``isoformat()`` produces a ``'T'`` separator, and ``'T'`` (0x54) sorts
        ABOVE ``' '`` (0x20) in the TEXT column's lexicographic comparison — so
        the row-value predicate matches the cursor's OWN row and everything above
        it.

        This runs both forms against the same seeded namespace: the typed one via
        the public ``scan_documents_page`` / ``DocumentCursor`` layer, the naive
        one as raw SQL of the identical shape with only the timestamp
        serialization changed. The failure mode is non-terminating, not
        wrong-answer — a walk chaining that cursor returns its anchor forever —
        so the assertion is that the two forms DISAGREE in exactly that
        direction.
        """
        result = _run_async(self._typed_vs_naive())

        assert result["anchor"] not in result["typed"], "the typed cursor did not exclude its own row"
        assert result["anchor"] in result["naive"], (
            "the naive isoformat bind did not return its own row — the contrast this "
            "test draws no longer holds, so the typed bind is no longer the thing keeping it out"
        )
        assert result["typed"] != result["naive"]
        # The naive bind does not merely include one extra row: it fails to
        # exclude anything at or above the anchor's whole second, so the tie
        # block comes back with it.
        assert set(result["typed"]) < set(result["naive"])

    async def _typed_vs_naive(self) -> dict[str, Any]:
        store = _seeded_store()
        coord = store.coord
        namespace_id, seed_plan = await _fresh_tie_corpus(coord, WHOLE_SECOND)

        full = await coord.scan_documents_page(namespace_id, limit=20, scan_bound=50)
        anchor = next(doc for doc in full if doc.id == seed_plan.tied_ids[0])

        typed_page = await coord.scan_documents_page(
            namespace_id, limit=20, after=(anchor.created_at, anchor.id), scan_bound=50
        )

        # The same statement shape the builder produces, with ONE thing changed:
        # the timestamp operand is hand-formatted instead of bound through the
        # column type. The id operand keeps the store's own encoding so the
        # contrast isolates the timestamp.
        cursor = await store.handle.sqlite.execute(
            "SELECT id FROM documents WHERE namespace_id = ? AND (created_at, id) < (?, ?) "
            "ORDER BY created_at DESC, id DESC",
            (
                uuid_to_text(namespace_id),
                as_utc(anchor.created_at).replace(tzinfo=None).isoformat(),
                uuid_to_text(anchor.id),
            ),
        )
        rows = await cursor.fetchall()
        return {
            "anchor": anchor.id,
            "typed": [doc.id for doc in typed_page],
            "naive": [UUID(row[0]) for row in rows],
        }

    def test_the_builder_binds_created_at_in_the_stored_serialization(self) -> None:
        """The rendered statement carries the space-separated form, not an ISO ``'T'``.

        The static half of the contrast above: compiling
        ``build_documents_scan_query`` with literal binds shows the operand this
        store compares against, without needing a row. Guards the one-line change
        (dropping the explicit column type from ``sa.literal``) that would put the
        wrong bytes in the predicate.
        """
        from sqlalchemy.dialects import sqlite

        anchor_id = uuid4()
        query = build_documents_scan_query(uuid4(), after=(WHOLE_SECOND.replace(tzinfo=None), anchor_id), scan_limit=5)
        rendered = str(query.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))

        assert "2026-01-31 12:30:00.000000" in rendered, rendered
        assert "2026-01-31T12:30:00" not in rendered, rendered
        # The id renders in this store's own 32-hex-undashed encoding, not the
        # dashed 36-character rendering that never matches the column.
        assert anchor_id.hex in rendered, rendered
        assert str(anchor_id) not in rendered, rendered


async def _fresh_tie_corpus(coord: StorageCoordinator, instant: datetime) -> tuple[UUID, ScanSeed]:
    """A brand-new namespace seeded with a tie-heavy six-row scan seed.

    Reuses the shared ``scan_seed`` plan (a four-row tie block flanked by two rows
    whose timestamp and id deliberately disagree) so the codec cases resume from
    the middle of a genuine tie rather than from a boundary any ordering would get
    right.
    """
    namespace_id = uuid4()
    await _create_namespace(coord, namespace_id)
    plan = scan_seed(6, instant=instant)
    for doc_id, created_at in plan.writes:
        await coord.create_document(
            Document(
                id=doc_id,
                namespace_id=namespace_id,
                content="codec row",
                checksum=f"codec-{doc_id.hex}",
                created_at=created_at,
                updated_at=created_at,
            )
        )
    return namespace_id, plan
