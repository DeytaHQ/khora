"""Property-based walk fuzzer for the document-enumeration surface (PostgreSQL leg).

The embedded twin (``tests/unit/filter/test_walk_fuzz.py``) carries the bulk of
the generated budget and the whole seed sweep, because it runs without services.
This module re-runs the same three properties — exactly-once/order/termination,
cursor stitching, limit invariance — against a live PostgreSQL, at a small budget,
because two things here are genuinely different and neither is checkable in
process:

* **The cursor round-trips through ``timestamptz`` and ``uuid``**, not through a
  lexicographically-compared TEXT column. The embedded store's whole class of
  serialization hazard (``'T'`` sorting above ``' '``, a missing ``.000000``
  sorting below its tie-mates) does not exist here — and this store has its own,
  opposite one, where a *wrong* operand type is silently correct rather than
  loudly broken. Neither store's behaviour predicts the other's.
* **The pushdown split is inverted.** This store's compiler pushes every
  enumerable key, INCLUDING both date keys, so a walk here is nearly all SQL
  where the embedded twin's is nearly all in-memory post-filter. The properties
  must hold identically over two very different physical plans, which is what
  running them twice buys.

What the differential does and does not prove is worth stating, because the
second bullet invites a stronger reading than it earns. It independently verifies
the *walk* — paging, cursor stitching, exactly-once, strictly-descending order,
honest termination — and it catches a too-RESTRICTIVE pushdown that drops a
keeper: such a row never enters the scanned window, so the post-filter cannot
recover it and the walk diverges from the oracle. It does NOT independently
verify filter or pushdown *semantics*. The coordinator's in-memory post-filter
and the oracle share the same ``compile_python`` predicate, re-checked over every
row on both sides, so a too-PERMISSIVE pushdown is masked by that shared re-check
and a bug inside ``compile_python`` itself is common-mode. Those are covered by
the recall filter fuzzer and the SQL-compiler superset checks, not here.

The eight-seed stability sweep is deliberately OMITTED here: it re-runs one
property under eight PRNG streams to rule out a lucky stream, which is a
statement about the *strategy*, not about the store, and the embedded leg already
makes it at a fraction of the round-trip cost.

Requires a running PostgreSQL (``make dev``). Skipped automatically when the
configured ``KHORA_DATABASE_URL`` is unreachable; the integration conftest turns
that skip into a hard failure when ``KHORA_PG_REQUIRED=1``, so a CI job with the
service down cannot pass by skipping.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from khora.core.models import Document, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentCursor, DocumentPage
from khora.db.session import run_migrations
from khora.storage.backends._documents_scan import build_documents_scan_query
from khora.storage.backends.postgresql import PostgreSQLBackend
from khora.storage.coordinator import StorageCoordinator
from tests.integration.matrix._conformance_lance import _run_async
from tests.test_helpers.document_order import id_ladder
from tests.test_helpers.document_scan import WHOLE_SECOND, ScanSeed, scan_seed
from tests.test_helpers.walk_fuzz import (
    CORPUS_SIZE,
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

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    # This repo's compose puts Postgres on 5434 (see compose.yaml); defaulting to
    # 5432 would make the whole class silently skip on a local `make test`.
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


pytestmark = [pytest.mark.integration]


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


skip_no_pg = pytest.mark.skipif(
    not _pg_reachable(),
    reason="PostgreSQL not reachable (run `make dev` first)",
)


# --------------------------------------------------------------------------- #
# The fixed corpus + the seeded PostgreSQL store.
# --------------------------------------------------------------------------- #
#
# Same shape as the embedded twin, and for the same reason: a module-level
# singleton resolved on the CALLER thread, with only the coroutines submitted to
# the dedicated loop thread. Hypothesis's ``function_scoped_fixture`` health check
# fires on a @given test that takes a per-test fixture, and a per-test connect /
# migrate / seed cycle would dominate the run anyway.
#
# Every namespace here is a fresh ``uuid4``, so the module is safe on the shared
# CI database: no test can see another's rows, and repeated runs never collide.

CORPUS_NAMESPACE_ID = uuid4()
CORPUS = build_walk_corpus(CORPUS_NAMESPACE_ID)

UNBOUNDED = CORPUS_SIZE + 1


def _budget(max_examples: int) -> settings:
    """The shared Hypothesis profile — small budgets, since each example is a live walk."""
    return settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )


class _SeededWalkStorePg:
    """A connected relational-only coordinator holding the frozen walk corpus."""

    def __init__(self, coord: StorageCoordinator, backend: PostgreSQLBackend) -> None:
        self.coord = coord
        self.backend = backend


async def _build_seeded_store() -> _SeededWalkStorePg:
    """Migrate, connect, and seed the corpus into a fresh namespace."""
    result = await run_migrations(DATABASE_URL)
    if not result.success:
        raise RuntimeError(f"migration failed: {result.error}")

    backend = PostgreSQLBackend(database_url=DATABASE_URL)
    await backend.connect()
    # ``vector=None``: enumeration is a relational-only read path, and it keeps
    # the mutation companion's ``delete_document`` scoped to the document row.
    coord = StorageCoordinator(relational=backend, vector=None)
    await coord.connect()

    await _create_namespace(coord, CORPUS_NAMESPACE_ID)
    for document in CORPUS.documents:
        await coord.create_document(document)

    # Materialization guard (fail loud, at build time): a seeder that silently
    # dropped rows would make the oracle and the walk agree on a too-small
    # corpus. Counted through the scan surface, so a broken one-page full read
    # is caught before any property depends on it.
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
    return _SeededWalkStorePg(coord, backend)


@lru_cache(maxsize=1)
def _seeded_store() -> _SeededWalkStorePg:
    """The process-wide seeded PostgreSQL store (migrated + seeded exactly once).

    Resolved on the CALLER thread, never from inside a coroutine already running
    on the loop thread — that would block the loop on a future only it can
    complete. Left open for the process lifetime, like the conformance helper's
    loop-thread coordinator.
    """
    return _run_async(_build_seeded_store())


async def _create_namespace(coord: StorageCoordinator, namespace_id: UUID) -> None:
    """Create a namespace whose row id and stable ``namespace_id`` agree."""
    await coord.create_namespace(
        MemoryNamespace(id=namespace_id, namespace_id=namespace_id, tenancy_mode=TenancyMode.SHARED)
    )


def _walk(*, limit: int, scan_bound: int, namespace_id: UUID | None = None, **kwargs: Any) -> list[DocumentPage]:
    """Drive a whole walk on the loop thread and return its pages.

    ``scan_bound`` is passed straight to ``scan_documents_page``; the facade's
    config-derived bound (``query.document_scan_overfetch_multiplier`` /
    ``document_scan_min_bound``) is deliberately bypassed, because no reachable
    config produces the tiny budgets that force a page boundary between almost
    every raw row.
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
    """Every page's shape invariants — the ``next_after``/``exhausted`` polarity."""
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
    """The concatenated ``(created_at, id)`` keys strictly descend across page seams."""
    keys = walked_keys(pages)
    for previous, current in zip(keys, keys[1:], strict=False):
        assert current < previous, f"walk order is not strictly descending: {previous!r} then {current!r}"


# --------------------------------------------------------------------------- #
# Property A — exactly-once + completeness + order + honest termination.
# --------------------------------------------------------------------------- #


@skip_no_pg
class TestWalkEnumerationPg:
    """One generated walk must enumerate exactly the oracle, once each, in order."""

    @given(
        filter_dict=walk_filter(),
        status=walk_status(),
        updated_before=walk_updated_before(),
        scan_bound=st.sampled_from([1, 2, 3, 5]),
    )
    @_budget(15)
    def test_walk_matches_the_oracle_exactly_once_and_in_order(
        self,
        filter_dict: dict[str, Any],
        status: str | None,
        updated_before: datetime | None,
        scan_bound: int,
    ) -> None:
        ast = validated_walk_ast(filter_dict)
        assume(ast is not None)
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


@skip_no_pg
class TestCursorStitchDifferentialPg:
    """A walk stitched from many ``timestamptz``/``uuid`` cursors equals one with none.

    The load-bearing property, and the one that justifies running any of this
    against a live server. The cursor here round-trips through real column types
    rather than through a text rendering, and the embedded twin's failure modes
    invert: a wrong-typed operand that the SQLite path rejects loudly is silently
    *correct* on asyncpg (its uuid parser skips dashes), while a naive
    ``datetime`` — harmless on the embedded store, whose column has no offset to
    lose — resolves here against whatever zone the process runs in. A
    ``scan_bound=1`` walk reassembles the whole answer from ~21 such round trips;
    the unbounded walk uses none and is the control.
    """

    @given(filter_dict=walk_filter(), status=walk_status(), updated_before=walk_updated_before())
    @_budget(15)
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


# --------------------------------------------------------------------------- #
# Property C — limit invariance.
# --------------------------------------------------------------------------- #


@skip_no_pg
class TestLimitInvariancePg:
    """``limit`` slices the answer into pages; it must not change the answer."""

    @given(filter_dict=walk_filter(), status=walk_status(), updated_before=walk_updated_before())
    @_budget(10)
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


@skip_no_pg
class TestWalkDeterminismPg:
    """The same walk, run twice over an unchanged namespace, must be the same walk.

    Worth running against a live server as well as in process: here each page is
    its own ``SELECT`` in its own session, so a repeated walk depends on the
    planner returning the same rows for the same keyset predicate — an ordering
    that is only total because of the ``id`` tie-break. A corpus with tie blocks
    and no tie-break would be free to reorder between runs and still satisfy every
    oracle comparison above.
    """

    def test_repeating_a_walk_reproduces_every_page_exactly(self) -> None:
        """Two identical walks agree on ids, page boundaries, cursors and flags."""
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


@skip_no_pg
class TestWalkDiscriminatesPg:
    """The corpus, the strategy and the budgets actually exercise what they claim.

    The embedded twin's ``>=50%`` multipage FRACTION guard is deliberately absent
    here: it assumes one RAW row scanned per budget unit, but this store pushes the
    filter into SQL, so ``scan_bound=1`` fetches one MATCHING row per page and the
    premise does not carry over. (The strict-subset fraction guard is absent from
    *both* legs — this strategy's true strict-subset rate hugs the ``30%`` floor,
    observed at 26% on a 100-draw sample, so a fraction over it is unsound wherever
    it runs.) Both properties are evidenced deterministically instead —
    boundary-crossing by ``test_unfiltered_walk_at_scan_bound_one_pages_the_whole_corpus``
    and discrimination by ``test_a_partial_filter_keeps_a_strict_subset`` — the same
    reasoning that keeps the eight-seed sweep on the embedded leg.
    """

    def test_a_partial_filter_keeps_a_strict_subset(self) -> None:
        """Explicit (non-Hypothesis) proof that the corpus discriminates on this store."""
        ast = to_walk_ast({"source_type": "library"})
        expected = walk_oracle(CORPUS.documents, filter_ast=ast)
        assert 0 < len(expected) < CORPUS_SIZE, expected
        assert walked_ids(_walk(limit=3, scan_bound=UNBOUNDED, filter_ast=ast)) == expected

    def test_unfiltered_walk_at_scan_bound_one_pages_the_whole_corpus(self) -> None:
        """A ``scan_bound=1`` walk takes exactly ``CORPUS_SIZE + 1`` pages, in pinned order.

        One raw row per page, plus the empty exhausted tail page that is the only
        sound termination signal.
        """
        pages = _walk(limit=3, scan_bound=1)

        assert len(pages) == CORPUS_SIZE + 1, [len(p) for p in pages]
        assert walked_ids(pages) == list(CORPUS.expected)
        assert [len(p) for p in pages[:-1]] == [1] * CORPUS_SIZE
        assert len(pages[-1]) == 0
        assert_page_contract(pages, limit=3)

    def test_this_store_pushes_every_enumerable_key(self) -> None:
        """The pushdown split is inverted here, and the difference is asserted, not assumed.

        This store's compiler backs — and pushes — all nine enumerable system keys
        plus every ``metadata`` path shape, so a page reports NO post-filtered
        keys and the raw window is already the answer. The embedded twin withholds
        both date keys (their TEXT format does not order against the compiler's
        binds), so the same filter there leaves scanned-but-rejected rows in the
        window.

        The consequence is worth naming rather than leaving implicit: the
        "matches split across pages with a rejected page between them" shape that
        the embedded leg pins is **not reachable on this store** through the
        enumerable key set, so that test lives there and only there. What this
        leg proves instead is that the identical properties hold over a plan
        where the post-filter narrows nothing at all.
        """
        for wire in (
            {"source_timestamp": {"$eq": "2026-06-01T00:00:00Z"}},
            {"created_at": {"$gte": "2026-01-01T00:00:00Z"}},
            {"metadata.a": {"b": "v"}},
            {"$or": [{"source_type": "library"}, {"metadata.tier": "gold"}]},
        ):
            page = _walk(limit=CORPUS_SIZE + 5, scan_bound=UNBOUNDED, filter_ast=to_walk_ast(wire))[0]
            assert page.post_filtered_keys == (), f"{wire!r} left post-filtered keys {page.post_filtered_keys}"


# --------------------------------------------------------------------------- #
# Mutation during a walk (deterministic — no snapshot spans pages).
# --------------------------------------------------------------------------- #


async def _seed_gap_ladder(coord: StorageCoordinator, namespace_id: UUID, ids: list[UUID], base: datetime) -> None:
    """Seed ``ids`` with strictly DESCENDING ``created_at``, ten seconds apart."""
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
    """Walk one row at a time, running ``mutations[i]`` after page ``i`` returns."""
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


async def _fresh_ladder(coord: StorageCoordinator, total: int) -> tuple[UUID, list[UUID], datetime]:
    """A brand-new namespace seeded with ``total`` gap-separated rows."""
    namespace_id = uuid4()
    await _create_namespace(coord, namespace_id)
    base = WHOLE_SECOND
    ids = id_ladder(total)
    await _seed_gap_ladder(coord, namespace_id, ids, base)
    return namespace_id, ids, base


@skip_no_pg
class TestMutationDuringWalkPg:
    """What a concurrent insert or delete does to a walk already in flight.

    ``scan_documents_page`` claims no consistent snapshot: each page is its own
    ``SELECT`` in its own session. On this store that is a statement about real
    concurrent transactions rather than about an embedded file, so the four exact
    answers below are worth pinning here as well as on the embedded twin — a
    future change to session or isolation handling would show up on this leg
    first.
    """

    def test_insert_above_the_cursor_is_not_seen(self) -> None:
        """A row newer than the cursor is already behind the walk — never served."""
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
        """A row older than the cursor is still ahead of the walk — served IN PLACE.

        The insert lands strictly between two existing rows, so it must appear at
        that exact position in the concatenation, not appended at the end.
        """
        result = _run_async(self._insert_below())
        ladder = result["ladder"]
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
        """A row deleted before the walk reaches it is never served, and leaves no hole."""
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
        """A row deleted AFTER it was served stays served, and the walk still advances.

        The keyset predicate is a comparison, not a lookup, so an anchor row that
        no longer exists is fine; a walk that re-read its anchor would break here.
        """
        result = _run_async(self._delete_behind())
        assert result["seen"] == result["ladder"]

    async def _delete_behind(self) -> dict[str, Any]:
        coord = _seeded_store().coord
        namespace_id, ladder, _base = await _fresh_ladder(coord, 5)

        async def mutate() -> None:
            await coord.delete_document(ladder[1], namespace_id=namespace_id)

        seen = await _walk_with_mutations(coord, namespace_id, {1: mutate})
        return {"seen": seen, "ladder": ladder}

    def test_all_four_mutations_in_one_walk(self) -> None:
        """The umbrella: every shape at once, in a single walk."""
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


# --------------------------------------------------------------------------- #
# Cursor codec — the DocumentCursor round trip on timestamptz + uuid.
# --------------------------------------------------------------------------- #


async def _fresh_tie_corpus(coord: StorageCoordinator, instant: datetime) -> tuple[UUID, ScanSeed]:
    """A brand-new namespace seeded with a tie-heavy six-row scan seed."""
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


@skip_no_pg
class TestCursorCodecPg:
    """A ``DocumentCursor`` fed back in resumes at the exact next row, on real column types.

    Both microsecond polarities are seeded here too, even though neither is the
    hazard on this store — ``created_at`` is a ``timestamptz``, not a
    lexicographically-compared TEXT column, so ``.000000`` and ``.123456`` are the
    same case. That is precisely why they are worth running: the assertion that
    the polarity does NOT matter here is what makes the embedded twin's opposite
    finding a store property rather than an unexplained difference.

    One asymmetry is deliberately NOT asserted, because on this store there is
    nothing to assert: a ``str`` where the cursor's ``uuid`` belongs is *silently
    accepted*. ``sqlalchemy.Uuid.bind_processor`` returns ``None`` whenever the
    dialect supports native UUIDs, so no ``.hex`` lookup runs, and asyncpg's own
    parser skips ``-`` outright — a dashed cursor decodes to the identical
    sixteen bytes and works. The same input is loud on the embedded store and
    harmless here, which is exactly why "build a cursor from a row this store
    returned, never by hand" is the rule that generalizes rather than any
    per-store type check.
    """

    def test_whole_second_tie_block_resume(self) -> None:
        """microsecond=0: resume from mid-tie excludes the anchor, keeps its tie-mate."""
        self._assert_mid_tie_resume(WHOLE_SECOND)

    def test_sub_second_tie_block_resume(self) -> None:
        """microsecond=.123456: identical outcome — the polarity is inert on timestamptz."""
        self._assert_mid_tie_resume(WHOLE_SECOND.replace(microsecond=123456))

    def _assert_mid_tie_resume(self, instant: datetime) -> None:
        result = _run_async(self._mid_tie_resume(instant))
        assert result["anchor"] not in result["resumed"], (
            "the cursor's own row came back — a resumed walk would never advance"
        )
        assert result["tie_mate"] in result["resumed"], (
            "the cursor's tie-mate was skipped — a resumed walk would lose rows"
        )
        assert result["resumed"] == result["expected_tail"]

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

        With ``limit=1`` there is nowhere for a wrong answer to hide behind a set
        comparison, and the rows on either side of the anchor share its
        ``created_at`` to the microsecond — so this lands correctly only if the
        cursor's ``id`` half is compared as a ``uuid``.
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

    def test_a_cursor_carried_into_another_zone_resumes_identically(self) -> None:
        """Converting the cursor's zone is a no-op here, and STRIPPING it is not portable.

        This is the contrast the embedded twin cannot draw, and the two stores
        make opposite adjustments look harmless. ``timestamptz`` holds an
        *instant*, so re-expressing the cursor in another zone selects the same
        row — asserted here. The embedded store holds wall clock with the
        writer's offset already discarded, so there the conversion MOVES the
        position while attaching or stripping ``tzinfo`` is the no-op. Neither
        rule generalizes; the one that holds on both is narrower than either:
        bind the value the store returned, unmodified.
        """
        result = _run_async(self._zone_shifted_resume())
        assert result["utc_created_at"].tzinfo is not None, "a timestamptz cursor must come back aware"
        assert result["shifted"] == result["utc"]

    async def _zone_shifted_resume(self) -> dict[str, Any]:
        from datetime import timezone

        coord = _seeded_store().coord
        namespace_id, seed_plan = await _fresh_tie_corpus(coord, WHOLE_SECOND)

        full = await coord.scan_documents_page(namespace_id, limit=20, scan_bound=50)
        anchor = next(doc for doc in full if doc.id == seed_plan.tied_ids[0])

        utc_page = await coord.scan_documents_page(
            namespace_id, limit=20, after=(anchor.created_at, anchor.id), scan_bound=50
        )
        shifted_at = anchor.created_at.astimezone(timezone(timedelta(hours=5, minutes=30)))
        shifted_page = await coord.scan_documents_page(
            namespace_id, limit=20, after=(shifted_at, anchor.id), scan_bound=50
        )
        return {
            "utc_created_at": anchor.created_at,
            "utc": [doc.id for doc in utc_page],
            "shifted": [doc.id for doc in shifted_page],
        }

    def test_the_builder_casts_both_cursor_operands_to_their_column_types(self) -> None:
        """The rendered statement casts the cursor operands, whatever their Python type.

        The static half: compiling ``build_documents_scan_query`` against the
        asyncpg dialect shows ``$n::TIMESTAMP WITH TIME ZONE`` and ``$n::UUID``
        because ``sa.literal`` names the column's type (see
        ``storage/backends/_documents_scan.py``). Drop that argument and a
        one-step-off operand renders as its own inferred type instead — a ``date``
        as ``::DATE``, a ``str`` id as ``::VARCHAR`` — which is how a cursor
        silently resolves against the wrong instant. Needs no server, so it fails
        fast and locally when the bind spelling regresses.
        """
        from sqlalchemy.dialects import postgresql

        rendered = str(
            build_documents_scan_query(uuid4(), after=(WHOLE_SECOND.date(), str(uuid4())), scan_limit=5).compile(
                dialect=postgresql.asyncpg.dialect()
            )
        )

        assert "::TIMESTAMP WITH TIME ZONE" in rendered, rendered
        assert "::DATE" not in rendered, rendered
        assert "::VARCHAR" not in rendered, rendered
        # The namespace scope and the cursor's id — the two UUID-typed operands.
        assert rendered.count("::UUID") == 2, rendered
