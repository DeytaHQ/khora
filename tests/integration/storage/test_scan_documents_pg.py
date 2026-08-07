"""``PostgreSQLBackend.scan_documents`` — the bounded keyset scan.

The shared half of this scan is proved by the embedded sibling
(``tests/unit/storage/backends/sqlite_lance/test_relational_scan_documents.py``):
both stores build the same statement through ``build_documents_scan_query``, and
the embedded lane runs without services, so it carries the cursor semantics.
What only a live server can show is here — that the cursor round-trips through
``timestamptz`` / ``uuid`` rather than through a text serialization, and that
this store's compiler reports a *different*, and correct, pushdown split for the
same filter.

Requires a running PostgreSQL (``make dev``). Skipped automatically when the
configured ``KHORA_DATABASE_URL`` is unreachable; the integration conftest turns
that skip into a hard failure when ``KHORA_PG_REQUIRED=1``, so a CI job with the
service down cannot pass by skipping.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from khora.core.models import MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.db.session import run_migrations
from khora.filter import CompilerRegistry
from khora.filter.execute import filter_leaf_keys

# ``_documents_compile_context`` is private, and imported on purpose: the
# split-equality test below must recompile with the *same* context the scan uses,
# or it would prove a property of some other context.
from khora.storage.backends.postgresql import PostgreSQLBackend, _documents_compile_context
from khora.storage.coordinator import StorageCoordinator
from tests.test_helpers.document_page_oracle import (
    assert_page_compliant,
    assert_walk_compliant,
    assert_walk_matches_expected,
)
from tests.test_helpers.document_scan import (
    WHOLE_SECOND,
    scan_seed,
    seed_documents,
    seed_varied,
    to_filter_ast,
    walk_scan,
    write_document,
)
from tests.test_helpers.document_scan_spy import (
    PROBE_VALUE,
    assert_split_honest,
    force_residual,
    sqlalchemy_sql_log,
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


@pytest.fixture(scope="module")
async def _run_migrations_once():
    result = await run_migrations(DATABASE_URL)
    assert result.success, f"Migrations failed: {result.error}"


@pytest.fixture
async def backend(_run_migrations_once):
    be = PostgreSQLBackend(database_url=DATABASE_URL)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


@pytest.fixture
async def namespace(backend: PostgreSQLBackend):
    """A fresh namespace per test, so a scan never sees another test's rows."""
    return await backend.create_namespace(MemoryNamespace())


@skip_no_pg
class TestScanDocumentsWindowPg:
    async def test_scan_limit_bounds_the_window(self, backend: PostgreSQLBackend, namespace) -> None:
        seed = scan_seed(6)
        await seed_documents(backend, namespace.id, seed)

        step = await backend.scan_documents(namespace.id, scan_limit=2)

        assert [d.id for d in step.documents] == seed.expected[:2]
        assert step.last_scanned == (step.documents[-1].created_at, step.documents[-1].id)
        assert step.exhausted is False

    async def test_a_full_window_is_not_yet_exhausted(self, backend: PostgreSQLBackend, namespace) -> None:
        """``exhausted`` means SQL ran short, not "the caller has seen everything".

        A window filled exactly to the bound cannot distinguish "six rows and no
        more" from "six rows and a seventh waiting", so it must report
        not-exhausted and let the next step find the empty tail. Reporting
        exhaustion here would silently truncate every namespace whose size is a
        multiple of the bound.
        """
        seed = scan_seed(6)
        await seed_documents(backend, namespace.id, seed)

        exact = await backend.scan_documents(namespace.id, scan_limit=6)
        assert len(exact.documents) == 6
        assert exact.exhausted is False

        over = await backend.scan_documents(namespace.id, scan_limit=7)
        assert len(over.documents) == 6
        assert over.exhausted is True

    async def test_scan_limit_below_one_is_rejected(self, backend: PostgreSQLBackend, namespace) -> None:
        """A zero bound would return an empty window that reports neither a resume
        position nor exhaustion — the one pair a walking caller cannot act on."""
        with pytest.raises(ValueError, match="scan_limit"):
            await backend.scan_documents(namespace.id, scan_limit=0)

    async def test_empty_window_reports_exhausted_without_a_position(
        self, backend: PostgreSQLBackend, namespace
    ) -> None:
        """Both the never-seeded namespace and the tail past the last row."""
        empty = await backend.scan_documents(namespace.id, scan_limit=5)
        assert empty.documents == []
        assert empty.last_scanned is None
        assert empty.exhausted is True

        seed = scan_seed(6)
        await seed_documents(backend, namespace.id, seed)
        full = await backend.scan_documents(namespace.id, scan_limit=10)
        oldest = full.documents[-1]

        tail = await backend.scan_documents(namespace.id, after=(oldest.created_at, oldest.id), scan_limit=5)
        assert tail.documents == []
        assert tail.last_scanned is None
        assert tail.exhausted is True

    async def test_status_and_updated_before_narrow_the_window(self, backend: PostgreSQLBackend, namespace) -> None:
        seed = scan_seed(6)
        cutoff = seed.tie_instant + timedelta(hours=1)
        for i, (doc_id, created_at) in enumerate(seed.writes):
            await write_document(
                backend,
                namespace.id,
                doc_id,
                created_at,
                status=DocumentStatus.COMPLETED if i % 2 == 0 else DocumentStatus.PENDING,
                updated_at=cutoff - timedelta(minutes=1) if i < 4 else cutoff + timedelta(minutes=1),
            )

        by_status = await backend.scan_documents(namespace.id, status=DocumentStatus.COMPLETED.value, scan_limit=10)
        assert {d.id for d in by_status.documents} == {
            doc_id for i, (doc_id, _) in enumerate(seed.writes) if i % 2 == 0
        }

        by_updated = await backend.scan_documents(namespace.id, updated_before=cutoff, scan_limit=10)
        assert {d.id for d in by_updated.documents} == {doc_id for i, (doc_id, _) in enumerate(seed.writes) if i < 4}


@skip_no_pg
class TestScanDocumentsCursorPg:
    async def test_walk_visits_every_document_exactly_once_in_total_order(
        self, backend: PostgreSQLBackend, namespace
    ) -> None:
        """One row per step across a tie block, chaining ``last_scanned``.

        ``scan_limit=1`` puts a cursor boundary between every pair of rows,
        including between rows that share a ``created_at`` to the microsecond —
        so every resume in this walk is a mid-tie resume, and the ``id DESC``
        leg decides all of them.
        """
        seed = scan_seed(6)
        await seed_documents(backend, namespace.id, seed)

        steps = await walk_scan(backend.scan_documents, namespace.id, scan_limit=1)
        seen = [d.id for step in steps for d in step.documents]

        assert len(seen) == len(set(seen))  # no document served twice
        assert set(seen) == set(seed.expected)  # every document served
        assert seen == seed.expected  # and in one total order across the concatenation
        assert steps[-1].documents == []
        assert steps[-1].last_scanned is None
        assert steps[-1].exhausted is True

    async def test_mid_tie_cursor_resumes_at_the_exact_next_row(self, backend: PostgreSQLBackend, namespace) -> None:
        """A position taken from the middle of a tie block, in one step.

        The rows on either side of the cursor share its ``created_at`` exactly,
        so this only lands on the right row if the ``id`` half of the position is
        compared as a ``uuid`` rather than as some rendering of one.
        """
        seed = scan_seed(6)
        await seed_documents(backend, namespace.id, seed)

        full = await backend.scan_documents(namespace.id, scan_limit=10)
        assert [d.id for d in full.documents] == seed.expected

        cursor_doc = next(d for d in full.documents if d.id == seed.tied_ids[0])
        step = await backend.scan_documents(namespace.id, after=(cursor_doc.created_at, cursor_doc.id), scan_limit=10)
        ids = [d.id for d in step.documents]

        assert cursor_doc.id not in ids, "the cursor's own row came back — a resumed walk would never advance"
        assert seed.tied_ids[1] in ids, "the cursor's tie-mate was skipped — a resumed walk would lose rows"
        assert ids == seed.expected[seed.expected.index(cursor_doc.id) + 1 :]

    async def test_filtered_walk_puts_a_cursor_and_a_compiled_fragment_in_one_statement(
        self, backend: PostgreSQLBackend, namespace
    ) -> None:
        """A cursor and a pushdown fragment in the same ``SELECT``, walked to the end.

        Every other test here passes ``after`` or ``filter_ast``, never both.
        This store's compiler emits a SQLAlchemy expression, so its operands are
        named by the same machinery that names the keyset's — there is no
        hand-written bind splice to collide, unlike the embedded store. What is
        worth proving is the rest of it: that the fragment's ``literal_column``
        references add no second FROM entry, and that a filtered walk still
        enumerates exactly once, in order, and terminates.
        """
        seed = scan_seed(6)
        await seed_varied(backend, namespace.id, seed)

        full = await backend.scan_documents(namespace.id, scan_limit=10)
        wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
        expected = [d.id for d in full.documents if d.source_type == "report" or d.title == "doc-1"]
        assert 1 < len(expected) < len(full.documents), "the filter must narrow, but not to a single row"

        steps = await walk_scan(
            backend.scan_documents,
            namespace.id,
            scan_limit=1,
            filter_ast=to_filter_ast(wire),
        )
        seen = [d.id for step in steps for d in step.documents]

        assert len(seen) == len(set(seen))
        assert seen == expected
        assert steps[-1].exhausted is True
        # Both leaves are real columns here, so this store pushes the whole
        # disjunction — the embedded sibling reports the same filter differently
        # only when a leaf is one it cannot back.
        assert steps[0].consumed_keys == frozenset({"source_type", "title"})

    async def test_cursor_read_off_a_row_is_timezone_aware(self, backend: PostgreSQLBackend, namespace) -> None:
        """This store's position is an aware instant, and the round trip is exact.

        ``created_at`` is ``timestamptz`` here, so a position read off a row
        carries a ``tzinfo`` and can be compared to the seeded instant directly —
        unlike the embedded store, whose ``DATETIME`` discards the writer's
        offset and reads back naive. The two shapes are why a position is
        store-local and must never be carried across stores.
        """
        seed = scan_seed(6)
        await seed_documents(backend, namespace.id, seed)

        step = await backend.scan_documents(namespace.id, scan_limit=10)
        # Guard before the unpack: ``last_scanned`` is ``DocumentScanKey | None``, so
        # a regression that returned no position would raise ``TypeError`` from the
        # tuple unpack rather than failing this test's actual subject.
        assert step.last_scanned is not None
        cursor_created_at, _ = step.last_scanned

        assert cursor_created_at.tzinfo is not None
        assert cursor_created_at == seed.tie_instant - timedelta(seconds=1)
        assert cursor_created_at.microsecond == 0


@skip_no_pg
class TestScanDocumentsSplitPg:
    async def test_date_system_keys_are_pushed_down_by_this_store(self, backend: PostgreSQLBackend, namespace) -> None:
        """``created_at`` pushes here, and the same filter does NOT push on the
        embedded store — the asymmetry is intended, not drift.

        This store keeps ``created_at`` in a real ``timestamptz``, so the
        compiler's bind orders against the stored value and the leaf is both
        consumed and enforced in SQL. The embedded store withholds the key
        because its TEXT format does not order against the same bind; its
        sibling test asserts the opposite answer for this same filter.
        """
        seed = scan_seed(6)
        await seed_documents(backend, namespace.id, seed)

        step = await backend.scan_documents(
            namespace.id,
            filter_ast=to_filter_ast({"created_at": {"$gte": "2999-01-01T00:00:00+00:00"}}),
            scan_limit=10,
        )

        assert step.consumed_keys == frozenset({"created_at"})
        # Consumed AND enforced: every seeded row is older than the bound.
        assert step.documents == []

    async def test_split_reports_only_the_leaves_sql_enforced(self, backend: PostgreSQLBackend, namespace) -> None:
        """A mixed filter: one pushable leaf, one this table cannot back.

        ``occurred_at`` is a recall-chunk key with no ``documents`` column, so it
        compiles to a match-all placeholder, stays out of ``consumed_keys``, and
        reaches the caller's post-filter.
        """
        seed = scan_seed(6)
        for i, (doc_id, created_at) in enumerate(seed.writes):
            await write_document(
                backend, namespace.id, doc_id, created_at, source_type="report" if i % 2 == 0 else "library"
            )

        step = await backend.scan_documents(
            namespace.id,
            filter_ast=to_filter_ast(
                {"source_type": {"$eq": "report"}, "occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}}
            ),
            scan_limit=10,
        )

        assert step.consumed_keys == frozenset({"source_type"})
        # The pushed leaf really did narrow the window; the unpushed one did not.
        assert {d.source_type for d in step.documents} == {"report"}
        assert len(step.documents) == 3

    async def test_last_scanned_is_the_final_raw_row_not_the_last_match(
        self, backend: PostgreSQLBackend, namespace
    ) -> None:
        """Resume from the last row SCANNED, not from the last row that matches.

        The window here deliberately ENDS on a row the caller's post-filter will
        reject: the filter is an ``$or`` mixing a pushable leaf with one this
        table cannot back, which the compiler defers as a whole rather than
        pushing half of a disjunction, so SQL narrows nothing and the oldest row
        — which does not satisfy the filter — is the last row of the raw window.

        A walk that resumed from the last *matching* row instead would re-scan
        the rejected gap on every step — and when a whole window is rejected
        there is no matching row to resume from at all, so such a walk cannot
        advance past a run of non-matching rows longer than one window. Taking
        the position from the raw window is what lets ``exhausted`` be the only
        termination signal.
        """
        newest, middle, oldest = (uuid4() for _ in range(3))
        base = datetime(2026, 1, 31, 12, 30, tzinfo=UTC)
        await write_document(backend, namespace.id, newest, base + timedelta(seconds=2), source_type="report")
        await write_document(backend, namespace.id, middle, base + timedelta(seconds=1), source_type="report")
        await write_document(backend, namespace.id, oldest, base, source_type="library")

        step = await backend.scan_documents(
            namespace.id,
            filter_ast=to_filter_ast(
                {
                    "$or": [
                        {"source_type": {"$eq": "report"}},
                        {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
                    ]
                }
            ),
            scan_limit=10,
        )

        # Nothing was pushed, so the raw window is the whole namespace and its
        # last row is the one the post-filter will drop.
        assert step.consumed_keys == frozenset()
        assert [d.id for d in step.documents] == [newest, middle, oldest]
        assert step.documents[-1].source_type == "library"

        last_row = step.documents[-1]
        assert step.last_scanned == (last_row.created_at, last_row.id)

        last_match = step.documents[1]
        assert step.last_scanned != (last_match.created_at, last_match.id)


@skip_no_pg
class TestScanDocumentsSplitHonestyPg:
    """Is the reported split HONEST? — checked at the driver boundary.

    Every other class here reads ``consumed_keys`` and believes it, which cannot
    distinguish an honest report from a *perf lie* (leaf reported as pushed, SQL
    never saw it, the caller's post-filter quietly covering) or a *correctness lie*
    (leaf reported as pushed and therefore, for any caller that trusted the report,
    enforced nowhere). Neither is visible on the result surface while the
    coordinator re-checks the full AST — hence a seam-level check.

    The spy listens on ``before_cursor_execute`` on this backend's
    ``AsyncEngine``, so the params recorded are the ones asyncpg receives.
    Assertions are on bound VALUES, never on column names: ``created_at``, ``id``,
    ``namespace_id`` and ``status`` appear in every statement's ORDER BY / keyset /
    scope whether or not a filter leaf pushed. See
    :mod:`tests.test_helpers.document_scan_spy`.
    """

    async def _seed_probe_corpus(self, backend: PostgreSQLBackend, namespace_id) -> int:
        """Seed the corpus with the probe literal on half the rows; return that count."""
        seed = scan_seed(6)
        for i, (doc_id, created_at) in enumerate(seed.writes):
            await write_document(
                backend,
                namespace_id,
                doc_id,
                created_at,
                source_type=PROBE_VALUE if i % 2 == 0 else "library",
            )
        return sum(1 for i in range(len(seed.writes)) if i % 2 == 0)

    async def test_a_pushed_leaf_really_reaches_the_statement(self, backend: PostgreSQLBackend, namespace) -> None:
        """The perf-lie catch: a leaf reported as pushed must be bound in the SQL."""
        matching = await self._seed_probe_corpus(backend, namespace.id)
        ast = to_filter_ast({"source_type": {"$eq": PROBE_VALUE}})

        with sqlalchemy_sql_log(backend._engine) as sql_log:  # noqa: SLF001 — no public driver-seam accessor
            step = await backend.scan_documents(namespace.id, filter_ast=ast, scan_limit=10)

        assert step.consumed_keys == frozenset({"source_type"})
        assert_split_honest(sql_log, pushed_values=[PROBE_VALUE])
        # And the pushdown narrowed for real, rather than leaving the work owed.
        assert len(step.documents) == matching
        assert {d.source_type for d in step.documents} == {PROBE_VALUE}

    async def test_a_naturally_deferred_leaf_never_reaches_the_statement(
        self, backend: PostgreSQLBackend, namespace
    ) -> None:
        """The correctness-lie catch WITHOUT monkeypatching anything.

        The stronger of the two residual tests, and the reason it exists separately:
        it needs no ``force_residual``, so it tests the store as it ships. The
        unpushable half is ``occurred_at`` (no ``documents`` column backs it) rather
        than ``created_at``, which is a real ``timestamptz`` here and pushes — and
        ``compile_postgres``'s all-or-nothing gate defers the whole ``$or`` rather
        than pushing half a disjunction. So ``source_type``, which this store
        normally pushes, must NOT be bound.

        ``occurred_at`` reaches the compiler at all only because this is the storage
        tier: the facade's ``_reject_non_enumerable_keys`` refuses it one level up.
        The gate is what this store's ``_documents_compile_context`` docstring names
        as the reason the pushdown is superset-safe — a match-all placeholder left
        inside a negation inverts into a match-nothing and wrongly EXCLUDES rows.
        """
        await self._seed_probe_corpus(backend, namespace.id)
        ast = to_filter_ast(
            {"$or": [{"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}}, {"source_type": {"$eq": PROBE_VALUE}}]}
        )

        with sqlalchemy_sql_log(backend._engine) as sql_log:  # noqa: SLF001
            step = await backend.scan_documents(namespace.id, filter_ast=ast, scan_limit=10)

        assert step.consumed_keys == frozenset()
        assert_split_honest(sql_log, residual_values=[PROBE_VALUE])
        # The whole disjunction deferred, so SQL narrowed nothing: the raw window is
        # the entire corpus and the post-filter owns every leaf.
        assert len(step.documents) == 6

    async def test_a_residual_leaf_never_reaches_the_statement(
        self, backend: PostgreSQLBackend, namespace, monkeypatch
    ) -> None:
        """The correctness-lie catch, same filter with the leaf forced residual.

        ``force_residual`` drops ``source_type`` from this store's
        ``field_mapping``, which ``compile_postgres`` reads as the pushdown
        whitelist, so the identical AST defers. Both halves must hold together: the
        report says not-consumed, and the operand is absent from the statement.

        This is also the only *uniform* forced-residual lever across the four
        stores. A ``SchemaCapabilities`` override would not work here — the
        postgres documents compiler never reads ``schema_capabilities`` at all.
        """
        await self._seed_probe_corpus(backend, namespace.id)
        from khora.storage.backends import postgresql as pg_module

        force_residual(monkeypatch, pg_module)
        ast = to_filter_ast({"source_type": {"$eq": PROBE_VALUE}})

        with sqlalchemy_sql_log(backend._engine) as sql_log:  # noqa: SLF001
            step = await backend.scan_documents(namespace.id, filter_ast=ast, scan_limit=10)

        assert step.consumed_keys == frozenset()
        assert_split_honest(sql_log, residual_values=[PROBE_VALUE])
        # Deferred means "narrowed nothing here": the raw window is the whole corpus.
        assert len(step.documents) == 6

    # The split this store reports, per AST shape. PINNED as well as recomputed:
    # the recompute alone is satisfiable by two compilers agreeing on the wrong
    # answer, including the degenerate "consumes nothing ever" that would make
    # every comparison ``frozenset() == frozenset()``. The date shapes are where
    # this store differs from the two SQLite tiers — ``created_at`` is a real
    # ``timestamptz`` here so it PUSHES, and ``occurred_at`` is the unbacked key.
    _SPLIT_SHAPES: dict[str, tuple[dict, frozenset[str]]] = {
        "pushed_whole": ({"source_type": {"$eq": PROBE_VALUE}}, frozenset({"source_type"})),
        "date_key_pushes_here": (
            {"source_type": {"$eq": PROBE_VALUE}, "created_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
            frozenset({"created_at", "source_type"}),
        ),
        "unbacked_key_is_the_residual": (
            {"source_type": {"$eq": PROBE_VALUE}, "occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
            frozenset({"source_type"}),
        ),
        "or_wholly_pushable": (
            {"$or": [{"source_type": {"$eq": PROBE_VALUE}}, {"title": {"$eq": "doc-1"}}]},
            frozenset({"source_type", "title"}),
        ),
        "or_wholly_deferred": (
            {"$or": [{"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}}, {"source_type": {"$eq": PROBE_VALUE}}]},
            frozenset(),
        ),
        "metadata_path": ({"metadata.tier": {"$eq": "gold"}}, frozenset({"metadata.tier"})),
    }

    @pytest.mark.parametrize(("wire", "expected"), _SPLIT_SHAPES.values(), ids=_SPLIT_SHAPES.keys())
    async def test_reported_split_equals_a_fresh_compile_of_the_same_ast(
        self, backend: PostgreSQLBackend, namespace, wire, expected
    ) -> None:
        """``consumed_keys`` is the compiler's answer, verbatim — no store-side edits.

        The scan may compile once and report; what it must not do is add or drop
        keys on the way out. A store that unioned in the keys it *hoped* were
        pushed, or stripped one it found inconvenient, still looks plausible on
        every row assertion here, because the caller re-checks the full AST
        regardless. Recomputed with the SAME compiler and context the scan uses.

        This is also how ``created_at`` pushdown is covered: its *value* is out of
        the spy's scope (asyncpg binds a ``datetime`` object where the SQLite tiers
        bind their own string serializations, so a value probe is a dialect quiz),
        but its membership in the split is exactly what the second shape compares —
        and it is the answer the embedded siblings give the other way round.
        """
        await self._seed_probe_corpus(backend, namespace.id)
        ast = to_filter_ast(wire)

        step = await backend.scan_documents(namespace.id, filter_ast=ast, scan_limit=10)

        assert step.consumed_keys == expected
        compiler = CompilerRegistry.get("relational.postgresql", "documents")
        assert step.consumed_keys == compiler(ast, _documents_compile_context()).consumed_keys


@skip_no_pg
class TestScanDocumentsPageSplitPg:
    """The page-level split: ``DocumentPage.post_filtered_keys`` over a live server.

    The coordinator derives it as ``sorted(filter_leaf_keys(ast) - consumed_keys)``
    unioned across the steps of a page, and this store's answer is the *opposite* of
    the embedded tiers' for the shape that matters: ``created_at`` pushes here, so
    it is NOT a residual, while ``occurred_at`` — which no ``documents`` column
    backs — is. The embedded sibling
    (``tests/unit/test_list_documents_enumeration.py``) asserts the mirror image on
    the same helper, which is what makes this leg worth its own run rather than a
    transcription.

    ``occurred_at`` reaches the compiler at all only because this is the storage
    tier: the facade's ``_reject_non_enumerable_keys`` refuses it one level up.
    """

    @pytest.fixture
    async def seeded(self, backend: PostgreSQLBackend, namespace):
        """A coordinator over a varied six-row namespace.

        Returns ``(coordinator, ns_id, rows)``. ``rows`` is the seed as plain dicts
        — the INDEPENDENT record of what was written, which the completeness
        matchers evaluate against. Nothing here reads it back out of the store.
        """
        seed = scan_seed(6)
        rows: list[dict] = []
        for i, (doc_id, created_at) in enumerate(seed.writes):
            fields = {"title": f"doc-{i}", "source_type": "report" if i % 2 == 0 else "library"}
            await write_document(backend, namespace.id, doc_id, created_at, **fields)
            rows.append({"id": doc_id, "created_at": created_at, **fields})
        return StorageCoordinator(relational=backend), namespace.id, rows

    # Wire filter, the residual the coordinator must REPORT, and an independent
    # plain-Python matcher over the seeded rows giving the id set it must RETURN.
    # The matcher is hand-written on purpose: deriving it with ``compile_python``
    # would reproduce the predicate the coordinator itself applies, so the
    # comparison could not fail. The residual is pinned as well as recomputed —
    # a recompute alone is satisfied by two computations agreeing on the wrong
    # answer, including the all-empty case that makes every comparison ``() == ()``.
    #
    # The date bound is ``WHOLE_SECOND`` so it BITES (the seed puts one row a second
    # below the tie instant, so ``$gte`` there keeps 5 of 6); a bound every row
    # satisfies would reduce completeness to "returns everything".
    _PAGE_SPLIT_SHAPES: dict[str, tuple[dict, tuple[str, ...], object]] = {
        "no_residual": (
            {"source_type": {"$eq": "report"}},
            (),
            lambda r: r["source_type"] == "report",
        ),
        "date_key_pushes_here": (
            {"created_at": {"$gte": WHOLE_SECOND.isoformat()}},
            (),
            lambda r: r["created_at"] >= WHOLE_SECOND,
        ),
        "unbacked_key_residual": (
            {"source_type": {"$eq": "report"}, "occurred_at": {"$gte": "2020-01-01T00:00:00+00:00"}},
            ("occurred_at",),
            # ``occurred_at`` is absent on every documents row, so a range compare on
            # it is false for all — but the compiler DEFERS it, leaving the
            # coordinator's post-filter to reject everything. Expect ZERO matches,
            # asserted below as a deliberate exception to the narrowing precondition.
            lambda _r: False,
        ),
        "or_wholly_deferred": (
            {"$or": [{"occurred_at": {"$gte": "2020-01-01T00:00:00+00:00"}}, {"source_type": {"$eq": "report"}}]},
            ("occurred_at", "source_type"),
            lambda r: r["source_type"] == "report",
        ),
    }

    @pytest.mark.parametrize(
        ("wire", "expected", "matcher"), _PAGE_SPLIT_SHAPES.values(), ids=_PAGE_SPLIT_SHAPES.keys()
    )
    async def test_page_reports_its_residual_and_returns_exactly_the_matches(
        self, seeded, wire, expected, matcher
    ) -> None:
        coordinator, namespace_id, rows = seeded
        ast = to_filter_ast(wire)
        expected_ids = [r["id"] for r in rows if matcher(r)]
        assert len(expected_ids) < len(rows), "the shape must narrow, or completeness proves nothing"

        page = await coordinator.scan_documents_page(namespace_id, filter_ast=ast, limit=100)

        # 1. The reported residual, pinned and recomputed against the compiler.
        assert page.post_filtered_keys == expected
        compiler = CompilerRegistry.get("relational.postgresql", "documents")
        consumed = compiler(ast, _documents_compile_context()).consumed_keys
        assert page.post_filtered_keys == tuple(sorted(filter_leaf_keys(ast) - consumed))
        # 2. COMPLETENESS — the assertion that can actually fail. Against the
        #    hand-written matcher over the seed, never against another enumeration
        #    call and never against ``compile_python`` (the coordinator's own).
        assert_walk_matches_expected(page, expected_ids)
        # 3. Order + the walk-control pair. Soundness rides along and is
        #    tautological here by construction; see the helper's docstring.
        assert_page_compliant(page, ast)

    async def test_a_multi_page_walk_loses_no_match_across_cursor_boundaries(self, seeded) -> None:
        """Completeness across a REAL walk, boundaries landing inside a tie block.

        The single-page cases above cannot see a cursor bug: one page never resumes.
        ``limit=2`` over a 5-row match set forces three pages whose boundaries fall
        inside the seed's ``created_at`` tie block — where a cursor that mis-binds
        its ``timestamptz`` or its ``uuid`` skips a tie-mate. That row simply never
        appears, so no per-returned-row assertion can notice it; only the comparison
        against the independently-known match set does.
        """
        coordinator, namespace_id, rows = seeded
        ast = to_filter_ast({"created_at": {"$gte": WHOLE_SECOND.isoformat()}})
        expected_ids = [r["id"] for r in rows if r["created_at"] >= WHOLE_SECOND]
        assert len(expected_ids) == 5, "5 of 6 rows sit at or above the tie instant"

        walked = []
        after = None
        while True:
            page = await coordinator.scan_documents_page(namespace_id, filter_ast=ast, limit=2, after=after)
            walked.append(page)
            assert len(walked) < 10, "walk did not terminate"
            if page.exhausted:
                break
            after = (page.next_after.created_at, page.next_after.id)

        assert len(walked) >= 3, "limit=2 over 5 matches must span multiple pages"
        assert_walk_compliant(walked, ast, expected_ids=expected_ids)


@skip_no_pg
class TestScanDocumentsNamespaceIsolationPg:
    """The scan's namespace scope, asserted rather than inherited.

    Every other class here runs against a fresh single-namespace fixture, so
    none of it can notice a scan that ignores its scope — on the shared CI
    database that failure would surface only as nondeterministic contamination
    from other tests' residue. This seeds a second namespace with the same
    varied corpus (every row a guaranteed filter hit) and walks the first at
    ``scan_limit=1``, so both the scope predicate and its AND-composition with
    the keyset predicate are load-bearing on every page. Deleting the
    namespace predicate from ``build_documents_scan_query`` must fail here.
    """

    async def test_scan_never_returns_another_namespaces_rows(self, backend: PostgreSQLBackend, namespace) -> None:
        seed = scan_seed(6)
        await seed_varied(backend, namespace.id, seed)
        other = await backend.create_namespace(MemoryNamespace())
        await seed_varied(backend, other.id, scan_seed(6))

        wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
        steps = await walk_scan(backend.scan_documents, namespace.id, scan_limit=1, filter_ast=to_filter_ast(wire))
        seen = [d for step in steps for d in step.documents]

        assert seen, "the filter must match rows in the scanned namespace for this test to bite"
        assert all(d.namespace_id == namespace.id for d in seen)

        unfiltered = await backend.scan_documents(namespace.id, scan_limit=50)
        assert len(unfiltered.documents) == 6
        assert all(d.namespace_id == namespace.id for d in unfiltered.documents)
