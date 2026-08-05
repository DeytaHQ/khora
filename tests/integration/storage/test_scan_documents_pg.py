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
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.core.models import Document, MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.db.session import run_migrations
from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from khora.storage.backends.postgresql import PostgreSQLBackend
from tests.test_helpers.document_scan import ScanSeed, scan_seed, walk_scan

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
    """A fresh namespace per test, so a scan never sees another test's rows.

    ``MemoryNamespace()`` draws a distinct row-level ``id`` and stable
    ``namespace_id``. That distinction is load-bearing: ``documents.namespace_id``
    is a foreign key onto ``memory_namespaces.id``, so the scan's argument is the
    row-level id despite its name, and equal ids would make a scan that resolved
    the wrong one of the pair indistinguishable from a correct one.
    """
    return await backend.create_namespace(MemoryNamespace())


@pytest.fixture
async def other_namespace(backend: PostgreSQLBackend):
    """A second tenant, so a scan has someone else's rows to exclude.

    This database is shared across the whole integration job, so a scan will
    always have *some* foreign rows around it. That is not an assertion — it is
    residue, and it would silently disappear the day the job runs this module
    alone. The isolation test seeds its own second tenant deliberately.
    """
    return await backend.create_namespace(MemoryNamespace())


def _filter_ast(wire: dict[str, Any]) -> Any:
    return parse_to_ast(RecallFilter.model_validate(wire))


async def _write(
    backend: PostgreSQLBackend, namespace_id: UUID, doc_id: UUID, created_at: datetime, **fields: Any
) -> None:
    """Insert one document through the production write API."""
    await backend.create_document(
        Document(
            id=doc_id,
            namespace_id=namespace_id,
            content="scanned content",
            checksum=f"scan-{doc_id.hex}",
            created_at=created_at,
            updated_at=fields.pop("updated_at", created_at),
            **fields,
        )
    )


async def _seed(backend: PostgreSQLBackend, namespace_id: UUID, seed: ScanSeed) -> None:
    for doc_id, created_at in seed.writes:
        await _write(backend, namespace_id, doc_id, created_at)


async def _seed_varied(backend: PostgreSQLBackend, namespace_id: UUID, seed: ScanSeed) -> None:
    """Seed the same corpus with attribute variety, so a filter can split it.

    Attributes are assigned by *write* index, which is deliberately not the
    enumeration order — every expectation below is therefore derived from the
    rows a scan actually returns, never from this loop's counter.
    """
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await _write(
            backend,
            namespace_id,
            doc_id,
            created_at,
            title=f"doc-{i}",
            source_type="report" if i % 2 == 0 else "library",
        )


@skip_no_pg
class TestScanDocumentsTenancyPg:
    async def test_scan_returns_only_the_scanned_namespaces_rows(
        self, backend: PostgreSQLBackend, namespace, other_namespace
    ) -> None:
        """The scan never crosses a namespace, filtered or resumed.

        The second tenant is seeded to be maximally confusable with the first:
        the same timestamps, the same ``source_type`` values and the same
        titles, so neither the enumeration order, the pushdown fragment nor the
        keyset cursor can tell the two apart. Only the namespace predicate can.
        Every other test in this module runs against a single tenant it created
        itself, where deleting that predicate outright changes no result.

        The two tenants are seeded here rather than inherited from whatever else
        the shared integration database happens to hold — foreign rows that
        arrive as residue are not an assertion, and would vanish the day this
        module runs alone.

        The filter is a top-level disjunction, matching the shape the embedded
        sibling uses to cover its raw-text fragment splice. This store composes
        its fragment as a SQLAlchemy expression, which carries its own grouping,
        so the precedence hazard does not arise here — the disjunction is kept
        for parity, and because it is the shape a real caller writes.
        """
        seed_a, seed_b = scan_seed(6), scan_seed(6)
        await _seed_varied(backend, namespace.id, seed_a)
        await _seed_varied(backend, other_namespace.id, seed_b)

        wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}

        def matching(doc) -> bool:
            return doc.source_type == "report" or doc.title == "doc-1"

        mine = (await backend.scan_documents(namespace.id, scan_limit=100)).documents
        theirs = (await backend.scan_documents(other_namespace.id, scan_limit=100)).documents
        # Non-vacuous only if the other tenant holds rows this filter would accept.
        assert [d.id for d in theirs] == seed_b.expected
        assert len([d for d in theirs if matching(d)]) > 1

        unresumed = await backend.scan_documents(namespace.id, filter_ast=_filter_ast(wire), scan_limit=100)
        assert {d.id for d in unresumed.documents} == {d.id for d in mine if matching(d)}
        assert {d.id for d in unresumed.documents}.isdisjoint(seed_b.expected)

        # The keyset predicate carries no namespace of its own, so the resumed
        # path needs its own assertion rather than inheriting the one above.
        cursor_row = mine[2]
        resumed = await backend.scan_documents(
            namespace.id,
            filter_ast=_filter_ast(wire),
            after=(cursor_row.created_at, cursor_row.id),
            scan_limit=100,
        )
        assert resumed.documents, "the resumed window must not be empty, or it asserts nothing"
        assert {d.id for d in resumed.documents}.isdisjoint(seed_b.expected)
        assert {d.id for d in resumed.documents} == {d.id for d in mine[3:] if matching(d)}


@skip_no_pg
class TestScanDocumentsWindowPg:
    async def test_scan_limit_bounds_the_window(self, backend: PostgreSQLBackend, namespace) -> None:
        seed = scan_seed(6)
        await _seed(backend, namespace.id, seed)

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
        await _seed(backend, namespace.id, seed)

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
        await _seed(backend, namespace.id, seed)
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
            await _write(
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
        await _seed(backend, namespace.id, seed)

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
        await _seed(backend, namespace.id, seed)

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
        await _seed_varied(backend, namespace.id, seed)

        full = await backend.scan_documents(namespace.id, scan_limit=10)
        wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
        expected = [d.id for d in full.documents if d.source_type == "report" or d.title == "doc-1"]
        assert 1 < len(expected) < len(full.documents), "the filter must narrow, but not to a single row"

        steps = await walk_scan(
            backend.scan_documents,
            namespace.id,
            scan_limit=1,
            filter_ast=_filter_ast(wire),
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
        await _seed(backend, namespace.id, seed)

        step = await backend.scan_documents(namespace.id, scan_limit=10)
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
        await _seed(backend, namespace.id, seed)

        step = await backend.scan_documents(
            namespace.id,
            filter_ast=_filter_ast({"created_at": {"$gte": "2999-01-01T00:00:00+00:00"}}),
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
            await _write(backend, namespace.id, doc_id, created_at, source_type="report" if i % 2 == 0 else "library")

        step = await backend.scan_documents(
            namespace.id,
            filter_ast=_filter_ast(
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
        await _write(backend, namespace.id, newest, base + timedelta(seconds=2), source_type="report")
        await _write(backend, namespace.id, middle, base + timedelta(seconds=1), source_type="report")
        await _write(backend, namespace.id, oldest, base, source_type="library")

        step = await backend.scan_documents(
            namespace.id,
            filter_ast=_filter_ast(
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
