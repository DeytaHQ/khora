"""``SQLiteLanceRelationalAdapter.scan_documents`` — the bounded keyset scan.

Runs against the real Alembic-migrated SQLite schema (no Docker, no services),
which matters more here than usual: this store keeps ``created_at`` as TEXT and
compares it lexicographically, so the scan's cursor is only correct if it is
bound in the store's own serialization. That property cannot be checked against
a mock, and it is shared with the PostgreSQL leg — both stores build the same
statement through ``build_documents_scan_query`` — so this locally-runnable
module carries most of the semantics and its PostgreSQL sibling
(``tests/integration/storage/test_scan_documents_pg.py``) covers what only a
live server can show.

Seeding goes through ``create_document``, the production write API, so every row
is serialized by the same path production writes take. Timestamps are pinned to
a whole second on purpose; see :mod:`tests.test_helpers.document_scan`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Document, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentStatus
from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from tests.test_helpers.document_scan import ScanSeed, scan_seed, walk_scan

# Lives in the unit lane next to the rest of this adapter's tests, and also
# carries the ``embedded`` marker so ``make test-embedded`` — the no-Docker
# command a developer reaches for to check this stack — actually selects it.
pytestmark = [
    pytest.mark.embedded,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

if _HAS_EMBEDDED:
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter


@pytest.fixture
async def adapter(migrated_sqlite_db, tmp_path):
    handle = EmbeddedStorageHandle(
        EmbeddedStorageHandleConfig(db_path=str(migrated_sqlite_db), lance_path=str(tmp_path / "khora.lance")),
    )
    adapter = SQLiteLanceRelationalAdapter(handle)
    await adapter.connect()
    try:
        yield adapter
    finally:
        await adapter.disconnect()
        await handle.disconnect()


@pytest.fixture
async def namespace(adapter):
    nid = uuid4()
    return await adapter.create_namespace(MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED))


def _filter_ast(wire: dict[str, Any]) -> Any:
    return parse_to_ast(RecallFilter.model_validate(wire))


async def _write(adapter: Any, namespace_id: UUID, doc_id: UUID, created_at: datetime, **fields: Any) -> None:
    """Insert one document through the production write API."""
    await adapter.create_document(
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


async def _seed(adapter: Any, namespace_id: UUID, seed: ScanSeed) -> None:
    for doc_id, created_at in seed.writes:
        await _write(adapter, namespace_id, doc_id, created_at)


# --------------------------------------------------------------------------- #
# The window bound
# --------------------------------------------------------------------------- #


async def test_scan_limit_bounds_the_window(adapter, namespace) -> None:
    seed = scan_seed(6)
    await _seed(adapter, namespace.id, seed)

    step = await adapter.scan_documents(namespace.id, scan_limit=2)

    assert [d.id for d in step.documents] == seed.expected[:2]
    assert step.last_scanned == (step.documents[-1].created_at, step.documents[-1].id)
    assert step.exhausted is False


async def test_a_full_window_is_not_yet_exhausted(adapter, namespace) -> None:
    """``exhausted`` means SQL ran short, not "the caller has seen everything".

    A window filled exactly to the bound cannot distinguish "six rows and no
    more" from "six rows and a seventh waiting", so it must report not-exhausted
    and let the next step find the empty tail. Reporting exhaustion here would
    silently truncate every namespace whose size is a multiple of the bound.
    """
    seed = scan_seed(6)
    await _seed(adapter, namespace.id, seed)

    exact = await adapter.scan_documents(namespace.id, scan_limit=6)
    assert len(exact.documents) == 6
    assert exact.exhausted is False

    over = await adapter.scan_documents(namespace.id, scan_limit=7)
    assert len(over.documents) == 6
    assert over.exhausted is True


async def test_scan_limit_below_one_is_rejected(adapter, namespace) -> None:
    """A zero bound would return an empty window that reports neither a resume
    position nor exhaustion — the one pair a walking caller cannot act on."""
    with pytest.raises(ValueError, match="scan_limit"):
        await adapter.scan_documents(namespace.id, scan_limit=0)


# --------------------------------------------------------------------------- #
# The keyset cursor
# --------------------------------------------------------------------------- #


async def test_walk_visits_every_document_exactly_once_in_total_order(adapter, namespace) -> None:
    """One row per step across a tie block, chaining ``last_scanned``.

    ``scan_limit=1`` puts a cursor boundary between every pair of rows,
    including between rows that share a ``created_at`` to the microsecond — so
    every resume in this walk is a mid-tie resume, and the ``id DESC`` leg
    decides all of them.
    """
    seed = scan_seed(6)
    await _seed(adapter, namespace.id, seed)

    steps = await walk_scan(adapter.scan_documents, namespace.id, scan_limit=1)
    seen = [d.id for step in steps for d in step.documents]

    assert len(seen) == len(set(seen))  # no document served twice
    assert set(seen) == set(seed.expected)  # every document served
    assert seen == seed.expected  # and in one total order across the concatenation
    assert steps[-1].documents == []
    assert steps[-1].last_scanned is None
    assert steps[-1].exhausted is True


async def test_whole_second_cursor_excludes_its_own_row_and_keeps_its_tie_mates(adapter, namespace) -> None:
    """The cursor round-trips exactly at ``.000000`` microseconds.

    This store writes ``created_at`` as TEXT and orders it lexicographically, so
    a cursor is only correct if it is serialized byte-for-byte the way the
    column was written. The two ways to get that wrong fail in opposite
    directions and neither raises, so both are asserted:

    * A hand-built ISO-8601 string sorts ABOVE every stored value, because
      ``'T'`` outranks the stored space separator. The window then matches the
      cursor's OWN row, and a walk chaining ``last_scanned`` never advances.
    * A space-separated form that omits the six-digit microsecond field — which
      is what ``str(datetime)`` produces at a whole second, and what a bare
      ``datetime`` handed to an untyped bind produces via the driver's
      deprecated adapter — sorts BELOW its tie-mates, silently skipping them.

    At non-zero microseconds the second form is byte-identical to the stored
    value and this test would pass against it, which is why the seed pins a
    whole second rather than sampling the clock.
    """
    seed = scan_seed(6)
    await _seed(adapter, namespace.id, seed)

    full = await adapter.scan_documents(namespace.id, scan_limit=10)
    assert [d.id for d in full.documents] == seed.expected

    cursor_doc = next(d for d in full.documents if d.id == seed.tied_ids[0])
    # The whole second survived the write/read round trip, so the divergence
    # between the stored form and a hand-formatted one is live in this corpus.
    assert cursor_doc.created_at.microsecond == 0
    assert cursor_doc.created_at.replace(tzinfo=UTC) == seed.tie_instant

    step = await adapter.scan_documents(
        namespace.id,
        after=(cursor_doc.created_at, cursor_doc.id),
        scan_limit=10,
    )
    ids = [d.id for d in step.documents]

    assert cursor_doc.id not in ids, "the cursor's own row came back — a resumed walk would never advance"
    assert seed.tied_ids[1] in ids, "the cursor's tie-mate was skipped — a resumed walk would lose rows"
    assert ids == seed.expected[seed.expected.index(cursor_doc.id) + 1 :]


async def test_empty_window_reports_exhausted_without_a_position(adapter, namespace) -> None:
    """Both the never-seeded namespace and the tail past the last row."""
    empty = await adapter.scan_documents(namespace.id, scan_limit=5)
    assert empty.documents == []
    assert empty.last_scanned is None
    assert empty.exhausted is True

    seed = scan_seed(6)
    await _seed(adapter, namespace.id, seed)
    full = await adapter.scan_documents(namespace.id, scan_limit=10)
    oldest = full.documents[-1]

    tail = await adapter.scan_documents(namespace.id, after=(oldest.created_at, oldest.id), scan_limit=5)
    assert tail.documents == []
    assert tail.last_scanned is None
    assert tail.exhausted is True


# --------------------------------------------------------------------------- #
# The compile split
# --------------------------------------------------------------------------- #


async def test_date_system_keys_are_not_pushed_down_by_this_store(adapter, namespace) -> None:
    """``created_at`` reaches the caller's post-filter here, and that is intended.

    This store withholds the date-valued system keys from its pushdown
    whitelist because its stored ``DATETIME`` text does not order against the
    compiler's ISO binds. The leaf therefore compiles to a match-all placeholder,
    stays out of ``consumed_keys``, and narrows nothing — asserted positively,
    because a widened whitelist would show up here as a suddenly-shorter window
    rather than as an error. PostgreSQL reports the same filter as pushed; the
    two answers differing is the intended split, not drift.
    """
    seed = scan_seed(6)
    await _seed(adapter, namespace.id, seed)

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=_filter_ast({"created_at": {"$gte": "2999-01-01T00:00:00+00:00"}}),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset()
    # Every row is older than the bound, so a pushed-down comparison would have
    # returned nothing at all.
    assert [d.id for d in step.documents] == seed.expected


async def test_split_reports_only_the_leaves_sql_enforced(adapter, namespace) -> None:
    """A mixed filter: one pushable leaf, one that must reach the post-filter."""
    seed = scan_seed(6)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await _write(adapter, namespace.id, doc_id, created_at, source_type="report" if i % 2 == 0 else "library")

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=_filter_ast({"source_type": {"$eq": "report"}, "created_at": {"$gte": "2026-01-01T00:00:00+00:00"}}),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset({"source_type"})
    # The pushed leaf really did narrow the window; the unpushed one did not.
    assert {d.source_type for d in step.documents} == {"report"}
    assert len(step.documents) == 3


# --------------------------------------------------------------------------- #
# What the position means
# --------------------------------------------------------------------------- #


async def test_last_scanned_is_the_final_raw_row_not_the_last_match(adapter, namespace) -> None:
    """Resume from the last row SCANNED, not from the last row that matches.

    The window here deliberately ENDS on a row the caller's post-filter will
    reject: the filter is an ``$or`` mixing a pushable leaf with an unbacked
    one, which the compiler defers as a whole rather than pushing half of a
    disjunction, so SQL narrows nothing and the oldest row — which does not
    satisfy the filter — is the last row of the raw window.

    A walk that resumed from the last *matching* row instead would re-scan the
    rejected gap on every step — and when a whole window is rejected there is no
    matching row to resume from at all, so such a walk cannot advance past a run
    of non-matching rows longer than one window. Taking the position from the
    raw window is what lets ``exhausted`` be the only termination signal.
    """
    newest, middle, oldest = (uuid4() for _ in range(3))
    base = datetime(2026, 1, 31, 12, 30, tzinfo=UTC)
    await _write(adapter, namespace.id, newest, base + timedelta(seconds=2), source_type="report")
    await _write(adapter, namespace.id, middle, base + timedelta(seconds=1), source_type="report")
    await _write(adapter, namespace.id, oldest, base, source_type="library")

    step = await adapter.scan_documents(
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

    # Nothing was pushed, so the raw window is the whole namespace and its last
    # row is the one the post-filter will drop.
    assert step.consumed_keys == frozenset()
    assert [d.id for d in step.documents] == [newest, middle, oldest]
    assert step.documents[-1].source_type == "library"

    last_row = step.documents[-1]
    assert step.last_scanned == (last_row.created_at, last_row.id)

    last_match = step.documents[1]
    assert step.last_scanned != (last_match.created_at, last_match.id)


# --------------------------------------------------------------------------- #
# The non-filter narrowing legs
# --------------------------------------------------------------------------- #


async def test_status_and_updated_before_narrow_the_window(adapter, namespace) -> None:
    seed = scan_seed(6)
    cutoff = seed.tie_instant + timedelta(hours=1)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await _write(
            adapter,
            namespace.id,
            doc_id,
            created_at,
            status=DocumentStatus.COMPLETED if i % 2 == 0 else DocumentStatus.PENDING,
            updated_at=cutoff - timedelta(minutes=1) if i < 4 else cutoff + timedelta(minutes=1),
        )

    by_status = await adapter.scan_documents(namespace.id, status=DocumentStatus.COMPLETED.value, scan_limit=10)
    assert {d.id for d in by_status.documents} == {doc_id for i, (doc_id, _) in enumerate(seed.writes) if i % 2 == 0}

    by_updated = await adapter.scan_documents(namespace.id, updated_before=cutoff, scan_limit=10)
    assert {d.id for d in by_updated.documents} == {doc_id for i, (doc_id, _) in enumerate(seed.writes) if i < 4}
