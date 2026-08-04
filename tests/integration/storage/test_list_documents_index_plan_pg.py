"""``list_documents`` must be planned as a sort-free index scan on PostgreSQL.

``list_documents`` pins a total order - ``ORDER BY created_at DESC, id DESC``.
The index that used to back it, ``ix_documents_namespace_created_at``, covers
only the ``created_at`` prefix of that order, so with ``namespace_id``
equality-constrained the planner still has to sort on the trailing ``id`` key
(an Incremental Sort on PG 13+). A first page barely notices. The cost lands on
full-drain offset pagination, which real callers perform - GC / session expiry,
session forget, and the agent-framework adapters that walk every document in a
namespace - because the sort is redone for every page.

Widening the index to ``(namespace_id, created_at, id)`` makes the residual
index order exactly ``(created_at ASC, id ASC)``, so a BACKWARD scan yields
``(created_at DESC, id DESC)`` directly and the sort node disappears.

These tests assert the PLAN SHAPE, not timings - timings are not portable
across CI runners, and a wall-clock threshold would flake. Plan shape is the
precise thing the wider index buys.

Two properties, both needed:

* the query is served by a backward scan of the 3-column index, with no
  ``Sort`` / ``Incremental Sort`` node anywhere in the plan;
* that assertion is NOT vacuous - the differential test rebuilds the old
  2-column index inside a rolled-back transaction and shows the same query
  *does* grow a sort node, so a plan without one is a real signal rather than
  something every plan would satisfy.

Requires a running PostgreSQL (``make dev``, port 5434). Skipped automatically
when the configured ``KHORA_DATABASE_URL`` is unreachable.

Run explicitly (the shell may leak a different URL)::

    KHORA_DATABASE_URL="postgresql://khora:khora@localhost:5434/khora" \
        UV_NO_SYNC=1 uv run pytest \
        tests/integration/storage/test_list_documents_index_plan_pg.py \
        -o addopts="" --no-cov -q
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy import event

from khora.core.models import MemoryNamespace
from khora.db.session import run_migrations
from khora.storage.backends.postgresql import PostgreSQLBackend

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    # This repo's compose puts Postgres on 5434 (see compose.yaml); defaulting
    # to 5432 would make the whole module silently skip on a local `make test`.
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


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


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _pg_reachable(), reason="PostgreSQL not reachable (run `make dev` first)"),
]

NEW_INDEX = "ix_documents_namespace_created_at_id"
OLD_INDEX = "ix_documents_namespace_created_at"

# Enough rows that a sequential scan plus a full sort is clearly the more
# expensive plan, so the planner reaches for the index on cost rather than on
# luck. Below roughly a thousand rows the two plans are close enough that a
# seq scan can win, which would make the sort-free assertion meaningless.
SEED_ROWS = 3000
PAGE_SIZE = 100

# Timestamp granularity: bulk ingest stamps many documents with the same
# ``created_at``, so ties are the realistic case and the trailing ``id`` key is
# load-bearing rather than decorative.
#
# Note this is the FAVORABLE case for the change. The ORM stamps ``created_at``
# per row, so production tie-groups are often size 1 and PG's win is smaller
# than a tie-heavy seed suggests. It does not affect the plan-shape assertions,
# which hold at any group size.
ROWS_PER_TIMESTAMP = 50

# WARNING - the seed's INSERT ORDER is load-bearing, not incidental.
#
# ``_rows`` writes ``i`` ascending while ``created_at`` descends, so physical
# heap order is almost exactly the reverse of index order and
# ``pg_stats.correlation`` lands near -1. PG interpolates an index scan's
# heap-access cost between its random and sequential bounds by that
# correlation, so a near-perfect value collapses the cost of the ~100 (or, at
# depth, ~2900) heap fetches these queries perform. Without it, default
# ``random_page_cost = 4.0`` makes an index scan roughly 2.5x more expensive
# than a seq scan plus top-N heapsort at this table size, and the Seq Scan
# guard below fires - for a reason that has nothing to do with the index being
# wrong.
#
# So: do NOT "tidy" this fixture by shuffling the seed or inserting in id
# order. If you need to change the write order, expect to raise the row counts
# to compensate, and re-run against real Postgres rather than reasoning about
# it - the cost model here is not intuitive.

# Documents seeded into OTHER namespaces, so the namespace filter is selective.
#
# These are the only thing that makes ``namespace_id`` a DISCRIMINATING key, and
# without that there is no cost reason for PG to prefer a 3-column index over a
# 1-column one on the same leading sort column. Not realism dressing - the
# assertions below are meaningless without them. Learned from a CI failure, so
# the reasoning is recorded rather than left to be rediscovered:
#
# ``documents`` also carries a single-column ``ix_documents_created_at`` (from an
# earlier temporal-search migration). Both it and the 3-column index lead on
# ``created_at``, so the near-perfect heap correlation described above discounts
# BOTH equally - heap access stops being the discriminator, and the tiebreak
# becomes index entry width: one timestamptz (~8 bytes) against
# uuid + timestamptz + uuid (~40). At ``OFFSET 2800`` the scan traverses ~2900
# entries, so that 3-4x page difference decides it and the narrower index wins.
# This is why the failure appeared at depth and not on the first page.
#
# Decoy rows break that tie. ``ix_documents_created_at`` carries no
# ``namespace_id``, so every tuple it returns needs a recheck - free when one
# namespace is 100% of the table, but real work once the table holds rows the
# scan must read and discard, while the 3-column index seeks straight to the
# namespace's range. Khora is multi-tenant, so that is also the production
# regime.
#
# Seed the decoys across the SAME timestamp range as the namespace under test
# (see ``seeded_namespace``): decoys bunched at one end would let a
# created_at-ordered scan skip them in a single range, leaving the filter
# selective in name only.
DECOY_NAMESPACES = 4
DECOY_ROWS_EACH = 3000

_INSERT_DOCUMENTS = sa.text(
    "INSERT INTO documents (id, namespace_id, content, checksum, status, created_at, updated_at) "
    "VALUES (:id, :ns, :content, :checksum, 'completed', :ts, :ts)"
)


def _rows(namespace_id: UUID, base: datetime, count: int, tag: str) -> list[dict[str, Any]]:
    return [
        {
            "id": uuid4(),
            "ns": namespace_id,
            "content": f"{tag} document {i}",
            "checksum": f"{tag}-{namespace_id}-{i}",
            "ts": base - timedelta(seconds=i // ROWS_PER_TIMESTAMP),
        }
        for i in range(count)
    ]


@pytest.fixture(scope="module")
async def _run_migrations_once():
    result = await run_migrations(DATABASE_URL)
    assert result.success, f"Migrations failed: {result.error}"


@pytest.fixture
async def backend(_run_migrations_once) -> AsyncIterator[PostgreSQLBackend]:
    be = PostgreSQLBackend(database_url=DATABASE_URL)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


@pytest.fixture
async def seeded_namespace(backend: PostgreSQLBackend) -> AsyncIterator[UUID]:
    """A namespace holding ``SEED_ROWS`` documents, with committed statistics.

    Sits inside a table that also holds several other namespaces of comparable
    size, so ``WHERE namespace_id = ?`` is selective - see ``DECOY_NAMESPACES``
    for why that matters to the plan.

    Documents go in by bulk INSERT rather than ``create_document`` - this
    fixture cares about table size and planner statistics, and thousands of
    round trips would dominate the test's runtime for no added coverage.
    """
    ns = await backend.create_namespace(MemoryNamespace())
    decoys = [await backend.create_namespace(MemoryNamespace()) for _ in range(DECOY_NAMESPACES)]
    engine = backend._engine
    assert engine is not None

    base = datetime.now(UTC)
    async with engine.begin() as conn:
        await conn.execute(_INSERT_DOCUMENTS, _rows(ns.id, base, SEED_ROWS, "plan-seed"))
        for decoy in decoys:
            # Same timestamp range as the namespace under test, so the decoys
            # interleave rather than sorting cleanly to one end - otherwise a
            # created_at-ordered scan could skip them all in one range and the
            # namespace filter would be selective in name only.
            await conn.execute(_INSERT_DOCUMENTS, _rows(decoy.id, base, DECOY_ROWS_EACH, "plan-decoy"))

    # ANALYZE must COMMIT - pg_statistic updates are MVCC-transactional, so an
    # autobegun-then-rolled-back connection would silently discard them and the
    # planner would still be working from empty-table estimates.
    async with engine.begin() as conn:
        await conn.execute(sa.text("ANALYZE documents"))

    try:
        yield ns.id
    finally:
        namespace_ids = [ns.id, *(d.id for d in decoys)]
        # Expanding IN rather than ``= ANY(:ns)``: SQLAlchemy renders one bind
        # per element, so the driver never has to infer a uuid[] array type for
        # a bare Python list. Teardown failing here would leak thousands of rows
        # into every later test's planner statistics.
        async with engine.begin() as conn:
            await conn.execute(
                sa.text("DELETE FROM documents WHERE namespace_id IN :ns").bindparams(
                    sa.bindparam("ns", expanding=True)
                ),
                {"ns": namespace_ids},
            )
            await conn.execute(
                sa.text("DELETE FROM memory_namespaces WHERE id IN :ns").bindparams(sa.bindparam("ns", expanding=True)),
                {"ns": namespace_ids},
            )


async def _capture_list_documents_sql(
    backend: PostgreSQLBackend, namespace_id: UUID, *, offset: int
) -> tuple[str, Any]:
    """Return the SQL ``list_documents`` actually emits, plus its parameters.

    Captured off the live cursor rather than hand-written here. A hand-written
    query would drift from the implementation the moment someone edits
    ``list_documents``, and the test would then happily assert a good plan for
    a query nobody runs.
    """
    engine = backend._engine
    assert engine is not None
    captured: list[tuple[str, Any]] = []

    def _capture(conn, cursor, statement, parameters, context, executemany) -> None:
        if "documents" in statement and "ORDER BY" in statement:
            captured.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        docs = await backend.list_documents(namespace_id, limit=PAGE_SIZE, offset=offset)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)

    assert len(docs) == PAGE_SIZE, f"seed too small: page at offset {offset} returned {len(docs)} rows"
    assert captured, "no SELECT against documents was captured"
    return captured[-1]


async def _explain(
    backend: PostgreSQLBackend,
    statement: str,
    parameters: Any,
    *,
    analyze: bool,
    settings: dict[str, str] | None = None,
) -> dict:
    """Run EXPLAIN over *statement* and return the root plan node.

    *settings* are applied with ``SET LOCAL`` inside the same transaction as
    the EXPLAIN, so they affect this plan and nothing else.
    """
    engine = backend._engine
    assert engine is not None
    options = "ANALYZE, FORMAT JSON" if analyze else "FORMAT JSON"
    async with engine.connect() as conn:
        trans = await conn.begin()
        try:
            for key, value in (settings or {}).items():
                await conn.exec_driver_sql(f"SET LOCAL {key} = {value}")
            result = await conn.exec_driver_sql(f"EXPLAIN ({options}) {statement}", parameters)
            raw = result.scalar()
        finally:
            await trans.rollback()
    plan = json.loads(raw) if isinstance(raw, str) else raw
    return plan[0]["Plan"]


def _walk(node: dict):
    """Yield every node in a PostgreSQL plan tree, root first."""
    yield node
    for child in node.get("Plans", []):
        yield from _walk(child)


def _node_types(root: dict) -> list[str]:
    return [n["Node Type"] for n in _walk(root)]


def _index_scans(root: dict) -> list[dict]:
    return [n for n in _walk(root) if n["Node Type"] in {"Index Scan", "Index Only Scan"}]


def _sort_nodes(root: dict) -> list[str]:
    return [t for t in _node_types(root) if "Sort" in t]


def _render(root: dict) -> str:
    """Compact one-line-per-node rendering of a plan tree, for failure output.

    Every assertion below reports this rather than just the node it tripped on.
    A plan-shape test that fails with only ``{'ix_documents_created_at'}`` tells
    you which index lost but not what the planner did instead - and since these
    tests only ever run in CI, a failure that needs a follow-up run to diagnose
    costs a whole cycle. The interesting attributes are the ones that explain a
    choice: which index, which direction, and what any sort is sorting on.
    """
    lines: list[str] = []

    def walk(node: dict, depth: int) -> None:
        parts = [node["Node Type"]]
        for key in ("Index Name", "Scan Direction", "Sort Key", "Presorted Key", "Filter"):
            if key in node:
                parts.append(f"{key}={node[key]}")
        for key in ("Actual Rows", "Plan Rows"):
            if key in node:
                parts.append(f"{key}={node[key]}")
        lines.append("  " * depth + " | ".join(str(p) for p in parts))
        for child in node.get("Plans", []):
            walk(child, depth + 1)

    walk(root, 0)
    return "\n" + "\n".join(lines)


class TestListDocumentsIndexPlanPg:
    async def test_query_is_a_sort_free_backward_index_scan(
        self, backend: PostgreSQLBackend, seeded_namespace: UUID
    ) -> None:
        """The first page is served straight off the 3-column index, no sort."""
        statement, parameters = await _capture_list_documents_sql(backend, seeded_namespace, offset=0)
        root = await _explain(backend, statement, parameters, analyze=True)
        types = _node_types(root)

        # Non-vacuity guard: a seq scan means the planner ignored every index,
        # so "no sort node above the index scan" would be trivially true of a
        # plan that has no index scan at all. Fail loudly instead.
        assert "Seq Scan" not in types, (
            f"planner chose a sequential scan - the seed ({SEED_ROWS} rows) is too small "
            f"or ANALYZE did not commit, so this test proves nothing. Plan:{_render(root)}"
        )

        scans = _index_scans(root)
        assert scans, f"expected an index scan, got plan:{_render(root)}"
        used = {s.get("Index Name") for s in scans}
        assert NEW_INDEX in used, f"expected {NEW_INDEX} to serve the query, plan used {used}. Plan:{_render(root)}"

        # The whole point of the change: no sort step.
        assert not _sort_nodes(root), (
            f"expected no Sort / Incremental Sort above the index scan, found {_sort_nodes(root)}. "
            f"That is the regression the 3-column index exists to prevent. Plan:{_render(root)}"
        )

        # A backward scan is the MECHANISM: the index is declared all-ASC, so
        # only reading it in reverse can yield (created_at DESC, id DESC). If
        # this ever reads Forward, the plan is satisfying the order some other
        # way and the sort-free assertion above is a coincidence.
        scan = next(s for s in scans if s.get("Index Name") == NEW_INDEX)
        assert scan.get("Scan Direction") == "Backward", (
            f"expected a Backward scan of {NEW_INDEX}, got {scan.get('Scan Direction')!r}"
        )

    async def test_deep_offset_page_can_be_served_without_a_sort(
        self, backend: PostgreSQLBackend, seeded_namespace: UUID
    ) -> None:
        """At depth, the index can still supply the whole order with no sort.

        Full-drain pagination is where the regression actually bit: the sort is
        redone once per page, so the cost is paid ``rows / page_size`` times.

        This asserts CAPABILITY, not the planner's unaided choice, and the
        distinction is deliberate. At the scale CI can afford (a few thousand
        rows) a deep page is genuinely cheaper as a bitmap heap scan plus a
        sort: sorting 3000 rows costs almost nothing, while an ordered index
        scan pays ~2900 random heap fetches. PostgreSQL picking that plan here
        is the cost model working correctly, not the index failing - and it
        flips the other way at production scale, where the sort is the
        expensive half. Pinning the planner's *choice* at CI scale would
        therefore be pinning an artifact of the seed size; it failed in CI
        exactly that way, first via the single-column ``ix_documents_created_at``
        and then via a bitmap plan.

        So the alternatives are switched off and the question narrowed to the
        one this migration actually answers: *when* an ordered scan is used, does
        the index supply ``(created_at DESC, id DESC)`` outright, or does a sort
        still have to be bolted on? Under the 2-column index the answer is a
        sort even with these settings - that is the differential test below.
        ``test_query_is_a_sort_free_backward_index_scan`` covers the unaided
        planner choice on the first page, so between them both properties are
        pinned.
        """
        deep_offset = SEED_ROWS - PAGE_SIZE * 2
        statement, parameters = await _capture_list_documents_sql(backend, seeded_namespace, offset=deep_offset)
        root = await _explain(
            backend,
            statement,
            parameters,
            analyze=True,
            # Leave only the ordered-index-scan path available. Without this the
            # planner reasonably prefers bitmap+sort at this row count.
            settings={"enable_bitmapscan": "off", "enable_seqscan": "off"},
        )

        used = {s.get("Index Name") for s in _index_scans(root)}
        assert NEW_INDEX in used, (
            f"expected {NEW_INDEX} to serve the deep page, plan used {used}. "
            "With bitmap and sequential scans disabled the only remaining paths are ordered "
            f"index scans, so a different index here means this one cannot supply the order. "
            f"Plan:{_render(root)}"
        )
        assert not _sort_nodes(root), (
            f"deep page grew a sort node: {_sort_nodes(root)}. The index was used but did not "
            f"supply the full ordering, which is the regression this migration exists to fix. "
            f"Plan:{_render(root)}"
        )

    async def test_old_two_column_index_would_reintroduce_a_sort(
        self, backend: PostgreSQLBackend, seeded_namespace: UUID
    ) -> None:
        """The differential: prove the sort-free assertion above is not vacuous.

        Rebuilds the superseded 2-column index in place of the 3-column one and
        re-plans the identical query. If the same query grows a sort node under
        the old index and loses it under the new one, the plan-shape assertions
        are measuring the index rather than something every plan satisfies.

        The swap runs inside a transaction that is ALWAYS rolled back. Index
        DDL is transactional on PostgreSQL, so the rollback restores both
        indexes exactly - nothing is left behind even if an assertion fails.
        ``CREATE INDEX`` here is deliberately non-concurrent: concurrent builds
        cannot run inside a transaction, and this one must be reverted.
        """
        statement, parameters = await _capture_list_documents_sql(backend, seeded_namespace, offset=0)

        engine = backend._engine
        assert engine is not None
        async with engine.connect() as conn:
            trans = await conn.begin()
            try:
                await conn.exec_driver_sql(f"DROP INDEX {NEW_INDEX}")
                await conn.exec_driver_sql(f"CREATE INDEX {OLD_INDEX} ON documents (namespace_id, created_at)")

                result = await conn.exec_driver_sql(f"EXPLAIN (FORMAT JSON) {statement}", parameters)
                raw = result.scalar()
                root = (json.loads(raw) if isinstance(raw, str) else raw)[0]["Plan"]
            finally:
                await trans.rollback()

        assert _sort_nodes(root), (
            "the 2-column index was expected to force a sort for "
            f"ORDER BY created_at DESC, id DESC, but the plan has none: {_node_types(root)}. "
            "Without that contrast the sort-free assertions in this module prove nothing."
        )
        # Guard the contrast itself: a sort node proves nothing if the widened
        # index somehow survived the swap and the plan sorted for an unrelated
        # reason. Its absence is what makes the sort attributable to the swap.
        assert NEW_INDEX not in {s.get("Index Name") for s in _index_scans(root)}, (
            f"{NEW_INDEX} was still in the plan during the differential, so the swap did not "
            f"take effect and the sort above is not attributable to it. Plan:{_render(root)}"
        )

        # And the indexes really did come back on rollback.
        async with engine.connect() as conn:
            result = await conn.execute(
                sa.text("SELECT indexname FROM pg_indexes WHERE tablename = 'documents' AND indexname = ANY(:names)"),
                {"names": [NEW_INDEX, OLD_INDEX]},
            )
            present = {row[0] for row in result.fetchall()}
        assert present == {NEW_INDEX}, f"rollback did not restore the index set, found {present}"
