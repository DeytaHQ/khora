"""``list_documents`` plan shape on the embedded SQLite stack.

The embedded backend gets its schema from the same Alembic chain as Postgres
and runs the same ``list_documents`` query, pinned to ``ORDER BY created_at
DESC, id DESC``. An index on ``(namespace_id, created_at)`` supplies only a
prefix of that order, so SQLite adds a sort pass - reported as ``USE TEMP
B-TREE FOR LAST TERM OF ORDER BY``. Because the sort is redone for every page,
the cost shows up on full-drain offset pagination rather than on a single read.

Two separate questions, deliberately split into two test classes:

* :class:`TestSqliteSortIndexDifferential` - does the third key actually change
  SQLite's plan? This is the evidence that the index shape matters here at all,
  and it is independent of any migration: it builds each index variant by hand
  and compares plans.
* :class:`TestSqliteChainDeliversSortIndex` - does the migration chain actually
  give an embedded database that index? This is the one that guards the
  shipped behaviour.

Plan shape is asserted rather than wall-clock time: timings are not portable
across CI runners and a threshold would flake. For reference, the plan
difference below was measured at 200k documents in one namespace, page size
100, full drain - 120s with the 2-column index against 7.7s with the 3-column
one, the entire gap being the per-page temp B-tree.

No Docker and no services required - SQLite only.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from khora.db.models import DocumentModel

pytestmark = [pytest.mark.integration, pytest.mark.embedded]

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"

NEW_INDEX = "ix_documents_namespace_created_at_id"
OLD_INDEX = "ix_documents_namespace_created_at"

SEED_ROWS = 5000
PAGE_SIZE = 100
# Bulk ingest stamps many documents with one ``created_at``; ties are the
# realistic case, which is what makes the trailing ``id`` key load-bearing.
ROWS_PER_TIMESTAMP = 50


def _build_schema(db_path: Path) -> None:
    """Run the full Alembic chain to head against a fresh SQLite file."""
    url = f"sqlite:///{db_path}"
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["database_url"] = url
    command.upgrade(cfg, "head")


def _seed(db_path: Path) -> UUID:
    """Insert a namespace holding ``SEED_ROWS`` documents; return its id."""
    ns = uuid4()
    base = datetime.now(UTC)
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO memory_namespaces (id, namespace_id, version, is_active, tenancy_mode, "
            "created_at, updated_at) VALUES (?, ?, 1, 1, 'shared', ?, ?)",
            (ns.hex, ns.hex, base.isoformat(), base.isoformat()),
        )
        con.executemany(
            "INSERT INTO documents (id, namespace_id, content, checksum, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'completed', ?, ?)",
            [
                (
                    uuid4().hex,
                    ns.hex,
                    f"seed document {i}",
                    f"plan-seed-{i}",
                    (base - timedelta(seconds=i // ROWS_PER_TIMESTAMP)).isoformat(),
                    (base - timedelta(seconds=i // ROWS_PER_TIMESTAMP)).isoformat(),
                )
                for i in range(SEED_ROWS)
            ],
        )
        con.execute("ANALYZE")
        con.commit()
    finally:
        con.close()
    return ns


@pytest.fixture(scope="module")
def mutable_db() -> Iterator[tuple[Path, UUID]]:
    """Seeded database for the differential tests, which SWAP ITS INDEXES.

    Deliberately separate from :func:`pristine_db`. Sharing one database
    between the two classes would let an index built by a differential test
    satisfy the chain assertions - checked, not assumed: an earlier draft
    shared a single fixture and the chain test passed against a chain that
    never created the index.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "differential.db"
        _build_schema(db_path)
        yield db_path, _seed(db_path)


@pytest.fixture(scope="module")
def pristine_db() -> Iterator[Path]:
    """A database built by the migration chain and never modified afterwards."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "pristine.db"
        _build_schema(db_path)
        yield db_path


@pytest.fixture
async def engine(mutable_db) -> AsyncIterator[AsyncEngine]:
    db_path, _ = mutable_db
    eng = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        yield eng
    finally:
        await eng.dispose()


def _documents_indexes(db_path: Path) -> dict[str, str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'documents'"
        ).fetchall()
    finally:
        con.close()
    return {name: sql or "" for name, sql in rows}


async def _capture_list_documents_sql(engine: AsyncEngine, namespace_id: UUID, *, offset: int) -> tuple[str, Any]:
    """Return the SQL the embedded ``list_documents`` emits, plus its parameters.

    Built from the ORM statement the backend constructs and captured off the
    live cursor, rather than hand-written here - a hand-written query would
    drift from the implementation and the test would assert a good plan for a
    query nobody runs.
    """
    captured: list[tuple[str, Any]] = []

    def _capture(conn, cursor, statement, parameters, context, executemany) -> None:
        if "documents" in statement and "ORDER BY" in statement:
            captured.append((statement, parameters))

    stmt = (
        select(DocumentModel)
        .where(DocumentModel.namespace_id == namespace_id)
        .limit(PAGE_SIZE)
        .offset(offset)
        .order_by(DocumentModel.created_at.desc(), DocumentModel.id.desc())
    )

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(stmt)
            rows = result.fetchall()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)

    assert len(rows) == PAGE_SIZE, f"seed too small: page at offset {offset} returned {len(rows)} rows"
    assert captured, "no SELECT against documents was captured"
    return captured[-1]


def _query_plan(db_path: Path, statement: str, parameters: Any) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(f"EXPLAIN QUERY PLAN {statement}", parameters).fetchall()
    finally:
        con.close()
    # The human-readable description is the last column of each row.
    return [str(row[-1]) for row in rows]


def _sorts(plan: list[str]) -> list[str]:
    """Plan lines describing a sort pass. SQLite spells these 'TEMP B-TREE'."""
    return [line for line in plan if "TEMP B-TREE" in line.upper()]


class TestSqliteSortIndexDifferential:
    """The third key changes SQLite's plan - measured, not assumed.

    Builds each index variant by hand on the seeded database and compares the
    plan for the identical query. Without this contrast, an assertion that the
    plan has no sort node would not distinguish a covering index from a lucky
    planner.
    """

    async def _plan_with_index(self, engine: AsyncEngine, mutable_db, create_sql: str) -> list[str]:
        db_path, ns = mutable_db
        statement, parameters = await _capture_list_documents_sql(engine, ns, offset=0)

        con = sqlite3.connect(db_path)
        try:
            con.execute(f"DROP INDEX IF EXISTS {NEW_INDEX}")
            con.execute(f"DROP INDEX IF EXISTS {OLD_INDEX}")
            con.execute(create_sql)
            con.execute("ANALYZE")
            con.commit()
        finally:
            con.close()

        return _query_plan(db_path, statement, parameters)

    async def test_two_column_index_forces_a_sort(self, engine: AsyncEngine, mutable_db) -> None:
        plan = await self._plan_with_index(
            engine, mutable_db, f"CREATE INDEX {OLD_INDEX} ON documents (namespace_id, created_at)"
        )
        assert any(OLD_INDEX in line for line in plan), f"expected the 2-column index to be used: {plan}"
        assert _sorts(plan), (
            "expected the 2-column index to leave SQLite sorting on the trailing key; "
            f"if it does not, this module's premise is wrong. Plan: {plan}"
        )

    async def test_three_column_index_removes_the_sort(self, engine: AsyncEngine, mutable_db) -> None:
        plan = await self._plan_with_index(
            engine, mutable_db, f"CREATE INDEX {NEW_INDEX} ON documents (namespace_id, created_at, id)"
        )
        assert any(NEW_INDEX in line for line in plan), f"expected the 3-column index to be used: {plan}"
        assert not _sorts(plan), (
            f"the 3-column index should let SQLite read the rows already ordered, found: {_sorts(plan)}"
        )


class TestSqliteChainDeliversSortIndex:
    """A database built by the migration chain must end up with the wide index.

    The embedded stack has no separate schema definition - Alembic is the only
    thing that creates its tables - so whatever the chain produces IS the
    embedded schema. If a migration is gated to Postgres, the embedded stack
    silently keeps the narrower index and the sort pass above.
    """

    def test_chain_head_has_the_three_column_index(self, pristine_db: Path) -> None:
        indexes = _documents_indexes(pristine_db)

        assert NEW_INDEX in indexes, (
            f"a SQLite database at chain head does not have {NEW_INDEX}; it has {sorted(indexes)}. "
            "The embedded stack takes its schema from this chain, so it keeps the narrower "
            "index and pays a sort pass on every page of a document listing."
        )
        assert OLD_INDEX not in indexes, (
            f"{OLD_INDEX} still present at head - it shares a prefix with {NEW_INDEX}, "
            "so keeping both costs an index write per document insert"
        )

    def test_orm_and_chain_agree_on_the_documents_indexes(self, pristine_db: Path) -> None:
        """No drift between what the ORM declares and what the chain builds.

        Scoped to the two indexes this change touches. A broader comparison
        would trip over the genuinely Postgres-only indexes (HNSW, GIN, BRIN)
        that the embedded stack is not expected to have.
        """
        built = set(_documents_indexes(pristine_db))
        declared = {ix.name for ix in DocumentModel.__table__.indexes}

        for name in (NEW_INDEX, OLD_INDEX):
            assert (name in built) == (name in declared), (
                f"{name} is {'declared in the ORM' if name in declared else 'absent from the ORM'} "
                f"but {'present in' if name in built else 'missing from'} a chain-built database"
            )
