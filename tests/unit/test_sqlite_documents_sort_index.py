"""Documents sort index on the raw SQLite backend.

The raw SQLite store keeps its whole schema in ``_SCHEMA_SQL`` and re-runs that
blob on every ``connect()``; there is no Alembic chain behind it. This module
guards the three things that arrangement makes fragile:

* ``list_documents`` pins ``ORDER BY created_at DESC, id DESC``. Its index has
  to carry all of ``(namespace_id, created_at, id)`` or SQLite re-sorts every
  page through a temp B-tree. Same argument as
  ``db/migrations/versions/054_documents_namespace_created_at_id.py``, which
  covers the ORM-backed stores and never reaches this one.
* An existing database is converted by the blob itself, not by a migration, so
  the create and the drop both have to be re-runnable and both have to be
  observable on a database that was opened before the change.
* ``_create_schema`` splits the blob on ``;``. That makes plain prose in the
  blob load-bearing, which is not obvious from reading it.

Plan shape is asserted, never wall-clock time or a speed-up ratio: the aggregate
ratios move with rows-per-namespace and a threshold would flake on CI runners.
Every assertion here is paired with the opposite index set on the same data, so
a test that would pass with or without the index fails its own differential.

SQLite only - no Docker, no services.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID, uuid4

import pytest

from khora.storage.backends.sqlite import (
    _SCHEMA_SQL,
    SQLiteRelationalBackend,
    SQLiteVectorBackend,
)

SORT_INDEX = "idx_docs_ns_created_id"
LEGACY_INDEX = "idx_docs_ns"

NAMESPACE_COUNT = 3
ROWS_PER_NAMESPACE = 400
# Bulk ingest stamps many documents with one ``created_at``; ties are the
# realistic case, and they are what makes the trailing ``id`` key load-bearing.
ROWS_PER_TIMESTAMP = 10
PAGE_SIZE = 100

BACKENDS = [SQLiteRelationalBackend, SQLiteVectorBackend]

# Both halves of the index swap, matched against the blob as text.
#
# ``\b`` is load-bearing: ``_`` is a word character, so ``idx_docs_ns\b`` cannot
# match inside ``idx_docs_ns_created_id`` or ``idx_docs_ns_external_id``. The
# decoy check in :class:`TestSchemaBlobStructure` pins that.
LEGACY_CREATE_RE = re.compile(r"CREATE\s+INDEX(?:\s+IF\s+NOT\s+EXISTS)?\s+idx_docs_ns\b")
LEGACY_DROP_RE = re.compile(r"DROP\s+INDEX\s+IF\s+EXISTS\s+idx_docs_ns\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingConnection:
    """Proxy around the backend's aiosqlite connection that records SQL.

    The plan assertions have to be made against the query the shipped
    ``list_documents`` actually emits. A locally rebuilt SELECT would drift the
    moment someone edits the backend, and the tests would then keep asserting a
    good plan for a query nobody runs.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.statements: list[tuple[str, list]] = []

    async def execute(self, sql: str, parameters: Any = None) -> Any:
        self.statements.append((sql, list(parameters or [])))
        if parameters is None:
            return await self._inner.execute(sql)
        return await self._inner.execute(sql, parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


async def _capture_sql(backend: Any, coro_factory: Any) -> tuple[str, list]:
    """Run a backend method and return the last SQL statement it emitted."""
    inner = backend._conn
    recorder = _RecordingConnection(inner)
    backend._conn = recorder
    try:
        result = await coro_factory()
    finally:
        backend._conn = inner
    assert recorder.statements, "the backend emitted no SQL"
    return result, recorder.statements[-1]


class SeededDB(NamedTuple):
    """What the seeded fixture hands a test."""

    backend: SQLiteRelationalBackend
    db_path: Path
    namespace_id: UUID
    #: Newest ``updated_at`` in the seed. Predicates are expressed relative to
    #: this rather than to ``datetime.now()``, so they mean the same thing no
    #: matter how long after collection the test runs.
    newest: datetime


def _seed(db_path: Path) -> tuple[list[UUID], datetime]:
    """Fill several namespaces with documents; return the ids and the newest stamp.

    More than one namespace so ``namespace_id = ?`` is a real selection rather
    than a whole-table match, and enough rows for the planner to have something
    to choose between.
    """
    namespaces = [uuid4() for _ in range(NAMESPACE_COUNT)]
    base = datetime.now(UTC)
    con = sqlite3.connect(db_path)
    try:
        for ns in namespaces:
            con.execute(
                "INSERT INTO memory_namespaces (id, namespace_id, version, is_active, tenancy_mode, "
                "created_at, updated_at) VALUES (?, ?, 1, 1, 'shared', ?, ?)",
                (str(ns), str(ns), base.isoformat(), base.isoformat()),
            )
            con.executemany(
                "INSERT INTO documents (id, namespace_id, content, checksum, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(uuid4()),
                        str(ns),
                        f"seed document {i}",
                        f"{ns}-{i}",
                        "completed" if i % 3 else "pending",
                        (base - timedelta(seconds=i // ROWS_PER_TIMESTAMP)).isoformat(),
                        (base - timedelta(seconds=i // ROWS_PER_TIMESTAMP)).isoformat(),
                    )
                    for i in range(ROWS_PER_NAMESPACE)
                ],
            )
        con.execute("ANALYZE")
        con.commit()
    finally:
        con.close()
    return namespaces, base


def _documents_indexes(db_path: Path) -> set[str]:
    con = sqlite3.connect(db_path)
    try:
        return {str(row[1]) for row in con.execute("PRAGMA index_list(documents)").fetchall()}
    finally:
        con.close()


def _use_legacy_indexes(db_path: Path) -> None:
    """Put the documents table back on the pre-change index set.

    The change is index-only, so reverting the two index statements reproduces
    the schema an older database was opened with - and gives every plan
    assertion below a differential on identical data.
    """
    con = sqlite3.connect(db_path)
    try:
        con.execute(f"DROP INDEX IF EXISTS {SORT_INDEX}")
        con.execute(f"CREATE INDEX IF NOT EXISTS {LEGACY_INDEX} ON documents(namespace_id)")
        con.execute("ANALYZE")
        con.commit()
    finally:
        con.close()


def _query_plan(db_path: Path, statement: str, parameters: list) -> list[str]:
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


# ``SEARCH documents USING COVERING INDEX idx_docs_ns_created_id (namespace_id=?)``
# and the non-covering ``SCAN``/``SEARCH ... USING INDEX <name>`` spellings. An
# unnamed automatic index does not match, which is intended: it is not one of
# this table's indexes.
PLAN_INDEX_RE = re.compile(r"USING\s+(?:COVERING\s+)?INDEX\s+(\w+)")


def _indexes_used(plan: list[str]) -> set[str]:
    """The index names a plan reads, extracted whole.

    Matching an index name as a naked substring of a plan line is unsound here:
    ``idx_docs_ns`` is a strict prefix of both ``idx_docs_ns_created_id`` and
    ``idx_docs_ns_external_id``, and every one of those lines occurs in this
    schema's plans. Same hazard the blob regexes above guard with ``\\b``.
    """
    return {match.group(1) for line in plan for match in PLAN_INDEX_RE.finditer(line)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def seeded(tmp_path: Path) -> AsyncIterator[SeededDB]:
    """A seeded database opened through the real backend.

    On disk rather than ``:memory:`` because the plan assertions run through a
    second connection, and because the conversion tests have to close and
    reopen the file.
    """
    db_path = tmp_path / "documents.db"
    backend = SQLiteRelationalBackend(str(db_path))
    # ``connect()`` inside the try: a failure there still leaves aiosqlite's
    # non-daemon worker thread running, which hangs the whole pytest process at
    # exit instead of failing the test.
    try:
        await backend.connect()
        namespaces, newest = _seed(db_path)
        yield SeededDB(backend, db_path, namespaces[1], newest)
    finally:
        await backend.disconnect()


# ---------------------------------------------------------------------------
# The schema blob is executed by splitting it on ";" - two traps live there
# ---------------------------------------------------------------------------


class TestSchemaBlobStructure:
    """Structural properties of ``_SCHEMA_SQL`` itself.

    Deliberately not asserting on comment prose: the wording is expected to
    change and a prose assertion is brittle for no safety value.
    """

    def test_legacy_index_is_dropped_and_never_recreated(self) -> None:
        """The blob must not build ``idx_docs_ns`` and drop it on every connect.

        A create left next to the drop is invisible to a plan assertion - the
        end state is identical - but it rebuilds and destroys a full index on
        every single ``connect()``.
        """
        assert LEGACY_DROP_RE.search(_SCHEMA_SQL), (
            f"the schema blob no longer drops {LEGACY_INDEX}; databases created before the sort "
            f"index landed keep a redundant strict prefix of {SORT_INDEX} forever, paying an "
            "index write per document insert"
        )
        assert not LEGACY_CREATE_RE.search(_SCHEMA_SQL), (
            f"the schema blob still creates {LEGACY_INDEX} alongside the drop - every connect() "
            "then builds a full index and immediately destroys it"
        )

    def test_the_legacy_index_pattern_does_not_match_the_surviving_indexes(self) -> None:
        """Positive and negative controls for the pattern used above.

        Without this, a pattern that matched nothing at all would make the
        assertion above pass for the wrong reason.
        """
        assert LEGACY_CREATE_RE.search(f"CREATE INDEX IF NOT EXISTS {LEGACY_INDEX} ON documents(namespace_id);")
        assert not LEGACY_CREATE_RE.search(
            f"CREATE INDEX IF NOT EXISTS {SORT_INDEX} ON documents(namespace_id, created_at, id);"
        )
        assert not LEGACY_CREATE_RE.search(
            "CREATE INDEX IF NOT EXISTS idx_docs_ns_external_id ON documents(namespace_id, external_id);"
        )

    def test_every_chunk_of_the_blob_survives_the_splitter(self) -> None:
        """Each chunk is valid SQL once the blob is split the way it is executed.

        ``_create_schema`` splits ``_SCHEMA_SQL`` on ``;``, which makes prose in
        the blob load-bearing: a semicolon anywhere inside a comment - leading,
        trailing, or a block comment - cuts the comment and hands the remainder
        to SQLite as bare prose. That fails on the very first connect, so the
        backend does not start at all.

        Executed rather than pattern-matched. A lexical rule would have to model
        SQLite's lexer to be complete, and an incomplete rule whose docstring
        claims completeness is exactly the defect being guarded against here.
        """
        con = sqlite3.connect(":memory:")
        try:
            for index, chunk in enumerate(_SCHEMA_SQL.split(";")):
                statement = chunk.strip()
                if not statement:
                    continue
                try:
                    con.execute(statement)
                # ``sqlite3.Error``, not ``OperationalError``: a cut comment
                # whose remainder parses on its own raises something else -
                # ``-- note; SELECT ?;`` leaves ``SELECT ?``, which prepares
                # cleanly and then raises ``ProgrammingError`` on the missing
                # binding. A narrow catch lets that escape as a bare traceback
                # with no chunk index and no chunk text.
                except sqlite3.Error as exc:
                    first_line = statement.splitlines()[0]
                    pytest.fail(
                        f"chunk {index} of _SCHEMA_SQL is not executable once the blob is split on "
                        f"the semicolon: {exc}. The chunk begins: {first_line!r}"
                    )
        finally:
            con.close()

    @pytest.mark.parametrize(
        "hazard",
        [
            pytest.param("-- a note; with a semicolon\nCREATE TABLE a (id TEXT)", id="leading-comment"),
            pytest.param("CREATE TABLE b (id TEXT); -- a note; with a semicolon", id="trailing-comment"),
            pytest.param("/* a note; with a semicolon */\nCREATE TABLE c (id TEXT)", id="block-comment"),
        ],
    )
    def test_a_semicolon_inside_a_comment_really_does_break_the_splitter(self, hazard: str) -> None:
        """The failure the test above prevents, in each shape it can take.

        Evidence that the executed check is not vacuous, and the reason the
        lexical version of this guard was dropped: only the first of these three
        shapes starts a line with ``--``.
        """
        con = sqlite3.connect(":memory:")
        try:
            with pytest.raises(sqlite3.OperationalError):
                for chunk in hazard.split(";"):
                    if chunk.strip():
                        con.execute(chunk.strip())
        finally:
            con.close()

    def test_comments_without_a_semicolon_are_not_a_hazard(self) -> None:
        """The boundary of the rule, so the guard is not widened onto safe prose.

        A comment-only chunk and a comment followed by a statement both execute
        cleanly - SQLite accepts them. Only a semicolon *inside* the comment
        breaks anything.
        """
        con = sqlite3.connect(":memory:")
        try:
            con.execute("-- a comment on its own is a valid statement")
            con.execute("-- a leading comment\nCREATE TABLE safe (id TEXT)")
            con.execute("/* a block comment */\nCREATE TABLE also_safe (id TEXT)")
        finally:
            con.close()


# ---------------------------------------------------------------------------
# What connect() leaves on disk
# ---------------------------------------------------------------------------


class TestDocumentsIndexSet:
    """The index set both backends build, fresh and on an existing database."""

    @pytest.mark.parametrize("backend_cls", BACKENDS, ids=lambda c: c.__name__)
    async def test_fresh_database_has_exactly_the_expected_indexes(self, backend_cls, tmp_path: Path) -> None:
        """The two partial indexes survive; the bare namespace index does not.

        ``idx_docs_checksum`` and ``idx_docs_ns_external_id`` also lead with
        ``namespace_id``, but their ``WHERE`` clauses make them a different
        index - dropping either would be a behaviour change, not a cleanup.
        """
        db_path = tmp_path / f"{backend_cls.__name__}.db"
        backend = backend_cls(str(db_path))
        try:
            await backend.connect()
            indexes = _documents_indexes(db_path)
        finally:
            await backend.disconnect()

        assert indexes == {
            "sqlite_autoindex_documents_1",
            "idx_docs_checksum",
            "idx_docs_ns_external_id",
            SORT_INDEX,
        }

    @pytest.mark.parametrize("backend_cls", BACKENDS, ids=lambda c: c.__name__)
    async def test_existing_database_converts_on_the_next_open(self, backend_cls, tmp_path: Path) -> None:
        """No migration guards this store - the blob has to do the conversion.

        Both backends run the same blob, so both have to convert; whichever one
        opens the file first is not something a caller controls.
        """
        db_path = tmp_path / f"{backend_cls.__name__}-existing.db"
        first = backend_cls(str(db_path))
        try:
            await first.connect()
            namespaces, _ = _seed(db_path)
        finally:
            await first.disconnect()

        _use_legacy_indexes(db_path)
        before = _documents_indexes(db_path)
        assert LEGACY_INDEX in before and SORT_INDEX not in before, (
            f"the pre-change index set was not reproduced, so the reopen below proves nothing: {before}"
        )

        second = backend_cls(str(db_path))
        try:
            await second.connect()
            after = _documents_indexes(db_path)
        finally:
            await second.disconnect()

        assert SORT_INDEX in after, f"reopening an existing database did not build {SORT_INDEX}: {after}"
        assert LEGACY_INDEX not in after, f"reopening an existing database did not drop {LEGACY_INDEX}: {after}"

        con = sqlite3.connect(db_path)
        try:
            surviving = con.execute(
                "SELECT COUNT(*) FROM documents WHERE namespace_id = ?", (str(namespaces[0]),)
            ).fetchone()[0]
        finally:
            con.close()
        assert surviving == ROWS_PER_NAMESPACE, "the conversion must not touch the rows"


# ---------------------------------------------------------------------------
# list_documents plan shape
# ---------------------------------------------------------------------------

# Every predicate combination ``list_documents`` can emit. An intervening
# predicate defeating the index is the most plausible silent regression here,
# so all three are pinned rather than just the bare namespace read.
PREDICATE_SHAPES = ["namespace", "namespace+status", "namespace+status+updated_before"]


def _predicate_kwargs(shape: str, newest: datetime) -> dict[str, Any]:
    """Build the kwargs for a shape, relative to the seed's newest row.

    Never ``datetime.now()``: the rows are stamped when the fixture runs, so a
    cutoff taken from the wall clock means something different depending on how
    long after collection the test executes. Anchoring on the seed keeps the
    predicate a live filter - it excludes the newest slice of rows - and keeps
    it deterministic.
    """
    kwargs: dict[str, Any] = {}
    if "status" in shape:
        kwargs["status"] = "completed"
    if "updated_before" in shape:
        kwargs["updated_before"] = newest - timedelta(seconds=1)
    return kwargs


@pytest.mark.parametrize("shape", PREDICATE_SHAPES)
class TestListDocumentsPlanShape:
    """``list_documents`` reads its page in index order, for every predicate set.

    The two tests are a matched pair on identical data: the second is the
    evidence that the first is not passing for free.
    """

    async def _captured(self, seeded: SeededDB, shape: str) -> tuple[str, list]:
        backend = seeded.backend
        rows, statement = await _capture_sql(
            backend,
            lambda: backend.list_documents(
                seeded.namespace_id, limit=PAGE_SIZE, offset=0, **_predicate_kwargs(shape, seeded.newest)
            ),
        )
        assert len(rows) == PAGE_SIZE, f"seed too small for shape {shape}: got {len(rows)} rows"
        return statement

    async def test_sort_index_serves_the_order(self, seeded: SeededDB, shape: str) -> None:
        statement, parameters = await self._captured(seeded, shape)

        plan = _query_plan(seeded.db_path, statement, parameters)

        assert SORT_INDEX in _indexes_used(plan), (
            f"{SORT_INDEX} is not used for shape {shape} - a listing that falls back to a table "
            f"scan reads the whole namespace to return one page. Plan: {plan}"
        )
        assert not _sorts(plan), (
            f"SQLite still sorts for shape {shape}, so the index is not covering the pinned order; "
            f"the sort is redone for every page of a drain. Plan: {plan}"
        )

    async def test_legacy_index_leaves_a_sort_pass(self, seeded: SeededDB, shape: str) -> None:
        """The differential: same query, same rows, pre-change index set."""
        statement, parameters = await self._captured(seeded, shape)

        _use_legacy_indexes(seeded.db_path)
        plan = _query_plan(seeded.db_path, statement, parameters)

        assert LEGACY_INDEX in _indexes_used(plan), f"expected the narrow index to be used: {plan}"
        assert _sorts(plan), (
            f"the narrow index was expected to leave SQLite sorting for shape {shape}; if it does "
            f"not, the premise of this module is wrong and the assertions above prove nothing. Plan: {plan}"
        )


# ---------------------------------------------------------------------------
# Namespace aggregates, which lose the index they used to read
# ---------------------------------------------------------------------------

AGGREGATES = ["count_documents", "get_last_activity_at", "get_document_stats"]


@pytest.mark.parametrize("method", AGGREGATES)
class TestNamespaceAggregatePlans:
    """The three namespace aggregates re-plan onto the sort index.

    They used to read the bare namespace index, so dropping it has to leave
    them on an index rather than on a table scan. Only the index NAME is
    asserted: whether SQLite also calls the read covering is a planner detail
    that depends on the projection and can shift under a SQLite upgrade,
    breaking the suite cosmetically for no added safety.

    Plans only, never timings. The measured effect runs from clearly faster
    (``get_last_activity_at``, ``get_document_stats``: the ``MAX(created_at)``
    stops reading the table) to slightly slower (bare ``count_documents``:
    wider index entries), and both ends scale with rows per namespace, so no
    ratio here is portable across machines or datasets.
    """

    async def test_aggregate_reads_the_sort_index(self, seeded: SeededDB, method: str) -> None:
        _, (statement, parameters) = await _capture_sql(
            seeded.backend, lambda: getattr(seeded.backend, method)(seeded.namespace_id)
        )

        plan = _query_plan(seeded.db_path, statement, parameters)

        assert SORT_INDEX in _indexes_used(plan), (
            f"{method} does not read {SORT_INDEX} - with the narrow namespace index dropped, "
            f"that leaves it scanning the table. Plan: {plan}"
        )

    async def test_aggregate_read_the_narrow_index_before(self, seeded: SeededDB, method: str) -> None:
        """The differential: these named a different index before the swap.

        Without this contrast, the assertion above could not tell a real
        re-plan from an aggregate that would name the sort index either way.

        Asserted as the whole index set, not as a membership test: the sort
        index is gone from this database, so "the plan does not name the sort
        index" cannot fail and proves nothing. What can fail - and is the
        regression worth catching - is the aggregate reading some *other*
        surviving index (``idx_docs_checksum``, ``idx_docs_ns_external_id``) or
        no index at all, either of which would make the pre-change state a
        different comparison than the one this class claims to draw.
        """
        _, (statement, parameters) = await _capture_sql(
            seeded.backend, lambda: getattr(seeded.backend, method)(seeded.namespace_id)
        )

        _use_legacy_indexes(seeded.db_path)
        plan = _query_plan(seeded.db_path, statement, parameters)

        assert _indexes_used(plan) == {LEGACY_INDEX}, (
            f"{method} was expected to read {LEGACY_INDEX} and nothing else while that index "
            f"exists; if it did not, the assertion above proves nothing about the swap. "
            f"Plan: {plan}"
        )
