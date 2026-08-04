"""Tests that Alembic migrations stay in sync with ORM models.

These tests ensure that:
1. All migration .py source files are committed (not just .pyc)
2. ORM models and migrations produce the same schema (no drift)
3. Composite indexes agree between the ORM and the migration that builds them
4. create_tables() emits a deprecation warning
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.db.models import Base

VERSIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "khora" / "db" / "migrations" / "versions"


def _make_mock_engine() -> tuple[MagicMock, AsyncMock]:
    """Create a mock SQLAlchemy async engine with a begin() context manager."""
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.run_sync = AsyncMock()
    mock_engine.begin = MagicMock(return_value=mock_conn)
    return mock_engine, mock_conn


def _read_all_migration_text() -> str:
    """Read and concatenate all migration source files."""
    parts: list[str] = []
    for py_file in VERSIONS_DIR.glob("*.py"):
        if py_file.name != "__init__.py":
            parts.append(py_file.read_text())
    return "".join(parts)


# ---------------------------------------------------------------------------
# Migration source file integrity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMigrationSourceFiles:
    """Verify that all migration .py source files are committed."""

    def test_versions_directory_exists(self):
        """The migrations/versions directory must exist."""
        assert VERSIONS_DIR.is_dir(), f"Missing migrations directory: {VERSIONS_DIR}"

    def test_no_orphan_pyc_files(self):
        """Every .pyc must have a corresponding .py source file."""
        pycache = VERSIONS_DIR / "__pycache__"
        if not pycache.exists():
            return  # No compiled files — nothing to check

        for pyc in pycache.glob("*.pyc"):
            # .pyc names are like "000_initial_schema.cpython-313.pyc"
            stem = pyc.stem.rsplit(".", 1)[0]  # Strip cpython-3xx suffix
            source = VERSIONS_DIR / f"{stem}.py"
            assert source.exists(), (
                f"Migration source file missing: {source.name}. "
                f"Only the compiled .pyc exists ({pyc.name}). "
                f"This means the migration will not run in production."
            )

    def test_all_migrations_have_revision(self):
        """Every migration .py file must define a revision variable."""
        for py_file in sorted(VERSIONS_DIR.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text()
            assert "revision" in content, f"Migration {py_file.name} is missing 'revision' attribute"

    def test_migration_chain_is_contiguous(self):
        """Each migration's down_revision must reference the previous one."""
        migrations: list[tuple[str, str | None]] = []

        for py_file in sorted(VERSIONS_DIR.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text()
            # Extract revision and down_revision from file content
            revision = None
            down_revision = None
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("revision") and "=" in stripped:
                    revision = stripped.split("=", 1)[1].strip().strip("'\"")
                elif stripped.startswith("down_revision") and "=" in stripped:
                    val = stripped.split("=", 1)[1].strip().strip("'\"")
                    down_revision = val if val != "None" else None
            if revision:
                migrations.append((revision, down_revision))

        assert len(migrations) > 0, "No migrations found"

        # First migration must have down_revision = None
        assert migrations[0][1] is None, f"First migration {migrations[0][0]} should have down_revision=None"

        # Each subsequent migration must point back to the previous
        for i in range(1, len(migrations)):
            current_rev, current_down = migrations[i]
            expected_down = migrations[i - 1][0]
            assert current_down == expected_down, (
                f"Migration chain broken: {current_rev} has "
                f"down_revision={current_down!r} but expected {expected_down!r}"
            )


# ---------------------------------------------------------------------------
# ORM / migration drift detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestORMMigrationDrift:
    """Verify ORM model columns have corresponding migration coverage."""

    def test_all_orm_tables_exist_in_migrations(self):
        """Every ORM table name should appear in at least one migration."""
        migration_text = _read_all_migration_text()

        # Check every ORM table is referenced in migrations
        missing_tables = []
        for table_name in Base.metadata.tables:
            if table_name not in migration_text:
                missing_tables.append(table_name)

        assert not missing_tables, (
            f"ORM tables not covered by any migration: {missing_tables}. "
            f"Run 'uv run alembic revision --autogenerate' to create a migration."
        )

    def test_all_orm_columns_referenced_in_migrations(self):
        """Every ORM column should be referenced alongside its table in migrations.

        Uses a table-scoped check: for each ORM column, we verify the column
        name appears in at least one migration file that also references its
        table. This avoids false negatives where a common column name (e.g.
        ``status``, ``created_at``) exists on another table's migration.
        """
        # Build per-file text for table-scoped matching
        migration_files: list[str] = []
        for py_file in VERSIONS_DIR.glob("*.py"):
            if py_file.name != "__init__.py":
                migration_files.append(py_file.read_text())

        missing_columns = []
        for table_name, table in Base.metadata.tables.items():
            # Find migration files that reference this table
            table_migrations = [m for m in migration_files if table_name in m]
            for column in table.columns:
                col_name = column.name
                # Column must appear in at least one migration that also
                # references its owning table
                if not any(col_name in m for m in table_migrations):
                    missing_columns.append(f"{table_name}.{col_name}")

        assert not missing_columns, (
            f"ORM columns not covered by any migration for their table: {missing_columns}. "
            f"Create an Alembic migration for these columns."
        )


# ---------------------------------------------------------------------------
# ORM / migration index agreement
# ---------------------------------------------------------------------------


class IndexOp(NamedTuple):
    """A single index DDL statement, with the properties worth asserting on."""

    action: str  # "create" | "drop"
    name: str
    columns: tuple[str, ...]  # empty for drops
    concurrent: bool
    in_autocommit_block: bool


class IndexOps(NamedTuple):
    """Index DDL from one dialect branch, in SOURCE ORDER.

    ``operations`` is the ordered record; ``created`` / ``dropped`` are derived
    views over it for the assertions that do not care about sequencing.
    """

    operations: tuple[IndexOp, ...]

    @property
    def created(self) -> dict[str, tuple[str, ...]]:
        return {op.name: op.columns for op in self.operations if op.action == "create"}

    @property
    def dropped(self) -> set[str]:
        return {op.name for op in self.operations if op.action == "drop"}

    def index_of(self, action: str, name: str) -> int | None:
        for i, op in enumerate(self.operations):
            if op.action == action and op.name == name:
                return i
        return None


def _mentions_is_postgres(test: ast.expr) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "_is_postgres" for n in ast.walk(test))


def _is_autocommit_block(item: ast.withitem) -> bool:
    call = item.context_expr
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "autocommit_block"


def _dialect_branches(func: ast.FunctionDef) -> dict[str, list[ast.stmt]]:
    """Split a migration function body into its per-dialect branches.

    Returns ``{"postgresql": [...], "other": [...]}`` for a dialect-gated
    migration, or ``{"": [...]}`` for one that issues the same DDL everywhere.

    Splitting matters because the two branches emit the SAME index names via
    DIFFERENT spellings. Scanning the whole function body into one dict keyed by
    index name lets whichever branch is visited last silently overwrite the
    other, so a Postgres branch declaring the wrong column ORDER would be masked
    by a correct SQLite branch. Both branches have to be checked on their own.

    Handles the two shapes this codebase uses:

    * ``if _is_postgres(): <A> else: <B>`` - and the negated spelling.
    * ``if not _is_postgres(): return`` followed by the Postgres DDL, i.e. the
      early-return form. Statements after the ``If`` belong to the branch that
      falls through, so a migration that silently no-ops on SQLite yields an
      EMPTY ``other`` branch and trips the assertions below rather than passing.
    """
    for i, stmt in enumerate(func.body):
        if isinstance(stmt, ast.If) and _mentions_is_postgres(stmt.test):
            negated = isinstance(stmt.test, ast.UnaryOp) and isinstance(stmt.test.op, ast.Not)
            rest = func.body[i + 1 :]
            if negated:
                return {"postgresql": stmt.orelse + rest, "other": stmt.body}
            return {"postgresql": stmt.body, "other": stmt.orelse + rest}
    return {"": func.body}


def _op_calls(nodes: list[ast.stmt]) -> list[tuple[ast.Call, bool]]:
    """Every ``op.*`` call in *nodes*, paired with whether it sits in an
    ``autocommit_block()``, in source order.

    Source order is why this is a hand-rolled descent rather than ``ast.walk``:
    ``walk`` is breadth-first, so it would interleave statements from different
    nesting depths and destroy the sequencing that ``create before drop`` needs.
    Results are sorted by position to stay honest regardless of traversal shape.
    """
    found: list[tuple[ast.Call, bool]] = []

    def descend(node: ast.AST, in_block: bool) -> None:
        if isinstance(node, ast.With):
            in_block = in_block or any(_is_autocommit_block(item) for item in node.items)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "op"
        ):
            found.append((node, in_block))
        for child in ast.iter_child_nodes(node):
            descend(child, in_block)

    for root in nodes:
        descend(root, False)

    return sorted(found, key=lambda pair: (pair[0].lineno, pair[0].col_offset))


def _scan(nodes: list[ast.stmt]) -> IndexOps:
    """Collect index DDL from a list of statements, in source order.

    Both spellings are recognised, since a dialect-gated migration typically
    uses raw SQL on one branch and the Alembic helpers on the other:

    * ``op.execute("CREATE INDEX ... ON t (a, b)")`` / ``op.execute("DROP INDEX ...")``
    * ``op.create_index("name", "t", ["a", "b"])`` / ``op.drop_index("name", ...)``
    """
    operations: list[IndexOp] = []

    for node, in_block in _op_calls(nodes):
        target = node.func
        assert isinstance(target, ast.Attribute)

        if target.attr == "execute" and node.args and isinstance(node.args[0], ast.Constant):
            statement = str(node.args[0].value)
            collapsed = " ".join(statement.split())
            upper = collapsed.upper()
            concurrent = "CONCURRENTLY" in upper
            if upper.startswith("CREATE INDEX"):
                head, _, cols = collapsed.partition("(")
                name = head.split()[-3]  # "... <name> ON <table>"
                columns = tuple(c.strip() for c in cols.rstrip(")").split(","))
                operations.append(IndexOp("create", name, columns, concurrent, in_block))
            elif upper.startswith("DROP INDEX"):
                operations.append(IndexOp("drop", collapsed.split()[-1], (), concurrent, in_block))

        elif target.attr == "create_index" and len(node.args) >= 3:
            name_node, cols_node = node.args[0], node.args[2]
            if isinstance(name_node, ast.Constant) and isinstance(cols_node, ast.List):
                columns = tuple(str(c.value) for c in cols_node.elts if isinstance(c, ast.Constant))
                operations.append(IndexOp("create", str(name_node.value), columns, False, in_block))

        elif target.attr == "drop_index" and node.args and isinstance(node.args[0], ast.Constant):
            operations.append(IndexOp("drop", str(node.args[0].value), (), False, in_block))

    return IndexOps(tuple(operations))


def _index_ops(migration_file: str, func_name: str) -> dict[str, IndexOps]:
    """Extract the index DDL a migration performs, KEYED BY DIALECT BRANCH.

    Parsed from the AST rather than by regex over the raw source, because the
    concurrent-index DDL is written as adjacent string literals that Python
    concatenates at parse time - a regex over the source text would see the
    statement split across two fragments and match neither. The AST hands back
    the already-joined constant.
    """
    source = (VERSIONS_DIR / migration_file).read_text()
    tree = ast.parse(source)
    func = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name),
        None,
    )
    assert func is not None, f"{migration_file} defines no {func_name}()"

    return {branch: _scan(body) for branch, body in _dialect_branches(func).items()}


def _orm_index_columns(table_name: str, index_name: str) -> tuple[str, ...] | None:
    """Ordered column names of an ORM ``Index``, or ``None`` if undeclared.

    Uses ``Index.expressions`` rather than ``Index.columns``: only the former
    is guaranteed to preserve declaration order, and for a composite index
    serving a sort, order is the entire point.
    """
    table = Base.metadata.tables[table_name]
    for index in table.indexes:
        if index.name == index_name:
            return tuple(c.name for c in index.expressions)
    return None


@pytest.mark.unit
class TestDocumentsCreatedAtIndexAgreement:
    """The ``documents`` sort-covering index must agree across ORM and migration.

    ``list_documents`` pins ``ORDER BY created_at DESC, id DESC``. The index
    that covers it is declared twice - once in the ORM (so autogenerate does
    not propose re-adding it) and once in the migration that builds it. If the
    two drift, nothing fails at runtime: the database still works, and the
    mismatch only surfaces later as a spurious autogenerate diff on an
    unrelated change. These assertions turn that into a red test instead.

    Column ORDER is asserted, not just membership. ``(namespace_id,
    created_at, id)`` covers the query; a permutation such as ``(namespace_id,
    id, created_at)`` does not, and a set comparison would wave it through.

    EVERY dialect branch is asserted independently. The migration emits the same
    index names through different spellings per dialect, and the embedded stack
    takes its schema from this chain and nothing else - so a branch that creates
    the wrong index, the wrong column order, or nothing at all is a real defect
    on that dialect even when its sibling branch is correct.
    """

    MIGRATION = "054_documents_namespace_created_at_id.py"
    NEW_INDEX = "ix_documents_namespace_created_at_id"
    OLD_INDEX = "ix_documents_namespace_created_at"
    EXPECTED_COLUMNS = ("namespace_id", "created_at", "id")

    def test_migration_is_branched_per_dialect(self):
        """Guard the guard: the per-branch assertions below need branches to exist.

        If the migration is ever collapsed to a single unbranched body, the
        parser yields one nameless branch and the loops below would assert
        against it once instead of twice - still correct, but quietly weaker
        than it reads. Pin the shape so that change is visible.
        """
        for func_name in ("upgrade", "downgrade"):
            branches = _index_ops(self.MIGRATION, func_name)
            assert set(branches) == {"postgresql", "other"}, (
                f"{self.MIGRATION} {func_name}() no longer splits per dialect; found branches {sorted(branches)}"
            )

    def test_orm_declares_the_sort_covering_index(self):
        """The ORM carries the 3-column index, in the covering order."""
        assert _orm_index_columns("documents", self.NEW_INDEX) == self.EXPECTED_COLUMNS

    def test_orm_no_longer_declares_the_superseded_index(self):
        """The 2-column index is gone from the ORM.

        Leaving both declared would have the ORM ask for two indexes sharing a
        prefix - write amplification on every document insert.
        """
        assert _orm_index_columns("documents", self.OLD_INDEX) is None

    def test_migration_creates_exactly_what_the_orm_declares(self):
        """Every branch of upgrade builds the ORM's index, same columns, same order."""
        for branch, ops in _index_ops(self.MIGRATION, "upgrade").items():
            created = ops.created
            assert self.NEW_INDEX in created, (
                f"{self.MIGRATION} upgrade() [{branch} branch] does not create {self.NEW_INDEX}; "
                f"it creates {sorted(created)}. A branch that skips the index leaves that dialect "
                "on the narrower one."
            )
            assert created[self.NEW_INDEX] == _orm_index_columns("documents", self.NEW_INDEX), (
                f"upgrade() [{branch} branch] disagrees with the ORM: "
                f"{created[self.NEW_INDEX]} vs {_orm_index_columns('documents', self.NEW_INDEX)}"
            )
            assert created[self.NEW_INDEX] == self.EXPECTED_COLUMNS, (
                f"upgrade() [{branch} branch] built {created[self.NEW_INDEX]}, "
                f"expected {self.EXPECTED_COLUMNS} - column order is what makes it cover the sort"
            )

    def test_migration_drops_the_superseded_index(self):
        """Every branch of upgrade removes the 2-column index the 3-column one subsumes."""
        for branch, ops in _index_ops(self.MIGRATION, "upgrade").items():
            dropped = ops.dropped
            assert self.OLD_INDEX in dropped, (
                f"upgrade() [{branch} branch] leaves {self.OLD_INDEX} in place; it shares a prefix "
                f"with {self.NEW_INDEX}, so both would be maintained on every insert"
            )

    def test_downgrade_restores_the_superseded_index(self):
        """Downgrade must put the 2-column index back, with its original columns.

        Not cosmetic. The migration that originally added the 2-column index
        drops it by an unconditional ``op.drop_index(...)`` - no
        ``if_exists=True`` - which errors if the index is absent - so a downgrade that walks past it would fail outright
        unless this migration restores it on the way down.

        Asserted per branch: the walk past that migration happens on whichever
        dialect the operator is running, so a restore present on only one of
        them leaves the other broken.
        """
        for branch, ops in _index_ops(self.MIGRATION, "downgrade").items():
            created, dropped = ops.created, ops.dropped
            assert created.get(self.OLD_INDEX) == ("namespace_id", "created_at"), (
                f"downgrade() [{branch} branch] must recreate {self.OLD_INDEX} on "
                f"(namespace_id, created_at); it creates {created}"
            )
            assert self.NEW_INDEX in dropped, f"downgrade() [{branch} branch] must remove the 3-column index"

    def test_replacement_index_is_built_before_the_old_one_is_dropped(self):
        """Create must precede drop, on every branch and in both directions.

        This is the migration's "never unindexed" invariant, and it is the whole
        reason the two statements are ordered rather than merged. Dropping first
        would leave ``get_last_activity_at()``'s ``MAX(created_at)`` with no
        index for the duration of a concurrent build - which on a table large
        enough to warrant a concurrent build is exactly when it matters. The
        column assertions above are indifferent to sequencing, so without this
        the invariant is documented in a docstring and enforced nowhere.
        """
        for func_name, new_first in (("upgrade", self.NEW_INDEX), ("downgrade", self.OLD_INDEX)):
            superseded = self.OLD_INDEX if func_name == "upgrade" else self.NEW_INDEX
            for branch, ops in _index_ops(self.MIGRATION, func_name).items():
                create_at = ops.index_of("create", new_first)
                drop_at = ops.index_of("drop", superseded)
                assert create_at is not None, f"{func_name}() [{branch} branch] never creates {new_first}"
                assert drop_at is not None, f"{func_name}() [{branch} branch] never drops {superseded}"
                assert create_at < drop_at, (
                    f"{func_name}() [{branch} branch] drops {superseded} before creating {new_first} "
                    f"(positions {drop_at} and {create_at}). The replacement must exist first, or the "
                    "namespace/created_at lookup runs unindexed for the length of the build."
                )

    def test_postgres_branch_builds_concurrently_inside_an_autocommit_block(self):
        """Both Postgres statements must be CONCURRENTLY, and both inside the block.

        A plain ``CREATE INDEX`` takes a lock that blocks writes on
        ``documents`` for the whole build, and ``CREATE INDEX CONCURRENTLY``
        cannot run inside a transaction at all - so the two properties are a
        pair, and losing either one silently converts a safe migration into an
        outage on a large table. Neither is visible to the column-level
        assertions above.

        Asserted only for the Postgres branch: concurrent builds are a Postgres
        feature, and the other branch deliberately uses plain DDL.
        """
        for func_name in ("upgrade", "downgrade"):
            postgres = _index_ops(self.MIGRATION, func_name)["postgresql"]
            assert postgres.operations, f"{func_name}() postgresql branch issues no index DDL"
            for op in postgres.operations:
                assert op.concurrent, (
                    f"{func_name}() postgresql branch runs a non-concurrent {op.action.upper()} on "
                    f"{op.name}; that locks out writes on documents for the length of the build"
                )
                assert op.in_autocommit_block, (
                    f"{func_name}() postgresql branch runs {op.action.upper()} {op.name} outside "
                    "op.get_context().autocommit_block(); CONCURRENTLY cannot run inside a transaction"
                )

    def test_non_postgres_branch_does_not_claim_concurrency(self):
        """The plain-DDL branch must stay plain.

        Guards the inverse of the test above: pasting the Postgres spelling into
        the other branch would emit SQL that SQLite cannot parse, and the
        embedded stack takes its schema from this chain and nothing else.
        """
        for func_name in ("upgrade", "downgrade"):
            other = _index_ops(self.MIGRATION, func_name)["other"]
            assert other.operations, f"{func_name}() non-postgres branch issues no index DDL"
            for op in other.operations:
                assert not op.concurrent, (
                    f"{func_name}() non-postgres branch marks {op.name} CONCURRENTLY; that is a Postgres-only feature"
                )


# ---------------------------------------------------------------------------
# create_tables() deprecation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateTablesDeprecation:
    """Verify that create_tables() emits a deprecation warning."""

    async def test_postgresql_backend_warns(self):
        """PostgreSQLBackend.create_tables() emits DeprecationWarning."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend("postgresql://localhost/test")
        mock_engine, _ = _make_mock_engine()
        backend._engine = mock_engine

        with patch("khora.storage.backends.postgresql.sync_enum_values", new_callable=AsyncMock):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await backend.create_tables()
                deprecation_warnings = [
                    x
                    for x in w
                    if issubclass(x.category, DeprecationWarning) and "create_tables() is deprecated" in str(x.message)
                ]
                assert len(deprecation_warnings) >= 1

    async def test_pgvector_backend_warns(self):
        """PgVectorBackend.create_tables() emits DeprecationWarning."""
        from khora.storage.backends.pgvector import PgVectorBackend

        backend = PgVectorBackend("postgresql://localhost/test")
        mock_engine, _ = _make_mock_engine()
        backend._engine = mock_engine

        with patch("khora.storage.backends.pgvector.sync_enum_values", new_callable=AsyncMock):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await backend.create_tables()
                assert len(w) >= 1
                deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deprecation_warnings) >= 1
                assert "create_tables() is deprecated" in str(deprecation_warnings[0].message)

    async def test_event_store_warns(self):
        """PostgreSQLEventStore.create_tables() emits DeprecationWarning."""
        from khora.storage.event_store import PostgreSQLEventStore

        store = PostgreSQLEventStore("postgresql://localhost/test")
        mock_engine, _ = _make_mock_engine()
        store._engine = mock_engine

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await store.create_tables()
            deprecation_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning) and "create_tables() is deprecated" in str(x.message)
            ]
            assert len(deprecation_warnings) >= 1

    async def test_init_db_warns(self):
        """init_db() emits DeprecationWarning."""
        from khora.db.session import DatabaseManager

        manager = DatabaseManager()
        mock_engine, _ = _make_mock_engine()

        with patch.object(manager, "get_engine", return_value=mock_engine):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await manager.init_db()
                deprecation_warnings = [
                    x
                    for x in w
                    if issubclass(x.category, DeprecationWarning) and "init_db() is deprecated" in str(x.message)
                ]
                assert len(deprecation_warnings) >= 1
