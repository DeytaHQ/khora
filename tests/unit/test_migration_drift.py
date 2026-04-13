"""Tests that Alembic migrations stay in sync with ORM models.

These tests ensure that:
1. All migration .py source files are committed (not just .pyc)
2. ORM models and migrations produce the same schema (no drift)
3. create_tables() emits a deprecation warning
"""

from __future__ import annotations

import warnings
from pathlib import Path
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
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "create_tables() is deprecated" in str(w[0].message)

    async def test_init_db_warns(self):
        """init_db() emits DeprecationWarning."""
        from khora.db.session import DatabaseManager

        manager = DatabaseManager()
        mock_engine, _ = _make_mock_engine()

        with patch.object(manager, "get_engine", return_value=mock_engine):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await manager.init_db()
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "init_db() is deprecated" in str(w[0].message)
