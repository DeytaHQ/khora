"""``026_widen_alembic_version_column``.

Pins the four branches of the migration so a future Alembic refactor
doesn't silently re-introduce VARCHAR(32) and break long revision IDs:

1. SQLite is a no-op (TEXT has no width).
2. Postgres on a fresh DB (column missing) is idempotent.
3. Postgres at-or-above 64 chars is idempotent.
4. Postgres below 64 issues the ALTER.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def migration():
    return importlib.import_module("khora.db.migrations.versions.026_widen_alembic_version_column")


class TestRevisionMetadata:
    def test_revision_id(self, migration) -> None:
        assert migration.revision == "026_widen_alembic_version_column"

    def test_down_revision(self, migration) -> None:
        assert migration.down_revision == "025_add_document_extraction_params"

    def test_target_width(self, migration) -> None:
        assert migration.TARGET_WIDTH == 64
        assert migration.VERSION_TABLE == "khora_alembic_version"


class TestUpgrade:
    def test_sqlite_is_noop(self, migration) -> None:
        """SQLite has no fixed-width VARCHAR — return early without any SQL."""
        bind = MagicMock()
        bind.dialect.name = "sqlite"
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.upgrade()
        exec_mock.assert_not_called()

    def test_postgres_fresh_db_is_noop(self, migration) -> None:
        """When the column is absent (current_width is None), do nothing."""
        bind = MagicMock()
        bind.dialect.name = "postgresql"
        bind.execute.return_value.scalar.return_value = None
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.upgrade()
        exec_mock.assert_not_called()

    def test_postgres_already_wide_is_noop(self, migration) -> None:
        """If existing width >= 64, the migration does not re-issue ALTER."""
        bind = MagicMock()
        bind.dialect.name = "postgresql"
        bind.execute.return_value.scalar.return_value = 64
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.upgrade()
        exec_mock.assert_not_called()

    def test_postgres_widens_when_narrow(self, migration) -> None:
        """If width < 64, issue ALTER COLUMN ... TYPE VARCHAR(64)."""
        bind = MagicMock()
        bind.dialect.name = "postgresql"
        bind.execute.return_value.scalar.return_value = 32
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.upgrade()
        exec_mock.assert_called_once()
        sql = str(exec_mock.call_args.args[0])
        assert "ALTER TABLE khora_alembic_version" in sql
        assert "VARCHAR(64)" in sql


class TestDowngrade:
    def test_sqlite_is_noop(self, migration) -> None:
        bind = MagicMock()
        bind.dialect.name = "sqlite"
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.downgrade()
        exec_mock.assert_not_called()

    def test_postgres_narrows_to_32(self, migration) -> None:
        bind = MagicMock()
        bind.dialect.name = "postgresql"
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.downgrade()
        exec_mock.assert_called_once()
        sql = str(exec_mock.call_args.args[0])
        assert "VARCHAR(32)" in sql
