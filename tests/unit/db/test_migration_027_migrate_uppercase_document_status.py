"""DYT-3736: ``027_migrate_uppercase_document_status``.

Pins the three branches of the migration:

1. SQLite is a no-op (no ENUM type, nothing to fix).
2. PostgreSQL executes an UPDATE that lowercases uppercase status rows.
3. Downgrade is a no-op (cannot recover original uppercase state).
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def migration():
    return importlib.import_module("khora.db.migrations.versions.027_migrate_uppercase_document_status")


class TestRevisionMetadata:
    def test_revision_id(self, migration) -> None:
        assert migration.revision == "027_migrate_uppercase_document_status"

    def test_down_revision(self, migration) -> None:
        assert migration.down_revision == "026_widen_alembic_version_column"


class TestUpgrade:
    def test_sqlite_is_noop(self, migration) -> None:
        """SQLite has no ENUM type — return early without any SQL."""
        bind = MagicMock()
        bind.dialect.name = "sqlite"
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.upgrade()
        exec_mock.assert_not_called()

    def test_postgres_executes_update(self, migration) -> None:
        """PostgreSQL path issues an UPDATE to lowercase uppercase status rows."""
        bind = MagicMock()
        bind.dialect.name = "postgresql"
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.upgrade()
        exec_mock.assert_called_once()

    def test_postgres_update_sql_is_correct(self, migration) -> None:
        """Verify the UPDATE uses LOWER() cast and filters on uppercase chars."""
        bind = MagicMock()
        bind.dialect.name = "postgresql"
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.upgrade()
        sql = str(exec_mock.call_args.args[0])
        assert "UPDATE documents" in sql
        assert "LOWER(status::text)::document_status" in sql
        assert "[A-Z]" in sql


class TestDowngrade:
    def test_is_noop(self, migration) -> None:
        """Downgrade cannot restore original uppercase values — it is a no-op."""
        bind = MagicMock()
        bind.dialect.name = "postgresql"
        with (
            patch.object(migration.op, "get_bind", return_value=bind),
            patch.object(migration.op, "execute") as exec_mock,
        ):
            migration.downgrade()
        exec_mock.assert_not_called()
