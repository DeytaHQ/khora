"""Unit tests for migration bundling (DYT-567).

Tests that Alembic migrations are bundled in the khora package and
can be run programmatically via run_migrations() / MemoryLake(run_migrations=True).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.db.session import MigrationResult, _run_migrations_sync, run_migrations
from khora.memory_lake import MemoryLake

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config() -> MagicMock:
    """Minimal mock KhoraConfig for MemoryLake tests."""
    cfg = MagicMock()
    cfg.database_url = "postgresql://localhost/testdb"
    cfg.llm.embedding_model = "text-embedding-3-small"
    return cfg


# ---------------------------------------------------------------------------
# MigrationResult dataclass
# ---------------------------------------------------------------------------


class TestMigrationResult:
    """Tests for the MigrationResult dataclass."""

    @pytest.mark.unit
    def test_construction(self):
        """All fields are set correctly on construction."""
        result = MigrationResult(
            success=True,
            current_revision="abc123",
            previous_revision="def456",
            migrations_run=3,
            elapsed_seconds=1.5,
            error=None,
        )
        assert result.success is True
        assert result.current_revision == "abc123"
        assert result.previous_revision == "def456"
        assert result.migrations_run == 3
        assert result.elapsed_seconds == 1.5
        assert result.error is None

    @pytest.mark.unit
    def test_default_error_is_none(self):
        """error field defaults to None when not provided."""
        result = MigrationResult(
            success=True,
            current_revision="abc",
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=0.1,
        )
        assert result.error is None

    @pytest.mark.unit
    def test_frozen(self):
        """MigrationResult is immutable (frozen=True)."""
        result = MigrationResult(
            success=True,
            current_revision=None,
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    @pytest.mark.unit
    def test_slots(self):
        """MigrationResult uses slots (no __dict__)."""
        result = MigrationResult(
            success=True,
            current_revision=None,
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=0.0,
        )
        assert not hasattr(result, "__dict__")

    @pytest.mark.unit
    def test_success_variant(self):
        """Typical success result."""
        result = MigrationResult(
            success=True,
            current_revision="head_rev",
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=2.3,
        )
        assert result.success is True
        assert result.error is None

    @pytest.mark.unit
    def test_failure_variant(self):
        """Typical failure result with error message."""
        result = MigrationResult(
            success=False,
            current_revision=None,
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=0.1,
            error="connection refused",
        )
        assert result.success is False
        assert result.error == "connection refused"


# ---------------------------------------------------------------------------
# _run_migrations_sync — no URL
# ---------------------------------------------------------------------------


class TestRunMigrationsSyncNoUrl:
    """Tests for _run_migrations_sync when no database URL is available."""

    @pytest.mark.unit
    def test_no_url_no_env(self, monkeypatch):
        """Returns failure when no URL passed and env var is empty."""
        monkeypatch.delenv("KHORA_DATABASE_URL", raising=False)
        result = _run_migrations_sync(None)
        assert result.success is False
        assert result.error is not None
        assert "database URL" in result.error.lower() or "KHORA_DATABASE_URL" in result.error


# ---------------------------------------------------------------------------
# _run_migrations_sync — programmatic config
# ---------------------------------------------------------------------------


class TestRunMigrationsSyncConfig:
    """Tests for _run_migrations_sync with a database URL."""

    @pytest.mark.unit
    @patch("alembic.command.upgrade")
    @patch("alembic.script.ScriptDirectory.from_config")
    @patch("alembic.config.Config")
    def test_programmatic_config(self, mock_config_cls, mock_from_config, mock_upgrade):
        """Verifies Config is built programmatically with correct options."""
        mock_cfg_instance = MagicMock()
        mock_config_cls.return_value = mock_cfg_instance
        mock_cfg_instance.attributes = {}

        mock_script = MagicMock()
        mock_script.get_current_head.return_value = "abc123"
        mock_from_config.return_value = mock_script

        result = _run_migrations_sync("postgresql://localhost/testdb")

        # Config() created without file path
        mock_config_cls.assert_called_once_with()

        # script_location points to db/migrations/
        migrations_dir = str(Path(__file__).parent.parent.parent / "src" / "khora" / "db" / "migrations")
        mock_cfg_instance.set_main_option.assert_any_call("script_location", migrations_dir)

        # URL passed via config.attributes
        assert mock_cfg_instance.attributes["database_url"] == "postgresql://localhost/testdb"

        # command.upgrade called
        mock_upgrade.assert_called_once_with(mock_cfg_instance, "head")

        assert result.success is True
        assert result.current_revision == "abc123"

    @pytest.mark.unit
    @patch("alembic.command.upgrade")
    @patch("alembic.script.ScriptDirectory.from_config")
    @patch("alembic.config.Config")
    def test_exception_handling(self, mock_config_cls, mock_from_config, mock_upgrade):
        """Returns failure MigrationResult when command.upgrade raises."""
        mock_cfg_instance = MagicMock()
        mock_config_cls.return_value = mock_cfg_instance
        mock_cfg_instance.attributes = {}

        mock_from_config.side_effect = RuntimeError("connection refused")

        result = _run_migrations_sync("postgresql://localhost/testdb")

        assert result.success is False
        assert result.error == "connection refused"
        assert result.elapsed_seconds >= 0


# ---------------------------------------------------------------------------
# run_migrations — async wrapper
# ---------------------------------------------------------------------------


class TestRunMigrationsAsync:
    """Tests for the async run_migrations wrapper."""

    @pytest.mark.unit
    async def test_delegates_to_sync(self):
        """run_migrations calls _run_migrations_sync with the database_url."""
        expected = MigrationResult(
            success=True,
            current_revision="abc",
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=0.5,
        )
        with patch(
            "khora.db.session._run_migrations_sync",
            return_value=expected,
        ) as mock_sync:
            result = await run_migrations("postgresql://localhost/testdb")

        mock_sync.assert_called_once_with("postgresql://localhost/testdb")
        assert result is expected

    @pytest.mark.unit
    async def test_delegates_none_url(self):
        """run_migrations passes None when no URL given."""
        expected = MigrationResult(
            success=False,
            current_revision=None,
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=0.0,
            error="No database URL",
        )
        with patch(
            "khora.db.session._run_migrations_sync",
            return_value=expected,
        ) as mock_sync:
            result = await run_migrations()

        mock_sync.assert_called_once_with(None)
        assert result.success is False


# ---------------------------------------------------------------------------
# MemoryLake.__init__ with run_migrations parameter
# ---------------------------------------------------------------------------


class TestMemoryLakeInitMigrations:
    """Tests for MemoryLake.__init__() run_migrations parameter."""

    @pytest.mark.unit
    def test_run_migrations_default_false(self):
        """run_migrations defaults to False."""
        with patch("khora.memory_lake.load_config", return_value=_mock_config()):
            lake = MemoryLake()
        assert lake._run_migrations is False

    @pytest.mark.unit
    def test_run_migrations_true(self):
        """run_migrations=True is stored on the instance."""
        with patch("khora.memory_lake.load_config", return_value=_mock_config()):
            lake = MemoryLake(run_migrations=True)
        assert lake._run_migrations is True


# ---------------------------------------------------------------------------
# MemoryLake.connect() — migration integration
# ---------------------------------------------------------------------------


class TestMemoryLakeConnectMigrations:
    """Tests for MemoryLake.connect() migration integration."""

    @pytest.mark.unit
    async def test_connect_runs_migrations_on_success(self):
        """When run_migrations=True, migrations run before engine creation."""
        migration_result = MigrationResult(
            success=True,
            current_revision="abc",
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=0.5,
        )
        mock_engine = MagicMock()
        mock_engine.connect = AsyncMock()

        call_order = []

        async def fake_run_migrations(url):
            call_order.append("migrations")
            return migration_result

        def fake_create_engine(*args, **kwargs):
            call_order.append("create_engine")
            return mock_engine

        with (
            patch("khora.memory_lake.load_config", return_value=_mock_config()),
            patch("khora.db.session.run_migrations", side_effect=fake_run_migrations),
            patch("khora.engines.create_engine", side_effect=fake_create_engine),
        ):
            lake = MemoryLake(run_migrations=True)
            await lake.connect()

        assert call_order == ["migrations", "create_engine"]
        assert lake._connected is True

    @pytest.mark.unit
    async def test_connect_raises_on_migration_failure(self):
        """When migrations fail, connect() raises RuntimeError."""
        migration_result = MigrationResult(
            success=False,
            current_revision=None,
            previous_revision=None,
            migrations_run=0,
            elapsed_seconds=0.1,
            error="migration table locked",
        )

        with (
            patch("khora.memory_lake.load_config", return_value=_mock_config()),
            patch("khora.db.session.run_migrations", AsyncMock(return_value=migration_result)),
        ):
            lake = MemoryLake(run_migrations=True)
            with pytest.raises(RuntimeError, match="migration table locked"):
                await lake.connect()

        assert lake._connected is False

    @pytest.mark.unit
    async def test_connect_skips_migrations_when_false(self):
        """When run_migrations=False (default), migrations are NOT called."""
        mock_engine = MagicMock()
        mock_engine.connect = AsyncMock()

        with (
            patch("khora.memory_lake.load_config", return_value=_mock_config()),
            patch("khora.engines.create_engine", return_value=mock_engine),
            patch("khora.db.session.run_migrations") as mock_mig,
        ):
            lake = MemoryLake()
            await lake.connect()

        mock_mig.assert_not_called()
        assert lake._connected is True


# ---------------------------------------------------------------------------
# db/__init__.py exports
# ---------------------------------------------------------------------------


class TestDbExports:
    """Tests that MigrationResult is importable from khora.db."""

    @pytest.mark.unit
    def test_migration_result_importable(self):
        """MigrationResult is exported from khora.db."""
        from khora.db import MigrationResult as MR

        assert MR is MigrationResult

    @pytest.mark.unit
    def test_run_migrations_importable(self):
        """run_migrations is exported from khora.db."""
        from khora.db import run_migrations as rm

        assert rm is run_migrations


# ---------------------------------------------------------------------------
# env.py module structure and migrations directory
# ---------------------------------------------------------------------------


class TestMigrationPackageStructure:
    """Tests for the bundled migration package structure."""

    @pytest.mark.unit
    def test_env_py_exists(self):
        """env.py exists in the migrations package."""
        env_path = Path(__file__).parent.parent.parent / "src" / "khora" / "db" / "migrations" / "env.py"
        assert env_path.exists(), f"env.py not found at {env_path}"

    @pytest.mark.unit
    def test_versions_dir_has_15_files(self):
        """versions/ directory contains 15 migration files."""
        versions_dir = Path(__file__).parent.parent.parent / "src" / "khora" / "db" / "migrations" / "versions"
        migration_files = sorted(versions_dir.glob("*.py"))
        # Filter out __pycache__ and __init__
        migration_files = [f for f in migration_files if not f.name.startswith("__")]
        assert len(migration_files) == 15, (
            f"Expected 15 migration files, found {len(migration_files)}: " f"{[f.name for f in migration_files]}"
        )

    @pytest.mark.unit
    def test_env_py_constants(self):
        """VERSION_TABLE and LOCK_ID constants are defined in env.py."""
        # Import indirectly to avoid triggering alembic context at module level
        env_path = Path(__file__).parent.parent.parent / "src" / "khora" / "db" / "migrations" / "env.py"
        source = env_path.read_text()
        assert "VERSION_TABLE" in source, "VERSION_TABLE constant not found in env.py"
        assert "LOCK_ID" in source, "LOCK_ID constant not found in env.py"

    @pytest.mark.unit
    def test_init_py_exists(self):
        """__init__.py exists making migrations a proper package."""
        init_path = Path(__file__).parent.parent.parent / "src" / "khora" / "db" / "migrations" / "__init__.py"
        assert init_path.exists(), f"__init__.py not found at {init_path}"
