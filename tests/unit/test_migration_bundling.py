"""Unit tests for migration bundling (DYT-567).

Tests that Alembic migrations are bundled in the khora package and
can be run programmatically via run_migrations() / MemoryLake(run_migrations=True).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import khora.db.migrations
from khora.db.session import MigrationResult, _run_migrations_sync, run_migrations
from khora.memory_lake import MemoryLake

# Derive migrations directory from the installed package — not relative to this test file
_MIGRATIONS_DIR = Path(khora.db.migrations.__file__).parent

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
            target_revision="abc123",
            current_revision="abc123",
            elapsed_seconds=1.5,
            error=None,
        )
        assert result.success is True
        assert result.target_revision == "abc123"
        assert result.current_revision == "abc123"
        assert result.elapsed_seconds == 1.5
        assert result.error is None

    @pytest.mark.unit
    def test_default_error_is_none(self):
        """error field defaults to None when not provided."""
        result = MigrationResult(
            success=True,
            target_revision="abc",
            current_revision="abc",
            elapsed_seconds=0.1,
        )
        assert result.error is None

    @pytest.mark.unit
    def test_frozen(self):
        """MigrationResult is immutable (frozen=True)."""
        result = MigrationResult(
            success=True,
            target_revision=None,
            current_revision=None,
            elapsed_seconds=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    @pytest.mark.unit
    def test_slots(self):
        """MigrationResult uses slots (no __dict__)."""
        result = MigrationResult(
            success=True,
            target_revision=None,
            current_revision=None,
            elapsed_seconds=0.0,
        )
        assert not hasattr(result, "__dict__")

    @pytest.mark.unit
    def test_success_variant(self):
        """Typical success result."""
        result = MigrationResult(
            success=True,
            target_revision="head_rev",
            current_revision="head_rev",
            elapsed_seconds=2.3,
        )
        assert result.success is True
        assert result.error is None

    @pytest.mark.unit
    def test_failure_variant(self):
        """Typical failure result with error message."""
        result = MigrationResult(
            success=False,
            target_revision=None,
            current_revision=None,
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
        mock_cfg_instance.set_main_option.assert_any_call("script_location", str(_MIGRATIONS_DIR))

        # URL passed via config.attributes
        assert mock_cfg_instance.attributes["database_url"] == "postgresql://localhost/testdb"

        # command.upgrade called
        mock_upgrade.assert_called_once_with(mock_cfg_instance, "head")

        assert result.success is True
        assert result.target_revision == "abc123"
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
        assert result.error == "RuntimeError: connection refused"
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
            target_revision="abc",
            current_revision="abc",
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
            target_revision=None,
            current_revision=None,
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
            target_revision="abc",
            current_revision="abc",
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
            target_revision=None,
            current_revision=None,
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
        env_path = _MIGRATIONS_DIR / "env.py"
        assert env_path.exists(), f"env.py not found at {env_path}"

    @pytest.mark.unit
    def test_versions_dir_has_15_files(self):
        """versions/ directory contains 15 migration files."""
        versions_dir = _MIGRATIONS_DIR / "versions"
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
        env_path = _MIGRATIONS_DIR / "env.py"
        source = env_path.read_text()
        assert "VERSION_TABLE" in source, "VERSION_TABLE constant not found in env.py"
        assert "LOCK_ID" in source, "LOCK_ID constant not found in env.py"

    @pytest.mark.unit
    def test_init_py_exists(self):
        """__init__.py exists making migrations a proper package."""
        init_path = _MIGRATIONS_DIR / "__init__.py"
        assert init_path.exists(), f"__init__.py not found at {init_path}"


# ---------------------------------------------------------------------------
# env.py — _acquire_advisory_lock logic
# ---------------------------------------------------------------------------


def _load_env_functions():
    """Load env.py functions without triggering module-level Alembic side effects."""
    import importlib
    import sys

    import alembic

    mock_context = MagicMock()
    mock_context.config = MagicMock()
    mock_context.config.config_file_name = None
    mock_context.config.attributes = {}
    mock_context.is_offline_mode.return_value = False
    mock_context.configure = MagicMock()
    mock_context.begin_transaction = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    mock_context.run_migrations = MagicMock()

    orig_attr = getattr(alembic, "context", None)
    orig_mod = sys.modules.get("alembic.context")
    alembic.context = mock_context
    sys.modules["alembic.context"] = mock_context

    mod_name = "khora.db.migrations.env"
    orig_env = sys.modules.pop(mod_name, None)

    try:
        with patch("asyncio.run"):
            mod = importlib.import_module(mod_name)
        return mod
    finally:
        sys.modules.pop(mod_name, None)
        if orig_env is not None:
            sys.modules[mod_name] = orig_env
        if orig_mod is not None:
            sys.modules["alembic.context"] = orig_mod
        elif "alembic.context" in sys.modules:
            del sys.modules["alembic.context"]
        if orig_attr is not None:
            alembic.context = orig_attr


class TestAcquireAdvisoryLock:
    """Tests for _acquire_advisory_lock in env.py."""

    @pytest.mark.unit
    def test_acquires_immediately(self):
        """Lock acquired on first try returns immediately."""
        env = _load_env_functions()
        conn = MagicMock()
        conn.execute.return_value.scalar.return_value = True

        env._acquire_advisory_lock(conn)

        conn.execute.assert_called_once()

    @pytest.mark.unit
    def test_retries_then_acquires(self):
        """Retries when lock not available, succeeds on subsequent attempt."""
        env = _load_env_functions()
        conn = MagicMock()
        # First call: not acquired. Second call: acquired.
        conn.execute.return_value.scalar.side_effect = [False, True]

        with patch("time.sleep"):
            env._acquire_advisory_lock(conn, timeout=60.0)

        assert conn.execute.call_count == 2

    @pytest.mark.unit
    def test_timeout_raises(self):
        """Raises ValueError when timeout is non-positive."""
        env = _load_env_functions()
        conn = MagicMock()

        with pytest.raises(ValueError, match="timeout must be positive"):
            env._acquire_advisory_lock(conn, timeout=0.0)

    @pytest.mark.unit
    def test_deadline_exceeded_raises(self):
        """Raises TimeoutError when lock cannot be acquired before deadline."""
        env = _load_env_functions()
        conn = MagicMock()
        conn.execute.return_value.scalar.return_value = False

        # First monotonic() sets deadline, second monotonic() exceeds it
        with patch("time.sleep"), patch("time.monotonic", side_effect=[0.0, 61.0]):
            with pytest.raises(TimeoutError, match="advisory lock"):
                env._acquire_advisory_lock(conn, timeout=60.0)


# ---------------------------------------------------------------------------
# env.py — _get_url() URL normalization
# ---------------------------------------------------------------------------


class TestGetUrl:
    """Tests for _get_url() URL normalization in env.py."""

    @pytest.mark.unit
    def test_postgresql_normalized_to_asyncpg(self):
        """postgresql:// is rewritten to postgresql+asyncpg://."""
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgresql://host/db"
        assert env._get_url() == "postgresql+asyncpg://host/db"

    @pytest.mark.unit
    def test_postgres_normalized_to_asyncpg(self):
        """postgres:// (Heroku-style) is rewritten to postgresql+asyncpg://."""
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgres://host/db"
        assert env._get_url() == "postgresql+asyncpg://host/db"

    @pytest.mark.unit
    def test_asyncpg_url_passes_through(self):
        """Already-normalized postgresql+asyncpg:// passes through unchanged."""
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgresql+asyncpg://host/db"
        assert env._get_url() == "postgresql+asyncpg://host/db"

    @pytest.mark.unit
    def test_no_url_raises_value_error(self, monkeypatch):
        """Raises ValueError when no URL is available."""
        monkeypatch.delenv("KHORA_DATABASE_URL", raising=False)
        env = _load_env_functions()
        env.config.attributes.clear()
        with pytest.raises(ValueError, match="No database URL"):
            env._get_url()


# ---------------------------------------------------------------------------
# env.py — run_migrations_offline()
# ---------------------------------------------------------------------------


class TestRunMigrationsOffline:
    """Tests for run_migrations_offline() in env.py."""

    @pytest.mark.unit
    def test_configures_with_version_table(self):
        """Offline mode calls context.configure with version_table."""
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgresql://host/db"

        env.run_migrations_offline()

        env.context.configure.assert_called_once()
        call_kwargs = env.context.configure.call_args[1]
        assert call_kwargs["version_table"] == env.VERSION_TABLE

    @pytest.mark.unit
    def test_runs_migrations_in_transaction(self):
        """Offline mode calls begin_transaction and run_migrations."""
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgresql://host/db"

        env.run_migrations_offline()

        env.context.begin_transaction.assert_called_once()
        env.context.run_migrations.assert_called_once()


# ---------------------------------------------------------------------------
# env.py — run_async_migrations()
# ---------------------------------------------------------------------------


class TestRunAsyncMigrations:
    """Tests for run_async_migrations() in env.py."""

    @staticmethod
    def _mock_async_engine():
        """Build a mock async engine with proper async context manager for connect()."""
        mock_conn = AsyncMock()
        # connect() returns an async context manager that yields mock_conn
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_conn
        mock_cm.__aexit__.return_value = False

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_cm
        mock_engine.dispose = AsyncMock()
        return mock_engine, mock_conn

    @pytest.mark.unit
    async def test_creates_engine_with_null_pool(self):
        """Async migrations create engine with NullPool."""
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgresql://host/db"

        mock_engine, _ = self._mock_async_engine()

        with patch.object(env, "create_async_engine", return_value=mock_engine) as mock_create:
            await env.run_async_migrations()

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs[1]["poolclass"] is env.pool.NullPool

    @pytest.mark.unit
    async def test_runs_sync_and_disposes(self):
        """Async migrations call connection.run_sync and engine.dispose."""
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgresql://host/db"

        mock_engine, mock_conn = self._mock_async_engine()

        with patch.object(env, "create_async_engine", return_value=mock_engine):
            await env.run_async_migrations()

        mock_conn.run_sync.assert_called_once_with(env.do_run_migrations)
        mock_engine.dispose.assert_awaited_once()


# ---------------------------------------------------------------------------
# MemoryLake.connect() — already connected (idempotent)
# ---------------------------------------------------------------------------


class TestMemoryLakeConnectIdempotent:
    """Tests for MemoryLake.connect() idempotency."""

    @pytest.mark.unit
    async def test_connect_already_connected_is_noop(self):
        """Calling connect() when already connected is a no-op."""
        mock_engine = MagicMock()
        mock_engine.connect = AsyncMock()

        with (
            patch("khora.memory_lake.load_config", return_value=_mock_config()),
            patch("khora.engines.create_engine", return_value=mock_engine) as mock_create,
        ):
            lake = MemoryLake(run_migrations=True)
            # Simulate already connected
            lake._connected = True
            lake._engine = mock_engine

            await lake.connect()

        # create_engine should NOT have been called (short-circuited)
        mock_create.assert_not_called()
        # engine.connect should NOT have been called again
        mock_engine.connect.assert_not_called()
