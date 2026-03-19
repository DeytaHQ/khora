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


# ---------------------------------------------------------------------------
# env.py — _seed_version_table logic
# ---------------------------------------------------------------------------


def _load_env_functions():
    """Load env.py functions without triggering module-level Alembic side effects.

    env.py executes ``context.is_offline_mode()`` at import time (lines 150-153)
    and reads ``context.config`` at module level (line 20).  We mock the entire
    alembic.context *attribute* on the alembic package so ``from alembic import context``
    resolves to our mock, then force-reimport the env module.
    """
    import importlib
    import sys

    import alembic

    # Build a mock that satisfies all module-level access patterns
    mock_context = MagicMock()
    mock_context.config = MagicMock()
    mock_context.config.config_file_name = None
    mock_context.config.attributes = {}
    mock_context.is_offline_mode.return_value = False
    mock_context.configure = MagicMock()
    mock_context.begin_transaction = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    mock_context.run_migrations = MagicMock()

    # Patch alembic.context at both the attribute and sys.modules level
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
        # Restore original state
        sys.modules.pop(mod_name, None)
        if orig_env is not None:
            sys.modules[mod_name] = orig_env
        if orig_mod is not None:
            sys.modules["alembic.context"] = orig_mod
        elif "alembic.context" in sys.modules:
            del sys.modules["alembic.context"]
        if orig_attr is not None:
            alembic.context = orig_attr


class TestSeedVersionTable:
    """Tests for _seed_version_table in env.py."""

    @pytest.mark.unit
    def test_skips_when_already_seeded(self):
        """Skips seeding if khora_alembic_version already has rows."""
        env = _load_env_functions()
        conn = MagicMock()
        # First query: SELECT 1 FROM khora_alembic_version — has data
        conn.execute.return_value.fetchone.return_value = (1,)

        env._seed_version_table(conn)

        # Only one execute call (the check), no INSERT
        assert conn.execute.call_count == 1

    @pytest.mark.unit
    def test_seeds_from_old_table(self):
        """Copies revision from alembic_version when khora table is empty."""
        env = _load_env_functions()
        conn = MagicMock()

        call_count = [0]

        def fake_execute(stmt, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                # khora_alembic_version is empty
                result.fetchone.return_value = None
            elif call_count[0] == 2:
                # alembic_version has a revision
                result.fetchone.return_value = ("abc123",)
            return result

        conn.execute.side_effect = fake_execute

        env._seed_version_table(conn)

        # 3 calls: check khora table, check old table, INSERT
        assert conn.execute.call_count == 3

    @pytest.mark.unit
    def test_skips_when_old_table_missing(self):
        """Skips seeding when alembic_version table doesn't exist."""
        env = _load_env_functions()
        conn = MagicMock()

        call_count = [0]

        def fake_execute(stmt, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                # khora_alembic_version is empty
                result.fetchone.return_value = None
            elif call_count[0] == 2:
                # alembic_version doesn't exist
                raise Exception("relation does not exist")
            return result

        conn.execute.side_effect = fake_execute

        env._seed_version_table(conn)

        # Only 2 calls — no INSERT
        assert conn.execute.call_count == 2

    @pytest.mark.unit
    def test_skips_when_old_table_empty(self):
        """Skips seeding when alembic_version exists but is empty."""
        env = _load_env_functions()
        conn = MagicMock()

        call_count = [0]

        def fake_execute(stmt, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                # khora_alembic_version is empty
                result.fetchone.return_value = None
            elif call_count[0] == 2:
                # alembic_version exists but is empty
                result.fetchone.return_value = None
            return result

        conn.execute.side_effect = fake_execute

        env._seed_version_table(conn)

        # Only 2 calls — no INSERT
        assert conn.execute.call_count == 2


# ---------------------------------------------------------------------------
# env.py — _acquire_advisory_lock logic
# ---------------------------------------------------------------------------


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
        """Raises TimeoutError when lock cannot be acquired within timeout."""
        env = _load_env_functions()
        conn = MagicMock()
        conn.execute.return_value.scalar.return_value = False

        with patch("time.sleep"), pytest.raises(TimeoutError, match="advisory lock"):
            env._acquire_advisory_lock(conn, timeout=0.0)


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
