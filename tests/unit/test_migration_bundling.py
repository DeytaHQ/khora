"""Unit tests for migration bundling.

Tests that Alembic migrations are bundled in the khora package and
can be run programmatically via run_migrations() / Khora(run_migrations=True).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import khora.db.migrations
from khora.db.session import MigrationResult, _DatabaseAheadError, _run_migrations_sync, run_migrations
from khora.khora import Khora

# Derive migrations directory from the installed package — not relative to this test file
_MIGRATIONS_DIR = Path(khora.db.migrations.__file__).parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config() -> MagicMock:
    """Minimal mock KhoraConfig for Khora tests."""
    from pydantic import SecretStr

    cfg = MagicMock()
    # database_url is SecretStr on KhoraConfig; tests must mock the same type
    # so unwrap call sites (Khora.connect()) don't AttributeError.
    cfg.database_url = SecretStr("postgresql://localhost/testdb")
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
        assert result.skipped is False
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
    def test_default_skipped_is_false(self):
        """skipped field defaults to False when not provided."""
        result = MigrationResult(
            success=True,
            target_revision="abc",
            current_revision="abc",
            elapsed_seconds=0.1,
        )
        assert result.skipped is False

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
        assert result.skipped is False
        assert result.error is None

    @pytest.mark.unit
    def test_skipped_variant(self):
        """Typical skipped result when DB is ahead."""
        result = MigrationResult(
            success=True,
            target_revision="head_rev",
            current_revision=None,
            elapsed_seconds=0.5,
            skipped=True,
        )
        assert result.success is True
        assert result.skipped is True
        assert result.current_revision is None
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
# Khora.__init__ with run_migrations parameter
# ---------------------------------------------------------------------------


class TestKhoraInitMigrations:
    """Tests for Khora.__init__() run_migrations parameter."""

    @pytest.mark.unit
    def test_run_migrations_default_false(self):
        """run_migrations defaults to False."""
        with patch("khora.khora.load_config", return_value=_mock_config()):
            kb = Khora()
        assert kb._run_migrations is False

    @pytest.mark.unit
    def test_run_migrations_true(self):
        """run_migrations=True is stored on the instance."""
        with patch("khora.khora.load_config", return_value=_mock_config()):
            kb = Khora(run_migrations=True)
        assert kb._run_migrations is True


# ---------------------------------------------------------------------------
# Khora.connect() — migration integration
# ---------------------------------------------------------------------------


class TestKhoraConnectMigrations:
    """Tests for Khora.connect() migration integration."""

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
            patch("khora.khora.load_config", return_value=_mock_config()),
            patch("khora.db.session.run_migrations", side_effect=fake_run_migrations),
            patch("khora.engines.create_engine", side_effect=fake_create_engine),
        ):
            kb = Khora(run_migrations=True)
            await kb.connect()

        assert call_order == ["migrations", "create_engine"]
        assert kb._connected is True

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
            patch("khora.khora.load_config", return_value=_mock_config()),
            patch("khora.db.session.run_migrations", AsyncMock(return_value=migration_result)),
        ):
            kb = Khora(run_migrations=True)
            with pytest.raises(RuntimeError, match="migration table locked"):
                await kb.connect()

        assert kb._connected is False

    @pytest.mark.unit
    async def test_connect_skips_migrations_when_false(self):
        """When run_migrations=False (default), migrations are NOT called."""
        mock_engine = MagicMock()
        mock_engine.connect = AsyncMock()

        with (
            patch("khora.khora.load_config", return_value=_mock_config()),
            patch("khora.engines.create_engine", return_value=mock_engine),
            patch("khora.db.session.run_migrations") as mock_mig,
        ):
            kb = Khora()
            await kb.connect()

        mock_mig.assert_not_called()
        assert kb._connected is True

    @pytest.mark.unit
    async def test_connect_skips_migrations_on_surrealdb_backend(self):
        """run_migrations=True is a no-op on backend=surrealdb (#713).

        SurrealDB has no Alembic chain (declarative schema via
        ``DEFINE IF NOT EXISTS``), so the migration runner must not fire
        on this backend even when the caller passes ``run_migrations=True``.
        Pre-fix, Khora.connect() unconditionally invoked the Postgres-shaped
        runner and raised ``RuntimeError: Database migration failed: No
        database URL.`` (or surfaced a stray ``KHORA_DATABASE_URL`` against
        a backend the user never configured).
        """
        cfg = _mock_config()
        cfg.database_url = None  # No PG URL — mirrors the reported repro
        cfg.storage.backend = "surrealdb"

        mock_engine = MagicMock()
        mock_engine.connect = AsyncMock()

        with (
            patch("khora.khora.load_config", return_value=cfg),
            patch("khora.engines.create_engine", return_value=mock_engine),
            patch("khora.db.session.run_migrations") as mock_mig,
        ):
            kb = Khora(run_migrations=True)
            await kb.connect()

        mock_mig.assert_not_called()
        assert kb._connected is True


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
    def test_versions_dir_has_37_files(self):
        """versions/ directory contains 37 migration files."""
        versions_dir = _MIGRATIONS_DIR / "versions"
        migration_files = sorted(versions_dir.glob("*.py"))
        # Filter out __pycache__ and __init__
        migration_files = [f for f in migration_files if not f.name.startswith("__")]
        assert len(migration_files) == 37, (
            f"Expected 37 migration files, found {len(migration_files)}: {[f.name for f in migration_files]}"
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
    mock_context.begin_transaction = MagicMock(
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
    )
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

        with patch("time.sleep") as mock_sleep, patch("random.uniform", return_value=0.05) as mock_uniform:
            env._acquire_advisory_lock(conn, timeout=60.0)

        assert conn.execute.call_count == 2
        # One retry → one sleep with jitter
        mock_sleep.assert_called_once()
        # attempt=0: high = min(2.0, 0.05 * 2^0) = 0.05
        mock_uniform.assert_called_once_with(0.05, 0.05)

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

        # monotonic() calls: (1) deadline=0+60=60, (2) check=30 (under deadline, retry),
        # (3) check=61 (over deadline, raise)
        with (
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 30.0, 61.0]),
            patch("random.uniform", return_value=0.05) as mock_uniform,
        ):
            with pytest.raises(TimeoutError, match="advisory lock"):
                env._acquire_advisory_lock(conn, timeout=60.0)

        # One retry before deadline exceeded → one uniform call
        # attempt=0: high = min(2.0, 0.05 * 2^0) = 0.05
        mock_uniform.assert_called_once_with(0.05, 0.05)

    @pytest.mark.unit
    def test_backoff_increases_exponentially(self):
        """Successive retries increase the upper bound exponentially, capped at max_delay."""
        env = _load_env_functions()
        conn = MagicMock()
        # Lock never acquired — 8 attempts then deadline exceeded
        conn.execute.return_value.scalar.return_value = False

        min_delay, max_delay = 0.05, 2.0
        num_retries = 8
        # monotonic: first call sets deadline, then num_retries checks under deadline,
        # then one final check that exceeds deadline
        mono_values = [0.0] + [1.0] * num_retries + [61.0]

        with (
            patch("time.sleep"),
            patch("time.monotonic", side_effect=mono_values),
            patch("random.uniform", return_value=0.01) as mock_uniform,
        ):
            with pytest.raises(TimeoutError):
                env._acquire_advisory_lock(conn, timeout=60.0, min_delay=min_delay, max_delay=max_delay)

        # Verify the upper bounds passed to random.uniform increase exponentially
        expected_highs = []
        for attempt in range(num_retries):
            high = min(max_delay, min_delay * (2**attempt))
            expected_highs.append(high)

        assert mock_uniform.call_count == num_retries
        for i, call in enumerate(mock_uniform.call_args_list):
            assert call == ((min_delay, expected_highs[i]),), (
                f"attempt {i}: expected uniform({min_delay}, {expected_highs[i]}), got uniform{call}"
            )

        # Verify the cap kicks in: attempts 6 and 7 should both be capped at max_delay
        assert expected_highs[6] == max_delay
        assert expected_highs[7] == max_delay
        # And earlier attempts are strictly less
        assert expected_highs[5] < max_delay

    @pytest.mark.unit
    def test_custom_delay_params(self):
        """Custom min_delay/max_delay are respected in random.uniform bounds."""
        env = _load_env_functions()
        conn = MagicMock()
        # Lock never acquired — 4 attempts then deadline exceeded
        conn.execute.return_value.scalar.return_value = False

        min_delay, max_delay = 0.1, 0.5
        num_retries = 4
        mono_values = [0.0] + [1.0] * num_retries + [61.0]

        with (
            patch("time.sleep"),
            patch("time.monotonic", side_effect=mono_values),
            patch("random.uniform", return_value=0.05) as mock_uniform,
        ):
            with pytest.raises(TimeoutError):
                env._acquire_advisory_lock(
                    conn,
                    timeout=60.0,
                    min_delay=min_delay,
                    max_delay=max_delay,
                )

        # Expected upper bounds with custom params:
        # attempt 0: min(0.5, 0.1 * 1) = 0.1
        # attempt 1: min(0.5, 0.1 * 2) = 0.2
        # attempt 2: min(0.5, 0.1 * 4) = 0.4
        # attempt 3: min(0.5, 0.1 * 8) = 0.5 (capped)
        expected_highs = [0.1, 0.2, 0.4, 0.5]

        assert mock_uniform.call_count == num_retries
        for i, call in enumerate(mock_uniform.call_args_list):
            assert call == ((min_delay, expected_highs[i]),), (
                f"attempt {i}: expected uniform({min_delay}, {expected_highs[i]}), got uniform{call}"
            )

        # Verify cap applied on last attempt
        assert expected_highs[-1] == max_delay

    @pytest.mark.unit
    def test_min_delay_gte_max_delay_raises(self):
        """Raises ValueError when min_delay >= max_delay."""
        env = _load_env_functions()
        conn = MagicMock()

        with pytest.raises(ValueError, match="min_delay"):
            env._acquire_advisory_lock(conn, min_delay=1.0, max_delay=0.5)

    @pytest.mark.unit
    def test_overflow_falls_back_to_max_delay(self):
        """OverflowError in exponentiation falls back to max_delay as upper bound."""
        env = _load_env_functions()
        conn = MagicMock()
        conn.execute.return_value.scalar.return_value = False

        min_delay, max_delay = 0.05, 2.0

        # Use a float subclass whose __mul__ raises OverflowError, simulating
        # what happens when min_delay * (2 ** attempt) overflows.
        class OverflowFloat(float):
            def __mul__(self, other):
                raise OverflowError("simulated overflow")

            def __rmul__(self, other):
                raise OverflowError("simulated overflow")

        overflow_min_delay = OverflowFloat(min_delay)

        # 1 retry then deadline exceeded
        mono_values = [0.0, 1.0, 61.0]

        with (
            patch("time.sleep"),
            patch("time.monotonic", side_effect=mono_values),
            patch("random.uniform", return_value=0.05) as mock_uniform,
        ):
            with pytest.raises(TimeoutError):
                env._acquire_advisory_lock(
                    conn,
                    timeout=60.0,
                    min_delay=overflow_min_delay,
                    max_delay=max_delay,
                )

        # The overflow fallback should use max_delay as the high bound
        mock_uniform.assert_called_once_with(overflow_min_delay, max_delay)


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

    @pytest.mark.unit
    async def test_dsn_redaction_suppresses_cause_chain(self):
        """TypeError fallback raises RuntimeError from None — __cause__ must not leak DSN.

        When exception type reconstruction raises TypeError (e.g. NoSuchModuleError),
        the fallback RuntimeError must use ``from None`` so the unredacted DSN is not
        retained in __cause__ and captured by Logfire / Sentry / traceback.print_exception.
        """
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgresql://user:secret@host/db"

        class _MultiArgError(Exception):
            """Simulates an exception type that requires >1 positional arg."""

            def __init__(self, name: str, extra: str) -> None:
                super().__init__(name, extra)

        dsn_error = _MultiArgError(
            "postgresql://user:secret@host/db",
            "some extra arg",
        )
        mock_engine, mock_conn = self._mock_async_engine()
        mock_conn.run_sync.side_effect = dsn_error

        with patch.object(env, "logger") as mock_logger:
            with patch.object(env, "create_async_engine", return_value=mock_engine):
                with pytest.raises(RuntimeError) as exc_info:
                    await env.run_async_migrations()

        raised = exc_info.value
        # __cause__ must be None — DSN must not leak via exception chain
        assert raised.__cause__ is None, "__cause__ must be suppressed (from None)"
        # The redacted message must not contain the plaintext credential
        assert "secret" not in str(raised)
        # logger.debug must only log the type name, not the exception message or DSN
        mock_logger.debug.assert_called_once()
        debug_args = " ".join(str(a) for a in mock_logger.debug.call_args[0])
        assert "secret" not in debug_args, "logger.debug must not log unredacted DSN"

    @pytest.mark.unit
    async def test_dsn_redaction_normal_exception_type_preserved(self):
        """When exception type can be reconstructed, original type is preserved with redacted message."""
        env = _load_env_functions()
        env.config.attributes["database_url"] = "postgresql://user:secret@host/db"

        dsn_error = ValueError("postgresql://user:secret@host/db")
        mock_engine, mock_conn = self._mock_async_engine()
        mock_conn.run_sync.side_effect = dsn_error

        with patch.object(env, "create_async_engine", return_value=mock_engine):
            with pytest.raises(ValueError) as exc_info:
                await env.run_async_migrations()

        raised = exc_info.value
        assert raised.__cause__ is None
        assert "secret" not in str(raised)


# ---------------------------------------------------------------------------
# env.py — do_run_migrations ahead-detection
# ---------------------------------------------------------------------------


class TestDoRunMigrationsAheadDetection:
    """Tests for ahead-detection logic in do_run_migrations()."""

    def _setup_conn(self, version_num: str | None, *, table_exists: bool = True) -> MagicMock:
        """Create a mock connection that returns the given version for the advisory
        lock query, information_schema existence check, version-num width check,
        and (optionally) the version table query.

        Execute call order after the fix:
          1. pg_try_advisory_xact_lock         → scalar() True
          2. information_schema.tables check   → scalar() table_exists
          3. information_schema.columns width  → scalar() 64 (only when table_exists)
          4. SELECT version_num                → fetchone() row/None (only when table_exists)
        """
        conn = MagicMock()
        # do_run_migrations branches on dialect.name — simulate Postgres.
        conn.dialect.name = "postgresql"

        lock_result = MagicMock()
        lock_result.scalar.return_value = True

        pg_catalog_result = MagicMock()
        pg_catalog_result.scalar.return_value = table_exists

        if table_exists:
            # Width check returns 64 — already wide enough, no ALTER issued.
            width_result = MagicMock()
            width_result.scalar.return_value = 64

            version_result = MagicMock()
            if version_num is not None:
                # Row behaves like a sequence/tuple: row[0] should return version_num.
                version_result.fetchone.return_value = (version_num,)
            else:
                version_result.fetchone.return_value = None
            conn.execute.side_effect = [lock_result, pg_catalog_result, width_result, version_result]
        else:
            conn.execute.side_effect = [lock_result, pg_catalog_result]

        return conn

    @pytest.mark.unit
    def test_skips_when_db_revision_unknown(self):
        """Raises _DatabaseAheadError when DB revision is not in the known set."""
        env = _load_env_functions()
        conn = self._setup_conn("unknown_rev_abc")

        mock_rev = MagicMock()
        mock_rev.revision = "known_rev_1"
        mock_script_dir = MagicMock()
        mock_script_dir.walk_revisions.return_value = [mock_rev]

        with patch.object(env.ScriptDirectory, "from_config", return_value=mock_script_dir):
            with pytest.raises(_DatabaseAheadError) as exc_info:
                env.do_run_migrations(conn)

        assert exc_info.value.current_revision == "unknown_rev_abc"
        env.context.run_migrations.assert_not_called()

    @pytest.mark.unit
    def test_proceeds_when_revision_known(self):
        """Runs migrations normally when DB revision is in the known set."""
        env = _load_env_functions()
        conn = self._setup_conn("known_rev")

        mock_rev = MagicMock()
        mock_rev.revision = "known_rev"
        mock_script_dir = MagicMock()
        mock_script_dir.walk_revisions.return_value = [mock_rev]

        with patch.object(env.ScriptDirectory, "from_config", return_value=mock_script_dir):
            env.do_run_migrations(conn)

        env.context.run_migrations.assert_called_once()

    @pytest.mark.unit
    def test_proceeds_when_no_current_revision(self):
        """Runs migrations normally when version table exists but is empty (no rows)."""
        env = _load_env_functions()
        conn = self._setup_conn(None, table_exists=True)

        env.do_run_migrations(conn)

        env.context.run_migrations.assert_called_once()

    @pytest.mark.unit
    def test_proceeds_when_version_table_absent(self):
        """Runs migrations normally on a fresh DB where the version table doesn't exist.

        Previously this would leave the PostgreSQL transaction in ABORTED state
        (InFailedSQLTransactionError) because querying a missing table inside an
        explicit transaction aborts it. The fix checks information_schema.tables first
        so no statement ever fails inside the transaction.
        """
        env = _load_env_functions()
        conn = self._setup_conn(None, table_exists=False)

        env.do_run_migrations(conn)

        # Exactly 2 execute calls: advisory lock + information_schema check — no version query
        assert conn.execute.call_count == 2
        # Verify the second call is the information_schema existence check by inspecting its
        # bound parameters — more reliable than parsing the SQL text() object string
        second_call_params = conn.execute.call_args_list[1][0][1]
        assert second_call_params == {"table": env.VERSION_TABLE}
        env.context.run_migrations.assert_called_once()

    @pytest.mark.unit
    def test_proceeds_when_version_select_fails(self):
        """Runs migrations normally when the version table exists but SELECT fails.

        Covers M1: version SELECT failure (e.g. permission denied) after table presence
        confirmed. The exception is swallowed and ahead-detection is skipped so that
        migrations can still proceed.
        """
        env = _load_env_functions()
        conn = MagicMock()
        conn.dialect.name = "postgresql"

        lock_result = MagicMock()
        lock_result.scalar.return_value = True

        existence_result = MagicMock()
        existence_result.scalar.return_value = True  # table exists

        # Width check: column already wide enough — no ALTER.
        width_result = MagicMock()
        width_result.scalar.return_value = 64

        # Fourth execute (version SELECT) raises an error
        conn.execute.side_effect = [lock_result, existence_result, width_result, Exception("permission denied")]

        env.do_run_migrations(conn)

        env.context.run_migrations.assert_called_once()

    @pytest.mark.unit
    def test_propagates_walk_revisions_error(self):
        """Exception from walk_revisions() propagates unhandled."""
        env = _load_env_functions()
        conn = self._setup_conn("some_rev")

        mock_script_dir = MagicMock()
        mock_script_dir.walk_revisions.side_effect = RuntimeError("corrupt history")

        with patch.object(env.ScriptDirectory, "from_config", return_value=mock_script_dir):
            with pytest.raises(RuntimeError, match="corrupt history"):
                env.do_run_migrations(conn)

        env.context.run_migrations.assert_not_called()

    @pytest.mark.unit
    def test_propagates_script_directory_error(self):
        """Exception from ScriptDirectory.from_config() propagates unhandled."""
        env = _load_env_functions()
        conn = self._setup_conn("some_rev")

        with patch.object(env.ScriptDirectory, "from_config", side_effect=RuntimeError("bad config")):
            with pytest.raises(RuntimeError, match="bad config"):
                env.do_run_migrations(conn)

        env.context.run_migrations.assert_not_called()

    @pytest.mark.unit
    def test_widens_version_num_when_too_narrow(self):
        """Issues ALTER TABLE when existing version_num column is narrower than 64.

        Covers: existing PostgreSQL deployments may have the version_num
        column at the Alembic default VARCHAR(32). env.py widens it in-place
        before running migrations so that subsequent revision IDs >32 chars fit.
        """
        env = _load_env_functions()
        conn = MagicMock()
        conn.dialect.name = "postgresql"

        lock_result = MagicMock()
        lock_result.scalar.return_value = True

        existence_result = MagicMock()
        existence_result.scalar.return_value = True

        # Width check returns 32 — too narrow, ALTER must be issued.
        width_result = MagicMock()
        width_result.scalar.return_value = 32

        alter_result = MagicMock()  # ALTER TABLE has no return value we use

        version_result = MagicMock()
        version_result.fetchone.return_value = ("known_rev",)

        conn.execute.side_effect = [
            lock_result,
            existence_result,
            width_result,
            alter_result,
            version_result,
        ]

        mock_rev = MagicMock()
        mock_rev.revision = "known_rev"
        mock_script_dir = MagicMock()
        mock_script_dir.walk_revisions.return_value = [mock_rev]

        with patch.object(env.ScriptDirectory, "from_config", return_value=mock_script_dir):
            env.do_run_migrations(conn)

        # Confirm ALTER TABLE was issued (4th execute call) targeting VARCHAR(64)
        alter_call = conn.execute.call_args_list[3]
        sql_text = str(alter_call.args[0])
        assert "ALTER TABLE" in sql_text
        assert "VARCHAR(64)" in sql_text
        env.context.run_migrations.assert_called_once()

    @pytest.mark.unit
    def test_skips_widen_when_already_wide(self):
        """Skips ALTER TABLE when version_num column is already at or above target width."""
        env = _load_env_functions()
        conn = self._setup_conn("known_rev")  # _setup_conn already uses width=64

        mock_rev = MagicMock()
        mock_rev.revision = "known_rev"
        mock_script_dir = MagicMock()
        mock_script_dir.walk_revisions.return_value = [mock_rev]

        with patch.object(env.ScriptDirectory, "from_config", return_value=mock_script_dir):
            env.do_run_migrations(conn)

        # Only 4 executes: lock, exists, width-check, version SELECT — no ALTER.
        assert conn.execute.call_count == 4


# ---------------------------------------------------------------------------
# _run_migrations_sync — _DatabaseAheadError handling
# ---------------------------------------------------------------------------


class TestRunMigrationsSyncSkippedAhead:
    """Tests for _run_migrations_sync handling _DatabaseAheadError from env.py."""

    @pytest.mark.unit
    @patch("alembic.command.upgrade")
    @patch("alembic.script.ScriptDirectory.from_config")
    @patch("alembic.config.Config")
    def test_returns_success_when_skipped_ahead(self, mock_config_cls, mock_from_config, mock_upgrade):
        """Returns success with skipped=True when env.py raises _DatabaseAheadError."""
        mock_cfg_instance = MagicMock()
        mock_config_cls.return_value = mock_cfg_instance
        mock_cfg_instance.attributes = {}

        mock_script = MagicMock()
        mock_script.get_current_head.return_value = "abc123"
        mock_from_config.return_value = mock_script

        # Simulate env.py raising _DatabaseAheadError during upgrade
        mock_upgrade.side_effect = _DatabaseAheadError("future_rev_xyz")

        result = _run_migrations_sync("postgresql://localhost/testdb")

        assert result.success is True
        assert result.skipped is True
        assert result.current_revision is None
        assert result.target_revision == "abc123"
        assert result.error is None


# ---------------------------------------------------------------------------
# Khora.connect() — already connected (idempotent)
# ---------------------------------------------------------------------------


class TestKhoraConnectIdempotent:
    """Tests for Khora.connect() idempotency."""

    @pytest.mark.unit
    async def test_connect_already_connected_is_noop(self):
        """Calling connect() when already connected is a no-op."""
        mock_engine = MagicMock()
        mock_engine.connect = AsyncMock()

        with (
            patch("khora.khora.load_config", return_value=_mock_config()),
            patch("khora.engines.create_engine", return_value=mock_engine) as mock_create,
        ):
            kb = Khora(run_migrations=True)
            # Simulate already connected
            kb._connected = True
            kb._engine = mock_engine

            await kb.connect()

        # create_engine should NOT have been called (short-circuited)
        mock_create.assert_not_called()
        # engine.connect should NOT have been called again
        mock_engine.connect.assert_not_called()
