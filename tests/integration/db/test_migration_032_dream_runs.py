"""Integration coverage for migration ``032_dream_runs``.

The dream-phase orchestrator (#649 / #651) checkpoints per-namespace run
state into ``khora_dream_runs`` so a crashed APPLY pass can be resumed
at the last committed op-seq. The migration is Postgres-only via the
same dialect gate as 029; on SQLite-backed sqlite_lance stacks the
embedded path mirrors state to a ``dream_runs.jsonl`` file sink and
the migration must be a clean no-op.

Tests pin:

1. Fresh PG: ``alembic upgrade head`` creates the table, its 16 columns
   with correct types, and the ``(namespace_id, started_at DESC)``
   composite index.
2. SQLite: upgrading head does NOT create the table (dialect gate).
3. Postgres downgrade -1 reverses cleanly (index dropped first, then
   table).
4. Skip-ahead: when the DB advertises an unknown future revision,
   ``run_migrations()`` returns ``MigrationResult(success=True,
   skipped=True)`` via the existing ``_DatabaseAheadError`` plumbing.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine

from khora.db.session import run_migrations

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


def _pg_reachable() -> bool:
    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.integration


_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    # Alembic uses configparser.BasicInterpolation; escape any literal '%' in
    # the URL (e.g. URL-encoded '?server_settings=search_path%3D...') so it
    # doesn't get interpreted as a config-interpolation token.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


# ---------------------------------------------------------------------------
# Postgres fixture — schema-isolated per test so the four tests can run in
# arbitrary order without leaking state. We point the async engine at a
# unique schema, run the chain there, then drop it on teardown.
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_url() -> Iterator[str]:
    """Yield the base Postgres URL; drop ``khora_dream_runs`` on teardown.

    We don't isolate per-schema because asyncpg rejects URL-embedded
    ``server_settings`` and threading per-test ``connect_args`` through the
    Alembic command API is fragile. Integration tests run serially anyway
    (per CLAUDE.md — `tests/integration/matrix/*` already DROP SCHEMA on
    shared PostgreSQL).
    """
    if not _pg_reachable():
        pytest.skip("PostgreSQL not reachable (run `make dev` first)")

    async def _reset() -> None:
        """Drop dream_runs table and reset alembic_version to 031 so tests
        don't poison each other via leftover stamps."""
        admin = create_async_engine(DATABASE_URL, isolation_level="AUTOCOMMIT")
        try:
            async with admin.connect() as conn:
                await conn.execute(sa.text("DROP TABLE IF EXISTS khora_dream_runs CASCADE"))
                # Reset alembic head to 031 so the next test's upgrade is a no-op
                # (032 is the head once migration lands; we want a clean baseline).
                await conn.execute(
                    sa.text(
                        "UPDATE khora_alembic_version SET version_num = "
                        "'031_session_id_indexes' WHERE version_num != "
                        "'031_session_id_indexes'"
                    )
                )
        finally:
            await admin.dispose()

    asyncio.run(_reset())
    try:
        yield DATABASE_URL
    finally:
        asyncio.run(_reset())


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMigration032OnPostgres:
    def test_migration_032_creates_dream_runs_table(self, pg_url: str) -> None:
        """Upgrade head on fresh PG creates the table with all 16 columns + index."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    # Table exists.
                    result = await conn.execute(
                        sa.text(
                            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                            "WHERE table_name = 'khora_dream_runs')"
                        )
                    )
                    assert result.scalar() is True

                    # All 16 columns present, with the data types we promised.
                    result = await conn.execute(
                        sa.text(
                            "SELECT column_name, data_type, is_nullable "
                            "FROM information_schema.columns "
                            "WHERE table_name = 'khora_dream_runs'"
                        )
                    )
                    cols = {row[0]: (row[1], row[2]) for row in result}
                    expected_cols = {
                        "run_id",
                        "namespace_id",
                        "trigger",
                        "mode",
                        "state",
                        "plan_hash",
                        "started_at",
                        "finished_at",
                        "last_committed_op_seq",
                        "heartbeat_at",
                        "total_ops",
                        "total_decisions",
                        "report_path",
                        "manifest_sha256",
                        "config_fingerprint",
                        "error",
                    }
                    assert set(cols.keys()) == expected_cols
                    assert len(cols) == 16

                    # Spot-check the load-bearing types.
                    assert cols["run_id"][0] == "uuid"
                    assert cols["namespace_id"][0] == "uuid"
                    assert cols["trigger"][0] == "character varying"
                    assert cols["state"][0] == "character varying"
                    assert cols["started_at"][0] == "timestamp with time zone"
                    assert cols["heartbeat_at"][0] == "timestamp with time zone"
                    assert cols["last_committed_op_seq"][0] == "integer"
                    assert cols["error"][0] == "jsonb"

                    # NOT NULL constraints.
                    assert cols["namespace_id"][1] == "NO"
                    assert cols["trigger"][1] == "NO"
                    assert cols["mode"][1] == "NO"
                    assert cols["state"][1] == "NO"
                    assert cols["started_at"][1] == "NO"
                    assert cols["heartbeat_at"][1] == "NO"
                    # finished_at / error / etc. are nullable.
                    assert cols["finished_at"][1] == "YES"
                    assert cols["error"][1] == "YES"

                    # Index exists on (namespace_id, started_at DESC).
                    result = await conn.execute(
                        sa.text(
                            "SELECT indexdef FROM pg_indexes "
                            "WHERE tablename = 'khora_dream_runs' "
                            "AND indexname = 'ix_khora_dream_runs_namespace_started'"
                        )
                    )
                    indexdef = result.scalar()
                    assert indexdef is not None
                    assert "namespace_id" in indexdef
                    assert "started_at" in indexdef
                    assert "DESC" in indexdef
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_migration_032_downgrade_reverses(self, pg_url: str) -> None:
        """upgrade head → downgrade to 031 leaves no table and no index behind.

        Targets revision 031 explicitly rather than ``-1`` because head may
        now be 033 (or later) once subsequent dream-phase migrations land —
        we need to walk back through 032 to verify its downgrade runs cleanly.
        """
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "031_session_id_indexes")

        async def check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(
                        sa.text(
                            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                            "WHERE table_name = 'khora_dream_runs')"
                        )
                    )
                    assert result.scalar() is False

                    result = await conn.execute(
                        sa.text(
                            "SELECT EXISTS(SELECT 1 FROM pg_indexes "
                            "WHERE indexname = 'ix_khora_dream_runs_namespace_started')"
                        )
                    )
                    assert result.scalar() is False
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_migration_032_unknown_revision_skips_ahead(self, pg_url: str) -> None:
        """If the DB is at an unknown future revision, run_migrations() skips."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        # Stamp the version table with an unknown future revision so env.py's
        # ahead-detection trips _DatabaseAheadError.
        future_rev = "999_pretend_future_revision"

        async def _stamp_future() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text("UPDATE khora_alembic_version SET version_num = :rev"),
                        {"rev": future_rev},
                    )
            finally:
                await engine.dispose()

        asyncio.run(_stamp_future())

        async def _try_migrate() -> None:
            result = await run_migrations(pg_url)
            assert result.success is True
            assert result.skipped is True
            assert result.error is None

        asyncio.run(_try_migrate())


class TestMigration032OnSqlite:
    def test_migration_032_idempotent_on_sqlite(self, sqlite_url: str) -> None:
        """Dialect gate: SQLite path runs the chain without creating the table."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    # Chain reached head (sanity).
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == "040_chunks_last_accessed_at"

                    # khora_dream_runs MUST NOT exist on SQLite — embedded path
                    # mirrors checkpoint state via a JSONL file sink.
                    result = await conn.execute(
                        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='khora_dream_runs'")
                    )
                    assert result.fetchone() is None
            finally:
                await engine.dispose()

        asyncio.run(check())
