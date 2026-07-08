"""Integration coverage for migration ``032_dream_runs``.

The dream-phase orchestrator (#649 / #651) checkpoints per-namespace run
state into ``khora_dream_runs`` so a crashed APPLY pass can be resumed
at the last committed op-seq. Since #896 the table is created on BOTH
dialects (the DDL is dialect-portable) so ``dream_history`` /
``dream_status`` work on the embedded sqlite_lance stack.

Tests pin:

1. Fresh PG: ``alembic upgrade head`` creates the table, its 17 columns
   with correct types, and the ``(namespace_id, started_at DESC)``
   composite index.
2. SQLite: upgrading head creates the table too (#896).
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
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

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
# Postgres fixture — wipe ``public`` and re-migrate to head per test so the
# tests run in arbitrary order against a shared PostgreSQL (the integration
# job migrates the service DB to head up front) without leaking state.
# ---------------------------------------------------------------------------


async def _reset_public_schema(eng: AsyncEngine) -> None:
    """Wipe ``public`` and pre-create the wide khora_alembic_version table.

    Mirrors ``test_migration_033_bitemporal.py``: alembic creates
    ``khora_alembic_version`` with the default ``VARCHAR(32)`` but several
    revision ids are wider. Pre-create the table with VARCHAR(64) so the chain
    applies cleanly.
    """
    async with eng.begin() as conn:
        r = await conn.execute(
            sa.text("SELECT typname FROM pg_type WHERE typnamespace = 'public'::regnamespace AND typtype = 'e'")
        )
        for (typname,) in r.fetchall():
            await conn.execute(sa.text(f"DROP TYPE IF EXISTS public.{typname} CASCADE"))
        await conn.execute(sa.text("DROP SCHEMA public CASCADE"))
        await conn.execute(sa.text("CREATE SCHEMA public"))
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(
            sa.text(
                "CREATE TABLE khora_alembic_version ("
                "  version_num VARCHAR(64) NOT NULL,"
                "  CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)"
                ")"
            )
        )


@pytest.fixture
def pg_url() -> Iterator[str]:
    """Wipe ``public``, re-migrate to head, and yield the base Postgres URL."""
    if not _pg_reachable():
        pytest.skip("PostgreSQL not reachable (run `make dev` first)")

    async def _setup() -> None:
        eng = create_async_engine(DATABASE_URL)
        try:
            await _reset_public_schema(eng)
        finally:
            await eng.dispose()
        result = await run_migrations(DATABASE_URL)
        assert result.success, f"migrations failed: {result.error}"

    asyncio.run(_setup())
    yield DATABASE_URL


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMigration032OnPostgres:
    def test_migration_032_creates_dream_runs_table(self, pg_url: str) -> None:
        """Upgrade head on fresh PG creates the table with all 17 columns + index."""
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

                    # All 17 columns present, with the data types we promised.
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
                        "graph_mirror_pending",
                    }
                    assert set(cols.keys()) == expected_cols
                    assert len(cols) == 17

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
    def test_migration_032_creates_table_on_sqlite(self, sqlite_url: str) -> None:
        """#896: the SQLite path now creates ``khora_dream_runs`` too."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    # Chain reached head (sanity).
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == "052_entities_source_chunk_ids_gin"

                    # khora_dream_runs exists on SQLite (#896) so dream_history /
                    # dream_status work on the embedded sqlite_lance stack.
                    result = await conn.execute(
                        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='khora_dream_runs'")
                    )
                    assert result.fetchone() is not None

                    # The composite index is created on SQLite too.
                    result = await conn.execute(
                        sa.text(
                            "SELECT name FROM sqlite_master WHERE type='index' "
                            "AND name='ix_khora_dream_runs_namespace_started'"
                        )
                    )
                    assert result.fetchone() is not None
            finally:
                await engine.dispose()

        asyncio.run(check())
