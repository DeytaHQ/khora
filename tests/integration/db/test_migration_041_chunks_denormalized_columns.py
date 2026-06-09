"""Coverage for migration ``041_khora_chunks_denormalized_columns``.

The migration adds eight nullable, denormalized document-grained columns to
the runtime-managed ``khora_chunks`` temporal store so recall filters can be
applied on the chunk row without a join:

    source_type, source_name, source_url, source_timestamp,
    external_id, content_type, source, title

``khora_chunks`` is NOT part of the Alembic-managed schema — it is created at
runtime by ``PgVectorTemporalStore.connect()``. The migration is therefore
guarded by ``has_table("khora_chunks")`` and exists only for existing
deployments where the table predates these columns. These tests cover both
shapes:

1. Postgres, existing-table path: we hand-create a ``khora_chunks`` table
   *without* the eight columns (mirroring a pre-migration deployment), run
   ``alembic upgrade head``, and assert the eight columns now exist with the
   promised types and that all are nullable with no default. ``downgrade``
   back to 040 drops exactly those eight.
2. Postgres, missing-table path: on a fresh DB where ``khora_chunks`` does not
   exist, the migration must early-return (``has_table`` guard) and the chain
   must still reach head without error.
3. SQLite: the migration is Postgres-only and early-returns; the full chain
   must reach head ``041`` cleanly.
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

# The eight columns this migration adds. ``source_timestamp`` is timezone-aware;
# the rest are string/text. All nullable, no default.
_NEW_COLUMNS = {
    "source_type",
    "source_name",
    "source_url",
    "source_timestamp",
    "external_id",
    "content_type",
    "source",
    "title",
}


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    # Alembic uses configparser.BasicInterpolation; escape any literal '%' in
    # the URL so it isn't read as a config-interpolation token.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


# A pre-migration ``khora_chunks`` table: the identity/temporal columns the
# runtime always created, but WITHOUT the eight denormalized columns. Creating
# this before upgrade exercises the migration's existing-deployment path. The
# ``content_tsv`` column + trigger mirror the runtime so the full re-run of the
# chain (043/044 ``DISABLE TRIGGER khora_chunks_content_tsv_update``) succeeds.
_LEGACY_KHORA_CHUNKS_DDL = """
CREATE TABLE khora_chunks (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    content_tsv TSVECTOR
)
"""

# The runtime's content_tsv trigger, byte-equivalent to
# ``PgVectorTemporalStore.connect()``. Seeding it lets the full re-run of the
# chain (043/044) DISABLE/ENABLE it without an UndefinedObjectError.
_TSV_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION khora_chunks_content_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""

_TSV_TRIGGER_DDL = """
CREATE TRIGGER khora_chunks_content_tsv_update
BEFORE INSERT OR UPDATE ON khora_chunks
FOR EACH ROW EXECUTE FUNCTION khora_chunks_content_tsv_trigger()
"""


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
    """Wipe ``public``, re-migrate to head, and yield the base Postgres URL.

    The schema-wipe makes the tests safe to run against shared PostgreSQL (the
    integration job migrates the service DB to head up front) without leaking
    state. ``khora_chunks`` is runtime-managed (not Alembic), so after the
    re-migrate it does not exist — each test seeds the legacy table itself.
    """
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
# Postgres
# ---------------------------------------------------------------------------


class TestMigration041OnPostgres:
    def test_adds_eight_columns_to_existing_table(self, pg_url: str) -> None:
        """Existing ``khora_chunks`` (missing the columns) gains all eight."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")  # bring the Alembic chain to head first

        async def _seed_legacy_table() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await conn.execute(sa.text("DROP TABLE IF EXISTS khora_chunks CASCADE"))
                    await conn.execute(sa.text(_LEGACY_KHORA_CHUNKS_DDL))
                    await conn.execute(sa.text(_TSV_FUNCTION_DDL))
                    await conn.execute(sa.text(_TSV_TRIGGER_DDL))
                    # Step the version table back so we can re-run 041 against
                    # the table we just created.
                    await conn.execute(
                        sa.text("UPDATE khora_alembic_version SET version_num = '040_chunks_last_accessed_at'")
                    )
            finally:
                await engine.dispose()

        asyncio.run(_seed_legacy_table())

        # Re-run the chain: 041 now sees an existing khora_chunks and adds cols.
        command.upgrade(cfg, "head")

        async def check_added() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(
                        sa.text(
                            "SELECT column_name, data_type, is_nullable, column_default "
                            "FROM information_schema.columns "
                            "WHERE table_name = 'khora_chunks'"
                        )
                    )
                    cols = {row[0]: (row[1], row[2], row[3]) for row in result}

                    # All eight present.
                    assert _NEW_COLUMNS <= set(cols.keys()), f"missing: {_NEW_COLUMNS - set(cols.keys())}"

                    # All eight nullable with no server default.
                    for name in _NEW_COLUMNS:
                        assert cols[name][1] == "YES", f"{name} should be nullable"
                        assert cols[name][2] is None, f"{name} should have no default"

                    # Load-bearing types after the chain reaches head:
                    # source_timestamp is timestamptz; source_url / title / source
                    # are text (migration 042 widens source varchar -> text); the
                    # rest are varchar.
                    assert cols["source_timestamp"][0] == "timestamp with time zone"
                    assert cols["source_url"][0] == "text"
                    assert cols["title"][0] == "text"
                    assert cols["source"][0] == "text"
                    assert cols["source_type"][0] == "character varying"
                    assert cols["source_name"][0] == "character varying"
                    assert cols["external_id"][0] == "character varying"
                    assert cols["content_type"][0] == "character varying"
            finally:
                await engine.dispose()

        asyncio.run(check_added())

    def test_downgrade_drops_the_eight_columns(self, pg_url: str) -> None:
        """``downgrade`` to 040 removes exactly the eight columns it added."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        async def _seed_legacy_table() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await conn.execute(sa.text("DROP TABLE IF EXISTS khora_chunks CASCADE"))
                    await conn.execute(sa.text(_LEGACY_KHORA_CHUNKS_DDL))
                    await conn.execute(sa.text(_TSV_FUNCTION_DDL))
                    await conn.execute(sa.text(_TSV_TRIGGER_DDL))
                    await conn.execute(
                        sa.text("UPDATE khora_alembic_version SET version_num = '040_chunks_last_accessed_at'")
                    )
            finally:
                await engine.dispose()

        asyncio.run(_seed_legacy_table())
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "040_chunks_last_accessed_at")

        async def check_dropped() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(
                        sa.text("SELECT column_name FROM information_schema.columns WHERE table_name = 'khora_chunks'")
                    )
                    cols = {row[0] for row in result}
                    # None of the eight remain.
                    assert _NEW_COLUMNS.isdisjoint(cols), f"still present after downgrade: {_NEW_COLUMNS & cols}"
                    # The legacy identity columns are untouched.
                    assert {"id", "namespace_id", "document_id", "content"} <= cols
            finally:
                await engine.dispose()

        asyncio.run(check_dropped())

    def test_no_op_when_table_absent(self, pg_url: str) -> None:
        """Fresh DB with no ``khora_chunks``: chain reaches head, no error.

        The ``has_table`` guard must early-return rather than fail when the
        runtime-managed table has not been created yet.
        """
        cfg = _make_config(pg_url)
        # pg_url fixture already dropped khora_chunks; just upgrade.
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    # Chain reached head.
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == "045_khora_try_timestamptz"

                    # The migration did not create the table.
                    result = await conn.execute(
                        sa.text(
                            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'khora_chunks')"
                        )
                    )
                    assert result.scalar() is False
            finally:
                await engine.dispose()

        asyncio.run(check())


# ---------------------------------------------------------------------------
# SQLite — Postgres-only migration must early-return and stay green
# ---------------------------------------------------------------------------


class TestMigration041OnSqlite:
    def test_chain_reaches_head_on_sqlite(self, sqlite_url: str) -> None:
        """Migration 041 is a clean no-op on SQLite; chain reaches head."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == "045_khora_try_timestamptz"
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_downgrade_is_clean_on_sqlite(self, sqlite_url: str) -> None:
        """upgrade head → downgrade to 040 is a clean no-op on SQLite."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "040_chunks_last_accessed_at")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == "040_chunks_last_accessed_at"
            finally:
                await engine.dispose()

        asyncio.run(check())
