"""Coverage for migration ``042_widen_khora_chunks_source_external_id``.

The migration widens two denormalized columns on the runtime-managed
``khora_chunks`` table so an upcoming backfill copying values down from the
parent ``documents`` table cannot truncate:

    source       VARCHAR(255) -> TEXT
    external_id  VARCHAR(255) -> VARCHAR(512)

``khora_chunks`` is NOT part of the Alembic-managed schema -- it is created at
runtime by ``PgVectorTemporalStore.connect()``. The migration is therefore
guarded by ``has_table("khora_chunks")`` and exists only for existing
deployments where the table predates this widening. These tests cover:

1. Postgres upgrade: seed a legacy ``khora_chunks`` (no denormalized columns),
   run the chain to head, and assert ``source`` is ``text`` and ``external_id``
   is ``character varying(512)`` (041 first adds them at 255, 042 widens).
2. Postgres clean downgrade: with no over-length values, ``downgrade`` back to
   041 narrows both columns to ``character varying(255)``.
3. Postgres guarded downgrade: with a ``source`` value longer than 255 chars,
   ``downgrade`` SKIPS the narrowing (no-op) so the over-length data is not
   truncated; the columns stay wide.
4. Postgres missing-table path: on a fresh DB where ``khora_chunks`` does not
   exist, the migration early-returns (``has_table`` guard) and the chain still
   reaches head.
5. SQLite: the migration is Postgres-only and early-returns; the full chain
   reaches head cleanly.
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
_HEAD = "051_documents_graph_mirror_pending"
_PREV = "041_khora_chunks_denormalized_columns"


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
# this before upgrade lets migration 041 add ``source`` / ``external_id`` at the
# narrow 255 width, which 042 then widens. The ``content_tsv`` column + trigger
# mirror the runtime so the full re-run of the chain (043/044 ``DISABLE TRIGGER
# khora_chunks_content_tsv_update``) succeeds.
_LEGACY_KHORA_CHUNKS_DDL = """
CREATE TABLE khora_chunks (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    occurred_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,
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


async def _seed_legacy_table_at_baseline(url: str) -> None:
    """Drop ``khora_chunks``, recreate it legacy-shaped, step version to 040."""
    engine = create_async_engine(url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.execute(sa.text("DROP TABLE IF EXISTS khora_chunks CASCADE"))
            await conn.execute(sa.text(_LEGACY_KHORA_CHUNKS_DDL))
            await conn.execute(sa.text(_TSV_FUNCTION_DDL))
            await conn.execute(sa.text(_TSV_TRIGGER_DDL))
            await conn.execute(sa.text("UPDATE khora_alembic_version SET version_num = '040_chunks_last_accessed_at'"))
            # The step-back above is metadata-only, so chunks.occurred_at (added by
            # migration 046 during the first upgrade-to-head) survives. Drop it so
            # the replayed ``upgrade head`` re-applies 046 cleanly — mirrors the
            # drop+recreate of khora_chunks above.
            await conn.execute(sa.text("ALTER TABLE chunks DROP COLUMN IF EXISTS occurred_at"))
    finally:
        await engine.dispose()


async def _source_external_id_widths(url: str) -> dict[str, tuple[str, int | None]]:
    """Return {column_name: (data_type, character_maximum_length)} for the two cols."""
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                sa.text(
                    "SELECT column_name, data_type, character_maximum_length "
                    "FROM information_schema.columns "
                    "WHERE table_name = 'khora_chunks' "
                    "AND column_name IN ('source', 'external_id')"
                )
            )
            return {row[0]: (row[1], row[2]) for row in result}
    finally:
        await engine.dispose()


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


class TestMigration042OnPostgres:
    def test_upgrade_widens_source_and_external_id(self, pg_url: str) -> None:
        """Chain to head: ``source`` becomes TEXT, ``external_id`` becomes VARCHAR(512)."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")  # bring the chain to head first

        asyncio.run(_seed_legacy_table_at_baseline(pg_url))

        # Re-run: 041 adds source/external_id at 255, then 042 widens them.
        command.upgrade(cfg, "head")

        widths = asyncio.run(_source_external_id_widths(pg_url))
        # source: varchar -> text (unbounded; no character_maximum_length).
        assert widths["source"][0] == "text"
        assert widths["source"][1] is None
        # external_id: varchar(255) -> varchar(512).
        assert widths["external_id"][0] == "character varying"
        assert widths["external_id"][1] == 512

    def test_clean_downgrade_narrows_both_columns(self, pg_url: str) -> None:
        """With no over-length values, downgrade narrows both back to VARCHAR(255)."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")
        asyncio.run(_seed_legacy_table_at_baseline(pg_url))
        command.upgrade(cfg, "head")

        command.downgrade(cfg, _PREV)

        widths = asyncio.run(_source_external_id_widths(pg_url))
        assert widths["source"][0] == "character varying"
        assert widths["source"][1] == 255
        assert widths["external_id"][0] == "character varying"
        assert widths["external_id"][1] == 255

    def test_guarded_downgrade_skips_when_value_would_truncate(self, pg_url: str) -> None:
        """An over-length ``source`` value makes downgrade skip the narrowing."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")
        asyncio.run(_seed_legacy_table_at_baseline(pg_url))
        command.upgrade(cfg, "head")  # widened: source TEXT, external_id VARCHAR(512)

        async def _insert_long_source() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await conn.execute(
                        sa.text(
                            "INSERT INTO khora_chunks (id, namespace_id, document_id, content, source) "
                            "VALUES (gen_random_uuid(), gen_random_uuid(), gen_random_uuid(), 'x', :s)"
                        ),
                        {"s": "a" * 300},
                    )
            finally:
                await engine.dispose()

        asyncio.run(_insert_long_source())

        # Downgrade must NOT raise and must NOT truncate -- it skips the narrowing.
        command.downgrade(cfg, _PREV)

        widths = asyncio.run(_source_external_id_widths(pg_url))
        # Columns stay wide because narrowing was skipped to avoid truncation.
        assert widths["source"][0] == "text"
        assert widths["external_id"][1] == 512

    def test_no_op_when_table_absent(self, pg_url: str) -> None:
        """Fresh DB with no ``khora_chunks``: chain reaches head, no error."""
        cfg = _make_config(pg_url)
        # pg_url fixture already dropped khora_chunks; just upgrade.
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == _HEAD

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


class TestMigration042OnSqlite:
    def test_chain_reaches_head_on_sqlite(self, sqlite_url: str) -> None:
        """Migration 042 is a clean no-op on SQLite; chain reaches head."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == _HEAD
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_downgrade_is_clean_on_sqlite(self, sqlite_url: str) -> None:
        """upgrade head -> downgrade to 041 is a clean no-op on SQLite."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, _PREV)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == _PREV
            finally:
                await engine.dispose()

        asyncio.run(check())
