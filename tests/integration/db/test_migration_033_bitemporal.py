"""Integration tests for migration 033: bi-temporal columns + partial indexes.

Validates that ``033_bitemporal_columns``:

* adds ``valid_to``, ``invalidated_at``, ``invalidated_by`` to both
  ``relationships`` and ``memory_facts`` (Postgres + SQLite),
* creates the ``ix_relationships_live`` and ``ix_memory_facts_live``
  partial indexes with the correct ``WHERE invalidated_at IS NULL``
  predicate (Postgres only),
* downgrades cleanly (drops columns + indexes),
* skips partial indexes silently on SQLite while still adding the
  columns,
* leaves the existing ``is_active=True`` filter behavior on
  ``memory_facts`` unchanged.

The migration depends on #651 / migration 032 landing first. While 032
is in flight, ``down_revision`` is a placeholder, and the full alembic
chain cannot resolve. These tests detect that condition and skip with a
clear reason — they will run normally once the rebase against the real
032 revision id is in place.

Run locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/db/test_migration_033_bitemporal.py -v \
        -m integration --no-cov
"""

from __future__ import annotations

import importlib
import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from khora.db.session import run_migrations

# ---------------------------------------------------------------------------
# Module-level skip if migration 032 hasn't been merged yet.
# ---------------------------------------------------------------------------

_MIGRATION_033 = importlib.import_module("khora.db.migrations.versions.033_bitemporal_columns")
_MIGRATIONS_DIR = Path(_MIGRATION_033.__file__).parent
_DOWN_REVISION = _MIGRATION_033.down_revision
_DOWN_REVISION_EXISTS = any(
    f.stem.startswith(_DOWN_REVISION) or _DOWN_REVISION in f.read_text()
    for f in _MIGRATIONS_DIR.glob("*.py")
    if f.name != "033_bitemporal_columns.py"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _DOWN_REVISION_EXISTS,
        reason=(
            f"migration 033 chains from {_DOWN_REVISION!r}, which is a "
            "placeholder for migration 032 (#651). Rebase this branch "
            "against 032's real revision id once #651 lands."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Postgres fixture
# ---------------------------------------------------------------------------


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


_PG_AVAILABLE = _pg_reachable()


async def _reset_public_schema(eng: AsyncEngine) -> None:
    """Wipe ``public`` and pre-create the wide khora_alembic_version table.

    Mirrors the workaround documented in ``test_chronicle_pg.py``: alembic
    creates ``khora_alembic_version`` with the default ``VARCHAR(32)`` but
    several revision ids are wider. Pre-create the table with VARCHAR(64)
    so the chain applies cleanly.
    """
    async with eng.begin() as conn:
        r = await conn.execute(
            text("SELECT typname FROM pg_type WHERE typnamespace = 'public'::regnamespace AND typtype = 'e'")
        )
        for (typname,) in r.fetchall():
            await conn.execute(text(f"DROP TYPE IF EXISTS public.{typname} CASCADE"))
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(
            text(
                "CREATE TABLE khora_alembic_version ("
                "  version_num VARCHAR(64) NOT NULL,"
                "  CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)"
                ")"
            )
        )


@pytest.fixture
async def pg_engine() -> AsyncIterator[AsyncEngine]:
    if not _PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable (run `make dev` first)")
    eng = create_async_engine(DATABASE_URL)
    await _reset_public_schema(eng)
    result = await run_migrations(DATABASE_URL)
    assert result.success, f"migrations failed: {result.error}"
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def sqlite_engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    db_path = tmp_path / "khora.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    result = await run_migrations(url)
    assert result.success, f"migrations failed: {result.error}"
    eng = create_async_engine(url)
    try:
        yield eng
    finally:
        await eng.dispose()


# ---------------------------------------------------------------------------
# Column-presence tests (PG)
# ---------------------------------------------------------------------------


async def _column_info(eng: AsyncEngine, table: str, column: str) -> tuple[str, str] | None:
    """Return ``(data_type, is_nullable)`` for ``table.column`` on PG, or None."""
    async with eng.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT data_type, is_nullable FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = :c"
                ),
                {"t": table, "c": column},
            )
        ).first()
    return (row[0], row[1]) if row else None


@pytest.mark.asyncio
async def test_migration_033_adds_columns_to_relationships(
    pg_engine: AsyncEngine,
) -> None:
    for col, expected_type in (
        ("valid_to", "timestamp with time zone"),
        ("invalidated_at", "timestamp with time zone"),
        ("invalidated_by", "uuid"),
    ):
        info = await _column_info(pg_engine, "relationships", col)
        assert info is not None, f"relationships.{col} missing after migration"
        data_type, is_nullable = info
        assert data_type == expected_type, f"relationships.{col}: expected {expected_type}, got {data_type}"
        assert is_nullable == "YES", f"relationships.{col} should be nullable"


@pytest.mark.asyncio
async def test_migration_033_adds_columns_to_memory_facts(
    pg_engine: AsyncEngine,
) -> None:
    for col, expected_type in (
        ("valid_to", "timestamp with time zone"),
        ("invalidated_at", "timestamp with time zone"),
        ("invalidated_by", "uuid"),
    ):
        info = await _column_info(pg_engine, "memory_facts", col)
        assert info is not None, f"memory_facts.{col} missing after migration"
        data_type, is_nullable = info
        assert data_type == expected_type, f"memory_facts.{col}: expected {expected_type}, got {data_type}"
        assert is_nullable == "YES", f"memory_facts.{col} should be nullable"


# ---------------------------------------------------------------------------
# Partial-index tests (PG)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_033_creates_partial_indexes(
    pg_engine: AsyncEngine,
) -> None:
    async with pg_engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE indexname IN ('ix_relationships_live', 'ix_memory_facts_live')"
                )
            )
        ).fetchall()
    found = {name: definition for name, definition in rows}

    assert "ix_relationships_live" in found, "ix_relationships_live missing"
    assert "ix_memory_facts_live" in found, "ix_memory_facts_live missing"

    rel_def = found["ix_relationships_live"].lower()
    assert "where (invalidated_at is null)" in rel_def
    assert "namespace_id" in rel_def
    assert "source_entity_id" in rel_def
    assert "target_entity_id" in rel_def
    assert "relationship_type" in rel_def

    fact_def = found["ix_memory_facts_live"].lower()
    assert "where (invalidated_at is null)" in fact_def
    assert "namespace_id" in fact_def
    assert "subject" in fact_def


# ---------------------------------------------------------------------------
# Downgrade test (PG) — apply, downgrade, columns + indexes gone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_033_downgrade_reverses(pg_engine: AsyncEngine) -> None:
    # Sanity: columns and indexes exist after upgrade.
    assert await _column_info(pg_engine, "relationships", "valid_to") is not None

    # Drive a single-step downgrade programmatically (no subprocess).
    from alembic import command
    from alembic.config import Config

    sync_url = DATABASE_URL.replace("+asyncpg", "")
    migrations_dir = Path(_MIGRATION_033.__file__).resolve().parents[0].parent
    cfg = Config()
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    cfg.set_main_option("version_table", "khora_alembic_version")
    cfg.set_main_option("version_table_schema", "public")
    command.downgrade(cfg, "-1")

    # Columns gone from both tables.
    for table in ("relationships", "memory_facts"):
        for col in ("valid_to", "invalidated_at", "invalidated_by"):
            info = await _column_info(pg_engine, table, col)
            assert info is None, f"{table}.{col} still present after downgrade"

    # Indexes gone.
    async with pg_engine.connect() as conn:
        names = {
            row[0]
            for row in (
                await conn.execute(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE indexname IN ('ix_relationships_live', 'ix_memory_facts_live')"
                    )
                )
            ).fetchall()
        }
    assert names == set(), f"indexes still present after downgrade: {names}"


# ---------------------------------------------------------------------------
# SQLite dialect-gating: columns added, partial indexes skipped silently.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_033_sqlite_dialect_gated(sqlite_engine: AsyncEngine) -> None:
    """On sqlite_lance, the columns get added; partial indexes are skipped.

    The migration's upgrade short-circuits before issuing
    ``CREATE INDEX CONCURRENTLY`` on non-Postgres dialects.
    """
    for table in ("relationships", "memory_facts"):
        async with sqlite_engine.connect() as conn:
            rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).fetchall()
        cols = {row[1] for row in rows}
        for expected in ("valid_to", "invalidated_at", "invalidated_by"):
            assert expected in cols, f"{table}.{expected} missing after migration on SQLite; have {sorted(cols)}"

    # No live-row partial indexes were created on SQLite.
    async with sqlite_engine.connect() as conn:
        idx_rows = (
            await conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name IN ('ix_relationships_live', 'ix_memory_facts_live')"
                )
            )
        ).fetchall()
    assert idx_rows == [], f"unexpected partial indexes on SQLite: {idx_rows}"


# ---------------------------------------------------------------------------
# Existing is_active filter behavior is unchanged post-migration.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_queries_still_work(pg_engine: AsyncEngine) -> None:
    """``is_active=True`` filter on memory_facts behaves identically post-033.

    Seeds two rows (one active, one supersession-style inactive) and verifies
    the existing predicate still partitions them correctly. The bi-temporal
    columns coexist; they do not change ``is_active`` semantics.
    """
    from uuid import uuid4

    ns_id = uuid4()
    active_id = uuid4()
    inactive_id = uuid4()

    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO memory_namespaces "
                "(id, namespace_id, name, description, created_at, updated_at) "
                "VALUES (:id, :id, 'test', '', NOW(), NOW())"
            ),
            {"id": ns_id},
        )
        await conn.execute(
            text(
                "INSERT INTO memory_facts "
                "(id, namespace_id, subject, predicate, object, fact_text, "
                " confidence, is_active, source_chunk_ids, created_at, updated_at) "
                "VALUES (:id, :ns, 'alice', 'works_at', 'acme', 'alice works at acme', "
                " 1.0, TRUE, ARRAY[]::uuid[], NOW(), NOW())"
            ),
            {"id": active_id, "ns": ns_id},
        )
        await conn.execute(
            text(
                "INSERT INTO memory_facts "
                "(id, namespace_id, subject, predicate, object, fact_text, "
                " confidence, is_active, source_chunk_ids, created_at, updated_at) "
                "VALUES (:id, :ns, 'alice', 'works_at', 'oldco', 'alice works at oldco', "
                " 1.0, FALSE, ARRAY[]::uuid[], NOW(), NOW())"
            ),
            {"id": inactive_id, "ns": ns_id},
        )

    async with pg_engine.connect() as conn:
        active_rows = (
            await conn.execute(
                text("SELECT id FROM memory_facts WHERE namespace_id = :ns AND is_active = TRUE"),
                {"ns": ns_id},
            )
        ).fetchall()
        all_rows = (
            await conn.execute(
                text("SELECT id, is_active, invalidated_at FROM memory_facts WHERE namespace_id = :ns"),
                {"ns": ns_id},
            )
        ).fetchall()

    assert {row[0] for row in active_rows} == {active_id}, "is_active filter changed shape post-033"
    # All existing rows backfill to invalidated_at = NULL.
    for _row_id, _is_active, invalidated_at in all_rows:
        assert invalidated_at is None
