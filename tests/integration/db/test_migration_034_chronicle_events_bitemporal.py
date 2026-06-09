"""Integration tests for migration 034: chronicle_events bi-temporal columns.

Validates that ``034_chronicle_events_bitemporal``:

* adds ``invalidated_at``, ``invalidated_by``, ``merged_into_event_id``
  to ``chronicle_events`` (Postgres + SQLite),
* creates the ``ix_chronicle_events_live`` partial composite index with
  the correct ``WHERE invalidated_at IS NULL`` predicate (Postgres
  only — dialect-gated on SQLite),
* enforces the ``merged_into_event_id`` self-FK with
  ``ON DELETE SET NULL`` semantics (Postgres),
* downgrades cleanly (drops columns + index + constraint).

Mirrors the test layout of migration 033's integration suite.

Run locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/db/test_migration_034_chronicle_events_bitemporal.py -v \
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
# Module-level skip if the down-revision hasn't been merged.
# ---------------------------------------------------------------------------

_MIGRATION_034 = importlib.import_module("khora.db.migrations.versions.034_chronicle_events_bitemporal")
_MIGRATIONS_DIR = Path(_MIGRATION_034.__file__).parent
_DOWN_REVISION = _MIGRATION_034.down_revision
_DOWN_REVISION_EXISTS = any(
    f.stem.startswith(_DOWN_REVISION) or _DOWN_REVISION in f.read_text()
    for f in _MIGRATIONS_DIR.glob("*.py")
    if f.name != "034_chronicle_events_bitemporal.py"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _DOWN_REVISION_EXISTS,
        reason=(
            f"migration 034 chains from {_DOWN_REVISION!r}; rebase against "
            "the real revision id of migration 033 once #680 lands."
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

    Same workaround documented in ``test_migration_033_bitemporal.py``:
    alembic uses ``VARCHAR(32)`` by default but several revision ids are
    wider. Pre-create with VARCHAR(64).
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
async def test_migration_034_adds_columns_to_chronicle_events(
    pg_engine: AsyncEngine,
) -> None:
    for col, expected_type in (
        ("invalidated_at", "timestamp with time zone"),
        ("invalidated_by", "uuid"),
        ("merged_into_event_id", "uuid"),
    ):
        info = await _column_info(pg_engine, "chronicle_events", col)
        assert info is not None, f"chronicle_events.{col} missing after migration"
        data_type, is_nullable = info
        assert data_type == expected_type, f"chronicle_events.{col}: expected {expected_type}, got {data_type}"
        assert is_nullable == "YES", f"chronicle_events.{col} should be nullable"


# ---------------------------------------------------------------------------
# Partial-index test (PG)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_034_creates_live_partial_index(
    pg_engine: AsyncEngine,
) -> None:
    async with pg_engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT indexname, indexdef FROM pg_indexes WHERE indexname = 'ix_chronicle_events_live'")
            )
        ).fetchall()
    found = {name: definition for name, definition in rows}
    assert "ix_chronicle_events_live" in found, "ix_chronicle_events_live missing"
    definition = found["ix_chronicle_events_live"].lower()
    assert "where (invalidated_at is null)" in definition
    assert "namespace_id" in definition
    # The migration indexes (namespace_id, referenced_date) — the real
    # column the Chronicle engine uses as the temporal anchor.
    assert "referenced_date" in definition


# ---------------------------------------------------------------------------
# Self-FK test (PG)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_034_self_fk_set_null_on_delete(
    pg_engine: AsyncEngine,
) -> None:
    """Deleting a canonical row sets ``merged_into_event_id`` to NULL on tails."""
    from uuid import uuid4

    ns_id = uuid4()
    chunk_id = uuid4()
    canonical_id = uuid4()
    tail_id = uuid4()

    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO memory_namespaces "
                "(id, namespace_id, version, is_active, created_at, updated_at) "
                "VALUES (:id, :id, 1, TRUE, NOW(), NOW())"
            ),
            {"id": ns_id},
        )
        await conn.execute(
            text(
                "INSERT INTO documents "
                "(id, namespace_id, content, source, status, chunk_count, metadata, created_at, updated_at) "
                "VALUES (:id, :ns, 'doc body', 'test', 'completed', 1, '{}'::jsonb, NOW(), NOW())"
            ),
            {"id": uuid4(), "ns": ns_id},
        )
        # Insert two chronicle_events rows. The first is canonical;
        # the second is the tail that references it.
        await conn.execute(
            text(
                "INSERT INTO chunks "
                "(id, namespace_id, document_id, chunk_index, content, created_at) "
                "SELECT :cid, :ns, id, 0, 'x', NOW() FROM documents LIMIT 1"
            ),
            {"cid": chunk_id, "ns": ns_id},
        )
        await conn.execute(
            text(
                "INSERT INTO chronicle_events "
                "(id, namespace_id, chunk_id, subject, verb, object, "
                " observation_date, referenced_date, confidence, source_text, created_at) "
                "VALUES (:id, :ns, :ch, 'alice', 'did', NULL, NOW(), NOW(), 1.0, '', NOW())"
            ),
            {"id": canonical_id, "ns": ns_id, "ch": chunk_id},
        )
        await conn.execute(
            text(
                "INSERT INTO chronicle_events "
                "(id, namespace_id, chunk_id, subject, verb, object, "
                " observation_date, referenced_date, confidence, source_text, "
                " invalidated_at, invalidated_by, merged_into_event_id, created_at) "
                "VALUES (:id, :ns, :ch, 'alice', 'did', NULL, NOW(), NOW(), 1.0, '', "
                "        NOW(), :iby, :merged, NOW())"
            ),
            {
                "id": tail_id,
                "ns": ns_id,
                "ch": chunk_id,
                "iby": uuid4(),
                "merged": canonical_id,
            },
        )

    # Sanity: tail's merged_into_event_id points at canonical.
    async with pg_engine.connect() as conn:
        before = (
            await conn.execute(
                text("SELECT merged_into_event_id FROM chronicle_events WHERE id = :id"),
                {"id": tail_id},
            )
        ).scalar_one()
    assert before == canonical_id

    # Delete the canonical row — the tail's FK should SET NULL.
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM chronicle_events WHERE id = :id"),
            {"id": canonical_id},
        )

    async with pg_engine.connect() as conn:
        after = (
            await conn.execute(
                text("SELECT merged_into_event_id FROM chronicle_events WHERE id = :id"),
                {"id": tail_id},
            )
        ).scalar_one()
    assert after is None, "expected ON DELETE SET NULL behavior on tail row"


# ---------------------------------------------------------------------------
# SQLite dialect-gating: columns added, partial index skipped silently.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_034_sqlite_dialect_gated(sqlite_engine: AsyncEngine) -> None:
    """On sqlite_lance, the columns get added; partial index is skipped."""
    async with sqlite_engine.connect() as conn:
        rows = (await conn.execute(text("PRAGMA table_info(chronicle_events)"))).fetchall()
    cols = {row[1] for row in rows}
    for expected in ("invalidated_at", "invalidated_by", "merged_into_event_id"):
        assert expected in cols, f"chronicle_events.{expected} missing on SQLite; have {sorted(cols)}"

    async with sqlite_engine.connect() as conn:
        idx_rows = (
            await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='index' AND name = 'ix_chronicle_events_live'")
            )
        ).fetchall()
    assert idx_rows == [], f"unexpected partial index on SQLite: {idx_rows}"


# ---------------------------------------------------------------------------
# Downgrade test (PG) — apply, downgrade, columns + index + constraint gone.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_034_downgrade_reverses(pg_engine: AsyncEngine) -> None:
    assert await _column_info(pg_engine, "chronicle_events", "invalidated_at") is not None

    import asyncio

    from alembic import command
    from alembic.config import Config

    # Downgrade to an explicit target revision (the revision immediately
    # before the migration under test), not a "-1" step count. The
    # pg_engine fixture upgrades to the current head, which is now several
    # revisions past 034, so "-1" would only revert head, not 034. Pinning
    # to the predecessor revision (033_bitemporal_columns) is robust as the
    # chain grows.
    sync_url = DATABASE_URL.replace("+asyncpg", "")
    migrations_dir = Path(_MIGRATION_034.__file__).resolve().parents[0].parent
    cfg = Config()
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    cfg.set_main_option("version_table", "khora_alembic_version")
    cfg.set_main_option("version_table_schema", "public")
    await asyncio.to_thread(command.downgrade, cfg, "033_bitemporal_columns")

    for col in ("invalidated_at", "invalidated_by", "merged_into_event_id"):
        info = await _column_info(pg_engine, "chronicle_events", col)
        assert info is None, f"chronicle_events.{col} still present after downgrade"

    async with pg_engine.connect() as conn:
        idx = (
            await conn.execute(text("SELECT indexname FROM pg_indexes WHERE indexname = 'ix_chronicle_events_live'"))
        ).fetchall()
    assert idx == [], f"partial index still present after downgrade: {idx}"
