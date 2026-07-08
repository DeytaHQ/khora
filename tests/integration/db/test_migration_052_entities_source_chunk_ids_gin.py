"""Live-PostgreSQL integration test for migration 052 — the GIN index on
``entities.source_chunk_ids`` (#1452).

Verifies on real Postgres that migration ``052_entities_source_chunk_ids_gin``:

1. ``alembic upgrade head`` builds ``ix_entities_source_chunk_ids_gin`` as a
   GIN index (the ``&&`` array-overlap pushdown of PR #1449 / #857 / #1448).
2. Downgrade to ``051`` drops the index (symmetric ``DROP INDEX
   CONCURRENTLY``).
3. Re-running the upgrade re-creates it — proving the ``IF NOT EXISTS`` re-run
   safety of the concurrent-index DDL.

The index is Postgres-only (``CREATE INDEX CONCURRENTLY`` + GIN); the migration
is a clean no-op on SQLite, so this test skips when Postgres is unreachable.

The lifecycle rewinds the shared dev DB's alembic revision, so the restore
``upgrade head`` runs in a ``finally`` block — the DB is never left rewound,
even on assertion failure.

Run with an explicit DB URL (the shell leaks a different one)::

    KHORA_DATABASE_URL="postgresql://khora:khora@localhost:5434/khora" \
        UV_NO_SYNC=1 uv run pytest \
        tests/integration/db/test_migration_052_entities_source_chunk_ids_gin.py \
        -o addopts="" -q
"""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine

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

_HEAD = "052_entities_source_chunk_ids_gin"
_PREV = "051_documents_graph_mirror_pending"

_INDEX_NAME = "ix_entities_source_chunk_ids_gin"


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


async def _index_rows(url: str) -> list[tuple[str, str]]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                sa.text("SELECT indexname, indexdef FROM pg_indexes WHERE indexname = :name"),
                {"name": _INDEX_NAME},
            )
            return [(row[0], row[1]) for row in result.fetchall()]
    finally:
        await engine.dispose()


class TestMigration052GinIndex:
    def test_gin_index_created_dropped_and_recreated(self) -> None:
        if not _pg_reachable():
            pytest.skip("PostgreSQL not reachable (run `make dev` first)")

        cfg = _make_config(DATABASE_URL)

        # Migration 052 builds the GIN index. Idempotent when the shared dev
        # DB is already at head.
        command.upgrade(cfg, _HEAD)
        rows = asyncio.run(_index_rows(DATABASE_URL))
        assert len(rows) == 1, f"expected the GIN index at head, found {rows}"
        assert rows[0][0] == _INDEX_NAME
        assert "USING gin" in rows[0][1], f"index is not GIN: {rows[0][1]}"

        try:
            # Downgrade drops just the GIN index (052 -> 051).
            command.downgrade(cfg, _PREV)
            assert asyncio.run(_index_rows(DATABASE_URL)) == []
        finally:
            # Always restore the true chain head so the shared dev DB is never
            # left rewound — "head" (not the pinned _HEAD) stays correct once a
            # future migration lands on top of 052.
            command.upgrade(cfg, "head")

        # Re-running the upgrade re-creates the index — proves IF NOT EXISTS
        # re-run safety.
        rows = asyncio.run(_index_rows(DATABASE_URL))
        assert len(rows) == 1
        assert rows[0][0] == _INDEX_NAME
        assert "USING gin" in rows[0][1]
