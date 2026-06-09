"""Regression test: replaying the full migration chain must stay idempotent.

Reproduces the failure mode where running the whole Alembic chain to head from
a freshly reset schema succeeded the first time but failed on a subsequent
full-chain replay. The repeated run must always return
``MigrationResult(success=True, error=None)`` — never a failure and never a
surfaced Alembic ``CommandError``.

Each iteration resets ``public`` from scratch and runs ``run_migrations()`` all
the way to head, so every iteration re-applies every migration step (not a
no-op against an already-migrated database). A pre-fix build fails on the
second iteration; the fix keeps every iteration green.

Run locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest \
        tests/integration/db/test_migration_replay_idempotent.py -v \
        -m integration --no-cov
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from khora.db.session import run_migrations

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Postgres connection handling (mirrors the migration 033 integration test).
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


async def _reset_public_schema() -> None:
    """Wipe ``public`` so the migration chain applies from a clean slate.

    Mirrors the workaround in ``test_migration_033_bitemporal.py``: drop any
    leftover enum types, drop and recreate the schema, re-create the ``vector``
    extension, and pre-create ``khora_alembic_version`` with VARCHAR(64) so the
    wide revision ids apply cleanly.
    """
    eng = create_async_engine(DATABASE_URL)
    try:
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
    finally:
        await eng.dispose()


# ---------------------------------------------------------------------------
# The regression test.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_chain_replay_is_idempotent() -> None:
    """Reset + migrate the full chain to head, repeated back-to-back.

    Each iteration wipes ``public`` and runs every migration step from scratch.
    Because every step re-runs each time (not a no-op against an
    already-migrated database), a non-idempotent step surfaces on the second
    iteration onward. Every iteration must return ``success=True`` with no
    error and without taking the ahead-skip path.
    """
    if not _PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable (run `make dev` first)")

    # Six full-chain replays: a non-idempotent step (e.g. an autocommit-block
    # migration whose version stamp matches zero rows on re-apply) fails
    # probabilistically per replay, so a robust iteration count keeps the
    # pre-fix failure near-certain rather than a flaky miss.
    iterations = 6
    head: str | None = None
    for i in range(1, iterations + 1):
        await _reset_public_schema()
        result = await run_migrations(DATABASE_URL)
        assert result.success is True, f"iteration #{i} failed: {result.error}"
        assert result.error is None, f"iteration #{i} surfaced an error: {result.error}"
        # A full-chain run to head is not the "database ahead" skip path.
        assert result.skipped is False, f"iteration #{i} unexpectedly took the ahead-skip path"
        assert result.target_revision is not None, f"iteration #{i} reported no target revision"
        if head is None:
            head = result.target_revision
        else:
            assert result.target_revision == head, (
                f"iteration #{i} target_revision drifted: {result.target_revision!r} != {head!r}"
            )

    # After the final clean run the version table points at exactly one
    # revision: head.
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.connect() as conn:
            rows = (await conn.execute(text("SELECT version_num FROM khora_alembic_version"))).fetchall()
    finally:
        await eng.dispose()
    assert [row[0] for row in rows] == [head], (
        f"khora_alembic_version should hold exactly [{head!r}] after replays, got {[r[0] for r in rows]}"
    )
