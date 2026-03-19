"""Alembic migration environment — programmatic + CLI compatible."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from logging.config import fileConfig

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from khora.db.models import Base

# ── Configuration ──────────────────────────────────────────────
config = context.config

# Only configure Python logging when running from alembic CLI (has .ini file)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

target_metadata = Base.metadata

# Dedicated version table — avoids collision with downstream apps
VERSION_TABLE = "khora_alembic_version"

# Advisory lock ID — fixed int64, unique to khora migrations
LOCK_ID = 0x4B484F5241  # "KHORA" in hex


def _get_url() -> str:
    """Resolve database URL from programmatic config or environment."""
    # Programmatic mode: URL injected via config attribute
    url = config.attributes.get("database_url", "")
    if not url:
        # CLI mode: fall back to env var
        url = os.getenv("KHORA_DATABASE_URL", "")
    if not url:
        raise ValueError("No database URL. Set KHORA_DATABASE_URL or pass database_url to run_migrations().")
    # Normalize to asyncpg
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _seed_version_table(connection: Connection) -> None:
    """One-time: copy stamp from 'alembic_version' to 'khora_alembic_version'.

    Existing deployments track migration state in the default
    'alembic_version' table. On first run with the new version table,
    copy the current revision so Alembic doesn't re-run everything.

    Idempotent — skips if khora_alembic_version already has rows,
    or if alembic_version doesn't exist / is empty.
    """
    # Check if khora_alembic_version already has data
    result = connection.execute(text(f"SELECT 1 FROM {VERSION_TABLE} LIMIT 1"))
    if result.fetchone() is not None:
        return  # already seeded

    # Check if old table exists and has data
    try:
        result = connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
        row = result.fetchone()
    except Exception:
        # alembic_version table doesn't exist (fresh database)
        return

    if row is not None:
        connection.execute(
            text(f"INSERT INTO {VERSION_TABLE} (version_num) VALUES (:v)"),
            {"v": row[0]},
        )
        logger.info("Seeded %s from alembic_version (revision: %s)", VERSION_TABLE, row[0])


def _acquire_advisory_lock(connection: Connection, timeout: float = 60.0) -> None:
    """Block until pg_advisory_xact_lock is acquired, with timeout.

    Transaction-scoped lock auto-releases on commit/rollback.
    """
    deadline = time.monotonic() + timeout
    while True:
        acquired = connection.execute(
            text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
            {"lock_id": LOCK_ID},
        ).scalar()
        if acquired:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Could not acquire migration advisory lock within {timeout}s. " "Another migration may be running."
            )
        logger.warning("Waiting for migration lock...")
        time.sleep(0.5)


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with advisory lock and version table seeding."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
    )

    with context.begin_transaction():
        _acquire_advisory_lock(connection)
        _seed_version_table(connection)
        context.run_migrations()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (SQL script generation)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    url = _get_url()
    connectable = create_async_engine(url, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
