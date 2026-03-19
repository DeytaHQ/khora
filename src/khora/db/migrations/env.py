"""Alembic migration environment — programmatic + CLI compatible."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from logging.config import fileConfig
import os
import random
import time

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

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

# Advisory lock ID — deterministic int64 from hashlib, unique to khora migrations
LOCK_ID = int.from_bytes(hashlib.md5(b"khora_migrations").digest()[:8], "big", signed=True)


def _get_url() -> str:
    """Resolve database URL from programmatic config or environment."""
    # Programmatic mode: URL injected via config attribute
    url = config.attributes.get("database_url", "")
    if not url:
        # CLI mode: fall back to env var
        url = os.getenv("KHORA_DATABASE_URL", "")
    if not url:
        raise ValueError("No database URL. Set KHORA_DATABASE_URL or pass database_url to run_migrations().")
    # Already normalized — nothing to do
    if "+asyncpg" in url:
        return url
    # Normalize to asyncpg
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _acquire_advisory_lock(
    connection: Connection,
    timeout: float = 60.0,
    min_delay: float = 0.05,
    max_delay: float = 2.0,
) -> None:
    """Block until pg_advisory_xact_lock is acquired, with timeout.

    Uses full jitter exponential backoff to decorrelate concurrent callers
    (algorithm: ``wait_random_exponential`` from tenacity / AWS Architecture Blog).

    Transaction-scoped lock auto-releases on commit/rollback.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if min_delay >= max_delay:
        raise ValueError(f"min_delay ({min_delay}) must be < max_delay ({max_delay})")
    deadline = time.monotonic() + timeout
    attempt = 0
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
        # Full jitter backoff — decorrelates concurrent callers
        # Algorithm: wait_random_exponential from tenacity / AWS Architecture Blog
        try:
            high = min(max_delay, min_delay * (2**attempt))
        except OverflowError:
            high = max_delay
        time.sleep(random.uniform(min_delay, high))
        attempt += 1


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with advisory lock."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
    )

    with context.begin_transaction():
        _acquire_advisory_lock(connection)
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
