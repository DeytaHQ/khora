"""Alembic migration environment — programmatic + CLI compatible."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
from logging.config import fileConfig

from alembic import context
from alembic.ddl.impl import DefaultImpl
from alembic.script import ScriptDirectory
from sqlalchemy import Column, MetaData, PrimaryKeyConstraint, String, Table, pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from khora.db.models import Base
from khora.db.session import _DatabaseAheadError

# ── Configuration ──────────────────────────────────────────────
config = context.config

# Only configure Python logging when running from alembic CLI (has .ini file)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

target_metadata = Base.metadata

# Dedicated version table — avoids collision with downstream apps
VERSION_TABLE = "khora_alembic_version"

# version_num column width. Alembic's default is 32, but Khora revision IDs
# (e.g. "022_promote_external_id_index_unique") exceed that. Widened to 64.
VERSION_NUM_LENGTH = 64

# Advisory lock ID — deterministic int64 from hashlib, unique to khora migrations
LOCK_ID = int.from_bytes(hashlib.md5(b"khora_migrations", usedforsecurity=False).digest()[:8], "big", signed=True)


# Override Alembic's hardcoded String(32) for the version table. This ensures
# fresh databases get a wider column on initial CREATE — without this, the
# first revision longer than 32 chars would fail on INSERT. Existing databases
# are widened by migration 026_widen_alembic_version_column.
def _version_table_impl(
    self: DefaultImpl,
    *,
    version_table: str,
    version_table_schema: str | None,
    version_table_pk: bool,
    **_kw: object,
) -> Table:
    vt = Table(
        version_table,
        MetaData(),
        Column("version_num", String(VERSION_NUM_LENGTH), nullable=False),
        schema=version_table_schema,
    )
    if version_table_pk:
        vt.append_constraint(PrimaryKeyConstraint("version_num", name=f"{version_table}_pkc"))
    return vt


DefaultImpl.version_table_impl = _version_table_impl  # type: ignore[method-assign]


def _get_url() -> str:
    """Resolve database URL from programmatic config or environment."""
    # Programmatic mode: URL injected via config attribute
    url = config.attributes.get("database_url", "")
    if not url:
        # CLI mode: check sqlalchemy.url (set via alembic.ini or Config.set_main_option)
        url = config.get_main_option("sqlalchemy.url") or ""
        # Ignore the placeholder used in alembic.ini
        if url.startswith("driver://"):
            url = ""
    if not url:
        # CLI mode: fall back to env var
        url = os.getenv("KHORA_DATABASE_URL", "")
    if not url:
        raise ValueError("No database URL. Set KHORA_DATABASE_URL or pass database_url to run_migrations().")

    # SQLite: normalize to aiosqlite for async engine
    if url.startswith("sqlite+aiosqlite://"):
        return url
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    # Postgres: already normalized — nothing to do
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
                f"Could not acquire migration advisory lock within {timeout}s. Another migration may be running."
            )
        logger.warning("Waiting for migration lock...")
        # Full jitter backoff — decorrelates concurrent callers
        # Algorithm: wait_random_exponential from tenacity / AWS Architecture Blog
        try:
            high = min(max_delay, min_delay * (2**attempt))
        except OverflowError:
            high = max_delay
        time.sleep(random.uniform(min_delay, high))  # noqa: S311
        attempt += 1


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with advisory lock (Postgres only)."""
    dialect_name = connection.dialect.name
    is_postgres = dialect_name == "postgresql"
    is_sqlite = dialect_name == "sqlite"

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        # SQLite requires batch mode for ALTER operations that involve FKs/constraints
        render_as_batch=is_sqlite,
    )

    with context.begin_transaction():
        if is_postgres:
            _acquire_advisory_lock(connection)

        # Ahead-detection: skip if DB is at a revision this version doesn't know.
        # Use information_schema.tables (SQL-standard, respects search_path) to check
        # existence before querying the version table. Querying a missing table inside
        # an explicit transaction puts PostgreSQL into ABORTED state, preventing
        # context.run_migrations() from running (InFailedSQLTransactionError).
        if is_sqlite:
            table_exists = connection.execute(
                text("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=:table)"),
                {"table": VERSION_TABLE},
            ).scalar()
        else:
            table_exists = connection.execute(
                text("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = :table)"),
                {"table": VERSION_TABLE},
            ).scalar()

        # Pre-migration widen: existing PostgreSQL deployments may have version_num
        # at the Alembic default VARCHAR(32). Khora revision IDs (e.g.
        # "022_promote_external_id_index_unique") exceed 32 chars, so the next
        # migration step would fail when Alembic writes the new revision. Widen
        # in-place before running migrations. Idempotent: skipped if already wide.
        if is_postgres and table_exists:
            current_width = connection.execute(
                text(
                    "SELECT character_maximum_length FROM information_schema.columns "
                    "WHERE table_name = :table AND column_name = 'version_num'"
                ),
                {"table": VERSION_TABLE},
            ).scalar()
            if current_width is not None and current_width < VERSION_NUM_LENGTH:
                connection.execute(
                    text(
                        f"ALTER TABLE {VERSION_TABLE} "  # noqa: S608
                        f"ALTER COLUMN version_num TYPE VARCHAR({VERSION_NUM_LENGTH})"
                    )
                )

        current_rev = None
        if table_exists:
            try:
                result = connection.execute(text(f"SELECT version_num FROM {VERSION_TABLE} LIMIT 1"))  # noqa: S608
                row = result.fetchone()
                current_rev = row[0] if row else None
            except Exception:
                # Version SELECT failed after table was confirmed present (e.g. permission
                # denied, transient error). Treat as no current revision so migrations
                # can still proceed.
                logger.warning(
                    "Could not read current revision from %s — proceeding without ahead-detection.",
                    VERSION_TABLE,
                )

        if current_rev is not None:
            known_revisions = {r.revision for r in ScriptDirectory.from_config(config).walk_revisions()}
            if current_rev not in known_revisions:
                logger.warning(
                    "Database at revision %s which is not recognized by this Khora version "
                    "— skipping migrations (database is ahead).",
                    current_rev,
                )
                raise _DatabaseAheadError(current_rev)

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
        # SQLite: enable foreign keys so FK constraints behave consistently
        # with Postgres during batch ALTER operations.
        if connection.dialect.name == "sqlite":
            await connection.execute(text("PRAGMA foreign_keys = ON"))
        await connection.run_sync(do_run_migrations)
        # Explicitly commit — Alembic's "non-transactional DDL" path on SQLite
        # doesn't issue COMMIT, and SQLAlchemy's async connection does not
        # auto-commit on close. Without this, all DDL is lost. Safe on Postgres:
        # the surrounding context.begin_transaction() already commits, and this
        # is a no-op after commit.
        await connection.commit()

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
