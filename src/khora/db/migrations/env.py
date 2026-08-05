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

from khora.config._secrets import redact_dsn
from khora.db.migrations._schema_config import (  # noqa: F401  (re-exported for callers importing via env.py)
    configured_embedding_dimension,
    configured_use_halfvec,
    full_precision_hnsw_supported,
)
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
    """Block until pg_advisory_lock is acquired, with timeout.

    Uses full jitter exponential backoff to decorrelate concurrent callers
    (algorithm: ``wait_random_exponential`` from tenacity / AWS Architecture Blog).

    Session-scoped lock — released explicitly in do_run_migrations' finally block.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if min_delay >= max_delay:
        raise ValueError(f"min_delay ({min_delay}) must be < max_delay ({max_delay})")
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        acquired = connection.execute(
            text("SELECT pg_try_advisory_lock(:lock_id)"),
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


def _release_advisory_lock(connection: Connection) -> None:
    """Release the session-scoped migration advisory lock. Best-effort."""
    try:
        connection.execute(text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": LOCK_ID})
    except Exception:
        logger.warning("Failed to release migration advisory lock — it will clear on connection close.")


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with advisory lock (Postgres only)."""
    dialect_name = connection.dialect.name
    is_postgres = dialect_name == "postgresql"
    is_sqlite = dialect_name == "sqlite"

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        # Autogenerate only: this makes `alembic revision --autogenerate` RENDER
        # alter operations in batch form. It has no effect at upgrade time and
        # does not drive the table rebuilds — those come from explicit
        # `op.batch_alter_table(...)` calls in the revision bodies. See the FK
        # pragma comment in run_async_migrations() for why that distinction
        # matters.
        render_as_batch=is_sqlite,
        # Commit each migration's DDL and version stamp atomically so a mid-chain
        # failure never leaves the recorded revision ahead of the applied schema.
        transaction_per_migration=True,
    )

    # Session-scoped advisory lock acquired BEFORE alembic's transaction
    # demarcation: with transaction_per_migration each migration commits its own
    # transaction, which would release a transaction-scoped lock after the first
    # step. A session-scoped lock is held for the whole run regardless and is
    # released deterministically in the finally below (same NullPool connection).
    if is_postgres:
        _acquire_advisory_lock(connection)
    try:
        with context.begin_transaction():
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
    finally:
        if is_postgres:
            _release_advisory_lock(connection)


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
    """Run migrations in 'online' mode with async engine.

    Connection-setup errors raised by the underlying driver often embed the
    plaintext DSN (asyncpg in particular formats the full
    ``postgresql://user:pass@host/db`` string into its
    ``InvalidPasswordError`` / ``CannotConnectNowError`` messages). Wrap the
    exception path with ``redact_dsn`` so the userinfo segment never reaches
    log sinks or Logfire span attributes.
    """
    url = _get_url()
    try:
        connectable = create_async_engine(url, poolclass=pool.NullPool)

        async with connectable.connect() as connection:
            # SQLite: foreign key enforcement MUST be OFF on the migration
            # connection.
            #
            # Ten revisions call ``op.batch_alter_table(...)`` explicitly in
            # their bodies, across four tables — ``documents``
            # (016 / 037 / 055 / 056), ``entities`` (008), ``memory_namespaces``
            # (001 / 010 / 011 / 012 / 013) and ``permissions`` (010). Re-derive
            # both the count and the list by grepping the versions/ directory,
            # not from this comment, if a revision is ever added.
            # (Not ``render_as_batch`` above: that is an *autogenerate rendering*
            # option and has no effect at upgrade time. Removing it would not
            # disable any of these rebuilds.) Batch mode performs SQLite's
            # documented table-rebuild procedure: create temp table, copy rows,
            # DROP TABLE, rename. That procedure's step 1 is "disable foreign
            # key constraints" — and Alembic implements the copy/drop/rename
            # steps but never issues that pragma itself, so it has to be done
            # here on its behalf.
            #
            # With enforcement left on, the rebuild's DROP TABLE performs an
            # implicit DELETE FROM that fires every inbound ON DELETE CASCADE,
            # transitively, before the rename puts the table back. Closures
            # against the schema AS IT STANDS AT HEAD — the figures that matter
            # for "may this pragma be turned back on today", which is the
            # decision this comment exists to inform:
            #   documents         ->  chunks, keyword_chunks, chronicle_events
            #                         (and, via the FTS triggers, chunks_fts)
            #   entities          ->  relationships, temporal_edges,
            #                         time_edge_links
            #   memory_namespaces ->  13 direct, 14 transitive, out of the
            #                         schema's 24 tables. The non-children are
            #                         permissions, khora_dream_runs,
            #                         khora_hook_subscriptions, the version
            #                         table, and the chunks_fts* shadow tables
            #                         (which empty anyway via the triggers).
            #   permissions       ->  nothing; it has no inbound FKs at all, so
            #                         010's rebuild of it is harmless either way.
            #
            # Those are NOT what the historical revisions destroyed — each one
            # only reaches what existed at its own point in the chain, and no
            # revision rebuilds memory_namespaces at head. Measured per revision:
            # 001 -> 8, 010 / 011 / 012 / 013 -> 11 each, 008 -> 3, 016 -> 1,
            # 037 -> 2, 055 and 056 -> 3. Do not quote the head figures as the
            # blast radius of a past revision.
            # On the embedded stack LanceDB still holds the embeddings, so the
            # result is orphaned vectors rather than merely missing rows.
            #
            # From early starting revisions it does not even get that far: the
            # cascade trips a constraint, the per-migration transaction rolls
            # back, and the database is stranded at its starting revision, unable
            # to upgrade at all. ``PRAGMA integrity_check`` still reports ok — the
            # file is fine, the chain simply cannot move. No gate caught any of
            # this because they all build the chain on an empty database, where
            # a zero-row cascade is invisible.
            #
            # This cannot be fixed inside a revision body. The pragma takes
            # effect only when no transaction is actually open, and pysqlite
            # defers the real BEGIN until the first DML — so an in-body pragma
            # *appears* to work (set before any DML it reports 0 and children
            # survive) yet silently becomes a no-op the moment any DML precedes
            # it, reporting the old value with no error. 056 runs its backfill
            # UPDATE before its batch copy, which is exactly that case. Under a
            # genuinely open transaction it is a no-op in either order.
            # ``PRAGMA defer_foreign_keys`` does not help either — it defers
            # violation *checking*, and a cascade is an action, not a violation.
            #
            # What this gives up: the migration run no longer rejects FK
            # violations it previously would have — a revision that deleted
            # parent rows would now orphan children silently instead of
            # cascading. Nothing in the chain does that today, and the
            # populated-database migration test (which asserts row counts and a
            # clean ``PRAGMA foreign_key_check`` after upgrading to head) is what
            # keeps it that way. The constraints themselves are unaffected: they
            # remain in the schema and are enforced on application connections.
            #
            # BEFORE YOU TURN THIS BACK ON — two things that look like reasons
            # to, and are not:
            #
            # 1. "Add a PRAGMA foreign_key_check guard and re-enable it." The
            #    check does not detect this bug. Measured: it returns **clean on
            #    the damaged database**, because a cascade deletes children
            #    precisely so that no violation remains — the result is a
            #    consistent database that is merely empty. Only row counts catch
            #    it, which is why the test asserts those. As a runtime gate it
            #    would be worse than useless: it would still miss the cascade
            #    while newly wedging the chain for anyone carrying a pre-existing
            #    orphan, including one created by this very bug in 016/037/055,
            #    who would have no remedy. The check earns its place in the test
            #    only as a control for the inverse risk named above.
            #
            # 2. "Enforcement during migrations sounds safer." It was tried. The
            #    ON setting arrived in #411 alongside render_as_batch, with a
            #    comment claiming it made batch ALTER behave "consistently with
            #    Postgres" — the rationale was backwards, since SQLite's own
            #    rebuild procedure requires enforcement OFF. No bug report and no
            #    test motivated it, and because every gate built the chain on an
            #    empty database it survived three further revisions. Re-enabling
            #    repeats #411.
            #
            # Set explicitly rather than relying on SQLite's default — the
            # default is overridable at compile time (SQLITE_DEFAULT_FOREIGN_KEYS),
            # so being explicit states a requirement instead of inheriting one.
            if connection.dialect.name == "sqlite":
                await connection.execute(text("PRAGMA foreign_keys = OFF"))
            await connection.run_sync(do_run_migrations)
            # Explicitly commit any transaction still open on the connection.
            # Still required after the transaction_per_migration flip:
            #   - SQLite's non-transactional-DDL path does not issue its own COMMIT.
            #   - On Postgres the outer context.begin_transaction() is now a
            #     nullcontext (it no longer commits), and each migration commits its
            #     own DDL + version stamp inside its per-migration transaction. This
            #     trailing commit flushes only the residual work SQLAlchemy autobegan
            #     outside those per-migration transactions — the ahead-detection /
            #     pre-widen reads and the advisory unlock — which is otherwise left
            #     pending in the no-migration-step edge case (e.g. an up-to-date DB).
            # SQLAlchemy's async connection does not auto-commit on close, so without
            # this that residual transaction would roll back. Harmless when nothing
            # is pending.
            await connection.commit()

        await connectable.dispose()
    except _DatabaseAheadError:
        # Sentinel: no DSN content, propagate unchanged for the session.py handler.
        raise
    except Exception as exc:
        redacted = redact_dsn(str(exc))
        if redacted != str(exc):
            # Re-raise with redacted message. Try to preserve the original
            # exception type, but some SQLAlchemy / asyncpg exception types
            # require multiple positional args (e.g. NoSuchModuleError(name))
            # — passing just the redacted message would raise TypeError.
            # Fall back to RuntimeError with ``from None`` so the unredacted
            # DSN is not retained in __cause__ and captured by observability
            # tools (Logfire, Sentry, traceback.print_exception).
            try:
                redacted_exc = type(exc)(redacted)
            except TypeError:
                logger.debug(
                    "Migration error type (original suppressed for DSN redaction): %s",
                    type(exc).__name__,
                )
                raise RuntimeError(redacted) from None
            raise redacted_exc.with_traceback(exc.__traceback__) from None
        raise


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
