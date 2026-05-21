"""Add chunker_info to khora_chunks.

Revision ID: 038_khora_chunks_chunker_info
Revises: 037_recall_response_format
Create Date: 2026-05-21

Schema changes:

* ``khora_chunks.chunker_info`` ``JSONB`` NOT NULL DEFAULT ``'{}'::jsonb`` —
  per-chunk metadata describing which chunker produced the chunk and
  with what params. Mirrors the ``chunks.chunker_info`` column added in
  migration 037, scoped to the temporal-store table managed by the
  vectorcypher engine.

PostgreSQL version requirement: ``chunker_info JSONB NOT NULL DEFAULT
'{}'::jsonb`` relies on the PG ≥11 ``ADD COLUMN ... NOT NULL DEFAULT``
fast-path to avoid a full ``khora_chunks`` table rewrite. The upgrade
asserts ``server_version_num >= 110000`` and raises on older Postgres
so we fail fast instead of silently triggering a multi-hour rewrite.

Lock-timeout safety: a Postgres-only ``SET lock_timeout = '5s'`` is
issued **before any DDL** so the ``ADD COLUMN`` AccessExclusiveLock
acquisition is bounded. The setting is scoped to the migration's
enclosing transaction (env.py wraps the whole upgrade in a single
transaction; the SET expires at COMMIT). On lock-timeout the upgrade
logs ``khora.migration.applied`` at ERROR with
``lock_timeout_tripped=True`` (detected by the Postgres SQLSTATE
``55P03`` / ``lock_not_available`` on the
``OperationalError.orig.pgcode`` — any other ``OperationalError``
class logs ``lock_timeout_tripped=False`` so dashboards don't conflate
deadlocks / connection drops / syntax errors with the lock-timeout
signal). Either path re-raises so Alembic rolls back the transaction.

Cross-dialect: the sqlite_lance fixture stack runs the full Alembic
chain. The Postgres ``JSONB`` type is replaced with portable ``sa.JSON``
(TEXT-backed) on SQLite, mirroring migration 037's pattern.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError

from alembic import op

revision: str = "038_khora_chunks_chunker_info"
down_revision: str | Sequence[str] | None = "037_recall_response_format"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_MIN_PG_VERSION = 110000  # PG 11 — chunker_info NOT NULL DEFAULT fast-path

# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _is_lock_timeout(exc: OperationalError) -> bool:
    """Distinguish a real lock_timeout trip from any other OperationalError.

    OperationalError is a broad SQLAlchemy class — it wraps deadlocks,
    connection drops, syntax errors, server shutdowns, AND lock_timeout
    failures. The structured log field ``lock_timeout_tripped`` must
    only be True for the latter so monitoring dashboards aren't misled.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return False
    return getattr(orig, "pgcode", None) == _PG_LOCK_NOT_AVAILABLE


def _assert_pg_version_ok() -> None:
    """Fail fast if Postgres is older than 11.

    ``khora_chunks.chunker_info NOT NULL DEFAULT '{}'::jsonb`` triggers
    a full table rewrite on PG ≤ 10 — a multi-hour outage on a large
    khora_chunks table. The hosted version meets this; we assert here
    so a misconfigured deploy fails in seconds rather than silently
    rewriting.
    """
    bind = op.get_bind()
    version_num = bind.execute(sa.text("SELECT current_setting('server_version_num')::int")).scalar()
    if version_num is None or int(version_num) < _MIN_PG_VERSION:
        raise RuntimeError(
            f"Migration {revision} requires PostgreSQL >= 11 "
            f"(server_version_num >= {_MIN_PG_VERSION}); got {version_num}. "
            "The khora_chunks.chunker_info NOT NULL DEFAULT fast-path requires PG 11+."
        )


def _upgrade_impl() -> None:
    """Perform the schema change."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("khora_chunks"):
        # ``khora_chunks`` is created at runtime by the vectorcypher
        # ``PgVectorTemporalStore.connect()`` (via ``metadata.create_all``)
        # and by ``SQLiteLanceTemporalStore``'s DDL — it is not part of the
        # Alembic-managed schema. On fresh deploys and on the sqlite_lance
        # test fixture the table doesn't exist when the chain runs; in that
        # case the new column lands when the runtime creates the table from
        # the updated ``khora_chunks_table`` definition. The migration is
        # still required for existing production deployments where
        # ``khora_chunks`` was created before this column existed.
        return

    is_pg = _is_postgres()

    if is_pg:
        _assert_pg_version_ok()
        # Bound the ADD COLUMN AccessExclusiveLock acquisition — a stuck
        # pg_stat_activity entry on khora_chunks cannot stall the deploy
        # past 5 seconds. Issued before any DDL.
        op.execute("SET lock_timeout = '5s'")

        op.add_column(
            "khora_chunks",
            sa.Column(
                "chunker_info",
                JSONB,
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
        )
    else:
        # SQLite: Postgres JSONB is unrenderable. Use sa.JSON (TEXT-backed)
        # with a JSON literal default — mirrors migration 037's pattern for
        # the corresponding ``chunks.chunker_info`` column.
        op.add_column(
            "khora_chunks",
            sa.Column(
                "chunker_info",
                sa.JSON,
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
        )


def upgrade() -> None:
    start = time.monotonic()
    try:
        _upgrade_impl()
    except OperationalError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.bind(
            migration_id=revision,
            duration_ms=duration_ms,
            lock_timeout_tripped=_is_lock_timeout(exc),
        ).error("khora.migration.applied")
        # Bare ``raise`` re-raises the active exception with the original
        # traceback preserved. env.py's wrapping context.begin_transaction()
        # rolls back all DDL from this revision.
        raise

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.bind(
        migration_id=revision,
        duration_ms=duration_ms,
        lock_timeout_tripped=False,
    ).info("khora.migration.applied")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("khora_chunks"):
        return
    op.drop_column("khora_chunks", "chunker_info")
