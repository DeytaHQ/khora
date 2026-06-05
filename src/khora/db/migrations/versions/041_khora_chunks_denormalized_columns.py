"""Add denormalized document-grained columns to khora_chunks.

Revision ID: 041_khora_chunks_denormalized_columns
Revises: 040_chunks_last_accessed_at
Create Date: 2026-06-05

Schema changes:

* Eight nullable columns on the ``khora_chunks`` temporal-store table,
  copied down from the parent document so recall filters can be applied
  directly on the chunk row without a join:

  - ``source_type`` ``VARCHAR`` — document source category (nullable; no
    default — chunks predating the backfill stay NULL).
  - ``source_name`` ``VARCHAR`` — human-readable source label.
  - ``source_url`` ``TEXT`` — origin URL (may be long).
  - ``source_timestamp`` ``TIMESTAMPTZ`` — the producer's verbatim event
    time, distinct from ``occurred_at`` (the chunk event-time).
  - ``external_id`` ``VARCHAR`` — upstream identifier.
  - ``content_type`` ``VARCHAR`` — MIME / content category.
  - ``source`` ``VARCHAR`` — raw source identifier.
  - ``title`` ``TEXT`` — document title (may be long).

  All eight are nullable with NO server_default. This is a pure catalog
  update: a nullable ``ADD COLUMN`` with no default is an instant catalog
  change on PG 11+ — no table rewrite — so we deliberately avoid the
  NOT-NULL-DEFAULT fast-path (and its ``server_version_num`` assert) used
  by migration 038.

Index note: NO indexes are created here. The indexed subset
(``source_type``, ``source_name``, ``source_timestamp`` (partial),
``external_id``, ``content_type``) is created by the runtime
``PgVectorTemporalStore.connect()`` via ``CREATE INDEX IF NOT EXISTS`` and
will be added against the populated columns in a later post-backfill step.
This migration only widens the catalog.

Lock-timeout safety: a Postgres-only ``SET lock_timeout = '5s'`` is issued
**before any DDL** so the ``ADD COLUMN`` AccessExclusiveLock acquisition is
bounded. The setting is scoped to the migration's enclosing transaction
(env.py wraps the whole upgrade in a single transaction; the SET expires at
COMMIT). On lock-timeout the upgrade logs ``khora.migration.applied`` at
ERROR with ``lock_timeout_tripped=True`` (detected by the Postgres SQLSTATE
``55P03`` / ``lock_not_available`` on ``OperationalError.orig.pgcode`` — any
other ``OperationalError`` class logs ``lock_timeout_tripped=False`` so
dashboards don't conflate deadlocks / connection drops with the lock-timeout
signal). Either path re-raises so Alembic rolls back the transaction.

Runtime-def convergence: ``khora_chunks`` is created at runtime by
``PgVectorTemporalStore.connect()`` (via ``metadata.create_all``) and by
``SQLiteLanceTemporalStore``'s DDL — it is not part of the Alembic-managed
schema. On fresh deploys and on the sqlite_lance test fixture the table
doesn't exist when the chain runs; the new columns then land from the
updated ``khora_chunks_table`` definition. This migration is required for
existing deployments where ``khora_chunks`` predates these columns.

Cross-dialect: this migration is Postgres-only and early-returns on other
dialects (SQLite). The sqlite_lance fixture stack runs the full Alembic
chain, so the early-return keeps it green; on SQLite the columns come from
the runtime table definition, not Alembic.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.exc import OperationalError

revision: str = "041_khora_chunks_denormalized_columns"
down_revision: str | Sequence[str] | None = "040_chunks_last_accessed_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"


def _new_columns() -> list[sa.Column]:
    """Build the eight denormalized columns fresh on each call.

    All nullable with no server_default. Seven are string/text;
    ``source_timestamp`` is timezone-aware. A ``Column`` object cannot be
    reused across multiple ``op.add_column`` invocations once bound, so we
    construct them here rather than at module scope.
    """
    return [
        sa.Column("source_type", sa.String(64), nullable=True),
        sa.Column("source_name", sa.String(255), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("external_id", sa.String(255), nullable=True),
        sa.Column("content_type", sa.String(128), nullable=True),
        sa.Column("source", sa.String(255), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
    ]


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
        # case the new columns land when the runtime creates the table from
        # the updated ``khora_chunks_table`` definition. The migration is
        # still required for existing production deployments where
        # ``khora_chunks`` was created before these columns existed.
        return

    # Bound the ADD COLUMN AccessExclusiveLock acquisition — a stuck
    # pg_stat_activity entry on khora_chunks cannot stall the deploy past
    # 5 seconds. Issued before any DDL. Nullable ADD COLUMN with no default
    # is an instant catalog update (no table rewrite) on PG 11+.
    op.execute("SET lock_timeout = '5s'")

    for column in _new_columns():
        op.add_column("khora_chunks", column)


def upgrade() -> None:
    if not _is_postgres():
        return

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
    if not _is_postgres():
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("khora_chunks"):
        return
    for column in reversed(_new_columns()):
        op.drop_column("khora_chunks", column.name)
