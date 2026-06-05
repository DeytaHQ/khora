"""Widen two denormalized ``khora_chunks`` columns to match their source widths.

Revision ID: 042_widen_khora_chunks_source_external_id
Revises: 041_khora_chunks_denormalized_columns
Create Date: 2026-06-05

Schema changes:

* ``khora_chunks.source`` ``VARCHAR(255)`` -> ``TEXT``
* ``khora_chunks.external_id`` ``VARCHAR(255)`` -> ``VARCHAR(512)``

Both columns were added (empty, no backfill) by the immediately preceding
migration 041. An upcoming write-path change plus a one-shot backfill will
copy these two fields down from the parent ``documents`` table, whose source
columns are wider than the 255-char placeholder 041 used:

  - ``documents.source`` is ``VARCHAR(1024)`` -> chunk side widened to ``TEXT``
    (``TEXT`` >= 1024, so no value can truncate).
  - ``documents.external_id`` is ``VARCHAR(512)`` -> chunk side widened to the
    matching ``VARCHAR(512)``.

Without this widening the backfill ``UPDATE khora_chunks ... FROM documents``
would raise ``StringDataRightTruncation`` on long values. The columns are
EMPTY today, so widening a ``varchar`` (and ``varchar`` -> ``text``) is a pure
catalog change in Postgres -- no table rewrite, instant on any table size,
zero risk.

Lock-timeout safety: a Postgres-only ``SET lock_timeout = '5s'`` is issued
**before any DDL** so the ``ALTER COLUMN`` AccessExclusiveLock acquisition is
bounded, mirroring migration 041. The setting is scoped to the migration's
enclosing transaction (env.py wraps the whole upgrade in a single
transaction; the SET expires at COMMIT). On lock-timeout the upgrade logs
``khora.migration.applied`` at ERROR with ``lock_timeout_tripped=True``
(detected via the Postgres SQLSTATE ``55P03`` / ``lock_not_available`` on
``OperationalError.orig.pgcode``); any other ``OperationalError`` logs
``lock_timeout_tripped=False`` so dashboards don't conflate deadlocks /
connection drops with the lock-timeout signal. Either path re-raises so
Alembic rolls back the transaction.

Cross-dialect: this migration is Postgres-only and early-returns on other
dialects (SQLite). The sqlite_lance fixture stack runs the full Alembic
chain, so the early-return keeps it green; on SQLite the columns come from
the runtime ``khora_chunks_table`` definition (which carries the correct
widened types), not Alembic.

Runtime-def convergence: ``khora_chunks`` is created at runtime by
``PgVectorTemporalStore.connect()`` (via ``metadata.create_all``) and is not
part of the Alembic-managed schema, so the migration is guarded by
``has_table("khora_chunks")`` and exists only for existing deployments where
the table predates this widening. Fresh deploys pick up the widened types
straight from the runtime table definition.

Downgrade narrowing-truncation guard: narrowing ``source`` back to
``VARCHAR(255)`` and ``external_id`` to ``VARCHAR(255)`` would truncate any
value longer than 255 chars (which only exists after the backfill runs).
``downgrade`` first probes for such a row and, if one is found, SKIPS the
narrowing as a no-op and logs a WARNING rather than silently destroying data.
NULLs are safe -- ``length(NULL)`` is NULL, never > 255.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.exc import OperationalError

revision: str = "042_widen_khora_chunks_source_external_id"
down_revision: str | Sequence[str] | None = "041_khora_chunks_denormalized_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# PostgreSQL SQLSTATE for "lock_not_available" -- what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _is_lock_timeout(exc: OperationalError) -> bool:
    """Distinguish a real lock_timeout trip from any other OperationalError.

    OperationalError is a broad SQLAlchemy class -- it wraps deadlocks,
    connection drops, syntax errors, server shutdowns, AND lock_timeout
    failures. The structured log field ``lock_timeout_tripped`` must only be
    True for the latter so monitoring dashboards aren't misled.
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
        # and is not part of the Alembic-managed schema. On fresh deploys and
        # on the sqlite_lance test fixture the table doesn't exist when the
        # chain runs; in that case the widened types land when the runtime
        # creates the table from the updated ``khora_chunks_table``
        # definition. The migration is still required for existing
        # deployments where ``khora_chunks`` was created with the narrower
        # placeholder widths from migration 041.
        return

    # Bound the ALTER COLUMN AccessExclusiveLock acquisition -- a stuck
    # pg_stat_activity entry on khora_chunks cannot stall the deploy past
    # 5 seconds. Issued before any DDL. Both columns are empty today, so
    # widening varchar (and varchar -> text) is an instant catalog change
    # with no table rewrite.
    op.execute("SET lock_timeout = '5s'")

    op.alter_column(
        "khora_chunks",
        "source",
        existing_type=sa.String(255),
        type_=sa.Text(),
        existing_nullable=True,
    )
    op.alter_column(
        "khora_chunks",
        "external_id",
        existing_type=sa.String(255),
        type_=sa.String(512),
        existing_nullable=True,
    )


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

    # Narrowing back to VARCHAR(255) would truncate any value longer than 255
    # chars (which can only exist once the backfill has run). Probe for such a
    # row first; if one is found, skip the narrowing rather than destroy data.
    # NULLs are safe -- length(NULL) is NULL, never > 255.
    has_long_value = (
        bind.execute(
            sa.text("SELECT 1 FROM khora_chunks WHERE length(source) > 255 OR length(external_id) > 255 LIMIT 1")
        ).first()
        is not None
    )
    if has_long_value:
        logger.bind(migration_id=revision).warning(
            "khora.migration.downgrade_skipped: khora_chunks has source/external_id "
            "values longer than 255 chars; skipping narrowing to avoid truncation"
        )
        return

    op.alter_column(
        "khora_chunks",
        "external_id",
        existing_type=sa.String(512),
        type_=sa.String(255),
        existing_nullable=True,
    )
    op.alter_column(
        "khora_chunks",
        "source",
        existing_type=sa.Text(),
        type_=sa.String(255),
        existing_nullable=True,
    )
