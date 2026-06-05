"""Backfill khora_chunks denormalized columns and build the filter indexes.

Revision ID: 044_khora_chunks_backfill_denormalized
Revises: 043_khora_chunks_metadata_backfill
Create Date: 2026-06-05

Migration 041 *added* eight nullable denormalized document-grained columns
to the runtime-managed ``khora_chunks`` temporal store. This migration
populates them on existing rows and creates the filterable-subset indexes
so recall filters resolve without a join. Three phases, in order:

* **PHASE 1 — backfill (autocommit, per-namespace, per-batch COMMIT).**
  Copies ``source_type``, ``source_name``, ``source_url``,
  ``source_timestamp``, ``external_id``, ``content_type``, ``source`` and
  ``title`` down from the parent ``documents`` row (joined on
  ``khora_chunks.document_id = documents.id``). The update is **batched
  per namespace** — ``SELECT DISTINCT namespace_id`` then one ``UPDATE``
  per namespace — so no single transaction spans the whole ~1M-row table:
  each per-ns UPDATE auto-commits inside the autocommit block, bounding
  open snapshots / MVCC bloat / advisory-lock contention. The
  ``AND kc.source_type IS NULL`` sentinel makes the backfill **idempotent
  and restartable** — re-running (or resuming after a crash mid-loop)
  touches zero already-populated rows, because ``documents.source_type``
  is ``NOT NULL`` so a backfilled chunk is always non-NULL there.

  All eight columns are copied verbatim. The widen migration sized
  ``khora_chunks.source`` and ``khora_chunks.external_id`` to the
  ``documents`` widths, so the full producer values fit in their entirety
  and every value stays filterable.

  The runtime ``BEFORE INSERT OR UPDATE`` trigger
  ``khora_chunks_content_tsv_update`` recomputes ``content_tsv`` on every
  UPDATE, including this backfill (which never touches ``content``). It is
  disabled once before the loop and re-enabled in a ``finally`` on every
  exit path. **The finally is load-bearing, not defensive:** the backfill
  runs in autocommit, so the DISABLE commits immediately and there is no
  outer transaction to roll it back. ``SET lock_timeout = '5s'`` bounds the
  ``AccessExclusiveLock`` the DISABLE takes. The DISABLE/ENABLE pair is
  guarded on ``information_schema.triggers`` and skipped if the trigger is
  absent (a 041 sidecar apply before any runtime boot has no trigger yet —
  and then no trigger fires on the UPDATE anyway).

  **Crash caveat:** a hard kill (SIGKILL / pod OOM) *between* the committed
  DISABLE and the ENABLE leaves the TSV trigger disabled until the
  migration is re-run (the re-run re-ENABLEs it; the ``source_type IS NULL``
  sentinel makes the re-run cheap). Acceptable in the off-peak deploy
  window — an operator who sees a stale-``content_tsv`` symptom should
  re-run this migration. We use ``DISABLE TRIGGER`` (not
  ``session_replication_role = replica``, which is strictly safer but
  needs superuser that managed-PG application roles lack).

* **PHASE 2 — indexes (autocommit).** Builds the five filter indexes with
  ``CREATE INDEX CONCURRENTLY IF NOT EXISTS``, names and definitions
  byte-identical to the runtime ``PgVectorTemporalStore.connect()`` so the
  two converge (whichever runs first wins). Built *after* the backfill so
  the btrees are populated, and *before* the VACUUM so the freshly-built
  trees are tidied. A ``CREATE INDEX CONCURRENTLY`` that fails midway can
  leave an INVALID index; ``IF NOT EXISTS`` does not retry it. Recovery is
  to ``DROP`` the invalid index and re-run the migration.

* **PHASE 3 — VACUUM (ANALYZE) (autocommit), last.** Reclaims the dead
  heap tuples the (non-HOT) backfill UPDATEs produced, tidies the
  just-built btrees, and refreshes planner stats for the new indexes.

Everything runs in autocommit: the backfill batches per namespace with
per-batch COMMITs, and ``CREATE INDEX CONCURRENTLY`` / ``VACUUM`` cannot
run inside a transaction. There is no transactional phase that relies on
env.py's rollback — which is exactly why the trigger ``finally`` ENABLE is
load-bearing.

``khora_chunks`` is created at runtime by ``PgVectorTemporalStore`` and is
not part of the Alembic-managed schema. This migration is Postgres-only,
early-returns on other dialects (SQLite), and is guarded by
``has_table("khora_chunks")`` so it is a clean no-op on fresh deploys
where the table doesn't exist yet (the runtime populates the columns
itself) and on the sqlite_lance test fixture.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.exc import OperationalError

revision: str = "044_khora_chunks_backfill_denormalized"
down_revision: str | Sequence[str] | None = "043_khora_chunks_metadata_backfill"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"

_TSV_TRIGGER = "khora_chunks_content_tsv_update"

# Per-namespace backfill UPDATE. Copies the eight denormalized columns down
# from the parent ``documents`` row verbatim — the widen migration sized
# ``source`` and ``external_id`` to the ``documents`` widths, so the full
# value fits in its entirety. ``AND kc.source_type IS NULL`` keeps the
# backfill idempotent / restartable.
_BACKFILL_SQL = sa.text(
    """
    UPDATE khora_chunks AS kc
    SET source_type      = d.source_type,
        source_name      = d.source_name,
        source_url       = d.source_url,
        source_timestamp = d.source_timestamp,
        external_id      = d.external_id,
        content_type     = d.content_type,
        source           = d.source,
        title            = d.title
    FROM documents AS d
    WHERE kc.document_id = d.id
      AND kc.namespace_id = :ns
      AND kc.source_type IS NULL
    """
)

_TRIGGER_PRESENT_SQL = sa.text(
    """
    SELECT 1 FROM information_schema.triggers
    WHERE event_object_table = 'khora_chunks'
      AND trigger_name = :trigger
    LIMIT 1
    """
)

# The five filter indexes — names and definitions byte-identical to the
# runtime ``PgVectorTemporalStore.connect()`` in
# ``engines/skeleton/backends/pgvector.py`` so the two converge.
_INDEX_SQL: tuple[str, ...] = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_ns_source_type "
    "ON khora_chunks (namespace_id, source_type)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_ns_source_name "
    "ON khora_chunks (namespace_id, source_name)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_ns_source_timestamp "
    "ON khora_chunks (namespace_id, source_timestamp) WHERE source_timestamp IS NOT NULL",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_ns_external_id "
    "ON khora_chunks (namespace_id, external_id)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_ns_content_type "
    "ON khora_chunks (namespace_id, content_type)",
)

# Drop order is the reverse of create order.
_DROP_INDEX_SQL: tuple[str, ...] = (
    "DROP INDEX CONCURRENTLY IF EXISTS ix_khora_chunks_ns_content_type",
    "DROP INDEX CONCURRENTLY IF EXISTS ix_khora_chunks_ns_external_id",
    "DROP INDEX CONCURRENTLY IF EXISTS ix_khora_chunks_ns_source_timestamp",
    "DROP INDEX CONCURRENTLY IF EXISTS ix_khora_chunks_ns_source_name",
    "DROP INDEX CONCURRENTLY IF EXISTS ix_khora_chunks_ns_source_type",
)


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


def _backfill() -> None:
    """Run the per-namespace backfill loop.

    Runs entirely inside an autocommit block: each per-namespace UPDATE
    auto-commits independently, and the trigger DISABLE/ENABLE commit
    around it. The ENABLE in ``finally`` is load-bearing — there is no
    enclosing transaction to roll the committed DISABLE back.
    """
    bind = op.get_bind()
    with op.get_context().autocommit_block():
        # Bound the AccessExclusiveLock the DISABLE TRIGGER takes. Plain
        # SET (not SET LOCAL) — there is no enclosing user transaction to
        # scope it to; the NullPool connection is disposed at migration end.
        op.execute("SET lock_timeout = '5s'")

        trigger_present = bind.execute(_TRIGGER_PRESENT_SQL, {"trigger": _TSV_TRIGGER}).scalar() is not None

        try:
            if trigger_present:
                op.execute(f"ALTER TABLE khora_chunks DISABLE TRIGGER {_TSV_TRIGGER}")
            namespace_ids = bind.execute(sa.text("SELECT DISTINCT namespace_id FROM khora_chunks")).scalars().all()
            for ns in namespace_ids:
                # Each per-ns UPDATE auto-commits inside the autocommit block.
                bind.execute(_BACKFILL_SQL, {"ns": ns})
        finally:
            if trigger_present:
                op.execute(f"ALTER TABLE khora_chunks ENABLE TRIGGER {_TSV_TRIGGER}")


def upgrade() -> None:
    if not _is_postgres():
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("khora_chunks"):
        # ``khora_chunks`` is created at runtime by
        # ``PgVectorTemporalStore.connect()``; on a fresh deploy the table
        # doesn't exist when migrations run and the runtime populates the
        # denormalized columns itself. Nothing to backfill.
        return

    start = time.monotonic()
    try:
        # ---- PHASE 1: per-namespace batched backfill (autocommit) ----
        _backfill()
    except OperationalError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.bind(
            migration_id=revision,
            duration_ms=duration_ms,
            lock_timeout_tripped=_is_lock_timeout(exc),
        ).error("khora.migration.applied")
        raise

    # ---- PHASE 2: the five filter indexes, CONCURRENTLY, after the
    #      backfill so the btrees are populated. Each runs as its own
    #      implicit transaction within the single autocommit block. ----
    with op.get_context().autocommit_block():
        for stmt in _INDEX_SQL:
            op.execute(stmt)

    # ---- PHASE 3: VACUUM (ANALYZE) last — reclaims the backfill dead
    #      tuples, tidies the just-built btrees, refreshes planner stats. ----
    with op.get_context().autocommit_block():
        op.execute("VACUUM (ANALYZE) khora_chunks")

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
    # Drop the five filter indexes only. The backfilled data is harmless and
    # is removed when 041's downgrade drops the columns themselves; nulling it
    # back out here would be pointless work and would itself fire the TSV
    # trigger / create more bloat. The natural rollback path is
    # ``downgrade -> 044`` (drop indexes) then ``-> 041`` (drop columns).
    with op.get_context().autocommit_block():
        for stmt in _DROP_INDEX_SQL:
            op.execute(stmt)
