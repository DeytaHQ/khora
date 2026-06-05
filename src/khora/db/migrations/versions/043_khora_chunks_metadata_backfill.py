"""Backfill khora_chunks.metadata by merging in the parent document's metadata.

Revision ID: 043_khora_chunks_metadata_backfill
Revises: 042_widen_khora_chunks_source_external_id
Create Date: 2026-06-05

Data change only — NO schema change. Every existing ``khora_chunks`` row
gets the parent ``documents.metadata`` JSONB merged into its own
``metadata``, with the chunk's own keys taking precedence on conflict
(``d.metadata || c.metadata`` — the right-hand operand wins in Postgres
JSONB concatenation). New chunks written after this migration already
carry the merged metadata from the ingest path; this migration only
catches chunks that predate that behavior.

Merge predicate / idempotency: the UPDATE is guarded by
``NOT (c.metadata @> d.metadata)`` so a chunk whose metadata already
contains every parent key/value is skipped. Re-running the migration (or
re-running within the same transaction) therefore updates zero rows once
the merge has landed — the operation converges. The ``@>`` containment
check is conservative: a chunk that overrides a parent key with a
different value does NOT contain the parent pair, so it is re-visited, but
the merge is stable (chunk key wins again, producing the identical row),
so this is correct, just a touch less selective.

Parent filter: only documents with a non-empty ``metadata`` participate
(``d.metadata IS NOT NULL AND d.metadata <> '{}'::jsonb``). A parent with
no metadata contributes nothing, so those joins are pruned up front.

Scale (~1M chunk rows): the backfill runs namespace-batched — one UPDATE
per distinct ``khora_chunks.namespace_id`` — rather than as a single
table-wide statement. Per-namespace statements take an index scan on the
``namespace_id`` index instead of a full-table hash join against
``documents``, keep each statement's working set bounded, and let the
structured log report progress per namespace. Note the whole Alembic
chain runs inside ONE transaction (``env.py:do_run_migrations`` wraps
``context.run_migrations()`` in a single ``context.begin_transaction()``),
so the row locks taken by each batch are held until the chain commits —
batching bounds per-statement cost and planning, not lock-hold duration.

Trigger suppression: ``khora_chunks`` carries a ``BEFORE INSERT OR UPDATE``
trigger (``khora_chunks_content_tsv_update``) that recomputes
``to_tsvector('english', content)`` on every updated row. This backfill
touches only ``metadata`` and never ``content``, so re-deriving the
tsvector for every row would be wasted CPU on a ~1M-row table. The trigger
is disabled for the duration of the backfill and re-enabled in a
``finally`` so it is restored on every exit path (success, lock-timeout,
or any error). ``DISABLE TRIGGER`` / ``ENABLE TRIGGER`` is itself
transactional, so a rolled-back migration leaves the trigger enabled.

Lock-timeout safety: a Postgres-only ``SET lock_timeout = '5s'`` is issued
before the trigger toggle so the brief ``ACCESS EXCLUSIVE`` lock that
``ALTER TABLE ... DISABLE TRIGGER`` acquires, and every row lock the
UPDATEs take, are bounded. The setting is scoped to the migration's
enclosing transaction (env.py wraps the whole upgrade in a single
transaction; the SET expires at COMMIT). On lock-timeout the upgrade logs
``khora.migration.applied`` at ERROR with ``lock_timeout_tripped=True``
(detected by the Postgres SQLSTATE ``55P03`` / ``lock_not_available`` on
``OperationalError.orig.pgcode`` — any other ``OperationalError`` class
logs ``lock_timeout_tripped=False`` so dashboards don't conflate deadlocks
/ connection drops with the lock-timeout signal). Either path re-raises so
Alembic rolls back the transaction.

Runtime-def convergence: ``khora_chunks`` is created at runtime by
``PgVectorTemporalStore.connect()`` (via ``metadata.create_all``), not by
the Alembic-managed schema. On fresh deploys and on the sqlite_lance test
fixture the table doesn't exist when the chain runs; the backfill then has
nothing to do and the ``has_table`` guard early-returns.

Cross-dialect: this migration is Postgres-only and early-returns on other
dialects (SQLite). The ``||`` JSONB merge, ``@>`` containment operator,
and ``DISABLE TRIGGER`` are Postgres-specific; the sqlite_lance fixture
stack runs the full Alembic chain, so the early-return keeps it green.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.exc import OperationalError

revision: str = "043_khora_chunks_metadata_backfill"
down_revision: str | Sequence[str] | None = "042_widen_khora_chunks_source_external_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"

# Name of the BEFORE INSERT OR UPDATE trigger on khora_chunks that recomputes
# content_tsv. Defined at runtime in
# ``src/khora/engines/skeleton/backends/pgvector.py``.
_CONTENT_TSV_TRIGGER = "khora_chunks_content_tsv_update"

# Namespace-batched merge: chunk keys win on conflict
# (``d.metadata || c.metadata`` — right operand wins in JSONB concat). The
# ``@>`` guard makes the statement a no-op for already-merged rows.
_MERGE_SQL = sa.text(
    "UPDATE khora_chunks c "
    "SET metadata = d.metadata || c.metadata "
    "FROM documents d "
    "WHERE c.document_id = d.id "
    "AND c.namespace_id = :namespace_id "
    "AND d.metadata IS NOT NULL "
    "AND d.metadata <> '{}'::jsonb "
    "AND NOT (c.metadata @> d.metadata)"
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


def _upgrade_impl() -> tuple[int, int]:
    """Run the namespace-batched backfill. Returns ``(namespaces, rows)``."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("khora_chunks") or not inspector.has_table("documents"):
        # ``khora_chunks`` is created at runtime by the vectorcypher
        # ``PgVectorTemporalStore.connect()`` (via ``metadata.create_all``),
        # not by the Alembic-managed schema. On fresh deploys and on the
        # sqlite_lance test fixture one or both tables may not exist when the
        # chain runs; new chunks already carry merged metadata from the ingest
        # path, so there is nothing to backfill here.
        return 0, 0

    # Bound every lock acquisition — the brief ACCESS EXCLUSIVE lock that
    # ``DISABLE TRIGGER`` takes and the row locks the UPDATEs take — so a stuck
    # pg_stat_activity entry on khora_chunks cannot stall the deploy past 5s.
    # Issued before the trigger toggle.
    op.execute("SET lock_timeout = '5s'")

    # The content_tsv trigger only matters when ``content`` changes; this
    # backfill touches only ``metadata``, so re-deriving the tsvector for every
    # updated row is wasted CPU on a ~1M-row table. Disable it for the backfill
    # and restore it in the ``finally`` on every exit path.
    op.execute(f"ALTER TABLE khora_chunks DISABLE TRIGGER {_CONTENT_TSV_TRIGGER}")
    try:
        namespace_ids = [row[0] for row in bind.execute(sa.text("SELECT DISTINCT namespace_id FROM khora_chunks"))]
        total_rows = 0
        for namespace_id in namespace_ids:
            result = bind.execute(_MERGE_SQL, {"namespace_id": namespace_id})
            total_rows += int(result.rowcount or 0)
        return len(namespace_ids), total_rows
    finally:
        op.execute(f"ALTER TABLE khora_chunks ENABLE TRIGGER {_CONTENT_TSV_TRIGGER}")


def upgrade() -> None:
    if not _is_postgres():
        return

    start = time.monotonic()
    # Initialize log fields up-front so the error path always emits a uniform
    # event for dashboards / alerts.
    namespaces_backfilled = 0
    rows_backfilled = 0
    try:
        namespaces_backfilled, rows_backfilled = _upgrade_impl()
    except OperationalError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.bind(
            migration_id=revision,
            duration_ms=duration_ms,
            lock_timeout_tripped=_is_lock_timeout(exc),
            namespaces_backfilled=namespaces_backfilled,
            rows_backfilled=rows_backfilled,
        ).error("khora.migration.applied")
        # Bare ``raise`` re-raises the active exception with the original
        # traceback preserved. env.py's wrapping context.begin_transaction()
        # rolls back the UPDATEs from this revision (and the trigger toggle).
        raise

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.bind(
        migration_id=revision,
        duration_ms=duration_ms,
        lock_timeout_tripped=False,
        namespaces_backfilled=namespaces_backfilled,
        rows_backfilled=rows_backfilled,
    ).info("khora.migration.applied")


def downgrade() -> None:
    """Irreversible data merge — no-op.

    The upgrade merges parent ``documents.metadata`` into each chunk's
    ``metadata`` (chunk keys winning). Once merged, the migration cannot
    distinguish keys that originated on the chunk from keys that were copied
    down from the parent, so the merge cannot be cleanly un-done. The
    downgrade is therefore a dialect-gated no-op.
    """
    if not _is_postgres():
        return
