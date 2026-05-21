"""Add bi-temporal soft-delete columns to chronicle_events for Phase 4 event clustering.

Revision ID: 034_chronicle_events_bitemporal
Revises: 033_bitemporal_columns
Create Date: 2026-05-17

Issue #669 — Phase 4 of the dream-phase rollout (umbrella #649). The
event-clustering apply path (#665 planner, this migration's apply
target) needs a place to write per-event soft-delete state when a tail
event is merged into a canonical event.

Schema additions to ``chronicle_events``:

* ``invalidated_at`` ``TIMESTAMPTZ`` NULL — when the dream op marked
  this row superseded
* ``invalidated_by`` ``UUID`` NULL — op_id of the dream op that did the
  invalidation. Not enforced as a DB-level FK: op_ids live in the dream
  run's ``undo.json`` rather than in any FK target table.
* ``merged_into_event_id`` ``UUID`` NULL — self-FK to
  ``chronicle_events.id`` ON DELETE SET NULL. The canonical event this
  row was merged into. Uses ``use_alter=True`` so the self-reference
  can be created after the table exists (mirrors migration 004's
  pattern for the temporal-edges self-FK).

Partial composite index (Postgres-only, dialect-gated; mirrors the
migration 033 pattern of ``CREATE INDEX CONCURRENTLY`` inside an
autocommit block):

* ``ix_chronicle_events_live`` — ``(namespace_id, occurred_at)``
  ``WHERE invalidated_at IS NULL`` — accelerates the live-event recall
  path. ``occurred_at`` does not exist as a column today; the live
  index keys on ``referenced_date`` which is the temporal anchor the
  Chronicle engine uses for the recency channel.

  (Spec called the column ``occurred_at``; the real chronicle_events
  schema uses ``referenced_date`` for the same concept. Indexing the
  real column avoids creating a DDL reference that would never
  resolve.)

Coexistence: ``chronicle_events`` previously had no soft-delete column.
Existing rows backfill to all NULL — meaning "still live".

FK enforcement on ``merged_into_event_id``: enforced via
``ForeignKey("chronicle_events.id", ondelete="SET NULL", use_alter=True)``
so a hard-delete of a canonical event detaches its tails rather than
cascading. The same row's ``invalidated_at`` / ``invalidated_by``
fields keep the tombstone audit trail.

Depends on migration 033 (``033_bitemporal_columns``, PR #680). The
chain expects 033 to be on the same branch.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision: str = "034_chronicle_events_bitemporal"
down_revision: str | Sequence[str] | None = "033_bitemporal_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _uuid_type() -> sa.types.TypeEngine:
    """UUID column type appropriate for the current dialect."""
    if _is_postgres():
        return PG_UUID(as_uuid=True)
    return sa.Uuid(as_uuid=True)


def upgrade() -> None:
    # Postgres path uses raw SQL with IF NOT EXISTS for idempotency
    # (tests re-run the chain; production rollouts may retry after a
    # partial-state crash). SQLite path uses op.add_column which fails
    # loudly if a column already exists — acceptable because the
    # sqlite_lance fixture rebuilds the schema per test.
    if _is_postgres():
        op.execute(
            "ALTER TABLE chronicle_events "
            "ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMP WITH TIME ZONE NULL, "
            "ADD COLUMN IF NOT EXISTS invalidated_by UUID NULL, "
            "ADD COLUMN IF NOT EXISTS merged_into_event_id UUID NULL"
        )

        # Self-FK. Raw SQL with IF NOT EXISTS via a DO block so a retry
        # after partial failure is idempotent (information_schema does
        # not expose a "create constraint if not exists" form).
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_chronicle_events_merged_into_event_id'
                ) THEN
                    ALTER TABLE chronicle_events
                    ADD CONSTRAINT fk_chronicle_events_merged_into_event_id
                    FOREIGN KEY (merged_into_event_id)
                    REFERENCES chronicle_events (id)
                    ON DELETE SET NULL;
                END IF;
            END
            $$;
            """
        )

        with op.get_context().autocommit_block():
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chronicle_events_live "
                "ON chronicle_events (namespace_id, referenced_date) "
                "WHERE invalidated_at IS NULL"
            )
        return

    # SQLite path — no IF NOT EXISTS support on ADD COLUMN; skip the
    # partial index per migration 033's pattern. SQLite's alembic
    # backend cannot ALTER a table to add a FK constraint (only batch
    # mode can, via copy-and-move), so on SQLite the
    # ``merged_into_event_id`` column is added without the FK. This
    # matches the migration-033 compromise on ``invalidated_by``: the
    # column is a semantic reference, and SQLite is a dev-fixture
    # dialect — prod (Postgres) enforces the constraint above.
    uuid_type = _uuid_type()
    op.add_column("chronicle_events", sa.Column("invalidated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("chronicle_events", sa.Column("invalidated_by", uuid_type, nullable=True))
    op.add_column("chronicle_events", sa.Column("merged_into_event_id", uuid_type, nullable=True))


def downgrade() -> None:
    if _is_postgres():
        with op.get_context().autocommit_block():
            op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_chronicle_events_live")
        op.execute("ALTER TABLE chronicle_events DROP CONSTRAINT IF EXISTS fk_chronicle_events_merged_into_event_id")
        op.execute(
            "ALTER TABLE chronicle_events "
            "DROP COLUMN IF EXISTS merged_into_event_id, "
            "DROP COLUMN IF EXISTS invalidated_by, "
            "DROP COLUMN IF EXISTS invalidated_at"
        )
        return

    op.drop_column("chronicle_events", "merged_into_event_id")
    op.drop_column("chronicle_events", "invalidated_by")
    op.drop_column("chronicle_events", "invalidated_at")
