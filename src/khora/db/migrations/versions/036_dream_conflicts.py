"""Add dream_conflicts table for the contradiction-detection op.

Revision ID: 036_dream_conflicts
Revises: 035_dream_communities
Create Date: 2026-05-18

Issue #672 — Phase 5.3 of the dream-phase rollout (umbrella #649). The
vectorcypher contradiction-detection op persists per-pair findings to
this table so a human triage queue (Phase 5.4, #673) can review them.
The op is report-only — it never mutates ``relationships``.

Schema (15 columns):

* ``id`` UUID PK
* ``namespace_id`` UUID NOT NULL — stable namespace id
* ``relationship_a_id`` UUID NOT NULL — first relationship (canonical
  order: ``str(rel_a) < str(rel_b)``)
* ``relationship_b_id`` UUID NOT NULL — second relationship
* ``source_entity_id`` UUID NOT NULL — denormalized for triage queries
* ``target_entity_id`` UUID NOT NULL
* ``relationship_type`` VARCHAR(64) NOT NULL
* ``similarity`` DOUBLE PRECISION NOT NULL — textual similarity score
  for the pair (``[0.0, 1.0]``)
* ``contradicting_keys`` TEXT[] NOT NULL DEFAULT '{}' — shared property
  keys whose stringified values disagreed
* ``reason`` VARCHAR(32) NOT NULL — one of ``low_similarity`` /
  ``property_contradiction`` / ``both``
* ``description_a_hash`` VARCHAR(16) NOT NULL — bounded_text_hash of
  the description, never raw text
* ``description_b_hash`` VARCHAR(16) NOT NULL
* ``detected_by_op_id`` UUID NOT NULL — op_id of the dream op that
  flagged the pair. Not enforced as a DB-level FK: op_ids live in the
  dream run's ``undo.json`` rather than in any FK target table — same
  pattern as migration 034's ``invalidated_by``.
* ``valid_from`` TIMESTAMPTZ NOT NULL — bi-temporal validity start;
  matches the migration-033 / 034 pattern.
* ``created_at`` TIMESTAMPTZ NOT NULL DEFAULT NOW()

UNIQUE ``(namespace_id, relationship_a_id, relationship_b_id)`` —
enforces idempotency on replay; the apply handler issues
``INSERT ... ON CONFLICT DO NOTHING`` against this constraint so the
same dream op never duplicates findings on a re-run.

Postgres-only via the same dialect gate as migration 029 / 032 / 034.
The embedded ``sqlite_lance`` stack runs the full Alembic chain against
SQLite; the migration is a clean no-op there. Apply-mode contradiction
detection is Postgres-only in v0.16.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "036_dream_conflicts"
down_revision: str | Sequence[str] | None = "035_dream_communities"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_NAME = "dream_conflicts"
UNIQUE_NAME = "uq_dream_conflicts_pair"
INDEX_NAME = "ix_dream_conflicts_namespace_detected"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        # sqlite_lance fixture path — apply-mode contradiction detection
        # is Postgres-only in v0.16. Skip silently per migration 032's
        # pattern.
        return

    op.create_table(
        TABLE_NAME,
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column("namespace_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("relationship_a_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("relationship_b_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("source_entity_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("target_entity_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("relationship_type", sa.String(length=64), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column(
            "contradicting_keys",
            ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("ARRAY[]::TEXT[]"),
        ),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("description_a_hash", sa.String(length=16), nullable=False),
        sa.Column("description_b_hash", sa.String(length=16), nullable=False),
        sa.Column("detected_by_op_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("valid_from", TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint(
            "namespace_id",
            "relationship_a_id",
            "relationship_b_id",
            name=UNIQUE_NAME,
        ),
    )

    op.create_index(
        INDEX_NAME,
        TABLE_NAME,
        [sa.text("namespace_id"), sa.text("detected_by_op_id")],
    )


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
    op.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
