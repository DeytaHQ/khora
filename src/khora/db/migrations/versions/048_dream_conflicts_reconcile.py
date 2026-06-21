"""Add reconcile-triage columns to dream_conflicts.

Revision ID: 048_dream_conflicts_reconcile
Revises: 047_dream_runs_graph_mirror_pending
Create Date: 2026-06-21

Issue #1281 - Phase 5 (final) of the dream-on-graph umbrella (#1282). The
contradiction-detection op (#672) was report-only: it INSERTed a per-pair
finding into ``dream_conflicts`` and never touched ``relationships``. #1281
promotes it to an opt-in two-LLM-judged reconcile op that soft-deletes the
losing edge on judge agreement and records the outcome of every other pair
(defer / keep) as a triage row.

This migration extends ``dream_conflicts`` with the reconcile outcome so the
same row carries both the detection finding and its (eventual) resolution:

* ``resolution`` VARCHAR(16) NOT NULL DEFAULT 'detected' - one of
  ``detected`` (report-only, no judge ran), ``invalidated`` (judges agreed,
  loser soft-deleted + mirrored), ``deferred`` (judges disagreed / timed out /
  ungrounded - no mutation), ``kept`` (judges agreed the pair is NOT a
  contradiction).
* ``loser_relationship_id`` UUID NULL - the soft-deleted edge on
  ``resolution='invalidated'``; NULL otherwise.
* ``winner_relationship_id`` UUID NULL - the surviving edge on
  ``resolution='invalidated'``; NULL otherwise.
* ``judge_rationale_hash`` VARCHAR(16) NULL - ``bounded_text_hash`` of the
  joint two-LLM rationale; never raw text.
* ``resolved_by_op_id`` UUID NULL - op_id of the reconcile dream op that set
  the resolution. Not a DB-level FK (same pattern as ``detected_by_op_id`` /
  migration 034's ``invalidated_by``).
* ``resolved_at`` TIMESTAMPTZ NULL - when the resolution was stamped.

The reconcile apply handler issues ``INSERT ... ON CONFLICT
(namespace_id, relationship_a_id, relationship_b_id) DO UPDATE`` so a pair
already detected report-only is upgraded in place to its resolution, and a
replay of the same reconcile op is idempotent.

Postgres-only via the same dialect gate as migrations 029 / 032 / 034 / 036.
The embedded ``sqlite_lance`` stack runs the full Alembic chain against
SQLite, where ``dream_conflicts`` is never created (migration 036 no-ops on
SQLite), so this migration is a clean no-op there too. Apply-mode contradiction
reconciliation is Postgres-only.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "048_dream_conflicts_reconcile"
down_revision: str | Sequence[str] | None = "047_dream_runs_graph_mirror_pending"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_NAME = "dream_conflicts"
_NEW_COLUMNS = (
    "resolution",
    "loser_relationship_id",
    "winner_relationship_id",
    "judge_rationale_hash",
    "resolved_by_op_id",
    "resolved_at",
)


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _existing_columns() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(TABLE_NAME)}


def upgrade() -> None:
    if not _is_postgres():
        # sqlite_lance fixture path - dream_conflicts is never created on
        # SQLite (migration 036 is Postgres-only). Skip silently.
        return

    # Idempotent on the live schema: the integration migration harness shares
    # one PostgreSQL instance across parallel test files (each resets via DROP
    # SCHEMA public CASCADE), so a plain ADD COLUMN can re-run against an
    # already-migrated table. Guard on the actual columns.
    existing = _existing_columns()

    if "resolution" not in existing:
        op.add_column(
            TABLE_NAME,
            sa.Column(
                "resolution",
                sa.String(length=16),
                nullable=False,
                server_default=sa.text("'detected'"),
            ),
        )
    if "loser_relationship_id" not in existing:
        op.add_column(TABLE_NAME, sa.Column("loser_relationship_id", PG_UUID(as_uuid=True), nullable=True))
    if "winner_relationship_id" not in existing:
        op.add_column(TABLE_NAME, sa.Column("winner_relationship_id", PG_UUID(as_uuid=True), nullable=True))
    if "judge_rationale_hash" not in existing:
        op.add_column(TABLE_NAME, sa.Column("judge_rationale_hash", sa.String(length=16), nullable=True))
    if "resolved_by_op_id" not in existing:
        op.add_column(TABLE_NAME, sa.Column("resolved_by_op_id", PG_UUID(as_uuid=True), nullable=True))
    if "resolved_at" not in existing:
        op.add_column(TABLE_NAME, sa.Column("resolved_at", TIMESTAMP(timezone=True), nullable=True))


def downgrade() -> None:
    if not _is_postgres():
        return
    existing = _existing_columns()
    for column in _NEW_COLUMNS:
        if column in existing:
            op.drop_column(TABLE_NAME, column)
