"""Migrate legacy uppercase document_status values to lowercase.

Revision ID: 027_migrate_uppercase_document_status
Revises: 026_widen_alembic_version_column
Create Date: 2026-05-05

DYT-3736: Staging contains ~19 000 rows with uppercase status values
(PENDING, PROCESSING, COMPLETED, FAILED, ARCHIVED) written by an early
Khora version that used enum .name instead of .value. The canonical
DocumentStatus enum defines lowercase values only. This migration
normalises all uppercase rows to their lowercase equivalents.

The lowercase enum values already exist (migration 014 added 'pending'
and 'archived'; 000 created 'processing', 'completed', 'failed'), so
this is a pure data update — no ALTER TYPE is needed.

PostgreSQL does not support DROP VALUE from an enum, so the uppercase
variants remain defined in the type but are now unused.

Idempotent: re-running is safe (the WHERE clause matches zero rows when
all statuses are already lowercase).
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "027_migrate_uppercase_document_status"
down_revision: str | Sequence[str] | None = "026_widen_alembic_version_column"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Normalise uppercase document_status rows to lowercase. Postgres-only."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        text(
            "UPDATE documents "
            "SET status = LOWER(status::text)::document_status "
            "WHERE status::text ~ '[A-Z]'"
        )
    )


def downgrade() -> None:
    """No-op — we cannot recover which rows were originally uppercase."""
