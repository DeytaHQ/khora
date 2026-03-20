"""Add missing 'pending' and 'archived' values to document_status enum.

Revision ID: 014_sync_document_status_enum
Revises: 013_drop_previous_version_id
Create Date: 2026-03-18

The Python DocumentStatus enum defines PENDING, PROCESSING, COMPLETED, FAILED,
and ARCHIVED, but databases created by older Khora versions (via create_tables()
or earlier migrations) may only have a subset of these values.

ALTER TYPE ... ADD VALUE IF NOT EXISTS is used so this migration is safe to run
on databases that already have the values.

Note: ADD VALUE cannot run inside a transaction, so autocommit mode is required.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers
revision: str = "014_sync_document_status_enum"
down_revision: str = "013_drop_previous_version_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ADD VALUE IF NOT EXISTS is idempotent — safe if values already present.
    # Must run outside a transaction block.
    op.execute("COMMIT")
    op.execute("ALTER TYPE document_status ADD VALUE IF NOT EXISTS 'pending'")
    op.execute("ALTER TYPE document_status ADD VALUE IF NOT EXISTS 'archived'")


def downgrade() -> None:
    # PostgreSQL does not support DROP VALUE from an enum type.
    # Downgrade is a no-op; the extra values are harmless.
    pass
