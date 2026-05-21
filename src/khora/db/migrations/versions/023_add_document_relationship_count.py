"""Add relationship_count column to documents table.

Revision ID: 023_add_document_relationship_count
Revises: 022_promote_external_id_index_unique
Create Date: 2026-04-25

Track persisted relationship count on Document so that
skipped DocumentResult entries can report the correct
relationships_created value instead of always returning 0.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "023_add_document_relationship_count"
down_revision: str | Sequence[str] | None = "022_promote_external_id_index_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add relationship_count column with default 0."""
    op.add_column("documents", sa.Column("relationship_count", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    """Remove relationship_count column."""
    op.drop_column("documents", "relationship_count")
