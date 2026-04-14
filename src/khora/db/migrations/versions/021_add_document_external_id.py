"""Add external_id column to documents table.

Revision ID: 021_add_document_external_id
Revises: 020_partial_index_dedup_active
Create Date: 2026-04-14

DYT-2427: Add external_id column (nullable) and partial composite index
(namespace_id, external_id) WHERE external_id IS NOT NULL for future
dedup-by-external_id support (Phase 2, ADR-050).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision: str = "021_add_document_external_id"
down_revision: str | Sequence[str] | None = "020_partial_index_dedup_active"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add external_id column and partial composite index."""
    op.add_column("documents", sa.Column("external_id", sa.String(), nullable=True))
    op.create_index(
        "ix_documents_namespace_external_id",
        "documents",
        ["namespace_id", "external_id"],
        postgresql_where=text("external_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove external_id column and index."""
    op.drop_index("ix_documents_namespace_external_id", table_name="documents")
    op.drop_column("documents", "external_id")
