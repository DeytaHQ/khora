"""Add partial index for dedup queries excluding failed documents.

Revision ID: 020_partial_index_dedup_active
Revises: 019_document_last_activity_index
Create Date: 2026-04-14

DYT-2381: Add partial index (namespace_id, checksum) WHERE status != 'failed'
on documents table. This allows dedup lookups to skip failed documents without
scanning rows that will never match.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "020_partial_index_dedup_active"
down_revision: str | Sequence[str] | None = "019_document_last_activity_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add partial index for dedup excluding failed docs."""
    op.create_index(
        "ix_documents_namespace_checksum_active",
        "documents",
        ["namespace_id", "checksum"],
        postgresql_where=text("status != 'failed'"),
    )


def downgrade() -> None:
    """Remove partial index."""
    op.drop_index("ix_documents_namespace_checksum_active", table_name="documents")
