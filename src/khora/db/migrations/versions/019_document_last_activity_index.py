"""Add composite index for document last activity queries.

Revision ID: 019_document_last_activity_index
Revises: 018_halfvec_hnsw_indexes
Create Date: 2026-04-07

Add composite index (namespace_id, created_at) on documents table
to optimize queries for namespace statistics, particularly get_last_activity_at()
which uses MAX(created_at) with namespace_id filtering.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "019_document_last_activity_index"
down_revision: str | Sequence[str] | None = "018_halfvec_hnsw_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add composite index for efficient last activity queries."""
    op.create_index(
        "ix_documents_namespace_created_at",
        "documents",
        ["namespace_id", "created_at"],
    )


def downgrade() -> None:
    """Remove composite index."""
    op.drop_index("ix_documents_namespace_created_at", "documents")
