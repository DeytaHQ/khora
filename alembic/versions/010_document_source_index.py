"""Add partial composite index on documents(namespace_id, source).

Revision ID: 010_document_source_index
Revises: 009_temporal_search_indexes
Create Date: 2026-03-03

Supports source-based document update detection: when a document is
re-ingested with the same source but different content, the old data is
cleaned up and replaced. The partial index (WHERE source != '') avoids
indexing documents without a source, which cannot be auto-updated.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "010_document_source_index"
down_revision: str | Sequence[str] | None = "009_temporal_search_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_documents_namespace_source",
        "documents",
        ["namespace_id", "source"],
        postgresql_where="source != ''",
    )


def downgrade() -> None:
    op.drop_index("ix_documents_namespace_source", table_name="documents")
