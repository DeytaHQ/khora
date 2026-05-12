"""Add extraction_config_hash column to documents table.

Revision ID: 015_add_extraction_config_hash
Revises: 014_sync_document_status_enum
Create Date: 2026-03-22

Supports ontology-aware re-extraction. The column stores a hash of the
extraction configuration used when a document was last processed, enabling
selective re-extraction when the ontology or extraction config changes.

Nullable so existing documents are unaffected — they will be back-filled on
their next extraction cycle.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "015_add_extraction_config_hash"
down_revision: str = "014_sync_document_status_enum"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("extraction_config_hash", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_documents_extraction_config_hash",
        "documents",
        ["extraction_config_hash"],
    )


def downgrade() -> None:
    op.drop_index("ix_documents_extraction_config_hash", table_name="documents")
    op.drop_column("documents", "extraction_config_hash")
