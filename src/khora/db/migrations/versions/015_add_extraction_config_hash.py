"""Add extraction_config_hash column to documents table.

Revision ID: 015_add_extraction_config_hash
Revises: 014_sync_document_status_enum
Create Date: 2026-03-21

Adds a nullable VARCHAR(255) column for tracking which extraction configuration
was used to process each document. NULL for legacy documents that pre-date
expertise-based extraction (ADR-022).
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
        sa.Column("extraction_config_hash", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "extraction_config_hash")
