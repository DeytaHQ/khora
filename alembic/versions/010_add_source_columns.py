"""Add source citation columns to chunks and documents.

Revision ID: 010_add_source_columns
Revises: 009_temporal_search_indexes
Create Date: 2026-03-03

Adds denormalized source columns to chunks for citation resolution,
and source_tool column to documents for canonical SaaS tool tracking.
Backfills chunk sources from parent documents.
"""

from collections.abc import Sequence

from sqlalchemy import Column, String, text

from alembic import op

revision: str = "010_add_source_columns"
down_revision: str | Sequence[str] | None = "009_temporal_search_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add source_tool column to documents first
    op.add_column("documents", Column("source_tool", String(64), nullable=False, server_default=""))

    # 2. Backfill documents source_tool from JSONB metadata
    op.execute(
        text(
            "UPDATE documents SET source_tool = metadata->>'source_tool' "
            "WHERE metadata->>'source_tool' IS NOT NULL "
            "AND metadata->>'source_tool' != ''"
        )
    )

    # 3. Add source columns to chunks (denormalized from parent document)
    op.add_column("chunks", Column("source_title", String(512), nullable=False, server_default=""))
    op.add_column("chunks", Column("source_url", String(1024), nullable=False, server_default=""))
    op.add_column("chunks", Column("source_type", String(64), nullable=False, server_default=""))
    op.add_column("chunks", Column("source_tool", String(64), nullable=False, server_default=""))

    # 4. Backfill chunk source columns from parent documents (including source_tool)
    op.execute(
        text(
            "UPDATE chunks SET "
            "source_title = d.title, "
            "source_url = d.source, "
            "source_type = d.source_type, "
            "source_tool = d.source_tool "
            "FROM documents d WHERE chunks.document_id = d.id"
        )
    )


def downgrade() -> None:
    op.drop_column("documents", "source_tool")
    op.drop_column("chunks", "source_tool")
    op.drop_column("chunks", "source_type")
    op.drop_column("chunks", "source_url")
    op.drop_column("chunks", "source_title")
