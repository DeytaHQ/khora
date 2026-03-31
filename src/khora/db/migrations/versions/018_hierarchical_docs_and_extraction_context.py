"""Add hierarchical document support and extraction context.

Revision ID: 018_hierarchy_ext_ctx
Revises: 017_temporal_coalesce_index
Create Date: 2026-03-26

Adds:
- parent_document_id: self-referential FK for document hierarchy
- hierarchy_depth: integer depth in document tree (0 = root)
- extraction_context: text prepended to LLM extraction prompt
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "018_hierarchy_ext_ctx"
down_revision: str = "017_temporal_coalesce_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("parent_document_id", sa.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("hierarchy_depth", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "documents",
        sa.Column("extraction_context", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_documents_parent_document_id",
        "documents",
        ["parent_document_id"],
    )
    op.create_foreign_key(
        "fk_documents_parent",
        "documents",
        "documents",
        ["parent_document_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_documents_parent", "documents", type_="foreignkey")
    op.drop_index("ix_documents_parent_document_id", table_name="documents")
    op.drop_column("documents", "extraction_context")
    op.drop_column("documents", "hierarchy_depth")
    op.drop_column("documents", "parent_document_id")
