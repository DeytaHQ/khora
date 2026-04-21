"""Promote external_id partial index to UNIQUE.

Revision ID: 022_promote_external_id_index_unique
Revises: 021_add_document_external_id
Create Date: 2026-04-21

DYT-2672: Promote the partial composite index (namespace_id, external_id)
WHERE external_id IS NOT NULL to a UNIQUE constraint, enabling idempotent
upsert-by-external_id.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "022_promote_external_id_index_unique"
down_revision: str | Sequence[str] | None = "021_add_document_external_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Replace non-unique partial index with a UNIQUE partial index."""
    op.drop_index("ix_documents_namespace_external_id", table_name="documents")
    if op.get_bind().dialect.name == "postgresql":
        op.create_index(
            "ix_documents_namespace_external_id_unique",
            "documents",
            ["namespace_id", "external_id"],
            unique=True,
            postgresql_where=text("external_id IS NOT NULL"),
        )
    else:
        op.create_index(
            "ix_documents_namespace_external_id_unique",
            "documents",
            ["namespace_id", "external_id"],
            unique=True,
            sqlite_where=text("external_id IS NOT NULL"),
        )


def downgrade() -> None:
    """Restore original non-unique partial index."""
    op.drop_index("ix_documents_namespace_external_id_unique", table_name="documents")
    if op.get_bind().dialect.name == "postgresql":
        op.create_index(
            "ix_documents_namespace_external_id",
            "documents",
            ["namespace_id", "external_id"],
            postgresql_where=text("external_id IS NOT NULL"),
        )
    else:
        op.create_index(
            "ix_documents_namespace_external_id",
            "documents",
            ["namespace_id", "external_id"],
            sqlite_where=text("external_id IS NOT NULL"),
        )
