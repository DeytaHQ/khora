"""Temporal search indexes and source_timestamp columns.

Revision ID: 009_temporal_search_indexes
Revises: 008_entity_dedup_and_indexes
Create Date: 2026-03-01

Adds indexes for temporal filtering pushdown and source_timestamp columns
for tracking original content timestamps separate from ingestion time.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import Column, DateTime, text

revision: str = "009_temporal_search_indexes"
down_revision: str | Sequence[str] | None = "008_entity_dedup_and_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Temporal filtering indexes for SQL pushdown
    op.create_index("ix_chunks_created_at", "chunks", ["created_at"])
    op.create_index("ix_chunks_ns_created", "chunks", ["namespace_id", "created_at"])
    op.create_index("ix_documents_created_at", "documents", ["created_at"])

    # Source timestamp columns (nullable — NULL means "use created_at")
    op.add_column("chunks", Column("source_timestamp", DateTime(timezone=True), nullable=True))
    op.add_column("documents", Column("source_timestamp", DateTime(timezone=True), nullable=True))

    # Index for source_timestamp temporal queries (partial index on both dialects)
    if op.get_bind().dialect.name == "postgresql":
        op.create_index(
            "ix_chunks_source_ts",
            "chunks",
            ["namespace_id", "source_timestamp"],
            postgresql_where=text("source_timestamp IS NOT NULL"),
        )
    else:
        op.create_index(
            "ix_chunks_source_ts",
            "chunks",
            ["namespace_id", "source_timestamp"],
            sqlite_where=text("source_timestamp IS NOT NULL"),
        )


def downgrade() -> None:
    op.drop_index("ix_chunks_source_ts", "chunks")
    op.drop_column("documents", "source_timestamp")
    op.drop_column("chunks", "source_timestamp")
    op.drop_index("ix_documents_created_at", "documents")
    op.drop_index("ix_chunks_ns_created", "chunks")
    op.drop_index("ix_chunks_created_at", "chunks")
