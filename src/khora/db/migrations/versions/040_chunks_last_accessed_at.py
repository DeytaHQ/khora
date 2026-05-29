"""Add nullable last_accessed_at column to chunks for recall reinforcement.

Revision ID: 040_chunks_last_accessed_at
Revises: 039_khora_chunks_content_tsv_gin
Create Date: 2026-05-28

Issue #855 - Chronicle reinforcement-on-recall. Each chunk gains a
``last_accessed_at`` TIMESTAMPTZ column that the engine updates after
returning the chunk in a recall result. When the feature flag
``KHORA_QUERY_CHRONICLE_ENABLE_RECALL_REINFORCEMENT=true`` is set, the
temporal-decay path reads ``max(source_timestamp, last_accessed_at)``
as the effective event time so frequently-recalled chunks stay fresh
even as their source_timestamp ages (Stanford generative-agents pattern).

The column is nullable - existing chunks land at NULL until first
recall and then naturally adopt the new behavior. Works on both
PostgreSQL and SQLite via Alembic's dialect-portable ``op.add_column``.
The partial index uses ``sqlite_where`` / ``postgresql_where`` so both
dialects skip NULL rows.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import Column, DateTime, text

revision: str = "040_chunks_last_accessed_at"
down_revision: str | Sequence[str] | None = "039_khora_chunks_content_tsv_gin"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", Column("last_accessed_at", DateTime(timezone=True), nullable=True))

    if op.get_bind().dialect.name == "postgresql":
        op.create_index(
            "ix_chunks_last_accessed_at",
            "chunks",
            ["namespace_id", "last_accessed_at"],
            postgresql_where=text("last_accessed_at IS NOT NULL"),
        )
    else:
        op.create_index(
            "ix_chunks_last_accessed_at",
            "chunks",
            ["namespace_id", "last_accessed_at"],
            sqlite_where=text("last_accessed_at IS NOT NULL"),
        )


def downgrade() -> None:
    op.drop_index("ix_chunks_last_accessed_at", "chunks")
    op.drop_column("chunks", "last_accessed_at")
