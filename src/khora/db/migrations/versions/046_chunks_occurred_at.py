"""Add nullable occurred_at column to chunks.

Revision ID: 046_chunks_occurred_at
Revises: 045_khora_try_timestamptz
Create Date: 2026-06-09

Each chunk gains a nullable ``occurred_at`` TIMESTAMPTZ column recording the
real-world event time the chunk's content refers to, distinct from its
ingestion ``created_at``. The column is nullable - existing chunks land at
NULL until a value is supplied. The plain ``op.add_column`` is
dialect-portable, so it materializes on both PostgreSQL and SQLite (the
sqlite_lance test fixtures run the full Alembic chain against SQLite).
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import Column, DateTime

revision: str = "046_chunks_occurred_at"
down_revision: str | Sequence[str] | None = "045_khora_try_timestamptz"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", Column("occurred_at", DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "occurred_at")
