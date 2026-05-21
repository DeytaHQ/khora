"""Add expression index for temporal COALESCE queries.

Revision ID: 017_temporal_coalesce_index
Revises: 016_widen_extraction_config_hash
Create Date: 2026-03-23

Temporal queries ("what happened in last 7 days?") take 40+ seconds
because search_similar and search_fulltext filter on
COALESCE(source_timestamp, created_at) but no index covers that expression.
PostgreSQL cannot use the existing (namespace_id, created_at) or
(namespace_id, source_timestamp) B-tree indexes for COALESCE, so every
temporal query does a sequential scan.

This adds a composite expression index that matches the exact COALESCE
used in the WHERE clauses, enabling index range scans.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "017_temporal_coalesce_index"
down_revision: str = "016_widen_extraction_config_hash"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute(
                text(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_ns_temporal "
                    "ON chunks (namespace_id, (COALESCE(source_timestamp, created_at)))"
                )
            )
    else:
        # SQLite supports expression indexes but not CONCURRENTLY.
        op.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_ns_temporal "
                "ON chunks (namespace_id, COALESCE(source_timestamp, created_at))"
            )
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_ns_temporal"))
    else:
        op.execute(text("DROP INDEX IF EXISTS ix_chunks_ns_temporal"))
