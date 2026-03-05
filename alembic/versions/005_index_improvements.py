"""Index improvements for search quality and temporal queries.

Revision ID: 005_index_improvements
Revises: 004_add_temporal_tables
Create Date: 2026-02-10

Adds:
- GIN index on khora_chunks.tags for array containment queries
- Composite index on khora_chunks(namespace_id, occurred_at) for temporal filtering
- Rebuild HNSW on khora_chunks with ef_construction=128 for better recall
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005_index_improvements"
down_revision: str | Sequence[str] | None = "004_add_temporal_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # khora_chunks is created at runtime by the skeleton engine, not by migrations.
    # Only create indexes if the table already exists.
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT EXISTS ("
            "  SELECT FROM information_schema.tables"
            "  WHERE table_name = 'khora_chunks'"
            ")"
        )
    )
    has_khora_chunks = result.scalar()

    if has_khora_chunks:
        # GIN index on tags for array containment queries (@>, &&)
        op.execute("CREATE INDEX IF NOT EXISTS ix_khora_chunks_tags_gin " "ON khora_chunks USING GIN (tags)")

        # Composite index for temporal filtering within namespace
        op.execute("CREATE INDEX IF NOT EXISTS ix_khora_chunks_ns_occurred " "ON khora_chunks (namespace_id, occurred_at)")

        # Rebuild HNSW index with higher ef_construction for better recall.
        # ef_construction=128 (up from default 64) improves recall at build time
        # with negligible query-time cost.
        op.execute("DROP INDEX IF EXISTS ix_khora_chunks_embedding_hnsw")
        op.execute(
            "CREATE INDEX ix_khora_chunks_embedding_hnsw "
            "ON khora_chunks USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 128)"
        )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_khora_chunks_embedding_hnsw")
    op.execute(
        "CREATE INDEX ix_khora_chunks_embedding_hnsw "
        "ON khora_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    op.execute("DROP INDEX IF EXISTS ix_khora_chunks_ns_occurred")
    op.execute("DROP INDEX IF EXISTS ix_khora_chunks_tags_gin")
