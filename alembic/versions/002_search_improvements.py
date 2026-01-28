"""HNSW indexes and tsvector full-text search.

Revision ID: 002_search_improvements
Revises: 001_namespace_versioning
Create Date: 2026-01-28

Replaces IVFFlat indexes with HNSW for better recall at scale,
and adds a generated tsvector column on chunks for PostgreSQL full-text search.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_search_improvements"
down_revision: str | Sequence[str] | None = "001_namespace_versioning"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- Drop old IVFFlat indexes ---
    op.drop_index("ix_chunks_embedding", table_name="chunks", if_exists=True)
    op.drop_index("ix_entities_embedding", table_name="entities", if_exists=True)

    # --- Create HNSW indexes ---
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        """
        CREATE INDEX ix_entities_embedding_hnsw
        ON entities USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    # --- Add tsvector column for full-text search ---
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )

    # --- GIN index on tsvector column ---
    op.execute(
        """
        CREATE INDEX ix_chunks_content_tsv
        ON chunks USING gin (content_tsv)
        """
    )


def downgrade() -> None:
    # --- Drop GIN index and tsvector column ---
    op.drop_index("ix_chunks_content_tsv", table_name="chunks", if_exists=True)
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")

    # --- Drop HNSW indexes ---
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks", if_exists=True)
    op.drop_index("ix_entities_embedding_hnsw", table_name="entities", if_exists=True)

    # --- Recreate IVFFlat indexes ---
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )
    op.execute(
        """
        CREATE INDEX ix_entities_embedding
        ON entities USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )
