"""Tune HNSW index parameters for better recall.

Revision ID: 007_hnsw_parameter_tuning
Revises: 006_uuid_as_uuid
Create Date: 2026-02-23

Changes:
- Rebuild chunks HNSW index with m=24, ef_construction=128 (was m=16, ef_construction=64)
- Rebuild entities HNSW index with m=24, ef_construction=128 (was m=16, ef_construction=64)
- Rebuild khora_chunks HNSW index with m=24 (was m=16; ef_construction already 128 from migration 005)

Uses CREATE INDEX CONCURRENTLY + DROP INDEX CONCURRENTLY for zero-downtime
migration. CONCURRENTLY cannot run inside a transaction, so each index
operation uses an autocommit block.

m=24 increases per-node connections for better graph connectivity (~20% larger index).
ef_construction=128 doubles the build-time search width for improved recall.
Together these yield meaningful retrieval quality improvement at moderate build cost.

Note: Three independent HNSW indexes exist across two table sets:
- chunks.embedding (main pgvector backend)
- entities.embedding (main pgvector backend)
- khora_chunks.embedding (skeleton engine backend)
All three are tuned consistently here.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "007_hnsw_parameter_tuning"
down_revision: str | Sequence[str] | None = "006_uuid_as_uuid"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Each index rebuild uses CREATE CONCURRENTLY (new name) then DROP CONCURRENTLY (old name).
    # CONCURRENTLY cannot run inside a transaction, so we use autocommit blocks.

    # --- chunks HNSW index (main pgvector backend) ---
    with op.get_context().autocommit_block():
        op.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_embedding_hnsw_v2 "
                "ON chunks USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 24, ef_construction = 128)"
            )
        )
    with op.get_context().autocommit_block():
        op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_embedding_hnsw"))
    with op.get_context().autocommit_block():
        op.execute(text("ALTER INDEX IF EXISTS ix_chunks_embedding_hnsw_v2 RENAME TO ix_chunks_embedding_hnsw"))

    # --- entities HNSW index (main pgvector backend) ---
    with op.get_context().autocommit_block():
        op.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_entities_embedding_hnsw_v2 "
                "ON entities USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 24, ef_construction = 128)"
            )
        )
    with op.get_context().autocommit_block():
        op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_entities_embedding_hnsw"))
    with op.get_context().autocommit_block():
        op.execute(text("ALTER INDEX IF EXISTS ix_entities_embedding_hnsw_v2 RENAME TO ix_entities_embedding_hnsw"))

    # --- khora_chunks HNSW index (skeleton engine backend) ---
    # khora_chunks is created at runtime by the skeleton engine; skip if absent.
    # ef_construction was already 128 from migration 005; only m changes 16 -> 24.
    conn = op.get_bind()
    has_khora_chunks = conn.execute(
        text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'khora_chunks')")
    ).scalar()
    if has_khora_chunks:
        with op.get_context().autocommit_block():
            op.execute(
                text(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_embedding_hnsw_v2 "
                    "ON khora_chunks USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 24, ef_construction = 128)"
                )
            )
        with op.get_context().autocommit_block():
            op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_khora_chunks_embedding_hnsw"))
        with op.get_context().autocommit_block():
            op.execute(
                text("ALTER INDEX IF EXISTS ix_khora_chunks_embedding_hnsw_v2 RENAME TO ix_khora_chunks_embedding_hnsw")
            )


def downgrade() -> None:
    # --- chunks HNSW index ---
    with op.get_context().autocommit_block():
        op.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_embedding_hnsw_v2 "
                "ON chunks USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
    with op.get_context().autocommit_block():
        op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_embedding_hnsw"))
    with op.get_context().autocommit_block():
        op.execute(text("ALTER INDEX IF EXISTS ix_chunks_embedding_hnsw_v2 RENAME TO ix_chunks_embedding_hnsw"))

    # --- entities HNSW index ---
    with op.get_context().autocommit_block():
        op.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_entities_embedding_hnsw_v2 "
                "ON entities USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
    with op.get_context().autocommit_block():
        op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_entities_embedding_hnsw"))
    with op.get_context().autocommit_block():
        op.execute(text("ALTER INDEX IF EXISTS ix_entities_embedding_hnsw_v2 RENAME TO ix_entities_embedding_hnsw"))

    # --- khora_chunks HNSW index (restore to m=16, ef_construction=128 per migration 005) ---
    with op.get_context().autocommit_block():
        op.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_embedding_hnsw_v2 "
                "ON khora_chunks USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 128)"
            )
        )
    with op.get_context().autocommit_block():
        op.execute(text("DROP INDEX CONCURRENTLY IF EXISTS ix_khora_chunks_embedding_hnsw"))
    with op.get_context().autocommit_block():
        op.execute(
            text("ALTER INDEX IF EXISTS ix_khora_chunks_embedding_hnsw_v2 RENAME TO ix_khora_chunks_embedding_hnsw")
        )
