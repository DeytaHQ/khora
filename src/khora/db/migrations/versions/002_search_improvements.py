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


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


# FTS5 trigger pattern mirrors src/khora/storage/backends/sqlite.py — keeps
# chunks_fts in sync with chunks via AFTER INSERT / DELETE / UPDATE triggers.
_SQLITE_FTS_SETUP = [
    # Virtual table with content linkage so rowid maps to chunks.rowid
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
    "content, content='chunks', content_rowid='rowid', tokenize='porter')",
    # Triggers keep FTS index in sync
    "CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN "
    "INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content); "
    "END",
    "CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN "
    "INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.rowid, old.content); "
    "END",
    "CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN "
    "INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.rowid, old.content); "
    "INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content); "
    "END",
]

_SQLITE_FTS_TEARDOWN = [
    "DROP TRIGGER IF EXISTS chunks_au",
    "DROP TRIGGER IF EXISTS chunks_ad",
    "DROP TRIGGER IF EXISTS chunks_ai",
    "DROP TABLE IF EXISTS chunks_fts",
]


def upgrade() -> None:
    if not _is_postgres():
        # SQLite: substitute HNSW vector indexes (LanceDB owns vectors) and
        # tsvector/GIN full-text search with FTS5 virtual table + triggers.
        for stmt in _SQLITE_FTS_SETUP:
            op.execute(stmt)
        return

    # --- Drop old IVFFlat indexes ---
    op.drop_index("ix_chunks_embedding", table_name="chunks", if_exists=True)
    op.drop_index("ix_entities_embedding", table_name="entities", if_exists=True)

    # --- Create HNSW indexes ---
    op.execute("""
        CREATE INDEX ix_chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """)
    op.execute("""
        CREATE INDEX ix_entities_embedding_hnsw
        ON entities USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """)

    # --- Add tsvector column for full-text search ---
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """)

    # --- GIN index on tsvector column ---
    op.execute("""
        CREATE INDEX ix_chunks_content_tsv
        ON chunks USING gin (content_tsv)
        """)


def downgrade() -> None:
    if not _is_postgres():
        for stmt in _SQLITE_FTS_TEARDOWN:
            op.execute(stmt)
        return

    # --- Drop GIN index and tsvector column ---
    op.drop_index("ix_chunks_content_tsv", table_name="chunks", if_exists=True)
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")

    # --- Drop HNSW indexes ---
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks", if_exists=True)
    op.drop_index("ix_entities_embedding_hnsw", table_name="entities", if_exists=True)

    # --- Recreate IVFFlat indexes ---
    op.execute("""
        CREATE INDEX ix_chunks_embedding
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """)
    op.execute("""
        CREATE INDEX ix_entities_embedding
        ON entities USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """)
