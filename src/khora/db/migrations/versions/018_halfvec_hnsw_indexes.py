"""Add halfvec HNSW expression indexes for float16 embeddings.

Revision ID: 018_halfvec_hnsw_indexes
Revises: 017_temporal_coalesce_index
Create Date: 2026-03-28

DYT-1346: Create halfvec HNSW expression indexes that cast embedding columns
to halfvec(1536), using halfvec_cosine_ops.  Float16 precision yields ~50%
smaller index size with minimal recall loss.  Requires pgvector >= 0.7.0.

Uses CREATE INDEX CONCURRENTLY (cannot run inside a transaction), so each
index operation uses an autocommit block.  Invalid indexes left behind by
interrupted builds are detected via pg_index.indisvalid, dropped, and
recreated.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

revision: str = "018_halfvec_hnsw_indexes"
down_revision: str | Sequence[str] | None = "017_temporal_coalesce_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Default embedding dimension. Deployments using a different
# embedding_dimension config value will need a custom migration
# to create indexes matching their dimension.
_EMBEDDING_DIMENSION = 1536

# Index definitions
_INDEXES = [
    {
        "name": "ix_chunks_embedding_halfvec_hnsw",
        "table": "chunks",
    },
    {
        "name": "ix_entities_embedding_halfvec_hnsw",
        "table": "entities",
    },
]


def _drop_invalid_index(index_name: str) -> None:
    """Drop an index if it exists and is marked invalid (indisvalid = false).

    An invalid index is left behind when CREATE INDEX CONCURRENTLY is
    interrupted.  We must remove it before re-creating, because
    IF NOT EXISTS will skip creation even for invalid indexes.
    """
    conn = op.get_bind()
    is_invalid = conn.execute(
        text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM pg_class c"
            "  JOIN pg_index i ON i.indexrelid = c.oid"
            "  WHERE c.relname = :name AND NOT i.indisvalid"
            ")"
        ),
        {"name": index_name},
    ).scalar()
    if is_invalid:
        with op.get_context().autocommit_block():
            op.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}"))


def upgrade() -> None:
    # Postgres-only: halfvec/HNSW are pgvector features. On SQLite, LanceDB
    # owns embedding storage and indexing.
    if op.get_bind().dialect.name != "postgresql":
        return
    for idx in _INDEXES:
        name = idx["name"]
        table = idx["table"]

        # If a previous interrupted build left an invalid index, remove it
        # so that IF NOT EXISTS doesn't skip the create.  The read query
        # in _drop_invalid_index runs inside the migration transaction
        # (read-only, safe); only the DROP uses autocommit_block because
        # DROP INDEX CONCURRENTLY cannot run inside a transaction.
        _drop_invalid_index(name)

        with op.get_context().autocommit_block():
            op.execute(
                text(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} "
                    f"ON {table} USING hnsw ((embedding::halfvec({_EMBEDDING_DIMENSION})) halfvec_cosine_ops) "
                    f"WITH (m = 24, ef_construction = 128)"
                )
            )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for idx in _INDEXES:
        name = idx["name"]
        with op.get_context().autocommit_block():
            op.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {name}"))
