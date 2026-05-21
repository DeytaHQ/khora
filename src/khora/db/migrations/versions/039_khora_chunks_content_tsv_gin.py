"""Add GIN index on khora_chunks.content_tsv for BM25 full-text search.

Revision ID: 039_khora_chunks_content_tsv_gin
Revises: 038_khora_chunks_chunker_info
Create Date: 2026-05-21

VectorCypher's batch ingest path writes chunks to the ``khora_chunks``
temporal-store table, which holds the populated ``content_tsv`` column
needed for BM25 / ts_rank queries. The runtime
``PgVectorTemporalStore.connect()`` creates the GIN index on first
process-start, but a deployment that runs ``alembic upgrade head`` from
a sidecar / job before any process opens the temporal store would query
``khora_chunks`` with no GIN index — turning every BM25 lookup into a
full sequential ``tsvector`` scan. Adding the index here ensures it
exists by the time migrations finish, regardless of which process
boots first.

``IF NOT EXISTS`` makes the migration converge cleanly with the
runtime ``CREATE INDEX IF NOT EXISTS`` in
``src/khora/engines/skeleton/backends/pgvector.py`` — whichever runs
first wins; the other becomes a no-op.

Concurrent index creation: Postgres ``CREATE INDEX CONCURRENTLY``
cannot run inside a transaction, so we open an autocommit block.

Cross-dialect: GIN indexes and ``CONCURRENTLY`` are Postgres-only;
SQLite uses an FTS5 virtual table instead. Skip silently on non-Postgres
so the sqlite_lance test fixtures pass.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "039_khora_chunks_content_tsv_gin"
down_revision: str | Sequence[str] | None = "038_khora_chunks_chunker_info"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("khora_chunks"):
        # ``khora_chunks`` is created at runtime by
        # ``PgVectorTemporalStore.connect()``; on a fresh deploy the
        # table may not exist when migrations run. The runtime
        # ``CREATE INDEX IF NOT EXISTS`` in pgvector.py will create the
        # index when the table is created.
        return
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_khora_chunks_content_tsv "
            "ON khora_chunks USING GIN (content_tsv)"
        )


def downgrade() -> None:
    if not _is_postgres():
        return
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_khora_chunks_content_tsv")
