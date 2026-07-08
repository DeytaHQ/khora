"""Add GIN index on entities.source_chunk_ids for the #1448 overlap pushdown.

Revision ID: 052_entities_source_chunk_ids_gin
Revises: 051_documents_graph_mirror_pending
Create Date: 2026-07-07

PR #1449 pushes the #857 recall entity projection down into the store; on
pgvector it compiles to an ``&&`` array-overlap
(``EntityModel.source_chunk_ids.overlap(...)``) against
``entities.source_chunk_ids`` (``ARRAY(UUID)``). Without an index that
predicate is a sequential scan of ``entities`` on every recall on graph-less
PostgreSQL stacks. A GIN index with the default ``array_ops`` operator class
serves the ``&&`` operator directly.

Concurrent index creation: Postgres ``CREATE INDEX CONCURRENTLY`` cannot run
inside a transaction, so we open an autocommit block. ``IF NOT EXISTS`` makes
the migration safe to re-run.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "052_entities_source_chunk_ids_gin"
down_revision: str | Sequence[str] | None = "051_documents_graph_mirror_pending"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    # GIN and ``CREATE INDEX CONCURRENTLY`` are Postgres-only. The
    # sqlite_lance test fixtures run the full Alembic chain against
    # SQLite, so we skip silently on non-Postgres dialects rather than
    # emit syntax SQLite cannot parse.
    if not _is_postgres():
        return
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_entities_source_chunk_ids_gin "
            "ON entities USING GIN (source_chunk_ids)"
        )


def downgrade() -> None:
    if not _is_postgres():
        return
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_entities_source_chunk_ids_gin")
