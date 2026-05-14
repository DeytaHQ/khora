"""Add BRIN index on chunks.created_at for archive-side analytics queries.

Revision ID: 029_chunks_created_at_brin
Revises: 028_typed_entity_recency_index
Create Date: 2026-05-14

Issue #593 — Phase D4. The ``chunks`` table grows roughly time-monotonically,
which is exactly the access pattern BRIN indexes are designed for. Long-range
analytics / export queries that today sequential-scan the table get a tiny
(KB-sized) index that doesn't compete with the HNSW vector indexes or any of
the existing B-trees.

``pages_per_range = 32`` is the default-conservative end of the BRIN tuning
spectrum — keeps the summary granular enough that range filters on a few
weeks of data still skip most of the table.

Concurrent index creation: Postgres ``CREATE INDEX CONCURRENTLY`` cannot run
inside a transaction, so we open an autocommit block. ``IF NOT EXISTS`` makes
the migration safe to re-run.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "029_chunks_created_at_brin"
down_revision: str | Sequence[str] | None = "028_typed_entity_recency_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_created_brin "
            "ON chunks USING BRIN (created_at) WITH (pages_per_range = 32)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_created_brin")
