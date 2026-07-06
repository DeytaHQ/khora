"""Add graph_mirror_pending column to documents.

Revision ID: 051_documents_graph_mirror_pending
Revises: 050_keyword_chunks
Create Date: 2026-07-06

Issue #1430 - durable marker for the external_id replace path, modeled on
the dream reconciler's ``khora_dream_runs.graph_mirror_pending`` (#1272,
migration 047). ``replace_document_extraction`` commits Postgres (new
chunks + COMPLETED status) first and mirrors to the graph afterwards,
outside any transaction. A graph failure in that window used to leave a
durable PG/graph divergence that only the next successful replace healed
(#884 made it observable, not recoverable). This column persists the
computed graph plan so the replace-mirror reconciler can replay it.

``graph_mirror_pending`` holds one JSON payload (retire / remap / strip
rows plus the net-new entities and relationships) shaped by
``khora.storage.replace_mirror.build_replace_mirror_payload``. NULL /
absent means "graph is in lockstep with PG for this document" - the
column is nullable and defaults to NULL so existing rows are untouched.

Dialect handling: the plain ``op.add_column`` materializes on both
PostgreSQL (``JSONB``) and SQLite (``JSON``) - the sqlite_lance test
fixtures run the full Alembic chain. The partial lookup index is
Postgres-only (``CREATE INDEX CONCURRENTLY`` in an autocommit block,
same shape as migration 031); markers are only ever written on
graph-backed PG stacks, so SQLite skips it silently.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "051_documents_graph_mirror_pending"
down_revision: str | Sequence[str] | None = "050_keyword_chunks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_NAME = "documents"
COLUMN_NAME = "graph_mirror_pending"
INDEX_NAME = "ix_documents_graph_mirror_pending"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _json_type() -> sa.types.TypeEngine:
    return JSONB() if _is_postgres() else sa.JSON()


def _has_column() -> bool:
    return COLUMN_NAME in {c["name"] for c in sa.inspect(op.get_bind()).get_columns(TABLE_NAME)}


def upgrade() -> None:
    # Idempotent on the live schema (same rationale as migration 047): the
    # integration migration harness shares one PostgreSQL instance across
    # parallel test files, so guard on the actual column rather than the
    # version table.
    if not _has_column():
        op.add_column(TABLE_NAME, sa.Column(COLUMN_NAME, _json_type(), nullable=True))

    if not _is_postgres():
        return

    # Partial index so the reconciler's "any pending markers in this
    # namespace?" probe stays O(pending) instead of scanning documents.
    # Pending markers are rare (one per failed graph mirror), so the index
    # is KB-sized.
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
            f"ON {TABLE_NAME} (namespace_id) "
            f"WHERE {COLUMN_NAME} IS NOT NULL"
        )


def downgrade() -> None:
    if _is_postgres():
        with op.get_context().autocommit_block():
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
    if _has_column():
        op.drop_column(TABLE_NAME, COLUMN_NAME)
