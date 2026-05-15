"""Add Postgres-only indexes on (namespace_id, session_id) and BRIN(session_id, created_at).

Revision ID: 031_session_id_indexes
Revises: 030_session_id_columns
Create Date: 2026-05-15

Issue #620 — companion to migration 030. Splits index creation out so it can
run via ``CREATE INDEX CONCURRENTLY`` inside an autocommit block, avoiding
table-lock contention on production traffic.

Indexes created (Postgres-only — SQLite-backed sqlite_lance stacks skip
silently like migration 029):

* ``ix_chunks_ns_session`` — partial B-tree on ``(namespace_id, session_id)``
  with ``WHERE session_id IS NOT NULL``. Covers session-scoped recall:
  ``WHERE namespace_id = ? AND session_id = ?``.
* ``ix_documents_ns_session`` — mirror of the chunks index for document-level
  filters (``Khora.forget_session`` cascade lookup).
* ``ix_chunks_session_created_brin`` — BRIN on ``(session_id, created_at)``.
  Sessions are append-only and time-correlated; BRIN gives a KB-sized
  summary that accelerates ``gc.expire_sessions(before=…)`` and
  time-bounded session replays without competing with the existing HNSW
  and partial-btree indexes.

No GIN on ``chunks.metadata`` — see the ticket for the cardinality
rationale.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "031_session_id_indexes"
down_revision: str | Sequence[str] | None = "030_session_id_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    # Concurrent index creation + partial / BRIN indexes are Postgres-only.
    # SQLite-backed sqlite_lance stacks skip silently — the column from 030
    # still exists and unindexed lookups are fine at SQLite's scale.
    if not _is_postgres():
        return

    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_ns_session "
            "ON chunks (namespace_id, session_id) "
            "WHERE session_id IS NOT NULL"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_ns_session "
            "ON documents (namespace_id, session_id) "
            "WHERE session_id IS NOT NULL"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_session_created_brin "
            "ON chunks USING BRIN (session_id, created_at) "
            "WITH (pages_per_range = 32)"
        )


def downgrade() -> None:
    if not _is_postgres():
        return
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_session_created_brin")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_documents_ns_session")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_ns_session")
