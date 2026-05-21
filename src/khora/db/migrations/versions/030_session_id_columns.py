"""Add nullable session_id UUID column to documents, chunks, and downstream tables.

Revision ID: 030_session_id_columns
Revises: 029_chunks_created_at_brin
Create Date: 2026-05-15

Issue #620 — first-class ``session_id`` column for agentic-framework adapters
(namespace = (framework, app, user), session = conversation/run, document =
turn). The field lives in metadata.custom["session_id"] today, which forces
JSONB sequential scans for session-scoped recalls. Promoting it to a real
column lets queries hit composite (namespace_id, session_id) indexes
(see migration 031).

The column is nullable on all five tables — existing rows naturally carry
NULL, and adapters that don't track sessions can keep ignoring the field.

Works on both PostgreSQL and SQLite. Index creation is split into migration
031 (Postgres-only ``CREATE INDEX CONCURRENTLY``).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "030_session_id_columns"
down_revision: str | Sequence[str] | None = "029_chunks_created_at_brin"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TABLES = ("documents", "chunks", "memory_events", "chronicle_events", "memory_facts")


def _uuid_type() -> sa.types.TypeEngine:
    """UUID column type appropriate for the current dialect.

    PostgreSQL uses the native ``UUID`` type; SQLite stores UUIDs as
    32-char TEXT (the same shape ``sa.Uuid`` lands on) so the sqlite_lance
    relational adapter can round-trip them via its existing ``Uuid``
    converters.
    """
    if op.get_bind().dialect.name == "postgresql":
        return PG_UUID(as_uuid=True)
    return sa.Uuid(as_uuid=True)


def upgrade() -> None:
    col_type = _uuid_type()
    for table in _TABLES:
        op.add_column(table, sa.Column("session_id", col_type, nullable=True))


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.drop_column(table, "session_id")
