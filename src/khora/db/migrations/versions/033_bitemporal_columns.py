"""Add bi-temporal soft-delete columns to relationships and memory_facts.

Revision ID: 033_bitemporal_columns
Revises: 032_dream_runs
Create Date: 2026-05-16

Issue #653 — Phase 0.3 of the dream-phase rollout (umbrella #649). v0.14
introduces three nullable columns on both ``relationships`` and
``memory_facts`` so that Phase 4 (v0.15) ``apply``-mode dream operations
have a place to write soft-delete state. v0.14 dream is dry-run only, so
the columns are read-only-NULL on the migration side. Existing rows
backfill to all NULL — meaning "still valid".

Schema additions per table (relationships, memory_facts):

* ``valid_to`` ``TIMESTAMPTZ`` NULL — closes the validity window
* ``invalidated_at`` ``TIMESTAMPTZ`` NULL — soft-delete timestamp
* ``invalidated_by`` ``UUID`` NULL — dream run that performed the
  invalidation (semantic FK to ``khora_dream_runs.run_id`` introduced
  in migration 032; not enforced as a DB-level FK in this migration —
  see "FK enforcement" below)

Partial indexes (Postgres-only, dialect-gated; mirrors the migration 029
pattern of ``CREATE INDEX CONCURRENTLY`` inside an autocommit block):

* ``ix_relationships_live`` —
  ``(namespace_id, source_entity_id, target_entity_id, relationship_type)``
  ``WHERE invalidated_at IS NULL``
* ``ix_memory_facts_live`` — ``(namespace_id, subject)``
  ``WHERE invalidated_at IS NULL``

Both accelerate the live-fact retrieval path without changing the
existing ``is_active`` filter behavior on ``memory_facts``.

Coexistence with ``is_active``: ``memory_facts.is_active`` (boolean) and
``invalidated_at IS NULL`` (new bi-temporal predicate) coexist in v0.14.
Both predicates filter "live" rows. Deprecation of ``is_active`` is a
v0.16+ concern; this migration does not touch it.

FK enforcement: ``invalidated_by`` is a semantic reference to
``khora_dream_runs.run_id`` (migration 032). It is NOT enforced as a
DB-level FK in this migration — matching the loose-coupling style of
migration 032 itself, which does not declare ``namespace_id`` as a FK
to ``memory_namespaces.id``. The constraint can be added in a later
migration if operational experience demands it.

Depends on migration 032 (``032_dream_runs``, PR #676). Must merge
after #676 lands on main.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "033_bitemporal_columns"
down_revision: str | Sequence[str] | None = "032_dream_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TABLES = ("relationships", "memory_facts")


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _uuid_type() -> sa.types.TypeEngine:
    """UUID column type appropriate for the current dialect.

    Matches the approach taken in migration 030 — Postgres uses the
    native UUID type; SQLite stores UUIDs as 32-char TEXT via
    ``sa.Uuid(as_uuid=True)``.
    """
    if _is_postgres():
        return PG_UUID(as_uuid=True)
    return sa.Uuid(as_uuid=True)


def upgrade() -> None:
    # Postgres path uses raw SQL with IF NOT EXISTS for idempotency
    # (tests run the chain repeatedly; production rollouts also need to
    # tolerate partial-state retry after a crash). SQLite path uses
    # op.add_column which fails loudly if columns exist — acceptable
    # because the sqlite_lance fixture stack rebuilds per test.
    if _is_postgres():
        for table in _TABLES:
            op.execute(
                f"ALTER TABLE {table} "
                "ADD COLUMN IF NOT EXISTS valid_to TIMESTAMP WITH TIME ZONE NULL, "
                "ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMP WITH TIME ZONE NULL, "
                "ADD COLUMN IF NOT EXISTS invalidated_by UUID NULL"
            )

        with op.get_context().autocommit_block():
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_relationships_live "
                "ON relationships (namespace_id, source_entity_id, target_entity_id, relationship_type) "
                "WHERE invalidated_at IS NULL"
            )
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_memory_facts_live "
                "ON memory_facts (namespace_id, subject) "
                "WHERE invalidated_at IS NULL"
            )
        return

    # SQLite path — no IF NOT EXISTS support on ADD COLUMN; skip partial
    # indexes per migration 031's pattern.
    uuid_type = _uuid_type()
    for table in _TABLES:
        op.add_column(table, sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True))
        op.add_column(table, sa.Column("invalidated_at", sa.DateTime(timezone=True), nullable=True))
        op.add_column(table, sa.Column("invalidated_by", uuid_type, nullable=True))


def downgrade() -> None:
    if _is_postgres():
        with op.get_context().autocommit_block():
            op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_memory_facts_live")
            op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_relationships_live")
        for table in reversed(_TABLES):
            op.execute(
                f"ALTER TABLE {table} "
                "DROP COLUMN IF EXISTS invalidated_by, "
                "DROP COLUMN IF EXISTS invalidated_at, "
                "DROP COLUMN IF EXISTS valid_to"
            )
        return

    for table in reversed(_TABLES):
        op.drop_column(table, "invalidated_by")
        op.drop_column(table, "invalidated_at")
        op.drop_column(table, "valid_to")
