"""Add graph_mirror_pending column to khora_dream_runs.

Revision ID: 047_dream_runs_graph_mirror_pending
Revises: 046_chunks_occurred_at
Create Date: 2026-06-19

Issue #1274 - Phase-2 foundation for dream-on-graph (umbrella #1282). The
post-commit graph mirror (#1272) advances the dream checkpoint inside the
PG apply transaction, *before* the graph write runs, so a failed mirror
leaves a committed-but-unmirrored op. The reconciler re-attempts those ops
from a per-op pending list. This migration adds the home for that list on
the relational run-state table.

``graph_mirror_pending`` holds a JSON array of pending op entries, each
shaped ``{"op_seq": int, "op_id": str, "op_type": str, "payload": {...}}``.
NULL / absent means "no ops awaiting mirror" - the column is nullable and
defaults to NULL so existing run rows are untouched.

Dialect-portable (#896): the plain ``op.add_column`` materializes on both
PostgreSQL (``JSONB``) and SQLite (``JSON``) - the sqlite_lance test
fixtures run the full Alembic chain against SQLite. The SurrealDB-unified
stack stores the same state on a ``DEFINE``-d ``khora_dream_runs`` table
(no Alembic) - see ``storage/backends/surrealdb/schema.py``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "047_dream_runs_graph_mirror_pending"
down_revision: str | Sequence[str] | None = "046_chunks_occurred_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_NAME = "khora_dream_runs"
COLUMN_NAME = "graph_mirror_pending"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _json_type() -> sa.types.TypeEngine:
    return JSONB() if _is_postgres() else sa.JSON()


def upgrade() -> None:
    op.add_column(TABLE_NAME, sa.Column(COLUMN_NAME, _json_type(), nullable=True))


def downgrade() -> None:
    op.drop_column(TABLE_NAME, COLUMN_NAME)
