"""Add khora_dream_communities for Phase 5.1 community-summary persistence.

Revision ID: 035_dream_communities
Revises: 034_chronicle_events_bitemporal
Create Date: 2026-05-18

Issue #670 — Phase 5.1 of the dream-phase rollout (umbrella #649). The
community-summary apply path persists LLM-grounded per-community
summaries with bi-temporal validity. The orchestrator's caller-owned
transaction writes directly into this table from
``apply_vectorcypher_community_summary``.

Postgres-only via the same dialect gate as migrations 029 / 032 /
033 / 034. The embedded ``sqlite_lance`` stack mirrors community state
to the JSONL undo sink — running the chain against SQLite must remain
a clean no-op.

Schema (10 columns):

* ``id`` UUID PK — deterministic UUID5 of ``(namespace_id, sorted
  member ids)`` so a replayed op finds the same row.
* ``namespace_id`` UUID NOT NULL — stable namespace id.
* ``op_id`` UUID NOT NULL — DreamOp.op_id that wrote this row. Carries
  through to the run's undo.json for downstream audit.
* ``member_ids`` UUID[] NOT NULL — community entity ids (top-k carried).
* ``payload`` JSONB NOT NULL — ``{text, claims[], summary_depth, model}``.
  Grounding-filtered claims only; the apply handler drops uncited or
  fabricated claims before INSERT.
* ``summary_depth`` INTEGER NOT NULL DEFAULT 1 — tag for retrieval
  pools so they refuse to mix summary depths (enforcement lives in
  retrieval; this column is the durable marker).
* ``valid_from`` TIMESTAMPTZ NOT NULL — bi-temporal start.
* ``valid_to`` TIMESTAMPTZ NULL — bi-temporal end (NULL = live).
* ``created_at`` / ``updated_at`` TIMESTAMPTZ NOT NULL.

Indexes:

* ``ix_khora_dream_communities_ns_live`` on
  ``(namespace_id, valid_to)`` ``WHERE valid_to IS NULL`` — accelerates
  the apply-handler's idempotent-replay short-circuit.
* ``ix_khora_dream_communities_op`` on ``op_id`` — covers
  undo-by-op_id lookups.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision: str = "035_dream_communities"
down_revision: str | Sequence[str] | None = "034_chronicle_events_bitemporal"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_NAME = "khora_dream_communities"
INDEX_LIVE = "ix_khora_dream_communities_ns_live"
INDEX_OP = "ix_khora_dream_communities_op"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return

    op.create_table(
        TABLE_NAME,
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column("namespace_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("op_id", PG_UUID(as_uuid=True), nullable=False),
        sa.Column("member_ids", ARRAY(PG_UUID(as_uuid=True)), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column(
            "summary_depth",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column("valid_from", TIMESTAMP(timezone=True), nullable=False),
        sa.Column("valid_to", TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.create_index(
        INDEX_LIVE,
        TABLE_NAME,
        ["namespace_id", "valid_to"],
        postgresql_where=sa.text("valid_to IS NULL"),
    )
    op.create_index(INDEX_OP, TABLE_NAME, ["op_id"])


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute(f"DROP INDEX IF EXISTS {INDEX_OP}")
    op.execute(f"DROP INDEX IF EXISTS {INDEX_LIVE}")
    op.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
