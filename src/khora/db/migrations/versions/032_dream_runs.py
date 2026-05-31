"""Add khora_dream_runs checkpoint table for dream-phase orchestrator.

Revision ID: 032_dream_runs
Revises: 031_session_id_indexes
Create Date: 2026-05-16

Issue #651 — Phase 0.2 of the dream-phase rollout (umbrella #649). The
dream orchestrator needs a per-namespace audit/checkpoint table so a
crashed APPLY pass can be resumed against the last committed op-seq
rather than restarted from scratch.

Created on BOTH dialects (#896). The table DDL is dialect-portable, so
the embedded ``sqlite_lance`` stack now persists run rows here too -
``Khora.dream_history`` / ``Khora.dream(...).status`` return real rows
on SQLite instead of ``[]``. UUID columns follow migration 030's
dialect helper (native ``UUID`` on Postgres, ``sa.Uuid`` TEXT on
SQLite); ``error`` is ``JSONB`` on Postgres and ``sa.JSON`` on SQLite.

Schema (16 columns):

* ``run_id`` UUID PK
* ``namespace_id`` UUID NOT NULL — stable namespace id, queried by
  ``Khora.dream_history(namespace_id)``
* ``trigger`` VARCHAR(32) NOT NULL — ``manual`` | ``resume`` |
  ``reconciler`` | etc.
* ``mode`` VARCHAR(16) NOT NULL — ``dry-run`` | ``apply``
* ``state`` VARCHAR(32) NOT NULL — ``init`` | ``planning`` |
  ``applying`` | ``completed`` | ``partial_failed`` | ``cancelled`` |
  ``crashed``
* ``plan_hash`` VARCHAR(64) — sha256 of canonicalised plan
* ``started_at`` TIMESTAMPTZ NOT NULL
* ``finished_at`` TIMESTAMPTZ
* ``last_committed_op_seq`` INTEGER DEFAULT -1 — resume cursor
* ``heartbeat_at`` TIMESTAMPTZ NOT NULL
* ``total_ops`` INTEGER DEFAULT 0
* ``total_decisions`` INTEGER DEFAULT 0
* ``report_path`` TEXT
* ``manifest_sha256`` VARCHAR(64)
* ``config_fingerprint`` VARCHAR(64)
* ``error`` JSONB

Index ``ix_khora_dream_runs_namespace_started`` on
``(namespace_id, started_at DESC)`` covers the dream-history listing
path.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "032_dream_runs"
down_revision: str | Sequence[str] | None = "031_session_id_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_NAME = "khora_dream_runs"
INDEX_NAME = "ix_khora_dream_runs_namespace_started"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _uuid_type() -> sa.types.TypeEngine:
    """UUID column type appropriate for the current dialect (mirrors 030)."""
    if _is_postgres():
        return PG_UUID(as_uuid=True)
    return sa.Uuid(as_uuid=True)


def _timestamp_type() -> sa.types.TypeEngine:
    if _is_postgres():
        return TIMESTAMP(timezone=True)
    return sa.DateTime(timezone=True)


def _json_type() -> sa.types.TypeEngine:
    return JSONB() if _is_postgres() else sa.JSON()


def upgrade() -> None:
    uuid_type = _uuid_type()
    ts_type = _timestamp_type()

    op.create_table(
        TABLE_NAME,
        sa.Column("run_id", uuid_type, primary_key=True),
        sa.Column("namespace_id", uuid_type, nullable=False),
        sa.Column("trigger", sa.String(length=32), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("plan_hash", sa.String(length=64), nullable=True),
        sa.Column("started_at", ts_type, nullable=False),
        sa.Column("finished_at", ts_type, nullable=True),
        sa.Column(
            "last_committed_op_seq",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("-1"),
        ),
        sa.Column("heartbeat_at", ts_type, nullable=False),
        sa.Column(
            "total_ops",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "total_decisions",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("0"),
        ),
        sa.Column("report_path", sa.Text(), nullable=True),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=True),
        sa.Column("config_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("error", _json_type(), nullable=True),
    )

    op.create_index(
        INDEX_NAME,
        TABLE_NAME,
        [sa.text("namespace_id"), sa.text("started_at DESC")],
    )


def downgrade() -> None:
    # IF EXISTS so a downgrade against a partial-state DB (e.g., the table
    # was DROPped out-of-band, leaving the alembic_version row pointing at
    # this revision) is idempotent rather than crashing.
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
    if _is_postgres():
        op.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
    else:
        op.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
