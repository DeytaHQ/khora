"""Add khora_hook_subscriptions table for durable semantic hooks (#599).

Revision ID: 049_hook_subscriptions
Revises: 048_dream_conflicts_reconcile
Create Date: 2026-06-21

Issue #599 - Phase 3 of the semantic-hooks rollout (refs #577, #580). The
``HookDispatcher`` keeps subscriptions in process memory, so a restart
loses every subscription. This table is the durable home for persistent
subscriptions: ``HookDispatcher.register_persistent`` writes a row,
``load_persistent`` re-reads them on startup so events delivered after a
restart still find the subscriber. The in-process callback path (today's
``subscribe``) is unchanged and never touches this table.

Created on BOTH dialects (#896). The DDL is dialect-portable so the
embedded ``sqlite_lance`` stack persists subscription rows too; UUID
columns follow migration 030/032's helper (native ``UUID`` on Postgres,
``sa.Uuid`` TEXT on SQLite), ``filter`` / ``delivery`` are ``JSONB`` on
Postgres and ``sa.JSON`` on SQLite. The SurrealDB-unified stack has no
Alembic - persistence there is out of scope for #599 (PG / SQLite only).

Schema (9 columns):

* ``id`` UUID PK - the subscription id (matches the in-process id)
* ``namespace_id`` UUID - scope; nullable (NULL = all namespaces)
* ``event_type`` TEXT NOT NULL - EventType value (e.g. ``entity.created``)
* ``filter`` JSONB - serialized SemanticFilter (NULL = no filter)
* ``delivery`` JSONB NOT NULL - webhook URL / queue config for the worker
* ``created_at`` TIMESTAMPTZ NOT NULL
* ``last_delivered_at`` TIMESTAMPTZ - stamped by the (out-of-scope) worker
* ``delivery_failure_count`` INTEGER DEFAULT 0
* ``paused_at`` TIMESTAMPTZ - set to pause delivery without deleting

Index ``ix_khora_hook_subscriptions_ns_event`` on
``(namespace_id, event_type)`` covers the load-on-startup filter.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "049_hook_subscriptions"
down_revision: str | Sequence[str] | None = "048_dream_conflicts_reconcile"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_NAME = "khora_hook_subscriptions"
INDEX_NAME = "ix_khora_hook_subscriptions_ns_event"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _uuid_type() -> sa.types.TypeEngine:
    if _is_postgres():
        return PG_UUID(as_uuid=True)
    return sa.Uuid(as_uuid=True)


def _timestamp_type() -> sa.types.TypeEngine:
    if _is_postgres():
        return TIMESTAMP(timezone=True)
    return sa.DateTime(timezone=True)


def _json_type() -> sa.types.TypeEngine:
    return JSONB() if _is_postgres() else sa.JSON()


def _has_table() -> bool:
    return sa.inspect(op.get_bind()).has_table(TABLE_NAME)


def _has_index() -> bool:
    return any(ix["name"] == INDEX_NAME for ix in sa.inspect(op.get_bind()).get_indexes(TABLE_NAME))


def upgrade() -> None:
    # Idempotent on the live schema: the integration migration harness shares
    # one PostgreSQL instance across parallel test files (each resets via
    # DROP SCHEMA public CASCADE), so a plain CREATE TABLE can re-run against
    # an already-migrated DB. Guard on the table, not the version row.
    if not _has_table():
        uuid_type = _uuid_type()
        ts_type = _timestamp_type()

        op.create_table(
            TABLE_NAME,
            sa.Column("id", uuid_type, primary_key=True),
            sa.Column("namespace_id", uuid_type, nullable=True),
            sa.Column("event_type", sa.Text(), nullable=False),
            sa.Column("filter", _json_type(), nullable=True),
            sa.Column("delivery", _json_type(), nullable=False),
            sa.Column("created_at", ts_type, nullable=False),
            sa.Column("last_delivered_at", ts_type, nullable=True),
            sa.Column(
                "delivery_failure_count",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column("paused_at", ts_type, nullable=True),
        )

    # Create the index unconditionally (guarded only on the index itself) so a
    # prior run that built the table but failed before this line still gets the
    # index on a re-run — the table-exists early-return used to skip it.
    if not _has_index():
        op.create_index(INDEX_NAME, TABLE_NAME, ["namespace_id", "event_type"])


def downgrade() -> None:
    # IF EXISTS so a downgrade against a partial-state DB (table DROPped
    # out-of-band, version row still pointing here) is idempotent.
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
    if _is_postgres():
        op.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
    else:
        op.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
