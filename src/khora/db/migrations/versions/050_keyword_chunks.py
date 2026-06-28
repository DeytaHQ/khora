"""Add keyword_chunks bipartite edge table for the keyword_ppr channel (#1391).

Revision ID: 050_keyword_chunks
Revises: 049_hook_subscriptions
Create Date: 2026-06-27

Issue #1391 - the experimental KET-RAG "text-keyword" retrieval channel
(``query.lexical_channel == "keyword_ppr"``). This table holds the
keyword -> chunk bipartite: one row per (keyword, chunk) with the keyword's
ingest-time IDF. Rows are written at ingest only when the channel is enabled
(default ``bm25`` deployments never touch it). At query time the channel loads
a namespace's edges (capped), builds chunk->chunk edges, and runs personalized
PageRank.

Created on BOTH dialects (Postgres + the embedded ``sqlite_lance`` stack). The
DDL is intentionally plain SQLAlchemy with no Postgres-only features, so no
dialect gating of the columns is needed beyond the portable UUID / timestamp
helpers (native ``UUID`` / ``TIMESTAMPTZ`` on Postgres, ``sa.Uuid`` /
``sa.DateTime`` on SQLite, matching migration 049). UUID columns use
``as_uuid=True`` so the on-disk format matches what the raw-aiosqlite adapters
write via ``uuid_to_text`` (32-char hex, no dashes). The SurrealDB-unified
stack has no Alembic chain and is out of scope.

Schema (6 columns):

* ``id`` UUID PK
* ``namespace_id`` UUID NOT NULL - FK to memory_namespaces.id (CASCADE)
* ``keyword`` TEXT NOT NULL
* ``chunk_id`` UUID NOT NULL - FK to chunks.id (CASCADE)
* ``idf`` FLOAT NOT NULL - ingest-time inverse document frequency
* ``created_at`` TIMESTAMPTZ NOT NULL

Indexes:
* ``ix_keyword_chunks_namespace_keyword`` on ``(namespace_id, keyword)`` -
  the per-query edge load filter.
* ``ix_keyword_chunks_namespace_chunk`` on ``(namespace_id, chunk_id)`` -
  per-chunk idempotent re-ingest deletes.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "050_keyword_chunks"
down_revision: str | Sequence[str] | None = "049_hook_subscriptions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TABLE_NAME = "keyword_chunks"
KEYWORD_INDEX = "ix_keyword_chunks_namespace_keyword"
CHUNK_INDEX = "ix_keyword_chunks_namespace_chunk"


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


def _has_table() -> bool:
    return sa.inspect(op.get_bind()).has_table(TABLE_NAME)


def _has_index(name: str) -> bool:
    return any(ix["name"] == name for ix in sa.inspect(op.get_bind()).get_indexes(TABLE_NAME))


def upgrade() -> None:
    # Idempotent on the live schema: the integration migration harness shares
    # one PostgreSQL instance across parallel test files (each resets via
    # DROP SCHEMA public CASCADE), so a plain CREATE TABLE can re-run against
    # an already-migrated DB. Guard on the table, not the version row.
    if not _has_table():
        uuid_type = _uuid_type()
        op.create_table(
            TABLE_NAME,
            sa.Column("id", uuid_type, primary_key=True),
            sa.Column(
                "namespace_id",
                uuid_type,
                sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("keyword", sa.Text(), nullable=False),
            sa.Column(
                "chunk_id",
                uuid_type,
                sa.ForeignKey("chunks.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("idf", sa.Float(), nullable=False),
            sa.Column("created_at", _timestamp_type(), nullable=False),
        )

    # Create indexes unconditionally (guarded only on the index itself) so a
    # prior run that built the table but failed before these lines still gets
    # them on a re-run.
    if not _has_index(KEYWORD_INDEX):
        op.create_index(KEYWORD_INDEX, TABLE_NAME, ["namespace_id", "keyword"])
    if not _has_index(CHUNK_INDEX):
        op.create_index(CHUNK_INDEX, TABLE_NAME, ["namespace_id", "chunk_id"])


def downgrade() -> None:
    # IF EXISTS so a downgrade against a partial-state DB is idempotent.
    op.execute(f"DROP INDEX IF EXISTS {KEYWORD_INDEX}")
    op.execute(f"DROP INDEX IF EXISTS {CHUNK_INDEX}")
    if _is_postgres():
        op.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
    else:
        op.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
