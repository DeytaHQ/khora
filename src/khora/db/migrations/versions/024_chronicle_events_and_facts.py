"""Add chronicle_events and memory_facts tables.

Revision ID: 024_chronicle_events_and_facts
Revises: 023_add_document_relationship_count
Create Date: 2026-04-16

Chronicle #1: Schema foundation for the Chronicle engine.

- chronicle_events: SVO event tuples with bi-temporal timestamps + optional
  pgvector embedding for event-channel similarity.
- memory_facts: atomic fact claims with supersession tracking
  (is_active, superseded_by self-FK).

Postgres path: native UUID, JSONB, pgvector + HNSW index on embedding.
SQLite path: TEXT for UUIDs, JSON for arrays, no vector column (LanceDB owns
vectors for the sqlite_lance backend; Chronicle #7 will wire that). Dialect
gating mirrors migration 002 / 004.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from alembic import op

revision: str = "024_chronicle_events_and_facts"
down_revision: str | Sequence[str] | None = "023_add_document_relationship_count"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    is_postgres = _is_postgres()
    uuid_t = UUID(as_uuid=False) if is_postgres else sa.String(36)
    uuid_arr_t = ARRAY(UUID(as_uuid=False)) if is_postgres else sa.JSON

    # ---- chronicle_events --------------------------------------------------
    chronicle_columns = [
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "chunk_id",
            uuid_t,
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("subject", sa.String(512), nullable=False),
        sa.Column("verb", sa.String(255), nullable=False),
        sa.Column("object", sa.String(512), nullable=True),
        sa.Column("observation_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("referenced_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("relative_offset", sa.String(255), nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("source_text", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    ]
    if is_postgres:
        # pgvector column lives only on Postgres; sqlite_lance keeps vectors
        # in LanceDB (Chronicle #7).
        chronicle_columns.insert(-1, sa.Column("embedding", Vector(1536), nullable=True))
    op.create_table("chronicle_events", *chronicle_columns)

    op.create_index(
        "ix_chronicle_events_namespace_referenced_date",
        "chronicle_events",
        ["namespace_id", "referenced_date"],
    )
    op.create_index(
        "ix_chronicle_events_namespace_subject",
        "chronicle_events",
        ["namespace_id", "subject"],
    )
    if is_postgres:
        op.execute(
            "CREATE INDEX ix_chronicle_events_embedding_hnsw "
            "ON chronicle_events USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )

    # ---- memory_facts ------------------------------------------------------
    op.create_table(
        "memory_facts",
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("subject", sa.String(512), nullable=False),
        sa.Column("predicate", sa.String(255), nullable=False),
        sa.Column("object", sa.String(512), nullable=False),
        sa.Column("fact_text", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column(
            "superseded_by",
            uuid_t,
            sa.ForeignKey("memory_facts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "source_chunk_ids",
            uuid_arr_t,
            nullable=False,
            server_default=sa.text("'{}'") if is_postgres else sa.text("'[]'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index(
        "ix_memory_facts_namespace_subject_active",
        "memory_facts",
        ["namespace_id", "subject", "is_active"],
    )
    op.create_index(
        "ix_memory_facts_superseded_by",
        "memory_facts",
        ["superseded_by"],
    )


def downgrade() -> None:
    is_postgres = _is_postgres()

    op.drop_index("ix_memory_facts_superseded_by", table_name="memory_facts")
    op.drop_index("ix_memory_facts_namespace_subject_active", table_name="memory_facts")
    op.drop_table("memory_facts")

    if is_postgres:
        op.execute("DROP INDEX IF EXISTS ix_chronicle_events_embedding_hnsw")
    op.drop_index("ix_chronicle_events_namespace_subject", table_name="chronicle_events")
    op.drop_index("ix_chronicle_events_namespace_referenced_date", table_name="chronicle_events")
    op.drop_table("chronicle_events")
