"""Add temporal tables for Khora engine.

Revision ID: 004_add_temporal_tables
Revises: 003_flexible_type_columns
Create Date: 2026-02-05

This migration adds the temporal infrastructure for the Khora engine:
- time_nodes: Hierarchical time graph (year → quarter → month → week → day)
- temporal_edges: Timestamped relationship edges with bi-temporal model
- time_edge_links: Join table for time-based navigation

Key features:
- BRIN indexes for efficient time-series queries (99% space savings vs btree)
- GiST support for temporal range operations
- Bi-temporal model tracking occurrence and ingestion time
- Edge invalidation for temporal fact updates
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from khora.db.migrations._schema_config import configured_embedding_dimension

# revision identifiers, used by Alembic.
revision: str = "004_add_temporal_tables"
down_revision: str | Sequence[str] | None = "003_flexible_type_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    is_postgres = _is_postgres()
    uuid_t = UUID(as_uuid=False) if is_postgres else sa.String(36)
    jsonb_t = JSONB if is_postgres else sa.JSON
    uuid_arr_t = ARRAY(UUID(as_uuid=False)) if is_postgres else sa.JSON

    # Create time_nodes table — summary_embedding is Postgres-only (pgvector).
    time_node_columns = [
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("granularity", sa.String(10), nullable=False, index=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "parent_id",
            uuid_t,
            sa.ForeignKey("time_nodes.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("summary_text", sa.Text, nullable=True),
        sa.Column("edge_count", sa.Integer, default=0),
        sa.Column("entity_count", sa.Integer, default=0),
        sa.Column("metadata", jsonb_t, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint(
            "namespace_id", "granularity", "start_time", name="uq_time_node_namespace_granularity_start"
        ),
    ]
    if is_postgres:
        # Size from the configured dimension (#1260); safe to edit — Alembic
        # tracks revision IDs, not body content, so only fresh creates change.
        time_node_columns.insert(
            8, sa.Column("summary_embedding", Vector(configured_embedding_dimension()), nullable=True)
        )
    op.create_table("time_nodes", *time_node_columns)

    # Create index for range queries on time_nodes
    op.create_index(
        "ix_time_nodes_namespace_range",
        "time_nodes",
        ["namespace_id", "start_time", "end_time"],
    )

    # Create temporal_edges table
    op.create_table(
        "temporal_edges",
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "source_entity_id",
            uuid_t,
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "target_entity_id",
            uuid_t,
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("relationship_type", sa.String(64), default="RELATES_TO", index=True),
        sa.Column("description", sa.Text, default=""),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_valid", sa.Boolean, default=True, index=True),
        sa.Column(
            "invalidated_by_id",
            uuid_t,
            sa.ForeignKey("temporal_edges.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("invalidation_reason", sa.Text, nullable=True),
        sa.Column("confidence", sa.Float, default=1.0),
        sa.Column("properties", jsonb_t, default=dict),
        sa.Column("source_document_ids", uuid_arr_t, default=list),
        sa.Column("source_chunk_ids", uuid_arr_t, default=list),
        sa.Column("metadata", jsonb_t, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # BRIN index is Postgres-only. SQLite gets a regular btree on the same column.
    if is_postgres:
        op.execute("""
            CREATE INDEX ix_temporal_edges_occurred_brin
            ON temporal_edges USING BRIN (occurred_at)
            """)
    else:
        op.create_index("ix_temporal_edges_occurred_brin", "temporal_edges", ["occurred_at"])

    # Create composite index for entity pair + time queries
    op.create_index(
        "ix_temporal_edges_entities_time",
        "temporal_edges",
        ["source_entity_id", "target_entity_id", "occurred_at"],
    )

    # Create index for validity range queries
    op.create_index(
        "ix_temporal_edges_valid_range",
        "temporal_edges",
        ["valid_from", "valid_until"],
    )

    # Create time_edge_links join table
    op.create_table(
        "time_edge_links",
        sa.Column(
            "time_node_id",
            uuid_t,
            sa.ForeignKey("time_nodes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "edge_id",
            uuid_t,
            sa.ForeignKey("temporal_edges.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("time_edge_links")
    op.drop_index("ix_temporal_edges_valid_range", table_name="temporal_edges")
    op.drop_index("ix_temporal_edges_entities_time", table_name="temporal_edges")
    op.drop_index("ix_temporal_edges_occurred_brin", table_name="temporal_edges")
    op.drop_table("temporal_edges")
    op.drop_index("ix_time_nodes_namespace_range", table_name="time_nodes")
    op.drop_table("time_nodes")
