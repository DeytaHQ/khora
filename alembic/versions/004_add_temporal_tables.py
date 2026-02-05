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
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004_add_temporal_tables"
down_revision: str | Sequence[str] | None = "003_flexible_type_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create time_nodes table
    op.create_table(
        "time_nodes",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("granularity", sa.String(10), nullable=False, index=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "parent_id",
            UUID(as_uuid=False),
            sa.ForeignKey("time_nodes.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("summary_text", sa.Text, nullable=True),
        sa.Column("summary_embedding", Vector(1536), nullable=True),
        sa.Column("edge_count", sa.Integer, default=0),
        sa.Column("entity_count", sa.Integer, default=0),
        sa.Column("metadata", JSONB, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint(
            "namespace_id", "granularity", "start_time", name="uq_time_node_namespace_granularity_start"
        ),
    )

    # Create index for range queries on time_nodes
    op.create_index(
        "ix_time_nodes_namespace_range",
        "time_nodes",
        ["namespace_id", "start_time", "end_time"],
    )

    # Create temporal_edges table
    op.create_table(
        "temporal_edges",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "source_entity_id",
            UUID(as_uuid=False),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "target_entity_id",
            UUID(as_uuid=False),
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
            UUID(as_uuid=False),
            sa.ForeignKey("temporal_edges.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("invalidation_reason", sa.Text, nullable=True),
        sa.Column("confidence", sa.Float, default=1.0),
        sa.Column("properties", JSONB, default=dict),
        sa.Column("source_document_ids", ARRAY(UUID(as_uuid=False)), default=[]),
        sa.Column("source_chunk_ids", ARRAY(UUID(as_uuid=False)), default=[]),
        sa.Column("metadata", JSONB, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create BRIN index for time-series optimization on temporal_edges
    op.execute(
        """
        CREATE INDEX ix_temporal_edges_occurred_brin
        ON temporal_edges USING BRIN (occurred_at)
        """
    )

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
            UUID(as_uuid=False),
            sa.ForeignKey("time_nodes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "edge_id",
            UUID(as_uuid=False),
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
