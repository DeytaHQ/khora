"""Initial database schema.

Revision ID: 000_initial_schema
Revises:
Create Date: 2026-01-27

Creates all base tables for Khora Memory Lake:
- Multi-tenancy: organizations, workspaces, memory_namespaces
- Documents: documents, chunks
- Entities: entities, relationships, episodes
- Event sourcing: memory_events
- ACL: permissions
- Sync: sync_checkpoints
- Expertise: expertise_definitions
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "000_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial schema."""
    # Create enum types
    op.execute("CREATE TYPE tenancy_mode AS ENUM ('shared', 'isolated')")
    op.execute("CREATE TYPE document_status AS ENUM ('pending', 'processing', 'completed', 'failed')")
    op.execute("""CREATE TYPE entity_type AS ENUM (
            'PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 'CONCEPT',
            'PRODUCT', 'TECHNOLOGY', 'DOCUMENT', 'PROJECT', 'TASK',
            'MEETING', 'DECISION', 'METRIC', 'GOAL', 'CUSTOM'
        )""")
    op.execute("""CREATE TYPE relationship_type AS ENUM (
            'RELATES_TO', 'WORKS_FOR', 'WORKS_WITH', 'MANAGES', 'REPORTS_TO',
            'OWNS', 'CREATED', 'MODIFIED', 'MENTIONED_IN', 'DISCUSSED_IN',
            'PARTICIPATED_IN', 'ATTENDED', 'DECIDED', 'ASSIGNED_TO', 'BLOCKED_BY',
            'DEPENDS_ON', 'PART_OF', 'LOCATED_IN', 'OCCURRED_AT', 'CAUSED',
            'INFLUENCED', 'SIMILAR_TO', 'OPPOSITE_OF', 'DERIVED_FROM', 'CUSTOM'
        )""")
    op.execute("""CREATE TYPE event_type AS ENUM (
            'DOCUMENT_CREATED', 'DOCUMENT_UPDATED', 'DOCUMENT_DELETED', 'DOCUMENT_PROCESSED',
            'ENTITY_CREATED', 'ENTITY_UPDATED', 'ENTITY_MERGED', 'ENTITY_DELETED',
            'RELATIONSHIP_CREATED', 'RELATIONSHIP_UPDATED', 'RELATIONSHIP_DELETED',
            'EPISODE_CREATED', 'EPISODE_UPDATED', 'EPISODE_DELETED',
            'MEMORY_QUERIED', 'MEMORY_RECALLED', 'MEMORY_CONSOLIDATED',
            'NAMESPACE_CREATED', 'NAMESPACE_UPDATED', 'NAMESPACE_DELETED'
        )""")

    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Organizations table
    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column(
            "tenancy_mode",
            postgresql.ENUM("shared", "isolated", name="tenancy_mode", create_type=False),
            server_default="shared",
        ),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Workspaces table
    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, index=True),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("organization_id", "slug", name="uq_workspace_org_slug"),
    )

    # Memory namespaces table (with versioning columns)
    op.create_table(
        "memory_namespaces",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, index=True),
        sa.Column("description", sa.Text, server_default=""),
        # Versioning columns
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "previous_version_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("config_overrides", postgresql.JSONB, server_default="{}"),
        sa.Column("sync_checkpoints", postgresql.JSONB, server_default="{}"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("workspace_id", "slug", "version", name="uq_namespace_workspace_slug_version"),
    )

    # Partial index for active namespaces
    op.create_index(
        "idx_namespace_active",
        "memory_namespaces",
        ["workspace_id", "slug"],
        postgresql_where=sa.text("is_active = true"),
    )

    # Documents table
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("pending", "processing", "completed", "failed", name="document_status", create_type=False),
            server_default="pending",
            index=True,
        ),
        sa.Column("source", sa.String(1024), server_default=""),
        sa.Column("source_type", sa.String(64), server_default=""),
        sa.Column("content_type", sa.String(128), server_default=""),
        sa.Column("title", sa.String(512), server_default=""),
        sa.Column("author", sa.String(255), server_default=""),
        sa.Column("language", sa.String(10), server_default="en"),
        sa.Column("checksum", sa.String(64), server_default="", index=True),
        sa.Column("size_bytes", sa.Integer, server_default="0"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("entity_count", sa.Integer, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index("ix_documents_namespace_checksum", "documents", ["namespace_id", "checksum"])

    # Chunks table
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("chunk_index", sa.Integer, server_default="0"),
        sa.Column("start_char", sa.Integer, server_default="0"),
        sa.Column("end_char", sa.Integer, server_default="0"),
        sa.Column("token_count", sa.Integer, server_default="0"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("embedding_model", sa.String(128), server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index("ix_chunks_document_index", "chunks", ["document_id", "chunk_index"])

    # Entities table
    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(512), nullable=False, index=True),
        sa.Column(
            "entity_type",
            postgresql.ENUM(
                "PERSON",
                "ORGANIZATION",
                "LOCATION",
                "EVENT",
                "CONCEPT",
                "PRODUCT",
                "TECHNOLOGY",
                "DOCUMENT",
                "PROJECT",
                "TASK",
                "MEETING",
                "DECISION",
                "METRIC",
                "GOAL",
                "CUSTOM",
                name="entity_type",
                create_type=False,
            ),
            server_default="CONCEPT",
            index=True,
        ),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("attributes", postgresql.JSONB, server_default="{}"),
        sa.Column("source_document_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=False)), server_default="{}"),
        sa.Column("source_chunk_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=False)), server_default="{}"),
        sa.Column("mention_count", sa.Integer, server_default="1"),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("embedding_model", sa.String(128), server_default=""),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index("ix_entities_namespace_name_type", "entities", ["namespace_id", "name", "entity_type"])

    # Relationships table
    op.create_table(
        "relationships",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "source_entity_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "target_entity_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "relationship_type",
            postgresql.ENUM(
                "RELATES_TO",
                "WORKS_FOR",
                "WORKS_WITH",
                "MANAGES",
                "REPORTS_TO",
                "OWNS",
                "CREATED",
                "MODIFIED",
                "MENTIONED_IN",
                "DISCUSSED_IN",
                "PARTICIPATED_IN",
                "ATTENDED",
                "DECIDED",
                "ASSIGNED_TO",
                "BLOCKED_BY",
                "DEPENDS_ON",
                "PART_OF",
                "LOCATED_IN",
                "OCCURRED_AT",
                "CAUSED",
                "INFLUENCED",
                "SIMILAR_TO",
                "OPPOSITE_OF",
                "DERIVED_FROM",
                "CUSTOM",
                name="relationship_type",
                create_type=False,
            ),
            server_default="RELATES_TO",
            index=True,
        ),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("properties", postgresql.JSONB, server_default="{}"),
        sa.Column("source_document_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=False)), server_default="{}"),
        sa.Column("source_chunk_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=False)), server_default="{}"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("weight", sa.Float, server_default="1.0"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index("ix_relationships_entities", "relationships", ["source_entity_id", "target_entity_id"])

    # Episodes table
    op.create_table(
        "episodes",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column("duration_seconds", sa.Integer, nullable=True),
        sa.Column("entity_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=False)), server_default="{}"),
        sa.Column("source_document_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=False)), server_default="{}"),
        sa.Column("source_chunk_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=False)), server_default="{}"),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("embedding_model", sa.String(128), server_default=""),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Memory events table
    op.create_table(
        "memory_events",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "event_type",
            postgresql.ENUM(
                "DOCUMENT_CREATED",
                "DOCUMENT_UPDATED",
                "DOCUMENT_DELETED",
                "DOCUMENT_PROCESSED",
                "ENTITY_CREATED",
                "ENTITY_UPDATED",
                "ENTITY_MERGED",
                "ENTITY_DELETED",
                "RELATIONSHIP_CREATED",
                "RELATIONSHIP_UPDATED",
                "RELATIONSHIP_DELETED",
                "EPISODE_CREATED",
                "EPISODE_UPDATED",
                "EPISODE_DELETED",
                "MEMORY_QUERIED",
                "MEMORY_RECALLED",
                "MEMORY_CONSOLIDATED",
                "NAMESPACE_CREATED",
                "NAMESPACE_UPDATED",
                "NAMESPACE_DELETED",
                name="event_type",
                create_type=False,
            ),
            nullable=False,
            index=True,
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column("resource_type", sa.String(64), nullable=False, index=True),
        sa.Column("resource_id", postgresql.UUID(as_uuid=False), nullable=False, index=True),
        sa.Column("data", postgresql.JSONB, server_default="{}"),
        sa.Column("previous_data", postgresql.JSONB, nullable=True),
        sa.Column("actor_id", sa.String(255), nullable=True),
        sa.Column("actor_type", sa.String(64), server_default="system"),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=False), nullable=True, index=True),
        sa.Column("version", sa.Integer, server_default="1"),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
    )

    op.create_index("ix_events_resource", "memory_events", ["resource_type", "resource_id"])
    op.create_index("ix_events_namespace_timestamp", "memory_events", ["namespace_id", "timestamp"])

    # Permissions table
    op.create_table(
        "permissions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("principal_type", sa.String(64), nullable=False, index=True),
        sa.Column("principal_id", sa.String(255), nullable=False, index=True),
        sa.Column("resource_type", sa.String(64), nullable=False, index=True),
        sa.Column("resource_id", postgresql.UUID(as_uuid=False), nullable=False, index=True),
        sa.Column("permission", sa.String(64), nullable=False),
        sa.Column("inherited_from_type", sa.String(64), nullable=True),
        sa.Column("inherited_from_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "principal_type", "principal_id", "resource_type", "resource_id", "permission", name="uq_permission"
        ),
    )

    op.create_index("ix_permissions_principal", "permissions", ["principal_type", "principal_id"])
    op.create_index("ix_permissions_resource", "permissions", ["resource_type", "resource_id"])

    # Sync checkpoints table
    op.create_table(
        "sync_checkpoints",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("source", sa.String(255), nullable=False, index=True),
        sa.Column("checkpoint", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("namespace_id", "source", name="uq_sync_checkpoint_namespace_source"),
    )

    # Expertise definitions table
    op.create_table(
        "expertise_definitions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("version", sa.String(32), server_default="1.0.0"),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("config", postgresql.JSONB, server_default="{}"),
        sa.Column("is_active", sa.Boolean, server_default="true", index=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("namespace_id", "name", name="uq_expertise_namespace_name"),
    )

    op.create_index("ix_expertise_namespace_active", "expertise_definitions", ["namespace_id", "is_active"])


def downgrade() -> None:
    """Drop all tables and types."""
    # Drop tables in reverse order (respect foreign keys)
    op.drop_table("expertise_definitions")
    op.drop_table("sync_checkpoints")
    op.drop_table("permissions")
    op.drop_table("memory_events")
    op.drop_table("episodes")
    op.drop_table("relationships")
    op.drop_table("entities")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("memory_namespaces")
    op.drop_table("workspaces")
    op.drop_table("organizations")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS event_type")
    op.execute("DROP TYPE IF EXISTS relationship_type")
    op.execute("DROP TYPE IF EXISTS entity_type")
    op.execute("DROP TYPE IF EXISTS document_status")
    op.execute("DROP TYPE IF EXISTS tenancy_mode")
