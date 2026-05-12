"""Initial database schema.

Revision ID: 000_initial_schema
Revises:
Create Date: 2026-01-27

Creates all base tables for Khora:
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
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "000_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# =============================================================================
# Dialect-aware type helpers
# =============================================================================
# On PostgreSQL we use the full feature set (JSONB, ARRAY, pgvector, ENUM types).
# On SQLite the Postgres-only types are substituted with portable equivalents:
# - JSONB  -> JSON (TEXT-backed; SQLAlchemy maps JSON to TEXT on SQLite)
# - ARRAY  -> JSON (stored as JSON array in TEXT)
# - ENUM   -> VARCHAR(64) with CHECK constraint implied at ORM layer
# - Vector -> column is omitted; LanceDB owns embedding storage for sqlite_lance
#
# This keeps functional parity for khora operations. Embeddings on SQLite
# live in a separate LanceDB table keyed by chunk/entity id (DYT-2728+).


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _uuid_col_type():
    """UUID column type — native uuid on Postgres, TEXT on SQLite."""
    return postgresql.UUID(as_uuid=False) if _is_postgres() else sa.String(36)


def _jsonb_type():
    """JSONB on Postgres, JSON (TEXT) on SQLite."""
    return postgresql.JSONB if _is_postgres() else sa.JSON


def _uuid_array_type():
    """Array of UUIDs on Postgres, JSON on SQLite."""
    return postgresql.ARRAY(postgresql.UUID(as_uuid=False)) if _is_postgres() else sa.JSON


def _jsonb_default():
    """server_default for JSONB/JSON columns."""
    return "{}" if _is_postgres() else sa.text("'{}'")


def _uuid_array_default():
    """server_default for UUID array columns (empty array)."""
    return "{}" if _is_postgres() else sa.text("'[]'")


def _enum_col(*values: str, name: str, default: str | None = None):
    """ENUM on Postgres (via CREATE TYPE), VARCHAR(64) on SQLite."""
    if _is_postgres():
        col_type = postgresql.ENUM(*values, name=name, create_type=False)
    else:
        col_type = sa.String(64)
    kwargs: dict = {}
    if default is not None:
        kwargs["server_default"] = default
    return col_type, kwargs


def upgrade() -> None:
    """Create initial schema."""
    is_postgres = _is_postgres()

    # Create enum types (Postgres only — SQLite uses VARCHAR)
    if is_postgres:
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

    uuid_t = _uuid_col_type()
    jsonb_t = _jsonb_type()
    uuid_arr_t = _uuid_array_type()
    jsonb_default = _jsonb_default()
    uuid_arr_default = _uuid_array_default()

    tenancy_col, tenancy_kwargs = _enum_col("shared", "isolated", name="tenancy_mode", default="shared")

    # Organizations table
    op.create_table(
        "organizations",
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("tenancy_mode", tenancy_col, **tenancy_kwargs),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Workspaces table
    op.create_table(
        "workspaces",
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "organization_id",
            uuid_t,
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, index=True),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("organization_id", "slug", name="uq_workspace_org_slug"),
    )

    # Memory namespaces table (with versioning columns)
    op.create_table(
        "memory_namespaces",
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "workspace_id",
            uuid_t,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, index=True),
        sa.Column("description", sa.Text, server_default=""),
        # Versioning columns
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "is_active",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("true") if is_postgres else sa.text("1"),
        ),
        sa.Column(
            "previous_version_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("config_overrides", jsonb_t, server_default=jsonb_default),
        sa.Column("sync_checkpoints", jsonb_t, server_default=jsonb_default),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("workspace_id", "slug", "version", name="uq_namespace_workspace_slug_version"),
    )

    # Partial index for active namespaces
    if is_postgres:
        op.create_index(
            "idx_namespace_active",
            "memory_namespaces",
            ["workspace_id", "slug"],
            postgresql_where=sa.text("is_active = true"),
        )
    else:
        # SQLite supports partial indexes too, via sqlite_where.
        op.create_index(
            "idx_namespace_active",
            "memory_namespaces",
            ["workspace_id", "slug"],
            sqlite_where=sa.text("is_active = 1"),
        )

    doc_status_col, doc_status_kwargs = _enum_col(
        "pending", "processing", "completed", "failed", name="document_status", default="pending"
    )

    # Documents table
    doc_columns = [
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("status", doc_status_col, index=True, **doc_status_kwargs),
        sa.Column("source", sa.String(1024), server_default=""),
        sa.Column("source_type", sa.String(64), server_default=""),
        sa.Column("content_type", sa.String(128), server_default=""),
        sa.Column("title", sa.String(512), server_default=""),
        sa.Column("author", sa.String(255), server_default=""),
        sa.Column("language", sa.String(10), server_default="en"),
        sa.Column("checksum", sa.String(64), server_default="", index=True),
        sa.Column("size_bytes", sa.Integer, server_default="0"),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("entity_count", sa.Integer, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    ]
    op.create_table("documents", *doc_columns)

    op.create_index("ix_documents_namespace_checksum", "documents", ["namespace_id", "checksum"])

    # Chunks table — embedding column is Postgres-only (pgvector). On SQLite,
    # LanceDB holds embeddings in a separate table keyed by chunk id.
    chunk_columns = [
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "document_id",
            uuid_t,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("chunk_index", sa.Integer, server_default="0"),
        sa.Column("start_char", sa.Integer, server_default="0"),
        sa.Column("end_char", sa.Integer, server_default="0"),
        sa.Column("token_count", sa.Integer, server_default="0"),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("embedding_model", sa.String(128), server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    ]
    if is_postgres:
        # Insert embedding column in its original position for Postgres schema parity.
        chunk_columns.insert(-2, sa.Column("embedding", Vector(1536), nullable=True))
    op.create_table("chunks", *chunk_columns)

    op.create_index("ix_chunks_document_index", "chunks", ["document_id", "chunk_index"])

    entity_type_col, entity_type_kwargs = _enum_col(
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
        default="CONCEPT",
    )

    # Entities table — embedding column is Postgres-only (pgvector).
    entity_columns = [
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(512), nullable=False, index=True),
        sa.Column("entity_type", entity_type_col, index=True, **entity_type_kwargs),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("attributes", jsonb_t, server_default=jsonb_default),
        sa.Column("source_document_ids", uuid_arr_t, server_default=uuid_arr_default),
        sa.Column("source_chunk_ids", uuid_arr_t, server_default=uuid_arr_default),
        sa.Column("mention_count", sa.Integer, server_default="1"),
        sa.Column("embedding_model", sa.String(128), server_default=""),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    ]
    if is_postgres:
        entity_columns.insert(-5, sa.Column("embedding", Vector(1536), nullable=True))
    op.create_table("entities", *entity_columns)

    op.create_index("ix_entities_namespace_name_type", "entities", ["namespace_id", "name", "entity_type"])

    rel_type_col, rel_type_kwargs = _enum_col(
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
        default="RELATES_TO",
    )

    # Relationships table
    op.create_table(
        "relationships",
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
        sa.Column("relationship_type", rel_type_col, index=True, **rel_type_kwargs),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("properties", jsonb_t, server_default=jsonb_default),
        sa.Column("source_document_ids", uuid_arr_t, server_default=uuid_arr_default),
        sa.Column("source_chunk_ids", uuid_arr_t, server_default=uuid_arr_default),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("weight", sa.Float, server_default="1.0"),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_index("ix_relationships_entities", "relationships", ["source_entity_id", "target_entity_id"])

    # Episodes table — embedding column is Postgres-only (pgvector).
    episode_columns = [
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column("duration_seconds", sa.Integer, nullable=True),
        sa.Column("entity_ids", uuid_arr_t, server_default=uuid_arr_default),
        sa.Column("source_document_ids", uuid_arr_t, server_default=uuid_arr_default),
        sa.Column("source_chunk_ids", uuid_arr_t, server_default=uuid_arr_default),
        sa.Column("embedding_model", sa.String(128), server_default=""),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    ]
    if is_postgres:
        episode_columns.insert(-3, sa.Column("embedding", Vector(1536), nullable=True))
    op.create_table("episodes", *episode_columns)

    event_type_col, event_type_kwargs = _enum_col(
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
    )

    # Memory events table
    op.create_table(
        "memory_events",
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("event_type", event_type_col, nullable=False, index=True, **event_type_kwargs),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column("resource_type", sa.String(64), nullable=False, index=True),
        sa.Column("resource_id", uuid_t, nullable=False, index=True),
        sa.Column("data", jsonb_t, server_default=jsonb_default),
        sa.Column("previous_data", jsonb_t, nullable=True),
        sa.Column("actor_id", sa.String(255), nullable=True),
        sa.Column("actor_type", sa.String(64), server_default="system"),
        sa.Column("correlation_id", uuid_t, nullable=True, index=True),
        sa.Column("version", sa.Integer, server_default="1"),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
    )

    op.create_index("ix_events_resource", "memory_events", ["resource_type", "resource_id"])
    op.create_index("ix_events_namespace_timestamp", "memory_events", ["namespace_id", "timestamp"])

    # Permissions table
    op.create_table(
        "permissions",
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column("principal_type", sa.String(64), nullable=False, index=True),
        sa.Column("principal_id", sa.String(255), nullable=False, index=True),
        sa.Column("resource_type", sa.String(64), nullable=False, index=True),
        sa.Column("resource_id", uuid_t, nullable=False, index=True),
        sa.Column("permission", sa.String(64), nullable=False),
        sa.Column("inherited_from_type", sa.String(64), nullable=True),
        sa.Column("inherited_from_id", uuid_t, nullable=True),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
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
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("source", sa.String(255), nullable=False, index=True),
        sa.Column("checkpoint", sa.Text, nullable=False),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("namespace_id", "source", name="uq_sync_checkpoint_namespace_source"),
    )

    # Expertise definitions table
    op.create_table(
        "expertise_definitions",
        sa.Column("id", uuid_t, primary_key=True),
        sa.Column(
            "namespace_id",
            uuid_t,
            sa.ForeignKey("memory_namespaces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("version", sa.String(32), server_default="1.0.0"),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("config", jsonb_t, server_default=jsonb_default),
        sa.Column(
            "is_active",
            sa.Boolean,
            server_default=sa.text("true") if is_postgres else sa.text("1"),
            index=True,
        ),
        sa.Column("metadata", jsonb_t, server_default=jsonb_default),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("namespace_id", "name", name="uq_expertise_namespace_name"),
    )

    op.create_index("ix_expertise_namespace_active", "expertise_definitions", ["namespace_id", "is_active"])


def downgrade() -> None:
    """Drop all tables and types."""
    is_postgres = _is_postgres()

    def _drop(table: str) -> None:
        # On SQLite (where migration 010's downgrade is a no-op), workspaces
        # and organizations may have already been dropped during upgrade.
        if is_postgres:
            op.drop_table(table)
        else:
            op.execute(f"DROP TABLE IF EXISTS {table}")

    # Drop tables in reverse order (respect foreign keys)
    _drop("expertise_definitions")
    _drop("sync_checkpoints")
    _drop("permissions")
    _drop("memory_events")
    _drop("episodes")
    _drop("relationships")
    _drop("entities")
    _drop("chunks")
    _drop("documents")
    _drop("memory_namespaces")
    _drop("workspaces")
    _drop("organizations")

    # Drop enum types (Postgres only)
    if _is_postgres():
        op.execute("DROP TYPE IF EXISTS event_type")
        op.execute("DROP TYPE IF EXISTS relationship_type")
        op.execute("DROP TYPE IF EXISTS entity_type")
        op.execute("DROP TYPE IF EXISTS document_status")
        op.execute("DROP TYPE IF EXISTS tenancy_mode")
