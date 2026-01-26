"""SQLAlchemy models for Khora Memory Lake.

This module defines the complete database schema including:
- Multi-tenancy (organizations, workspaces, namespaces)
- Documents and chunks with vector embeddings
- Entities, relationships, and episodes
- Event sourcing log
- ACL and permissions
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from khora.core.models.document import DocumentStatus
from khora.core.models.entity import EntityType, RelationshipType
from khora.core.models.event import EventType
from khora.core.models.tenancy import TenancyMode


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
        list[str]: ARRAY(String),
    }


# =============================================================================
# Multi-Tenancy Models
# =============================================================================


class OrganizationModel(Base):
    """Organization - top-level tenant."""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    tenancy_mode: Mapped[str] = mapped_column(
        Enum(TenancyMode, name="tenancy_mode", create_constraint=True),
        default=TenancyMode.SHARED,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    workspaces: Mapped[list[WorkspaceModel]] = relationship("WorkspaceModel", back_populates="organization")

    def __repr__(self) -> str:
        return f"<Organization(id={self.id!r}, name={self.name!r})>"


class WorkspaceModel(Base):
    """Workspace within an organization."""

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    organization: Mapped[OrganizationModel] = relationship("OrganizationModel", back_populates="workspaces")
    namespaces: Mapped[list[MemoryNamespaceModel]] = relationship("MemoryNamespaceModel", back_populates="workspace")

    __table_args__ = (UniqueConstraint("organization_id", "slug", name="uq_workspace_org_slug"),)

    def __repr__(self) -> str:
        return f"<Workspace(id={self.id!r}, name={self.name!r})>"


class MemoryNamespaceModel(Base):
    """Memory namespace for isolating memories."""

    __tablename__ = "memory_namespaces"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    workspace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    config_overrides: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    sync_checkpoints: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    workspace: Mapped[WorkspaceModel] = relationship("WorkspaceModel", back_populates="namespaces")
    documents: Mapped[list[DocumentModel]] = relationship("DocumentModel", back_populates="namespace")
    chunks: Mapped[list[ChunkModel]] = relationship("ChunkModel", back_populates="namespace")
    entities: Mapped[list[EntityModel]] = relationship("EntityModel", back_populates="namespace")
    relationships: Mapped[list[RelationshipModel]] = relationship("RelationshipModel", back_populates="namespace")
    episodes: Mapped[list[EpisodeModel]] = relationship("EpisodeModel", back_populates="namespace")
    events: Mapped[list[MemoryEventModel]] = relationship("MemoryEventModel", back_populates="namespace")
    expertise_definitions: Mapped[list[ExpertiseDefinitionModel]] = relationship(
        "ExpertiseDefinitionModel", back_populates="namespace"
    )

    __table_args__ = (UniqueConstraint("workspace_id", "slug", name="uq_namespace_workspace_slug"),)

    def __repr__(self) -> str:
        return f"<MemoryNamespace(id={self.id!r}, name={self.name!r})>"


# =============================================================================
# Document Models
# =============================================================================


class DocumentModel(Base):
    """Document to be processed and stored."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    namespace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(DocumentStatus, name="document_status", create_constraint=True),
        default=DocumentStatus.PENDING,
        index=True,
    )

    # Metadata
    source: Mapped[str] = mapped_column(String(1024), default="")
    source_type: Mapped[str] = mapped_column(String(64), default="")
    content_type: Mapped[str] = mapped_column(String(128), default="")
    title: Mapped[str] = mapped_column(String(512), default="")
    author: Mapped[str] = mapped_column(String(255), default="")
    language: Mapped[str] = mapped_column(String(10), default="en")
    checksum: Mapped[str] = mapped_column(String(64), default="", index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    # Processing info
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel", back_populates="documents")
    chunks: Mapped[list[ChunkModel]] = relationship(
        "ChunkModel", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_documents_namespace_checksum", "namespace_id", "checksum"),)

    def __repr__(self) -> str:
        return f"<Document(id={self.id!r}, title={self.title!r})>"


class ChunkModel(Base):
    """Chunk of text from a document with embedding."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    namespace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Chunk metadata
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    start_char: Mapped[int] = mapped_column(Integer, default=0)
    end_char: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    # Embedding (pgvector)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128), default="")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel", back_populates="chunks")
    document: Mapped[DocumentModel] = relationship("DocumentModel", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
        # Vector similarity index (using IVFFlat for approximate nearest neighbor)
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id!r}, document_id={self.document_id!r}, index={self.chunk_index})>"


# =============================================================================
# Entity Models
# =============================================================================


class EntityModel(Base):
    """Entity extracted from documents (stored in both PostgreSQL and Neo4j)."""

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    namespace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(
        Enum(EntityType, name="entity_type", create_constraint=True),
        default=EntityType.CONCEPT,
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, default="")

    # Attributes and sources
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source_document_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), default=list)
    source_chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), default=list)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)

    # Embedding for entity similarity
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128), default="")

    # Temporal validity
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Confidence
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel", back_populates="entities")

    __table_args__ = (
        Index("ix_entities_namespace_name_type", "namespace_id", "name", "entity_type"),
        Index(
            "ix_entities_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Entity(id={self.id!r}, name={self.name!r}, type={self.entity_type})>"


class RelationshipModel(Base):
    """Relationship between entities (stored in both PostgreSQL and Neo4j)."""

    __tablename__ = "relationships"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    namespace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_entity_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_entity_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relationship_type: Mapped[str] = mapped_column(
        Enum(RelationshipType, name="relationship_type", create_constraint=True),
        default=RelationshipType.RELATES_TO,
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, default="")

    # Properties and sources
    properties: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source_document_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), default=list)
    source_chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), default=list)

    # Temporal validity
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Confidence and weight
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel", back_populates="relationships")

    __table_args__ = (Index("ix_relationships_entities", "source_entity_id", "target_entity_id"),)

    def __repr__(self) -> str:
        return f"<Relationship(id={self.id!r}, type={self.relationship_type})>"


class EpisodeModel(Base):
    """Episodic memory representing a temporal event."""

    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    namespace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")

    # Temporal bounds
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Associated entities
    entity_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), default=list)

    # Sources
    source_document_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), default=list)
    source_chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), default=list)

    # Embedding
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128), default="")

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel", back_populates="episodes")

    def __repr__(self) -> str:
        return f"<Episode(id={self.id!r}, name={self.name!r})>"


# =============================================================================
# Event Sourcing Model
# =============================================================================


class MemoryEventModel(Base):
    """Immutable event log for event sourcing."""

    __tablename__ = "memory_events"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    namespace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(
        Enum(EventType, name="event_type", create_constraint=True),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)

    # Resource reference
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False, index=True)

    # Event data
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    previous_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Actor info
    actor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    actor_type: Mapped[str] = mapped_column(String(64), default="system")

    # Correlation
    correlation_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True, index=True)

    # Version
    version: Mapped[int] = mapped_column(Integer, default=1)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel", back_populates="events")

    __table_args__ = (
        Index("ix_events_resource", "resource_type", "resource_id"),
        Index("ix_events_namespace_timestamp", "namespace_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<MemoryEvent(id={self.id!r}, type={self.event_type})>"


# =============================================================================
# ACL / Permissions Model
# =============================================================================


class PermissionModel(Base):
    """Permission entry for ACL-based access control."""

    __tablename__ = "permissions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))

    # The principal (who has the permission)
    principal_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # user, role, api_key
    principal_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # The resource (what the permission applies to)
    resource_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )  # organization, workspace, namespace
    resource_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False, index=True)

    # The permission level
    permission: Mapped[str] = mapped_column(String(64), nullable=False)  # read, write, admin, owner

    # Inheritance
    inherited_from_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    inherited_from_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        UniqueConstraint(
            "principal_type", "principal_id", "resource_type", "resource_id", "permission", name="uq_permission"
        ),
        Index("ix_permissions_principal", "principal_type", "principal_id"),
        Index("ix_permissions_resource", "resource_type", "resource_id"),
    )

    def __repr__(self) -> str:
        return f"<Permission({self.principal_type}:{self.principal_id} -> {self.resource_type}:{self.resource_id} = {self.permission})>"


# =============================================================================
# Sync Checkpoint Model (for incremental updates)
# =============================================================================


class SyncCheckpointModel(Base):
    """Sync checkpoint for tracking incremental updates from external sources."""

    __tablename__ = "sync_checkpoints"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    namespace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(255), nullable=False, index=True)  # e.g., "github", "notion", "slack"
    checkpoint: Mapped[str] = mapped_column(Text, nullable=False)  # Source-specific checkpoint value
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    __table_args__ = (UniqueConstraint("namespace_id", "source", name="uq_sync_checkpoint_namespace_source"),)

    def __repr__(self) -> str:
        return f"<SyncCheckpoint(namespace_id={self.namespace_id!r}, source={self.source!r})>"


# =============================================================================
# Expertise Definition Model
# =============================================================================


class ExpertiseDefinitionModel(Base):
    """Stored expertise configuration for a namespace.

    Allows namespaces to have custom expertise definitions that control
    entity extraction, relationship types, correlation rules, and inference rules.
    """

    __tablename__ = "expertise_definitions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    namespace_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    description: Mapped[str] = mapped_column(Text, default="")

    # The full expertise configuration as JSONB
    # Contains: entity_types, relationship_types, tool_schemas, correlation_rules,
    # inference_rules, confidence, expansion, system_prompt, extraction_prompt
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Status
    is_active: Mapped[bool] = mapped_column(default=True, index=True)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship(
        "MemoryNamespaceModel", back_populates="expertise_definitions"
    )

    __table_args__ = (
        UniqueConstraint("namespace_id", "name", name="uq_expertise_namespace_name"),
        Index("ix_expertise_namespace_active", "namespace_id", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<ExpertiseDefinition(id={self.id!r}, name={self.name!r}, version={self.version!r})>"
