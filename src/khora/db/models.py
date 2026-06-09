"""SQLAlchemy models for Khora.

This module defines the complete database schema including:
- Multi-tenancy (namespaces)
- Documents and chunks with vector embeddings
- Entities, relationships, and episodes
- Event sourcing log
- ACL and permissions
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID as UUIDType
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from khora.core.models.document import DocumentStatus
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


class MemoryNamespaceModel(Base):
    """Memory namespace for isolating memories.

    Namespace is the sole data isolation boundary.

    Supports versioning for data replacement workflows:
    - version: Incremental version number (starts at 1)
    - is_active: Whether this is the current active version
    """

    __tablename__ = "memory_namespaces"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    # Stable ID shared across all versions of a namespace (for external references)
    namespace_id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    tenancy_mode: Mapped[str] = mapped_column(
        Enum(TenancyMode, name="tenancy_mode", create_constraint=True, values_callable=lambda e: [m.value for m in e]),
        default=TenancyMode.SHARED,
    )

    # Versioning fields
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config_overrides: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    sync_checkpoints: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    documents: Mapped[list[DocumentModel]] = relationship("DocumentModel", back_populates="namespace")
    chunks: Mapped[list[ChunkModel]] = relationship("ChunkModel", back_populates="namespace")
    entities: Mapped[list[EntityModel]] = relationship("EntityModel", back_populates="namespace")
    relationships: Mapped[list[RelationshipModel]] = relationship("RelationshipModel", back_populates="namespace")
    episodes: Mapped[list[EpisodeModel]] = relationship("EpisodeModel", back_populates="namespace")
    events: Mapped[list[MemoryEventModel]] = relationship("MemoryEventModel", back_populates="namespace")
    expertise_definitions: Mapped[list[ExpertiseDefinitionModel]] = relationship(
        "ExpertiseDefinitionModel", back_populates="namespace"
    )

    __table_args__ = (
        UniqueConstraint("namespace_id", "version", name="uq_namespace_stable_id_version"),
        Index(
            "idx_namespace_stable_active",
            "namespace_id",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
    )

    def __repr__(self) -> str:
        return f"<MemoryNamespace(id={self.id!r})>"


# =============================================================================
# Document Models
# =============================================================================


class DocumentModel(Base):
    """Document to be processed and stored."""

    __tablename__ = "documents"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            create_constraint=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        default=DocumentStatus.PENDING,
        index=True,
    )

    # Metadata
    source: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, default="library")
    source_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    external_id: Mapped[str | None] = mapped_column(String(512), nullable=True, default=None)

    # Processing info
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)
    relationship_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Ontology-aware re-extraction
    extraction_config_hash: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    # Extraction parameters for deferred/crash-recovery processing.
    extraction_params: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Session attribution for agentic-framework adapters (#620).
    # Nullable — populated when the caller passes ``session_id`` to remember/submit_batch.
    # Indexed via migration 031 (Postgres-only); see docs/migrations.md.
    session_id: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel", back_populates="documents")
    chunks: Mapped[list[ChunkModel]] = relationship(
        "ChunkModel", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_namespace_checksum", "namespace_id", "checksum"),
        Index(
            "ix_documents_namespace_checksum_active",
            "namespace_id",
            "checksum",
            postgresql_where=text("status != 'failed'"),
        ),
        Index("ix_documents_namespace_source_type", "namespace_id", "source_type"),
        Index("ix_documents_namespace_created_at", "namespace_id", "created_at"),
        Index(
            "ix_documents_namespace_external_id_unique",
            "namespace_id",
            "external_id",
            unique=True,
            postgresql_where=text("external_id IS NOT NULL"),
        ),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id!r}, title={self.title!r})>"


class ChunkModel(Base):
    """Chunk of text from a document with embedding."""

    __tablename__ = "chunks"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Chunk metadata
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    start_char: Mapped[int] = mapped_column(Integer, default=0)
    end_char: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    chunker_info: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        # Portable JSON literal — JSONB on Postgres accepts plain '{}' without the
        # explicit ``::jsonb`` cast (implicit cast on column type), and SQLite's
        # JSON alias accepts the same literal. Keeps any code path that calls
        # ``Base.metadata.create_all()`` (deprecated but still callable) emitting
        # valid DDL on the sqlite_lance fixture.
        server_default=text("'{}'"),
        default=dict,
    )

    # Embedding (pgvector)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128), default="")

    # Full-text search (generated tsvector column)
    content_tsv: Mapped[Any | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    source_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Reinforcement-on-recall (#855): NULL until first recall. When the
    # ``chronicle_enable_recall_reinforcement`` flag is set, the Chronicle
    # decay path treats ``max(source_timestamp, last_accessed_at)`` as the
    # effective event time so frequently-recalled chunks stay fresh.
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Real-world event time the chunk's content refers to, distinct from created_at.
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Session attribution for agentic-framework adapters (#620).
    # Inherited from the parent document at chunking time.
    session_id: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel", back_populates="chunks")
    document: Mapped[DocumentModel] = relationship("DocumentModel", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
        # Vector similarity index (HNSW for better recall)
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 24, "ef_construction": 128},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN index for full-text search
        Index(
            "ix_chunks_content_tsv",
            "content_tsv",
            postgresql_using="gin",
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

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(
        String(64),
        default="CONCEPT",
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, default="")

    # Attributes and sources
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source_document_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    source_chunk_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
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
        UniqueConstraint("namespace_id", "name", "entity_type", name="uq_entities_namespace_name_type"),
        Index(
            "ix_entities_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 24, "ef_construction": 128},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_entities_namespace_mentions", "namespace_id", mention_count.desc()),
        # Partial indexes for temporal filtering (only rows with non-NULL values)
        Index("ix_entities_valid_from", "valid_from", postgresql_where="valid_from IS NOT NULL"),
        Index("ix_entities_valid_until", "valid_until", postgresql_where="valid_until IS NOT NULL"),
    )

    def __repr__(self) -> str:
        return f"<Entity(id={self.id!r}, name={self.name!r}, type={self.entity_type})>"


class RelationshipModel(Base):
    """Relationship between entities (stored in Neo4j; Postgres table exists but is not actively written)."""

    __tablename__ = "relationships"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_entity_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_entity_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relationship_type: Mapped[str] = mapped_column(
        String(64),
        default="RELATES_TO",
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, default="")

    # Properties and sources
    properties: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source_document_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    source_chunk_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)

    # Temporal validity
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Bi-temporal soft-delete (migration 033, dream-phase Phase 0.3, #653).
    # Populated by Phase 4 apply-mode dream runs in v0.15; NULL = still valid.
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_by: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True)

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

    __table_args__ = (
        Index("ix_relationships_entities", "source_entity_id", "target_entity_id"),
        Index("ix_relationships_namespace_type", "namespace_id", "relationship_type"),
        Index("ix_relationships_target_source", "target_entity_id", "source_entity_id"),
    )

    def __repr__(self) -> str:
        return f"<Relationship(id={self.id!r}, type={self.relationship_type})>"


class EpisodeModel(Base):
    """Episodic memory representing a temporal event."""

    __tablename__ = "episodes"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")

    # Temporal bounds
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Associated entities
    entity_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)

    # Sources
    source_document_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    source_chunk_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)

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

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(
        Enum(EventType, name="event_type", create_constraint=True, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)

    # Resource reference
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    # Event data
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    previous_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Actor info
    actor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    actor_type: Mapped[str] = mapped_column(String(64), default="system")

    # Correlation
    correlation_id: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)

    # Version
    version: Mapped[int] = mapped_column(Integer, default=1)

    # Session attribution for agentic-framework adapters (#620).
    session_id: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True)

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

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # The principal (who has the permission)
    principal_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # user, role, api_key
    principal_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # The resource (what the permission applies to)
    # resource_type must always be 'namespace' (sole isolation boundary).
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    # The permission level
    permission: Mapped[str] = mapped_column(String(64), nullable=False)  # read, write, admin, owner

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

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
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

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
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


# =============================================================================
# Temporal Models for Khora Engine
# =============================================================================


class TimeGranularity:
    """Time granularity levels for hierarchical time graph."""

    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"


class TimeNodeModel(Base):
    """Hierarchical time graph node for temporal navigation.

    Enables efficient temporal queries by organizing time into a hierarchy:
    Year → Quarter → Month → Week → Day

    Each node can have a summary embedding for temporal range queries.
    """

    __tablename__ = "time_nodes"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Time hierarchy
    granularity: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # year, quarter, month, week, day
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    parent_id: Mapped[UUIDType | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("time_nodes.id", ondelete="CASCADE"), nullable=True, index=True
    )

    # Display name (e.g., "2024", "Q1 2024", "January 2024", "Week 1 2024", "2024-01-15")
    name: Mapped[str] = mapped_column(String(64), nullable=False)

    # Temporal summary for range queries
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    # Stats for optimization
    edge_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Self-referential relationship for hierarchy
    parent: Mapped[TimeNodeModel | None] = relationship("TimeNodeModel", remote_side=[id], backref="children")

    __table_args__ = (
        # Unique constraint: one node per namespace/granularity/start_time
        UniqueConstraint("namespace_id", "granularity", "start_time", name="uq_time_node_namespace_granularity_start"),
        # Index for range queries
        Index("ix_time_nodes_namespace_range", "namespace_id", "start_time", "end_time"),
    )

    def __repr__(self) -> str:
        return f"<TimeNode(id={self.id!r}, name={self.name!r}, granularity={self.granularity})>"


class TemporalEdgeModel(Base):
    """Temporal relationship edge with explicit timestamps.

    Unlike RelationshipModel which collapses multiple observations into one edge,
    TemporalEdgeModel stores each timestamped observation as a distinct edge.
    This enables precise temporal queries like "who worked with whom in Q1 2024?"
    """

    __tablename__ = "temporal_edges"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_entity_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_entity_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relationship_type: Mapped[str] = mapped_column(String(64), default="RELATES_TO", index=True)
    description: Mapped[str] = mapped_column(Text, default="")

    # Bi-temporal model: when did this happen vs when did we learn about it
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Temporal validity window (when is this fact considered true)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Edge invalidation tracking
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    invalidated_by_id: Mapped[UUIDType | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("temporal_edges.id", ondelete="SET NULL"), nullable=True
    )
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Confidence and source tracking
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    properties: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source_document_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    source_chunk_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel")
    source_entity: Mapped[EntityModel] = relationship("EntityModel", foreign_keys=[source_entity_id])
    target_entity: Mapped[EntityModel] = relationship("EntityModel", foreign_keys=[target_entity_id])
    invalidated_by: Mapped[TemporalEdgeModel | None] = relationship("TemporalEdgeModel", remote_side=[id])

    __table_args__ = (
        # BRIN index for time-series optimization (99% space savings vs btree)
        Index("ix_temporal_edges_occurred_brin", "occurred_at", postgresql_using="brin"),
        # Composite index for entity pair + time queries
        Index(
            "ix_temporal_edges_entities_time",
            "source_entity_id",
            "target_entity_id",
            "occurred_at",
        ),
        # Index for validity queries
        Index("ix_temporal_edges_valid_range", "valid_from", "valid_until"),
    )

    def __repr__(self) -> str:
        return f"<TemporalEdge(id={self.id!r}, type={self.relationship_type}, occurred_at={self.occurred_at})>"


class TimeEdgeLinkModel(Base):
    """Link between time nodes and temporal edges for efficient time-based navigation.

    This join table enables queries like "give me all edges in January 2024"
    without scanning the entire temporal_edges table.
    """

    __tablename__ = "time_edge_links"

    time_node_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("time_nodes.id", ondelete="CASCADE"), primary_key=True
    )
    edge_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("temporal_edges.id", ondelete="CASCADE"), primary_key=True
    )

    # Relationships
    time_node: Mapped[TimeNodeModel] = relationship("TimeNodeModel")
    edge: Mapped[TemporalEdgeModel] = relationship("TemporalEdgeModel")

    def __repr__(self) -> str:
        return f"<TimeEdgeLink(time_node_id={self.time_node_id!r}, edge_id={self.edge_id!r})>"


# =============================================================================
# Chronicle Engine Models (events + atomic facts)
# =============================================================================


class ChronicleEventModel(Base):
    """Structured event extracted from chunks for the Chronicle engine.

    Events are SVO (subject-verb-object) tuples with bi-temporal information:
    ``observation_date`` (when ingested) and ``referenced_date`` (when the
    event occurred per source text). Used for high-precision temporal
    reasoning queries (Chronos pattern, 95.6% LongMemEval).
    """

    __tablename__ = "chronicle_events"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # SVO triple. ``object`` is reserved-ish in SQL grammars; the Python attr
    # is ``object_`` but the column is named ``object`` for natural querying.
    subject: Mapped[str] = mapped_column(String(512), nullable=False)
    verb: Mapped[str] = mapped_column(String(255), nullable=False)
    object_: Mapped[str | None] = mapped_column("object", String(512), nullable=True)

    # Bi-temporal timestamps
    observation_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    referenced_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    relative_offset: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Metadata
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Embedding for event-channel similarity (pgvector)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    # Session attribution for agentic-framework adapters (#620).
    session_id: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Bi-temporal soft-delete (migration 034, dream-phase Phase 4, #669).
    # Populated by Phase 4 apply-mode dream runs that soft-merge near-duplicate
    # events. NULL = still live. ``merged_into_event_id`` is a self-FK to the
    # canonical event the row was merged into; ``ON DELETE SET NULL`` so a hard
    # delete of a canonical row detaches its tails rather than cascading.
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_by: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    merged_into_event_id: Mapped[UUIDType | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "chronicle_events.id",
            name="fk_chronicle_events_merged_into_event_id",
            ondelete="SET NULL",
            use_alter=True,
        ),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel")

    __table_args__ = (
        Index("ix_chronicle_events_namespace_referenced_date", "namespace_id", "referenced_date"),
        Index("ix_chronicle_events_namespace_subject", "namespace_id", "subject"),
        Index(
            "ix_chronicle_events_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<ChronicleEvent(id={self.id!r}, {self.subject} {self.verb} {self.object_})>"


class MemoryFactModel(Base):
    """Atomic memory fact extracted from chunks for the Chronicle engine.

    Each fact is a self-contained SVO claim that can be independently
    verified, superseded, or deleted (EMem EDU pattern, 84.9% LongMemEval).
    Contradiction resolution sets ``is_active=False`` and points
    ``superseded_by`` at the replacement fact.
    """

    __tablename__ = "memory_facts"

    id: Mapped[UUIDType] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace_id: Mapped[UUIDType] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_namespaces.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Atomic SVO claim
    subject: Mapped[str] = mapped_column(String(512), nullable=False)
    predicate: Mapped[str] = mapped_column(String(255), nullable=False)
    object_: Mapped[str] = mapped_column("object", String(512), nullable=False)
    fact_text: Mapped[str] = mapped_column(Text, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    # Supersession tracking
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    superseded_by: Mapped[UUIDType | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_facts.id", ondelete="SET NULL"), nullable=True
    )

    # Source tracking
    source_chunk_ids: Mapped[list[UUIDType]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)

    # Session attribution for agentic-framework adapters (#620).
    session_id: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Bi-temporal soft-delete (migration 033, dream-phase Phase 0.3, #653).
    # Coexists with ``is_active`` in v0.14; deprecation of ``is_active`` is a
    # v0.16+ concern. NULL = still valid.
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    invalidated_by: Mapped[UUIDType | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Relationships
    namespace: Mapped[MemoryNamespaceModel] = relationship("MemoryNamespaceModel")
    superseding_fact: Mapped[MemoryFactModel | None] = relationship("MemoryFactModel", remote_side=[id])

    __table_args__ = (
        Index("ix_memory_facts_namespace_subject_active", "namespace_id", "subject", "is_active"),
        Index("ix_memory_facts_superseded_by", "superseded_by"),
    )

    def __repr__(self) -> str:
        return f"<MemoryFact(id={self.id!r}, {self.subject} {self.predicate} {self.object_})>"
