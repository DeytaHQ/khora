"""Document and chunk models for Khora.

Documents represent source content that is chunked, embedded, and stored
for semantic search and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"  # Waiting to be processed
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Successfully processed
    FAILED = "failed"  # Processing failed
    ARCHIVED = "archived"  # Archived, not actively used


@dataclass(slots=True, frozen=True)
class DocumentSource:
    """Lightweight document metadata projection for source attribution.

    Returned by read methods when ``include_sources=True``.
    Contains only the fields needed for display/linking — no content,
    processing stats, or mutable state.
    """

    id: UUID
    title: str = ""
    source: str = ""
    source_type: str = ""
    created_at: datetime | None = None
    source_timestamp: datetime | None = None


@dataclass
class Document:
    """A document to be processed and stored in Khora.

    Documents are the primary input unit. They are chunked, embedded,
    and stored for retrieval. Entities and relationships are extracted
    from documents during processing.
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    content: str = ""
    external_id: str | None = None
    status: DocumentStatus = DocumentStatus.PENDING

    # Source/provenance fields (flat).
    title: str | None = None
    source: str | None = None
    source_type: str = "library"
    source_name: str | None = None
    source_url: str | None = None
    content_type: str | None = None
    author: str | None = None
    language: str | None = None
    checksum: str | None = None
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Processing info
    chunk_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    error_message: str | None = None

    # Extraction config tracking (max 255 chars; accommodates compound keys)
    extraction_config_hash: str | None = None

    # Extraction parameters stored for deferred/crash-recovery processing.
    # Contains skill_name, entity_types, relationship_types, expertise (as dict),
    # chunk_strategy, and max_chunks_in_flight so the pending processor can
    # reconstruct the original extraction intent without hardcoding defaults.
    extraction_params: dict[str, Any] | None = None

    # Maximum length for extraction_config_hash (matches DB column String(255))
    _EXTRACTION_HASH_MAX_LEN: int = field(default=255, init=False, repr=False, compare=False)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = None
    source_timestamp: datetime | None = None

    # Session attribution for agentic-framework adapters (#620).
    # Stable public API — coordinate changes with khora-cli, khora-explorer.
    session_id: UUID | None = None

    # Maximum length for external_id (matches DB column String(512))
    _EXTERNAL_ID_MAX_LEN: int = field(default=512, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.external_id is not None:
            if not self.external_id.strip():
                raise ValueError("external_id must be None or a non-blank string")
            if len(self.external_id) > self._EXTERNAL_ID_MAX_LEN:
                raise ValueError(
                    f"external_id must be at most {self._EXTERNAL_ID_MAX_LEN} characters, got {len(self.external_id)}"
                )
        if self.extraction_config_hash is not None and len(self.extraction_config_hash) > self._EXTRACTION_HASH_MAX_LEN:
            raise ValueError(
                f"extraction_config_hash must be at most {self._EXTRACTION_HASH_MAX_LEN} characters, "
                f"got {len(self.extraction_config_hash)}"
            )

    @property
    def is_processed(self) -> bool:
        """Check if the document has been successfully processed."""
        return self.status == DocumentStatus.COMPLETED

    def mark_processing(self) -> None:
        """Mark the document as currently processing."""
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.now(UTC)

    def mark_completed(self, chunk_count: int, entity_count: int, relationship_count: int = 0) -> None:
        """Mark the document as successfully processed."""
        self.status = DocumentStatus.COMPLETED
        self.chunk_count = chunk_count
        self.entity_count = entity_count
        self.relationship_count = relationship_count
        self.processed_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)
        self.error_message = None

    def mark_failed(self, error: str) -> None:
        """Mark the document as failed."""
        self.status = DocumentStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.now(UTC)


@dataclass(slots=True)
class Chunk:
    """A chunk of text from a document with its embedding.

    Chunks are the unit of storage and retrieval for vector search.
    Each chunk has an embedding vector for semantic similarity search.
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    content: str = ""

    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    chunker_info: dict[str, Any] = field(default_factory=dict)

    # Embedding vector (stored in pgvector)
    embedding: list[float] | None = None
    embedding_model: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_timestamp: datetime | None = None
    # Reinforcement-on-recall (#855). NULL until the chunk is first
    # returned by a recall path that has reinforcement enabled.
    last_accessed_at: datetime | None = None

    # Session attribution propagated from the parent document (#620).
    # Stable public API — coordinate changes with khora-cli, khora-explorer.
    session_id: UUID | None = None

    # Populated by Khora when include_sources=True
    source_document: DocumentSource | None = None

    @property
    def has_embedding(self) -> bool:
        """Check if the chunk has an embedding."""
        return self.embedding is not None and len(self.embedding) > 0
