"""Document and chunk models for Khora Memory Lake.

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


@dataclass(slots=True)
class DocumentMetadata:
    """Metadata associated with a document."""

    source: str = ""  # Source identifier (URL, file path, etc.)
    source_type: str = ""  # Type of source (file, url, api, etc.)
    source_tool: str = ""  # Canonical SaaS tool identifier (see core.models.source.SourceTool)
    content_type: str = ""  # MIME type or content classification
    title: str = ""
    author: str = ""
    language: str = "en"
    checksum: str = ""  # For change detection
    size_bytes: int = 0
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A document to be processed and stored in the memory lake.

    Documents are the primary input unit. They are chunked, embedded,
    and stored for retrieval. Entities and relationships are extracted
    from documents during processing.
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    content: str = ""
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    status: DocumentStatus = DocumentStatus.PENDING

    # Processing info
    chunk_count: int = 0
    entity_count: int = 0
    error_message: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = None
    source_timestamp: datetime | None = None

    @property
    def is_processed(self) -> bool:
        """Check if the document has been successfully processed."""
        return self.status == DocumentStatus.COMPLETED

    def mark_processing(self) -> None:
        """Mark the document as currently processing."""
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.now(UTC)

    def mark_completed(self, chunk_count: int, entity_count: int) -> None:
        """Mark the document as successfully processed."""
        self.status = DocumentStatus.COMPLETED
        self.chunk_count = chunk_count
        self.entity_count = entity_count
        self.processed_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)
        self.error_message = None

    def mark_failed(self, error: str) -> None:
        """Mark the document as failed."""
        self.status = DocumentStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.now(UTC)


@dataclass(slots=True)
class ChunkMetadata:
    """Metadata associated with a chunk."""

    document_id: UUID = field(default_factory=uuid4)
    chunk_index: int = 0  # Position in document
    start_char: int = 0  # Start character offset
    end_char: int = 0  # End character offset
    token_count: int = 0
    custom: dict[str, Any] = field(default_factory=dict)


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
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Embedding vector (stored in pgvector)
    embedding: list[float] | None = None
    embedding_model: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_timestamp: datetime | None = None

    @property
    def has_embedding(self) -> bool:
        """Check if the chunk has an embedding."""
        return self.embedding is not None and len(self.embedding) > 0
