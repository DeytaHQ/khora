"""Incremental update manager for Khora.

Handles change detection and incremental processing of documents.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

if TYPE_CHECKING:
    from khora.storage import StorageCoordinator


@dataclass
class ChangeDetectionResult:
    """Result of change detection for a batch of documents."""

    new_documents: list[dict[str, Any]] = field(default_factory=list)
    updated_documents: list[dict[str, Any]] = field(default_factory=list)
    unchanged_documents: list[dict[str, Any]] = field(default_factory=list)
    deleted_document_ids: list[UUID] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.new_documents or self.updated_documents or self.deleted_document_ids)

    @property
    def total_changes(self) -> int:
        """Get total number of changes."""
        return len(self.new_documents) + len(self.updated_documents) + len(self.deleted_document_ids)


class IncrementalUpdateManager:
    """Manager for incremental document updates.

    Provides checksum-based change detection and incremental
    processing of document changes.
    """

    def __init__(self, storage: StorageCoordinator) -> None:
        """Initialize the incremental update manager.

        Args:
            storage: StorageCoordinator instance
        """
        self._storage = storage

    def compute_checksum(self, content: str) -> str:
        """Compute SHA-256 checksum of content.

        Args:
            content: Content to checksum

        Returns:
            Hex digest of checksum
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def detect_changes(
        self,
        namespace_id: UUID,
        documents: list[dict[str, Any]],
        *,
        detect_deletions: bool = False,
        existing_source_ids: list[str] | None = None,
    ) -> ChangeDetectionResult:
        """Detect changes in a batch of documents.

        Args:
            namespace_id: Namespace to check against
            documents: List of document dicts with 'content' and optionally 'source_id'
            detect_deletions: Whether to detect deleted documents
            existing_source_ids: List of source IDs for deletion detection

        Returns:
            ChangeDetectionResult with categorized documents
        """
        result = ChangeDetectionResult()

        for doc in documents:
            content = doc.get("content", "")
            checksum = self.compute_checksum(content)

            # Check for existing document by checksum
            existing = await self._storage.get_document_by_checksum(namespace_id, checksum)

            if existing is None:
                # New document
                doc["_checksum"] = checksum
                result.new_documents.append(doc)
            elif not existing.is_processed:
                # Document exists but not processed (retry)
                doc["_checksum"] = checksum
                doc["_existing_id"] = str(existing.id)
                result.updated_documents.append(doc)
            else:
                # Unchanged
                result.unchanged_documents.append(doc)

        # Detect deletions if requested
        if detect_deletions and existing_source_ids:
            current_source_ids = {doc.get("source_id") for doc in documents if doc.get("source_id")}
            deleted_ids = set(existing_source_ids) - current_source_ids
            # Note: Would need to implement source_id lookup in storage
            # For now, this is a placeholder
            logger.debug(f"Deletion detection found {len(deleted_ids)} potential deletions")

        logger.info(
            f"Change detection: {len(result.new_documents)} new, "
            f"{len(result.updated_documents)} updated, "
            f"{len(result.unchanged_documents)} unchanged"
        )

        return result

    async def process_incremental(
        self,
        namespace_id: UUID,
        changes: ChangeDetectionResult,
        *,
        skill_name: str = "general_entities",
    ) -> dict[str, Any]:
        """Process incremental changes.

        Args:
            namespace_id: Target namespace
            changes: Change detection result
            skill_name: Extraction skill to use

        Returns:
            Processing results
        """
        from khora.pipelines.flows.ingest import ingest_documents

        if not changes.has_changes:
            return {
                "processed": 0,
                "skipped": len(changes.unchanged_documents),
            }

        # Resolve namespace_id to internal row-level id at the public API boundary
        namespace_id = await self._storage.resolve_namespace(namespace_id)

        # Process new and updated documents
        documents_to_process = changes.new_documents + changes.updated_documents

        result = await ingest_documents(
            namespace_id=namespace_id,
            documents=documents_to_process,
            storage=self._storage,
            skill_name=skill_name,
        )

        # Handle deletions
        if changes.deleted_document_ids:
            for doc_id in changes.deleted_document_ids:
                await self._storage.delete_document(doc_id, namespace_id=namespace_id)
            logger.info(f"Deleted {len(changes.deleted_document_ids)} documents")

        return {
            "new_processed": len(changes.new_documents),
            "updated_processed": len(changes.updated_documents),
            "deleted": len(changes.deleted_document_ids),
            "unchanged": len(changes.unchanged_documents),
            **result,
        }
