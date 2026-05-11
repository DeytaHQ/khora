"""Source synchronization flows for Khora.

Provides incremental sync from external sources with checkpoint tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from ..registry import pipeline

if TYPE_CHECKING:
    from khora.storage import StorageCoordinator


async def get_sync_checkpoint(
    namespace_id: UUID,
    source: str,
    storage: StorageCoordinator,
) -> str | None:
    """Get the last sync checkpoint for a source."""
    return await storage.get_sync_checkpoint(namespace_id, source)


async def set_sync_checkpoint(
    namespace_id: UUID,
    source: str,
    checkpoint: str,
    storage: StorageCoordinator,
) -> None:
    """Set the sync checkpoint for a source."""
    await storage.set_sync_checkpoint(namespace_id, source, checkpoint)


async def fetch_from_source(
    source: str,
    connector_config: dict[str, Any],
    checkpoint: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch documents from an external source.

    This is a placeholder that should be replaced with actual connector
    implementations for different sources (GitHub, Notion, etc.)

    Returns:
        Tuple of (documents, new_checkpoint)
    """
    # This would be implemented by specific connectors
    # For now, return empty results
    logger.warning(f"No connector implemented for source: {source}")
    return [], checkpoint


@pipeline("sync_source", description="Sync from external source", tags=["sync"])
async def sync_source(
    namespace_id: UUID,
    source: str,
    storage: StorageCoordinator | None = None,
    *,
    connector_config: dict[str, Any] | None = None,
    skill_name: str = "general_entities",
    **kwargs,
) -> dict[str, Any]:
    """Sync documents from an external source.

    Performs incremental sync using checkpoints to track progress.

    Args:
        namespace_id: Target namespace
        source: Source name (e.g., "github", "notion")
        storage: StorageCoordinator instance
        connector_config: Configuration for the source connector
        skill_name: Extraction skill to use

    Returns:
        Summary of sync results
    """
    from .ingest import ingest_documents

    if storage is None:
        raise ValueError("storage is required")

    # Resolve namespace_id to internal row-level id at the public API boundary
    namespace_id = await storage.resolve_namespace(namespace_id)

    connector_config = connector_config or {}

    logger.info(f"Starting sync from {source} into namespace {namespace_id}")

    # Get last checkpoint
    checkpoint = await get_sync_checkpoint(namespace_id, source, storage)
    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint[:50]}...")

    # Fetch documents from source
    documents, new_checkpoint = await fetch_from_source(source, connector_config, checkpoint)

    if not documents:
        logger.info(f"No new documents from {source}")
        return {
            "source": source,
            "documents_fetched": 0,
            "documents_processed": 0,
        }

    logger.info(f"Fetched {len(documents)} documents from {source}")

    # Ingest the documents
    result = await ingest_documents(
        namespace_id=namespace_id,
        documents=documents,
        storage=storage,
        skill_name=skill_name,
        **kwargs,
    )

    # Update checkpoint
    if new_checkpoint:
        await set_sync_checkpoint(namespace_id, source, new_checkpoint, storage)
        logger.info(f"Updated checkpoint: {new_checkpoint[:50]}...")

    return {
        "source": source,
        "documents_fetched": len(documents),
        "documents_processed": result["processed_documents"],
        "checkpoint": new_checkpoint,
        **result,
    }


@pipeline("sync_all", description="Sync from all configured sources", tags=["sync"])
async def sync_all_sources(
    namespace_id: UUID,
    sources: list[dict[str, Any]],
    storage: StorageCoordinator | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Sync from multiple sources.

    Args:
        namespace_id: Target namespace
        sources: List of source configs with 'name' and 'config' keys
        storage: StorageCoordinator instance

    Returns:
        Summary of all sync results
    """
    if storage is None:
        raise ValueError("storage is required")

    results = []
    for source_config in sources:
        source_name = source_config.get("name", "unknown")
        connector_config = source_config.get("config", {})

        try:
            result = await sync_source(
                namespace_id=namespace_id,
                source=source_name,
                storage=storage,
                connector_config=connector_config,
                **kwargs,
            )
            results.append({"source": source_name, "status": "success", **result})
        except Exception as e:
            logger.error(f"Sync failed for {source_name}: {e}")
            results.append({"source": source_name, "status": "failed", "error": str(e)})

    return {
        "namespace_id": str(namespace_id),
        "sources_synced": len([r for r in results if r["status"] == "success"]),
        "sources_failed": len([r for r in results if r["status"] == "failed"]),
        "results": results,
    }
