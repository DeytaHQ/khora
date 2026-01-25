"""Sync management API routes for Khora Memory Lake."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from khora.api.deps import ACLEnforcerDep, MemoryLakeDep, PrincipalDep
from khora.pipelines import PipelineManager

router = APIRouter(prefix="/sync", tags=["sync"])


# =============================================================================
# Request/Response Models
# =============================================================================


class IngestRequest(BaseModel):
    """Request to ingest documents."""

    namespace_id: UUID
    documents: list[dict[str, Any]] = Field(..., description="List of documents with 'content' key")
    skill_name: str = Field("general_entities", description="Extraction skill to use")


class IngestResponse(BaseModel):
    """Response from ingestion."""

    total_documents: int
    processed_documents: int
    skipped_documents: int
    total_chunks: int
    total_entities: int
    total_relationships: int


class SyncRequest(BaseModel):
    """Request to sync from an external source."""

    namespace_id: UUID
    source: str = Field(..., description="Source name (e.g., 'github', 'notion')")
    connector_config: dict[str, Any] = Field(default_factory=dict, description="Connector configuration")
    skill_name: str = Field("general_entities", description="Extraction skill to use")


class SyncResponse(BaseModel):
    """Response from sync operation."""

    source: str
    documents_fetched: int
    documents_processed: int
    checkpoint: str | None


class CheckpointResponse(BaseModel):
    """Sync checkpoint response."""

    namespace_id: str
    source: str
    checkpoint: str | None


# =============================================================================
# Routes
# =============================================================================


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    enforcer: ACLEnforcerDep,
) -> IngestResponse:
    """Ingest documents into the memory lake.

    This endpoint runs the full ingestion pipeline:
    1. Stage documents (checksum-based change detection)
    2. Chunk documents
    3. Generate embeddings
    4. Extract entities and relationships
    """
    manager = PipelineManager(storage=lake.storage)

    try:
        run = await manager.run_ingestion(
            request.namespace_id,
            request.documents,
            skill_name=request.skill_name,
        )

        result = run.metadata.get("result", {})
        return IngestResponse(
            total_documents=result.get("total_documents", len(request.documents)),
            processed_documents=result.get("processed_documents", 0),
            skipped_documents=result.get("skipped_documents", 0),
            total_chunks=result.get("total_chunks", 0),
            total_entities=result.get("total_entities", 0),
            total_relationships=result.get("total_relationships", 0),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post("/source", response_model=SyncResponse)
async def sync_source(
    request: SyncRequest,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    enforcer: ACLEnforcerDep,
) -> SyncResponse:
    """Sync documents from an external source.

    Performs incremental sync using checkpoints to track progress.
    """
    manager = PipelineManager(storage=lake.storage)

    try:
        run = await manager.run_sync(
            request.namespace_id,
            request.source,
            connector_config=request.connector_config,
        )

        result = run.metadata.get("result", {})
        return SyncResponse(
            source=request.source,
            documents_fetched=result.get("documents_fetched", 0),
            documents_processed=result.get("documents_processed", 0),
            checkpoint=result.get("checkpoint"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {str(e)}",
        )


@router.get("/checkpoint/{namespace_id}/{source}", response_model=CheckpointResponse)
async def get_checkpoint(
    namespace_id: UUID,
    source: str,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> CheckpointResponse:
    """Get the last sync checkpoint for a source."""
    checkpoint = await lake.storage.get_sync_checkpoint(namespace_id, source)

    return CheckpointResponse(
        namespace_id=str(namespace_id),
        source=source,
        checkpoint=checkpoint,
    )


@router.put("/checkpoint/{namespace_id}/{source}")
async def set_checkpoint(
    namespace_id: UUID,
    source: str,
    checkpoint: str,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    enforcer: ACLEnforcerDep,
) -> CheckpointResponse:
    """Set the sync checkpoint for a source."""
    await lake.storage.set_sync_checkpoint(namespace_id, source, checkpoint)

    return CheckpointResponse(
        namespace_id=str(namespace_id),
        source=source,
        checkpoint=checkpoint,
    )


@router.get("/pipelines")
async def list_pipelines(
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> list[dict[str, Any]]:
    """List available pipelines."""
    from khora.pipelines.registry import get_registry

    registry = get_registry()
    pipelines = registry.all_pipelines()

    return [
        {
            "name": p.name,
            "description": p.description,
            "tags": p.tags,
            "version": p.version,
        }
        for p in pipelines
    ]
