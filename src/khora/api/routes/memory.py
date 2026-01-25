"""Memory CRUD API routes for Khora Memory Lake."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from khora.api.deps import ACLEnforcerDep, MemoryLakeDep, PrincipalDep
from khora.query import SearchMode

router = APIRouter(prefix="/memory", tags=["memory"])


# =============================================================================
# Request/Response Models
# =============================================================================


class RememberRequest(BaseModel):
    """Request to store a memory."""

    content: str = Field(..., description="Content to remember")
    namespace_id: UUID | None = Field(None, description="Target namespace (uses default if not specified)")
    title: str = Field("", description="Optional title")
    source: str = Field("", description="Optional source identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    skill_name: str = Field("general_entities", description="Extraction skill to use")


class RememberResponse(BaseModel):
    """Response from remember operation."""

    document_id: str
    namespace_id: str
    chunks_created: int
    entities_extracted: int
    relationships_created: int


class RecallRequest(BaseModel):
    """Request to recall memories."""

    query: str = Field(..., description="Search query")
    namespace_id: UUID | None = Field(None, description="Namespace to search (uses default if not specified)")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")
    mode: str = Field("hybrid", description="Search mode: vector, graph, hybrid, all")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")


class ChunkResult(BaseModel):
    """A chunk in recall results."""

    id: str
    content: str
    document_id: str
    score: float


class EntityResult(BaseModel):
    """An entity in recall results."""

    id: str
    name: str
    entity_type: str
    description: str
    score: float


class RecallResponse(BaseModel):
    """Response from recall operation."""

    query: str
    namespace_id: str
    chunks: list[ChunkResult]
    entities: list[EntityResult]
    context_text: str


class ForgetRequest(BaseModel):
    """Request to forget a memory."""

    document_id: UUID = Field(..., description="Document ID to forget")
    namespace_id: UUID | None = Field(None, description="Namespace for verification")


class ForgetResponse(BaseModel):
    """Response from forget operation."""

    deleted: bool
    document_id: str


# =============================================================================
# Routes
# =============================================================================


@router.post("/remember", response_model=RememberResponse)
async def remember(
    request: RememberRequest,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    enforcer: ACLEnforcerDep,
) -> RememberResponse:
    """Store content in the memory lake.

    Processes the content through the ingestion pipeline:
    1. Creates a document
    2. Chunks the content
    3. Generates embeddings
    4. Extracts entities and relationships
    """
    try:
        result = await lake.remember(
            request.content,
            namespace=request.namespace_id,
            title=request.title,
            source=request.source,
            metadata=request.metadata,
            skill_name=request.skill_name,
        )

        return RememberResponse(
            document_id=str(result.document_id),
            namespace_id=str(result.namespace_id),
            chunks_created=result.chunks_created,
            entities_extracted=result.entities_extracted,
            relationships_created=result.relationships_created,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remember: {str(e)}",
        )


@router.post("/recall", response_model=RecallResponse)
async def recall(
    request: RecallRequest,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> RecallResponse:
    """Recall memories relevant to a query.

    Searches across vector, graph, and keyword indexes
    and returns ranked results.
    """
    # Map mode string to enum
    mode_map = {
        "vector": SearchMode.VECTOR,
        "graph": SearchMode.GRAPH,
        "hybrid": SearchMode.HYBRID,
        "all": SearchMode.ALL,
    }
    mode = mode_map.get(request.mode.lower(), SearchMode.HYBRID)

    try:
        result = await lake.recall(
            request.query,
            namespace=request.namespace_id,
            limit=request.limit,
            mode=mode,
            min_similarity=request.min_similarity,
        )

        # Convert to response format
        chunks = [
            ChunkResult(
                id=str(chunk.id),
                content=chunk.content,
                document_id=str(chunk.document_id),
                score=score,
            )
            for chunk, score in result.chunks
        ]

        entities = [
            EntityResult(
                id=str(entity.id),
                name=entity.name,
                entity_type=entity.entity_type.value,
                description=entity.description,
                score=score,
            )
            for entity, score in result.entities
        ]

        return RecallResponse(
            query=result.query,
            namespace_id=str(result.namespace_id),
            chunks=chunks,
            entities=entities,
            context_text=result.context_text,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to recall: {str(e)}",
        )


@router.delete("/forget", response_model=ForgetResponse)
async def forget(
    request: ForgetRequest,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    enforcer: ACLEnforcerDep,
) -> ForgetResponse:
    """Remove a memory from the lake."""
    try:
        deleted = await lake.forget(
            request.document_id,
            namespace=request.namespace_id,
        )

        return ForgetResponse(
            deleted=deleted,
            document_id=str(request.document_id),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to forget: {str(e)}",
        )


@router.get("/documents/{document_id}")
async def get_document(
    document_id: UUID,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> dict[str, Any]:
    """Get a document by ID."""
    document = await lake.storage.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    return {
        "id": str(document.id),
        "namespace_id": str(document.namespace_id),
        "status": document.status.value,
        "title": document.metadata.title,
        "source": document.metadata.source,
        "chunk_count": document.chunk_count,
        "entity_count": document.entity_count,
        "created_at": document.created_at.isoformat(),
        "processed_at": document.processed_at.isoformat() if document.processed_at else None,
    }


@router.get("/entities")
async def list_entities(
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    namespace_id: UUID | None = Query(None),
    entity_type: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """List entities in a namespace."""
    entities = await lake.list_entities(
        namespace=namespace_id,
        entity_type=entity_type,
        limit=limit,
    )

    return [
        {
            "id": str(e.id),
            "name": e.name,
            "entity_type": e.entity_type.value,
            "description": e.description,
            "mention_count": e.mention_count,
            "confidence": e.confidence,
        }
        for e in entities
    ]


@router.get("/entities/{entity_id}")
async def get_entity(
    entity_id: UUID,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> dict[str, Any]:
    """Get an entity by ID."""
    entity = await lake.get_entity(entity_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity not found: {entity_id}",
        )

    return {
        "id": str(entity.id),
        "namespace_id": str(entity.namespace_id),
        "name": entity.name,
        "entity_type": entity.entity_type.value,
        "description": entity.description,
        "attributes": entity.attributes,
        "mention_count": entity.mention_count,
        "confidence": entity.confidence,
        "source_document_ids": [str(d) for d in entity.source_document_ids],
        "created_at": entity.created_at.isoformat(),
    }


@router.get("/entities/{entity_id}/related")
async def get_related_entities(
    entity_id: UUID,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    max_depth: int = Query(2, ge=1, le=5),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Get entities related to a given entity."""
    related = await lake.find_related_entities(
        entity_id,
        max_depth=max_depth,
        limit=limit,
    )

    return [
        {
            "id": str(e.id),
            "name": e.name,
            "entity_type": e.entity_type.value,
            "relevance_score": score,
        }
        for e, score in related
    ]
