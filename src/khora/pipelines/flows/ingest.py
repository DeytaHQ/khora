"""Two-phase ingestion flow for Khora Memory Lake.

Phase 1 (Staging): Fast parallel fetch, checksum-based change detection
Phase 2 (Enrichment): Chunk, embed, extract entities, integrate graph
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger
from prefect import flow, task

from ..registry import pipeline

if TYPE_CHECKING:
    from khora.core.models import Document
    from khora.storage import StorageCoordinator


@task(name="compute_checksum")
def compute_checksum(content: str) -> str:
    """Compute SHA-256 checksum of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@task(name="stage_document")
async def stage_document(
    doc_input: dict[str, Any],
    namespace_id: UUID,
    storage: StorageCoordinator,
) -> Document | None:
    """Stage a document for processing.

    Checks if document already exists (by checksum) and creates it if new.

    Returns:
        Document if new or updated, None if unchanged
    """
    from khora.core.models import Document, DocumentMetadata

    content = doc_input.get("content", "")
    checksum = compute_checksum(content)

    # Check for existing document
    existing = await storage.get_document_by_checksum(namespace_id, checksum)
    if existing and existing.is_processed:
        logger.debug(f"Document unchanged (checksum={checksum[:8]}...)")
        return None

    # Create document
    metadata = DocumentMetadata(
        source=doc_input.get("source", ""),
        source_type=doc_input.get("source_type", "manual"),
        content_type=doc_input.get("content_type", "text/plain"),
        title=doc_input.get("title", ""),
        author=doc_input.get("author", ""),
        language=doc_input.get("language", "en"),
        checksum=checksum,
        size_bytes=len(content.encode("utf-8")),
        custom=doc_input.get("metadata", {}),
    )

    document = Document(
        namespace_id=namespace_id,
        content=content,
        metadata=metadata,
    )

    return await storage.create_document(document)


@task(name="process_document")
async def process_document(
    document: Document,
    storage: StorageCoordinator,
    *,
    chunk_strategy: str = "semantic",
    chunk_size: int = 512,
    embedding_model: str = "text-embedding-3-small",
    extraction_model: str = "gpt-4o-mini",
    skill_name: str = "general_entities",
) -> dict[str, Any]:
    """Process a document through the enrichment pipeline.

    Steps:
    1. Chunk the document
    2. Generate embeddings for chunks
    3. Extract entities and relationships
    4. Store everything
    """
    from ..tasks import chunk_document, embed_chunks, extract_entities

    # Mark as processing
    document.mark_processing()
    await storage.update_document(document)

    try:
        # Step 1: Chunk
        chunks = await chunk_document(
            document,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
        )
        logger.info(f"Document {document.id}: created {len(chunks)} chunks")

        # Step 2: Embed
        chunks = await embed_chunks(chunks, model=embedding_model)
        logger.info(f"Document {document.id}: generated embeddings")

        # Step 3: Extract entities
        entities, relationships = await extract_entities(
            chunks,
            skill_name=skill_name,
            model=extraction_model,
        )
        logger.info(f"Document {document.id}: extracted {len(entities)} entities, {len(relationships)} relationships")

        # Step 4: Store chunks
        await storage.create_chunks_batch(chunks)

        # Step 5: Store entities
        stored_entities = []
        for entity in entities:
            # Check for existing entity (dedup)
            existing = await storage.get_entity_by_name(
                document.namespace_id,
                entity.name,
                entity.entity_type.value,
            )
            if existing:
                existing.merge_with(entity)
                stored = await storage.update_entity(existing)
            else:
                stored = await storage.create_entity(entity)
            stored_entities.append(stored)

        # Step 6: Store relationships
        for relationship in relationships:
            await storage.create_relationship(relationship)

        # Mark as completed
        document.mark_completed(len(chunks), len(entities))
        await storage.update_document(document)

        return {
            "document_id": str(document.id),
            "chunks": len(chunks),
            "entities": len(entities),
            "relationships": len(relationships),
        }

    except Exception as e:
        document.mark_failed(str(e))
        await storage.update_document(document)
        raise


@pipeline("ingest", description="Two-phase document ingestion", tags=["ingestion"])
@flow(name="ingest_documents", log_prints=True)
async def ingest_documents(
    namespace_id: UUID,
    documents: list[dict[str, Any]],
    storage: StorageCoordinator | None = None,
    *,
    skill_name: str = "general_entities",
    chunk_strategy: str = "semantic",
    chunk_size: int = 512,
    embedding_model: str = "text-embedding-3-small",
    extraction_model: str = "gpt-4o-mini",
    **kwargs,
) -> dict[str, Any]:
    """Two-phase document ingestion flow.

    Phase 1: Stage documents (checksum-based change detection)
    Phase 2: Process changed documents (chunk, embed, extract)

    Args:
        namespace_id: Target namespace
        documents: List of document dicts with 'content' and optional metadata
        storage: StorageCoordinator instance
        skill_name: Extraction skill to use
        chunk_strategy: Chunking strategy
        chunk_size: Target chunk size
        embedding_model: Model for embeddings
        extraction_model: Model for extraction

    Returns:
        Summary of ingestion results
    """
    if storage is None:
        raise ValueError("storage is required")

    logger.info(f"Starting ingestion of {len(documents)} documents into namespace {namespace_id}")

    # Phase 1: Stage documents
    staged_docs = []
    for doc_input in documents:
        doc = await stage_document(doc_input, namespace_id, storage)
        if doc:
            staged_docs.append(doc)

    logger.info(f"Phase 1 complete: {len(staged_docs)} documents to process")

    if not staged_docs:
        return {
            "total_documents": len(documents),
            "processed_documents": 0,
            "skipped_documents": len(documents),
            "total_chunks": 0,
            "total_entities": 0,
            "total_relationships": 0,
        }

    # Phase 2: Process staged documents
    results = []
    for doc in staged_docs:
        result = await process_document(
            doc,
            storage,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            embedding_model=embedding_model,
            extraction_model=extraction_model,
            skill_name=skill_name,
        )
        results.append(result)

    # Aggregate results
    total_chunks = sum(r["chunks"] for r in results)
    total_entities = sum(r["entities"] for r in results)
    total_relationships = sum(r["relationships"] for r in results)

    logger.info(f"Ingestion complete: {len(results)} documents processed")

    return {
        "total_documents": len(documents),
        "processed_documents": len(results),
        "skipped_documents": len(documents) - len(results),
        "total_chunks": total_chunks,
        "total_entities": total_entities,
        "total_relationships": total_relationships,
    }
