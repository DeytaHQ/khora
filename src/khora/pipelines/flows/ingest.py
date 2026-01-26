"""Two-phase ingestion flow for Khora Memory Lake.

Phase 1 (Staging): Fast parallel fetch, checksum-based change detection
Phase 2 (Enrichment): Chunk, embed, extract entities, integrate graph
Phase 3 (Expansion, optional): Semantic expansion, entity unification, relationship inference

Supports parallel document processing with configurable concurrency.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from ..registry import pipeline

if TYPE_CHECKING:
    from khora.core.models import Document
    from khora.extraction.skills import ExpertiseConfig
    from khora.storage import StorageCoordinator


@task(name="compute_checksum")
def compute_checksum(content: str) -> str:
    """Compute SHA-256 checksum of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@task(name="stage_document", cache_policy=NO_CACHE)
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

    # Check for existing document - skip if any document with same checksum exists
    existing = await storage.get_document_by_checksum(namespace_id, checksum)
    if existing:
        logger.debug(f"Document unchanged (checksum={checksum[:8]}..., status={existing.status})")
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


@task(name="process_document", cache_policy=NO_CACHE)
async def process_document(
    document: Document,
    storage: StorageCoordinator,
    *,
    chunk_strategy: str = "semantic",
    chunk_size: int = 512,
    embedding_model: str = "text-embedding-3-small",
    extraction_model: str = "gpt-4o-mini",
    skill_name: str = "general_entities",
    expertise: ExpertiseConfig | str | None = None,
    max_concurrent_extractions: int = 10,
    enable_expansion: bool = False,
    extraction_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process a document through the enrichment pipeline.

    Steps:
    1. Chunk the document
    2. Generate embeddings for chunks (batched)
    3. Extract entities and relationships (parallel)
    4. (Optional) Semantic expansion - unify entities, infer relationships
    5. Store everything (batched)

    Args:
        document: Document to process
        storage: Storage coordinator
        chunk_strategy: Chunking strategy
        chunk_size: Target chunk size
        embedding_model: Model for embeddings
        extraction_model: Model for extraction
        skill_name: Legacy skill name (ignored if expertise provided)
        expertise: ExpertiseConfig, expertise name, or file path
        max_concurrent_extractions: Maximum concurrent LLM extractions
        enable_expansion: Whether to run semantic expansion
        extraction_context: Context dict for prompt template rendering
    """
    from ..tasks import chunk_document, embed_chunks, extract_entities

    # Resolve expertise if needed
    resolved_expertise: ExpertiseConfig | None = None
    if expertise is not None:
        from khora.extraction.skills import ExpertiseConfig as EC
        from khora.extraction.skills import load_expertise

        if isinstance(expertise, EC):
            resolved_expertise = expertise
        elif isinstance(expertise, str):
            try:
                resolved_expertise = load_expertise(expertise)
            except Exception:
                pass

    # Check if expansion is enabled in expertise config
    if resolved_expertise and resolved_expertise.expansion.enabled:
        enable_expansion = True

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

        # Step 2: Embed (already batched internally)
        chunks = await embed_chunks(chunks, model=embedding_model)
        logger.info(f"Document {document.id}: generated embeddings")

        # Step 3: Extract entities (parallel extraction across chunks)
        entities, relationships = await extract_entities(
            chunks,
            skill_name=skill_name,
            expertise=resolved_expertise,
            model=extraction_model,
            max_concurrent=max_concurrent_extractions,
            context=extraction_context,
        )
        logger.info(f"Document {document.id}: extracted {len(entities)} entities, {len(relationships)} relationships")

        # Step 4 (Optional): Semantic expansion
        inferred_relationships = []
        if enable_expansion and resolved_expertise:
            from khora.extraction.expansion import SemanticExpander

            expander = SemanticExpander(expertise=resolved_expertise)
            expansion_result = await expander.expand(
                entities=entities,
                relationships=relationships,
                namespace_id=document.namespace_id,
            )

            entities = expansion_result.entities
            relationships = expansion_result.relationships
            inferred_relationships = expansion_result.inferred_relationships

            logger.info(
                f"Document {document.id}: expansion unified to {len(entities)} entities, "
                f"inferred {len(inferred_relationships)} relationships"
            )

        # Step 4: Store chunks (batched)
        await storage.create_chunks_batch(chunks)

        # Step 5: Store entities with deduplication
        # Process entities concurrently but with semaphore to avoid overwhelming the DB
        entity_semaphore = asyncio.Semaphore(20)

        async def store_entity(entity):
            async with entity_semaphore:
                existing = await storage.get_entity_by_name(
                    document.namespace_id,
                    entity.name,
                    entity.entity_type.value,
                )
                if existing:
                    existing.merge_with(entity)
                    return await storage.update_entity(existing)
                else:
                    return await storage.create_entity(entity)

        await asyncio.gather(*[store_entity(e) for e in entities])

        # Step 6: Store relationships concurrently
        async def store_relationship(rel):
            async with entity_semaphore:
                return await storage.create_relationship(rel)

        all_relationships = relationships + inferred_relationships
        if all_relationships:
            await asyncio.gather(*[store_relationship(r) for r in all_relationships])

        # Mark as completed
        document.mark_completed(len(chunks), len(entities))
        await storage.update_document(document)

        return {
            "document_id": str(document.id),
            "chunks": len(chunks),
            "entities": len(entities),
            "relationships": len(relationships),
            "inferred_relationships": len(inferred_relationships),
        }

    except Exception as e:
        document.mark_failed(str(e))
        await storage.update_document(document)
        raise


@pipeline("ingest", description="Two-phase document ingestion with optional expansion", tags=["ingestion"])
@flow(name="ingest_documents", log_prints=True)
async def ingest_documents(
    namespace_id: UUID,
    documents: list[dict[str, Any]],
    storage: StorageCoordinator | None = None,
    *,
    skill_name: str = "general_entities",
    expertise: ExpertiseConfig | str | None = None,
    chunk_strategy: str = "semantic",
    chunk_size: int = 512,
    embedding_model: str = "text-embedding-3-small",
    extraction_model: str = "gpt-4o-mini",
    max_concurrent_documents: int = 5,
    max_concurrent_extractions: int = 10,
    enable_expansion: bool = False,
    extraction_context: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Two-phase document ingestion flow with parallel processing.

    Phase 1: Stage documents (checksum-based change detection)
    Phase 2: Process changed documents in parallel (chunk, embed, extract)
    Phase 3 (Optional): Semantic expansion (entity unification, relationship inference)

    Args:
        namespace_id: Target namespace
        documents: List of document dicts with 'content' and optional metadata
        storage: StorageCoordinator instance
        skill_name: Legacy extraction skill to use (ignored if expertise provided)
        expertise: ExpertiseConfig, expertise name string, or file path
        chunk_strategy: Chunking strategy
        chunk_size: Target chunk size
        embedding_model: Model for embeddings
        extraction_model: Model for extraction
        max_concurrent_documents: Maximum documents to process in parallel
        max_concurrent_extractions: Maximum concurrent LLM extractions per document
        enable_expansion: Whether to run semantic expansion
        extraction_context: Context dict for prompt template rendering

    Returns:
        Summary of ingestion results
    """
    if storage is None:
        raise ValueError("storage is required")

    logger.info(f"Starting ingestion of {len(documents)} documents into namespace {namespace_id}")

    # Phase 1: Stage documents (can run in parallel too)
    staging_semaphore = asyncio.Semaphore(max_concurrent_documents * 2)

    async def stage_with_limit(doc_input):
        async with staging_semaphore:
            return await stage_document(doc_input, namespace_id, storage)

    staged_results = await asyncio.gather(*[stage_with_limit(doc) for doc in documents])
    staged_docs = [doc for doc in staged_results if doc is not None]

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

    # Phase 2: Process staged documents in parallel with controlled concurrency
    doc_semaphore = asyncio.Semaphore(max_concurrent_documents)

    async def process_with_limit(doc):
        async with doc_semaphore:
            return await process_document(
                doc,
                storage,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                embedding_model=embedding_model,
                extraction_model=extraction_model,
                skill_name=skill_name,
                expertise=expertise,
                max_concurrent_extractions=max_concurrent_extractions,
                enable_expansion=enable_expansion,
                extraction_context=extraction_context,
            )

    results = await asyncio.gather(
        *[process_with_limit(doc) for doc in staged_docs],
        return_exceptions=True,
    )

    # Filter out exceptions and count errors
    successful_results = []
    error_count = 0
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Document processing failed: {result}")
            error_count += 1
        else:
            successful_results.append(result)

    # Aggregate results
    total_chunks = sum(r["chunks"] for r in successful_results)
    total_entities = sum(r["entities"] for r in successful_results)
    total_relationships = sum(r["relationships"] for r in successful_results)
    total_inferred = sum(r.get("inferred_relationships", 0) for r in successful_results)

    logger.info(f"Ingestion complete: {len(successful_results)} documents processed, {error_count} errors")

    return {
        "total_documents": len(documents),
        "processed_documents": len(successful_results),
        "skipped_documents": len(documents) - len(staged_docs),
        "failed_documents": error_count,
        "total_chunks": total_chunks,
        "total_entities": total_entities,
        "total_relationships": total_relationships,
        "total_inferred_relationships": total_inferred,
    }
