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
        logger.debug(f"Document {document.id}: created {len(chunks)} chunks")

        # Step 2: Embed (already batched internally)
        chunks = await embed_chunks(chunks, model=embedding_model)
        logger.debug(f"Document {document.id}: generated embeddings")

        # Step 3: Extract entities (parallel extraction across chunks)
        entities, relationships = await extract_entities(
            chunks,
            skill_name=skill_name,
            expertise=resolved_expertise,
            model=extraction_model,
            max_concurrent=max_concurrent_extractions,
            context=extraction_context,
        )
        logger.debug(f"Document {document.id}: extracted {len(entities)} entities, {len(relationships)} relationships")

        # Step 4 (Optional): Semantic expansion
        inferred_relationships = []
        if enable_expansion and resolved_expertise:
            from khora.extraction.expansion import SemanticExpander

            # Determine inference mode from expertise config
            inference_mode = resolved_expertise.expansion.inference_mode

            # For incremental mode, fetch existing entities/relationships from storage
            # to enable cross-document inference
            expansion_entities = list(entities)
            expansion_relationships = list(relationships)

            if inference_mode == "incremental":
                # Query existing entities and relationships from the namespace
                existing_entities = await storage.list_entities(document.namespace_id, limit=1000)
                existing_relationships = await storage.list_relationships(document.namespace_id, limit=5000)

                # Add existing data to expansion context
                expansion_entities.extend(existing_entities)
                expansion_relationships.extend(existing_relationships)

                logger.debug(
                    f"Document {document.id}: incremental mode - added {len(existing_entities)} existing entities, "
                    f"{len(existing_relationships)} existing relationships to expansion context"
                )

            # For batch mode, skip inference (only do unification on current doc)
            # Inference will be run separately after all documents are processed
            enable_inference = inference_mode != "batch" and inference_mode != "none"

            expander = SemanticExpander(
                expertise=resolved_expertise,
                enable_inference=enable_inference,
            )
            expansion_result = await expander.expand(
                entities=expansion_entities,
                relationships=expansion_relationships,
                namespace_id=document.namespace_id,
            )

            # Only keep entities from current document (not the existing ones we added)
            # The existing entities are already stored
            if inference_mode == "incremental":
                current_entity_ids = {e.id for e in entities}
                entities = [e for e in expansion_result.entities if e.id in current_entity_ids]
            else:
                entities = expansion_result.entities

            relationships = expansion_result.relationships
            inferred_relationships = expansion_result.inferred_relationships

            logger.debug(
                f"Document {document.id}: expansion unified to {len(entities)} entities, "
                f"inferred {len(inferred_relationships)} relationships (mode={inference_mode})"
            )

        # Step 4: Store chunks (batched)
        await storage.create_chunks_batch(chunks)

        # Step 5: Store entities with deduplication
        # Process entities concurrently but with semaphore to avoid overwhelming the DB
        entity_semaphore = asyncio.Semaphore(20)

        # Track mapping from original entity IDs to stored entity IDs (for dedup)
        entity_id_mapping: dict[str, str] = {}

        async def store_entity(entity):
            async with entity_semaphore:
                original_id = str(entity.id)
                existing = await storage.get_entity_by_name(
                    document.namespace_id,
                    entity.name,
                    entity.entity_type.value,
                )
                if existing:
                    existing.merge_with(entity)
                    await storage.update_entity(existing)
                    # Map original ID to existing entity's ID
                    entity_id_mapping[original_id] = str(existing.id)
                    return existing
                else:
                    await storage.create_entity(entity)
                    # ID stays the same for new entities
                    entity_id_mapping[original_id] = original_id
                    return entity

        await asyncio.gather(*[store_entity(e) for e in entities])

        # Step 6: Store relationships concurrently
        # Update relationship entity IDs to use deduplicated entity IDs
        async def store_relationship(rel):
            async with entity_semaphore:
                # Remap source and target entity IDs if they were deduplicated
                source_id = str(rel.source_entity_id)
                target_id = str(rel.target_entity_id)

                mapped_source = entity_id_mapping.get(source_id)
                mapped_target = entity_id_mapping.get(target_id)

                # Skip if either entity wasn't found (shouldn't happen but be safe)
                if not mapped_source or not mapped_target:
                    logger.debug(
                        f"Skipping relationship {rel.relationship_type}: "
                        f"missing entity mapping (source={source_id}, target={target_id})"
                    )
                    return None

                # Update the relationship with mapped IDs
                from uuid import UUID

                rel.source_entity_id = UUID(mapped_source)
                rel.target_entity_id = UUID(mapped_target)

                return await storage.create_relationship(rel)

        all_relationships = relationships + inferred_relationships
        if all_relationships:
            results = await asyncio.gather(*[store_relationship(r) for r in all_relationships])
            # Count successfully stored relationships
            stored_count = sum(1 for r in results if r is not None)
            if stored_count < len(all_relationships):
                logger.debug(
                    f"Stored {stored_count}/{len(all_relationships)} relationships "
                    f"({len(all_relationships) - stored_count} skipped due to missing entity mappings)"
                )

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


@task(name="run_batch_inference", cache_policy=NO_CACHE)
async def run_batch_inference(
    namespace_id: UUID,
    storage: StorageCoordinator,
    expertise: ExpertiseConfig,
    *,
    max_entities: int = 10000,
    max_relationships: int = 50000,
) -> dict[str, Any]:
    """Run batch inference on the entire namespace.

    This should be called after all documents are ingested when using
    inference_mode="batch". It queries all entities and relationships
    from the namespace and runs inference rules to create new relationships.

    Args:
        namespace_id: Namespace to run inference on
        storage: Storage coordinator
        expertise: ExpertiseConfig with inference rules
        max_entities: Maximum entities to load
        max_relationships: Maximum relationships to load

    Returns:
        Summary of inference results
    """
    from khora.extraction.expansion import SemanticExpander

    logger.info(f"Starting batch inference for namespace {namespace_id}")

    # Load all entities and relationships from storage
    entities = await storage.list_entities(namespace_id, limit=max_entities)
    relationships = await storage.list_relationships(namespace_id, limit=max_relationships)

    logger.info(f"Loaded {len(entities)} entities and {len(relationships)} relationships")

    if not entities:
        return {
            "entities": 0,
            "relationships": 0,
            "inferred_relationships": 0,
        }

    # Create expander with inference enabled
    expander = SemanticExpander(
        expertise=expertise,
        enable_unification=False,  # Entities already unified during ingestion
        enable_inference=True,
    )

    # Run expansion (inference only)
    expansion_result = await expander.expand(
        entities=entities,
        relationships=relationships,
        namespace_id=namespace_id,
    )

    # Store inferred relationships
    inferred_count = 0
    if expansion_result.inferred_relationships:
        for rel in expansion_result.inferred_relationships:
            try:
                await storage.create_relationship(rel)
                inferred_count += 1
            except Exception as e:
                logger.warning(f"Failed to store inferred relationship: {e}")

    logger.info(f"Batch inference complete: inferred {inferred_count} new relationships")

    return {
        "entities": len(entities),
        "relationships": len(relationships),
        "inferred_relationships": inferred_count,
    }
