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
    from datetime import datetime

    from khora.core.models import Document, Entity
    from khora.extraction.expansion.entity_index import EntityIndex
    from khora.extraction.skills import ExpertiseConfig
    from khora.storage import StorageCoordinator


@task(name="compute_checksum")
def compute_checksum(content: str) -> str:
    """Compute SHA-256 checksum of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_source_timestamp(metadata: dict[str, Any]) -> datetime | None:
    """Extract the original timestamp from source metadata.

    Looks for common timestamp fields and parses them.
    Priority: sent_at > created_at > timestamp > date
    """
    from datetime import datetime

    # Common timestamp field names in order of preference
    timestamp_fields = ["sent_at", "created_at", "timestamp", "date", "occurred_at", "started_at"]

    for field in timestamp_fields:
        if field in metadata and metadata[field]:
            value = metadata[field]
            try:
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    # Try ISO format first
                    if "T" in value:
                        # Handle ISO format with or without timezone
                        if value.endswith("Z"):
                            return datetime.fromisoformat(value.replace("Z", "+00:00"))
                        return datetime.fromisoformat(value)
                    # Try date-only format
                    return datetime.fromisoformat(value + "T00:00:00+00:00")
            except (ValueError, TypeError):
                continue
    return None


@task(name="stage_document", cache_policy=NO_CACHE)
async def stage_document(
    doc_input: dict[str, Any],
    namespace_id: UUID,
    storage: StorageCoordinator,
) -> Document | None:
    """Stage a document for processing.

    Checks if document already exists (by checksum) and creates it if new.
    Uses source system timestamp for created_at when available.

    Returns:
        Document if new or updated, None if unchanged
    """
    from datetime import UTC, datetime

    from khora.core.models import Document, DocumentMetadata

    content = doc_input.get("content", "")
    checksum = compute_checksum(content)

    # Check for existing document - skip if any document with same checksum exists
    existing = await storage.get_document_by_checksum(namespace_id, checksum)
    if existing:
        logger.debug(f"Document unchanged (checksum={checksum[:8]}..., status={existing.status})")
        return None

    # Extract custom metadata
    custom_metadata = doc_input.get("metadata", {})

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
        custom=custom_metadata,
    )

    # Use source timestamp if available, otherwise use current time
    source_timestamp = _extract_source_timestamp(custom_metadata)
    created_at = source_timestamp or datetime.now(UTC)

    document = Document(
        namespace_id=namespace_id,
        content=content,
        metadata=metadata,
        created_at=created_at,
        updated_at=created_at,  # Set updated_at to source time too
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
    entity_index: EntityIndex | None = None,
) -> dict[str, Any]:
    """Process a document through the enrichment pipeline.

    Steps:
    1. Chunk the document
    2. Generate embeddings for chunks (batched)
    3. Extract entities and relationships (parallel)
    4. (Optional) Semantic expansion - unify entities, infer relationships
    5. Store everything (batched)

    When *entity_index* is provided (smart mode), skips per-document DB
    fetches and O(n^2) cross-document unification.  Instead, does O(1)
    within-doc exact dedup via the shared index.  Cross-document resolution
    and inference are deferred to ``run_smart_resolution``.

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
        entity_index: Shared EntityIndex for smart mode (skip per-doc DB loads)
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
        from uuid import uuid4 as _uuid4

        from khora.telemetry.instrument import pipeline_stage

        _run_id = _uuid4()
        _ns_id = document.namespace_id

        # Step 1: Chunk
        async with pipeline_stage("ingestion", "chunking", _run_id, namespace_id=_ns_id):
            chunks = await chunk_document(
                document,
                strategy=chunk_strategy,
                chunk_size=chunk_size,
            )
        logger.debug(f"Document {document.id}: created {len(chunks)} chunks")

        # Steps 2 & 3: Embed + Extract concurrently (both depend only on chunks)
        async def _embed_with_telemetry():
            async with pipeline_stage(
                "ingestion", "embedding", _run_id, namespace_id=_ns_id, extra_metadata={"chunk_count": len(chunks)}
            ):
                return await embed_chunks(chunks, model=embedding_model)

        async def _extract_with_telemetry():
            async with pipeline_stage(
                "ingestion", "extraction", _run_id, namespace_id=_ns_id, extra_metadata={"chunk_count": len(chunks)}
            ):
                return await extract_entities(
                    chunks,
                    skill_name=skill_name,
                    expertise=resolved_expertise,
                    model=extraction_model,
                    max_concurrent=max_concurrent_extractions,
                    context=extraction_context,
                )

        embedded_chunks, (entities, relationships) = await asyncio.gather(
            _embed_with_telemetry(), _extract_with_telemetry()
        )
        chunks = embedded_chunks
        logger.debug(f"Document {document.id}: generated embeddings")
        logger.debug(f"Document {document.id}: extracted {len(entities)} entities, {len(relationships)} relationships")

        # Step 4 (Optional): Semantic expansion
        inferred_relationships = []
        inference_mode = resolved_expertise.expansion.inference_mode if resolved_expertise else "none"

        if entity_index is not None and inference_mode == "smart":
            # Smart mode: within-doc exact dedup via shared EntityIndex.
            # Cross-document resolution + inference deferred to run_smart_resolution().
            deduped_entities = []
            for entity in entities:
                existing = entity_index.add(entity)
                if existing is not None:
                    # Merge into existing (already in index)
                    existing.merge_with(entity)
                else:
                    deduped_entities.append(entity)
            if len(entities) != len(deduped_entities):
                logger.debug(f"Document {document.id}: smart dedup {len(entities)} -> {len(deduped_entities)} entities")
            entities = deduped_entities

        elif enable_expansion and resolved_expertise:
            from khora.extraction.expansion import SemanticExpander

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
        async with pipeline_stage(
            "ingestion",
            "storage",
            _run_id,
            namespace_id=_ns_id,
            extra_metadata={"chunk_count": len(chunks), "entity_count": len(entities)},
        ):
            await storage.create_chunks_batch(chunks)

        # Step 5: Store entities with deduplication
        async with pipeline_stage(
            "ingestion",
            "entity_storage",
            _run_id,
            namespace_id=_ns_id,
            input_count=len(entities),
        ) as _es_ctx:
            # Process entities concurrently but with semaphore to avoid overwhelming the DB
            entity_semaphore = asyncio.Semaphore(20)

            # Track mapping from original entity IDs to stored entity IDs (for dedup)
            entity_id_mapping: dict[str, str] = {}

            async def store_entity(entity) -> tuple[Entity, bool]:
                """Store entity and return (entity, needs_embedding)."""
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
                        # Only generate embedding if not already present
                        needs_embedding = not existing.embedding
                        return existing, needs_embedding
                    else:
                        await storage.create_entity(entity)
                        # ID stays the same for new entities
                        entity_id_mapping[original_id] = original_id
                        # New entities always need embeddings
                        return entity, True

            store_results = await asyncio.gather(*[store_entity(e) for e in entities])
            # Collect entities that need embeddings
            entities_needing_embeddings = [e for e, needs in store_results if needs]
            _es_ctx["output_count"] = len(store_results)

        # Step 5b: Generate and store entity embeddings
        if entities_needing_embeddings:
            async with pipeline_stage(
                "ingestion",
                "entity_embedding",
                _run_id,
                namespace_id=_ns_id,
                input_count=len(entities_needing_embeddings),
            ) as _ee_ctx:
                from khora.extraction.embedders import LiteLLMEmbedder

                embedder = LiteLLMEmbedder(model=embedding_model)
                # Create entity text representations for embedding
                entity_texts = [
                    f"{e.name}: {e.description}" if e.description else e.name for e in entities_needing_embeddings
                ]
                # Generate embeddings in batch
                entity_embeddings = await embedder.embed_batch(entity_texts)
                # Update entities with embeddings (single transaction)
                updates = [
                    (entity.id, embedding, embedding_model)
                    for entity, embedding in zip(entities_needing_embeddings, entity_embeddings)
                ]
                await storage.update_entity_embeddings_batch(updates)
                _ee_ctx["output_count"] = len(entities_needing_embeddings)
            logger.debug(
                f"Document {document.id}: generated embeddings for {len(entities_needing_embeddings)} entities"
            )

        # Step 6: Store relationships in batch
        # Remap entity IDs to use deduplicated entity IDs, then batch-store
        all_relationships = relationships + inferred_relationships
        if all_relationships:
            from uuid import UUID

            valid_relationships = []
            skipped = 0
            for rel in all_relationships:
                source_id = str(rel.source_entity_id)
                target_id = str(rel.target_entity_id)

                mapped_source = entity_id_mapping.get(source_id)
                mapped_target = entity_id_mapping.get(target_id)

                if not mapped_source or not mapped_target:
                    logger.debug(
                        f"Skipping relationship {rel.relationship_type}: "
                        f"missing entity mapping (source={source_id}, target={target_id})"
                    )
                    skipped += 1
                    continue

                rel.source_entity_id = UUID(mapped_source)
                rel.target_entity_id = UUID(mapped_target)
                valid_relationships.append(rel)

            if valid_relationships:
                stored_count = await storage.create_relationships_batch(valid_relationships)
            else:
                stored_count = 0

            if skipped > 0:
                logger.debug(
                    f"Stored {stored_count}/{len(all_relationships)} relationships "
                    f"({skipped} skipped due to missing entity mappings)"
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

    # Resolve expertise early to determine inference mode
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

    inference_mode = resolved_expertise.expansion.inference_mode if resolved_expertise else "none"
    is_smart = inference_mode == "smart"

    # Smart mode: create shared EntityIndex, optionally pre-load existing entities
    shared_entity_index: EntityIndex | None = None
    if is_smart and resolved_expertise:
        from khora.extraction.expansion.entity_index import EntityIndex as EI

        shared_entity_index = EI()
        if resolved_expertise.expansion.preload_existing:
            existing_entities = await storage.list_entities(namespace_id, limit=50000)
            for e in existing_entities:
                shared_entity_index.add(e)
            if existing_entities:
                logger.info(f"Smart mode: pre-loaded {len(existing_entities)} existing entities into index")

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
                entity_index=shared_entity_index,
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

    # Phase 3 (Smart mode): Post-ingestion cross-document resolution + inference
    smart_resolution_result: dict[str, Any] = {}
    if is_smart and shared_entity_index and resolved_expertise and successful_results:
        logger.info("Starting smart post-ingestion resolution...")
        smart_resolution_result = await run_smart_resolution(
            namespace_id,
            storage,
            shared_entity_index,
            resolved_expertise,
            embedding_model=embedding_model,
        )
        total_entities = smart_resolution_result.get("entities_resolved", total_entities)
        total_inferred = smart_resolution_result.get("inferred_relationships", total_inferred)

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
        "per_document_results": successful_results,
        **({"smart_resolution": smart_resolution_result} if smart_resolution_result else {}),
    }


@task(name="run_smart_resolution", cache_policy=NO_CACHE)
async def run_smart_resolution(
    namespace_id: UUID,
    storage: StorageCoordinator,
    entity_index: EntityIndex,
    expertise: ExpertiseConfig,
    *,
    embedding_model: str = "text-embedding-3-small",
) -> dict[str, Any]:
    """Post-ingestion cross-document entity resolution and relationship inference.

    Called once after all documents have been processed in smart mode.
    Uses the shared EntityIndex for blocked (O(n*k)) matching instead
    of O(n^2) pairwise comparisons.

    Steps:
        1. Run CrossToolUnifier with token blocking via EntityIndex
        2. Apply merge results to storage (batch upsert)
        3. Load all relationships once
        4. Run RelationshipInferrer on the full resolved graph
        5. Store inferred relationships (batch)

    Args:
        namespace_id: Namespace to resolve
        storage: Storage coordinator
        entity_index: Populated EntityIndex from ingestion
        expertise: ExpertiseConfig with rules
        embedding_model: Model name for entity embeddings

    Returns:
        Summary of resolution results
    """
    from khora.extraction.expansion import SemanticExpander
    from khora.extraction.expansion.relationship_inferrer import to_relationship
    from khora.telemetry.instrument import pipeline_stage

    all_entities = entity_index.get_all_entities()
    logger.info(f"Smart resolution: {len(all_entities)} entities in index " f"({entity_index.stats()})")

    if not all_entities:
        return {"entities_resolved": 0, "entities_merged": 0, "inferred_relationships": 0}

    # Phase 1: Cross-document entity unification with blocking
    async with pipeline_stage(
        "ingestion",
        "smart_resolution",
        namespace_id=namespace_id,
        input_count=len(all_entities),
    ) as _sr_ctx:
        expander = SemanticExpander(
            expertise=expertise,
            enable_unification=True,
            enable_inference=False,  # Inference done separately below
        )
        expansion_result = await expander.expand(
            entities=all_entities,
            relationships=[],  # No relationships needed for unification
            namespace_id=namespace_id,
            entity_index=entity_index,
        )
        _sr_ctx["output_count"] = len(expansion_result.entities)

    resolved_entities = expansion_result.entities
    entity_mapping = expansion_result.entity_mapping
    entities_merged = expansion_result.merged_entity_count

    logger.info(
        f"Smart resolution: unified {len(all_entities)} -> {len(resolved_entities)} " f"({entities_merged} merged)"
    )

    # Phase 2: Batch upsert resolved entities to storage
    batch_size = expertise.expansion.batch_storage_size
    await storage.upsert_entities_batch(namespace_id, resolved_entities, batch_size=batch_size)

    # Generate embeddings for entities missing them
    entities_needing_embeddings = [e for e in resolved_entities if not e.embedding]
    if entities_needing_embeddings:
        from khora.extraction.embedders import LiteLLMEmbedder

        embedder = LiteLLMEmbedder(model=embedding_model)
        entity_texts = [f"{e.name}: {e.description}" if e.description else e.name for e in entities_needing_embeddings]
        entity_embeddings = await embedder.embed_batch(entity_texts)
        updates = [
            (entity.id, embedding, embedding_model)
            for entity, embedding in zip(entities_needing_embeddings, entity_embeddings)
        ]
        await storage.update_entity_embeddings_batch(updates)
        logger.debug(f"Smart resolution: generated embeddings for {len(entities_needing_embeddings)} entities")

    # Phase 3: Load all relationships and remap merged entity IDs
    relationships = await storage.list_relationships(namespace_id, limit=50000)
    if entity_mapping:
        for rel in relationships:
            new_source = entity_mapping.get(rel.source_entity_id, rel.source_entity_id)
            new_target = entity_mapping.get(rel.target_entity_id, rel.target_entity_id)
            rel.source_entity_id = new_source
            rel.target_entity_id = new_target

    # Phase 4: Relationship inference on full resolved graph (single pass)
    from khora.extraction.expansion.relationship_inferrer import RelationshipInferrer

    inferrer = RelationshipInferrer(
        expertise=expertise,
        min_confidence=expertise.confidence.min_inferred,
    )
    inferred = inferrer.infer(
        resolved_entities,
        relationships,
        depth=expertise.expansion.depth,
    )

    # Phase 5: Store inferred relationships (batch)
    inferred_count = 0
    if inferred:
        inferred_rels = [to_relationship(inf, namespace_id) for inf in inferred]
        inferred_count = await storage.create_relationships_batch(inferred_rels, batch_size=batch_size)

    logger.info(
        f"Smart resolution complete: {len(resolved_entities)} entities, "
        f"{entities_merged} merged, {inferred_count} inferred relationships"
    )

    return {
        "entities_resolved": len(resolved_entities),
        "entities_merged": entities_merged,
        "inferred_relationships": inferred_count,
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
    logger.info("Creating SemanticExpander...")
    expander = SemanticExpander(
        expertise=expertise,
        enable_unification=False,  # Entities already unified during ingestion
        enable_inference=True,
    )
    logger.info("SemanticExpander created, starting expansion...")

    # Run expansion (inference only)
    expansion_result = await expander.expand(
        entities=entities,
        relationships=relationships,
        namespace_id=namespace_id,
    )
    logger.info(f"Expansion complete: {expansion_result.inferred_relationship_count} inferred")

    # Store inferred relationships (batch)
    inferred_count = 0
    if expansion_result.inferred_relationships:
        try:
            inferred_count = await storage.create_relationships_batch(expansion_result.inferred_relationships)
        except Exception as e:
            logger.warning(f"Failed to store inferred relationships in batch: {e}")

    logger.info(f"Batch inference complete: inferred {inferred_count} new relationships")

    return {
        "entities": len(entities),
        "relationships": len(relationships),
        "inferred_relationships": inferred_count,
    }


@task(name="backfill_entity_embeddings", cache_policy=NO_CACHE)
async def backfill_entity_embeddings(
    namespace_id: UUID,
    storage: StorageCoordinator,
    *,
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_entities: int = 50000,
) -> dict[str, Any]:
    """Backfill embeddings for entities that don't have them.

    This is useful for fixing entities created before entity embedding
    generation was implemented. It queries entities from Neo4j via the
    graph backend and generates embeddings for storage in PostgreSQL.

    Args:
        namespace_id: Namespace to process
        storage: Storage coordinator
        embedding_model: Model to use for embeddings
        batch_size: Batch size for embedding generation
        max_entities: Maximum entities to process

    Returns:
        Summary of backfill results
    """
    from khora.extraction.embedders import LiteLLMEmbedder

    logger.info(f"Starting entity embedding backfill for namespace {namespace_id}")

    # Get all entities from the namespace
    entities = await storage.list_entities(namespace_id, limit=max_entities)
    logger.info(f"Found {len(entities)} entities")

    if not entities:
        return {"total_entities": 0, "entities_updated": 0}

    # Filter to entities without embeddings
    # Note: We check the vector backend directly since graph doesn't store embeddings
    entities_needing_embeddings = []
    for entity in entities:
        if not entity.embedding:
            # Also ensure entity exists in PostgreSQL, create if not
            if storage.vector:
                exists = await storage.vector.entity_exists(entity.id)
                if not exists:
                    await storage.vector.create_entity(entity)
            entities_needing_embeddings.append(entity)

    logger.info(f"Found {len(entities_needing_embeddings)} entities needing embeddings")

    if not entities_needing_embeddings:
        return {"total_entities": len(entities), "entities_updated": 0}

    # Create embedder
    embedder = LiteLLMEmbedder(model=embedding_model, batch_size=batch_size)

    # Process in batches
    total_updated = 0
    for i in range(0, len(entities_needing_embeddings), batch_size):
        batch = entities_needing_embeddings[i : i + batch_size]

        # Create text representations
        texts = [f"{e.name}: {e.description}" if e.description else e.name for e in batch]

        # Generate embeddings
        embeddings = await embedder.embed_batch(texts)

        # Update entities
        for entity, embedding in zip(batch, embeddings):
            await storage.update_entity_embedding(entity.id, embedding, embedding_model)
            total_updated += 1

        logger.debug(f"Updated {total_updated}/{len(entities_needing_embeddings)} entity embeddings")

    logger.info(f"Entity embedding backfill complete: updated {total_updated} entities")

    return {
        "total_entities": len(entities),
        "entities_updated": total_updated,
    }
