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

from khora._accel import normalize_entity_names_batch

from ..registry import pipeline

if TYPE_CHECKING:
    from datetime import datetime

    from khora.core.models import Chunk, Document, Entity, Relationship
    from khora.engines.skeleton.backends import TemporalVectorStore
    from khora.extraction.embedders import Embedder
    from khora.extraction.expansion.entity_index import EntityIndex
    from khora.extraction.skills import ExpertiseConfig
    from khora.storage import StorageCoordinator


def _should_skip_entity_embedding(
    entity,
    skip_types: list[str],
    mention_threshold: int,
) -> bool:
    """Check if an entity should skip embedding generation.

    Low-value entity types (DATE, URL, EMAIL) don't benefit from vector
    similarity search. Skipping embedding saves API cost and storage.

    Args:
        entity: Entity to check.
        skip_types: Entity type names to skip (case-insensitive).
        mention_threshold: When 0, skip ALL entities of the listed types
            regardless of mention count.  When >0, only skip if
            ``entity.mention_count <= mention_threshold``.
    """
    if not skip_types:
        return False
    entity_type = entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type)
    _skip_upper = frozenset(t.upper() for t in skip_types)
    if entity_type.upper() not in _skip_upper:
        return False
    # threshold=0 means skip all entities of these types unconditionally
    if mention_threshold == 0:
        return True
    return entity.mention_count <= mention_threshold


def _find_entity_key(normalized_name: str, all_entities: dict[str, Any]) -> str | None:
    """Find entity key by exact match first, then fuzzy Levenshtein matching."""
    from khora._accel import levenshtein_similarity

    # Exact prefix match first (fast path)
    exact = next((k for k in all_entities if k.startswith(f"{normalized_name}:")), None)
    if exact:
        return exact
    # Fuzzy match: compare normalized name against entity name portion of key
    best_key = None
    best_sim = 0.0
    for k in all_entities:
        entity_name = k.split(":")[0]
        sim = levenshtein_similarity(normalized_name, entity_name)
        if sim > best_sim and sim > 0.8:
            best_sim = sim
            best_key = k
    return best_key


async def _extract_cross_chunk_relationships(
    chunks: list,
    entities_by_chunk: dict,
    extractor,
    extraction_context: dict,
    *,
    max_windows: int = 50,
) -> list:
    """Extract relationships spanning chunk boundaries via overlapping windows.

    Creates overlapping windows of 2 consecutive chunks and runs the extractor
    on the combined text to find relationships between entities that cross chunk
    boundaries. Deduplicates results across windows by (source, rel_type, target).

    This is opt-in — requires ``cross_chunk_extraction=True`` in extraction_context.
    Increases LLM calls by up to ``len(chunks) - 1`` (capped at ``max_windows``).

    Args:
        chunks: Chunk objects with ``.content``, ``.id``, and optional ``.metadata.chunk_index``
        entities_by_chunk: Mapping of chunk_id to list of entity name strings in that chunk
        extractor: Initialized entity extractor (e.g. LLMEntityExtractor)
        extraction_context: Context dict; must contain ``cross_chunk_extraction=True``
        max_windows: Maximum windows to process (cap on extra LLM calls)

    Returns:
        Deduplicated list of ExtractedRelationship objects found across windows
    """
    if not extraction_context.get("cross_chunk_extraction", False):
        return []
    if len(chunks) < 2:
        return []

    def _chunk_index(c) -> int:
        if hasattr(c, "metadata") and c.metadata and hasattr(c.metadata, "chunk_index"):
            return c.metadata.chunk_index or 0
        return 0

    sorted_chunks = sorted(chunks, key=_chunk_index)
    seen_triples: set[tuple[str, str, str]] = set()
    new_relationships: list = []

    for i in range(min(len(sorted_chunks) - 1, max_windows)):
        chunk_a = sorted_chunks[i]
        chunk_b = sorted_chunks[i + 1]

        names_a = entities_by_chunk.get(chunk_a.id, [])
        names_b = entities_by_chunk.get(chunk_b.id, [])
        all_names = list(dict.fromkeys(names_a + names_b))  # dedupe, preserve order

        if not all_names:
            continue

        combined_text = f"{chunk_a.content}\n\n{chunk_b.content}"
        window_ctx = {**extraction_context, "known_entities": all_names}

        try:
            results = await extractor.extract_multi(
                [combined_text],
                batch_size=1,
                max_input_tokens=None,
                context=window_ctx,
            )
        except Exception as exc:
            logger.warning(f"Cross-chunk extraction failed for window {i}: {exc}")
            continue

        if not results:
            continue

        for extracted_rel in results[0].relationships:
            triple = (
                extracted_rel.source_entity.lower(),
                (extracted_rel.relationship_type or "").upper(),
                extracted_rel.target_entity.lower(),
            )
            if triple not in seen_triples:
                seen_triples.add(triple)
                new_relationships.append(extracted_rel)

    return new_relationships


async def stream_extract_and_embed_entities(
    chunks: list[Chunk],
    embedder: Embedder,
    *,
    skill_name: str = "general_entities",
    expertise: ExpertiseConfig | None = None,
    model: str = "gpt-4o-mini",
    max_concurrent_extractions: int = 20,
    extraction_context: dict[str, Any] | None = None,
    extraction_timeout: int = 120,
    extraction_max_retries: int = 3,
    extraction_retry_wait: float = 2.0,
    embedding_batch_size: int = 100,
    extraction_batch_size: int = 10,
    extraction_max_tokens: int | None = None,
    skip_embedding_entity_types: list[str] | None = None,
    skip_embedding_mention_threshold: int = 1,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities from chunks and embed them in a streaming fashion.

    This function overlaps extraction and embedding work by using an async queue.
    As entities are extracted from chunks, they are immediately queued for embedding,
    allowing the embedding process to start while extraction continues.

    Args:
        chunks: Chunks to extract entities from
        embedder: Embedder for generating entity embeddings
        skill_name: Extraction skill to use
        expertise: Optional ExpertiseConfig
        model: LLM model for extraction
        max_concurrent_extractions: Maximum concurrent LLM extractions
        extraction_context: Optional context for prompt templates
        extraction_timeout: Timeout for extraction calls
        extraction_max_retries: Max retries for extraction
        extraction_retry_wait: Base wait time between retries
        embedding_batch_size: Batch size for embedding operations
        extraction_batch_size: Max texts per LLM extraction call

    Returns:
        Tuple of (entities with embeddings, relationships)
    """
    from khora.core.models import Entity, Relationship
    from khora.core.models.entity import EntityType, RelationshipType
    from khora.extraction.extractors import LLMEntityExtractor
    from khora.extraction.skills.registry import get_default_registry

    if not chunks:
        return [], []

    # Entity queue for streaming between extraction and embedding
    entity_queue: asyncio.Queue[Entity | None] = asyncio.Queue()
    embedded_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    extraction_complete = asyncio.Event()

    # Resolve expertise and skill
    registry = get_default_registry()
    skill = registry.get_or_default(skill_name)

    if expertise:
        min_entity_confidence = expertise.confidence.min_entity
        min_relationship_confidence = expertise.confidence.min_relationship
    else:
        min_entity_confidence = skill.min_entity_confidence
        min_relationship_confidence = skill.min_relationship_confidence

    async def extraction_task() -> None:
        """Extract entities from chunks and queue them for embedding."""
        try:
            extractor_kwargs = dict(
                model=model,
                max_concurrent=max_concurrent_extractions,
                timeout=extraction_timeout,
                max_retries=extraction_max_retries,
                retry_wait=extraction_retry_wait,
            )
            if extraction_max_tokens is not None:
                extractor_kwargs["max_tokens"] = extraction_max_tokens
            extractor = LLMEntityExtractor(**extractor_kwargs)

            texts = [chunk.content for chunk in chunks]

            if expertise:
                results = await extractor.extract_multi(
                    texts,
                    expertise=expertise,
                    context=extraction_context,
                    batch_size=extraction_batch_size,
                    max_input_tokens=None,
                )
            else:
                results = await extractor.extract_multi(
                    texts,
                    entity_types=skill.entity_types,
                    batch_size=extraction_batch_size,
                    max_input_tokens=None,
                )

            # Process results and queue entities
            all_entities: dict[str, Entity] = {}
            chunk_entity_keys: dict[UUID, list[str]] = {}  # chunk_id -> entity keys

            # 2.10: Batch-normalize all entity names upfront (single FFI call)
            _all_names: set[str] = set()
            for _r in results:
                for _e in _r.entities:
                    _all_names.add(_e.name)
                for _rel in _r.relationships:
                    _all_names.add(_rel.source_entity)
                    _all_names.add(_rel.target_entity)
            _names_list = list(_all_names)
            _normalized_names = normalize_entity_names_batch(_names_list) if _names_list else []
            _norm_cache: dict[str, str] = dict(zip(_names_list, _normalized_names))

            for chunk, result in zip(chunks, results):
                chunk_keys: list[str] = []
                for extracted in result.entities:
                    if extracted.confidence < min_entity_confidence:
                        continue

                    key = f"{_norm_cache[extracted.name]}:{extracted.entity_type}"
                    chunk_keys.append(key)
                    if key in all_entities:
                        existing = all_entities[key]
                        existing.mention_count += 1
                        if chunk.document_id not in existing.source_document_ids:
                            existing.source_document_ids.append(chunk.document_id)
                        if chunk.id not in existing.source_chunk_ids:
                            existing.source_chunk_ids.append(chunk.id)
                        # Use earliest known valid_from (prefer LLM-extracted over chunk ts)
                        _t = extracted.temporal
                        _vf = _parse_temporal_date(_t.valid_from) if _t else None
                        candidate_from = _vf or chunk.created_at
                        if existing.valid_from and candidate_from and candidate_from < existing.valid_from:
                            existing.valid_from = candidate_from
                        # Widen valid_until to latest extracted value
                        _vu = _parse_temporal_date(_t.valid_until) if _t else None
                        if _vu and (not existing.valid_until or _vu > existing.valid_until):
                            existing.valid_until = _vu
                    else:
                        try:
                            entity_type: EntityType | str = EntityType(extracted.entity_type)
                        except ValueError:
                            entity_type = extracted.entity_type or "CONCEPT"

                        # Prefer LLM-extracted temporal bounds over chunk timestamp
                        _t = extracted.temporal
                        _vf = _parse_temporal_date(_t.valid_from) if _t else None
                        _vu = _parse_temporal_date(_t.valid_until) if _t else None

                        entity = Entity(
                            namespace_id=chunk.namespace_id,
                            name=_norm_cache[extracted.name],
                            entity_type=entity_type,
                            description=extracted.description,
                            attributes=extracted.attributes,
                            source_document_ids=[chunk.document_id],
                            source_chunk_ids=[chunk.id],
                            confidence=extracted.confidence,
                            valid_from=_vf or chunk.created_at,
                            valid_until=_vu,
                        )
                        all_entities[key] = entity
                        # Queue entity for embedding as soon as it's extracted
                        await entity_queue.put(entity)

                chunk_entity_keys[chunk.id] = chunk_keys

                # Process relationships
                for extracted_rel in result.relationships:
                    if extracted_rel.confidence < min_relationship_confidence:
                        continue

                    try:
                        rel_type: RelationshipType | str = RelationshipType(extracted_rel.relationship_type)
                    except ValueError:
                        rel_type = extracted_rel.relationship_type or "RELATES_TO"

                    norm_source = _norm_cache[extracted_rel.source_entity]
                    norm_target = _norm_cache[extracted_rel.target_entity]
                    source_key = _find_entity_key(norm_source, all_entities)
                    target_key = _find_entity_key(norm_target, all_entities)

                    if source_key and target_key:
                        # Prefer LLM-extracted temporal bounds for relationships
                        _rt = extracted_rel.temporal
                        _rvf = _parse_temporal_date(_rt.valid_from if _rt else None)
                        _rvu = _parse_temporal_date(_rt.valid_until if _rt else None)

                        relationship = Relationship(
                            namespace_id=chunk.namespace_id,
                            source_entity_id=all_entities[source_key].id,
                            target_entity_id=all_entities[target_key].id,
                            relationship_type=rel_type,
                            description=extracted_rel.description,
                            properties=extracted_rel.properties,
                            source_document_ids=[chunk.document_id],
                            source_chunk_ids=[chunk.id],
                            confidence=extracted_rel.confidence,
                            valid_from=_rvf or chunk.created_at,
                            valid_until=_rvu,
                        )
                        all_relationships.append(relationship)

            # 1.1: Add co-occurrence edges between entities in the same chunk
            existing_pairs: set[tuple[UUID, UUID]] = {
                (r.source_entity_id, r.target_entity_id) for r in all_relationships
            }
            existing_pairs |= {(r.target_entity_id, r.source_entity_id) for r in all_relationships}
            co_occurrence_count = 0
            _MAX_COOCCURRENCE_PER_CHUNK = 15  # Cap O(n^2) explosion
            for chunk in chunks:
                keys = chunk_entity_keys.get(chunk.id, [])
                unique_keys = list(dict.fromkeys(keys))  # dedupe, preserve order
                if len(unique_keys) < 2:
                    continue
                chunk_co_count = 0
                for i, key_a in enumerate(unique_keys):
                    if chunk_co_count >= _MAX_COOCCURRENCE_PER_CHUNK:
                        break
                    for key_b in unique_keys[i + 1 :]:
                        if chunk_co_count >= _MAX_COOCCURRENCE_PER_CHUNK:
                            break
                        if key_a not in all_entities or key_b not in all_entities:
                            continue
                        ent_a = all_entities[key_a]
                        ent_b = all_entities[key_b]
                        pair = (min(ent_a.id, ent_b.id), max(ent_a.id, ent_b.id))
                        if pair in existing_pairs or (pair[1], pair[0]) in existing_pairs:
                            continue
                        existing_pairs.add(pair)
                        all_relationships.append(
                            Relationship(
                                namespace_id=chunk.namespace_id,
                                source_entity_id=ent_a.id,
                                target_entity_id=ent_b.id,
                                relationship_type="ASSOCIATED_WITH",
                                description="Co-occurs in same chunk",
                                properties={},
                                source_document_ids=[chunk.document_id],
                                source_chunk_ids=[chunk.id],
                                confidence=0.4,
                            )
                        )
                        chunk_co_count += 1
                        co_occurrence_count += 1
            if co_occurrence_count:
                logger.debug(f"Added {co_occurrence_count} co-occurrence edges")

            # 3.3: Cross-chunk relationship extraction (opt-in, disabled by default)
            if extraction_context and extraction_context.get("cross_chunk_extraction", False):
                _entities_by_chunk: dict[UUID, list[str]] = {
                    cid: [k.split(":")[0] for k in keys] for cid, keys in chunk_entity_keys.items()
                }
                cross_rels_raw = await _extract_cross_chunk_relationships(
                    chunks, _entities_by_chunk, extractor, extraction_context
                )
                if cross_rels_raw:
                    logger.info(f"Cross-chunk extraction: found {len(cross_rels_raw)} candidate relationships")
                    added_cross = 0
                    for extracted_rel in cross_rels_raw:
                        norm_src = _norm_cache.get(
                            extracted_rel.source_entity,
                            extracted_rel.source_entity.lower().strip(),
                        )
                        norm_tgt = _norm_cache.get(
                            extracted_rel.target_entity,
                            extracted_rel.target_entity.lower().strip(),
                        )
                        source_key = _find_entity_key(norm_src, all_entities)
                        target_key = _find_entity_key(norm_tgt, all_entities)
                        if not source_key or not target_key:
                            continue
                        ent_a = all_entities[source_key]
                        ent_b = all_entities[target_key]
                        pair = (min(ent_a.id, ent_b.id), max(ent_a.id, ent_b.id))
                        if pair in existing_pairs or (pair[1], pair[0]) in existing_pairs:
                            continue
                        existing_pairs.add(pair)
                        try:
                            cross_rel_type: RelationshipType | str = RelationshipType(extracted_rel.relationship_type)
                        except ValueError:
                            cross_rel_type = extracted_rel.relationship_type or "RELATES_TO"
                        all_relationships.append(
                            Relationship(
                                namespace_id=chunks[0].namespace_id,
                                source_entity_id=ent_a.id,
                                target_entity_id=ent_b.id,
                                relationship_type=cross_rel_type,
                                description=extracted_rel.description,
                                properties=getattr(extracted_rel, "properties", {}),
                                source_document_ids=[chunks[0].document_id],
                                source_chunk_ids=[],
                                confidence=extracted_rel.confidence,
                            )
                        )
                        added_cross += 1
                    logger.debug(f"Cross-chunk extraction: added {added_cross} new relationships")

        finally:
            # Signal completion
            await entity_queue.put(None)
            extraction_complete.set()

    async def embedding_task() -> None:
        """Consume entities from queue and embed them in batches."""
        batch: list[Entity] = []

        async def embed_batch() -> None:
            if not batch:
                return
            try:
                texts = [f"{e.name}: {e.description}" if e.description else e.name for e in batch]
                embeddings = await embedder.embed_batch(texts)
                for entity, embedding in zip(batch, embeddings):
                    entity.embedding = embedding
                    embedded_entities.append(entity)
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")
                # Still add entities without embeddings
                embedded_entities.extend(batch)
            batch.clear()

        while True:
            try:
                # Use a timeout to allow periodic batch flushes
                entity = await asyncio.wait_for(entity_queue.get(), timeout=0.5)
            except TimeoutError:
                # Flush current batch on timeout
                await embed_batch()
                if extraction_complete.is_set() and entity_queue.empty():
                    break
                continue

            if entity is None:
                # Extraction complete, flush remaining batch
                await embed_batch()
                break

            # Skip embedding for low-value entity types
            _skip_types = skip_embedding_entity_types or []
            if _skip_types and _should_skip_entity_embedding(entity, _skip_types, skip_embedding_mention_threshold):
                embedded_entities.append(entity)  # Add without embedding
                continue

            batch.append(entity)
            if len(batch) >= embedding_batch_size:
                await embed_batch()

    # Run extraction and embedding concurrently
    await asyncio.gather(extraction_task(), embedding_task())

    return embedded_entities, all_relationships


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


def _parse_temporal_date(value: str | None) -> datetime | None:
    """Parse an ISO date string from LLM-extracted temporal info.

    Returns a datetime if parseable, None otherwise.
    """
    if not value:
        return None
    from datetime import datetime

    try:
        if "T" in value:
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value)
        return datetime.fromisoformat(value + "T00:00:00+00:00")
    except (ValueError, TypeError):
        return None


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


async def stage_documents_batch(
    doc_inputs: list[dict[str, Any]],
    namespace_id: UUID,
    storage: StorageCoordinator,
) -> list[Document | None]:
    """Batch stage documents with single-query checksum dedup.

    Computes all checksums upfront and checks for existing documents
    in a single batch query instead of N individual queries.

    Returns:
        List parallel to doc_inputs: Document if new, None if unchanged
    """
    from datetime import UTC, datetime

    from khora.core.models import Document, DocumentMetadata

    if not doc_inputs:
        return []

    # Compute all checksums upfront
    checksums = [compute_checksum(doc.get("content", "")) for doc in doc_inputs]

    # Single batch query for existing documents (replaces N individual queries)
    existing = await storage.get_documents_by_checksums(namespace_id, checksums)

    results: list[Document | None] = []
    for doc_input, checksum in zip(doc_inputs, checksums):
        if checksum in existing:
            logger.debug(f"Document unchanged (checksum={checksum[:8]}..., status={existing[checksum].status})")
            results.append(None)
            continue

        content = doc_input.get("content", "")
        custom_metadata = doc_input.get("metadata", {})
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

        source_timestamp = _extract_source_timestamp(custom_metadata)
        created_at = source_timestamp or datetime.now(UTC)

        document = Document(
            namespace_id=namespace_id,
            content=content,
            metadata=metadata,
            created_at=created_at,
            updated_at=created_at,
        )

        doc = await storage.create_document(document)
        results.append(doc)

    return results


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
    max_concurrent_extractions: int = 20,
    enable_expansion: bool = False,
    extraction_context: dict[str, Any] | None = None,
    entity_index: EntityIndex | None = None,
    shared_embedder: Any | None = None,
    temporal_store: TemporalVectorStore | None = None,
    extraction_timeout: int = 120,
    extraction_max_retries: int = 3,
    extraction_retry_wait: float = 2.0,
    extraction_batch_size: int = 10,
    extraction_max_tokens: int | None = None,
    skip_embedding_entity_types: list[str] | None = None,
    skip_embedding_mention_threshold: int = 1,
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
        extraction_batch_size: Max texts per LLM extraction call
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

        # T-2: Propagate document source timestamp to chunks
        if document.created_at:
            for chunk in chunks:
                chunk.created_at = document.created_at

        # R-2: Prepend document title for embeddings (better embedding space separation)
        doc_title = document.metadata.title if document.metadata and hasattr(document.metadata, "title") else ""

        # R-4: Propagate document title into chunk metadata for reranker context
        if doc_title:
            for chunk in chunks:
                if chunk.metadata and isinstance(chunk.metadata.custom, dict):
                    chunk.metadata.custom.setdefault("title", doc_title)

        original_contents: dict[UUID, str] = {}
        if doc_title:
            for chunk in chunks:
                original_contents[chunk.id] = chunk.content
                chunk.content = f"{doc_title}: {chunk.content}"

        # Steps 2 & 3: Embed + Extract concurrently (both depend only on chunks)
        async def _embed_with_telemetry():
            async with pipeline_stage(
                "ingestion", "embedding", _run_id, namespace_id=_ns_id, extra_metadata={"chunk_count": len(chunks)}
            ):
                return await embed_chunks(chunks, model=embedding_model, shared_embedder=shared_embedder)

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
                    timeout=extraction_timeout,
                    max_retries=extraction_max_retries,
                    retry_wait=extraction_retry_wait,
                    extraction_batch_size=extraction_batch_size,
                    max_tokens=extraction_max_tokens,
                )

        embedded_chunks, (entities, relationships) = await asyncio.gather(
            _embed_with_telemetry(), _extract_with_telemetry()
        )
        chunks = embedded_chunks

        # R-2: Restore original content after embedding
        if original_contents:
            for chunk in chunks:
                if chunk.id in original_contents:
                    chunk.content = original_contents[chunk.id]

        logger.debug(f"Document {document.id}: generated embeddings")
        logger.debug(f"Document {document.id}: {len(entities)} entities, {len(relationships)} relationships extracted")

        # Step 4 (Optional): Semantic expansion
        inferred_relationships = []
        inference_mode = resolved_expertise.expansion.inference_mode if resolved_expertise else "none"

        dedup_id_mapping: dict[str, str] = {}

        if entity_index is not None and inference_mode == "smart":
            # Smart mode: within-doc exact dedup via shared EntityIndex.
            # Cross-document resolution + inference deferred to run_smart_resolution().
            deduped_entities = []
            for entity in entities:
                existing = entity_index.add(entity)
                if existing is not None:
                    # Merge into existing (already in index)
                    existing.merge_with(entity)
                    # Map the dropped entity's ID to the surviving entity's ID
                    dedup_id_mapping[str(entity.id)] = str(existing.id)
                else:
                    deduped_entities.append(entity)
            if len(entities) != len(deduped_entities):
                logger.debug(
                    f"Document {document.id}: smart dedup {len(entities)} -> {len(deduped_entities)} entities "
                    f"({len(dedup_id_mapping)} cross-doc duplicates)"
                )
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

        # Step 4 & 5: Store chunks and entities in parallel
        # Chunks go to pgvector, entities go to graph+vector - independent writes
        async def _store_chunks():
            """Store chunks to vector backend and optional temporal store."""
            async with pipeline_stage(
                "ingestion",
                "storage",
                _run_id,
                namespace_id=_ns_id,
                extra_metadata={"chunk_count": len(chunks), "entity_count": len(entities)},
            ):
                await storage.create_chunks_batch(chunks)

                # Also write to khora_chunks for VectorCypher/Skeleton engines
                if temporal_store is not None:
                    from datetime import UTC, datetime

                    from khora.engines.skeleton.backends import TemporalChunk

                    doc_metadata: dict[str, Any] = {}
                    if document.metadata and document.metadata.custom:
                        doc_metadata = document.metadata.custom

                    occurred_at = _extract_source_timestamp(doc_metadata) or getattr(document, "created_at", None)

                    temporal_chunks = [
                        TemporalChunk(
                            id=chunk.id,
                            namespace_id=chunk.namespace_id,
                            document_id=chunk.document_id,
                            content=chunk.content,
                            embedding=chunk.embedding,
                            occurred_at=occurred_at,
                            created_at=datetime.now(UTC),
                            source_system=doc_metadata.get("source_system"),
                            author=doc_metadata.get("author"),
                            channel=doc_metadata.get("channel"),
                            tags=doc_metadata.get("tags", []),
                            confidence=1.0,
                            metadata=(
                                {
                                    "chunk_index": chunk.metadata.chunk_index,
                                    "start_char": chunk.metadata.start_char,
                                    "end_char": chunk.metadata.end_char,
                                    "token_count": chunk.metadata.token_count,
                                    **chunk.metadata.custom,
                                }
                                if chunk.metadata
                                else {}
                            ),
                        )
                        for chunk in chunks
                    ]
                    await temporal_store.create_chunks_batch(temporal_chunks)

        async def _store_entities() -> tuple[list[tuple[Entity, bool]], dict[str, str], list[Entity]]:
            """Store entities with deduplication. Returns (results, id_mapping, entities_needing_embeddings)."""
            async with pipeline_stage(
                "ingestion",
                "entity_storage",
                _run_id,
                namespace_id=_ns_id,
                input_count=len(entities),
            ) as _es_ctx:
                # Track mapping from original entity IDs to stored entity IDs (for dedup)
                entity_id_mapping: dict[str, str] = {}
                # Pre-seed with cross-document dedup mappings from smart mode
                if entity_index is not None and inference_mode == "smart":
                    entity_id_mapping.update(dedup_id_mapping)

                # Save pre-upsert IDs (Neo4j may sync entity.id to a different value on MERGE)
                pre_upsert_ids = [str(e.id) for e in entities]
                logger.debug(f"Document {document.id}: upserting {len(entities)} entities")

                # Batch upsert: single MERGE operation instead of N+1 individual lookups
                upsert_results = await storage.upsert_entities_batch(document.namespace_id, entities)

                logger.debug(
                    f"Document {document.id}: upsert returned {len(upsert_results)} results "
                    f"(expected {len(pre_upsert_ids)})"
                )
                if len(upsert_results) != len(pre_upsert_ids):
                    # Log details to diagnose the mismatch
                    result_ids = [str(e.id) for e, _ in upsert_results]
                    missing_ids = set(pre_upsert_ids) - set(result_ids)
                    extra_ids = set(result_ids) - set(pre_upsert_ids)
                    logger.warning(
                        f"Document {document.id}: upsert result count mismatch - "
                        f"sent {len(pre_upsert_ids)}, got {len(upsert_results)} "
                        f"(missing: {len(missing_ids)}, extra: {len(extra_ids)})"
                    )

                store_results: list[tuple[Entity, bool]] = []

                # Batch-normalize entity names for mapping (single FFI call)
                _store_names = list({e.name for e, _ in upsert_results} | {e.name for e in entities})
                _store_normalized = normalize_entity_names_batch(_store_names) if _store_names else []
                _store_norm = dict(zip(_store_names, _store_normalized))

                # Build name+type -> stored_id mapping from upsert results
                name_type_to_stored: dict[str, str] = {}
                for entity, is_new in upsert_results:
                    et = entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type)
                    key = f"{_store_norm[entity.name]}:{et}"
                    stored_id = str(entity.id)
                    name_type_to_stored[key] = stored_id
                    entity_id_mapping[stored_id] = stored_id
                    needs_embedding = is_new or not entity.embedding
                    store_results.append((entity, needs_embedding))

                # Map every original entity ID to its stored counterpart by name+type
                for orig_entity in entities:
                    et = (
                        orig_entity.entity_type.value
                        if hasattr(orig_entity.entity_type, "value")
                        else str(orig_entity.entity_type)
                    )
                    key = f"{_store_norm[orig_entity.name]}:{et}"
                    stored_id = name_type_to_stored.get(key)
                    if stored_id:
                        entity_id_mapping[str(orig_entity.id)] = stored_id
                # Collect entities that need embeddings
                entities_needing = [e for e, needs in store_results if needs]
                _es_ctx["output_count"] = len(store_results)
                return store_results, entity_id_mapping, entities_needing

        # Run chunk and entity storage in parallel
        _, (store_results, entity_id_mapping, entities_needing_embeddings) = await asyncio.gather(
            _store_chunks(),
            _store_entities(),
        )

        # Step 5b + Step 6: Entity embeddings and relationship storage run in parallel
        # since relationships don't depend on entity embeddings

        async def _embed_entities() -> int:
            """Generate and store entity embeddings. Returns count embedded."""
            if not entities_needing_embeddings:
                return 0

            # Filter out low-value entity types that don't benefit from vector search
            _skip_types = skip_embedding_entity_types or []
            if _skip_types:
                embeddable = [
                    e
                    for e in entities_needing_embeddings
                    if not _should_skip_entity_embedding(e, _skip_types, skip_embedding_mention_threshold)
                ]
                skipped = len(entities_needing_embeddings) - len(embeddable)
                if skipped:
                    logger.debug(
                        f"Document {document.id}: skipped embedding for {skipped} low-value entities "
                        f"(types={_skip_types}, threshold={skip_embedding_mention_threshold})"
                    )
            else:
                embeddable = entities_needing_embeddings

            if not embeddable:
                return 0

            async with pipeline_stage(
                "ingestion",
                "entity_embedding",
                _run_id,
                namespace_id=_ns_id,
                input_count=len(embeddable),
            ) as _ee_ctx:
                from khora.extraction.embedders import LiteLLMEmbedder

                embedder = shared_embedder or LiteLLMEmbedder(model=embedding_model)
                entity_texts = [f"{e.name}: {e.description}" if e.description else e.name for e in embeddable]
                entity_embeddings = await embedder.embed_batch(entity_texts)
                updates = [
                    (entity.id, embedding, embedding_model) for entity, embedding in zip(embeddable, entity_embeddings)
                ]
                await storage.update_entity_embeddings_batch(updates)
                _ee_ctx["output_count"] = len(embeddable)
            logger.debug(f"Document {document.id}: generated embeddings for {len(embeddable)} entities")
            return len(embeddable)

        async def _store_relationships() -> tuple[int, int]:
            """Remap and batch-store relationships. Returns (stored_count, skipped)."""
            all_relationships = relationships + inferred_relationships
            if not all_relationships:
                return 0, 0
            from uuid import UUID

            valid_relationships = []
            skipped = 0
            for rel in all_relationships:
                source_id = str(rel.source_entity_id)
                target_id = str(rel.target_entity_id)

                mapped_source = entity_id_mapping.get(source_id)
                mapped_target = entity_id_mapping.get(target_id)

                if not mapped_source or not mapped_target:
                    skipped += 1
                    continue

                rel.source_entity_id = UUID(mapped_source)
                rel.target_entity_id = UUID(mapped_target)
                valid_relationships.append(rel)

            count = 0
            if valid_relationships:
                count = await storage.create_relationships_batch(valid_relationships)

            if skipped > 0:
                logger.warning(
                    f"Stored {count}/{len(all_relationships)} relationships "
                    f"({skipped} skipped due to missing entity mappings)"
                )
            return count, skipped

        # Run embedding and relationship storage concurrently
        _, (stored_count, _skipped) = await asyncio.gather(
            _embed_entities(),
            _store_relationships(),
        )

        # Mark as completed
        document.mark_completed(len(chunks), len(entities))
        await storage.update_document(document)

        return {
            "document_id": str(document.id),
            "chunks": len(chunks),
            "entities": len(entities),
            "relationships": stored_count,
            "extracted_relationships": len(relationships),
            "inferred_relationships": len(inferred_relationships),
        }

    except Exception as e:
        document.mark_failed(str(e))
        await storage.update_document(document)
        raise


@pipeline("ingest", description="Two-phase document ingestion with optional expansion", tags=["ingestion"])
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
    max_concurrent_documents: int = 10,
    max_concurrent_extractions: int = 20,
    enable_expansion: bool = False,
    extraction_context: dict[str, Any] | None = None,
    skip_resolution: bool = False,
    shared_embedder: Any | None = None,
    shared_entity_index: Any | None = None,
    temporal_store: TemporalVectorStore | None = None,
    extraction_timeout: int = 120,
    extraction_max_retries: int = 3,
    extraction_retry_wait: float = 2.0,
    extraction_batch_size: int = 10,
    extraction_max_tokens: int | None = None,
    skip_embedding_entity_types: list[str] | None = None,
    skip_embedding_mention_threshold: int = 1,
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
        skip_resolution: If True, skip Phase 3 (smart resolution). Useful when the
            caller runs resolution separately after all batches are processed.
        extraction_batch_size: Max texts per LLM extraction call (default 10)
        extraction_max_tokens: Max tokens for LLM extraction response. If None, uses extractor default.

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
    if shared_entity_index is not None:
        pass  # Caller provided a pre-populated index; use it as-is
    elif is_smart and resolved_expertise:
        from khora.extraction.expansion.entity_index import EntityIndex as EI

        shared_entity_index = EI()
        if resolved_expertise.expansion.preload_existing:
            existing_entities = await storage.list_entities(namespace_id, limit=50000)
            for e in existing_entities:
                shared_entity_index.add(e)
            if existing_entities:
                logger.info(f"Smart mode: pre-loaded {len(existing_entities)} existing entities into index")

    # Phase 1: Batch checksum dedup + stage new documents
    # Single batch query replaces N individual get_document_by_checksum calls.
    staged_results = await stage_documents_batch(documents, namespace_id, storage)
    staged_docs = [doc for doc in staged_results if doc is not None]

    # Build mapping from staged doc ID to original dict for per-doc overrides
    # (e.g. _skill_name, _extraction_context set by callers like Genesis)
    doc_originals: dict[UUID, dict[str, Any]] = {}
    for orig, staged in zip(documents, staged_results):
        if staged is not None:
            doc_originals[staged.id] = orig

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
    # Share a single embedder across all documents to preserve the embedding cache
    if shared_embedder is None:
        from khora.extraction.embedders import LiteLLMEmbedder

        shared_embedder = LiteLLMEmbedder(model=embedding_model)

    doc_semaphore = asyncio.Semaphore(max_concurrent_documents)

    async def process_with_limit(doc):
        async with doc_semaphore:
            # Check for per-doc overrides from the original document dict
            orig = doc_originals.get(doc.id, {})
            doc_skill = orig.get("_skill_name", skill_name)
            doc_context = orig.get("_extraction_context", extraction_context)
            doc_model = orig.get("_extraction_model", extraction_model)
            return await process_document(
                doc,
                storage,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                embedding_model=embedding_model,
                extraction_model=doc_model,
                skill_name=doc_skill,
                expertise=expertise,
                max_concurrent_extractions=max_concurrent_extractions,
                enable_expansion=enable_expansion,
                extraction_context=doc_context,
                entity_index=shared_entity_index,
                shared_embedder=shared_embedder,
                temporal_store=temporal_store,
                extraction_timeout=extraction_timeout,
                extraction_max_retries=extraction_max_retries,
                extraction_retry_wait=extraction_retry_wait,
                extraction_batch_size=extraction_batch_size,
                extraction_max_tokens=extraction_max_tokens,
                skip_embedding_entity_types=skip_embedding_entity_types,
                skip_embedding_mention_threshold=skip_embedding_mention_threshold,
            )

    results = await asyncio.gather(
        *[process_with_limit(doc) for doc in staged_docs],
        return_exceptions=True,
    )

    if shared_entity_index is not None:
        logger.info(f"Entity index size after processing: {len(shared_entity_index)} entities")

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
    if is_smart and shared_entity_index and resolved_expertise and successful_results and not skip_resolution:
        logger.info("Starting smart post-ingestion resolution...")
        smart_resolution_result = await run_smart_resolution(
            namespace_id,
            storage,
            shared_entity_index,
            resolved_expertise,
            embedding_model=embedding_model,
            shared_embedder=shared_embedder,
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


async def run_smart_resolution(
    namespace_id: UUID,
    storage: StorageCoordinator,
    entity_index: EntityIndex,
    expertise: ExpertiseConfig,
    *,
    embedding_model: str = "text-embedding-3-small",
    shared_embedder: Any | None = None,
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
    logger.info(f"Smart resolution: {len(all_entities)} entities in index ({entity_index.stats()})")

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

    logger.info(f"Smart resolution: unified {len(all_entities)} -> {len(resolved_entities)} ({entities_merged} merged)")

    # Phase 2: Batch upsert resolved entities to storage
    batch_size = expertise.expansion.batch_storage_size
    await storage.upsert_entities_batch(namespace_id, resolved_entities, batch_size=batch_size)

    # Generate embeddings for entities missing them
    entities_needing_embeddings = [e for e in resolved_entities if not e.embedding]
    if entities_needing_embeddings:
        from khora.extraction.embedders import LiteLLMEmbedder

        embedder = shared_embedder or LiteLLMEmbedder(model=embedding_model)
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

    # --- Inference diagnostics ---
    from collections import Counter

    logger.info(f"Smart resolution: loaded {len(relationships)} relationships from storage")

    if relationships:
        rel_type_dist = Counter(
            str(r.relationship_type.value if hasattr(r.relationship_type, "value") else r.relationship_type)
            for r in relationships
        )
        logger.info(f"Smart resolution: relationship types: {dict(rel_type_dist.most_common(15))}")

    if resolved_entities:
        ent_type_dist = Counter(
            str(e.entity_type.value if hasattr(e.entity_type, "value") else e.entity_type) for e in resolved_entities
        )
        logger.info(f"Smart resolution: entity types: {dict(ent_type_dist.most_common(15))}")

    # Check entity ID overlap between relationships and resolved_entities
    rel_entity_ids = set()
    for rel in relationships:
        rel_entity_ids.add(str(rel.source_entity_id))
        rel_entity_ids.add(str(rel.target_entity_id))
    resolved_ids = {str(e.id) for e in resolved_entities}
    matched = len(rel_entity_ids & resolved_ids)
    unmatched = len(rel_entity_ids - resolved_ids)
    logger.info(
        f"Smart resolution: entity ID overlap: {matched}/{len(rel_entity_ids)} "
        f"({unmatched} relationship entity IDs NOT in resolved entities)"
    )

    # Debug: show sample IDs if overlap is suspiciously low
    if len(rel_entity_ids) > 0 and matched / len(rel_entity_ids) < 0.5:
        sample_rel_ids = list(rel_entity_ids)[:5]
        sample_resolved_ids = list(resolved_ids)[:5]
        logger.warning(
            f"Low entity ID overlap! Sample rel IDs: {sample_rel_ids}, Sample resolved IDs: {sample_resolved_ids}"
        )

    # Phase 4: Relationship inference on full resolved graph (single pass)
    from khora.extraction.expansion.relationship_inferrer import RelationshipInferrer
    from khora.extraction.expansion.rule_engine import RuleEvaluationContext

    inferrer = RelationshipInferrer(
        expertise=expertise,
        min_confidence=expertise.confidence.min_inferred,
    )

    # Pre-check: count rule engine matches (for diagnostics)
    rule_engine_matches = 0
    if expertise.inference_rules:
        context = RuleEvaluationContext.from_data(resolved_entities, relationships)
        matches = inferrer._rule_engine.evaluate_inference_rules(context)
        rule_engine_matches = len(matches)
        logger.info(f"Smart resolution: rule engine produced {rule_engine_matches} raw matches")

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

    # Build diagnostics for caller
    diagnostics = {
        "relationships_loaded": len(relationships),
        "entities_resolved": len(resolved_entities),
        "relationship_types": dict(rel_type_dist) if relationships else {},
        "entity_types": dict(ent_type_dist) if resolved_entities else {},
        "entity_id_overlap": {
            "matched": matched,
            "total": len(rel_entity_ids),
            "unmatched": unmatched,
        },
        "rule_engine_matches": rule_engine_matches,
    }

    return {
        "entities_resolved": len(resolved_entities),
        "entities_merged": entities_merged,
        "inferred_relationships": inferred_count,
        "diagnostics": diagnostics,
    }


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

        # Update entities in batch
        updates = [(entity.id, embedding, embedding_model) for entity, embedding in zip(batch, embeddings)]
        total_updated += await storage.update_entity_embeddings_batch(updates)

        logger.debug(f"Updated {total_updated}/{len(entities_needing_embeddings)} entity embeddings")

    logger.info(f"Entity embedding backfill complete: updated {total_updated} entities")

    return {
        "total_entities": len(entities),
        "entities_updated": total_updated,
    }
