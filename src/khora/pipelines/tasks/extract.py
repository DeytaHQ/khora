"""Entity extraction task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from khora.core.models import Chunk, Entity, Relationship
    from khora.extraction.skills import ExpertiseConfig


async def extract_entities(
    chunks: list[Chunk],
    *,
    skill_name: str = "general_entities",
    expertise: ExpertiseConfig | str | None = None,
    model: str = "gpt-4o-mini",
    max_concurrent: int = 10,
    context: dict[str, Any] | None = None,
    timeout: int = 120,
    max_retries: int = 3,
    retry_wait: float = 2.0,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from chunks.

    Uses batch extraction for parallel processing of multiple chunks.
    Supports both legacy skills and new expertise configurations.

    Args:
        chunks: Chunks to extract from
        skill_name: Extraction skill to use (legacy, ignored if expertise provided)
        expertise: ExpertiseConfig, expertise name string, or file path
        model: LLM model for extraction
        max_concurrent: Maximum concurrent extractions
        context: Optional context dict for prompt template rendering
        timeout: Request timeout in seconds
        max_retries: Maximum retries on failure
        retry_wait: Base wait time for exponential backoff between retries

    Returns:
        Tuple of (entities, relationships)
    """
    from khora.core.models import Entity, Relationship
    from khora.core.models.entity import EntityType, RelationshipType
    from khora.extraction.extractors import LLMEntityExtractor
    from khora.extraction.skills import ExpertiseConfig
    from khora.extraction.skills.registry import get_default_registry

    if not chunks:
        return [], []

    # Resolve expertise configuration
    resolved_expertise: ExpertiseConfig | None = None
    if expertise is not None:
        if isinstance(expertise, ExpertiseConfig):
            resolved_expertise = expertise
        elif isinstance(expertise, str):
            # Load from string (file path or builtin name)
            from khora.extraction.skills import load_expertise

            try:
                resolved_expertise = load_expertise(expertise)
            except Exception:
                # Fall back to registry lookup
                registry = get_default_registry()
                resolved_expertise = registry.get_expertise(expertise)

    # Get legacy skill for backward compatibility
    registry = get_default_registry()
    skill = registry.get_or_default(skill_name)

    # If expertise provided, use its confidence thresholds
    if resolved_expertise:
        min_entity_confidence = resolved_expertise.confidence.min_entity
        min_relationship_confidence = resolved_expertise.confidence.min_relationship
    else:
        min_entity_confidence = skill.min_entity_confidence
        min_relationship_confidence = skill.min_relationship_confidence

    # Create extractor with concurrency limit and timeout settings
    extractor = LLMEntityExtractor(
        model=model,
        max_concurrent=max_concurrent,
        timeout=timeout,
        max_retries=max_retries,
        retry_wait=retry_wait,
    )

    # Extract from all chunks using grouped multi-extraction (fewer LLM calls)
    # Groups 3-5 chunks per LLM call, reducing API round-trips proportionally
    texts = [chunk.content for chunk in chunks]

    if resolved_expertise:
        results = await extractor.extract_multi(
            texts,
            expertise=resolved_expertise,
            context=context,
            batch_size=1,
        )
    else:
        results = await extractor.extract_multi(
            texts,
            entity_types=skill.entity_types,
            batch_size=1,
        )

    # Process results
    all_entities: dict[str, Entity] = {}  # name -> entity (for dedup)
    all_relationships: list[Relationship] = []

    for chunk, result in zip(chunks, results):
        # Process entities
        for extracted in result.entities:
            if extracted.confidence < min_entity_confidence:
                continue

            # Deduplicate by name
            key = f"{extracted.name}:{extracted.entity_type}"
            if key in all_entities:
                # Merge into existing
                existing = all_entities[key]
                existing.mention_count += 1
                if chunk.document_id not in existing.source_document_ids:
                    existing.source_document_ids.append(chunk.document_id)
                if chunk.id not in existing.source_chunk_ids:
                    existing.source_chunk_ids.append(chunk.id)
                # Update valid_from to earliest timestamp
                if existing.valid_from and chunk.created_at < existing.valid_from:
                    existing.valid_from = chunk.created_at
            else:
                # Create new entity — preserve original type string from LLM
                try:
                    entity_type: EntityType | str = EntityType(extracted.entity_type)
                except ValueError:
                    entity_type = extracted.entity_type or "CONCEPT"

                entity = Entity(
                    namespace_id=chunk.namespace_id,
                    name=extracted.name,
                    entity_type=entity_type,
                    description=extracted.description,
                    attributes=extracted.attributes,
                    source_document_ids=[chunk.document_id],
                    source_chunk_ids=[chunk.id],
                    confidence=extracted.confidence,
                    valid_from=chunk.created_at,  # Inherit source timestamp
                )
                all_entities[key] = entity

        # Process relationships
        for extracted_rel in result.relationships:
            if extracted_rel.confidence < min_relationship_confidence:
                continue

            # Preserve original type string from LLM
            try:
                rel_type: RelationshipType | str = RelationshipType(extracted_rel.relationship_type)
            except ValueError:
                rel_type = extracted_rel.relationship_type or "RELATES_TO"

            # Find source and target entities
            source_key = next(
                (k for k in all_entities if k.startswith(f"{extracted_rel.source_entity}:")),
                None,
            )
            target_key = next(
                (k for k in all_entities if k.startswith(f"{extracted_rel.target_entity}:")),
                None,
            )

            if source_key and target_key:
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
                    valid_from=chunk.created_at,  # Inherit source timestamp
                )
                all_relationships.append(relationship)

    return list(all_entities.values()), all_relationships
