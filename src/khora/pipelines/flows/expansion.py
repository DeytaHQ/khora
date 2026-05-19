"""Standalone semantic expansion flow for knowledge graph enhancement.

Runs semantic expansion (entity unification, relationship inference) on
existing entities and relationships in a namespace.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from ..registry import pipeline

if TYPE_CHECKING:
    from khora.core.models import Entity, Relationship
    from khora.extraction.expansion import ExpansionResult
    from khora.extraction.skills import ExpertiseConfig
    from khora.storage import StorageCoordinator


async def load_entities(
    namespace_id: UUID,
    storage: StorageCoordinator,
    *,
    limit: int = 1000,
) -> list[Entity]:
    """Load entities from storage for expansion."""
    return await storage.list_entities(namespace_id, limit=limit)


async def load_relationships(
    namespace_id: UUID,
    storage: StorageCoordinator,
    *,
    limit: int = 5000,
) -> list[Relationship]:
    """Load relationships from storage for expansion."""
    return await storage.list_relationships(namespace_id, limit=limit)


async def run_expansion(
    entities: list[Entity],
    relationships: list[Relationship],
    namespace_id: UUID,
    expertise: ExpertiseConfig | None = None,
    *,
    inference_depth: int = 2,
) -> ExpansionResult:
    """Run semantic expansion on entities and relationships.

    Args:
        entities: Entities to expand
        relationships: Existing relationships
        namespace_id: Namespace ID
        expertise: Expertise configuration
        inference_depth: Number of inference passes

    Returns:
        Expansion result
    """
    from khora.extraction.expansion import SemanticExpander

    expander = SemanticExpander(
        expertise=expertise,
        inference_depth=inference_depth,
    )

    return await expander.expand(
        entities=entities,
        relationships=relationships,
        namespace_id=namespace_id,
    )


async def store_expansion_results(
    result: ExpansionResult,
    storage: StorageCoordinator,
) -> dict[str, int]:
    """Store expansion results (merged entities, inferred relationships).

    Args:
        result: Expansion result to store
        storage: Storage coordinator

    Returns:
        Statistics about stored items
    """
    stored_entities = 0
    stored_relationships = 0

    # Update merged entities
    entity_semaphore = asyncio.Semaphore(40)

    async def update_entity(entity):
        nonlocal stored_entities
        async with entity_semaphore:
            await storage.update_entity(entity, namespace_id=entity.namespace_id)
            stored_entities += 1

    if result.merged_entity_count > 0:
        # Only update entities that were actually modified by merges
        merged_ids = set(result.entity_mapping.values()) if result.entity_mapping else set()
        modified_entities = [e for e in result.entities if e.id in merged_ids] if merged_ids else result.entities
        await asyncio.gather(*[update_entity(e) for e in modified_entities])

    # Store inferred relationships in batch
    if result.inferred_relationships:
        stored_relationships = await storage.create_relationships_batch(result.inferred_relationships)

    return {
        "updated_entities": stored_entities,
        "stored_relationships": stored_relationships,
    }


@pipeline("expand_knowledge", description="Semantic expansion of knowledge graph", tags=["expansion", "enrichment"])
async def expand_knowledge_graph(
    namespace_id: UUID,
    storage: StorageCoordinator | None = None,
    *,
    expertise: ExpertiseConfig | str | None = None,
    inference_depth: int = 2,
    max_entities: int = 1000,
    max_relationships: int = 5000,
    store_results: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Expand a namespace's knowledge graph with semantic enrichment.

    Runs semantic expansion on existing entities and relationships:
    - Cross-tool entity unification (merge duplicates)
    - Relationship inference (infer new relationships from patterns)

    Args:
        namespace_id: Target namespace
        storage: StorageCoordinator instance
        expertise: ExpertiseConfig, expertise name, or file path
        inference_depth: Number of inference passes
        max_entities: Maximum entities to process
        max_relationships: Maximum relationships to process
        store_results: Whether to persist results to storage

    Returns:
        Summary of expansion results
    """
    if storage is None:
        raise ValueError("storage is required")

    # Resolve expertise
    resolved_expertise: ExpertiseConfig | None = None
    if expertise is not None:
        from khora.extraction.skills import ExpertiseConfig as EC
        from khora.extraction.skills import load_expertise

        if isinstance(expertise, EC):
            resolved_expertise = expertise
        elif isinstance(expertise, str):
            try:
                resolved_expertise = load_expertise(expertise)
            except Exception as e:
                logger.warning(f"Failed to load expertise '{expertise}': {e}")

    logger.info(f"Starting knowledge graph expansion for namespace {namespace_id}")

    # Load existing data in parallel
    entities, relationships = await asyncio.gather(
        load_entities(namespace_id, storage, limit=max_entities),
        load_relationships(namespace_id, storage, limit=max_relationships),
    )

    logger.info(f"Loaded {len(entities)} entities and {len(relationships)} relationships")

    if not entities:
        return {
            "original_entities": 0,
            "original_relationships": 0,
            "unified_entities": 0,
            "merged_count": 0,
            "inferred_relationships": 0,
            "stored": False,
        }

    # Run expansion
    result = await run_expansion(
        entities,
        relationships,
        namespace_id,
        expertise=resolved_expertise,
        inference_depth=inference_depth,
    )

    logger.info(
        f"Expansion complete: {result.total_entities} entities "
        f"({result.merged_entity_count} merged), "
        f"{result.inferred_relationship_count} relationships inferred"
    )

    # Store results if requested
    stored = False
    if store_results and (result.merged_entity_count > 0 or result.inferred_relationship_count > 0):
        store_stats = await store_expansion_results(result, storage)
        stored = True
        logger.info(
            f"Stored expansion results: {store_stats['updated_entities']} entities, "
            f"{store_stats['stored_relationships']} relationships"
        )

    return {
        "original_entities": result.original_entity_count,
        "original_relationships": result.original_relationship_count,
        "unified_entities": result.total_entities,
        "merged_count": result.merged_entity_count,
        "inferred_relationships": result.inferred_relationship_count,
        "stored": stored,
    }


@pipeline("unify_entities", description="Cross-tool entity unification only", tags=["expansion", "unification"])
async def unify_entities(
    namespace_id: UUID,
    storage: StorageCoordinator | None = None,
    *,
    expertise: ExpertiseConfig | str | None = None,
    max_entities: int = 1000,
    store_results: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Unify entities across tools without relationship inference.

    A lighter-weight expansion that only runs cross-tool entity unification
    to merge duplicate entities.

    Args:
        namespace_id: Target namespace
        storage: StorageCoordinator instance
        expertise: ExpertiseConfig with correlation rules
        max_entities: Maximum entities to process
        store_results: Whether to persist results to storage

    Returns:
        Summary of unification results
    """
    if storage is None:
        raise ValueError("storage is required")

    # Resolve expertise
    resolved_expertise: ExpertiseConfig | None = None
    if expertise is not None:
        from khora.extraction.skills import ExpertiseConfig as EC
        from khora.extraction.skills import load_expertise

        if isinstance(expertise, EC):
            resolved_expertise = expertise
        elif isinstance(expertise, str):
            try:
                resolved_expertise = load_expertise(expertise)
            except Exception as e:
                logger.warning(f"Failed to load expertise '{expertise}': {e}")

    logger.info(f"Starting entity unification for namespace {namespace_id}")

    # Load entities and relationships
    entities = await load_entities(namespace_id, storage, limit=max_entities)
    relationships = await load_relationships(namespace_id, storage, limit=max_entities * 5)

    if not entities:
        return {
            "original_entities": 0,
            "unified_entities": 0,
            "merged_count": 0,
            "stored": False,
        }

    # Run unification only
    from khora.extraction.expansion import CrossToolUnifier

    unifier = CrossToolUnifier(expertise=resolved_expertise)
    result = await unifier.unify(entities, relationships, storage=storage)

    logger.info(f"Unification complete: {len(result.unified_entities)} entities ({result.entities_merged} merged)")

    # Store results if requested
    stored = False
    if store_results and result.entities_merged > 0:
        import asyncio

        entity_semaphore = asyncio.Semaphore(40)

        async def update_entity(entity):
            async with entity_semaphore:
                await storage.update_entity(entity, namespace_id=entity.namespace_id)

        await asyncio.gather(*[update_entity(e) for e in result.unified_entities])
        stored = True

    return {
        "original_entities": len(entities),
        "unified_entities": len(result.unified_entities),
        "merged_count": result.entities_merged,
        "merge_groups": len(result.merge_groups),
        "stored": stored,
    }
