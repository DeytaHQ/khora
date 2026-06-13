"""Semantic expander for knowledge graph enhancement.

Orchestrates cross-tool entity unification and relationship inference
to enrich extracted knowledge graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from .cross_tool_unifier import CrossToolUnifier
from .relationship_inferrer import RelationshipInferrer, to_relationship

if TYPE_CHECKING:
    from khora.core.models import Entity, Relationship
    from khora.extraction.skills import ExpertiseConfig

    from .entity_index import EntityIndex


@dataclass
class ExpansionResult:
    """Result of semantic expansion."""

    # Unified entities (after deduplication)
    entities: list[Entity] = field(default_factory=list)

    # Updated existing relationships
    relationships: list[Relationship] = field(default_factory=list)

    # New inferred relationships
    inferred_relationships: list[Relationship] = field(default_factory=list)

    # Statistics
    original_entity_count: int = 0
    merged_entity_count: int = 0
    original_relationship_count: int = 0
    inferred_relationship_count: int = 0

    # Mapping for provenance tracking
    entity_mapping: dict[UUID, UUID] = field(default_factory=dict)

    @property
    def total_entities(self) -> int:
        """Total entities after expansion."""
        return len(self.entities)

    @property
    def total_relationships(self) -> int:
        """Total relationships after expansion."""
        return len(self.relationships) + len(self.inferred_relationships)

    @property
    def all_relationships(self) -> list[Relationship]:
        """All relationships (existing + inferred)."""
        return self.relationships + self.inferred_relationships


class SemanticExpander:
    """Orchestrates semantic expansion of knowledge graphs.

    Combines:
    - Cross-tool entity unification
    - Relationship inference
    - LLM-powered expansion (future)

    Example usage:
        from khora.extraction.expansion import SemanticExpander
        from khora.extraction.skills import load_expertise

        expertise = load_expertise("saas_expert")
        expander = SemanticExpander(expertise=expertise)

        result = await expander.expand(
            entities=extracted_entities,
            relationships=extracted_relationships,
        )
    """

    def __init__(
        self,
        expertise: ExpertiseConfig | None = None,
        *,
        enable_unification: bool | None = None,
        enable_inference: bool | None = None,
        inference_depth: int = 2,
        embedding_threshold: float = 0.85,
        fuzzy_threshold: float = 0.85,
        min_inference_confidence: float = 0.3,
    ) -> None:
        """Initialize the semantic expander.

        Args:
            expertise: ExpertiseConfig with expansion rules
            enable_unification: Whether to run cross-tool unification. ``None`` (default)
                defers to the expertise config; an explicit ``True``/``False`` wins.
            enable_inference: Whether to run relationship inference. ``None`` (default)
                defers to the expertise config; an explicit ``True``/``False`` wins.
            inference_depth: Number of inference passes
            embedding_threshold: Similarity threshold for embedding matching
            fuzzy_threshold: Threshold for fuzzy string matching
            min_inference_confidence: Minimum confidence for inferred relationships
        """
        self._expertise = expertise
        self._inference_depth = inference_depth

        # Honor explicit kwargs (#1124). The expertise config supplies defaults ONLY when
        # the caller did not pass a value - a None sentinel distinguishes "not passed" from
        # an explicit False. ingest.py passes enable_inference=False to suppress per-doc
        # inference in batch mode; that flag must not be silently overridden.
        if expertise and expertise.expansion:
            if enable_unification is None:
                enable_unification = expertise.expansion.cross_tool_unification
            if enable_inference is None:
                enable_inference = expertise.expansion.relationship_inference
            self._inference_depth = expertise.expansion.depth

        self._enable_unification = True if enable_unification is None else enable_unification
        self._enable_inference = True if enable_inference is None else enable_inference

        if expertise and expertise.confidence:
            min_inference_confidence = expertise.confidence.min_inferred

        # Initialize components
        self._unifier = CrossToolUnifier(
            expertise=expertise,
            embedding_threshold=embedding_threshold,
            fuzzy_threshold=fuzzy_threshold,
        )
        self._inferrer = RelationshipInferrer(
            expertise=expertise,
            min_confidence=min_inference_confidence,
        )

    async def expand(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        *,
        namespace_id: UUID | None = None,
        entity_index: EntityIndex | None = None,
        storage: Any = None,
    ) -> ExpansionResult:
        """Expand the knowledge graph.

        Runs unification and inference phases based on configuration.

        Args:
            entities: Entities to expand
            relationships: Relationships to expand
            namespace_id: Namespace ID for new relationships

        Returns:
            ExpansionResult with expanded graph
        """
        # Bound here (not inside Phase 1) so Phase 2 telemetry works when
        # unification is disabled but inference is enabled - see #1122.
        import time as _time

        from khora.telemetry import get_collector

        result = ExpansionResult(
            original_entity_count=len(entities),
            original_relationship_count=len(relationships),
        )

        if not entities:
            return result

        # Determine namespace
        if namespace_id is None and entities:
            namespace_id = entities[0].namespace_id

        current_entities = list(entities)
        current_relationships = list(relationships)
        logger.debug(
            f"Starting expansion with {len(current_entities)} entities, {len(current_relationships)} relationships"
        )

        # Phase 1: Cross-tool entity unification
        if self._enable_unification:
            logger.debug("Running cross-tool entity unification...")
            _t0 = _time.perf_counter()
            unification_result = await self._unifier.unify(
                current_entities,
                current_relationships,
                use_embeddings=True,
                use_fuzzy=True,
                entity_index=entity_index,
                storage=storage,
            )
            get_collector().record_pipeline_stage(
                pipeline="expansion",
                stage="cross_tool_unification",
                latency_ms=(_time.perf_counter() - _t0) * 1000,
                input_count=len(current_entities),
                output_count=len(unification_result.unified_entities),
                namespace_id=namespace_id,
                metadata={"merged": unification_result.entities_merged},
            )

            current_entities = unification_result.unified_entities
            current_relationships = unification_result.updated_relationships
            result.entity_mapping = unification_result.entity_mapping
            result.merged_entity_count = unification_result.entities_merged

            logger.debug(
                f"Unified {result.original_entity_count} entities into {len(current_entities)} "
                f"({result.merged_entity_count} merged)"
            )

        # Phase 2: Relationship inference
        inferred_relationships: list[Relationship] = []
        if self._enable_inference and self._expertise:
            logger.debug(f"Running relationship inference (depth={self._inference_depth})...")
            _t0 = _time.perf_counter()
            inferred = self._inferrer.infer(
                current_entities,
                current_relationships,
                depth=self._inference_depth,
            )

            # Convert to domain relationships
            inferred_relationships = [to_relationship(inf, namespace_id) for inf in inferred]
            result.inferred_relationship_count = len(inferred_relationships)

            get_collector().record_pipeline_stage(
                pipeline="expansion",
                stage="relationship_inference",
                latency_ms=(_time.perf_counter() - _t0) * 1000,
                input_count=len(current_entities) + len(current_relationships),
                output_count=len(inferred_relationships),
                namespace_id=namespace_id,
                metadata={"depth": self._inference_depth},
            )

            logger.debug(f"Inferred {len(inferred_relationships)} new relationships")

        # Build final result
        result.entities = current_entities
        result.relationships = current_relationships
        result.inferred_relationships = inferred_relationships

        return result

    def expand_sync(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        *,
        namespace_id: UUID | None = None,
    ) -> ExpansionResult:
        """Synchronous version of expand.

        Useful for non-async contexts or when LLM expansion is not needed.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.expand(entities, relationships, namespace_id=namespace_id)
        )

    @classmethod
    def from_expertise(cls, expertise: ExpertiseConfig) -> SemanticExpander:
        """Create expander from expertise configuration.

        Args:
            expertise: ExpertiseConfig to use

        Returns:
            Configured SemanticExpander
        """
        return cls(
            expertise=expertise,
            enable_unification=expertise.expansion.cross_tool_unification,
            enable_inference=expertise.expansion.relationship_inference,
            inference_depth=expertise.expansion.depth,
        )

    @classmethod
    def from_expertise_name(cls, name: str) -> SemanticExpander:
        """Create expander from expertise name.

        Args:
            name: Name of expertise to load

        Returns:
            Configured SemanticExpander
        """
        from khora.extraction.skills import load_expertise

        expertise = load_expertise(f"builtin:{name}")
        return cls.from_expertise(expertise)
