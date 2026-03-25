"""LLM-powered entity type inference."""

from __future__ import annotations

from loguru import logger

from khora.extraction.skills.base import EntityTypeConfig

from ..llm import OntologyLLM
from . import prompts
from .domain import DomainResult


class EntityInferrer:
    """Infer entity types from data samples using LLM analysis."""

    def __init__(self, llm: OntologyLLM) -> None:
        self._llm = llm

    async def infer(
        self,
        domain: DomainResult,
        formatted_samples: str,
    ) -> list[EntityTypeConfig]:
        """Infer entity types from domain analysis and data samples.

        Args:
            domain: Result from domain detection.
            formatted_samples: Pre-formatted sample text.

        Returns:
            List of inferred EntityTypeConfig objects.
        """
        # Scale target range based on domain complexity
        if domain.ontology_scope == "large":
            min_types, max_types = 15, 30
        elif domain.ontology_scope == "multiple":
            min_types, max_types = 10, 25
        else:
            min_types, max_types = 5, 18

        system = prompts.ENTITY_INFERENCE_SYSTEM.format(
            domain=domain.primary_domain,
            min_types=min_types,
            max_types=max_types,
        )
        user = prompts.ENTITY_INFERENCE_USER.format(
            domain=domain.primary_domain,
            key_concepts=", ".join(domain.key_concepts[:15]),
            samples=formatted_samples,
        )

        result = await self._llm.complete(system=system, user=user, temperature=0.3)

        raw_types = result.get("entity_types", [])
        entity_types = [EntityTypeConfig.from_dict(et) for et in raw_types if isinstance(et, dict)]

        reasoning = result.get("reasoning", "")
        if reasoning:
            logger.debug(f"Entity inference reasoning: {reasoning}")

        logger.info(f"Inferred {len(entity_types)} entity types for {domain.primary_domain}")
        return entity_types
