"""Generate domain-specific system prompts for the final ontology YAML."""

from __future__ import annotations

from loguru import logger

from khora.extraction.skills.base import EntityTypeConfig, RelationshipTypeConfig

from ..llm import OntologyLLM
from . import prompts
from .domain import DomainResult


class PromptGenerator:
    """Generate a domain-specific system prompt for the extraction skill."""

    def __init__(self, llm: OntologyLLM) -> None:
        self._llm = llm

    async def generate(
        self,
        domain: DomainResult,
        entity_types: list[EntityTypeConfig],
        relationship_types: list[RelationshipTypeConfig],
    ) -> str:
        """Generate a system prompt for the final ontology.

        Args:
            domain: Domain detection result.
            entity_types: Inferred entity types.
            relationship_types: Inferred relationship types.

        Returns:
            The generated system prompt string.
        """
        entity_names = [et.name for et in entity_types]
        rel_names = [rt.name for rt in relationship_types]

        system = prompts.PROMPT_GENERATION_SYSTEM.format(domain=domain.primary_domain)
        user = prompts.PROMPT_GENERATION_USER.format(
            domain=domain.primary_domain,
            entity_type_names=", ".join(entity_names),
            relationship_type_names=", ".join(rel_names),
            key_concepts=", ".join(domain.key_concepts[:15]),
        )

        result = await self._llm.complete(system=system, user=user, temperature=0.7)

        system_prompt = result.get("system_prompt", "")
        reasoning = result.get("reasoning", "")
        if reasoning:
            logger.debug(f"Prompt generation reasoning: {reasoning}")

        logger.info(f"Generated system prompt ({len(system_prompt)} chars)")
        return system_prompt
