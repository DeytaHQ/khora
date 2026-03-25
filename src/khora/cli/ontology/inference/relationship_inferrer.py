"""LLM-powered relationship type inference."""

from __future__ import annotations

from loguru import logger

from khora.extraction.skills.base import EntityTypeConfig, RelationshipTypeConfig

from ..llm import OntologyLLM
from . import prompts
from .domain import DomainResult


class RelationshipInferrer:
    """Infer relationship types from entity types and data samples."""

    def __init__(self, llm: OntologyLLM) -> None:
        self._llm = llm

    async def infer(
        self,
        domain: DomainResult,
        entity_types: list[EntityTypeConfig],
        formatted_samples: str,
    ) -> list[RelationshipTypeConfig]:
        """Infer relationship types from domain, entity types, and samples.

        Args:
            domain: Result from domain detection.
            entity_types: Previously inferred entity types.
            formatted_samples: Pre-formatted sample text.

        Returns:
            List of inferred RelationshipTypeConfig objects.
        """
        entity_names = [et.name for et in entity_types]
        n_entities = len(entity_names)
        min_rels = max(3, int(n_entities * 1.2))
        max_rels = max(5, int(n_entities * 2.5))

        system = prompts.RELATIONSHIP_INFERENCE_SYSTEM.format(
            domain=domain.primary_domain,
            entity_type_names=", ".join(entity_names),
            min_rels=min_rels,
            max_rels=max_rels,
        )
        user = prompts.RELATIONSHIP_INFERENCE_USER.format(
            domain=domain.primary_domain,
            entity_type_names=", ".join(entity_names),
            samples=formatted_samples,
        )

        result = await self._llm.complete(system=system, user=user, temperature=0.3)

        raw_types = result.get("relationship_types", [])
        rel_types = [RelationshipTypeConfig.from_dict(rt) for rt in raw_types if isinstance(rt, dict)]

        # Validate references
        valid = []
        for rt in rel_types:
            bad_refs = [t for t in rt.source_types + rt.target_types if t != "*" and t not in entity_names]
            if bad_refs:
                logger.warning(f"Dropping relationship '{rt.name}': references unknown types {bad_refs}")
            else:
                valid.append(rt)

        reasoning = result.get("reasoning", "")
        if reasoning:
            logger.debug(f"Relationship inference reasoning: {reasoning}")

        logger.info(
            f"Inferred {len(valid)} relationship types " f"(dropped {len(rel_types) - len(valid)} with bad references)"
        )
        return valid
