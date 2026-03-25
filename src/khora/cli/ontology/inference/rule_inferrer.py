"""LLM-powered correlation and inference rule generation."""

from __future__ import annotations

from loguru import logger

from khora.extraction.skills.base import (
    CorrelationRule,
    EntityTypeConfig,
    InferenceRule,
    RelationshipTypeConfig,
)

from ..llm import OntologyLLM
from . import prompts
from .domain import DomainResult


class RuleInferrer:
    """Generate correlation and inference rules from entity/relationship types."""

    def __init__(self, llm: OntologyLLM) -> None:
        self._llm = llm

    async def infer(
        self,
        domain: DomainResult,
        entity_types: list[EntityTypeConfig],
        relationship_types: list[RelationshipTypeConfig],
    ) -> tuple[list[CorrelationRule], list[InferenceRule]]:
        """Generate rules from domain context and type definitions.

        Args:
            domain: Result from domain detection.
            entity_types: Inferred entity types.
            relationship_types: Inferred relationship types.

        Returns:
            Tuple of (correlation_rules, inference_rules).
        """
        entity_names = [et.name for et in entity_types]
        rel_names = [rt.name for rt in relationship_types]

        system = prompts.RULE_INFERENCE_SYSTEM.format(domain=domain.primary_domain)
        user = prompts.RULE_INFERENCE_USER.format(
            domain=domain.primary_domain,
            entity_type_names=", ".join(entity_names),
            relationship_type_names=", ".join(rel_names),
        )

        result = await self._llm.complete(system=system, user=user, temperature=0.3)

        # Parse correlation rules
        raw_corr = result.get("correlation_rules", [])
        corr_rules = [CorrelationRule.from_dict(cr) for cr in raw_corr if isinstance(cr, dict)]

        # Parse inference rules
        raw_inf = result.get("inference_rules", [])
        inf_rules = [InferenceRule.from_dict(ir) for ir in raw_inf if isinstance(ir, dict)]

        reasoning = result.get("reasoning", "")
        if reasoning:
            logger.debug(f"Rule inference reasoning: {reasoning}")

        logger.info(f"Generated {len(corr_rules)} correlation rules, {len(inf_rules)} inference rules")
        return corr_rules, inf_rules
