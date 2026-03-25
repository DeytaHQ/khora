"""LLM-based domain detection for ontology construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ..llm import OntologyLLM
from . import prompts


@dataclass(slots=True)
class DomainResult:
    """Result of domain detection analysis."""

    primary_domain: str = "General"
    secondary_domains: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["English"])
    data_structure: str = "unstructured"
    ontology_scope: str = "single"
    scope_reasoning: str = ""
    estimated_entity_types: int = 10
    estimated_relationship_types: int = 15
    key_concepts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainResult:
        return cls(
            primary_domain=data.get("primary_domain", "General"),
            secondary_domains=data.get("secondary_domains", []),
            languages=data.get("languages", ["English"]),
            data_structure=data.get("data_structure", "unstructured"),
            ontology_scope=data.get("ontology_scope", "single"),
            scope_reasoning=data.get("scope_reasoning", ""),
            estimated_entity_types=data.get("estimated_entity_types", 10),
            estimated_relationship_types=data.get("estimated_relationship_types", 15),
            key_concepts=data.get("key_concepts", []),
        )


class DomainDetector:
    """Detect the domain and characteristics of data sources using LLM analysis."""

    def __init__(self, llm: OntologyLLM) -> None:
        self._llm = llm

    async def detect(
        self,
        formatted_samples: str,
        source_count: int,
        total_chars: int,
    ) -> DomainResult:
        """Analyze data samples and detect the domain.

        Args:
            formatted_samples: Pre-formatted sample text from DataSampler.
            source_count: Number of data sources.
            total_chars: Total characters across all samples.

        Returns:
            DomainResult with detected domain characteristics.
        """
        user_prompt = prompts.DOMAIN_DETECTION_USER.format(
            source_count=source_count,
            total_chars=total_chars,
            samples=formatted_samples,
        )

        result = await self._llm.complete(
            system=prompts.DOMAIN_DETECTION_SYSTEM,
            user=user_prompt,
            temperature=0.3,
        )

        domain = DomainResult.from_dict(result)
        logger.info(
            f"Domain detected: {domain.primary_domain} "
            f"(scope={domain.ontology_scope}, "
            f"~{domain.estimated_entity_types} entity types)"
        )
        return domain
