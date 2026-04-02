"""Rule-based entity extractor.

Runs regex patterns against text to extract entities and relationships
before the LLM extractor. Fast (no API calls) and produces high-confidence
results for well-defined patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .base import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)

if TYPE_CHECKING:
    from khora.extraction.skills.base import ExpertiseConfig


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------


@dataclass
class _Pattern:
    name: str
    pattern: re.Pattern[str]
    entity_type: str
    confidence: float = 0.95
    group: int = 1  # capture group to use as entity name


_BUILTIN_PATTERNS: list[_Pattern] = [
    _Pattern(
        name="section_number",
        pattern=re.compile(r"(?:Section|Sec\.)\s*(\d+[A-Za-z]?\.\d+[A-Za-z]?\.?(?:\d+\.?)*)"),
        entity_type="CODE_SECTION",
        confidence=0.95,
    ),
    _Pattern(
        name="ordinance_ref",
        pattern=re.compile(r"Ord\.?\s*No\.?\s*([\d,]+)\s*,\s*Eff\.?\s*(\d{1,2}/\d{1,2}/\d{2,4})"),
        entity_type="ORDINANCE",
        confidence=0.95,
    ),
    _Pattern(
        name="fee_amount",
        pattern=re.compile(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"),
        entity_type="FEE_SCHEDULE",
        confidence=0.80,
        group=0,  # full match including $
    ),
]


class RuleBasedExtractor(EntityExtractor):
    """Extract entities using regex patterns — runs before LLM."""

    def __init__(
        self,
        *,
        expertise: ExpertiseConfig | None = None,
        use_builtins: bool = True,
    ) -> None:
        self._expertise = expertise
        self._use_builtins = use_builtins

    def _get_patterns(self) -> list[_Pattern]:
        patterns: list[_Pattern] = []
        if self._use_builtins:
            patterns.extend(_BUILTIN_PATTERNS)
        # Add patterns from expertise correlation_rules
        if self._expertise and hasattr(self._expertise, "correlation_rules"):
            for rule in self._expertise.correlation_rules:
                if hasattr(rule, "pattern") and rule.pattern:
                    try:
                        patterns.append(
                            _Pattern(
                                name=rule.name,
                                pattern=re.compile(rule.pattern),
                                entity_type=rule.entity_types[0] if rule.entity_types else "CONCEPT",
                                confidence=getattr(rule, "confidence", 0.9),
                            )
                        )
                    except re.error:
                        pass
        return patterns

    def _extract_from_text(self, text: str) -> ExtractionResult:
        entities: list[ExtractedEntity] = []
        relationships: list[ExtractedRelationship] = []
        seen_entities: set[tuple[str, str]] = set()

        for pat in self._get_patterns():
            for match in pat.pattern.finditer(text):
                name = match.group(pat.group) if pat.group < len(match.groups()) + 1 else match.group(0)
                if not name:
                    continue

                key = (name, pat.entity_type)
                if key in seen_entities:
                    continue
                seen_entities.add(key)

                attrs: dict[str, Any] = {}
                # For ordinances, capture effective date
                if pat.name == "ordinance_ref" and match.lastindex and match.lastindex >= 2:
                    name = match.group(1).replace(",", "")
                    attrs["effective_date"] = match.group(2)

                entities.append(
                    ExtractedEntity(
                        name=name,
                        entity_type=pat.entity_type,
                        attributes=attrs,
                        confidence=pat.confidence,
                        source_text=match.group(0),
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )

                # If the expertise rule specifies a relationship to create
                if self._expertise and hasattr(self._expertise, "correlation_rules"):
                    for rule in self._expertise.correlation_rules:
                        if (
                            rule.name == pat.name
                            and hasattr(rule, "creates_relationship")
                            and rule.creates_relationship
                        ):
                            relationships.append(
                                ExtractedRelationship(
                                    source_entity=name,
                                    target_entity=name,
                                    relationship_type=rule.creates_relationship,
                                    confidence=pat.confidence,
                                )
                            )

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            metadata={"extractor": "rule_based"},
        )

    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        if expertise:
            self._expertise = expertise
        result = self._extract_from_text(text)
        # Filter by requested entity types
        if entity_types:
            type_set = set(entity_types)
            result.entities = [e for e in result.entities if e.entity_type in type_set]
        return result

    async def extract_batch(
        self,
        texts: list[str],
        *,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[ExtractionResult]:
        if expertise:
            self._expertise = expertise
        results = [self._extract_from_text(t) for t in texts]
        if entity_types:
            type_set = set(entity_types)
            for r in results:
                r.entities = [e for e in r.entities if e.entity_type in type_set]
        return results
