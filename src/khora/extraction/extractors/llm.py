"""LLM-based entity extraction using LiteLLM."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from loguru import logger

from .base import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelationship,
    ExtractionResult,
    TemporalInfo,
)

if TYPE_CHECKING:
    from khora.config import LiteLLMConfig


# Default entity types to extract
DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "TECHNOLOGY"]

# Extraction prompt template with temporal awareness
EXTRACTION_PROMPT = """Extract entities, relationships, and temporal information from the following text.

Entity types to extract: {entity_types}

Text:
{text}

Return a JSON object with the following structure:
{{
    "entities": [
        {{
            "name": "entity name (canonical form, properly capitalized)",
            "entity_type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT|TECHNOLOGY|PRODUCT|DATE|etc",
            "description": "brief description of the entity",
            "attributes": {{"key": "value"}},
            "aliases": ["alternative names", "nicknames", "abbreviations"],
            "temporal": {{
                "mentioned_at": "when entity is mentioned (if temporal context exists)",
                "valid_from": "ISO date or null if entity validity period is mentioned",
                "valid_until": "ISO date or null if entity validity period ends"
            }}
        }}
    ],
    "relationships": [
        {{
            "source_entity": "source entity name (must match an entity above)",
            "target_entity": "target entity name (must match an entity above)",
            "relationship_type": "WORKS_FOR|KNOWS|MANAGES|REPORTS_TO|COLLABORATES_WITH|OWNS|PART_OF|LOCATED_IN|RELATES_TO|DEPENDS_ON|IMPLEMENTS|PRECEDES|FOLLOWS|ASSOCIATED_WITH|etc",
            "description": "brief description of relationship",
            "temporal": {{
                "occurred_at": "when relationship occurred/started",
                "valid_from": "ISO date or null if relationship has time bounds",
                "valid_until": "ISO date or null if relationship ended"
            }}
        }}
    ],
    "events": [
        {{
            "description": "what happened",
            "occurred_at": "when it occurred (ISO date or descriptive)",
            "participants": ["entity names involved"],
            "event_type": "MEETING|DECISION|MILESTONE|ANNOUNCEMENT|INCIDENT|etc"
        }}
    ]
}}

Guidelines:
- Use canonical entity names (e.g., "Jennifer Walsh" not "Jenny", "Acme Corporation" not "Acme Corp")
- Include aliases for entities that have multiple names/abbreviations
- Extract temporal information when dates, times, or relative time references appear
- For events, capture the when, who, and what
- Be thorough but precise - only extract entities that are clearly mentioned
- Ensure relationship source/target names match extracted entity names exactly

Return ONLY valid JSON, no other text."""


class LLMEntityExtractor(EntityExtractor):
    """LLM-based entity extractor using LiteLLM.

    Uses an LLM to extract entities and relationships from text
    through structured JSON output.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.3,  # Lower for more consistent extraction
        max_tokens: int = 4000,
        timeout: int = 60,
        max_retries: int = 3,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize the LLM entity extractor.

        Args:
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum retries on failure
            max_concurrent: Maximum concurrent extractions
        """
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @classmethod
    def from_config(cls, config: LiteLLMConfig) -> LLMEntityExtractor:
        """Create extractor from LiteLLM configuration.

        Args:
            config: LiteLLMConfig instance

        Returns:
            Configured LLMEntityExtractor
        """
        return cls(
            model=config.model,
            temperature=0.3,  # Override for extraction
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            max_concurrent=config.max_concurrent_llm_calls,
        )

    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            entity_types: Optional list of entity types to extract

        Returns:
            ExtractionResult containing entities and relationships
        """
        if not text.strip():
            return ExtractionResult()

        entity_types = entity_types or DEFAULT_ENTITY_TYPES

        try:
            import litellm
        except ImportError:
            raise RuntimeError("litellm package not installed. Run: pip install litellm")

        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(entity_types),
            text=text[:8000],  # Truncate very long texts
        )

        async with self._semaphore:
            for attempt in range(self._max_retries):
                try:
                    response = await litellm.acompletion(
                        model=self._model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert entity extraction system. Extract entities and relationships from text and return them as structured JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        timeout=self._timeout,
                        response_format={"type": "json_object"},
                    )

                    content = response.choices[0].message.content
                    return self._parse_response(content)

                except Exception as e:
                    if attempt < self._max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(f"Extraction attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Extraction failed after {self._max_retries} attempts: {e}")
                        return ExtractionResult(metadata={"error": str(e)})

    async def extract_batch(
        self,
        texts: list[str],
        *,
        entity_types: list[str] | None = None,
    ) -> list[ExtractionResult]:
        """Extract from multiple texts concurrently.

        Args:
            texts: List of texts to extract from
            entity_types: Optional list of entity types to extract

        Returns:
            List of ExtractionResult objects
        """
        if not texts:
            return []

        tasks = [self.extract(text, entity_types=entity_types) for text in texts]
        return await asyncio.gather(*tasks)

    def _parse_response(self, content: str) -> ExtractionResult:
        """Parse the LLM response into an ExtractionResult."""
        try:
            # Try to parse as JSON
            data = json.loads(content)

            entities = []
            for e in data.get("entities", []):
                # Parse temporal info if present
                temporal = None
                if "temporal" in e and e["temporal"]:
                    t = e["temporal"]
                    temporal = TemporalInfo(
                        mentioned_at=t.get("mentioned_at"),
                        valid_from=t.get("valid_from"),
                        valid_until=t.get("valid_until"),
                    )

                entities.append(
                    ExtractedEntity(
                        name=e.get("name", ""),
                        entity_type=e.get("entity_type", "CONCEPT"),
                        description=e.get("description", ""),
                        attributes=e.get("attributes", {}),
                        aliases=e.get("aliases", []),
                        temporal=temporal,
                        confidence=e.get("confidence", 0.9),
                    )
                )

            relationships = []
            for r in data.get("relationships", []):
                # Parse temporal info if present
                temporal = None
                if "temporal" in r and r["temporal"]:
                    t = r["temporal"]
                    temporal = TemporalInfo(
                        occurred_at=t.get("occurred_at"),
                        valid_from=t.get("valid_from"),
                        valid_until=t.get("valid_until"),
                    )

                relationships.append(
                    ExtractedRelationship(
                        source_entity=r.get("source_entity", ""),
                        target_entity=r.get("target_entity", ""),
                        relationship_type=r.get("relationship_type", "RELATES_TO"),
                        description=r.get("description", ""),
                        properties=r.get("properties", {}),
                        temporal=temporal,
                        confidence=r.get("confidence", 0.9),
                    )
                )

            events = []
            for ev in data.get("events", []):
                events.append(
                    ExtractedEvent(
                        description=ev.get("description", ""),
                        event_type=ev.get("event_type", "EVENT"),
                        occurred_at=ev.get("occurred_at"),
                        participants=ev.get("participants", []),
                        confidence=ev.get("confidence", 0.9),
                    )
                )

            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                events=events,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction response as JSON: {e}")
            # Try to extract JSON from the response
            return self._extract_json_from_text(content)

    def _extract_json_from_text(self, text: str) -> ExtractionResult:
        """Try to extract JSON from text that may contain other content."""
        import re

        # Look for JSON object in the text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._parse_response(json.dumps(data))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not extract valid JSON from response")
        return ExtractionResult(metadata={"raw_response": text[:500]})
