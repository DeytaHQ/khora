"""LLM-based entity extraction using LiteLLM."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

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
    from khora.extraction.skills import ExpertiseConfig


# Default entity types to extract
DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "TECHNOLOGY"]

# Default system prompt for extraction
DEFAULT_SYSTEM_PROMPT = """You are an expert entity extraction system. Extract entities and relationships from text and return them as structured JSON."""

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
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            entity_types: Optional list of entity types to extract
            expertise: Optional ExpertiseConfig for domain-specific extraction
            context: Optional context dict for prompt template rendering

        Returns:
            ExtractionResult containing entities and relationships
        """
        if not text.strip():
            return ExtractionResult()

        # Determine entity types from expertise or fallback
        if expertise:
            entity_types = expertise.get_entity_type_names() or DEFAULT_ENTITY_TYPES
        else:
            entity_types = entity_types or DEFAULT_ENTITY_TYPES

        try:
            import litellm
        except ImportError:
            raise RuntimeError("litellm package not installed. Run: pip install litellm")

        # Render prompts based on expertise
        system_prompt = self._render_system_prompt(expertise, context)
        extraction_prompt = self._render_extraction_prompt(text, entity_types, expertise, context)

        async with self._semaphore:
            for attempt in range(self._max_retries):
                try:
                    import time as _time

                    _t0 = _time.perf_counter()
                    response = await litellm.acompletion(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": extraction_prompt},
                        ],
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        timeout=self._timeout,
                        response_format={"type": "json_object"},
                    )
                    _latency = (_time.perf_counter() - _t0) * 1000

                    # Record telemetry
                    from khora.telemetry import get_collector

                    usage = getattr(response, "usage", None)
                    get_collector().record_llm_call(
                        operation="entity_extraction",
                        model=self._model,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                        latency_ms=_latency,
                    )

                    content = response.choices[0].message.content
                    result = self._parse_response(content)

                    # Apply confidence filtering from expertise if available
                    if expertise:
                        result = self._filter_by_confidence(result, expertise)

                    return result

                except Exception as e:
                    if attempt < self._max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(f"Extraction attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Extraction failed after {self._max_retries} attempts: {e}")
                        return ExtractionResult(metadata={"error": str(e)})

    def _render_system_prompt(
        self,
        expertise: ExpertiseConfig | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Render the system prompt, optionally using expertise config."""
        if not expertise or not expertise.system_prompt:
            return DEFAULT_SYSTEM_PROMPT

        try:
            from khora.extraction.skills.composer import ExpertiseComposer

            composer = ExpertiseComposer()
            return composer.render_prompt(
                expertise.system_prompt,
                expertise=expertise,
                context=context,
            )
        except Exception as e:
            logger.warning(f"Failed to render system prompt: {e}")
            return expertise.system_prompt or DEFAULT_SYSTEM_PROMPT

    def _build_tool_context(self, expertise: ExpertiseConfig | None, context: dict[str, Any] | None) -> str:
        """Build tool-specific context block for the extraction prompt.

        When expertise has tool_schemas populated and the context identifies
        a source_tool, this injects structured field knowledge so the LLM
        understands the data format it's extracting from.

        Args:
            expertise: Optional ExpertiseConfig with tool_schemas
            context: Optional context dict with source_tool key

        Returns:
            Tool context string to prepend to the text, or empty string
        """
        if not expertise or not expertise.tool_schemas or not context:
            return ""

        source_tool = context.get("source_tool", "")
        if not source_tool:
            return ""

        schema = expertise.tool_schemas.get(source_tool)
        if not schema:
            return ""

        lines = [f"\nSOURCE CONTEXT: This content comes from {source_tool}."]
        for obj_type, obj_schema in schema.items():
            if not isinstance(obj_schema, dict):
                continue
            fields = obj_schema.get("fields", [])
            if fields:
                lines.append(f"  {obj_type} fields: {', '.join(str(f) for f in fields)}")
            for key, values in obj_schema.items():
                if key != "fields" and isinstance(values, list):
                    lines.append(f"  {key}: {', '.join(str(v) for v in values)}")

        # Add attribute schema hints from entity types
        if expertise.entity_types:
            lines.append("\nEXPECTED ENTITY ATTRIBUTES:")
            for et in expertise.entity_types:
                required = et.attributes.get("required", [])
                optional = et.attributes.get("optional", [])
                if required or optional:
                    parts = []
                    if required:
                        parts.append(f"required: {', '.join(required)}")
                    if optional:
                        parts.append(f"optional: {', '.join(optional)}")
                    lines.append(f"  {et.name}: {'; '.join(parts)}")

        return "\n".join(lines)

    def _render_extraction_prompt(
        self,
        text: str,
        entity_types: list[str],
        expertise: ExpertiseConfig | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Render the extraction prompt, optionally using expertise config."""
        # Build tool context for SaaS-aware extraction
        tool_context = self._build_tool_context(expertise, context)

        # If expertise has a custom extraction prompt, use it
        if expertise and expertise.extraction_prompt:
            try:
                from khora.extraction.skills.composer import ExpertiseComposer

                composer = ExpertiseComposer()
                prompt_context = {
                    **(context or {}),
                    "text": text[:8000],
                    "entity_types": entity_types,
                    "tool_context": tool_context,
                }
                return composer.render_prompt(
                    expertise.extraction_prompt,
                    expertise=expertise,
                    context=prompt_context,
                )
            except Exception as e:
                logger.warning(f"Failed to render extraction prompt: {e}")

        # Use default extraction prompt with optional tool context
        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(entity_types),
            text=text[:8000],  # Truncate very long texts
        )
        if tool_context:
            prompt = tool_context + "\n\n" + prompt
        return prompt

    def _filter_by_confidence(
        self,
        result: ExtractionResult,
        expertise: ExpertiseConfig,
    ) -> ExtractionResult:
        """Filter extraction results by confidence thresholds from expertise."""
        min_entity = expertise.confidence.min_entity
        min_relationship = expertise.confidence.min_relationship

        filtered_entities = [e for e in result.entities if e.confidence >= min_entity]
        filtered_relationships = [r for r in result.relationships if r.confidence >= min_relationship]

        return ExtractionResult(
            entities=filtered_entities,
            relationships=filtered_relationships,
            events=result.events,
            metadata=result.metadata,
        )

    async def extract_batch(
        self,
        texts: list[str],
        *,
        entity_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[ExtractionResult]:
        """Extract from multiple texts concurrently.

        Args:
            texts: List of texts to extract from
            entity_types: Optional list of entity types to extract
            expertise: Optional ExpertiseConfig for domain-specific extraction
            context: Optional context dict for prompt template rendering

        Returns:
            List of ExtractionResult objects
        """
        if not texts:
            return []

        tasks = [self.extract(text, entity_types=entity_types, expertise=expertise, context=context) for text in texts]
        return await asyncio.gather(*tasks)

    async def extract_multi(
        self,
        texts: list[str],
        *,
        entity_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
        batch_size: int = 5,
    ) -> list[ExtractionResult]:
        """Extract entities from multiple texts in grouped LLM calls.

        Groups texts into batches and sends each batch as a single LLM call,
        reducing API round-trips by up to batch_size times.

        Args:
            texts: List of texts to extract from
            entity_types: Optional list of entity types to extract
            expertise: Optional ExpertiseConfig for domain-specific extraction
            context: Optional context dict for prompt template rendering
            batch_size: Number of texts per LLM call

        Returns:
            List of ExtractionResult objects (one per input text)
        """
        if not texts:
            return []

        if expertise:
            entity_types = expertise.get_entity_type_names() or DEFAULT_ENTITY_TYPES
        else:
            entity_types = entity_types or DEFAULT_ENTITY_TYPES

        try:
            import litellm
        except ImportError:
            raise RuntimeError("litellm package not installed. Run: pip install litellm")

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        all_results: list[ExtractionResult] = []

        # Build system prompt from expertise if available
        system_prompt = self._render_system_prompt(expertise, context)
        tool_context = self._build_tool_context(expertise, context)

        async def _run_batch(batch: list[str]) -> list[ExtractionResult]:
            results = await self._extract_multi_batch(
                batch,
                entity_types,
                litellm,
                system_prompt=system_prompt,
                tool_context=tool_context,
            )
            if expertise:
                results = [self._filter_by_confidence(r, expertise) for r in results]
            return results

        batch_results = await asyncio.gather(*[_run_batch(b) for b in batches])
        for results in batch_results:
            all_results.extend(results)

        return all_results

    async def _extract_multi_batch(
        self,
        texts: list[str],
        entity_types: list[str],
        litellm: Any,
        *,
        system_prompt: str | None = None,
        tool_context: str | None = None,
    ) -> list[ExtractionResult]:
        """Extract from a batch of texts in a single LLM call."""
        sections = "\n".join(f"=== SECTION {i + 1} ===\n{text[:4000]}" for i, text in enumerate(texts))

        tool_prefix = f"{tool_context}\n\n" if tool_context else ""
        prompt = f"""{tool_prefix}Extract entities, relationships, and events from each text section below.

Entity types to find: {", ".join(entity_types)}

{sections}

Return a JSON object with a "sections" array, one object per section:
{{"sections": [
    {{"entities": [...], "relationships": [...], "events": [...]}},
    ...
]}}

Each section follows the same entity/relationship/event format.
Return ONLY valid JSON, no other text."""

        async with self._semaphore:
            for attempt in range(self._max_retries):
                try:
                    import time as _time

                    _t0 = _time.perf_counter()
                    response = await litellm.acompletion(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        timeout=self._timeout,
                        response_format={"type": "json_object"},
                    )
                    _latency = (_time.perf_counter() - _t0) * 1000

                    # Record telemetry
                    from khora.telemetry import get_collector

                    usage = getattr(response, "usage", None)
                    get_collector().record_llm_call(
                        operation="entity_extraction_multi",
                        model=self._model,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                        latency_ms=_latency,
                        metadata={"batch_size": len(texts)},
                    )

                    content = response.choices[0].message.content
                    data = json.loads(content)
                    sections_data = data.get("sections", [])

                    results: list[ExtractionResult] = []
                    for i, text in enumerate(texts):
                        if i < len(sections_data):
                            section_json = json.dumps(sections_data[i])
                            results.append(self._parse_response(section_json))
                        else:
                            results.append(ExtractionResult())

                    return results

                except Exception as e:
                    if attempt < self._max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            f"Multi-extraction attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Multi-extraction failed after {self._max_retries} attempts: {e}")
                        return [ExtractionResult(metadata={"error": str(e)}) for _ in texts]

        return [ExtractionResult() for _ in texts]

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

                # Ensure attributes is a dict (LLM sometimes returns a list)
                attrs = e.get("attributes", {})
                if not isinstance(attrs, dict):
                    attrs = {}

                entities.append(
                    ExtractedEntity(
                        name=e.get("name", ""),
                        entity_type=e.get("entity_type", "CONCEPT"),
                        description=e.get("description", ""),
                        attributes=attrs,
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
