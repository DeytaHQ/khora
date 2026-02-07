"""LLM-based entity extraction using LiteLLM."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from loguru import logger
from tenacity import AsyncRetrying, before_sleep_log, stop_after_attempt, wait_exponential

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

# Default relationship types to use
DEFAULT_RELATIONSHIP_TYPES = [
    "WORKS_FOR",
    "KNOWS",
    "MANAGES",
    "PART_OF",
    "LOCATED_IN",
    "DEPENDS_ON",
    "IMPLEMENTS",
    "RELATES_TO",
    "ASSOCIATED_WITH",
]

# Default system prompt for extraction
DEFAULT_SYSTEM_PROMPT = """You are an expert entity extraction system. Extract entities and relationships from text and return them as structured JSON."""

# Extraction prompt template with temporal awareness
EXTRACTION_PROMPT = """Extract entities, relationships, and temporal information from the following text.

Entity types to extract: {entity_types}
Relationship types to use: {relationship_types}

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

    # Models that require json_schema format instead of json_object
    # OpenAI models with structured output support need explicit json_schema
    # to ensure additionalProperties: false is properly set on all nested objects
    MODELS_REQUIRING_JSON_SCHEMA: set[str] = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o1",
        "o1-mini",
        "o1-preview",
        "o3-mini",
    }

    # Model input token multipliers for adaptive batching
    # Multiplier is applied to max_tokens to get max_input_tokens budget
    # Higher values for large context models (128K+), lower for smaller context
    MODEL_INPUT_MULTIPLIERS: dict[str, int] = {
        # Large context models (128K+) - can be more aggressive
        "gpt-4o": 8,
        "gpt-4o-2024-05-13": 8,
        "gpt-4o-2024-08-06": 8,
        "gpt-4o-2024-11-20": 8,
        "gpt-4.1": 8,  # 1M context
        "gpt-4.1-mini": 5,  # 1M context
        "gpt-4.1-nano": 5,  # 1M context
        "gpt-4o-mini": 5,
        "gpt-4o-mini-2024-07-18": 5,
        "o1": 8,
        "o1-mini": 5,
        "o3-mini": 5,
        # Medium context models (32K)
        "gpt-4-turbo": 4,
        "gpt-4-turbo-preview": 4,
        # Smaller context models (8K-16K) - conservative
        "gpt-4": 2,
        "gpt-3.5-turbo": 2,
        # Claude models
        "claude-3-opus": 8,
        "claude-3-sonnet": 8,
        "claude-3-haiku": 5,
    }
    DEFAULT_INPUT_MULTIPLIER = 3  # Fallback for unknown models

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.3,  # Lower for more consistent extraction
        max_tokens: int = 4000,
        timeout: int = 60,
        max_retries: int = 3,
        max_concurrent: int = 10,
        retry_wait: float = 1.0,
    ) -> None:
        """Initialize the LLM entity extractor.

        Args:
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum retries on failure
            max_concurrent: Maximum concurrent extractions
            retry_wait: Base wait time (seconds) for exponential backoff between retries
        """
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_wait = retry_wait
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses ~3 chars per token as a conservative heuristic for English text.
        This slightly overestimates to provide safety margin.
        """
        return len(text) // 3

    def _get_input_multiplier(self) -> int:
        """Get input token multiplier based on model.

        Returns the multiplier to apply to max_tokens for calculating
        the input token budget for adaptive batching.
        """
        # Check exact match first
        if self._model in self.MODEL_INPUT_MULTIPLIERS:
            return self.MODEL_INPUT_MULTIPLIERS[self._model]
        # Check prefix matches (e.g., "gpt-4o-mini" matches "gpt-4o-mini-...")
        for model_prefix, multiplier in self.MODEL_INPUT_MULTIPLIERS.items():
            if self._model.startswith(model_prefix):
                return multiplier
        return self.DEFAULT_INPUT_MULTIPLIER

    def _create_adaptive_batches(
        self,
        texts: list[str],
        max_batch_size: int,
        max_input_tokens: int,
        prompt_overhead: int = 500,
    ) -> list[list[str]]:
        """Create batches that fit within token budget.

        Groups texts greedily until hitting the input token budget or
        max batch size, whichever comes first.

        Args:
            texts: List of texts to batch
            max_batch_size: Maximum number of texts per batch
            max_input_tokens: Token budget for input
            prompt_overhead: Estimated tokens for system prompt and instructions

        Returns:
            List of text batches
        """
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = prompt_overhead

        for text in texts:
            # Match truncation in _extract_multi_batch (4000 chars)
            text_tokens = self._estimate_tokens(text[:4000])

            # Check if adding this text would exceed budget or max batch size
            if current_batch and (
                current_tokens + text_tokens > max_input_tokens or len(current_batch) >= max_batch_size
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = prompt_overhead

            current_batch.append(text)
            current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _get_response_format(self) -> dict[str, Any]:
        """Get the appropriate response_format based on the model.

        Some models (like gpt-5-nano) require json_schema format with
        strict structured outputs, while others work with json_object.
        """
        if self._model in self.MODELS_REQUIRING_JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "entity_type": {"type": "string"},
                                        "description": {"type": "string"},
                                        "aliases": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": ["name", "entity_type", "description", "aliases"],
                                    "additionalProperties": False,
                                },
                            },
                            "relationships": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_entity": {"type": "string"},
                                        "target_entity": {"type": "string"},
                                        "relationship_type": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["source_entity", "target_entity", "relationship_type", "description"],
                                    "additionalProperties": False,
                                },
                            },
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {"type": "string"},
                                        "event_type": {"type": "string"},
                                        "occurred_at": {"type": ["string", "null"]},
                                        "participants": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": ["description", "event_type", "occurred_at", "participants"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["entities", "relationships", "events"],
                        "additionalProperties": False,
                    },
                },
            }
        return {"type": "json_object"}

    def _get_multi_response_format(self) -> dict[str, Any]:
        """Get the appropriate response_format for multi-section batch extraction.

        Similar to _get_response_format but wraps entities/relationships/events
        in a "sections" array for batch processing.
        """
        if self._model in self.MODELS_REQUIRING_JSON_SCHEMA:
            section_schema = {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "entity_type": {"type": "string"},
                                "description": {"type": "string"},
                                "aliases": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["name", "entity_type", "description", "aliases"],
                            "additionalProperties": False,
                        },
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_entity": {"type": "string"},
                                "target_entity": {"type": "string"},
                                "relationship_type": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["source_entity", "target_entity", "relationship_type", "description"],
                            "additionalProperties": False,
                        },
                    },
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "event_type": {"type": "string"},
                                "occurred_at": {"type": ["string", "null"]},
                                "participants": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["description", "event_type", "occurred_at", "participants"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["entities", "relationships", "events"],
                "additionalProperties": False,
            }
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "multi_extraction_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sections": {
                                "type": "array",
                                "items": section_schema,
                            },
                        },
                        "required": ["sections"],
                        "additionalProperties": False,
                    },
                },
            }
        return {"type": "json_object"}

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
            retry_wait=config.retry_wait,
        )

    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExtractionResult:  # type: ignore[invalid-return-type]
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

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=self._retry_wait, min=self._retry_wait, max=10),
                before_sleep=before_sleep_log(logger, "WARNING"),
                reraise=True,
            ):
                with attempt:
                    async with self._semaphore:
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
                            response_format=self._get_response_format(),
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
                    finish_reason = getattr(response.choices[0], "finish_reason", "unknown")

                    # Check for truncated response (hit max_tokens limit)
                    if finish_reason == "length":
                        model_used = getattr(response, "model", self._model)
                        logger.warning(
                            f"LLM response truncated (finish_reason=length) in extraction. "
                            f"Model: {model_used}. Consider increasing max_tokens."
                        )
                        return ExtractionResult(
                            metadata={"error": "truncated_response", "finish_reason": finish_reason}
                        )

                    result = self._parse_response(content)

                    # Apply confidence filtering from expertise if available
                    if expertise:
                        result = self._filter_by_confidence(result, expertise)

                    return result
        except Exception as e:
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
            logger.debug(
                f"Using custom extraction_prompt from expertise '{expertise.name}' "
                f"({len(expertise.extraction_prompt)} chars)"
            )
            try:
                from khora.extraction.skills.composer import ExpertiseComposer

                composer = ExpertiseComposer()
                # Get relationship types from expertise
                relationship_types = expertise.get_relationship_type_names() or DEFAULT_RELATIONSHIP_TYPES

                prompt_context = {
                    **(context or {}),
                    "text": text[:8000],
                    "entity_types": entity_types,
                    "relationship_types": relationship_types,
                    "tool_context": tool_context,
                }
                return composer.render_prompt(
                    expertise.extraction_prompt,
                    expertise=expertise,
                    context=prompt_context,
                )
            except Exception as e:
                logger.warning(f"Failed to render extraction prompt: {e}")

        # Use default extraction prompt
        logger.debug("Using DEFAULT_EXTRACTION_PROMPT (no custom prompt in expertise)")
        # Use default extraction prompt with optional tool context
        # Get relationship types from expertise or defaults
        if expertise:
            relationship_types = expertise.get_relationship_type_names() or DEFAULT_RELATIONSHIP_TYPES
        else:
            relationship_types = DEFAULT_RELATIONSHIP_TYPES

        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(entity_types),
            relationship_types=", ".join(relationship_types),
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
        """Extract from multiple texts using adaptive token-budget batching.

        Uses extract_multi() to group texts into batches based on token budgets,
        reducing API round-trips while avoiding context overflow. Falls back to
        single-document extraction if batch extraction fails.

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

        # Use extract_multi for efficient token-budget-based batching
        # This reduces API calls by grouping texts that fit within model context
        return await self.extract_multi(
            texts,
            entity_types=entity_types,
            expertise=expertise,
            context=context,
        )

    async def extract_multi(
        self,
        texts: list[str],
        *,
        entity_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
        batch_size: int = 5,
        max_input_tokens: int | None = None,
    ) -> list[ExtractionResult]:
        """Extract entities from multiple texts in grouped LLM calls.

        Groups texts into batches using adaptive token-budget-based batching,
        reducing API round-trips while avoiding context overflow.

        Args:
            texts: List of texts to extract from
            entity_types: Optional list of entity types to extract
            expertise: Optional ExpertiseConfig for domain-specific extraction
            context: Optional context dict for prompt template rendering
            batch_size: Maximum number of texts per LLM call (fallback/cap)
            max_input_tokens: Token budget for input. If None, auto-calculated
                from max_tokens using model-aware multipliers.

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

        # Calculate max_input_tokens from model if not provided
        if max_input_tokens is None:
            multiplier = self._get_input_multiplier()
            max_input_tokens = self._max_tokens * multiplier
            logger.debug(
                f"Using adaptive batching: model={self._model}, "
                f"multiplier={multiplier}x, max_input_tokens={max_input_tokens}"
            )

        # Create adaptive batches based on token budget
        batches = self._create_adaptive_batches(
            texts,
            max_batch_size=batch_size,
            max_input_tokens=max_input_tokens,
        )
        logger.debug(f"Created {len(batches)} batches from {len(texts)} texts")
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
                expertise=expertise,
                context=context,
            )

            # Check if batch failed (all results have errors) - fallback to single extraction
            all_failed = all(r.metadata.get("error") for r in results)
            if all_failed and len(batch) > 1:
                logger.info(
                    f"Batch extraction failed for {len(batch)} texts, falling back to single-document extraction"
                )
                # Extract documents one at a time
                single_results = []
                for text in batch:
                    result = await self.extract(
                        text,
                        entity_types=entity_types,
                        expertise=expertise,
                        context=context,
                    )
                    single_results.append(result)
                results = single_results

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
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[ExtractionResult]:  # type: ignore[invalid-return-type]
        """Extract from a batch of texts in a single LLM call."""
        sections = "\n".join(f"=== SECTION {i + 1} ===\n{text[:4000]}" for i, text in enumerate(texts))

        # If expertise has custom extraction prompt, use it with multi-section adaptation
        if expertise and expertise.extraction_prompt:
            from khora.extraction.skills.composer import ExpertiseComposer

            composer = ExpertiseComposer()
            # Append multi-section response format to the text
            multi_text = (
                sections
                + """

## MULTI-SECTION RESPONSE FORMAT:
Return a JSON object with a "sections" array, one object per input section:
{"sections": [
    {"entities": [...], "relationships": [...], "events": [...]},
    ...
]}
Each section follows the entity/relationship format from the instructions above."""
            )

            # Get relationship types from expertise
            relationship_types = expertise.get_relationship_type_names() or DEFAULT_RELATIONSHIP_TYPES

            prompt_context = {
                **(context or {}),
                "text": multi_text,
                "entity_types": entity_types,
                "relationship_types": relationship_types,
                "tool_context": tool_context or "",
            }
            try:
                prompt = composer.render_prompt(
                    expertise.extraction_prompt,
                    expertise=expertise,
                    context=prompt_context,
                )
            except Exception as e:
                logger.warning(f"Failed to render extraction prompt for batch: {e}")
                # Fall through to default prompt below
                expertise = None

        # Fallback to hardcoded prompt (existing behavior)
        if not expertise or not expertise.extraction_prompt:
            # Get relationship types - expertise may exist but without custom extraction_prompt
            if expertise:
                relationship_types = expertise.get_relationship_type_names() or DEFAULT_RELATIONSHIP_TYPES
            else:
                relationship_types = DEFAULT_RELATIONSHIP_TYPES

            tool_prefix = f"{tool_context}\n\n" if tool_context else ""
            prompt = f"""{tool_prefix}Extract entities, relationships, and events from each text section below.

Entity types to find: {", ".join(entity_types)}
Relationship types to use: {", ".join(relationship_types)}

{sections}

Return a JSON object with a "sections" array, one object per section:
{{"sections": [
    {{"entities": [...], "relationships": [...], "events": [...]}},
    ...
]}}

Each section follows the same entity/relationship/event format.
Return ONLY valid JSON, no other text."""

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=self._retry_wait, min=self._retry_wait, max=10),
                before_sleep=before_sleep_log(logger, "WARNING"),
                reraise=True,
            ):
                with attempt:
                    async with self._semaphore:
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
                            response_format=self._get_multi_response_format(),
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
                    finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
                    model_used = getattr(response, "model", self._model)

                    if not content:
                        # Log more details about the response for debugging
                        logger.warning(
                            f"Empty response content from LLM in batch extraction. "
                            f"Model: {model_used}, finish_reason: {finish_reason}, "
                            f"response keys: {list(vars(response).keys()) if hasattr(response, '__dict__') else 'N/A'}"
                        )
                        return [
                            ExtractionResult(metadata={"error": "empty_response", "finish_reason": finish_reason})
                            for _ in texts
                        ]

                    # Check for truncated response (hit max_tokens limit)
                    if finish_reason == "length":
                        logger.warning(
                            f"LLM response truncated (finish_reason=length) in batch extraction. "
                            f"Model: {model_used}, batch_size: {len(texts)}. "
                            f"Consider increasing max_tokens or reducing batch size."
                        )
                        # Don't retry - truncation will happen again. Return empty results.
                        return [
                            ExtractionResult(metadata={"error": "truncated_response", "finish_reason": finish_reason})
                            for _ in texts
                        ]

                    # Parse JSON with error handling - don't retry on parse errors
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError as json_err:
                        # Log details to help diagnose the issue
                        logger.warning(
                            f"JSON parse error in batch extraction (finish_reason={finish_reason}): {json_err}. "
                            f"Model: {model_used}, content_length: {len(content)}, "
                            f"content_preview: {content[:200]}..."
                        )
                        # Raise to trigger retry - model may produce valid JSON on next attempt
                        raise

                    if not isinstance(data, dict):
                        logger.warning(f"Batch response is not a dict: {type(data)}")
                        return [ExtractionResult(metadata={"error": "invalid_response_type"}) for _ in texts]
                    sections_data = data.get("sections", [])

                    # Handle flat format: LLM returned {"entities": [...], "relationships": [...]}
                    # instead of {"sections": [...]}.  Treat as single-section response.
                    if not sections_data and (data.get("entities") or data.get("relationships")):
                        sections_data = [data]

                    results: list[ExtractionResult] = []
                    for i, text in enumerate(texts):
                        if i < len(sections_data):
                            results.append(self._parse_response(sections_data[i]))
                        else:
                            results.append(ExtractionResult())

                    return results
        except Exception as e:
            logger.error(f"Multi-extraction failed after {self._max_retries} attempts: {e}")
            return [ExtractionResult(metadata={"error": str(e)}) for _ in texts]

    def _compute_entity_confidence(self, entity_data: dict[str, Any]) -> float:
        """Compute confidence score for an entity based on extraction quality heuristics.

        QUALITY FIX: Instead of hardcoding 0.9, compute confidence based on:
        - Name quality (length, proper capitalization)
        - Description presence and length
        - Entity type specificity
        - Attribute completeness

        Returns:
            Confidence score in [0.5, 1.0] range
        """
        # If LLM provided explicit confidence, use it
        if entity_data.get("confidence") is not None:
            return float(entity_data["confidence"])

        score = 0.5  # Base score

        name = entity_data.get("name") or ""
        description = entity_data.get("description") or ""
        entity_type = entity_data.get("entity_type") or ""
        aliases = entity_data.get("aliases") or []

        # Name quality (max +0.2)
        if len(name) >= 2:
            score += 0.1
        if name and len(name) >= 3 and name[0].isupper():  # Proper capitalization
            score += 0.1

        # Description quality (max +0.2)
        if len(description) >= 10:
            score += 0.1
        if len(description) >= 30:
            score += 0.1

        # Entity type specificity (max +0.05)
        generic_types = {"CONCEPT", "THING", "OTHER", "UNKNOWN", "ENTITY"}
        if entity_type and entity_type.upper() not in generic_types:
            score += 0.05

        # Aliases indicate thorough extraction (max +0.05)
        if aliases and len(aliases) > 0:
            score += 0.05

        return min(1.0, score)

    def _compute_relationship_confidence(self, rel_data: dict[str, Any], entity_names: set[str]) -> float:
        """Compute confidence score for a relationship based on extraction quality.

        QUALITY FIX: Instead of hardcoding 0.9, compute confidence based on:
        - Source/target entity validity
        - Relationship type specificity
        - Description presence

        Args:
            rel_data: Relationship data from LLM
            entity_names: Set of entity names extracted in the same batch

        Returns:
            Confidence score in [0.5, 1.0] range
        """
        # If LLM provided explicit confidence, use it
        if rel_data.get("confidence") is not None:
            return float(rel_data["confidence"])

        score = 0.5  # Base score

        source = rel_data.get("source_entity") or ""
        target = rel_data.get("target_entity") or ""
        rel_type = rel_data.get("relationship_type") or ""
        description = rel_data.get("description") or ""

        # Entity reference validity (max +0.25)
        # Higher confidence if source/target match extracted entities
        if source in entity_names:
            score += 0.125
        if target in entity_names:
            score += 0.125

        # Relationship type specificity (max +0.15)
        generic_rels = {"RELATES_TO", "ASSOCIATED_WITH", "CONNECTED_TO", "RELATED"}
        if rel_type and rel_type.upper() not in generic_rels:
            score += 0.15

        # Description quality (max +0.1)
        if len(description) >= 10:
            score += 0.05
        if len(description) >= 25:
            score += 0.05

        return min(1.0, score)

    def _compute_event_confidence(self, event_data: dict[str, Any]) -> float:
        """Compute confidence score for an event based on extraction quality.

        QUALITY FIX: Instead of hardcoding 0.9, compute confidence based on:
        - Description quality
        - Temporal information presence
        - Participant count

        Returns:
            Confidence score in [0.5, 1.0] range
        """
        # If LLM provided explicit confidence, use it
        if event_data.get("confidence") is not None:
            return float(event_data["confidence"])

        score = 0.5  # Base score

        description = event_data.get("description") or ""
        occurred_at = event_data.get("occurred_at")
        participants = event_data.get("participants") or []

        # Description quality (max +0.2)
        if len(description) >= 10:
            score += 0.1
        if len(description) >= 30:
            score += 0.1

        # Temporal information (max +0.15)
        if occurred_at:
            score += 0.15

        # Participants (max +0.15)
        if len(participants) >= 1:
            score += 0.075
        if len(participants) >= 2:
            score += 0.075

        return min(1.0, score)

    def _parse_response(self, content: str | dict | None) -> ExtractionResult:
        """Parse the LLM response into an ExtractionResult.

        Accepts either a JSON string or a pre-parsed dict to avoid
        unnecessary json.dumps/json.loads round-trips in batch mode.
        """
        try:
            # Handle None or empty content
            if content is None or content == "":
                logger.warning("Empty response content from LLM")
                return ExtractionResult(metadata={"error": "empty_response"})

            # Accept pre-parsed dict directly (from extract_multi_batch)
            data = content if isinstance(content, dict) else json.loads(content)

            # Ensure data is actually a dict (not a string that parsed as string)
            if not isinstance(data, dict):
                logger.warning(f"Response parsed but is not a dict: {type(data)}")
                return ExtractionResult(metadata={"error": "invalid_response_type", "raw": str(data)[:500]})

            # First pass: collect entity names for relationship confidence scoring
            entity_names: set[str] = set()
            for e in data.get("entities", []):
                if isinstance(e, dict) and e.get("name"):
                    entity_names.add(e["name"])

            entities = []
            for e in data.get("entities", []):
                # Skip malformed entities (LLM sometimes returns strings instead of dicts)
                if not isinstance(e, dict):
                    logger.debug(f"Skipping malformed entity (not a dict): {type(e)}")
                    continue

                # Parse temporal info if present
                temporal = None
                if "temporal" in e and e["temporal"]:
                    t = e["temporal"]
                    if isinstance(t, dict):
                        temporal = TemporalInfo(
                            mentioned_at=t.get("mentioned_at"),
                            valid_from=t.get("valid_from"),
                            valid_until=t.get("valid_until"),
                        )

                # Ensure attributes is a dict (LLM sometimes returns a list)
                attrs = e.get("attributes", {})
                if not isinstance(attrs, dict):
                    attrs = {}

                # QUALITY FIX: Use heuristic confidence instead of hardcoded 0.9
                confidence = self._compute_entity_confidence(e)

                entities.append(
                    ExtractedEntity(
                        name=e.get("name") or "",
                        entity_type=e.get("entity_type") or "CONCEPT",
                        description=e.get("description") or "",
                        attributes=attrs,
                        aliases=e.get("aliases") or [],
                        temporal=temporal,
                        confidence=confidence,
                    )
                )

            relationships = []
            for r in data.get("relationships", []):
                # Skip malformed relationships (LLM sometimes returns strings instead of dicts)
                if not isinstance(r, dict):
                    logger.debug(f"Skipping malformed relationship (not a dict): {type(r)}")
                    continue

                # Parse temporal info if present
                temporal = None
                if "temporal" in r and r["temporal"]:
                    t = r["temporal"]
                    if isinstance(t, dict):
                        temporal = TemporalInfo(
                            occurred_at=t.get("occurred_at"),
                            valid_from=t.get("valid_from"),
                            valid_until=t.get("valid_until"),
                        )

                # QUALITY FIX: Use heuristic confidence instead of hardcoded 0.9
                confidence = self._compute_relationship_confidence(r, entity_names)

                relationships.append(
                    ExtractedRelationship(
                        source_entity=r.get("source_entity") or "",
                        target_entity=r.get("target_entity") or "",
                        relationship_type=r.get("relationship_type") or "RELATES_TO",
                        description=r.get("description") or "",
                        properties=r.get("properties") or {},
                        temporal=temporal,
                        confidence=confidence,
                    )
                )

            events = []
            for ev in data.get("events", []):
                # Skip malformed events (LLM sometimes returns strings instead of dicts)
                if not isinstance(ev, dict):
                    logger.debug(f"Skipping malformed event (not a dict): {type(ev)}")
                    continue

                # QUALITY FIX: Use heuristic confidence instead of hardcoded 0.9
                confidence = self._compute_event_confidence(ev)

                events.append(
                    ExtractedEvent(
                        description=ev.get("description") or "",
                        event_type=ev.get("event_type") or "EVENT",
                        occurred_at=ev.get("occurred_at"),
                        participants=ev.get("participants") or [],
                        confidence=confidence,
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
