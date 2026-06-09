"""LLM-based entity extraction using LiteLLM."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, stop_after_delay, wait_exponential

from khora.config.llm import get_shared_session, llm_call_timeout
from khora.extraction.embedders._request_telemetry import (
    parse_rate_limit_headers,
    set_connector_attributes,
    set_rate_limit_attributes,
)

from .base import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelationship,
    ExtractionResult,
    TemporalInfo,
)

# ---------------------------------------------------------------------------
# Regex patterns for tiered extraction (A-4)
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_DATE_RE = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_SINGLE_CAPITALIZED_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Markdown code-fence pattern: ```json\n...\n``` or ```\n...\n```
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)```\s*$", re.DOTALL)


def _strip_json_fences(content: str) -> str:
    """Strip markdown code fences and repair common JSON issues from LLM output.

    Models like Claude wrap JSON in ```json ... ``` fences even when
    asked for raw JSON, and sometimes produce malformed JSON (trailing
    commas, unquoted values, etc.).  This strips fences and applies
    lightweight repairs so json.loads() succeeds.
    """
    content = content.strip()
    m = _CODE_FENCE_RE.match(content)
    if m:
        content = m.group(1).strip()
    return _repair_json(content)


# Trailing comma before closing brace/bracket: ,\s*} or ,\s*]
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _repair_json(content: str) -> str:
    """Repair common JSON malformations from LLM output.

    Handles:
    - Trailing commas before } or ] (most common Claude issue)
    - Single-line // comments
    - Unquoted NaN/Infinity values
    """
    # Remove single-line comments (// ...)
    content = re.sub(r"//[^\n]*", "", content)
    # Fix trailing commas: [1, 2,] -> [1, 2]  or {"a": 1,} -> {"a": 1}
    content = _TRAILING_COMMA_RE.sub(r"\1", content)
    return content


if TYPE_CHECKING:
    from khora.config import LiteLLMConfig
    from khora.extraction.skills import ExpertiseConfig

# Default system prompt for extraction — kept static and long for prefix caching.
# OpenAI automatically caches identical prompt prefixes (~50% latency reduction).
# By putting all static guidelines in the system prompt, every call shares the same
# prefix regardless of document content.
DEFAULT_SYSTEM_PROMPT = """\
You are an expert entity extraction system. Extract entities and relationships from text and return them as structured JSON.

Guidelines:
- Extract all named entities mentioned or implied in the text — if a person, organization, location, or concept is referenced even indirectly, extract it
- Use canonical entity names (e.g., "Jennifer Walsh" not "Jenny", "Acme Corporation" not "Acme Corp")
- Include aliases for entities that have multiple names/abbreviations
- Extract temporal information when dates, times, or relative time references appear
- For STATE_CHANGE detection: when text indicates transitions ("switched from X to Y", "no longer X", "used to X", "previously X but now Y"), extract a STATE_CHANGE entity with these required attributes: {"entity_affected": "name of entity whose state changed", "previous_state": "old value", "new_state": "new value", "attribute_changed": "what changed (e.g. job_title, location, instrument)", "transition_date": "ISO date or null"}. Set valid_from to the transition date. Use INVOLVES to link it to the affected entity
- For EVENT detection: when text describes specific occurrences, extract the event with date, participants, and location when available
- Use temporal relationships (PRECEDES, FOLLOWS, INVOLVES) to connect events and state changes to other entities
- Ensure relationship source/target names match extracted entity names exactly
- RELATIONSHIP DENSITY: For N extracted entities, aim to identify N to 2N relationships between them. Include both explicit relationships (stated directly) and implicit ones (inferred from context, co-occurrence, or logical connection)
- For every pair of extracted entities that have any direct or implied connection, create a relationship. It is better to have a weak relationship than no relationship
- Use ASSOCIATED_WITH or RELATES_TO for weaker/implied connections when a more specific type doesn't fit
- Before returning, verify that each extracted entity has at least one relationship connecting it to another entity. If an entity appears isolated, re-examine the text for implicit connections (e.g., co-location, temporal co-occurrence, shared attributes, being mentioned in the same document)

Return ONLY valid JSON, no other text."""

# Extraction prompt template — dynamic content only (guidelines moved to system prompt).
# Structured-output models already enforce the JSON schema via response_format,
# so we skip the JSON example to save ~400-500 tokens per call.
EXTRACTION_PROMPT_STRUCTURED = """\
Extract entities, relationships, and temporal information from the following text.
{document_context}
Entity types to extract: {entity_types}
Relationship types to use: {relationship_types}

Text:
{text}"""

# Full extraction prompt for models without structured output support —
# includes inline JSON schema example since json_object mode doesn't enforce structure.
EXTRACTION_PROMPT = """\
Extract entities, relationships, and temporal information from the following text.
{document_context}
Entity types to extract: {entity_types}
Relationship types to use: {relationship_types}

Text:
{text}

Return a JSON object with the following structure:
{{
    "entities": [
        {{
            "name": "entity name (canonical form, properly capitalized)",
            "entity_type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT|STATE_CHANGE|TECHNOLOGY|PRODUCT|DATE|etc",
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
}}"""

# Second-pass prompt for focused relationship extraction (G-6: two-pass extraction)
# Reference: DeepStruct (Wang et al., 2022) shows 30-40% more relationships with two-pass
RELATIONSHIP_EXTRACTION_PROMPT = """The following entities were extracted from a text. Identify ALL relationships between them.

Entities:
{entity_list}

Original text:
{text}

Relationship types to use: {relationship_types}

Return a JSON object:
{{
    "relationships": [
        {{
            "source_entity": "entity name (must match an entity above)",
            "target_entity": "entity name (must match an entity above)",
            "relationship_type": "type from the list above or a specific type",
            "description": "brief description of relationship",
            "confidence": null,
            "temporal": null
        }}
    ]
}}

Guidelines:
- Only use entity names from the list above
- Focus on relationships that are implied but not explicitly stated
- Consider co-occurrence, organizational, temporal, and causal relationships
- Use ASSOCIATED_WITH or RELATES_TO for weaker connections
- Do NOT repeat relationships that are obvious from the text — focus on non-obvious connections

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
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.4",
        "gpt-5.4-mini",
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
        "gpt-5-nano": 5,  # 1M context
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
    _BATCH_FAILURE_THRESHOLD = 2  # Circuit breaker trips after this many consecutive batch failures

    # Tracks models we've already warned about (to avoid log spam across many
    # extractor instances configured with the same off-allowlist model).
    _WARNED_NON_ALLOWLIST_MODELS: set[str] = set()

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.0,  # Deterministic extraction for reproducible entity sets
        max_tokens: int = 16384,
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
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # Persists across extract_batch calls so the circuit breaker can actually trip
        self._consecutive_batch_failures = 0

        # Register litellm failure callback to log 429 rate-limit responses
        self._register_rate_limit_callback()

        # Warn once per process if the configured model isn't on the
        # json_schema allowlist. Off-allowlist models fall back to a looser
        # json_object response_format, which means entity/relationship key
        # names depend on whatever the LLM emits. The parser accepts both
        # `entity_type` and `type` (and `source`/`target`) so extraction
        # still works, but adding the model to the allowlist remains the
        # cleanest path when the provider supports strict structured output.
        # See issue #839.
        if model not in self.MODELS_REQUIRING_JSON_SCHEMA and model not in self._WARNED_NON_ALLOWLIST_MODELS:
            self._WARNED_NON_ALLOWLIST_MODELS.add(model)
            logger.warning(
                "Extraction model '{}' is not on the json_schema allowlist; "
                "falling back to json_object response format. "
                "Entity/relationship types should still parse - the parser "
                "accepts both 'entity_type' and 'type' keys - but you may "
                "want to add this model to MODELS_REQUIRING_JSON_SCHEMA if "
                "its provider supports strict structured output.",
                model,
            )

    @staticmethod
    def _register_rate_limit_callback() -> None:
        """Register a litellm failure callback to log 429 rate-limit responses."""
        import litellm

        def _on_failure(kwargs: dict[str, Any], response: Any, start_time: Any, end_time: Any) -> None:  # noqa: ANN401
            if hasattr(response, "status_code") and response.status_code == 429:
                retry_after = getattr(response, "headers", {}).get("retry-after", "unknown")
                logger.warning(
                    "OpenAI rate limit hit (429): retry-after=%s, model=%s",
                    retry_after,
                    kwargs.get("model", "unknown"),
                )

        # Append without clobbering existing callbacks
        if not isinstance(litellm.failure_callback, list):
            litellm.failure_callback = []
        if not any(getattr(cb, "__name__", None) == "_on_failure" for cb in litellm.failure_callback):
            litellm.failure_callback.append(_on_failure)

    @staticmethod
    def _log_rate_limit_headers(response: Any) -> None:  # noqa: ANN401
        """Log rate-limit headers from an LLM response at DEBUG level."""
        parsed = parse_rate_limit_headers(response)
        rl_remaining = parsed.get("ratelimit.remaining_requests", "")
        rl_limit = parsed.get("ratelimit.limit_requests", "")
        if rl_limit or rl_remaining:
            logger.debug("Rate limit: {}/{} remaining requests", rl_remaining, rl_limit)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses ~3 chars per token as a conservative heuristic for English text.
        This slightly overestimates to provide safety margin.
        """
        return len(text) // 3

    @staticmethod
    def _regex_extract(text: str) -> ExtractionResult:
        """Extract entities from short text using regex patterns.

        Used for texts under the tiered extraction threshold (default 200 chars)
        to avoid LLM API calls for content that's too short for rich extraction.
        """
        entities: list[ExtractedEntity] = []
        seen_names: set[str] = set()

        def _add(name: str, etype: str, conf: float = 0.6) -> None:
            lower = name.lower().strip()
            if lower and lower not in seen_names and len(lower) > 1:
                seen_names.add(lower)
                entities.append(ExtractedEntity(name=name.strip(), entity_type=etype, confidence=conf))

        for m in _EMAIL_RE.finditer(text):
            _add(m.group(), "EMAIL", 0.9)
        for m in _URL_RE.finditer(text):
            _add(m.group(), "URL", 0.9)
        for m in _DATE_RE.finditer(text):
            _add(m.group(), "DATE", 0.8)
        for m in _PROPER_NOUN_RE.finditer(text):
            _add(m.group(), "PERSON", 0.5)

        # Create pairwise CO_OCCURS_WITH relationships for co-occurring entities
        relationships: list[ExtractedRelationship] = []
        if len(entities) >= 2:
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    relationships.append(
                        ExtractedRelationship(
                            source_entity=e1.name,
                            target_entity=e2.name,
                            relationship_type="CO_OCCURS_WITH",
                            description="Co-occurs in same text",
                            confidence=0.4,
                        )
                    )

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            metadata={"extraction_method": "regex"},
        )

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
        current_density_limit = max_batch_size  # Track most restrictive limit in batch

        for text in texts:
            # Match truncation in _extract_multi_batch (4000 chars)
            text_tokens = self._estimate_tokens(text[:4000])

            # Density-based limit: shorter texts can be batched more aggressively.
            # Short messages (e.g. Slack) are packed densely to minimize
            # API round-trips while staying within context limits.
            text_len = len(text)
            if text_len < 150:
                density_limit = 50  # Very short texts (Slack reactions, status updates)
            elif text_len < 300:
                density_limit = 30
            elif text_len < 800:
                density_limit = 15
            elif text_len < 2000:
                density_limit = 8
            else:
                density_limit = 3

            effective_limit = min(max_batch_size, density_limit)

            # Check if adding this text would exceed budget or effective batch size
            if current_batch and (
                current_tokens + text_tokens > max_input_tokens
                or len(current_batch) >= min(current_density_limit, effective_limit)
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = prompt_overhead
                current_density_limit = max_batch_size

            current_batch.append(text)
            current_tokens += text_tokens
            # Track the most restrictive density limit seen in this batch
            current_density_limit = min(current_density_limit, effective_limit)

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
                                        "temporal": {
                                            "anyOf": [
                                                {
                                                    "type": "object",
                                                    "properties": {
                                                        "valid_from": {"type": ["string", "null"]},
                                                        "valid_until": {"type": ["string", "null"]},
                                                        "mentioned_at": {"type": ["string", "null"]},
                                                    },
                                                    "required": ["valid_from", "valid_until", "mentioned_at"],
                                                    "additionalProperties": False,
                                                },
                                                {"type": "null"},
                                            ],
                                        },
                                    },
                                    "required": ["name", "entity_type", "description", "aliases", "temporal"],
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
                                        "confidence": {"type": ["number", "null"]},
                                        "temporal": {
                                            "anyOf": [
                                                {
                                                    "type": "object",
                                                    "properties": {
                                                        "valid_from": {"type": ["string", "null"]},
                                                        "valid_until": {"type": ["string", "null"]},
                                                        "occurred_at": {"type": ["string", "null"]},
                                                    },
                                                    "required": ["valid_from", "valid_until", "occurred_at"],
                                                    "additionalProperties": False,
                                                },
                                                {"type": "null"},
                                            ],
                                        },
                                    },
                                    "required": [
                                        "source_entity",
                                        "target_entity",
                                        "relationship_type",
                                        "description",
                                        "confidence",
                                        "temporal",
                                    ],
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
                                "temporal": {
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "properties": {
                                                "valid_from": {"type": ["string", "null"]},
                                                "valid_until": {"type": ["string", "null"]},
                                                "mentioned_at": {"type": ["string", "null"]},
                                            },
                                            "required": ["valid_from", "valid_until", "mentioned_at"],
                                            "additionalProperties": False,
                                        },
                                        {"type": "null"},
                                    ],
                                },
                            },
                            "required": ["name", "entity_type", "description", "aliases", "temporal"],
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
                                "confidence": {"type": ["number", "null"]},
                                "temporal": {
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "properties": {
                                                "valid_from": {"type": ["string", "null"]},
                                                "valid_until": {"type": ["string", "null"]},
                                                "occurred_at": {"type": ["string", "null"]},
                                            },
                                            "required": ["valid_from", "valid_until", "occurred_at"],
                                            "additionalProperties": False,
                                        },
                                        {"type": "null"},
                                    ],
                                },
                            },
                            "required": [
                                "source_entity",
                                "target_entity",
                                "relationship_type",
                                "description",
                                "confidence",
                                "temporal",
                            ],
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
            temperature=0.0,  # Deterministic extraction for reproducible entity sets
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
        relationship_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExtractionResult:  # type: ignore[invalid-return-type]
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            entity_types: Optional list of entity types to extract
            relationship_types: Optional list of relationship types to extract
            expertise: Optional ExpertiseConfig for domain-specific extraction
            context: Optional context dict for prompt template rendering

        Returns:
            ExtractionResult containing entities and relationships
        """
        if not text.strip():
            return ExtractionResult()

        # Resolve entity types: explicit param > expertise
        if entity_types is None and expertise:
            entity_types = expertise.get_entity_type_names() or None

        # Resolve relationship types: explicit param > expertise
        if relationship_types is None and expertise:
            relationship_types = expertise.get_relationship_type_names() or None

        try:
            import litellm
        except ImportError:
            raise RuntimeError("litellm package not installed. Run: pip install litellm")

        # Render prompts based on expertise
        system_prompt = self._render_system_prompt(expertise, context)
        extraction_prompt = self._render_extraction_prompt(
            text, entity_types, expertise, context, relationship_types=relationship_types
        )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries) | stop_after_delay(180),
                wait=wait_exponential(multiplier=self._retry_wait, min=self._retry_wait, max=10),
                before_sleep=lambda retry_state: logger.warning(
                    "Retrying LLM call (attempt {}) after {}: {}",
                    retry_state.attempt_number,
                    (
                        type(retry_state.outcome.exception()).__name__
                        if retry_state.outcome and retry_state.outcome.failed
                        else "unknown"
                    ),
                    (
                        str(retry_state.outcome.exception())[:200]
                        if retry_state.outcome and retry_state.outcome.failed
                        else "unknown"
                    ),
                ),
                reraise=True,
            ):
                with attempt:
                    async with self._semaphore:
                        import time as _time

                        from khora.telemetry import get_collector, trace_span

                        _t0 = _time.perf_counter()
                        # Don't cap below the configured max_tokens — JSON structure overhead
                        # is constant regardless of input size. Let the model use its full budget.
                        effective_max_tokens = self._max_tokens

                        with trace_span("khora.extraction.llm_call", model=self._model, call_type="single") as llm_span:
                            set_connector_attributes(llm_span, get_shared_session())
                            response = await asyncio.wait_for(
                                litellm.acompletion(
                                    model=self._model,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": extraction_prompt},
                                    ],
                                    temperature=self._temperature,
                                    max_tokens=effective_max_tokens,
                                    timeout=self._timeout,
                                    num_retries=0,
                                    response_format=self._get_response_format(),
                                    shared_session=get_shared_session(),
                                ),
                                llm_call_timeout(self._timeout),
                            )
                            set_rate_limit_attributes(llm_span, response)
                        _latency = (_time.perf_counter() - _t0) * 1000
                        self._log_rate_limit_headers(response)

                        # Record telemetry
                        usage = getattr(response, "usage", None)
                        _pt = getattr(usage, "prompt_tokens", 0) or 0
                        _ct = getattr(usage, "completion_tokens", 0) or 0
                        _tt = getattr(usage, "total_tokens", 0) or 0
                        get_collector().record_llm_call(
                            operation="entity_extraction",
                            model=self._model,
                            prompt_tokens=_pt,
                            completion_tokens=_ct,
                            total_tokens=_tt,
                            latency_ms=_latency,
                        )

                        from khora.khora import LLMUsage
                        from khora.telemetry.context import record_usage

                        record_usage(
                            LLMUsage(
                                operation="entity_extraction",
                                model=self._model,
                                prompt_tokens=_pt,
                                completion_tokens=_ct,
                                total_tokens=_tt,
                                latency_ms=_latency,
                            )
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

                    # Two-pass relationship extraction (G-6): if first pass
                    # extracted many entities but few relationships, run a
                    # focused second pass to catch missed connections.
                    # Reference: DeepStruct (Wang et al., 2022)
                    if self._should_run_second_pass(result):
                        second_pass = await self._extract_additional_relationships(
                            result.entities,
                            text,
                            relationship_types=relationship_types,
                        )
                        if second_pass:
                            result = self._merge_relationships(result, second_pass)

                    # Apply confidence filtering from expertise if available
                    if expertise:
                        result = self._filter_by_confidence(result, expertise)

                    return result
        except Exception as e:
            logger.error(f"Extraction failed after {self._max_retries} attempts: {e}")
            return ExtractionResult(metadata={"error": str(e)})

    @staticmethod
    def _should_run_second_pass(result: ExtractionResult) -> bool:
        """Check if a second-pass relationship extraction should run.

        Cost guard: only trigger when the first pass extracted many entities
        but few relationships, suggesting missed connections.

        Args:
            result: First-pass extraction result

        Returns:
            True if second pass should run
        """
        num_entities = len(result.entities)
        num_relationships = len(result.relationships)
        return num_entities >= 2 and num_relationships < num_entities - 1

    async def _extract_additional_relationships(
        self,
        entities: list[ExtractedEntity],
        text: str,
        *,
        relationship_types: list[str] | None = None,
    ) -> list[ExtractedRelationship]:
        """Run a focused second-pass LLM call to find relationships between entities.

        Sends the entity list back to the LLM with a targeted prompt asking
        specifically about relationships. This catches 30-40% more connections
        than a single pass (DeepStruct, Wang et al. 2022).

        Args:
            entities: Entities from the first pass
            text: Original text chunk
            relationship_types: Allowed relationship types

        Returns:
            List of additional relationships found
        """
        if not entities:
            return []

        try:
            import litellm
        except ImportError:
            return []

        entity_list = "\n".join(f"- {e.name} ({e.entity_type})" for e in entities)
        rel_types = relationship_types or []

        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            entity_list=entity_list,
            text=text[:8000],
            relationship_types=", ".join(rel_types),
        )

        try:
            async with self._semaphore:
                import time as _time

                from khora.telemetry import get_collector, trace_span

                _t0 = _time.perf_counter()
                with trace_span(
                    "khora.extraction.llm_call", model=self._model, call_type="relationship_second_pass"
                ) as llm_span:
                    set_connector_attributes(llm_span, get_shared_session())
                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=self._model,
                            messages=[
                                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=self._temperature,
                            max_tokens=self._max_tokens,
                            timeout=self._timeout,
                            response_format=self._get_response_format(),
                            shared_session=get_shared_session(),
                        ),
                        llm_call_timeout(self._timeout),
                    )
                    set_rate_limit_attributes(llm_span, response)
                _latency = (_time.perf_counter() - _t0) * 1000
                self._log_rate_limit_headers(response)

                usage = getattr(response, "usage", None)
                _pt = getattr(usage, "prompt_tokens", 0) or 0
                _ct = getattr(usage, "completion_tokens", 0) or 0
                _tt = getattr(usage, "total_tokens", 0) or 0
                get_collector().record_llm_call(
                    operation="relationship_extraction_second_pass",
                    model=self._model,
                    prompt_tokens=_pt,
                    completion_tokens=_ct,
                    total_tokens=_tt,
                    latency_ms=_latency,
                )

                from khora.khora import LLMUsage
                from khora.telemetry.context import record_usage

                record_usage(
                    LLMUsage(
                        operation="relationship_extraction_second_pass",
                        model=self._model,
                        prompt_tokens=_pt,
                        completion_tokens=_ct,
                        total_tokens=_tt,
                        latency_ms=_latency,
                    )
                )

            content = response.choices[0].message.content
            if not content:
                return []

            data = json.loads(_strip_json_fences(content))
            if not isinstance(data, dict):
                return []

            entity_names = {e.name for e in entities}
            relationships = []
            for r in data.get("relationships", []):
                if not isinstance(r, dict):
                    continue
                source = r.get("source_entity") or r.get("source") or ""
                target = r.get("target_entity") or r.get("target") or ""
                # Only accept relationships referencing known entities
                if source not in entity_names or target not in entity_names:
                    continue
                confidence = self._compute_relationship_confidence(r, entity_names)
                relationships.append(
                    ExtractedRelationship(
                        source_entity=source,
                        target_entity=target,
                        relationship_type=r.get("relationship_type") or r.get("type") or "RELATES_TO",
                        description=r.get("description", ""),
                        confidence=confidence,
                    )
                )

            logger.debug(f"Second-pass extraction found {len(relationships)} additional relationships")
            return relationships

        except Exception as e:
            logger.warning(f"Second-pass relationship extraction failed: {e}")
            return []

    @staticmethod
    def _merge_relationships(
        result: ExtractionResult,
        additional: list[ExtractedRelationship],
    ) -> ExtractionResult:
        """Merge second-pass relationships into the first-pass result.

        Deduplicates by (source_entity, target_entity, relationship_type).

        Args:
            result: First-pass extraction result
            additional: Relationships from the second pass

        Returns:
            Updated ExtractionResult with merged relationships
        """
        existing = {(r.source_entity, r.target_entity, r.relationship_type) for r in result.relationships}
        merged = list(result.relationships)
        added = 0
        for rel in additional:
            key = (rel.source_entity, rel.target_entity, rel.relationship_type)
            if key not in existing:
                merged.append(rel)
                existing.add(key)
                added += 1

        if added:
            logger.debug(f"Merged {added} new relationships from second pass")

        return ExtractionResult(
            entities=result.entities,
            relationships=merged,
            events=result.events,
            metadata={**result.metadata, "second_pass_relationships": added},
        )

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

    @staticmethod
    def _build_document_context(context: dict[str, Any] | None) -> str:
        """Build temporal document context block for extraction prompts.

        When document_created_at is available in the context dict, generates
        instructions for the LLM to resolve relative temporal references.

        Args:
            context: Optional context dict that may contain document_created_at

        Returns:
            Document context string to include in the prompt, or empty string
        """
        if not context:
            return ""

        created_at = context.get("document_created_at")
        if not created_at:
            return ""

        # Convert to string if it's a datetime object
        if hasattr(created_at, "isoformat"):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = str(created_at)

        lines = [
            "\nDOCUMENT CONTEXT:",
            f"- Document created: {created_at_str}",
        ]

        source_tool = context.get("source_tool")
        if source_tool:
            lines.append(f"- Source: {source_tool}")

        lines.append(
            "\nWhen you encounter relative temporal references (e.g., 'yesterday', "
            "'last week', 'three days ago'), resolve them to absolute ISO 8601 dates "
            "using the document creation date as reference. Always provide ISO 8601 "
            "dates in temporal fields, not relative references."
        )

        return "\n".join(lines)

    def _render_extraction_prompt(
        self,
        text: str,
        entity_types: list[str],
        expertise: ExpertiseConfig | None,
        context: dict[str, Any] | None,
        *,
        relationship_types: list[str] | None = None,
    ) -> str:
        """Render the extraction prompt, optionally using expertise config.

        Args:
            text: Text to extract from
            entity_types: Resolved entity types to extract
            expertise: Optional expertise config
            context: Optional context dict
            relationship_types: Resolved relationship types (defaults applied if None)
        """
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

        # Build document temporal context from the context dict
        document_context = self._build_document_context(context)

        # Use the stripped prompt for structured-output models (JSON schema already enforced).
        # This saves ~400-500 tokens per call since the JSON example is redundant.
        prompt_template = (
            EXTRACTION_PROMPT_STRUCTURED if self._model in self.MODELS_REQUIRING_JSON_SCHEMA else EXTRACTION_PROMPT
        )

        prompt = prompt_template.format(
            entity_types=", ".join(entity_types),
            relationship_types=", ".join(relationship_types or []),
            text=text[:8000],  # Truncate very long texts
            document_context=document_context,
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
        relationship_types: list[str] | None = None,
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
            relationship_types: Optional list of relationship types to extract
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
            relationship_types=relationship_types,
            expertise=expertise,
            context=context,
        )

    async def extract_multi(
        self,
        texts: list[str],
        *,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
        batch_size: int = 5,
        max_input_tokens: int | None = None,
        tiered_extraction: bool = True,
        tier1_max_chars: int = 20,
    ) -> list[ExtractionResult]:
        """Extract entities from multiple texts in grouped LLM calls.

        Groups texts into batches using adaptive token-budget-based batching,
        reducing API round-trips while avoiding context overflow.

        When tiered_extraction=True, texts shorter than tier1_max_chars are
        processed via fast regex extraction instead of LLM. The default
        threshold (20 chars) only skips truly trivial texts — all substantive
        content goes through LLM extraction for proper entity/relationship
        discovery. Short texts are packed together in batched LLM calls.

        Args:
            texts: List of texts to extract from
            entity_types: Optional list of entity types to extract
            relationship_types: Optional list of relationship types to extract
            expertise: Optional ExpertiseConfig for domain-specific extraction
            context: Optional context dict for prompt template rendering
            batch_size: Maximum number of texts per LLM call (fallback/cap)
            max_input_tokens: Token budget for input. If None, auto-calculated
                from max_tokens using model-aware multipliers.
            tiered_extraction: If True, use regex for short texts (<tier1_max_chars)
            tier1_max_chars: Character threshold for regex-only extraction (default 20)

        Returns:
            List of ExtractionResult objects (one per input text)
        """
        if not texts:
            return []

        # Tiered extraction: separate short texts for regex-only processing
        if tiered_extraction:
            regex_results: dict[int, ExtractionResult] = {}
            llm_indices: list[int] = []
            llm_texts: list[str] = []
            for i, text in enumerate(texts):
                if len(text) < tier1_max_chars:
                    regex_results[i] = self._regex_extract(text)
                else:
                    llm_indices.append(i)
                    llm_texts.append(text)

            if regex_results:
                logger.debug(f"Tiered extraction: {len(regex_results)} texts via regex, {len(llm_texts)} via LLM")

            if not llm_texts:
                return [regex_results[i] for i in range(len(texts))]

            # Process LLM texts through the normal pipeline
            llm_results = await self.extract_multi(
                llm_texts,
                entity_types=entity_types,
                relationship_types=relationship_types,
                expertise=expertise,
                context=context,
                batch_size=batch_size,
                max_input_tokens=max_input_tokens,
                tiered_extraction=False,  # Don't recurse
            )

            # Reassemble in original order
            combined: list[ExtractionResult] = []
            llm_idx = 0
            for i in range(len(texts)):
                if i in regex_results:
                    combined.append(regex_results[i])
                else:
                    combined.append(llm_results[llm_idx])
                    llm_idx += 1
            return combined

        # Resolve entity types: explicit param > expertise
        if entity_types is None and expertise:
            entity_types = expertise.get_entity_type_names() or None

        # Resolve relationship types: explicit param > expertise
        if relationship_types is None and expertise:
            relationship_types = expertise.get_relationship_type_names() or None

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
        logger.debug(
            "Extraction starting: model={}, timeout={}s, max_tokens={}, texts={}, batches={}",
            self._model,
            self._timeout,
            self._max_tokens,
            len(texts),
            len(batches),
        )
        all_results: list[ExtractionResult] = []

        # Build system prompt from expertise if available
        system_prompt = self._render_system_prompt(expertise, context)
        tool_context = self._build_tool_context(expertise, context)

        def _is_truncated(r: ExtractionResult) -> bool:
            return r.metadata.get("error") == "truncated_response"

        async def _bisect_and_extract(batch: list[str]) -> list[ExtractionResult]:
            """Recursively bisect a batch when output truncation is detected.

            Returns unfiltered results; filtering is applied by the caller.
            Complexity: O(log2(N)) depth, at most 2N LLM calls worst-case.
            """
            if len(batch) == 0:
                return []

            if len(batch) == 1:
                # Floor: can't split further — use single-doc extraction
                logger.debug("Bisection floor: extracting single text individually")
                result = await self.extract(
                    batch[0],
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    expertise=expertise,
                    context=context,
                )
                return [result]

            mid = len(batch) // 2
            left_batch, right_batch = batch[:mid], batch[mid:]
            logger.info(
                "Bisecting batch of {} → [{}, {}] to resolve output truncation",
                len(batch),
                len(left_batch),
                len(right_batch),
            )

            async def _process_half(half: list[str]) -> list[ExtractionResult]:
                half_results = await self._extract_multi_batch(
                    half,
                    entity_types,
                    litellm,
                    system_prompt=system_prompt,
                    tool_context=tool_context,
                    expertise=expertise,
                    context=context,
                    relationship_types=relationship_types,
                )
                if any(_is_truncated(r) for r in half_results):
                    return await _bisect_and_extract(half)
                return half_results

            left_results, right_results = await asyncio.gather(
                _process_half(left_batch),
                _process_half(right_batch),
            )
            return left_results + right_results

        # Circuit breaker: after consecutive batch failures, skip batch mode entirely.
        # _consecutive_batch_failures is an instance attribute so it persists across
        # extract_batch invocations, making the circuit breaker actually reachable.

        async def _run_batch(batch: list[str]) -> list[ExtractionResult]:
            # Circuit breaker tripped — go straight to single-doc extraction
            if self._consecutive_batch_failures >= self._BATCH_FAILURE_THRESHOLD and len(batch) > 1:
                logger.warning(
                    "Circuit breaker active: skipping batch mode after {} consecutive failures, "
                    "extracting {} texts individually",
                    self._consecutive_batch_failures,
                    len(batch),
                )
                single_results = []
                for text in batch:
                    result = await self.extract(
                        text,
                        entity_types=entity_types,
                        relationship_types=relationship_types,
                        expertise=expertise,
                        context=context,
                    )
                    single_results.append(result)
                self._consecutive_batch_failures = 0  # Reset: single-doc fallback succeeded
                if expertise:
                    single_results = [self._filter_by_confidence(r, expertise) for r in single_results]
                return single_results

            results = await self._extract_multi_batch(
                batch,
                entity_types,
                litellm,
                system_prompt=system_prompt,
                tool_context=tool_context,
                expertise=expertise,
                context=context,
                relationship_types=relationship_types,
            )

            # --- Truncation handling: bisect instead of falling back to single-doc ---
            truncated_indices = [i for i, r in enumerate(results) if _is_truncated(r)]
            if truncated_indices:
                if len(truncated_indices) == len(batch):
                    # All results truncated
                    if len(batch) == 1:
                        # Floor: single-doc extraction
                        result = await self.extract(
                            batch[0],
                            entity_types=entity_types,
                            relationship_types=relationship_types,
                            expertise=expertise,
                            context=context,
                        )
                        results = [result]
                    else:
                        results = await _bisect_and_extract(batch)
                else:
                    # Partial: keep successes, bisect only the truncated texts
                    failed_texts = [batch[i] for i in truncated_indices]
                    bisected = await _bisect_and_extract(failed_texts)
                    for new_i, orig_i in enumerate(truncated_indices):
                        results[orig_i] = bisected[new_i]
                self._consecutive_batch_failures = 0  # Bisection succeeded — clear failure streak
                if expertise:
                    results = [self._filter_by_confidence(r, expertise) for r in results]
                return results

            # --- Non-truncation failure handling: circuit breaker + single-doc fallback ---
            all_failed = all(r.metadata.get("error") for r in results)
            if all_failed and len(batch) > 1:
                self._consecutive_batch_failures += 1
                logger.info(
                    "Batch extraction failed for {} texts (consecutive failures: {}), "
                    "falling back to single-document extraction",
                    len(batch),
                    self._consecutive_batch_failures,
                )
                single_results = []
                for text in batch:
                    result = await self.extract(
                        text,
                        entity_types=entity_types,
                        relationship_types=relationship_types,
                        expertise=expertise,
                        context=context,
                    )
                    single_results.append(result)
                results = single_results
            else:
                # Successful batch — reset circuit breaker
                self._consecutive_batch_failures = 0

            if expertise:
                results = [self._filter_by_confidence(r, expertise) for r in results]
            return results

        # Process batches in waves: concurrent within each wave, circuit breaker
        # checked between waves.  This restores most of the throughput of the
        # original asyncio.gather() while letting the breaker trip before the
        # next wave launches.  Worst-case waste: (wave_size - 1) extra doomed
        # API calls in the wave where the first failure occurs.
        wave_size = 4
        for wave_start in range(0, len(batches), wave_size):
            if self._consecutive_batch_failures >= self._BATCH_FAILURE_THRESHOLD:
                # Circuit breaker tripped between waves — remaining batches
                # go through single-doc extraction via _run_batch's own check.
                for batch in batches[wave_start:]:
                    results = await _run_batch(batch)
                    all_results.extend(results)
                break
            wave = batches[wave_start : wave_start + wave_size]
            wave_results = await asyncio.gather(*[_run_batch(b) for b in wave])
            for results in wave_results:
                all_results.extend(results)

        error_count = sum(1 for r in all_results if r.metadata.get("error"))
        logger.debug(
            "Extraction complete: {} results ({} errors) from {} texts in {} batches",
            len(all_results),
            error_count,
            len(texts),
            len(batches),
        )
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
        relationship_types: list[str] | None = None,
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
            tool_prefix = f"{tool_context}\n\n" if tool_context else ""
            prompt = f"""{tool_prefix}Extract entities, relationships, and events from each text section below.

Entity types to find: {", ".join(entity_types)}
Relationship types to use: {", ".join(relationship_types or [])}

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
                stop=stop_after_attempt(self._max_retries) | stop_after_delay(180),
                wait=wait_exponential(multiplier=self._retry_wait, min=self._retry_wait, max=10),
                before_sleep=lambda retry_state: logger.warning(
                    "Retrying LLM call (attempt {}) after {}: {}",
                    retry_state.attempt_number,
                    (
                        type(retry_state.outcome.exception()).__name__
                        if retry_state.outcome and retry_state.outcome.failed
                        else "unknown"
                    ),
                    (
                        str(retry_state.outcome.exception())[:200]
                        if retry_state.outcome and retry_state.outcome.failed
                        else "unknown"
                    ),
                ),
                reraise=True,
            ):
                with attempt:
                    async with self._semaphore:
                        import time as _time

                        from khora.telemetry import get_collector, trace_span

                        _t0 = _time.perf_counter()
                        with trace_span(
                            "khora.extraction.llm_call",
                            model=self._model,
                            call_type="multi_batch",
                            batch_size=len(texts),
                        ) as llm_span:
                            set_connector_attributes(llm_span, get_shared_session())
                            response = await asyncio.wait_for(
                                litellm.acompletion(
                                    model=self._model,
                                    messages=[
                                        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                                        {"role": "user", "content": prompt},
                                    ],
                                    temperature=self._temperature,
                                    max_tokens=self._max_tokens,
                                    timeout=self._timeout,
                                    num_retries=0,
                                    response_format=self._get_multi_response_format(),
                                    shared_session=get_shared_session(),
                                ),
                                llm_call_timeout(self._timeout),
                            )
                            set_rate_limit_attributes(llm_span, response)
                        _latency = (_time.perf_counter() - _t0) * 1000
                        self._log_rate_limit_headers(response)

                        # Record telemetry
                        usage = getattr(response, "usage", None)
                        _pt = getattr(usage, "prompt_tokens", 0) or 0
                        _ct = getattr(usage, "completion_tokens", 0) or 0
                        _tt = getattr(usage, "total_tokens", 0) or 0
                        logger.debug(
                            "Batch extraction complete: {} texts in {:.0f}ms, prompt={} completion={} tokens",
                            len(texts),
                            _latency,
                            _pt,
                            _ct,
                        )
                        get_collector().record_llm_call(
                            operation="entity_extraction_multi",
                            model=self._model,
                            prompt_tokens=_pt,
                            completion_tokens=_ct,
                            total_tokens=_tt,
                            latency_ms=_latency,
                            metadata={"batch_size": len(texts)},
                        )

                        from khora.khora import LLMUsage
                        from khora.telemetry.context import record_usage

                        record_usage(
                            LLMUsage(
                                operation="entity_extraction_multi",
                                model=self._model,
                                prompt_tokens=_pt,
                                completion_tokens=_ct,
                                total_tokens=_tt,
                                latency_ms=_latency,
                                batch_size=len(texts),
                            )
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
                        data = json.loads(_strip_json_fences(content))
                    except json.JSONDecodeError as json_err:
                        # "Unterminated string" = output token limit hit mid-JSON (deterministic).
                        # Retrying will produce the same result — return truncation error so the
                        # caller can bisect the batch instead.
                        if "Unterminated string" in str(json_err):
                            logger.warning(
                                f"LLM response truncated (Unterminated string, "
                                f"finish_reason={finish_reason}) in batch extraction. "
                                f"Model: {model_used}, batch_size: {len(texts)}. "
                                f"Bisection will handle this."
                            )
                            return [
                                ExtractionResult(
                                    metadata={"error": "truncated_response", "finish_reason": finish_reason}
                                )
                                for _ in texts
                            ]
                        # Other JSON errors — log and retry (model may produce valid JSON next attempt)
                        logger.warning(
                            f"JSON parse error in batch extraction (finish_reason={finish_reason}): {json_err}. "
                            f"Model: {model_used}, content_length: {len(content)}, "
                            f"content_preview: {content[:200]}..."
                        )
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
        entity_type = entity_data.get("entity_type") or entity_data.get("type") or ""
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

        source = rel_data.get("source_entity") or rel_data.get("source") or ""
        target = rel_data.get("target_entity") or rel_data.get("target") or ""
        rel_type = rel_data.get("relationship_type") or rel_data.get("type") or ""
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
            data = content if isinstance(content, dict) else json.loads(_strip_json_fences(content))

            # Ensure data is actually a dict (not a string that parsed as string)
            if not isinstance(data, dict):
                logger.warning(f"Response parsed but is not a dict: {type(data)}")
                return ExtractionResult(metadata={"error": "invalid_response_type", "raw": str(data)[:500]})

            # First pass: collect entity names for relationship confidence scoring.
            # Strip whitespace so " " (whitespace-only) is treated as empty.
            entity_names: set[str] = set()
            for e in data.get("entities", []):
                if isinstance(e, dict):
                    raw = e.get("name") or ""
                    nm = raw.strip() if isinstance(raw, str) else ""
                    if nm:
                        entity_names.add(nm)

            entities = []
            skipped_entities_empty_name = 0
            for e in data.get("entities", []):
                # Skip malformed entities (LLM sometimes returns strings instead of dicts)
                if not isinstance(e, dict):
                    logger.debug(f"Skipping malformed entity (not a dict): {type(e)}")
                    continue

                # Skip entities with empty / whitespace / missing / non-string name (#894).
                raw_name = e.get("name") or ""
                name = raw_name.strip() if isinstance(raw_name, str) else ""
                if not name:
                    skipped_entities_empty_name += 1
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
                        name=name,
                        entity_type=e.get("entity_type") or e.get("type") or "CONCEPT",
                        description=e.get("description") or "",
                        attributes=attrs,
                        aliases=e.get("aliases") or [],
                        temporal=temporal,
                        confidence=confidence,
                    )
                )

            relationships = []
            skipped_relationships_empty_endpoint = 0
            for r in data.get("relationships", []):
                # Skip malformed relationships (LLM sometimes returns strings instead of dicts)
                if not isinstance(r, dict):
                    logger.debug(f"Skipping malformed relationship (not a dict): {type(r)}")
                    continue

                # Skip relationships with empty / whitespace / missing source or target name (#894).
                raw_src = r.get("source_entity") or r.get("source") or ""
                raw_tgt = r.get("target_entity") or r.get("target") or ""
                src = raw_src.strip() if isinstance(raw_src, str) else ""
                tgt = raw_tgt.strip() if isinstance(raw_tgt, str) else ""
                if not src or not tgt:
                    skipped_relationships_empty_endpoint += 1
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
                        source_entity=src,
                        target_entity=tgt,
                        relationship_type=r.get("relationship_type") or r.get("type") or "RELATES_TO",
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
                        event_type=ev.get("event_type") or ev.get("type") or "EVENT",
                        occurred_at=ev.get("occurred_at"),
                        participants=ev.get("participants") or [],
                        confidence=confidence,
                    )
                )

            # Track empty-name skips (#894). Surface counts and a forward-compatible
            # degradation entry on ExtractionResult.metadata so the call stack can
            # propagate them to RememberResult.metadata["degradations"].
            metadata: dict[str, Any] = {}
            if skipped_entities_empty_name or skipped_relationships_empty_endpoint:
                logger.debug(
                    "llm.entity_parser: skipped {ents} entities and {rels} relationships with empty/missing name",
                    ents=skipped_entities_empty_name,
                    rels=skipped_relationships_empty_endpoint,
                )
                metadata["skipped_entities_empty_name"] = skipped_entities_empty_name
                metadata["skipped_relationships_empty_endpoint"] = skipped_relationships_empty_endpoint
                metadata["degradations"] = [
                    {
                        "component": "llm.entity_parser",
                        "reason": "empty_name",
                        "skipped_entities": skipped_entities_empty_name,
                        "skipped_relationships": skipped_relationships_empty_endpoint,
                    }
                ]

            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                events=events,
                metadata=metadata,
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
                # Pass dict directly — _parse_response accepts dicts and skips
                # json.loads. Re-serializing with json.dumps would re-enter
                # _parse_response as a string, risking mutual recursion if
                # _strip_json_fences mangles the output back into invalid JSON.
                return self._parse_response(data)
            except json.JSONDecodeError:
                pass

        logger.warning("Could not extract valid JSON from response")
        return ExtractionResult(metadata={"raw_response": text[:500]})
