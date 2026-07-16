# Extractors

Extractors identify entities and relationships from text using LLM-based extraction. This document covers the extraction system and configuration.

## Overview

The extraction pipeline:
1. Takes chunk text as input
2. Calls an LLM with a structured extraction prompt
3. Parses JSON output into entities and relationships
4. Returns structured `ExtractionResult`

## LLMEntityExtractor

The primary extractor uses LiteLLM for LLM access with JSON schema output.

### Configuration

```python
from khora.extraction.extractors import LLMEntityExtractor

extractor = LLMEntityExtractor(
    model="gpt-4o-mini",      # LLM model
    temperature=0.0,           # Deterministic extraction (default)
    max_tokens=16384,          # Output token limit (default)
    timeout=60,                # Request timeout (seconds)
    max_retries=3,             # Retry count
    max_concurrent=10,         # Parallel extractions
)
```

### From LiteLLM Config

```python
from khora.config import LiteLLMConfig

config = LiteLLMConfig(
    model="gpt-4o-mini",
    max_tokens=16384,
    timeout=60,
    max_retries=3,
    max_concurrent_llm_calls=10,
)

extractor = LLMEntityExtractor.from_config(config)
```

## Extraction Result

```python
@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
    events: list[ExtractedEvent]
    metadata: dict[str, Any]
```

### ExtractedEntity

```python
@dataclass
class ExtractedEntity:
    name: str                    # "Albert Einstein"
    entity_type: str             # "PERSON"
    description: str             # "Theoretical physicist"
    attributes: dict[str, Any]   # {"nationality": "German"}
    aliases: list[str]           # ["A. Einstein", "Einstein"]
    temporal: TemporalInfo | None  # Validity period
    confidence: float            # 0.0-1.0
```

### ExtractedRelationship

```python
@dataclass
class ExtractedRelationship:
    source_entity: str           # Entity name (must match an entity)
    target_entity: str           # Entity name (must match an entity)
    relationship_type: str       # "WORKS_FOR"
    description: str             # "Employed as professor"
    properties: dict[str, Any]   # {"start_date": "1933"}
    temporal: TemporalInfo | None  # Validity period
    confidence: float            # 0.0-1.0
```

### ExtractedEvent

```python
@dataclass
class ExtractedEvent:
    description: str             # "Awarded Nobel Prize"
    event_type: str              # "AWARD"
    occurred_at: str | None      # "1921-11-09"
    participants: list[str]      # Entity names
    confidence: float            # 0.0-1.0
```

## Default Extraction Prompts

The live prompt is split in two (see `DEFAULT_SYSTEM_PROMPT` and the prompt templates in `src/khora/extraction/extractors/llm.py`):

- **`DEFAULT_SYSTEM_PROMPT`** - a fully static system message carrying only the guidelines: canonical entity names, aliases, temporal extraction, STATE_CHANGE and EVENT detection rules, and the RELATIONSHIP DENSITY instruction ("For N extracted entities, aim to identify N to 2N relationships", including implicit/co-occurrence connections).
- **A user prompt template** carrying the dynamic content: `{entity_types}`, `{relationship_types}`, `{document_context}`, and `{text}`.

Two user-prompt variants exist:

- **`EXTRACTION_PROMPT_STRUCTURED`** - used for models on the structured-output allowlist. It contains no JSON example; the output schema is enforced through the `response_format` (`json_schema`) API parameter instead, saving ~400-500 tokens per call.
- **`EXTRACTION_PROMPT`** - the full prompt with an inline JSON schema example, used for models without structured-output support.

## Prompt Optimization for Prefix Caching

Extraction prompts are structured to maximize prefix caching hits with LLM providers. The static guidelines live in the system message; everything per-call - entity types, relationship types, document context, and the text itself - lives in the user message, and the output schema is passed via the `response_format` API parameter rather than embedded in any prompt:

```text
System: {static guidelines - identical across all calls}

User: Extract entities ... {document_context}
      Entity types to extract: {entity_types}
      Relationship types to use: {relationship_types}
      Text: {text}
```

Keeping the system message free of per-call content is what makes the prefix cacheable even when entity types vary between calls. When processing hundreds of documents, LLM providers (OpenAI, Anthropic) cache the shared prefix, reducing per-call latency and cost. The improvement is most significant with GPT-4o (automatic prefix caching) and Claude models.

## Entity Types

Entity types must be provided by the caller - Khora does not define defaults.
Pass `entity_types` and `relationship_types` explicitly to `remember()` / `remember_batch()`.

Custom types can be specified via expertise configuration.

## Temporal Awareness

The extractor captures temporal information:

```python
@dataclass
class TemporalInfo:
    mentioned_at: str | None     # When mentioned in text
    occurred_at: str | None      # When event/relationship happened
    valid_from: str | None       # Start of validity period
    valid_until: str | None      # End of validity period
```

Example extraction:
```json
{
    "name": "Acme CEO Position",
    "entity_type": "ROLE",
    "temporal": {
        "valid_from": "2020-01-01",
        "valid_until": "2023-12-31"
    }
}
```

## Expertise Integration

Extractors can use `ExpertiseConfig` for domain-specific extraction:

```python
from khora.extraction.skills import load_expertise

expertise = load_expertise("saas_expert")

result = await extractor.extract(
    text,
    expertise=expertise,
)
```

With expertise:
- Entity types from config
- Custom system prompt
- Custom extraction prompt (Jinja2 templates)
- Confidence filtering

See [Expertise System](expertise-system.md) for details.

## Prompt Customization

### Custom System Prompt

```python
expertise = ExpertiseConfig(
    name="custom",
    system_prompt="""You are an expert at extracting information
    about SaaS companies and their products. Focus on:
    - Company names and founders
    - Product features and pricing
    - Customer relationships
    """,
)
```

### Custom Extraction Prompt (Jinja2)

Custom prompts render in a Jinja `ImmutableSandboxedEnvironment`; unsafe constructs (dunder/private attribute access, mutating methods) are rejected and raise `SecurityError`.

```python
expertise = ExpertiseConfig(
    name="custom",
    extraction_prompt="""Extract information about {{ company_name }}.

    Entity types: {{ entity_types | join(', ') }}

    Text:
    {{ text }}

    Focus on products, pricing, and customer mentions.
    Return JSON with entities and relationships.
    """,
)

result = await extractor.extract(
    text,
    expertise=expertise,
    context={"company_name": "Acme Corp"},
)
```

## Confidence Filtering

Results can be filtered by confidence:

```python
expertise = ExpertiseConfig(
    name="strict",
    confidence=ConfidenceConfig(
        min_entity=0.7,        # Only entities with 70%+ confidence
        min_relationship=0.6,  # Only relationships with 60%+ confidence
    ),
)
```

## Concurrency Control

Extraction uses a semaphore for rate limiting. Slot acquisition is wall-clock-bounded via `_acquire_slot()` to prevent a wedged LLM call from parking all other extraction coroutines indefinitely:

```python
class LLMEntityExtractor:
    def __init__(self, ..., max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def extract(self, text: str, ...):
        async with self._acquire_slot():  # wall-clock-bounded acquire
            # LLM call here
            ...
```

Batch size is adaptive based on the model's context window. For unknown models the default multiplier is 3x `max_tokens` for input budget; known large-context models (gpt-4o, claude-3-opus) use up to 8x. `extract_multi` uses this budget to group texts greedily, so the effective batch size varies with input length rather than being fixed at 5.

`extract_multi` dispatches batches in waves: `extraction_wave_size` (config `llm.extraction_wave_size`, env `KHORA_LLM_EXTRACTION_WAVE_SIZE`, default 20) bounds how many extraction batches dispatch concurrently per wave, and the circuit breaker is evaluated between waves. Raising it above `max_concurrent_llm_calls` has no effect - the semaphore still caps in-flight calls.

## Batch Extraction

Extract from multiple texts concurrently:

```python
results = await extractor.extract_batch(
    texts=[chunk.content for chunk in chunks],
    expertise=expertise,
)

for chunk, result in zip(chunks, results):
    # Process each chunk's entities
    for entity in result.entities:
        ...
```

### Multi-Chunk Extraction (`extract_multi`)

For documents with many chunks, `extract_multi` groups chunks into batches (typically 5 per batch) and sends each batch as a single LLM call. All batches run concurrently, bounded by the extractor's semaphore:

```python
# 15 chunks → 3 batches of 5 → 3 concurrent LLM calls
# Instead of 3 sequential calls, all 3 run at the same time
results = await extractor.extract_multi(
    texts=[chunk.content for chunk in chunks],
    expertise=expertise,
)
```

This means extraction time scales with the single slowest batch, not the sum of all batches. For a document with 15 chunks, this is roughly 3-5x faster than sequential processing.

### Two-Pass Relationship Extraction

A second-pass relationship extraction targets sparse graphs where entities were extracted but relationships between them were missed. The trigger is `num_entities >= 2 and num_relationships < num_entities - 1` (it never fires for texts with 0 or 1 entity). The two extraction paths differ in gating:

- **Single-text `extract()`** runs the second pass automatically on the trigger, using `RELATIONSHIP_EXTRACTION_PROMPT`. Not configurable.
- **Batched `extract_multi()`** - the path `remember()` / `remember_batch()` ingest uses - runs a batched, relationships-only second pass that is **OFF by default** (#1409/#1420). Enable it via `pipeline.extraction_second_pass` (env `KHORA_PIPELINES_EXTRACTION_SECOND_PASS=true`). When on, under-connected sections get one extra batched relationship-only LLM call, recovering ~30-40% more connections at extra cost; the default keeps the ingest cost profile flat.

The second pass uses a relationships-only `response_format` schema, so the model is not forced to emit empty `entities`/`events` arrays. A failed second pass never fails the extraction: first-pass relationships are kept and an ADR-001 `Degradation` (`component="extraction.llm.second_pass"`, `reason="second_pass_failed"`) is recorded in `ExtractionResult.metadata["degradations"]` (#1412).

## JSON Parsing

The extractor handles various JSON output formats:

```python
def _parse_response(self, content: str) -> ExtractionResult:
    try:
        data = json.loads(content)
        # Parse entities, relationships, events
        ...
    except json.JSONDecodeError:
        # Try to extract JSON from mixed content
        return self._extract_json_from_text(content)
```

## Error Handling

Extraction uses retry with exponential backoff, driven by `tenacity`'s `AsyncRetrying`. Retries stop after `max_retries` attempts or 180 seconds total (`stop_after_attempt(self._max_retries) | stop_after_delay(180)`), with `wait_exponential(multiplier=retry_wait, min=retry_wait, max=10)` backoff (default `retry_wait=1.0`):

```python
from tenacity import AsyncRetrying, stop_after_attempt, stop_after_delay, wait_exponential

async for attempt in AsyncRetrying(
    stop=stop_after_attempt(self._max_retries) | stop_after_delay(180),
    wait=wait_exponential(multiplier=self._retry_wait, min=self._retry_wait, max=10),
):
    with attempt:
        async with self._acquire_slot():
            response = await litellm.acompletion(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            result = self._parse_response(response.choices[0].message.content)
```

### Thinking-Model Budget Floor

Thinking models (Gemini 2.5, o1, o3) spend a large portion of their output budget on hidden reasoning tokens before emitting JSON. With the default budget, that leaves too little room for extraction output. For these models the first-attempt `max_tokens` is floored at 32768 regardless of the configured value.

### Truncation Auto-Retry

When the response `finish_reason` is `length` or `MAX_TOKENS`, the extractor automatically retries once with double the `max_tokens` budget. If the response is still truncated after the doubled-budget retry, the call is not silently treated as an empty-extraction success - it returns an `ExtractionResult` with a structured ADR-001 `Degradation` attached:

```python
degradation: Degradation = {
    "component": "extraction.llm",
    "reason": "truncated_response",
    "detail": f"finish_reason={finish_reason}, model={model_used}",
}
```

This is logged at `ERROR` level and included in `ExtractionResult.metadata["degradations"]`.

### Non-Retryable Auth Failures

`litellm.AuthenticationError` and `litellm.PermissionDeniedError` are never retried -- they represent deterministic failures that will not resolve with backoff. Missing-credentials errors surfaced as `InternalServerError` (e.g. "Missing credentials. Please pass an api_key...") are also detected and fast-failed. `asyncio.CancelledError` and `TimeoutError` are similarly non-retryable.

## API Usage

### Via Khora

```python
result = await kb.remember(
    content,
    namespace=ns.namespace_id,
    expertise=expertise,            # preferred: ExpertiseConfig or name string
    # skill_name="general_entities" is legacy and ignored when expertise= is provided
    entity_types=["PERSON", "ORG"],
    relationship_types=["WORKS_AT"],
)
```

`extraction_model` isn't a per-call kwarg on `kb.remember()`. Set the
extraction model globally via `KhoraConfig.llm.model` (or env var
`KHORA_LLM_MODEL`) at construction time.

### In Pipeline Tasks

```python
from khora.pipelines.tasks import extract_entities

entities, relationships = await extract_entities(
    chunks,
    expertise=expertise,            # preferred; skill_name= is legacy and ignored when expertise= is set
    model="gpt-4o-mini",
    max_concurrent=10,
)
```

### Direct Extractor Usage

```python
from khora.extraction.extractors import LLMEntityExtractor

extractor = LLMEntityExtractor(model="gpt-4o-mini")

result = await extractor.extract(
    text="Einstein worked at Princeton University starting in 1933.",
    entity_types=["PERSON", "ORGANIZATION"],
)

for entity in result.entities:
    print(f"{entity.name} ({entity.entity_type})")
```

## Next Steps

- [Expertise System](expertise-system.md) - Domain configuration
- [Semantic Expansion](semantic-expansion.md) - Entity unification
- [Knowledge Graph](../data-models/knowledge-graph.md) - Entity storage
