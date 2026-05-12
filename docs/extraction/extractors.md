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
    temperature=0.3,           # Low for consistent extraction
    max_tokens=4000,           # Output token limit
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
    max_tokens=4000,
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

## Default Extraction Prompt

The extractor uses a structured prompt for consistent JSON output:

```python
EXTRACTION_PROMPT = """Extract entities, relationships, and temporal information from the following text.

Entity types to extract: {entity_types}

Text:
{text}

Return a JSON object with the following structure:
{
    "entities": [
        {
            "name": "entity name (canonical form, properly capitalized)",
            "entity_type": "PERSON|ORGANIZATION|...",
            "description": "brief description",
            "attributes": {"key": "value"},
            "aliases": ["alternative names"],
            "temporal": {
                "mentioned_at": "when entity is mentioned",
                "valid_from": "ISO date or null",
                "valid_until": "ISO date or null"
            }
        }
    ],
    "relationships": [
        {
            "source_entity": "source entity name",
            "target_entity": "target entity name",
            "relationship_type": "WORKS_FOR|KNOWS|...",
            "description": "brief description",
            "temporal": {
                "occurred_at": "when relationship occurred",
                "valid_from": "ISO date or null",
                "valid_until": "ISO date or null"
            }
        }
    ],
    "events": [
        {
            "description": "what happened",
            "occurred_at": "when (ISO date or descriptive)",
            "participants": ["entity names"],
            "event_type": "MEETING|DECISION|..."
        }
    ]
}

Guidelines:
- Use canonical entity names (e.g., "Jennifer Walsh" not "Jenny")
- Include aliases for entities with multiple names
- Extract temporal information when dates appear
- Ensure relationship source/target names match entity names exactly

Return ONLY valid JSON, no other text."""
```

## Prompt Optimization for Prefix Caching

Extraction prompts are structured to maximize prefix caching hits with LLM providers. Static instruction content (entity types, guidelines, output schema) is placed in the system message, and variable content (the actual text to extract from) is placed in the user message:

```
System: You are an entity extractor.
        Entity types: {entity_types}
        Guidelines: {static_instructions}
        Output schema: {json_schema}

User: Extract from this text: {variable_text}
```

When processing hundreds of documents with the same extraction configuration, the system message is identical across calls. LLM providers (OpenAI, Anthropic) cache this prefix, reducing per-call latency and cost. The improvement is most significant with GPT-4o (automatic prefix caching) and Claude models.

## Entity Types

Entity types must be provided by the caller — Khora does not define defaults.
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

Extraction uses a semaphore for rate limiting:

```python
class LLMEntityExtractor:
    def __init__(self, ..., max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def extract(self, text: str, ...):
        async with self._semaphore:
            # LLM call here
            ...
```

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

The extractor automatically runs a second-pass relationship extraction when `num_relationships < num_entities - 1`, using `RELATIONSHIP_EXTRACTION_PROMPT`. This targets sparse graphs where entities were extracted but relationships between them were missed. The second pass is automatic and not configurable — it triggers whenever the entity-to-relationship ratio suggests missing connections.

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

Extraction uses retry with exponential backoff:

```python
for attempt in range(self._max_retries):
    try:
        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return self._parse_response(response.choices[0].message.content)
    except Exception as e:
        if attempt < self._max_retries - 1:
            await asyncio.sleep(2 ** attempt)
        else:
            return ExtractionResult(metadata={"error": str(e)})
```

## API Usage

### Via Khora

```python
result = await kb.remember(
    content,
    extraction_model="gpt-4o-mini",
    skill_name="general_entities",
)
```

### In Pipeline Tasks

```python
from khora.pipelines.tasks import extract_entities

entities, relationships = await extract_entities(
    chunks,
    skill_name="general_entities",
    expertise=expertise,
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
