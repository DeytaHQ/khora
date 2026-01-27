# Ingestion Pipeline

Khora uses a two-phase (optionally three-phase) ingestion pipeline orchestrated by Prefect. This document describes the pipeline architecture and configuration.

## Pipeline Overview

```
Documents
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│                      Phase 1: Staging                              │
│                                                                    │
│  For each document (parallel with semaphore):                     │
│    1. Compute SHA-256 checksum                                    │
│    2. Check for existing document with same checksum              │
│    3. Extract source timestamp from metadata                       │
│    4. Create Document with PENDING status                          │
│                                                                    │
│  Output: List of new documents (skipping duplicates)               │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│                     Phase 2: Enrichment                            │
│                                                                    │
│  For each staged document (parallel with semaphore):              │
│    1. Mark document as PROCESSING                                  │
│    2. Chunk document into segments                                 │
│    3. Generate embeddings for all chunks (batched)                 │
│    4. Extract entities and relationships (parallel LLM calls)      │
│    5. Store chunks in pgvector                                     │
│    6. Store entities in Neo4j (with deduplication)                 │
│    7. Store relationships in Neo4j                                 │
│    8. Mark document as COMPLETED                                   │
│                                                                    │
│  Output: Processing statistics                                     │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼ (optional)
┌───────────────────────────────────────────────────────────────────┐
│                   Phase 3: Expansion                               │
│                                                                    │
│  If enable_expansion=True:                                        │
│    1. Cross-tool entity unification                               │
│       - Exact name matching                                        │
│       - Fuzzy string matching (Levenshtein)                        │
│       - Embedding similarity matching                              │
│    2. Relationship inference                                       │
│       - Apply inference rules from expertise config                │
│       - Create inferred relationships                              │
│                                                                    │
│  Output: Unified entities, inferred relationships                  │
└───────────────────────────────────────────────────────────────────┘
```

## Phase 1: Staging

### Checksum Deduplication

Documents are deduplicated by content checksum:

```python
from hashlib import sha256

def compute_checksum(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()

# Check for existing
existing = await storage.get_document_by_checksum(namespace_id, checksum)
if existing:
    return None  # Skip duplicate
```

### Source Timestamp Extraction

Timestamps are extracted from source metadata for accurate temporal ordering:

```python
# Priority order for timestamp fields
timestamp_fields = [
    "sent_at",      # Email sent time
    "created_at",   # Creation time
    "timestamp",    # Generic timestamp
    "date",         # Date field
    "occurred_at",  # Event time
    "started_at",   # Start time
]

# Documents use source timestamp, not ingestion time
document = Document(
    content=content,
    created_at=source_timestamp or datetime.now(UTC),
)
```

### Document Creation

```python
# Create document with metadata
document = Document(
    namespace_id=namespace_id,
    content=content,
    metadata=DocumentMetadata(
        source=doc_input.get("source", ""),
        source_type=doc_input.get("source_type", "manual"),
        title=doc_input.get("title", ""),
        checksum=checksum,
        custom=doc_input.get("metadata", {}),
    ),
    status=DocumentStatus.PENDING,
)

await storage.create_document(document)
```

## Phase 2: Enrichment

### Step 1: Chunking

```python
from khora.pipelines.tasks import chunk_document

chunks = await chunk_document(
    document,
    strategy="semantic",  # or "fixed", "recursive"
    chunk_size=512,
)
```

See [Chunkers](chunkers.md) for chunking strategies.

### Step 2: Embedding

```python
from khora.pipelines.tasks import embed_chunks

chunks = await embed_chunks(
    chunks,
    model="text-embedding-3-small",
)
```

Embedding is batched internally for efficiency.

### Step 3: Extraction

```python
from khora.pipelines.tasks import extract_entities

entities, relationships = await extract_entities(
    chunks,
    skill_name="general_entities",
    expertise=expertise_config,
    model="gpt-4o-mini",
    max_concurrent=10,
)
```

Extraction runs in parallel across chunks with semaphore control.

### Step 4: Storage

```python
# Store chunks (batched)
await storage.create_chunks_batch(chunks)

# Store entities with deduplication
for entity in entities:
    existing = await storage.get_entity_by_name(
        namespace_id, entity.name, entity.entity_type
    )
    if existing:
        existing.merge_with(entity)
        await storage.update_entity(existing)
    else:
        await storage.create_entity(entity)

# Store relationships
for relationship in relationships:
    await storage.create_relationship(relationship)
```

### Entity ID Remapping

When entities are deduplicated, relationship IDs are remapped:

```python
# Track original → deduplicated ID mapping
entity_id_mapping = {}

for entity in entities:
    existing = await storage.get_entity_by_name(...)
    if existing:
        entity_id_mapping[str(entity.id)] = str(existing.id)
    else:
        entity_id_mapping[str(entity.id)] = str(entity.id)

# Remap relationship source/target IDs
for relationship in relationships:
    relationship.source_entity_id = UUID(
        entity_id_mapping[str(relationship.source_entity_id)]
    )
    relationship.target_entity_id = UUID(
        entity_id_mapping[str(relationship.target_entity_id)]
    )
```

## Phase 3: Expansion (Optional)

When `enable_expansion=True`, additional processing runs:

### Cross-Tool Unification

```python
from khora.extraction.expansion import SemanticExpander

expander = SemanticExpander(
    expertise=expertise,
    enable_inference=True,
)

result = await expander.expand(
    entities=entities,
    relationships=relationships,
    namespace_id=namespace_id,
)
```

### Inference Modes

Three modes control when inference runs:

| Mode | Description |
|------|-------------|
| `none` | No inference (unification only) |
| `incremental` | Infer per-document (with existing graph context) |
| `batch` | Infer after all documents processed |

```python
# Incremental mode: fetch existing graph for context
if inference_mode == "incremental":
    existing_entities = await storage.list_entities(namespace_id)
    existing_relationships = await storage.list_relationships(namespace_id)
    expansion_entities.extend(existing_entities)
    expansion_relationships.extend(existing_relationships)
```

### Batch Inference

For batch mode, run inference separately after ingestion:

```python
from khora.pipelines.flows.ingest import run_batch_inference

result = await run_batch_inference(
    namespace_id=namespace_id,
    storage=storage,
    expertise=expertise,
    max_entities=10000,
    max_relationships=50000,
)
```

## Concurrency Control

### Document Concurrency

```python
# Maximum documents processed in parallel
max_concurrent_documents = 5

doc_semaphore = asyncio.Semaphore(max_concurrent_documents)

async def process_with_limit(doc):
    async with doc_semaphore:
        return await process_document(doc, ...)

# Process all documents
results = await asyncio.gather(
    *[process_with_limit(doc) for doc in staged_docs],
    return_exceptions=True,
)
```

### Extraction Concurrency

```python
# Maximum concurrent LLM calls
max_concurrent_extractions = 10

# Semaphore passed to extractor
extractor = LLMEntityExtractor(max_concurrent=max_concurrent_extractions)
```

### Staging Concurrency

```python
# Staging runs faster, so higher concurrency
staging_semaphore = asyncio.Semaphore(max_concurrent_documents * 2)
```

## Error Handling

### Per-Document Errors

Errors are captured per-document, allowing other documents to continue:

```python
results = await asyncio.gather(
    *[process_with_limit(doc) for doc in staged_docs],
    return_exceptions=True,  # Don't fail on individual errors
)

# Separate successful results from errors
successful_results = []
error_count = 0

for result in results:
    if isinstance(result, Exception):
        logger.error(f"Document processing failed: {result}")
        error_count += 1
    else:
        successful_results.append(result)
```

### Document Status on Failure

```python
try:
    # Process document
    ...
except Exception as e:
    document.mark_failed(str(e))
    await storage.update_document(document)
    raise
```

## API Usage

### Basic Ingestion

```python
from khora import MemoryLake

async with MemoryLake() as lake:
    result = await lake.remember(
        "Content to ingest...",
        title="Document Title",
        source="manual",
    )
```

### Batch Ingestion

```python
documents = [
    {"content": "Doc 1 content", "title": "Doc 1"},
    {"content": "Doc 2 content", "title": "Doc 2"},
]

result = await lake.remember_batch(
    documents,
    max_concurrent_documents=5,
    enable_expansion=True,
)
```

### Direct Pipeline Call

```python
from khora.pipelines.flows import ingest_documents

result = await ingest_documents(
    namespace_id=namespace_id,
    documents=documents,
    storage=storage,
    expertise="saas_expert",
    chunk_strategy="semantic",
    chunk_size=512,
    embedding_model="text-embedding-3-small",
    extraction_model="gpt-4o-mini",
    max_concurrent_documents=5,
    max_concurrent_extractions=10,
    enable_expansion=True,
)
```

## Result Structure

```python
{
    "total_documents": 100,        # Input documents
    "processed_documents": 95,     # Successfully processed
    "skipped_documents": 3,        # Duplicates (by checksum)
    "failed_documents": 2,         # Processing errors
    "total_chunks": 450,           # Chunks created
    "total_entities": 200,         # Entities extracted
    "total_relationships": 150,    # Relationships extracted
    "total_inferred_relationships": 25,  # From expansion
}
```

## Prefect Integration

The pipeline uses Prefect for orchestration:

```python
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

@task(name="stage_document", cache_policy=NO_CACHE)
async def stage_document(doc_input, namespace_id, storage):
    ...

@task(name="process_document", cache_policy=NO_CACHE)
async def process_document(document, storage, **kwargs):
    ...

@flow(name="ingest_documents", log_prints=True)
async def ingest_documents(namespace_id, documents, storage, **kwargs):
    # Phase 1: Stage
    staged_docs = await asyncio.gather(...)

    # Phase 2: Enrich
    results = await asyncio.gather(...)

    return summary
```

## Next Steps

- [Chunkers](chunkers.md) - Text splitting strategies
- [Embedders](embedders.md) - Embedding generation
- [Extractors](extractors.md) - Entity extraction
- [Semantic Expansion](semantic-expansion.md) - Unification and inference
