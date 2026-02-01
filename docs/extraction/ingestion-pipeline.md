# Ingestion Pipeline

When you call `remember()`, a lot happens behind the scenes. Your content gets deduplicated, split into chunks, converted to vectors, analyzed for entities and relationships, and stored across three different databases. This document explains that pipeline in detail.

## The Three Phases

```
Your Documents
      |
      v
+--------------------------------------------------+
|  PHASE 1: STAGING                                |
|                                                  |
|  "Have we seen this before?"                     |
|                                                  |
|  - Compute checksum                              |
|  - Skip duplicates                               |
|  - Extract source timestamps                     |
|  - Create document records                       |
+--------------------------------------------------+
      |
      v
+--------------------------------------------------+
|  PHASE 2: ENRICHMENT                             |
|                                                  |
|  "What's in this content?"                       |
|                                                  |
|  - Chunk into segments                           |
|  - Generate embeddings                           |
|  - Extract entities & relationships              |
|  - Store everything                              |
+--------------------------------------------------+
      |
      v  (optional)
+--------------------------------------------------+
|  PHASE 3: EXPANSION                              |
|                                                  |
|  "How does this connect to what we know?"        |
|                                                  |
|  - Merge duplicate entities                      |
|  - Infer new relationships                       |
+--------------------------------------------------+
```

## Phase 1: Staging

Before doing any expensive processing, we answer a simple question: is this content actually new?

### Deduplication by Checksum

Every document gets a SHA-256 hash of its content:

```python
from hashlib import sha256

checksum = sha256(content.encode("utf-8")).hexdigest()

# Check if we've seen this exact content before
existing = await storage.get_document_by_checksum(namespace_id, checksum)
if existing:
    # Skip it - we already have this
    return None
```

This catches duplicates even if the filename or metadata changed. Same content = same checksum = skip.

### Source Timestamps

When did this content actually originate? Not when it was ingested - when was it created at the source?

```python
# We look for timestamps in the metadata, in priority order
timestamp_fields = [
    "sent_at",      # Emails
    "created_at",   # General creation time
    "timestamp",    # Generic
    "date",
    "occurred_at",  # Events
]

for field in timestamp_fields:
    if field in metadata:
        source_timestamp = parse_datetime(metadata[field])
        break
```

This matters for temporal queries - "what was discussed last week?" should find content from last week, not content that was ingested last week about events from six months ago.

### Document Creation

Finally, we create the document record:

```python
document = Document(
    namespace_id=namespace_id,
    content=content,
    metadata=DocumentMetadata(
        title=title,
        source=source,
        checksum=checksum,
    ),
    status=DocumentStatus.PENDING,
    created_at=source_timestamp or now()
)

await storage.create_document(document)
```

Staging runs in parallel with controlled concurrency - typically `2 * max_concurrent_documents` since it's lightweight.

## Phase 2: Enrichment

This is where content becomes knowledge. Each staged document goes through four steps.

### Step 1: Chunking

Documents get split into segments optimized for embedding and retrieval:

```python
chunks = await chunk_document(
    document,
    strategy="semantic",  # or "fixed", "recursive"
    chunk_size=512,       # tokens
    chunk_overlap=50      # overlap between chunks
)
```

Why chunk? Embedding models have token limits, and retrieval works better with focused content. A 10,000-word document as a single embedding is too diluted - split it into coherent 500-token pieces.

See [Chunkers](chunkers.md) for the different strategies.

### Step 2: Embedding

Each chunk gets converted to a vector:

```python
chunks = await embed_chunks(
    chunks,
    model="text-embedding-3-small"  # 1536 dimensions
)
```

These vectors capture semantic meaning. Similar concepts get similar vectors, enabling "find content like this" queries.

Embedding is batched internally - instead of 100 API calls for 100 chunks, we make ~3 calls with batches of 32.

### Step 3: Extraction

An LLM reads each chunk and extracts structured knowledge:

```python
entities, relationships = await extract_entities(
    chunks,
    model="gpt-4o-mini",
    skill="general_entities",  # or domain-specific
    max_concurrent=10          # parallel LLM calls
)
```

**Entities** - Named things worth remembering:
```python
Entity(
    name="Einstein",
    entity_type="PERSON",
    description="Theoretical physicist"
)
```

**Relationships** - Connections between entities:
```python
Relationship(
    source="Einstein",
    target="Theory of Relativity",
    type="DEVELOPED"
)
```

Extraction runs in parallel across chunks, with semaphores preventing API rate limits.

### Step 4: Storage

Now we save everything to its appropriate home:

**Chunks → pgvector:**
```python
await storage.create_chunks_batch(chunks)
```

**Entities → Neo4j + pgvector:**
```python
# Per-entity approach (incremental/batch modes):
for entity in entities:
    existing = await storage.get_entity_by_name(
        namespace_id, entity.name, entity.entity_type
    )
    if existing:
        existing.merge_with(entity)
        await storage.update_entity(existing)
    else:
        await storage.create_entity(entity)

# Batch approach (smart mode post-resolution):
await storage.upsert_entities_batch(namespace_id, resolved_entities, batch_size=50)
# Uses UNWIND + MERGE in Neo4j, INSERT ... ON CONFLICT in PostgreSQL
```

**Entity Embeddings → pgvector:**
```python
# Generate embeddings for entity similarity search
entity_texts = [f"{e.name}: {e.description}" for e in entities]
embeddings = await embedder.embed_batch(entity_texts)

for entity, embedding in zip(entities, embeddings):
    await storage.update_entity_embedding(entity.id, embedding, model)
```

**Relationships → Neo4j:**
```python
# Per-relationship approach:
for relationship in relationships:
    await storage.create_relationship(relationship)

# Batch approach (smart mode):
await storage.create_relationships_batch(relationships, batch_size=50)
```

### Entity ID Remapping

When entities get deduplicated (merged with existing), we need to update relationship references:

```python
# Track: original extraction ID → actual stored ID
id_mapping = {}

for entity in extracted_entities:
    existing = await storage.get_entity_by_name(...)
    if existing:
        id_mapping[entity.id] = existing.id  # Maps to existing
    else:
        id_mapping[entity.id] = entity.id    # Maps to self

# Update relationships to use the actual IDs
for rel in relationships:
    rel.source_entity_id = id_mapping[rel.source_entity_id]
    rel.target_entity_id = id_mapping[rel.target_entity_id]
```

Without this, relationships would point to entity IDs that don't exist (the extracted IDs that got merged away).

### Document Status

Throughout enrichment, the document's status tracks progress:

```
PENDING → PROCESSING → COMPLETED
              |
              +----→ FAILED (on error)
```

If processing fails, the error message is stored in the document's metadata, and other documents continue processing.

## Phase 3: Expansion (Optional)

After basic enrichment, expansion can enhance your knowledge graph.

### Entity Unification

The same entity might appear differently across documents:

```
Document 1: "Microsoft Corporation"
Document 2: "Microsoft"
Document 3: "MSFT"
```

The unifier recognizes these as the same entity using three matching strategies:

```python
expander = SemanticExpander(expertise=expertise)

# Three matching strategies:
# 1. Exact: "Microsoft" == "Microsoft"
# 2. Fuzzy: "Microsft" ~= "Microsoft" (edit distance)
# 3. Embedding: "the Redmond tech giant" ≈ "Microsoft" (semantic)

result = await expander.expand(
    entities=entities,
    relationships=relationships,
    namespace_id=namespace_id,
    entity_index=entity_index,  # Optional: enables efficient blocked matching
)
```

### Relationship Inference

Some relationships can be inferred from what we know:

```
Known:     Alice WORKS_FOR Acme
           Bob WORKS_FOR Acme
                 |
                 v
Inferred:  Alice COLLEAGUE_OF Bob
```

Inference rules are defined in the expertise configuration.

### Inference Modes

Four modes control when inference runs:

| Mode | When It Runs | Complexity | Use Case |
|------|-------------|-----------|----------|
| `smart` (default) | Per-doc dedup during ingestion, full resolution once after all docs | O(n * k) | Large imports, production use |
| `incremental` | Per document, with existing graph context | O(n^2) per doc | Small graphs, trickle feeds |
| `batch` | After all documents, on full graph | O(n^2) once | Legacy bulk imports |
| `none` | Never | O(1) | Unification only |

**Smart mode** (the default) uses a shared in-memory `EntityIndex` that grows as documents are processed. During ingestion, each entity gets an O(1) exact-match check against the index. After all documents are processed, a single cross-document resolution pass runs with token-blocked fuzzy and embedding matching -- O(n * k) instead of O(n^2).

```python
# Smart mode is the default:
result = await ingest_documents(
    namespace_id, documents, storage,
    expertise="saas_expert",
    enable_expansion=True,
    # inference_mode="smart" is the default
)
```

**Incremental mode** fetches the existing graph for context on each document:

```python
if inference_mode == "incremental":
    existing = await storage.list_entities(namespace_id)
    expansion_context.extend(existing)
```

This works well for small graphs but becomes slow as entity counts grow, since every document triggers O(n^2) pairwise comparisons.

See [Semantic Expansion](semantic-expansion.md) for full details on each mode and the `EntityIndex`.

## Concurrency Control

The pipeline uses semaphores to prevent overwhelming your system or hitting API limits:

```python
# Document-level: 5 docs processing at once
doc_semaphore = asyncio.Semaphore(5)

# Extraction-level: 10 LLM calls at once
extraction_semaphore = asyncio.Semaphore(10)

# Staging-level: 10 docs staging at once (it's fast)
staging_semaphore = asyncio.Semaphore(10)
```

These are configurable:

```python
result = await lake.remember_batch(
    documents,
    max_concurrent_documents=5,
    max_concurrent_extractions=10
)
```

## Error Handling

Errors don't stop the batch - they're captured per-document:

```python
results = await asyncio.gather(
    *[process(doc) for doc in docs],
    return_exceptions=True  # Don't fail the whole batch
)

for doc, result in zip(docs, results):
    if isinstance(result, Exception):
        doc.mark_failed(str(result))
        await storage.update_document(doc)
        # Continue with other documents
```

Check for failures in the result:

```python
result = await lake.remember_batch(documents)

print(f"Processed: {result['processed_documents']}")
print(f"Skipped (duplicates): {result['skipped_documents']}")
print(f"Failed: {result['failed_documents']}")
```

## Usage Examples

### Simple Ingestion

```python
result = await lake.remember(
    "Your content here...",
    title="Document Title",
    source="manual"
)

print(f"Document: {result.document_id}")
print(f"Chunks: {result.chunks_created}")
print(f"Entities: {result.entities_extracted}")
```

### Batch Ingestion

```python
documents = [
    {"content": "Doc 1...", "title": "Doc 1", "source": "upload"},
    {"content": "Doc 2...", "title": "Doc 2", "source": "upload"},
]

result = await lake.remember_batch(
    documents,
    max_concurrent_documents=5,
    enable_expansion=True
)
```

### With Custom Configuration

```python
result = await lake.remember(
    content,
    chunk_strategy="recursive",
    chunk_size=1024,
    embedding_model="text-embedding-3-large",
    extraction_model="claude-sonnet-4-20250514",
    expertise="technical_docs"
)
```

### Direct Pipeline Call

For more control:

```python
from khora.pipelines.flows import ingest_documents

result = await ingest_documents(
    namespace_id=namespace_id,
    documents=documents,
    storage=storage,
    chunk_strategy="semantic",
    embedding_model="text-embedding-3-small",
    extraction_model="gpt-4o-mini",
    expertise="saas_expert",
    max_concurrent_documents=5,
    max_concurrent_extractions=10,
    enable_expansion=True,
    inference_mode="smart"   # default; also "incremental", "batch", "none"
)
```

## Backfilling Entity Embeddings

If you have entities created before entity embedding was implemented:

```python
from khora.pipelines.flows import backfill_entity_embeddings

result = await backfill_entity_embeddings(
    namespace_id=namespace_id,
    storage=storage,
    embedding_model="text-embedding-3-small",
    batch_size=100
)

print(f"Updated {result['entities_updated']} entities")
```

This is a one-time migration for existing data.

## Result Structure

```python
{
    "total_documents": 100,
    "processed_documents": 95,
    "skipped_documents": 3,        # Duplicates
    "failed_documents": 2,
    "total_chunks": 450,
    "total_entities": 200,
    "total_relationships": 150,
    "total_inferred_relationships": 25
}
```

## What's Next?

- **[Chunkers](chunkers.md)** - Text splitting strategies
- **[Embedders](embedders.md)** - Vector generation
- **[Extractors](extractors.md)** - Entity extraction
- **[Semantic Expansion](semantic-expansion.md)** - Unification and inference
- **[Expertise System](expertise-system.md)** - Domain configuration
