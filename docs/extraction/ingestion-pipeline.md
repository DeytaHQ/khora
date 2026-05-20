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
    title=title,
    source=source,
    checksum=checksum,
    status=DocumentStatus.PENDING,
    created_at=source_timestamp or now(),
)

await storage.create_document(document)
```

Staging runs in parallel with controlled concurrency - typically `2 * max_concurrent_documents` since it's lightweight.

## Phase 2: Enrichment (Staged Batch Pipeline)

This is where content becomes knowledge. The pipeline uses a **staged batch architecture** — instead of processing each document independently through all steps, all documents flow through each stage together:

```
Stage 1: chunk(Doc1, Doc2, ..., DocN)              ← all docs chunked first
Stage 2: embed(all chunks) ∥ extract(all chunks)   ← parallel across all
Stage 3: store(all results)                        ← batch writes
```

Embedding and extraction run **concurrently** via `asyncio.gather` since they both depend only on chunks — extraction doesn't need embeddings, and embedding doesn't need entities. The staged approach provides better LLM API utilization (larger concurrent batches), more efficient batch database writes, and enables cross-document entity deduplication on the full set.

> **Note:** Chunking now runs in `asyncio.to_thread()` when using spaCy, avoiding blocking the event loop during sentence splitting.

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

Each returned `ChunkResult` stamps `metadata["chunker"]` with its registered strategy name (`"fixed"`, `"recursive"`, `"semantic"`, `"conversation"`) so downstream code — and the persisted chunk row — knows how a chunk was produced without re-running the chunker.

See [Chunkers](chunkers.md) for the different strategies.

### Steps 2 & 3: Embedding + Extraction (Parallel)

After chunking, embedding and extraction run at the same time via `asyncio.gather`. The slower operation (usually extraction) determines wall-clock time, while the faster one (usually embedding) completes "for free" in the background.

**Embedding** converts each chunk to a vector:

```python
chunks = await embed_chunks(
    chunks,
    model="text-embedding-3-small"  # 1536 dimensions
)
```

These vectors capture semantic meaning. Similar concepts get similar vectors, enabling "find content like this" queries.

Embedding is batched internally - instead of 100 API calls for 100 chunks, we make a few concurrent calls with batches of up to 200. When there are more texts than the batch size, sub-batches run concurrently (up to `embed_concurrency`, default 20) rather than sequentially.

**Extraction** has an LLM read each chunk and extract structured knowledge:

```python
entities, relationships = await extract_entities(
    chunks,
    model="gpt-4o-mini",
    skill="general_entities",  # or domain-specific
    max_concurrent=20          # parallel LLM calls
)
```

When using `extract_multi` (the default for multi-chunk documents), chunks are grouped into batches of ~5 and each batch is sent as a single LLM call. All batches run concurrently (bounded by the extractor's semaphore), so 15 chunks across 3 batches complete in roughly the time of 1 batch instead of 3 sequential calls.

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

# Update all embeddings in a single transaction
updates = [(e.id, emb, model) for e, emb in zip(entities, embeddings)]
await storage.update_entity_embeddings_batch(updates, namespace_id=namespace_id)
```

**Relationships → Neo4j:**
```python
# Batch approach (used by default):
await storage.create_relationships_batch(relationships, batch_size=50)
# Uses UNWIND + CREATE in Neo4j — one transaction instead of N individual writes
```

> Since v0.16.0 (#769) every storage read/write requires `namespace_id` as a kwarg-only argument (or as the leading positional on the small set of methods like `get_document_by_checksum` / `list_entities` / `list_relationships` where it has always been positional). The `StorageCoordinator.{relational,vector,graph,event_store}` attrs are wrapped in `NamespaceRequiredProxy` and emit a `DeprecationWarning` once per role per process; they refuse read calls without `namespace_id=`. Internal canonical refs are `self._{relational,vector,graph,event_store}`. The public attrs disappear in v0.17 — engines that talk to them go through the namespace-scoped coordinator facade instead.

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

### Selective Extraction with Importance Scoring

Not all chunks are worth sending to an LLM for entity extraction. The `ChunkImportanceScorer` (in `src/khora/extraction/importance.py`) scores each chunk on four signals:

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| Entity density | 35% | Capitalized phrases, proper nouns (0-1) |
| Information density | 25% | Type-token ratio (unique words / total words) |
| Position | 20% | First/last chunks score highest |
| Length | 20% | 50-300 words is the sweet spot |

When `selective_extraction=True` (the default), only the top chunks by importance score get full LLM extraction. Remaining chunks get lightweight co-occurrence edges (`CO_OCCURS_WITH` relationships) extracted via regex, at confidence 0.4. This reduces LLM extraction costs by 30-50% with minimal recall loss.

```python
result = await ingest_documents(
    namespace_id, documents, storage,
    selective_extraction=True,        # default
    extraction_importance_ratio=0.7,  # top 70% get LLM extraction
    extraction_min_importance=0.2,    # minimum score threshold
)
```

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
# Document-level: 10 docs processing at once
doc_semaphore = asyncio.Semaphore(10)

# Extraction-level: 20 LLM calls at once
extraction_semaphore = asyncio.Semaphore(20)

# Staging-level: 20 docs staging at once (it's fast)
staging_semaphore = asyncio.Semaphore(20)
```

**Neo4j write-level coordination** prevents lock contention during storage:

| Write type | Mechanism | Concurrency | Why |
|-----------|-----------|-------------|-----|
| Entity writes | `_EntityKeyGate` (key-aware) | 12 concurrent, serializes overlapping keys | MERGE transactions on the same `(namespace_id, name, entity_type)` would cause Neo4j lock contention and ~1 s retry backoff |
| Relationship writes | `asyncio.Semaphore` | 8 concurrent | CREATE transactions don't contend — each is a new edge |

These are configurable:

```python
result = await kb.remember_batch(
    documents,
    max_concurrent_documents=10,
    max_concurrent_extractions=20
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
result = await kb.remember_batch(documents)

print(f"Processed: {result['processed_documents']}")
print(f"Skipped (duplicates): {result['skipped_documents']}")
print(f"Failed: {result['failed_documents']}")
```

## Usage Examples

### Simple Ingestion

```python
result = await kb.remember(
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

result = await kb.remember_batch(
    documents,
    max_concurrent=10,
)
```

`remember_batch` delegates to `ingest_documents` internally, which means batch calls get all the benefits of the full pipeline: shared `EntityIndex` for cross-document entity deduplication, smart mode resolution, and parallel document processing. This is a significant improvement over calling `remember()` in a loop, which would miss cross-document dedup entirely.

### With Custom Configuration

```python
result = await kb.remember(
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
    max_concurrent_documents=10,
    max_concurrent_extractions=20,
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
    batch_size=200
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
    "total_inferred_relationships": 25,
    "per_document_results": [      # One entry per successfully processed document
        {"document_id": "uuid", "chunks": 5, "entities": 3, "relationships": 2, ...},
        ...
    ]
}
```

## Canonical metadata fields per source

Connector authors should populate `metadata.custom` with the canonical
fields below. The ingestion pipeline reads them in
`_extract_source_timestamp` to build `TemporalChunk.occurred_at`, which
drives temporal recency scoring + per-source decay. The shape is exported
as `khora.pipelines.ConnectorMetadata` (a `TypedDict`).

| Source | Upstream field | Map to `metadata.custom` key | Notes |
|---|---|---|---|
| Slack message | `ts` (epoch float seconds) | `sent_at` (ISO 8601 UTC) | not `event_ts` |
| Slack edit | `message.edited.ts` | `valid_from = sent_at`, `valid_until = edited.ts` | bitemporal mirror |
| Gmail | `internalDate` (epoch ms) | `sent_at` | prefer over `Date` header |
| Google Calendar event | `start.dateTime` | `occurred_at` | event time, not creation time |
| Salesforce Activity | `ActivityDate` | `occurred_at` | not `CreatedDate` |
| Salesforce record edit | `LastModifiedDate` | `updated_at` | secondary |
| Jira/Linear issue | `createdAt` | `created_at` | comments use `createdAt` as `sent_at` |

Run `validate_connector_metadata()` in your connector CI before calling
`Khora.remember()`:

```python
from khora.pipelines import validate_connector_metadata

metadata = {"sent_at": "2026-05-13T14:00:00Z", "source_system": "slack"}
warnings = validate_connector_metadata(metadata, source_type="slack")
assert not warnings, warnings
await kb.remember(content, namespace=ns, metadata=metadata, ...)
```

## What's Next?

- **[Chunkers](chunkers.md)** - Text splitting strategies
- **[Embedders](embedders.md)** - Vector generation
- **[Extractors](extractors.md)** - Entity extraction
- **[Semantic Expansion](semantic-expansion.md)** - Unification and inference
- **[Expertise System](expertise-system.md)** - Domain configuration
