# Extraction Pipeline Overview

When you store content in Khora, it doesn't just sit there as text. The extraction pipeline transforms your raw content into structured, searchable knowledge. This is where the magic happens.

## What the Pipeline Does

Think of it as three questions Khora asks about your content:

```
Your Document
      |
      v
+------------------------------------------+
|  "Have we seen this before?"             |  STAGING
|                                          |
|  Check duplicates, create record         |
+------------------------------------------+
      |
      v
+------------------------------------------+
|  "What's in here?"                       |  ENRICHMENT
|                                          |
|  Split, embed, extract knowledge         |
+------------------------------------------+
      |
      v
+------------------------------------------+
|  "How does this connect?"                |  EXPANSION
|                                          |
|  Merge entities, infer relationships     |
+------------------------------------------+
```

## Phase 1: Staging

Before we do any expensive processing, we check if this content is new:

```
Content arrives
      |
      v
  +-----------+     +---------------+
  |  Compute  | --> |  Check if     |
  |  checksum |     |  exists       |
  +-----------+     +---------------+
                           |
              +------------+------------+
              |                         |
         (duplicate)               (new content)
              |                         |
              v                         v
         Skip it              Create Document record
```

**Why this matters**: If you accidentally upload the same file twice, Khora recognizes it and skips the duplicate. The checksum is computed from the content itself, so even if the filename changes, duplicates are caught.

Staging runs in parallel with controlled concurrency - you can ingest thousands of documents without overwhelming your system.

## Phase 2: Enrichment

This is where content becomes knowledge. Each document goes through four transformations:

### 1. Chunking - Break It Apart

Raw documents are too large for embedding models and too unwieldy for retrieval. Chunking splits them into digestible pieces:

```
Original Document (5000 tokens)
            |
            v
+-------+ +-------+ +-------+ +-------+ +-------+
|Chunk 1| |Chunk 2| |Chunk 3| |Chunk 4| |Chunk 5|
| 512t  | | 512t  | | 512t  | | 512t  | | 456t  |
+-------+ +-------+ +-------+ +-------+ +-------+
     \______/  \______/  \______/
       overlap   overlap   overlap
```

Three chunking strategies are available:

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Fixed** | Split by token count | Predictable sizing |
| **Semantic** | Split at sentence boundaries | Natural language |
| **Recursive** | Try paragraphs, then lines, then sentences | Structured docs |

The overlap (typically 50 tokens) ensures context isn't lost at chunk boundaries.

### 2. Embedding - Capture Meaning

Each chunk gets converted to a vector - a list of numbers that captures its semantic meaning:

```
"Einstein developed the theory of relativity"
                    |
                    v
            [0.021, -0.156, 0.089, 0.334, ...]
                 (1536 dimensions)
```

Similar concepts get similar vectors. "Einstein's relativity theory" and "the physicist's work on space-time" will have vectors pointing in similar directions, even though the words are different.

We use LiteLLM to support multiple embedding providers:
- OpenAI (`text-embedding-3-small` default)
- Cohere
- Others

### 3. Extraction - Find the Knowledge

An LLM reads each chunk and extracts structured knowledge:

**Entities** - Named things worth remembering:
```python
Entity(
    name="Albert Einstein",
    entity_type="PERSON",
    description="German-born physicist, developed relativity",
    confidence=0.95
)
```

**Relationships** - How entities connect:
```python
Relationship(
    source="Albert Einstein",
    target="Theory of Relativity",
    relationship_type="DEVELOPED",
    confidence=0.90
)
```

**Episodes** - Events that happened:
```python
Episode(
    name="Nobel Prize Award",
    description="Einstein received the Nobel Prize in Physics",
    occurred_at="1921-11-09",
    participants=["Albert Einstein", "Nobel Committee"]
)
```

The extraction model (default: `gpt-4o-mini`) outputs structured JSON, making parsing reliable.

### 4. Storage - Put It Where It Belongs

Each piece of knowledge goes to its optimal storage backend:

```
                  Extracted Knowledge
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
    +-----------+   +-----------+   +-----------+
    |PostgreSQL |   | pgvector  |   |   Neo4j   |
    |           |   |           |   |           |
    | Documents |   | Chunk     |   | Entity    |
    | Metadata  |   | embeddings|   | nodes     |
    | Events    |   | Entity    |   | Relation  |
    |           |   | embeddings|   | edges     |
    +-----------+   +-----------+   +-----------+
```

## Phase 3: Expansion (Optional)

After basic enrichment, we can enhance the knowledge graph.

### Entity Unification

The same entity might be mentioned different ways across documents:

```
Document 1: "Microsoft Corporation"
Document 2: "Microsoft"
Document 3: "MSFT"
                |
                v
    +-------------------------+
    |  Cross-Tool Unifier     |
    |                         |
    |  Exact:  "Microsoft" == "Microsoft"
    |  Fuzzy:  "Microsft" ~= "Microsoft"
    |  Embed:  "the Redmond giant" ≈ "Microsoft"
    +-------------------------+
                |
                v
    Single unified "Microsoft" entity
    with merged attributes and sources
```

In smart mode (the default), within-document dedup happens during ingestion via an in-memory `EntityIndex` with O(1) lookups. Cross-document resolution runs once after all documents are processed, using token blocking to reduce pairwise comparisons from O(n^2) to O(n * k).

### Relationship Inference

Some relationships can be inferred from existing data:

```
Known:     Alice WORKS_FOR Acme Corp
           Bob WORKS_FOR Acme Corp
                |
                v
Inferred:  Alice COLLEAGUE_OF Bob
```

Inference rules are configurable through the expertise system. In smart mode, inference runs once on the fully resolved entity graph rather than per-document.

## Putting It Together

Here's the complete flow for `remember()`:

```python
result = await lake.remember(
    content="Einstein published his theory of general relativity in 1915...",
    title="Physics History",
    chunk_strategy="semantic",
    enable_expansion=True
)

# Result
RememberResult(
    document_id=UUID("..."),
    chunks_created=3,
    entities_extracted=5,
    relationships_extracted=4
)
```

Behind the scenes:
1. **Staging**: Checksum computed, no duplicate found, document created
2. **Chunking**: Split into 3 chunks using semantic boundaries
3. **Embedding + Extraction** (concurrent): 3 chunk embeddings generated *while* 5 entities and 4 relationships are extracted by LLM — the slower step determines wall-clock time
4. **Storage**: Chunks → pgvector, Entities → Neo4j + pgvector (parallel writes), Relationships → Neo4j (batch)
5. **Entity Embeddings**: 5 entity embeddings generated and stored in a single batch transaction
6. **Expansion**: Entity dedup merged 2 entities, inferred 1 relationship

## Configuration

### Default Settings

```python
chunk_strategy = "semantic"
chunk_size = 512          # tokens per chunk
chunk_overlap = 50        # overlap between chunks

embedding_model = "text-embedding-3-small"
embedding_dimension = 1536

extraction_model = "gpt-4o-mini"
```

### Per-Document Overrides

```python
await lake.remember(
    content,
    chunk_strategy="recursive",
    chunk_size=1024,
    embedding_model="text-embedding-3-large",
    extraction_model="claude-sonnet-4-20250514",
    expertise="technical_docs"  # Domain-specific extraction
)
```

### Batch Processing

For large ingestions:

```python
results = await lake.remember_batch(
    documents,
    max_concurrent_documents=10,     # Process 10 docs at once
    max_concurrent_extractions=20,   # Max 20 LLM calls in flight
    enable_expansion=True
)
```

## Error Handling

Documents that fail processing get marked, but don't stop the batch:

```python
# Document status lifecycle
PENDING -> PROCESSING -> COMPLETED
               |
               +-------> FAILED (error message stored)
```

Failed documents can be retried later without re-processing successful ones.

## What's Next?

- **[Ingestion Pipeline](ingestion-pipeline.md)** - Detailed pipeline mechanics
- **[Chunkers](chunkers.md)** - Text splitting strategies
- **[Embedders](embedders.md)** - Vector generation
- **[Extractors](extractors.md)** - Entity and relationship extraction
- **[Expertise System](expertise-system.md)** - Domain-specific configuration
- **[Semantic Expansion](semantic-expansion.md)** - Entity unification and inference
