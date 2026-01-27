# Extraction Pipeline Overview

Khora's extraction pipeline transforms raw content into structured knowledge. This document provides an overview of the extraction components and their interactions.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Extraction Pipeline                                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        Phase 1: Staging                                  ││
│  │                                                                          ││
│  │   Input          Checksum         Dedup           Document               ││
│  │  Content   →    Compute     →    Check      →    Creation               ││
│  │                                                                          ││
│  │  (Parallel staging with controlled concurrency)                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                       Phase 2: Enrichment                                ││
│  │                                                                          ││
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         ││
│  │   │ Chunker  │ →  │ Embedder │ →  │Extractor │ →  │  Store   │          ││
│  │   │          │    │          │    │          │    │          │          ││
│  │   │ - Fixed  │    │ - LiteLLM│    │ - LLM    │    │ - Chunks │          ││
│  │   │ - Seman- │    │ - OpenAI │    │ - Schema │    │ - Embeds │          ││
│  │   │   tic    │    │ - Cohere │    │ - JSON   │    │ - Nodes  │          ││
│  │   │ - Recur- │    │          │    │          │    │ - Edges  │          ││
│  │   │   sive   │    │          │    │          │    │          │          ││
│  │   └──────────┘    └──────────┘    └──────────┘    └──────────┘         ││
│  │                                                                          ││
│  │  (Parallel document processing with semaphores)                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                  Phase 3: Expansion (Optional)                           ││
│  │                                                                          ││
│  │   ┌────────────────────┐    ┌────────────────────┐                      ││
│  │   │  Cross-Tool        │    │   Relationship     │                      ││
│  │   │  Unifier           │ →  │   Inferrer         │                      ││
│  │   │                    │    │                    │                      ││
│  │   │  - Exact match     │    │  - Pattern rules   │                      ││
│  │   │  - Fuzzy match     │    │  - Transitive      │                      ││
│  │   │  - Embedding sim   │    │  - Configurable    │                      ││
│  │   └────────────────────┘    └────────────────────┘                      ││
│  │                                                                          ││
│  │  (Entity deduplication and relationship inference)                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Components

### Chunkers

Split documents into segments optimized for embedding and retrieval.

| Chunker | Description |
|---------|-------------|
| `FixedChunker` | Token-based splitting with overlap |
| `SemanticChunker` | Respects sentence/paragraph boundaries |
| `RecursiveChunker` | Hierarchical splitting (LangChain-style) |

See [Chunkers](chunkers.md) for details.

### Embedders

Generate vector embeddings for semantic search.

| Embedder | Description |
|----------|-------------|
| `LiteLLMEmbedder` | Unified interface to multiple providers |

Supports OpenAI, Cohere, and other embedding providers via LiteLLM.

See [Embedders](embedders.md) for details.

### Extractors

Extract structured knowledge from text.

| Extractor | Description |
|-----------|-------------|
| `LLMEntityExtractor` | LLM-based entity and relationship extraction |

Uses structured JSON output for reliable extraction.

See [Extractors](extractors.md) for details.

### Expertise System

Configure domain-specific extraction behavior.

| Component | Description |
|-----------|-------------|
| `ExpertiseConfig` | Complete domain knowledge definition |
| `EntityTypeConfig` | Define entity types and attributes |
| `RelationshipTypeConfig` | Define relationship types and constraints |
| `CorrelationRule` | Cross-tool entity matching rules |
| `InferenceRule` | Pattern-based relationship inference |

See [Expertise System](expertise-system.md) for details.

### Semantic Expansion

Enhance the knowledge graph through unification and inference.

| Component | Description |
|-----------|-------------|
| `SemanticExpander` | Orchestrates expansion phases |
| `CrossToolUnifier` | Deduplicate entities across sources |
| `RelationshipInferrer` | Infer relationships from patterns |

See [Semantic Expansion](semantic-expansion.md) for details.

## Configuration

### Default Settings

```python
# Chunking
chunk_strategy = "semantic"
chunk_size = 512          # tokens
chunk_overlap = 50        # tokens

# Embedding
embedding_model = "text-embedding-3-small"
embedding_dimension = 1536

# Extraction
extraction_model = "gpt-4o-mini"
```

### Per-Ingestion Configuration

```python
result = await lake.remember(
    content,
    chunk_strategy="recursive",
    chunk_size=1024,
    embedding_model="text-embedding-3-large",
    extraction_model="claude-sonnet-4-20250514",
    expertise="saas_expert",
)
```

### Batch Ingestion

```python
result = await lake.remember_batch(
    documents,
    max_concurrent_documents=5,
    max_concurrent_extractions=10,
    enable_expansion=True,
)
```

## Pipeline Orchestration

The extraction pipeline is orchestrated by Prefect:

```python
from khora.pipelines.flows import ingest_documents

result = await ingest_documents(
    namespace_id=namespace_id,
    documents=documents,
    storage=storage,
    expertise="saas_expert",
    chunk_strategy="semantic",
    enable_expansion=True,
)
```

### Concurrency Control

Concurrency is controlled at multiple levels:

```python
# Document-level concurrency
max_concurrent_documents = 5     # Process 5 docs simultaneously

# Extraction-level concurrency
max_concurrent_extractions = 10  # Max parallel LLM calls

# Staging concurrency
staging_semaphore = max_concurrent_documents * 2
```

### Error Handling

Documents that fail processing are marked with `FAILED` status:

```python
try:
    # Process document
    await process_document(document, storage, ...)
except Exception as e:
    document.mark_failed(str(e))
    await storage.update_document(document)
    # Continue with other documents
```

## Output

### RememberResult

```python
@dataclass
class RememberResult:
    document_id: UUID
    chunks_created: int
    entities_extracted: int
    relationships_extracted: int
```

### Batch Ingestion Result

```python
{
    "total_documents": 100,
    "processed_documents": 95,
    "skipped_documents": 3,     # Duplicates
    "failed_documents": 2,
    "total_chunks": 450,
    "total_entities": 200,
    "total_relationships": 150,
    "total_inferred_relationships": 25,
}
```

## Pipeline Registry

Pipelines are registered for discovery:

```python
from khora.pipelines.registry import pipeline

@pipeline("ingest", description="Document ingestion", tags=["ingestion"])
@flow(name="ingest_documents")
async def ingest_documents(...):
    ...

# List available pipelines
from khora.pipelines.registry import list_pipelines
pipelines = list_pipelines()
```

## Next Steps

- [Ingestion Pipeline](ingestion-pipeline.md) - Two-phase ingestion details
- [Chunkers](chunkers.md) - Text splitting strategies
- [Embedders](embedders.md) - Embedding generation
- [Extractors](extractors.md) - Entity extraction
- [Expertise System](expertise-system.md) - Domain configuration
- [Semantic Expansion](semantic-expansion.md) - Entity unification and inference
