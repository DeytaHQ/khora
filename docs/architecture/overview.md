# Architecture Overview

Khora is a Memory Lake system that unifies three storage paradigms into a cohesive knowledge management platform. This document describes the high-level system design, core components, and data flow.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MemoryLake API                                  │
│                         (Library + FastAPI Service)                          │
│                                                                              │
│  remember() ─────────────┐                          ┌────────── recall()    │
│  forget() ──────────────┼──────────────────────────┼──────── list_entities()│
│  remember_batch() ──────┘                          └────── find_related()   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │   HybridQuery    │   │   Pipeline       │   │      ACL         │        │
│  │     Engine       │   │   Manager        │   │    Enforcer      │        │
│  │                  │   │   (Prefect)      │   │                  │        │
│  │ - Vector search  │   │ - Ingestion      │   │ - Permission     │        │
│  │ - Graph search   │   │ - Sync flows     │   │   checking       │        │
│  │ - Keyword search │   │ - Chunking       │   │ - Inheritance    │        │
│  │ - RRF fusion     │   │ - Embedding      │   │                  │        │
│  │ - Query under-   │   │ - Extraction     │   │                  │        │
│  │   standing       │   │                  │   │                  │        │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘        │
│           │                      │                      │                   │
├───────────┴──────────────────────┴──────────────────────┴───────────────────┤
│                          StorageCoordinator                                  │
│                   (Orchestrates all storage backends)                        │
├─────────┬───────────────────┬───────────────────┬───────────────────────────┤
│         │                   │                   │                            │
│  ┌──────┴──────┐     ┌──────┴──────┐     ┌──────┴──────┐     ┌────────────┐ │
│  │ PostgreSQL  │     │  pgvector   │     │   Neo4j     │     │   Event    │ │
│  │             │     │             │     │             │     │   Store    │ │
│  │ - Documents │     │ - Chunk     │     │ - Entity    │     │            │ │
│  │ - Tenancy   │     │   embeddings│     │   nodes     │     │ - Immutable│ │
│  │ - Metadata  │     │ - Entity    │     │ - Relations │     │   events   │ │
│  │ - Sync      │     │   embeddings│     │ - Traversal │     │ - Audit    │ │
│  │   checkpts  │     │ - IVFFlat   │     │             │     │   trail    │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │     LiteLLM      │
                            │                  │
                            │ - Embeddings     │
                            │ - Extraction     │
                            │ - Understanding  │
                            │                  │
                            │ OpenAI, Claude,  │
                            │ Cohere, etc.     │
                            └──────────────────┘
```

## Core Components

### MemoryLake

The primary API for all memory operations. Located at `src/khora/memory_lake.py`.

**Responsibilities:**
- Provide a simple, unified interface (`remember`, `recall`, `forget`)
- Manage connection lifecycle to all backends
- Handle namespace resolution and creation
- Orchestrate the ingestion pipeline for new memories

**Key Methods:**
- `remember(content, ...)` - Store content with automatic chunking, embedding, and entity extraction
- `recall(query, ...)` - Search memories using hybrid search with optional agentic exploration
- `forget(document_id)` - Remove a document and its associated chunks
- `remember_batch(documents, ...)` - Efficiently process multiple documents in parallel

### StorageCoordinator

Orchestrates all storage backends through a unified interface. Located at `src/khora/storage/coordinator.py`.

**Responsibilities:**
- Delegate operations to appropriate backends
- Handle cross-backend transactions
- Provide health checking for all backends
- Manage batch operations efficiently

**Backend Delegation:**
- **Relational operations** → PostgreSQL (documents, tenancy, sync checkpoints)
- **Vector operations** → pgvector (chunk/entity similarity search)
- **Graph operations** → Neo4j (entities, relationships, traversal)
- **Event operations** → PostgreSQL Event Store

### HybridQueryEngine

Combines multiple search methods with intelligent fusion. Located at `src/khora/query/engine.py`.

**Responsibilities:**
- Execute parallel searches across vector, graph, and keyword backends
- Apply Reciprocal Rank Fusion (RRF) to combine results
- Provide LLM-based query understanding
- Support temporal filtering and recency bias
- Enable agentic multi-step exploration

**Search Pipeline (7 steps):**
1. Query Understanding - Extract intent, entities, temporal refs (single LLM call)
2. Entity Linking - Link query mentions to stored entities
3. Multi-source Search - Vector, graph, keyword in parallel
4. RRF Fusion - Combine ranked results
5. Temporal Filtering - Apply time constraints
6. Reranking (optional) - Neural re-ranking
7. Final Limiting - Return top results

### PipelineManager

Manages Prefect-based ingestion workflows. Located at `src/khora/pipelines/`.

**Responsibilities:**
- Register and execute ingestion pipelines
- Handle two-phase ingestion (staging + enrichment)
- Manage concurrency controls
- Track pipeline execution status

## Data Flow

### Ingestion Path

```
Content Input
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: Staging                              │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Compute    │ → │   Checksum   │ → │    Create    │       │
│  │   Checksum   │    │  Dedup Check │    │   Document   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
│  (Parallel staging with semaphore-controlled concurrency)        │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 2: Enrichment                           │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Chunk     │ → │   Generate   │ → │   Extract    │       │
│  │   Document   │    │  Embeddings  │    │  Entities    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Store     │    │    Store     │    │    Store     │       │
│  │   Chunks     │    │  Embeddings  │    │   Entities   │       │
│  │ (pgvector)   │    │  (pgvector)  │    │   (Neo4j)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
│  (Parallel processing with configurable max_concurrent)          │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 3: Expansion (Optional)                       │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │   Cross-Tool │ → │ Relationship │                            │
│  │  Unification │    │  Inference   │                            │
│  └──────────────┘    └──────────────┘                           │
│                                                                  │
│  (Entity dedup, pattern-based relationship inference)            │
└─────────────────────────────────────────────────────────────────┘
```

### Query Path

```
Query Input
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Query Understanding                            │
│                   (Single LLM Call)                              │
│                                                                  │
│  Extracts: intent, entities, temporal refs, keywords,           │
│            source priority, search strategy, follow-up queries   │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Entity Linking                                 │
│                                                                  │
│  Links query entity mentions to stored entities using:          │
│  - Exact matching                                                │
│  - Fuzzy matching (Levenshtein)                                  │
│  - Embedding similarity                                          │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Parallel Search                                │
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   Vector    │   │    Graph    │   │   Keyword   │           │
│  │   Search    │   │   Search    │   │   (BM25)    │           │
│  │             │   │             │   │             │           │
│  │  pgvector   │   │   Neo4j     │   │  In-memory  │           │
│  │  similarity │   │  traversal  │   │   index     │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                    │
│         └────────────────┬┘─────────────────┘                   │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   Reciprocal Rank     │                          │
│              │      Fusion (RRF)     │                          │
│              │                       │                          │
│              │ score = Σ(w/(k+rank)) │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Post-Processing                                │
│                                                                  │
│  - Temporal filtering (before/after/between)                    │
│  - Recency bias (exponential decay)                             │
│  - Optional neural reranking                                    │
│  - Result limiting                                               │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
   QueryResult
   (chunks, entities, graph_context, metadata)
```

## Protocol-Based Design

Khora uses Python protocols to define backend interfaces, enabling:

- **Swappability**: Replace Neo4j with another graph database
- **Testing**: Mock backends for unit tests
- **Extensibility**: Add new backend types without modifying core code

**Key Protocols** (defined in `src/khora/storage/backends/base.py`):
- `RelationalBackendProtocol` - Documents, tenancy, metadata
- `VectorBackendProtocol` - Embeddings and similarity search
- `GraphBackendProtocol` - Entities, relationships, traversal
- `EventStoreProtocol` - Immutable event log

## Configuration Hierarchy

```
Environment Variables
        │
        ▼
┌───────────────────┐
│   KhoraConfig     │ ← KHORA_DATABASE_URL, KHORA_NEO4J_URL, etc.
│                   │
│ - database_url    │
│ - neo4j_url       │
│ - llm settings    │
│ - api settings    │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  StorageConfig    │ ← Programmatic override
│                   │
│ - postgresql_url  │
│ - pgvector_url    │
│ - neo4j settings  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Namespace Config  │ ← Per-namespace overrides
│    Overrides      │
│                   │
│ - custom settings │
│ - per-tenant LLM  │
└───────────────────┘
```

## Next Steps

- [Storage Backends](storage-backends.md) - Deep dive into PostgreSQL, pgvector, and Neo4j
- [Multi-Tenancy](multi-tenancy.md) - Organization, Workspace, Namespace hierarchy
- [Event Sourcing](event-sourcing.md) - Immutable event log and temporal queries
