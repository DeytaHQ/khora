# Architecture Overview

Khora is a **Memory Lake** - a system that remembers everything you tell it and helps you find what you need later. Unlike a simple database or search engine, Khora understands your content at multiple levels: the literal words, the concepts and meanings, and the relationships between things.

## The Big Picture

At its heart, Khora combines three different ways of storing and finding information:

```
                              You
                               |
                               v
                    +-------------------+
                    |    MemoryLake     |
                    |                   |
                    |  remember()       |  <- Store new knowledge
                    |  recall()         |  <- Find what you need
                    |  forget()         |  <- Remove when needed
                    +--------+----------+
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
    +---------------+ +-------------+ +--------------+
    |  PostgreSQL   | |  pgvector   | |    Neo4j     |
    |               | |             | |              |
    |  The facts    | |  The        | |  The         |
    |  Documents,   | |  meaning    | |  connections |
    |  metadata,    | |  Semantic   | |  Who knows   |
    |  who owns     | |  similarity | |  whom, what  |
    |  what         | |  search     | |  relates to  |
    +---------------+ +-------------+ |  what        |
                                      +--------------+
```

**PostgreSQL** is your source of truth - it stores the actual documents, tracks who owns what, and keeps an immutable log of everything that happens.

**pgvector** enables semantic search - when you ask "what do we know about machine learning?", it finds content that's *conceptually* related, even if it doesn't contain those exact words.

**Neo4j** captures relationships - people, organizations, concepts, and how they connect. When you ask "who works with Alice?", it traverses a graph of knowledge to find the answer.

## How Data Flows Through Khora

### Storing Knowledge (Ingestion)

When you call `remember()`, your content goes through a pipeline that extracts meaning:

```
Your Content
     |
     v
+--------------------------------------------+
|           Phase 1: Staging                 |
|                                            |
|   "Did we see this before?"                |
|                                            |
|   - Compute a checksum                     |
|   - Check for duplicates                   |
|   - Create a document record               |
+--------------------------------------------+
     |
     v
+--------------------------------------------+
|          Phase 2: Enrichment               |
|                                            |
|   "What does this content contain?"        |
|                                            |
|   1. CHUNK - Split into digestible pieces  |
|   2. EMBED - Convert to vectors            |
|   3. EXTRACT - Find entities & relations   |
|   4. STORE - Save everywhere it belongs    |
+--------------------------------------------+
     |
     v
+--------------------------------------------+
|    Phase 3: Expansion (Optional)           |
|                                            |
|   "How does this connect to what we know?" |
|                                            |
|   - Merge duplicate entities               |
|   - Infer new relationships                |
|   - Smart mode: O(1) per-doc dedup,        |
|     single post-ingestion resolution pass  |
+--------------------------------------------+
```

The key insight: content is stored *multiple times in different forms*. The original text lives in PostgreSQL. Vector embeddings of each chunk live in pgvector. Extracted entities and their relationships live in Neo4j. This redundancy is intentional - each storage backend excels at different types of queries.

### Finding Knowledge (Query)

When you call `recall()`, Khora searches all three backends in parallel and combines the results:

```
Your Question
     |
     v
+--------------------------------------------+
|          Query Understanding               |
|                                            |
|   One LLM call figures out:                |
|   - What are you really asking?            |
|   - Any people/places/things mentioned?    |
|   - Is there a time component?             |
|   - What search strategy fits best?        |
+--------------------------------------------+
     |
     +------------------+------------------+
     |                  |                  |
     v                  v                  v
+-----------+    +-----------+    +-----------+
|  Vector   |    |   Graph   |    |  Keyword  |
|  Search   |    |   Search  |    |   Search  |
|           |    |           |    |           |
|  "What's  |    |  "What's  |    |  "What    |
|  similar  |    |  connected|    |   matches |
|  in       |    |  to these |    |   these   |
|  meaning?"|    |  entities"|    |   words?" |
+-----------+    +-----------+    +-----------+
     |                  |                  |
     +------------------+------------------+
                        |
                        v
            +------------------------+
            |   Reciprocal Rank      |
            |   Fusion (RRF)         |
            |                        |
            |   Combine rankings     |
            |   intelligently        |
            +------------------------+
                        |
                        v
                 Your Results
```

This hybrid approach means you get the best of all worlds. Semantic search finds conceptually relevant content. Graph search finds related entities and follows connections. Keyword search catches exact matches that might otherwise be missed.

## The Core Components

### MemoryLake

Your primary interface. Lives at `src/khora/memory_lake.py`.

```python
from khora import MemoryLake

async with MemoryLake() as lake:
    # Store something
    result = await lake.remember(
        "Einstein developed relativity while working at the patent office.",
        title="Einstein Biography"
    )

    # Find it later
    results = await lake.recall("Who developed the theory of relativity?")

    # Remove if needed
    await lake.forget(result.document_id)
```

MemoryLake handles all the complexity of coordinating three databases, running extraction pipelines, and combining search results. You just tell it what to remember and what to recall.

### StorageCoordinator

The traffic controller. Lives at `src/khora/storage/coordinator.py`.

When MemoryLake needs to store an entity, StorageCoordinator knows that it should go to Neo4j for graph queries *and* pgvector for similarity search. When you delete a document, it ensures cleanup happens everywhere.

### HybridQueryEngine

The search brain. Lives at `src/khora/query/engine.py`.

This component orchestrates the multi-source search pipeline shown above. It runs searches in parallel, applies Reciprocal Rank Fusion to combine results, and handles temporal filtering and reranking.

### PipelineManager

The extraction orchestrator. Lives at `src/khora/pipelines/`.

Built on Prefect, it manages the ingestion workflow - chunking documents, generating embeddings, extracting entities. It handles concurrency, retries, and progress tracking.

## Multi-Tenancy: Who Owns What

Khora supports multiple isolated data spaces through a hierarchy:

```
Organization
     |
     +-- Workspace
     |        |
     |        +-- Namespace (your data lives here)
     |        |
     |        +-- Namespace (version 2, for replacements)
     |
     +-- Workspace (different team)
              |
              +-- Namespace
```

**Organizations** are top-level containers (your company).

**Workspaces** group related namespaces (a team, a project).

**Namespaces** hold actual data. They can be versioned - create a new version, populate it, then swap it in atomically. Great for full rebuilds.

## Event Sourcing: Nothing is Forgotten

Every change to Khora is recorded as an immutable event:

```python
MemoryEvent(
    event_type="document.created",
    resource_type="document",
    resource_id=doc_id,
    actor_id="user:alice",
    data={"title": "Meeting Notes", "source": "upload"},
    timestamp="2024-01-15T10:30:00Z"
)
```

This enables:
- **Audit trails** - Who changed what, when?
- **Temporal queries** - What did we know last Tuesday?
- **Debugging** - Replay events to understand issues

## Configuration: Layers of Overrides

Settings flow from general to specific:

```
Environment Variables (KHORA_DATABASE_URL, etc.)
          |
          v
    KhoraConfig  <-- Application-wide defaults
          |
          v
   StorageConfig  <-- Can override storage settings
          |
          v
 Namespace Config  <-- Per-namespace overrides (different LLM, etc.)
```

This lets you have different configurations for different use cases while maintaining sensible defaults.

## Protocol-Based Design

Each storage backend implements a protocol (interface). This means you could swap Neo4j for a different graph database, or pgvector for a different vector store, without changing the rest of the system:

```python
class GraphBackendProtocol(Protocol):
    async def create_entity(self, entity: Entity) -> None: ...
    async def get_entity(self, id: UUID) -> Entity | None: ...
    async def find_related(self, entity_id: UUID, ...) -> list[Entity]: ...
```

## What's Next?

- **[Storage Backends](storage-backends.md)** - How PostgreSQL, pgvector, and Neo4j work together
- **[Multi-Tenancy](multi-tenancy.md)** - Organizations, workspaces, and namespaces in detail
- **[Event Sourcing](event-sourcing.md)** - The immutable event log and what you can do with it
