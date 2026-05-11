# Architecture Overview

Khora is a **Khora** - a system that remembers everything you tell it and helps you find what you need later. Unlike a simple database or search engine, Khora understands your content at multiple levels: the literal words, the concepts and meanings, and the relationships between things.

## The Big Picture

At its heart, Khora combines three different ways of storing and finding information:

```
                              You
                               |
                               v
                    +-------------------+
                    |    Khora     |
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
    |  PostgreSQL   | |  pgvector   | |  Graph DB    |
    |               | |             | |              |
    |  The facts    | |  The        | |  The         |
    |  Documents,   | |  meaning    | |  connections |
    |  metadata,    | |  Semantic   | |  Who knows   |
    |  who owns     | |  similarity | |  whom, what  |
    |  what         | |  search     | |  relates to  |
    +---------------+ +-------------+ |  what        |
                                      +--------------+

    Alternative: SurrealDB (unified backend — all three roles in one DB)
```

**PostgreSQL** is your source of truth - it stores the actual documents, tracks who owns what, and keeps an immutable log of everything that happens.

**pgvector** enables semantic search - when you ask "what do we know about machine learning?", it finds content that's *conceptually* related, even if it doesn't contain those exact words.

**Graph DB** (Neo4j, Kuzu, Memgraph, or SurrealDB) captures relationships - people, organizations, concepts, and how they connect. When you ask "who works with Alice?", it traverses a graph of knowledge to find the answer.

**SurrealDB** is an alternative unified backend that can serve all three roles (relational, vector, graph) in a single database, simplifying deployment at the cost of specialization.

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
|     Phase 2: Enrichment (Staged Batch)     |
|                                            |
|   "What does this content contain?"        |
|                                            |
|   Stage 1: CHUNK all documents             |
|   Stage 2: EMBED + EXTRACT in parallel     |
|            (asyncio.gather)                |
|   Stage 3: STORE in batch writes           |
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

### Khora

Your primary interface. Lives at `src/khora/khora.py`.

```python
from khora import Khora

async with Khora() as lake:
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

Khora handles all the complexity of coordinating three databases, running extraction pipelines, and combining search results. You just tell it what to remember and what to recall.

### StorageCoordinator

The traffic controller. Lives at `src/khora/storage/coordinator.py`.

When Khora needs to store an entity, StorageCoordinator knows that it should go to Neo4j for graph queries *and* pgvector for similarity search. When you delete a document, it ensures cleanup happens everywhere. It also provides `transaction()` for atomic multi-backend writes with savepoint support.

### HybridQueryEngine

The search brain. Lives at `src/khora/query/engine.py`.

This component orchestrates the multi-source search pipeline shown above. It runs searches in parallel, applies Reciprocal Rank Fusion to combine results, and handles temporal filtering and reranking.

### Ingestion Pipeline

The extraction orchestrator. Lives at `src/khora/pipelines/flows/ingest.py`.

A native async Python pipeline that manages the ingestion workflow — chunking documents, generating embeddings, extracting entities. It uses a staged batch architecture where all documents flow through each stage together, with `asyncio.gather` for parallel embed+extract and batch database writes for storage.

## Multi-Tenancy: Who Owns What

Khora isolates data through **namespaces** — the sole unit of isolation. There is no organization or workspace hierarchy within khora; higher-level grouping is the consuming service's responsibility.

```
Namespace A  (your data lives here)
Namespace B  (another dataset)
Namespace A' (version 2 of A, for zero-downtime rebuilds)
```

**Namespaces** hold actual data and can be versioned — create a new version, populate it, then swap it in atomically.

Each namespace has two IDs:
- **`namespace_id`** — Stable across all versions. Use this in your application.
- **`id`** — Row-level, changes per version. Used internally for FK references.

Public API methods accept `namespace_id` and resolve to the active version's `id` automatically via `resolve_namespace()`. You can also look up a namespace by its stable ID:

```python
ns = await lake.get_namespace_by_stable_id(stable_namespace_id)
```

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

## Observability: Logfire / OpenTelemetry

Khora includes optional Logfire instrumentation for OTEL-compatible distributed tracing. Install with `pip install khora[logfire]`.

When present, spans are emitted for:
- LLM extraction calls
- Entity deduplication passes
- Skeleton build phases
- Ingestion pipeline stages

Two APIs are available:
- **`@trace` decorator** — Automatic span creation per function. Auto-captures arguments as span attributes.
- **`trace_span()` context manager** — For complex methods needing mid-function attributes.

When Logfire is not installed, both APIs short-circuit to zero-overhead no-ops. Khora never calls `logfire.configure()` — that's the consumer's responsibility.

```python
from khora.telemetry import trace, trace_span

@trace("khora.search", exclude={"query"}, result=lambda r: {"count": len(r)})
async def search(query: str, namespace_id: UUID) -> list:
    ...
```

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
