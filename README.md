# Khora

> *"Khora is the receptacle, the space, the matrix in which all things come to be."*
> *— Plato, Timaeus*

In Plato's cosmology, **Khora** (χώρα) is the primordial receptacle—neither being nor non-being, but the space that receives all forms and gives them place. It is the nurse of becoming, the womb of the cosmos where the eternal Forms find material expression. Khora does not impose form; it receives, holds, and makes manifestation possible.

This project embodies that philosophy: **Khora is a memory lake**—a receptacle for knowledge that receives information from disparate sources, holds it in structured form, and enables its retrieval through multiple paths of inquiry. Just as Plato's Khora mediates between the intelligible and sensible worlds, this Memory Lake bridges raw data and meaningful knowledge through semantic extraction, graph relationships, and temporal context.

---

## Overview

Khora is a **Memory Lake** system that combines three storage paradigms:

- **Knowledge Graph** (Neo4j) — Entities and their relationships
- **Vector Database** (pgvector) — Semantic embeddings for similarity search
- **Relational Database** (PostgreSQL) — Documents, events, and metadata

It supports **multi-tenancy** with hierarchical isolation (Organization → Workspace → Namespace), **event sourcing** for complete audit trails, and **hybrid search** combining vector similarity, graph traversal, and keyword matching.

### Key Features

- **Library-First Design**: Use as a Python library or deploy as a FastAPI service
- **Pluggable Engines**: Choose GraphRAG (knowledge graphs) or Skeleton (temporal-first)
- **Hybrid Search**: Vector + graph + keyword search with Reciprocal Rank Fusion
- **Multi-Tenancy**: Namespace-level isolation (shared mode with ACLs designed but not yet active — see `docs/design/namespace-optimization-plan.md`)
- **Event Sourcing**: Immutable event log for temporal queries and audit trails
- **LiteLLM Integration**: Unified access to OpenAI, Anthropic, Google, and other providers
- **Prefect Pipelines**: Orchestrated ingestion with checksum-based change detection
- **Semantic Extraction**: LLM-powered entity and relationship extraction

---

## Pluggable Engines

Khora supports three engines with different strengths:

| Engine | Focus | Best For | LLM Cost |
|--------|-------|----------|----------|
| **GraphRAG** | Knowledge graphs | Knowledge bases, entity exploration | Higher |
| **VectorCypher** | Hybrid retrieval | Multi-hop queries, complex relationships | Medium |
| **Skeleton Construction** | Temporal events | Chat logs, events, cost-sensitive apps | 5-10x lower |

### GraphRAG Engine (Default)

Full knowledge graph construction with entity and relationship extraction:

```python
from khora import MemoryLake

async with MemoryLake("postgresql://...", engine="graphrag") as lake:
    # Extracts entities and relationships from all content
    result = await lake.remember("Einstein developed relativity in 1905.")
    print(f"Extracted {result.entities_extracted} entities")

    # Graph-based retrieval
    entities = await lake.list_entities(entity_type="PERSON")
    related = await lake.find_related_entities(entity_id, max_depth=2)
```

**Requirements:** PostgreSQL + pgvector + Neo4j/Kuzu/Memgraph

### Skeleton Construction Engine (Temporal-First)

Cost-optimized engine with bi-temporal model and skeleton indexing:

```python
from khora import MemoryLake

async with MemoryLake("postgresql://...", engine="skeleton") as lake:
    # Store with temporal metadata
    result = await lake.remember(
        "Team standup notes",
        metadata={
            "occurred_at": "2024-01-15T09:00:00Z",
            "author": "alice@company.com",
            "channel": "engineering"
        }
    )

    # Temporal and structured filtering
    results = await lake.recall(
        "What decisions were made?",
        temporal_filter={
            "occurred_after": "2024-01-01",
            "author": "alice@company.com"
        },
        hybrid_alpha=0.7  # 70% vector, 30% BM25
    )
```

**Requirements:** PostgreSQL + pgvector only (Neo4j optional)

### VectorCypher Engine (Hybrid Retrieval)

Combines vector similarity search with Cypher graph traversal for complex multi-hop queries:

```python
from khora import MemoryLake
from khora.engines.vectorcypher import VectorCypherConfig

async with MemoryLake(
    "postgresql://...",
    engine="vectorcypher",
    engine_kwargs={"vectorcypher_config": VectorCypherConfig(
        skeleton_core_ratio=0.70,        # 70% get full KG extraction
        fusion_simple_vector_weight=0.8, # Vector-heavy for simple queries
        fusion_complex_graph_weight=0.7, # Graph-heavy for complex queries
    )},
) as lake:
    # Store with temporal context
    result = await lake.remember(
        "Meeting notes from Q1 planning with John and Sarah",
        title="Q1 Planning",
        metadata={"author": "alice@company.com"},
    )

    # Multi-hop retrieval: automatically routes to graph traversal
    results = await lake.recall(
        "How are John and Sarah connected through projects?",
        graph_depth=2,
    )
```

**Requirements:** PostgreSQL + pgvector + Neo4j

See [Engine Comparison](docs/engines/engine-comparison.md) for detailed guidance.

---

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

| Topic | Description |
|-------|-------------|
| **Engines** | |
| [Skeleton Construction Engine](docs/engines/skeleton-engine.md) | Temporal-first engine documentation |
| [VectorCypher Engine](docs/engines/vectorcypher-engine.md) | Hybrid vector+graph engine documentation |
| [Engine Comparison](docs/engines/engine-comparison.md) | GraphRAG vs Skeleton vs VectorCypher comparison |
| [Temporal Model](docs/engines/temporal-model.md) | Bi-temporal design deep dive |
| [Skeleton Indexing](docs/engines/skeleton-indexing.md) | Cost optimization via PageRank |
| [Hybrid Search](docs/engines/hybrid-search.md) | Vector + BM25 fusion |
| **Architecture** | |
| [Overview](docs/architecture/overview.md) | System design, components, data flow |
| [Storage Backends](docs/architecture/storage-backends.md) | PostgreSQL, pgvector, Neo4j configuration |
| [Multi-Tenancy](docs/architecture/multi-tenancy.md) | Organization → Workspace → Namespace hierarchy |
| [Event Sourcing](docs/architecture/event-sourcing.md) | Immutable event log, audit trails |
| **Data Models** | |
| [Overview](docs/data-models/overview.md) | Model relationships and purposes |
| [Documents & Chunks](docs/data-models/documents-chunks.md) | Content storage and chunking |
| [Knowledge Graph](docs/data-models/knowledge-graph.md) | Entities, relationships, episodes |
| [Events](docs/data-models/events.md) | MemoryEvent types and usage |
| **Extraction Pipeline** | |
| [Overview](docs/extraction/overview.md) | Pipeline components and flow |
| [Ingestion Pipeline](docs/extraction/ingestion-pipeline.md) | Two-phase ingestion with Prefect |
| [Chunkers](docs/extraction/chunkers.md) | Fixed, semantic, recursive chunking |
| [Embedders](docs/extraction/embedders.md) | LiteLLM-based embedding generation |
| [Extractors](docs/extraction/extractors.md) | LLM entity and relationship extraction |
| [Expertise System](docs/extraction/expertise-system.md) | Domain-specific extraction configuration |
| [Semantic Expansion](docs/extraction/semantic-expansion.md) | Entity unification and relationship inference |
| **Query Engine** | |
| [Overview](docs/query-engine/overview.md) | HybridQueryEngine architecture |
| [Search Modes](docs/query-engine/search-modes.md) | Vector, graph, keyword, hybrid search |
| [Query Understanding](docs/query-engine/query-understanding.md) | LLM-based query analysis |
| [Fusion](docs/query-engine/fusion.md) | Reciprocal Rank Fusion (RRF) |
| [Temporal Queries](docs/query-engine/temporal-queries.md) | Time filtering and recency bias |
| [Agentic Search](docs/query-engine/agentic-search.md) | Multi-step exploration |
| **Performance** | |
| [Rust Acceleration](docs/architecture/performance-optimization.md) | Native Rust extensions for CPU-bound operations |
| [Performance Optimization](docs/architecture/performance-optimization.md) | Query caching, batch operations, entity resolution |
| **Planning** | |
| [Roadmap](docs/roadmap.md) | Future improvements and features |
| **References** | |
| [References](docs/REFERENCES.md) | Research papers and inspirations |
| [Changelog](CHANGELOG.md) | Release history and migration notes |

---

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for package management
- PostgreSQL with pgvector extension
- Neo4j (optional, for graph features)

### Quick Install

```bash
# Clone and install
git clone https://github.com/DeytaHQ/khora.git
cd khora
uv sync --all-extras

# Install pre-commit hooks
uv run prek install
```

### Start Development Databases

```bash
# Start PostgreSQL and Neo4j via Docker
make dev

# Run database migrations
uv run alembic upgrade head
```

---

## Usage

### As a Library

The primary interface is the `MemoryLake` class:

```python
from khora import MemoryLake, SearchMode

async def main():
    # Simplest - reads KHORA_DATABASE_URL from environment
    async with MemoryLake() as lake:
        # Store a memory
        result = await lake.remember(
            "Albert Einstein developed the theory of relativity in 1905.",
            title="Einstein Biography",
            source="wikipedia",
        )
        print(f"Stored document: {result.document_id}")
        print(f"Extracted {result.entities_extracted} entities")

        # Recall relevant memories
        memories = await lake.recall(
            "Who developed relativity?",
            limit=5,
            mode=SearchMode.HYBRID,  # vector + graph + keyword
        )
        print(f"Found {len(memories.chunks)} relevant chunks")
        print(f"Context: {memories.context_text}")

        # Explore entity relationships
        entities = await lake.list_entities(entity_type="PERSON")
        for entity in entities:
            related = await lake.find_related_entities(entity.id, max_depth=2)
            print(f"{entity.name} is related to {len(related)} entities")

        # Forget a memory
        await lake.forget(result.document_id)

import asyncio
asyncio.run(main())
```

### Simplified Constructor

The `MemoryLake` constructor supports multiple initialization patterns:

```python
from khora import MemoryLake, KhoraConfig

# 1. From environment variables (KHORA_DATABASE_URL)
lake = MemoryLake()

# 2. Explicit database URL
lake = MemoryLake("postgresql://localhost/mydb")

# 3. With graph backend
lake = MemoryLake(
    "postgresql://localhost/mydb",
    graph_url="bolt://localhost:7687",
)

# 4. Custom embedding model
lake = MemoryLake(
    "postgresql://localhost/mydb",
    embedding_model="text-embedding-3-large",
)

# 5. Full configuration object (for advanced use)
config = KhoraConfig(
    database_url="postgresql://localhost/mydb",
    neo4j_url="bolt://localhost:7687",
)
lake = MemoryLake(config)
```

### Batch Ingestion

For efficient bulk document ingestion:

```python
from khora import MemoryLake

async with MemoryLake(database_url) as lake:
    # Batch ingestion with automatic optimization
    result = await lake.remember_batch(
        [
            {"content": "Document 1 text...", "title": "Doc 1"},
            {"content": "Document 2 text...", "title": "Doc 2"},
            {"content": "Document 3 text...", "title": "Doc 3"},
        ],
        deduplicate=True,           # Cross-document entity deduplication
        infer_relationships=True,   # Relationship inference after ingestion
        on_progress=lambda done, total: print(f"Progress: {done}/{total}"),
    )

    print(f"Processed: {result.processed}/{result.total} documents")
    print(f"Chunks: {result.chunks}, Entities: {result.entities}")
    print(f"Relationships: {result.relationships}")
```

### Raw Search (No LLM Features)

For benchmarks or simple searches without LLM overhead:

```python
# Skip query understanding, entity linking, reranking, HyDE
results = await lake.recall(
    "search query",
    mode=SearchMode.ALL,
    raw=True,  # Disables all LLM features
)
```

### Search Modes

```python
from khora import MemoryLake, SearchMode

async with MemoryLake() as lake:
    # Vector-only search (semantic similarity)
    results = await lake.recall("quantum physics", mode=SearchMode.VECTOR)

    # Graph-only search (entity relationships)
    results = await lake.recall("Einstein collaborators", mode=SearchMode.GRAPH)

    # Hybrid search (combines all sources with RRF)
    results = await lake.recall("relativity theory", mode=SearchMode.HYBRID)

    # All sources (returns results from each separately)
    results = await lake.recall("physics discoveries", mode=SearchMode.ALL)
```

### Multi-Tenancy

```python
from khora import MemoryLake

async with MemoryLake() as lake:
    # Simple: Get or create a namespace by name
    namespace_id = await lake.ensure_namespace("physics", description="Physics research")

    # Store memories in specific namespace
    await lake.remember(
        "Important research findings...",
        namespace=namespace_id,
    )

    # Query within namespace (isolated from other namespaces)
    results = await lake.recall("findings", namespace=namespace_id)

    # Get namespace statistics
    stats = await lake.stats(namespace=namespace_id)
    print(f"Documents: {stats.documents}, Entities: {stats.entities}")
```

### As a Service

```bash
# Start the API server
uv run khora serve --reload
```

#### API Endpoints

**Memory Operations:**
```bash
# Store a memory
curl -X POST http://localhost:8100/memory/remember \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Einstein developed relativity in 1905.",
    "title": "Physics History",
    "skill_name": "general_entities"
  }'

# Recall memories
curl -X POST http://localhost:8100/memory/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who developed relativity?",
    "limit": 10,
    "mode": "hybrid"
  }'

# Get a document
curl http://localhost:8100/memory/documents/{document_id}

# List entities
curl "http://localhost:8100/memory/entities?entity_type=PERSON&limit=50"

# Get related entities
curl "http://localhost:8100/memory/entities/{entity_id}/related?max_depth=2"

# Forget a memory
curl -X DELETE http://localhost:8100/memory/forget \
  -H "Content-Type: application/json" \
  -d '{"document_id": "uuid-here"}'
```

**Namespace Management:**
```bash
# Create organization
curl -X POST http://localhost:8100/namespaces/organizations \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp", "slug": "acme"}'

# Create workspace
curl -X POST http://localhost:8100/namespaces/workspaces \
  -H "Content-Type: application/json" \
  -d '{"organization_id": "org-uuid", "name": "Research"}'

# Create namespace
curl -X POST http://localhost:8100/namespaces/ \
  -H "Content-Type: application/json" \
  -d '{"workspace_id": "ws-uuid", "name": "Physics"}'
```

**Sync & Pipelines:**
```bash
# Ingest documents
curl -X POST http://localhost:8100/sync/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "namespace_id": "ns-uuid",
    "documents": [{"content": "Document text..."}],
    "skill_name": "general_entities"
  }'

# List available pipelines
curl http://localhost:8100/sync/pipelines
```

**Health Checks:**
```bash
curl http://localhost:8100/status        # Service status
curl http://localhost:8100/health        # Health check
curl http://localhost:8100/health/ready  # Readiness probe
curl http://localhost:8100/health/live   # Liveness probe
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              MemoryLake API                                  │
│                         (Library + FastAPI Service)                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │    Query     │   │  Pipelines   │   │  VectorCypher│   │   Config     │   │
│  │   Engine     │   │  (Prefect)   │   │   Router     │   │   Resolver   │   │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   │
│         │                  │                  │                  │           │
├─────────┴──────────────────┴──────────────────┴──────────────────┴───────────┤
│                          Storage Coordinator                                 │
├─────────┬───────────────────┬───────────────────┬────────────────────────────┤
│         │                   │                   │                            │
│  ┌──────┴──────┐     ┌──────┴──────┐     ┌──────┴──────┐     ┌────────────┐  │
│  │ PostgreSQL  │     │  pgvector   │     │   Neo4j    │     │  LiteLLM   │   │
│  │  (Events,   │     │ (Embeddings)│     │  (Graph)   │     │  (Models)  │   │
│  │ Documents)  │     │             │     │            │     │            │   │
│  └─────────────┘     └─────────────┘     └────────────┘     └────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Purpose |
|-----------|---------|
| `MemoryLake` | Primary API for remember/recall/forget operations |
| `StorageCoordinator` | Orchestrates all storage backends |
| `HybridQueryEngine` | Combines vector, graph, and keyword search |
| `PipelineManager` | Manages Prefect ingestion flows |
| `ACLEnforcer` | Cross-layer permission enforcement |

### Storage Backends

| Backend | Technology | Purpose |
|---------|------------|---------|
| Relational | PostgreSQL | Documents, events, permissions, metadata |
| Vector | pgvector | Embeddings for semantic similarity search |
| Graph | Neo4j | Entity nodes and relationship edges |
| Event Store | PostgreSQL | Immutable event log for sourcing |

### Data Flow

1. **Ingestion** (Three-Phase Pipeline)
   - Phase 1: Stage documents, compute checksums, detect duplicates
   - Phase 2: Chunk text, then generate embeddings and extract entities concurrently
   - Phase 3 (optional): Cross-document entity unification and relationship inference

2. **Query** (Hybrid Search)
   - Execute vector, graph, and keyword searches in parallel
   - Apply Reciprocal Rank Fusion to combine results
   - Filter by ACL and temporal context

3. **Event Sourcing**
   - All changes recorded as immutable events
   - Enables temporal queries ("state as of date X")
   - Complete audit trail for compliance

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_DATABASE_URL` | PostgreSQL connection URL | Required |
| `KHORA_NEO4J_URL` | Neo4j connection URL | `bolt://localhost:7687` |
| `KHORA_NEO4J_USER` | Neo4j username | `neo4j` |
| `KHORA_NEO4J_PASSWORD` | Neo4j password | Required for Neo4j |
| `KHORA_DEBUG` | Enable debug mode | `false` |
| `KHORA_API_HOST` | API server host | `127.0.0.1` |
| `KHORA_API_PORT` | API server port | `8100` |
| `KHORA_AUTH_ENABLED` | Enable authentication | `true` |
| `OPENAI_API_KEY` | OpenAI API key (for embeddings) | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (for extraction) | - |

### LiteLLM Configuration

Khora uses LiteLLM for unified model access. Configure in `examples/config/litellm/`:

```yaml
# examples/config/litellm/openai.yaml
model: "gpt-4o-mini"
api_key_env: "OPENAI_API_KEY"
temperature: 0.7
max_tokens: 8192
embedding_model: "text-embedding-3-small"
```

```yaml
# examples/config/litellm/claude.yaml
model: "claude-sonnet-4-20250514"
api_key_env: "ANTHROPIC_API_KEY"
temperature: 0.7
max_tokens: 8192

# Router with fallbacks
model_list:
  - model_name: claude-sonnet-4
    litellm_params:
      model: claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: claude-sonnet-4
    litellm_params:
      model: claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

### Extraction Skills

Configure entity extraction in your code:

```python
from khora.extraction.skills import ExtractionSkill

skill = ExtractionSkill(
    name="custom_entities",
    description="Extract domain-specific entities",
    entity_types=["COMPANY", "PRODUCT", "TECHNOLOGY"],
    relationship_types=["DEVELOPS", "COMPETES_WITH", "USES"],
)

await lake.remember(content, skill_name="custom_entities")
```

---

## Project Structure

```
khora/
├── src/khora/
│   ├── __init__.py              # Package exports
│   ├── memory_lake.py           # Primary MemoryLake class
│   ├── api/                     # FastAPI application
│   │   ├── app.py               # App factory with lifespan
│   │   ├── deps.py              # Dependency injection
│   │   └── routes/              # API endpoints
│   │       ├── memory.py        # Remember/recall/forget
│   │       ├── namespaces.py    # Multi-tenancy management
│   │       ├── sync.py          # Ingestion pipelines
│   │       └── status.py        # Health checks
│   ├── acl/                     # Access control
│   │   ├── checker.py           # Permission checking
│   │   └── enforcer.py          # Cross-layer enforcement
│   ├── chat/                    # Conversational context
│   │   ├── engine.py            # Chat engine
│   │   ├── history.py           # Conversation history
│   │   ├── persona.py           # Persona management
│   │   └── prompt.py            # Prompt construction
│   ├── cli/                     # Command-line interface
│   ├── config/                  # Configuration
│   │   ├── schema.py            # Pydantic settings
│   │   ├── llm.py               # LiteLLM configuration
│   │   └── resolver.py          # Hierarchical config
│   ├── core/models/             # Domain models
│   │   ├── document.py          # Document, Chunk
│   │   ├── entity.py            # Entity, Relationship
│   │   ├── event.py             # MemoryEvent (sourcing)
│   │   └── tenancy.py           # Org, Workspace, Namespace
│   ├── db/                      # Database layer
│   │   ├── models.py            # SQLAlchemy ORM
│   │   └── session.py           # DatabaseManager + async sessions
│   ├── extraction/              # Content processing
│   │   ├── chunkers/            # Text chunking strategies
│   │   ├── embedders/           # Embedding generation
│   │   ├── extractors/          # Entity extraction
│   │   └── skills/              # Extraction configurations
│   ├── pipelines/               # Prefect workflows
│   │   ├── flows/               # Ingestion and sync flows
│   │   ├── tasks/               # Individual pipeline tasks
│   │   ├── manager.py           # Pipeline orchestration
│   │   └── registry.py          # Pipeline registration
│   ├── query/                   # Search engine
│   │   ├── engine.py            # HybridQueryEngine
│   │   ├── fusion.py            # Reciprocal Rank Fusion
│   │   └── temporal.py          # Time-based queries
│   └── storage/                 # Storage backends
│       ├── backends/            # PostgreSQL, pgvector, Neo4j
│       ├── coordinator.py       # Backend orchestration + TransactionContext
│       ├── event_store.py       # Event sourcing
│       └── factory.py           # Storage initialization + shared pools
├── tests/                       # Test suite
├── alembic/                     # Database migrations
├── examples/config/             # Example configurations
├── compose.yaml                 # Development databases
└── pyproject.toml               # Project configuration
```

---

## Development

### Commands

```bash
# Start development server
uv run khora serve --reload --no-auth

# Run tests with coverage
make test

# Format code
make format

# Run linting
make lint

# Run all pre-commit hooks
make prek

# Start development databases
make dev

# Stop development databases
make down
```

### Database Migrations

```bash
# Run all migrations
uv run alembic upgrade head

# Create a new migration
uv run alembic revision --autogenerate -m "Add new table"

# Rollback one migration
uv run alembic downgrade -1
```

### Testing

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/unit/test_api.py -v

# Run with markers
uv run pytest -m unit        # Unit tests only
uv run pytest -m integration # Integration tests
uv run pytest -m e2e         # End-to-end tests
```

---

## API Reference

### MemoryLake Class

#### Constructor

```python
class MemoryLake:
    def __init__(
        self,
        database_url: str | KhoraConfig | None = None,
        *,
        graph_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """Initialize the Memory Lake.

        Args:
            database_url: PostgreSQL URL, KhoraConfig, or None (reads from env)
            graph_url: Optional Neo4j URL (bolt://user:pass@host:port)
            embedding_model: Embedding model to use
        """
```

#### Core Methods

```python
    async def remember(
        self,
        content: str,
        *,
        namespace: str | UUID | None = None,
        title: str = "",
        source: str = "",
        metadata: dict = {},
        skill_name: str = "general_entities",
    ) -> RememberResult:
        """Store content in the memory lake."""

    async def remember_batch(
        self,
        documents: list[dict],
        *,
        namespace: str | UUID | None = None,
        skill_name: str = "general_entities",
        max_concurrent: int = 5,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Store multiple documents with automatic optimization."""

    async def recall(
        self,
        query: str,
        *,
        namespace: str | UUID | None = None,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        agentic: bool = False,
        raw: bool = False,  # Skip all LLM features
    ) -> RecallResult:
        """Recall memories relevant to a query."""

    async def forget(
        self,
        document_id: UUID,
        *,
        namespace: str | UUID | None = None,
    ) -> bool:
        """Remove a memory from the lake."""
```

#### Convenience Methods

```python
    async def ensure_namespace(
        self,
        name: str,
        *,
        description: str = "",
    ) -> UUID:
        """Get or create a namespace by name."""

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""

    async def list_documents(
        self,
        *,
        namespace: str | UUID | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """List documents in a namespace."""

    async def search_entities(
        self,
        query: str,
        *,
        namespace: str | UUID | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity."""

    async def stats(
        self,
        *,
        namespace: str | UUID | None = None,
    ) -> Stats:
        """Get document/chunk/entity/relationship counts."""

    async def list_entities(
        self,
        *,
        namespace: str | UUID | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List entities in a namespace."""

    async def find_related_entities(
        self,
        entity_id: UUID,
        *,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity."""
```

### Data Classes

```python
@dataclass
class RememberResult:
    """Result of a remember() operation."""
    document_id: UUID
    namespace_id: UUID
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    metadata: dict[str, Any]

@dataclass
class RecallResult:
    """Result of a recall() operation."""
    query: str
    namespace_id: UUID
    chunks: list[tuple[Chunk, float]]
    entities: list[tuple[Entity, float]]
    context_text: str
    metadata: dict[str, Any]

@dataclass
class BatchResult:
    """Result of remember_batch() operation."""
    total: int        # Total documents submitted
    processed: int    # Successfully processed
    skipped: int      # Skipped (duplicates)
    failed: int       # Failed to process
    chunks: int       # Total chunks created
    entities: int     # Total entities extracted
    relationships: int # Total relationships created

@dataclass
class Stats:
    """Namespace statistics from stats()."""
    documents: int
    chunks: int
    entities: int
    relationships: int
```

### Search Modes

| Mode | Description |
|------|-------------|
| `VECTOR` | Semantic similarity search using embeddings |
| `GRAPH` | Entity and relationship traversal |
| `HYBRID` | Combined vector + graph + keyword with RRF fusion |
| `ALL` | All sources (vector, graph, keyword) |

### Entity Types

| Type | Description |
|------|-------------|
| `PERSON` | Individual people |
| `ORGANIZATION` | Companies, institutions |
| `LOCATION` | Places, addresses |
| `CONCEPT` | Abstract ideas, theories |
| `EVENT` | Occurrences, incidents |
| `TECHNOLOGY` | Tools, platforms, languages |
| `PRODUCT` | Goods, services |
| `DOCUMENT` | Referenced documents |
| `OTHER` | Uncategorized entities |

### Changes in v0.3.1

| Feature | Description |
|---------|-------------|
| MMR diversity | Enabled by default — Rust-accelerated diversity selection prevents same-document dominance |
| Pre-normalized embeddings | Embeddings L2-normalized at ingest; scoring uses dot product (~3x faster) |
| Entity dedup constraint | `UNIQUE(namespace_id, name, entity_type)` with automatic dedup migration |
| Adaptive top-k | "Very focused" tier (complexity < 0.3 → 3 chunks) for precise single-entity queries |
| Slack extraction skill | Built-in `slack.yaml` with DM recipient extraction and MESSAGED relationships |
| Two-pass extraction | Triggers when entity-to-relationship ratio is low, not just `< 2` relationships |
| Temporal indexes | Neo4j relationship temporal indexes + PostgreSQL partial indexes on valid_from/valid_until |
| HNSW tuning | m=24, ef_construction=128 for improved vector recall |

### Changes in v0.3.0

| API | Status |
|-----|--------|
| `lake.storage` | Stable public API (no longer deprecated) |
| `lake.query_engine` | **Removed** — use `lake.recall(raw=True)` for unprocessed search |
| `remember_batch_legacy()` | **Removed** — use `remember_batch()` |
| `TransactionContext` | New — atomic multi-backend operations via `coordinator.transaction()` |
| `khora[nlp]` extra | New — install for spaCy-powered sentence splitting |
| Shared connection pools | New — backends sharing the same database URL reuse one engine pool |

---

## License

Copyright (c) 2024-2026 Deyta. All rights reserved.
