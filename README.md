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
- **Hybrid Search**: Vector + graph + keyword search with Reciprocal Rank Fusion
- **Multi-Tenancy**: Shared mode with ACLs or complete tenant isolation
- **Event Sourcing**: Immutable event log for temporal queries and audit trails
- **LiteLLM Integration**: Unified access to OpenAI, Anthropic, Google, and other providers
- **Prefect Pipelines**: Orchestrated ingestion with checksum-based change detection
- **Semantic Extraction**: LLM-powered entity and relationship extraction

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
from khora import MemoryLake

async def main():
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
            mode="hybrid",  # vector + graph + keyword
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
    # Create organizational hierarchy
    org = await lake.storage.create_organization(
        Organization(name="Acme Corp", slug="acme")
    )
    workspace = await lake.storage.create_workspace(
        Workspace(organization_id=org.id, name="Research", slug="research")
    )
    namespace = await lake.storage.create_namespace(
        MemoryNamespace(workspace_id=workspace.id, name="Physics", slug="physics")
    )

    # Store memories in specific namespace
    await lake.remember(
        "Important research findings...",
        namespace=namespace.id,
    )

    # Query within namespace (isolated from other namespaces)
    results = await lake.recall("findings", namespace=namespace.id)
```

### As a Service

```bash
# Start the API server
uv run khora serve --reload

# Or with Docker
docker compose up
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
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MemoryLake API                                  │
│                         (Library + FastAPI Service)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │    Query     │   │  Pipelines   │   │     ACL      │   │   Config     │ │
│  │   Engine     │   │  (Prefect)   │   │   Enforcer   │   │   Resolver   │ │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘ │
│         │                  │                  │                  │          │
├─────────┴──────────────────┴──────────────────┴──────────────────┴──────────┤
│                          Storage Coordinator                                 │
├─────────┬───────────────────┬───────────────────┬───────────────────────────┤
│         │                   │                   │                            │
│  ┌──────┴──────┐     ┌──────┴──────┐     ┌──────┴──────┐     ┌────────────┐ │
│  │ PostgreSQL  │     │  pgvector   │     │   Neo4j    │     │  LiteLLM   │ │
│  │  (Events,   │     │ (Embeddings)│     │  (Graph)   │     │  (Models)  │ │
│  │ Documents)  │     │             │     │            │     │            │ │
│  └─────────────┘     └─────────────┘     └────────────┘     └────────────┘ │
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

1. **Ingestion** (Two-Phase Pipeline)
   - Phase 1: Stage documents, compute checksums, detect changes
   - Phase 2: Chunk text, generate embeddings, extract entities

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
│   │   └── session.py           # Async session management
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
│       ├── coordinator.py       # Backend orchestration
│       ├── event_store.py       # Event sourcing
│       └── factory.py           # Storage initialization
├── tests/                       # Test suite
├── alembic/                     # Database migrations
├── examples/config/             # Example configurations
├── docker-compose.yml           # Development services
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

```python
class MemoryLake:
    async def remember(
        self,
        content: str,
        *,
        namespace: UUID | None = None,
        title: str = "",
        source: str = "",
        metadata: dict = {},
        skill_name: str = "general_entities",
    ) -> RememberResult:
        """Store content in the memory lake."""

    async def recall(
        self,
        query: str,
        *,
        namespace: UUID | None = None,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.5,
    ) -> RecallResult:
        """Recall memories relevant to a query."""

    async def forget(
        self,
        document_id: UUID,
        *,
        namespace: UUID | None = None,
    ) -> bool:
        """Remove a memory from the lake."""

    async def list_entities(
        self,
        *,
        namespace: UUID | None = None,
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

### Search Modes

| Mode | Description |
|------|-------------|
| `VECTOR` | Semantic similarity search using embeddings |
| `GRAPH` | Entity and relationship traversal |
| `KEYWORD` | Full-text keyword search |
| `HYBRID` | Combined search with RRF fusion |
| `ALL` | Returns results from all sources separately |

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

---

## License

Copyright (c) 2024-2025 Deyta. All rights reserved.
