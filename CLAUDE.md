# Khora - Development Guide

Khora is Deyta's Memory Lake - a system combining knowledge graph (Neo4j), vector database (pgvector), and relational database (PostgreSQL) for unified knowledge storage and retrieval.

## Quick Reference

### Commands
```bash
# Development
uv run khora serve --reload      # Start dev server with hot-reload
uv run khora serve --no-auth     # Start without authentication
make test                         # Run tests with coverage
make prek                         # Run pre-commit hooks
make format                       # Format code (black, isort, ruff)
make lint                         # Check linting
make dev                          # Start development databases
make down                         # Stop development databases

# Database
uv run alembic upgrade head       # Run migrations
uv run alembic revision --autogenerate -m "description"  # Create migration
```

### Project Structure
```
src/khora/
├── __init__.py              # Package exports (MemoryLake, SearchMode)
├── memory_lake.py           # Primary MemoryLake class
├── api/                     # FastAPI application
│   ├── app.py               # App factory with lifespan
│   ├── deps.py              # Dependency injection
│   └── routes/              # API endpoints
│       ├── memory.py        # Remember/recall/forget
│       ├── namespaces.py    # Multi-tenancy management
│       ├── sync.py          # Ingestion pipelines
│       └── status.py        # Health checks
├── acl/                     # Access control
│   ├── checker.py           # Permission checking with inheritance
│   └── enforcer.py          # Cross-layer enforcement
├── cli/                     # Click CLI commands
├── config/                  # Configuration
│   ├── schema.py            # Pydantic settings
│   ├── llm.py               # LiteLLM configuration
│   └── resolver.py          # Hierarchical config resolution
├── core/models/             # Domain models
│   ├── document.py          # Document, Chunk
│   ├── entity.py            # Entity, Relationship, Episode
│   ├── event.py             # MemoryEvent (event sourcing)
│   └── tenancy.py           # Organization, Workspace, Namespace
├── db/                      # Database layer
│   ├── models.py            # SQLAlchemy ORM models
│   └── session.py           # Async session management
├── extraction/              # Content processing
│   ├── chunkers/            # Fixed, semantic, recursive chunkers
│   ├── embedders/           # LiteLLM-based embeddings
│   ├── extractors/          # LLM entity extraction
│   └── skills/              # Extraction skill registry
├── pipelines/               # Prefect workflows
│   ├── flows/               # Ingestion and sync flows
│   ├── tasks/               # Chunk, embed, extract tasks
│   ├── manager.py           # Pipeline orchestration
│   └── registry.py          # Pipeline registration
├── query/                   # Search engine
│   ├── engine.py            # HybridQueryEngine
│   ├── fusion.py            # Reciprocal Rank Fusion
│   └── temporal.py          # Time-based queries
├── storage/                 # Storage backends
│   ├── backends/            # PostgreSQL, pgvector, Neo4j
│   ├── coordinator.py       # Backend orchestration
│   ├── event_store.py       # Event sourcing
│   └── factory.py           # Storage initialization
└── logging_config.py        # Loguru setup
```

## Architecture

### Core Components
- **MemoryLake**: Primary API for remember/recall/forget operations
- **StorageCoordinator**: Orchestrates PostgreSQL, pgvector, and Neo4j
- **HybridQueryEngine**: Combines vector, graph, and keyword search with RRF
- **PipelineManager**: Manages Prefect ingestion flows
- **ACLEnforcer**: Cross-layer permission enforcement

### Storage Backends
- **PostgreSQL**: Documents, events, permissions, metadata
- **pgvector**: Embeddings for semantic similarity search
- **Neo4j**: Entity nodes and relationship edges

### Data Flow
1. **Ingestion**: Two-phase pipeline (staging + enrichment)
2. **Query**: Parallel search with Reciprocal Rank Fusion
3. **Event Sourcing**: Immutable event log for temporal queries

## Code Style

- Python 3.13+
- Line length: 120 characters
- Black for formatting
- isort with black profile
- ruff for linting
- Type hints throughout

## Testing

- pytest with pytest-asyncio
- Coverage minimum: 30%
- Markers: @pytest.mark.unit, @pytest.mark.integration, @pytest.mark.e2e
- Fixtures in tests/conftest.py

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| KHORA_DATABASE_URL | PostgreSQL/pgvector connection URL | Required |
| KHORA_NEO4J_URL | Neo4j connection URL (bolt://user:pass@host:port) | - |
| KHORA_DEBUG | Enable debug mode | false |
| KHORA_API_HOST | API server host | 127.0.0.1 |
| KHORA_API_PORT | API server port | 8100 |
| KHORA_AUTH_ENABLED | Enable authentication | true |
| OPENAI_API_KEY | OpenAI API key (for embeddings) | - |
| ANTHROPIC_API_KEY | Anthropic API key (for extraction) | - |

**URL formats:**
- PostgreSQL: `postgresql://user:password@host:port/database`
- Neo4j: `bolt://user:password@host:port` or `bolt://user:password@host:port/database`

**Note:** Programmatic configuration takes priority over environment variables.

## Library Usage

```python
from khora import MemoryLake, SearchMode

# Simple usage - uses KHORA_DATABASE_URL and KHORA_NEO4J_URL env vars
async with MemoryLake() as lake:
    # Store a memory
    result = await lake.remember("Content to store", title="Title")

    # Recall memories
    memories = await lake.recall("query", mode=SearchMode.HYBRID)

    # Forget a memory
    await lake.forget(result.document_id)

# Programmatic configuration (overrides env vars)
from khora.config import KhoraConfig
from khora.storage import StorageConfig

config = KhoraConfig(
    database_url="postgresql://user:pass@localhost:5432/mydb",
    neo4j_url="bolt://localhost:7687",
)
async with MemoryLake(config=config) as lake:
    ...

# Or override storage directly
storage_config = StorageConfig(
    postgresql_url="postgresql://...",
    neo4j_url="bolt://...",
    neo4j_user="neo4j",
    neo4j_password="secret",
)
async with MemoryLake(storage_config=storage_config) as lake:
    ...
```

## API Endpoints

### Memory Operations
- `POST /memory/remember` - Store content
- `POST /memory/recall` - Search memories
- `DELETE /memory/forget` - Remove memory
- `GET /memory/documents/{id}` - Get document
- `GET /memory/entities` - List entities
- `GET /memory/entities/{id}/related` - Related entities

### Namespace Management
- `POST /namespaces/organizations` - Create organization
- `POST /namespaces/workspaces` - Create workspace
- `POST /namespaces/` - Create namespace

### Sync & Pipelines
- `POST /sync/ingest` - Ingest documents
- `POST /sync/source` - Sync from source
- `GET /sync/pipelines` - List pipelines

### Health Checks
- `GET /status` - Service status
- `GET /health` - Health check
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
