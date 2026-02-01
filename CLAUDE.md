# Khora - Development Guide

Khora is Deyta's Memory Lake - a system combining knowledge graphs, vector database (pgvector), and relational database (PostgreSQL) for unified knowledge storage and retrieval. Supports multiple graph backends (Neo4j, Kuzu, Memgraph, ArcadeDB) and vector backends (pgvector, ArcadeDB).

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
├── __init__.py                  # Package exports (MemoryLake, SearchMode)
├── __main__.py                  # Entry point
├── memory_lake.py               # Primary MemoryLake class (remember/recall/forget/remember_batch)
├── logging_config.py            # Loguru setup
├── api/                         # FastAPI application
│   ├── app.py                   # App factory with lifespan
│   ├── deps.py                  # Dependency injection
│   └── routes/
│       ├── memory.py            # Remember/recall/forget + entity CRUD
│       ├── namespaces.py        # Org/workspace/namespace management
│       ├── sync.py              # Ingestion pipelines + sync checkpoints
│       └── status.py            # Health checks (status, health, ready, live)
├── acl/                         # Access control
│   ├── checker.py               # Permission checking with inheritance
│   └── enforcer.py              # Cross-layer enforcement
├── chat/                        # Conversational interface
│   ├── engine.py                # ChatEngine (persona-driven responses)
│   ├── history.py               # HistoryManager (turn management + compression)
│   ├── persona.py               # PersonaConfig (behavior, style, chat settings)
│   └── prompt.py                # PromptGenerator (system prompt construction)
├── cli/
│   ├── __init__.py              # Click CLI group
│   └── server.py                # `khora serve` command
├── config/
│   ├── schema.py                # KhoraConfig + StorageSettings + GraphConfig union + QuerySettings
│   ├── llm.py                   # LiteLLM wrapper (acompletion, aembedding, router)
│   └── resolver.py              # Hierarchical config resolution
├── core/models/                 # Domain models
│   ├── document.py              # Document, Chunk, DocumentMetadata, DocumentStatus
│   ├── entity.py                # Entity, Relationship, Episode, EntityType
│   ├── event.py                 # MemoryEvent (event sourcing)
│   ├── schemas.py               # Extensible attribute schemas (Person, Organization, etc.)
│   ├── source.py                # Source taxonomy (SourceTool, aliases, registry)
│   └── tenancy.py               # Organization, Workspace, MemoryNamespace
├── db/
│   ├── models.py                # SQLAlchemy ORM models
│   └── session.py               # Async session management (asyncpg)
├── extraction/                  # Content processing
│   ├── entity_resolution.py     # Entity deduplication and resolution
│   ├── chunkers/
│   │   ├── base.py              # Chunker base class
│   │   ├── fixed.py             # Fixed-size token chunking
│   │   ├── semantic.py          # Embedding-based semantic chunking
│   │   ├── recursive.py         # Recursive text splitting
│   │   └── conversation.py      # Conversation-aware chunking (time gaps, message groups)
│   ├── embedders/
│   │   ├── base.py              # Embedder base class
│   │   └── litellm.py           # LiteLLM embedding (batched, with telemetry)
│   ├── extractors/
│   │   ├── base.py              # Extractor base class
│   │   └── llm.py               # LLM entity extraction (single + multi-batch)
│   ├── expansion/               # Knowledge graph enrichment
│   │   ├── expander.py          # SemanticExpander (orchestrates expansion)
│   │   ├── cross_tool_unifier.py # Cross-tool entity unification
│   │   ├── relationship_inferrer.py # Infer implicit relationships
│   │   └── rule_engine.py       # Configurable rule-based expansion
│   └── skills/                  # Extraction skill system
│       ├── base.py              # ExpertiseConfig, EntityTypeConfig, RelationshipTypeConfig
│       ├── registry.py          # Skill registry (get/register skills)
│       ├── loader.py            # YAML skill loader
│       └── composer.py          # Skill composition
├── pipelines/                   # Processing pipelines
│   ├── manager.py               # PipelineManager (ingestion orchestration)
│   ├── registry.py              # Pipeline registration
│   ├── incremental.py           # Incremental sync support
│   ├── flows/
│   │   ├── ingest.py            # Document ingestion flow (chunk → embed → extract → expand → store)
│   │   ├── expansion.py         # Post-extraction graph expansion flow
│   │   └── sync.py              # External source sync flow
│   └── tasks/
│       ├── chunk.py             # Chunking task
│       ├── embed.py             # Embedding task
│       └── extract.py           # Entity extraction task
├── query/                       # Search engine
│   ├── engine.py                # HybridQueryEngine (orchestrates all search)
│   ├── understanding.py         # LLM query understanding (entities, temporal, expansion)
│   ├── linking.py               # Entity linking (exact, fuzzy, embedding match)
│   ├── keyword.py               # BM25/fulltext keyword search
│   ├── fusion.py                # Reciprocal Rank Fusion
│   ├── reranking.py             # Neural reranking (cross-encoder, LLM)
│   ├── hyde.py                  # Hypothetical Document Embeddings
│   ├── agentic.py               # Multi-step agentic search
│   ├── temporal.py              # Time-based query filters
│   ├── metrics.py               # SearchMetrics (per-query performance stats)
│   ├── cache.py                 # Query result caching
│   └── message_extract.py       # Message content extraction
├── storage/                     # Storage backends
│   ├── coordinator.py           # StorageCoordinator (backend orchestration)
│   ├── factory.py               # Storage initialization + backend selection
│   ├── event_store.py           # Event sourcing (immutable event log)
│   ├── expertise_store.py       # Expertise definition CRUD
│   ├── optimize.py              # Post-ingestion index optimization
│   └── backends/
│       ├── base.py              # GraphBackend + VectorBackend base classes
│       ├── mixins.py            # Shared backend mixins
│       ├── postgresql.py        # PostgreSQL (documents, events, tenancy, metadata)
│       ├── pgvector.py          # pgvector (embeddings, vector similarity search)
│       ├── neo4j.py             # Neo4j graph backend
│       ├── kuzu.py              # Kuzu embedded graph backend
│       ├── memgraph.py          # Memgraph graph backend
│       └── arcadedb.py          # ArcadeDB graph + vector backend
└── telemetry/                   # Internal telemetry
    ├── __init__.py              # init_telemetry/shutdown_telemetry/get_collector
    ├── config.py                # TelemetryConfig (from env)
    ├── models.py                # LLMEvent, StorageEvent, PipelineEvent
    ├── tables.py                # SQLAlchemy Core table definitions
    ├── session.py               # Separate async engine for telemetry DB
    ├── collector.py             # TelemetryCollector (async buffer + flush loop)
    ├── noop.py                  # NoOpCollector (zero-cost when disabled)
    └── instrument.py            # Decorators: @instrument_llm, @instrument_storage, pipeline_stage
```

## Architecture

### Core Components
- **MemoryLake**: Primary API for `remember()` / `recall()` / `forget()` / `remember_batch()` operations
- **StorageCoordinator**: Orchestrates PostgreSQL, pgvector, and the active graph backend
- **HybridQueryEngine**: Multi-stage query pipeline (understanding → linking → search → fusion → reranking)
- **ChatEngine**: Persona-driven conversational interface over MemoryLake
- **PipelineManager**: Manages ingestion and sync flows
- **SemanticExpander**: Post-extraction knowledge graph enrichment (relationship inference, cross-tool unification, rule engine)
- **ACLEnforcer**: Cross-layer permission enforcement with hierarchical inheritance

### Storage Backends

**Relational (always PostgreSQL):**
- Documents, events, permissions, tenancy hierarchy, sync checkpoints

**Vector (selectable):**
- **pgvector** (default): Embeddings and vector similarity search via PostgreSQL extension
- **ArcadeDB**: Vector storage via ArcadeDB's embedding support

**Graph (selectable via `storage.graph.backend`):**
- **Neo4j** (default): Client-server graph database (bolt:// protocol)
- **Kuzu**: Embedded graph database (local directory, no server needed)
- **Memgraph**: In-memory graph database (bolt:// protocol)
- **ArcadeDB**: Multi-model database (HTTP API, supports Cypher or Gremlin)

All graph backends implement a common `GraphBackend` interface: entity/relationship CRUD, neighborhood traversal, fulltext search.

### Multi-Tenancy Model

```
Organization (tenancy_mode: shared|isolated)
  └── Workspace
        └── MemoryNamespace (config_overrides, versioning)
```

Each namespace isolates documents, chunks, entities, and relationships. Namespaces support per-namespace configuration overrides and custom expertise definitions.

### Data Flow

**Ingestion pipeline** (`remember()` / `POST /sync/ingest`):
1. **Document creation** — checksum-based deduplication
2. **Chunking** — fixed, semantic, recursive, or conversation-aware
3. **Embedding** — batched LiteLLM embedding generation
4. **Entity extraction** — LLM-based extraction using configurable skills
5. **Expansion** — relationship inference, cross-tool unification, rule engine
6. **Storage** — chunks → pgvector, entities/relationships → graph backend, documents → PostgreSQL
7. **Index optimization** — post-ingestion index maintenance

**Query pipeline** (`recall()` / `POST /memory/recall`):
1. **Query understanding** — LLM analyzes query for entities, temporal refs, and generates expansions
2. **Entity linking** — matches extracted entity mentions to stored entities (exact, fuzzy, embedding)
3. **HyDE** (optional) — generates hypothetical documents for improved embedding search
4. **Parallel search** — vector similarity + graph traversal + keyword (BM25/fulltext)
5. **Reciprocal Rank Fusion** — weighted fusion of search results
6. **Reranking** — cross-encoder or LLM-based reranking of fused candidates
7. **Agentic search** (optional) — multi-step exploration with follow-up queries

**Event sourcing**: Immutable `MemoryEvent` log in PostgreSQL for temporal queries and audit.

### Extraction System

**Chunkers**: `fixed` (token count), `semantic` (embedding similarity boundaries), `recursive` (text splitting), `conversation` (time-gap and message-group aware).

**Skills**: YAML-configured extraction profiles defining entity types, relationship types, and extraction prompts. Skills are composable and stored per-namespace via `ExpertiseStore`. Default: `general_entities`.

**Entity resolution**: Deduplication of extracted entities across documents.

**Expansion** (`SemanticExpander`):
- `CrossToolUnifier`: Merges entities from different extraction tools/runs
- `RelationshipInferrer`: Infers implicit relationships from entity co-occurrence and attributes
- `RuleEngine`: Configurable rules for domain-specific graph enrichment

**Attribute schemas**: Pydantic-validated attribute schemas per entity type (Person, Organization, Location, etc.). Extensible via `register_attribute_schema()`.

**Source taxonomy**: Controlled vocabulary for source types (`SourceTool` enum + dynamic registry). Downstream projects register domain-specific tools via `register_source_type()`.

### Chat Engine

The `ChatEngine` provides conversational access to the memory lake:
- **PersonaConfig**: Defines behavior, response style, and chat parameters
- **HistoryManager**: Turn management with compression for long conversations
- **PromptGenerator**: Constructs system prompts from persona config + retrieved context
- Supports agentic search mode for deeper exploration during conversations

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
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`
- Fixtures in `tests/conftest.py`

## Environment Variables

### Core
| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_DATABASE_URL` | PostgreSQL/pgvector connection URL | Required |
| `KHORA_NEO4J_URL` | Neo4j connection URL (bolt://user:pass@host:port) | - |
| `KHORA_DEBUG` | Enable debug mode | `false` |
| `KHORA_ENVIRONMENT` | Environment: development, staging, production | `development` |
| `KHORA_API_HOST` | API server host | `127.0.0.1` |
| `KHORA_API_PORT` | API server port | `8000` |
| `KHORA_AUTH_ENABLED` | Enable authentication | `true` |

### LLM
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (for embeddings/LLM) | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (for extraction) | - |
| `KHORA_LLM__MODEL` | Primary LLM model | `gpt-4o-mini` |
| `KHORA_LLM__EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `KHORA_LLM__EMBEDDING_DIMENSION` | Embedding vector dimension | `1536` |

### Storage (new-style backend configs)
| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_STORAGE__GRAPH__BACKEND` | Graph backend: `neo4j`, `kuzu`, `memgraph`, `arcadedb` | `neo4j` |
| `KHORA_STORAGE__VECTOR__BACKEND` | Vector backend: `pgvector`, `arcadedb` | `pgvector` |

### Query Pipeline
| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_QUERY__DEFAULT_MODE` | Default search mode: `vector`, `graph`, `hybrid`, `all` | `hybrid` |
| `KHORA_QUERY__MIN_CHUNK_SIMILARITY` | Minimum chunk similarity threshold (0.0 = no filtering) | `0.05` |
| `KHORA_QUERY__MIN_ENTITY_SIMILARITY` | Minimum entity similarity threshold | `0.05` |
| `KHORA_QUERY__ENABLE_UNDERSTANDING` | Enable LLM query understanding | `true` |
| `KHORA_QUERY__ENABLE_ENTITY_LINKING` | Enable entity linking | `true` |
| `KHORA_QUERY__ENABLE_RERANKING` | Enable neural reranking | `true` |
| `KHORA_QUERY__RERANKING_METHOD` | Reranking method: `cross_encoder`, `llm` | `cross_encoder` |
| `KHORA_QUERY__ENABLE_KEYWORD_SEARCH` | Enable keyword search (runs in hybrid and all modes) | `true` |
| `KHORA_QUERY__ENABLE_HYDE` | Enable HyDE query expansion | `false` |
| `KHORA_QUERY__VECTOR_WEIGHT` | Weight for vector search in fusion | `0.5` |
| `KHORA_QUERY__GRAPH_WEIGHT` | Weight for graph search in fusion | `0.3` |
| `KHORA_QUERY__KEYWORD_WEIGHT` | Weight for keyword search in fusion | `0.2` |

### Pipelines
| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_PIPELINES__CHUNKING_STRATEGY` | Chunking strategy: `fixed`, `semantic`, `recursive` | `semantic` |
| `KHORA_PIPELINES__CHUNK_SIZE` | Target chunk size in tokens | `512` |
| `KHORA_PIPELINES__EXTRACT_ENTITIES` | Extract entities from documents | `true` |

### Telemetry
| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_TELEMETRY_DATABASE_URL` | PostgreSQL URL for telemetry DB (enables telemetry when set) | - |
| `KHORA_TELEMETRY_SERVICE_NAME` | Service name tag in telemetry events | `khora` |

**URL formats:**
- PostgreSQL: `postgresql://user:password@host:port/database`
- Neo4j: `bolt://user:password@host:port` or `bolt://user:password@host:port/database`
- Kuzu: Local directory path (e.g., `./kuzu_db`)
- Memgraph: `bolt://user:password@host:port`
- ArcadeDB: `http://user:password@host:port`

**Note:** Programmatic configuration takes priority over environment variables. Nested config uses `__` delimiter (e.g., `KHORA_QUERY__ENABLE_HYDE=true`).

## Library Usage

```python
from khora import MemoryLake, SearchMode

# Simple usage - uses KHORA_DATABASE_URL and KHORA_NEO4J_URL env vars
async with MemoryLake() as lake:
    # Store a memory
    result = await lake.remember("Content to store", title="Title")

    # Recall memories (hybrid search with query understanding, entity linking, reranking)
    memories = await lake.recall("query", mode=SearchMode.HYBRID)

    # Agentic recall (multi-step exploration with follow-up queries)
    memories = await lake.recall("complex query", agentic=True)

    # Batch ingestion
    results = await lake.remember_batch([
        {"content": "Doc 1", "title": "First"},
        {"content": "Doc 2", "title": "Second"},
    ], max_concurrent=5)

    # Forget a memory
    await lake.forget(result.document_id)

    # Entity operations
    entities = await lake.list_entities(entity_type="PERSON")
    related = await lake.find_related_entities(entity_id, max_depth=2)

# Programmatic configuration with multi-backend storage
from khora.config import KhoraConfig
from khora.config.schema import StorageSettings, KuzuConfig, PgVectorConfig

config = KhoraConfig(
    database_url="postgresql://user:pass@localhost:5432/mydb",
    storage=StorageSettings(
        graph=KuzuConfig(database_path="./my_kuzu_db"),
        vector=PgVectorConfig(url="postgresql://user:pass@localhost:5432/mydb"),
    ),
)
async with MemoryLake(config=config) as lake:
    ...

# Chat engine with persona
from khora.chat import ChatEngine
from khora.chat.persona import PersonaConfig

persona = PersonaConfig(...)
chat = ChatEngine(persona=persona, memory_lake=lake, agentic_search=True)
response = await chat.chat("What do you know about X?", namespace_id=ns_id)
```

## API Endpoints

### Memory Operations
- `POST /memory/remember` - Store content (with extraction skill selection)
- `POST /memory/recall` - Search memories (vector/graph/hybrid/all modes)
- `DELETE /memory/forget` - Remove a memory
- `GET /memory/documents/{id}` - Get document details
- `GET /memory/entities` - List entities (filter by type, namespace)
- `GET /memory/entities/{id}` - Get entity details with attributes
- `GET /memory/entities/{id}/related` - Get related entities (configurable depth)

### Namespace Management
- `POST /namespaces/organizations` - Create organization
- `GET /namespaces/organizations/{id}` - Get organization
- `POST /namespaces/workspaces` - Create workspace
- `GET /namespaces/workspaces/{id}` - Get workspace
- `GET /namespaces/organizations/{id}/workspaces` - List workspaces in org
- `POST /namespaces/` - Create namespace
- `GET /namespaces/{id}` - Get namespace
- `GET /namespaces/workspaces/{id}/namespaces` - List namespaces in workspace

### Sync & Pipelines
- `POST /sync/ingest` - Ingest documents (full pipeline)
- `POST /sync/source` - Sync from external source (incremental)
- `GET /sync/checkpoint/{namespace_id}/{source}` - Get sync checkpoint
- `PUT /sync/checkpoint/{namespace_id}/{source}` - Set sync checkpoint
- `GET /sync/pipelines` - List registered pipelines

### Health Checks
- `GET /status` - Service status with version
- `GET /health` - Health check
- `GET /health/ready` - Readiness probe (component checks)
- `GET /health/live` - Liveness probe

## Telemetry

The `khora.telemetry` module records LLM usage, storage operations, and pipeline performance to a **separate** PostgreSQL database. It is enabled by setting `KHORA_TELEMETRY_DATABASE_URL`.

### How it works

- **Disabled by default**: When the env var is unset, a zero-cost `NoOpCollector` is used — all record methods are no-ops.
- **Non-blocking**: Events are buffered in memory and flushed every 5 seconds (or 100 events) via a background `asyncio.Task`.
- **Separate DB**: Telemetry uses its own `AsyncEngine` and auto-creates tables on startup (no Alembic).
- **Tables**: `llm_events`, `storage_events`, `pipeline_events` in the telemetry database.

### Instrumenting new code

```python
# Record an LLM call
from khora.telemetry import get_collector
get_collector().record_llm_call(
    operation="my_operation",
    model="gpt-4o-mini",
    prompt_tokens=120,
    completion_tokens=350,
    total_tokens=470,
    latency_ms=812.3,
)

# Use the pipeline_stage context manager
from khora.telemetry.instrument import pipeline_stage
async with pipeline_stage("my_pipeline", "my_stage", run_id):
    await do_work()

# Use decorators
from khora.telemetry.instrument import instrument_llm, instrument_storage

@instrument_llm("my_llm_operation")
async def call_llm(): ...

@instrument_storage("postgresql", "my_storage_op")
async def store_data(): ...
```
