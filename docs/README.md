# Khora Documentation

Khora is a knowledge memory library. This directory contains everything beyond the quickstart in the top-level [README](../README.md).

## Getting started

- [Configuration](configuration.md) - `KHORA_*` environment variables, `KhoraConfig`, installation extras.
- [API reference](api-reference.md) - public `Khora` methods and result types.
- [Observability](observability.md) - OTel spans/metrics, `[otel]` and `[logfire]` paths, `configure_telemetry()`.
- [Migrations](migrations.md) - Alembic workflow for library users (PostgreSQL backends only).
- [Consumers](consumers.md) - how downstream packages consume khora's public API.

## Architecture

- [Overview](architecture/overview.md) - the three-backend model (PostgreSQL, pgvector, graph DB) and data flow.
- [Storage backends](architecture/storage-backends.md) - PostgreSQL, pgvector, Neo4j, SurrealDB, AGE, Memgraph, Neptune.
- [Multi-tenancy](architecture/multi-tenancy.md) - namespaces, isolation modes.
- [Event sourcing](architecture/event-sourcing.md) - the immutable audit log.
- [Rust acceleration](architecture/rust-acceleration.md) - optional `khora-accel` extensions.
- [Performance optimization](architecture/performance-optimization.md) - pool sizing, ef_search, batch strategies.

## Engines

Pluggable retrieval strategies that implement `MemoryEngineProtocol`.

- [Engine comparison](engines/engine-comparison.md) - pick the right engine.
- [VectorCypher](engines/vectorcypher-engine.md) - default, hybrid vector + Cypher + BM25.
- [Skeleton](engines/skeleton-engine.md) - lightweight, no graph DB required.
- [Skeleton indexing](engines/skeleton-indexing.md) - how Skeleton indexes content.
- [Chronicle](engines/chronicle-engine.md) - temporal-semantic memory.
- [Temporal model](engines/temporal-model.md) - time-as-first-class-citizen mechanics.
- [Hybrid search](engines/hybrid-search.md) - the shared hybrid retrieval primitive.

## Extraction

3-phase ingestion pipeline: stage → enrich → expand.

- [Overview](extraction/overview.md) - the big picture.
- [Ingestion pipeline](extraction/ingestion-pipeline.md) - flow, concurrency, failure handling.
- [Chunkers](extraction/chunkers.md) - fixed / semantic / recursive strategies.
- [Conversation chunking](extraction/conversation-chunking.md) - message-aware grouping.
- [Embedders](extraction/embedders.md) - vector generation via LiteLLM.
- [Extractors](extraction/extractors.md) - entity and relationship extraction.
- [Expertise system](extraction/expertise-system.md) - `ExpertiseConfig` for domain-specific extraction.
- [Semantic expansion](extraction/semantic-expansion.md) - entity unification and relationship inference.

## Query engine

- [Overview](query-engine/overview.md) - how `recall()` routes through the engine.
- [Search modes](query-engine/search-modes.md) - `vector`, `graph`, `hybrid`, `all`.
- [Fusion](query-engine/fusion.md) - Reciprocal Rank Fusion and weighting.
- [Recall semantics](query-engine/recall-semantics.md) - score vs order contract, `min_similarity` floors, abstention signals.
- [Query understanding](query-engine/query-understanding.md) - HyDE, intent detection.
- [Agentic search](query-engine/agentic-search.md) - multi-step retrieval.
- [Temporal queries](query-engine/temporal-queries.md) - relative-date SQL pushdown.
- [Retrieval tuning](query-engine/retrieval-tuning.md) - practical knobs.

## Data models

- [Overview](data-models/overview.md) - documents, chunks, entities, events.
- [Documents and chunks](data-models/documents-chunks.md).
- [Knowledge graph](data-models/knowledge-graph.md) - entities and relationships.
- [Events](data-models/events.md) - the append-only event log.

## Hooks

- [Semantic hooks](hooks/semantic-hooks.md) - subscribe to extraction events with 3-level semantic filtering.

## Dream phase

Background knowledge-consolidation cycle that runs between recalls to deduplicate entities, resolve contradictions, summarize communities, and compact facts.

- [Dream phase](dream-phase.md) - architecture, ops, orchestration, and configuration.

## Integrations

Adapters for agentic frameworks. Install the matching extra, then import from `khora.integrations.<name>`.

- [CrewAI](integrations/crewai.md) - `KhoraMemory` for CrewAI agents (`khora[crewai]`).
- [LangGraph](integrations/langgraph.md) - `KhoraStore` semantic long-term memory for LangGraph (`khora[langgraph]`).
- [Google ADK](integrations/google_adk.md) - `KhoraMemoryService` for Google Agent Development Kit (`khora[google-adk]`).
- [OpenAI Agents SDK](integrations/openai_agents.md) - `KhoraSession`, `khora_recall_tool`, `KhoraMemoryHooks` (`khora[openai-agents]`).
- [LlamaIndex](integrations/llamaindex.md) - `KhoraRetriever`, `KhoraMemoryBlock`, `KhoraChatStore` (`khora[llamaindex]`).
- [Hermes](integrations/hermes.md) - event-bus adapter for Hermes-compatible message brokers (`khora[hermes]`).

## Process

- [Release process](RELEASE.md) - how versions are tagged and published.
- [Roadmap](roadmap.md) - what's next.
