# Roadmap

Khora is under active development. This document outlines where we're headed - from immediate improvements to longer-term research directions.

## Near-Term: Performance & Polish

These improvements are focused on making Khora faster and more reliable for production use.

### Query Performance

| Item | Why It Matters |
|------|----------------|
| **HNSW Index Support** | Done. Migration 005 rebuilds the vector index as HNSW with `ef_construction=128`. See [Performance Optimization](architecture/performance-optimization.md). |
| **Coherence Scoring** | Done (v0.3.5). Bigram coherence scoring for confounder rejection in VectorCypher's RRF fusion (`coherence_weight=0.1`). |
| **Streaming Pipeline** | Done (v0.3.2). `streaming_pipeline=True` default in `VectorCypherConfig`. |
| **Incremental Index Updates** | Currently, adding new vectors requires rebuilding indexes. Incremental updates would make real-time ingestion practical. |

### Ingestion Performance

| Item | Status | Why It Matters |
|------|--------|----------------|
| **Smart Entity Resolution** | Done | Two-phase architecture with token-blocked matching. Replaces O(n^2) per-document entity resolution with O(1) per-doc dedup + single O(n*k) post-ingestion pass. Batch storage operations reduce N+1 queries. See [Performance Optimization](architecture/performance-optimization.md). |
| **Staged Batch Pipeline** | Done | Documents processed through stages (chunk → embed+extract → store) as batches rather than individually. Better LLM utilization, batch DB writes, cross-document dedup on full set. |
| **Bulk Mode** | Done | `bulk_mode=True` defers HNSW indexes and Neo4j constraints for faster initial data loading. See [Storage Backends](architecture/storage-backends.md). |
| **Prefix Caching Optimization** | Done | Extraction prompts restructured for LLM prefix caching — static instructions first, variable content last. Reduces per-call latency with OpenAI and Anthropic. |
| **Async Pipeline Fixes** | Done | Blocking sync calls (spaCy chunking, SHA-256, JSON parsing) fixed with `asyncio.to_thread()`. |
| **Streaming Ingestion** | Planned | Process documents as they arrive rather than waiting for batches. Critical for real-time applications. |
| **Distributed Workers** | Planned | Scale extraction across multiple machines. Currently limited by single-process throughput. |
| **GPU-Accelerated Embedding** | Planned | Local embedding models can be 10-100x faster on GPU. Important for cost-sensitive deployments. |

### Storage Efficiency

| Item | Why It Matters |
|------|----------------|
| **Document-Level Dedup** | Done. SHA-256 checksum deduplication skips already-ingested documents. Chunk-level dedup (identical text across documents) is not yet implemented. |
| **SurrealDB Unified Backend** | Done (Phase 1). SurrealDB serves as relational + vector + graph backend in a single database. See [Storage Backends](architecture/storage-backends.md). |
| **Token-Aware Embedding Truncation** | Done. Embedder truncates at sentence boundaries before API call, avoiding silent truncation or errors. |
| **Vector Quantization** | 1536-dimension float32 vectors are 6KB each. Quantization can reduce this to 768 bytes with minimal quality loss. pgvector's `halfvec` type is supported for float16 storage. |
| **Cold Storage Tiering** | Older, rarely-accessed data could move to cheaper storage while staying queryable. |

## Medium-Term: Features & Capabilities

These additions would significantly expand what you can do with Khora.

### Smarter Queries

| Item | Why It Matters |
|------|----------------|
| **Conversational Context** | Done. Chat module at `src/khora/chat/` provides multi-turn conversations where follow-up questions understand context. |
| **Result Explanations** | Explain *why* a result matched - which entities linked, which keywords hit, how the graph was traversed. |
| **Faceted Search** | Filter results by entity type, source document, date range, confidence score. Essential for exploratory search. |

### Better Extraction

| Item | Why It Matters |
|------|----------------|
| **Multi-Modal Support** | Extract knowledge from images (diagrams, screenshots), PDFs (preserving layout), even video transcripts. |
| **Domain-Specific Models** | Fine-tuned extraction models for specific domains (legal, medical, technical) would dramatically improve accuracy. |
| **Coreference Resolution** | "Einstein... he... the physicist" should all resolve to the same entity. Current extraction misses these connections. |

### Knowledge Graph Features

| Item | Why It Matters |
|------|----------------|
| **Graph Visualization** | An interactive explorer for browsing entities and relationships. Hard to understand a knowledge graph without seeing it. |
| **Community Detection** | Automatically identify clusters of related entities. "These 15 people form a research group." |
| **Temporal Views** | See how the graph looked at a specific point in time. Replay knowledge evolution. |

### Multi-Tenancy & Operations

| Item | Why It Matters |
|------|----------------|
| **Cross-Namespace Search** | Query across multiple namespaces with proper access control. Currently each namespace is isolated. |
| **Usage Analytics** | Track queries, storage, and costs per tenant. Essential for billing and capacity planning. |
| **Namespace Templates** | Pre-configured setups for common use cases (customer support, research, documentation). |

## Integrations

Khora becomes more valuable when it connects to where your data lives.

### Data Sources

| Source | Notes |
|--------|-------|
| **Slack** | Conversations contain enormous institutional knowledge. Message threading requires special handling. |
| **Notion** | Hierarchical pages with rich formatting. Need to preserve structure while extracting content. |
| **GitHub** | Code, issues, PRs, discussions. Technical knowledge scattered across repositories. |
| **Google Drive** | Documents, spreadsheets, presentations. Auth is the hard part. |
| **Confluence** | Enterprise wikis. Often the canonical source for company knowledge. |
| **Linear** | Issue tracking with rich metadata. Project context that's often missing from docs. |

### LLM Ecosystem

| Integration | Notes |
|-------------|-------|
| **Local LLMs (Ollama, vLLM)** | Run extraction on your own hardware for cost or privacy. Quality varies by model. |
| **Model Router** | Automatically pick the right model for each task - cheap for embedding, capable for extraction. |
| **Fine-Tuned Models** | Use your own extraction models trained on domain-specific data. |

### Observability

| Integration | Notes |
|-------------|-------|
| **OpenTelemetry / Logfire** | Done. Logfire instrumentation (OTEL-based) added for LLM extraction calls, entity dedup, skeleton build, and ingestion pipeline. Optional: `pip install khora[logfire]`. `@trace` decorator and `trace_span()` context manager for span creation. Zero overhead when Logfire is not installed. |
| **Prometheus Metrics** | Query latency, ingestion throughput, queue depths. Standard monitoring. |
| **Query Analytics** | What are users searching for? Which queries return empty? Drives improvement priorities. |

### Deployment

| Option | Notes |
|--------|-------|
| **Kubernetes Helm Charts** | Production-ready K8s deployment with proper resource limits, health checks, scaling. |
| **Terraform Modules** | Infrastructure as code for AWS/GCP/Azure. Reproducible deployments. |
| **Managed Service** | Hosted Khora - just an API endpoint, we handle the infrastructure. Longer-term goal. |

## Developer Experience

Making Khora easier to use and integrate.

### API Evolution

| Item | Notes |
|------|-------|
| **GraphQL API** | Some use cases fit better with GraphQL than REST. Especially graph exploration. |
| **Webhooks** | Get notified when documents are processed, entities are created, etc. Enables reactive architectures. |
| **Generated SDKs** | Auto-generate TypeScript, Python, Go clients from OpenAPI spec. |

### Tooling

| Item | Notes |
|------|-------|
| **Admin Dashboard** | Web UI for managing namespaces, viewing stats, monitoring pipelines. |
| **Data Explorer** | Browse stored content, entities, relationships without writing code. |
| **Debug Tools** | See exactly what happened during a query or ingestion. Which chunker? What entities? |

## Research Directions

Longer-term ideas we're exploring. Less certain timelines.

### Advanced Retrieval

| Area | What We're Exploring |
|------|----------------------|
| **HyDE (Hypothetical Document Embedding)** | Done. Implemented in v0.3.1, disabled by default (`enable_hyde=False`). Generate a hypothetical answer, embed that, search for similar real content. Improves recall for question-style queries. |
| **Self-Query** | Let the LLM write its own filters based on the query. "Find documents about AI from last month" → automatic date filter. |
| **Contextual Compression** | Before returning chunks, compress them to just the relevant parts. Reduces noise in results. |

### Knowledge Graph Research

| Area | What We're Exploring |
|------|----------------------|
| **Ontology Learning** | Automatically discover entity types and relationship patterns from data. No manual schema definition. |
| **Link Prediction** | Predict likely relationships that aren't explicitly stated. "If Alice works with Bob at Acme, they probably know each other." |
| **Temporal Reasoning** | Answer questions about "what was true when" - not just current state. |

### Semantic Understanding

| Area | What We're Exploring |
|------|----------------------|
| **Claim Extraction** | Extract verifiable claims: "X said Y about Z on date D." Foundation for fact-checking. |
| **Contradiction Detection** | Find conflicting information across documents. Critical for accuracy. |
| **Multi-Document Summarization** | Synthesize information from many sources into coherent summaries. |

## Contributing

We welcome contributions! Particularly valuable areas:

1. **Source Integrations** - Connectors for data sources you use
2. **Extraction Skills** - Domain-specific expertise configs (legal, medical, etc.)
3. **Benchmarks** - Help us measure and improve performance
4. **Documentation** - Examples, guides, improvements

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Feedback

Have suggestions? Priorities are driven by community needs.

- Open an issue on GitHub
- Start a discussion
- Contact the maintainers

What would make Khora more useful for you?
