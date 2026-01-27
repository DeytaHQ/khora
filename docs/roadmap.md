# Roadmap

This document outlines future improvements and enhancements planned for Khora.

## Performance Improvements

### Query Performance

- [ ] **HNSW Index Support** - Replace IVFFlat with HNSW for better recall/latency tradeoff
- [ ] **Query Caching** - Cache frequent query embeddings and results
- [ ] **Incremental Indexing** - Update indexes without full rebuilds
- [ ] **Parallel Graph Queries** - Optimize Neo4j traversal for concurrent access

### Ingestion Performance

- [ ] **Streaming Ingestion** - Process documents as they arrive vs batch
- [ ] **Distributed Processing** - Scale ingestion across multiple workers
- [ ] **GPU Embedding** - Accelerate embedding generation
- [ ] **Incremental Entity Updates** - Update entities without full re-extraction

### Storage Optimization

- [ ] **Chunk Deduplication** - Deduplicate identical chunks across documents
- [ ] **Embedding Quantization** - Reduce storage with quantized vectors
- [ ] **Graph Partitioning** - Partition large graphs for better performance
- [ ] **Cold Storage Tiering** - Move old data to cheaper storage

## Feature Enhancements

### Query Capabilities

- [ ] **Conversational Memory** - Multi-turn query context
- [ ] **Query Explanation** - Explain why results matched
- [ ] **Faceted Search** - Filter by entity type, source, date
- [ ] **Saved Searches** - Store and replay query configurations

### Extraction Improvements

- [ ] **Multi-Modal Extraction** - Extract from images, PDFs, videos
- [ ] **Relation Extraction Models** - Fine-tuned models for specific domains
- [ ] **Active Learning** - Improve extraction with user feedback
- [ ] **Coreference Resolution** - Better entity linking across mentions

### Graph Enhancements

- [ ] **Graph Visualization** - Interactive knowledge graph explorer
- [ ] **Path Explanations** - Explain relationship paths
- [ ] **Community Detection** - Identify entity clusters
- [ ] **Temporal Graph Views** - Visualize graph state over time

### Multi-Tenancy

- [ ] **Cross-Namespace Queries** - Search across multiple namespaces
- [ ] **Namespace Templates** - Pre-configured namespace setups
- [ ] **Usage Analytics** - Per-tenant usage tracking
- [ ] **Rate Limiting** - Per-tenant query limits

## Integration Opportunities

### Data Sources

- [ ] **Slack Integration** - Native Slack message ingestion
- [ ] **Notion Integration** - Native Notion page sync
- [ ] **Google Drive** - Document sync from Drive
- [ ] **GitHub** - Repository and issue ingestion
- [ ] **Linear** - Issue and project tracking sync
- [ ] **Confluence** - Wiki page ingestion

### LLM Providers

- [ ] **Local LLM Support** - Ollama, vLLM integration
- [ ] **Model Router** - Automatic model selection based on task
- [ ] **Cost Optimization** - Route to cheaper models when possible
- [ ] **Fine-Tuned Models** - Support for custom extraction models

### Observability

- [ ] **OpenTelemetry** - Distributed tracing
- [ ] **Prometheus Metrics** - Performance metrics export
- [ ] **Query Analytics** - Query pattern analysis
- [ ] **Pipeline Monitoring** - Real-time ingestion monitoring

### Deployment

- [ ] **Kubernetes Helm Charts** - Production-ready K8s deployment
- [ ] **Docker Compose Production** - Optimized compose for production
- [ ] **Terraform Modules** - Infrastructure as code
- [ ] **Managed Service** - Hosted Khora offering

## Developer Experience

### API Improvements

- [ ] **GraphQL API** - Alternative to REST
- [ ] **Webhook Notifications** - Event-based notifications
- [ ] **API Versioning** - Stable versioned endpoints
- [ ] **SDK Generation** - Auto-generated client SDKs

### Tooling

- [ ] **CLI Enhancements** - Better debugging commands
- [ ] **Admin Dashboard** - Web-based administration
- [ ] **Data Explorer** - Browse stored content
- [ ] **Migration Tools** - Data migration utilities

### Documentation

- [ ] **Interactive Examples** - Runnable code samples
- [ ] **Video Tutorials** - Getting started guides
- [ ] **API Reference** - Auto-generated API docs
- [ ] **Best Practices Guide** - Production deployment guide

## Research Areas

### Advanced Retrieval

- [ ] **Hypothetical Document Embedding (HyDE)** - Generate hypothetical answers for better retrieval
- [ ] **Self-Query** - Let LLM write its own filters
- [ ] **Contextual Compression** - Compress chunks before return
- [ ] **Reranking Models** - Neural reranking for better precision

### Knowledge Graph

- [ ] **Ontology Learning** - Automatic schema discovery
- [ ] **Link Prediction** - Predict missing relationships
- [ ] **Knowledge Completion** - Fill gaps in entities
- [ ] **Temporal Reasoning** - Reason about time-based facts

### Semantic Understanding

- [ ] **Claim Extraction** - Extract verifiable claims
- [ ] **Contradiction Detection** - Identify conflicting information
- [ ] **Summary Generation** - Multi-document summarization
- [ ] **Question Generation** - Generate questions from content

## Contributing

We welcome contributions! Areas of particular interest:

1. **Source Integrations** - Add connectors for new data sources
2. **Extraction Skills** - Domain-specific expertise configurations
3. **Performance Benchmarks** - Help measure and improve performance
4. **Documentation** - Improve examples and guides

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Feedback

Have suggestions for the roadmap? Please:
- Open an issue on GitHub
- Start a discussion
- Contact the maintainers

Priorities are based on community feedback and use cases.
