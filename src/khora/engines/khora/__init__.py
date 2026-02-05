"""Khora engine - temporal-first memory engine.

The Khora engine is optimized for:
- Temporal queries: Filtering by date, relative time references ("yesterday" in message context)
- Fast/cheap ingestion: Batch + incremental with skeleton-based indexing
- High precision: Bi-temporal model tracking occurrence and ingestion time
- Cost efficiency: PageRank-based core selection, lazy entity expansion
- Structured field filtering: Filter on occurred_at, not just chunk.created_at
- Multiple backends: PostgreSQL+pgvector (default) and Weaviate (advanced hybrid search)

Usage:
    # Default backend (pgvector)
    async with MemoryLake(db_url, engine="khora") as lake:
        await lake.remember("content", title="Doc")
        results = await lake.recall("query", temporal_filter=TemporalFilter.relative_days(-1))

    # Weaviate backend (advanced filtering)
    async with MemoryLake(
        db_url,
        engine="khora",
        backend="weaviate",
        weaviate_url="http://localhost:8080",
    ) as lake:
        results = await lake.recall(
            "query",
            filters={"occurred_at": {"gte": "2024-01-01"}, "author": {"eq": "alice"}},
            hybrid_alpha=0.7,
        )
"""

from khora.engines.khora.engine import KhoraEngine

__all__ = ["KhoraEngine"]
