"""Khora - Deyta's memory lake and materialization of knowledge.

Khora provides a unified interface for:
- Storing and retrieving knowledge artifacts
- Materializing data transformations
- Building memory graphs and relationships

Example usage:
    # Simplest - from env vars (KHORA_DATABASE_URL)
    from khora import MemoryLake

    async with MemoryLake() as lake:
        await lake.remember("Important information to store")
        results = await lake.recall("query about information")

    # Common - explicit database URL
    async with MemoryLake("postgresql://localhost/mydb") as lake:
        await lake.remember("content", title="My Document")
        results = await lake.recall("query", limit=20)

    # With graph backend
    async with MemoryLake(
        "postgresql://localhost/mydb",
        graph_url="bolt://localhost:7687",
    ) as lake:
        results = await lake.recall("query", mode=SearchMode.GRAPH)

    # Batch ingestion with automatic optimization
    async with MemoryLake(database_url) as lake:
        result = await lake.remember_batch(documents)
        print(f"Processed {result.processed} docs, {result.entities} entities")

    # Raw search without LLM features (for benchmarks)
    results = await lake.recall(query, mode=SearchMode.ALL, raw=True)
"""

from .cli import main
from .config import KhoraConfig
from .engines import create_engine, list_engines, register_engine
from .memory_lake import BatchResult, MemoryLake, RecallResult, RememberResult, Stats
from .query import SearchMode

__version__ = "0.1.0"

__all__ = [
    "main",
    "MemoryLake",
    "RememberResult",
    "RecallResult",
    "BatchResult",
    "Stats",
    "SearchMode",
    "KhoraConfig",
    # Engine functions
    "create_engine",
    "list_engines",
    "register_engine",
]
