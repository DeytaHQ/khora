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

    # Chronicle engine — temporal-semantic recall, no graph DB needed
    async with MemoryLake("postgresql://localhost/mydb", engine="chronicle") as lake:
        ns = await lake.create_namespace()
        await lake.remember(
            "Alice met Bob at the conference on March 15th.",
            namespace=ns.namespace_id,
            entity_types=["PERSON", "EVENT"],
            relationship_types=["ATTENDED"],
        )
        result = await lake.recall("Who did Alice meet?", namespace=ns.namespace_id)
"""

from .config import KhoraConfig
from .core.models.document import DocumentSource
from .core.models.event import EventType
from .engines import create_engine, list_engines, register_engine
from .exceptions import KhoraError
from .extraction.skills import EntityTypeConfig, ExpertiseConfig, RelationshipTypeConfig
from .hooks import SemanticFilter
from .memory_lake import (
    BatchHandle,
    BatchResult,
    DocumentResult,
    LLMUsage,
    MemoryLake,
    RecallResult,
    RememberResult,
    Stats,
)
from .query import SearchMode

__version__ = __import__("importlib").metadata.version("khora")

__all__ = [
    "KhoraError",
    "MemoryLake",
    "LLMUsage",
    "RememberResult",
    "RecallResult",
    "BatchResult",
    "BatchHandle",
    "DocumentResult",
    "Stats",
    "SearchMode",
    "KhoraConfig",
    "DocumentSource",
    # Engine functions
    "create_engine",
    "list_engines",
    "register_engine",
    # Expertise types — stable public API (ADR-022)
    "ExpertiseConfig",
    "EntityTypeConfig",
    "RelationshipTypeConfig",
    # Semantic hooks
    "EventType",
    "SemanticFilter",
]
