"""Khora — knowledge graph + vector + SQL storage library.

Khora provides a unified interface for:
- Storing and retrieving knowledge artifacts
- Materializing data transformations
- Building memory graphs and relationships

Example usage:
    # Simplest - from env vars (KHORA_DATABASE_URL)
    from khora import Khora

    async with Khora() as kb:
        await kb.remember("Important information to store")
        results = await kb.recall("query about information")

    # Common - explicit database URL
    async with Khora("postgresql://localhost/mydb") as kb:
        await kb.remember("content", title="My Document")
        results = await kb.recall("query", limit=20)

    # With graph backend
    async with Khora(
        "postgresql://localhost/mydb",
        graph_url="bolt://localhost:7687",
    ) as kb:
        results = await kb.recall("query", mode=SearchMode.GRAPH)

    # Batch ingestion with automatic optimization
    async with Khora(database_url) as kb:
        result = await kb.remember_batch(documents)
        print(f"Processed {result.processed} docs, {result.entities} entities")

    # Chronicle engine — temporal-semantic recall, no graph DB needed
    async with Khora("postgresql://localhost/mydb", engine="chronicle") as kb:
        ns = await kb.create_namespace()
        await kb.remember(
            "Alice met Bob at the conference on March 15th.",
            namespace=ns.namespace_id,
            entity_types=["PERSON", "EVENT"],
            relationship_types=["ATTENDED"],
        )
        result = await kb.recall("Who did Alice meet?", namespace=ns.namespace_id)
"""

from . import integrations
from .config import KhoraConfig
from .core.models.document import DocumentSource
from .core.models.event import EventType
from .core.recall_context import context_text
from .dream import DreamConfig, DreamMode, DreamResult, DreamRunInfo, DreamScope, OpKind
from .engines import create_engine, list_engines, register_engine
from .exceptions import EngineCapabilityError, KhoraError
from .extraction.skills import EntityTypeConfig, ExpertiseConfig, RelationshipTypeConfig
from .filter import (
    SYSTEM_KEYS,
    DateOps,
    Op,
    RecallFilter,
    RecallFilterUnsupportedError,
    RecallFilterValidationError,
    StringOps,
)
from .hooks import SemanticFilter
from .khora import (
    BatchHandle,
    BatchResult,
    DocumentProjection,
    DocumentResult,
    Khora,
    LLMUsage,
    RecallChunk,
    RecallEntity,
    RecallRelationship,
    RecallResult,
    RememberResult,
    Stats,
)
from .search_mode import SearchMode

__version__ = __import__("importlib").metadata.version("khora")

__all__ = [
    "EngineCapabilityError",
    "KhoraError",
    "Khora",
    "LLMUsage",
    "RememberResult",
    "RecallResult",
    "DocumentProjection",
    "RecallChunk",
    "RecallEntity",
    "RecallRelationship",
    "context_text",
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
    # Expertise types — stable public API
    "ExpertiseConfig",
    "EntityTypeConfig",
    "RelationshipTypeConfig",
    # Semantic hooks
    "EventType",
    "SemanticFilter",
    # Integrations (adapter Protocols, registry, types) — see issue #619
    "integrations",
    # Dream-phase scaffolding — see issues #649 / #650
    "DreamConfig",
    "DreamMode",
    "DreamScope",
    "DreamResult",
    "DreamRunInfo",
    "OpKind",
    # Deterministic recall filter
    "RecallFilter",
    "StringOps",
    "DateOps",
    "RecallFilterValidationError",
    "RecallFilterUnsupportedError",
    "Op",
    "SYSTEM_KEYS",
]
