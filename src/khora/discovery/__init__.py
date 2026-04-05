"""Interactive datasource discovery for Khora.

When no --source/-s is provided to the ontology construct command,
the discovery agent helps users find, validate, and pull datasources
from the internet using Perplexity (search) and Firecrawl (scraping).
"""

from __future__ import annotations

from .memory import DiscoveryMemory, MemoryEntry
from .state import (
    AgentPhase,
    DiscoveredSource,
    FetchAttempt,
    FetchMethod,
    FetchResult,
    SessionState,
    SourceStatus,
    SourceType,
)

__all__ = [
    "AgentPhase",
    "DiscoveredSource",
    "DiscoveryMemory",
    "FetchAttempt",
    "FetchMethod",
    "FetchResult",
    "MemoryEntry",
    "SessionState",
    "SourceStatus",
    "SourceType",
]
