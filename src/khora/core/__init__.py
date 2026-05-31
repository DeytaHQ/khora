"""Core domain models and types for Khora."""

from __future__ import annotations

from .diagnostics import Degradation, ErrorRecord, SkipReason
from .models.document import Chunk, Document
from .models.entity import Entity, Episode, Relationship
from .models.event import EventType, MemoryEvent
from .models.tenancy import MemoryNamespace, TenancyMode
from .recall_abstention import compute_abstention_signals

__all__ = [
    # Tenancy
    "MemoryNamespace",
    "TenancyMode",
    # Document
    "Document",
    "Chunk",
    # Entity
    "Entity",
    "Episode",
    "Relationship",
    # Event
    "MemoryEvent",
    "EventType",
    # Recall helpers
    "compute_abstention_signals",
    # Failure-observability TypedDicts (ADR-001)
    "Degradation",
    "ErrorRecord",
    "SkipReason",
]
