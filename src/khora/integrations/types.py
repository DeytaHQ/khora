"""Shared return types for ``khora.integrations`` adapters.

These types are part of khora's stable public API surface — adapter
implementations and downstream consumers rely on the field names. Add
fields; never rename or remove without a coordinated release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(slots=True, frozen=True)
class RetrievedNode:
    """A single retrieval result surfaced to a framework retriever.

    Adapter-facing shape returned by :meth:`RetrieverAdapter.aretrieve`.
    The framework (LangGraph, LlamaIndex, ...) is responsible for
    converting these into its own node / document types.

    Attributes:
        id: The khora chunk / entity ID this node came from.
        text: Renderable text payload. For chunks this is the chunk
            content; for entities, a short summary line.
        score: Retrieval score (higher = more relevant). Engine-defined
            scale; comparable within a single result list only.
        metadata: Arbitrary engine metadata (source URI, namespace ID,
            entity types, ...). Frameworks pass this through to their
            own metadata field.
    """

    id: UUID
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
