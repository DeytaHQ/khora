"""Mixin base classes for storage backends.

Provide default implementations for batch/aggregate operations so new
backends get them for free and can override with optimized versions.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

# ---------------------------------------------------------------------------
# Serialization helpers (shared across graph backends)
# ---------------------------------------------------------------------------


def serialize_dict(value: dict[str, Any] | None) -> str | None:
    """Serialize a dict to JSON string for property-store backends.

    Graph databases typically only support primitive property values;
    nested dicts must be stored as JSON strings.
    """
    if value is None:
        return None
    return json.dumps(value)


def deserialize_dict(value: str | dict[str, Any] | None) -> dict[str, Any]:
    """Deserialize a JSON string back to dict.

    Handles both string (serialized) and dict (legacy/native) values.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def element_to_dict(element: Any) -> dict[str, Any]:
    """Safely convert a graph element (Node, Relationship, or raw value) to a dict.

    Handles Neo4j Node/Relationship objects, Kùzu results, plain dicts, etc.
    """
    if isinstance(element, dict):
        return element
    # neo4j.graph.Node / Relationship expose _properties
    if hasattr(element, "_properties"):
        return dict(element._properties)
    # Some driver versions make Node/Relationship a Mapping
    if hasattr(element, "items"):
        try:
            return dict(element.items())
        except Exception:
            pass
    # Last resort — avoid crashing on unexpected types
    return {"_raw": str(element)}


# ---------------------------------------------------------------------------
# Parse helpers for record→domain model conversions
# ---------------------------------------------------------------------------


def parse_uuid(val: Any) -> UUID:
    """Parse a UUID from a string or UUID."""
    if isinstance(val, UUID):
        return val
    return UUID(str(val))


def parse_uuid_list(val: list | None) -> list[UUID]:
    """Parse a list of UUIDs."""
    if not val:
        return []
    return [parse_uuid(v) for v in val]


def parse_datetime(val: str | datetime | None, default: datetime | None = None) -> datetime | None:
    """Parse an ISO datetime string or pass through a datetime."""
    if val is None:
        return default
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(val)


# ---------------------------------------------------------------------------
# GraphBackendBase
# ---------------------------------------------------------------------------


class GraphBackendBase:
    """Mixin providing default N+1 implementations for batch/aggregate graph ops.

    Concrete graph backends inherit this alongside their protocol implementation.
    Backends that support native batch queries should override these methods.
    """

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:  # type: ignore[unresolved-reference]  # noqa: F821
        """Default N+1 implementation — subclasses should override for efficiency."""
        if not entity_ids:
            return {}
        result: dict[UUID, Any] = {}
        for eid in entity_ids:
            entity = await self.get_entity(eid)  # type: ignore[attr-defined]
            if entity is not None:
                result[eid] = entity
        return result

    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Default N+1 implementation — subclasses should override for efficiency."""
        if not entity_ids:
            return {}
        result: dict[UUID, dict[str, Any]] = {}
        for eid in entity_ids:
            neighborhood = await self.get_neighborhood(  # type: ignore[attr-defined]
                eid,
                depth=depth,
                relationship_types=relationship_types,
                limit=limit_per_entity,
            )
            result[eid] = neighborhood
        return result

    async def count_entities(self, namespace_id: UUID) -> int:
        """Default implementation using list_entities — subclasses should override."""
        entities = await self.list_entities(namespace_id, limit=100_000)  # type: ignore[attr-defined]
        return len(entities)


# ---------------------------------------------------------------------------
# VectorBackendBase
# ---------------------------------------------------------------------------


class VectorBackendBase:
    """Mixin providing default implementations for aggregate vector ops.

    Concrete vector backends inherit this alongside their protocol implementation.
    """

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Default implementation — subclasses should override for efficiency."""
        # Fallback: this is intentionally inefficient; override in real backends
        return 0

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """Default implementation — subclasses should override."""
        return []
