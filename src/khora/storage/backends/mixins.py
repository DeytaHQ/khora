"""Mixin base classes for storage backends.

Provide default implementations for batch/aggregate operations so new
backends get them for free and can override with optimized versions.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from khora.core.models import Entity

# ---------------------------------------------------------------------------
# Async session mixin (shared across SQL-based backends)
# ---------------------------------------------------------------------------

AsyncSessionFactory = async_sessionmaker[AsyncSession]


class AsyncSessionMixin:
    """Mixin providing shared async session management."""

    _session_factory: AsyncSessionFactory | None

    def _get_session(self) -> AsyncSession:
        """Get a new database session."""
        if self._session_factory is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._session_factory()


# ---------------------------------------------------------------------------
# Deadlock retry decorator (shared across SQL-based backends)
# ---------------------------------------------------------------------------


def _is_deadlock_error(exc: BaseException) -> bool:
    """Check if exception is a database deadlock."""
    error_str = str(exc).lower()
    return "deadlock" in error_str or "serialization" in error_str


retry_on_deadlock = retry(
    retry=retry_if_exception(_is_deadlock_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, max=2),
    reraise=True,
)


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
        except Exception as e:
            logger.debug(f"Failed to convert graph element via items(): {e}")
    # Last resort — avoid crashing on unexpected types
    return {"_raw": str(element)}


# ---------------------------------------------------------------------------
# Cypher label/relationship-type sanitizer (shared across graph backends)
# ---------------------------------------------------------------------------

_CYPHER_LABEL_RE = re.compile(r"[^A-Za-z0-9_]")


def sanitize_cypher_label(label: str) -> str:
    """Sanitize a string for safe use as a Cypher relationship type or node label.

    Strips all characters except alphanumeric and underscore. Converts to
    UPPER_SNAKE_CASE. Falls back to ``RELATES_TO`` if the result is empty.

    This **must** be applied to any user-controlled value interpolated into
    Cypher query patterns like ``[r:TYPE]`` where parameterization is not
    possible.
    """
    sanitized = _CYPHER_LABEL_RE.sub("_", label.strip())
    return sanitized.upper() if sanitized else "RELATES_TO"


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

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        """Default N+1 implementation — subclasses should override for efficiency.

        Scoped to ``namespace_id`` to prevent cross-tenant IDOR (IGR-223).
        """
        if not entity_ids:
            return {}
        result: dict[UUID, Any] = {}
        for eid in entity_ids:
            entity = await self.get_entity(eid, namespace_id=namespace_id)  # type: ignore[attr-defined]
            if entity is not None:
                result[eid] = entity
        return result

    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Default N+1 implementation — subclasses should override for efficiency.

        Traversal is scoped to ``namespace_id`` so it never visits a node in
        a different namespace (IGR-223).
        """
        if not entity_ids:
            return {}
        result: dict[UUID, dict[str, Any]] = {}
        for eid in entity_ids:
            neighborhood = await self.get_neighborhood(  # type: ignore[attr-defined]
                eid,
                namespace_id=namespace_id,
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

    async def count_relationships(self, namespace_id: UUID) -> int:
        """Default — subclasses should override."""
        raise NotImplementedError


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
