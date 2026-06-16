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
    from khora.core.models import Entity, Relationship

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
    """Parse a list of UUIDs, dropping null / empty elements.

    Graph backends store provenance ids (``source_document_ids`` etc.) as
    list properties that can carry stray ``None`` / ``""`` entries on
    partial or externally-written data. Filtering them here keeps a single
    malformed element from aborting the whole deserialize (#1237).
    """
    if not val:
        return []
    return [parse_uuid(v) for v in val if v]


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

        Scoped to ``namespace_id`` to prevent cross-tenant IDOR (IDOR family).
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
        a different namespace (IDOR family).
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

    # -- Batch write defaults (#1149) ----------------------------------------
    # Default N+1 implementations built on the per-record CRUD primitives
    # every graph backend already provides. Without them, the coordinator's
    # hasattr gates silently skipped the graph write on backends that lack a
    # native batch path (Neptune, AGE): entities landed only in the vector
    # mirror and relationships were written nowhere. Backends with a native
    # batch path (Neo4j, Memgraph, SurrealDB, sqlite_lance) override these.

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 100,
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]:
        """Default N+1 upsert - subclasses should override for efficiency.

        Matches on ``(namespace_id, name, entity_type)`` with MERGE semantics
        mirroring the native batch paths: longest description wins, source
        ids are unioned, mention counts are summed, confidence takes the max,
        attributes are replaced. On match the INPUT entity's ``id`` is synced
        in place to the stored id so relationship endpoints built from
        extraction-time ids resolve (the #806 id-remap contract).

        ``batch_size`` is accepted for signature parity with the native batch
        paths and is irrelevant here (per-record writes). ``bulk_mode`` skips
        the existence probe - used for fresh namespaces where nothing can
        match. A mid-batch failure propagates (never a silent drop); the
        coordinator's #868 partial-failure accounting covers the
        vector-committed/graph-failed case.
        """
        results: list[tuple[Entity, bool]] = []
        for entity in entities:
            existing = None
            if not bulk_mode:
                existing = await self.get_entity_by_name(  # type: ignore[attr-defined]
                    namespace_id, entity.name, entity.entity_type
                )
            if existing is None:
                await self.create_entity(entity)  # type: ignore[attr-defined]
                results.append((entity, True))
                continue
            if len(entity.description or "") > len(existing.description or ""):
                existing.description = entity.description
            existing.source_document_ids = existing.source_document_ids + [
                d for d in entity.source_document_ids if d not in existing.source_document_ids
            ]
            existing.source_chunk_ids = existing.source_chunk_ids + [
                c for c in entity.source_chunk_ids if c not in existing.source_chunk_ids
            ]
            existing.mention_count += entity.mention_count
            existing.confidence = max(existing.confidence, entity.confidence)
            existing.attributes = entity.attributes
            existing.updated_at = entity.updated_at
            await self.update_entity(existing, namespace_id=namespace_id)  # type: ignore[attr-defined]
            entity.id = existing.id
            results.append((entity, False))
        return results

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 100,
    ) -> int:
        """Default N+1 create - subclasses should override for efficiency.

        Returns the number of relationships written. ``batch_size`` is
        accepted for signature parity and is irrelevant here. A mid-batch
        failure propagates (never a silent drop): relationships written
        before the failure are persisted and the caller sees the exception.
        """
        count = 0
        for relationship in relationships:
            await self.create_relationship(relationship)  # type: ignore[attr-defined]
            count += 1
        return count

    # -- Forget-cascade cleanup (#923) --------------------------------------
    # Default implementations built on the per-record CRUD primitives every
    # graph backend already provides. They let the forget cascade clean up
    # orphaned entities/relationships on backends that do not (yet) ship a
    # native batch delete or source-strip (SurrealDB, Memgraph, Neptune, AGE,
    # sqlite_lance). Backends with a native batch path (Neo4j, pgvector)
    # override these. Signatures mirror pgvector's so the cascade can call
    # either store uniformly.

    async def delete_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> int:
        """Hard-delete entities by id, scoped to ``namespace_id`` (IDOR family)."""
        deleted = 0
        for eid in entity_ids:
            if await self.delete_entity(eid, namespace_id=namespace_id):  # type: ignore[attr-defined]
                deleted += 1
        return deleted

    async def delete_relationships_batch(self, relationship_ids: list[UUID], *, namespace_id: UUID) -> int:
        """Hard-delete relationships by id, scoped to ``namespace_id`` (IDOR family)."""
        deleted = 0
        for rid in relationship_ids:
            if await self.delete_relationship(rid, namespace_id=namespace_id):  # type: ignore[attr-defined]
                deleted += 1
        return deleted

    async def strip_document_from_entities(
        self, entity_ids: list[UUID], document_id: UUID, *, namespace_id: UUID
    ) -> int:
        """Strip ``document_id`` from survivor entities' ``source_document_ids``.

        Per-record read-modify-write fallback scoped to ``namespace_id``
        (IDOR family). Backends with a native bulk update override this.
        """
        updated = 0
        for eid in entity_ids:
            entity = await self.get_entity(eid, namespace_id=namespace_id)  # type: ignore[attr-defined]
            if entity is None:
                continue
            if document_id in (entity.source_document_ids or []):
                entity.source_document_ids = [d for d in entity.source_document_ids if d != document_id]
                await self.update_entity(entity, namespace_id=namespace_id)  # type: ignore[attr-defined]
                updated += 1
        return updated

    async def strip_document_from_relationships(
        self, relationship_ids: list[UUID], document_id: UUID, *, namespace_id: UUID
    ) -> int:
        """Strip ``document_id`` from survivor relationships' ``source_document_ids``.

        No graph backend exposes a single-relationship update primitive, so
        this deletes the survivor and recreates it with the document id
        removed. Scoped to ``namespace_id`` (IDOR family).
        """
        updated = 0
        for rid in relationship_ids:
            rel = await self.get_relationship(rid, namespace_id=namespace_id)  # type: ignore[attr-defined]
            if rel is None:
                continue
            if document_id in (rel.source_document_ids or []):
                rel.source_document_ids = [d for d in rel.source_document_ids if d != document_id]
                await self.delete_relationship(rid, namespace_id=namespace_id)  # type: ignore[attr-defined]
                await self.create_relationship(rel)  # type: ignore[attr-defined]
                updated += 1
        return updated


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
