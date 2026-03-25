"""SurrealDB event store adapter for Khora.

Implements EventStoreProtocol using SurrealQL, delegating connection
lifecycle to SurrealDBConnection.  Record IDs follow the SurrealDB convention:
``memory_event:⟨uuid⟩``.  All UUIDs are converted to ``str`` at the boundary
and parsed back on read.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from loguru import logger

from khora.core.models.event import EventType, MemoryEvent
from khora.storage.backends.surrealdb._helpers import (
    _parse_dt,
    _parse_uuid,
)
from khora.storage.backends.surrealdb.connection import SurrealDBConnection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLE = "memory_event"


def _event_record_id(uid: UUID) -> str:
    """Build a SurrealDB record ID string: ``memory_event:⟨uuid⟩``."""
    return f"{_TABLE}:⟨{uid!s}⟩"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SurrealDBEventStoreAdapter:
    """Event store backend backed by SurrealDB.

    Fulfils :class:`~khora.storage.backends.base.EventStoreProtocol`
    without importing SQLAlchemy.  The adapter delegates all I/O to a
    :class:`SurrealDBConnection` instance.
    """

    def __init__(self, connection: SurrealDBConnection) -> None:
        self._conn = connection

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SurrealDBEventStoreAdapter:
        """Create an adapter from a configuration dictionary.

        Expected keys mirror :class:`SurrealDBConnection.__init__` kwargs:
        ``mode``, ``path``, ``url``, ``namespace``, ``database``, ``user``,
        ``password``.  All are optional and fall back to SurrealDBConnection
        defaults.
        """
        conn = SurrealDBConnection(
            mode=config.get("mode", "memory"),
            path=config.get("path"),
            url=config.get("url"),
            namespace=config.get("namespace", "khora"),
            database=config.get("database", "default"),
            user=config.get("user", "root"),
            password=config.get("password", "root"),
        )
        return cls(connection=conn)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def create_tables(self) -> None:
        """Create SurrealDB tables and indexes (idempotent).

        Schema is also auto-initialized on connect(), so this is
        safe to call multiple times.
        """
        from .schema import initialize_schema

        await initialize_schema(self._conn)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to SurrealDB."""
        await self._conn.connect()

    async def disconnect(self) -> None:
        """Close the SurrealDB connection."""
        await self._conn.disconnect()

    async def is_healthy(self) -> bool:
        """Delegate health check to the connection."""
        return await self._conn.is_healthy()

    # ------------------------------------------------------------------
    # SQLAlchemy compatibility shim
    # ------------------------------------------------------------------

    def _get_session(self) -> None:
        """No-op — SurrealDB does not use SQLAlchemy sessions."""
        return None

    # ------------------------------------------------------------------
    # Event operations
    # ------------------------------------------------------------------

    async def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append a single event to the log."""
        rid = _event_record_id(event.id)
        row = await self._conn.query_one(
            "CREATE $rid SET "
            "namespace_id = $namespace_id, "
            "event_type = $event_type, "
            "timestamp = $timestamp, "
            "resource_type = $resource_type, "
            "resource_id = $resource_id, "
            "data = $data, "
            "previous_data = $previous_data, "
            "actor_id = $actor_id, "
            "actor_type = $actor_type, "
            "correlation_id = $correlation_id, "
            "version = $version, "
            "metadata_ = $metadata_",
            {
                "rid": rid,
                "namespace_id": str(event.namespace_id),
                "event_type": event.event_type.value if isinstance(event.event_type, EventType) else event.event_type,
                "timestamp": event.timestamp,
                "resource_type": event.resource_type,
                "resource_id": str(event.resource_id),
                "data": event.data or {},
                "previous_data": event.previous_data,
                "actor_id": event.actor_id,
                "actor_type": event.actor_type,
                "correlation_id": str(event.correlation_id) if event.correlation_id else None,
                "version": event.version,
                "metadata_": event.metadata or {},
            },
        )
        if row is None:
            raise RuntimeError(f"Failed to append event {event.id}")
        logger.debug("Appended event {} (type={})", event.id, event.event_type.value)
        return self._row_to_event(row)

    async def append_events_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events in a batch using INSERT INTO."""
        if not events:
            return []

        records = []
        for event in events:
            records.append(
                {
                    "id": _event_record_id(event.id),
                    "namespace_id": str(event.namespace_id),
                    "event_type": (
                        event.event_type.value if isinstance(event.event_type, EventType) else event.event_type
                    ),
                    "timestamp": event.timestamp,
                    "resource_type": event.resource_type,
                    "resource_id": str(event.resource_id),
                    "data": event.data or {},
                    "previous_data": event.previous_data,
                    "actor_id": event.actor_id,
                    "actor_type": event.actor_type,
                    "correlation_id": str(event.correlation_id) if event.correlation_id else None,
                    "version": event.version,
                    "metadata_": event.metadata or {},
                }
            )

        rows = await self._conn.query(
            "INSERT INTO memory_event $records",
            {"records": records},
        )
        logger.debug("Appended {} events in batch", len(events))

        if rows:
            return [self._row_to_event(r) for r in rows]
        # Fall back to returning the original events if INSERT doesn't return rows
        return events

    async def get_events(
        self,
        namespace_id: UUID,
        *,
        event_types: list[str] | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEvent]:
        """Query events from the log with dynamic filters."""
        clauses: list[str] = ["namespace_id = $namespace_id"]
        bindings: dict[str, Any] = {"namespace_id": str(namespace_id)}

        if event_types:
            clauses.append("event_type IN $event_types")
            bindings["event_types"] = event_types

        if resource_type:
            clauses.append("resource_type = $resource_type")
            bindings["resource_type"] = resource_type

        if resource_id:
            clauses.append("resource_id = $resource_id")
            bindings["resource_id"] = str(resource_id)

        if after:
            clauses.append("timestamp > $after")
            bindings["after"] = after

        if before:
            clauses.append("timestamp < $before")
            bindings["before"] = before

        where = " AND ".join(clauses)
        bindings["lim"] = limit
        bindings["off"] = offset

        rows = await self._conn.query(
            f"SELECT * FROM {_TABLE} WHERE {where} ORDER BY timestamp DESC LIMIT $lim START $off",
            bindings,
        )
        return [self._row_to_event(r) for r in rows]

    async def get_events_for_resource(
        self,
        resource_type: str,
        resource_id: UUID,
        *,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Get all events for a specific resource."""
        rows = await self._conn.query(
            f"SELECT * FROM {_TABLE} "
            "WHERE resource_type = $resource_type AND resource_id = $resource_id "
            "ORDER BY timestamp DESC LIMIT $lim",
            {
                "resource_type": resource_type,
                "resource_id": str(resource_id),
                "lim": limit,
            },
        )
        return [self._row_to_event(r) for r in rows]

    async def get_latest_event(
        self,
        resource_type: str,
        resource_id: UUID,
    ) -> MemoryEvent | None:
        """Get the latest event for a resource."""
        row = await self._conn.query_one(
            f"SELECT * FROM {_TABLE} "
            "WHERE resource_type = $resource_type AND resource_id = $resource_id "
            "ORDER BY timestamp DESC LIMIT 1",
            {
                "resource_type": resource_type,
                "resource_id": str(resource_id),
            },
        )
        if row is None:
            return None
        return self._row_to_event(row)

    async def count_events(
        self,
        namespace_id: UUID,
        *,
        event_types: list[str] | None = None,
        after: datetime | None = None,
    ) -> int:
        """Count events matching criteria."""
        clauses: list[str] = ["namespace_id = $namespace_id"]
        bindings: dict[str, Any] = {"namespace_id": str(namespace_id)}

        if event_types:
            clauses.append("event_type IN $event_types")
            bindings["event_types"] = event_types

        if after:
            clauses.append("timestamp > $after")
            bindings["after"] = after

        where = " AND ".join(clauses)

        row = await self._conn.query_one(
            f"SELECT count() AS total FROM {_TABLE} WHERE {where} GROUP ALL",
            bindings,
        )
        if row is None:
            return 0
        return int(row.get("total", 0))

    # ------------------------------------------------------------------
    # Row → domain model
    # ------------------------------------------------------------------

    def _row_to_event(self, row: dict[str, Any]) -> MemoryEvent:
        """Convert a SurrealDB row dict to a domain MemoryEvent."""
        event_type_raw = row.get("event_type", "document.created")
        correlation_raw = row.get("correlation_id")

        return MemoryEvent(
            id=_parse_uuid(row["id"]),
            namespace_id=UUID(row["namespace_id"]) if isinstance(row["namespace_id"], str) else row["namespace_id"],
            event_type=EventType(event_type_raw) if isinstance(event_type_raw, str) else event_type_raw,
            timestamp=_parse_dt(row.get("timestamp")) or row.get("timestamp"),
            resource_type=row.get("resource_type", ""),
            resource_id=(
                UUID(row["resource_id"]) if isinstance(row.get("resource_id"), str) else row.get("resource_id")
            ),
            data=row.get("data") or {},
            previous_data=row.get("previous_data"),
            actor_id=row.get("actor_id"),
            actor_type=row.get("actor_type", "system"),
            correlation_id=(
                UUID(correlation_raw) if correlation_raw and isinstance(correlation_raw, str) else correlation_raw
            ),
            version=row.get("version", 1),
            metadata=row.get("metadata_") or {},
        )
