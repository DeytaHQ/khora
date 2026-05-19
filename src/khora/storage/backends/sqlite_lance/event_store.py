"""SQLite event store adapter.

Append-only event log backed by the ``memory_events`` SQLite table
(created by Alembic migration ``000_initial_schema`` under the
dialect-gated path).  Implements
:class:`~khora.storage.backends.base.EventStoreProtocol` without
SQLAlchemy — direct aiosqlite against the shared
:class:`EmbeddedStorageHandle`.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.core.models.event import EventType, MemoryEvent

from ._helpers import from_json_text, iso8601, to_json_text, uuid_to_text

if TYPE_CHECKING:
    from .connection import EmbeddedStorageHandle


_TABLE = "memory_events"

# Column list kept in one place so SELECTs and _row_to_event stay in sync.
# Note: on SQLite the JSONB-typed columns (data/previous_data/metadata) are
# stored as TEXT (see _enum_col / jsonb_t dialect gating in 000_initial_schema).
_COLUMNS = (
    "id",
    "namespace_id",
    "event_type",
    "timestamp",
    "resource_type",
    "resource_id",
    "data",
    "previous_data",
    "actor_id",
    "actor_type",
    "correlation_id",
    "version",
    "metadata",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE}"  # noqa: S608


def _event_type_value(et: EventType | str) -> str:
    """Canonical string form of an event type for storage."""
    return et.value if isinstance(et, EventType) else et


def _event_to_row(event: MemoryEvent) -> tuple:
    """Convert a domain event to a tuple of SQLite-bound values."""
    return (
        uuid_to_text(event.id),
        uuid_to_text(event.namespace_id),
        _event_type_value(event.event_type),
        iso8601(event.timestamp),
        event.resource_type,
        uuid_to_text(event.resource_id),
        to_json_text(event.data or {}),
        to_json_text(event.previous_data) if event.previous_data is not None else None,
        event.actor_id,
        event.actor_type,
        uuid_to_text(event.correlation_id) if event.correlation_id is not None else None,
        event.version,
        to_json_text(event.metadata or {}),
    )


class SQLiteLanceEventStoreAdapter:
    """Append-only event store backed by SQLite.

    Shares the :class:`EmbeddedStorageHandle` with the other sqlite_lance
    adapters — the handle owns the aiosqlite connection and its lifecycle
    (:meth:`connect` / :meth:`disconnect`).  Writes are append-only: no
    update or delete helpers are exposed.
    """

    def __init__(self, handle: EmbeddedStorageHandle) -> None:
        self._handle = handle

    # ------------------------------------------------------------------
    # Connection lifecycle (delegated to the shared handle)
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await self._handle.connect()

    async def disconnect(self) -> None:
        await self._handle.disconnect()

    async def is_healthy(self) -> bool:
        return await self._handle.is_healthy()

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    async def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append a single event. Returns the stored event."""
        placeholders = ", ".join("?" for _ in _COLUMNS)
        sql = f"INSERT INTO {_TABLE} ({', '.join(_COLUMNS)}) VALUES ({placeholders})"  # noqa: S608
        conn = self._handle.sqlite
        await conn.execute(sql, _event_to_row(event))
        await conn.commit()
        logger.debug("Appended event {} (type={})", event.id, _event_type_value(event.event_type))
        return event

    async def append_events_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events in a single transaction."""
        if not events:
            return []
        placeholders = ", ".join("?" for _ in _COLUMNS)
        sql = f"INSERT INTO {_TABLE} ({', '.join(_COLUMNS)}) VALUES ({placeholders})"  # noqa: S608
        rows = [_event_to_row(e) for e in events]
        conn = self._handle.sqlite
        await conn.executemany(sql, rows)
        await conn.commit()
        logger.debug("Appended {} events in batch", len(events))
        return events

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

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
        """Return events matching the filters, newest first."""
        clauses: list[str] = ["namespace_id = ?"]
        params: list[Any] = [uuid_to_text(namespace_id)]

        if event_types:
            # Normalize EventType members to their stored string values.
            values = [_event_type_value(et) for et in event_types]
            clauses.append(f"event_type IN ({', '.join('?' for _ in values)})")
            params.extend(values)
        if resource_type is not None:
            clauses.append("resource_type = ?")
            params.append(resource_type)
        if resource_id is not None:
            clauses.append("resource_id = ?")
            params.append(uuid_to_text(resource_id))
        if after is not None:
            clauses.append("timestamp > ?")
            params.append(iso8601(after))
        if before is not None:
            clauses.append("timestamp < ?")
            params.append(iso8601(before))

        where = " AND ".join(clauses)
        sql = f"{_SELECT} WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"  # noqa: S608
        params.append(limit)
        params.append(offset)

        conn = self._handle.sqlite
        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_event(r) for r in rows]

    async def get_events_for_resource(
        self,
        resource_type: str,
        resource_id: UUID,
        *,
        namespace_id: UUID,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Return all events for a specific resource, newest first.

        Scoped to ``namespace_id`` so cross-tenant audit-log access is
        impossible (IGR-221 / IGR-223 family). Returns an empty list when
        the resource belongs to a different namespace.
        """
        sql = (
            f"{_SELECT} WHERE resource_type = ? AND resource_id = ? "  # noqa: S608
            "AND namespace_id = ? "
            "ORDER BY timestamp DESC LIMIT ?"
        )
        conn = self._handle.sqlite
        async with conn.execute(
            sql,
            (resource_type, uuid_to_text(resource_id), uuid_to_text(namespace_id), limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_event(r) for r in rows]

    async def get_latest_event(
        self,
        resource_type: str,
        resource_id: UUID,
        *,
        namespace_id: UUID,
    ) -> MemoryEvent | None:
        """Return the most recent event for the given resource, or None.

        Scoped to ``namespace_id`` so cross-tenant audit-log access is
        impossible (IGR-221 / IGR-223 family). Returns ``None`` when the
        resource belongs to a different namespace.
        """
        sql = (
            f"{_SELECT} WHERE resource_type = ? AND resource_id = ? "  # noqa: S608
            "AND namespace_id = ? "
            "ORDER BY timestamp DESC LIMIT 1"
        )
        conn = self._handle.sqlite
        async with conn.execute(sql, (resource_type, uuid_to_text(resource_id), uuid_to_text(namespace_id))) as cursor:
            row = await cursor.fetchone()
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
        """Count events matching the filters."""
        clauses: list[str] = ["namespace_id = ?"]
        params: list[Any] = [uuid_to_text(namespace_id)]

        if event_types:
            values = [_event_type_value(et) for et in event_types]
            clauses.append(f"event_type IN ({', '.join('?' for _ in values)})")
            params.extend(values)
        if after is not None:
            clauses.append("timestamp > ?")
            params.append(iso8601(after))

        where = " AND ".join(clauses)
        sql = f"SELECT COUNT(*) FROM {_TABLE} WHERE {where}"  # noqa: S608

        conn = self._handle.sqlite
        async with conn.execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    # ------------------------------------------------------------------
    # Row → domain model
    # ------------------------------------------------------------------

    def _row_to_event(self, row: Any) -> MemoryEvent:
        """Convert an aiosqlite Row (or tuple) to a domain MemoryEvent."""
        # aiosqlite.Row supports both index- and name-based access.
        event_type_raw = row["event_type"]
        correlation_raw = row["correlation_id"]
        previous_raw = row["previous_data"]

        event_type: EventType | str
        try:
            event_type = EventType(event_type_raw) if isinstance(event_type_raw, str) else event_type_raw
        except ValueError:
            # Unknown/legacy values — preserve the raw string rather than failing.
            event_type = event_type_raw

        return MemoryEvent(
            id=UUID(row["id"]),
            namespace_id=UUID(row["namespace_id"]),
            event_type=event_type,  # type: ignore[arg-type]
            timestamp=datetime.fromisoformat(row["timestamp"])
            if row["timestamp"]
            else datetime.fromisoformat("1970-01-01T00:00:00+00:00"),
            resource_type=row["resource_type"],
            resource_id=UUID(row["resource_id"]),
            data=from_json_text(row["data"]) if row["data"] else {},
            previous_data=from_json_text(previous_raw) if previous_raw else None,
            actor_id=row["actor_id"],
            actor_type=row["actor_type"] or "system",
            correlation_id=UUID(correlation_raw) if correlation_raw else None,
            version=row["version"] if row["version"] is not None else 1,
            metadata=from_json_text(row["metadata"]) if row["metadata"] else {},
        )
