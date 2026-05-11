"""PostgreSQL-based event store for event sourcing.

Provides an append-only event log for all Khora operations,
enabling audit trails, temporal queries, and event replay.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from khora.core.models import MemoryEvent
from khora.core.models.event import EventType
from khora.db.models import Base, MemoryEventModel
from khora.storage.backends.mixins import AsyncSessionMixin


class PostgreSQLEventStore(AsyncSessionMixin):
    """PostgreSQL-based event store.

    Stores all memory events in an append-only log for event sourcing.
    """

    def __init__(
        self,
        database_url: str,
        *,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_pre_ping: bool = False,
        engine: AsyncEngine | None = None,
    ) -> None:
        """Initialize the event store.

        Args:
            database_url: PostgreSQL connection URL
            echo: Enable SQL echo logging
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_pre_ping: Enable pool pre-ping to detect stale connections
            engine: Optional shared engine (skip dispose on disconnect)
        """
        # Convert to async URL if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)

        self._database_url = database_url
        self._echo = echo
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_pre_ping = pool_pre_ping
        self._engine: AsyncEngine | None = engine
        self._engine_shared: bool = engine is not None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        """Establish connection to the database."""
        if self._session_factory is not None:
            return

        logger.info("Connecting event store...")
        if self._engine is None:
            self._engine = create_async_engine(
                self._database_url,
                echo=self._echo,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_pre_ping=self._pool_pre_ping,
            )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Event store connected")

    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            logger.info("Disconnecting event store...")
            if not self._engine_shared:
                await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Event store disconnected")

    async def is_healthy(self) -> bool:
        """Check if the store is healthy."""
        if self._engine is None or self._session_factory is None:
            return False
        try:
            async with self._session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"Event store health check failed: {e}")
            return False

    async def create_tables(self) -> None:
        """Create database tables.

        .. deprecated::
            Use ``run_migrations()`` instead. ``create_tables()`` bypasses
            Alembic and masks missing migrations — tests pass locally but
            production breaks. Will be removed in a future release.
        """
        import warnings

        warnings.warn(
            "create_tables() is deprecated. Use khora.db.run_migrations() instead. "
            "create_tables() bypasses Alembic and masks missing migrations.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._engine is None:
            raise RuntimeError("Event store not connected. Call connect() first.")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def append_event(self, event: MemoryEvent, *, session: AsyncSession | None = None) -> MemoryEvent:
        """Append an event to the log."""
        if session is not None:
            return await self._append_event_with(session, event)
        async with self._get_session() as own_session:
            return await self._append_event_with(own_session, event, commit=True)

    async def _append_event_with(
        self, session: AsyncSession, event: MemoryEvent, *, commit: bool = False
    ) -> MemoryEvent:
        model = MemoryEventModel(
            id=event.id,
            namespace_id=event.namespace_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            data=event.data,
            previous_data=event.previous_data,
            actor_id=event.actor_id,
            actor_type=event.actor_type,
            correlation_id=event.correlation_id,
            version=event.version,
            metadata_=event.metadata,
        )
        session.add(model)
        if commit:
            await session.commit()
        else:
            await session.flush()
        await session.refresh(model)
        return self._model_to_domain(model)

    async def append_events_batch(
        self, events: list[MemoryEvent], *, session: AsyncSession | None = None
    ) -> list[MemoryEvent]:
        """Append multiple events in a batch."""
        if not events:
            return []

        if session is not None:
            return await self._append_events_batch_with(session, events)
        async with self._get_session() as own_session:
            return await self._append_events_batch_with(own_session, events, commit=True)

    async def _append_events_batch_with(
        self, session: AsyncSession, events: list[MemoryEvent], *, commit: bool = False
    ) -> list[MemoryEvent]:
        models = [
            MemoryEventModel(
                id=event.id,
                namespace_id=event.namespace_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                data=event.data,
                previous_data=event.previous_data,
                actor_id=event.actor_id,
                actor_type=event.actor_type,
                correlation_id=event.correlation_id,
                version=event.version,
                metadata_=event.metadata,
            )
            for event in events
        ]
        session.add_all(models)
        if commit:
            await session.commit()
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
        """Query events from the log."""
        async with self._get_session() as session:
            query = select(MemoryEventModel).where(MemoryEventModel.namespace_id == namespace_id)

            if event_types:
                query = query.where(MemoryEventModel.event_type.in_(event_types))
            if resource_type:
                query = query.where(MemoryEventModel.resource_type == resource_type)
            if resource_id:
                query = query.where(MemoryEventModel.resource_id == resource_id)
            if after:
                query = query.where(MemoryEventModel.timestamp > after)
            if before:
                query = query.where(MemoryEventModel.timestamp < before)

            query = query.order_by(MemoryEventModel.timestamp.desc()).limit(limit).offset(offset)

            result = await session.execute(query)
            return [self._model_to_domain(m) for m in result.scalars().all()]

    async def get_events_for_resource(
        self,
        resource_type: str,
        resource_id: UUID,
        *,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Get all events for a specific resource."""
        async with self._get_session() as session:
            query = (
                select(MemoryEventModel)
                .where(
                    MemoryEventModel.resource_type == resource_type,
                    MemoryEventModel.resource_id == resource_id,
                )
                .order_by(MemoryEventModel.timestamp.desc())
                .limit(limit)
            )
            result = await session.execute(query)
            return [self._model_to_domain(m) for m in result.scalars().all()]

    async def get_latest_event(
        self,
        resource_type: str,
        resource_id: UUID,
    ) -> MemoryEvent | None:
        """Get the latest event for a resource."""
        async with self._get_session() as session:
            query = (
                select(MemoryEventModel)
                .where(
                    MemoryEventModel.resource_type == resource_type,
                    MemoryEventModel.resource_id == resource_id,
                )
                .order_by(MemoryEventModel.timestamp.desc())
                .limit(1)
            )
            result = await session.execute(query)
            model = result.scalar_one_or_none()
            return self._model_to_domain(model) if model else None

    async def count_events(
        self,
        namespace_id: UUID,
        *,
        event_types: list[str] | None = None,
        after: datetime | None = None,
    ) -> int:
        """Count events matching criteria."""
        async with self._get_session() as session:
            query = select(func.count(MemoryEventModel.id)).where(MemoryEventModel.namespace_id == namespace_id)

            if event_types:
                query = query.where(MemoryEventModel.event_type.in_(event_types))
            if after:
                query = query.where(MemoryEventModel.timestamp > after)

            result = await session.execute(query)
            return result.scalar_one()

    def _model_to_domain(self, model: MemoryEventModel) -> MemoryEvent:
        """Convert MemoryEventModel to domain MemoryEvent."""
        return MemoryEvent(
            id=model.id,
            namespace_id=model.namespace_id,
            event_type=EventType(model.event_type) if isinstance(model.event_type, str) else model.event_type,
            timestamp=model.timestamp,
            resource_type=model.resource_type,
            resource_id=model.resource_id,
            data=model.data,
            previous_data=model.previous_data,
            actor_id=model.actor_id,
            actor_type=model.actor_type,
            correlation_id=model.correlation_id,
            version=model.version,
            metadata=model.metadata_,
        )
