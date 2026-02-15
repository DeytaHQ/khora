"""Hierarchical time graph builder for the Khora engine.

This module implements TG-RAG-inspired time hierarchy for efficient temporal navigation:
- Year → Quarter → Month → Week → Day structure
- Auto-creation of time nodes on demand
- Navigation methods for temporal queries
- Lazy summary embedding computation
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from khora.db.models import TimeGranularity, TimeNodeModel

if TYPE_CHECKING:
    pass


@dataclass
class TimeNode:
    """In-memory representation of a time node."""

    id: UUID
    namespace_id: UUID
    granularity: str
    start_time: datetime
    end_time: datetime
    parent_id: UUID | None
    name: str
    edge_count: int = 0
    entity_count: int = 0


class TimeHierarchyBuilder:
    """Builds and navigates hierarchical time graphs.

    The time hierarchy enables efficient temporal queries by organizing time into
    a tree structure:

        2024 (year)
        ├── Q1 2024 (quarter)
        │   ├── January 2024 (month)
        │   │   ├── Week 1 (week)
        │   │   │   ├── 2024-01-01 (day)
        │   │   │   ├── 2024-01-02 (day)
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        └── ...

    This allows:
    - Fast range queries ("what happened in Q1 2024?")
    - Drill-down from coarse to fine granularity
    - Roll-up summaries for high-level temporal views
    """

    def __init__(self, session: AsyncSession):
        """Initialize the builder.

        Args:
            session: SQLAlchemy async session
        """
        self._session = session

    async def get_or_create_day_node(self, namespace_id: UUID, dt: datetime) -> TimeNode:
        """Get or create a day node for the given datetime.

        This also ensures all ancestor nodes (week, month, quarter, year) exist.

        Args:
            namespace_id: Namespace for the time nodes
            dt: Datetime to create node for (uses date portion only)

        Returns:
            Day-level TimeNode
        """
        # Normalize to start of day in UTC
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if day_start.tzinfo is None:
            day_start = day_start.replace(tzinfo=UTC)
        day_end = day_start + timedelta(days=1)

        # Check if day node exists
        existing = await self._get_node(namespace_id, TimeGranularity.DAY, day_start)
        if existing:
            return existing

        # Ensure ancestors exist first (called for side effects)
        week_node = await self._ensure_week_node(namespace_id, day_start)
        await self._ensure_month_node(namespace_id, day_start)
        await self._ensure_quarter_node(namespace_id, day_start)
        await self._ensure_year_node(namespace_id, day_start)

        # Create day node (parent is the week)
        name = day_start.strftime("%Y-%m-%d")
        day_node = await self._create_node(
            namespace_id=namespace_id,
            granularity=TimeGranularity.DAY,
            start_time=day_start,
            end_time=day_end,
            parent_id=week_node.id,
            name=name,
        )

        logger.debug(f"Created time node: {name} (day) under week {week_node.name}")
        return day_node

    async def get_nodes_in_range(
        self,
        namespace_id: UUID,
        start: datetime,
        end: datetime,
        granularity: str = TimeGranularity.DAY,
    ) -> list[TimeNode]:
        """Get all time nodes within a date range.

        Args:
            namespace_id: Namespace to query
            start: Range start (inclusive)
            end: Range end (exclusive)
            granularity: Level of granularity to return

        Returns:
            List of TimeNodes in chronological order
        """
        stmt = (
            select(TimeNodeModel)
            .where(
                TimeNodeModel.namespace_id == namespace_id,
                TimeNodeModel.granularity == granularity,
                TimeNodeModel.start_time >= start,
                TimeNodeModel.start_time < end,
            )
            .order_by(TimeNodeModel.start_time)
        )

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        return [self._model_to_node(row) for row in rows]

    async def get_children(self, node: TimeNode) -> list[TimeNode]:
        """Get child nodes of a time node.

        Args:
            node: Parent time node

        Returns:
            List of child TimeNodes in chronological order
        """
        stmt = select(TimeNodeModel).where(TimeNodeModel.parent_id == node.id).order_by(TimeNodeModel.start_time)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        return [self._model_to_node(row) for row in rows]

    async def get_parent(self, node: TimeNode) -> TimeNode | None:
        """Get parent node of a time node.

        Args:
            node: Child time node

        Returns:
            Parent TimeNode or None if at root level
        """
        if node.parent_id is None:
            return None

        stmt = select(TimeNodeModel).where(TimeNodeModel.id == node.parent_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()

        return self._model_to_node(row) if row else None

    async def get_covering_node(
        self,
        namespace_id: UUID,
        start: datetime,
        end: datetime,
    ) -> TimeNode | None:
        """Get the smallest time node that fully covers a date range.

        Useful for finding the appropriate level for a temporal query.

        Args:
            namespace_id: Namespace to query
            start: Range start
            end: Range end

        Returns:
            TimeNode that covers the range, or None
        """
        # Try from coarsest to finest granularity
        for granularity in [
            TimeGranularity.YEAR,
            TimeGranularity.QUARTER,
            TimeGranularity.MONTH,
            TimeGranularity.WEEK,
            TimeGranularity.DAY,
        ]:
            stmt = (
                select(TimeNodeModel)
                .where(
                    TimeNodeModel.namespace_id == namespace_id,
                    TimeNodeModel.granularity == granularity,
                    TimeNodeModel.start_time <= start,
                    TimeNodeModel.end_time >= end,
                )
                .limit(1)
            )

            result = await self._session.execute(stmt)
            row = result.scalar_one_or_none()

            if row:
                return self._model_to_node(row)

        return None

    async def increment_edge_count(self, node_id: UUID, delta: int = 1) -> None:
        """Increment the edge count for a time node.

        Args:
            node_id: Time node ID
            delta: Amount to increment by
        """
        stmt = select(TimeNodeModel).where(TimeNodeModel.id == node_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()

        if row:
            row.edge_count = (row.edge_count or 0) + delta
            await self._session.flush()

    async def increment_entity_count(self, node_id: UUID, delta: int = 1) -> None:
        """Increment the entity count for a time node.

        Args:
            node_id: Time node ID
            delta: Amount to increment by
        """
        stmt = select(TimeNodeModel).where(TimeNodeModel.id == node_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()

        if row:
            row.entity_count = (row.entity_count or 0) + delta
            await self._session.flush()

    # =========================================================================
    # Private helper methods
    # =========================================================================

    async def _get_node(
        self,
        namespace_id: UUID,
        granularity: str,
        start_time: datetime,
    ) -> TimeNode | None:
        """Get a node by namespace, granularity, and start time."""
        stmt = select(TimeNodeModel).where(
            TimeNodeModel.namespace_id == namespace_id,
            TimeNodeModel.granularity == granularity,
            TimeNodeModel.start_time == start_time,
        )

        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()

        return self._model_to_node(row) if row else None

    async def _create_node(
        self,
        namespace_id: UUID,
        granularity: str,
        start_time: datetime,
        end_time: datetime,
        parent_id: UUID | None,
        name: str,
    ) -> TimeNode:
        """Create a new time node."""
        node_id = uuid4()
        model = TimeNodeModel(
            id=node_id,
            namespace_id=namespace_id,
            granularity=granularity,
            start_time=start_time,
            end_time=end_time,
            parent_id=parent_id,
            name=name,
            edge_count=0,
            entity_count=0,
        )

        self._session.add(model)
        await self._session.flush()

        return TimeNode(
            id=node_id,
            namespace_id=namespace_id,
            granularity=granularity,
            start_time=start_time,
            end_time=end_time,
            parent_id=parent_id,
            name=name,
        )

    async def _ensure_year_node(self, namespace_id: UUID, dt: datetime) -> TimeNode:
        """Ensure a year node exists."""
        year_start = datetime(dt.year, 1, 1, tzinfo=UTC)
        year_end = datetime(dt.year + 1, 1, 1, tzinfo=UTC)

        existing = await self._get_node(namespace_id, TimeGranularity.YEAR, year_start)
        if existing:
            return existing

        name = str(dt.year)
        return await self._create_node(
            namespace_id=namespace_id,
            granularity=TimeGranularity.YEAR,
            start_time=year_start,
            end_time=year_end,
            parent_id=None,  # Year is root
            name=name,
        )

    async def _ensure_quarter_node(self, namespace_id: UUID, dt: datetime) -> TimeNode:
        """Ensure a quarter node exists."""
        quarter = (dt.month - 1) // 3 + 1
        quarter_start_month = (quarter - 1) * 3 + 1
        quarter_start = datetime(dt.year, quarter_start_month, 1, tzinfo=UTC)

        if quarter == 4:
            quarter_end = datetime(dt.year + 1, 1, 1, tzinfo=UTC)
        else:
            quarter_end = datetime(dt.year, quarter_start_month + 3, 1, tzinfo=UTC)

        existing = await self._get_node(namespace_id, TimeGranularity.QUARTER, quarter_start)
        if existing:
            return existing

        # Ensure year parent exists
        year_node = await self._ensure_year_node(namespace_id, dt)

        name = f"Q{quarter} {dt.year}"
        return await self._create_node(
            namespace_id=namespace_id,
            granularity=TimeGranularity.QUARTER,
            start_time=quarter_start,
            end_time=quarter_end,
            parent_id=year_node.id,
            name=name,
        )

    async def _ensure_month_node(self, namespace_id: UUID, dt: datetime) -> TimeNode:
        """Ensure a month node exists."""
        month_start = datetime(dt.year, dt.month, 1, tzinfo=UTC)
        _, last_day = monthrange(dt.year, dt.month)
        month_end = month_start + timedelta(days=last_day)

        existing = await self._get_node(namespace_id, TimeGranularity.MONTH, month_start)
        if existing:
            return existing

        # Ensure quarter parent exists
        quarter_node = await self._ensure_quarter_node(namespace_id, dt)

        name = dt.strftime("%B %Y")
        return await self._create_node(
            namespace_id=namespace_id,
            granularity=TimeGranularity.MONTH,
            start_time=month_start,
            end_time=month_end,
            parent_id=quarter_node.id,
            name=name,
        )

    async def _ensure_week_node(self, namespace_id: UUID, dt: datetime) -> TimeNode:
        """Ensure a week node exists (ISO week)."""
        # Get ISO week start (Monday)
        iso_calendar = dt.isocalendar()
        week_start = datetime.fromisocalendar(iso_calendar[0], iso_calendar[1], 1).replace(tzinfo=UTC)
        week_end = week_start + timedelta(days=7)

        existing = await self._get_node(namespace_id, TimeGranularity.WEEK, week_start)
        if existing:
            return existing

        # Ensure month parent exists (use week start date's month)
        month_node = await self._ensure_month_node(namespace_id, week_start)

        name = f"Week {iso_calendar[1]} {iso_calendar[0]}"
        return await self._create_node(
            namespace_id=namespace_id,
            granularity=TimeGranularity.WEEK,
            start_time=week_start,
            end_time=week_end,
            parent_id=month_node.id,
            name=name,
        )

    def _model_to_node(self, model: TimeNodeModel) -> TimeNode:
        """Convert a database model to a TimeNode dataclass."""
        return TimeNode(
            id=model.id,
            namespace_id=model.namespace_id,
            granularity=model.granularity,
            start_time=model.start_time,
            end_time=model.end_time,
            parent_id=model.parent_id,
            name=model.name,
            edge_count=model.edge_count or 0,
            entity_count=model.entity_count or 0,
        )


__all__ = ["TimeHierarchyBuilder", "TimeNode"]
