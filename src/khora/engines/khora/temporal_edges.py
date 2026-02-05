"""Temporal edge storage for the Khora engine.

This module implements Graphiti-inspired bi-temporal edge storage:
- Occurrence time: When the fact/event happened
- Ingestion time: When we learned about it
- Validity windows: When the fact is considered true
- Edge invalidation: Tracking contradicting facts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from khora.db.models import TemporalEdgeModel, TimeEdgeLinkModel
from khora.engines.khora.time_hierarchy import TimeHierarchyBuilder

if TYPE_CHECKING:
    pass


@dataclass
class TemporalEdge:
    """In-memory representation of a temporal edge."""

    id: UUID
    namespace_id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    description: str = ""

    # Bi-temporal fields
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Validity tracking
    is_valid: bool = True
    invalidated_by_id: UUID | None = None
    invalidation_reason: str | None = None

    # Source tracking
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    source_document_ids: list[UUID] = field(default_factory=list)
    source_chunk_ids: list[UUID] = field(default_factory=list)


class TemporalEdgeStorage:
    """Storage layer for temporal edges with time hierarchy integration.

    Provides:
    - Creation of timestamped edges (not collapsed like RelationshipModel)
    - Automatic time node linking for efficient temporal queries
    - Edge invalidation for handling contradicting facts
    - Queries by time window, entity pair, or relationship type
    """

    def __init__(self, session: AsyncSession):
        """Initialize the storage.

        Args:
            session: SQLAlchemy async session
        """
        self._session = session
        self._time_builder = TimeHierarchyBuilder(session)

    async def create_edge(
        self,
        edge: TemporalEdge,
        *,
        check_conflicts: bool = True,
    ) -> TemporalEdge:
        """Create a temporal edge.

        Args:
            edge: Edge to create
            check_conflicts: If True, check for and invalidate conflicting edges

        Returns:
            Created edge with ID assigned
        """
        # Check for conflicting edges if requested
        if check_conflicts:
            await self._handle_conflicts(edge)

        # Create the edge
        edge_id = edge.id or uuid4()
        model = TemporalEdgeModel(
            id=str(edge_id),
            namespace_id=str(edge.namespace_id),
            source_entity_id=str(edge.source_entity_id),
            target_entity_id=str(edge.target_entity_id),
            relationship_type=edge.relationship_type,
            description=edge.description,
            occurred_at=edge.occurred_at,
            ingested_at=edge.ingested_at,
            valid_from=edge.valid_from,
            valid_until=edge.valid_until,
            is_valid=edge.is_valid,
            confidence=edge.confidence,
            properties=edge.properties,
            source_document_ids=[str(id) for id in edge.source_document_ids],
            source_chunk_ids=[str(id) for id in edge.source_chunk_ids],
        )

        self._session.add(model)
        await self._session.flush()

        # Link to time hierarchy
        await self._link_to_time_node(edge.namespace_id, edge_id, edge.occurred_at)

        edge.id = edge_id
        logger.debug(
            f"Created temporal edge: {edge.source_entity_id} --[{edge.relationship_type}@{edge.occurred_at}]--> {edge.target_entity_id}"
        )

        return edge

    async def create_edges_batch(
        self,
        edges: list[TemporalEdge],
        *,
        check_conflicts: bool = False,  # Disabled for batch performance
    ) -> list[TemporalEdge]:
        """Create multiple temporal edges in batch.

        Args:
            edges: Edges to create
            check_conflicts: If True, check for conflicts (slower)

        Returns:
            Created edges with IDs assigned
        """
        created = []
        for edge in edges:
            created_edge = await self.create_edge(edge, check_conflicts=check_conflicts)
            created.append(created_edge)

        return created

    async def get_edge(self, edge_id: UUID) -> TemporalEdge | None:
        """Get an edge by ID.

        Args:
            edge_id: Edge ID

        Returns:
            Edge or None if not found
        """
        stmt = select(TemporalEdgeModel).where(TemporalEdgeModel.id == str(edge_id))
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()

        return self._model_to_edge(row) if row else None

    async def invalidate_edge(
        self,
        edge_id: UUID,
        *,
        invalidated_by: UUID | None = None,
        reason: str | None = None,
    ) -> bool:
        """Invalidate an edge (mark as no longer valid).

        Args:
            edge_id: Edge to invalidate
            invalidated_by: New edge that contradicts this one
            reason: Human-readable reason for invalidation

        Returns:
            True if edge was invalidated
        """
        stmt = select(TemporalEdgeModel).where(TemporalEdgeModel.id == str(edge_id))
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()

        if not row:
            return False

        row.is_valid = False
        row.invalidated_by_id = str(invalidated_by) if invalidated_by else None
        row.invalidation_reason = reason

        await self._session.flush()
        logger.debug(f"Invalidated edge {edge_id}: {reason}")

        return True

    async def get_edges_by_time_range(
        self,
        namespace_id: UUID,
        start: datetime,
        end: datetime,
        *,
        relationship_type: str | None = None,
        include_invalid: bool = False,
        limit: int = 1000,
    ) -> list[TemporalEdge]:
        """Get edges within a time range.

        Args:
            namespace_id: Namespace to query
            start: Range start (inclusive)
            end: Range end (exclusive)
            relationship_type: Optional filter by relationship type
            include_invalid: If True, include invalidated edges
            limit: Maximum number of edges to return

        Returns:
            List of edges in chronological order
        """
        conditions = [
            TemporalEdgeModel.namespace_id == str(namespace_id),
            TemporalEdgeModel.occurred_at >= start,
            TemporalEdgeModel.occurred_at < end,
        ]

        if relationship_type:
            conditions.append(TemporalEdgeModel.relationship_type == relationship_type)

        if not include_invalid:
            conditions.append(TemporalEdgeModel.is_valid == True)  # noqa: E712

        stmt = select(TemporalEdgeModel).where(and_(*conditions)).order_by(TemporalEdgeModel.occurred_at).limit(limit)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        return [self._model_to_edge(row) for row in rows]

    async def get_edges_by_entity(
        self,
        entity_id: UUID,
        namespace_id: UUID,
        *,
        direction: str = "both",  # "outgoing", "incoming", "both"
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        include_invalid: bool = False,
        limit: int = 100,
    ) -> list[TemporalEdge]:
        """Get edges connected to an entity.

        Args:
            entity_id: Entity to find edges for
            namespace_id: Namespace to query
            direction: Which edges to find ("outgoing", "incoming", "both")
            time_start: Optional time range start
            time_end: Optional time range end
            include_invalid: If True, include invalidated edges
            limit: Maximum number of edges to return

        Returns:
            List of edges in chronological order
        """
        entity_id_str = str(entity_id)

        if direction == "outgoing":
            entity_condition = TemporalEdgeModel.source_entity_id == entity_id_str
        elif direction == "incoming":
            entity_condition = TemporalEdgeModel.target_entity_id == entity_id_str
        else:  # both
            entity_condition = (TemporalEdgeModel.source_entity_id == entity_id_str) | (
                TemporalEdgeModel.target_entity_id == entity_id_str
            )

        conditions = [
            TemporalEdgeModel.namespace_id == str(namespace_id),
            entity_condition,
        ]

        if time_start:
            conditions.append(TemporalEdgeModel.occurred_at >= time_start)
        if time_end:
            conditions.append(TemporalEdgeModel.occurred_at < time_end)
        if not include_invalid:
            conditions.append(TemporalEdgeModel.is_valid == True)  # noqa: E712

        stmt = select(TemporalEdgeModel).where(and_(*conditions)).order_by(TemporalEdgeModel.occurred_at).limit(limit)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        return [self._model_to_edge(row) for row in rows]

    async def get_edges_by_entity_pair(
        self,
        source_entity_id: UUID,
        target_entity_id: UUID,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        include_invalid: bool = False,
        limit: int = 100,
    ) -> list[TemporalEdge]:
        """Get edges between two specific entities.

        Args:
            source_entity_id: Source entity
            target_entity_id: Target entity
            namespace_id: Namespace to query
            relationship_type: Optional filter by relationship type
            include_invalid: If True, include invalidated edges
            limit: Maximum number of edges to return

        Returns:
            List of edges in chronological order
        """
        conditions = [
            TemporalEdgeModel.namespace_id == str(namespace_id),
            TemporalEdgeModel.source_entity_id == str(source_entity_id),
            TemporalEdgeModel.target_entity_id == str(target_entity_id),
        ]

        if relationship_type:
            conditions.append(TemporalEdgeModel.relationship_type == relationship_type)
        if not include_invalid:
            conditions.append(TemporalEdgeModel.is_valid == True)  # noqa: E712

        stmt = select(TemporalEdgeModel).where(and_(*conditions)).order_by(TemporalEdgeModel.occurred_at).limit(limit)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        return [self._model_to_edge(row) for row in rows]

    async def get_valid_at(
        self,
        namespace_id: UUID,
        point_in_time: datetime,
        *,
        entity_id: UUID | None = None,
        relationship_type: str | None = None,
        limit: int = 1000,
    ) -> list[TemporalEdge]:
        """Get edges that were valid at a specific point in time.

        Uses the valid_from/valid_until fields to determine validity.

        Args:
            namespace_id: Namespace to query
            point_in_time: The point in time to query
            entity_id: Optional filter by entity
            relationship_type: Optional filter by relationship type
            limit: Maximum number of edges to return

        Returns:
            List of edges valid at that time
        """
        conditions = [
            TemporalEdgeModel.namespace_id == str(namespace_id),
            TemporalEdgeModel.is_valid == True,  # noqa: E712
            # valid_from <= point_in_time < valid_until
            (TemporalEdgeModel.valid_from.is_(None)) | (TemporalEdgeModel.valid_from <= point_in_time),
            (TemporalEdgeModel.valid_until.is_(None)) | (TemporalEdgeModel.valid_until > point_in_time),
        ]

        if entity_id:
            conditions.append(
                (TemporalEdgeModel.source_entity_id == str(entity_id))
                | (TemporalEdgeModel.target_entity_id == str(entity_id))
            )

        if relationship_type:
            conditions.append(TemporalEdgeModel.relationship_type == relationship_type)

        stmt = select(TemporalEdgeModel).where(and_(*conditions)).order_by(TemporalEdgeModel.occurred_at).limit(limit)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        return [self._model_to_edge(row) for row in rows]

    async def delete_edge(self, edge_id: UUID) -> bool:
        """Delete an edge.

        Args:
            edge_id: Edge to delete

        Returns:
            True if deleted
        """
        stmt = select(TemporalEdgeModel).where(TemporalEdgeModel.id == str(edge_id))
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()

        if not row:
            return False

        await self._session.delete(row)
        await self._session.flush()

        return True

    async def delete_edges_by_entity(self, entity_id: UUID, namespace_id: UUID) -> int:
        """Delete all edges connected to an entity.

        Args:
            entity_id: Entity whose edges to delete
            namespace_id: Namespace

        Returns:
            Number of edges deleted
        """
        stmt = select(TemporalEdgeModel).where(
            TemporalEdgeModel.namespace_id == str(namespace_id),
            (TemporalEdgeModel.source_entity_id == str(entity_id))
            | (TemporalEdgeModel.target_entity_id == str(entity_id)),
        )

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        for row in rows:
            await self._session.delete(row)

        await self._session.flush()
        return len(rows)

    # =========================================================================
    # Private helper methods
    # =========================================================================

    async def _link_to_time_node(
        self,
        namespace_id: UUID,
        edge_id: UUID,
        occurred_at: datetime,
    ) -> None:
        """Link an edge to its time node in the hierarchy."""
        # Get or create the day node
        day_node = await self._time_builder.get_or_create_day_node(namespace_id, occurred_at)

        # Create the link
        link = TimeEdgeLinkModel(
            time_node_id=str(day_node.id),
            edge_id=str(edge_id),
        )
        self._session.add(link)

        # Update edge count
        await self._time_builder.increment_edge_count(day_node.id)

        await self._session.flush()

    async def _handle_conflicts(self, new_edge: TemporalEdge) -> None:
        """Check for and handle conflicting edges.

        A conflict is detected when:
        - Same source/target entities
        - Same relationship type
        - New edge occurs after existing edge
        - The relationship type implies mutual exclusivity (e.g., WORKS_FOR)
        """
        # Get existing edges between the same entity pair
        existing = await self.get_edges_by_entity_pair(
            new_edge.source_entity_id,
            new_edge.target_entity_id,
            new_edge.namespace_id,
            relationship_type=new_edge.relationship_type,
            include_invalid=False,
        )

        # Check for conflicts based on relationship semantics
        exclusive_types = {
            "WORKS_FOR",
            "REPORTS_TO",
            "MANAGES",
            "MARRIED_TO",
            "CEO_OF",
            "PRESIDENT_OF",
            "LOCATED_AT",
            "HEADQUARTERED_IN",
        }

        if new_edge.relationship_type.upper() in exclusive_types:
            for old_edge in existing:
                # If new edge is more recent, invalidate old one
                if new_edge.occurred_at > old_edge.occurred_at:
                    await self.invalidate_edge(
                        old_edge.id,
                        invalidated_by=new_edge.id,
                        reason=f"Superseded by newer {new_edge.relationship_type} edge",
                    )

    def _model_to_edge(self, model: TemporalEdgeModel) -> TemporalEdge:
        """Convert a database model to a TemporalEdge dataclass."""
        return TemporalEdge(
            id=UUID(model.id),
            namespace_id=UUID(model.namespace_id),
            source_entity_id=UUID(model.source_entity_id),
            target_entity_id=UUID(model.target_entity_id),
            relationship_type=model.relationship_type,
            description=model.description,
            occurred_at=model.occurred_at,
            ingested_at=model.ingested_at,
            valid_from=model.valid_from,
            valid_until=model.valid_until,
            is_valid=model.is_valid,
            invalidated_by_id=UUID(model.invalidated_by_id) if model.invalidated_by_id else None,
            invalidation_reason=model.invalidation_reason,
            confidence=model.confidence,
            properties=model.properties or {},
            source_document_ids=[UUID(id) for id in (model.source_document_ids or [])],
            source_chunk_ids=[UUID(id) for id in (model.source_chunk_ids or [])],
        )


__all__ = ["TemporalEdge", "TemporalEdgeStorage"]
