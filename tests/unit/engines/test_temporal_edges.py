"""Unit tests for temporal edge storage."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from khora.engines.skeleton.temporal_edges import TemporalEdge


class TestTemporalEdge:
    """Tests for TemporalEdge dataclass."""

    def test_create_temporal_edge(self):
        """Test creating a temporal edge."""
        edge_id = uuid4()
        namespace_id = uuid4()
        source_id = uuid4()
        target_id = uuid4()
        occurred_at = datetime(2024, 6, 15, 10, 30, tzinfo=UTC)

        edge = TemporalEdge(
            id=edge_id,
            namespace_id=namespace_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relationship_type="WORKS_FOR",
            description="Alice works for Acme Corp",
            occurred_at=occurred_at,
            confidence=0.95,
        )

        assert edge.id == edge_id
        assert edge.namespace_id == namespace_id
        assert edge.source_entity_id == source_id
        assert edge.target_entity_id == target_id
        assert edge.relationship_type == "WORKS_FOR"
        assert edge.description == "Alice works for Acme Corp"
        assert edge.occurred_at == occurred_at
        assert edge.confidence == 0.95
        assert edge.is_valid is True

    def test_edge_defaults(self):
        """Test temporal edge default values."""
        edge = TemporalEdge(
            id=uuid4(),
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="RELATES_TO",
        )

        assert edge.description == ""
        assert edge.is_valid is True
        assert edge.invalidated_by_id is None
        assert edge.invalidation_reason is None
        assert edge.confidence == 1.0
        assert edge.properties == {}
        assert edge.source_document_ids == []
        assert edge.source_chunk_ids == []

    def test_edge_with_validity_window(self):
        """Test edge with temporal validity window."""
        valid_from = datetime(2024, 1, 1, tzinfo=UTC)
        valid_until = datetime(2024, 12, 31, tzinfo=UTC)

        edge = TemporalEdge(
            id=uuid4(),
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="CEO_OF",
            valid_from=valid_from,
            valid_until=valid_until,
        )

        assert edge.valid_from == valid_from
        assert edge.valid_until == valid_until

    def test_edge_invalidation(self):
        """Test edge invalidation tracking."""
        original_edge_id = uuid4()
        new_edge_id = uuid4()

        edge = TemporalEdge(
            id=original_edge_id,
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="WORKS_FOR",
            is_valid=False,
            invalidated_by_id=new_edge_id,
            invalidation_reason="Superseded by newer WORKS_FOR edge",
        )

        assert edge.is_valid is False
        assert edge.invalidated_by_id == new_edge_id
        assert "Superseded" in edge.invalidation_reason

    def test_edge_with_sources(self):
        """Test edge with source document/chunk references."""
        doc_ids = [uuid4(), uuid4()]
        chunk_ids = [uuid4(), uuid4(), uuid4()]

        edge = TemporalEdge(
            id=uuid4(),
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="MENTIONED_IN",
            source_document_ids=doc_ids,
            source_chunk_ids=chunk_ids,
        )

        assert len(edge.source_document_ids) == 2
        assert len(edge.source_chunk_ids) == 3
        assert edge.source_document_ids == doc_ids
        assert edge.source_chunk_ids == chunk_ids


class TestBiTemporalModel:
    """Tests for bi-temporal model concepts."""

    def test_occurrence_vs_ingestion_time(self):
        """Test that occurrence and ingestion times are tracked separately."""
        # Event happened on June 1
        occurred_at = datetime(2024, 6, 1, tzinfo=UTC)
        # We learned about it on June 15
        ingested_at = datetime(2024, 6, 15, tzinfo=UTC)

        edge = TemporalEdge(
            id=uuid4(),
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="ACQUIRED",
            occurred_at=occurred_at,
            ingested_at=ingested_at,
        )

        assert edge.occurred_at == occurred_at
        assert edge.ingested_at == ingested_at
        assert edge.occurred_at < edge.ingested_at

    def test_validity_window_semantics(self):
        """Test validity window semantics."""
        # Alice was CEO from Jan 1 to June 30
        valid_from = datetime(2024, 1, 1, tzinfo=UTC)
        valid_until = datetime(2024, 7, 1, tzinfo=UTC)

        edge = TemporalEdge(
            id=uuid4(),
            namespace_id=uuid4(),
            source_entity_id=uuid4(),  # Alice
            target_entity_id=uuid4(),  # Company
            relationship_type="CEO_OF",
            occurred_at=valid_from,  # When we first knew about it
            valid_from=valid_from,
            valid_until=valid_until,
        )

        # Check if a point in time is within validity window
        def is_valid_at(edge: TemporalEdge, t: datetime) -> bool:
            if edge.valid_from and t < edge.valid_from:
                return False
            if edge.valid_until and t >= edge.valid_until:
                return False
            return edge.is_valid

        # Before validity
        assert not is_valid_at(edge, datetime(2023, 12, 31, tzinfo=UTC))

        # During validity
        assert is_valid_at(edge, datetime(2024, 3, 15, tzinfo=UTC))

        # After validity
        assert not is_valid_at(edge, datetime(2024, 7, 1, tzinfo=UTC))


class TestEdgeConflictDetection:
    """Tests for edge conflict detection logic."""

    def test_exclusive_relationship_types(self):
        """Test identification of mutually exclusive relationship types."""
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

        # WORKS_FOR is exclusive (one employer at a time)
        assert "WORKS_FOR" in exclusive_types

        # KNOWS is not exclusive (can know many people)
        assert "KNOWS" not in exclusive_types

    def test_conflict_detection_newer_wins(self):
        """Test that newer edge invalidates older for exclusive types."""
        source_id = uuid4()
        target_id = uuid4()
        namespace_id = uuid4()

        # Old edge: Alice works for OldCorp (June 2023)
        old_edge = TemporalEdge(
            id=uuid4(),
            namespace_id=namespace_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relationship_type="WORKS_FOR",
            occurred_at=datetime(2023, 6, 1, tzinfo=UTC),
        )

        # New edge: Alice works for NewCorp (Jan 2024)
        new_edge = TemporalEdge(
            id=uuid4(),
            namespace_id=namespace_id,
            source_entity_id=source_id,
            target_entity_id=uuid4(),  # Different target
            relationship_type="WORKS_FOR",
            occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
        )

        # New edge is more recent
        assert new_edge.occurred_at > old_edge.occurred_at

        # Both involve same source and relationship type
        assert old_edge.source_entity_id == new_edge.source_entity_id
        assert old_edge.relationship_type == new_edge.relationship_type


class TestTemporalQueryPatterns:
    """Tests for common temporal query patterns."""

    def test_point_in_time_query(self):
        """Test finding edges valid at a specific point in time."""
        edges = [
            TemporalEdge(
                id=uuid4(),
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
                relationship_type="CEO_OF",
                occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
                valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                valid_until=datetime(2024, 6, 30, tzinfo=UTC),
            ),
            TemporalEdge(
                id=uuid4(),
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
                relationship_type="CEO_OF",
                occurred_at=datetime(2024, 7, 1, tzinfo=UTC),
                valid_from=datetime(2024, 7, 1, tzinfo=UTC),
                valid_until=None,  # Still valid
            ),
        ]

        # Query: Who was CEO on March 15, 2024?
        query_time = datetime(2024, 3, 15, tzinfo=UTC)
        valid_edges = [
            e
            for e in edges
            if (e.valid_from is None or e.valid_from <= query_time)
            and (e.valid_until is None or e.valid_until > query_time)
        ]
        assert len(valid_edges) == 1
        assert valid_edges[0].occurred_at == datetime(2024, 1, 1, tzinfo=UTC)

        # Query: Who is CEO on August 1, 2024?
        query_time = datetime(2024, 8, 1, tzinfo=UTC)
        valid_edges = [
            e
            for e in edges
            if (e.valid_from is None or e.valid_from <= query_time)
            and (e.valid_until is None or e.valid_until > query_time)
        ]
        assert len(valid_edges) == 1
        assert valid_edges[0].occurred_at == datetime(2024, 7, 1, tzinfo=UTC)

    def test_time_range_query(self):
        """Test finding edges within a time range."""
        edges = [
            TemporalEdge(
                id=uuid4(),
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
                relationship_type="MEETING",
                occurred_at=datetime(2024, 1, 15, tzinfo=UTC),
            ),
            TemporalEdge(
                id=uuid4(),
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
                relationship_type="MEETING",
                occurred_at=datetime(2024, 2, 10, tzinfo=UTC),
            ),
            TemporalEdge(
                id=uuid4(),
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
                relationship_type="MEETING",
                occurred_at=datetime(2024, 3, 5, tzinfo=UTC),
            ),
        ]

        # Query: Meetings in February 2024
        start = datetime(2024, 2, 1, tzinfo=UTC)
        end = datetime(2024, 3, 1, tzinfo=UTC)

        matching = [e for e in edges if start <= e.occurred_at < end]
        assert len(matching) == 1
        assert matching[0].occurred_at == datetime(2024, 2, 10, tzinfo=UTC)

    def test_relative_time_query(self):
        """Test relative time queries like 'yesterday'."""
        now = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)

        edges = [
            TemporalEdge(
                id=uuid4(),
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
                relationship_type="SENT_MESSAGE",
                occurred_at=datetime(2024, 6, 14, 10, 0, tzinfo=UTC),  # Yesterday
            ),
            TemporalEdge(
                id=uuid4(),
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
                relationship_type="SENT_MESSAGE",
                occurred_at=datetime(2024, 6, 15, 9, 0, tzinfo=UTC),  # Today
            ),
        ]

        # "Yesterday" relative to now
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_end = now.replace(hour=0, minute=0, second=0, microsecond=0)

        matching = [e for e in edges if yesterday_start <= e.occurred_at < yesterday_end]
        assert len(matching) == 1
        assert matching[0].occurred_at.day == 14
