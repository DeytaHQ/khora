"""Unit tests for temporal database models."""

from datetime import UTC, datetime
from uuid import uuid4

from khora.db.models import (
    TemporalEdgeModel,
    TimeEdgeLinkModel,
    TimeGranularity,
    TimeNodeModel,
)


class TestTimeGranularityConstants:
    """Test TimeGranularity class constants."""

    def test_year(self):
        assert TimeGranularity.YEAR == "year"

    def test_quarter(self):
        assert TimeGranularity.QUARTER == "quarter"

    def test_month(self):
        assert TimeGranularity.MONTH == "month"

    def test_week(self):
        assert TimeGranularity.WEEK == "week"

    def test_day(self):
        assert TimeGranularity.DAY == "day"


class TestTimeNodeModel:
    """Tests for TimeNodeModel."""

    def test_repr(self):
        """Test string representation."""
        node = TimeNodeModel(
            id=str(uuid4()),
            namespace_id=str(uuid4()),
            granularity=TimeGranularity.MONTH,
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            end_time=datetime(2024, 2, 1, tzinfo=UTC),
            name="January 2024",
        )
        repr_str = repr(node)
        assert "TimeNode" in repr_str
        assert "January 2024" in repr_str
        assert "month" in repr_str


class TestTemporalEdgeModel:
    """Tests for TemporalEdgeModel."""

    def test_repr(self):
        """Test string representation."""
        now = datetime.now(UTC)
        edge = TemporalEdgeModel(
            id=str(uuid4()),
            namespace_id=str(uuid4()),
            source_entity_id=str(uuid4()),
            target_entity_id=str(uuid4()),
            relationship_type="WORKS_FOR",
            occurred_at=now,
        )
        repr_str = repr(edge)
        assert "TemporalEdge" in repr_str
        assert "WORKS_FOR" in repr_str


class TestTimeEdgeLinkModel:
    """Tests for TimeEdgeLinkModel."""

    def test_repr(self):
        """Test string representation."""
        time_node_id = str(uuid4())
        edge_id = str(uuid4())
        link = TimeEdgeLinkModel(
            time_node_id=time_node_id,
            edge_id=edge_id,
        )
        repr_str = repr(link)
        assert "TimeEdgeLink" in repr_str
        assert time_node_id in repr_str
        assert edge_id in repr_str
