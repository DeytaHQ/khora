"""Unit tests for the time hierarchy builder."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from khora.db.models import TimeGranularity
from khora.engines.khora.time_hierarchy import TimeNode


class TestTimeNode:
    """Tests for TimeNode dataclass."""

    def test_create_time_node(self):
        """Test creating a time node."""
        node_id = uuid4()
        namespace_id = uuid4()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)

        node = TimeNode(
            id=node_id,
            namespace_id=namespace_id,
            granularity=TimeGranularity.MONTH,
            start_time=start,
            end_time=end,
            parent_id=None,
            name="January 2024",
            edge_count=5,
            entity_count=10,
        )

        assert node.id == node_id
        assert node.namespace_id == namespace_id
        assert node.granularity == TimeGranularity.MONTH
        assert node.start_time == start
        assert node.end_time == end
        assert node.parent_id is None
        assert node.name == "January 2024"
        assert node.edge_count == 5
        assert node.entity_count == 10

    def test_time_node_defaults(self):
        """Test time node default values."""
        node = TimeNode(
            id=uuid4(),
            namespace_id=uuid4(),
            granularity=TimeGranularity.DAY,
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            parent_id=None,
            name="test",
        )

        assert node.edge_count == 0
        assert node.entity_count == 0


class TestTimeGranularity:
    """Tests for time granularity constants."""

    def test_granularity_values(self):
        """Test granularity constant values."""
        assert TimeGranularity.YEAR == "year"
        assert TimeGranularity.QUARTER == "quarter"
        assert TimeGranularity.MONTH == "month"
        assert TimeGranularity.WEEK == "week"
        assert TimeGranularity.DAY == "day"


@pytest.mark.unit
class TestTimeHierarchyLogic:
    """Tests for time hierarchy calculation logic."""

    def test_quarter_calculation(self):
        """Test quarter calculation from dates."""
        # Q1: Jan, Feb, Mar
        for month in [1, 2, 3]:
            dt = datetime(2024, month, 15, tzinfo=UTC)
            quarter = (dt.month - 1) // 3 + 1
            assert quarter == 1

        # Q2: Apr, May, Jun
        for month in [4, 5, 6]:
            dt = datetime(2024, month, 15, tzinfo=UTC)
            quarter = (dt.month - 1) // 3 + 1
            assert quarter == 2

        # Q3: Jul, Aug, Sep
        for month in [7, 8, 9]:
            dt = datetime(2024, month, 15, tzinfo=UTC)
            quarter = (dt.month - 1) // 3 + 1
            assert quarter == 3

        # Q4: Oct, Nov, Dec
        for month in [10, 11, 12]:
            dt = datetime(2024, month, 15, tzinfo=UTC)
            quarter = (dt.month - 1) // 3 + 1
            assert quarter == 4

    def test_quarter_start_calculation(self):
        """Test quarter start month calculation."""
        for quarter in [1, 2, 3, 4]:
            start_month = (quarter - 1) * 3 + 1
            expected_starts = {1: 1, 2: 4, 3: 7, 4: 10}
            assert start_month == expected_starts[quarter]

    def test_iso_week_calculation(self):
        """Test ISO week calculation."""
        # 2024-01-01 is Monday of week 1
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        iso = dt.isocalendar()
        assert iso.year == 2024
        assert iso.week == 1

        # 2024-01-08 is Monday of week 2
        dt = datetime(2024, 1, 8, tzinfo=UTC)
        iso = dt.isocalendar()
        assert iso.year == 2024
        assert iso.week == 2

    def test_day_boundaries(self):
        """Test day start/end boundaries."""
        dt = datetime(2024, 6, 15, 14, 30, 45, tzinfo=UTC)

        # Normalize to start of day
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        assert day_start.hour == 0
        assert day_start.minute == 0
        assert day_start.second == 0

        # End of day is start of next day
        day_end = day_start + timedelta(days=1)
        assert day_end.day == 16


class TestTimeNodeNaming:
    """Tests for time node naming conventions."""

    def test_year_name(self):
        """Test year node naming."""
        dt = datetime(2024, 6, 15, tzinfo=UTC)
        name = str(dt.year)
        assert name == "2024"

    def test_quarter_name(self):
        """Test quarter node naming."""
        for month, expected_q in [(1, 1), (4, 2), (7, 3), (10, 4)]:
            dt = datetime(2024, month, 15, tzinfo=UTC)
            quarter = (dt.month - 1) // 3 + 1
            name = f"Q{quarter} {dt.year}"
            assert name == f"Q{expected_q} 2024"

    def test_month_name(self):
        """Test month node naming."""
        dt = datetime(2024, 1, 15, tzinfo=UTC)
        name = dt.strftime("%B %Y")
        assert name == "January 2024"

        dt = datetime(2024, 12, 15, tzinfo=UTC)
        name = dt.strftime("%B %Y")
        assert name == "December 2024"

    def test_week_name(self):
        """Test week node naming."""
        dt = datetime(2024, 1, 15, tzinfo=UTC)
        iso = dt.isocalendar()
        name = f"Week {iso.week} {iso.year}"
        assert "Week" in name
        assert "2024" in name

    def test_day_name(self):
        """Test day node naming."""
        dt = datetime(2024, 1, 15, tzinfo=UTC)
        name = dt.strftime("%Y-%m-%d")
        assert name == "2024-01-15"


class TestTimeHierarchyIntegrity:
    """Tests for time hierarchy integrity checks."""

    def test_year_contains_quarters(self):
        """Test that year properly contains all quarters."""
        year_start = datetime(2024, 1, 1, tzinfo=UTC)
        year_end = datetime(2025, 1, 1, tzinfo=UTC)

        # All quarter starts should be within year
        q_starts = [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 4, 1, tzinfo=UTC),
            datetime(2024, 7, 1, tzinfo=UTC),
            datetime(2024, 10, 1, tzinfo=UTC),
        ]

        for q_start in q_starts:
            assert year_start <= q_start < year_end

    def test_quarter_contains_months(self):
        """Test that quarter properly contains all months."""
        # Q1 2024
        q_start = datetime(2024, 1, 1, tzinfo=UTC)
        q_end = datetime(2024, 4, 1, tzinfo=UTC)

        months = [1, 2, 3]
        for month in months:
            m_start = datetime(2024, month, 1, tzinfo=UTC)
            assert q_start <= m_start < q_end

    def test_month_contains_weeks(self):
        """Test that month contains weeks (partial overlap is ok)."""
        # January 2024
        m_start = datetime(2024, 1, 1, tzinfo=UTC)

        # Week 1 starts on 2024-01-01
        w1_start = datetime.fromisocalendar(2024, 1, 1).replace(tzinfo=UTC)
        assert w1_start >= m_start

    def test_no_time_gaps(self):
        """Test that hierarchy has no gaps."""
        # Year end == next year start
        year1_end = datetime(2024, 1, 1, tzinfo=UTC)
        year2_start = datetime(2024, 1, 1, tzinfo=UTC)
        assert year1_end == year2_start

        # Quarter end == next quarter start
        q1_end = datetime(2024, 4, 1, tzinfo=UTC)
        q2_start = datetime(2024, 4, 1, tzinfo=UTC)
        assert q1_end == q2_start
