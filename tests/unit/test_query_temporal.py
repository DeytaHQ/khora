"""Unit tests for query/temporal.py — TemporalFilter and TemporalQuery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from khora.query.temporal import TemporalFilter, TemporalOperator, TemporalQuery


class TestTemporalOperator:
    """Tests for TemporalOperator enum."""

    def test_values(self) -> None:
        """All expected operators exist."""
        assert TemporalOperator.BEFORE == "before"
        assert TemporalOperator.AFTER == "after"
        assert TemporalOperator.BETWEEN == "between"
        assert TemporalOperator.DURING == "during"
        assert TemporalOperator.OVERLAPS == "overlaps"


class TestTemporalFilter:
    """Tests for TemporalFilter."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        f = TemporalFilter()
        assert f.operator == TemporalOperator.AFTER
        assert f.start_time is None
        assert f.end_time is None
        assert f.relative_days is None
        assert f.relative_hours is None

    def test_date_aliases(self) -> None:
        """start_date and end_date aliases work."""
        now = datetime.now()
        f = TemporalFilter(start_date=now)
        assert f.start_time == now
        assert f.start_date == now

    def test_auto_detect_between(self) -> None:
        """Providing both start and end auto-detects BETWEEN operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        f = TemporalFilter(start_time=start, end_time=end)
        assert f.operator == TemporalOperator.BETWEEN

    def test_auto_detect_before(self) -> None:
        """Providing only end_time auto-detects BEFORE operator."""
        end = datetime(2024, 6, 1)
        f = TemporalFilter(end_time=end)
        assert f.operator == TemporalOperator.BEFORE

    def test_last_days(self) -> None:
        """Factory method last_days creates correct filter."""
        f = TemporalFilter.last_days(7)
        assert f.operator == TemporalOperator.AFTER
        assert f.start_time is not None
        # start_time should be roughly 7 days ago
        delta = datetime.now() - f.start_time
        assert 6.9 < delta.total_seconds() / 86400 < 7.1

    def test_last_hours(self) -> None:
        """Factory method last_hours creates correct filter."""
        f = TemporalFilter.last_hours(24)
        assert f.operator == TemporalOperator.AFTER
        assert f.start_time is not None

    def test_before(self) -> None:
        """Factory method before creates correct filter."""
        t = datetime(2024, 6, 1)
        f = TemporalFilter.before(t)
        assert f.operator == TemporalOperator.BEFORE
        assert f.end_time == t

    def test_after(self) -> None:
        """Factory method after creates correct filter."""
        t = datetime(2024, 6, 1)
        f = TemporalFilter.after(t)
        assert f.operator == TemporalOperator.AFTER
        assert f.start_time == t

    def test_between(self) -> None:
        """Factory method between creates correct filter."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        f = TemporalFilter.between(start, end)
        assert f.operator == TemporalOperator.BETWEEN
        assert f.start_time == start
        assert f.end_time == end

    def test_get_effective_times_absolute(self) -> None:
        """get_effective_times with absolute times returns them directly."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        f = TemporalFilter(start_time=start, end_time=end)
        s, e = f.get_effective_times()
        assert s == start
        assert e == end

    def test_get_effective_times_relative_days(self) -> None:
        """get_effective_times with relative_days computes start time."""
        f = TemporalFilter(relative_days=7)
        s, e = f.get_effective_times()
        assert s is not None
        delta = datetime.now() - s
        assert 6.9 < delta.total_seconds() / 86400 < 7.1
        assert e is None

    def test_get_effective_times_relative_hours(self) -> None:
        """get_effective_times with relative_hours computes start time."""
        f = TemporalFilter(relative_hours=12)
        s, e = f.get_effective_times()
        assert s is not None
        delta = datetime.now() - s
        assert 11.9 < delta.total_seconds() / 3600 < 12.1

    def test_matches_before(self) -> None:
        """matches with BEFORE operator."""
        end = datetime(2024, 6, 1)
        f = TemporalFilter(operator=TemporalOperator.BEFORE, end_time=end)
        assert f.matches(datetime(2024, 5, 1)) is True
        assert f.matches(datetime(2024, 7, 1)) is False

    def test_matches_after(self) -> None:
        """matches with AFTER operator."""
        start = datetime(2024, 1, 1)
        f = TemporalFilter(operator=TemporalOperator.AFTER, start_time=start)
        assert f.matches(datetime(2024, 6, 1)) is True
        assert f.matches(datetime(2023, 6, 1)) is False

    def test_matches_between(self) -> None:
        """matches with BETWEEN operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        f = TemporalFilter.between(start, end)
        assert f.matches(datetime(2024, 3, 1)) is True
        assert f.matches(datetime(2024, 7, 1)) is False
        assert f.matches(datetime(2023, 11, 1)) is False
        # Boundaries are inclusive
        assert f.matches(datetime(2024, 1, 1)) is True
        assert f.matches(datetime(2024, 6, 1)) is True

    def test_matches_between_missing_bounds(self) -> None:
        """BETWEEN with missing bounds returns True."""
        f = TemporalFilter(operator=TemporalOperator.BETWEEN)
        assert f.matches(datetime(2024, 3, 1)) is True

    def test_matches_unknown_operator(self) -> None:
        """Unknown operators default to True."""
        f = TemporalFilter(operator=TemporalOperator.DURING)
        assert f.matches(datetime(2024, 3, 1)) is True

    def test_timezone_normalization(self) -> None:
        """Timezone-aware datetimes are normalized for comparison."""
        # UTC+5 time
        tz_plus5 = timezone(timedelta(hours=5))
        aware_time = datetime(2024, 6, 1, 12, 0, 0, tzinfo=tz_plus5)

        # Create filter with naive UTC time
        f = TemporalFilter.after(datetime(2024, 6, 1, 6, 0, 0))  # 6 AM UTC

        # 12:00 UTC+5 = 7:00 UTC, which is after 6:00 UTC
        assert f.matches(aware_time) is True

    def test_normalize_tz_none(self) -> None:
        """_normalize_tz with None returns None."""
        assert TemporalFilter._normalize_tz(None) is None

    def test_normalize_tz_naive(self) -> None:
        """_normalize_tz with naive datetime returns it unchanged."""
        dt = datetime(2024, 6, 1, 12, 0, 0)
        result = TemporalFilter._normalize_tz(dt)
        assert result == dt
        assert result.tzinfo is None

    def test_normalize_tz_aware(self) -> None:
        """_normalize_tz with aware datetime converts to naive UTC."""
        tz_plus5 = timezone(timedelta(hours=5))
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=tz_plus5)
        result = TemporalFilter._normalize_tz(dt)
        assert result.tzinfo is None
        # 12:00 UTC+5 = 07:00 UTC
        assert result.hour == 7


class TestTemporalQuery:
    """Tests for TemporalQuery."""

    def test_init(self) -> None:
        """Test default initialization."""
        tq = TemporalQuery(query="test query")
        assert tq.query == "test query"
        assert tq.filters == []
        assert tq.recency_weight == 0.0
        assert tq.decay_days == 30.0
        assert tq.context_window_days is None

    def test_add_filter(self) -> None:
        """add_filter appends and returns self for chaining."""
        tq = TemporalQuery(query="test")
        f = TemporalFilter.last_days(7)
        result = tq.add_filter(f)
        assert result is tq
        assert len(tq.filters) == 1
        assert tq.filters[0] is f

    def test_with_recency_bias(self) -> None:
        """with_recency_bias sets weight and decay."""
        tq = TemporalQuery(query="test")
        result = tq.with_recency_bias(weight=0.5, decay_days=14.0)
        assert result is tq
        assert tq.recency_weight == 0.5
        assert tq.decay_days == 14.0

    def test_calculate_recency_score_no_bias(self) -> None:
        """No recency bias returns 1.0."""
        tq = TemporalQuery(query="test", recency_weight=0.0)
        score = tq.calculate_recency_score(datetime.now())
        assert score == 1.0

    def test_calculate_recency_score_recent(self) -> None:
        """Very recent timestamps get high scores."""
        tq = TemporalQuery(query="test")
        tq.with_recency_bias(weight=0.5, decay_days=30.0)
        score = tq.calculate_recency_score(datetime.utcnow())
        # Recent item: decay ≈ 1.0, score ≈ (1-0.5) + 0.5*1.0 = 1.0
        assert score > 0.95

    def test_calculate_recency_score_old(self) -> None:
        """Old timestamps get lower scores."""
        tq = TemporalQuery(query="test")
        tq.with_recency_bias(weight=0.5, decay_days=30.0)
        old_time = datetime.utcnow() - timedelta(days=90)
        score = tq.calculate_recency_score(old_time)
        # 90 days with 30-day half-life: decay = 0.5^3 = 0.125
        # score = 0.5 + 0.5 * 0.125 = 0.5625
        assert 0.5 < score < 0.6

    def test_calculate_recency_score_decay(self) -> None:
        """Half-life works correctly: score at decay_days is predictable."""
        tq = TemporalQuery(query="test")
        tq.with_recency_bias(weight=1.0, decay_days=30.0)
        half_life_time = datetime.utcnow() - timedelta(days=30)
        score = tq.calculate_recency_score(half_life_time)
        # With weight=1.0: score = (1-1.0) + 1.0 * 0.5^1 = 0.5
        assert abs(score - 0.5) < 0.02

    def test_calculate_recency_score_aware_datetime(self) -> None:
        """Timezone-aware datetime is handled correctly."""
        tq = TemporalQuery(query="test")
        tq.with_recency_bias(weight=0.5, decay_days=30.0)
        aware_time = datetime.now(UTC)
        score = tq.calculate_recency_score(aware_time)
        assert score > 0.9

    def test_get_context_filter_none(self) -> None:
        """No context_window_days returns None."""
        tq = TemporalQuery(query="test")
        assert tq.get_context_filter() is None

    def test_get_context_filter(self) -> None:
        """context_window_days creates a TemporalFilter."""
        tq = TemporalQuery(query="test", context_window_days=7)
        f = tq.get_context_filter()
        assert f is not None
        assert f.operator == TemporalOperator.AFTER
        assert f.start_time is not None
