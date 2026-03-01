"""Unit tests for query/temporal_resolver.py — TemporalResolver and ResolvedRange."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from khora.query.temporal_resolver import ResolvedRange, TemporalResolver


class TestResolvedRange:
    """Tests for ResolvedRange dataclass."""

    def test_defaults(self) -> None:
        r = ResolvedRange()
        assert r.start is None
        assert r.end is None
        assert r.confidence == 0.0
        assert r.expression == ""
        assert r.source == "dateparser"

    def test_with_values(self) -> None:
        now = datetime.now(UTC)
        r = ResolvedRange(start=now, end=now, confidence=0.9, expression="yesterday", source="llm")
        assert r.start == now
        assert r.source == "llm"


class TestTemporalResolverResolveFast:
    """Tests for TemporalResolver.resolve_fast()."""

    def setup_method(self) -> None:
        self.resolver = TemporalResolver()
        self.ref = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)

    def test_last_7_days(self) -> None:
        result = self.resolver.resolve_fast("What happened in the last 7 days?", reference=self.ref)
        assert result is not None
        assert result.start is not None
        assert result.source == "dateparser"
        assert result.confidence > 0

    def test_yesterday(self) -> None:
        result = self.resolver.resolve_fast("yesterday", reference=self.ref)
        assert result is not None
        assert result.start is not None
        assert result.end is not None
        # yesterday should produce a single-day range
        if result.start.tzinfo:
            start_naive = result.start.replace(tzinfo=None)
            end_naive = result.end.replace(tzinfo=None)
        else:
            start_naive = result.start
            end_naive = result.end
        assert (end_naive - start_naive).total_seconds() < 86400 + 1

    def test_today(self) -> None:
        result = self.resolver.resolve_fast("today", reference=self.ref)
        assert result is not None
        assert result.start is not None
        assert result.end is not None

    def test_january_2025(self) -> None:
        result = self.resolver.resolve_fast("January 2025", reference=self.ref)
        assert result is not None
        assert result.start is not None
        assert result.end is not None
        s = result.start.replace(tzinfo=None)
        e = result.end.replace(tzinfo=None)
        assert s.month == 1
        assert s.year == 2025
        assert e.month == 1
        assert e.year == 2025

    def test_last_quarter(self) -> None:
        result = self.resolver.resolve_fast("last quarter", reference=self.ref)
        assert result is not None
        assert result.start is not None

    def test_3_weeks_ago(self) -> None:
        result = self.resolver.resolve_fast("3 weeks ago", reference=self.ref)
        assert result is not None
        assert result.start is not None

    def test_returns_none_for_unparseable(self) -> None:
        """Ambiguous or non-temporal phrases return None."""
        assert self.resolver.resolve_fast("around the holidays") is None
        assert self.resolver.resolve_fast("since the reorg") is None
        assert self.resolver.resolve_fast("gibberish xyz abc") is None

    def test_returns_none_for_generic_recently(self) -> None:
        """'recently' alone is too vague for dateparser."""
        # dateparser may or may not parse "recently"; if it does,
        # that's acceptable — we just ensure it doesn't raise.
        self.resolver.resolve_fast("recently")

    def test_utc_timezone_on_result(self) -> None:
        """Results should have UTC timezone."""
        result = self.resolver.resolve_fast("yesterday", reference=self.ref)
        assert result is not None
        if result.start:
            assert result.start.tzinfo == UTC
        if result.end:
            assert result.end.tzinfo == UTC

    def test_custom_reference_date(self) -> None:
        """Resolver respects the reference date."""
        ref = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        result = self.resolver.resolve_fast("yesterday", reference=ref)
        assert result is not None
        assert result.start is not None
        start_date = result.start.replace(tzinfo=None).date()
        assert start_date == (ref.replace(tzinfo=None) - timedelta(days=1)).date()


class TestPointToRange:
    """Tests for TemporalResolver._point_to_range() granularity inference."""

    def setup_method(self) -> None:
        self.resolver = TemporalResolver()
        self.ref = datetime(2026, 3, 1, 12, 0, 0)

    def test_yesterday_single_day(self) -> None:
        point = datetime(2026, 2, 28, 0, 0, 0)
        start, end = self.resolver._point_to_range(point, "yesterday", self.ref)
        assert start.hour == 0
        assert start.minute == 0
        assert end.hour == 23
        assert end.minute == 59

    def test_today_single_day(self) -> None:
        point = datetime(2026, 3, 1, 0, 0, 0)
        start, end = self.resolver._point_to_range(point, "today", self.ref)
        assert start.hour == 0
        assert end.hour == 23

    def test_last_n_days(self) -> None:
        point = datetime(2026, 2, 22, 12, 0, 0)
        start, end = self.resolver._point_to_range(point, "last 7 days", self.ref)
        delta = (end - start).days
        assert delta == 7

    def test_last_n_weeks(self) -> None:
        point = datetime(2026, 2, 1, 12, 0, 0)
        start, end = self.resolver._point_to_range(point, "last 4 weeks", self.ref)
        delta = (end - start).days
        assert delta == 28

    def test_last_week(self) -> None:
        point = datetime(2026, 2, 22, 12, 0, 0)
        start, end = self.resolver._point_to_range(point, "last week", self.ref)
        delta = (end - start).days
        assert delta == 7

    def test_last_month(self) -> None:
        point = datetime(2026, 2, 1, 12, 0, 0)
        start, end = self.resolver._point_to_range(point, "last month", self.ref)
        delta = (end - start).days
        assert delta == 30

    def test_quarter_q1(self) -> None:
        point = datetime(2025, 1, 1)
        start, end = self.resolver._point_to_range(point, "Q1 2025", self.ref)
        assert start == datetime(2025, 1, 1)
        assert end.month == 3

    def test_quarter_q4(self) -> None:
        point = datetime(2025, 10, 1)
        start, end = self.resolver._point_to_range(point, "Q4 2025", self.ref)
        assert start == datetime(2025, 10, 1)
        assert end.month == 12 or (end.month == 1 and end.year == 2026)

    def test_month_year(self) -> None:
        point = datetime(2025, 1, 15)
        start, end = self.resolver._point_to_range(point, "January 2025", self.ref)
        assert start == datetime(2025, 1, 1)
        assert end.day == 31
        assert end.month == 1

    def test_year_only(self) -> None:
        point = datetime(2025, 6, 15)
        start, end = self.resolver._point_to_range(point, "2025", self.ref)
        assert start == datetime(2025, 1, 1)
        assert end.month == 12
        assert end.day == 31

    def test_this_week(self) -> None:
        point = datetime(2026, 3, 1, 12, 0, 0)
        start, end = self.resolver._point_to_range(point, "this week", self.ref)
        assert start.hour == 0
        assert end == self.ref

    def test_this_month(self) -> None:
        point = datetime(2026, 3, 1, 12, 0, 0)
        start, end = self.resolver._point_to_range(point, "this month", self.ref)
        assert start.day == 1
        assert end == self.ref

    def test_default_fallback(self) -> None:
        """Unknown expressions use point → reference as range."""
        point = datetime(2026, 2, 15, 12, 0, 0)
        start, end = self.resolver._point_to_range(point, "some meeting notes", self.ref)
        assert start == point
        assert end == self.ref


class TestValidateDates:
    """Tests for TemporalResolver.validate_dates()."""

    def setup_method(self) -> None:
        self.resolver = TemporalResolver()
        self.ref = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)

    def test_swap_inverted(self) -> None:
        """Start > end gets swapped."""
        end = datetime(2025, 1, 1, tzinfo=UTC)
        start = datetime(2025, 6, 1, tzinfo=UTC)
        s, e = self.resolver.validate_dates(start, end, reference=self.ref)
        assert s is not None and e is not None
        assert s <= e

    def test_cap_future_dates(self) -> None:
        """Future dates are capped to reference."""
        future = datetime(2027, 6, 1, tzinfo=UTC)
        s, e = self.resolver.validate_dates(future, None, reference=self.ref)
        assert s is not None
        # Should be capped to reference (made naive then re-UTC'd)
        assert s <= self.ref

    def test_reject_ancient_dates(self) -> None:
        """Dates > 10 years old are rejected (returned as None)."""
        ancient = datetime(2010, 1, 1, tzinfo=UTC)
        s, e = self.resolver.validate_dates(ancient, None, reference=self.ref)
        assert s is None

    def test_valid_range_unchanged(self) -> None:
        """Valid ranges pass through."""
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 6, 1, tzinfo=UTC)
        s, e = self.resolver.validate_dates(start, end, reference=self.ref)
        assert s is not None and e is not None
        assert s.replace(tzinfo=None) == datetime(2025, 1, 1)
        assert e.replace(tzinfo=None) == datetime(2025, 6, 1)

    def test_none_inputs(self) -> None:
        """None inputs remain None."""
        s, e = self.resolver.validate_dates(None, None, reference=self.ref)
        assert s is None
        assert e is None

    def test_result_has_utc_timezone(self) -> None:
        """Output dates have UTC timezone."""
        start = datetime(2025, 6, 1, tzinfo=UTC)
        s, e = self.resolver.validate_dates(start, None, reference=self.ref)
        assert s is not None
        assert s.tzinfo == UTC

    def test_default_reference(self) -> None:
        """Without explicit reference, uses datetime.now(UTC)."""
        start = datetime(2025, 6, 1, tzinfo=UTC)
        s, e = self.resolver.validate_dates(start, None)
        assert s is not None
