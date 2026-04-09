"""Tests for temporal SQL WHERE pushdown via resolve_temporal_filter.

Verifies that relative date expressions ("last 7 days", "this week",
"yesterday") are resolved into TemporalFilter instances with absolute
datetime ranges suitable for SQL WHERE clause filtering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from khora.query.temporal_resolver import resolve_temporal_filter, to_query_temporal_filter


@pytest.mark.unit
class TestResolveTemporalFilter:
    """Tests for the resolve_temporal_filter helper."""

    def test_last_7_days_produces_filter(self) -> None:
        """'last 7 days' should produce a filter with occurred_after ~7 days ago."""
        result = resolve_temporal_filter("What happened in the last 7 days?")
        assert result is not None
        assert result.occurred_after is not None
        delta = datetime.now(UTC) - result.occurred_after
        assert 6.5 < delta.total_seconds() / 86400 < 7.5

    def test_last_week_produces_filter(self) -> None:
        result = resolve_temporal_filter("What was discussed last week?")
        assert result is not None
        assert result.occurred_after is not None

    def test_yesterday_produces_filter(self) -> None:
        result = resolve_temporal_filter("What happened yesterday?")
        assert result is not None
        assert result.occurred_after is not None

    def test_non_temporal_returns_none(self) -> None:
        result = resolve_temporal_filter("What is the capital of France?")
        assert result is None

    def test_explicit_signal_with_filter_passed_through(self) -> None:
        """If temporal_signal already has a filter, use it directly."""
        from khora.engines.skeleton.backends import TemporalFilter

        existing = TemporalFilter(occurred_after=datetime(2026, 1, 1, tzinfo=UTC))
        mock_signal = MagicMock()
        mock_signal.temporal_filter = existing
        mock_signal.is_temporal = True

        result = resolve_temporal_filter("query", mock_signal)
        assert result is existing

    def test_non_temporal_signal_returns_none(self) -> None:
        mock_signal = MagicMock()
        mock_signal.temporal_filter = None
        mock_signal.is_temporal = False

        result = resolve_temporal_filter("query", mock_signal)
        assert result is None

    def test_this_month_produces_filter(self) -> None:
        result = resolve_temporal_filter("Show me this month's progress")
        assert result is not None
        assert result.occurred_after is not None

    def test_3_days_ago_produces_filter(self) -> None:
        result = resolve_temporal_filter("What happened 3 days ago?")
        assert result is not None
        assert result.occurred_after is not None

    def test_returns_skeleton_temporal_filter_type(self) -> None:
        """resolve_temporal_filter returns the skeleton TemporalFilter type."""
        from khora.engines.skeleton.backends import TemporalFilter

        result = resolve_temporal_filter("What happened in the last 3 days?")
        assert result is not None
        assert isinstance(result, TemporalFilter)

    def test_occurred_before_is_set(self) -> None:
        """Both occurred_after and occurred_before should be set for range queries."""
        result = resolve_temporal_filter("What happened in the last 7 days?")
        assert result is not None
        assert result.occurred_after is not None
        assert result.occurred_before is not None
        assert result.occurred_before > result.occurred_after


@pytest.mark.unit
class TestToQueryTemporalFilter:
    """Tests for the to_query_temporal_filter converter."""

    def test_converts_skeleton_to_query_filter(self) -> None:
        """Skeleton TemporalFilter is converted to query TemporalFilter."""
        from khora.engines.skeleton.backends import TemporalFilter as SkeletonTF
        from khora.query.temporal import TemporalFilter as QueryTF

        after = datetime(2026, 1, 1, tzinfo=UTC)
        before = datetime(2026, 2, 1, tzinfo=UTC)
        skeleton = SkeletonTF(occurred_after=after, occurred_before=before)

        result = to_query_temporal_filter(skeleton)
        assert result is not None
        assert isinstance(result, QueryTF)
        assert result.start_time == after
        assert result.end_time == before

    def test_returns_none_for_empty_filter(self) -> None:
        """Returns None when skeleton filter has no time bounds."""
        from khora.engines.skeleton.backends import TemporalFilter as SkeletonTF

        skeleton = SkeletonTF()
        result = to_query_temporal_filter(skeleton)
        assert result is None
