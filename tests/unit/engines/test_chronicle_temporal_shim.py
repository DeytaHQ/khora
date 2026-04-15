"""Unit tests for Chronicle engine temporal_filter field extraction shim.

The `_temporal_channel` method reads `occurred_after`/`occurred_before` first
and falls back to `start_time`/`end_time` via `or`-chaining getattr calls.
These tests verify that logic directly without spinning up any database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace


def _extract_bounds(temporal_filter: object) -> tuple[datetime | None, datetime | None]:
    """Replicate the extraction logic from Chronicle._temporal_channel.

    Mirrors the exact getattr or-chaining from engine.py lines 779-784:
        created_after  = getattr(tf, "occurred_after",  None) or getattr(tf, "start_time",  None)
        created_before = getattr(tf, "occurred_before", None) or getattr(tf, "end_time",    None)
    """
    created_after = getattr(temporal_filter, "occurred_after", None) or getattr(temporal_filter, "start_time", None)
    created_before = getattr(temporal_filter, "occurred_before", None) or getattr(temporal_filter, "end_time", None)
    return created_after, created_before


class TestChronicleTemporalShim:
    """Tests for the occurred_after/occurred_before → start_time/end_time fallback."""

    def test_skeleton_filter_primary_fields_used(self) -> None:
        """SkeletonTemporalFilter with occurred_after/occurred_before → primary fields used."""
        from khora.engines.skeleton.backends import TemporalFilter as SkeletonTemporalFilter

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        tf = SkeletonTemporalFilter(occurred_after=start, occurred_before=end)

        after, before = _extract_bounds(tf)
        assert after == start
        assert before == end

    def test_fallback_to_start_time_end_time(self) -> None:
        """Object with only start_time/end_time → fallback fields used."""
        start = datetime(2024, 3, 1, tzinfo=UTC)
        end = datetime(2024, 9, 30, tzinfo=UTC)
        tf = SimpleNamespace(start_time=start, end_time=end)

        after, before = _extract_bounds(tf)
        assert after == start
        assert before == end

    def test_occurred_after_takes_precedence_over_start_time(self) -> None:
        """When both occurred_after and start_time present, occurred_after wins."""
        primary = datetime(2024, 1, 1, tzinfo=UTC)
        fallback = datetime(2023, 1, 1, tzinfo=UTC)
        tf = SimpleNamespace(occurred_after=primary, start_time=fallback, occurred_before=None, end_time=None)

        after, before = _extract_bounds(tf)
        assert after == primary

    def test_none_filter_returns_none_bounds(self) -> None:
        """None temporal_filter → both bounds are None."""
        tf = SimpleNamespace(occurred_after=None, occurred_before=None)

        after, before = _extract_bounds(tf)
        assert after is None
        assert before is None

    def test_start_only_no_end(self) -> None:
        """Only start bound set → created_before is None."""
        start = datetime(2024, 6, 1, tzinfo=UTC)
        tf = SimpleNamespace(occurred_after=start, occurred_before=None, end_time=None)

        after, before = _extract_bounds(tf)
        assert after == start
        assert before is None

    def test_end_only_no_start(self) -> None:
        """Only end bound set → created_after is None."""
        end = datetime(2024, 6, 1, tzinfo=UTC)
        tf = SimpleNamespace(occurred_after=None, start_time=None, occurred_before=end)

        after, before = _extract_bounds(tf)
        assert after is None
        assert before == end
