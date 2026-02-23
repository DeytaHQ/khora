"""Temporal query support for Khora Memory Lake."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


class TemporalOperator(str, Enum):
    """Temporal query operators."""

    BEFORE = "before"
    AFTER = "after"
    BETWEEN = "between"
    DURING = "during"  # Within a specific time period
    OVERLAPS = "overlaps"  # Overlaps with a time range


@dataclass
class TemporalFilter:
    """Filter for temporal queries."""

    operator: TemporalOperator = TemporalOperator.AFTER
    start_time: datetime | None = None
    end_time: datetime | None = None

    # For relative time queries
    relative_days: int | None = None
    relative_hours: int | None = None

    # Alias properties for consistency
    @property
    def start_date(self) -> datetime | None:
        """Alias for start_time."""
        return self.start_time

    @property
    def end_date(self) -> datetime | None:
        """Alias for end_time."""
        return self.end_time

    def __init__(
        self,
        operator: TemporalOperator = TemporalOperator.AFTER,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        relative_days: int | None = None,
        relative_hours: int | None = None,
    ) -> None:
        """Initialize with flexible date/time naming."""
        self.operator = operator
        self.start_time = start_time or start_date
        self.end_time = end_time or end_date
        self.relative_days = relative_days
        self.relative_hours = relative_hours

        # Auto-detect operator if not specified
        if self.start_time and self.end_time:
            self.operator = TemporalOperator.BETWEEN
        elif self.end_time and not self.start_time:
            self.operator = TemporalOperator.BEFORE

    @classmethod
    def last_days(cls, days: int) -> TemporalFilter:
        """Create a filter for the last N days."""
        return cls(
            operator=TemporalOperator.AFTER,
            start_time=datetime.now(UTC) - timedelta(days=days),
        )

    @classmethod
    def last_hours(cls, hours: int) -> TemporalFilter:
        """Create a filter for the last N hours."""
        return cls(
            operator=TemporalOperator.AFTER,
            start_time=datetime.now(UTC) - timedelta(hours=hours),
        )

    @classmethod
    def before(cls, time: datetime) -> TemporalFilter:
        """Create a filter for before a specific time."""
        return cls(operator=TemporalOperator.BEFORE, end_time=time)

    @classmethod
    def after(cls, time: datetime) -> TemporalFilter:
        """Create a filter for after a specific time."""
        return cls(operator=TemporalOperator.AFTER, start_time=time)

    @classmethod
    def between(cls, start: datetime, end: datetime) -> TemporalFilter:
        """Create a filter for a time range."""
        return cls(operator=TemporalOperator.BETWEEN, start_time=start, end_time=end)

    def get_effective_times(self) -> tuple[datetime | None, datetime | None]:
        """Get the effective start and end times."""
        start = self.start_time
        end = self.end_time

        # Handle relative times
        if self.relative_days is not None:
            start = datetime.now(UTC) - timedelta(days=self.relative_days)
        if self.relative_hours is not None:
            start = datetime.now(UTC) - timedelta(hours=self.relative_hours)

        return start, end

    def matches(self, timestamp: datetime) -> bool:
        """Check if a timestamp matches this filter.

        Handles timezone-aware and timezone-naive datetime comparison
        by normalizing both to the same timezone awareness.
        """
        start, end = self.get_effective_times()

        # Normalize timezone awareness for comparison
        ts = self._normalize_tz(timestamp)
        start_norm = self._normalize_tz(start) if start else None
        end_norm = self._normalize_tz(end) if end else None

        if self.operator == TemporalOperator.BEFORE:
            return end_norm is not None and ts < end_norm
        elif self.operator == TemporalOperator.AFTER:
            return start_norm is not None and ts > start_norm
        elif self.operator == TemporalOperator.BETWEEN:
            if start_norm is None or end_norm is None:
                return True
            return start_norm <= ts <= end_norm
        else:
            return True

    @staticmethod
    def _normalize_tz(dt: datetime | None) -> datetime | None:
        """Normalize datetime to naive UTC for comparison.

        Converts timezone-aware datetimes to UTC then strips tzinfo.
        Leaves timezone-naive datetimes as-is (assumes UTC).
        """
        if dt is None:
            return None

        if dt.tzinfo is not None:
            # Convert to UTC and make naive
            from datetime import UTC

            utc_dt = dt.astimezone(UTC)
            return utc_dt.replace(tzinfo=None)
        else:
            # Already naive, assume UTC
            return dt


@dataclass
class TemporalQuery:
    """Query with temporal context."""

    query: str
    filters: list[TemporalFilter] = field(default_factory=list)

    # Temporal weighting
    recency_weight: float = 0.0  # 0 = no recency bias, 1 = strong recency bias
    decay_days: float = 30.0  # Half-life for recency decay

    # Context window
    context_window_days: int | None = None  # Limit context to recent period

    def add_filter(self, filter: TemporalFilter) -> TemporalQuery:
        """Add a temporal filter."""
        self.filters.append(filter)
        return self

    def with_recency_bias(self, weight: float = 0.3, decay_days: float = 30.0) -> TemporalQuery:
        """Add recency bias to scoring."""
        self.recency_weight = weight
        self.decay_days = decay_days
        return self

    def calculate_recency_score(self, timestamp: datetime) -> float:
        """Calculate recency score for a timestamp.

        Uses exponential decay with configurable half-life.
        Handles timezone-aware and timezone-naive datetime comparison.
        """
        if self.recency_weight == 0:
            return 1.0

        import math

        # Normalize both to naive UTC for comparison
        now = datetime.now(UTC).replace(tzinfo=None)
        ts = timestamp

        if ts.tzinfo is not None:
            ts = ts.astimezone(UTC).replace(tzinfo=None)

        age_days = (now - ts).total_seconds() / (24 * 60 * 60)

        # Exponential decay: score = 0.5^(age/half_life)
        decay = math.pow(0.5, age_days / self.decay_days)

        # Blend with recency weight
        return (1 - self.recency_weight) + (self.recency_weight * decay)

    def get_context_filter(self) -> TemporalFilter | None:
        """Get a filter for the context window."""
        if self.context_window_days is None:
            return None
        return TemporalFilter.last_days(self.context_window_days)


# ---------------------------------------------------------------------------
# Batch helpers — convert datetimes to epoch seconds and call Rust-accelerated
# batch operations from _accel.py.  Used by query/engine.py hot paths.
# ---------------------------------------------------------------------------


def _dt_to_epoch(dt: datetime | None) -> float | None:
    """Convert a datetime to epoch seconds, normalizing timezone."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    epoch = datetime(1970, 1, 1)
    return (dt - epoch).total_seconds()


def batch_filter_chunks(
    chunks_and_scores: list[tuple[Any, float]],
    temporal_filter: TemporalFilter,
) -> list[tuple[Any, float]]:
    """Filter a list of (chunk, score) by temporal_filter.matches(), batched.

    Extracts created_at from each chunk, converts to epoch seconds, and uses
    the Rust-accelerated batch_temporal_filter for the comparison.

    Falls back to per-item TemporalFilter.matches() if chunks have no created_at.
    """
    from khora._accel import batch_temporal_filter

    if not chunks_and_scores:
        return []

    start, end = temporal_filter.get_effective_times()
    operator = temporal_filter.operator.value  # e.g. "before", "after", "between"
    start_secs = _dt_to_epoch(TemporalFilter._normalize_tz(start))
    end_secs = _dt_to_epoch(TemporalFilter._normalize_tz(end))

    # Collect epoch seconds from chunk.created_at
    timestamps: list[float] = []
    valid = True
    for chunk, _score in chunks_and_scores:
        created_at = getattr(chunk, "created_at", None)
        if created_at is None:
            valid = False
            break
        ts = _dt_to_epoch(TemporalFilter._normalize_tz(created_at))
        if ts is None:
            valid = False
            break
        timestamps.append(ts)

    if not valid:
        # Fallback: per-item filtering (e.g. missing created_at)
        return [(c, s) for c, s in chunks_and_scores if temporal_filter.matches(c.created_at)]

    mask = batch_temporal_filter(timestamps, operator, start_secs, end_secs)
    return [pair for pair, keep in zip(chunks_and_scores, mask) if keep]


def batch_apply_recency(
    chunks_and_scores: list[tuple[Any, float]],
    recency_weight: float,
    decay_days: float,
) -> list[tuple[Any, float]]:
    """Multiply chunk scores by recency scores, batched.

    Extracts created_at from each chunk, converts to epoch seconds, and uses
    the Rust-accelerated batch_recency_scores. Returns a new list of
    (chunk, score * recency_score) sorted descending.
    """
    from khora._accel import batch_recency_scores

    if not chunks_and_scores or recency_weight == 0.0:
        return chunks_and_scores

    now_secs = _dt_to_epoch(datetime.now(UTC).replace(tzinfo=None))
    if now_secs is None:
        return chunks_and_scores

    timestamps: list[float] = []
    valid = True
    for chunk, _score in chunks_and_scores:
        created_at = getattr(chunk, "created_at", None)
        if created_at is None:
            valid = False
            break
        ts = _dt_to_epoch(TemporalFilter._normalize_tz(created_at))
        if ts is None:
            valid = False
            break
        timestamps.append(ts)

    if not valid:
        # Fallback: per-item scoring
        tq = TemporalQuery(query="").with_recency_bias(recency_weight, decay_days)
        result = [(c, s * tq.calculate_recency_score(c.created_at)) for c, s in chunks_and_scores]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    scores = batch_recency_scores(timestamps, now_secs, decay_days, recency_weight)
    result = [(c, s * rs) for (c, s), rs in zip(chunks_and_scores, scores)]
    result.sort(key=lambda x: x[1], reverse=True)
    return result
