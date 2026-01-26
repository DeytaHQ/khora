"""Temporal query support for Khora Memory Lake."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


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
            start_time=datetime.now() - timedelta(days=days),
        )

    @classmethod
    def last_hours(cls, hours: int) -> TemporalFilter:
        """Create a filter for the last N hours."""
        return cls(
            operator=TemporalOperator.AFTER,
            start_time=datetime.now() - timedelta(hours=hours),
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
            start = datetime.now() - timedelta(days=self.relative_days)
        if self.relative_hours is not None:
            start = datetime.now() - timedelta(hours=self.relative_hours)

        return start, end

    def matches(self, timestamp: datetime) -> bool:
        """Check if a timestamp matches this filter."""
        start, end = self.get_effective_times()

        if self.operator == TemporalOperator.BEFORE:
            return end is not None and timestamp < end
        elif self.operator == TemporalOperator.AFTER:
            return start is not None and timestamp > start
        elif self.operator == TemporalOperator.BETWEEN:
            if start is None or end is None:
                return True
            return start <= timestamp <= end
        else:
            return True


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
        """
        if self.recency_weight == 0:
            return 1.0

        now = datetime.now()
        if timestamp.tzinfo:
            from datetime import UTC

            now = datetime.now(UTC)

        age_days = (now - timestamp).total_seconds() / (24 * 60 * 60)

        # Exponential decay: score = 0.5^(age/half_life)
        import math

        decay = math.pow(0.5, age_days / self.decay_days)

        # Blend with recency weight
        return (1 - self.recency_weight) + (self.recency_weight * decay)

    def get_context_filter(self) -> TemporalFilter | None:
        """Get a filter for the context window."""
        if self.context_window_days is None:
            return None
        return TemporalFilter.last_days(self.context_window_days)
