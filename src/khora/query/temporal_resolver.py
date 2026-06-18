"""Two-tier temporal resolver for Khora query engine.

Provides fast dateparser-based resolution (~0.25ms) with LLM fallback
for natural language temporal expressions like "last week", "yesterday",
"January 2025", "Q3 last year", etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from loguru import logger


@dataclass
class ResolvedRange:
    """Result of temporal resolution."""

    start: datetime | None = None
    end: datetime | None = None
    confidence: float = 0.0
    expression: str = ""
    source: Literal["dateparser", "llm"] = "dateparser"


class TemporalResolver:
    """Two-tier temporal resolver: fast dateparser + LLM fallback."""

    def resolve_fast(self, query: str, reference: datetime | None = None) -> ResolvedRange | None:
        """Resolve temporal expressions using dateparser (~0.25ms).

        Uses dateparser.parse() with languages=['en'], PREFER_DATES_FROM='past',
        RELATIVE_BASE=reference.

        Returns None if dateparser can't parse (which should trigger LLM fallback).
        """
        try:
            import dateparser
        except ImportError:
            logger.debug("dateparser not installed, skipping fast resolution")
            return None

        if reference is None:
            reference = datetime.now(UTC)
        # Strip timezone for dateparser RELATIVE_BASE (it expects naive)
        ref_naive = reference.replace(tzinfo=None) if reference.tzinfo else reference

        settings = {
            "PREFER_DATES_FROM": "past",
            "RELATIVE_BASE": ref_naive,
            "RETURN_AS_TIMEZONE_AWARE": False,
        }

        # Try to parse the entire query, then individual temporal phrases
        parsed = dateparser.parse(query, languages=["en"], settings=settings)
        if parsed is None:
            # Try extracting temporal phrases with regex
            temporal_patterns = [
                r"last\s+\d+\s+\w+",
                r"last\s+\w+",
                r"this\s+\w+",
                r"\d+\s+\w+\s+ago",
                r"yesterday",
                r"today",
                r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}",
                r"q[1-4]\s*\d{4}",
                r"q[1-4]\s+last\s+year",
                r"last\s+quarter",
                r"20\d{2}",
            ]
            combined = "|".join(f"({p})" for p in temporal_patterns)
            match = re.search(combined, query, re.IGNORECASE)
            if match:
                matched_text = match.group()
                parsed = dateparser.parse(matched_text, languages=["en"], settings=settings)
                if parsed is None:
                    # dateparser can't parse but regex matched a known pattern
                    # (e.g. "last 7 days", "last quarter") — use reference as
                    # the point and let _point_to_range handle it via its own
                    # regex-based granularity inference.
                    start, end = self._point_to_range(ref_naive, matched_text, ref_naive)
                    return ResolvedRange(
                        start=start.replace(tzinfo=UTC) if start else None,
                        end=end.replace(tzinfo=UTC) if end else None,
                        confidence=0.75,
                        expression=matched_text,
                        source="dateparser",
                    )

            if parsed is None:
                return None

        # Determine what expression was matched for range inference
        expression = query.strip()
        start, end = self._point_to_range(parsed, expression, ref_naive)

        return ResolvedRange(
            start=start.replace(tzinfo=UTC) if start else None,
            end=end.replace(tzinfo=UTC) if end else None,
            confidence=0.85,
            expression=expression,
            source="dateparser",
        )

    def _point_to_range(self, point: datetime, expression: str, reference: datetime) -> tuple[datetime, datetime]:
        """Infer a range from a point date based on the expression granularity.

        "January 2025" -> Jan 1 - Jan 31
        "yesterday" -> 00:00 - 23:59:59
        "last week" -> 7 days ago - now
        "last month" -> start of last month - end of last month
        "last quarter" -> quarter boundaries
        "3 weeks ago" -> 21 days ago - now
        "2025" -> Jan 1 - Dec 31
        "today" -> 00:00 - 23:59:59
        """
        expr_lower = expression.lower().strip()

        # "yesterday" / "today" -> single day
        if "yesterday" in expr_lower or "today" in expr_lower:
            start = point.replace(hour=0, minute=0, second=0, microsecond=0)
            end = point.replace(hour=23, minute=59, second=59, microsecond=999999)
            return start, end

        # "last N days/weeks/months" or "N days/weeks/months ago"
        duration_match = re.search(r"(?:last\s+)?(\d+)\s+(day|week|month|year)s?\s*(?:ago)?", expr_lower)
        if duration_match:
            count = int(duration_match.group(1))
            unit = duration_match.group(2)
            if unit == "day":
                delta = timedelta(days=count)
            elif unit == "week":
                delta = timedelta(weeks=count)
            elif unit == "month":
                delta = timedelta(days=count * 30)
            elif unit == "year":
                delta = timedelta(days=count * 365)
            else:
                delta = timedelta(days=count)
            return reference - delta, reference

        # "last week"
        if "last week" in expr_lower:
            return reference - timedelta(days=7), reference

        # "last month"
        if "last month" in expr_lower:
            return reference - timedelta(days=30), reference

        # "last quarter"
        if "last quarter" in expr_lower or re.search(r"q[1-4]", expr_lower):
            quarter_match = re.search(r"q([1-4])", expr_lower)
            if quarter_match:
                q = int(quarter_match.group(1))
                # Check if a year is specified
                year_match = re.search(r"20\d{2}", expr_lower)
                year = int(year_match.group()) if year_match else reference.year
                month_start = (q - 1) * 3 + 1
                start = datetime(year, month_start, 1)
                if q == 4:
                    end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
                else:
                    end = datetime(year, month_start + 3, 1) - timedelta(seconds=1)
                return start, end
            # Generic "last quarter": 90 days ago
            return reference - timedelta(days=90), reference

        # Month + year: "January 2025", "march 2024"
        month_year_match = re.search(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})",
            expr_lower,
        )
        if month_year_match:
            import calendar

            month_name = month_year_match.group(1)
            year = int(month_year_match.group(2))
            month_map = {
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "may": 5,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
            }
            month = month_map[month_name]
            _, last_day = calendar.monthrange(year, month)
            start = datetime(year, month, 1)
            end = datetime(year, month, last_day, 23, 59, 59)
            return start, end

        # Year only: "2025"
        year_match = re.match(r"^20\d{2}$", expr_lower.strip())
        if year_match:
            year = int(year_match.group())
            return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)

        # "this week"
        if "this week" in expr_lower:
            # Start of current week (Monday)
            days_since_monday = reference.weekday()
            start = (reference - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            return start, reference

        # "this month"
        if "this month" in expr_lower:
            start = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return start, reference

        # Default: point date as start, reference as end
        if point < reference:
            return point, reference
        else:
            return reference, point

    def validate_dates(
        self,
        start: datetime | None,
        end: datetime | None,
        reference: datetime | None = None,
    ) -> tuple[datetime | None, datetime | None]:
        """Validate and fix date ranges.

        - Swap if start > end
        - Cap future dates to now
        - Reject dates > 10 years old (return None)
        """
        if reference is None:
            reference = datetime.now(UTC)
        ref_naive = reference.replace(tzinfo=None) if reference.tzinfo else reference

        def _to_naive(dt: datetime | None) -> datetime | None:
            if dt is None:
                return None
            return dt.replace(tzinfo=None) if dt.tzinfo else dt

        s = _to_naive(start)
        e = _to_naive(end)

        # Swap if inverted
        if s is not None and e is not None and s > e:
            s, e = e, s

        # Cap future dates
        if s is not None and s > ref_naive:
            s = ref_naive
        if e is not None and e > ref_naive:
            e = ref_naive

        # Reject dates > 10 years old
        ten_years_ago = ref_naive - timedelta(days=3650)
        if s is not None and s < ten_years_ago:
            s = None
        if e is not None and e < ten_years_ago:
            e = None

        # Restore UTC timezone
        if s is not None:
            s = s.replace(tzinfo=UTC)
        if e is not None:
            e = e.replace(tzinfo=UTC)

        return s, e


# ---------------------------------------------------------------------------
# Public helper — converts a temporal signal + query into a pushdown filter
# ---------------------------------------------------------------------------


def resolve_temporal_filter(
    query: str,
    temporal_signal: Any = None,
) -> Any | None:
    """Attempt to resolve a temporal query into a SQL-pushdown TemporalFilter.

    Parses relative date expressions ("last 7 days", "this week", "since Monday")
    into absolute datetime ranges suitable for WHERE clause filtering.

    Returns a ``khora.core.temporal.ChunkTemporalFilter`` with
    ``occurred_after`` / ``occurred_before`` set, or *None* if the query
    doesn't contain parseable temporal expressions.

    This is intentionally engine-agnostic: callers that need the
    ``khora.query.temporal.TemporalFilter`` (``start_time`` / ``end_time``)
    can convert via :func:`to_query_temporal_filter`.
    """
    # 1. If the signal already carries an explicit filter, pass it through.
    if temporal_signal is not None and getattr(temporal_signal, "temporal_filter", None) is not None:
        return temporal_signal.temporal_filter

    # 2. Only attempt resolution for queries with temporal intent.
    if temporal_signal is not None and not getattr(temporal_signal, "is_temporal", False):
        return None

    # 3. Use TemporalResolver (dateparser path — no LLM, ~0.25 ms).
    try:
        resolver = TemporalResolver()
        resolved = resolver.resolve_fast(query)
        if resolved and resolved.start:
            from khora.core.temporal import ChunkTemporalFilter as SkeletonTemporalFilter

            return SkeletonTemporalFilter(
                occurred_after=resolved.start,
                occurred_before=resolved.end,
            )
    except Exception:
        logger.debug("resolve_temporal_filter: dateparser resolution failed", exc_info=True)

    return None


def to_query_temporal_filter(
    skeleton_filter: Any,
) -> Any | None:
    """Convert a ChunkTemporalFilter to a ``khora.query.temporal.TemporalFilter``.

    Useful for engines (VectorCypher, Chronicle) that pass the filter into
    ``HybridQueryEngine`` or ``_temporal_channel`` which read
    ``start_time`` / ``end_time``.

    Returns *None* if the input has no usable time bounds.
    """
    occurred_after = getattr(skeleton_filter, "occurred_after", None)
    occurred_before = getattr(skeleton_filter, "occurred_before", None)
    if occurred_after is None and occurred_before is None:
        return None

    from khora.query.temporal import TemporalFilter as QueryTemporalFilter

    return QueryTemporalFilter(
        start_time=occurred_after,
        end_time=occurred_before,
    )
