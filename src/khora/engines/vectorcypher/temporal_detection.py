"""Temporal query detection and signal classification.

Three-tier cascade for detecting temporal intent in queries:
1. Aho-Corasick dictionary lookup (~1-10μs via Rust, or Python fallback)
2. Model2Vec embedding centroid (~40-50μs, optional)
3. LLM-based query understanding (existing, not invoked here)

Each detected query is classified into a TemporalCategory that drives
retrieval parameters (recency weight, sort order, decay rate).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from khora._accel import detect_temporal_category

if TYPE_CHECKING:
    pass


class TemporalCategory(str, Enum):
    """Classification of temporal query intent."""

    NONE = "none"
    EXPLICIT = "explicit"  # "before April 2024", parseable dates
    STATE_QUERY = "state_query"  # "currently", "now", "does X play"
    ORDINAL = "ordinal"  # "first", "which came earlier"
    AGGREGATE = "aggregate"  # "how many total", "all instances"
    RECENCY = "recency"  # "latest", "most recent"
    CHANGE = "change"  # "changed", "used to", "still"


# Map integer category IDs (from Rust/Python fallback) to enum values
CATEGORY_MAP: dict[int, TemporalCategory] = {
    0: TemporalCategory.NONE,
    1: TemporalCategory.EXPLICIT,
    2: TemporalCategory.STATE_QUERY,
    3: TemporalCategory.ORDINAL,
    4: TemporalCategory.AGGREGATE,
    5: TemporalCategory.RECENCY,
    6: TemporalCategory.CHANGE,
}


@dataclass(frozen=True)
class RetrievalParams:
    """Category-specific retrieval parameters."""

    recency_weight: float
    temporal_sort: bool
    decay_days_override: int | None = None
    recency_floor: float = 0.5  # Default floor for multiplicative recency


# Category → retrieval behavior mapping
# Weights control the *multiplicative* recency exponent applied to RRF scores.
# Higher weight = stronger penalty for stale chunks (score *= recency^(exp*w)).
# Conservative values protect non-temporal categories (implicit_inference,
# abstention) while still discriminating temporal ones.
RETRIEVAL_PARAMS: dict[TemporalCategory, RetrievalParams] = {
    TemporalCategory.NONE: RetrievalParams(recency_weight=0.1, temporal_sort=False, recency_floor=0.5),
    TemporalCategory.EXPLICIT: RetrievalParams(recency_weight=0.3, temporal_sort=False, recency_floor=0.4),
    TemporalCategory.STATE_QUERY: RetrievalParams(recency_weight=0.6, temporal_sort=True, recency_floor=0.3),
    TemporalCategory.ORDINAL: RetrievalParams(
        recency_weight=0.3, temporal_sort=True, decay_days_override=7, recency_floor=0.4
    ),
    TemporalCategory.AGGREGATE: RetrievalParams(recency_weight=0.0, temporal_sort=False, recency_floor=0.5),
    TemporalCategory.RECENCY: RetrievalParams(
        recency_weight=0.6, temporal_sort=True, decay_days_override=3, recency_floor=0.3
    ),
    TemporalCategory.CHANGE: RetrievalParams(
        recency_weight=0.5, temporal_sort=True, decay_days_override=14, recency_floor=0.4
    ),
}


@dataclass(frozen=True)
class TemporalSignal:
    """Result of temporal query detection."""

    is_temporal: bool
    category: TemporalCategory
    confidence: float  # 0.0-1.0
    source: str  # "dictionary", "semantic", "none"
    temporal_filter: Any | None = None  # TemporalFilter for EXPLICIT category


# Regex for extracting explicit dates from queries
_DATE_EXTRACT_RE = re.compile(
    r"(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})"
    r"|(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2},?\s+\d{4}\b)"
    r"|(\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{4}\b)",
    re.IGNORECASE,
)

# Semantic detection threshold (near-zero FN, ~30% FP acceptable)
SEMANTIC_THRESHOLD = 0.20


class TemporalDetector:
    """Three-tier cascade temporal query detector."""

    def __init__(self, *, semantic_enabled: bool = False, centroid: Any | None = None):
        self._semantic_enabled = semantic_enabled
        self._centroid = centroid  # Pre-normalized Model2Vec centroid (ndarray)

    def detect(self, query: str, *, query_embedding: list[float] | None = None) -> TemporalSignal:
        """Detect temporal intent in a query.

        Args:
            query: The user query text.
            query_embedding: Optional pre-computed query embedding for Tier 2 semantic check.

        Returns:
            TemporalSignal with category, confidence, and optional temporal filter.
        """
        # Tier 1: Aho-Corasick dictionary (always runs)
        cat_id = detect_temporal_category(query)
        if cat_id > 0:
            category = CATEGORY_MAP[cat_id]
            temporal_filter = None
            if category == TemporalCategory.EXPLICIT:
                temporal_filter = self._extract_date_filter(query)
            return TemporalSignal(
                is_temporal=True,
                category=category,
                confidence=0.9,
                source="dictionary",
                temporal_filter=temporal_filter,
            )

        # Tier 2: Model2Vec centroid (optional, only if Tier 1 missed)
        similarity = 0.0
        if self._semantic_enabled and self._centroid is not None and query_embedding is not None:
            try:
                import numpy as np

                similarity = float(np.dot(query_embedding, self._centroid))
                if similarity > SEMANTIC_THRESHOLD:
                    return TemporalSignal(
                        is_temporal=True,
                        category=TemporalCategory.STATE_QUERY,  # default for semantic
                        confidence=similarity,
                        source="semantic",
                        temporal_filter=None,
                    )
            except ImportError:
                pass

        # No temporal signal detected
        return TemporalSignal(
            is_temporal=False,
            category=TemporalCategory.NONE,
            confidence=1.0 - similarity,
            source="none",
            temporal_filter=None,
        )

    def _extract_date_filter(self, query: str) -> Any | None:
        """Extract a TemporalFilter from explicit date mentions in the query."""
        from khora.engines.skeleton.backends import TemporalFilter

        date_match = _DATE_EXTRACT_RE.search(query)
        if not date_match:
            return None

        date_str = date_match.group(0)
        try:
            parsed_dt = self._parse_datetime(date_str)
        except ValueError:
            return None

        query_lower = query.lower()
        if "before" in query_lower:
            return TemporalFilter(occurred_before=parsed_dt)
        elif "after" in query_lower or "since" in query_lower:
            return TemporalFilter(occurred_after=parsed_dt)
        else:
            # Within ±30 days of the mentioned date
            return TemporalFilter(
                occurred_after=parsed_dt - timedelta(days=30),
                occurred_before=parsed_dt + timedelta(days=30),
            )

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse a datetime string from various formats."""
        try:
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            pass
        for fmt in (
            "%Y/%m/%d",
            "%B %d, %Y",
            "%B %d %Y",
            "%d %B %Y",
        ):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {value}")


def get_retrieval_params(signal: TemporalSignal) -> RetrievalParams:
    """Get retrieval parameters for a temporal signal."""
    return RETRIEVAL_PARAMS[signal.category]


__all__ = [
    "CATEGORY_MAP",
    "RETRIEVAL_PARAMS",
    "RetrievalParams",
    "TemporalCategory",
    "TemporalDetector",
    "TemporalSignal",
    "get_retrieval_params",
]
