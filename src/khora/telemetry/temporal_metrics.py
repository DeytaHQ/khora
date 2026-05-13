"""Phase A temporal-recency metrics (issue #567).

Counters and one histogram that track the synthetic RECENCY/CHANGE date
floor, the parallel recency channel, and the recency skew of recall
results. All labels are bounded enums — ``category`` from
:class:`TemporalCategory` (7 values) and ``vetoed`` from ``{true,false}``.

Cardinality discipline: never attach ``namespace_id`` here. It is a span
attribute only. See ``docs/telemetry-contract.md``.
"""

from __future__ import annotations

import threading
from typing import Any

from .metrics import metric_counter, metric_histogram

_lock = threading.Lock()
_floor_applied: Any | None = None
_recency_channel_fired: Any | None = None
_top1_age_days: Any | None = None


def _get_floor_applied() -> Any:
    global _floor_applied
    if _floor_applied is None:
        with _lock:
            if _floor_applied is None:
                _floor_applied = metric_counter(
                    "khora.query.temporal.floor_applied_total",
                    unit="1",
                    description=("Number of times a synthetic RECENCY/CHANGE temporal filter was applied to a query."),
                )
    return _floor_applied


def _get_recency_channel_fired() -> Any:
    global _recency_channel_fired
    if _recency_channel_fired is None:
        with _lock:
            if _recency_channel_fired is None:
                _recency_channel_fired = metric_counter(
                    "khora.query.temporal.recency_channel_fired_total",
                    unit="1",
                    description="Number of times the parallel recency channel was queried.",
                )
    return _recency_channel_fired


def _get_top1_age_days() -> Any:
    global _top1_age_days
    if _top1_age_days is None:
        with _lock:
            if _top1_age_days is None:
                _top1_age_days = metric_histogram(
                    "khora.recall.recency.query_to_top1_age_days",
                    unit="d",
                    description=(
                        "Age in days of the top-1 chunk in a recall result. Log-bucketed: [0,1,7,30,90,365,3650]."
                    ),
                )
    return _top1_age_days


def record_floor_applied(*, category: str, vetoed: bool) -> None:
    """Record one synthetic-floor decision.

    ``category`` is a :class:`TemporalCategory` value (string). ``vetoed``
    is True when an anti-recency token (``ever``, ``all``, ``history``,
    ...) suppressed the floor.
    """
    _get_floor_applied().add(1, attributes={"category": category, "vetoed": "true" if vetoed else "false"})


def record_recency_channel_fired(*, category: str) -> None:
    """Record one invocation of the parallel recency channel."""
    _get_recency_channel_fired().add(1, attributes={"category": category})


def record_top1_age_days(age_days: float) -> None:
    """Record the age in days of the top-1 chunk for a recall result.

    No labels — this is a global recency-skew signal.
    """
    _get_top1_age_days().record(float(age_days))
