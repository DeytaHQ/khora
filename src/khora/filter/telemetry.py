"""OTel counters for the deterministic recall filter — ``@internal``.

Two service-level counters that surface how the deterministic recall
filter behaves at compile/recall time. They complement the per-recall
``filter.*`` span attributes (on ``khora.recall``) with aggregate signals
SREs can alert on without subscribing to sampled spans.

The instruments use the same lazy-init + :class:`threading.Lock` pattern as
:mod:`khora.telemetry.aggregate_metrics`: each is created on first use behind
a double-checked lock so a cold import never builds an instrument, and so the
no-op meter (no real ``MeterProvider`` installed) stays free to call.

Cardinality discipline (matching the rest of khora's metrics): ``namespace_id``
is **never** an attribute here — it lives on span attributes only.
"""

from __future__ import annotations

import threading
from typing import Any

from khora.telemetry.metrics import metric_counter

_lock = threading.Lock()
_under_filled_counter: Any | None = None
_graph_channel_empty_counter: Any | None = None


def _get_under_filled_counter() -> Any:
    global _under_filled_counter
    if _under_filled_counter is None:
        with _lock:
            if _under_filled_counter is None:
                _under_filled_counter = metric_counter(
                    "khora.recall.filter.under_filled",
                    unit="1",
                    description=("Filtered recalls that returned fewer results than the requested limit."),
                )
    return _under_filled_counter


def _get_graph_channel_empty_counter() -> Any:
    global _graph_channel_empty_counter
    if _graph_channel_empty_counter is None:
        with _lock:
            if _graph_channel_empty_counter is None:
                _graph_channel_empty_counter = metric_counter(
                    "khora.recall.filter.graph_channel_empty",
                    unit="1",
                    description=(
                        "Filtered recalls whose graph channel returned no candidates "
                        "(the filter narrowed the graph side to empty)."
                    ),
                )
    return _graph_channel_empty_counter


def record_under_filled() -> None:
    """Record one filtered recall that returned fewer results than the requested limit.

    Fired once per recall (no attributes) when a caller filter narrowed the
    candidate set below the requested ``k``. Never pass ``namespace_id``.
    """
    _get_under_filled_counter().add(1)


def record_graph_channel_empty() -> None:
    """Record one filtered recall whose graph channel was narrowed to empty.

    Fired once per recall (no attributes) when a caller filter eliminated every
    graph-channel candidate. Never pass ``namespace_id``.
    """
    _get_graph_channel_empty_counter().add(1)
