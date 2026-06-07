"""OTel counters for the deterministic recall filter — ``@internal``.

Three service-level counters that surface how the deterministic recall
filter behaves at compile/recall time. They complement the per-recall
``filter.*`` span attributes (on ``khora.recall``) with aggregate signals
SREs can alert on without subscribing to sampled spans.

The instruments use the same lazy-init + :class:`threading.Lock` pattern as
:mod:`khora.telemetry.aggregate_metrics`: each is created on first use behind
a double-checked lock so a cold import never builds an instrument, and so the
no-op meter (no real ``MeterProvider`` installed) stays free to call.

Cardinality discipline (matching the rest of khora's metrics): ``namespace_id``
is **never** an attribute here — it lives on span attributes only. The single
bounded label exposed is ``op`` (the metadata leaf's comparison operator, a
member of the closed :class:`~khora.filter.model.Op` enum).
"""

from __future__ import annotations

import threading
from typing import Any

from khora.telemetry.metrics import metric_counter

_lock = threading.Lock()
_unindexed_metadata_counter: Any | None = None
_under_filled_counter: Any | None = None
_graph_channel_empty_counter: Any | None = None


def _get_unindexed_metadata_counter() -> Any:
    global _unindexed_metadata_counter
    if _unindexed_metadata_counter is None:
        with _lock:
            if _unindexed_metadata_counter is None:
                _unindexed_metadata_counter = metric_counter(
                    "khora.recall.filter.unindexed_metadata",
                    unit="1",
                    description=(
                        "Metadata leaves in a deterministic recall filter that compile to an "
                        "unindexed JSONB column access. Split by op (the leaf's comparison operator)."
                    ),
                )
    return _unindexed_metadata_counter


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


def record_unindexed_metadata(*, op: str) -> None:
    """Record one metadata leaf that compiled to an unindexed JSONB access.

    Fired once per metadata leaf, so a filter with N metadata predicates emits
    N observations (each leaf is a distinct unindexed-column access). ``op`` is
    the leaf's comparison operator wire literal (an :class:`~khora.filter.model.Op`
    value, e.g. ``"$eq"``) — the only attribute, and bounded by that closed enum.
    Never pass ``namespace_id`` or a metadata key-path (both unbounded).
    """
    _get_unindexed_metadata_counter().add(1, attributes={"op": op})
