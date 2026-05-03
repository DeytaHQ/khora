"""Aggregate (service-level) OTel metric instruments.

These complement the Neo4j-pool-scoped metrics in
``storage/backends/neo4j.py`` by exposing service-wide signals that
downstream SREs can use to compute SLOs (recall/ingest latency, LLM
token spend, log-queue health) without subscribing to sampled spans.

Cardinality discipline (Phase 0): ``namespace_id`` is intentionally
**not** exposed as an attribute on any metric here — span attributes
remain the right place for that, since metrics buckets explode at the
~hundreds of distinct namespaces seen in production.

Attribute names follow OTel semantic conventions where applicable
(``db.*``, ``gen_ai.*``).
"""

from __future__ import annotations

import threading
from typing import Any

from .metrics import metric_counter, metric_gauge_callback, metric_histogram

_lock = threading.Lock()
_recall_histogram: Any | None = None
_ingest_histogram: Any | None = None
_llm_tokens_counter: Any | None = None
_llm_cost_counter: Any | None = None
_log_queue_gauge_registered = False
_log_handler_error_count = 0


def _get_recall_histogram() -> Any:
    global _recall_histogram
    if _recall_histogram is None:
        with _lock:
            if _recall_histogram is None:
                _recall_histogram = metric_histogram(
                    "khora.memory.recall.duration",
                    unit="s",
                    description="End-to-end MemoryLake.recall() latency.",
                )
    return _recall_histogram


def _get_ingest_histogram() -> Any:
    global _ingest_histogram
    if _ingest_histogram is None:
        with _lock:
            if _ingest_histogram is None:
                _ingest_histogram = metric_histogram(
                    "khora.memory.ingest.duration",
                    unit="s",
                    description="End-to-end MemoryLake.remember()/remember_batch() latency.",
                )
    return _ingest_histogram


def _get_llm_tokens_counter() -> Any:
    global _llm_tokens_counter
    if _llm_tokens_counter is None:
        with _lock:
            if _llm_tokens_counter is None:
                _llm_tokens_counter = metric_counter(
                    "khora.llm.tokens",
                    unit="tokens",
                    description="LLM tokens consumed, split by kind=prompt|completion.",
                )
    return _llm_tokens_counter


def _get_llm_cost_counter() -> Any:
    global _llm_cost_counter
    if _llm_cost_counter is None:
        with _lock:
            if _llm_cost_counter is None:
                _llm_cost_counter = metric_counter(
                    "khora.llm.cost_usd",
                    unit="USD",
                    description="Estimated LLM spend in USD.",
                )
    return _llm_cost_counter


def record_recall_duration(seconds: float, *, engine: str, mode: str, status: str) -> None:
    """Record one ``khora.recall()`` invocation.

    Attributes intentionally bounded: ``engine`` (small enum),
    ``mode`` (SearchMode enum), ``status`` (success|error).
    """
    _get_recall_histogram().record(
        seconds,
        attributes={"engine": engine, "mode": mode, "status": status},
    )


def record_ingest_duration(seconds: float, *, stage: str, status: str) -> None:
    """Record one ``remember()`` / ``remember_batch()`` invocation.

    ``stage`` is currently always ``"end_to_end"``; per-stage breakdown
    (chunk/embed/extract/store) is a Phase 5+ refinement.
    """
    _get_ingest_histogram().record(
        seconds,
        attributes={"stage": stage, "status": status},
    )


def record_llm_call_metrics(
    *,
    model: str,
    operation: str,
    status: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float | None = None,
) -> None:
    """Emit aggregate LLM metrics for a single call.

    Tokens fire as two separate observations split by ``kind`` so a
    single counter can answer either prompt-only or total queries.
    Cost is skipped when not provided (most call sites don't compute
    it inline today).
    """
    base_attrs = {"gen_ai.request.model": model, "operation": operation, "status": status}
    if prompt_tokens:
        attrs = dict(base_attrs)
        attrs["kind"] = "prompt"
        _get_llm_tokens_counter().add(int(prompt_tokens), attributes=attrs)
    if completion_tokens:
        attrs = dict(base_attrs)
        attrs["kind"] = "completion"
        _get_llm_tokens_counter().add(int(completion_tokens), attributes=attrs)
    if cost_usd is not None and cost_usd > 0:
        _get_llm_cost_counter().add(float(cost_usd), attributes=base_attrs)


# ---------------------------------------------------------------------------
# Log queue depth (loguru error proxy)
# ---------------------------------------------------------------------------
#
# loguru 0.7.3's enqueue=True path uses ``multiprocessing.SimpleQueue`` which
# exposes no ``qsize()``. We therefore surface a *handler error count* as a
# proxy for queue health -- mirroring tests/soak/test_soak.py PR #493. A
# healthy queue produces zero formatter / sink errors; saturation or sink
# breakage shows up as non-zero.


def _increment_log_handler_errors() -> None:
    """Bump the log-handler error counter (called from a counting sink)."""
    global _log_handler_error_count
    _log_handler_error_count += 1


def _observe_log_queue(_options: Any) -> Any:
    from opentelemetry.metrics import Observation

    yield Observation(_log_handler_error_count)


def register_log_queue_depth_gauge() -> None:
    """Register the log-queue-depth observable gauge (idempotent)."""
    global _log_queue_gauge_registered
    if _log_queue_gauge_registered:
        return
    metric_gauge_callback(
        "khora.log.queue.depth",
        [_observe_log_queue],
        unit="records",
        description="Loguru handler error count (queue-health proxy; SimpleQueue has no qsize).",
    )
    _log_queue_gauge_registered = True
