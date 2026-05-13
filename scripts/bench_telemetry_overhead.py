"""Benchmark trace_span / @trace overhead on the no-provider path.

The OTel-first migration trades a logfire ``_HAS_LOGFIRE`` short-circuit
for a call into OTel's ``ProxyTracer.start_as_current_span``. The proxy
returns a ``NonRecordingSpan`` quickly, but it's not literally zero —
this script measures the per-call cost and flags regressions.

Run::

    uv run python scripts/bench_telemetry_overhead.py

The script prints a comparison table and exits non-zero if any
benchmark exceeds the regression budget. CI uses this as a soft gate.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from khora.telemetry import trace_span
from khora.telemetry.trace_decorator import trace

# Budget per call (microseconds) on the no-provider (ProxyTracer) path.
# Measured baseline as of 0.10.8 / opentelemetry-api 1.27 is ~2.8 µs/call
# (3 µs total - 0.4 µs noop CM). We allow ~2x headroom to accommodate
# OTel SDK version variance; a real regression (4x slowdown, ~10 µs)
# would fail this gate. At 5 µs/call, even an unusually chatty 50-span
# recall pays only 0.25 ms — far below any meaningful threshold for a
# knowledge memory library where ops take 10-100+ ms (DB I/O, LLM).
REGRESSION_BUDGET_PER_CALL_US = 5.0

ITERATIONS = 200_000


def _baseline_loop() -> float:
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        pass
    return (time.perf_counter() - start) / ITERATIONS * 1_000_000


def _trace_span_loop() -> float:
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        with trace_span("khora.bench") as span:
            span.set_attribute("foo", "bar")
    return (time.perf_counter() - start) / ITERATIONS * 1_000_000


@trace("khora.bench_decorated")
def _decorated(x: int) -> int:
    return x + 1


def _trace_decorator_loop() -> float:
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        _decorated(0)
    return (time.perf_counter() - start) / ITERATIONS * 1_000_000


@contextmanager
def _noop_cm():
    yield


def _noop_cm_loop() -> float:
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        with _noop_cm() as _x:
            pass
    return (time.perf_counter() - start) / ITERATIONS * 1_000_000


def main() -> int:
    print(f"Iterations: {ITERATIONS:,}")
    print()
    print(f"{'benchmark':<30} {'µs/call':>10}  {'note':<30}")
    print("-" * 75)
    baseline = _baseline_loop()
    print(f"{'empty loop':<30} {baseline:>10.3f}  {'pure for-loop':<30}")
    noop = _noop_cm_loop()
    print(f"{'noop context manager':<30} {noop:>10.3f}  {'@contextmanager + yield':<30}")
    ts = _trace_span_loop()
    print(f"{'trace_span (no provider)':<30} {ts:>10.3f}  {'proxy tracer NonRecordingSpan':<30}")
    dec = _trace_decorator_loop()
    print(f"{'@trace decorator (no provider)':<30} {dec:>10.3f}  {'sig.bind + trace_span':<30}")

    print()
    delta_ts = ts - noop
    print(f"trace_span overhead vs noop CM:  {delta_ts:+.3f} µs/call")
    if delta_ts > REGRESSION_BUDGET_PER_CALL_US:
        print(
            f"FAIL: trace_span overhead {delta_ts:.3f} µs/call exceeds budget {REGRESSION_BUDGET_PER_CALL_US} µs/call"
        )
        return 1
    print("OK: trace_span overhead within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
