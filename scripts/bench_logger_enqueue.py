"""Microbenchmark: loguru sync vs enqueue=True under an async event loop.

This script reproduces the latent async-correctness bug fixed in DYT-2050.
A synchronous loguru sink (the default) blocks the calling thread during the
write, which means inside ``async def`` code every ``logger.info(...)`` call
stalls the event loop for the duration of the I/O. Switching the sink to
``enqueue=True`` hands records to a background thread and returns almost
immediately — at the cost of a small per-call pickle + IPC overhead — so the
loop can keep running other coroutines while logs drain in the background.

We emit 10_000 ``logger.info`` calls in batches of N_BATCH (=100) and, for
each batch, record the wall time the producer was blocked inside
``logger.info`` calls before yielding back to the loop. That blocked time
*is* the event-loop stall the bug is about. We deliberately do NOT run a
free-running lag probe: when an inter-batch pause is present the probe gets
flooded with thousands of near-zero ``sleep(0)`` samples between bursts, and
the real stalls drown in noise at the tail.

Six scenarios across two sink types and three duty cycles:

* **fast-file**: a buffered on-disk tempfile in text mode. Per-call cost is
  mostly userspace memcpy. This is where ``enqueue=True`` looks *bad* — its
  pickle + ``multiprocessing.SimpleQueue`` pipe write dominates what would
  otherwise be a free write.
* **slow-sink**: a Python callable sink that ``time.sleep(50us)`` per record,
  modelling realistic write latency (network log shippers, TTY flushes,
  rotating-file rotation events, anything making real syscalls).
* **handler/slow**: ``slow-sink`` plus a 5ms ``await asyncio.sleep`` between
  batches, modelling a request handler doing other async work between log
  bursts. This is the realistic Khora / Peras workload. Without *some* idle
  time for the background thread to drain, ``slow-sink`` applies sustained
  backpressure through the IPC pipe and enqueue's wins evaporate.

Run with::

    uv run python scripts/bench_logger_enqueue.py

## Measured on 2026-04-08 (Linux 6.8, Python 3.13, loguru 0.7.3)

scenario                  wall_ms  batch_p50_us  batch_p95_us  batch_p99_us  batch_max_us
fast-file/sync              72.13         711.9         756.7         777.4         852.3
fast-file/enqueue          370.38        3683.0        4264.3        4416.7        4550.3
slow-sink/sync            1486.07       15032.1       15301.5       15466.8       15556.0
slow-sink/enqueue         1538.69       15322.7       15812.2       16448.5       16762.9
handler/slow/sync         2057.80       15074.2       17295.3       18712.5       20750.2
handler/slow/enqueue      1592.83       10404.9       10876.1       11055.5       11140.4

``batch_*_us`` is the wall time the producer spent inside a batch of N_BATCH
logger.info calls before yielding — i.e. the time the event loop was stalled
on logging. Numbers above are from a single representative run; run-to-run
variance on a shared machine is ≈15% for ``slow-sink``/``handler/slow`` and
<5% for ``fast-file``. Directional conclusions (fast-file is ~5x slower
under enqueue; handler/slow p99 and wall time drop by ~25-40% and ~23%
respectively) are stable across runs.

What the table actually says:

1. **fast-file**: enqueue is ~5x slower per batch. The cost of pickle +
   ``multiprocessing.SimpleQueue`` pipe write dominates a buffered memcpy.
   We accept this overhead because production sinks are not free.
2. **slow-sink (no idle)**: enqueue does NOT help — both scenarios stall the
   loop ~15ms per batch. Cause: the IPC pipe fills immediately, the producer
   blocks waiting for the background thread to drain, and the background
   thread is itself bottlenecked on the slow sink. Sustained log throughput
   above sink capacity defeats enqueue. Important footnote, not a regression.
3. **handler/slow (5ms idle between batches)**: this is the realistic case.
   sync stalls the loop ~15ms per batch (every batch); enqueue cuts that
   materially at p99 AND overall wall time drops by a similar amount. The
   5ms inter-batch idle gives the background thread room to drain, so the
   IPC pipe stays mostly empty and each enqueue push hits the cheap
   no-contention path.

Bottom line: ``enqueue=True`` is the correct default for an async-first
library. It removes the worst stalls in handler-shaped code, costs measurable
but acceptable overhead on fast sinks, and leaves the throughput-saturated
case neither helped nor harmed. ``khora.logging_config.setup_logging`` now
passes ``enqueue=True`` on all sinks it installs and registers
``logger.complete`` via ``atexit`` so queued records drain on clean shutdown.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import tempfile
import time
from typing import Any

from loguru import logger

# Total log calls per scenario.
N_MESSAGES = 10_000
# Batch size: how many logger.info calls the producer makes before yielding
# back to the loop. Each batch boundary is a measurement point — we record
# how long the producer was blocked inside that batch.
N_BATCH = 100
# Slow-sink write latency (per record). Models a network log shipper round
# trip or a TTY flush. 50us is conservative; real network sinks are often
# 200us-1ms per write.
SLOW_SINK_SLEEP_S = 50e-6
# Inter-batch idle for the handler-shaped scenario. Models an async handler
# doing other work (DB query, HTTP call) between log bursts. 5ms is a
# reasonable lower bound for the "quiet" portion of a request.
HANDLER_IDLE_S = 5e-3


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile (0-100) of ``values`` in microseconds."""
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank percentile — fine for a microbenchmark.
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k] * 1_000_000


def _stats(batch_stalls: list[float], wall_ms: float) -> dict[str, Any]:
    return {
        "wall_ms": wall_ms,
        "batch_p50_us": _percentile(batch_stalls, 50),
        "batch_p95_us": _percentile(batch_stalls, 95),
        "batch_p99_us": _percentile(batch_stalls, 99),
        "batch_max_us": (max(batch_stalls) * 1_000_000) if batch_stalls else 0.0,
        "batch_mean_us": (statistics.fmean(batch_stalls) * 1_000_000 if batch_stalls else 0.0),
        "n_batches": len(batch_stalls),
    }


async def _emit_and_measure(
    n: int,
    batch: int,
    inter_batch_sleep_s: float,
) -> list[float]:
    """Emit ``n`` logger.info calls in ``batch``-sized chunks.

    Returns the list of per-batch producer-blocked times (seconds). Between
    batches, sleeps ``inter_batch_sleep_s`` to model a handler doing other
    work; passes ``0.0`` for back-to-back logging.
    """
    stalls: list[float] = []
    batch_start = time.perf_counter()
    for i in range(n):
        logger.info("benchmark message {}", i)
        if (i + 1) % batch == 0:
            stalls.append(time.perf_counter() - batch_start)
            if inter_batch_sleep_s > 0:
                await asyncio.sleep(inter_batch_sleep_s)
            else:
                await asyncio.sleep(0)
            batch_start = time.perf_counter()
    return stalls


def _slow_sink(message: Any) -> None:
    """Callable sink that sleeps to model real-world write latency.

    Loguru invokes this for every record. The sleep is the point: a real
    network logger or TTY flush is not free, and the sync path blocks the
    event loop for the full duration.
    """
    time.sleep(SLOW_SINK_SLEEP_S)


def _configure_fast_file_sink(enqueue: bool) -> Any:
    """Install a single buffered tempfile sink. Returns the file handle."""
    logger.remove()
    tf = tempfile.NamedTemporaryFile(mode="w", delete=False)
    logger.add(
        tf,
        level="INFO",
        format="{time} | {level} | {message}",
        enqueue=enqueue,
    )
    return tf


def _configure_slow_sink(enqueue: bool) -> None:
    """Install a single slow-callable sink that sleeps 50us per record."""
    logger.remove()
    logger.add(
        _slow_sink,
        level="INFO",
        format="{time} | {level} | {message}",
        enqueue=enqueue,
    )


def _teardown(tempfile_handle: Any = None) -> None:
    """Drain the queue and remove all sinks. Idempotent across scenarios."""
    logger.complete()
    logger.remove()
    if tempfile_handle is not None:
        try:
            tempfile_handle.close()
            os.unlink(tempfile_handle.name)
        except OSError:
            pass


def run_fast_file(enqueue: bool) -> dict[str, Any]:
    tf = _configure_fast_file_sink(enqueue)
    try:
        start = time.perf_counter()
        stalls = asyncio.run(_emit_and_measure(N_MESSAGES, N_BATCH, inter_batch_sleep_s=0.0))
        # Include drain in wall_ms so enqueue=True does not get an unfair
        # head start by hiding work in the background thread.
        logger.complete()
        wall_ms = (time.perf_counter() - start) * 1000.0
        return _stats(stalls, wall_ms)
    finally:
        _teardown(tf)


def run_slow_sink(enqueue: bool) -> dict[str, Any]:
    _configure_slow_sink(enqueue)
    try:
        start = time.perf_counter()
        stalls = asyncio.run(_emit_and_measure(N_MESSAGES, N_BATCH, inter_batch_sleep_s=0.0))
        logger.complete()
        wall_ms = (time.perf_counter() - start) * 1000.0
        return _stats(stalls, wall_ms)
    finally:
        _teardown()


def run_handler_slow(enqueue: bool) -> dict[str, Any]:
    _configure_slow_sink(enqueue)
    try:
        start = time.perf_counter()
        stalls = asyncio.run(_emit_and_measure(N_MESSAGES, N_BATCH, inter_batch_sleep_s=HANDLER_IDLE_S))
        logger.complete()
        wall_ms = (time.perf_counter() - start) * 1000.0
        return _stats(stalls, wall_ms)
    finally:
        _teardown()


def _format_row(label: str, r: dict[str, Any]) -> str:
    return (
        f"{label:<22} {r['wall_ms']:>10.2f} "
        f"{r['batch_p50_us']:>13.1f} "
        f"{r['batch_p95_us']:>13.1f} "
        f"{r['batch_p99_us']:>13.1f} "
        f"{r['batch_max_us']:>13.1f}"
    )


def main() -> None:
    header = (
        f"{'scenario':<22} {'wall_ms':>10} "
        f"{'batch_p50_us':>13} "
        f"{'batch_p95_us':>13} "
        f"{'batch_p99_us':>13} "
        f"{'batch_max_us':>13}"
    )
    print(header)
    print("-" * len(header))
    print(_format_row("fast-file/sync", run_fast_file(enqueue=False)))
    print(_format_row("fast-file/enqueue", run_fast_file(enqueue=True)))
    print(_format_row("slow-sink/sync", run_slow_sink(enqueue=False)))
    print(_format_row("slow-sink/enqueue", run_slow_sink(enqueue=True)))
    print(_format_row("handler/slow/sync", run_handler_slow(enqueue=False)))
    print(_format_row("handler/slow/enqueue", run_handler_slow(enqueue=True)))


if __name__ == "__main__":
    main()
