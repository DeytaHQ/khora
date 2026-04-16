"""Cold-pool Neo4j acquire-duration smoke test.

Satisfies DYT-2624 AC2: verify ``khora.neo4j.pool.acquire_duration``
records *real* pool acquire time (not query time + retry delays) by
comparing against wall-clock for N parallel ``RETURN 1`` calls on a
cold pool.

Usage::

    uv run python scripts/bench_neo4j_pool_acquire.py \\
        --url bolt://localhost:7687 \\
        --user neo4j \\
        --password password \\
        --pool-size 5 \\
        --concurrency 20 \\
        --timeout 2.0

The script:

1. Opens a fresh ``Neo4jBackend`` (cold pool — first N requests will
   all wait to create connections).
2. Fires ``--concurrency`` parallel ``RETURN 1`` calls and measures
   each call's wall-clock outside the instrumentation.
3. Reads the per-call ``_InstrumentedSession.last_acquire`` values.
4. Asserts recorded acquire p99 < wall-clock p99 (acquire is strictly
   less than total).
5. Fires a second burst with ``concurrency = pool_size + 5`` and
   ``--timeout=0.5`` to force pool exhaustion; asserts that the
   ``khora.neo4j.pool.timeout`` counter increments by at least 5.

Exits 1 on assertion failure so this can be wired into CI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from typing import Any

# Local imports deferred until after we set sys.path (uv run handles
# this automatically, but running as a raw script should still work).
try:
    from neo4j.exceptions import ConnectionAcquisitionTimeoutError

    from khora.storage.backends.neo4j import Neo4jBackend
except ImportError as exc:  # pragma: no cover
    print(f"Failed to import khora/neo4j: {exc}", file=sys.stderr)
    print("Run via `uv run python scripts/bench_neo4j_pool_acquire.py ...`", file=sys.stderr)
    sys.exit(2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--url", required=True, help="Neo4j bolt URL")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="connection_acquisition_timeout (seconds)",
    )
    return parser.parse_args()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


async def _one_call(backend: Neo4jBackend) -> tuple[float, float]:
    """Return (wall_clock_s, recorded_acquire_s) for a single RETURN 1."""
    t0 = time.monotonic()
    acquire: float = 0.0
    async with backend._session() as session:
        await session.run("RETURN 1")
        acquire = session.last_acquire
    wall = time.monotonic() - t0
    return wall, acquire


async def _burst(backend: Neo4jBackend, n: int) -> list[tuple[float, float]]:
    results = await asyncio.gather(*[_one_call(backend) for _ in range(n)], return_exceptions=True)
    return [r for r in results if isinstance(r, tuple)]


async def _burst_allow_timeout(backend: Neo4jBackend, n: int) -> tuple[list[tuple[float, float]], int]:
    """Return (completed, timeout_count)."""
    results = await asyncio.gather(*[_one_call(backend) for _ in range(n)], return_exceptions=True)
    completed: list[tuple[float, float]] = []
    timeouts = 0
    for r in results:
        if isinstance(r, tuple):
            completed.append(r)
        elif isinstance(r, ConnectionAcquisitionTimeoutError):
            timeouts += 1
    return completed, timeouts


def _read_counter_value(backend: Neo4jBackend) -> int | None:
    """Best-effort read of the timeout counter's integer value.

    When logfire is installed, the counter may be an opentelemetry
    ``Counter`` instance that doesn't expose its current value. We wrap
    the ``.add`` method below to track increments ourselves instead; the
    helper returns None on the non-logfire path (counter is a no-op).
    """
    counter = backend._timeout_counter
    return getattr(counter, "_bench_count", None)


def _wrap_counter_to_track_adds(backend: Neo4jBackend) -> None:
    counter = backend._timeout_counter
    counter._bench_count = 0  # type: ignore[attr-defined]
    original_add = counter.add

    def tracked_add(value: int | float = 1, **kwargs: Any) -> None:
        counter._bench_count += int(value)  # type: ignore[attr-defined]
        return original_add(value, **kwargs)

    counter.add = tracked_add  # type: ignore[method-assign]


async def _main(args: argparse.Namespace) -> int:
    print(f"Neo4j pool acquire smoke test: url={args.url} pool_size={args.pool_size} concurrency={args.concurrency}")

    # First backend: warm burst, timeout generous — verifies acquire <= wall.
    backend = Neo4jBackend(
        url=args.url,
        user=args.user,
        password=args.password,
        max_connection_pool_size=args.pool_size,
        connection_acquisition_timeout=args.timeout,
        pool_sampler_enabled=True,
        pool_sampler_interval_ms=100,
    )
    _wrap_counter_to_track_adds(backend)

    await backend.connect()
    try:
        samples = await _burst(backend, args.concurrency)
        if len(samples) != args.concurrency:
            print(f"Burst 1: expected {args.concurrency} successes, got {len(samples)}", file=sys.stderr)
            return 1

        walls = [w for w, _ in samples]
        acquires = [a for _, a in samples if a > 0]

        wall_p50, wall_p99 = _percentile(walls, 50), _percentile(walls, 99)
        acq_p50 = _percentile(acquires, 50) if acquires else float("nan")
        acq_p99 = _percentile(acquires, 99) if acquires else float("nan")

        print(f"\n=== Burst 1 (concurrency={args.concurrency}, cold pool, generous timeout) ===")
        print(f"  wall-clock  p50={wall_p50 * 1000:.2f} ms   p99={wall_p99 * 1000:.2f} ms")
        print(f"  acquire     p50={acq_p50 * 1000:.2f} ms   p99={acq_p99 * 1000:.2f} ms")
        print(f"  mean wall   = {statistics.mean(walls) * 1000:.2f} ms")
        print(f"  mean acquire= {statistics.mean(acquires) * 1000:.2f} ms" if acquires else "  (no acquires recorded)")

        # Sanity: acquire p99 should be <= wall p99 (it's a subset of wall)
        if acquires and acq_p99 > wall_p99 + 0.001:  # 1ms slack
            print(
                f"FAIL: recorded acquire p99 ({acq_p99 * 1000:.2f} ms) exceeds "
                f"wall-clock p99 ({wall_p99 * 1000:.2f} ms)",
                file=sys.stderr,
            )
            return 1
    finally:
        await backend.disconnect()

    # Second backend: saturate to force timeouts.
    saturate_n = args.pool_size + 5
    backend2 = Neo4jBackend(
        url=args.url,
        user=args.user,
        password=args.password,
        max_connection_pool_size=args.pool_size,
        connection_acquisition_timeout=0.5,
    )
    _wrap_counter_to_track_adds(backend2)
    await backend2.connect()
    try:
        print(f"\n=== Burst 2 (concurrency={saturate_n * 3}, tight timeout=0.5s, expect ≥5 timeouts) ===")

        # Crank concurrency high so at least 5 requests time out waiting.
        # Hold acquired connections briefly inside the task to force queue
        # build-up beyond pool_size.
        async def hog() -> tuple[float, float] | BaseException:
            try:
                t0 = time.monotonic()
                async with backend2._session() as s:
                    await s.run("RETURN 1")
                    await asyncio.sleep(0.8)  # keep connection busy longer than timeout
                    acquire = s.last_acquire
                wall = time.monotonic() - t0
                return wall, acquire
            except ConnectionAcquisitionTimeoutError as e:
                return e

        results = await asyncio.gather(*[hog() for _ in range(saturate_n * 3)], return_exceptions=True)
        timeouts = sum(1 for r in results if isinstance(r, ConnectionAcquisitionTimeoutError))
        print(f"  timeouts observed: {timeouts}")
        counter_val = _read_counter_value(backend2)
        print(f"  timeout counter value: {counter_val}")
        if timeouts < 5:
            print(f"FAIL: expected ≥5 timeouts with pool_size={args.pool_size}, got {timeouts}", file=sys.stderr)
            return 1
        if counter_val is not None and counter_val != timeouts:
            print(
                f"FAIL: timeout counter ({counter_val}) disagrees with observed timeouts ({timeouts})",
                file=sys.stderr,
            )
            return 1
    finally:
        await backend2.disconnect()

    print("\nAll assertions passed.")
    return 0


def main() -> None:
    args = _parse_args()
    # Optional logfire export — no-op if khora[logfire] isn't installed.
    token = os.environ.get("KHORA_LOGFIRE_TOKEN")
    if token:
        try:
            import logfire  # noqa: F401

            logfire.configure(token=token, send_to_logfire=True)
        except ImportError:
            pass
    exit_code = asyncio.run(_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
