"""Cold-pool Neo4j acquire-duration smoke test (AC2).

Verifies that ``khora.neo4j.pool.acquire_duration`` records strictly less
than wall-clock for N parallel ``RETURN 1`` calls on a cold pool.

Usage::

    uv run python scripts/bench_neo4j_pool_acquire.py \\
        --url bolt://localhost:7687 \\
        --user neo4j \\
        --password password \\
        --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from typing import Any

from khora.storage.backends.neo4j import Neo4jBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    return parser.parse_args()


def _pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


async def _one_call(backend: Neo4jBackend, recorded: list[float]) -> float:
    t0 = time.perf_counter()
    async with backend._session() as session:
        await session.run("RETURN 1")
        recorded.append(session.last_acquire)
    return time.perf_counter() - t0


async def _main(args: argparse.Namespace) -> int:
    backend = Neo4jBackend(url=args.url, user=args.user, password=args.password)
    recorded: list[float] = []
    # Monkey-patch the acquire histogram to capture every .record() call.
    original_record = backend._acquire_duration_histogram.record

    def spy_record(value: float, **kwargs: Any) -> None:
        recorded.append(value)
        original_record(value, **kwargs)

    backend._acquire_duration_histogram.record = spy_record  # type: ignore[method-assign]

    await backend.connect()
    try:
        wall_times = await asyncio.gather(*[_one_call(backend, recorded) for _ in range(args.concurrency)])
    finally:
        await backend.disconnect()

    wall_p50, wall_p99 = _pct(wall_times, 50), _pct(wall_times, 99)
    acq_p50, acq_p99 = _pct(recorded, 50), _pct(recorded, 99)

    print(f"wall-clock  p50={wall_p50 * 1000:.2f}ms  p99={wall_p99 * 1000:.2f}ms")
    print(f"acquire     p50={acq_p50 * 1000:.2f}ms  p99={acq_p99 * 1000:.2f}ms")
    print(f"mean wall    = {statistics.mean(wall_times) * 1000:.2f}ms")
    print(f"mean acquire = {statistics.mean(recorded) * 1000:.2f}ms")

    if acq_p99 >= wall_p99:
        print(f"FAIL: acquire p99 ({acq_p99 * 1000:.2f}ms) >= wall p99 ({wall_p99 * 1000:.2f}ms)", file=sys.stderr)
        return 1
    print("OK: acquire p99 < wall p99")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main(_parse_args())))


if __name__ == "__main__":
    main()
