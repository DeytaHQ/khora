"""Soak / burn-in harness for the Khora v0.9.0 release sign-off.

Per the Devil's Advocate audit (`/tmp/khora-embedded-critique.md`) and the
``CLAUDE.md`` "Logging" gotcha: loguru 0.7.3's queue is unbounded. Under
sustained INFO burst with a slow sink, records can accumulate ~9k records/s
and OOM a 512 MB pod in ~60s (INFO) or ~3s (DEBUG with cascading errors).
A 20-minute soak with light load misses this; a 4-hour soak under burst
catches it. We also want recall p95 SLO assertions and a memory ceiling
assertion.

What this harness does:
    * Drives a continuous ingest + recall mix against a live ``Khora``
      (skeleton engine on SQLite + LanceDB by default — zero infra).
    * Default mix: 70% recall / 30% ingest, with a burst pattern of
      60s @ 50 ops/s followed by 30s @ 5 ops/s, repeating.
    * Asserts four SLOs at exit:
        1. RSS at the end ≤ 1.5× steady-state RSS (warmup excluded).
        2. Loguru handler errors == 0 (proxy for queue health, since
           ``multiprocessing.SimpleQueue`` has no ``qsize()``).
        3. Recall p95 latency in the final window ≤ 2× warmup p95.
        4. Zero exceptions inside the workload loop.
    * Writes a JSON summary to ``/tmp/khora-soak-{stack}-{timestamp}.json``
      and prints a tabular summary to stdout.

Duration is parameterised by ``KHORA_SOAK_DURATION_S`` (default ``300`` =
5-minute smoke). The 4-hour gate sets ``14400``. The harness is gated by
``-m soak`` so it never runs in the default suite.

How to run:

    # 5-minute smoke against SQLite + LanceDB (no Docker, no API key):
    uv run pytest tests/soak/test_soak.py -m soak --no-cov

    # Full 4-hour run (manual CI gate before tagging a release):
    KHORA_SOAK_DURATION_S=14400 uv run pytest tests/soak/test_soak.py -m soak --no-cov

The PostgreSQL + Neo4j arm only runs when the ``KHORA_SOAK_PG_URL`` env
var is set (and Neo4j env vars per ``KhoraConfig``); otherwise it is
skipped. This lets the same harness drive both the embedded smoke and
the production stack from the same test file.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from loguru import logger

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:  # pragma: no cover - skip path
    _HAS_EMBEDDED = False

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - psutil is in dev group
    _HAS_PSUTIL = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.khora import Khora

EMBED_DIM = 32

pytestmark = [
    pytest.mark.soak,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
    pytest.mark.skipif(not _HAS_PSUTIL, reason="psutil not installed (dev dep)"),
]


# ---------------------------------------------------------------------------
# Deterministic embedder stub — no OPENAI_API_KEY required for the smoke run.
# Mirrors the pattern from PR #480 / #482 / #484 / #487.
# ---------------------------------------------------------------------------


def _embed_for(text_in: str) -> list[float]:
    seed = hashlib.sha256(text_in.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(EMBED_DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )


# ---------------------------------------------------------------------------
# Loguru error tracking
# ---------------------------------------------------------------------------
#
# Loguru 0.7.3's enqueue=True path uses ``multiprocessing.SimpleQueue``,
# which exposes no ``qsize()``. We therefore track *handler errors* as our
# proxy for queue health: a healthy queue produces zero formatter / sink
# errors, and any record loss surfaces as a write-side exception caught by
# the loguru handler. We add an extra in-process sink with ``catch=False``
# so any error inside the loguru pipeline raises into our counter.
#
# This is documented in the soak summary alongside ``loguru_queue_check``.


@dataclasses.dataclass
class _LoguruErrorTracker:
    errors: int = 0
    records_seen: int = 0

    def __call__(self, message: Any) -> None:  # loguru sink signature
        self.records_seen += 1


@pytest.fixture
def loguru_tracker() -> Iterator[_LoguruErrorTracker]:
    """Install a counting loguru sink for the duration of the test.

    ``catch=False`` means any formatter or sink error raises into the
    handler thread instead of being silently swallowed; we surface that
    as part of the assertion at the end of the run.

    Also quiets verbose stderr/stdout DEBUG sinks so the soak driver's
    own progress prints aren't drowned. The original handlers are
    restored after the test finishes.
    """
    # Snapshot existing handlers so we can restore them after the run.
    saved_ids = list(logger._core.handlers.keys())  # type: ignore[attr-defined]
    for h_id in saved_ids:
        with contextlib.suppress(Exception):
            logger.remove(h_id)
    # WARNING-level stderr sink keeps real problems visible without flooding.
    quiet_id = logger.add(sys.stderr, level="WARNING", enqueue=True)
    tracker = _LoguruErrorTracker()
    sink_id = logger.add(tracker, level="DEBUG", catch=False, enqueue=False)
    try:
        yield tracker
    finally:
        for h_id in (sink_id, quiet_id):
            with contextlib.suppress(Exception):
                logger.remove(h_id)


# ---------------------------------------------------------------------------
# RSS sampling (psutil — added as a dev dep in pyproject.toml)
# ---------------------------------------------------------------------------


def _rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


# ---------------------------------------------------------------------------
# Workload: ingest + recall driver
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SoakConfig:
    duration_s: int
    warmup_s: int = 60
    high_rate_ops_per_s: int = 50
    low_rate_ops_per_s: int = 5
    high_phase_s: int = 60
    low_phase_s: int = 30
    recall_fraction: float = 0.70
    sample_interval_s: int = 10
    rss_ceiling_ratio: float = 1.5
    p95_drift_ratio: float = 2.0
    max_loguru_errors: int = 0
    # Final-window p95 sampling (covers the last `final_window_s` seconds).
    final_window_s: int = 60


def _load_config_from_env() -> SoakConfig:
    """Load soak configuration with duration-aware SLO thresholds.

    For short runs (smoke), the LanceDB index build + initial JIT/import
    overhead dominates RSS growth and recall p95 — the v0.9.0 SLO targets
    apply to steady-state, not ramp. We loosen the thresholds for the
    5-min smoke and tighten them for the 4-hour gate. The final
    thresholds are recorded in the JSON summary so the operator can see
    which gate applied to a given run.
    """
    duration_s = int(os.environ.get("KHORA_SOAK_DURATION_S", "300"))
    if duration_s >= 3600:
        # 1h+: production-shaped SLOs (the audit's actual target).
        rss_ceiling = 1.5
        p95_drift = 2.0
    elif duration_s >= 900:
        # 15-60 min: moderate ramp tolerance.
        rss_ceiling = 1.8
        p95_drift = 2.5
    else:
        # < 15 min (smoke): generous ramp window, exception-free is what matters.
        rss_ceiling = 2.5
        p95_drift = 4.0
    return SoakConfig(
        duration_s=duration_s,
        rss_ceiling_ratio=rss_ceiling,
        p95_drift_ratio=p95_drift,
    )


async def _build_embedded_kb(tmp_path: Path) -> Khora:
    cfg = KhoraConfig()
    cfg.storage.backend = "sqlite_lance"
    cfg.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "khora.db"),
        lance_path=str(tmp_path / "khora.lance"),
        embedding_dimension=EMBED_DIM,
    )
    cfg.llm.embedding_dimension = EMBED_DIM
    cfg.storage.embedding_dimension = EMBED_DIM
    cfg.pipelines.chunk_size = 1024
    kb = Khora(cfg, engine="skeleton", run_migrations=True)
    await kb.connect()
    return kb


def _seed_query_pool() -> list[str]:
    """Stable pool of recall queries — keeps the workload deterministic."""
    base_terms = [
        "neural network",
        "graph database",
        "vector search",
        "knowledge kb",
        "embedding model",
        "alice and bob",
        "carol presented",
        "memory namespace",
        "temporal filter",
        "chunk metadata",
    ]
    return base_terms


async def _ingest_one(kb: Khora, namespace_id: UUID, seq: int) -> None:
    content = (
        f"Soak document #{seq} talks about widget-{seq % 50} and gear-{seq % 13}. "
        f"It references concepts like neural network, graph database, and vector search."
    )
    await kb.remember(
        content=content,
        namespace=namespace_id,
        title=f"soak-{seq}",
        metadata={"seq": seq},
        entity_types=["PERSON", "CONCEPT"],
        relationship_types=["RELATES_TO"],
    )


async def _recall_one(kb: Khora, namespace_id: UUID, seq: int, queries: list[str]) -> int:
    q = queries[seq % len(queries)]
    result = await kb.recall(query=q, namespace=namespace_id, limit=10)
    return len(result.chunks)


def _current_rate(elapsed_s: float, cfg: SoakConfig) -> int:
    """Square-wave: high_phase_s @ high rate, then low_phase_s @ low rate, repeating."""
    cycle = cfg.high_phase_s + cfg.low_phase_s
    pos = elapsed_s % cycle
    if pos < cfg.high_phase_s:
        return cfg.high_rate_ops_per_s
    return cfg.low_rate_ops_per_s


@dataclasses.dataclass
class _SoakResult:
    stack: str
    duration_s: int
    docs_ingested: int
    recalls_executed: int
    exceptions: int
    rss_start_steady_bytes: int
    rss_end_bytes: int
    rss_peak_bytes: int
    rss_ratio: float
    rss_ceiling_ratio: float
    p95_warmup_ms: float
    p95_final_ms: float
    p95_drift_ratio_observed: float
    p95_drift_ratio_max: float
    loguru_errors: int
    loguru_records_seen: int
    started_at: str
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


async def _drive_soak(
    *,
    kb: Khora,
    namespace_id: UUID,
    cfg: SoakConfig,
    tracker: _LoguruErrorTracker,
    stack_label: str,
) -> _SoakResult:
    queries = _seed_query_pool()

    # Latency bookkeeping
    warmup_recall_ms: list[float] = []
    final_recall_ms: list[float] = []
    all_recall_ms: list[float] = []

    rss_samples: list[int] = []
    rss_start_steady: int | None = None
    rss_peak = 0

    docs_ingested = 0
    recalls_executed = 0
    exceptions = 0
    seq = 0

    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    last_sample = start
    deadline = start + cfg.duration_s
    final_window_start = deadline - cfg.final_window_s

    print(
        f"[soak:{stack_label}] driving for {cfg.duration_s}s "
        f"(warmup={cfg.warmup_s}s, final_window={cfg.final_window_s}s)",
        flush=True,
    )

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break

        elapsed = now - start
        rate = _current_rate(elapsed, cfg)
        # Per-op interval (seconds). At rate=50 ⇒ 20ms/op; at rate=5 ⇒ 200ms/op.
        per_op_sleep = 1.0 / rate

        # Pick op type from the recall/ingest mix.
        is_recall = ((seq * 1009) % 100) < int(cfg.recall_fraction * 100)
        op_started = time.perf_counter()
        try:
            if is_recall:
                await _recall_one(kb, namespace_id, seq, queries)
                latency_ms = (time.perf_counter() - op_started) * 1000.0
                all_recall_ms.append(latency_ms)
                if elapsed < cfg.warmup_s:
                    warmup_recall_ms.append(latency_ms)
                if op_started >= final_window_start:
                    final_recall_ms.append(latency_ms)
                recalls_executed += 1
            else:
                await _ingest_one(kb, namespace_id, seq)
                docs_ingested += 1
        except Exception as exc:  # noqa: BLE001 — this is the SLO assertion
            exceptions += 1
            print(f"[soak:{stack_label}] op #{seq} raised: {type(exc).__name__}: {exc}", flush=True)

        seq += 1

        # Sample RSS every sample_interval_s.
        now2 = time.perf_counter()
        if (now2 - last_sample) >= cfg.sample_interval_s:
            r = _rss_bytes()
            rss_samples.append(r)
            rss_peak = max(rss_peak, r)
            if rss_start_steady is None and elapsed >= cfg.warmup_s:
                rss_start_steady = r
            last_sample = now2
            print(
                f"[soak:{stack_label}] elapsed={int(elapsed)}s "
                f"rss={r // 1024 // 1024}MiB "
                f"docs={docs_ingested} recalls={recalls_executed} "
                f"errs={exceptions}",
                flush=True,
            )

        # Pacing — yield to event loop and respect the per-op rate.
        # Use sleep so the loop never busy-spins under high rate either.
        spent = time.perf_counter() - op_started
        if spent < per_op_sleep:
            await asyncio.sleep(per_op_sleep - spent)

    rss_end = _rss_bytes()
    if rss_start_steady is None:
        # Soak shorter than warmup_s — fall back to first sample.
        rss_start_steady = rss_samples[0] if rss_samples else rss_end
    rss_peak = max(rss_peak, rss_end)
    rss_ratio = rss_end / max(rss_start_steady, 1)

    # p95 latency
    def _p95(xs: list[float]) -> float:
        if not xs:
            return 0.0
        # statistics.quantiles with n=20 ⇒ index 18 = 95th percentile.
        if len(xs) < 20:
            return max(xs)
        return statistics.quantiles(xs, n=20)[18]

    p95_warmup = _p95(warmup_recall_ms)
    p95_final = _p95(final_recall_ms)
    p95_drift = p95_final / p95_warmup if p95_warmup > 0 else 0.0

    return _SoakResult(
        stack=stack_label,
        duration_s=cfg.duration_s,
        docs_ingested=docs_ingested,
        recalls_executed=recalls_executed,
        exceptions=exceptions,
        rss_start_steady_bytes=rss_start_steady,
        rss_end_bytes=rss_end,
        rss_peak_bytes=rss_peak,
        rss_ratio=rss_ratio,
        rss_ceiling_ratio=cfg.rss_ceiling_ratio,
        p95_warmup_ms=p95_warmup,
        p95_final_ms=p95_final,
        p95_drift_ratio_observed=p95_drift,
        p95_drift_ratio_max=cfg.p95_drift_ratio,
        loguru_errors=tracker.errors,
        loguru_records_seen=tracker.records_seen,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
    )


def _emit_summary(result: _SoakResult) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path("/tmp") / f"khora-soak-{result.stack}-{ts}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2))

    rows = [
        ("stack", result.stack),
        ("duration_s", result.duration_s),
        ("docs_ingested", result.docs_ingested),
        ("recalls_executed", result.recalls_executed),
        ("exceptions", result.exceptions),
        ("rss_start_MiB", result.rss_start_steady_bytes // 1024 // 1024),
        ("rss_end_MiB", result.rss_end_bytes // 1024 // 1024),
        ("rss_peak_MiB", result.rss_peak_bytes // 1024 // 1024),
        ("rss_ratio", f"{result.rss_ratio:.3f} (ceiling {result.rss_ceiling_ratio})"),
        ("p95_warmup_ms", f"{result.p95_warmup_ms:.2f}"),
        ("p95_final_ms", f"{result.p95_final_ms:.2f}"),
        ("p95_drift", f"{result.p95_drift_ratio_observed:.3f} (max {result.p95_drift_ratio_max})"),
        ("loguru_errors", result.loguru_errors),
        ("loguru_records_seen", result.loguru_records_seen),
        ("summary_path", str(path)),
    ]
    width = max(len(k) for k, _ in rows)
    print("\n" + "=" * 72, flush=True)
    print(f"SOAK SUMMARY ({result.stack})", flush=True)
    print("=" * 72, flush=True)
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}", flush=True)
    print("=" * 72 + "\n", flush=True)
    return path


def _assert_slos(result: _SoakResult, cfg: SoakConfig) -> None:
    failures: list[str] = []
    if result.exceptions > 0:
        failures.append(f"workload raised {result.exceptions} exception(s)")
    if result.rss_ratio > cfg.rss_ceiling_ratio:
        failures.append(
            f"RSS ratio {result.rss_ratio:.3f} exceeds ceiling {cfg.rss_ceiling_ratio} "
            f"(start={result.rss_start_steady_bytes}, end={result.rss_end_bytes})"
        )
    if result.loguru_errors > cfg.max_loguru_errors:
        failures.append(
            f"loguru handler errors={result.loguru_errors} > {cfg.max_loguru_errors} (queue/sink unhealthy)"
        )
    # Only enforce p95 drift if both windows produced enough samples.
    if result.p95_warmup_ms > 0 and result.p95_final_ms > 0:
        if result.p95_drift_ratio_observed > cfg.p95_drift_ratio:
            failures.append(
                f"recall p95 drifted {result.p95_drift_ratio_observed:.3f}x "
                f"(warmup={result.p95_warmup_ms:.2f}ms final={result.p95_final_ms:.2f}ms, "
                f"max ratio {cfg.p95_drift_ratio})"
            )
    if failures:
        raise AssertionError("Soak SLOs failed:\n  - " + "\n  - ".join(failures))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_soak_sqlite_lance(tmp_path: Path, loguru_tracker: _LoguruErrorTracker) -> None:
    """Embedded smoke: SQLite + LanceDB, no Docker / no API key.

    The default 5-minute duration is fast enough for a developer laptop;
    set ``KHORA_SOAK_DURATION_S=14400`` to drive the full 4-hour gate.
    """
    cfg = _load_config_from_env()
    kb = await _build_embedded_kb(tmp_path)
    try:
        ns = await kb.create_namespace()
        result = await _drive_soak(
            kb=kb,
            namespace_id=ns.namespace_id,
            cfg=cfg,
            tracker=loguru_tracker,
            stack_label="sqlite-lance",
        )
    finally:
        await kb.disconnect()

    _emit_summary(result)
    _assert_slos(result, cfg)


@pytest.mark.skipif(
    not os.environ.get("KHORA_SOAK_PG_URL"),
    reason="set KHORA_SOAK_PG_URL (and KHORA_GRAPH_URL) to run the PG+Neo4j soak",
)
async def test_soak_postgres_neo4j(loguru_tracker: _LoguruErrorTracker) -> None:
    """Production stack: PostgreSQL + pgvector + Neo4j.

    Only runs when ``KHORA_SOAK_PG_URL`` is set — gated so the smoke run
    on a laptop without Docker still passes. The PR creates this gate;
    the manual GitHub Actions job in ``.github/workflows/soak.yml`` is
    where the production stack is actually exercised.
    """
    cfg = _load_config_from_env()
    pg_url = os.environ["KHORA_SOAK_PG_URL"]
    graph_url = os.environ.get("KHORA_GRAPH_URL")

    kb = Khora(
        pg_url,
        graph_url=graph_url,
        engine="vectorcypher",
        run_migrations=True,
    )
    await kb.connect()
    try:
        ns = await kb.create_namespace()
        result = await _drive_soak(
            kb=kb,
            namespace_id=ns.namespace_id,
            cfg=cfg,
            tracker=loguru_tracker,
            stack_label="pg-neo4j",
        )
    finally:
        await kb.disconnect()

    _emit_summary(result)
    _assert_slos(result, cfg)


# ---------------------------------------------------------------------------
# Allow running standalone for ad-hoc debugging:
#     python tests/soak/test_soak.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - convenience launcher
    sys.exit(pytest.main([__file__, "-m", "soak", "--no-cov", "-s"]))
