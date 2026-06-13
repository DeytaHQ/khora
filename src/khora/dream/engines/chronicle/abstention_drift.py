"""Chronicle abstention-threshold drift report (#652, part of #649).

A read-only dream-phase audit op that compares the configured chronicle
abstention thresholds against the observed distribution of recall
results and emits a textual recommendation. The op:

- never mutates state, never calls an LLM
- emits a single :class:`DreamOp` whose ``outputs`` carry the report
- refuses silently when the observed sample count is below the
  configured floor (``DreamConfig.abstention_drift_min_samples``)
- records the rationale as a SHA1[:8] hash on the span attribute — raw
  text never reaches the collector (CLAUDE.md cardinality + redaction
  rule)

Sample source: a bounded per-namespace ring buffer populated through
:func:`record_abstention_sample`. The chronicle engine is not yet wired
to call it (zero-cost when unused); operators that want offline drift
analytics can wire it in their own subscription or test harness today.
The op also degrades to "insufficient_data" when no samples were
recorded, so the read-side never crashes on a cold namespace.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from khora.dream.plan import DreamOp, OpKind
from khora.telemetry import bounded_text_hash, trace_span

if TYPE_CHECKING:
    from khora.dream.config import DreamConfig
    from khora.engines.chronicle.engine import ChronicleEngine


# ---------------------------------------------------------------------------
# In-process ring buffer
# ---------------------------------------------------------------------------

# Module-level: one ring buffer per namespace. Each entry is
# ``(top_score, combined_score, chunk_count)``. The buffer is bounded
# at ``DreamConfig.abstention_drift_sample_cap`` (default 1024) so
# memory stays predictable regardless of recall volume.
_SAMPLES: dict[UUID, deque[tuple[float, float, int]]] = {}
_SAMPLES_LOCK = Lock()


def record_abstention_sample(
    namespace_id: UUID,
    *,
    top_score: float,
    combined_score: float,
    chunk_count: int,
    cap: int = 1024,
) -> None:
    """Record one recall observation for later drift analysis.

    Cheap (a single deque append under a short lock). Callers that want
    drift telemetry wire this into their recall path; the chronicle
    engine itself does not call it by default — Phase 1.1 ships the
    read-only reporter and leaves wiring as an operator decision.
    """
    with _SAMPLES_LOCK:
        buf = _SAMPLES.get(namespace_id)
        if buf is None or buf.maxlen != cap:
            buf = deque(buf or (), maxlen=cap)
            _SAMPLES[namespace_id] = buf
        buf.append((float(top_score), float(combined_score), int(chunk_count)))


def reset_abstention_samples(namespace_id: UUID | None = None) -> None:
    """Drop recorded samples for a namespace (or all) — test helper."""
    with _SAMPLES_LOCK:
        if namespace_id is None:
            _SAMPLES.clear()
        else:
            _SAMPLES.pop(namespace_id, None)


def _snapshot_samples(namespace_id: UUID) -> list[tuple[float, float, int]]:
    with _SAMPLES_LOCK:
        buf = _SAMPLES.get(namespace_id)
        return list(buf) if buf else []


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile over a sorted copy of ``values``.

    Matches the convention used by ``statistics.quantiles`` but keeps the
    dependency surface trivially small. ``p`` ∈ [0.0, 1.0].
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    rank = p * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def _build_recommendation(
    *,
    top_score_p90: float,
    combined_score_p90: float,
    configured_min_top_score: float,
    configured_combined_threshold: float,
) -> tuple[str, str]:
    """Return ``(direction, rationale)`` for the operator.

    ``direction`` is one of ``"lower"``, ``"raise"``, or ``"calibrated"``.
    The rationale is plain English so it renders well in dashboards;
    the collector sink only ever sees the SHA1[:8] hash.
    """
    # Strict threshold: observed p90 of top_score is much lower than the
    # configured floor — most recalls trip ``top_score_low`` even on
    # good answers. Recommend lowering the floor.
    if configured_min_top_score > 0 and top_score_p90 * 1.5 < configured_min_top_score:
        suggested = round(top_score_p90 * 0.9, 3)
        rationale = (
            f"observed p90 top_score is {top_score_p90:.3f} but "
            f"abstention_min_top_score is {configured_min_top_score:.3f} — "
            f"most recalls fire top_score_low even on good answers; "
            f"consider lowering to ~{suggested:.3f}"
        )
        return "lower", rationale

    # Over-abstaining: observed p90 of combined_score is much higher than
    # the configured ``should_abstain`` threshold. ``should_abstain``
    # fires when ``combined_score >= threshold`` (recall_abstention.py),
    # so a p90 far above the threshold means a large share of recalls are
    # at or above it — the gate trips frequently. If that over-abstention
    # is unintended, raise the threshold above the bulk of the
    # distribution.
    if configured_combined_threshold > 0 and combined_score_p90 > configured_combined_threshold * 1.5:
        suggested = round(combined_score_p90 * 1.05, 3)
        rationale = (
            f"observed p90 combined_score is {combined_score_p90:.3f} but "
            f"abstention_combined_threshold is {configured_combined_threshold:.3f} — "
            f"a large share of recalls trip should_abstain (the gate fires "
            f"frequently); raise the threshold to ~{suggested:.3f} if this "
            f"over-abstention is unintended"
        )
        return "raise", rationale

    # Under-abstaining: the configured threshold sits well above the
    # observed combined_score distribution, so ``combined_score >=
    # threshold`` almost never holds — the gate never trips. Lower the
    # threshold toward the distribution if abstention is meant to fire.
    if configured_combined_threshold > 0 and combined_score_p90 * 1.5 < configured_combined_threshold:
        suggested = round(combined_score_p90 * 1.05, 3)
        rationale = (
            f"observed p90 combined_score is {combined_score_p90:.3f} but "
            f"abstention_combined_threshold is {configured_combined_threshold:.3f} — "
            f"should_abstain never trips (the threshold is above the observed "
            f"distribution); consider lowering to ~{suggested:.3f}"
        )
        return "lower", rationale

    return "calibrated", "thresholds look calibrated against observed distribution"


# ---------------------------------------------------------------------------
# Plan helper
# ---------------------------------------------------------------------------


async def plan_chronicle_abstention_drift(
    namespace_id: UUID,
    *,
    engine: ChronicleEngine,
    config: DreamConfig,
    sample_rate: float | None = None,
) -> DreamOp:
    """Build a single read-only drift-report :class:`DreamOp`.

    Args:
        namespace_id: which namespace to read samples for.
        engine: the live chronicle engine — only its configured
            thresholds are read (``_abstention_min_top_score``,
            ``_abstention_combined_threshold``, ``_abstention_min_chunks``).
        config: dream config — provides the sample-count floor and the
            ring-buffer cap.
        sample_rate: reserved for future use; the op currently consumes
            every recorded sample (no down-sampling).

    Returns:
        A :class:`DreamOp` with ``op_type=CHRONICLE_ABSTENTION_DRIFT_REPORT``
        and ``decision`` in ``{"recommend", "insufficient_data"}``. The
        ``outputs`` tuple carries a single report dict; nothing else is
        mutated.
    """
    _ = sample_rate  # reserved; ring buffer already bounds memory
    samples = _snapshot_samples(namespace_id)
    sample_count = len(samples)
    min_samples = config.abstention_drift_min_samples

    configured = {
        "abstention_min_top_score": engine._abstention_min_top_score,
        "abstention_combined_threshold": engine._abstention_combined_threshold,
        "abstention_min_chunks": engine._abstention_min_chunks,
    }

    if sample_count < min_samples:
        report: dict[str, Any] = {
            "sample_count": sample_count,
            "min_samples": min_samples,
            "configured_thresholds": configured,
        }
        rationale = (
            f"observed {sample_count} samples, below the configured floor "
            f"of {min_samples} — refusing to emit a recommendation"
        )
        with trace_span(
            "khora.dream.chronicle.abstention_drift",
            namespace_id=str(namespace_id),
            decision="insufficient_data",
            sample_count=sample_count,
            min_samples=min_samples,
            rationale_hash=bounded_text_hash(rationale),
        ):
            pass
        return DreamOp(
            op_id=uuid4(),
            phase="audit",
            op_type=OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT,
            decision="insufficient_data",
            rationale=rationale,
            outputs=(report,),
            namespace_id=namespace_id,
        )

    top_scores = [t for (t, _c, _n) in samples]
    combined_scores = [c for (_t, c, _n) in samples]
    chunk_counts = [float(n) for (_t, _c, n) in samples]

    top_p50 = _percentile(top_scores, 0.50)
    top_p90 = _percentile(top_scores, 0.90)
    top_p99 = _percentile(top_scores, 0.99)
    comb_p50 = _percentile(combined_scores, 0.50)
    comb_p90 = _percentile(combined_scores, 0.90)
    comb_p99 = _percentile(combined_scores, 0.99)
    chunks_p50 = _percentile(chunk_counts, 0.50)
    chunks_p90 = _percentile(chunk_counts, 0.90)
    chunks_p99 = _percentile(chunk_counts, 0.99)

    direction, rationale = _build_recommendation(
        top_score_p90=top_p90,
        combined_score_p90=comb_p90,
        configured_min_top_score=engine._abstention_min_top_score,
        configured_combined_threshold=engine._abstention_combined_threshold,
    )

    report = {
        "sample_count": sample_count,
        "configured_thresholds": configured,
        "observed": {
            "top_score": {"p50": top_p50, "p90": top_p90, "p99": top_p99},
            "combined_score": {"p50": comb_p50, "p90": comb_p90, "p99": comb_p99},
            "chunk_count": {"p50": chunks_p50, "p90": chunks_p90, "p99": chunks_p99},
        },
        "recommendation": {
            "direction": direction,
            "rationale": rationale,
        },
    }

    # Hash the free-text rationale before it ever reaches a span
    # attribute. Numeric percentiles are bounded-cardinality and safe to
    # record verbatim.
    with trace_span(
        "khora.dream.chronicle.abstention_drift",
        namespace_id=str(namespace_id),
        decision="recommend",
        direction=direction,
        sample_count=sample_count,
        top_score_p90=top_p90,
        combined_score_p90=comb_p90,
        rationale_hash=bounded_text_hash(rationale),
    ):
        pass

    return DreamOp(
        op_id=uuid4(),
        phase="audit",
        op_type=OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT,
        decision="recommend",
        rationale=rationale,
        outputs=(report,),
        namespace_id=namespace_id,
    )


__all__ = [
    "plan_chronicle_abstention_drift",
    "record_abstention_sample",
    "reset_abstention_samples",
]
