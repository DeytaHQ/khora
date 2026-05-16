"""Tests for the chronicle abstention-drift dream op (#652)."""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from khora.dream.config import DreamConfig
from khora.dream.engines.chronicle import (
    plan_chronicle_abstention_drift,
    record_abstention_sample,
    reset_abstention_samples,
)
from khora.dream.plan import DreamOp, OpKind


def _fake_engine(
    *,
    min_top_score: float = 0.3,
    combined_threshold: float = 0.5,
    min_chunks: int = 1,
) -> SimpleNamespace:
    """Minimal stand-in for ChronicleEngine — only the threshold attrs."""
    return SimpleNamespace(
        _abstention_min_top_score=min_top_score,
        _abstention_combined_threshold=combined_threshold,
        _abstention_min_chunks=min_chunks,
    )


def _populate(
    namespace_id: UUID,
    *,
    n: int,
    top_score: float,
    combined_score: float,
    chunk_count: int = 3,
    cap: int = 1024,
) -> None:
    rng = random.Random(42)
    for _ in range(n):
        # Tiny jitter so the distribution has interior values for the
        # percentile calculation rather than a degenerate spike.
        jitter = rng.uniform(-0.01, 0.01)
        record_abstention_sample(
            namespace_id,
            top_score=top_score + jitter,
            combined_score=combined_score + jitter,
            chunk_count=chunk_count,
            cap=cap,
        )


@pytest.fixture(autouse=True)
def _reset_samples() -> Iterator[None]:
    reset_abstention_samples()
    yield
    reset_abstention_samples()


@pytest.mark.asyncio
async def test_recommendation_when_threshold_too_strict() -> None:
    """p90 top_score 0.18 vs configured 0.3 — recommend lower."""
    ns = uuid4()
    _populate(ns, n=1500, top_score=0.18, combined_score=0.20)
    engine = _fake_engine(min_top_score=0.3, combined_threshold=0.5)

    op = await plan_chronicle_abstention_drift(ns, engine=engine, config=DreamConfig())

    assert op.decision == "recommend"
    assert op.op_type == OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT
    report = op.outputs[0]
    assert report["recommendation"]["direction"] == "lower"
    assert "top_score" in report["recommendation"]["rationale"]
    assert "0.3" in report["recommendation"]["rationale"]
    # Configured thresholds round-trip into the report verbatim.
    assert report["configured_thresholds"]["abstention_min_top_score"] == 0.3


@pytest.mark.asyncio
async def test_recommendation_when_threshold_too_lax() -> None:
    """p90 combined_score 0.85 vs configured 0.5 — recommend raise."""
    ns = uuid4()
    # top_score above the floor so we don't trigger the lower-branch.
    _populate(ns, n=1500, top_score=0.7, combined_score=0.85)
    engine = _fake_engine(min_top_score=0.3, combined_threshold=0.5)

    op = await plan_chronicle_abstention_drift(ns, engine=engine, config=DreamConfig())

    assert op.decision == "recommend"
    report = op.outputs[0]
    assert report["recommendation"]["direction"] == "raise"
    assert "combined_score" in report["recommendation"]["rationale"]


@pytest.mark.asyncio
async def test_no_recommendation_when_calibrated() -> None:
    """Distribution sits at the configured thresholds — calibrated."""
    ns = uuid4()
    # p90 top_score ~0.32 — slightly above configured 0.3 floor (not 1.5x below).
    # p90 combined_score ~0.45 — below configured 0.5 threshold (not 1.5x above).
    _populate(ns, n=1500, top_score=0.32, combined_score=0.45)
    engine = _fake_engine(min_top_score=0.3, combined_threshold=0.5)

    op = await plan_chronicle_abstention_drift(ns, engine=engine, config=DreamConfig())

    assert op.decision == "recommend"
    report = op.outputs[0]
    assert report["recommendation"]["direction"] == "calibrated"
    assert "calibrated" in report["recommendation"]["rationale"]


@pytest.mark.asyncio
async def test_insufficient_data() -> None:
    """Below the min-samples floor → decision='insufficient_data'."""
    ns = uuid4()
    _populate(ns, n=50, top_score=0.18, combined_score=0.20)
    engine = _fake_engine()

    op = await plan_chronicle_abstention_drift(ns, engine=engine, config=DreamConfig())

    assert op.decision == "insufficient_data"
    report = op.outputs[0]
    assert report["sample_count"] == 50
    assert report["min_samples"] == 1000
    # Recommendation block omitted on this path — only configured + sample counts.
    assert "recommendation" not in report


@pytest.mark.asyncio
async def test_emits_span_attributes_via_bounded_hash() -> None:
    """Rationale text never appears verbatim as a span attribute."""
    ns = uuid4()
    _populate(ns, n=1500, top_score=0.18, combined_score=0.20)
    engine = _fake_engine(min_top_score=0.3, combined_threshold=0.5)

    captured: list[tuple[str, dict[str, object]]] = []

    class _Span:
        def __enter__(self) -> _Span:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _capture(name: str, **kwargs: object) -> _Span:
        captured.append((name, dict(kwargs)))
        return _Span()

    target = "khora.dream.engines.chronicle.abstention_drift.trace_span"
    with patch(target, side_effect=_capture):
        op = await plan_chronicle_abstention_drift(ns, engine=engine, config=DreamConfig())

    assert captured, "expected a trace_span call"
    span_name, attrs = captured[0]
    assert span_name == "khora.dream.chronicle.abstention_drift"
    assert "rationale_hash" in attrs
    # The raw rationale must never appear as a span attribute value.
    rationale = op.rationale
    assert rationale
    for value in attrs.values():
        assert value != rationale, "raw rationale leaked into span attributes"
    # And no attribute should *contain* a chunk of the raw rationale.
    needle = rationale[:20]
    for value in attrs.values():
        if isinstance(value, str):
            assert needle not in value


@pytest.mark.asyncio
async def test_dream_op_shape_round_trips_json() -> None:
    """A returned DreamOp serializes cleanly via dataclasses.asdict + json."""
    ns = uuid4()
    _populate(ns, n=1500, top_score=0.18, combined_score=0.20)
    engine = _fake_engine()

    op = await plan_chronicle_abstention_drift(ns, engine=engine, config=DreamConfig())

    payload = dataclasses.asdict(op)

    def _default(obj: object) -> object:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, OpKind):
            return obj.value
        raise TypeError(f"unsupported type: {type(obj).__name__}")

    encoded = json.dumps(payload, default=_default)
    decoded = json.loads(encoded)

    assert decoded["op_type"] == OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT.value
    assert decoded["decision"] == "recommend"
    assert decoded["namespace_id"] == str(ns)
    assert decoded["outputs"][0]["recommendation"]["direction"] == "lower"
    # Round-trip remains a dataclass-compatible dict (no exotic types).
    assert isinstance(decoded["outputs"], list)
    assert isinstance(decoded["outputs"][0]["observed"]["top_score"]["p90"], float)


def test_ring_buffer_cap_enforced() -> None:
    """The bounded ring buffer never exceeds the configured cap."""
    ns = uuid4()
    for i in range(2000):
        record_abstention_sample(
            ns,
            top_score=0.1 + (i % 10) * 0.05,
            combined_score=0.2,
            chunk_count=1,
            cap=512,
        )
    # The internal snapshot helper reflects the cap.
    from khora.dream.engines.chronicle.abstention_drift import _snapshot_samples

    snap = _snapshot_samples(ns)
    assert len(snap) == 512


def test_dream_op_kind_enum_member_exposed() -> None:
    """The new enum value is wired into the public OpKind surface."""
    assert OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT.value == "chronicle_abstention_drift_report"


def test_dream_op_default_dataclass_returns_op() -> None:
    """Sanity: DreamOp is a frozen dataclass and accepts the field set we use."""
    op = DreamOp(
        op_id=uuid4(),
        phase="audit",
        op_type=OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT,
        decision="recommend",
        outputs=({"foo": 1},),
    )
    assert op.decision == "recommend"
