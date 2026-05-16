"""Tests for ``DreamCollectorSink`` — OTel spans + metrics (#666)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from khora.dream.events import (
    DreamOperationEvent,
    DreamPhaseCompleted,
    DreamRationale,
    DreamRunCompleted,
    DreamRunFailed,
    DreamRunStarted,
)
from khora.dream.report.collector_sink import DreamCollectorSink

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPO_ROOT / "docs" / "telemetry-contract.json"


def _now() -> datetime:
    return datetime.now(UTC)


def _op_event() -> DreamOperationEvent:
    return DreamOperationEvent(
        op_id=uuid4(),
        run_id=uuid4(),
        phase="audit",
        op_type="dedupe_entities",
        inputs={},
        outputs={},
        decision="merge",
        rationale=DreamRationale(strategy="cosine_above_threshold", rationale_hash="abcd1234"),
        started_at=_now(),
        duration_ms=5.0,
        namespace_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_collector_emit_run_started_does_not_raise() -> None:
    sink = DreamCollectorSink()
    rid, ns = uuid4(), uuid4()
    await sink.emit(DreamRunStarted(run_id=rid, namespace_id=ns, mode="dry-run", trigger="manual", started_at=_now()))


@pytest.mark.asyncio
async def test_collector_emit_full_lifecycle() -> None:
    """All six event kinds should pass through without raising."""
    sink = DreamCollectorSink()
    rid, ns = uuid4(), uuid4()
    await sink.emit(DreamRunStarted(run_id=rid, namespace_id=ns, mode="dry-run", trigger="manual", started_at=_now()))
    await sink.emit(_op_event())
    await sink.emit(
        DreamPhaseCompleted(
            run_id=rid, namespace_id=ns, phase="audit", outcome="success", ops_total=1, duration_ms=10.0
        )
    )
    await sink.emit(DreamRunCompleted(run_id=rid, namespace_id=ns, mode="dry-run", duration_ms=20.0, ops_total=1))


@pytest.mark.asyncio
async def test_collector_emit_failed_path() -> None:
    sink = DreamCollectorSink()
    rid, ns = uuid4(), uuid4()
    await sink.emit(DreamRunStarted(run_id=rid, namespace_id=ns, mode="dry-run", trigger="manual", started_at=_now()))
    await sink.emit(
        DreamRunFailed(
            run_id=rid,
            namespace_id=ns,
            mode="dry-run",
            duration_ms=15.0,
            error_hash="cafebabe",
            error_type="TimeoutError",
        )
    )


def test_hash_text_uses_bounded_text_hash() -> None:
    """The sink exposes the canonical hash function."""
    from khora.telemetry import bounded_text_hash

    assert DreamCollectorSink.hash_text("some raw text") == bounded_text_hash("some raw text")


def test_no_dream_metric_carries_namespace_id_label() -> None:
    """Cardinality rule: no dream metric may declare ``namespace_id`` as a label."""
    contract = json.loads(CONTRACT_PATH.read_text())
    for metric in contract["metrics"]:
        if not metric["name"].startswith("khora.dream."):
            continue
        labels = metric.get("labels") or []
        label_names = {entry["name"] for entry in labels}
        assert "namespace_id" not in label_names, (
            f"Metric {metric['name']!r} declares namespace_id as a label — violates CLAUDE.md cardinality rule."
        )


def test_dream_metrics_declared_with_expected_label_sets() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    by_name = {m["name"]: m for m in contract["metrics"]}

    expected = {
        "khora.dream.runs_total": {"trigger", "outcome"},
        "khora.dream.run.duration": {"trigger", "outcome"},
        "khora.dream.phase.duration": {"phase", "outcome"},
        "khora.dream.ops_total": {"phase", "op_type", "decision"},
        "khora.dream.op.duration": {"phase", "op_type"},
        "khora.dream.llm.tokens": {"direction", "model"},
        "khora.dream.undo_invocations_total": {"op_type", "outcome"},
        "khora.dream.subscription.overflow_total": {"subscription_class"},
        "khora.dream.report.write_failures_total": {"reason"},
    }
    for name, labels in expected.items():
        m = by_name[name]
        actual = {entry["name"] for entry in m.get("labels") or []}
        assert actual == labels, f"{name}: labels {actual} != expected {labels}"
