"""Round-trip + validation tests for ``khora.dream.events`` (#666)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.dream.events import (
    DreamOperationEvent,
    DreamPhaseCompleted,
    DreamPhaseStarted,
    DreamRationale,
    DreamRunCompleted,
    DreamRunFailed,
    DreamRunStarted,
    UndoHandle,
)


def _now() -> datetime:
    return datetime.now(UTC)


def test_dream_rationale_roundtrips_through_json() -> None:
    r = DreamRationale(
        strategy="cosine_above_threshold",
        score=0.91,
        threshold=0.85,
        rationale_hash="abcd1234",
    )
    blob = r.model_dump(mode="json")
    restored = DreamRationale.model_validate(blob)
    assert restored == r


def test_undo_handle_kind_is_bounded() -> None:
    u = UndoHandle(kind="split", payload_ref="undo/op.json")
    assert u.kind == "split"

    with pytest.raises(ValueError):
        UndoHandle(kind="unsafe_kind", payload_ref="undo/op.json")  # type: ignore[arg-type]


def test_dream_run_started_roundtrip() -> None:
    rid, ns = uuid4(), uuid4()
    ev = DreamRunStarted(
        run_id=rid,
        namespace_id=ns,
        mode="dry-run",
        trigger="manual",
        started_at=_now(),
    )
    blob = json.dumps(ev.model_dump(mode="json"))
    restored = DreamRunStarted.model_validate(json.loads(blob))
    assert restored.run_id == rid
    assert restored.namespace_id == ns
    assert restored.mode == "dry-run"


def test_dream_operation_event_uses_hashes_not_raw_text() -> None:
    """Op payload exposes a hash field for rationale — never raw text on the model."""
    ev = DreamOperationEvent(
        op_id=uuid4(),
        run_id=uuid4(),
        phase="audit",
        op_type="dedupe_entities",
        inputs={"candidate_a": uuid4()},
        outputs={},
        decision="merge",
        rationale=DreamRationale(strategy="llm_verifier", rationale_hash="deadbeef"),
        started_at=_now(),
        duration_ms=12.5,
        namespace_id=uuid4(),
        text_refs={"entity_a_name": "abcd1234"},
    )
    blob = ev.model_dump(mode="json")
    assert blob["rationale"]["rationale_hash"] == "deadbeef"
    assert blob["text_refs"]["entity_a_name"] == "abcd1234"


def test_phase_and_run_completion_models() -> None:
    rid, ns = uuid4(), uuid4()
    pc = DreamPhaseCompleted(
        run_id=rid,
        namespace_id=ns,
        phase="audit",
        outcome="success",
        ops_total=3,
        duration_ms=100.0,
    )
    rc = DreamRunCompleted(
        run_id=rid,
        namespace_id=ns,
        mode="dry-run",
        duration_ms=200.0,
        ops_total=3,
    )
    rf = DreamRunFailed(
        run_id=rid,
        namespace_id=ns,
        mode="dry-run",
        duration_ms=50.0,
        error_hash="abcd1234",
        error_type="TimeoutError",
    )
    assert pc.outcome == "success"
    assert rc.ops_total == 3
    assert rf.error_type == "TimeoutError"


def test_phase_started_required_fields() -> None:
    with pytest.raises(ValueError):
        DreamPhaseStarted()  # type: ignore[call-arg]


def test_negative_duration_rejected() -> None:
    """``duration_ms`` is constrained to non-negative — clock-skew bugs are loud."""
    with pytest.raises(ValueError):
        DreamRunCompleted(
            run_id=uuid4(),
            namespace_id=uuid4(),
            mode="dry-run",
            duration_ms=-1.0,
            ops_total=0,
        )
