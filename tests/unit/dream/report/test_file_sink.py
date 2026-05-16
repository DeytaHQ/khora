"""Tests for ``DreamFileSink`` (#666)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
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
from khora.dream.report.base import DreamReportSchemaMismatchError
from khora.dream.report.file_sink import (
    SCHEMA_VERSION,
    DreamFileSink,
    expire_dream_reports,
    load_manifest,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _started(run_id, namespace_id):
    return DreamRunStarted(
        run_id=run_id,
        namespace_id=namespace_id,
        mode="dry-run",
        trigger="manual",
        started_at=_now(),
    )


def _op(run_id, namespace_id, *, with_undo: bool = False, raw_input: str | None = None):
    return DreamOperationEvent(
        op_id=uuid4(),
        run_id=run_id,
        phase="audit",
        op_type="dedupe_entities",
        inputs={"sample": raw_input or "alpha"},
        outputs={},
        decision="merge",
        rationale=DreamRationale(strategy="cosine_above_threshold", score=0.9, rationale_hash="abcd1234"),
        source_llm_call_ids=[],
        undo=UndoHandle(kind="split", payload_ref="undo/x.json") if with_undo else None,
        started_at=_now(),
        duration_ms=5.0,
        namespace_id=namespace_id,
        text_refs={"sample": "12345678"},
    )


@pytest.mark.asyncio
async def test_clean_run_finalizes_jsonl_and_writes_artifacts(tmp_path: Path) -> None:
    sink = DreamFileSink(tmp_path)
    rid, ns = uuid4(), uuid4()
    await sink.emit(_started(rid, ns))
    await sink.emit(DreamPhaseStarted(run_id=rid, namespace_id=ns, phase="audit", started_at=_now()))
    await sink.emit(_op(rid, ns, with_undo=True))
    await sink.emit(
        DreamPhaseCompleted(
            run_id=rid, namespace_id=ns, phase="audit", outcome="success", ops_total=1, duration_ms=10.0
        )
    )
    await sink.emit(DreamRunCompleted(run_id=rid, namespace_id=ns, mode="dry-run", duration_ms=20.0, ops_total=1))
    await sink.close()

    run_dirs = list(tmp_path.glob(f"{ns}/*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / f"{rid}.events.jsonl").exists()
    assert not (run_dir / f"{rid}.events.partial").exists()
    assert (run_dir / f"{rid}.manifest.json").exists()
    assert (run_dir / f"{rid}.summary.md").exists()
    assert (run_dir / f"{rid}.undo.json").exists()


@pytest.mark.asyncio
async def test_crash_leaves_events_crashed_jsonl(tmp_path: Path) -> None:
    """No DreamRunCompleted before close() → ``.events.crashed.jsonl``."""
    sink = DreamFileSink(tmp_path)
    rid, ns = uuid4(), uuid4()
    await sink.emit(_started(rid, ns))
    await sink.emit(_op(rid, ns))
    # Abrupt: no completed, no failed event. Operator's process died.
    await sink.close()

    run_dir = next((tmp_path / str(ns)).iterdir())
    assert (run_dir / f"{rid}.events.crashed.jsonl").exists()
    assert not (run_dir / f"{rid}.events.jsonl").exists()


@pytest.mark.asyncio
async def test_failed_run_writes_summary_with_error_hash(tmp_path: Path) -> None:
    sink = DreamFileSink(tmp_path)
    rid, ns = uuid4(), uuid4()
    await sink.emit(_started(rid, ns))
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
    await sink.close()

    run_dir = next((tmp_path / str(ns)).iterdir())
    summary_text = (run_dir / f"{rid}.summary.md").read_text()
    assert "cafebabe" in summary_text
    assert "TimeoutError" in summary_text
    # Crashed marker: failed-without-Completed → crashed jsonl.
    assert (run_dir / f"{rid}.events.crashed.jsonl").exists()


@pytest.mark.asyncio
async def test_manifest_schema_version_stamped(tmp_path: Path) -> None:
    sink = DreamFileSink(tmp_path)
    rid, ns = uuid4(), uuid4()
    await sink.emit(_started(rid, ns))
    await sink.emit(DreamRunCompleted(run_id=rid, namespace_id=ns, mode="dry-run", duration_ms=1.0, ops_total=0))
    await sink.close()

    run_dir = next((tmp_path / str(ns)).iterdir())
    manifest_path = run_dir / f"{rid}.manifest.json"
    loaded = load_manifest(manifest_path)
    assert loaded["schema"] == SCHEMA_VERSION
    assert loaded["completed"] is True


def test_load_manifest_rejects_unknown_schema(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"schema": "dream-report/999"}))
    with pytest.raises(DreamReportSchemaMismatchError):
        load_manifest(bad)


@pytest.mark.asyncio
async def test_redact_text_all_strips_raw_strings(tmp_path: Path) -> None:
    sink = DreamFileSink(tmp_path, redact_text="all")
    rid, ns = uuid4(), uuid4()
    await sink.emit(_started(rid, ns))
    await sink.emit(_op(rid, ns, raw_input="SECRET_VALUE"))
    await sink.emit(DreamRunCompleted(run_id=rid, namespace_id=ns, mode="dry-run", duration_ms=1.0, ops_total=1))
    await sink.close()

    run_dir = next((tmp_path / str(ns)).iterdir())
    jsonl = (run_dir / f"{rid}.events.jsonl").read_text()
    assert "SECRET_VALUE" not in jsonl


@pytest.mark.asyncio
async def test_redact_text_none_keeps_raw_strings(tmp_path: Path) -> None:
    sink = DreamFileSink(tmp_path, redact_text="none")
    rid, ns = uuid4(), uuid4()
    await sink.emit(_started(rid, ns))
    await sink.emit(_op(rid, ns, raw_input="VERBATIM_TEXT"))
    await sink.emit(DreamRunCompleted(run_id=rid, namespace_id=ns, mode="dry-run", duration_ms=1.0, ops_total=1))
    await sink.close()

    run_dir = next((tmp_path / str(ns)).iterdir())
    jsonl = (run_dir / f"{rid}.events.jsonl").read_text()
    assert "VERBATIM_TEXT" in jsonl


@pytest.mark.asyncio
async def test_expire_dream_reports_removes_old_dirs(tmp_path: Path) -> None:
    ns = uuid4()
    old = tmp_path / str(ns) / "2024-01-01"
    keep = tmp_path / str(ns) / "2099-12-31"
    old.mkdir(parents=True)
    keep.mkdir(parents=True)
    (old / "x.txt").write_text("x")
    (keep / "x.txt").write_text("x")

    cutoff = datetime(2025, 1, 1, tzinfo=UTC)
    deleted = await expire_dream_reports(base_dir=tmp_path, before=cutoff)
    assert deleted == 1
    assert not old.exists()
    assert keep.exists()
