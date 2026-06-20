"""Orchestrator unit tests (#661).

The orchestrator is exercised against an in-process stub engine plugin
that owns a small, deterministic plan. The Postgres-backed
``khora_dream_runs`` persistence path is covered by the integration
tests (``test_orchestrator_e2e.py``); these unit tests focus on the
state-machine, safety floor, and cancellation semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.config import DreamConfig
from khora.dream.engines import registry as registry_mod
from khora.dream.engines.registry import _validate_no_forbidden_ops
from khora.dream.exceptions import DreamDisabledError, DreamForbiddenOpError
from khora.dream.orchestrator import DreamOrchestrator, request_cancel
from khora.dream.plan import DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.report.base import ReportSink


class _CapturingSink(ReportSink):
    """Sink that records every event it receives — no I/O."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)


class _StubPlugin:
    """``DreamCapable``-shaped plugin returning a hand-rolled plan."""

    def __init__(self, ops: list[DreamOp]) -> None:
        self._ops = ops

    @property
    def dream_capabilities(self) -> frozenset[OpKind]:
        return frozenset({op.op_type for op in self._ops})

    async def plan_dream(
        self,
        kb: Any,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        config: DreamConfig,
        expertise: Any = None,
    ) -> DreamPlan:
        del kb, scope, config, expertise
        return DreamPlan(
            plan_id=uuid4(),
            namespace_id=namespace_id,
            ops=tuple(self._ops),
        )

    async def apply_dream(
        self,
        plan: DreamPlan,
        *,
        checkpoint: Any = None,
        on_progress: Any = None,
    ) -> Any:
        del plan, checkpoint, on_progress
        raise NotImplementedError  # Orchestrator iterates per-op itself in Phase 1.


class _FakeKB:
    """Minimal Khora-shaped object for the orchestrator's needs."""

    def __init__(self) -> None:
        self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
        self._engine_name = "stub"
        self.storage = _FakeCoordinator()


class _FakeCoordinator:
    """Coordinator stub whose ``transaction()`` raises so the orchestrator
    falls through every persistence path as a no-op."""

    def transaction(self) -> Any:
        raise RuntimeError("no SQL backend connected (test stub)")


def _audit_op(op_type: OpKind = OpKind.CHRONICLE_TOMBSTONE_AUDIT) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="audit",
        op_type=op_type,
        decision="audit_complete",
        rationale="unit-test op",
        outputs=({"total": 0},),
        started_at=datetime.now(UTC),
        duration_ms=1.0,
        namespace_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_disabled_raises() -> None:
    """kb.dream() with enabled=False raises DreamDisabledError."""
    from khora.dream.api import dream

    kb = _FakeKB()
    kb._config = SimpleNamespace(dream=DreamConfig(enabled=False))

    with pytest.raises(DreamDisabledError):
        await dream(kb, uuid4())


@pytest.mark.asyncio
async def test_dream_rejects_invalid_mode() -> None:
    """Unknown modes raise ValueError before any plan runs."""
    from khora.dream.api import dream

    kb = _FakeKB()
    with pytest.raises(ValueError, match="mode must be"):
        await dream(kb, uuid4(), mode="commit")


# ---------------------------------------------------------------------------
# Orchestrator: dry-run path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_returns_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run wires through every planned op and produces a DreamResult."""
    plugin = _StubPlugin([_audit_op(), _audit_op(OpKind.VECTORCYPHER_ORPHAN_REPORT)])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    kb = _FakeKB()
    sink = _CapturingSink()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[sink])

    namespace_id = uuid4()
    result = await orch.run(namespace_id, mode="dry-run")

    assert result.run.mode == "dry-run"
    assert result.run.namespace_id == namespace_id
    # Two op kinds → two OpSummary entries, each with planned=1.
    assert {s.op_type for s in result.ops} == {
        str(OpKind.CHRONICLE_TOMBSTONE_AUDIT),
        str(OpKind.VECTORCYPHER_ORPHAN_REPORT),
    }
    assert all(s.planned == 1 for s in result.ops)
    # Sink saw RunStarted + PhaseStarted("plan") + PhaseCompleted("plan")
    # + PhaseStarted("report") + 2x OpEvent + PhaseCompleted("report")
    # + RunCompleted.
    event_types = [type(e).__name__ for e in sink.events]
    assert event_types.count("DreamRunStarted") == 1
    assert event_types.count("DreamRunCompleted") == 1
    assert event_types.count("DreamOperationEvent") == 2


@pytest.mark.asyncio
async def test_apply_mode_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply mode on Phase-1 read-only ops increments ``applied`` counts."""
    plugin = _StubPlugin([_audit_op()])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    result = await orch.run(uuid4(), mode="apply")
    assert result.run.mode == "apply"
    assert result.ops[0].applied == 1


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_between_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flipped cancel flag halts the run after the next op boundary."""
    ops = [_audit_op() for _ in range(5)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    kb = _FakeKB()
    sink = _CapturingSink()

    # Mid-run cancel: a sink that flips the cancel flag after seeing
    # the first op event drives the orchestrator out of its loop.
    class _CancelOnFirstOp(ReportSink):
        def __init__(self) -> None:
            self.seen = 0

        async def emit(self, event: Any) -> None:
            if type(event).__name__ == "DreamOperationEvent":
                self.seen += 1
                if self.seen == 1:
                    await request_cancel(event.run_id)

    cancel_sink = _CancelOnFirstOp()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[sink, cancel_sink])

    result = await orch.run(uuid4(), mode="apply")
    assert result is not None
    # Cancellation happens after the first apply op fires.
    op_events = [e for e in sink.events if type(e).__name__ == "DreamOperationEvent"]
    assert 1 <= len(op_events) < 5


# ---------------------------------------------------------------------------
# Safety floor
# ---------------------------------------------------------------------------


def test_safety_floor_rejects_forbidden_op() -> None:
    """A synthetic plan with a forbidden op kind raises before any apply."""
    # The OpKind enum only contains permitted ops, so synthesize the
    # forbidden op via a SimpleNamespace shaped like DreamOp. The
    # validator stringifies ``op.op_type`` and pattern-matches.
    forbidden = SimpleNamespace(op_id=uuid4(), op_type="delete_document")
    plan = SimpleNamespace(ops=(forbidden,))
    with pytest.raises(DreamForbiddenOpError, match="delete_document"):
        _validate_no_forbidden_ops(plan)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_safety_floor_aborts_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin that returns a forbidden op aborts the run with DreamForbiddenOpError."""

    class _BadPlugin:
        @property
        def dream_capabilities(self) -> frozenset[OpKind]:
            return frozenset()

        async def plan_dream(
            self,
            kb: Any,
            namespace_id: UUID,
            *,
            scope: Any,
            config: Any,
            expertise: Any = None,
        ) -> DreamPlan:
            del kb, scope, config, expertise
            bad_op = SimpleNamespace(
                op_id=uuid4(),
                op_type="delete_document",
                phase="apply",
                inputs=(),
                outputs=(),
                decision="",
                rationale="",
                source_llm_call_ids=(),
                undo=None,
                started_at=None,
                duration_ms=None,
                namespace_id=namespace_id,
            )
            return DreamPlan(plan_id=uuid4(), namespace_id=namespace_id, ops=(bad_op,))  # type: ignore[arg-type]

        async def apply_dream(self, plan: DreamPlan, **kwargs: Any) -> Any:
            raise NotImplementedError

    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": _BadPlugin()})

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    with pytest.raises(DreamForbiddenOpError):
        await orch.run(uuid4(), mode="dry-run")


# ---------------------------------------------------------------------------
# Status / history (no Postgres available — must degrade to empty)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_returns_none_without_postgres() -> None:
    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
    assert await orch.status(uuid4()) is None


@pytest.mark.asyncio
async def test_history_limit() -> None:
    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
    # The stub coordinator raises on transaction(); history must catch
    # that and return an empty list rather than crash.
    assert await orch.history(uuid4(), limit=5) == []


# ---------------------------------------------------------------------------
# Resume (no real DB — exercises the orchestrator path, not the SQL update)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_from_run_id_carried_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """``resume_from`` is propagated as the run_id on the RunStarted event."""
    plugin = _StubPlugin([_audit_op()])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    kb = _FakeKB()
    sink = _CapturingSink()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[sink])

    resume_id = uuid4()
    result = await orch.run(uuid4(), mode="apply", resume_from=resume_id)
    assert result.run.run_id == resume_id
    started = next(e for e in sink.events if type(e).__name__ == "DreamRunStarted")
    assert started.trigger == "resume"
    assert started.run_id == resume_id


@pytest.mark.asyncio
async def test_init_run_row_skips_record_run_on_resume() -> None:
    """On resume the run row already exists; _init_run_row must not re-run
    record_run, which would reset state / last_committed_op_seq on a backend
    whose write is not conflict-preserving (the SurrealDB UPSERT), replaying
    already-committed ops."""
    orch = DreamOrchestrator(_FakeKB(), DreamConfig(enabled=True), sinks=[])
    calls: list[str] = []

    class _MockStore:
        async def record_run(self, *args: object, **kwargs: object) -> None:
            calls.append("record_run")

    orch._run_store_cache = _MockStore()  # type: ignore[assignment]
    orch._run_store_resolved = True

    await orch._init_run_row(uuid4(), uuid4(), "apply", is_resume=True)
    assert calls == [], "record_run must be skipped on resume"

    await orch._init_run_row(uuid4(), uuid4(), "apply", is_resume=False)
    assert calls == ["record_run"], "record_run must run on a fresh start"


# ---------------------------------------------------------------------------
# Cancel registry hygiene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_cancel_returns_false_for_unknown_run() -> None:
    assert await request_cancel(uuid4()) is False
