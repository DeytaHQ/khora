"""Apply-phase orchestrator unit tests (#667).

Covers the Phase 4 apply path:

- Per-op handler dispatch + sequencing.
- ``last_committed_op_seq`` checkpoint update after each commit.
- Incremental ``undo.json`` writes via :class:`DreamFileSink`.
- The :envvar:`KHORA_DREAM_DISABLE_APPLY` kill-switch.
- ``chunk_id`` runtime safety assertion.
- ``fact_compaction_retention_days`` floor enforced at config load.
- Resume-from-checkpoint semantics.

Postgres-backed integration tests live in
``tests/integration/dream/test_orchestrator_apply_e2e.py``.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.config import DreamConfig
from khora.dream.engines import registry as registry_mod
from khora.dream.exceptions import DreamApplyDisabled, DreamForbiddenOpError
from khora.dream.orchestrator import DreamOrchestrator
from khora.dream.plan import DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.report.base import ReportSink
from khora.dream.report.file_sink import UNDO_SCHEMA_VERSION, DreamFileSink
from khora.dream.result import UndoRecord
from khora.dream.safety import _assert_no_chunk_id_mutation

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CapturingSink(ReportSink):
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)


class _StubPlugin:
    """Plugin that returns a hand-rolled plan from a list of ops."""

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
        return DreamPlan(plan_id=uuid4(), namespace_id=namespace_id, ops=tuple(self._ops))

    async def apply_dream(self, plan: DreamPlan, **kwargs: Any) -> Any:
        del plan, kwargs
        raise NotImplementedError


class _NoopSession:
    """Session stub with a no-op ``execute``.

    ``bind=None`` keeps the orchestrator's ``_is_postgres`` check treating
    this as Postgres-equivalent. ``_init_run_row`` (ungated since #896)
    now issues a run-row INSERT through any session, so the stub must
    accept ``execute`` calls without touching real SQL.
    """

    def __init__(self) -> None:
        self.bind = None

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(first=lambda: None, all=lambda: [])


class _FakeCoordinator:
    """Coordinator stub that surfaces sessions to per-op handlers.

    ``transaction()`` is an async context manager yielding an object
    with a ``session`` attribute. Sessions are SimpleNamespaces so the
    ``_is_postgres`` check in the orchestrator returns False — we exercise
    the apply-handler path without touching real SQL.
    """

    def __init__(self) -> None:
        self.tx_open_count = 0
        self.tx_close_count = 0

    def transaction(self) -> Any:
        return _FakeTxnCtx(self)


class _FakeTxnCtx:
    def __init__(self, coordinator: _FakeCoordinator) -> None:
        self._coordinator = coordinator

    async def __aenter__(self) -> Any:
        self._coordinator.tx_open_count += 1
        return SimpleNamespace(session=_NoopSession())

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._coordinator.tx_close_count += 1
        return None


class _FakeKB:
    def __init__(self) -> None:
        self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
        self._engine_name = "stub"
        self.storage = _FakeCoordinator()


def _op(op_type: OpKind = OpKind.VECTORCYPHER_DEDUPE_ENTITIES) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=op_type,
        decision="merge",
        rationale="test",
        outputs=({"merged": 1},),
        started_at=datetime.now(UTC),
        duration_ms=1.0,
        namespace_id=uuid4(),
    )


def _make_handler(undo_before: dict[str, Any] | None = None) -> Any:
    """Build an apply handler that records its calls and returns an UndoRecord."""
    calls: list[dict[str, Any]] = []

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        calls.append({"op_id": op.op_id, "session": session, "coordinator": coordinator})
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before=dict(undo_before) if undo_before is not None else {"entity_id": str(uuid4())},
            applied_at=datetime.now(UTC),
        )

    _handler.calls = calls  # type: ignore[attr-defined]
    return _handler


def _install_handler(
    monkeypatch: pytest.MonkeyPatch,
    op_kind: OpKind,
    handler: Any,
) -> None:
    """Install ``handler`` as the apply handler for ``op_kind``.

    Bypasses the importlib-based ``get_apply_handler`` lookup by
    monkey-patching the registry's lookup result directly.
    """

    def _stub_get_apply_handler(op_type: OpKind | str) -> Any:
        from khora.dream.engines.registry import _FORBIDDEN_OP_KINDS

        if str(op_type) in _FORBIDDEN_OP_KINDS:
            raise DreamForbiddenOpError(f"forbidden: {op_type}")
        if str(op_type) == str(op_kind):
            return handler
        return None

    monkeypatch.setattr(registry_mod, "get_apply_handler", _stub_get_apply_handler)
    import khora.dream.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "get_apply_handler", _stub_get_apply_handler)


# ---------------------------------------------------------------------------
# 1. Per-op dispatch + sequencing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_phase_calls_per_op_handler_in_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    ops = [_op() for _ in range(3)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler = _make_handler()
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_DEDUPE_ENTITIES, handler)

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
    await orch.run(uuid4(), mode="apply")

    calls = handler.calls  # type: ignore[attr-defined]
    assert len(calls) == 3
    assert [c["op_id"] for c in calls] == [op.op_id for op in ops]


# ---------------------------------------------------------------------------
# 2. Checkpoint update after each commit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_phase_writes_checkpoint_after_each_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    ops = [_op() for _ in range(3)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler = _make_handler()
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_DEDUPE_ENTITIES, handler)

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    recorded: list[tuple[UUID, int]] = []

    async def _record(session: Any, run_id: UUID, seq: int) -> None:
        del session
        recorded.append((run_id, seq))

    monkeypatch.setattr(orch, "_record_committed_in_session", _record)

    await orch.run(uuid4(), mode="apply")

    assert len(recorded) == 3
    seqs = [seq for _, seq in recorded]
    assert seqs == [0, 1, 2]


# ---------------------------------------------------------------------------
# 3. Incremental undo.json writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_phase_writes_undo_json_incrementally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ops = [_op() for _ in range(3)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler = _make_handler(undo_before={"row_id": "abc", "name": "before"})
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_DEDUPE_ENTITIES, handler)

    file_sink = DreamFileSink(base_dir=tmp_path, redact_text="none")

    write_calls: list[int] = []
    original_write = file_sink.write_undo_incremental

    def _spy_write(records: Any, **kwargs: Any) -> None:
        write_calls.append(len(records))
        original_write(records, **kwargs)

    monkeypatch.setattr(file_sink, "write_undo_incremental", _spy_write)

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[file_sink])
    namespace_id = uuid4()
    result = await orch.run(namespace_id, mode="apply")

    assert write_calls == [1, 2, 3]

    run_dir = tmp_path / str(namespace_id)
    undo_files = list(run_dir.rglob("*.undo.json"))
    assert len(undo_files) == 1
    payload = json.loads(undo_files[0].read_text())
    assert payload["schema_version"] == UNDO_SCHEMA_VERSION
    assert payload["run_id"] == str(result.run.run_id)
    assert len(payload["ops"]) == 3
    for entry in payload["ops"]:
        assert entry["before"] == {"row_id": "abc", "name": "before"}
        assert entry["op_type"] == str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)


# ---------------------------------------------------------------------------
# 4. Kill-switch blocks apply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_switch_blocks_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KHORA_DREAM_DISABLE_APPLY", "1")

    handler = _make_handler()
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_DEDUPE_ENTITIES, handler)

    ops = [_op()]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    with pytest.raises(DreamApplyDisabled, match="KHORA_DREAM_DISABLE_APPLY"):
        await orch.run(uuid4(), mode="apply")

    assert handler.calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_kill_switch_falsey_values_do_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """``KHORA_DREAM_DISABLE_APPLY=0`` / ``=false`` / ``=""`` must NOT block."""
    for falsey in ("", "0", "false", "False", "no"):
        monkeypatch.setenv("KHORA_DREAM_DISABLE_APPLY", falsey)
        kb = _FakeKB()
        orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
        assert orch._apply_disabled is False, f"falsey {falsey!r} incorrectly tripped kill-switch"


# ---------------------------------------------------------------------------
# 5. chunk_id assertion
# ---------------------------------------------------------------------------


def test_chunk_id_assertion_fires_on_violation() -> None:
    record = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_dedupe_entities",
        before={"chunk_id": "11111111-1111-1111-1111-111111111111", "name": "x"},
        applied_at=datetime.now(UTC),
    )
    with pytest.raises(DreamForbiddenOpError, match="chunk_id"):
        _assert_no_chunk_id_mutation(record)


def test_chunk_id_assertion_passes_when_absent() -> None:
    record = UndoRecord(
        op_id=uuid4(),
        op_type="vectorcypher_dedupe_entities",
        before={"entity_id": "abc", "name": "x"},
        applied_at=datetime.now(UTC),
    )
    _assert_no_chunk_id_mutation(record)


@pytest.mark.asyncio
async def test_chunk_id_assertion_aborts_apply_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A handler that returns a chunk_id-bearing undo aborts the run."""
    ops = [_op()]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    bad_handler = _make_handler(undo_before={"chunk_id": "11111111-1111-1111-1111-111111111111"})
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_DEDUPE_ENTITIES, bad_handler)

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
    with pytest.raises(DreamForbiddenOpError, match="chunk_id"):
        await orch.run(uuid4(), mode="apply")


# ---------------------------------------------------------------------------
# 6. Retention-days floor
# ---------------------------------------------------------------------------


def test_retention_days_floor_rejected_at_config_load() -> None:
    with pytest.raises(ValueError, match="fact_compaction_retention_days must be >= 7"):
        DreamConfig(fact_compaction_retention_days=3)


def test_retention_days_floor_accepts_exact_floor() -> None:
    cfg = DreamConfig(fact_compaction_retention_days=7)
    assert cfg.fact_compaction_retention_days == 7


def test_retention_days_floor_apply_mode_validator() -> None:
    """Even with the field validator bypassed, the model-level apply-mode
    validator catches an under-floor value paired with apply mode."""
    with pytest.raises(ValueError):
        DreamConfig(
            enabled=True,
            default_mode="apply",
            fact_compaction_retention_days=1,
        )


# ---------------------------------------------------------------------------
# 7. Resume from last_committed_op_seq
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_phase_resumes_from_last_committed_op_seq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = [_op() for _ in range(4)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler = _make_handler()
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_DEDUPE_ENTITIES, handler)

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    async def _fake_read_last_committed(run_id: UUID) -> int:
        del run_id
        return 1  # Ops 0 and 1 are already committed; resume at 2.

    monkeypatch.setattr(orch, "_read_last_committed", _fake_read_last_committed)

    await orch.run(uuid4(), mode="apply")

    assert len(handler.calls) == 2  # type: ignore[attr-defined]
    assert [c["op_id"] for c in handler.calls] == [ops[2].op_id, ops[3].op_id]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Pass-through behaviour for Phase 1 audit ops (no apply handler)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_phase_audit_op_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """An op without an apply handler advances the checkpoint, no undo, no handler."""
    audit_op = _op(OpKind.CHRONICLE_TOMBSTONE_AUDIT)
    plugin = _StubPlugin([audit_op])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler_called = []

    def _no_handler(op_type: OpKind | str) -> Any:
        del op_type
        handler_called.append(True)
        return None

    monkeypatch.setattr(registry_mod, "get_apply_handler", _no_handler)
    import khora.dream.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "get_apply_handler", _no_handler)

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
    result = await orch.run(uuid4(), mode="apply")
    assert result.run.mode == "apply"
    assert handler_called


# ---------------------------------------------------------------------------
# Lock-hold invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_is_held_during_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-namespace lock context wraps both plan and apply phases.

    The orchestrator's ``_lock`` is an asynccontextmanager. We assert
    the run flows through it by spying on entry / exit and verifying
    that the apply handlers fire between entry and exit.
    """
    ops = [_op() for _ in range(2)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    events: list[str] = []
    handler_op_ids: list[UUID] = []

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del coordinator, session
        events.append(f"handler:{op.op_id}")
        handler_op_ids.append(op.op_id)
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={"x": 1},
            applied_at=datetime.now(UTC),
        )

    def _lookup(op_type: OpKind | str) -> Any:
        del op_type
        return _handler

    monkeypatch.setattr(registry_mod, "get_apply_handler", _lookup)
    import khora.dream.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "get_apply_handler", _lookup)

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    original_lock = orch._lock

    @contextlib.asynccontextmanager
    async def _spy_lock(namespace_id: UUID) -> Any:
        events.append("lock-acquired")
        try:
            async with original_lock(namespace_id):
                yield
        finally:
            events.append("lock-released")

    monkeypatch.setattr(orch, "_lock", _spy_lock)

    await orch.run(uuid4(), mode="apply")

    acquired_idx = events.index("lock-acquired")
    released_idx = events.index("lock-released")
    handler_indices = [i for i, e in enumerate(events) if e.startswith("handler:")]
    assert handler_indices, "no handlers ran"
    assert acquired_idx < min(handler_indices)
    assert max(handler_indices) < released_idx
