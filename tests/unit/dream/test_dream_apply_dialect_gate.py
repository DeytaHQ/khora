"""Dialect-gate guard for the vectorcypher apply handlers (#875).

The four vectorcypher mutation apply handlers
(``apply_vectorcypher_dedupe_entities`` and siblings) bind raw
``uuid.UUID`` values into ``session.execute``; only PostgreSQL handles
those natively. Running them against a SQLite session (the
``sqlite_lance`` test stack) used to crash with
``sqlite3.ProgrammingError: type 'UUID' is not supported``. The fix is
an orchestrator-level dialect gate that raises ``DreamBackendUnsupported``
before the handler runs, logs a warning, advances the checkpoint, and
records the op as ``skipped``.

Covers:
  * ``_assert_backend_supported`` raises on sqlite for every gated op kind.
  * The orchestrator's ``_apply_phase`` catches the exception and marks
    the op as skipped (no ``sqlite3.ProgrammingError`` leaks).
  * Non-gated ops (audit pass-throughs) still run on sqlite.
  * Sessions with no dialect bind are treated as Postgres-equivalent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

import khora.dream.engines.registry as registry_mod
from khora.dream.config import DreamConfig
from khora.dream.exceptions import DreamBackendUnsupported
from khora.dream.orchestrator import (
    _POSTGRES_ONLY_OP_KINDS,
    DreamOrchestrator,
    _assert_backend_supported,
)
from khora.dream.plan import DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.report import ReportSink
from khora.dream.result import UndoRecord

# ---------------------------------------------------------------------------
# Test doubles (mirror the shape used in test_orchestrator_apply.py)
# ---------------------------------------------------------------------------


class _CapturingSink(ReportSink):
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, event: Any) -> None:
        self.events.append(event)


class _SqliteSession:
    """Session stub whose dialect reports as sqlite. Any execute() call would crash."""

    def __init__(self) -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
        self.execute_called = False

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        # This simulates the real sqlite3.ProgrammingError surface area.
        self.execute_called = True
        raise AssertionError("execute() must not be reached: dialect gate should fire first")


class _SqliteTxnCtx:
    def __init__(self, coordinator: _SqliteCoordinator) -> None:
        self._coordinator = coordinator

    async def __aenter__(self) -> Any:
        return SimpleNamespace(session=self._coordinator.session)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _SqliteCoordinator:
    """Coordinator whose transaction yields a sqlite-dialect session."""

    def __init__(self, *, graph: Any | None = None) -> None:
        self.session = _SqliteSession()
        self._graph = graph

    def transaction(self) -> Any:
        return _SqliteTxnCtx(self)


class _StubPlugin:
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


class _FakeKB:
    def __init__(self, coordinator: _SqliteCoordinator) -> None:
        self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
        self._engine_name = "stub"
        self.storage = coordinator


def _op(op_type: OpKind) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=op_type,
        decision="planned",
        rationale="test",
        outputs=({"merges": []},),
        started_at=datetime.now(UTC),
        duration_ms=1.0,
        namespace_id=uuid4(),
    )


async def _exploding_handler(_op: DreamOp, **_kwargs: Any) -> UndoRecord:
    raise AssertionError("handler must not be invoked when dialect gate fires")


def _install_handler(monkeypatch: pytest.MonkeyPatch, op_kind: OpKind, handler: Any) -> None:
    def _stub(op_type: OpKind | str) -> Any:
        return handler if str(op_type) == str(op_kind) else None

    monkeypatch.setattr(registry_mod, "get_apply_handler", _stub)
    import khora.dream.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "get_apply_handler", _stub)


# ---------------------------------------------------------------------------
# Unit-level tests on the helper itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_kind_str", sorted(_POSTGRES_ONLY_OP_KINDS))
def test_assert_backend_supported_raises_on_sqlite(op_kind_str: str) -> None:
    """Every gated op kind must trip the dialect gate on sqlite."""
    session = _SqliteSession()
    with pytest.raises(DreamBackendUnsupported, match=op_kind_str):
        _assert_backend_supported(session, op_kind_str)


def test_assert_backend_supported_allows_postgresql() -> None:
    """A postgresql session passes the gate."""
    session = SimpleNamespace(bind=SimpleNamespace(dialect=SimpleNamespace(name="postgresql")))
    _assert_backend_supported(session, OpKind.VECTORCYPHER_DEDUPE_ENTITIES)


def test_assert_backend_supported_passes_when_dialect_unknown() -> None:
    """No bind / unreadable dialect is treated as Postgres-equivalent.

    Test stubs in the existing apply suite use ``SimpleNamespace(bind=None)``;
    treating them as Postgres-equivalent keeps the legacy test surface working
    without forcing every fixture to fake a dialect.
    """
    session = SimpleNamespace(bind=None)
    _assert_backend_supported(session, OpKind.VECTORCYPHER_DEDUPE_ENTITIES)


def test_assert_backend_supported_ignores_non_gated_ops() -> None:
    """Audit / non-gated op kinds never trip the gate even on sqlite."""
    session = _SqliteSession()
    _assert_backend_supported(session, OpKind.CHRONICLE_TOMBSTONE_AUDIT)
    _assert_backend_supported(session, OpKind.VECTORCYPHER_NORMALIZE_SCHEMA)


# ---------------------------------------------------------------------------
# Orchestrator-level integration: the apply loop catches and marks skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_phase_marks_op_skipped_on_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running a gated op against a sqlite session marks it skipped, no crash."""
    op = _op(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)
    plugin = _StubPlugin([op])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_DEDUPE_ENTITIES, _exploding_handler)

    coordinator = _SqliteCoordinator()
    kb = _FakeKB(coordinator)
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    result = await orch.run(uuid4(), mode="apply")

    # No execute() call leaked through the gate.
    assert coordinator.session.execute_called is False
    # The op was counted as skipped, not applied.
    summaries = {s.op_type: s for s in result.ops}
    summary = summaries[str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)]
    assert summary.planned == 1
    assert summary.applied == 0
    assert summary.skipped == 1
    assert summary.failed == 0


@pytest.mark.asyncio
async def test_apply_phase_does_not_leak_sqlite_programming_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must NOT surface sqlite3.ProgrammingError to the caller."""
    import sqlite3

    op = _op(OpKind.VECTORCYPHER_PRUNE_EDGES)
    plugin = _StubPlugin([op])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    async def _crashing_handler(_op: DreamOp, **_kwargs: Any) -> UndoRecord:
        # If the gate ever fails to fire, this is what a real apply handler
        # would do on its first bind. Asserting the orchestrator surfaces
        # neither this nor DreamBackendUnsupported.
        raise sqlite3.ProgrammingError("type 'UUID' is not supported")

    _install_handler(monkeypatch, OpKind.VECTORCYPHER_PRUNE_EDGES, _crashing_handler)

    coordinator = _SqliteCoordinator()
    kb = _FakeKB(coordinator)
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    # Must NOT raise sqlite3.ProgrammingError.
    result = await orch.run(uuid4(), mode="apply")

    summary = next(s for s in result.ops if s.op_type == str(OpKind.VECTORCYPHER_PRUNE_EDGES))
    assert summary.skipped == 1
    assert summary.applied == 0


@pytest.mark.asyncio
async def test_apply_phase_runs_audit_op_on_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit ops (no apply handler) still pass through on sqlite without the gate firing."""
    audit_op = _op(OpKind.CHRONICLE_TOMBSTONE_AUDIT)
    plugin = _StubPlugin([audit_op])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})
    # No apply handler installed - registry's lookup returns None for an
    # audit op kind.

    def _no_handler(_op_type: OpKind | str) -> Any:
        return None

    monkeypatch.setattr(registry_mod, "get_apply_handler", _no_handler)
    import khora.dream.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "get_apply_handler", _no_handler)

    coordinator = _SqliteCoordinator()
    kb = _FakeKB(coordinator)
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    result = await orch.run(uuid4(), mode="apply")

    summary = next(s for s in result.ops if s.op_type == str(OpKind.CHRONICLE_TOMBSTONE_AUDIT))
    # Audit ops are reported as applied because they pass through cleanly.
    assert summary.skipped == 0


@pytest.mark.asyncio
async def test_skip_count_per_op_type_drains_in_plan_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two gated ops + one audit op produces skipped=2 / applied=1 / planned=3."""
    gated_a = _op(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)
    gated_b = _op(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)
    audit = _op(OpKind.CHRONICLE_TOMBSTONE_AUDIT)
    plugin = _StubPlugin([gated_a, gated_b, audit])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    def _lookup(op_type: OpKind | str) -> Any:
        if str(op_type) == str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES):
            return _exploding_handler
        return None

    monkeypatch.setattr(registry_mod, "get_apply_handler", _lookup)
    import khora.dream.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "get_apply_handler", _lookup)

    coordinator = _SqliteCoordinator()
    kb = _FakeKB(coordinator)
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])

    result = await orch.run(uuid4(), mode="apply")

    summaries = {s.op_type: s for s in result.ops}
    dedupe_sum = summaries[str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES)]
    audit_sum = summaries[str(OpKind.CHRONICLE_TOMBSTONE_AUDIT)]
    assert dedupe_sum.planned == 2
    assert dedupe_sum.skipped == 2
    assert dedupe_sum.applied == 0
    assert audit_sum.planned == 1
    assert audit_sum.skipped == 0
