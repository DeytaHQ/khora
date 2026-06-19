"""Dream LLM token-budget enforcer (#1270).

The dream orchestrator caps LLM spend with two budgets read off
:class:`khora.dream.config.DreamConfig`:

- ``llm_max_tokens_per_run`` - hard per-run ceiling.
- ``llm_max_tokens_per_namespace_per_day`` - rolling-day ceiling shared
  across runs in the same namespace.

The accumulator is fed by the same ``record_usage`` path that
``record_llm_call`` / ``acompletion`` populate. The enforcer is checked
*before* each LLM-using op: once a budget is exhausted the orchestrator
skips the remaining LLM ops, records an ``llm_budget_exhausted`` skip
reason, and emits ``khora.dream.llm.throttled_total`` (no namespace_id
label - cardinality rule). Already-applied ops stay committed.

Runs entirely against in-process stubs - no Docker, no real LLM, no
Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream import orchestrator as orchestrator_mod
from khora.dream.config import DreamConfig
from khora.dream.engines import registry as registry_mod
from khora.dream.orchestrator import DreamOrchestrator, _reset_namespace_llm_budgets
from khora.dream.plan import DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.result import UndoRecord

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _NoopSession:
    def __init__(self) -> None:
        self.bind = None

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(first=lambda: None, all=lambda: [])


class _FakeCoordinator:
    def __init__(self) -> None:
        self._graph = None

    def transaction(self) -> Any:
        return _FakeTxnCtx()


class _FakeTxnCtx:
    async def __aenter__(self) -> Any:
        return SimpleNamespace(session=_NoopSession())

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FakeKB:
    def __init__(self) -> None:
        self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
        self._engine_name = "stub"
        self.storage = _FakeCoordinator()


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


def _llm_op(namespace_id: UUID) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="mutation",
        op_type=OpKind.VECTORCYPHER_COMMUNITY_SUMMARY,
        decision="planned",
        rationale="test",
        inputs=({"community_id": str(uuid4())},),
        outputs=(),
        started_at=datetime.now(UTC),
        duration_ms=1.0,
        namespace_id=namespace_id,
    )


def _make_llm_handler(tokens_per_call: int) -> Any:
    """Apply handler that spends ``tokens_per_call`` via the usage path.

    Mirrors how ``acompletion`` records spend: it appends an ``LLMUsage``
    entry into the context-local accumulator that the orchestrator reads
    back to update its budget buckets.
    """
    calls: list[UUID] = []

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del coordinator, session
        calls.append(op.op_id)
        from khora.khora import LLMUsage
        from khora.telemetry.context import record_usage

        record_usage(
            LLMUsage(
                operation="dream_community_summary",
                model="gpt-4o-mini",
                prompt_tokens=tokens_per_call // 2,
                completion_tokens=tokens_per_call - tokens_per_call // 2,
                total_tokens=tokens_per_call,
                latency_ms=1.0,
            )
        )
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={"community_id": str(uuid4())},
            applied_at=datetime.now(UTC),
        )

    _handler.calls = calls  # type: ignore[attr-defined]
    return _handler


def _install_handler(monkeypatch: pytest.MonkeyPatch, op_kind: OpKind, handler: Any) -> None:
    def _lookup(op_type: OpKind | str) -> Any:
        if str(op_type) == str(op_kind):
            return handler
        return None

    monkeypatch.setattr(registry_mod, "get_apply_handler", _lookup)
    monkeypatch.setattr(orchestrator_mod, "get_apply_handler", _lookup)


@pytest.fixture(autouse=True)
def _clear_budgets() -> Any:
    """Reset the process-global per-namespace-per-day bucket between tests."""
    _reset_namespace_llm_budgets()
    yield
    _reset_namespace_llm_budgets()


# ---------------------------------------------------------------------------
# Per-run budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_run_budget_stops_fanning_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run exceeding ``llm_max_tokens_per_run`` stops calling the LLM after
    the budget is reached; later ops are skipped, not applied."""
    namespace_id = uuid4()
    ops = [_llm_op(namespace_id) for _ in range(5)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler = _make_llm_handler(tokens_per_call=100)
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_COMMUNITY_SUMMARY, handler)

    kb = _FakeKB()
    # Budget allows two calls (100 + 100 = 200); the third op exceeds it.
    cfg = DreamConfig(enabled=True, llm_max_tokens_per_run=200)
    orch = DreamOrchestrator(kb, cfg, sinks=[])

    result = await orch.run(namespace_id, mode="apply")

    # Two ops ran, three were skipped by the budget.
    assert len(handler.calls) == 2  # type: ignore[attr-defined]

    summary = result.ops[0]
    assert summary.op_type == OpKind.VECTORCYPHER_COMMUNITY_SUMMARY.value
    assert summary.applied == 2
    assert summary.skipped == 3

    reasons = result.metadata["skip_reasons"]
    budget_reasons = [r for r in reasons if r["reason"] == "llm_budget_exhausted"]
    assert len(budget_reasons) == 3
    assert budget_reasons[0]["op_kind"] == OpKind.VECTORCYPHER_COMMUNITY_SUMMARY.value


@pytest.mark.asyncio
async def test_under_budget_run_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run comfortably under both budgets applies every op and records no
    budget skip reasons."""
    namespace_id = uuid4()
    ops = [_llm_op(namespace_id) for _ in range(3)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler = _make_llm_handler(tokens_per_call=10)
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_COMMUNITY_SUMMARY, handler)

    kb = _FakeKB()
    cfg = DreamConfig(enabled=True, llm_max_tokens_per_run=10_000)
    orch = DreamOrchestrator(kb, cfg, sinks=[])

    result = await orch.run(namespace_id, mode="apply")

    assert len(handler.calls) == 3  # type: ignore[attr-defined]
    summary = result.ops[0]
    assert summary.applied == 3
    assert summary.skipped == 0
    reasons = result.metadata["skip_reasons"]
    assert [r for r in reasons if r["reason"] == "llm_budget_exhausted"] == []


@pytest.mark.asyncio
async def test_budget_zero_means_disabled_no_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A budget of 0 means 'disabled' (no cap) - mirrors the hooks convention
    where 0 disables the per-subscription budget."""
    namespace_id = uuid4()
    ops = [_llm_op(namespace_id) for _ in range(3)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler = _make_llm_handler(tokens_per_call=10_000)
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_COMMUNITY_SUMMARY, handler)

    kb = _FakeKB()
    cfg = DreamConfig(
        enabled=True,
        llm_max_tokens_per_run=0,
        llm_max_tokens_per_namespace_per_day=0,
    )
    orch = DreamOrchestrator(kb, cfg, sinks=[])

    result = await orch.run(namespace_id, mode="apply")

    assert len(handler.calls) == 3  # type: ignore[attr-defined]
    assert result.ops[0].applied == 3
    assert result.ops[0].skipped == 0


# ---------------------------------------------------------------------------
# Per-namespace-per-day budget (persists across runs in-process)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_namespace_per_day_budget_spans_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-namespace-per-day bucket accumulates across runs: a second run
    in the same namespace is throttled by spend from the first run."""
    namespace_id = uuid4()
    handler = _make_llm_handler(tokens_per_call=100)
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_COMMUNITY_SUMMARY, handler)

    kb = _FakeKB()
    # Per-run cap is generous; the per-day cap (100) is the binding one.
    # The budget is reactive (fed by record_llm_call) so an op is refused
    # once accumulated day spend has reached the cap.
    cfg = DreamConfig(
        enabled=True,
        llm_max_tokens_per_run=10_000,
        llm_max_tokens_per_namespace_per_day=100,
    )

    # First run: one op (100 tokens) applies, day bucket now at 100 (== cap).
    plugin1 = _StubPlugin([_llm_op(namespace_id)])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin1})
    orch1 = DreamOrchestrator(kb, cfg, sinks=[])
    r1 = await orch1.run(namespace_id, mode="apply")
    assert r1.ops[0].applied == 1
    assert len(handler.calls) == 1  # type: ignore[attr-defined]

    # Second run: the day bucket (100) is already at the cap -> throttled.
    plugin2 = _StubPlugin([_llm_op(namespace_id)])
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin2})
    orch2 = DreamOrchestrator(kb, cfg, sinks=[])
    r2 = await orch2.run(namespace_id, mode="apply")

    # Handler not called again - still only one total call.
    assert len(handler.calls) == 1  # type: ignore[attr-defined]
    assert r2.ops[0].applied == 0
    assert r2.ops[0].skipped == 1
    budget_reasons = [r for r in r2.metadata["skip_reasons"] if r["reason"] == "llm_budget_exhausted"]
    assert len(budget_reasons) == 1
    assert "per_namespace_per_day" in budget_reasons[0]["detail"]


@pytest.mark.asyncio
async def test_throttle_metric_emitted_on_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    """A budget breach emits ``khora.dream.llm.throttled_total`` once per
    skipped op."""
    namespace_id = uuid4()
    ops = [_llm_op(namespace_id) for _ in range(3)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    handler = _make_llm_handler(tokens_per_call=100)
    _install_handler(monkeypatch, OpKind.VECTORCYPHER_COMMUNITY_SUMMARY, handler)

    throttle_adds: list[int] = []

    class _FakeCounter:
        def add(self, value: int, attributes: Any = None) -> None:
            del attributes
            throttle_adds.append(value)

    monkeypatch.setattr(orchestrator_mod, "_THROTTLE_COUNTER", _FakeCounter())

    kb = _FakeKB()
    cfg = DreamConfig(enabled=True, llm_max_tokens_per_run=100)
    orch = DreamOrchestrator(kb, cfg, sinks=[])

    await orch.run(namespace_id, mode="apply")

    # First op applies (spends 100), next two breach -> two throttle events.
    assert len(handler.calls) == 1  # type: ignore[attr-defined]
    assert sum(throttle_adds) == 2


@pytest.mark.asyncio
async def test_non_llm_ops_never_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-LLM mutation ops are never gated by the LLM budget even when the
    per-run budget is tiny."""
    namespace_id = uuid4()

    def _dedupe_op() -> DreamOp:
        return DreamOp(
            op_id=uuid4(),
            phase="mutation",
            op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
            decision="merge",
            rationale="test",
            outputs=({"merged": 1},),
            started_at=datetime.now(UTC),
            duration_ms=1.0,
            namespace_id=namespace_id,
        )

    ops = [_dedupe_op() for _ in range(3)]
    plugin = _StubPlugin(ops)
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    calls: list[UUID] = []

    async def _handler(op: DreamOp, *, coordinator: Any, session: Any) -> UndoRecord:
        del coordinator, session
        calls.append(op.op_id)
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={"entity_id": str(uuid4())},
            applied_at=datetime.now(UTC),
        )

    _install_handler(monkeypatch, OpKind.VECTORCYPHER_DEDUPE_ENTITIES, _handler)

    kb = _FakeKB()
    cfg = DreamConfig(enabled=True, llm_max_tokens_per_run=1)
    orch = DreamOrchestrator(kb, cfg, sinks=[])

    result = await orch.run(namespace_id, mode="apply")

    assert len(calls) == 3
    assert result.ops[0].applied == 3
    assert [r for r in result.metadata["skip_reasons"] if r["reason"] == "llm_budget_exhausted"] == []
