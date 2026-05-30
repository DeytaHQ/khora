"""Observability tests for ``DreamResult.metadata['skip_reasons']`` (#876).

Repros the user-visible bug:
``kb.dream(scope=DreamScope(op_kinds=(OpKind.CLUSTER_EVENTS,)))`` returns
successfully with ``len(result.ops) == 0`` and no signal explaining why.
After the fix, ``result.metadata["skip_reasons"]`` carries one entry per
dropped or no-candidate op so callers can distinguish:

- (a) requested op kind not owned by the active engine plugin
  -> ``"op_not_supported_by_engine"``
- (b) planner ran but its source query returned no rows
  -> ``"no_candidates"``

Runs entirely against in-process stubs and an in-memory SQLite session.
No Docker, no LLM, no Postgres.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from khora.dream.api import dream as dream_api
from khora.dream.config import DreamConfig
from khora.dream.engines import registry as registry_mod
from khora.dream.engines.chronicle import plan_chronicle_event_clustering
from khora.dream.orchestrator import DreamOrchestrator
from khora.dream.plan import DreamOp, DreamPlan, DreamScope, OpKind

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub harness (mirrors the shape used in test_orchestrator.py).
# ---------------------------------------------------------------------------


class _StubPlugin:
    """Plugin that owns exactly the ops it's constructed with."""

    def __init__(self, owned: frozenset[OpKind], ops: list[DreamOp]) -> None:
        self._owned = owned
        self._ops = ops

    @property
    def dream_capabilities(self) -> frozenset[OpKind]:
        return self._owned

    async def plan_dream(
        self,
        kb: Any,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        config: DreamConfig,
        expertise: Any = None,
    ) -> DreamPlan:
        del kb, config, expertise
        skip_reasons: list[dict[str, Any]] = []
        wanted = registry_mod._resolved_scope(
            scope,
            self._owned,
            skip_reasons=skip_reasons,
            engine_name="stub",
        )
        ops = [op for op in self._ops if op.op_type in wanted]
        return DreamPlan(
            plan_id=uuid4(),
            namespace_id=namespace_id,
            ops=tuple(ops),
            metadata={"skip_reasons": skip_reasons} if skip_reasons else {},
        )

    async def apply_dream(self, plan: DreamPlan, **kwargs: Any) -> Any:
        del plan, kwargs
        raise NotImplementedError  # orchestrator iterates per-op itself


class _FakeCoordinator:
    def transaction(self) -> Any:
        raise RuntimeError("no SQL backend connected (test stub)")


class _FakeKB:
    def __init__(self) -> None:
        self._config = SimpleNamespace(dream=DreamConfig(enabled=True))
        self._engine_name = "stub"
        self.storage = _FakeCoordinator()


def _audit_op(op_type: OpKind) -> DreamOp:
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
# _resolved_scope direct unit
# ---------------------------------------------------------------------------


def test_resolved_scope_records_dropped_op_kinds() -> None:
    """Unsupported op kinds are appended as ``op_not_supported_by_engine`` entries."""
    capabilities = frozenset({OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT})
    scope = DreamScope(op_kinds=(OpKind.CLUSTER_EVENTS, OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT))
    reasons: list[dict[str, Any]] = []

    resolved = registry_mod._resolved_scope(
        scope,
        capabilities,
        skip_reasons=reasons,
        engine_name="chronicle",
    )

    assert resolved == frozenset({OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT})
    assert len(reasons) == 1
    entry = reasons[0]
    assert entry["op_kind"] == OpKind.CLUSTER_EVENTS.value
    assert entry["reason"] == "op_not_supported_by_engine"
    assert "chronicle" in entry["detail"]


def test_resolved_scope_no_drops_no_reasons() -> None:
    """When every requested op is supported, the reasons list stays empty."""
    capabilities = frozenset({OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT})
    scope = DreamScope(op_kinds=(OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT,))
    reasons: list[dict[str, Any]] = []

    resolved = registry_mod._resolved_scope(
        scope,
        capabilities,
        skip_reasons=reasons,
        engine_name="chronicle",
    )

    assert resolved == capabilities
    assert reasons == []


def test_resolved_scope_emits_warning_log(caplog: pytest.LogCaptureFixture) -> None:
    """Dropped op kinds trigger exactly one ``logger.warning`` per resolve call."""
    # Loguru's caplog integration is opt-in; we install a temporary
    # propagating handler so pytest's caplog sees the warning.
    import logging

    from loguru import logger

    sink_id = logger.add(
        lambda msg: logging.getLogger("khora.dream.registry.test").warning(msg.strip()),
        level="WARNING",
        format="{message}",
    )
    try:
        with caplog.at_level(logging.WARNING, logger="khora.dream.registry.test"):
            registry_mod._resolved_scope(
                DreamScope(op_kinds=(OpKind.CLUSTER_EVENTS, OpKind.CHRONICLE_EVENT_CLUSTERING)),
                frozenset(),
                skip_reasons=[],
                engine_name="chronicle",
            )
    finally:
        logger.remove(sink_id)

    matched = [r for r in caplog.records if "dropping op_kinds" in r.getMessage()]
    assert len(matched) == 1, f"expected one warning, got {len(matched)}: {[r.getMessage() for r in caplog.records]}"


# ---------------------------------------------------------------------------
# End-to-end via the public dream() API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_cluster_events_surfaces_skip_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repro of the #876 user flow.

    ``kb.dream(scope=DreamScope(op_kinds=(OpKind.CLUSTER_EVENTS,)))`` against
    a stub plugin that doesn't own ``CLUSTER_EVENTS`` returns
    ``len(result.ops) == 0`` AND ``result.metadata["skip_reasons"]``
    explains why.
    """
    plugin = _StubPlugin(
        owned=frozenset({OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT}),
        ops=[],
    )
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    kb = _FakeKB()
    result = await dream_api(
        kb,
        uuid4(),
        scope=DreamScope(op_kinds=(OpKind.CLUSTER_EVENTS,)),
    )

    assert result.ops == ()
    skip_reasons = result.metadata.get("skip_reasons")
    assert isinstance(skip_reasons, list)
    assert len(skip_reasons) == 1
    entry = skip_reasons[0]
    assert entry["op_kind"] == OpKind.CLUSTER_EVENTS.value
    assert entry["reason"] == "op_not_supported_by_engine"


@pytest.mark.asyncio
async def test_dream_supported_op_no_skip_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every requested op is satisfied, ``skip_reasons`` is empty."""
    op = _audit_op(OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT)
    plugin = _StubPlugin(
        owned=frozenset({OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT}),
        ops=[op],
    )
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    kb = _FakeKB()
    result = await dream_api(
        kb,
        uuid4(),
        scope=DreamScope(op_kinds=(OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT,)),
        mode="apply",
    )

    assert len(result.ops) == 1
    assert result.ops[0].applied == 1
    assert result.metadata.get("skip_reasons") == []


@pytest.mark.asyncio
async def test_dream_apply_path_backward_compatible_metadata_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plan_hash`` + ``plan_payload`` continue to live in metadata after the fix."""
    op = _audit_op(OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT)
    plugin = _StubPlugin(
        owned=frozenset({OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT}),
        ops=[op],
    )
    monkeypatch.setattr(registry_mod, "_REGISTRY", {"stub": plugin})

    kb = _FakeKB()
    orch = DreamOrchestrator(kb, DreamConfig(enabled=True), sinks=[])
    result = await orch.run(uuid4(), mode="apply")

    assert "plan_hash" in result.metadata
    assert "plan_payload" in result.metadata
    assert "skip_reasons" in result.metadata
    assert result.metadata["skip_reasons"] == []


# ---------------------------------------------------------------------------
# plan_chronicle_event_clustering: no_candidates surface
# ---------------------------------------------------------------------------


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """In-memory SQLite with the minimal ``chronicle_events`` schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    sa.text(
                        "CREATE TABLE chronicle_events ("
                        "id TEXT PRIMARY KEY, "
                        "namespace_id TEXT NOT NULL, "
                        "chunk_id TEXT NOT NULL, "
                        "subject TEXT NOT NULL, "
                        "verb TEXT NOT NULL, "
                        "object TEXT, "
                        "observation_date TEXT NOT NULL, "
                        "referenced_date TEXT, "
                        "confidence REAL NOT NULL DEFAULT 1.0, "
                        "embedding TEXT"
                        ")"
                    )
                )
            )
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_plan_chronicle_event_clustering_no_candidates_when_empty(
    session: AsyncSession,
) -> None:
    """Empty ``chronicle_events`` -> ``no_candidates`` skip reason."""
    skip_reasons: list[dict[str, Any]] = []
    ops = await plan_chronicle_event_clustering(
        uuid4(),
        session=session,
        config=DreamConfig(),
        _skip_reasons=skip_reasons,
    )

    assert ops == ()
    assert len(skip_reasons) == 1
    entry = skip_reasons[0]
    assert entry["op_kind"] == OpKind.CHRONICLE_EVENT_CLUSTERING.value
    assert entry["reason"] == "no_candidates"
    assert entry["detail"] is not None


@pytest.mark.asyncio
async def test_plan_chronicle_event_clustering_no_reasons_without_optin(
    session: AsyncSession,
) -> None:
    """The ``_skip_reasons`` kwarg is opt-in; the return shape is unchanged when omitted."""
    ops = await plan_chronicle_event_clustering(
        uuid4(),
        session=session,
        config=DreamConfig(),
    )
    assert ops == ()
