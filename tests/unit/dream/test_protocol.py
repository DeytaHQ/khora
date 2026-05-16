"""Runtime-check tests for :class:`khora.dream.protocol.DreamCapable`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID, uuid4

from khora.dream.config import DreamConfig
from khora.dream.plan import Checkpoint, DreamPlan, DreamScope, OpKind
from khora.dream.protocol import DreamCapable
from khora.dream.result import (
    DreamDiff,
    DreamProgress,
    DreamResult,
    DreamRunInfo,
)
from khora.extraction.skills.base import ExpertiseConfig


def _make_run_info() -> DreamRunInfo:
    from datetime import UTC, datetime

    return DreamRunInfo(
        run_id=uuid4(),
        namespace_id=uuid4(),
        mode="dry-run",
        started_at=datetime.now(UTC),
    )


class _ConformingEngine:
    """A minimal engine that structurally implements DreamCapable."""

    @property
    def dream_capabilities(self) -> frozenset[OpKind]:
        return frozenset({OpKind.DEDUPE_ENTITIES})

    async def plan_dream(
        self,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        config: DreamConfig,
        expertise: ExpertiseConfig | None = None,
    ) -> DreamPlan:
        return DreamPlan(plan_id=uuid4(), namespace_id=namespace_id)

    async def apply_dream(
        self,
        plan: DreamPlan,
        *,
        checkpoint: Checkpoint | None = None,
        on_progress: Callable[[DreamProgress], Awaitable[None]] | None = None,
    ) -> DreamResult:
        return DreamResult(run=_make_run_info(), diff=DreamDiff())


class _MissingApply:
    """Missing apply_dream → must NOT pass isinstance(DreamCapable)."""

    @property
    def dream_capabilities(self) -> frozenset[OpKind]:
        return frozenset()

    async def plan_dream(
        self,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        config: DreamConfig,
        expertise: ExpertiseConfig | None = None,
    ) -> DreamPlan:
        return DreamPlan(plan_id=uuid4(), namespace_id=namespace_id)


class _MissingPlan:
    """Missing plan_dream → must NOT pass isinstance(DreamCapable)."""

    @property
    def dream_capabilities(self) -> frozenset[OpKind]:
        return frozenset()

    async def apply_dream(
        self,
        plan: DreamPlan,
        *,
        checkpoint: Checkpoint | None = None,
        on_progress: Callable[[DreamProgress], Awaitable[None]] | None = None,
    ) -> DreamResult:
        return DreamResult(run=_make_run_info(), diff=DreamDiff())


class _MissingCapabilities:
    """Missing the dream_capabilities property."""

    async def plan_dream(
        self,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        config: DreamConfig,
        expertise: ExpertiseConfig | None = None,
    ) -> DreamPlan:
        return DreamPlan(plan_id=uuid4(), namespace_id=namespace_id)

    async def apply_dream(
        self,
        plan: DreamPlan,
        *,
        checkpoint: Checkpoint | None = None,
        on_progress: Callable[[DreamProgress], Awaitable[None]] | None = None,
    ) -> DreamResult:
        return DreamResult(run=_make_run_info(), diff=DreamDiff())


def test_protocol_isinstance_check() -> None:
    """A class implementing all three members passes isinstance."""
    assert isinstance(_ConformingEngine(), DreamCapable)


def test_protocol_missing_apply_fails() -> None:
    """A class missing apply_dream does NOT pass isinstance."""
    assert not isinstance(_MissingApply(), DreamCapable)


def test_protocol_missing_plan_fails() -> None:
    """A class missing plan_dream does NOT pass isinstance."""
    assert not isinstance(_MissingPlan(), DreamCapable)


def test_protocol_missing_capabilities_fails() -> None:
    """A class missing dream_capabilities does NOT pass isinstance."""
    assert not isinstance(_MissingCapabilities(), DreamCapable)


def test_protocol_runtime_check_with_property() -> None:
    """dream_capabilities must be a property returning frozenset."""
    engine = _ConformingEngine()
    assert isinstance(engine.dream_capabilities, frozenset)
    assert OpKind.DEDUPE_ENTITIES in engine.dream_capabilities


def test_checkpoint_dataclass_shape() -> None:
    """Checkpoint has the run_id / last_committed_op_seq / plan_hash trio."""
    cp = Checkpoint(run_id=uuid4(), last_committed_op_seq=3, plan_hash="abc123")
    assert cp.last_committed_op_seq == 3
    assert cp.plan_hash == "abc123"
