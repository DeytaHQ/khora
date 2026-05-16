"""Dream backend capability Protocol.

Engines opt into dream-phase by structurally implementing :class:`DreamCapable`.
The orchestrator runtime-checks ``isinstance(engine, DreamCapable)`` and raises
if an engine without dream support is passed.

``plan_dream`` is pure (no writes); ``apply_dream`` is the destructive phase.
A plan hash (``sha1(canonical_json(plan))``) lets a resuming ``apply_dream``
validate that the world hasn't changed under the orchestrator.

Kept as a **separate** Protocol from
:class:`khora.engines.protocol.MemoryEngineProtocol` so engines that don't
want to grow a dream surface aren't forced to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from khora.dream.config import DreamConfig
    from khora.dream.plan import Checkpoint, DreamPlan, DreamScope, OpKind
    from khora.dream.result import DreamProgress, DreamResult
    from khora.extraction.skills.base import ExpertiseConfig


@runtime_checkable
class DreamCapable(Protocol):
    """Engines opt into dream-phase by implementing this Protocol.

    The orchestrator runtime-checks ``isinstance(engine, DreamCapable)``
    and raises if an engine without dream support is passed.

    Methods:
        plan_dream: pure read-side — inspects state and produces a
            :class:`DreamPlan`. Must not mutate.
        apply_dream: destructive — executes the plan and returns a
            :class:`DreamResult`. Honors ``checkpoint`` for resume.

    Properties:
        dream_capabilities: the subset of :class:`OpKind` values this
            engine supports. The orchestrator filters scope ops against
            this set before dispatching.
    """

    @property
    def dream_capabilities(self) -> frozenset[OpKind]:
        """Op kinds this engine can plan/apply. Empty set = no support."""
        ...

    async def plan_dream(
        self,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        config: DreamConfig,
        expertise: ExpertiseConfig | None = None,
    ) -> DreamPlan:
        """Build a :class:`DreamPlan` for ``namespace_id`` under ``scope``.

        Must be free of writes — callers may invoke this in dry-run mode
        without expecting side effects.
        """
        ...

    async def apply_dream(
        self,
        plan: DreamPlan,
        *,
        checkpoint: Checkpoint | None = None,
        on_progress: Callable[[DreamProgress], Awaitable[None]] | None = None,
    ) -> DreamResult:
        """Execute ``plan``, optionally resuming from ``checkpoint``.

        When ``checkpoint`` is provided, the engine must validate that
        ``checkpoint.plan_hash`` matches the hash of ``plan`` before
        applying any ops, and skip ops up to ``last_committed_op_seq``.
        """
        ...
