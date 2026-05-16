"""Dream orchestrator — placeholder.

The real orchestrator (plan → execute → diff → report) lands in #661.
This module ships only the class signature so call sites can already
import :class:`DreamOrchestrator` and downstream tickets can wire it up
without churning import paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.dream.config import DreamConfig
from khora.dream.plan import DreamPlan, DreamScope, OpKind
from khora.dream.result import DreamProgress, DreamResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from khora.khora import Khora


_NOT_WIRED = "Dream orchestrator not yet wired — see #649 phase 0.5 / #661"


class DreamOrchestrator:
    """Coordinates a single dream run end-to-end.

    Construction is cheap; all real work happens inside :meth:`run`.
    Method bodies raise :class:`NotImplementedError` in Phase 0.1.
    """

    def __init__(
        self,
        kb: Khora,
        config: DreamConfig,
    ) -> None:
        self._kb = kb
        self._config = config

    async def plan(
        self,
        namespace_id: UUID,
        *,
        scope: DreamScope | None = None,
        ops: Iterable[OpKind] | None = None,
    ) -> DreamPlan:
        """Build a :class:`DreamPlan` for the given namespace + scope."""
        raise NotImplementedError(_NOT_WIRED)

    async def run(
        self,
        namespace_id: UUID,
        *,
        mode: str = "dry-run",
        scope: DreamScope | None = None,
        ops: Iterable[OpKind] | None = None,
        on_progress: Callable[[DreamProgress], None] | None = None,
        resume_from: UUID | None = None,
    ) -> DreamResult:
        """Plan and execute a dream run."""
        raise NotImplementedError(_NOT_WIRED)

    async def status(self, run_id: UUID) -> dict[str, Any]:
        """Return live or post-mortem status for ``run_id``."""
        raise NotImplementedError(_NOT_WIRED)

    async def history(
        self,
        namespace_id: UUID,
        *,
        limit: int = 20,
    ) -> list[DreamResult]:
        """Return the most recent :class:`DreamResult`s for the namespace."""
        raise NotImplementedError(_NOT_WIRED)
