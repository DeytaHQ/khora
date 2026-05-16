"""Entry-point bodies for ``Khora.dream`` / ``dream_status`` / ``dream_history``.

The :class:`khora.Khora` class re-exposes these as bound methods. Phase
0.1 keeps the bodies as :class:`NotImplementedError` stubs — wiring
lands in #661.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from khora.dream.config import DreamConfig
from khora.dream.orchestrator import DreamOrchestrator
from khora.dream.plan import DreamScope, OpKind
from khora.dream.result import DreamProgress, DreamResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from khora.khora import Khora


_NOT_WIRED = "Dream orchestrator not yet wired — see #649 phase 0.5 / #661"


async def dream(
    kb: Khora,
    namespace: str | UUID,
    *,
    mode: str = "dry-run",
    scope: DreamScope | None = None,
    ops: Iterable[OpKind] | None = None,
    config: DreamConfig | None = None,
    on_progress: Callable[[DreamProgress], None] | None = None,
    resume_from: UUID | None = None,
) -> DreamResult:
    """Run a dream-phase pass over ``namespace``.

    Constructs a :class:`DreamOrchestrator` and delegates. Phase 0.1
    stub: raises :class:`NotImplementedError` until #661 lands.
    """
    effective_config = config if config is not None else kb._config.dream
    _ = DreamOrchestrator(kb, effective_config)
    raise NotImplementedError(_NOT_WIRED)


async def dream_status(kb: Khora, run_id: UUID) -> dict[str, object]:
    """Return live or post-mortem status for a dream run."""
    _ = kb
    _ = run_id
    raise NotImplementedError(_NOT_WIRED)


async def dream_history(
    kb: Khora,
    namespace: str | UUID,
    *,
    limit: int = 20,
) -> list[DreamResult]:
    """Return recent dream-run results for ``namespace`` (newest first)."""
    _ = kb
    _ = namespace
    _ = limit
    raise NotImplementedError(_NOT_WIRED)
