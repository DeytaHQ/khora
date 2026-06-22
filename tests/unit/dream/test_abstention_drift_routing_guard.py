"""Guard test for the chronicle abstention-drift routing check (#1331).

Before #1331 the registry used ``hasattr(engine, "_abstention_min_top_score")``
to detect whether the abstention-drift op was routed to a real ChronicleEngine.
#1331 gives VectorCypher those attrs too, so the guard now checks the active
engine NAME instead. These tests pin both legs: chronicle plans the op,
vectorcypher raises.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.dream.config import DreamConfig
from khora.dream.engines.registry import _ChroniclePlugin
from khora.dream.plan import DreamScope, OpKind
from khora.exceptions import KhoraError


class _EngineAttrs:
    """Both engines now expose these (#1331); presence no longer disambiguates."""

    _abstention_min_top_score = 0.3
    _abstention_combined_threshold = 0.5
    _abstention_min_chunks = 1


class _KB:
    """Minimal Khora-shaped stub: only ``_engine_name`` + ``_get_engine``."""

    def __init__(self, engine_name: str) -> None:
        self._engine_name = engine_name

    def _get_engine(self) -> object:
        return _EngineAttrs()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_abstention_drift_routes_on_chronicle() -> None:
    """engine_name == 'chronicle' → the op is planned (no raise)."""
    plugin = _ChroniclePlugin()
    scope = DreamScope(op_kinds=(OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT,))

    plan = await plugin.plan_dream(
        _KB("chronicle"),  # type: ignore[arg-type]
        uuid4(),
        scope=scope,
        config=DreamConfig(),
    )

    assert any(op.op_type == OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT for op in plan.ops)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_abstention_drift_rejects_vectorcypher() -> None:
    """engine_name == 'vectorcypher' → misconfig, raises despite the attrs."""
    plugin = _ChroniclePlugin()
    scope = DreamScope(op_kinds=(OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT,))

    with pytest.raises(KhoraError, match="not a ChronicleEngine"):
        await plugin.plan_dream(
            _KB("vectorcypher"),  # type: ignore[arg-type]
            uuid4(),
            scope=scope,
            config=DreamConfig(),
        )
