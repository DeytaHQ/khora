"""#833 mode contract tests for ChronicleEngine.

Chronicle implements VECTOR, HYBRID, and ALL. KEYWORD and GRAPH raise
``EngineCapabilityError`` - KEYWORD doesn't fit chronicle's design
(temporal scoring is its differentiator), and GRAPH is impossible
(chronicle has no graph backend).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.chronicle.engine import ChronicleEngine
from khora.exceptions import EngineCapabilityError
from khora.query import SearchMode


def _build_engine_with_stubs() -> ChronicleEngine:
    """Construct a ChronicleEngine bypassing connect() with minimal stubs.

    The mode-contract guard fires before any storage call, so the storage
    stub stays untouched on unsupported modes.
    """
    cfg = MagicMock()
    cfg.storage.backend = "pgvector"
    cfg.query = None
    cfg.telemetry_database_url = None

    engine = ChronicleEngine.__new__(ChronicleEngine)
    engine._config = cfg
    engine._backend = "pgvector"
    engine._storage = AsyncMock()
    engine._embedder = AsyncMock()
    engine._connected = True
    engine._router_enabled = False
    engine._router = MagicMock()
    return engine


def test_supported_modes_declaration() -> None:
    """Chronicle's supported_modes set is exactly {VECTOR, HYBRID, ALL}."""
    assert ChronicleEngine.supported_modes == frozenset({SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL})


@pytest.mark.parametrize("mode", [SearchMode.KEYWORD, SearchMode.GRAPH])
async def test_recall_unsupported_mode_raises(mode: SearchMode) -> None:
    """KEYWORD and GRAPH raise EngineCapabilityError before any storage I/O."""
    engine = _build_engine_with_stubs()
    with pytest.raises(EngineCapabilityError) as excinfo:
        await engine.recall("q", uuid4(), mode=mode)
    assert excinfo.value.engine_name == "chronicle"
    assert excinfo.value.mode is mode
    assert mode not in excinfo.value.supported_modes
    # Storage was never touched.
    engine._storage.search_similar_chunks.assert_not_called()
