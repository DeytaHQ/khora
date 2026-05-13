"""Unit tests for ``SearchMode`` + Skeleton recall mode dispatch.

Pre-fix, ``SkeletonConstructionEngine.recall`` referenced a non-existent
``SearchMode.KEYWORD`` member. Any call with ``mode != VECTOR`` (HYBRID is
the default) crashed with ``AttributeError`` before reaching the temporal
store. These tests:

1. Pin the enum surface — ``KEYWORD`` exists alongside the four older
   members and ``ALL``/``HYBRID``/``VECTOR``/``GRAPH`` keep their identity.
2. Drive ``Skeleton.recall(mode=...)`` against an in-memory stub for each
   mode and assert the engine forwards the documented ``hybrid_alpha``
   default to the temporal store (the bug surface).

The stub avoids spinning up Postgres/Weaviate/SurrealDB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.skeleton.engine import SkeletonConstructionEngine
from khora.query import SearchMode


def test_search_mode_has_keyword_member() -> None:
    """SearchMode must expose KEYWORD."""
    # Attribute access — what the buggy ``elif`` does at engine.py:441.
    assert hasattr(SearchMode, "KEYWORD")
    assert SearchMode.KEYWORD is SearchMode["KEYWORD"]


def test_search_mode_membership_is_complete() -> None:
    """Adding KEYWORD must not displace any existing member."""
    names = {m.name for m in SearchMode}
    assert names == {"VECTOR", "GRAPH", "HYBRID", "ALL", "KEYWORD"}


def _build_engine_with_stubs() -> tuple[SkeletonConstructionEngine, AsyncMock]:
    """Construct an engine with embedder + temporal store stubs.

    The engine's ``__init__`` only touches ``config`` to derive a storage
    config; it does no network I/O until ``connect()``. We bypass
    ``connect()`` and inject the two collaborators ``recall()`` actually
    needs (``_embedder`` and ``_temporal_store``) directly.
    """
    cfg = MagicMock()
    cfg.storage.backend = "pgvector"

    engine = SkeletonConstructionEngine.__new__(SkeletonConstructionEngine)
    engine._config = cfg
    engine._backend_type = "pgvector"
    engine._weaviate_url = None
    engine._storage_config = MagicMock()
    engine._storage = None
    engine._connected = True

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    engine._embedder = embedder

    temporal_store = AsyncMock()
    temporal_store.search = AsyncMock(return_value=[])
    engine._temporal_store = temporal_store

    return engine, temporal_store


@pytest.mark.parametrize(
    ("mode", "expected_alpha"),
    [
        (SearchMode.VECTOR, 1.0),
        (SearchMode.KEYWORD, 0.0),
        (SearchMode.HYBRID, 0.7),
        (SearchMode.GRAPH, 0.7),  # Falls into the else branch (treated as HYBRID).
        (SearchMode.ALL, 0.7),  # Same.
    ],
)
async def test_recall_resolves_hybrid_alpha_per_mode(mode: SearchMode, expected_alpha: float) -> None:
    """Every SearchMode value must resolve a ``hybrid_alpha`` without crashing.

    Previously, ``mode=HYBRID`` (the Khora default) raised
    ``AttributeError: SearchMode has no attribute 'KEYWORD'`` because the
    ``elif`` RHS was evaluated whenever the ``if`` branch was False.
    """
    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    result = await engine.recall("alpha", namespace_id, mode=mode)

    assert result.metadata["hybrid_alpha"] == expected_alpha
    temporal_store.search.assert_awaited_once()
    forwarded = temporal_store.search.await_args.kwargs["hybrid_alpha"]
    assert forwarded == expected_alpha


async def test_recall_explicit_hybrid_alpha_overrides_mode_default() -> None:
    """An explicit ``hybrid_alpha`` short-circuits the mode-based defaulting.

    This is the path the integration test
    ``test_skeleton_recall_with_metadata_filter`` relies on.
    """
    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    result = await engine.recall("alpha", namespace_id, mode=SearchMode.HYBRID, hybrid_alpha=0.25)

    assert result.metadata["hybrid_alpha"] == 0.25
    forwarded = temporal_store.search.await_args.kwargs["hybrid_alpha"]
    assert forwarded == 0.25
