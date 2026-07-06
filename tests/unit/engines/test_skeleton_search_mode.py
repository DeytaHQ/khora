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
    ],
)
async def test_recall_resolves_hybrid_alpha_per_mode(mode: SearchMode, expected_alpha: float) -> None:
    """Every supported SearchMode resolves a ``hybrid_alpha`` without crashing.

    Skeleton's supported_modes is ``{VECTOR, HYBRID, KEYWORD}`` (#833).
    GRAPH and ALL raise ``EngineCapabilityError`` - covered in the
    dedicated test below.
    """
    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    result = await engine.recall("alpha", namespace_id, mode=mode)

    assert result.engine_info["hybrid_alpha"] == expected_alpha
    temporal_store.search.assert_awaited_once()
    forwarded = temporal_store.search.await_args.kwargs["hybrid_alpha"]
    assert forwarded == expected_alpha


@pytest.mark.parametrize("mode", [SearchMode.GRAPH, SearchMode.ALL])
async def test_recall_unsupported_mode_raises(mode: SearchMode) -> None:
    """#833: Skeleton refuses GRAPH (no graph backend) and ALL (no extra
    channels beyond HYBRID). Both raise ``EngineCapabilityError``."""
    from khora.exceptions import EngineCapabilityError

    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    with pytest.raises(EngineCapabilityError) as excinfo:
        await engine.recall("alpha", namespace_id, mode=mode)
    assert excinfo.value.engine_name == "skeleton"
    assert excinfo.value.mode is mode
    # Storage was never touched.
    temporal_store.search.assert_not_awaited()


async def test_recall_explicit_hybrid_alpha_overrides_mode_default() -> None:
    """An explicit ``hybrid_alpha`` short-circuits the mode-based defaulting.

    This is the path the integration test
    ``test_skeleton_recall_with_metadata_filter`` relies on.
    """
    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    result = await engine.recall("alpha", namespace_id, mode=SearchMode.HYBRID, hybrid_alpha=0.25)

    assert result.engine_info["hybrid_alpha"] == 0.25
    forwarded = temporal_store.search.await_args.kwargs["hybrid_alpha"]
    assert forwarded == 0.25


async def test_recall_chunk_scores_are_min_max_normalized() -> None:
    """#834: ``RecallChunk.score`` must be a min-max normalized rank in [0, 1].

    Skeleton previously surfaced ``result.combined_score or result.similarity``
    on an arbitrary scale (e.g. 0.013-0.015 for raw cosine). The unified
    contract: top chunk = 1.0, bottom chunk = 0.0 when 2+ chunks are returned.
    """
    from khora.storage.temporal import TemporalChunk, TemporalSearchResult

    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    raw_scores = [0.01335, 0.01317, 0.01299]
    results = [
        TemporalSearchResult(
            chunk=TemporalChunk(
                id=uuid4(),
                namespace_id=namespace_id,
                document_id=uuid4(),
                content=f"chunk-{i}",
            ),
            similarity=score,
        )
        for i, score in enumerate(raw_scores)
    ]
    temporal_store.search = AsyncMock(return_value=results)

    recall_result = await engine.recall("alpha", namespace_id, mode=SearchMode.VECTOR)

    assert len(recall_result.chunks) == 3
    assert recall_result.chunks[0].score == 1.0
    assert recall_result.chunks[-1].score == 0.0
    # Middle chunk lands strictly between.
    assert 0.0 < recall_result.chunks[1].score < 1.0


async def test_recall_chunk_scores_single_chunk_is_one() -> None:
    """Edge case: a single chunk gets score=1.0 (degenerate min-max)."""
    from khora.storage.temporal import TemporalChunk, TemporalSearchResult

    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    results = [
        TemporalSearchResult(
            chunk=TemporalChunk(
                id=uuid4(),
                namespace_id=namespace_id,
                document_id=uuid4(),
                content="solo",
            ),
            similarity=0.42,
        )
    ]
    temporal_store.search = AsyncMock(return_value=results)

    recall_result = await engine.recall("alpha", namespace_id, mode=SearchMode.VECTOR)

    assert len(recall_result.chunks) == 1
    assert recall_result.chunks[0].score == 1.0


async def test_recall_chunk_scores_all_tied_collapse_to_one() -> None:
    """Edge case: when max == min, every chunk collapses to 1.0."""
    from khora.storage.temporal import TemporalChunk, TemporalSearchResult

    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    results = [
        TemporalSearchResult(
            chunk=TemporalChunk(
                id=uuid4(),
                namespace_id=namespace_id,
                document_id=uuid4(),
                content=f"tied-{i}",
            ),
            similarity=0.5,
        )
        for i in range(3)
    ]
    temporal_store.search = AsyncMock(return_value=results)

    recall_result = await engine.recall("alpha", namespace_id, mode=SearchMode.VECTOR)

    assert [c.score for c in recall_result.chunks] == [1.0, 1.0, 1.0]
