"""Unit tests for the temporal-recency additions to PgVectorBackend.

Covers two Phase-A changes for issue #567:

* ``search_recent_chunks`` — pure recency-sorted parallel channel for RECENCY
  / CHANGE queries. Returns ``(chunk, None)`` tuples so RRF callers can detect
  the absence of a cosine score.
* ``_probe_iterative_scan_supported`` — one-time capability probe for the
  pgvector >= 0.8 ``hnsw.iterative_scan`` setting. Cached on the instance,
  swallows errors from older pgvector builds.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.storage.backends.pgvector import PgVectorBackend


def _make_backend() -> PgVectorBackend:
    """Construct a backend without touching the network / SQLAlchemy."""
    return PgVectorBackend.__new__(PgVectorBackend)


def _make_chunk_model(*, idx: int) -> SimpleNamespace:
    """Stand-in for a SQLAlchemy ChunkModel row."""
    now = datetime(2026, 5, 1, 12, 0, idx, tzinfo=UTC)
    return SimpleNamespace(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=f"chunk-{idx}",
        chunk_index=idx,
        start_char=0,
        end_char=10,
        token_count=2,
        metadata_={},
        chunker_info={},
        embedding=None,
        embedding_model="test-model",
        created_at=now,
        source_timestamp=None,
    )


def _patch_session_with_query_capture(backend: PgVectorBackend, models: list) -> dict:
    """Stub ``_get_session`` so ``session.execute(<select>)`` is captured.

    Returns a dict with two keys:
        * ``execute``: the AsyncMock standing in for session.execute
        * ``scalars_models``: the list returned by ``result.scalars().all()``

    The mocked execute returns a result whose ``.scalars().all()`` yields
    *models*. Callers can inspect ``execute.call_args_list`` to assert the
    SQL shape on the SQLAlchemy Select object.
    """
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=models)

    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars)

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)

    cm = AsyncMock()
    cm.__aenter__.return_value = session
    cm.__aexit__.return_value = False
    backend._get_session = MagicMock(return_value=cm)  # type: ignore[attr-defined]
    return {"execute": session.execute, "scalars_models": models}


# ---------------------------------------------------------------------------
# search_recent_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_recent_chunks_returns_none_similarity() -> None:
    """Each returned tuple has ``None`` in the similarity slot — the recency
    channel does not have a cosine score."""
    backend = _make_backend()
    models = [_make_chunk_model(idx=i) for i in range(3)]
    _patch_session_with_query_capture(backend, models)

    results = await backend.search_recent_chunks(uuid4(), limit=5)

    assert len(results) == 3
    for _chunk, sim in results:
        assert sim is None


@pytest.mark.asyncio
async def test_search_recent_chunks_uses_coalesce_desc_order_and_limit() -> None:
    """The compiled SQL must ORDER BY COALESCE(source_timestamp, created_at) DESC
    and apply LIMIT — that's the whole contract of this method."""
    backend = _make_backend()
    _patch_session_with_query_capture(backend, [])

    await backend.search_recent_chunks(uuid4(), limit=7)

    # First and only execute() call should be the SELECT
    execute_mock = backend._get_session().__aenter__.return_value.execute  # type: ignore[attr-defined]
    assert execute_mock.await_count == 1
    select_obj = execute_mock.await_args.args[0]

    compiled = str(select_obj.compile(compile_kwargs={"literal_binds": True}))
    # ORDER BY uses COALESCE expression, DESC, and the LIMIT is honored.
    assert "coalesce" in compiled.lower()
    assert "source_timestamp" in compiled.lower()
    assert "created_at" in compiled.lower()
    assert "desc" in compiled.lower()
    assert "limit 7" in compiled.lower()


@pytest.mark.asyncio
async def test_search_recent_chunks_created_after_is_optional() -> None:
    """When ``created_after`` is None, the WHERE has no temporal lower bound."""
    backend = _make_backend()
    _patch_session_with_query_capture(backend, [])

    await backend.search_recent_chunks(uuid4(), limit=5, created_after=None)

    execute_mock = backend._get_session().__aenter__.return_value.execute  # type: ignore[attr-defined]
    select_obj = execute_mock.await_args.args[0]
    compiled = str(select_obj.compile(compile_kwargs={"literal_binds": True})).lower()

    # No >= comparison should be emitted when created_after is None. The only
    # WHERE predicate is namespace_id =.
    assert ">=" not in compiled


@pytest.mark.asyncio
async def test_search_recent_chunks_applies_created_after_floor() -> None:
    """When ``created_after`` is provided, a WHERE COALESCE(...) >= bound is emitted."""
    backend = _make_backend()
    _patch_session_with_query_capture(backend, [])

    floor = datetime(2026, 5, 1, tzinfo=UTC)
    await backend.search_recent_chunks(uuid4(), limit=5, created_after=floor)

    execute_mock = backend._get_session().__aenter__.return_value.execute  # type: ignore[attr-defined]
    select_obj = execute_mock.await_args.args[0]
    compiled = str(select_obj.compile(compile_kwargs={"literal_binds": True})).lower()

    assert ">=" in compiled
    # The bound expression is COALESCE(source_timestamp, created_at) — assert
    # both columns appear in the WHERE alongside the comparison.
    assert "coalesce" in compiled


@pytest.mark.asyncio
async def test_search_recent_chunks_empty_namespace_returns_empty_list() -> None:
    backend = _make_backend()
    _patch_session_with_query_capture(backend, [])

    results = await backend.search_recent_chunks(uuid4(), limit=10)

    assert results == []


# ---------------------------------------------------------------------------
# _probe_iterative_scan_supported
# ---------------------------------------------------------------------------


def _patch_engine_connect(backend: PgVectorBackend, *, raise_exc: Exception | None = None) -> AsyncMock:
    """Patch ``backend._engine.connect()`` to a context-manager that either
    returns a working conn (whose ``execute`` succeeds) or raises
    *raise_exc* from ``conn.execute``."""
    conn = AsyncMock()
    if raise_exc is None:
        result = MagicMock()
        result.scalar = MagicMock(return_value="off")
        conn.execute = AsyncMock(return_value=result)
    else:
        conn.execute = AsyncMock(side_effect=raise_exc)

    cm = AsyncMock()
    cm.__aenter__.return_value = conn
    cm.__aexit__.return_value = False

    engine = MagicMock()
    engine.connect = MagicMock(return_value=cm)
    backend._engine = engine  # type: ignore[attr-defined]
    return conn.execute


@pytest.mark.asyncio
async def test_probe_iterative_scan_supported_true_when_show_succeeds() -> None:
    backend = _make_backend()
    _patch_engine_connect(backend)

    assert await backend._probe_iterative_scan_supported() is True


@pytest.mark.asyncio
async def test_probe_iterative_scan_supported_false_when_show_raises() -> None:
    """pgvector < 0.8 has no ``hnsw.iterative_scan`` GUC — SHOW raises."""
    backend = _make_backend()
    _patch_engine_connect(backend, raise_exc=RuntimeError('unrecognized configuration parameter "hnsw.iterative_scan"'))

    assert await backend._probe_iterative_scan_supported() is False


@pytest.mark.asyncio
async def test_probe_iterative_scan_supported_caches_result() -> None:
    """Probe runs exactly once per backend instance — second call must not
    touch the engine."""
    backend = _make_backend()
    execute_mock = _patch_engine_connect(backend)

    first = await backend._probe_iterative_scan_supported()
    second = await backend._probe_iterative_scan_supported()

    assert first is True
    assert second is True
    assert execute_mock.await_count == 1


@pytest.mark.asyncio
async def test_probe_iterative_scan_supported_caches_false_too() -> None:
    """Even when the probe decides ``False``, it must cache the result so
    every search_similar call doesn't keep paying the SHOW round-trip."""
    backend = _make_backend()
    execute_mock = _patch_engine_connect(backend, raise_exc=RuntimeError("bad GUC"))

    first = await backend._probe_iterative_scan_supported()
    second = await backend._probe_iterative_scan_supported()

    assert first is False
    assert second is False
    assert execute_mock.await_count == 1


@pytest.mark.asyncio
async def test_probe_iterative_scan_returns_false_when_engine_none() -> None:
    """If ``connect()`` was never called, the probe must not crash."""
    backend = _make_backend()
    backend._engine = None  # type: ignore[attr-defined]

    assert await backend._probe_iterative_scan_supported() is False
