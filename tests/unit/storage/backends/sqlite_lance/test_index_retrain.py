"""Tests for IVF-PQ index retraining on corpus growth (DYT-3580).

The vector adapter trains the LanceDB ANN index lazily on first search
once the table has at least ``_ANN_INDEX_THRESHOLD`` (5000) rows. Before
DYT-3580 the ``_chunks_indexed`` flag became sticky and the index was
never rebuilt, so a long-running embedded process that ingested 50k
chunks queried against an index trained on the first 5k.

These tests cover:

1. First search past the threshold trains the index inline; the row count
   at training is recorded.
2. After the corpus has grown by ``retrain_factor`` (default 2x), the next
   search schedules a background retrain.
3. Sub-factor growth does NOT trigger a retrain.
4. ``retrain_factor`` is configurable.

We mock LanceDB's ``count_rows`` and ``create_index`` so the tests don't
pay real training cost or need 5000-row tables on disk.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import lancedb  # noqa: F401
    import pyarrow  # noqa: F401

    _HAS_LANCEDB = True
except ImportError:
    _HAS_LANCEDB = False

pytestmark = pytest.mark.skipif(not _HAS_LANCEDB, reason="lancedb not installed")


if _HAS_LANCEDB:
    from khora.db.session import run_migrations
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.vector import SQLiteLanceVectorAdapter


async def _build_adapter(
    tmp_path: Path, *, retrain_factor: float = 2.0
) -> tuple[EmbeddedStorageHandle, SQLiteLanceVectorAdapter]:
    db_path = tmp_path / "k.db"
    lance_path = tmp_path / "k.lance"
    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    assert result.success, result.error
    cfg = EmbeddedStorageHandleConfig(
        db_path=str(db_path),
        lance_path=str(lance_path),
        embedding_dimension=8,
        retrain_factor=retrain_factor,
    )
    h = EmbeddedStorageHandle(cfg)
    a = SQLiteLanceVectorAdapter(h)
    await a.connect()
    return h, a


def _fake_table(row_count: int) -> MagicMock:
    """Build a stand-in for the LanceDB AsyncTable.

    Only ``count_rows`` and ``create_index`` are exercised by the retrain
    path; we don't need a real Arrow schema.
    """
    tbl = MagicMock()
    tbl.count_rows = AsyncMock(return_value=row_count)
    tbl.create_index = AsyncMock(return_value=None)
    return tbl


async def _wait_for_retrain(adapter: SQLiteLanceVectorAdapter) -> None:
    task = adapter._chunks_retrain_task
    if task is not None:
        await task


class TestRetrainTrigger:
    async def test_initial_train_at_threshold(self, tmp_path: Path):
        """Before any training, hitting the search path with >= 5000 rows
        trains the index inline and records the row count."""
        handle, adapter = await _build_adapter(tmp_path)
        try:
            tbl = _fake_table(row_count=5_000)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()

            assert adapter._chunks_at_last_index == 5_000
            assert tbl.create_index.await_count == 1
        finally:
            await adapter.disconnect()

    async def test_retrain_when_corpus_doubles(self, tmp_path: Path):
        """After initial train at 5k, growing to 10k triggers a retrain."""
        handle, adapter = await _build_adapter(tmp_path)
        try:
            tbl = _fake_table(row_count=5_000)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
            assert adapter._chunks_at_last_index == 5_000
            initial_calls = tbl.create_index.await_count

            # Corpus has now doubled. Trigger second check.
            tbl.count_rows = AsyncMock(return_value=10_000)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
                # Background task — wait for it to finish.
                await _wait_for_retrain(adapter)

            assert adapter._chunks_at_last_index == 10_000
            assert tbl.create_index.await_count == initial_calls + 1
        finally:
            await adapter.disconnect()

    async def test_no_retrain_below_factor(self, tmp_path: Path):
        """100 extra rows past 10k (total 10.1k) is far below the 2x
        threshold (would need 20k) — no retrain."""
        handle, adapter = await _build_adapter(tmp_path)
        try:
            tbl = _fake_table(row_count=10_000)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
            assert adapter._chunks_at_last_index == 10_000
            initial_calls = tbl.create_index.await_count

            tbl.count_rows = AsyncMock(return_value=10_100)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
                await _wait_for_retrain(adapter)

            assert adapter._chunks_at_last_index == 10_000  # unchanged
            assert tbl.create_index.await_count == initial_calls  # no retrain
        finally:
            await adapter.disconnect()

    async def test_configurable_retrain_factor(self, tmp_path: Path):
        """retrain_factor=1.5 triggers retrain at 7.5k rows."""
        handle, adapter = await _build_adapter(tmp_path, retrain_factor=1.5)
        try:
            tbl = _fake_table(row_count=5_000)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
            assert adapter._chunks_at_last_index == 5_000
            initial_calls = tbl.create_index.await_count

            # 7.4k — still below 1.5x = 7.5k.
            tbl.count_rows = AsyncMock(return_value=7_400)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
                await _wait_for_retrain(adapter)
            assert adapter._chunks_at_last_index == 5_000
            assert tbl.create_index.await_count == initial_calls

            # 7.5k — at the threshold, retrain.
            tbl.count_rows = AsyncMock(return_value=7_500)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
                await _wait_for_retrain(adapter)
            assert adapter._chunks_at_last_index == 7_500
            assert tbl.create_index.await_count == initial_calls + 1
        finally:
            await adapter.disconnect()

    async def test_retrain_disabled_when_factor_le_one(self, tmp_path: Path):
        """retrain_factor=1.0 disables retraining entirely."""
        handle, adapter = await _build_adapter(tmp_path, retrain_factor=1.0)
        try:
            tbl = _fake_table(row_count=5_000)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
            initial_calls = tbl.create_index.await_count

            # Even at 100x growth, no retrain.
            tbl.count_rows = AsyncMock(return_value=500_000)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()
                await _wait_for_retrain(adapter)

            assert tbl.create_index.await_count == initial_calls
        finally:
            await adapter.disconnect()

    async def test_no_train_below_threshold(self, tmp_path: Path):
        """Below 5000 rows in auto/ivf_pq mode, no index is built and the
        marker stays unset so the next search reconsiders."""
        handle, adapter = await _build_adapter(tmp_path)
        try:
            tbl = _fake_table(row_count=4_999)
            with patch.object(adapter, "_chunks_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_chunks_index()

            assert adapter._chunks_at_last_index is None
            assert tbl.create_index.await_count == 0
        finally:
            await adapter.disconnect()

    async def test_entities_index_retrains_too(self, tmp_path: Path):
        """The same retrain logic applies to entities_vec — it's the same
        bug pattern. Doubled corpus triggers a rebuild."""
        handle, adapter = await _build_adapter(tmp_path)
        try:
            tbl = _fake_table(row_count=5_000)
            with patch.object(adapter, "_entities_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_entities_index()
            assert adapter._entities_at_last_index == 5_000
            initial_calls = tbl.create_index.await_count

            tbl.count_rows = AsyncMock(return_value=10_000)
            with patch.object(adapter, "_entities_table", AsyncMock(return_value=tbl)):
                await adapter._maybe_build_entities_index()
                task = adapter._entities_retrain_task
                if task is not None:
                    await task

            assert adapter._entities_at_last_index == 10_000
            assert tbl.create_index.await_count == initial_calls + 1
        finally:
            await adapter.disconnect()
