"""Integration tests for three sqlite_lance + Skeleton temporal fixes.

#1068 - UTC-normalize occurred_at before storing so tz-aware timestamps
        match UTC-normalized recall bounds in lexicographic TEXT comparison.
#1070 - stats(namespace).chunks returns the actual khora_chunks count,
        not the always-zero relational chunks count.
#1182 - search_recent_chunks populates embeddings from LanceDB and returns
        recency-ordered (Chunk, None) tuples.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.temporal import ChunkTemporalFilter, TemporalChunk
from tests.integration._sqlite_lance_fixtures import (
    EMBED_DIM,
    build_sqlite_lance_coordinator,
    fake_embedding,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IST = timezone(timedelta(hours=5, minutes=30))  # UTC+5:30 (India Standard Time)
_EST = timezone(timedelta(hours=-5))  # UTC-5 (Eastern Standard Time)


def _make_temporal_chunk(
    namespace_id: UUID,
    *,
    content: str,
    occurred_at: datetime,
) -> TemporalChunk:
    from uuid import uuid4

    doc_id = uuid4()
    return TemporalChunk(
        id=None,  # type: ignore[arg-type]
        namespace_id=namespace_id,
        document_id=doc_id,
        content=content,
        embedding=fake_embedding(content),
        occurred_at=occurred_at,
        created_at=datetime.now(UTC),
    )


async def _build_temporal_store(coord):
    """Extract the SQLiteLanceTemporalStore from a coordinator built with sqlite_lance."""
    from khora.config import KhoraConfig
    from khora.storage.temporal.sqlite_lance import SQLiteLanceTemporalStore

    cfg = KhoraConfig()
    cfg.storage.backend = "sqlite_lance"
    cfg.llm.embedding_dimension = EMBED_DIM

    # Build the store via the coordinator factory so it shares the handle.
    store = await coord.temporal_store("sqlite_lance", cfg)
    assert isinstance(store, SQLiteLanceTemporalStore)
    return store


# ---------------------------------------------------------------------------
# #1068 — UTC normalization of occurred_at
# ---------------------------------------------------------------------------


class TestUtcNormalization:
    """#1068: tz-aware occurred_at is UTC-normalized at write time."""

    async def test_in_window_tz_aware_ist_chunk_returned(self, tmp_path: Path) -> None:
        """A chunk whose occurred_at is in-window by instant is returned even
        though its raw offset string sorts differently than the UTC bound."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            from khora.core.models import MemoryNamespace

            ns = await coord.create_namespace(MemoryNamespace())
            store = await _build_temporal_store(coord)

            # Event happened at 2024-06-05 15:30:00 UTC, expressed in IST (+05:30)
            # which renders as "2024-06-05T21:00:00+05:30".  The raw string prefix
            # "21:00" > "15:00" (the UTC bound), so the old broken path would drop
            # the row from a lexicographic filter("...+00:00" boundary).
            occurred_ist = datetime(2024, 6, 5, 21, 0, 0, tzinfo=_IST)
            chunk = _make_temporal_chunk(ns.id, content="in-window IST event", occurred_at=occurred_ist)
            await store.create_chunk(chunk)

            # Recall window: 2024-06-05 00:00 UTC … 2024-06-06 00:00 UTC (includes the event)
            window_start = datetime(2024, 6, 5, 0, 0, 0, tzinfo=UTC)
            window_end = datetime(2024, 6, 6, 0, 0, 0, tzinfo=UTC)
            results = await store.search(
                ns.id,
                fake_embedding("in-window IST event"),
                limit=10,
                temporal_filter=ChunkTemporalFilter(
                    occurred_after=window_start,
                    occurred_before=window_end,
                ),
            )
            assert len(results) == 1, "The IST chunk is within the UTC window by instant — it must be returned"
            assert results[0].chunk.content == "in-window IST event"

        finally:
            with contextlib.suppress(Exception):
                await store.disconnect()
            with contextlib.suppress(Exception):
                await coord.disconnect()

    async def test_out_of_window_chunk_excluded(self, tmp_path: Path) -> None:
        """A chunk outside the window is excluded regardless of its timezone offset."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            from khora.core.models import MemoryNamespace

            ns = await coord.create_namespace(MemoryNamespace())
            store = await _build_temporal_store(coord)

            # 2024-06-07 10:00 UTC — outside the window below.
            occurred_utc = datetime(2024, 6, 7, 10, 0, 0, tzinfo=UTC)
            chunk = _make_temporal_chunk(ns.id, content="out-of-window chunk", occurred_at=occurred_utc)
            await store.create_chunk(chunk)

            window_start = datetime(2024, 6, 5, 0, 0, 0, tzinfo=UTC)
            window_end = datetime(2024, 6, 6, 0, 0, 0, tzinfo=UTC)
            results = await store.search(
                ns.id,
                fake_embedding("out-of-window chunk"),
                limit=10,
                temporal_filter=ChunkTemporalFilter(
                    occurred_after=window_start,
                    occurred_before=window_end,
                ),
            )
            assert len(results) == 0, "Chunk outside window must be excluded"

        finally:
            with contextlib.suppress(Exception):
                await store.disconnect()
            with contextlib.suppress(Exception):
                await coord.disconnect()

    async def test_in_window_vs_out_of_window_with_different_offsets(self, tmp_path: Path) -> None:
        """Two chunks: one in-window (IST), one out-of-window (EST). Only the
        in-window chunk survives the filter."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            from khora.core.models import MemoryNamespace

            ns = await coord.create_namespace(MemoryNamespace())
            store = await _build_temporal_store(coord)

            # In window: 2024-06-05T15:00:00 UTC expressed in IST → T20:30+05:30
            in_window_ist = datetime(2024, 6, 5, 20, 30, 0, tzinfo=_IST)
            # Out of window: 2024-06-04T22:00:00 UTC expressed in EST → T17:00-05:00
            out_of_window_est = datetime(2024, 6, 4, 17, 0, 0, tzinfo=_EST)

            await store.create_chunks_batch(
                [
                    _make_temporal_chunk(ns.id, content="alpha in-window", occurred_at=in_window_ist),
                    _make_temporal_chunk(ns.id, content="beta out-of-window", occurred_at=out_of_window_est),
                ]
            )

            window_start = datetime(2024, 6, 5, 0, 0, 0, tzinfo=UTC)
            window_end = datetime(2024, 6, 6, 0, 0, 0, tzinfo=UTC)

            # Search for in-window chunk
            results = await store.search(
                ns.id,
                fake_embedding("alpha in-window"),
                limit=10,
                temporal_filter=ChunkTemporalFilter(
                    occurred_after=window_start,
                    occurred_before=window_end,
                ),
            )
            contents = {r.chunk.content for r in results}
            assert "alpha in-window" in contents, "In-window IST chunk must be returned"
            assert "beta out-of-window" not in contents, "Out-of-window EST chunk must be excluded"

        finally:
            with contextlib.suppress(Exception):
                await store.disconnect()
            with contextlib.suppress(Exception):
                await coord.disconnect()


# ---------------------------------------------------------------------------
# #1070 — stats(namespace).chunks counts khora_chunks
# ---------------------------------------------------------------------------


class TestStatsChunkCount:
    """#1070: stats(namespace).chunks returns the actual khora_chunks count."""

    async def test_stats_chunks_nonzero_on_sqlite_lance_temporal(self, tmp_path: Path) -> None:
        """Chunks written to khora_chunks via the temporal store are counted
        by SQLiteLanceTemporalStore.count_chunks."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            from khora.core.models import MemoryNamespace

            ns = await coord.create_namespace(MemoryNamespace())
            store = await _build_temporal_store(coord)

            now = datetime.now(UTC)
            await store.create_chunks_batch(
                [
                    _make_temporal_chunk(ns.id, content="chunk one", occurred_at=now),
                    _make_temporal_chunk(ns.id, content="chunk two", occurred_at=now),
                ]
            )

            count = await store.count_chunks(ns.id)
            assert count == 2, f"count_chunks should be 2, got {count}"

        finally:
            with contextlib.suppress(Exception):
                await store.disconnect()
            with contextlib.suppress(Exception):
                await coord.disconnect()


# ---------------------------------------------------------------------------
# #1182 — search_recent_chunks on sqlite_lance
# ---------------------------------------------------------------------------


class TestSearchRecentChunks:
    """#1182: search_recent_chunks returns embedding-bearing, recency-ordered chunks."""

    async def test_search_recent_chunks_returns_embedding_bearing_tuples(self, tmp_path: Path) -> None:
        """search_recent_chunks populates chunk.embedding from LanceDB."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            from khora.core.models import MemoryNamespace

            ns = await coord.create_namespace(MemoryNamespace())
            store = await _build_temporal_store(coord)

            now = datetime.now(UTC)
            await store.create_chunks_batch(
                [
                    _make_temporal_chunk(ns.id, content="recent alpha", occurred_at=now),
                    _make_temporal_chunk(ns.id, content="older beta", occurred_at=now - timedelta(hours=2)),
                ]
            )

            results = await store.search_recent_chunks(ns.id, limit=10)

            assert len(results) == 2, f"Expected 2 results, got {len(results)}"
            # Protocol shape: (Chunk, None)
            for chunk, score in results:
                assert score is None, "search_recent_chunks returns (Chunk, None) tuples"
                # Embedding is populated so the cosine gate in the caller works.
                assert chunk.embedding is not None, f"Chunk {chunk.id} is missing embedding"
                assert len(chunk.embedding) == EMBED_DIM

        finally:
            with contextlib.suppress(Exception):
                await store.disconnect()
            with contextlib.suppress(Exception):
                await coord.disconnect()

    async def test_search_recent_chunks_recency_order(self, tmp_path: Path) -> None:
        """search_recent_chunks returns chunks in descending recency order."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            from khora.core.models import MemoryNamespace

            ns = await coord.create_namespace(MemoryNamespace())
            store = await _build_temporal_store(coord)

            base = datetime(2024, 6, 10, 12, 0, 0, tzinfo=UTC)
            # Insert out-of-order to confirm ordering is by occurred_at, not insert order.
            await store.create_chunks_batch(
                [
                    _make_temporal_chunk(ns.id, content="oldest", occurred_at=base - timedelta(hours=4)),
                    _make_temporal_chunk(ns.id, content="newest", occurred_at=base),
                    _make_temporal_chunk(ns.id, content="middle", occurred_at=base - timedelta(hours=2)),
                ]
            )

            results = await store.search_recent_chunks(ns.id, limit=10)
            contents = [chunk.content for chunk, _ in results]
            assert contents == ["newest", "middle", "oldest"], (
                f"Expected recency order [newest, middle, oldest], got {contents}"
            )

        finally:
            with contextlib.suppress(Exception):
                await store.disconnect()
            with contextlib.suppress(Exception):
                await coord.disconnect()

    async def test_search_recent_chunks_created_after_filter(self, tmp_path: Path) -> None:
        """created_after narrows results to chunks at or after the bound."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            from khora.core.models import MemoryNamespace

            ns = await coord.create_namespace(MemoryNamespace())
            store = await _build_temporal_store(coord)

            base = datetime(2024, 6, 10, 12, 0, 0, tzinfo=UTC)
            await store.create_chunks_batch(
                [
                    _make_temporal_chunk(ns.id, content="before cutoff", occurred_at=base - timedelta(hours=3)),
                    _make_temporal_chunk(ns.id, content="at cutoff", occurred_at=base),
                    _make_temporal_chunk(ns.id, content="after cutoff", occurred_at=base + timedelta(hours=1)),
                ]
            )

            results = await store.search_recent_chunks(ns.id, limit=10, created_after=base)
            contents = {chunk.content for chunk, _ in results}
            assert "before cutoff" not in contents, "Chunk before cutoff must be excluded"
            assert "at cutoff" in contents, "Chunk at cutoff must be included"
            assert "after cutoff" in contents, "Chunk after cutoff must be included"

        finally:
            with contextlib.suppress(Exception):
                await store.disconnect()
            with contextlib.suppress(Exception):
                await coord.disconnect()

    async def test_search_recent_chunks_limit_respected(self, tmp_path: Path) -> None:
        """search_recent_chunks respects the limit parameter."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            from khora.core.models import MemoryNamespace

            ns = await coord.create_namespace(MemoryNamespace())
            store = await _build_temporal_store(coord)

            base = datetime.now(UTC)
            await store.create_chunks_batch(
                [
                    _make_temporal_chunk(ns.id, content=f"chunk-{i}", occurred_at=base - timedelta(hours=i))
                    for i in range(5)
                ]
            )

            results = await store.search_recent_chunks(ns.id, limit=3)
            assert len(results) == 3, f"limit=3 must return exactly 3 results, got {len(results)}"

        finally:
            with contextlib.suppress(Exception):
                await store.disconnect()
            with contextlib.suppress(Exception):
                await coord.disconnect()
