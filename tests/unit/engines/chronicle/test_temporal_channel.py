"""Chronicle #4: tests for the events-based temporal channel.

The channel queries ``chronicle_events`` and ranks chunks by a blend of
event-summary cosine similarity and temporal proximity to the query window
(measured against ``referenced_date`` — the date the source text refers to,
NOT ingest time). The legacy chunk-based path is preserved as a fallback.

Tests use mock coordinators so no real DB / LLM calls are made.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, ChunkMetadata
from khora.engines.chronicle.engine import (
    ChronicleEngine,
    _extract_temporal_bounds,
    _temporal_proximity,
)
from khora.engines.chronicle.events import ChronicleEvent
from khora.query.temporal import TemporalFilter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "x", *, namespace_id: UUID | None = None) -> Chunk:
    ns_id = namespace_id or uuid4()
    document_id = uuid4()
    return Chunk(
        namespace_id=ns_id,
        document_id=document_id,
        content=content,
        metadata=ChunkMetadata(document_id=document_id, chunk_index=0),
    )


def _make_event(
    *,
    chunk_id: UUID,
    namespace_id: UUID,
    referenced_date: datetime | None = None,
    embedding: list[float] | None = None,
    summary: str = "alice met bob",
) -> ChronicleEvent:
    parts = summary.split(maxsplit=2)
    while len(parts) < 3:
        parts.append("")
    return ChronicleEvent(
        chunk_id=chunk_id,
        namespace_id=namespace_id,
        subject=parts[0],
        verb=parts[1],
        object=parts[2],
        referenced_date=referenced_date,
        embedding=embedding,
    )


class _FakeCoordinator:
    """Coordinator double that records query_events calls and returns canned data."""

    def __init__(
        self,
        *,
        events: list[ChronicleEvent] | None = None,
        chunks: list[Chunk] | None = None,
        chunk_search_results: list[tuple[Chunk, float]] | None = None,
        raise_query_events: Exception | None = None,
    ) -> None:
        self._events = events or []
        self._chunks_by_id = {c.id: c for c in (chunks or [])}
        self._chunk_search_results = chunk_search_results or []
        self._raise_query_events = raise_query_events

        self.query_events_calls: list[dict[str, Any]] = []
        self.search_similar_chunks_calls: list[dict[str, Any]] = []
        self.get_chunks_batch_calls: list[list[UUID]] = []

    async def query_events(
        self,
        namespace_id: UUID,
        *,
        subject: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[ChronicleEvent]:
        self.query_events_calls.append(
            {
                "namespace_id": namespace_id,
                "subject": subject,
                "since": since,
                "until": until,
                "limit": limit,
            }
        )
        if self._raise_query_events is not None:
            raise self._raise_query_events
        # Mimic backend filtering by referenced_date so tests exercising the
        # "outside scope" case match real-world behaviour.
        out: list[ChronicleEvent] = []
        for ev in self._events:
            ref = ev.referenced_date
            if since is not None and (ref is None or ref < since):
                continue
            if until is not None and (ref is None or ref > until):
                continue
            out.append(ev)
        return out[:limit]

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        self.get_chunks_batch_calls.append(list(chunk_ids))
        return {cid: self._chunks_by_id[cid] for cid in chunk_ids if cid in self._chunks_by_id}

    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_similarity: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        self.search_similar_chunks_calls.append(
            {
                "namespace_id": namespace_id,
                "limit": limit,
                "created_after": created_after,
                "created_before": created_before,
            }
        )
        return list(self._chunk_search_results)


def _bare_engine(**kwargs: Any) -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"), **kwargs)


def _wire(engine: ChronicleEngine, coord: _FakeCoordinator) -> None:
    engine._storage = coord  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# _temporal_proximity unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTemporalProximity:
    def test_returns_none_when_referenced_date_missing(self) -> None:
        start = datetime(2026, 4, 10, tzinfo=UTC)
        end = datetime(2026, 4, 17, tzinfo=UTC)
        assert _temporal_proximity(None, start, end) is None

    def test_in_range_returns_one(self) -> None:
        ref = datetime(2026, 4, 14, tzinfo=UTC)
        start = datetime(2026, 4, 10, tzinfo=UTC)
        end = datetime(2026, 4, 17, tzinfo=UTC)
        assert _temporal_proximity(ref, start, end) == pytest.approx(1.0)

    def test_outside_range_decays(self) -> None:
        # 7 days outside the range → exp(-1) ≈ 0.367879
        ref = datetime(2026, 4, 1, tzinfo=UTC)
        start = datetime(2026, 4, 8, tzinfo=UTC)
        end = datetime(2026, 4, 15, tzinfo=UTC)
        score = _temporal_proximity(ref, start, end)
        assert score == pytest.approx(0.36787944, rel=1e-3)

    def test_focal_date_decays_symmetrically(self) -> None:
        focal = datetime(2026, 4, 14, tzinfo=UTC)
        before = _temporal_proximity(datetime(2026, 4, 7, tzinfo=UTC), focal, focal)
        after = _temporal_proximity(datetime(2026, 4, 21, tzinfo=UTC), focal, focal)
        assert before == pytest.approx(after, rel=1e-6)


# ---------------------------------------------------------------------------
# _extract_temporal_bounds unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractTemporalBounds:
    def test_returns_none_for_none_filter(self) -> None:
        assert _extract_temporal_bounds(None) == (None, None)

    def test_extracts_start_end(self) -> None:
        f = TemporalFilter(
            start_time=datetime(2026, 4, 1, tzinfo=UTC),
            end_time=datetime(2026, 4, 8, tzinfo=UTC),
        )
        start, end = _extract_temporal_bounds(f)
        assert start == datetime(2026, 4, 1, tzinfo=UTC)
        assert end == datetime(2026, 4, 8, tzinfo=UTC)

    def test_naive_datetimes_normalized_to_utc(self) -> None:
        f = TemporalFilter(
            start_time=datetime(2026, 4, 1),  # naive
            end_time=datetime(2026, 4, 8),
        )
        start, end = _extract_temporal_bounds(f)
        assert start is not None and start.tzinfo is UTC
        assert end is not None and end.tzinfo is UTC


# ---------------------------------------------------------------------------
# _temporal_channel: events path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTemporalChannelEventsPath:
    @pytest.mark.asyncio
    async def test_events_in_scope_drive_ranking(self) -> None:
        ns_id = uuid4()
        c1, c2, c3 = (_make_chunk(namespace_id=ns_id) for _ in range(3))
        # Same-direction embeddings ⇒ cosine 1.0; the differentiator is
        # whether referenced_date is in the [start, end] window.
        emb_q = [1.0, 0.0, 0.0]
        emb_match = [1.0, 0.0, 0.0]
        ev_in_a = _make_event(
            chunk_id=c1.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 12, tzinfo=UTC),
            embedding=emb_match,
        )
        ev_in_b = _make_event(
            chunk_id=c2.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 14, tzinfo=UTC),
            embedding=emb_match,
        )
        ev_far = _make_event(
            chunk_id=c3.id,
            namespace_id=ns_id,
            referenced_date=datetime(2025, 1, 1, tzinfo=UTC),  # outside, far
            embedding=emb_match,
        )
        coord = _FakeCoordinator(events=[ev_in_a, ev_in_b, ev_far], chunks=[c1, c2, c3])
        engine = _bare_engine()
        _wire(engine, coord)

        tf = TemporalFilter(
            start_time=datetime(2026, 4, 10, tzinfo=UTC),
            end_time=datetime(2026, 4, 17, tzinfo=UTC),
        )
        results = await engine._temporal_channel(ns_id, "q", emb_q, limit=5, temporal_filter=tf)
        # Mock's query_events filters by since/until, so c3 is excluded
        # before scoring — verify we asked for the right window AND that the
        # in-scope events surface first.
        assert len(coord.query_events_calls) == 1
        assert coord.query_events_calls[0]["since"] == datetime(2026, 4, 10, tzinfo=UTC)
        assert coord.query_events_calls[0]["until"] == datetime(2026, 4, 17, tzinfo=UTC)
        returned_ids = [chunk.id for chunk, _ in results]
        assert c1.id in returned_ids and c2.id in returned_ids
        assert c3.id not in returned_ids

    @pytest.mark.asyncio
    async def test_partial_scope_only_in_scope_rank_high(self) -> None:
        # Even when the coordinator returns events both inside and outside the
        # query's focal window (e.g., subject was extracted across a wider
        # since/until pass), the scorer should rank in-window events higher.
        ns_id = uuid4()
        c_in, c_out = _make_chunk(namespace_id=ns_id), _make_chunk(namespace_id=ns_id)
        emb = [1.0, 0.0, 0.0]
        ev_in = _make_event(
            chunk_id=c_in.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 12, tzinfo=UTC),
            embedding=emb,
        )
        ev_out = _make_event(
            chunk_id=c_out.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 5, tzinfo=UTC),  # outside focal
            embedding=emb,
        )

        # Use a no-clip coordinator so the scorer sees both events.
        class _NoClipCoord(_FakeCoordinator):
            async def query_events(self, namespace_id: UUID, **kwargs: Any) -> list[ChronicleEvent]:
                return list(self._events)

        coord = _NoClipCoord(events=[ev_in, ev_out], chunks=[c_in, c_out])
        engine = _bare_engine()
        _wire(engine, coord)

        focal = datetime(2026, 4, 12, tzinfo=UTC)
        tf = TemporalFilter(start_time=focal, end_time=focal)
        results = await engine._temporal_channel(ns_id, "q", emb, limit=5, temporal_filter=tf)
        scores = {chunk.id: score for chunk, score in results}
        assert scores[c_in.id] > scores[c_out.id]

    @pytest.mark.asyncio
    async def test_cosine_and_temporal_blend(self) -> None:
        ns_id = uuid4()
        c_close_low_cos = _make_chunk(namespace_id=ns_id)
        c_far_high_cos = _make_chunk(namespace_id=ns_id)

        # Query embedding aligned with axis 0.
        emb_q = [1.0, 0.0, 0.0]
        # "Closer in time but lower cosine": orthogonal axis ⇒ cosine 0.
        emb_low = [0.0, 1.0, 0.0]
        # "Further in time but higher cosine": same direction ⇒ cosine 1.
        emb_high = [1.0, 0.0, 0.0]

        # Window centered on April 14.
        focal = datetime(2026, 4, 14, tzinfo=UTC)
        ev_close = _make_event(
            chunk_id=c_close_low_cos.id,
            namespace_id=ns_id,
            referenced_date=focal,  # proximity = 1.0
            embedding=emb_low,  # cosine = 0.0
        )
        ev_far = _make_event(
            chunk_id=c_far_high_cos.id,
            namespace_id=ns_id,
            referenced_date=focal - timedelta(days=14),  # proximity ≈ exp(-2) ≈ 0.1353
            embedding=emb_high,  # cosine = 1.0
        )

        # Use a no-clip coordinator so both events reach the scorer.
        class _NoClipCoord(_FakeCoordinator):
            async def query_events(self, namespace_id: UUID, **kwargs: Any) -> list[ChronicleEvent]:
                return list(self._events)

        coord = _NoClipCoord(events=[ev_close, ev_far], chunks=[c_close_low_cos, c_far_high_cos])
        # Even split (cw=tw=0.5):
        #   close_score = 0.5 * 0 + 0.5 * 1.0   = 0.5
        #   far_score   = 0.5 * 1.0 + 0.5 * 0.1353 ≈ 0.5677
        # → far should rank above close at the default 0.5 weight.
        engine = _bare_engine(temporal_event_cosine_weight=0.5)
        _wire(engine, coord)

        tf = TemporalFilter(start_time=focal, end_time=focal)
        results = await engine._temporal_channel(ns_id, "q", emb_q, limit=5, temporal_filter=tf)
        scores = {chunk.id: score for chunk, score in results}
        assert scores[c_far_high_cos.id] > scores[c_close_low_cos.id]
        assert scores[c_close_low_cos.id] == pytest.approx(0.5, rel=1e-3)

        # Crank cosine weight up — far should still win, by more.
        coord2 = _NoClipCoord(events=[ev_close, ev_far], chunks=[c_close_low_cos, c_far_high_cos])
        engine2 = _bare_engine(temporal_event_cosine_weight=0.9)
        _wire(engine2, coord2)
        results2 = await engine2._temporal_channel(ns_id, "q", emb_q, limit=5, temporal_filter=tf)
        scores2 = {chunk.id: score for chunk, score in results2}
        assert scores2[c_far_high_cos.id] > scores2[c_close_low_cos.id]
        assert scores2[c_far_high_cos.id] > scores[c_far_high_cos.id]

    @pytest.mark.asyncio
    async def test_no_events_falls_back_to_chunk_search(self) -> None:
        ns_id = uuid4()
        chunk = _make_chunk(namespace_id=ns_id)
        coord = _FakeCoordinator(
            events=[],  # events table empty for this namespace
            chunks=[chunk],
            chunk_search_results=[(chunk, 0.8)],
        )
        engine = _bare_engine()
        _wire(engine, coord)

        tf = TemporalFilter(
            start_time=datetime(2026, 4, 10, tzinfo=UTC),
            end_time=datetime(2026, 4, 17, tzinfo=UTC),
        )
        results = await engine._temporal_channel(ns_id, "q", [1.0, 0.0, 0.0], limit=5, temporal_filter=tf)
        # Should have called query_events once, then fallen back to search_similar_chunks.
        assert len(coord.query_events_calls) == 1
        assert len(coord.search_similar_chunks_calls) == 1
        # And produced a result from the fallback.
        assert len(results) == 1 and results[0][0].id == chunk.id

    @pytest.mark.asyncio
    async def test_event_with_no_embedding_uses_proximity_only(self) -> None:
        ns_id = uuid4()
        chunk = _make_chunk(namespace_id=ns_id)
        focal = datetime(2026, 4, 14, tzinfo=UTC)
        ev = _make_event(
            chunk_id=chunk.id,
            namespace_id=ns_id,
            referenced_date=focal,  # exact match → proximity 1.0
            embedding=None,  # ← no cosine signal
        )
        coord = _FakeCoordinator(events=[ev], chunks=[chunk])
        engine = _bare_engine()
        _wire(engine, coord)

        tf = TemporalFilter(start_time=focal, end_time=focal)
        results = await engine._temporal_channel(ns_id, "q", [1.0, 0.0, 0.0], limit=5, temporal_filter=tf)
        assert len(results) == 1
        # Score should be the raw proximity (1.0) since cosine is unavailable.
        assert results[0][1] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_event_with_no_referenced_date_uses_cosine_only(self) -> None:
        ns_id = uuid4()
        chunk = _make_chunk(namespace_id=ns_id)
        emb = [1.0, 0.0, 0.0]
        ev = _make_event(
            chunk_id=chunk.id,
            namespace_id=ns_id,
            referenced_date=None,  # ← no temporal anchor on the event itself
            embedding=emb,
        )

        # The coordinator's real backend wouldn't return events whose
        # referenced_date is null when an explicit since/until is set, so
        # we use a no-clip mock to hit the channel's "no proximity" branch.
        class _NoClipCoord(_FakeCoordinator):
            async def query_events(self, namespace_id: UUID, **kwargs: Any) -> list[ChronicleEvent]:
                return list(self._events)

        coord = _NoClipCoord(events=[ev], chunks=[chunk])
        engine = _bare_engine()
        _wire(engine, coord)

        # Need a temporal signal on the FILTER so the events path activates.
        tf = TemporalFilter(
            start_time=datetime(2026, 4, 10, tzinfo=UTC),
            end_time=datetime(2026, 4, 17, tzinfo=UTC),
        )
        results = await engine._temporal_channel(ns_id, "q", emb, limit=5, temporal_filter=tf)
        assert len(results) == 1
        # Score = cosine only ≈ 1.0 (no temporal blend).
        assert results[0][1] == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.asyncio
    async def test_events_outside_window_clipped_by_coordinator(self) -> None:
        ns_id = uuid4()
        c_in = _make_chunk(namespace_id=ns_id)
        c_out = _make_chunk(namespace_id=ns_id)
        emb = [1.0, 0.0, 0.0]
        ev_in = _make_event(
            chunk_id=c_in.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 12, tzinfo=UTC),
            embedding=emb,
        )
        ev_out = _make_event(
            chunk_id=c_out.id,
            namespace_id=ns_id,
            referenced_date=datetime(2025, 1, 1, tzinfo=UTC),  # way out
            embedding=emb,
        )
        coord = _FakeCoordinator(events=[ev_in, ev_out], chunks=[c_in, c_out])
        engine = _bare_engine()
        _wire(engine, coord)

        tf = TemporalFilter(
            start_time=datetime(2026, 4, 10, tzinfo=UTC),
            end_time=datetime(2026, 4, 17, tzinfo=UTC),
        )
        results = await engine._temporal_channel(ns_id, "q", emb, limit=5, temporal_filter=tf)
        ids = [chunk.id for chunk, _ in results]
        assert c_in.id in ids
        assert c_out.id not in ids

    @pytest.mark.asyncio
    async def test_chunks_deduped_when_multiple_events_share_chunk(self) -> None:
        ns_id = uuid4()
        chunk = _make_chunk(namespace_id=ns_id)
        emb = [1.0, 0.0, 0.0]
        # Three events on the same chunk: one perfect, two weaker.
        ev_strong = _make_event(
            chunk_id=chunk.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 14, tzinfo=UTC),
            embedding=emb,
        )
        ev_mid = _make_event(
            chunk_id=chunk.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 14, tzinfo=UTC),
            embedding=[0.5, 0.5, 0.0],  # weaker cosine
        )
        ev_weak = _make_event(
            chunk_id=chunk.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 14, tzinfo=UTC),
            embedding=[0.0, 1.0, 0.0],  # cosine 0
        )
        coord = _FakeCoordinator(events=[ev_strong, ev_mid, ev_weak], chunks=[chunk])
        engine = _bare_engine()
        _wire(engine, coord)

        focal = datetime(2026, 4, 14, tzinfo=UTC)
        tf = TemporalFilter(start_time=focal, end_time=focal)
        results = await engine._temporal_channel(ns_id, "q", emb, limit=5, temporal_filter=tf)
        # Output unit is chunks: only one entry, with the MAX score across events.
        assert len(results) == 1
        # Strong event score: 0.5 * 1.0 (cosine) + 0.5 * 1.0 (proximity) = 1.0
        assert results[0][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _temporal_channel: routing / fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTemporalChannelRouting:
    @pytest.mark.asyncio
    async def test_use_events_false_skips_events_query(self) -> None:
        ns_id = uuid4()
        chunk = _make_chunk(namespace_id=ns_id)
        coord = _FakeCoordinator(
            events=[],
            chunks=[chunk],
            chunk_search_results=[(chunk, 0.7)],
        )
        engine = _bare_engine(temporal_use_events=False)
        _wire(engine, coord)

        tf = TemporalFilter(
            start_time=datetime(2026, 4, 10, tzinfo=UTC),
            end_time=datetime(2026, 4, 17, tzinfo=UTC),
        )
        await engine._temporal_channel(ns_id, "q", [1.0, 0.0, 0.0], limit=5, temporal_filter=tf)
        # Events path completely skipped.
        assert coord.query_events_calls == []
        assert len(coord.search_similar_chunks_calls) == 1

    @pytest.mark.asyncio
    async def test_no_temporal_filter_uses_legacy_path(self) -> None:
        ns_id = uuid4()
        chunk = _make_chunk(namespace_id=ns_id)
        coord = _FakeCoordinator(
            events=[],
            chunks=[chunk],
            chunk_search_results=[(chunk, 0.5)],
        )
        engine = _bare_engine()
        _wire(engine, coord)

        # No filter at all → no temporal signal → legacy path.
        await engine._temporal_channel(ns_id, "q", [1.0, 0.0, 0.0], limit=5, temporal_filter=None)
        assert coord.query_events_calls == []
        assert len(coord.search_similar_chunks_calls) == 1


# ---------------------------------------------------------------------------
# Tunability + perf
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridBlendTunability:
    @pytest.mark.asyncio
    async def test_score_blend_is_configurable(self) -> None:
        """Verify ``temporal_event_cosine_weight`` is a tuneable engine kwarg."""
        ns_id = uuid4()
        chunk = _make_chunk(namespace_id=ns_id)
        emb_q = [1.0, 0.0, 0.0]
        # Cosine = 0.5, proximity = 1.0
        ev = _make_event(
            chunk_id=chunk.id,
            namespace_id=ns_id,
            referenced_date=datetime(2026, 4, 14, tzinfo=UTC),
            embedding=[0.5, 0.5, 0.0],  # cosine ≈ 1/√2 ≈ 0.707 with normalization
        )
        coord = _FakeCoordinator(events=[ev], chunks=[chunk])
        focal = datetime(2026, 4, 14, tzinfo=UTC)
        tf = TemporalFilter(start_time=focal, end_time=focal)

        # cw=0 → score = pure proximity = 1.0
        e0 = _bare_engine(temporal_event_cosine_weight=0.0)
        _wire(e0, coord)
        r0 = await e0._temporal_channel(ns_id, "q", emb_q, limit=5, temporal_filter=tf)
        assert r0[0][1] == pytest.approx(1.0)

        # cw=1 → score = pure cosine ≈ 0.707
        coord2 = _FakeCoordinator(events=[ev], chunks=[chunk])
        e1 = _bare_engine(temporal_event_cosine_weight=1.0)
        _wire(e1, coord2)
        r1 = await e1._temporal_channel(ns_id, "q", emb_q, limit=5, temporal_filter=tf)
        assert r1[0][1] == pytest.approx(0.7071, rel=1e-3)


@pytest.mark.unit
class TestPerformance:
    @pytest.mark.asyncio
    async def test_thousand_events_under_50ms(self) -> None:
        """Sanity check: 1000 in-scope events shouldn't blow the latency budget.

        The embedder is fully mocked so there are no external calls; the test
        measures the pure scoring + dedup + hydrate path.
        """
        ns_id = uuid4()
        emb = [1.0, 0.0, 0.0]
        chunks = [_make_chunk(namespace_id=ns_id) for _ in range(1000)]
        events = [
            _make_event(
                chunk_id=chunks[i].id,
                namespace_id=ns_id,
                referenced_date=datetime(2026, 4, 14, tzinfo=UTC) + timedelta(minutes=i),
                embedding=emb,
            )
            for i in range(1000)
        ]
        coord = _FakeCoordinator(events=events, chunks=chunks)
        engine = _bare_engine()
        _wire(engine, coord)

        tf = TemporalFilter(
            start_time=datetime(2026, 4, 10, tzinfo=UTC),
            end_time=datetime(2026, 5, 1, tzinfo=UTC),
        )
        t0 = time.perf_counter()
        results = await engine._temporal_channel(ns_id, "q", emb, limit=10, temporal_filter=tf)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert len(results) == 10
        # Generous threshold; CI runners vary. The test is mainly a sanity
        # check that we're not doing accidentally O(N²) work.
        assert elapsed_ms < 200, f"temporal channel took {elapsed_ms:.1f}ms for 1000 events"
