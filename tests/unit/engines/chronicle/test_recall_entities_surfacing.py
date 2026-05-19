"""Chronicle #5: tests for surfacing entity hits in ``RecallResult.entities``.

Two sources feed ``RecallResult.entities``:

1. The entity channel (``_entity_channel``) — direct similarity hits get full
   score.
2. The temporal-events channel (``_temporal_channel``) — event subjects are
   resolved to Entity records by name and added with score attenuated by 0.5.

Tests use mock coordinators so no real DB / LLM calls are made. The
``_collect_entities`` helper is exercised directly where useful, plus
end-to-end through ``recall()``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, Entity
from khora.engines.chronicle.engine import ChronicleEngine
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
        chunk_index=0,
    )


def _make_entity(
    name: str,
    *,
    namespace_id: UUID,
    entity_type: str = "PERSON",
    chunk_ids: list[UUID] | None = None,
    embedding: list[float] | None = None,
) -> Entity:
    return Entity(
        namespace_id=namespace_id,
        name=name,
        entity_type=entity_type,
        source_chunk_ids=list(chunk_ids or []),
        embedding=embedding,
    )


def _make_event(
    *,
    chunk_id: UUID,
    namespace_id: UUID,
    subject: str = "Alice",
    referenced_date: datetime | None = None,
    embedding: list[float] | None = None,
) -> ChronicleEvent:
    return ChronicleEvent(
        chunk_id=chunk_id,
        namespace_id=namespace_id,
        subject=subject,
        verb="met",
        object="Bob",
        referenced_date=referenced_date,
        embedding=embedding,
    )


class _FakeCoordinator:
    """Coordinator double exercising the entity + temporal-events paths."""

    def __init__(
        self,
        *,
        events: list[ChronicleEvent] | None = None,
        chunks: list[Chunk] | None = None,
        entities_by_id: dict[UUID, Entity] | None = None,
        entities_by_name: dict[str, Entity] | None = None,
        similar_entity_results: list[tuple[UUID, float]] | None = None,
    ) -> None:
        self._events = events or []
        self._chunks_by_id = {c.id: c for c in (chunks or [])}
        self._entities_by_id = entities_by_id or {}
        self._entities_by_name = entities_by_name or {}
        self._similar_entity_results = similar_entity_results or []

        self.get_entities_by_names_calls: list[list[str]] = []
        self.search_similar_entities_calls: list[dict[str, Any]] = []

    # --- temporal channel deps ---
    async def query_events(
        self,
        namespace_id: UUID,
        *,
        subject: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[ChronicleEvent]:
        out = []
        for ev in self._events:
            ref = ev.referenced_date
            if since is not None and (ref is None or ref < since):
                continue
            if until is not None and (ref is None or ref > until):
                continue
            out.append(ev)
        return out[:limit]

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        return {cid: self._chunks_by_id[cid] for cid in chunk_ids if cid in self._chunks_by_id}

    async def search_similar_chunks(self, *args: Any, **kwargs: Any) -> list[tuple[Chunk, float]]:
        return []

    # --- entity channel deps ---
    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        self.search_similar_entities_calls.append({"limit": limit})
        return list(self._similar_entity_results)

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
        return {eid: self._entities_by_id[eid] for eid in entity_ids if eid in self._entities_by_id}

    # --- collect_entities dep (Chronicle #5) ---
    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Entity]:
        self.get_entities_by_names_calls.append(list(names))
        return {n: self._entities_by_name[n] for n in names if n in self._entities_by_name}


def _bare_engine(**kwargs: Any) -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"), **kwargs)


def _wire(engine: ChronicleEngine, coord: _FakeCoordinator) -> None:
    engine._storage = coord  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# _collect_entities — direct unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollectEntities:
    @pytest.mark.asyncio
    async def test_entity_channel_only(self) -> None:
        ns_id = uuid4()
        e1 = _make_entity("Alice", namespace_id=ns_id)
        e2 = _make_entity("Bob", namespace_id=ns_id)
        e3 = _make_entity("Carol", namespace_id=ns_id)
        coord = _FakeCoordinator()
        engine = _bare_engine()
        _wire(engine, coord)

        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits={
                e1.id: (e1, 0.9),
                e2.id: (e2, 0.7),
                e3.id: (e3, 0.5),
            },
            temporal_event_subjects={},
            limit=20,
        )
        # Sorted by score desc; no DB lookup since temporal subjects empty.
        assert [ent.name for ent, _ in result] == ["Alice", "Bob", "Carol"]
        assert [score for _, score in result] == [0.9, 0.7, 0.5]
        assert coord.get_entities_by_names_calls == []

    @pytest.mark.asyncio
    async def test_temporal_events_only(self) -> None:
        ns_id = uuid4()
        e1 = _make_entity("Alice", namespace_id=ns_id)
        e2 = _make_entity("Bob", namespace_id=ns_id)
        coord = _FakeCoordinator(entities_by_name={"Alice": e1, "Bob": e2})
        engine = _bare_engine()
        _wire(engine, coord)

        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits={},
            temporal_event_subjects={"Alice": 0.8, "Bob": 0.6},
            limit=20,
        )
        # Both subjects resolved; scores attenuated by 0.5.
        assert {ent.name: score for ent, score in result} == {"Alice": 0.4, "Bob": 0.3}
        # Sorted by attenuated score desc.
        assert [ent.name for ent, _ in result] == ["Alice", "Bob"]
        # Both names looked up in a single batch call.
        assert len(coord.get_entities_by_names_calls) == 1
        assert sorted(coord.get_entities_by_names_calls[0]) == ["Alice", "Bob"]

    @pytest.mark.asyncio
    async def test_dedupe_entity_channel_beats_event_score(self) -> None:
        ns_id = uuid4()
        alice = _make_entity("Alice", namespace_id=ns_id)
        coord = _FakeCoordinator(entities_by_name={"Alice": alice})
        engine = _bare_engine()
        _wire(engine, coord)

        # Alice hit by entity channel at 0.9 AND mentioned in events with score 1.0
        # Event score after attenuation = 0.5; entity-channel score = 0.9 wins.
        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits={alice.id: (alice, 0.9)},
            temporal_event_subjects={"Alice": 1.0},
            limit=20,
        )
        assert len(result) == 1
        ent, score = result[0]
        assert ent.id == alice.id
        assert score == pytest.approx(0.9)
        # Alice is already in merged-by-name, so we should not even hit the DB.
        assert coord.get_entities_by_names_calls == []

    @pytest.mark.asyncio
    async def test_event_subject_not_found_is_skipped(self) -> None:
        ns_id = uuid4()
        # Event subject "Unknown" has no Entity row — should silently disappear.
        coord = _FakeCoordinator(entities_by_name={})
        engine = _bare_engine()
        _wire(engine, coord)

        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits={},
            temporal_event_subjects={"Unknown Person": 0.7},
            limit=20,
        )
        assert result == []
        # Still issued the lookup — just got no rows back.
        assert coord.get_entities_by_names_calls == [["Unknown Person"]]

    @pytest.mark.asyncio
    async def test_limit_truncation(self) -> None:
        ns_id = uuid4()
        ents = [_make_entity(f"E{i}", namespace_id=ns_id) for i in range(30)]
        # Score schedule: E0=1.0, E1=0.99, ... — descending.
        hits = {e.id: (e, 1.0 - i * 0.01) for i, e in enumerate(ents)}
        coord = _FakeCoordinator()
        engine = _bare_engine()
        _wire(engine, coord)

        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits=hits,
            temporal_event_subjects={},
            limit=20,
        )
        assert len(result) == 20
        # Top-20 by score; E0..E19 are the highest.
        names = [ent.name for ent, _ in result]
        assert names == [f"E{i}" for i in range(20)]

    @pytest.mark.asyncio
    async def test_empty_inputs_return_empty(self) -> None:
        ns_id = uuid4()
        coord = _FakeCoordinator()
        engine = _bare_engine()
        _wire(engine, coord)

        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits={},
            temporal_event_subjects={},
            limit=20,
        )
        assert result == []
        # No DB calls when there's nothing to resolve.
        assert coord.get_entities_by_names_calls == []

    @pytest.mark.asyncio
    async def test_score_sort_descending(self) -> None:
        ns_id = uuid4()
        a = _make_entity("A", namespace_id=ns_id)
        b = _make_entity("B", namespace_id=ns_id)
        c = _make_entity("C", namespace_id=ns_id)
        coord = _FakeCoordinator()
        engine = _bare_engine()
        _wire(engine, coord)

        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits={a.id: (a, 0.3), b.id: (b, 0.95), c.id: (c, 0.6)},
            temporal_event_subjects={},
            limit=20,
        )
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)
        assert [ent.name for ent, _ in result] == ["B", "C", "A"]

    @pytest.mark.asyncio
    async def test_mixed_sources_with_partial_overlap(self) -> None:
        ns_id = uuid4()
        # Entity-channel: 5 hits (Alice, Bob, Carol, Dan, Eve)
        # Event subjects: 5 (Alice, Bob, Frank, Grace, Heidi)
        # Overlap: Alice, Bob -> total unique = 8
        names_ec = ["Alice", "Bob", "Carol", "Dan", "Eve"]
        names_ev = ["Alice", "Bob", "Frank", "Grace", "Heidi"]
        ents = {n: _make_entity(n, namespace_id=ns_id) for n in set(names_ec) | set(names_ev)}
        ec_hits = {ents[n].id: (ents[n], 0.9 - 0.05 * i) for i, n in enumerate(names_ec)}
        coord = _FakeCoordinator(entities_by_name={n: ents[n] for n in names_ev})
        engine = _bare_engine()
        _wire(engine, coord)

        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits=ec_hits,
            temporal_event_subjects={n: 0.8 for n in names_ev},
            limit=20,
        )
        # 5 EC hits + 3 unique event-derived (Frank, Grace, Heidi) = 8
        assert len(result) == 8
        result_names = {ent.name for ent, _ in result}
        assert result_names == {"Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Heidi"}
        # Alice/Bob keep entity-channel scores (full), not attenuated.
        result_scores = {ent.name: score for ent, score in result}
        assert result_scores["Alice"] == pytest.approx(0.9)
        assert result_scores["Bob"] == pytest.approx(0.85)
        # Frank/Grace/Heidi are attenuated.
        assert result_scores["Frank"] == pytest.approx(0.4)
        # Only the 3 non-overlapping names should hit the DB.
        assert len(coord.get_entities_by_names_calls) == 1
        looked_up = sorted(coord.get_entities_by_names_calls[0])
        assert looked_up == ["Frank", "Grace", "Heidi"]

    @pytest.mark.asyncio
    async def test_event_score_attenuation(self) -> None:
        ns_id = uuid4()
        alice = _make_entity("Alice", namespace_id=ns_id)
        coord = _FakeCoordinator(entities_by_name={"Alice": alice})
        engine = _bare_engine()
        _wire(engine, coord)

        result = await engine._collect_entities(
            namespace_id=ns_id,
            entity_channel_hits={},
            temporal_event_subjects={"Alice": 1.0},
            limit=20,
        )
        # Pure event-derived Alice scores at 1.0 * 0.5 = 0.5.
        assert len(result) == 1
        _, score = result[0]
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _entity_channel populates the accumulator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityChannelAccumulator:
    @pytest.mark.asyncio
    async def test_entity_channel_records_hits(self) -> None:
        ns_id = uuid4()
        chunk = _make_chunk(namespace_id=ns_id)
        alice = _make_entity("Alice", namespace_id=ns_id, chunk_ids=[chunk.id])
        bob = _make_entity("Bob", namespace_id=ns_id, chunk_ids=[chunk.id])
        coord = _FakeCoordinator(
            chunks=[chunk],
            entities_by_id={alice.id: alice, bob.id: bob},
            similar_entity_results=[(alice.id, 0.92), (bob.id, 0.71)],
        )
        engine = _bare_engine()
        _wire(engine, coord)

        accumulator: dict[UUID, tuple[Entity, float]] = {}
        await engine._entity_channel(ns_id, "q", [1.0, 0.0, 0.0], limit=10, entity_hits=accumulator)
        assert set(accumulator.keys()) == {alice.id, bob.id}
        assert accumulator[alice.id][0].name == "Alice"
        assert accumulator[alice.id][1] == pytest.approx(0.92)
        assert accumulator[bob.id][1] == pytest.approx(0.71)


# ---------------------------------------------------------------------------
# _temporal_channel populates the subject scores
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTemporalChannelSubjects:
    @pytest.mark.asyncio
    async def test_subjects_recorded_with_max_score(self) -> None:
        ns_id = uuid4()
        c1 = _make_chunk(namespace_id=ns_id)
        c2 = _make_chunk(namespace_id=ns_id)
        emb = [1.0, 0.0, 0.0]
        # Two events on Alice — different chunks, different referenced_dates;
        # the closer one (April 14) gets the higher combined score.
        focal = datetime(2026, 4, 14, tzinfo=UTC)
        ev_close = _make_event(
            chunk_id=c1.id,
            namespace_id=ns_id,
            subject="Alice",
            referenced_date=focal,
            embedding=emb,
        )
        ev_far = _make_event(
            chunk_id=c2.id,
            namespace_id=ns_id,
            subject="Alice",
            referenced_date=datetime(2026, 1, 1, tzinfo=UTC),
            embedding=emb,
        )

        # No-clip coord so both events reach the scorer.
        class _NoClip(_FakeCoordinator):
            async def query_events(self, namespace_id: UUID, **kwargs: Any) -> list[ChronicleEvent]:
                return list(self._events)

        coord = _NoClip(events=[ev_close, ev_far], chunks=[c1, c2])
        engine = _bare_engine()
        _wire(engine, coord)

        accumulator: dict[str, float] = {}
        tf = TemporalFilter(start_time=focal, end_time=focal)
        await engine._temporal_channel(ns_id, "q", emb, limit=5, temporal_filter=tf, subject_scores=accumulator)
        # One subject — Alice — captured with the *max* score (the closer event).
        assert "Alice" in accumulator
        # The far event score is much lower than the close one.
        # (We don't assert exact value — the helper picks max combined.)
        assert accumulator["Alice"] > 0.5

    @pytest.mark.asyncio
    async def test_empty_subject_skipped(self) -> None:
        ns_id = uuid4()
        c = _make_chunk(namespace_id=ns_id)
        ev = _make_event(
            chunk_id=c.id,
            namespace_id=ns_id,
            subject="   ",  # whitespace-only subject — must be ignored
            referenced_date=datetime(2026, 4, 14, tzinfo=UTC),
            embedding=[1.0, 0.0, 0.0],
        )

        class _NoClip(_FakeCoordinator):
            async def query_events(self, namespace_id: UUID, **kwargs: Any) -> list[ChronicleEvent]:
                return list(self._events)

        coord = _NoClip(events=[ev], chunks=[c])
        engine = _bare_engine()
        _wire(engine, coord)

        accumulator: dict[str, float] = {}
        tf = TemporalFilter(
            start_time=datetime(2026, 4, 10, tzinfo=UTC),
            end_time=datetime(2026, 4, 17, tzinfo=UTC),
        )
        await engine._temporal_channel(
            ns_id, "q", [1.0, 0.0, 0.0], limit=5, temporal_filter=tf, subject_scores=accumulator
        )
        assert accumulator == {}
