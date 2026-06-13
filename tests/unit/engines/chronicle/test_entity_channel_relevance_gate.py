"""Regression: chronicle entity channel relevance gate + degradation recording.

Covers two defects in ``ChronicleEngine._entity_channel`` (issues #1143, #1157):

#1143 - The semantic relevance gate was dead code. ``batch_cosine_similarity``
        returns ``(index, score)`` tuples sorted descending, but the channel
        positionally zipped that result against ``embeddings_with_idx`` and
        called ``float(sim)`` on each element. ``float()`` of a tuple raises
        ``TypeError`` on the first iteration; the bare ``except Exception``
        fallback then set every similarity to 1.0, so semantically irrelevant
        entity-adjacent chunks flowed into RRF fusion at full score. These
        tests pin that a low-similarity chunk is dropped and a high-similarity
        chunk is attenuated by its cosine.

#1157 - The channel caught storage exceptions internally and returned ``[]``,
        making a failing channel indistinguishable from "no matching entities":
        no ``Degradation`` recorded, no ``degraded_total`` metric, WARNING logs
        without ``exc_info``. These tests pin that each catch point records a
        ``chronicle.entity`` degradation and bumps the counter, mirroring the
        BM25 path.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from loguru import logger as loguru_logger

from khora.config import KhoraConfig
from khora.config.schema import QuerySettings
from khora.core.models import Chunk, Entity
from khora.engines.chronicle.engine import ChronicleEngine
from khora.query import SearchMode


@pytest.fixture
def loguru_caplog() -> Iterator[str]:
    """Bridge loguru WARNING records into pytest's stdlib caplog."""
    bridge_name = "khora.test.entity_channel_gate"

    def _sink(message: Any) -> None:
        record = message.record
        level = record["level"].no
        logging.getLogger(bridge_name).log(level, record["message"])

    sink_id = loguru_logger.add(_sink, level="WARNING", format="{message}")
    try:
        yield bridge_name
    finally:
        loguru_logger.remove(sink_id)


def _unit(vec: list[float]) -> list[float]:
    """L2-normalize so cosine == dot product (embeddings are normalized at ingest)."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _make_chunk(*, namespace_id: UUID, embedding: list[float] | None) -> Chunk:
    return Chunk(
        namespace_id=namespace_id,
        document_id=uuid4(),
        content="x",
        chunk_index=0,
        embedding=embedding,
    )


def _make_entity(*, namespace_id: UUID, name: str, chunk_ids: list[UUID]) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=namespace_id,
        name=name,
        entity_type="THING",
        source_chunk_ids=chunk_ids,
    )


class _EntityCoord:
    """Coordinator double for the entity channel, configurable to raise per-call."""

    def __init__(
        self,
        *,
        entity_search_results: list[tuple[UUID, float]] | None = None,
        entities: dict[UUID, Entity] | None = None,
        chunks: dict[UUID, Chunk] | None = None,
        search_similar_entities_raises: Exception | None = None,
        get_entities_batch_raises: Exception | None = None,
        get_chunks_batch_raises: Exception | None = None,
    ) -> None:
        self._entity_search_results = entity_search_results or []
        self._entities = entities or {}
        self._chunks = chunks or {}
        self._search_similar_entities_raises = search_similar_entities_raises
        self._get_entities_batch_raises = get_entities_batch_raises
        self._get_chunks_batch_raises = get_chunks_batch_raises

    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        return []

    async def search_fulltext_chunks(
        self,
        namespace_id: UUID,
        query: str,
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        return []

    async def query_events(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[UUID, float]]:
        if self._search_similar_entities_raises is not None:
            raise self._search_similar_entities_raises
        return list(self._entity_search_results)

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        if self._get_entities_batch_raises is not None:
            raise self._get_entities_batch_raises
        return {eid: self._entities[eid] for eid in entity_ids if eid in self._entities}

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        if self._get_chunks_batch_raises is not None:
            raise self._get_chunks_batch_raises
        return {cid: self._chunks[cid] for cid in chunk_ids if cid in self._chunks}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Any]:
        return {}


def _bare_engine(**kwargs: Any) -> ChronicleEngine:
    cfg = KhoraConfig(
        database_url="postgresql://localhost/test",
        query=QuerySettings(enable_reranking=False),
    )
    return ChronicleEngine(cfg, **kwargs)


def _wire(engine: ChronicleEngine, coord: _EntityCoord) -> None:
    engine._storage = coord  # type: ignore[assignment]
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    engine._embedder = embedder  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# #1143 - the relevance gate must actually run
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_low_similarity_chunk_dropped_by_relevance_gate() -> None:
    """A chunk semantically orthogonal to the query is dropped by the < 0.3 gate.

    Pre-fix the gate was dead (all sims forced to 1.0 by the swallowed
    ``TypeError``), so the orthogonal chunk leaked through at full score.
    """
    ns_id = uuid4()

    # Query embedding aligns with the "relevant" axis.
    query_embedding = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    relevant = _make_chunk(namespace_id=ns_id, embedding=_unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    irrelevant = _make_chunk(namespace_id=ns_id, embedding=_unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    ent_relevant = _make_entity(namespace_id=ns_id, name="Relevant", chunk_ids=[relevant.id])
    ent_irrelevant = _make_entity(namespace_id=ns_id, name="Irrelevant", chunk_ids=[irrelevant.id])

    coord = _EntityCoord(
        entity_search_results=[(ent_relevant.id, 0.9), (ent_irrelevant.id, 0.9)],
        entities={ent_relevant.id: ent_relevant, ent_irrelevant.id: ent_irrelevant},
        chunks={relevant.id: relevant, irrelevant.id: irrelevant},
    )
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    results = await engine._entity_channel(ns_id, "q", query_embedding, limit=10)

    returned_ids = {chunk.id for chunk, _ in results}
    assert relevant.id in returned_ids, "Query-aligned chunk should pass the relevance gate"
    assert irrelevant.id not in returned_ids, (
        "Orthogonal (sim < 0.3) chunk should be dropped by the relevance gate; "
        "if it leaked through, the gate is dead (#1143)"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relevance_gate_attenuates_score_by_cosine() -> None:
    """A partially-aligned chunk's score is attenuated by its real cosine, not 1.0.

    The query and chunk are at 45 degrees (cosine ~0.707). The channel score is
    ``entity_similarity * cosine``. Pre-fix every cosine was 1.0, so the score
    would equal the entity similarity unattenuated.
    """
    ns_id = uuid4()

    query_embedding = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # 45-degree chunk -> cosine ~0.7071, above the 0.3 gate.
    partial = _make_chunk(namespace_id=ns_id, embedding=_unit([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    entity_sim = 0.8
    ent = _make_entity(namespace_id=ns_id, name="Partial", chunk_ids=[partial.id])

    coord = _EntityCoord(
        entity_search_results=[(ent.id, entity_sim)],
        entities={ent.id: ent},
        chunks={partial.id: partial},
    )
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    results = await engine._entity_channel(ns_id, "q", query_embedding, limit=10)

    assert len(results) == 1
    _chunk, score = results[0]
    expected = entity_sim * (1.0 / math.sqrt(2.0))  # entity_sim * cos(45deg)
    assert score == pytest.approx(expected, abs=1e-3), (
        f"Score should be entity_sim*cosine={expected:.4f}, got {score:.4f}; "
        f"if it equals entity_sim ({entity_sim}), the cosine was forced to 1.0 (#1143)"
    )


# ---------------------------------------------------------------------------
# #1157 - storage failures inside the entity channel must record a degradation
# ---------------------------------------------------------------------------


def _seeded_coord(ns_id: UUID, **raises: Exception) -> _EntityCoord:
    """A coord that gets far enough to hit any of the three catch points."""
    chunk = _make_chunk(namespace_id=ns_id, embedding=_unit([1.0] * 8))
    ent = _make_entity(namespace_id=ns_id, name="E", chunk_ids=[chunk.id])
    return _EntityCoord(
        entity_search_results=[(ent.id, 0.9)],
        entities={ent.id: ent},
        chunks={chunk.id: chunk},
        **raises,
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raise_kwarg",
    [
        "search_similar_entities_raises",
        "get_entities_batch_raises",
        "get_chunks_batch_raises",
    ],
)
async def test_entity_channel_storage_failure_records_degradation(
    raise_kwarg: str, caplog: pytest.LogCaptureFixture, loguru_caplog: str
) -> None:
    """Each storage call in the entity channel records a chronicle.entity degradation."""
    ns_id = uuid4()
    coord = _seeded_coord(ns_id, **{raise_kwarg: RuntimeError("backend offline")})
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    with caplog.at_level(logging.WARNING, logger=loguru_caplog):
        result = await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID)

    degradations = result.engine_info["degradations"]
    entity_entries = [d for d in degradations if d["component"] == "chronicle.entity"]
    assert len(entity_entries) == 1, (
        f"Expected exactly one chronicle.entity degradation for {raise_kwarg}; got {degradations!r}"
    )
    assert entity_entries[0]["reason"] == "channel_exception"
    assert entity_entries[0]["exception"] == "RuntimeError"

    warn_logs = [r for r in caplog.records if r.levelno >= logging.WARNING and "entity" in r.getMessage().lower()]
    assert warn_logs, "Expected a WARNING-level log when the entity channel fails"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_entity_channel_failure_bumps_metric_counter() -> None:
    """``khora.chronicle.channel.degraded_total{channel=entity}`` is bumped on failure."""
    ns_id = uuid4()
    coord = _seeded_coord(ns_id, search_similar_entities_raises=RuntimeError("offline"))
    engine = _bare_engine(router_enabled=False)
    _wire(engine, coord)

    with patch("khora.engines.chronicle.engine._CHANNEL_DEGRADED_COUNTER") as mock_counter:
        await engine.recall("test", ns_id, limit=5, mode=SearchMode.HYBRID)

    entity_calls = [
        call for call in mock_counter.add.call_args_list if call.kwargs.get("attributes", {}).get("channel") == "entity"
    ]
    assert entity_calls, "Expected an entity degraded_total bump"
    assert entity_calls[0].args[0] == 1
    assert entity_calls[0].kwargs["attributes"]["reason"] == "channel_exception"
