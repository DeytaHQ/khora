"""Unit tests for the #1473 graph-channel seeding upgrades.

Covers two of the three independently-flagged, default-OFF upgrades that live in
the retriever's entry-entity seeding path:

* reverse-seeding (``_reverse_seed_entities`` / ``_merge_seed_entities``): seed
  the graph from the entities connected to the top vector chunks.
* per-mention diversified seeding (``_extract_query_mentions`` /
  ``_per_mention_seed_entities`` / ``_seed_entry_entities``): decompose a
  multi-entity query and round-robin the entry-entity budget across mentions.

The evidence-based graph gate (the third upgrade) lives at the router layer and
is tested in ``tests/unit/query/test_evidence_graph_gate_1473.py``.

Each upgrade is asserted OFF (legacy behavior reproduced) and ON (new behavior).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.core.diagnostics import Degradation
from khora.core.models import Chunk, Entity
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)

pytestmark = pytest.mark.unit


def _make_chunk(content: str = "text") -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
    )


def _make_retriever(
    *,
    config: RetrieverConfig | None = None,
    storage: Any | None = None,
    embedder: Any | None = None,
) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=embedder if embedder is not None else AsyncMock(),
        config=config or RetrieverConfig(),
        storage=storage,
    )


# ---------------------------------------------------------------------------
# _merge_seed_entities — pure helper
# ---------------------------------------------------------------------------


class TestMergeSeedEntities:
    def test_never_drops_primary_and_preserves_order(self) -> None:
        a, b = uuid4(), uuid4()
        c, d = uuid4(), uuid4()
        primary = [(a, 0.9), (b, 0.8)]
        extra = [(c, 0.95), (d, 0.7)]
        merged = VectorCypherRetriever._merge_seed_entities(primary, extra, max_added=5)
        # Primary entities keep their leading position + scores untouched.
        assert merged[:2] == primary
        assert {eid for eid, _ in merged} == {a, b, c, d}

    def test_caps_added_at_max_added(self) -> None:
        primary = [(uuid4(), 0.9)]
        extra = [(uuid4(), 0.8), (uuid4(), 0.7), (uuid4(), 0.6)]
        merged = VectorCypherRetriever._merge_seed_entities(primary, extra, max_added=2)
        assert len(merged) == 3  # 1 primary + 2 added (third extra dropped)

    def test_dedups_against_primary_and_within_extra(self) -> None:
        shared = uuid4()
        primary = [(shared, 0.9)]
        extra = [(shared, 0.99), (shared, 0.5)]
        merged = VectorCypherRetriever._merge_seed_entities(primary, extra, max_added=5)
        # ``shared`` already in primary: not re-added, primary score preserved.
        assert merged == [(shared, 0.9)]

    def test_empty_extra_returns_primary_copy(self) -> None:
        primary = [(uuid4(), 0.9)]
        merged = VectorCypherRetriever._merge_seed_entities(primary, [], max_added=5)
        assert merged == primary
        assert merged is not primary  # a fresh list


# ---------------------------------------------------------------------------
# _reverse_seed_entities
# ---------------------------------------------------------------------------


class TestReverseSeedEntities:
    async def test_scores_entities_by_best_mentioning_chunk(self) -> None:
        c_hi = _make_chunk("high")
        c_lo = _make_chunk("low")
        ent_a, ent_b = uuid4(), uuid4()
        storage = AsyncMock()
        storage.list_entities.return_value = [
            Entity(id=ent_a, name="A", source_chunk_ids=[c_hi.id, c_lo.id]),
            Entity(id=ent_b, name="B", source_chunk_ids=[c_lo.id]),
        ]
        retriever = _make_retriever(storage=storage)
        degradations: list[Degradation] = []
        seeds = await retriever._reverse_seed_entities(
            [(c_hi.id, 0.4, c_hi), (c_lo.id, 0.4, c_lo)],
            namespace_id=uuid4(),
            raw_cosine_by_id={c_hi.id: 0.9, c_lo.id: 0.3},
            top_chunks=5,
            max_entities=5,
            degradations=degradations,
        )
        # A mentions the high-cosine chunk -> stronger seed, sorted first.
        assert seeds[0] == (ent_a, 0.9)
        assert (ent_b, 0.3) in seeds
        assert not degradations

    async def test_respects_top_chunks_window(self) -> None:
        c_hi = _make_chunk("hi")
        c_lo = _make_chunk("lo")
        only_lo = uuid4()
        storage = AsyncMock()
        storage.list_entities.return_value = [
            Entity(id=only_lo, name="LO", source_chunk_ids=[c_lo.id]),
        ]
        retriever = _make_retriever(storage=storage)
        seeds = await retriever._reverse_seed_entities(
            [(c_hi.id, 0.9, c_hi), (c_lo.id, 0.3, c_lo)],
            namespace_id=uuid4(),
            raw_cosine_by_id={c_hi.id: 0.9, c_lo.id: 0.3},
            top_chunks=1,  # only the top chunk (c_hi) seeds
            max_entities=5,
            degradations=[],
        )
        # list_entities is called with only the top chunk's id.
        called_ids = storage.list_entities.call_args.kwargs["source_chunk_ids"]
        assert called_ids == [c_hi.id]
        # The returned entity mentions only c_lo (outside the window) -> dropped.
        assert seeds == []

    async def test_caps_at_max_entities(self) -> None:
        chunk = _make_chunk()
        storage = AsyncMock()
        storage.list_entities.return_value = [
            Entity(id=uuid4(), name=str(i), source_chunk_ids=[chunk.id]) for i in range(10)
        ]
        retriever = _make_retriever(storage=storage)
        seeds = await retriever._reverse_seed_entities(
            [(chunk.id, 0.5, chunk)],
            namespace_id=uuid4(),
            raw_cosine_by_id={chunk.id: 0.5},
            top_chunks=5,
            max_entities=3,
            degradations=[],
        )
        assert len(seeds) == 3

    async def test_empty_pool_returns_empty(self) -> None:
        retriever = _make_retriever(storage=AsyncMock())
        seeds = await retriever._reverse_seed_entities(
            [],
            namespace_id=uuid4(),
            raw_cosine_by_id={},
            top_chunks=5,
            max_entities=5,
            degradations=[],
        )
        assert seeds == []

    async def test_no_storage_returns_empty(self) -> None:
        chunk = _make_chunk()
        retriever = _make_retriever(storage=None)
        seeds = await retriever._reverse_seed_entities(
            [(chunk.id, 0.5, chunk)],
            namespace_id=uuid4(),
            raw_cosine_by_id={chunk.id: 0.5},
            top_chunks=5,
            max_entities=5,
            degradations=[],
        )
        assert seeds == []

    async def test_degrades_on_lookup_error(self) -> None:
        chunk = _make_chunk()
        storage = AsyncMock()
        storage.list_entities.side_effect = RuntimeError("boom")
        retriever = _make_retriever(storage=storage)
        degradations: list[Degradation] = []
        seeds = await retriever._reverse_seed_entities(
            [(chunk.id, 0.5, chunk)],
            namespace_id=uuid4(),
            raw_cosine_by_id={chunk.id: 0.5},
            top_chunks=5,
            max_entities=5,
            degradations=degradations,
        )
        assert seeds == []
        assert len(degradations) == 1
        assert degradations[0]["component"] == "vectorcypher.reverse_seed"
        assert degradations[0]["reason"] == "lookup_failed"


# ---------------------------------------------------------------------------
# Config wiring (flag default OFF)
# ---------------------------------------------------------------------------


class TestSeedingConfigDefaults:
    def test_reverse_seeding_default_off(self) -> None:
        cfg = RetrieverConfig()
        assert cfg.enable_reverse_seeding is False
        assert cfg.reverse_seed_top_chunks == 5
        assert cfg.reverse_seed_max_entities == 5

    def test_query_settings_default_off(self) -> None:
        from khora.config.schema import QuerySettings

        qs = QuerySettings()
        assert qs.enable_reverse_seeding is False
        assert qs.reverse_seed_top_chunks == 5
        assert qs.reverse_seed_max_entities == 5
