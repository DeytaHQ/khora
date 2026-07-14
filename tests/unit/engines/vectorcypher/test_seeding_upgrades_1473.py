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
    _extract_query_mentions,
    _round_robin_seeds,
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
# _extract_query_mentions
# ---------------------------------------------------------------------------


class TestExtractQueryMentions:
    def test_skips_sentence_initial_and_finds_proper_nouns(self) -> None:
        assert _extract_query_mentions("How is Alice connected to Bob?") == ["Alice", "Bob"]

    def test_multi_word_proper_noun_is_one_mention(self) -> None:
        assert _extract_query_mentions("What did Project Phoenix decide?") == ["Project Phoenix"]

    def test_quoted_spans_any_position(self) -> None:
        mentions = _extract_query_mentions('compare "red team" and "blue team"')
        assert mentions == ["red team", "blue team"]

    def test_all_caps_acronym_excluded(self) -> None:
        # Matches the router's ``not .isupper()`` acronym exclusion.
        assert _extract_query_mentions("what is the API doing") == []

    def test_dedup_case_insensitive(self) -> None:
        assert _extract_query_mentions("did Alice meet Alice") == ["Alice"]

    def test_single_mention_query(self) -> None:
        assert _extract_query_mentions("tell me about Kubernetes") == ["Kubernetes"]


# ---------------------------------------------------------------------------
# _round_robin_seeds
# ---------------------------------------------------------------------------


class TestRoundRobinSeeds:
    def test_interleaves_across_mentions(self) -> None:
        a0, a1 = uuid4(), uuid4()
        b0, b1 = uuid4(), uuid4()
        merged = _round_robin_seeds([[(a0, 0.9), (a1, 0.5)], [(b0, 0.8), (b1, 0.4)]], cap=10)
        # rank-0 of each mention comes before any rank-1 entity.
        assert merged[:2] == [(a0, 0.9), (b0, 0.8)]
        assert set(merged[2:]) == {(a1, 0.5), (b1, 0.4)}

    def test_caps_total(self) -> None:
        lists = [[(uuid4(), 0.9)] for _ in range(6)]
        merged = _round_robin_seeds(lists, cap=3)
        assert len(merged) == 3

    def test_dedups_shared_entity(self) -> None:
        shared = uuid4()
        other = uuid4()
        merged = _round_robin_seeds([[(shared, 0.9)], [(shared, 0.8), (other, 0.7)]], cap=10)
        ids = [eid for eid, _ in merged]
        assert ids.count(shared) == 1
        assert other in ids

    def test_uneven_lists(self) -> None:
        a0 = uuid4()
        b0, b1 = uuid4(), uuid4()
        merged = _round_robin_seeds([[(a0, 0.9)], [(b0, 0.8), (b1, 0.4)]], cap=10)
        assert {eid for eid, _ in merged} == {a0, b0, b1}


# ---------------------------------------------------------------------------
# _seed_entry_entities / _per_mention_seed_entities
# ---------------------------------------------------------------------------


class TestSeedEntryEntities:
    async def test_flag_off_uses_global_search(self) -> None:
        """Flag OFF: byte-identical to a direct _vector_search_entities call."""
        ent = uuid4()
        storage = AsyncMock()
        storage.search_similar_entities.return_value = [(ent, 0.7)]
        embedder = AsyncMock()
        retriever = _make_retriever(
            config=RetrieverConfig(enable_per_mention_seeding=False),
            storage=storage,
            embedder=embedder,
        )
        seeds = await retriever._seed_entry_entities(
            query="How is Alice connected to Bob?",
            query_embedding=[0.1, 0.2],
            namespace_id=uuid4(),
            entry_limit=10,
            degradations=[],
        )
        assert seeds == [(ent, 0.7)]
        # Global path: exactly one entity search, no per-mention embeds.
        storage.search_similar_entities.assert_awaited_once()
        embedder.embed.assert_not_called()

    async def test_single_mention_query_uses_global_even_when_on(self) -> None:
        ent = uuid4()
        storage = AsyncMock()
        storage.search_similar_entities.return_value = [(ent, 0.7)]
        embedder = AsyncMock()
        retriever = _make_retriever(
            config=RetrieverConfig(enable_per_mention_seeding=True),
            storage=storage,
            embedder=embedder,
        )
        seeds = await retriever._seed_entry_entities(
            query="tell me about Kubernetes",  # one mention
            query_embedding=[0.1, 0.2],
            namespace_id=uuid4(),
            entry_limit=10,
            degradations=[],
        )
        assert seeds == [(ent, 0.7)]
        embedder.embed.assert_not_called()

    async def test_multi_mention_query_seeds_per_mention(self) -> None:
        alice_ent, bob_ent = uuid4(), uuid4()
        storage = AsyncMock()

        def _search(_ns: Any, emb: Any, *, limit: int, min_similarity: float) -> list:
            # Distinct results keyed by the (mention) embedding. Plain function
            # used as an AsyncMock side_effect: its return value is the awaited
            # result of ``search_similar_entities``.
            if emb == [1.0]:
                return [(alice_ent, 0.9)]
            return [(bob_ent, 0.8)]

        storage.search_similar_entities.side_effect = _search
        embedder = AsyncMock()
        embedder.embed.side_effect = [[1.0], [2.0]]  # Alice -> [1.0], Bob -> [2.0]
        retriever = _make_retriever(
            config=RetrieverConfig(enable_per_mention_seeding=True),
            storage=storage,
            embedder=embedder,
        )
        seeds = await retriever._seed_entry_entities(
            query="How is Alice connected to Bob?",
            query_embedding=[0.5],
            namespace_id=uuid4(),
            entry_limit=10,
            degradations=[],
        )
        ids = {eid for eid, _ in seeds}
        assert ids == {alice_ent, bob_ent}  # every mention contributed
        assert embedder.embed.await_count == 2

    async def test_per_mention_degrades_to_global_on_embed_error(self) -> None:
        ent = uuid4()
        storage = AsyncMock()
        storage.search_similar_entities.return_value = [(ent, 0.7)]
        embedder = AsyncMock()
        embedder.embed.side_effect = RuntimeError("embed boom")
        retriever = _make_retriever(
            config=RetrieverConfig(enable_per_mention_seeding=True),
            storage=storage,
            embedder=embedder,
        )
        degradations: list[Degradation] = []
        seeds = await retriever._seed_entry_entities(
            query="How is Alice connected to Bob?",
            query_embedding=[0.5],
            namespace_id=uuid4(),
            entry_limit=10,
            degradations=degradations,
        )
        # Degraded to the global single-embedding search.
        assert seeds == [(ent, 0.7)]
        assert any(d["component"] == "vectorcypher.per_mention_seed" for d in degradations)

    async def test_respects_max_mentions_cap(self) -> None:
        storage = AsyncMock()
        storage.search_similar_entities.return_value = [(uuid4(), 0.7)]
        embedder = AsyncMock()
        embedder.embed.return_value = [1.0]
        retriever = _make_retriever(
            config=RetrieverConfig(enable_per_mention_seeding=True, per_mention_max_mentions=2),
            storage=storage,
            embedder=embedder,
        )
        await retriever._seed_entry_entities(
            # 4 mentions separated by lowercase words (so they don't merge into
            # one multi-word proper-noun phrase).
            query="did Alice meet Bob and Carol and Dave",
            query_embedding=[0.5],
            namespace_id=uuid4(),
            entry_limit=10,
            degradations=[],
        )
        # Only the first 2 mentions were embedded (cap).
        assert embedder.embed.await_count == 2


# ---------------------------------------------------------------------------
# Config wiring (flag default OFF)
# ---------------------------------------------------------------------------


class TestSeedingConfigDefaults:
    def test_reverse_seeding_default_off(self) -> None:
        cfg = RetrieverConfig()
        assert cfg.enable_reverse_seeding is False
        assert cfg.reverse_seed_top_chunks == 5
        assert cfg.reverse_seed_max_entities == 5

    def test_per_mention_default_off(self) -> None:
        cfg = RetrieverConfig()
        assert cfg.enable_per_mention_seeding is False
        assert cfg.per_mention_max_mentions == 4

    def test_query_settings_default_off(self) -> None:
        from khora.config.schema import QuerySettings

        qs = QuerySettings()
        assert qs.enable_reverse_seeding is False
        assert qs.reverse_seed_top_chunks == 5
        assert qs.reverse_seed_max_entities == 5
        assert qs.enable_per_mention_seeding is False
        assert qs.per_mention_max_mentions == 4
