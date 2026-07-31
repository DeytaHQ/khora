"""Union fusion strategies + quota-aware CE admission (#1518).

``fusion_mode="union_best_rank"`` is a rank-preserving interleave that exposes
each enabled channel's head at fusion exit (spec §4.1); ``union_mnz`` is the
CombMNZ count-boost diagnostic sibling (§4.2). Because the recency boost and
coherence blend re-sort between fusion exit and the cross-encoder window slice,
the interleave's exposure property does NOT survive to CE admission - so the
union modes pair with quota-aware CE-window admission in ``_select_rerank_window``
(co-primary), which guarantees each active channel its top-``floor(top_n/n)`` by
fusion-exit channel rank regardless of the intervening re-sorts.

These tests assert:

1. union_best_rank interleaving: each channel's top-k is exposed at fusion exit.
2. disabled-channel and empty-channel handling.
3. union_mnz count-boost math.
4. tie-break ordering by the (best_rank, channel_count, priority, norm) hierarchy.
5. per-channel-limit truncation.
6. quota-aware admission rescues a channel head that plain rrf would drop.
7. rrf/calibrated are byte-identical: dispatch untouched, admission is a no-op.
8. config validators reject empty/duplicate channels.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.engines.vectorcypher.fusion import (
    FusedResult,
    union_best_rank_fusion,
    union_mnz_fusion,
    weighted_rrf_normalized,
)


def _item(name: str) -> Any:
    return type("Item", (), {"name": name})()


def _order(results: list[FusedResult]) -> list[UUID]:
    return [r.item_id for r in results]


def _channel(prefix: str, n: int) -> list[tuple[UUID, float, Any]]:
    """A channel of ``n`` results with descending scores (rank order)."""
    return [(uuid4(), float(n - i), _item(f"{prefix}{i + 1}")) for i in range(n)]


@pytest.mark.unit
class TestUnionBestRankInterleave:
    def test_blocks_by_best_rank_exposing_each_channel_head(self) -> None:
        # Three disjoint channels of 3 items each. The interleave must order by
        # best_rank first, so all rank-1 items lead (ordered by channel
        # priority), then all rank-2, then all rank-3.
        vector = _channel("v", 3)
        graph = _channel("g", 3)
        bm25 = _channel("b", 3)
        fused = union_best_rank_fusion(
            [("vector", vector), ("graph", graph), ("bm25", bm25)],
        )
        order = _order(fused)
        # rank-1 block (priority vector < graph < bm25), then rank-2, then rank-3.
        expected = [
            vector[0][0], graph[0][0], bm25[0][0],
            vector[1][0], graph[1][0], bm25[1][0],
            vector[2][0], graph[2][0], bm25[2][0],
        ]  # fmt: skip
        assert order == expected

    def test_each_channel_top_m_within_first_m_times_num_channels(self) -> None:
        # The exposure property: channel c's top-m sits within the first
        # m * len(channels) positions at fusion exit.
        vector = _channel("v", 5)
        graph = _channel("g", 5)
        bm25 = _channel("b", 5)
        channels = [("vector", vector), ("graph", graph), ("bm25", bm25)]
        order = _order(union_best_rank_fusion(channels))
        for _name, results in channels:
            for m in range(1, 6):
                head_ids = {cid for cid, _s, _i in results[:m]}
                positions = [order.index(cid) for cid in head_ids]
                assert max(positions) < m * len(channels)

    def test_synthetic_rrf_score_strictly_decreasing_positive(self) -> None:
        fused = union_best_rank_fusion([("vector", _channel("v", 4))], rrf_k=60)
        scores = [r.rrf_score for r in fused]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0 / 61)

    def test_corroboration_promotes_multichannel_item(self) -> None:
        # An item at rank 1 in vector AND graph must outrank a vector-only rank-1
        # peer via the channel_count (desc) tiebreak.
        shared, solo = uuid4(), uuid4()
        vector = [(shared, 0.9, _item("s")), (solo, 0.8, _item("x"))]
        graph = [(shared, 5.0, _item("s"))]
        order = _order(union_best_rank_fusion([("vector", vector), ("graph", graph)]))
        assert order[0] == shared  # best_rank 1 in both, channel_count 2 wins
        assert order.index(shared) < order.index(solo)

    def test_provenance_backfilled_including_bm25(self) -> None:
        v, g, b = uuid4(), uuid4(), uuid4()
        fused = {
            r.item_id: r
            for r in union_best_rank_fusion(
                [
                    ("vector", [(v, 0.9, _item("v"))]),
                    ("graph", [(g, 5.0, _item("g"))]),
                    ("bm25", [(b, 2.0, _item("b"))]),
                ]
            )
        }
        assert fused[v].vector_rank == 1 and fused[v].graph_rank is None and fused[v].bm25_rank is None
        assert fused[g].graph_rank == 1 and fused[g].vector_rank is None
        assert fused[b].bm25_rank == 1 and fused[b].vector_rank is None
        assert fused[b].bm25_score == pytest.approx(2.0)


@pytest.mark.unit
class TestChannelHandling:
    def test_empty_channel_contributes_nothing(self) -> None:
        vector = _channel("v", 2)
        fused = union_best_rank_fusion([("vector", vector), ("graph", [])])
        assert len(fused) == 2
        assert all(r.graph_rank is None for r in fused)

    def test_priority_order_follows_channel_list_order(self) -> None:
        # Same items, reversed channel priority -> reversed rank-1 block order.
        v1, b1 = uuid4(), uuid4()
        vector = [(v1, 0.9, _item("v"))]
        bm25 = [(b1, 3.0, _item("b"))]
        order_vg = _order(union_best_rank_fusion([("vector", vector), ("bm25", bm25)]))
        order_gv = _order(union_best_rank_fusion([("bm25", bm25), ("vector", vector)]))
        assert order_vg == [v1, b1]
        assert order_gv == [b1, v1]

    def test_per_channel_limit_truncates(self) -> None:
        vector = _channel("v", 5)
        fused = union_best_rank_fusion([("vector", vector)], per_channel_limit=2)
        assert len(fused) == 2
        assert _order(fused) == [vector[0][0], vector[1][0]]

    def test_empty_everything(self) -> None:
        assert union_best_rank_fusion([]) == []
        assert union_mnz_fusion([], {}) == []


@pytest.mark.unit
class TestUnionMnz:
    def test_count_boost_math(self) -> None:
        # shared: present in vector rank1 + graph rank1 (channel_count 2).
        # solo: vector rank2 only (channel_count 1).
        shared, solo = uuid4(), uuid4()
        vector = [(shared, 0.9, _item("s")), (solo, 0.8, _item("x"))]
        graph = [(shared, 5.0, _item("s"))]
        weights = {"vector": 0.6, "graph": 0.4}
        fused = {
            r.item_id: r.rrf_score for r in union_mnz_fusion([("vector", vector), ("graph", graph)], weights, rrf_k=60)
        }
        # shared: 2 * (0.6/61 + 0.4/61); solo: 1 * (0.6/62).
        assert fused[shared] == pytest.approx(2 * (0.6 / 61 + 0.4 / 61))
        assert fused[solo] == pytest.approx(1 * (0.6 / 62))
        assert fused[shared] > fused[solo]

    def test_single_channel_singletons_not_boosted(self) -> None:
        # The structural objection: lone single-channel items are multiplied by 1.
        a, b = uuid4(), uuid4()
        vector = [(a, 0.9, _item("a"))]
        bm25 = [(b, 3.0, _item("b"))]
        weights = {"vector": 0.6, "bm25": 0.3}
        fused = {r.item_id: r.rrf_score for r in union_mnz_fusion([("vector", vector), ("bm25", bm25)], weights)}
        assert fused[a] == pytest.approx(1 * (0.6 / 61))
        assert fused[b] == pytest.approx(1 * (0.3 / 61))


@pytest.mark.unit
class TestTieBreak:
    def test_full_three_key_cascade(self) -> None:
        # Two items both at best_rank 1:
        #  - A: rank1 in vector only (channel_count 1, priority vector=0)
        #  - B: rank1 in graph only  (channel_count 1, priority graph=1)
        # best_rank ties (1==1), channel_count ties (1==1) -> priority breaks it:
        # vector (0) before graph (1).
        a, b = uuid4(), uuid4()
        vector = [(a, 0.5, _item("a"))]
        graph = [(b, 9.0, _item("b"))]
        order = _order(union_best_rank_fusion([("vector", vector), ("graph", graph)]))
        assert order == [a, b]

    def test_order_is_deterministic(self) -> None:
        # Identical scenarios produce identical order across runs. Key 4 of the
        # hierarchy (in-channel min-max norm, desc) is a defensive tiebreak that
        # is in fact unreachable for distinct items: keys 1-3 tying implies the
        # same best channel at the same rank, and ranks are unique within a
        # channel, so two distinct items can never reach key 4. str(item_id)
        # (key 5) is the effective final tiebreak and guarantees determinism.
        vector = _channel("v", 3)
        graph = _channel("g", 3)
        first = _order(union_best_rank_fusion([("vector", vector), ("graph", graph)]))
        second = _order(union_best_rank_fusion([("vector", vector), ("graph", graph)]))
        assert first == second


def _retriever(fusion_mode: str, **config_kwargs: Any):
    from unittest.mock import AsyncMock

    from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever

    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(fusion_mode=fusion_mode, **config_kwargs),
        storage=None,
    )


def _fr(rrf_score: float, *, vector_rank: int | None = None, bm25_rank: int | None = None) -> FusedResult:
    return FusedResult(
        item_id=uuid4(),
        item=_item("x"),
        rrf_score=rrf_score,
        vector_rank=vector_rank,
        bm25_rank=bm25_rank,
    )


@pytest.mark.unit
class TestQuotaAwareAdmission:
    def test_rescues_channel_head_dropped_by_post_boost_order(self) -> None:
        # Post-boost order (by rrf_score) buries the bm25 head b1 at position 6,
        # so the plain top_n=4 slice would drop it. Quota (top_n//2 = 2 per
        # active channel) guarantees bm25's top-2 (only b1) a window slot.
        retriever = _retriever("union_best_rank")
        v1 = _fr(0.99, vector_rank=1)
        v2 = _fr(0.98, vector_rank=2)
        v3 = _fr(0.97, vector_rank=3)
        v4 = _fr(0.96, vector_rank=4)
        v5 = _fr(0.95, vector_rank=5)
        b1 = _fr(0.10, bm25_rank=1)  # bm25 head, demoted by the recency boost
        fused = [v1, v2, v3, v4, v5, b1]

        window, remainder = retriever._select_rerank_window(fused, top_n=4)
        window_ids = {r.item_id for r in window}
        assert b1.item_id in window_ids  # rescued
        assert len(window) == 4
        # window is a partition preserving post-boost order.
        assert window + remainder == [r for r in fused if r.item_id in window_ids] + [
            r for r in fused if r.item_id not in window_ids
        ]
        assert set(_order(window)) | set(_order(remainder)) == {r.item_id for r in fused}

    def test_plain_rrf_would_drop_the_head(self) -> None:
        # Same pool under rrf: b1 is NOT rescued (proves the scenario is real).
        retriever = _retriever("rrf")
        v = [_fr(0.99 - 0.01 * i, vector_rank=i + 1) for i in range(5)]
        b1 = _fr(0.10, bm25_rank=1)
        fused = [*v, b1]
        window, _remainder = retriever._select_rerank_window(fused, top_n=4)
        assert b1.item_id not in {r.item_id for r in window}

    def test_guaranteed_set_fits_within_top_n(self) -> None:
        # n_active * floor(top_n/n_active) <= top_n, so the window is never
        # over-filled even when every item is a distinct channel head.
        retriever = _retriever("union_best_rank")
        vector = [_fr(0.9 - 0.01 * i, vector_rank=i + 1) for i in range(10)]
        bm25 = [_fr(0.5 - 0.01 * i, bm25_rank=i + 1) for i in range(10)]
        fused = [*vector, *bm25]
        window, remainder = retriever._select_rerank_window(fused, top_n=6)
        assert len(window) == 6
        assert len(window) + len(remainder) == len(fused)

    def test_no_active_channels_falls_back_to_plain_slice(self) -> None:
        # Union mode but provenance-less FusedResults (e.g. the SIMPLE path):
        # admission must degrade to the historic slice.
        retriever = _retriever("union_best_rank")
        fused = [_fr(0.9), _fr(0.8), _fr(0.7)]
        window, remainder = retriever._select_rerank_window(fused, top_n=2)
        assert window == fused[:2]
        assert remainder == fused[2:]


@pytest.mark.unit
class TestByteIdenticalDefault:
    def test_rrf_and_calibrated_admission_is_plain_slice(self) -> None:
        # The new admission path must be a byte-identical no-op for the default
        # and calibrated modes: identical objects, identical partition.
        for mode in ("rrf", "calibrated"):
            retriever = _retriever(mode)
            fused = [_fr(0.9, vector_rank=1), _fr(0.8, bm25_rank=1), _fr(0.7, vector_rank=2)]
            window, remainder = retriever._select_rerank_window(fused, top_n=2)
            assert window == fused[:2]
            assert remainder == fused[2:]

    def _chunk(self, cid: UUID):
        from khora.core.models import Chunk

        return Chunk(id=cid, namespace_id=uuid4(), document_id=uuid4(), content="x")

    def test_fuse_results_rrf_two_channel_matches_weighted_rrf_normalized(self) -> None:
        retriever = _retriever("rrf")
        v_id, g_id = uuid4(), uuid4()
        vector = [(v_id, 0.9, self._chunk(v_id)), (g_id, 0.2, self._chunk(g_id))]
        graph = [(g_id, 10.0, self._chunk(g_id))]
        got = retriever._fuse_results(
            vector_chunks=vector, graph_chunks=graph, use_normalization=True, fusion_mode="rrf"
        )
        expected = weighted_rrf_normalized(
            vector_results=vector,
            graph_results=graph,
            k=retriever._config.rrf_k,
            vector_weight=retriever._config.vector_weight,
            graph_weight=retriever._config.graph_weight,
        )
        assert _order(got) == _order(expected)
        assert [r.rrf_score for r in got] == [r.rrf_score for r in expected]
        # rrf path never populates bm25 provenance (proves union branch untaken).
        assert all(r.bm25_rank is None for r in got)

    def test_fuse_results_rrf_three_channel_leaves_bm25_provenance_none(self) -> None:
        # Under rrf the 3-channel branch (not the union branch) runs; it does not
        # back-fill bm25_rank. The union branch is the only path that would.
        retriever = _retriever("rrf")
        ids = [uuid4() for _ in range(3)]
        vector = [(ids[0], 0.9, self._chunk(ids[0]))]
        graph = [(ids[1], 5.0, self._chunk(ids[1]))]
        bm25 = [(ids[2], 3.0, self._chunk(ids[2]))]
        got = retriever._fuse_results(
            vector_chunks=vector,
            graph_chunks=graph,
            bm25_chunks=bm25,
            use_normalization=True,
            fusion_mode="rrf",
        )
        assert all(r.bm25_rank is None for r in got)
        assert {r.item_id for r in got} == set(ids)


@pytest.mark.unit
class TestFuseResultsUnionDispatch:
    def _chunk(self, cid: UUID):
        from khora.core.models import Chunk

        return Chunk(id=cid, namespace_id=uuid4(), document_id=uuid4(), content="x")

    def test_union_three_channel_dispatch_backfills_bm25(self) -> None:
        retriever = _retriever("union_best_rank")
        ids = [uuid4() for _ in range(3)]
        vector = [(ids[0], 0.9, self._chunk(ids[0]))]
        graph = [(ids[1], 5.0, self._chunk(ids[1]))]
        bm25 = [(ids[2], 3.0, self._chunk(ids[2]))]
        got = retriever._fuse_results(
            vector_chunks=vector,
            graph_chunks=graph,
            bm25_chunks=bm25,
            use_normalization=True,
            fusion_mode="union_best_rank",
        )
        by_id = {r.item_id: r for r in got}
        assert by_id[ids[2]].bm25_rank == 1  # union branch back-fills bm25
        # rank-1 block ordered by channel priority vector < graph < bm25.
        assert _order(got) == ids

    def test_union_disabled_channel_excluded(self) -> None:
        # union_rank_channels=[vector, bm25] drops the graph channel entirely.
        retriever = _retriever("union_best_rank", union_rank_channels=["vector", "bm25"])
        v_id, g_id, b_id = uuid4(), uuid4(), uuid4()
        vector = [(v_id, 0.9, self._chunk(v_id))]
        graph = [(g_id, 5.0, self._chunk(g_id))]
        bm25 = [(b_id, 3.0, self._chunk(b_id))]
        got = retriever._fuse_results(
            vector_chunks=vector,
            graph_chunks=graph,
            bm25_chunks=bm25,
            use_normalization=True,
            fusion_mode="union_best_rank",
        )
        assert {r.item_id for r in got} == {v_id, b_id}  # graph-only chunk absent

    def test_union_exclude_bm25_only_filters_lexical_singletons(self) -> None:
        retriever = _retriever("union_best_rank")
        v_id, b_id = uuid4(), uuid4()
        vector = [(v_id, 0.9, self._chunk(v_id))]
        bm25 = [(b_id, 3.0, self._chunk(b_id))]
        got = retriever._fuse_results(
            vector_chunks=vector,
            graph_chunks=[],
            bm25_chunks=bm25,
            use_normalization=True,
            exclude_bm25_only=True,
            fusion_mode="union_best_rank",
        )
        assert {r.item_id for r in got} == {v_id}  # bm25-only chunk excluded


@pytest.mark.unit
class TestConfigValidators:
    def test_defaults_are_all_channels(self) -> None:
        from khora.config.schema import QuerySettings

        qs = QuerySettings()
        assert qs.fusion_mode == "rrf"
        assert qs.union_rank_channels == ["vector", "graph", "bm25"]
        assert qs.union_rank_per_channel_limit is None

    def test_empty_channels_rejected(self) -> None:
        from pydantic import ValidationError

        from khora.config.schema import QuerySettings

        with pytest.raises(ValidationError):
            QuerySettings(union_rank_channels=[])

    def test_duplicate_channels_rejected(self) -> None:
        from pydantic import ValidationError

        from khora.config.schema import QuerySettings

        with pytest.raises(ValidationError):
            QuerySettings(union_rank_channels=["vector", "vector"])

    def test_per_channel_limit_must_be_ge_one(self) -> None:
        from pydantic import ValidationError

        from khora.config.schema import QuerySettings

        with pytest.raises(ValidationError):
            QuerySettings(union_rank_per_channel_limit=0)

    def test_new_fusion_modes_accepted(self) -> None:
        from khora.config.schema import QuerySettings

        assert QuerySettings(fusion_mode="union_best_rank").fusion_mode == "union_best_rank"
        assert QuerySettings(fusion_mode="union_mnz").fusion_mode == "union_mnz"
