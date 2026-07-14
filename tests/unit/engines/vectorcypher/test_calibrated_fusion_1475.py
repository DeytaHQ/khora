"""Score-calibrated convex fusion (#1475).

Rank-only weighted RRF buries a lone strong-cosine vector hit below weak-cosine
chunks that merely co-occur across channels, because summed ``weight/(k+rank)``
sees only positions, never magnitudes. #1441 made the true raw cosine available
on every chunk, unblocking a magnitude-aware alternative fusion (behind the
``query.fusion_mode="calibrated"`` flag).

These tests assert:

1. The hermetic burial proof: a 0.95-cosine lone vector hit ranks BELOW
   0.26-cosine graph-co-occurrence chunks under RRF, and ABOVE them under
   calibrated fusion.
2. Calibrated OFF (``fusion_mode="rrf"``) is byte-identical to the RRF path.
3. The convex-combination math and the N-list (3-channel) variant.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.engines.vectorcypher.fusion import (
    FusedResult,
    score_calibrated_fusion,
    score_calibrated_fusion_nlist,
    weighted_rrf_normalized,
)


def _item(name: str) -> Any:
    return type("Item", (), {"name": name})()


def _order(results: list[FusedResult]) -> list[UUID]:
    return [r.item_id for r in results]


@pytest.mark.unit
class TestBurialProof:
    """The lone strong-cosine winner must not be buried under co-occurrence."""

    def _scenario(self) -> tuple[list, list, UUID, UUID, UUID]:
        # V: 0.95-cosine lone vector hit, ABSENT from the graph channel.
        # G1/G2: weak 0.26/0.24-cosine chunks that ALSO co-occur in the graph
        # channel (mentions-scale scores 10/8). This is the multi-channel
        # presence that rank fusion over-rewards.
        v_id, g1_id, g2_id = uuid4(), uuid4(), uuid4()
        vector_results = [
            (v_id, 0.95, _item("V")),
            (g1_id, 0.26, _item("G1")),
            (g2_id, 0.24, _item("G2")),
        ]
        graph_results = [
            (g1_id, 10.0, _item("G1")),
            (g2_id, 8.0, _item("G2")),
        ]
        return vector_results, graph_results, v_id, g1_id, g2_id

    def test_rrf_buries_the_lone_vector_winner(self) -> None:
        vector_results, graph_results, v_id, g1_id, g2_id = self._scenario()
        order = _order(
            weighted_rrf_normalized(
                vector_results=vector_results,
                graph_results=graph_results,
                vector_weight=0.6,
                graph_weight=0.4,
            )
        )
        # The multi-channel weak chunk G1 outranks the lone strong hit V:
        # that is the burial the ticket targets.
        assert order.index(g1_id) < order.index(v_id)

    def test_calibrated_surfaces_the_lone_vector_winner(self) -> None:
        vector_results, graph_results, v_id, g1_id, g2_id = self._scenario()
        fused = score_calibrated_fusion(
            vector_results=vector_results,
            graph_results=graph_results,
            vector_weight=0.6,
            graph_weight=0.4,
        )
        order = _order(fused)
        # Calibrated fusion weighs magnitude: the 0.95 hit leads, above BOTH
        # 0.26/0.24-cosine graph-co-occurrence chunks.
        assert order[0] == v_id
        assert order.index(v_id) < order.index(g1_id)
        assert order.index(v_id) < order.index(g2_id)
        # And the top score is exactly vector_weight * 1.0 (channel-normalized).
        assert fused[0].rrf_score == pytest.approx(0.6)


@pytest.mark.unit
class TestFuseResultsDispatch:
    """``_fuse_results`` dispatches on ``fusion_mode`` and threads raw cosines."""

    def _retriever(self, fusion_mode: str):
        from unittest.mock import AsyncMock

        from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever

        return VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=None,
            embedder=AsyncMock(),
            config=RetrieverConfig(fusion_mode=fusion_mode),
            storage=None,
        )

    def _chunk(self, cid: UUID):
        from khora.core.models import Chunk

        return Chunk(id=cid, namespace_id=uuid4(), document_id=uuid4(), content="x")

    def _scenario(self):
        v_id, g1_id, g2_id = uuid4(), uuid4(), uuid4()
        vector_chunks = [
            (v_id, 0.95, self._chunk(v_id)),
            (g1_id, 0.26, self._chunk(g1_id)),
            (g2_id, 0.24, self._chunk(g2_id)),
        ]
        graph_chunks = [
            (g1_id, 10.0, self._chunk(g1_id)),
            (g2_id, 8.0, self._chunk(g2_id)),
        ]
        return vector_chunks, graph_chunks, v_id, g1_id, g2_id

    def test_default_mode_matches_weighted_rrf_normalized_order(self) -> None:
        vector_chunks, graph_chunks, *_ = self._scenario()
        retriever = self._retriever("rrf")
        got = _order(
            retriever._fuse_results(
                vector_chunks=vector_chunks,
                graph_chunks=graph_chunks,
                use_normalization=True,
                fusion_mode="rrf",
            )
        )
        expected = _order(
            weighted_rrf_normalized(
                vector_results=vector_chunks,
                graph_results=graph_chunks,
                k=retriever._config.rrf_k,
                vector_weight=retriever._config.vector_weight,
                graph_weight=retriever._config.graph_weight,
            )
        )
        assert got == expected

    def test_calibrated_mode_surfaces_winner(self) -> None:
        vector_chunks, graph_chunks, v_id, g1_id, _ = self._scenario()
        retriever = self._retriever("calibrated")
        got = _order(
            retriever._fuse_results(
                vector_chunks=vector_chunks,
                graph_chunks=graph_chunks,
                use_normalization=True,
                fusion_mode="calibrated",
                raw_cosine_by_id={cid: score for cid, score, _ in vector_chunks},
            )
        )
        assert got[0] == v_id
        assert got.index(v_id) < got.index(g1_id)


@pytest.mark.unit
class TestConvexMath:
    def test_two_channel_convex_combination(self) -> None:
        a, b = uuid4(), uuid4()
        # a: vector top (norm 1.0) + graph bottom (norm 0.0).
        # b: vector bottom (norm 0.0) + graph top (norm 1.0).
        vector_results = [(a, 0.9, _item("a")), (b, 0.1, _item("b"))]
        graph_results = [(b, 5.0, _item("b")), (a, 1.0, _item("a"))]
        fused = {
            r.item_id: r.rrf_score
            for r in score_calibrated_fusion(vector_results, graph_results, vector_weight=0.7, graph_weight=0.3)
        }
        assert fused[a] == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)
        assert fused[b] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)

    def test_single_element_channel_normalizes_to_one(self) -> None:
        # A lone vector hit min-max normalizes to 1.0 (max == min branch).
        v = uuid4()
        fused = score_calibrated_fusion([(v, 0.42, _item("v"))], [], vector_weight=0.6, graph_weight=0.4)
        assert fused[0].rrf_score == pytest.approx(0.6)

    def test_provenance_backfilled(self) -> None:
        a, b = uuid4(), uuid4()
        fused = {r.item_id: r for r in score_calibrated_fusion([(a, 0.9, _item("a"))], [(b, 5.0, _item("b"))])}
        assert fused[a].vector_rank == 1 and fused[a].graph_rank is None
        assert fused[b].graph_rank == 1 and fused[b].vector_rank is None
        assert fused[a].vector_score == pytest.approx(0.9)
        assert fused[b].graph_score == pytest.approx(5.0)

    def test_empty_inputs(self) -> None:
        assert score_calibrated_fusion([], []) == []
        assert score_calibrated_fusion_nlist([]) == []


@pytest.mark.unit
class TestNList:
    def test_three_channel_convex(self) -> None:
        a, b, c = uuid4(), uuid4(), uuid4()
        vector = [(a, 0.9, _item("a"))]
        graph = [(b, 3.0, _item("b"))]
        bm25 = [(c, 7.0, _item("c"))]
        fused = {
            r.item_id: r.rrf_score for r in score_calibrated_fusion_nlist([(vector, 0.5), (graph, 0.3), (bm25, 0.2)])
        }
        # Each is the lone (hence norm=1.0) member of its channel → weight.
        assert fused[a] == pytest.approx(0.5)
        assert fused[b] == pytest.approx(0.3)
        assert fused[c] == pytest.approx(0.2)

    def test_multichannel_membership_accumulates(self) -> None:
        shared = uuid4()
        vector = [(shared, 0.9, _item("s"))]
        graph = [(shared, 4.0, _item("s"))]
        fused = score_calibrated_fusion_nlist([(vector, 0.6), (graph, 0.4)])
        # Present in both single-element channels (each norm 1.0) → 0.6 + 0.4.
        assert fused[0].rrf_score == pytest.approx(1.0)
