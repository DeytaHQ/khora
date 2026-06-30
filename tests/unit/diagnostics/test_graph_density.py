"""Tests for the PPR graph-density audit (#598)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.diagnostics.graph_density import (
    GraphStats,
    _connected_components,
    _stats_from_lists,
    compute_graph_stats,
)


@pytest.mark.unit
def test_empty_namespace_returns_zeroed_stats() -> None:
    ns = uuid4()
    stats = _stats_from_lists(ns, [], [])
    assert stats.num_entities == 0
    assert stats.num_components == 0
    assert stats.meets_ppr_threshold is False


@pytest.mark.unit
def test_isolated_nodes_have_n_components_zero_degree() -> None:
    ns = uuid4()
    a, b, c = uuid4(), uuid4(), uuid4()
    stats = _stats_from_lists(ns, [a, b, c], [])
    assert stats.num_entities == 3
    assert stats.num_relationships == 0
    assert stats.mean_degree == 0.0
    assert stats.median_degree == 0.0
    # 3 singletons → 3 components. The old gate green-lit this via the
    # ≥3-components arm; the #1377 gate requires actual connectivity, so an
    # edgeless graph (median_degree 0, largest_cc_fraction 1/3) fails.
    assert stats.num_components == 3
    assert stats.largest_cc_size == 1
    assert stats.meets_ppr_threshold is False


@pytest.mark.unit
def test_chain_graph_one_component() -> None:
    ns = uuid4()
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    # a-b-c-d chain: 4 nodes, 3 edges, 1 component, mean degree 1.5
    stats = _stats_from_lists(
        ns,
        [a, b, c, d],
        [(a, b, "WORKS_FOR"), (b, c, "WORKS_FOR"), (c, d, "WORKS_FOR")],
    )
    assert stats.num_entities == 4
    assert stats.num_relationships == 3
    assert stats.num_components == 1
    assert stats.largest_cc_size == 4
    assert stats.largest_cc_fraction == 1.0
    assert stats.mean_degree == 1.5
    # Connected + semantic, but mean_degree_largest_cc (1.5) is below the
    # core-degree threshold → fails the density conjunct.
    assert stats.meets_ppr_threshold is False


@pytest.mark.unit
def test_dense_largest_cc_meets_threshold() -> None:
    """K_5 (complete graph on 5 nodes): degree 4 each.

    Then add a 6th isolated node — still only 2 components, but
    mean degree in the largest CC is 4. Add one more edge inside the
    K_5 to bump mean-degree past 5? K_5 already has mean=4. Make it K_6:
    6 nodes, every pair connected = 15 edges, every node degree 5.
    """
    ns = uuid4()
    nodes = [uuid4() for _ in range(6)]
    edges = [(nodes[i], nodes[j], "DISCOVERED") for i in range(6) for j in range(i + 1, 6)]
    stats = _stats_from_lists(ns, nodes, edges)
    assert stats.num_components == 1
    assert stats.mean_degree_largest_cc == 5.0
    assert stats.non_generic_edge_fraction == 1.0
    # Connected, fully linked, semantic edges, dense core → passes all conjuncts.
    assert stats.meets_ppr_threshold is True


@pytest.mark.unit
def test_dangling_edge_dropped_silently() -> None:
    ns = uuid4()
    a, b = uuid4(), uuid4()
    ghost = uuid4()  # not in entity_ids
    stats = _stats_from_lists(ns, [a, b], [(a, b, "WORKS_FOR"), (a, ghost, "WORKS_FOR")])
    # Only the valid edge survives; the dangling one is ignored.
    assert stats.num_relationships == 1
    assert stats.num_components == 1
    assert stats.largest_cc_size == 2


@pytest.mark.unit
def test_self_loops_ignored() -> None:
    ns = uuid4()
    a, b = uuid4(), uuid4()
    stats = _stats_from_lists(ns, [a, b], [(a, a, "WORKS_FOR"), (a, b, "WORKS_FOR")])
    # Self-loop is dropped — only a-b counts.
    assert stats.num_relationships == 1


@pytest.mark.unit
def test_two_components_below_threshold() -> None:
    """Two small triangles: 6 nodes, 6 edges, 2 components, mean_deg_in_largest_cc=2."""
    ns = uuid4()
    a, b, c, d, e, f = (uuid4() for _ in range(6))
    edges = [
        (a, b, "WORKS_FOR"),
        (b, c, "WORKS_FOR"),
        (c, a, "WORKS_FOR"),
        (d, e, "WORKS_FOR"),
        (e, f, "WORKS_FOR"),
        (f, d, "WORKS_FOR"),
    ]
    stats = _stats_from_lists(ns, [a, b, c, d, e, f], edges)
    assert stats.num_components == 2
    assert stats.largest_cc_size == 3
    assert stats.mean_degree_largest_cc == 2.0
    # largest_cc_fraction is only 0.5 and the core degree (2.0) is below the
    # threshold → fails.
    assert stats.meets_ppr_threshold is False


@pytest.mark.unit
def test_edgeless_hundred_node_graph_fails_gate() -> None:
    """#1377 misfire A: 100 entities, 0 relationships must NOT meet the gate.

    The old gate green-lit this via num_components >= 3 (each singleton is a
    component). PPR on an edgeless graph has nothing to walk.
    """
    ns = uuid4()
    nodes = [uuid4() for _ in range(100)]
    stats = _stats_from_lists(ns, nodes, [])
    assert stats.num_components == 100
    assert stats.num_relationships == 0
    assert stats.median_degree == 0.0
    assert stats.non_generic_edge_fraction == 0.0
    assert stats.meets_ppr_threshold is False


@pytest.mark.unit
def test_noise_clique_vs_semantic_clique_differ() -> None:
    """#1377 misfire B: identical topology, different edge type → different verdict.

    A dense clique of pure CO_OCCURS_WITH noise must fail; the same clique built
    from semantic DISCOVERED edges must pass. Edge type is no longer invisible.
    """
    ns = uuid4()
    nodes = [uuid4() for _ in range(6)]
    pairs = [(nodes[i], nodes[j]) for i in range(6) for j in range(i + 1, 6)]

    noise = _stats_from_lists(ns, nodes, [(a, b, "CO_OCCURS_WITH") for a, b in pairs])
    semantic = _stats_from_lists(ns, nodes, [(a, b, "DISCOVERED") for a, b in pairs])

    # Identical topology → identical structural stats.
    assert noise.mean_degree_largest_cc == semantic.mean_degree_largest_cc
    assert noise.largest_cc_fraction == semantic.largest_cc_fraction

    assert noise.non_generic_edge_fraction == 0.0
    assert semantic.non_generic_edge_fraction == 1.0
    assert noise.meets_ppr_threshold is False
    assert semantic.meets_ppr_threshold is True
    # The verdicts differ — edge quality is now a signal.
    assert noise.meets_ppr_threshold != semantic.meets_ppr_threshold


@pytest.mark.unit
def test_connected_components_finds_all() -> None:
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    adj = {a: {b}, b: {a}, c: {d}, d: {c}}
    comps = _connected_components(adj, {a, b, c, d})
    sizes = sorted(len(c) for c in comps)
    assert sizes == [2, 2]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compute_graph_stats_uses_coordinator_helpers() -> None:
    """End-to-end: compute_graph_stats() must use coordinator.list_entities /
    list_relationships (the helpers added in #587 that work on every backend)
    and never crash on an empty namespace.
    """
    ns = uuid4()
    entity = MagicMock()
    entity.id = uuid4()

    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=[entity])
    storage.list_relationships = AsyncMock(return_value=[])

    stats = await compute_graph_stats(storage, ns, entity_limit=10, relationship_limit=10)

    assert isinstance(stats, GraphStats)
    assert stats.num_entities == 1
    assert stats.num_components == 1  # the single isolated node
    storage.list_entities.assert_awaited_once_with(ns, limit=10)
    storage.list_relationships.assert_awaited_once_with(ns, limit=10)
