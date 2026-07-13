"""Unit tests for ``khora.query.degree_stats`` (#1477).

Covers the pure degree-histogram builder and the epoch-invalidated cache that
feeds frontier-budgeted adaptive depth.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.query.degree_stats import DegreeStatsCache, build_degree_stats

pytestmark = pytest.mark.unit


class TestBuildDegreeStats:
    def test_empty_graph(self) -> None:
        stats = build_degree_stats([], [])
        assert stats.num_entities == 0
        assert stats.mean_degree == 0.0
        assert stats.max_degree == 0
        assert stats.degree_by_entity == {}

    def test_star_graph_degrees(self) -> None:
        # One hub connected to 4 leaves: hub degree 4, each leaf degree 1.
        hub = uuid4()
        leaves = [uuid4() for _ in range(4)]
        ids = [hub, *leaves]
        edges = [(hub, leaf) for leaf in leaves]

        stats = build_degree_stats(ids, edges)

        assert stats.num_entities == 5
        assert stats.degree_by_entity[hub] == 4
        assert all(stats.degree_by_entity[leaf] == 1 for leaf in leaves)
        assert stats.max_degree == 4
        # 2|E| / |V| = 8 / 5.
        assert stats.mean_degree == pytest.approx(8 / 5)
        assert stats.median_degree == 1

    def test_self_loops_and_dangling_edges_dropped(self) -> None:
        a, b, ghost = uuid4(), uuid4(), uuid4()
        # self-loop on a, real a-b edge, and an edge to a ghost not in the id set.
        edges = [(a, a), (a, b), (a, ghost)]
        stats = build_degree_stats([a, b], edges)
        # Only the a-b edge survives: each has degree 1.
        assert stats.degree_by_entity[a] == 1
        assert stats.degree_by_entity[b] == 1

    def test_duplicate_edges_count_once(self) -> None:
        a, b = uuid4(), uuid4()
        # Two parallel a-b edges collapse to a single distinct neighbor.
        stats = build_degree_stats([a, b], [(a, b), (b, a), (a, b)])
        assert stats.degree_by_entity[a] == 1
        assert stats.degree_by_entity[b] == 1

    def test_seed_degree_sum_uses_mean_for_unknown(self) -> None:
        a, b = uuid4(), uuid4()
        stats = build_degree_stats([a, b], [(a, b)])  # both degree 1, mean 1.0
        unknown = uuid4()
        # a (deg 1) + unknown (charged mean 1.0) = 2.0.
        assert stats.seed_degree_sum([a, unknown]) == pytest.approx(2.0)

    def test_seed_degree_sum_empty(self) -> None:
        a = uuid4()
        stats = build_degree_stats([a], [])
        assert stats.seed_degree_sum([]) == 0.0


class TestDegreeStatsCache:
    def test_miss_then_hit_under_same_epoch(self) -> None:
        cache = DegreeStatsCache()
        ns = uuid4()
        stats = build_degree_stats([uuid4()], [])

        assert cache.get(ns, epoch=1) is None
        cache.set(ns, epoch=1, stats=stats)
        assert cache.get(ns, epoch=1) is stats

    def test_epoch_bump_invalidates(self) -> None:
        # A write bumps the epoch; the stale entry must read as a miss so the
        # caller recomputes against the new graph state.
        cache = DegreeStatsCache()
        ns = uuid4()
        stats = build_degree_stats([uuid4()], [])
        cache.set(ns, epoch=1, stats=stats)

        assert cache.get(ns, epoch=2) is None  # epoch advanced -> miss
        assert cache.get(ns, epoch=1) is stats  # old epoch still resolves pre-overwrite

    def test_per_namespace_isolation(self) -> None:
        cache = DegreeStatsCache()
        ns_a, ns_b = uuid4(), uuid4()
        stats_a = build_degree_stats([uuid4()], [])
        cache.set(ns_a, epoch=1, stats=stats_a)
        assert cache.get(ns_b, epoch=1) is None

    def test_lru_eviction(self) -> None:
        cache = DegreeStatsCache(max_namespaces=2)
        ns1, ns2, ns3 = uuid4(), uuid4(), uuid4()
        s = build_degree_stats([uuid4()], [])
        cache.set(ns1, epoch=1, stats=s)
        cache.set(ns2, epoch=1, stats=s)
        cache.set(ns3, epoch=1, stats=s)  # evicts ns1 (oldest)
        assert cache.get(ns1, epoch=1) is None
        assert cache.get(ns2, epoch=1) is s
        assert cache.get(ns3, epoch=1) is s

    def test_clear(self) -> None:
        cache = DegreeStatsCache()
        ns = uuid4()
        cache.set(ns, epoch=1, stats=build_degree_stats([uuid4()], []))
        cache.clear()
        assert cache.get(ns, epoch=1) is None
