"""Per-namespace degree statistics for frontier-budgeted adaptive depth (#1477).

The old adaptive-depth rule keyed traversal depth on entry-entity COUNT alone
(a two-threshold step function). On power-law graphs the sign is wrong for hub
seeds: a query anchored to two high-degree entities gets DEEPER traversal
exactly when its one-hop frontier will explode, while a query anchored to many
low-degree entities gets capped SHALLOW when deeper would be cheap.

This module supplies the missing signal - the actual degree of the seed
entities - so the depth rule can predict frontier size and pick a depth that
stays under a budget instead of stepping on seed count.

Design
------
* :class:`DegreeStats` is the per-namespace summary: the per-entity degree map
  (the "histogram" data) plus the mean degree (the graph's branching factor).
  The per-entity map is what distinguishes a hub seed from a low-degree seed;
  the mean is the fallback degree for a seed not in the map (e.g. dropped past
  the ``list_relationships`` cap) and the per-hop branching factor.
* :class:`DegreeStatsCache` caches one ``DegreeStats`` per namespace, keyed on
  the same monotonic per-namespace write-epoch the #1469 recall cache uses. A
  write bumps the epoch, so the next recall recomputes; between writes the
  stats are reused and never recomputed per recall. Building the stats is the
  caller's job (it needs an async ``list_relationships`` round-trip); this cache
  only stores and epoch-invalidates.
"""

from __future__ import annotations

import statistics
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DegreeStats:
    """Per-namespace degree summary used to predict BFS frontier growth.

    ``degree_by_entity`` maps an entity id to its undirected degree (number of
    distinct neighbors). ``mean_degree`` is ``2|E| / |V|`` over the same edge
    set - the graph's average branching factor and the fallback degree for a
    seed absent from the map.
    """

    num_entities: int
    mean_degree: float
    median_degree: float
    max_degree: int
    degree_by_entity: dict[UUID, int]

    def seed_degree_sum(self, seed_ids: list[UUID]) -> float:
        """Sum of the seeds' degrees, using ``mean_degree`` for unknown seeds.

        This is the predicted size of the one-hop BFS frontier: the union of
        the seeds' neighborhoods. Summing (rather than unioning) slightly
        over-counts shared neighbors, which is the conservative direction for a
        budget - it never under-predicts an explosion.
        """
        if not seed_ids:
            return 0.0
        return sum(float(self.degree_by_entity.get(sid, self.mean_degree)) for sid in seed_ids)


def build_degree_stats(
    entity_ids: list[UUID],
    edges: list[tuple[UUID, UUID]],
) -> DegreeStats:
    """Compute :class:`DegreeStats` from entity ids and undirected edges.

    Self-loops and edges whose endpoints are not both in ``entity_ids`` are
    dropped (a namespace can carry dangling relationship rows pointing at
    since-deleted entities). Degree is the count of DISTINCT neighbors, matching
    how a BFS frontier dedups.
    """
    n = len(entity_ids)
    if n == 0:
        return DegreeStats(
            num_entities=0,
            mean_degree=0.0,
            median_degree=0.0,
            max_degree=0,
            degree_by_entity={},
        )

    valid = set(entity_ids)
    adj: dict[UUID, set[UUID]] = defaultdict(set)
    for src, tgt in edges:
        if src in valid and tgt in valid and src != tgt:
            adj[src].add(tgt)
            adj[tgt].add(src)

    degree_by_entity = {eid: len(adj.get(eid, ())) for eid in entity_ids}
    degrees = list(degree_by_entity.values())
    return DegreeStats(
        num_entities=n,
        mean_degree=sum(degrees) / n,
        median_degree=statistics.median(degrees),
        max_degree=max(degrees),
        degree_by_entity=degree_by_entity,
    )


class DegreeStatsCache:
    """One :class:`DegreeStats` per namespace, epoch-invalidated (#1469 signal).

    ``get`` returns the cached stats only when they were stored under the
    namespace's current write-epoch; a stale (pre-write) entry is treated as a
    miss so the caller recomputes. The cache never fetches on its own - the
    caller builds the stats from an async ``list_relationships`` round-trip and
    hands them to :meth:`set`. Bounded by an LRU cap on namespace count.
    """

    def __init__(self, *, max_namespaces: int = 256) -> None:
        self._max = max(int(max_namespaces), 1)
        self._store: OrderedDict[UUID, tuple[int, DegreeStats]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, namespace_id: UUID, epoch: int) -> DegreeStats | None:
        """Return cached stats for the namespace iff stored under ``epoch``."""
        with self._lock:
            entry = self._store.get(namespace_id)
            if entry is None:
                return None
            stored_epoch, stats = entry
            if stored_epoch != epoch:
                # Stale: a write bumped the epoch since these stats were built.
                return None
            self._store.move_to_end(namespace_id)
            return stats

    def set(self, namespace_id: UUID, epoch: int, stats: DegreeStats) -> None:
        """Cache ``stats`` for the namespace under ``epoch`` (LRU-evicting)."""
        with self._lock:
            self._store[namespace_id] = (epoch, stats)
            self._store.move_to_end(namespace_id)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
