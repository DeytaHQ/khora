"""Graph density reporter for the PPR decision gate (Issue #598).

Measures whether a namespace's entity/relationship graph is dense enough
that Personalized PageRank would meaningfully differentiate passages
versus BFS + reciprocal-rank fusion (today's VectorCypher retrieval).

Decision criteria from #598:

- Median namespace has ≥3 connected components, **or**
- Mean degree ≥5 in the largest connected component.

If neither holds, PPR converges near-uniform and the swap from
BFS+RRF → PPR (issue #542) is not worth the complexity.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from khora.storage.coordinator import StorageCoordinator


@dataclass(frozen=True, slots=True)
class GraphStats:
    """Per-namespace graph-density summary.

    Use the bottom-three fields (``num_components``, ``largest_cc_size``,
    ``mean_degree_largest_cc``) to apply the #598 decision gate.
    """

    namespace_id: UUID
    num_entities: int  # |V|
    num_relationships: int  # |E|
    mean_degree: float  # 2|E| / |V| (undirected)
    median_degree: float
    num_components: int
    largest_cc_size: int
    largest_cc_fraction: float  # largest_cc_size / num_entities
    mean_degree_largest_cc: float
    # Decision flag — True when the namespace meets #598's "worth doing PPR" bar.
    meets_ppr_threshold: bool


def _build_adjacency(
    entity_ids: list[UUID],
    edges: list[tuple[UUID, UUID]],
) -> tuple[dict[UUID, set[UUID]], set[UUID]]:
    """Build an undirected adjacency dict and a set of valid node IDs.

    Edges whose endpoints aren't in ``entity_ids`` are silently dropped — a
    real namespace can have dangling relationship rows pointing at since-
    deleted entities, and we don't want to crash the audit on those.
    """
    valid = set(entity_ids)
    adj: dict[UUID, set[UUID]] = defaultdict(set)
    for src, tgt in edges:
        if src in valid and tgt in valid and src != tgt:
            adj[src].add(tgt)
            adj[tgt].add(src)
    return adj, valid


def _connected_components(adj: dict[UUID, set[UUID]], nodes: set[UUID]) -> list[set[UUID]]:
    """Return all connected components as sets of node IDs.

    Singleton nodes (entities with no relationships at all) form their
    own component — they count toward num_components and degrade the
    largest_cc_fraction, both of which match the #598 framing.
    """
    seen: set[UUID] = set()
    components: list[set[UUID]] = []
    for node in nodes:
        if node in seen:
            continue
        comp: set[UUID] = set()
        queue: deque[UUID] = deque([node])
        while queue:
            cur = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            for neighbor in adj.get(cur, ()):
                if neighbor not in seen:
                    queue.append(neighbor)
        components.append(comp)
    return components


def _stats_from_lists(
    namespace_id: UUID,
    entity_ids: list[UUID],
    edges: list[tuple[UUID, UUID]],
) -> GraphStats:
    """Compute GraphStats from already-fetched entity ids and edges.

    Separated from the storage-coupled :func:`compute_graph_stats` so the
    same logic can be tested without spinning up a backend.
    """
    n = len(entity_ids)
    if n == 0:
        return GraphStats(
            namespace_id=namespace_id,
            num_entities=0,
            num_relationships=0,
            mean_degree=0.0,
            median_degree=0.0,
            num_components=0,
            largest_cc_size=0,
            largest_cc_fraction=0.0,
            mean_degree_largest_cc=0.0,
            meets_ppr_threshold=False,
        )

    adj, valid = _build_adjacency(entity_ids, edges)
    degrees = [len(adj.get(node, ())) for node in entity_ids]
    components = _connected_components(adj, valid)
    largest = max(components, key=len) if components else set()
    largest_size = len(largest)

    if largest_size > 0:
        largest_degrees = [len(adj.get(n_, ())) for n_ in largest]
        # 2|E_largest_cc| / |V_largest_cc| — equivalent to mean(degrees in subgraph)
        mean_deg_largest = sum(largest_degrees) / largest_size
    else:
        mean_deg_largest = 0.0

    num_relationships = sum(degrees) // 2  # each edge counted twice
    mean_deg = sum(degrees) / n
    median_deg = statistics.median(degrees)

    meets = len(components) >= 3 or mean_deg_largest >= 5.0

    return GraphStats(
        namespace_id=namespace_id,
        num_entities=n,
        num_relationships=num_relationships,
        mean_degree=mean_deg,
        median_degree=median_deg,
        num_components=len(components),
        largest_cc_size=largest_size,
        largest_cc_fraction=largest_size / n,
        mean_degree_largest_cc=mean_deg_largest,
        meets_ppr_threshold=meets,
    )


async def compute_graph_stats(
    storage: StorageCoordinator,
    namespace_id: UUID,
    *,
    entity_limit: int = 100_000,
    relationship_limit: int = 1_000_000,
) -> GraphStats:
    """Compute graph-density stats for one namespace.

    Uses the coordinator's ``list_entities`` / ``list_relationships``
    helpers (added in #587) so this works on every backend that exposes
    the entities/relationships tables — including chronicle+PG-only.

    Limits default high so a typical workload returns the full graph in
    one round-trip; tune down for spot-checks on huge namespaces.
    """
    entities = await storage.list_entities(namespace_id, limit=entity_limit)
    relationships = await storage.list_relationships(namespace_id, limit=relationship_limit)
    entity_ids = [e.id for e in entities]
    edges = [(r.source_entity_id, r.target_entity_id) for r in relationships]
    return _stats_from_lists(namespace_id, entity_ids, edges)
