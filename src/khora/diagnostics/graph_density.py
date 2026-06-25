"""Graph density reporter for the PPR decision gate (Issue #598).

Measures whether a namespace's entity/relationship graph is dense enough
that Personalized PageRank would meaningfully differentiate passages
versus BFS + reciprocal-rank fusion (today's VectorCypher retrieval).

Decision criteria (#1377 revision of #598): PPR is "worth doing" only when
the graph is genuinely connected, genuinely linked, and held together by
semantic edges rather than co-occurrence noise. All four conjuncts must hold:

- ``largest_cc_fraction >= 0.5`` — the graph is mostly one connected
  component, not a pile of shards.
- ``median_degree >= 1.0`` — most entities are actually linked, so a swarm
  of singletons can't masquerade as communities.
- ``non_generic_edge_fraction >= 0.5`` — at least half the edges carry a
  semantic relationship type, not a generic ``CO_OCCURS_WITH`` /
  ``ASSOCIATED_WITH`` co-occurrence edge.
- ``mean_degree_largest_cc >= GATE_MIN_CORE_DEGREE`` — the core component is
  dense enough for PPR to differentiate.

The old gate (``num_components >= 3 OR mean_degree_largest_cc >= 5.0``)
green-lit PPR on a 0-edge graph (each singleton counted as a component) and
could not tell a pure-co-occurrence-noise clique from a semantic one (it
discarded ``relationship_type``). See #1377.

CAVEAT — capped slice: these stats are computed over a recency-ordered slice
(``list_entities`` limit 100k / ``list_relationships`` limit 1M), so they are
only trustworthy when ``num_entities < entity_limit``. Past the cap the verdict
reflects a biased sub-graph, not the real one. Fixing that needs a real
sampling strategy (deferred follow-up, see #1377 part 3).
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from khora.storage.coordinator import StorageCoordinator

# Generic / co-occurrence relationship types that carry little semantic signal.
# PPR over a graph held together only by these collapses toward co-occurrence
# noise, so they are excluded from the non-generic-edge fraction. Compared
# case-insensitively (relationship types are free-text). MENTIONED_IN is NOT
# listed: it is an Entity->Chunk edge, never returned by list_relationships.
_GENERIC_EDGE_TYPES = frozenset({"CO_OCCURS_WITH", "ASSOCIATED_WITH", "RELATES_TO", "CONNECTED_TO", "RELATED"})

# Minimum mean degree in the largest connected component for the gate's
# density conjunct. AND-combined with the connectivity/edge-quality terms
# below, so it can be less strict than the old standalone 5.0 threshold;
# 3.0 is a tunable judgment call.
GATE_MIN_CORE_DEGREE = 3.0


@dataclass(frozen=True, slots=True)
class GraphStats:
    """Per-namespace graph-density summary.

    ``meets_ppr_threshold`` is the decision flag; it AND-combines
    ``largest_cc_fraction``, ``median_degree``, ``non_generic_edge_fraction``
    and ``mean_degree_largest_cc`` (see the module docstring for the gate).
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
    # Fraction of valid edges (same edges the degree math counts) whose type is
    # NOT in _GENERIC_EDGE_TYPES; 0.0 when there are no edges.
    non_generic_edge_fraction: float
    # Decision flag — True when the namespace meets the "worth doing PPR" bar.
    meets_ppr_threshold: bool


def _build_adjacency(
    entity_ids: list[UUID],
    edges: list[tuple[UUID, UUID, str]],
) -> tuple[dict[UUID, set[UUID]], set[UUID], int, int]:
    """Build an undirected adjacency dict and a set of valid node IDs.

    Edges whose endpoints aren't in ``entity_ids`` are silently dropped — a
    real namespace can have dangling relationship rows pointing at since-
    deleted entities, and we don't want to crash the audit on those.

    Also returns the count of valid edges (valid endpoints, non-self-loop) and
    of those that are non-generic (semantic) so the caller can compute
    ``non_generic_edge_fraction`` over the exact same edge set the degree math
    counts.
    """
    valid = set(entity_ids)
    adj: dict[UUID, set[UUID]] = defaultdict(set)
    valid_edges = 0
    non_generic_edges = 0
    for src, tgt, rel_type in edges:
        if src in valid and tgt in valid and src != tgt:
            adj[src].add(tgt)
            adj[tgt].add(src)
            valid_edges += 1
            if rel_type.upper() not in _GENERIC_EDGE_TYPES:
                non_generic_edges += 1
    return adj, valid, valid_edges, non_generic_edges


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
    edges: list[tuple[UUID, UUID, str]],
) -> GraphStats:
    """Compute GraphStats from already-fetched entity ids and typed edges.

    Each edge is ``(source_entity_id, target_entity_id, relationship_type)``.
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
            non_generic_edge_fraction=0.0,
            meets_ppr_threshold=False,
        )

    adj, valid, valid_edges, non_generic_edges = _build_adjacency(entity_ids, edges)
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
    non_generic_fraction = non_generic_edges / valid_edges if valid_edges else 0.0

    meets = (
        largest_size / n >= 0.5
        and median_deg >= 1.0
        and non_generic_fraction >= 0.5
        and mean_deg_largest >= GATE_MIN_CORE_DEGREE
    )

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
        non_generic_edge_fraction=non_generic_fraction,
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

    CAVEAT: the result is computed over the recency-ordered slice these limits
    define, so it is only trustworthy when ``num_entities < entity_limit``.
    Past the cap the verdict reflects a biased sub-graph (deferred follow-up:
    a real sampling strategy, see #1377 part 3).
    """
    entities = await storage.list_entities(namespace_id, limit=entity_limit)
    relationships = await storage.list_relationships(namespace_id, limit=relationship_limit)
    entity_ids = [e.id for e in entities]
    edges = [(r.source_entity_id, r.target_entity_id, r.relationship_type) for r in relationships]
    return _stats_from_lists(namespace_id, entity_ids, edges)
