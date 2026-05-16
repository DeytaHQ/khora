"""Query-time Personalized PageRank retrieval (HippoRAG 2 style).

Replaces the BFS + reciprocal-rank-fusion graph channel with a PPR walk
over the namespace entity graph, seeded from the entities the query
resolves to.  Chunks are then scored by the PR-weighted sum over the
entities that mention them.

Default OFF: gated by ``RetrieverConfig.enable_ppr_retrieval`` and
``KhoraConfig.query.enable_ppr_retrieval``.  When entry entities or the
entity graph are empty the helpers return an empty list so the caller
falls back to the vector-only path.

References:
- HippoRAG 2 (Feb 2025): https://arxiv.org/abs/2502.14802
- ``khora._accel.pagerank`` already accepts a personalization vector
  (Rust kernel; pure-Python fallback when the extension is absent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from khora._accel import pagerank as _pagerank
from khora.core.models import Chunk, ChunkMetadata, Entity
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from khora.storage.coordinator import StorageCoordinator


# Cap how many entities the PPR graph is built over.  Khora deployments
# typically have ~50-500 entities per namespace (see #598 density audit);
# this is a safety upper bound to avoid pathological graphs blowing up
# memory at query time.
_MAX_ENTITIES_FOR_PPR = 5000
_MAX_RELATIONSHIPS_FOR_PPR = 50_000


@dataclass(frozen=True, slots=True)
class _PPRGraph:
    """Indexed entity graph ready for PPR.

    ``entity_id_to_idx`` maps ``Entity.id`` to its row index in the PR
    score vector.  ``edges`` is the bidirectional, weighted edge list
    fed to ``_accel.pagerank``.
    """

    entities: list[Entity]
    entity_id_to_idx: dict[UUID, int]
    edges: list[tuple[int, int, float]]


def build_personalization_vector(
    entry_entities: list[tuple[UUID, float]],
    entity_id_to_idx: dict[UUID, int],
) -> list[float]:
    """Project entry-entity scores onto a length-``n`` personalization vector.

    Entry entities not present in the graph (e.g. dangling refs after a
    delete) are silently dropped.  When no entry entity survives the
    projection the returned vector is all zeros — ``_accel.pagerank``
    treats that as a signal to fall back to uniform teleport (i.e. the
    PR collapses to standard PageRank), which is the right behaviour
    when there's nothing to personalize on.
    """
    n = len(entity_id_to_idx)
    vec = [0.0] * n
    for eid, score in entry_entities:
        idx = entity_id_to_idx.get(eid)
        if idx is None:
            continue
        # Use the recall similarity (or 1.0 if zero/negative) as the
        # seed weight.  Negatives would be clipped by the Rust kernel
        # but we'd rather not feed it junk in the first place.
        vec[idx] = max(float(score), 1e-6)
    return vec


def build_ppr_graph(
    entities: list[Entity],
    relationships: list[tuple[UUID, UUID, float]],
) -> _PPRGraph:
    """Index entities and edges into the dense form expected by ``pagerank``.

    Self-loops and edges with endpoints outside the entity set are
    dropped (matches the dangling-edge handling in
    ``khora.diagnostics.graph_density``).  Each (src, dst) relationship
    becomes two directed edges (src→dst and dst→src) so PPR can flow
    in both directions — relationships in khora are conceptually
    undirected for retrieval purposes.
    """
    entity_id_to_idx: dict[UUID, int] = {e.id: i for i, e in enumerate(entities)}
    edges: list[tuple[int, int, float]] = []
    for src_id, tgt_id, weight in relationships:
        s = entity_id_to_idx.get(src_id)
        t = entity_id_to_idx.get(tgt_id)
        if s is None or t is None or s == t:
            continue
        w = float(weight) if weight > 0 else 1.0
        edges.append((s, t, w))
        edges.append((t, s, w))
    return _PPRGraph(entities=entities, entity_id_to_idx=entity_id_to_idx, edges=edges)


def score_chunks_via_ppr(
    pr_scores: list[float],
    entities: list[Entity],
    *,
    top_entities: int,
    chunk_similarity: dict[UUID, float] | None = None,
) -> list[tuple[UUID, float]]:
    """Score chunks by the PR-weighted sum over their mentioning entities.

    HippoRAG 2 scores a passage by summing the personalized-PR mass of
    every entity it mentions.  Khora records the entity-chunk linkage
    in ``Entity.source_chunk_ids``; that's the data we use here so the
    function works on every backend (Neo4j-less stacks included).

    Args:
        pr_scores: PR score per entity, indexed identically to ``entities``.
        entities: Entities indexed identically to ``pr_scores``.
        top_entities: Take only the top-K PR-scored entities — keeps the
            chunk fan-out bounded on large graphs (HippoRAG 2 §4.3).
        chunk_similarity: Optional ``chunk_id → cosine similarity to
            query`` map.  When provided, the chunk's final score is
            ``ppr_mass * (1 + similarity)`` — small multiplicative
            blend, never overwhelms the PR signal.

    Returns:
        ``[(chunk_id, score), ...]`` sorted by score descending.
    """
    if not pr_scores or not entities:
        return []
    # Rank entities by PR; take top-K so the chunk fan-out is bounded.
    ranked = sorted(range(len(entities)), key=lambda i: pr_scores[i], reverse=True)
    ranked = ranked[: max(1, top_entities)]

    chunk_score: dict[UUID, float] = {}
    for idx in ranked:
        mass = pr_scores[idx]
        if mass <= 0.0:
            continue
        for cid in entities[idx].source_chunk_ids:
            chunk_score[cid] = chunk_score.get(cid, 0.0) + mass

    if chunk_similarity:
        for cid, base in list(chunk_score.items()):
            sim = chunk_similarity.get(cid, 0.0)
            chunk_score[cid] = base * (1.0 + max(0.0, sim))

    return sorted(chunk_score.items(), key=lambda kv: kv[1], reverse=True)


async def ppr_retrieve_chunks(
    *,
    storage: StorageCoordinator,
    namespace_id: UUID,
    entry_entities: list[tuple[UUID, float]],
    damping: float,
    max_iter: int,
    tol: float,
    top_entities: int,
    chunk_similarity: dict[UUID, float] | None = None,
    limit: int,
) -> tuple[list[tuple[UUID, float, Chunk]], dict[UUID, float]]:
    """Run query-time PPR over the namespace graph and score chunks.

    Returns ``(chunk_results, entity_scores)`` where ``chunk_results`` is
    in the shape ``_vectorcypher_retrieve`` already passes to the
    fusion layer — list of ``(chunk_id, score, Chunk)`` tuples — and
    ``entity_scores`` is the per-entity PR map (kept for telemetry +
    downstream entity ranking).  On any degenerate input (no entities,
    no entry entities, all-zero personalization) returns ``([], {})``
    so the caller falls back to the vector-only path.
    """
    with trace_span(
        "khora.vectorcypher.ppr_retrieve",
        namespace_id=str(namespace_id),
        entry_count=len(entry_entities),
    ) as span:
        if not entry_entities:
            span.set_attribute("fallback_reason", "no_entry_entities")
            return [], {}

        entities = await storage.list_entities(namespace_id, limit=_MAX_ENTITIES_FOR_PPR)
        if not entities:
            span.set_attribute("fallback_reason", "empty_entity_graph")
            return [], {}

        relationships = await storage.list_relationships(namespace_id, limit=_MAX_RELATIONSHIPS_FOR_PPR)
        edge_triples: list[tuple[UUID, UUID, float]] = [
            (r.source_entity_id, r.target_entity_id, r.weight) for r in relationships
        ]

        graph = build_ppr_graph(entities, edge_triples)
        personalization = build_personalization_vector(entry_entities, graph.entity_id_to_idx)

        if sum(personalization) <= 0.0:
            # None of the query entities matched a graph node — fall back.
            span.set_attribute("fallback_reason", "no_seed_overlap")
            return [], {}

        n = len(graph.entities)
        span.set_attribute("entity_count", n)
        span.set_attribute("edge_count", len(graph.edges))

        pr_scores = _pagerank(
            n=n,
            edges=graph.edges,
            damping=damping,
            max_iter=max_iter,
            tol=tol,
            personalization=personalization,
        )

        entity_score_map: dict[UUID, float] = {graph.entities[i].id: pr_scores[i] for i in range(n)}

        ranked_chunks = score_chunks_via_ppr(
            pr_scores,
            graph.entities,
            top_entities=top_entities,
            chunk_similarity=chunk_similarity,
        )
        # Cap to ``limit`` chunk IDs before fetching, then hydrate to Chunks.
        chunk_ids = [cid for cid, _ in ranked_chunks[:limit]]
        span.set_attribute("chunk_count", len(chunk_ids))
        if not chunk_ids:
            return [], entity_score_map

        chunks_map = await storage.get_chunks_batch(chunk_ids)
        score_map = dict(ranked_chunks)
        results: list[tuple[UUID, float, Chunk]] = []
        for cid in chunk_ids:
            chunk = chunks_map.get(cid)
            if chunk is None:
                continue
            # Normalize the chunk shape to what fusion expects (matching
            # _vector_search_chunks / _fetch_chunks_from_entities).
            results.append(
                (
                    cid,
                    score_map[cid],
                    Chunk(
                        id=chunk.id,
                        namespace_id=chunk.namespace_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata=ChunkMetadata(
                            custom={
                                "occurred_at": (
                                    chunk.metadata.custom.get("occurred_at")
                                    if isinstance(chunk.metadata, ChunkMetadata)
                                    else None
                                ),
                                "ppr_score": score_map[cid],
                                **(
                                    chunk.metadata.custom
                                    if isinstance(chunk.metadata, ChunkMetadata)
                                    else (chunk.metadata or {})
                                ),
                            }
                        ),
                        created_at=getattr(chunk, "created_at", None),
                    ),
                )
            )
        return results, entity_score_map


__all__ = [
    "build_personalization_vector",
    "build_ppr_graph",
    "score_chunks_via_ppr",
    "ppr_retrieve_chunks",
]
