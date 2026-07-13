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

import asyncio
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger

from khora._accel import pagerank as _pagerank
from khora.core.diagnostics import Degradation
from khora.core.models import Chunk, Entity, Relationship
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from khora.storage.coordinator import StorageCoordinator


# Cap how many entities the PPR graph is built over.  Khora deployments
# typically have ~50-500 entities per namespace (see #598 density audit);
# this is a safety upper bound to avoid pathological graphs blowing up
# memory at query time.
_MAX_ENTITIES_FOR_PPR = 5000
_MAX_RELATIONSHIPS_FOR_PPR = 50_000


# Bounded LRU cache of the query-INDEPENDENT base graph slice
# (``list_entities`` / ``list_relationships``), keyed on
# ``(namespace_id, write_epoch)`` (#1476). Repeated queries on a slowly-changing
# namespace reuse the slice and skip the two DB round-trips, while the walk (seed
# personalization + PPR) still runs per query. The write-epoch is the #1469
# recall-cache epoch, bumped on ANY namespace write, so a post-write query
# captures a fresh epoch and misses — a stale slice is never served (it shares
# the #1469 epoch-eviction semantics: an epoch reset only after >1000 namespaces
# evict the epoch map, by which point the tiny slice LRU has long dropped the
# entry). Small by design: each entry may hold up to _MAX_ENTITIES_FOR_PPR
# entities + _MAX_RELATIONSHIPS_FOR_PPR relationships, so the cap trades memory
# for hit rate. Entries are read-only downstream (augmentation builds fresh
# lists), so sharing one across concurrent recalls is safe.
_GRAPH_SLICE_CACHE_MAX = 8
_graph_slice_cache: OrderedDict[tuple[UUID, int], tuple[list[Entity], list[Relationship]]] = OrderedDict()
_graph_slice_lock = threading.Lock()


def _graph_slice_cache_get(key: tuple[UUID, int]) -> tuple[list[Entity], list[Relationship]] | None:
    with _graph_slice_lock:
        hit = _graph_slice_cache.get(key)
        if hit is not None:
            _graph_slice_cache.move_to_end(key)
        return hit


def _graph_slice_cache_put(key: tuple[UUID, int], value: tuple[list[Entity], list[Relationship]]) -> None:
    with _graph_slice_lock:
        _graph_slice_cache[key] = value
        _graph_slice_cache.move_to_end(key)
        while len(_graph_slice_cache) > _GRAPH_SLICE_CACHE_MAX:
            _graph_slice_cache.popitem(last=False)


def _clear_graph_slice_cache() -> None:
    """Drop every cached base slice (test hook)."""
    with _graph_slice_lock:
        _graph_slice_cache.clear()


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


def _record_ppr_degradation(
    out_degradations: list[Degradation] | None,
    *,
    reason: str,
    detail: str,
) -> None:
    """Append an ADR-001 ``Degradation`` for a silently-dropped PPR channel.

    The graph channel falling back to vector-only is not an error — recall
    still returns chunks — so this logs at ``WARNING`` and records a
    structured degradation rather than raising. Mirrors the
    ``vectorcypher.version_filter`` precedent. No metric counter is emitted
    (would trip the telemetry-contract drift gate); the ``Degradation``
    record plus the span ``fallback_reason`` attribute satisfy ADR-001.
    """
    logger.warning("PPR graph channel returned nothing ({}); falling back to vector-only", reason)
    if out_degradations is not None:
        out_degradations.append(
            Degradation(
                component="vectorcypher.ppr",
                reason=reason,
                detail=detail,
            )
        )


async def _augment_with_seed_neighborhood(
    *,
    storage: StorageCoordinator,
    namespace_id: UUID,
    entry_entities: list[tuple[UUID, float]],
    entities: list[Entity],
    relationships: list[Relationship],
    neighborhood_per_seed_limit: int,
    max_neighborhood_entities: int,
) -> tuple[list[Entity], list[Relationship]]:
    """Merge the query seeds + their 1-hop neighborhood into an at-cap slice.

    Only called when the global slice hit a cap and may have excluded the
    seeds (#1373). Guarantees every resolvable ``entry_entity`` survives into
    the entity set so ``build_personalization_vector`` cannot sum to zero on a
    populated namespace. ``get_entities_batch`` has a pgvector fallback (seeds
    present even on graph-less stacks); ``get_entity_relationships`` returns
    ``[]`` when no graph backend is wired — fine, the global slice still covers
    small/medium namespaces and isolated seeds get teleport mass.

    ``max_neighborhood_entities`` bounds how far augmentation may *grow* the
    set; the effective bound is ``max(max_neighborhood_entities, len(slice))``
    so the base slice (the multi-hop backbone) is never shrunk below what PPR
    would have walked without augmentation.
    """
    # Preserve order while de-duplicating seed ids.
    seed_ids = list(dict.fromkeys(eid for eid, _ in entry_entities))

    # Gather the 1-hop neighborhood of each seed (graph backends only; graph-less
    # returns [] per seed) so we can pull in the *neighbor* entities too — an edge
    # only survives build_ppr_graph if both endpoints are in the entity set, so a
    # neighborhood edge whose far end is outside the slice would otherwise be
    # dropped as dangling, isolating the seed (it would still get teleport mass,
    # but the 1-hop mass flow would be lost).
    #
    # return_exceptions=True so a single seed's transient graph error degrades to
    # "no neighborhood for that seed" rather than aborting the whole augmentation
    # (and recall) — the seed still survives via the batch fetch below + teleport
    # mass. Matches the module's "degrades, never crashes" contract.
    gathered = await asyncio.gather(
        *(
            storage.get_entity_relationships(
                sid,
                namespace_id=namespace_id,
                direction="both",
                limit=neighborhood_per_seed_limit,
            )
            for sid in seed_ids
        ),
        return_exceptions=True,
    )
    neighborhoods: list[list[Relationship]] = []
    for sid, result in zip(seed_ids, gathered, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "PPR seed neighborhood fetch failed for {} ({}); skipping its 1-hop edges",
                sid,
                type(result).__name__,
            )
            neighborhoods.append([])
        else:
            neighborhoods.append(result)

    # Entities to batch-fetch: the seeds (guarantees the #1373 invariant) plus the
    # far endpoints of their edges (so the neighborhood contributes to mass flow).
    slice_ids = {e.id for e in entities}
    fetch_ids = list(seed_ids)
    seen_fetch = set(seed_ids)
    for neighborhood in neighborhoods:
        for rel in neighborhood:
            for endpoint in (rel.source_entity_id, rel.target_entity_id):
                if endpoint in seen_fetch or endpoint in slice_ids:
                    continue
                seen_fetch.add(endpoint)
                fetch_ids.append(endpoint)

    fetched = await storage.get_entities_batch(fetch_ids, namespace_id=namespace_id)

    # Merge into the entity set, de-duplicated by Entity.id. Seeds first, then the
    # rest of the fetched neighbors, then the slice — so the trim below keeps the
    # seeds when the augmented set overflows the bound.
    by_id: dict[UUID, Entity] = {}
    for sid in seed_ids:
        ent = fetched.get(sid)
        if ent is not None:
            by_id[ent.id] = ent
    for ent in fetched.values():
        by_id.setdefault(ent.id, ent)
    for ent in entities:
        by_id.setdefault(ent.id, ent)

    # Bound how far augmentation may grow the set, but never shrink it below the
    # base slice — the global slice IS the multi-hop backbone, so re-capping it
    # smaller than _MAX_ENTITIES_FOR_PPR would silently truncate PPR. The bound
    # therefore only caps the *augmentation growth* (seeds + neighbors) on top of
    # the slice; effective_bound >= len(slice) always.
    effective_bound = max(max_neighborhood_entities, len(entities))
    if len(by_id) > effective_bound:
        # Keep seeds first, then fill (fetched neighbors + slice) up to the bound.
        trimmed: dict[UUID, Entity] = {}
        for sid in seed_ids:
            ent = by_id.get(sid)
            if ent is not None:
                trimmed[ent.id] = ent
        for eid, ent in by_id.items():
            if len(trimmed) >= effective_bound:
                break
            trimmed.setdefault(eid, ent)
        by_id = trimmed

    merged_entities = list(by_id.values())

    seen_rel_ids: set[UUID] = set()
    merged_relationships = list(relationships)
    for rel in relationships:
        seen_rel_ids.add(rel.id)
    for neighborhood in neighborhoods:
        for rel in neighborhood:
            if rel.id in seen_rel_ids:
                continue
            seen_rel_ids.add(rel.id)
            merged_relationships.append(rel)

    return merged_entities, merged_relationships


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
    neighborhood_per_seed_limit: int = 64,
    max_neighborhood_entities: int = 2000,
    early_stop_patience: int = 3,
    early_stop_margin: int = 10,
    graph_cache_epoch: int | None = None,
    out_degradations: list[Degradation] | None = None,
) -> tuple[list[tuple[UUID, float, Chunk]], dict[UUID, float]]:
    """Run query-time PPR over the namespace graph and score chunks.

    Returns ``(chunk_results, entity_scores)`` where ``chunk_results`` is
    in the shape ``_vectorcypher_retrieve`` already passes to the
    fusion layer — list of ``(chunk_id, score, Chunk)`` tuples — and
    ``entity_scores`` is the per-entity PR map (kept for telemetry +
    downstream entity ranking).  On any degenerate input (no entities,
    no entry entities, all-zero personalization) returns ``([], {})``
    so the caller falls back to the vector-only path.

    The PPR graph is built from the global namespace slice
    (``list_entities`` / ``list_relationships`` capped at
    ``_MAX_ENTITIES_FOR_PPR`` / ``_MAX_RELATIONSHIPS_FOR_PPR``).  HippoRAG-2
    PPR needs the *whole* graph for multi-hop mass flow, so on small and
    medium namespaces (below the caps) the slice is used verbatim.  When the
    slice hits a cap it may have excluded the query's seed entities (the
    entity slice is ordered ``BY name``), which would make
    ``build_personalization_vector`` sum to zero and silently drop the graph
    channel (#1373).  In that case the slice is *augmented* with the seed
    entities and their 1-hop neighborhood so every resolvable seed survives
    into the graph; the global slice still provides the multi-hop backbone.

    Args:
        neighborhood_per_seed_limit: Max relationships fetched per seed when
            augmenting an at-cap slice (``get_entity_relationships`` ``limit``).
        max_neighborhood_entities: Upper bound on how far augmentation may grow
            the entity set; the effective bound is
            ``max(max_neighborhood_entities, len(slice))`` so the base slice is
            never shrunk. Seeds are kept first when trimming.
        early_stop_patience: Top-k rank-stability early-stop patience (#1476).
            The PPR power iteration halts once the top ``top_entities +
            early_stop_margin`` entity ordering is unchanged for this many
            consecutive iterations instead of running to global-L1 convergence
            (~2-4x fewer iterations on production graph shapes, top-``top_entities``
            byte-identical). ``0`` disables the early-stop (legacy global-L1 only).
        early_stop_margin: Extra entities beyond ``top_entities`` tracked for the
            early-stop stability check, so a node just outside the retrieved set
            cannot climb in after the walk halts. Inert when ``early_stop_patience``
            is ``0``.
        graph_cache_epoch: When provided, the query-independent base graph slice
            (``list_entities`` / ``list_relationships``) is cached keyed on
            ``(namespace_id, graph_cache_epoch)`` (#1476), so repeated queries on
            a slowly-changing namespace skip the two DB round-trips. The epoch is
            the #1469 recall-cache write-epoch (bumped on any namespace write), so
            a stale slice is never served. ``None`` disables the cache (every
            recall re-fetches — the pre-#1476 behaviour).
        out_degradations: When provided, a structured :class:`Degradation`
            (ADR-001) is appended whenever the graph channel returns nothing on
            a genuine degenerate condition (no seed overlap / still-empty graph
            channel after augmentation), so the silent drop is observable.
    """
    with trace_span(
        "khora.vectorcypher.ppr_retrieve",
        namespace_id=str(namespace_id),
        entry_count=len(entry_entities),
    ) as span:
        if not entry_entities:
            span.set_attribute("fallback_reason", "no_entry_entities")
            return [], {}

        # #1476: reuse the cached query-independent base slice when the namespace
        # write-epoch is unchanged, skipping both DB round-trips. The empty-graph
        # short-circuit only runs on a miss (the cache never holds an empty slice).
        cache_key = (namespace_id, graph_cache_epoch) if graph_cache_epoch is not None else None
        cached_slice = _graph_slice_cache_get(cache_key) if cache_key is not None else None
        if cached_slice is not None:
            entities, relationships = cached_slice
            span.set_attribute("graph_cache_hit", True)
        else:
            entities = await storage.list_entities(namespace_id, limit=_MAX_ENTITIES_FOR_PPR)
            if not entities:
                span.set_attribute("fallback_reason", "empty_entity_graph")
                return [], {}

            relationships = await storage.list_relationships(namespace_id, limit=_MAX_RELATIONSHIPS_FOR_PPR)
            if cache_key is not None:
                _graph_slice_cache_put(cache_key, (entities, relationships))
            span.set_attribute("graph_cache_hit", False)

        # #1373: the global slice is ordered query-independently (entities BY
        # name, relationships by created_at DESC) and capped. Below the caps the
        # slice is the whole namespace, so PPR walks the full graph (HippoRAG-2
        # multi-hop mass flow) — behavior is byte-identical to pre-#1373, no
        # extra round-trips. At a cap the slice may have excluded the query's
        # seeds, which would zero the personalization vector and silently drop
        # the graph channel; augment with the seeds + their 1-hop neighborhood
        # so every resolvable seed survives. The global slice still provides the
        # multi-hop backbone; isolated seeds get teleport mass.
        if len(entities) >= _MAX_ENTITIES_FOR_PPR or len(relationships) >= _MAX_RELATIONSHIPS_FOR_PPR:
            entities, relationships = await _augment_with_seed_neighborhood(
                storage=storage,
                namespace_id=namespace_id,
                entry_entities=entry_entities,
                entities=entities,
                relationships=relationships,
                neighborhood_per_seed_limit=neighborhood_per_seed_limit,
                max_neighborhood_entities=max_neighborhood_entities,
            )
            span.set_attribute("seed_augmented", True)

        edge_triples: list[tuple[UUID, UUID, float]] = [
            (r.source_entity_id, r.target_entity_id, r.weight) for r in relationships
        ]

        graph = build_ppr_graph(entities, edge_triples)
        personalization = build_personalization_vector(entry_entities, graph.entity_id_to_idx)

        if sum(personalization) <= 0.0:
            # None of the query entities matched a graph node — fall back.
            # Even after seed augmentation this can happen on a genuinely
            # degenerate namespace (e.g. the seeds were deleted between vector
            # resolution and the graph read). Record an ADR-001 degradation so
            # the silently-dropped graph channel is observable (#1373).
            span.set_attribute("fallback_reason", "no_seed_overlap")
            _record_ppr_degradation(
                out_degradations,
                reason="no_seed_overlap",
                detail=(
                    "no query seed entity survived into the PPR graph after seed "
                    "augmentation; graph channel returned nothing — recall continues "
                    "on the vector-only path"
                ),
            )
            return [], {}

        n = len(graph.entities)
        span.set_attribute("entity_count", n)
        span.set_attribute("edge_count", len(graph.edges))

        # #1476: top-k rank-stability early-stop. Track the top
        # ``top_entities + margin`` ordering; the retrieved set (top
        # ``top_entities``) stays byte-identical to the full-iteration result
        # (guarded by a parity test) while the walk halts ~2-4x sooner. Disabled
        # (rank_k=None) when patience is 0 → legacy global-L1 behaviour.
        rank_k = (top_entities + max(0, early_stop_margin)) if early_stop_patience > 0 else None
        span.set_attribute("early_stop_rank_k", rank_k if rank_k is not None else -1)

        pr_scores = _pagerank(
            n=n,
            edges=graph.edges,
            damping=damping,
            max_iter=max_iter,
            tol=tol,
            personalization=personalization,
            rank_k=rank_k,
            stable_iters=early_stop_patience,
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
            # PPR ran but no scored entity carried a source chunk (e.g.
            # graph-less + isolated seeds with no chunk linkage). The graph
            # channel still returns nothing, so record an ADR-001 degradation
            # so the empty channel is observable rather than silent (#1373).
            span.set_attribute("fallback_reason", "empty_graph_channel")
            _record_ppr_degradation(
                out_degradations,
                reason="empty_graph_channel",
                detail=(
                    "PPR scored entities but none carried a source chunk; graph "
                    "channel returned no chunks — recall continues on the "
                    "vector-only path"
                ),
            )
            return [], entity_score_map

        chunks_map = await storage.get_chunks_batch(chunk_ids, namespace_id=namespace_id)
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
                        metadata={
                            "occurred_at": (
                                chunk.metadata.get("occurred_at") if isinstance(chunk.metadata, dict) else None
                            ),
                            "ppr_score": score_map[cid],
                            **(chunk.metadata if isinstance(chunk.metadata, dict) else {}),
                        },
                        created_at=getattr(chunk, "created_at", None),
                        occurred_at=chunk.occurred_at,
                        source_timestamp=chunk.source_timestamp,
                    ),
                )
            )
        if not results:
            # PPR scored chunk ids but hydration returned nothing — the chunk
            # store does not have the rows the entity graph referenced (the
            # #1372 silent-drop symptom: graph and chunk stores diverged). The
            # graph channel returns nothing despite a non-empty candidate set,
            # so record an ADR-001 degradation rather than dropping silently.
            span.set_attribute("fallback_reason", "chunk_hydration_empty")
            _record_ppr_degradation(
                out_degradations,
                reason="chunk_hydration_empty",
                detail=(
                    f"PPR scored {len(chunk_ids)} chunk ids but hydration returned 0 "
                    "results — chunk store / entity-graph mismatch; recall continues "
                    "on the vector-only path"
                ),
            )
        return results, entity_score_map


__all__ = [
    "build_personalization_vector",
    "build_ppr_graph",
    "score_chunks_via_ppr",
    "ppr_retrieve_chunks",
]
