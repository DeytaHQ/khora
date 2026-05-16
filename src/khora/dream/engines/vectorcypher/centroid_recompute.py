"""Vectorcypher centroid embedding recompute (#660, Phase 2.2 of #649).

For each cluster of entities proposed for merge (see the Phase 2.1 dedupe
op), this planner picks how the post-merge canonical embedding should be
produced. The decision is per-cluster:

* **centroid** — every pair of names is within ``centroid_lev_threshold``
  Levenshtein distance (variants of the same surface form, e.g.
  ``"OpenAI"`` / ``"Open AI"``). Compute the weighted mean of the
  cluster's embeddings (weights = ``mention_count``) and L2-renormalize
  via :func:`khora._accel.normalize_embeddings_batch`.
* **re_embed** — names are lexically distant but semantically aligned
  (e.g. ``"IBM"`` / ``"International Business Machines"``). Plan a
  re-embedding of the canonical name (highest ``mention_count``); the
  actual embedding call is deferred to apply mode in v0.15.
* **skip_multimodal** — intra-cluster pairwise cosine drops below
  ``centroid_min_intra_cluster_cosine``. The cluster spans more than one
  concept; the merge itself is the bug. Emit a finding, plan nothing.

The op is dry-run only in v0.14 — ``mode="apply"`` raises
:class:`NotImplementedError`. Apply lands in v0.15 (see #649 Phase 4 /
#668).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

# Note: `from rapidfuzz.distance import Levenshtein` is deferred into
# `_max_pairwise_lev_distance` below. rapidfuzz ships with the `[accel]`
# optional extra, and importing it at module top would break the
# examples-smoke CI job (which installs per-adapter extras only).
from khora import _accel
from khora.dream.plan import DreamOp, OpKind
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from khora.core.models.entity import Entity
    from khora.storage.coordinator import StorageCoordinator


_PHASE = "mutation_plan"

# Default embedding model used in the re_embed plan output when the
# caller didn't supply one. Mirrors :attr:`LLMSettings.embedding_model`.
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


async def plan_vectorcypher_centroid_recompute(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    merge_clusters: list[list[UUID]],
    mode: str = "dry-run",
    lev_threshold: int = 2,
    min_intra_cluster_cosine: float = 0.88,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> tuple[DreamOp, ...]:
    """Plan a canonical post-merge embedding for each merge cluster.

    Args:
        namespace_id: Stable namespace identifier.
        coordinator: Storage coordinator. Used to load entity rows for
            each cluster member.
        merge_clusters: Lists of entity ids that the Phase 2.1 dedupe op
            (or any other source) proposes to merge. One :class:`DreamOp`
            is emitted per cluster.
        mode: ``"dry-run"`` only in v0.14. ``"apply"`` raises
            :class:`NotImplementedError` — apply mode lands in v0.15
            (#649 Phase 4 / #668).
        lev_threshold: Maximum pairwise Levenshtein distance among names
            for the centroid branch. Default 2.
        min_intra_cluster_cosine: Minimum pairwise cosine within the
            cluster's embeddings. Below this the cluster is judged
            multimodal. Default 0.88.
        embedding_model: Model identifier recorded in the plan output for
            ``re_embed`` clusters. The embedding call itself is deferred
            to apply mode.

    Returns:
        One :class:`DreamOp` per cluster. ``decision`` is one of
        ``"centroid"``, ``"re_embed"``, or ``"skip_multimodal"``. Each
        op's ``inputs`` carries ``{cluster_size, member_ids,
        canonical_entity_id}``; ``outputs`` carries either
        ``{new_embedding_vector, source_count}`` (centroid) or
        ``{new_embedding_text, embedding_model, source_count}``
        (re_embed). ``skip_multimodal`` ops have empty outputs.

    Raises:
        NotImplementedError: When ``mode="apply"`` — that path lands in
            v0.15.
    """
    if mode != "dry-run":
        raise NotImplementedError("apply mode lands in v0.15 — see #649 phase 4 / #668")

    ops: list[DreamOp] = []
    for cluster_ids in merge_clusters:
        ops.append(
            await _plan_cluster(
                namespace_id=namespace_id,
                coordinator=coordinator,
                cluster_ids=cluster_ids,
                lev_threshold=lev_threshold,
                min_intra_cluster_cosine=min_intra_cluster_cosine,
                embedding_model=embedding_model,
            )
        )
    return tuple(ops)


async def _plan_cluster(
    *,
    namespace_id: UUID,
    coordinator: StorageCoordinator,
    cluster_ids: list[UUID],
    lev_threshold: int,
    min_intra_cluster_cosine: float,
    embedding_model: str,
) -> DreamOp:
    op_id = uuid4()
    started_at_wall = _utcnow()
    t0 = time.perf_counter()

    with trace_span(
        "khora.dream.vectorcypher.centroid_recompute",
        op_id=str(op_id),
        namespace_id=str(namespace_id),
        phase=_PHASE,
        cluster_size=len(cluster_ids),
    ) as span:
        members = await _load_members(coordinator, namespace_id, cluster_ids)

        if len(members) < 2:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            span.set_attribute("decision", "skip_singleton")
            return DreamOp(
                op_id=op_id,
                phase=_PHASE,
                op_type=OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
                inputs=({"cluster_size": len(cluster_ids), "member_ids": [str(i) for i in cluster_ids]},),
                outputs=(),
                decision="skip_singleton",
                rationale="Cluster has fewer than 2 resolvable members.",
                started_at=started_at_wall,
                duration_ms=duration_ms,
                namespace_id=namespace_id,
            )

        canonical = max(members, key=lambda e: e.mention_count)
        names = [m.name for m in members]
        embeddings = [m.embedding for m in members if m.embedding]

        min_cosine = _min_pairwise_cosine(embeddings) if len(embeddings) >= 2 else 1.0
        max_lev = _max_pairwise_lev_distance(names)

        span.set_attribute("min_pairwise_cosine", float(min_cosine))
        span.set_attribute("max_pairwise_lev", max_lev)

        base_inputs: dict[str, Any] = {
            "cluster_size": len(members),
            "member_ids": [str(m.id) for m in members],
            "canonical_entity_id": str(canonical.id),
        }

        if min_cosine < min_intra_cluster_cosine:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            span.set_attribute("decision", "skip_multimodal")
            return DreamOp(
                op_id=op_id,
                phase=_PHASE,
                op_type=OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
                inputs=(base_inputs,),
                outputs=(),
                decision="skip_multimodal",
                rationale=(
                    f"Intra-cluster cosine {min_cosine:.4f} < {min_intra_cluster_cosine:.4f}; cluster is multimodal."
                ),
                started_at=started_at_wall,
                duration_ms=duration_ms,
                namespace_id=namespace_id,
            )

        if max_lev <= lev_threshold:
            new_vec = _weighted_centroid(embeddings, [m.mention_count for m in members if m.embedding])
            duration_ms = (time.perf_counter() - t0) * 1000.0
            span.set_attribute("decision", "centroid")
            return DreamOp(
                op_id=op_id,
                phase=_PHASE,
                op_type=OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
                inputs=(base_inputs,),
                outputs=(
                    {
                        "new_embedding_vector": new_vec,
                        "source_count": len(embeddings),
                    },
                ),
                decision="centroid",
                rationale=(
                    f"Max pairwise Levenshtein {max_lev} <= {lev_threshold}; "
                    f"weighted centroid over {len(embeddings)} embeddings."
                ),
                started_at=started_at_wall,
                duration_ms=duration_ms,
                namespace_id=namespace_id,
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        span.set_attribute("decision", "re_embed")
        return DreamOp(
            op_id=op_id,
            phase=_PHASE,
            op_type=OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
            inputs=(base_inputs,),
            outputs=(
                {
                    "new_embedding_text": canonical.name,
                    "embedding_model": embedding_model,
                    "source_count": len(members),
                },
            ),
            decision="re_embed",
            rationale=(
                f"Max pairwise Levenshtein {max_lev} > {lev_threshold} but "
                f"intra-cluster cosine {min_cosine:.4f} >= "
                f"{min_intra_cluster_cosine:.4f}; re-embed canonical name."
            ),
            started_at=started_at_wall,
            duration_ms=duration_ms,
            namespace_id=namespace_id,
        )


async def _load_members(
    coordinator: StorageCoordinator,
    namespace_id: UUID,
    cluster_ids: list[UUID],
) -> list[Entity]:
    """Return only the entities that actually live in this namespace."""
    wanted = set(cluster_ids)
    entities = await coordinator.list_entities(namespace_id, limit=100_000)
    return [e for e in entities if e.id in wanted]


def _min_pairwise_cosine(embeddings: list[list[float]]) -> float:
    """Min pairwise cosine over a list of pre-normalised embeddings.

    Falls back to 1.0 (trivially safe) when there are fewer than two
    embeddings to compare.
    """
    n = len(embeddings)
    if n < 2:
        return 1.0
    # ``pairwise_cosine_above_threshold`` returns only pairs >= threshold;
    # we want the minimum pair, so pass threshold=-1.0 to get them all
    # and pick the floor.
    pairs = _accel.pairwise_cosine_above_threshold(embeddings, threshold=-1.0)
    if not pairs:
        return 1.0
    return min(score for _, _, score in pairs)


def _max_pairwise_lev_distance(names: list[str]) -> int:
    """Largest Levenshtein distance among any two names in the cluster."""
    if len(names) < 2:
        return 0
    from rapidfuzz.distance import Levenshtein

    worst = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = Levenshtein.distance(names[i], names[j])
            if d > worst:
                worst = d
    return worst


def _weighted_centroid(
    embeddings: list[list[float]],
    weights: list[int],
) -> list[float]:
    """Weighted mean of ``embeddings`` then L2-renormalize.

    Uses :func:`khora._accel.normalize_embeddings_batch` so the result
    matches the rest of the ingest pipeline's normalisation contract
    (pre-normalised vectors, dot == cosine).
    """
    dim = len(embeddings[0])
    total_weight = float(sum(weights)) or 1.0
    mean = [0.0] * dim
    for vec, w in zip(embeddings, weights, strict=True):
        if w <= 0:
            continue
        scale = w / total_weight
        for k, v in enumerate(vec):
            mean[k] += scale * v
    return _accel.normalize_embeddings_batch([mean])[0]


def _utcnow():
    from datetime import UTC, datetime

    return datetime.now(UTC)
