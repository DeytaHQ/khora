"""Shadow-scoring A/B harness (#1479).

A thin, observe-only seam that runs a CANDIDATE ranking alongside the live
(INCUMBENT) ranking on the same recall and records the divergence, without ever
changing the results the caller receives. It lets a retrieval-quality change be
A/B'd in production (per-namespace, behind ``KHORA_QUERY_SHADOW_SCORING``)
before it is promoted to the live path.

Design constraints (see #1479):

* **Observe-only.** The returned ``RecallResult.chunks`` are ALWAYS the
  incumbent's order. This module never reorders the live result - it only
  computes a shadow order and emits a comparison under
  ``engine_info["shadow_scoring"]`` (``engine_info`` keys are free-form and NOT
  telemetry-contract-gated, so no span/metric is added here).
* **Zero-cost when OFF.** The engine guards every call to this module behind
  ``if config.query.shadow_scoring``. When the flag is False (the default),
  nothing in this module runs.
* **Minimal seam.** It taps the retriever's ALREADY-scored candidate list
  (the ``list[(chunk, score)]`` the engine hands to ``RecallChunk``
  construction). It does NOT refactor the retriever's channel/scoring stages -
  that restructuring is #1480. Until then the only candidate strategy that can
  be expressed over the incumbent's final scored candidates is a re-ordering of
  them; a genuinely independent candidate scoring pipeline (its own fusion /
  boost stages) is deferred to #1480. See ``build_candidate_order``.

The divergence report is intentionally small and JSON-serializable: a rank
correlation (Spearman rho over the shared candidates), top-k overlap (Jaccard of
the top-k document sets), and the list of chunks whose rank moved the most.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from khora.core.models import Chunk

__all__ = [
    "build_candidate_order",
    "compute_shadow_report",
]

# How many "biggest mover" chunks to surface in the report. Bounded so the
# report stays small even on a large candidate pool.
_MAX_MOVERS = 10
# Default k for the top-k overlap metric when the caller does not pin one.
_DEFAULT_TOPK = 10


def build_candidate_order(
    scored_candidates: list[tuple[Chunk, float]],
    *,
    strategy: str,
) -> list[tuple[Chunk, float]]:
    """Produce a CANDIDATE ordering of the incumbent's scored candidates.

    ``scored_candidates`` is the incumbent's final ordered ``(chunk, score)``
    list - the same list the engine turns into ``RecallChunk`` objects. The
    incumbent order is the list order itself (fusion + boosts + rerank);
    ``score`` is the absolute relevance value, NOT the sort key (#1433).

    Strategies (deliberately small until #1480 lands a pluggable candidate
    pipeline):

    * ``"score_sort"`` - re-order strictly by descending ``score``. This is the
      exact #1433/#1463-class divergence: sorting hybrid chunks by ``.score``
      floats graph-only chunks (score >= 1.0) above vector-matched chunks
      (score < 1.0), demoting the true top hit. Shipping it as the default
      candidate makes the harness observably non-trivial on any hybrid recall
      and doubles as a live tripwire for that regression class.
    * ``"identity"`` - return the incumbent order unchanged (rho = 1.0, overlap
      = 1.0). Useful as a self-check / no-op candidate.

    The returned list is a re-ordering of the SAME ``(chunk, score)`` tuples;
    no chunk is added or dropped, so the shadow set is always comparable to the
    incumbent set.
    """
    if strategy == "identity":
        return list(scored_candidates)
    if strategy == "score_sort":
        # Stable sort by descending score. Ties keep incumbent relative order,
        # so the candidate differs from the incumbent ONLY where scores
        # disagree with the incumbent rank - which is precisely the signal.
        return sorted(scored_candidates, key=lambda cs: cs[1], reverse=True)
    raise ValueError(
        f"unknown shadow-scoring candidate strategy {strategy!r}; expected one of 'score_sort', 'identity'"
    )


def _spearman_rho(incumbent_ids: list[UUID], candidate_ids: list[UUID]) -> float | None:
    """Spearman rank correlation between two orderings of the same id set.

    Returns ``None`` when it is undefined (fewer than 2 shared items). ``1.0``
    means identical order, ``-1.0`` fully reversed.

    Pure-Python (no numpy/accel dependency): the candidate pool is bounded by
    the recall limit / over-fetch window, so this is a handful of elements.
    """
    candidate_set = set(candidate_ids)
    common = [cid for cid in incumbent_ids if cid in candidate_set]
    n = len(common)
    if n < 2:
        return None
    inc_rank = {cid: i for i, cid in enumerate(incumbent_ids)}
    cand_rank = {cid: i for i, cid in enumerate(candidate_ids)}
    d2 = sum((inc_rank[cid] - cand_rank[cid]) ** 2 for cid in common)
    # Standard Spearman rho = 1 - 6*sum(d^2) / (n*(n^2-1)). No ties within
    # either ranking (each id appears once), so the simple form is exact.
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _topk_overlap(
    incumbent_ids: list[UUID],
    candidate_ids: list[UUID],
    *,
    k: int,
) -> float:
    """Jaccard overlap of the top-``k`` id sets. 1.0 = identical top-k."""
    top_inc = set(incumbent_ids[:k])
    top_cand = set(candidate_ids[:k])
    if not top_inc and not top_cand:
        return 1.0
    union = top_inc | top_cand
    return len(top_inc & top_cand) / len(union)


def compute_shadow_report(
    scored_candidates: list[tuple[Chunk, float]],
    candidate_order: list[tuple[Chunk, float]],
    *,
    strategy: str,
    limit: int,
) -> dict[str, Any]:
    """Compare the incumbent order to a candidate order; return a small report.

    ``scored_candidates`` is the incumbent order; ``candidate_order`` is the
    output of :func:`build_candidate_order`. Both contain the same chunks. The
    report is JSON-serializable and safe to place directly under
    ``engine_info["shadow_scoring"]``.

    Report shape::

        {
          "strategy": "score_sort",
          "candidate_count": 42,
          "spearman_rho": 0.83 | None,     # rank correlation over shared ids
          "topk": 10,
          "topk_overlap": 0.7,             # Jaccard of top-k CHUNK id sets
          "topk_doc_overlap": 0.8,         # Jaccard of top-k DOCUMENT id sets
          "moved": [                       # biggest rank movers (<= _MAX_MOVERS)
            {"chunk_id": "...", "document_id": "...",
             "incumbent_rank": 0, "candidate_rank": 5, "delta": 5,
             "score": 0.0164},
            ...
          ],
          "identical": false,              # true iff order is byte-identical
        }

    ``moved`` sorts by absolute rank delta descending, then by incumbent rank
    for a stable order; only chunks that actually moved are listed.
    """
    k = min(limit, _DEFAULT_TOPK) if limit > 0 else _DEFAULT_TOPK

    inc_ids = [chunk.id for chunk, _ in scored_candidates]
    cand_ids = [chunk.id for chunk, _ in candidate_order]

    inc_docs = [chunk.document_id for chunk, _ in scored_candidates]
    cand_docs = [chunk.document_id for chunk, _ in candidate_order]

    cand_rank = {cid: i for i, cid in enumerate(cand_ids)}
    score_by_id = {chunk.id: score for chunk, score in scored_candidates}
    doc_by_id = {chunk.id: chunk.document_id for chunk, _ in scored_candidates}

    movers: list[dict[str, Any]] = []
    for inc_pos, cid in enumerate(inc_ids):
        cand_pos = cand_rank.get(cid)
        if cand_pos is None or cand_pos == inc_pos:
            continue
        movers.append(
            {
                "chunk_id": str(cid),
                "document_id": str(doc_by_id.get(cid)),
                "incumbent_rank": inc_pos,
                "candidate_rank": cand_pos,
                "delta": cand_pos - inc_pos,
                "score": round(float(score_by_id.get(cid, 0.0)), 6),
            }
        )
    movers.sort(key=lambda m: (-abs(m["delta"]), m["incumbent_rank"]))

    return {
        "strategy": strategy,
        "candidate_count": len(scored_candidates),
        "spearman_rho": _spearman_rho(inc_ids, cand_ids),
        "topk": k,
        "topk_overlap": round(_topk_overlap(inc_ids, cand_ids, k=k), 6),
        "topk_doc_overlap": round(_topk_overlap(inc_docs, cand_docs, k=k), 6),
        "moved": movers[:_MAX_MOVERS],
        "identical": inc_ids == cand_ids,
    }
