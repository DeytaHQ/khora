"""Recall-result abstention-signal computation.

Pure function consumed by both the chronicle and vectorcypher engines to
produce the ``abstention_signals`` sub-dict carried in
``RecallResult.engine_info``.
"""

from __future__ import annotations

from typing import Any, Literal


def clip01(value: float) -> float:
    """Clamp ``value`` to the ``[0.0, 1.0]`` closed interval."""
    return max(0.0, min(1.0, value))


def compute_abstention_signals(
    *,
    chunk_count: int,
    top_vector_score: float,
    entity_count: int,
    min_chunks: int = 1,
    min_top_score: float = 0.3,
    combined_threshold: float = 0.5,
    weight_entities_empty: float = 0.3,
    weight_chunks_below_min: float = 0.4,
    weight_top_score_low: float = 0.3,
    mode: Literal["cosine_floor", "weighted"] = "cosine_floor",
) -> dict[str, Any]:
    """Compute passive abstention signals for downstream answer-generation.

    Args:
        chunk_count: Number of chunks returned by the engine.
        top_vector_score: **Pre-rerank, pre-fusion** raw vector cosine of the
            top-ranked semantic-channel chunk (0.0 when no chunks). Must
            NOT be the post-fusion display score because cross-encoder
            reranking compresses scores into a narrow high-side band
            even for off-topic queries (issue #809) - feeding the
            post-rerank score would make ``top_score_low`` silently
            useless. Both Chronicle and VectorCypher engines capture
            ``max_raw_cosine`` from the semantic channel for exactly
            this purpose.
        entity_count: Number of entities returned by the engine.
        min_chunks: Chunk count below which ``chunks_below_min`` fires.
        min_top_score: Top-chunk score below which ``top_score_low`` fires.
        combined_threshold: Combined-score threshold at or above which
            ``should_abstain`` is True in ``mode="weighted"``.
        weight_entities_empty: Weight of ``entities_empty`` in
            ``combined_score`` (``mode="weighted"`` decision only).
        weight_chunks_below_min: Weight of ``chunks_below_min`` in
            ``combined_score`` (``mode="weighted"`` decision only).
        weight_top_score_low: Weight of ``top_score_low`` in
            ``combined_score`` (``mode="weighted"`` decision only).
        mode: ``should_abstain`` derivation. ``"cosine_floor"`` (default)
            abstains when the topicality floor fires on its own
            (``top_score_low``) OR when retrieval came back genuinely empty
            (``chunks_empty AND entities_empty``); chunk/entity counts only
            measure retrieval liveness, not relevance, on a populated
            namespace. ``"weighted"`` is the legacy escape hatch:
            ``combined_score >= combined_threshold``. Both modes return the
            same four flags and the same ``combined_score``; only the
            ``should_abstain`` derivation differs (issue #1331).

    Returns:
        Dict with four boolean signal flags, a weighted ``combined_score``
        in [0.0, 1.0], and a ``should_abstain`` convenience flag. The flag
        set and ``combined_score`` are a documented public contract and are
        identical across modes.
    """
    entities_empty = entity_count == 0
    chunks_empty = chunk_count == 0
    chunks_below_min = chunk_count < min_chunks
    top_score_low = top_vector_score < min_top_score

    combined = (
        weight_entities_empty * float(entities_empty)
        + weight_chunks_below_min * float(chunks_below_min)
        + weight_top_score_low * float(top_score_low)
    )

    if mode == "weighted":
        should_abstain = combined >= combined_threshold
    elif mode == "cosine_floor":
        # cosine_floor (default): the topicality floor decides on its own;
        # chunk/entity COUNTS only matter in the genuinely-empty case.
        should_abstain = top_score_low or (chunks_empty and entities_empty)
    else:
        raise ValueError(f"unknown abstention mode: {mode!r}")

    return {
        "entities_empty": entities_empty,
        "chunks_empty": chunks_empty,
        "chunks_below_min": chunks_below_min,
        "top_score_low": top_score_low,
        "combined_score": combined,
        "should_abstain": should_abstain,
    }


def compute_confidence(
    *,
    top_cosine: float,
    top_score_gap: float,
    target_cosine: float = 0.5,
    target_gap: float = 0.1,
    mode: Literal["legacy", "raw_cosine"] = "legacy",
) -> float:
    """Calibrated retrieval confidence in ``[0.0, 1.0]`` (issue #1331).

    Blends the absolute top cosine (how topical the best hit is) with the
    score gap between the top two hits (how decisively the engine separates
    the winner). Both inputs are absolute cosines after #1319.

    ``mode="legacy"`` (default, unchanged):
    ``confidence = 0.8 * clip01(top_cosine / target_cosine)
    + 0.2 * clip01(top_score_gap / target_gap)``. The cosine term SATURATES
    at ``target_cosine`` (default 0.5), so a 0.5-cosine and a 0.95-cosine top
    hit read identically.

    ``mode="raw_cosine"`` (#1475): the cosine term uses the FULL [0,1] cosine
    magnitude (``0.8 * clip01(top_cosine)``) so it no longer ceilings at 0.5,
    and the caller supplies ``top_score_gap`` as the true raw-cosine gap rather
    than a post-fusion display-score gap. The gap term keeps the ``target_gap``
    saturation.
    """
    if mode == "raw_cosine":
        cosine_component = clip01(top_cosine)
    else:
        cosine_component = clip01(top_cosine / target_cosine) if target_cosine > 0 else 0.0
    gap_component = clip01(top_score_gap / target_gap) if target_gap > 0 else 0.0
    return 0.8 * cosine_component + 0.2 * gap_component
