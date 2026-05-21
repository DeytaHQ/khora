"""Recall-result abstention-signal computation.

Pure function consumed by both the chronicle and vectorcypher engines to
produce the ``abstention_signals`` sub-dict carried in
``RecallResult.engine_info`` per ADR-098 §1 #24.
"""

from __future__ import annotations

from typing import Any


def compute_abstention_signals(
    *,
    chunk_count: int,
    top_chunk_score: float,
    entity_count: int,
    min_chunks: int = 1,
    min_top_score: float = 0.3,
    combined_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute passive abstention signals for downstream answer-generation.

    Args:
        chunk_count: Number of chunks returned by the engine.
        top_chunk_score: Score of the top-ranked chunk (0.0 if none).
        entity_count: Number of entities returned by the engine.
        min_chunks: Chunk count below which ``chunks_below_min`` fires.
        min_top_score: Top-chunk score below which ``top_score_low`` fires.
        combined_threshold: Combined-score threshold at or above which
            ``should_abstain`` is True.

    Returns:
        Dict with four boolean signal flags, a weighted ``combined_score``
        in [0.0, 1.0], and a ``should_abstain`` convenience flag. Shape and
        weighting match ADR-098 §1 #24 and ``ChronicleEngine._compute_abstention_signals``.
    """
    entities_empty = entity_count == 0
    chunks_empty = chunk_count == 0
    chunks_below_min = chunk_count < min_chunks
    top_score_low = top_chunk_score < min_top_score

    combined = 0.3 * float(entities_empty) + 0.4 * float(chunks_below_min) + 0.3 * float(top_score_low)

    return {
        "entities_empty": entities_empty,
        "chunks_empty": chunks_empty,
        "chunks_below_min": chunks_below_min,
        "top_score_low": top_score_low,
        "combined_score": combined,
        "should_abstain": combined >= combined_threshold,
    }
