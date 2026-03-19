"""Search quality metrics for observability.

Captures per-stage latency, result counts, score distributions,
and feature flags for every query execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from loguru import logger


@dataclass
class StageTimer:
    """Tracks start/end of a pipeline stage."""

    _start: float = 0.0
    _end: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        if self._end <= self._start:
            return 0.0
        return (self._end - self._start) * 1000


@dataclass
class SearchMetrics:
    """Metrics collected during a single query execution."""

    # Stage timers
    understanding_timer: StageTimer = field(default_factory=StageTimer)
    linking_timer: StageTimer = field(default_factory=StageTimer)
    search_timer: StageTimer = field(default_factory=StageTimer)
    fusion_timer: StageTimer = field(default_factory=StageTimer)
    reranking_timer: StageTimer = field(default_factory=StageTimer)
    total_timer: StageTimer = field(default_factory=StageTimer)

    # Multi-stage pipeline timers
    stage1_recall_timer: StageTimer = field(default_factory=StageTimer)
    stage2_normalize_timer: StageTimer = field(default_factory=StageTimer)
    stage3_filter_timer: StageTimer = field(default_factory=StageTimer)
    stage4_rerank_timer: StageTimer = field(default_factory=StageTimer)
    stage5_diversity_timer: StageTimer = field(default_factory=StageTimer)

    # Result counts per source
    vector_chunk_count: int = 0
    graph_chunk_count: int = 0
    keyword_chunk_count: int = 0
    vector_entity_count: int = 0
    graph_entity_count: int = 0

    # Post-fusion counts
    fused_chunk_count: int = 0
    fused_entity_count: int = 0
    final_chunk_count: int = 0
    final_entity_count: int = 0

    # Multi-stage pipeline counts
    stage1_candidate_count: int = 0
    stage2_normalized_count: int = 0
    stage3_filtered_count: int = 0
    stage4_reranked_count: int = 0
    stage5_final_count: int = 0

    # Multi-stage enabled flag
    multi_stage_enabled: bool = False

    # Score distributions (after fusion)
    chunk_score_min: float = 0.0
    chunk_score_max: float = 0.0
    chunk_score_mean: float = 0.0

    # Feature flags active during this query
    features: dict[str, bool] = field(default_factory=dict)

    def set_chunk_scores(self, scores: list[float]) -> None:
        """Compute score statistics from a list of fused chunk scores."""
        if not scores:
            return
        self.chunk_score_min = min(scores)
        self.chunk_score_max = max(scores)
        self.chunk_score_mean = sum(scores) / len(scores)

    def log(self) -> None:
        """Emit a structured log line with all metrics."""
        log_data = {
            "latency_total_ms": round(self.total_timer.elapsed_ms, 2),
            "latency_understanding_ms": round(self.understanding_timer.elapsed_ms, 2),
            "latency_linking_ms": round(self.linking_timer.elapsed_ms, 2),
            "latency_search_ms": round(self.search_timer.elapsed_ms, 2),
            "latency_fusion_ms": round(self.fusion_timer.elapsed_ms, 2),
            "latency_reranking_ms": round(self.reranking_timer.elapsed_ms, 2),
            "vector_chunks": self.vector_chunk_count,
            "graph_chunks": self.graph_chunk_count,
            "keyword_chunks": self.keyword_chunk_count,
            "fused_chunks": self.fused_chunk_count,
            "final_chunks": self.final_chunk_count,
            "final_entities": self.final_entity_count,
            "score_min": round(self.chunk_score_min, 4),
            "score_max": round(self.chunk_score_max, 4),
            "score_mean": round(self.chunk_score_mean, 4),
        }

        # Add multi-stage metrics if enabled
        if self.multi_stage_enabled:
            log_data.update(
                {
                    "multi_stage": True,
                    "stage1_recall_ms": round(self.stage1_recall_timer.elapsed_ms, 2),
                    "stage2_normalize_ms": round(self.stage2_normalize_timer.elapsed_ms, 2),
                    "stage3_filter_ms": round(self.stage3_filter_timer.elapsed_ms, 2),
                    "stage4_rerank_ms": round(self.stage4_rerank_timer.elapsed_ms, 2),
                    "stage5_diversity_ms": round(self.stage5_diversity_timer.elapsed_ms, 2),
                    "stage1_candidates": self.stage1_candidate_count,
                    "stage3_filtered": self.stage3_filtered_count,
                    "stage4_reranked": self.stage4_reranked_count,
                    "stage5_final": self.stage5_final_count,
                }
            )

        log_data.update(self.features)
        logger.bind(search_metrics=True).info("Search query completed", **log_data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics for inclusion in API response metadata."""
        result = {
            "latency_ms": {
                "total": round(self.total_timer.elapsed_ms, 2),
                "understanding": round(self.understanding_timer.elapsed_ms, 2),
                "linking": round(self.linking_timer.elapsed_ms, 2),
                "search": round(self.search_timer.elapsed_ms, 2),
                "fusion": round(self.fusion_timer.elapsed_ms, 2),
                "reranking": round(self.reranking_timer.elapsed_ms, 2),
            },
            "counts": {
                "vector_chunks": self.vector_chunk_count,
                "graph_chunks": self.graph_chunk_count,
                "keyword_chunks": self.keyword_chunk_count,
                "vector_entities": self.vector_entity_count,
                "graph_entities": self.graph_entity_count,
                "fused_chunks": self.fused_chunk_count,
                "fused_entities": self.fused_entity_count,
                "final_chunks": self.final_chunk_count,
                "final_entities": self.final_entity_count,
            },
            "scores": {
                "min": round(self.chunk_score_min, 4),
                "max": round(self.chunk_score_max, 4),
                "mean": round(self.chunk_score_mean, 4),
            },
            "features": self.features,
        }

        # Add multi-stage metrics if enabled
        if self.multi_stage_enabled:
            result["multi_stage"] = {
                "enabled": True,
                "latency_ms": {
                    "stage1_recall": round(self.stage1_recall_timer.elapsed_ms, 2),
                    "stage2_normalize": round(self.stage2_normalize_timer.elapsed_ms, 2),
                    "stage3_filter": round(self.stage3_filter_timer.elapsed_ms, 2),
                    "stage4_rerank": round(self.stage4_rerank_timer.elapsed_ms, 2),
                    "stage5_diversity": round(self.stage5_diversity_timer.elapsed_ms, 2),
                },
                "counts": {
                    "stage1_candidates": self.stage1_candidate_count,
                    "stage2_normalized": self.stage2_normalized_count,
                    "stage3_filtered": self.stage3_filtered_count,
                    "stage4_reranked": self.stage4_reranked_count,
                    "stage5_final": self.stage5_final_count,
                },
            }

        return result
