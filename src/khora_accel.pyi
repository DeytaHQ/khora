"""Type stubs for the khora-accel Rust extension module (PyO3).

This stub allows type checkers (ty, mypy) to resolve imports when the
native extension is not installed (e.g. in CI without a Rust toolchain).
"""

from __future__ import annotations

from typing import Any

# -- BM25 index ---------------------------------------------------------------

class RustBM25Index:
    def __init__(self) -> None: ...
    def add_document(self, doc_id: int, tokens: list[str]) -> None: ...
    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]: ...
    def document_count(self) -> int: ...

# -- Cosine similarity --------------------------------------------------------

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float: ...
def batch_cosine_similarity(
    query: Any,  # numpy ndarray
    candidates: Any,  # numpy ndarray
    threshold: float = 0.0,
) -> list[tuple[int, float]]: ...
def pairwise_cosine_above_threshold(
    embeddings: Any,  # numpy ndarray
    threshold: float,
) -> list[tuple[int, int, float]]: ...
def normalize_embeddings_batch(vectors: list[list[float]]) -> list[list[float]]: ...
def batch_dot_product(query: Any, candidates: Any, threshold: float = 0.0) -> list[tuple[int, float]]: ...

# -- String similarity --------------------------------------------------------

def levenshtein_similarity(s1: str, s2: str) -> float: ...
def sequence_match_ratio(s1: str, s2: str) -> float: ...
def batch_levenshtein(query: str, candidates: list[str], threshold: float = 0.0) -> list[tuple[int, float]]: ...
def batch_sequence_match(query: str, candidates: list[str], threshold: float = 0.0) -> list[tuple[int, float]]: ...
def normalize_entity_name(name: str) -> str: ...
def normalize_entity_names_batch(names: list[str]) -> list[str]: ...

# -- BM25 / search ------------------------------------------------------------
# (RustBM25Index class above)

# -- PageRank ------------------------------------------------------------------

def pagerank(
    n: int,
    edges: list[tuple[int, int, float]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> list[float]: ...
def build_chunk_edges(
    n_chunks: int,
    keyword_chunk_ids: list[list[int]],
    idf_scores: list[float],
) -> list[tuple[int, int, float]]: ...

# -- RRF -----------------------------------------------------------------------

def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]: ...
def weighted_rrf(ranked_lists: list[tuple[float, list[str]]], k: int = 60) -> list[tuple[str, float]]: ...
def normalize_scores(scores: list[float]) -> list[float]: ...
def weighted_rrf_normalized(
    vector_results: list[tuple[str, float]],
    graph_results: list[tuple[str, float]],
    k: int = 60,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list[tuple[str, float]]: ...
def weighted_rrf_normalized_with_provenance(
    vector_results: list[tuple[str, float]],
    graph_results: list[tuple[str, float]],
    k: int = 60,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list[tuple[str, float, int]]: ...

# -- Temporal -----------------------------------------------------------------

def batch_temporal_filter(
    timestamps_secs: list[float], operator: str, start_secs: float | None = None, end_secs: float | None = None
) -> list[bool]: ...
def batch_recency_scores(
    timestamps_secs: list[float], now_secs: float, decay_days: float, recency_weight: float
) -> list[float]: ...

# -- Community detection ------------------------------------------------------

def detect_communities(
    n: int, edges: list[tuple[int, int, float]], resolution: float = 1.0, max_iter: int = 10
) -> list[int]: ...

# -- MMR diversity ------------------------------------------------------------

def mmr_diversity_select(embeddings: Any, scores: list[float], lambda_param: float, k: int) -> list[int]: ...

# -- Entity resolution ---------------------------------------------------------

def resolve_entities_batch(
    new_names: list[str],
    existing_names: list[str],
    existing_aliases: list[list[str]],
    threshold: float = 0.85,
) -> list[tuple[int, float, str] | None]: ...
def resolve_entities_enhanced(
    new_names: list[str],
    new_types: list[str],
    existing_names: list[str],
    existing_aliases: list[list[str]],
    existing_types: list[str],
    type_thresholds_keys: list[str],
    type_thresholds_vals: list[float],
    default_threshold: float = 0.85,
) -> list[tuple[int, float, str] | None]: ...

# -- Keyword extraction --------------------------------------------------------

def extract_keywords(content: str) -> list[str]: ...
def extract_keywords_batch(contents: list[str]) -> list[list[str]]: ...
