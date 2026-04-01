"""Query engine for Khora Memory Lake.

Provides hybrid search combining vector, graph, and keyword search
with Reciprocal Rank Fusion for result combination.
"""

from __future__ import annotations

from .engine import HybridQueryEngine, QueryConfig, QueryResult, SearchMode
from .engine_profiles import apply_engine_profile, get_engine_profile, list_engine_profiles
from .fusion import reciprocal_rank_fusion
from .keyword import BM25Index, KeywordSearcher, build_keyword_index, normalize_bm25_score, tokenize
from .linking import EntityLinker, LinkedEntity, LinkingResult, link_query_entities
from .reranking import (
    CrossEncoderReranker,
    LLMReranker,
    RerankCandidate,
    Reranker,
    RerankResult,
    create_reranker,
    rerank_chunks,
    rerank_entities,
)
from .temporal import TemporalFilter, TemporalQuery
from .understanding import (
    EntityMention,
    QueryIntent,
    QueryUnderstanding,
    TemporalReference,
    UnderstandingResult,
)

__all__ = [
    "HybridQueryEngine",
    "QueryConfig",
    "QueryResult",
    "SearchMode",
    "reciprocal_rank_fusion",
    # Engine profiles
    "apply_engine_profile",
    "get_engine_profile",
    "list_engine_profiles",
    "TemporalFilter",
    "TemporalQuery",
    # Query understanding
    "QueryUnderstanding",
    "UnderstandingResult",
    "QueryIntent",
    "EntityMention",
    "TemporalReference",
    # Entity linking
    "EntityLinker",
    "LinkedEntity",
    "LinkingResult",
    "link_query_entities",
    # Reranking
    "Reranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "RerankCandidate",
    "RerankResult",
    "create_reranker",
    "rerank_chunks",
    "rerank_entities",
    # Keyword search
    "BM25Index",
    "KeywordSearcher",
    "build_keyword_index",
    "normalize_bm25_score",
    "tokenize",
]
