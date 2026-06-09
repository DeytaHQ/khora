"""Query engine for Khora.

Provides hybrid search combining vector, graph, and keyword search
with Reciprocal Rank Fusion for result combination.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from khora.search_mode import SearchMode
from .keyword import BM25Index, KeywordSearcher, build_keyword_index, normalize_bm25_score, tokenize
from .router import QueryComplexity, QueryComplexityRouter, RouterConfig, RoutingDecision
from .temporal import TemporalFilter, TemporalQuery
from .understanding import (
    EntityMention,
    QueryIntent,
    QueryUnderstanding,
    TemporalReference,
    UnderstandingResult,
)

if TYPE_CHECKING:
    from .engine import HybridQueryEngine, QueryConfig, QueryResult
    from .fusion import reciprocal_rank_fusion
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

# Symbols lazily loaded from heavy submodules — resolved on first access via __getattr__.
_LAZY: dict[str, str] = {
    # .engine
    "HybridQueryEngine": ".engine",
    "QueryConfig": ".engine",
    "QueryResult": ".engine",
    # .fusion
    "reciprocal_rank_fusion": ".fusion",
    # .linking
    "EntityLinker": ".linking",
    "LinkedEntity": ".linking",
    "LinkingResult": ".linking",
    "link_query_entities": ".linking",
    # .reranking
    "CrossEncoderReranker": ".reranking",
    "LLMReranker": ".reranking",
    "RerankCandidate": ".reranking",
    "Reranker": ".reranking",
    "RerankResult": ".reranking",
    "create_reranker": ".reranking",
    "rerank_chunks": ".reranking",
    "rerank_entities": ".reranking",
}

_lazy_cache: dict[str, object] = {}


def __getattr__(name: str) -> object:
    if name in _LAZY:
        if name not in _lazy_cache:
            mod = importlib.import_module(_LAZY[name], package=__name__)
            _lazy_cache[name] = getattr(mod, name)
        return _lazy_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__ + list(_LAZY)


__all__ = [
    "HybridQueryEngine",
    "QueryConfig",
    "QueryResult",
    "SearchMode",
    "reciprocal_rank_fusion",
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
    # Query routing
    "QueryComplexity",
    "QueryComplexityRouter",
    "RouterConfig",
    "RoutingDecision",
]
