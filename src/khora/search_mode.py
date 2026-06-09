"""Leaf module for SearchMode enum — no heavy query-engine dependencies."""

from enum import Enum, auto


class SearchMode(Enum):
    """Search mode for the query engine."""

    VECTOR = auto()  # Vector similarity only
    GRAPH = auto()  # Graph traversal only
    HYBRID = auto()  # Combine vector and graph
    ALL = auto()  # Vector, graph, and keyword
    KEYWORD = auto()  # BM25 / keyword-only (used by Skeleton)
