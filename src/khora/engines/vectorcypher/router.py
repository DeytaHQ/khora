"""Query complexity router for VectorCypher engine.

Routes queries to appropriate search paths based on complexity heuristics:
- SIMPLE: Vector-only search (faster, for simple factual queries)
- MODERATE: Shallow graph traversal (depth=1)
- COMPLEX: Full VectorCypher with deep graph traversal (depth=2-3)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


class QueryComplexity(Enum):
    """Query complexity levels."""

    SIMPLE = "simple"  # Vector-only search
    MODERATE = "moderate"  # Shallow graph (depth=1)
    COMPLEX = "complex"  # Full VectorCypher (depth=2-3)


@dataclass
class RoutingDecision:
    """Result of query routing."""

    complexity: QueryComplexity
    use_graph: bool
    graph_depth: int
    confidence: float
    reasoning: str
    suggested_entry_limit: int = 10


@dataclass
class RouterConfig:
    """Configuration for the query router."""

    enabled: bool = True
    use_llm: bool = False  # Use heuristics by default (faster)
    llm_model: str = "gpt-4o-mini"

    # Depth settings
    simple_depth: int = 0
    moderate_depth: int = 1
    complex_depth: int = 2

    # Entry entity limits
    simple_entry_limit: int = 5
    moderate_entry_limit: int = 10
    complex_entry_limit: int = 15

    # Heuristic thresholds
    multi_entity_threshold: int = 2  # Number of potential entities to trigger complex


class QueryComplexityRouter:
    """Routes queries based on complexity heuristics.

    The router analyzes query patterns to determine the optimal search strategy:
    - Simple queries (what, who, when alone) -> Vector-only (fastest)
    - Moderate queries (single relationship) -> Shallow graph
    - Complex queries (multi-hop, comparisons) -> Full VectorCypher
    """

    # Patterns indicating complex queries
    RELATIONSHIP_PATTERNS = [
        r"\b(between|related|connected|linked|associated)\b",
        r"\b(relationship|connection|link|tie|association)\b",
        r"\b(how\s+does?|how\s+is|how\s+are)\b.*\b(relate|connect|link|associated)\b",
    ]

    COMPARISON_PATTERNS = [
        r"\b(vs\.?|versus|compare|compared|comparison|differ|different|difference)\b",
        r"\b(similar|similarity|similarities)\b.*\b(between|and)\b",
        r"\b(contrast|contrasting)\b",
    ]

    MULTI_HOP_PATTERNS = [
        r"\b(through|via|across|spanning)\b",
        r"\b(chain|path|route|sequence)\b",
        r"\b(indirect|indirectly)\b",
        r"\b(how\s+many\s+degrees?|hops?)\b",
    ]

    AGGREGATION_PATTERNS = [
        r"\b(all|every|each|total|sum|count|number\s+of|how\s+many)\b",
        r"\b(list|enumerate|show\s+me\s+all)\b",
    ]

    TEMPORAL_PATTERNS = [
        r"\b(over\s+time|timeline|history|evolution|changed?|changes?)\b",
        r"\b(before|after|during|since|until|throughout)\b",
        r"\b(first|last|earliest|latest|most\s+recent)\b",
    ]

    # Patterns indicating simple queries
    SIMPLE_QUESTION_PATTERNS = [
        r"^what\s+is\b",
        r"^who\s+is\b",
        r"^when\s+(was|did|is)\b",
        r"^where\s+is\b",
        r"^define\b",
        r"^tell\s+me\s+about\b",
    ]

    def __init__(self, config: RouterConfig | None = None):
        """Initialize the router.

        Args:
            config: Router configuration
        """
        self._config = config or RouterConfig()

        # Compile patterns for efficiency
        self._relationship_re = [re.compile(p, re.IGNORECASE) for p in self.RELATIONSHIP_PATTERNS]
        self._comparison_re = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self._multi_hop_re = [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS]
        self._aggregation_re = [re.compile(p, re.IGNORECASE) for p in self.AGGREGATION_PATTERNS]
        self._temporal_re = [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS]
        self._simple_re = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_QUESTION_PATTERNS]

    async def route(self, query: str) -> RoutingDecision:
        """Route a query to the appropriate search path.

        Args:
            query: The user's query

        Returns:
            RoutingDecision with complexity level and parameters
        """
        if not self._config.enabled:
            # Default to moderate if routing disabled
            return RoutingDecision(
                complexity=QueryComplexity.MODERATE,
                use_graph=True,
                graph_depth=self._config.moderate_depth,
                confidence=1.0,
                reasoning="Routing disabled, using default moderate path",
                suggested_entry_limit=self._config.moderate_entry_limit,
            )

        if self._config.use_llm:
            return await self._llm_route(query)

        return self._heuristic_route(query)

    def _heuristic_route(self, query: str) -> RoutingDecision:
        """Route query using pattern-based heuristics (fast, no LLM).

        Args:
            query: The user's query

        Returns:
            RoutingDecision based on pattern matching
        """
        # Score different complexity indicators
        complexity_score = 0.0
        reasons: list[str] = []

        # Check for multi-entity mentions (named entities, quoted terms)
        entity_count = self._count_potential_entities(query)
        if entity_count >= self._config.multi_entity_threshold:
            complexity_score += 0.4
            reasons.append(f"multiple entities ({entity_count})")

        # Check relationship patterns
        if any(p.search(query) for p in self._relationship_re):
            complexity_score += 0.3
            reasons.append("relationship keywords")

        # Check comparison patterns
        if any(p.search(query) for p in self._comparison_re):
            complexity_score += 0.3
            reasons.append("comparison keywords")

        # Check multi-hop patterns
        if any(p.search(query) for p in self._multi_hop_re):
            complexity_score += 0.4
            reasons.append("multi-hop keywords")

        # Check aggregation patterns
        if any(p.search(query) for p in self._aggregation_re):
            complexity_score += 0.2
            reasons.append("aggregation keywords")

        # Check temporal patterns
        if any(p.search(query) for p in self._temporal_re):
            complexity_score += 0.2
            reasons.append("temporal keywords")

        # Check for simple question patterns (reduces score)
        if any(p.search(query) for p in self._simple_re):
            complexity_score -= 0.3
            reasons.append("simple question pattern")

        # Query length as a factor (longer queries often more complex)
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 0.1
            reasons.append("long query")
        elif word_count < 6:
            complexity_score -= 0.1
            reasons.append("short query")

        # Determine complexity level
        if complexity_score >= 0.5:
            complexity = QueryComplexity.COMPLEX
            depth = self._config.complex_depth
            entry_limit = self._config.complex_entry_limit
        elif complexity_score >= 0.2:
            complexity = QueryComplexity.MODERATE
            depth = self._config.moderate_depth
            entry_limit = self._config.moderate_entry_limit
        else:
            complexity = QueryComplexity.SIMPLE
            depth = self._config.simple_depth
            entry_limit = self._config.simple_entry_limit

        use_graph = complexity != QueryComplexity.SIMPLE
        reasoning = "; ".join(reasons) if reasons else "no complexity indicators"

        decision = RoutingDecision(
            complexity=complexity,
            use_graph=use_graph,
            graph_depth=depth,
            confidence=min(1.0, 0.5 + abs(complexity_score)),
            reasoning=f"Heuristic: {reasoning} (score={complexity_score:.2f})",
            suggested_entry_limit=entry_limit,
        )

        logger.debug(f"Query routed: {complexity.value} (score={complexity_score:.2f}, graph_depth={depth})")

        return decision

    def _count_potential_entities(self, query: str) -> int:
        """Count potential entity mentions in the query.

        Looks for:
        - Capitalized words (proper nouns)
        - Quoted strings
        - Known entity patterns (dates, numbers with units)
        """
        count = 0

        # Count capitalized sequences (potential proper nouns)
        # Exclude sentence-initial capitals
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and word and word[0].isupper():
                # Count consecutive capitalized words as one entity
                if i > 1 and words[i - 1] and words[i - 1][0].isupper():
                    continue  # Part of previous entity
                count += 1

        # Count quoted strings
        count += len(re.findall(r'"[^"]+"|\'[^\']+\'', query))

        return count

    async def _llm_route(self, query: str) -> RoutingDecision:
        """Route query using LLM classification (more accurate, slower).

        Args:
            query: The user's query

        Returns:
            RoutingDecision from LLM classification
        """
        # For now, fall back to heuristics
        # In production, this would call an LLM to classify the query
        logger.debug("LLM routing requested but not implemented, using heuristics")
        return self._heuristic_route(query)


__all__ = [
    "QueryComplexity",
    "QueryComplexityRouter",
    "RouterConfig",
    "RoutingDecision",
]
