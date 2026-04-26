"""Query complexity router shared across engines.

Routes queries to appropriate search paths based on complexity heuristics:
- SIMPLE: Lightweight retrieval (vectorcypher: vector-only; chronicle: skip BM25+entity)
- MODERATE: Shallow graph traversal (vectorcypher depth=1; chronicle: all channels)
- COMPLEX: Full retrieval (vectorcypher depth=2-3; chronicle: all channels)

The router uses a two-phase approach:
1. Fast heuristic-based classification (regex patterns, query structure)
2. Optional LLM classification when heuristic confidence is low

Telemetry is logged for all routing decisions to enable analysis and tuning.

Originally lived under :mod:`khora.engines.vectorcypher.router`; relocated here
for reuse across engines (Chronicle #6). The vectorcypher path remains as a
re-export shim for back-compat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from khora.config.llm import LiteLLMConfig
    from khora.query.temporal_detection import TemporalSignal


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
    llm_config: LiteLLMConfig | None = None  # LLM config for LLM-based routing
    llm_model: str = "gpt-4o-mini"  # Fallback model if llm_config not provided

    # Confidence threshold for LLM fallback
    llm_confidence_threshold: float = 0.85  # Use LLM if heuristic confidence below this

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

    # Adaptive depth settings
    adaptive_depth_enabled: bool = True
    adaptive_depth_high_entity_threshold: int = 10  # Shallow depth if >= this many entities
    adaptive_depth_low_entity_threshold: int = 2  # Deeper depth if <= this many entities


class QueryComplexityRouter:
    """Routes queries based on complexity heuristics with optional LLM fallback.

    The router analyzes query patterns to determine the optimal search strategy:
    - Simple queries (what, who, when alone) -> Vector-only (fastest)
    - Moderate queries (single relationship) -> Shallow graph
    - Complex queries (multi-hop, comparisons) -> Full VectorCypher

    Two-phase routing:
    1. Fast heuristic analysis (regex patterns, query structure, entity detection)
    2. Optional LLM classification when heuristic confidence is low

    All routing decisions are logged with telemetry for analysis and tuning.
    """

    # Patterns indicating complex queries
    RELATIONSHIP_PATTERNS = [
        r"\b(between|related|connected|linked|associated)\b",
        r"\b(relationship|connection|link|tie|association)\b",
        r"\b(how\s+does?|how\s+is|how\s+are)\b.*\b(relate|connect|link|associated)\b",
        r"\b(works?\s+with|collaborates?\s+with|interacts?\s+with)\b",
        r"\b(depends?\s+on|requires?|needs?)\b.*\b(to|for)\b",
    ]

    COMPARISON_PATTERNS = [
        r"\b(vs\.?|versus|compare|compared|comparison|differ|different|difference)\b",
        r"\b(similar|similarity|similarities)\b.*\b(between|and)\b",
        r"\b(contrast|contrasting)\b",
        r"\b(better|worse|more|less)\s+than\b",
        r"\b(advantages?|disadvantages?|pros?|cons?)\b.*\b(of|between)\b",
    ]

    MULTI_HOP_PATTERNS = [
        r"\b(through|via|across|spanning)\b",
        r"\b(chain|path|route|sequence)\b",
        r"\b(indirect|indirectly)\b",
        r"\b(how\s+many\s+degrees?|hops?)\b",
        r"\b(leads?\s+to|results?\s+in|causes?|affects?)\b",
        r"\b(upstream|downstream)\b",
        r"\b(transitively|recursively)\b",
    ]

    AGGREGATION_PATTERNS = [
        r"\b(all|every|each|total|sum|count|number\s+of|how\s+many)\b",
        r"\b(list|enumerate|show\s+me\s+all)\b",
        r"\b(overview|summary|summarize)\b",
        r"\b(most\s+common|frequently|popular)\b",
    ]

    TEMPORAL_PATTERNS = [
        r"\b(over\s+time|timeline|history|evolution|changed?|changes?)\b",
        r"\b(before|after|during|since|until|throughout)\b",
        r"\b(first|last|earliest|latest|most\s+recent)\b",
        r"\b(trend|trends|trending)\b",
        r"\b(growth|decline|increase|decrease)\b",
    ]

    CAUSAL_PATTERNS = [
        r"\b(why|because|reason|cause|caused)\b",
        r"\b(impact|effect|consequence|result)\b",
        r"\b(leads?\s+to|results?\s+in)\b",
        r"\b(due\s+to|owing\s+to|thanks\s+to)\b",
    ]

    COUNTERFACTUAL_PATTERNS = [
        r"\bif\b.*\bhad\s+not\b",
        r"\bwhat\s+would\b",
        r"\bwhat\s+if\b",
        r"\bhypothetical\b",
        r"\binstead\s+of\b",
        r"\bwithout\s+(having|doing|being)\b",
        r"\bhad\s+(they|he|she|it|we)\s+not\b",
        r"\bdo\b.*\b(both|start|share|have\s+in\s+common)\b",
    ]

    HIERARCHICAL_PATTERNS = [
        r"\b(parent|child|ancestor|descendant)\b",
        r"\b(belongs?\s+to|part\s+of|contains?|includes?)\b",
        r"\b(subcategory|supercategory|subset|superset)\b",
        r"\b(under|above|within)\b",
    ]

    # Patterns indicating simple queries (reduces complexity score)
    SIMPLE_QUESTION_PATTERNS = [
        r"^what\s+is\b",
        r"^who\s+is\b",
        r"^when\s+(was|did|is)\b",
        r"^where\s+is\b",
        r"^define\b",
        r"^tell\s+me\s+about\b",
        r"^explain\s+what\b",
        r"^describe\b",
    ]

    FACTUAL_PATTERNS = [
        r"^(what|who|when|where|which)\b.*\?$",
        r"\b(name|title|date|location|address)\s+of\b",
        r"\bwhat\s+does\s+\w+\s+mean\b",
    ]

    # LLM prompt for classification
    LLM_CLASSIFICATION_PROMPT = """Classify this search query's complexity for a knowledge graph retrieval system.

Query: "{query}"

Classify as one of:
- SIMPLE: Direct lookup, single entity, factual question (e.g., "What is Python?", "Who is CEO of Apple?")
- MODERATE: Single relationship, basic connection (e.g., "Who works at Google?", "What products does Apple make?")
- COMPLEX: Multi-hop traversal, comparison, aggregation, causal reasoning (e.g., "How is A connected to B through C?", "Compare X and Y", "What caused the incident?")

Respond with ONLY the classification word (SIMPLE, MODERATE, or COMPLEX) and a brief reason.
Format: CLASSIFICATION|reason

Example responses:
SIMPLE|Direct factual lookup about a single entity
MODERATE|Single relationship query between two entities
COMPLEX|Multi-hop query requiring graph traversal"""

    def __init__(
        self,
        config: RouterConfig | None = None,
        *,
        use_llm: bool | None = None,
        llm_config: LiteLLMConfig | None = None,
    ):
        """Initialize the router.

        Args:
            config: Router configuration
            use_llm: Override for LLM routing (takes precedence over config)
            llm_config: LLM configuration for LLM-based routing
        """
        self._config = config or RouterConfig()

        # Allow overrides via constructor arguments
        if use_llm is not None:
            self._config.use_llm = use_llm
        if llm_config is not None:
            self._config.llm_config = llm_config

        # Compile patterns for efficiency
        self._relationship_re = [re.compile(p, re.IGNORECASE) for p in self.RELATIONSHIP_PATTERNS]
        self._comparison_re = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self._multi_hop_re = [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS]
        self._aggregation_re = [re.compile(p, re.IGNORECASE) for p in self.AGGREGATION_PATTERNS]
        self._temporal_re = [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS]
        self._causal_re = [re.compile(p, re.IGNORECASE) for p in self.CAUSAL_PATTERNS]
        self._hierarchical_re = [re.compile(p, re.IGNORECASE) for p in self.HIERARCHICAL_PATTERNS]
        self._counterfactual_re = [re.compile(p, re.IGNORECASE) for p in self.COUNTERFACTUAL_PATTERNS]
        self._simple_re = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_QUESTION_PATTERNS]
        self._factual_re = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]

        # Routing decision cache for telemetry analysis
        self._routing_stats: dict[str, int] = {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
            "llm_fallback": 0,
        }

    async def route(
        self,
        query: str,
        *,
        temporal_signal: TemporalSignal | None = None,
    ) -> RoutingDecision:
        """Route a query to the appropriate search path.

        Uses a two-phase approach:
        1. Fast heuristic-based classification
        2. LLM fallback if enabled and heuristic confidence is low

        Args:
            query: The user's query
            temporal_signal: Optional temporal detection signal; when present and
                ``is_temporal`` is True the heuristic guarantees at least MODERATE
                routing so the graph path is activated for cross-session entity linking.

        Returns:
            RoutingDecision with complexity level and parameters
        """
        if not self._config.enabled:
            # Default to moderate if routing disabled
            decision = RoutingDecision(
                complexity=QueryComplexity.MODERATE,
                use_graph=True,
                graph_depth=self._config.moderate_depth,
                confidence=1.0,
                reasoning="Routing disabled, using default moderate path",
                suggested_entry_limit=self._config.moderate_entry_limit,
            )
            self._log_routing_decision(query, decision, "disabled")
            return decision

        # Phase 1: Fast heuristic classification
        heuristic_decision = self._heuristic_route(query, temporal_signal=temporal_signal)

        # Phase 2: LLM fallback if enabled and confidence is low
        if self._config.use_llm and heuristic_decision.confidence < self._config.llm_confidence_threshold:
            self._routing_stats["llm_fallback"] += 1
            try:
                llm_decision = await self._llm_route(query, heuristic_decision)
                self._log_routing_decision(query, llm_decision, "llm")
                return llm_decision
            except Exception as e:
                logger.warning(f"LLM routing failed, using heuristic result: {e}")
                # Fall back to heuristic decision
                heuristic_decision.reasoning += f" (LLM fallback failed: {e})"

        self._log_routing_decision(query, heuristic_decision, "heuristic")
        return heuristic_decision

    def _log_routing_decision(
        self,
        query: str,
        decision: RoutingDecision,
        method: str,
    ) -> None:
        """Log routing decision with telemetry.

        Args:
            query: Original query
            decision: Routing decision made
            method: Method used (heuristic, llm, disabled)
        """
        # Update stats
        self._routing_stats[decision.complexity.value] += 1

        # Log for analysis - use info level for visibility
        logger.info(
            f"Query routed: complexity={decision.complexity.value}, "
            f"confidence={decision.confidence:.2f}, "
            f"depth={decision.graph_depth}, "
            f"use_graph={decision.use_graph}, "
            f"method={method}"
        )

        # Debug level for full details
        logger.debug(
            f"Routing details: query='{query[:100]}...', "
            f"entry_limit={decision.suggested_entry_limit}, "
            f"reasoning={decision.reasoning}"
        )

    def _heuristic_route(
        self,
        query: str,
        *,
        temporal_signal: TemporalSignal | None = None,
    ) -> RoutingDecision:
        """Route query using pattern-based heuristics (fast, no LLM).

        The heuristic scoring system:
        - Positive scores indicate complexity (relationship, comparison, multi-hop, etc.)
        - Negative scores indicate simplicity (factual questions, short queries)
        - Final score determines routing: <0.2 SIMPLE, 0.2-0.5 MODERATE, >=0.5 COMPLEX
        - Confidence is derived from the strength of the signal

        Args:
            query: The user's query
            temporal_signal: Optional temporal detection signal; when present and
                temporal the score is boosted to at least MODERATE.

        Returns:
            RoutingDecision based on pattern matching
        """
        # Score different complexity indicators
        complexity_score = 0.0
        pattern_matches = 0  # Track how many pattern types matched for confidence
        reasons: list[str] = []

        # Check for multi-entity mentions (named entities, quoted terms)
        entity_count = self._count_potential_entities(query)
        if entity_count >= self._config.multi_entity_threshold:
            complexity_score += 0.4
            pattern_matches += 1
            reasons.append(f"multiple entities ({entity_count})")
        elif entity_count == 1:
            # Single entity suggests simpler query
            complexity_score -= 0.1
            reasons.append("single entity")

        # Check relationship patterns (moderate complexity indicator)
        if any(p.search(query) for p in self._relationship_re):
            complexity_score += 0.3
            pattern_matches += 1
            reasons.append("relationship keywords")

        # Check comparison patterns (high complexity indicator)
        if any(p.search(query) for p in self._comparison_re):
            complexity_score += 0.35
            pattern_matches += 1
            reasons.append("comparison keywords")

        # Check multi-hop patterns (high complexity indicator)
        if any(p.search(query) for p in self._multi_hop_re):
            complexity_score += 0.4
            pattern_matches += 1
            reasons.append("multi-hop keywords")

        # Check causal patterns (high complexity indicator)
        if any(p.search(query) for p in self._causal_re):
            complexity_score += 0.35
            pattern_matches += 1
            reasons.append("causal keywords")

        # Check counterfactual patterns (high complexity — needs both original
        # state and change event, typically spanning multiple graph hops)
        if any(p.search(query) for p in self._counterfactual_re):
            complexity_score += 0.35
            pattern_matches += 1
            reasons.append("counterfactual keywords")

        # Check hierarchical patterns (moderate complexity indicator)
        if any(p.search(query) for p in self._hierarchical_re):
            complexity_score += 0.25
            pattern_matches += 1
            reasons.append("hierarchical keywords")

        # Check aggregation patterns (moderate complexity indicator)
        if any(p.search(query) for p in self._aggregation_re):
            complexity_score += 0.2
            pattern_matches += 1
            reasons.append("aggregation keywords")

        # Check temporal patterns (moderate complexity indicator)
        if any(p.search(query) for p in self._temporal_re):
            complexity_score += 0.2
            pattern_matches += 1
            reasons.append("temporal keywords")

        # Check for simple question patterns (reduces score)
        if any(p.search(query) for p in self._simple_re):
            complexity_score -= 0.3
            pattern_matches += 1
            reasons.append("simple question pattern")

        # Check for factual patterns (reduces score)
        if any(p.search(query) for p in self._factual_re):
            complexity_score -= 0.2
            pattern_matches += 1
            reasons.append("factual query pattern")

        # Query structural analysis
        word_count = len(query.split())
        sentence_count = len(re.findall(r"[.!?]+", query)) + 1

        # Query length as a factor
        if word_count > 25:
            complexity_score += 0.15
            reasons.append("long query")
        elif word_count > 15:
            complexity_score += 0.05
        elif word_count < 5:
            complexity_score -= 0.15
            reasons.append("very short query")
        elif word_count < 8:
            complexity_score -= 0.05
            reasons.append("short query")

        # Multiple sentences suggest compound query
        if sentence_count > 1:
            complexity_score += 0.1 * (sentence_count - 1)
            reasons.append(f"multi-sentence ({sentence_count})")

        # Count question words - multiple WHs suggest complex query
        question_words = len(re.findall(r"\b(what|who|when|where|why|how|which)\b", query, re.IGNORECASE))
        if question_words > 1:
            complexity_score += 0.1 * (question_words - 1)
            reasons.append(f"multiple question words ({question_words})")

        # Temporal signal override: if the upstream temporal detector flagged this
        # query as temporal, ensure at least MODERATE routing so the graph path is
        # activated for cross-session entity linking.  Without this, short temporal
        # queries (e.g. "When did Alice move?") get penalized by short-query and
        # simple-question deductions, land below 0.2, and route to vector-only
        # search — which cannot resolve cross-session entity references.
        if temporal_signal is not None and temporal_signal.is_temporal:
            if complexity_score < 0.35:  # Would be SIMPLE or barely MODERATE
                original_score = complexity_score
                complexity_score = max(complexity_score, 0.35)  # Solidly MODERATE for graph path
                reasons.append("temporal signal boosted to MODERATE")
                logger.debug(f"Temporal signal boosted complexity from {original_score:.2f} to {complexity_score:.2f}")

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

        # Calculate confidence based on pattern match strength
        # More pattern matches = higher confidence in the classification
        # Score far from thresholds = higher confidence
        base_confidence = 0.5

        # Boost confidence based on number of pattern matches
        pattern_confidence = min(0.3, pattern_matches * 0.05)

        # Boost confidence based on distance from decision thresholds
        if complexity == QueryComplexity.COMPLEX:
            threshold_distance = complexity_score - 0.5
        elif complexity == QueryComplexity.SIMPLE:
            threshold_distance = 0.2 - complexity_score
        else:
            # MODERATE: further from both boundaries = higher confidence
            threshold_distance = min(complexity_score - 0.2, 0.5 - complexity_score)
        threshold_confidence = min(0.2, threshold_distance * 0.4)

        confidence = min(1.0, base_confidence + pattern_confidence + threshold_confidence)

        reasoning = "; ".join(reasons) if reasons else "no complexity indicators"

        decision = RoutingDecision(
            complexity=complexity,
            use_graph=use_graph,
            graph_depth=depth,
            confidence=confidence,
            reasoning=f"Heuristic: {reasoning} (score={complexity_score:.2f})",
            suggested_entry_limit=entry_limit,
        )

        return decision

    def _count_potential_entities(self, query: str) -> int:
        """Count potential entity mentions in the query.

        Looks for:
        - Capitalized words/phrases (proper nouns)
        - Quoted strings
        - Known entity patterns (dates, numbers with units)
        - CamelCase or snake_case identifiers (technical entities)
        """
        count = 0

        # Count capitalized sequences (potential proper nouns)
        # Exclude sentence-initial capitals
        words = query.split()
        i = 0
        while i < len(words):
            word = words[i]
            # Skip first word (sentence-initial capital doesn't count)
            if i > 0 and word and word[0].isupper() and not word.isupper():
                # Check for multi-word entity (consecutive capitals)
                entity_length = 1
                while i + entity_length < len(words):
                    next_word = words[i + entity_length]
                    if next_word and next_word[0].isupper() and not next_word.isupper():
                        entity_length += 1
                    else:
                        break
                count += 1
                i += entity_length
                continue
            i += 1

        # Count quoted strings
        count += len(re.findall(r'"[^"]+"|\'[^\']+\'', query))

        # Count CamelCase identifiers (technical entities)
        count += len(re.findall(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b", query))

        # Count snake_case identifiers (technical entities)
        count += len(re.findall(r"\b[a-z]+_[a-z_]+\b", query))

        # Count @mentions or #tags
        count += len(re.findall(r"[@#]\w+", query))

        return count

    async def _llm_route(
        self,
        query: str,
        heuristic_result: RoutingDecision | None = None,
    ) -> RoutingDecision:
        """Route query using LLM classification (more accurate, slower).

        Uses a lightweight LLM call to classify query complexity when
        heuristic confidence is low. The LLM provides a second opinion
        that can override or confirm the heuristic result.

        Args:
            query: The user's query
            heuristic_result: Optional heuristic result for context

        Returns:
            RoutingDecision from LLM classification
        """
        try:
            from khora.config.llm import LiteLLMConfig, acompletion
        except ImportError:
            logger.warning("LiteLLM not available, falling back to heuristics")
            return heuristic_result or self._heuristic_route(query)

        # Build LLM config
        llm_config = self._config.llm_config
        if llm_config is None:
            llm_config = LiteLLMConfig(
                model=self._config.llm_model,
                temperature=0.0,  # Deterministic for classification
                max_tokens=100,  # Short response expected
                timeout=5,  # Fast timeout for routing
            )
        else:
            # Override settings for fast, deterministic classification
            llm_config = LiteLLMConfig(
                model=llm_config.model,
                temperature=0.0,
                max_tokens=100,
                timeout=5,
            )

        # Format the prompt
        prompt = self.LLM_CLASSIFICATION_PROMPT.format(query=query[:500])  # Truncate long queries

        try:
            response = await acompletion(
                prompt=prompt,
                config=llm_config,
                _telemetry_op="query_routing",
            )

            # Parse the response
            response_text = response.strip()
            parts = response_text.split("|", 1)
            classification = parts[0].strip().upper()
            llm_reasoning = parts[1].strip() if len(parts) > 1 else "LLM classification"

            # Map to QueryComplexity
            if classification == "SIMPLE":
                complexity = QueryComplexity.SIMPLE
                depth = self._config.simple_depth
                entry_limit = self._config.simple_entry_limit
            elif classification == "MODERATE":
                complexity = QueryComplexity.MODERATE
                depth = self._config.moderate_depth
                entry_limit = self._config.moderate_entry_limit
            elif classification == "COMPLEX":
                complexity = QueryComplexity.COMPLEX
                depth = self._config.complex_depth
                entry_limit = self._config.complex_entry_limit
            else:
                # Unknown response, fall back to heuristic
                logger.warning(f"Unexpected LLM response '{classification}', using heuristic")
                if heuristic_result:
                    heuristic_result.reasoning += f" (LLM unclear: {response_text})"
                    return heuristic_result
                return self._heuristic_route(query)

            use_graph = complexity != QueryComplexity.SIMPLE

            # Build reasoning that combines LLM and heuristic insights
            if heuristic_result:
                reasoning = (
                    f"LLM: {llm_reasoning}; "
                    f"Heuristic suggested {heuristic_result.complexity.value} "
                    f"(confidence={heuristic_result.confidence:.2f})"
                )
            else:
                reasoning = f"LLM: {llm_reasoning}"

            return RoutingDecision(
                complexity=complexity,
                use_graph=use_graph,
                graph_depth=depth,
                confidence=0.9,  # High confidence from LLM
                reasoning=reasoning,
                suggested_entry_limit=entry_limit,
            )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            if heuristic_result:
                heuristic_result.reasoning += f" (LLM error: {e})"
                return heuristic_result
            return self._heuristic_route(query)

    def compute_adaptive_depth(
        self,
        entry_entity_count: int,
        base_depth: int = 2,
    ) -> int:
        """Adjust graph traversal depth based on entry entity count.

        This prevents explosion when many entities are found (shallow traversal)
        and enables deeper exploration when few entities are found.

        Args:
            entry_entity_count: Number of entry entities found
            base_depth: Base depth from routing decision

        Returns:
            Adjusted depth value
        """
        if not self._config.adaptive_depth_enabled:
            return base_depth

        if entry_entity_count >= self._config.adaptive_depth_high_entity_threshold:
            # Many entities: shallow depth to avoid explosion
            adjusted = min(base_depth, 1)
            logger.debug(
                f"Adaptive depth: {entry_entity_count} entities >= "
                f"{self._config.adaptive_depth_high_entity_threshold}, "
                f"reducing depth {base_depth} -> {adjusted}"
            )
            return adjusted
        elif entry_entity_count <= self._config.adaptive_depth_low_entity_threshold:
            # Few entities: deeper traversal to find more context
            adjusted = min(base_depth + 1, self._config.complex_depth + 1)
            logger.debug(
                f"Adaptive depth: {entry_entity_count} entities <= "
                f"{self._config.adaptive_depth_low_entity_threshold}, "
                f"increasing depth {base_depth} -> {adjusted}"
            )
            return adjusted

        return base_depth

    def get_routing_stats(self) -> dict[str, int]:
        """Get routing decision statistics for analysis.

        Returns:
            Dict mapping complexity level to count
        """
        return self._routing_stats.copy()

    def reset_routing_stats(self) -> None:
        """Reset routing statistics."""
        self._routing_stats = {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
            "llm_fallback": 0,
        }


__all__ = [
    "QueryComplexity",
    "QueryComplexityRouter",
    "RouterConfig",
    "RoutingDecision",
]
