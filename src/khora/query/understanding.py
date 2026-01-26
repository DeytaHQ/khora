"""Query understanding module for Khora Memory Lake.

Provides LLM-based query interpretation including:
- Intent detection
- Entity mention extraction
- Temporal reference detection
- Query expansion/reformulation
- Keyword extraction
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from khora.config.llm import LiteLLMConfig


class QueryIntent(Enum):
    """Types of query intent."""

    SEARCH = auto()  # General search for information
    QUESTION = auto()  # Specific question expecting an answer
    TEMPORAL = auto()  # Query about time-based events
    COMPARISON = auto()  # Query comparing entities or concepts
    NAVIGATION = auto()  # Query to find specific entities
    UNKNOWN = auto()


@dataclass
class EntityMention:
    """An entity mentioned in the query."""

    name: str
    entity_type: str  # PERSON, ORGANIZATION, CONCEPT, etc.
    confidence: float = 1.0
    start_pos: int | None = None
    end_pos: int | None = None


@dataclass
class TemporalReference:
    """A temporal reference in the query."""

    type: str  # relative, absolute, range
    value: str  # "last week", "2024-01-15", etc.
    normalized: str | None = None  # ISO format or semantic (e.g., "past_7_days")


@dataclass
class UnderstandingResult:
    """Result of query understanding."""

    original_query: str
    intent: QueryIntent
    entities: list[EntityMention] = field(default_factory=list)
    temporal_references: list[TemporalReference] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_temporal(self) -> bool:
        """Check if query has temporal references."""
        return len(self.temporal_references) > 0

    @property
    def has_entities(self) -> bool:
        """Check if query mentions entities."""
        return len(self.entities) > 0

    def get_all_queries(self) -> list[str]:
        """Get original query plus all expansions."""
        return [self.original_query] + self.expanded_queries


QUERY_UNDERSTANDING_PROMPT = """You are a query understanding system for a corporate memory lake containing:
- Slack messages, Linear issues, Notion documents
- Attio CRM records, Gong call transcripts
- BambooHR employee data, GitHub activity

Analyze the following query and extract structured information.

Query: {query}

Respond with a JSON object containing:
{{
    "intent": "search|question|temporal|comparison|navigation",
    "entities": [
        {{
            "name": "entity name",
            "type": "PERSON|ORGANIZATION|CONCEPT|PRODUCT|TECHNOLOGY|LOCATION|EVENT",
            "confidence": 0.0-1.0
        }}
    ],
    "temporal_references": [
        {{
            "type": "relative|absolute|range",
            "value": "original text",
            "normalized": "ISO date or semantic like past_7_days"
        }}
    ],
    "expanded_queries": [
        "alternative phrasing 1",
        "alternative phrasing 2"
    ],
    "keywords": ["keyword1", "keyword2"],
    "confidence": 0.0-1.0
}}

Guidelines:
- Extract ALL mentioned entities (people, companies, products, concepts)
- Detect temporal references like "last week", "yesterday", "in January"
- Generate 2-3 alternative query phrasings that capture the same intent
- Extract important keywords for full-text search
- Be precise about entity types based on context

Respond ONLY with the JSON object, no explanation."""


class QueryUnderstanding:
    """LLM-based query understanding for enhanced search."""

    def __init__(
        self,
        llm_config: LiteLLMConfig | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize query understanding.

        Args:
            llm_config: LiteLLM configuration
            model: Optional model override (defaults to config model)
        """
        self._llm_config = llm_config
        self._model = model

    async def understand(
        self,
        query: str,
        *,
        expand_query: bool = True,
        extract_entities: bool = True,
        detect_temporal: bool = True,
    ) -> UnderstandingResult:
        """Understand a query using LLM.

        Args:
            query: The query to understand
            expand_query: Whether to generate query expansions
            extract_entities: Whether to extract entity mentions
            detect_temporal: Whether to detect temporal references

        Returns:
            UnderstandingResult with extracted information
        """
        from khora.config.llm import LiteLLMConfig, acompletion

        config = self._llm_config or LiteLLMConfig()
        if self._model:
            config = LiteLLMConfig(
                model=self._model,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000,
            )
        else:
            config = LiteLLMConfig(
                model=config.model,
                temperature=0.1,
                max_tokens=1000,
            )

        try:
            prompt = QUERY_UNDERSTANDING_PROMPT.format(query=query)
            response = await acompletion(prompt, config)

            # Parse JSON response
            result = self._parse_response(response, query)

            # Filter based on settings
            if not expand_query:
                result.expanded_queries = []
            if not extract_entities:
                result.entities = []
            if not detect_temporal:
                result.temporal_references = []

            return result

        except Exception as e:
            logger.warning(f"Query understanding failed: {e}")
            # Return basic result on failure
            return UnderstandingResult(
                original_query=query,
                intent=QueryIntent.SEARCH,
                keywords=self._extract_keywords_simple(query),
                confidence=0.5,
            )

    def _parse_response(self, response: str, original_query: str) -> UnderstandingResult:
        """Parse the LLM response into UnderstandingResult.

        Args:
            response: Raw LLM response
            original_query: Original query text

        Returns:
            Parsed UnderstandingResult
        """
        # Clean up response (handle markdown code blocks)
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse query understanding response: {response[:100]}")
            return UnderstandingResult(
                original_query=original_query,
                intent=QueryIntent.SEARCH,
                confidence=0.5,
            )

        # Map intent
        intent_map = {
            "search": QueryIntent.SEARCH,
            "question": QueryIntent.QUESTION,
            "temporal": QueryIntent.TEMPORAL,
            "comparison": QueryIntent.COMPARISON,
            "navigation": QueryIntent.NAVIGATION,
        }
        intent = intent_map.get(data.get("intent", "search").lower(), QueryIntent.UNKNOWN)

        # Parse entities
        entities = []
        for e in data.get("entities", []):
            entities.append(
                EntityMention(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "CONCEPT"),
                    confidence=e.get("confidence", 1.0),
                )
            )

        # Parse temporal references
        temporal_refs = []
        for t in data.get("temporal_references", []):
            temporal_refs.append(
                TemporalReference(
                    type=t.get("type", "relative"),
                    value=t.get("value", ""),
                    normalized=t.get("normalized"),
                )
            )

        return UnderstandingResult(
            original_query=original_query,
            intent=intent,
            entities=entities,
            temporal_references=temporal_refs,
            expanded_queries=data.get("expanded_queries", []),
            keywords=data.get("keywords", []),
            confidence=data.get("confidence", 1.0),
        )

    def _extract_keywords_simple(self, query: str) -> list[str]:
        """Simple keyword extraction without LLM.

        Args:
            query: Query text

        Returns:
            List of keywords
        """
        # Simple stopword removal and tokenization
        stopwords = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
        }

        # Tokenize and filter
        words = query.lower().split()
        keywords = [
            w.strip(".,!?;:'\"()[]{}")
            for w in words
            if w.lower().strip(".,!?;:'\"()[]{}") not in stopwords and len(w) > 2
        ]

        return keywords
