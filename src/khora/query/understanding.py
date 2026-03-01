"""Query understanding module for Khora Memory Lake.

Provides comprehensive LLM-based query interpretation in a SINGLE request:
- Intent detection and complexity assessment
- Entity mention extraction with relationship hints
- Temporal reference detection with computed ISO dates
- Query expansion/reformulation
- Keyword extraction for BM25
- Source prioritization (slack, linear, notion, attio, gong, github)
- Search strategy recommendations
- Follow-up query suggestions for agentic search
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING

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
    AGGREGATION = auto()  # Query requiring aggregation/summary
    UNKNOWN = auto()


class AnswerType(Enum):
    """Expected answer type."""

    LIST = auto()  # List of items
    SUMMARY = auto()  # Narrative summary
    FACT = auto()  # Specific fact or value
    EXPLANATION = auto()  # Detailed explanation
    COMPARISON = auto()  # Side-by-side comparison
    TIMELINE = auto()  # Chronological sequence
    UNKNOWN = auto()


@dataclass
class EntityMention:
    """An entity mentioned in the query."""

    name: str
    entity_type: str  # PERSON, ORGANIZATION, CONCEPT, etc.
    confidence: float = 1.0
    aliases: list[str] = field(default_factory=list)  # Alternative names/spellings
    context_hint: str = ""  # Additional context for disambiguation


@dataclass
class RelationshipHint:
    """A relationship to explore in the graph."""

    from_entity: str
    relationship_type: str  # WORKS_WITH, MENTIONED_IN, RELATED_TO, etc.
    to_entity: str | None = None  # None means "find related entities"
    importance: float = 1.0


@dataclass
class TemporalReference:
    """A temporal reference in the query.

    The LLM extracts temporal references and computes actual ISO date bounds.
    """

    type: str  # relative, absolute, range
    text: str  # Original text "last week", "yesterday", etc.
    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass
class SourcePriority:
    """Priority hints for data sources."""

    slack: float = 1.0
    linear: float = 1.0
    notion: float = 1.0
    attio: float = 1.0
    gong: float = 1.0
    github: float = 1.0
    bamboohr: float = 1.0

    def get_top_sources(self, n: int = 3) -> list[str]:
        """Get top N prioritized sources."""
        sources = [
            ("slack", self.slack),
            ("linear", self.linear),
            ("notion", self.notion),
            ("attio", self.attio),
            ("gong", self.gong),
            ("github", self.github),
            ("bamboohr", self.bamboohr),
        ]
        sources.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sources[:n] if s[1] > 0]


@dataclass
class SearchStrategy:
    """Recommended search strategy."""

    use_vector: bool = True
    use_graph: bool = True
    use_keyword: bool = True

    # Weights (should sum to ~1.0)
    vector_weight: float = 0.4
    graph_weight: float = 0.3
    keyword_weight: float = 0.3

    # Graph-specific
    graph_depth: int = 2
    explore_neighborhoods: bool = True

    # Reasoning
    strategy_reasoning: str = ""


@dataclass
class FollowUpQuery:
    """A suggested follow-up query for deeper exploration."""

    query: str
    reasoning: str
    target_sources: list[str] = field(default_factory=list)
    priority: float = 1.0


@dataclass
class UnderstandingResult:
    """Comprehensive result of query understanding - extracted in single LLM call."""

    original_query: str
    intent: QueryIntent
    answer_type: AnswerType = AnswerType.UNKNOWN

    # Core extractions
    entities: list[EntityMention] = field(default_factory=list)
    relationships: list[RelationshipHint] = field(default_factory=list)
    temporal_references: list[TemporalReference] = field(default_factory=list)

    # Search optimization
    expanded_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source_priority: SourcePriority = field(default_factory=SourcePriority)
    search_strategy: SearchStrategy = field(default_factory=SearchStrategy)

    # Source-aware filtering — tools with priority < 0.1 are actively excluded
    source_filters: list[str] = field(default_factory=list)

    # Agentic search support
    follow_up_queries: list[FollowUpQuery] = field(default_factory=list)
    requires_multi_step: bool = False
    complexity_score: float = 0.5  # 0-1, higher = more complex

    # Metadata
    confidence: float = 1.0
    reasoning: str = ""  # LLM's reasoning about the query

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

    def get_entity_names(self) -> list[str]:
        """Get all entity names including aliases."""
        names = []
        for e in self.entities:
            names.append(e.name)
            names.extend(e.aliases)
        return names


# Comprehensive prompt that extracts everything in one shot
COMPREHENSIVE_UNDERSTANDING_PROMPT = """You are an expert query understanding system for a corporate memory lake.

DATA SOURCES AVAILABLE:
- Slack: Team messages, channels, threads, reactions
- Linear: Issues, projects, cycles, comments, labels
- Notion: Documents, wikis, databases, pages
- Attio: CRM records, companies, contacts, deals, meetings
- Gong: Sales calls, recordings, transcripts, key moments
- GitHub: Repositories, PRs, issues, commits, code reviews
- BambooHR: Employee data, org structure, time off

CURRENT DATETIME: {current_datetime}

QUERY: {query}

Analyze this query comprehensively and return a JSON object with ALL of the following:

{{
    "intent": "search|question|temporal|comparison|navigation|aggregation",
    "answer_type": "list|summary|fact|explanation|comparison|timeline",

    "entities": [
        {{
            "name": "exact name",
            "type": "PERSON|ORGANIZATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|LOCATION|EVENT|TEAM|CHANNEL",
            "confidence": 0.0-1.0,
            "aliases": ["alternative names", "nicknames", "abbreviations"],
            "context_hint": "additional context for disambiguation"
        }}
    ],

    "relationships": [
        {{
            "from_entity": "entity name",
            "relationship_type": "WORKS_ON|MENTIONED_IN|RELATED_TO|CREATED|OWNS|MANAGES|REPORTS_TO|COLLABORATES_WITH|DISCUSSED_IN|BLOCKED_BY",
            "to_entity": "target entity or null to discover",
            "importance": 0.0-1.0
        }}
    ],

    "temporal": [
        {{
            "type": "relative|absolute|range",
            "text": "original temporal phrase",
            "start_date": "ISO 8601 datetime or null",
            "end_date": "ISO 8601 datetime or null"
        }}
    ],

    "expanded_queries": [
        "semantically equivalent rephrasing 1",
        "rephrasing targeting different vocabulary 2",
        "more specific version if query is vague"
    ],

    "keywords": ["important", "search", "terms", "for", "bm25"],

    "source_priority": {{
        "slack": 0.0-1.0,
        "linear": 0.0-1.0,
        "notion": 0.0-1.0,
        "attio": 0.0-1.0,
        "gong": 0.0-1.0,
        "github": 0.0-1.0,
        "bamboohr": 0.0-1.0
    }},

    "search_strategy": {{
        "use_vector": true/false,
        "use_graph": true/false,
        "use_keyword": true/false,
        "vector_weight": 0.0-1.0,
        "graph_weight": 0.0-1.0,
        "keyword_weight": 0.0-1.0,
        "graph_depth": 1-3,
        "explore_neighborhoods": true/false,
        "reasoning": "why this strategy"
    }},

    "follow_up_queries": [
        {{
            "query": "specific follow-up to explore deeper",
            "reasoning": "why this follow-up helps",
            "target_sources": ["slack", "linear"],
            "priority": 0.0-1.0
        }}
    ],

    "requires_multi_step": true/false,
    "complexity_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of your analysis"
}}

GUIDELINES:
1. ENTITIES: Extract ALL mentioned entities. Include likely aliases (e.g., "JavaScript" -> ["JS", "Javascript"]). Use context_hint for ambiguous names.

2. RELATIONSHIPS: Infer likely relationships to explore. If someone asks about a person's work, suggest WORKS_ON relationships. For project questions, suggest RELATED_TO, BLOCKED_BY.

3. TEMPORAL: Compute actual ISO dates from the current datetime. Handle:
   - "last week" -> 7 days ago to now
   - "yesterday" -> that day's full range
   - "Q3" -> July 1 to Sept 30
   - "recently" -> last 14 days
   - Use null for open bounds

4. SOURCE PRIORITY: Set higher weights (0.8-1.0) for likely relevant sources:
   - Technical questions -> github, linear high
   - People questions -> slack, bamboohr high
   - Sales/deals -> attio, gong high
   - Documentation -> notion high
   - Set 0.0-0.3 for unlikely sources

5. SEARCH STRATEGY:
   - Entity-heavy queries -> higher graph_weight
   - Keyword-specific queries -> higher keyword_weight
   - Semantic/conceptual queries -> higher vector_weight
   - Weights should roughly sum to 1.0

6. FOLLOW-UP QUERIES: Generate 2-4 queries that would help if initial results are insufficient:
   - Target under-represented sources
   - Explore specific entities found
   - Narrow down time ranges
   - Try alternative phrasings

7. COMPLEXITY: Set requires_multi_step=true and high complexity_score for:
   - Questions requiring information synthesis
   - Comparisons across multiple entities
   - Queries spanning multiple data sources
   - Aggregation or trend analysis

Respond with ONLY the JSON object, no markdown, no explanation."""

# Lightweight prompt for non-agentic queries that skips follow-ups, source priority detail,
# and agentic exploration hints. Cuts token usage ~50% for typical recall queries.
LIGHTWEIGHT_UNDERSTANDING_PROMPT = """You are a query understanding system for a memory lake.

CURRENT DATETIME: {current_datetime}

QUERY: {query}

Analyze this query and return a JSON object:

{{
    "intent": "search|question|temporal|comparison|navigation|aggregation",
    "answer_type": "list|summary|fact|explanation|comparison|timeline",
    "entities": [
        {{
            "name": "exact name",
            "type": "PERSON|ORGANIZATION|PRODUCT|PROJECT|TECHNOLOGY|CONCEPT|LOCATION|EVENT|TEAM",
            "confidence": 0.0-1.0,
            "aliases": ["alternative names"]
        }}
    ],
    "relationships": [
        {{
            "from_entity": "entity name",
            "relationship_type": "WORKS_ON|MENTIONED_IN|RELATED_TO|CREATED|OWNS|MANAGES",
            "to_entity": "target entity or null",
            "importance": 0.0-1.0
        }}
    ],
    "temporal": [
        {{
            "type": "relative|absolute|range",
            "text": "original temporal phrase",
            "start_date": "ISO 8601 or null",
            "end_date": "ISO 8601 or null"
        }}
    ],
    "expanded_queries": ["rephrasing 1", "rephrasing 2"],
    "keywords": ["important", "search", "terms"],
    "search_strategy": {{
        "vector_weight": 0.0-1.0,
        "graph_weight": 0.0-1.0,
        "keyword_weight": 0.0-1.0,
        "graph_depth": 1-3,
        "reasoning": "brief strategy note"
    }},
    "complexity_score": 0.0-1.0,
    "reasoning": "brief analysis"
}}

GUIDELINES:
- Extract ALL mentioned entities with aliases
- Compute ISO dates from current datetime for temporal references
- Entity-heavy queries -> higher graph_weight; semantic queries -> higher vector_weight
- Weights should roughly sum to 1.0

Respond with ONLY the JSON object."""


class QueryUnderstanding:
    """Comprehensive LLM-based query understanding.

    Extracts ALL information in a single LLM call for efficiency:
    - Intent, entities, relationships
    - Temporal references with computed dates
    - Query expansions and keywords
    - Source prioritization
    - Search strategy recommendations
    - Follow-up queries for agentic search
    """

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
        lightweight: bool = False,
    ) -> UnderstandingResult:
        """Understand a query comprehensively using a single LLM call.

        Args:
            query: The query to understand
            expand_query: Whether to include query expansions in result
            extract_entities: Whether to include entity mentions in result
            detect_temporal: Whether to include temporal references in result
            lightweight: Use a smaller prompt that skips follow-ups/source priority

        Returns:
            UnderstandingResult with all extracted information
        """
        from khora.config.llm import LiteLLMConfig, acompletion

        config = self._llm_config or LiteLLMConfig()
        model = self._model or config.model

        # Lightweight mode uses fewer tokens for the prompt and response
        max_tokens = 1200 if lightweight else 2000

        # Use appropriate settings for structured extraction
        extraction_config = LiteLLMConfig(
            model=model,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=max_tokens,
        )

        try:
            current_dt = datetime.utcnow().isoformat() + "Z"
            template = LIGHTWEIGHT_UNDERSTANDING_PROMPT if lightweight else COMPREHENSIVE_UNDERSTANDING_PROMPT
            prompt = template.format(
                query=query,
                current_datetime=current_dt,
            )
            response = await acompletion(prompt, extraction_config)

            # Parse comprehensive JSON response
            result = self._parse_comprehensive_response(response, query)

            # Filter based on settings (but we extracted everything efficiently)
            if not expand_query:
                result.expanded_queries = []
            if not extract_entities:
                result.entities = []
                result.relationships = []
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
                confidence=0.3,
            )

    def _parse_comprehensive_response(self, response: str, original_query: str) -> UnderstandingResult:
        """Parse the comprehensive LLM response.

        Args:
            response: Raw LLM response
            original_query: Original query text

        Returns:
            Fully populated UnderstandingResult
        """
        # Clean up response (handle markdown code blocks)
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            if lines[-1].strip() == "```":
                response = "\n".join(lines[1:-1])
            else:
                response = "\n".join(lines[1:])
            response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse query understanding response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return UnderstandingResult(
                original_query=original_query,
                intent=QueryIntent.SEARCH,
                confidence=0.3,
            )

        # Parse intent
        intent_map = {
            "search": QueryIntent.SEARCH,
            "question": QueryIntent.QUESTION,
            "temporal": QueryIntent.TEMPORAL,
            "comparison": QueryIntent.COMPARISON,
            "navigation": QueryIntent.NAVIGATION,
            "aggregation": QueryIntent.AGGREGATION,
        }
        intent = intent_map.get(data.get("intent", "search").lower(), QueryIntent.UNKNOWN)

        # Parse answer type
        answer_map = {
            "list": AnswerType.LIST,
            "summary": AnswerType.SUMMARY,
            "fact": AnswerType.FACT,
            "explanation": AnswerType.EXPLANATION,
            "comparison": AnswerType.COMPARISON,
            "timeline": AnswerType.TIMELINE,
        }
        answer_type = answer_map.get(data.get("answer_type", "summary").lower(), AnswerType.UNKNOWN)

        # Parse entities
        entities = []
        for e in data.get("entities", []):
            entities.append(
                EntityMention(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "CONCEPT"),
                    confidence=float(e.get("confidence", 1.0)),
                    aliases=e.get("aliases", []),
                    context_hint=e.get("context_hint", ""),
                )
            )

        # Parse relationships
        relationships = []
        for r in data.get("relationships", []):
            relationships.append(
                RelationshipHint(
                    from_entity=r.get("from_entity", ""),
                    relationship_type=r.get("relationship_type", "RELATED_TO"),
                    to_entity=r.get("to_entity"),
                    importance=float(r.get("importance", 1.0)),
                )
            )

        # Parse temporal references
        temporal_refs = []
        for t in data.get("temporal", []):
            start_date = self._parse_iso_date(t.get("start_date"))
            end_date = self._parse_iso_date(t.get("end_date"))
            temporal_refs.append(
                TemporalReference(
                    type=t.get("type", "relative"),
                    text=t.get("text", ""),
                    start_date=start_date,
                    end_date=end_date,
                )
            )

        # Parse source priority
        sp_data = data.get("source_priority", {})
        source_priority = SourcePriority(
            slack=float(sp_data.get("slack", 1.0)),
            linear=float(sp_data.get("linear", 1.0)),
            notion=float(sp_data.get("notion", 1.0)),
            attio=float(sp_data.get("attio", 1.0)),
            gong=float(sp_data.get("gong", 1.0)),
            github=float(sp_data.get("github", 1.0)),
            bamboohr=float(sp_data.get("bamboohr", 1.0)),
        )

        # Parse search strategy
        ss_data = data.get("search_strategy", {})
        search_strategy = SearchStrategy(
            use_vector=ss_data.get("use_vector", True),
            use_graph=ss_data.get("use_graph", True),
            use_keyword=ss_data.get("use_keyword", True),
            vector_weight=float(ss_data.get("vector_weight", 0.4)),
            graph_weight=float(ss_data.get("graph_weight", 0.3)),
            keyword_weight=float(ss_data.get("keyword_weight", 0.3)),
            graph_depth=int(ss_data.get("graph_depth", 2)),
            explore_neighborhoods=ss_data.get("explore_neighborhoods", True),
            strategy_reasoning=ss_data.get("reasoning", ""),
        )

        # Parse follow-up queries
        follow_ups = []
        for fq in data.get("follow_up_queries", []):
            follow_ups.append(
                FollowUpQuery(
                    query=fq.get("query", ""),
                    reasoning=fq.get("reasoning", ""),
                    target_sources=fq.get("target_sources", []),
                    priority=float(fq.get("priority", 1.0)),
                )
            )

        # Compute source_filters — tools with priority < 0.1 should be excluded
        source_filters = []
        sp_fields = {
            "slack": source_priority.slack,
            "linear": source_priority.linear,
            "notion": source_priority.notion,
            "attio": source_priority.attio,
            "gong": source_priority.gong,
            "github": source_priority.github,
            "bamboohr": source_priority.bamboohr,
        }
        for tool_name, weight in sp_fields.items():
            if weight < 0.1:
                source_filters.append(tool_name)

        return UnderstandingResult(
            original_query=original_query,
            intent=intent,
            answer_type=answer_type,
            entities=entities,
            relationships=relationships,
            temporal_references=temporal_refs,
            expanded_queries=data.get("expanded_queries", []),
            keywords=data.get("keywords", []),
            source_priority=source_priority,
            search_strategy=search_strategy,
            source_filters=source_filters,
            follow_up_queries=follow_ups,
            requires_multi_step=data.get("requires_multi_step", False),
            complexity_score=float(data.get("complexity_score", 0.5)),
            confidence=float(data.get("confidence", 1.0)),
            reasoning=data.get("reasoning", ""),
        )

    def _parse_iso_date(self, date_str: str | None) -> datetime | None:
        """Parse an ISO 8601 date string from LLM output.

        Args:
            date_str: ISO date string or None

        Returns:
            datetime object or None
        """
        if not date_str or date_str == "null":
            return None

        try:
            date_str = date_str.strip()

            # Remove trailing Z
            if date_str.endswith("Z"):
                date_str = date_str[:-1]

            # Try common formats
            parsed_dt: datetime | None = None
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%d",
            ]:
                try:
                    parsed_dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue

            # Last resort
            if parsed_dt is None:
                parsed_dt = datetime.fromisoformat(date_str)

            # Validate the parsed date
            if parsed_dt is not None:
                now = datetime.now()
                # Reject dates in the far future (> 1 year from now)
                if parsed_dt > now + timedelta(days=365):
                    logger.warning(f"Rejected future date from LLM: {date_str}")
                    return None
                # Reject dates before 2000
                if parsed_dt.year < 2000:
                    logger.warning(f"Rejected ancient date from LLM: {date_str}")
                    return None

            return parsed_dt

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse ISO date '{date_str}': {e}")
            return None

    def _extract_keywords_simple(self, query: str) -> list[str]:
        """Simple keyword extraction without LLM (fallback).

        Args:
            query: Query text

        Returns:
            List of keywords
        """
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
            "about",
            "tell",
            "find",
            "show",
            "get",
        }

        words = query.lower().split()
        keywords = [
            w.strip(".,!?;:'\"()[]{}")
            for w in words
            if w.lower().strip(".,!?;:'\"()[]{}") not in stopwords and len(w) > 2
        ]

        return keywords
