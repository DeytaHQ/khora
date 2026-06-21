"""Temporal query detection and signal classification.

Three-tier cascade for detecting temporal intent in queries:
1. Aho-Corasick dictionary lookup (~1-10μs via Rust, or Python fallback)
2. Model2Vec embedding centroid (~40-50μs, optional)
3. LLM-based query understanding (existing, not invoked here)

Each detected query is classified into a TemporalCategory that drives
retrieval parameters (recency weight, sort order, decay rate).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

from khora._accel import detect_temporal_category
from khora.telemetry.metrics import metric_counter as _metric_counter

if TYPE_CHECKING:
    from khora.core.diagnostics import Degradation


class TemporalCategory(str, Enum):
    """Classification of temporal query intent."""

    NONE = "none"
    EXPLICIT = "explicit"  # "before April 2024", parseable dates
    STATE_QUERY = "state_query"  # "currently", "now", "does X play"
    ORDINAL = "ordinal"  # "first", "which came earlier"
    AGGREGATE = "aggregate"  # "how many total", "all instances"
    RECENCY = "recency"  # "latest", "most recent"
    CHANGE = "change"  # "changed", "used to", "still"


# Map integer category IDs (from Rust/Python fallback) to enum values
CATEGORY_MAP: dict[int, TemporalCategory] = {
    0: TemporalCategory.NONE,
    1: TemporalCategory.EXPLICIT,
    2: TemporalCategory.STATE_QUERY,
    3: TemporalCategory.ORDINAL,
    4: TemporalCategory.AGGREGATE,
    5: TemporalCategory.RECENCY,
    6: TemporalCategory.CHANGE,
}


@dataclass(frozen=True)
class RetrievalParams:
    """Category-specific retrieval parameters.

    ``default_window_days`` is the synthetic date floor applied when the
    user types a bare temporal adjective ("latest", "recent") that the
    dateparser-based resolver cannot translate into a SQL filter. Only
    used when ``QuerySettings.temporal_recency_floor_enabled`` is True
    AND no anti-recency token is present in the query (see
    :data:`ANTI_RECENCY_TOKENS`). ``None`` means "do not synthesize a
    floor for this category."

    ``prefer_current`` is decoupled from ``temporal_sort``. It controls
    whether the retriever filters out entities/edges whose ``valid_until``
    has passed. Set True only for categories where historical entities
    are wrong (STATE_QUERY, RECENCY, CHANGE). ORDINAL queries
    ("which came first") need historical entities to answer correctly,
    so ``prefer_current`` is False there even though ``temporal_sort``
    is True.
    """

    recency_weight: float
    temporal_sort: bool
    decay_days_override: int | None = None
    recency_floor: float = 0.5  # Default floor for multiplicative recency
    default_window_days: int | None = None
    prefer_current: bool = False


# Category → retrieval behavior mapping
# Weights control the *multiplicative* recency exponent applied to RRF scores.
# Higher weight = stronger penalty for stale chunks (score *= recency^(exp*w)).
# Conservative values protect non-temporal categories (implicit_inference,
# abstention) while still discriminating temporal ones.
RETRIEVAL_PARAMS: dict[TemporalCategory, RetrievalParams] = {
    TemporalCategory.NONE: RetrievalParams(
        recency_weight=0.0, temporal_sort=False, recency_floor=0.5, prefer_current=False
    ),
    TemporalCategory.EXPLICIT: RetrievalParams(
        recency_weight=0.3, temporal_sort=False, recency_floor=0.5, prefer_current=False
    ),
    TemporalCategory.STATE_QUERY: RetrievalParams(
        recency_weight=0.5, temporal_sort=True, recency_floor=0.3, prefer_current=True
    ),
    TemporalCategory.ORDINAL: RetrievalParams(
        recency_weight=0.3,
        temporal_sort=True,
        decay_days_override=None,
        recency_floor=0.5,
        # ORDINAL queries ("which came first", "earliest") need historical
        # entities — filtering by valid_until would discard the very rows
        # that answer the question. Keep prefer_current=False here even
        # though temporal_sort is True.
        prefer_current=False,
    ),
    TemporalCategory.AGGREGATE: RetrievalParams(
        recency_weight=0.0, temporal_sort=False, recency_floor=0.5, prefer_current=False
    ),
    TemporalCategory.RECENCY: RetrievalParams(
        recency_weight=0.5,
        temporal_sort=True,
        decay_days_override=3,
        recency_floor=0.3,
        # LoCoMo --small benchmark showed counterfactual_accuracy regressed
        # 16.7pp with a 14d window because counterfactual queries ask about
        # past hypothetical states that the floor excluded. 30d gives the
        # floor headroom for slightly-older content while still being a
        # meaningful "recent" cutoff. Plus the LLM disambiguation tier (see
        # ``classify_temporal_intent_llm``) catches counterfactual phrasings
        # that the anti-recency token list misses.
        default_window_days=30,
        prefer_current=True,
    ),
    TemporalCategory.CHANGE: RetrievalParams(
        recency_weight=0.4,
        temporal_sort=True,
        decay_days_override=14,
        recency_floor=0.3,
        default_window_days=60,
        prefer_current=True,
    ),
}


# Tokens that signal the user wants historical / "all-time" results.
# When any of these is present, RECENCY / CHANGE categories MUST NOT apply
# a synthetic date floor — even if the category dictionary fires on a
# separate token in the same query. Caller checks via
# :func:`has_anti_recency_token`.
#
# Examples of queries that SHOULD veto the floor:
#   "what action items have we ever discussed for Phoenix"
#   "show me all the meetings we've had over time"
#   "any history of the budget conversation since the beginning"
ANTI_RECENCY_TOKENS: frozenset[str] = frozenset(
    {
        # Bare single-word tokens — these only trigger when they appear
        # word-bounded in the query. Devil's-Advocate review flagged that
        # bare "all"/"any"/"every"/"entire" are too common in legitimate
        # recency queries ("latest from all channels", "any new emails")
        # to be safe singletons. We keep only tokens that unambiguously
        # signal historical scope.
        "ever",
        "history",
        "throughout",
        "previously",
        "originally",
        "initially",
        "hypothetically",
        # Multi-word phrases — these are unambiguous: a user who types
        # "all-time" or "since the beginning" is asking for historical
        # scope, not freshness.
        "history of",
        "any time",
        "anytime",
        "since the beginning",
        "over time",
        "all-time",
        "all time",
        "all the time",
        "of all time",
        "every single",
        "entire history",
        # Counterfactual phrasings — LoCoMo --small showed a 16.7pp
        # counterfactual_accuracy regression because the 14d floor
        # excluded historical chunks these queries need. Veto the floor
        # when the query is hypothetical-past in nature. The LLM
        # disambiguation tier catches the cases this lexicon misses.
        "would have",
        "would not have",
        "wouldn't have",
        "had we",
        "if we had",
        "if i had",
        "if they had",
        "if it had",
        "should have",
        "could have",
        "might have",
        "in the past",
        "back when",
        "back in",
        "at one point",
        "at some point",
    }
)

# Compiled regex — word-boundary match for single-word tokens, raw substring
# match for multi-word phrases (these are already word-bounded by their spaces).
_ANTI_RECENCY_SINGLE = frozenset(t for t in ANTI_RECENCY_TOKENS if " " not in t and "-" not in t)
_ANTI_RECENCY_MULTI = tuple(t for t in ANTI_RECENCY_TOKENS if " " in t or "-" in t)
_ANTI_RECENCY_WORD_RE = re.compile(r"\b(" + "|".join(_ANTI_RECENCY_SINGLE) + r")\b", re.IGNORECASE)


def has_anti_recency_token(query: str) -> bool:
    """Return True iff *query* contains a token that vetoes the recency floor.

    Used by the RECENCY / CHANGE call sites in the retriever to suppress
    the synthetic ``default_window_days`` filter when the user explicitly
    asks for historical or all-time scope.
    """
    if not query:
        return False
    lowered = query.lower()
    if any(phrase in lowered for phrase in _ANTI_RECENCY_MULTI):
        return True
    return _ANTI_RECENCY_WORD_RE.search(query) is not None


@dataclass(frozen=True)
class TemporalSignal:
    """Result of temporal query detection."""

    is_temporal: bool
    category: TemporalCategory
    confidence: float  # 0.0-1.0
    source: str  # "dictionary", "semantic", "none"
    temporal_filter: Any | None = None  # TemporalFilter for EXPLICIT category


# Regex for extracting explicit dates from queries
_DATE_EXTRACT_RE = re.compile(
    r"(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})"
    r"|(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2},?\s+\d{4}\b)"
    r"|(\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{4}\b)",
    re.IGNORECASE,
)

# Semantic detection threshold (near-zero FN, ~30% FP acceptable)
SEMANTIC_THRESHOLD = 0.20


# Tier-2 semantic-fallback degradation counter (ADR-001, #981). When the
# opt-in LLM temporal classifier fails or times out, the detector keeps the
# keyword result (NONE) and bumps this counter. NO namespace_id label -
# cardinality rule. The same event is appended to
# RecallResult.engine_info['degradations'].
_TEMPORAL_SEMANTIC_DEGRADED_COUNTER = _metric_counter(
    "khora.vectorcypher.temporal_semantic_fallback.degraded_total",
    unit="1",
    description="Tier-2 LLM temporal-category fallback failures (degraded to keyword tier), by reason.",
)


class TemporalDetector:
    """Three-tier cascade temporal query detector.

    ``detect`` runs the keyword tier only (sync, zero LLM cost). The opt-in
    Tier-2 LLM fallback (#981) - which classifies German / paraphrased queries
    the English keyword dictionary misses - is reached via the async
    :meth:`detect_async` and only when the detector is built with
    ``llm_enabled=True``.
    """

    def __init__(
        self,
        *,
        semantic_enabled: bool = False,
        centroid: Any | None = None,
        llm_enabled: bool = False,
        llm_model: str | None = None,
        llm_timeout: float = 3.0,
    ):
        self._semantic_enabled = semantic_enabled
        self._centroid = centroid  # Pre-normalized Model2Vec centroid (ndarray)
        self._llm_enabled = llm_enabled
        self._llm_model = llm_model
        self._llm_timeout = llm_timeout

    def detect(self, query: str, *, query_embedding: list[float] | None = None) -> TemporalSignal:
        """Detect temporal intent in a query.

        Args:
            query: The user query text.
            query_embedding: Optional pre-computed query embedding for Tier 2 semantic check.

        Returns:
            TemporalSignal with category, confidence, and optional temporal filter.
        """
        # Tier 1: Aho-Corasick dictionary (always runs)
        cat_id = detect_temporal_category(query)
        if cat_id > 0:
            category = CATEGORY_MAP[cat_id]
            temporal_filter = None
            if category == TemporalCategory.EXPLICIT:
                temporal_filter = self._extract_date_filter(query)
            return TemporalSignal(
                is_temporal=True,
                category=category,
                confidence=0.9,
                source="dictionary",
                temporal_filter=temporal_filter,
            )

        # Tier 2: Model2Vec centroid (optional, only if Tier 1 missed)
        similarity = 0.0
        if self._semantic_enabled and self._centroid is not None and query_embedding is not None:
            try:
                import numpy as np

                similarity = float(np.dot(query_embedding, self._centroid))
                if similarity > SEMANTIC_THRESHOLD:
                    return TemporalSignal(
                        is_temporal=True,
                        category=TemporalCategory.STATE_QUERY,  # default for semantic
                        confidence=similarity,
                        source="semantic",
                        temporal_filter=None,
                    )
            except ImportError:
                pass

        # No temporal signal detected
        return TemporalSignal(
            is_temporal=False,
            category=TemporalCategory.NONE,
            confidence=1.0 - similarity,
            source="none",
            temporal_filter=None,
        )

    async def detect_async(
        self,
        query: str,
        *,
        query_embedding: list[float] | None = None,
        degradations: list[Degradation] | None = None,
    ) -> TemporalSignal:
        """Detect temporal intent, consulting the Tier-2 LLM fallback (#981).

        Runs the keyword tier first (via :meth:`detect`). When that returns
        NONE *and* the detector was built with ``llm_enabled=True``, route the
        query to an LLM classifier that resolves German / paraphrased temporal
        intent the English keyword dictionary cannot match. On any LLM failure
        or timeout the result degrades back to the keyword signal (NONE) and a
        structured :class:`~khora.core.diagnostics.Degradation` is appended to
        ``degradations`` if a list was supplied (ADR-001).

        Zero LLM cost when ``llm_enabled`` is False or the keyword tier already
        fired - the call is skipped entirely.
        """
        keyword_signal = self.detect(query, query_embedding=query_embedding)
        if keyword_signal.category is not TemporalCategory.NONE:
            return keyword_signal

        if not self._llm_enabled or not query.strip():
            return keyword_signal

        try:
            category, confidence = await classify_temporal_category_llm(
                query, model=self._llm_model, timeout=self._llm_timeout
            )
        except Exception as exc:
            logger.warning(
                "Tier-2 temporal semantic fallback failed; degrading to keyword tier: {}",
                exc,
                exc_info=True,
            )
            _TEMPORAL_SEMANTIC_DEGRADED_COUNTER.add(1, {"reason": "llm_failed"})
            if degradations is not None:
                degradations.append(
                    {
                        "component": "vectorcypher.temporal_semantic_fallback",
                        "reason": "llm_failed",
                        "detail": "Tier-2 LLM temporal classifier raised; kept keyword (NONE) result.",
                        "exception": repr(exc),
                    }
                )
            return keyword_signal

        if category is TemporalCategory.NONE:
            return keyword_signal

        temporal_filter = None
        if category == TemporalCategory.EXPLICIT:
            temporal_filter = self._extract_date_filter(query)
        return TemporalSignal(
            is_temporal=True,
            category=category,
            confidence=confidence,
            source="semantic",
            temporal_filter=temporal_filter,
        )

    def _extract_date_filter(self, query: str) -> Any | None:
        """Extract a ChunkTemporalFilter from explicit date mentions in the query."""
        from khora.core.temporal import ChunkTemporalFilter

        date_match = _DATE_EXTRACT_RE.search(query)
        if not date_match:
            return None

        date_str = date_match.group(0)
        try:
            parsed_dt = self._parse_datetime(date_str)
        except ValueError:
            return None

        query_lower = query.lower()
        if "before" in query_lower:
            return ChunkTemporalFilter(occurred_before=parsed_dt)
        elif "after" in query_lower or "since" in query_lower:
            return ChunkTemporalFilter(occurred_after=parsed_dt)
        else:
            # Within ±30 days of the mentioned date
            return ChunkTemporalFilter(
                occurred_after=parsed_dt - timedelta(days=30),
                occurred_before=parsed_dt + timedelta(days=30),
            )

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse a datetime string from various formats."""
        try:
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            pass
        for fmt in (
            "%Y/%m/%d",
            "%B %d, %Y",
            "%B %d %Y",
            "%d %B %Y",
        ):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {value}")


def get_retrieval_params(signal: TemporalSignal) -> RetrievalParams:
    """Get retrieval parameters for a temporal signal."""
    return RETRIEVAL_PARAMS[signal.category]


# ---------------------------------------------------------------------------
# Tier-3 LLM disambiguation
# ---------------------------------------------------------------------------
#
# The Aho-Corasick dictionary + anti-recency token list catch the clear cases.
# But LoCoMo benchmarks showed a 16.7pp counterfactual regression because
# phrasings like "what would the team have decided if X had happened last
# quarter" trip RECENCY on "last quarter" while being structurally historical.
# A short LLM call disambiguates: RECENT / HISTORICAL / COUNTERFACTUAL /
# NEUTRAL. Output is cached per-query so the LLM cost is bounded by query
# distinct-count, not query rate.
#
# Gated by ``RetrieverConfig.temporal_llm_disambiguation_enabled`` and only
# invoked when an ambiguity-trigger token is present — most "latest action
# items" queries skip the call entirely.


class TemporalIntent(str, Enum):
    """LLM-classified temporal intent of a query."""

    RECENT = "recent"  # User wants recent results — apply the floor.
    HISTORICAL = "historical"  # User wants past-state results — veto the floor.
    COUNTERFACTUAL = "counterfactual"  # Hypothetical past — veto the floor.
    NEUTRAL = "neutral"  # No temporal preference — veto the floor.


# Tokens that signal a query MIGHT be misclassified by the Aho-Corasick tier.
# When ANY of these appears in a query that also fires RECENCY/CHANGE, route
# to the LLM disambiguator for a final call. Bounded substrings, not regex.
_AMBIGUITY_TRIGGER_TOKENS: frozenset[str] = frozenset(
    {
        "would",
        "could",
        "should",
        "might",
        "if ",
        "unless",
        "imagine",
        "suppose",
        "what if",
        "previously",
        "originally",
        "earlier",
        "back in",
        "back when",
        "prior to",
        "before the",
        "in the past",
    }
)


def has_ambiguity_trigger(query: str) -> bool:
    """Return True iff the query contains a token that might fool the
    dictionary tier and warrants LLM disambiguation."""
    if not query:
        return False
    lowered = query.lower()
    return any(t in lowered for t in _AMBIGUITY_TRIGGER_TOKENS)


_TEMPORAL_INTENT_PROMPT = """Classify the temporal intent of this query.

Categories:
- RECENT: user wants results from the recent past (last few days/weeks).
  Examples: "latest action items", "recent emails", "what did the team
  decide this morning".
- HISTORICAL: user wants results from the distant past, all-time, or
  historical archive. Examples: "show me the entire history of the
  Phoenix project", "what was our policy in 2021", "old discussions
  about pricing".
- COUNTERFACTUAL: user asks about a hypothetical past state — what
  WOULD have happened, what IF X HAD occurred. Examples: "what would
  have happened if we'd shipped on time", "if Alice had taken the
  Italy job", "should we have decided differently".
- NEUTRAL: no strong temporal preference — the query is about content,
  not time.

Query: {query}

Respond with EXACTLY one word: RECENT, HISTORICAL, COUNTERFACTUAL, or
NEUTRAL.
"""


# Process-level cache: query string → (intent, confidence). Bounded.
# Used by classify_temporal_intent_llm; tests can clear it with
# ``_TEMPORAL_INTENT_CACHE.clear()``.
_TEMPORAL_INTENT_CACHE: dict[str, tuple[TemporalIntent, float]] = {}
_TEMPORAL_INTENT_CACHE_MAX_SIZE = 1024


async def classify_temporal_intent_llm(
    query: str,
    *,
    model: str | None = None,
    timeout: float = 3.0,
) -> tuple[TemporalIntent, float]:
    """Classify ``query`` into a :class:`TemporalIntent` via a small LLM call.

    Caches results per-query (process-local) so repeated identical
    queries cost zero. Returns ``(intent, confidence)`` — confidence is
    1.0 on cache hits and on parsable LLM responses; 0.0 when the LLM
    response can't be parsed (caller should treat as NEUTRAL).

    Cost: one short completion (~50 tokens out, fast model). Uses
    ``khora.config.llm.acompletion`` so it inherits the same retry,
    timeout, and telemetry behavior as other LLM calls in khora.

    Caller is responsible for gating the call on a feature flag — this
    function does NOT check whether disambiguation is enabled.
    """
    cache_key = query.strip().lower()
    if cache_key in _TEMPORAL_INTENT_CACHE:
        return _TEMPORAL_INTENT_CACHE[cache_key]

    try:
        from khora.config.llm import LiteLLMConfig, acompletion
    except ImportError:
        return TemporalIntent.NEUTRAL, 0.0

    llm_config = LiteLLMConfig(
        model=model or "gpt-4o-mini",
        temperature=0.0,
        max_tokens=20,
        timeout=timeout,
    )

    try:
        response = await acompletion(
            prompt=_TEMPORAL_INTENT_PROMPT.format(query=query[:500]),
            config=llm_config,
            _telemetry_op="temporal_intent_classification",
        )
    except Exception:
        return TemporalIntent.NEUTRAL, 0.0

    response_text = (response or "").strip().upper()
    # Take only the first word — the prompt asks for one word; some
    # models add a period or trailing explanation.
    first_word = response_text.split()[0].rstrip(".,;:!?") if response_text else ""

    intent_map = {
        "RECENT": TemporalIntent.RECENT,
        "HISTORICAL": TemporalIntent.HISTORICAL,
        "COUNTERFACTUAL": TemporalIntent.COUNTERFACTUAL,
        "NEUTRAL": TemporalIntent.NEUTRAL,
    }
    intent = intent_map.get(first_word, TemporalIntent.NEUTRAL)
    confidence = 1.0 if first_word in intent_map else 0.0

    # Bounded cache: evict oldest when full. dict preserves insertion
    # order in Python 3.7+, so popitem(last=False) gives us FIFO.
    if len(_TEMPORAL_INTENT_CACHE) >= _TEMPORAL_INTENT_CACHE_MAX_SIZE:
        try:
            oldest_key = next(iter(_TEMPORAL_INTENT_CACHE))
            del _TEMPORAL_INTENT_CACHE[oldest_key]
        except StopIteration:
            pass
    _TEMPORAL_INTENT_CACHE[cache_key] = (intent, confidence)

    return intent, confidence


# ---------------------------------------------------------------------------
# Tier-2 LLM temporal-category classifier (#981)
# ---------------------------------------------------------------------------
#
# The Aho-Corasick keyword tier is an English-only, phrase-literal dictionary.
# German queries ("Was ist der letzte Deploy?") and paraphrased-English queries
# ("evolved over time", "timeline of milestones") that carry real temporal
# intent collapse to NONE because no dictionary phrase matches. This classifier
# resolves them into the correct six-way TemporalCategory via one short LLM
# call. It is the opt-in Tier-2 fallback reached from
# ``TemporalDetector.detect_async`` and only fires when the keyword tier
# returned NONE - so the cost is bounded by the count of distinct
# keyword-missed queries, not by query rate (results are cached per-query).
#
# Distinct from ``classify_temporal_intent_llm`` (RECENT/HISTORICAL/...): that
# one disambiguates the recency *floor* for queries the keyword tier already
# classified; this one *classifies* queries the keyword tier missed entirely.


_TEMPORAL_CATEGORY_PROMPT = """Classify the temporal-retrieval intent of this query.
The query may be in any language.

Categories:
- EXPLICIT: names a specific date, month, year, or bounded period.
  Examples: "What did Acme ship in March 2024?", "events since last week".
- STATE_QUERY: asks for the CURRENT state of something.
  Examples: "Who is currently leading the project?", "who is the account manager?".
- ORDINAL: asks about order, sequence, or which came first/earliest.
  Examples: "Which version came first?", "give me a timeline of the milestones".
- AGGREGATE: asks for a count or total over time.
  Examples: "How many releases did we ship in total?".
- RECENCY: asks for the most recent / latest thing.
  Examples: "What's the most recent deploy?", "latest news".
- CHANGE: asks how something changed, evolved, or what it used to be.
  Examples: "What did Bob's role used to be?", "how did Alice's role evolve over time?".
- NONE: no temporal-retrieval intent - the query is about content, not time.

Query: {query}

Respond with EXACTLY one word: EXPLICIT, STATE_QUERY, ORDINAL, AGGREGATE,
RECENCY, CHANGE, or NONE.
"""


# Process-level cache: query string -> (category, confidence). Bounded FIFO.
# Tests clear it via ``_TEMPORAL_CATEGORY_CACHE.clear()``.
_TEMPORAL_CATEGORY_CACHE: dict[str, tuple[TemporalCategory, float]] = {}
_TEMPORAL_CATEGORY_CACHE_MAX_SIZE = 1024

_CATEGORY_WORD_MAP: dict[str, TemporalCategory] = {
    "EXPLICIT": TemporalCategory.EXPLICIT,
    "STATE_QUERY": TemporalCategory.STATE_QUERY,
    "ORDINAL": TemporalCategory.ORDINAL,
    "AGGREGATE": TemporalCategory.AGGREGATE,
    "RECENCY": TemporalCategory.RECENCY,
    "CHANGE": TemporalCategory.CHANGE,
    "NONE": TemporalCategory.NONE,
}


async def classify_temporal_category_llm(
    query: str,
    *,
    model: str | None = None,
    timeout: float = 3.0,
) -> tuple[TemporalCategory, float]:
    """Classify ``query`` into a :class:`TemporalCategory` via a small LLM call.

    Caches results per-query (process-local) so repeated identical queries
    cost zero. Returns ``(category, confidence)`` - confidence is 1.0 on
    cache hits and parsable responses, 0.0 (and ``NONE``) when the response
    can't be parsed.

    Cost: one short completion (~20 tokens out, fast model). Uses
    ``khora.config.llm.acompletion`` so it inherits the same retry, timeout,
    and telemetry behavior as other LLM calls.

    Raises rather than swallowing transport errors - the caller
    (``TemporalDetector.detect_async``) owns the degrade-to-keyword decision
    so it can record the ADR-001 Degradation. ``ImportError`` (litellm absent)
    is the one exception treated as a clean NONE.
    """
    cache_key = query.strip().lower()
    if cache_key in _TEMPORAL_CATEGORY_CACHE:
        return _TEMPORAL_CATEGORY_CACHE[cache_key]

    try:
        from khora.config.llm import LiteLLMConfig, acompletion
    except ImportError:
        return TemporalCategory.NONE, 0.0

    llm_config = LiteLLMConfig(
        model=model or "gpt-4o-mini",
        temperature=0.0,
        max_tokens=20,
        timeout=timeout,
    )

    response = await acompletion(
        prompt=_TEMPORAL_CATEGORY_PROMPT.format(query=query[:500]),
        config=llm_config,
        _telemetry_op="temporal_category_classification",
    )

    response_text = (response or "").strip().upper()
    first_word = response_text.split()[0].rstrip(".,;:!?") if response_text else ""
    category = _CATEGORY_WORD_MAP.get(first_word, TemporalCategory.NONE)
    confidence = 1.0 if first_word in _CATEGORY_WORD_MAP else 0.0

    if len(_TEMPORAL_CATEGORY_CACHE) >= _TEMPORAL_CATEGORY_CACHE_MAX_SIZE:
        try:
            oldest_key = next(iter(_TEMPORAL_CATEGORY_CACHE))
            del _TEMPORAL_CATEGORY_CACHE[oldest_key]
        except StopIteration:
            pass
    _TEMPORAL_CATEGORY_CACHE[cache_key] = (category, confidence)

    return category, confidence


__all__ = [
    "ANTI_RECENCY_TOKENS",
    "CATEGORY_MAP",
    "RETRIEVAL_PARAMS",
    "RetrievalParams",
    "TemporalCategory",
    "TemporalDetector",
    "TemporalIntent",
    "TemporalSignal",
    "classify_temporal_category_llm",
    "classify_temporal_intent_llm",
    "get_retrieval_params",
    "has_ambiguity_trigger",
    "has_anti_recency_token",
]
