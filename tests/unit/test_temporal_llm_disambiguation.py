"""Unit tests for the LLM temporal-intent disambiguation tier.

PR #571 follow-up: the LoCoMo benchmark surfaced a 16.7pp counterfactual
regression because phrasings like "what would have happened if X had
shipped last quarter" trip RECENCY on "last quarter" while being
structurally historical. The LLM tier resolves this when the
Aho-Corasick + token-list tiers can't.

These tests pin the LLM call shape, the cache behavior, and the
ambiguity-trigger detection — but mock the LLM itself.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from khora.query.temporal_detection import (
    _TEMPORAL_INTENT_CACHE,
    ANTI_RECENCY_TOKENS,
    TemporalIntent,
    classify_temporal_intent_llm,
    has_ambiguity_trigger,
    has_anti_recency_token,
)


@pytest.fixture(autouse=True)
def _clear_intent_cache():
    """Each test starts with an empty per-query cache."""
    _TEMPORAL_INTENT_CACHE.clear()
    yield
    _TEMPORAL_INTENT_CACHE.clear()


# ---------------------------------------------------------------------------
# has_ambiguity_trigger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "what would have happened if we shipped last week",  # would, if
        "should we have decided differently in Q3",  # should
        "could we have prevented the outage",  # could
        "might we have missed something earlier",  # might, earlier
        "if Alice had taken the Italy job",  # had... wait, "if "
        "imagine the team had agreed yesterday",  # imagine
        "suppose pricing went down last quarter",  # suppose
        "what if the deal had closed in May",  # what if
        "back when the team was three people",  # back when
        "in the past few standups, what was decided",  # in the past
        "previously we had a different policy",  # previously
        "originally the spec said something else",  # originally
        "prior to the rebrand, how did we describe X",  # prior to
        "before the team grew, how did we run standups",  # before the
    ],
)
def test_ambiguity_trigger_detected(query: str) -> None:
    assert has_ambiguity_trigger(query), f"expected trigger on {query!r}"


@pytest.mark.parametrize(
    "query",
    [
        "latest action items",
        "recent emails about pricing",
        "what are the newest decisions",
        "show me action items from yesterday's standup",
    ],
)
def test_ambiguity_trigger_negative(query: str) -> None:
    assert not has_ambiguity_trigger(query), f"unexpected trigger on {query!r}"


def test_ambiguity_trigger_empty() -> None:
    assert has_ambiguity_trigger("") is False


# ---------------------------------------------------------------------------
# classify_temporal_intent_llm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classifier_parses_recent() -> None:
    with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="RECENT")):
        intent, confidence = await classify_temporal_intent_llm("latest action items")
    assert intent == TemporalIntent.RECENT
    assert confidence == 1.0


@pytest.mark.asyncio
async def test_classifier_parses_historical() -> None:
    with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="HISTORICAL")):
        intent, confidence = await classify_temporal_intent_llm("show me the entire history of the Phoenix project")
    assert intent == TemporalIntent.HISTORICAL
    assert confidence == 1.0


@pytest.mark.asyncio
async def test_classifier_parses_counterfactual() -> None:
    with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="COUNTERFACTUAL")):
        intent, confidence = await classify_temporal_intent_llm("what would have happened if we'd shipped on time")
    assert intent == TemporalIntent.COUNTERFACTUAL
    assert confidence == 1.0


@pytest.mark.asyncio
async def test_classifier_parses_neutral() -> None:
    with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="NEUTRAL")):
        intent, confidence = await classify_temporal_intent_llm("what is our pricing")
    assert intent == TemporalIntent.NEUTRAL
    assert confidence == 1.0


@pytest.mark.asyncio
async def test_classifier_handles_trailing_punctuation() -> None:
    """Some models add a period after the one-word answer."""
    with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="RECENT.")):
        intent, confidence = await classify_temporal_intent_llm("latest emails")
    assert intent == TemporalIntent.RECENT
    assert confidence == 1.0


@pytest.mark.asyncio
async def test_classifier_unparsable_response_returns_neutral_zero() -> None:
    with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="The answer depends on...")):
        intent, confidence = await classify_temporal_intent_llm("ambiguous query")
    assert intent == TemporalIntent.NEUTRAL
    assert confidence == 0.0


@pytest.mark.asyncio
async def test_classifier_exception_returns_neutral_zero() -> None:
    """LLM call failure must NOT raise — return neutral/0.0 so the caller
    can fall back to the dictionary tier."""
    with patch(
        "khora.config.llm.acompletion",
        new=AsyncMock(side_effect=TimeoutError("upstream slow")),
    ):
        intent, confidence = await classify_temporal_intent_llm("query")
    assert intent == TemporalIntent.NEUTRAL
    assert confidence == 0.0


@pytest.mark.asyncio
async def test_classifier_caches_per_query() -> None:
    """Repeated identical queries hit the cache — only one LLM call."""
    mock_completion = AsyncMock(return_value="HISTORICAL")
    with patch("khora.config.llm.acompletion", new=mock_completion):
        i1, c1 = await classify_temporal_intent_llm("show history of Phoenix")
        i2, c2 = await classify_temporal_intent_llm("show history of Phoenix")
        i3, c3 = await classify_temporal_intent_llm("SHOW HISTORY OF PHOENIX")  # case
    assert i1 == i2 == i3 == TemporalIntent.HISTORICAL
    # Cache is keyed on lowered+stripped query; case variants share an entry.
    mock_completion.assert_called_once()


# ---------------------------------------------------------------------------
# Counterfactual phrasings now in ANTI_RECENCY_TOKENS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # The exact LoCoMo-shaped counterfactual phrasings that broke
        # the v1 floor synthesis. These should now be vetoed by the
        # dictionary tier without needing the LLM.
        "what would have happened if we shipped last quarter",
        "what would not have happened with the new pricing",
        "if we had decided differently at the offsite",
        "if I had taken that call earlier",
        "if they had escalated when the bug was reported",
        "should have escalated last sprint",
        "could have invested in retention then",
        "might have caught the regression sooner",
        "back when we still had the old team",
        "back in 2022 when the product was unstable",
        "at one point we considered Postgres",
        "at some point the team changed direction",
        "originally the spec called for json",
        "initially we leaned toward async",
        "hypothetically we should have shipped earlier",
        "in the past, this was handled differently",
    ],
)
def test_counterfactual_phrasings_in_anti_recency_set(query: str) -> None:
    assert has_anti_recency_token(query), (
        f"counterfactual phrasing {query!r} not in ANTI_RECENCY_TOKENS — "
        "LoCoMo benchmark regression will reappear if this slips."
    )


def test_required_counterfactual_tokens_present() -> None:
    """Sanity-check guard: future contributors must not drop the
    counterfactual phrasings that the LoCoMo regression depends on."""
    must_contain = {
        "would have",
        "if we had",
        "should have",
        "could have",
        "in the past",
        "back when",
        "back in",
        "hypothetically",
        "previously",
        "originally",
    }
    missing = must_contain - ANTI_RECENCY_TOKENS
    assert not missing, f"counterfactual tokens missing from ANTI_RECENCY_TOKENS: {missing}"
