"""Direct unit tests for ``khora.query.temporal_detection.has_anti_recency_token``.

Devil's-Advocate demand #2 (PR #571 follow-up review): the token list
that vetoes the synthetic RECENCY floor must be unit-tested per token,
with explicit false-positive coverage for common adjectives that share
surface form with anti-recency markers ("all", "any", "every", "entire").
"""

from __future__ import annotations

import pytest

from khora.query.temporal_detection import (
    ANTI_RECENCY_TOKENS,
    has_anti_recency_token,
)

# ---------------------------------------------------------------------------
# Positive: each anti-recency token should be detected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # Single-word tokens
        "have we ever discussed Phoenix",
        "what's the history on this account",
        "throughout the engagement we noticed",
        # Multi-word phrases
        "show me the history of the budget conversation",
        "any time we talked about pricing",
        "anytime someone mentioned Q3",
        "since the beginning of the project",
        "how has this evolved over time",
        "all-time leaders in opens",
        "all time most-clicked subjects",
        "all the time someone asked",
        "the best email of all time",
        "every single time it failed",
        "the entire history of this thread",
    ],
)
def test_anti_recency_token_detected(query: str) -> None:
    assert has_anti_recency_token(query), f"expected veto on {query!r}"


# ---------------------------------------------------------------------------
# Negative: legitimate recency queries that share surface tokens but are
# NOT historical-scope must NOT trigger the veto. These were the strings
# the Devil's Advocate flagged as silently-truncated by the v1 token list.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "latest from all the slack channels",  # "all" alone is fine
        "any updates on the budget",  # "any" alone is fine
        "show me every meeting from last week",  # "every" alone is fine
        "the entire team's standup notes from this morning",  # "entire" alone is fine
        "what are the latest action items",
        "recent emails about pricing",
        "newest decisions from the leadership offsite",
        "most recent slack thread on Phoenix",
    ],
)
def test_legitimate_recency_query_not_vetoed(query: str) -> None:
    assert not has_anti_recency_token(query), f"unexpected veto on {query!r}"


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_empty_query_is_not_vetoed() -> None:
    assert has_anti_recency_token("") is False
    assert has_anti_recency_token("   ") is False


def test_case_insensitive() -> None:
    assert has_anti_recency_token("EVER") is True
    assert has_anti_recency_token("History Of The Budget") is True
    assert has_anti_recency_token("All-Time Top Performers") is True


def test_word_boundary_for_single_word_tokens() -> None:
    # "however" contains "ever" as substring but is not "ever" as a word.
    # The regex uses \b so this must not trigger.
    assert has_anti_recency_token("however the deal went sideways") is False
    # but a real "ever" still trips.
    assert has_anti_recency_token("have we ever discussed this") is True


def test_token_set_contains_canonical_phrases() -> None:
    """Sanity: the token set should include the canonical anti-recency
    multi-word phrases. Future contributors must not silently drop them."""
    must_contain = {
        "ever",
        "history",
        "history of",
        "since the beginning",
        "over time",
        "all-time",
        "of all time",
    }
    missing = must_contain - ANTI_RECENCY_TOKENS
    assert not missing, f"required tokens dropped from ANTI_RECENCY_TOKENS: {missing}"


def test_token_set_does_not_contain_overly_broad_singletons() -> None:
    """Devil's-Advocate review removed bare 'all'/'any'/'every'/'entire'
    because they false-positive on common recency queries. Keep them out."""
    must_not_contain = {"all", "any", "every", "entire"}
    overlap = must_not_contain & ANTI_RECENCY_TOKENS
    assert not overlap, (
        f"ANTI_RECENCY_TOKENS contains overly-broad singleton(s): {overlap}. "
        "These cause silent veto of legitimate recency queries like "
        "'latest from all channels' / 'any new emails'."
    )
