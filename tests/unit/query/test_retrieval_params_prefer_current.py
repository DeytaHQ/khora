"""Unit tests for the ``prefer_current`` field on ``RetrievalParams``.

Issue #569 decoupled ``prefer_current`` from ``temporal_sort`` so that
ORDINAL queries ("which came first") retain historical entities even
though they sort temporally. STATE_QUERY / RECENCY / CHANGE remain the
only categories that filter out expired entities.
"""

from __future__ import annotations

import pytest

from khora.query.temporal_detection import (
    RETRIEVAL_PARAMS,
    RetrievalParams,
    TemporalCategory,
)

# ---------------------------------------------------------------------------
# Dataclass default
# ---------------------------------------------------------------------------


def test_prefer_current_defaults_false() -> None:
    """``prefer_current`` defaults to False so omitted call sites don't
    silently start filtering historical entities."""
    params = RetrievalParams(recency_weight=0.0, temporal_sort=False)
    assert params.prefer_current is False


# ---------------------------------------------------------------------------
# Per-category assignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category, expected_prefer_current",
    [
        (TemporalCategory.NONE, False),
        (TemporalCategory.EXPLICIT, False),
        (TemporalCategory.STATE_QUERY, True),
        (TemporalCategory.ORDINAL, False),
        (TemporalCategory.AGGREGATE, False),
        (TemporalCategory.RECENCY, True),
        (TemporalCategory.CHANGE, True),
    ],
)
def test_per_category_prefer_current(category: TemporalCategory, expected_prefer_current: bool) -> None:
    """Each TemporalCategory has the documented ``prefer_current`` value."""
    params = RETRIEVAL_PARAMS[category]
    assert params.prefer_current is expected_prefer_current, (
        f"{category.value}: expected prefer_current={expected_prefer_current}, got {params.prefer_current}"
    )


# ---------------------------------------------------------------------------
# Regression: ORDINAL must NOT filter historical entities
# ---------------------------------------------------------------------------


def test_ordinal_decoupled_from_temporal_sort() -> None:
    """Regression for #569: ORDINAL queries ("which came first") sort
    temporally but must keep historical entities. Before this change,
    ``prefer_current=_tp.temporal_sort`` collapsed both flags, so ORDINAL
    silently filtered out the very rows needed to answer the question.
    """
    params = RETRIEVAL_PARAMS[TemporalCategory.ORDINAL]
    assert params.temporal_sort is True
    assert params.prefer_current is False


def test_state_query_and_recency_still_filter_current() -> None:
    """STATE_QUERY and RECENCY remain coupled — both filter expired entities.
    Decoupling must not regress these existing behaviors.
    """
    state_params = RETRIEVAL_PARAMS[TemporalCategory.STATE_QUERY]
    recency_params = RETRIEVAL_PARAMS[TemporalCategory.RECENCY]
    change_params = RETRIEVAL_PARAMS[TemporalCategory.CHANGE]

    assert state_params.temporal_sort is True
    assert state_params.prefer_current is True
    assert recency_params.temporal_sort is True
    assert recency_params.prefer_current is True
    assert change_params.temporal_sort is True
    assert change_params.prefer_current is True


def test_non_temporal_categories_do_not_filter() -> None:
    """NONE / EXPLICIT / AGGREGATE all keep prefer_current=False."""
    for cat in (
        TemporalCategory.NONE,
        TemporalCategory.EXPLICIT,
        TemporalCategory.AGGREGATE,
    ):
        params = RETRIEVAL_PARAMS[cat]
        assert params.prefer_current is False, cat.value
