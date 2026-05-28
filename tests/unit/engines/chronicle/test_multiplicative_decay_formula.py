"""Chronicle #852: pin the multiplicative decay formula.

The formula is

    final_score = relevance * ((1 - w) + w * retention)

where ``retention`` is the Ebbinghaus curve ``exp(-ln(2) * age / half_life)``
in [0, 1]. The max age penalty is ``w`` when ``retention -> 0``; a fresh
memory (retention -> 1) is unscored. We pin both boundary cases.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.chronicle.engine import _apply_temporal_decay


def _chunk(*, source_timestamp: datetime, content: str = "x") -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        created_at=datetime.now(UTC),
        source_timestamp=source_timestamp,
    )


def test_fully_faded_memory_keeps_one_minus_w_of_relevance() -> None:
    """retention -> 0 (very old chunk) means score = relevance * (1 - w)."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    # Age many half-lives, so retention is effectively zero.
    ancient = _chunk(source_timestamp=now - timedelta(days=3650))  # 10 years

    [(_, score)] = _apply_temporal_decay(
        [(ancient, 1.0)],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
    )
    # retention ~ 0, so final ~ relevance * (1 - 0.3) = 0.7
    assert score == pytest.approx(0.7, abs=1e-6)


def test_fresh_memory_keeps_full_relevance() -> None:
    """retention -> 1 (just-happened event) means score = relevance * 1.0."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    fresh = _chunk(source_timestamp=now)  # age = 0

    [(_, score)] = _apply_temporal_decay(
        [(fresh, 1.0)],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
    )
    assert score == pytest.approx(1.0, abs=1e-6)


def test_one_half_life_retention_is_half() -> None:
    """At exactly one half-life of age, retention = 0.5; score = relevance * (1 - w + 0.5w)."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    one_half_life = _chunk(source_timestamp=now - timedelta(hours=168))

    [(_, score)] = _apply_temporal_decay(
        [(one_half_life, 1.0)],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
    )
    # final = 1.0 * (0.7 + 0.3 * 0.5) = 0.85
    expected = (1 - 0.3) + 0.3 * 0.5
    assert score == pytest.approx(expected, abs=1e-6)


def test_decay_weight_zero_is_noop() -> None:
    """decay_weight=0 must short-circuit and return the input unchanged."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    chunks = [
        (_chunk(source_timestamp=now - timedelta(days=30)), 0.42),
        (_chunk(source_timestamp=now), 0.99),
    ]

    out = _apply_temporal_decay(
        chunks,
        decay_weight=0.0,
        half_life_hours=168.0,
        reference_time=now,
    )
    assert out is chunks  # short-circuit returns same object
