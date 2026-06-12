"""Chronicle #855: with reinforcement on, decay uses max(source_timestamp, last_accessed_at).

Without reinforcement, a year-old chunk is fully faded regardless of how
recently it was recalled. With reinforcement, a year-old chunk that was
recalled five minutes ago should score as fresh.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.chronicle.engine import _apply_temporal_decay, _compute_recency_multipliers


def _chunk(*, source_timestamp: datetime, last_accessed_at: datetime | None = None) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="x",
        created_at=datetime.now(UTC),
        source_timestamp=source_timestamp,
        last_accessed_at=last_accessed_at,
    )


def test_reinforcement_off_uses_only_source_timestamp() -> None:
    """With reinforcement disabled, last_accessed_at is ignored."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    year_old_recently_recalled = _chunk(
        source_timestamp=now - timedelta(days=365),
        last_accessed_at=now,
    )

    [(_, score)] = _apply_temporal_decay(
        [(year_old_recently_recalled, 1.0)],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
        enable_reinforcement=False,
    )
    # 365 days = ~52 half-lives; retention essentially 0 -> score ~ 0.7.
    assert score == pytest.approx(0.7, abs=1e-3)


def test_reinforcement_on_uses_max_of_source_and_last_accessed() -> None:
    """With reinforcement enabled, a recently-recalled old chunk is fresh."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    year_old_recently_recalled = _chunk(
        source_timestamp=now - timedelta(days=365),
        last_accessed_at=now,
    )

    [(_, score)] = _apply_temporal_decay(
        [(year_old_recently_recalled, 1.0)],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
        enable_reinforcement=True,
    )
    # effective age = max(year-old, now) = now -> retention 1.0 -> score 1.0.
    assert score == pytest.approx(1.0, abs=1e-6)


def test_reinforcement_on_falls_back_to_created_at_when_both_null() -> None:
    """When both source_timestamp and last_accessed_at are NULL, created_at wins."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    chunk = Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="x",
        created_at=now,
        source_timestamp=None,
        last_accessed_at=None,
    )

    [(_, score)] = _apply_temporal_decay(
        [(chunk, 1.0)],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
        enable_reinforcement=True,
    )
    assert score == pytest.approx(1.0, abs=1e-6)


def test_reinforcement_on_picks_newest_when_both_set() -> None:
    """When both fields are set, the more recent one drives decay."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    # source is older than last_accessed - last_accessed wins.
    chunk_newer_access = _chunk(
        source_timestamp=now - timedelta(days=30),
        last_accessed_at=now - timedelta(hours=1),
    )
    # source is newer than last_accessed - source wins (atypical but valid).
    chunk_newer_source = _chunk(
        source_timestamp=now - timedelta(hours=1),
        last_accessed_at=now - timedelta(days=30),
    )

    [(_, s_access), (_, s_source)] = _apply_temporal_decay(
        [(chunk_newer_access, 1.0), (chunk_newer_source, 1.0)],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
        enable_reinforcement=True,
    )
    # Both chunks are effectively ~1 hour old: scores should match.
    assert s_access == pytest.approx(s_source, abs=1e-6)
    # And both should be close to 1.0 (almost no decay at 1 hour into a 168h half-life).
    assert s_access > 0.99


def test_reinforcement_on_handles_naive_source_with_aware_last_accessed() -> None:
    """#1145: a tz-naive source_timestamp must not crash max() against a tz-aware last_accessed_at.

    coerce_source_timestamp('2026-01-15T10:30:00') returns a naive datetime and
    the sqlite_lance backend round-trips it verbatim, while last_accessed_at is
    always stamped tz-aware by _reinforce_last_accessed. Before the fix this
    raised 'TypeError: can't compare offset-naive and offset-aware datetimes'.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    chunk = _chunk(
        source_timestamp=datetime(2026, 1, 15, 10, 30),  # naive, as coerced from ISO string
        last_accessed_at=now - timedelta(hours=1),  # aware, as stamped by reinforcement
    )

    [(_, score)] = _apply_temporal_decay(
        [(chunk, 1.0)],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
        enable_reinforcement=True,
    )
    # last_accessed_at (1h ago, UTC) is more recent than the naive-as-UTC
    # source_timestamp -> nearly no decay.
    assert score > 0.99


def test_recency_multipliers_handle_naive_source_with_aware_last_accessed() -> None:
    """#1145: same naive-vs-aware crash in _compute_recency_multipliers."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    chunk = _chunk(
        source_timestamp=datetime(2026, 1, 15, 10, 30),  # naive
        last_accessed_at=now - timedelta(hours=1),  # aware
    )

    mults = _compute_recency_multipliers(
        [chunk],
        decay_weight=0.3,
        half_life_hours=168.0,
        reference_time=now,
        enable_reinforcement=True,
    )
    assert mults[chunk.id] == pytest.approx(1.0, abs=0.01)
