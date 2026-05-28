"""Chronicle #848: ``_apply_temporal_decay`` must use event time, not ingest time.

When a user supplies an event timestamp via ``metadata['occurred_at']``, it
flows through the pipeline onto ``Chunk.source_timestamp``. The decay scorer
must read that field (falling back to ``created_at`` only when missing),
otherwise backfilled / batched ingest treats every memory as "fresh" because
``created_at`` is always ``now()`` at ingest time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from khora.core.models import Chunk
from khora.engines.chronicle.engine import (
    DEFAULT_CHRONICLE_DECAY_WEIGHT,
    DEFAULT_CHRONICLE_HALF_LIFE_HOURS,
    _apply_temporal_decay,
)


def _chunk(*, source_timestamp: datetime | None, created_at: datetime) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="x",
        created_at=created_at,
        source_timestamp=source_timestamp,
    )


def test_decay_prefers_source_timestamp_over_created_at() -> None:
    """Two chunks with identical created_at but different source_timestamp
    must receive different decay multipliers.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    # Both ingested "now" - same created_at.
    fresh_event = _chunk(
        source_timestamp=now - timedelta(hours=1),
        created_at=now,
    )
    stale_event = _chunk(
        source_timestamp=now - timedelta(days=30),
        created_at=now,
    )

    rescored = _apply_temporal_decay(
        [(fresh_event, 1.0), (stale_event, 1.0)],
        decay_weight=DEFAULT_CHRONICLE_DECAY_WEIGHT,
        half_life_hours=DEFAULT_CHRONICLE_HALF_LIFE_HOURS,
        reference_time=now,
    )

    # Sorted by score desc - fresh event must outrank the 30-day-old one.
    scores = {chunk.id: score for chunk, score in rescored}
    assert scores[fresh_event.id] > scores[stale_event.id], (
        f"fresh={scores[fresh_event.id]} stale={scores[stale_event.id]}"
    )
    # And the stale chunk's score should be meaningfully degraded (>5% drop).
    assert scores[stale_event.id] < 0.95 * scores[fresh_event.id]


def test_decay_falls_back_to_created_at_when_no_source_timestamp() -> None:
    """When source_timestamp is None, created_at is the age reference."""
    now = datetime(2026, 5, 28, tzinfo=UTC)
    fresh = _chunk(source_timestamp=None, created_at=now - timedelta(hours=1))
    stale = _chunk(source_timestamp=None, created_at=now - timedelta(days=30))

    rescored = _apply_temporal_decay(
        [(fresh, 1.0), (stale, 1.0)],
        decay_weight=DEFAULT_CHRONICLE_DECAY_WEIGHT,
        half_life_hours=DEFAULT_CHRONICLE_HALF_LIFE_HOURS,
        reference_time=now,
    )

    scores = {chunk.id: score for chunk, score in rescored}
    assert scores[fresh.id] > scores[stale.id]


def test_decay_ignores_stale_created_at_when_source_timestamp_is_fresh() -> None:
    """Backfill scenario: ingested today, but the event happened today.

    The 6-month-old event ingested today must outrank a 6-month-old created_at
    chunk that ALSO has a 6-month-old source_timestamp.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    # Backfilled today, but the event is fresh.
    backfilled_fresh = _chunk(
        source_timestamp=now - timedelta(hours=1),
        created_at=now - timedelta(days=180),
    )
    # Old ingest AND old event.
    truly_stale = _chunk(
        source_timestamp=now - timedelta(days=180),
        created_at=now - timedelta(days=180),
    )

    rescored = _apply_temporal_decay(
        [(backfilled_fresh, 1.0), (truly_stale, 1.0)],
        decay_weight=DEFAULT_CHRONICLE_DECAY_WEIGHT,
        half_life_hours=DEFAULT_CHRONICLE_HALF_LIFE_HOURS,
        reference_time=now,
    )

    scores = {chunk.id: score for chunk, score in rescored}
    assert scores[backfilled_fresh.id] > scores[truly_stale.id]
