"""Issue #1144: fact reconciliation must supersede by EVENT time, not ingestion order.

Before this fix, ``ChronicleEngine._reconcile_facts`` applied the LLM-chosen
UPDATE/DELETE by unconditionally superseding the target with the just-extracted
fact - "last ingested wins". Chronicle supports backfilled / out-of-order
ingest (``occurred_at``, #848), so re-importing an OLDER conversation after a
NEWER one would deactivate the newer active fact and tombstone it. The active
fact set then silently answers with stale state.

After the fix, supersession is ordered by each fact's effective event time
(``event_time`` anchored on the source chunk's ``occurred_at`` /
``source_timestamp``, falling back to ``created_at`` / ingestion time when no
event time is carried). When the LLM picks a target whose event time is
strictly NEWER than the new fact's, the engine inverts the direction: the new
(older) fact is written as a superseded historical record and the newer target
stays active. Ties / missing event-times fall back to ingestion order (new
supersedes old), preserving prior behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.chronicle.compression import (
    FactOperation,
    MemoryFact,
    ReconcileAction,
)
from khora.engines.chronicle.engine import ChronicleEngine


def _fact(
    *,
    obj: str,
    fact_id=None,
    event_time: datetime | None = None,
    created_at: datetime | None = None,
) -> MemoryFact:
    return MemoryFact(
        id=fact_id or uuid4(),
        subject="alice",
        predicate="works_at",
        object_=obj,
        fact_text=f"alice works_at {obj}",
        event_time=event_time,
        created_at=created_at,
    )


def _wire(engine: ChronicleEngine, *, existing: list[MemoryFact], action: ReconcileAction):
    storage = MagicMock()
    storage.query_active_facts_for_subject = AsyncMock(return_value=list(existing))
    storage.write_facts = AsyncMock(side_effect=lambda facts, **_: [f.id for f in facts])
    storage.supersede_fact = AsyncMock()

    compressor = AsyncMock()
    compressor.reconcile_fact = AsyncMock(return_value=action)

    engine._get_compressor = lambda *_, **__: compressor  # type: ignore[assignment]
    engine._get_storage = lambda: storage  # type: ignore[assignment]
    return storage


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backfilled_older_fact_does_not_supersede_newer_active_fact() -> None:
    """The regression: older real-world fact ingested later must NOT win.

    Active fact = 2026 ('Initech'), event_time newer. Backfilled fact = 2024
    ('Acme'), event_time older, extracted/ingested later. The LLM says UPDATE
    targeting the active 2026 fact. The engine must refuse to deactivate it and
    instead invert: the older fact is written superseded-by the active one.
    """
    ns_id = uuid4()
    active = _fact(obj="initech", event_time=datetime(2026, 1, 1, tzinfo=UTC))
    backfilled = _fact(obj="acme", event_time=datetime(2024, 1, 1, tzinfo=UTC))

    engine = ChronicleEngine.__new__(ChronicleEngine)
    storage = _wire(
        engine,
        existing=[active],
        action=ReconcileAction(op=FactOperation.UPDATE, target=active),
    )

    persisted, errors = await engine._reconcile_facts([backfilled], ns_id, expertise=None)

    assert errors == 0
    # The newer active fact must NOT be superseded by the older backfilled one.
    superseded_targets = [c.args[0] for c in storage.supersede_fact.await_args_list]
    assert active.id not in superseded_targets, "newer active fact wrongly superseded by older backfilled fact"

    # Inverted direction: the older fact is recorded superseded-by the newer
    # active fact (history preserved, active set stays correct).
    assert storage.supersede_fact.await_args_list, "expected the older fact to be recorded as superseded"
    old_id, new_id = storage.supersede_fact.await_args_list[0].args[:2]
    assert old_id == backfilled.id
    assert new_id == active.id
    assert backfilled.is_active is False
    assert backfilled.superseded_by == active.id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_newer_fact_ingested_later_still_supersedes_older_active_fact() -> None:
    """Normal case must NOT invert: newer event-time fact supersedes the older one."""
    ns_id = uuid4()
    active = _fact(obj="acme", event_time=datetime(2024, 1, 1, tzinfo=UTC))
    newer = _fact(obj="initech", event_time=datetime(2026, 1, 1, tzinfo=UTC))

    engine = ChronicleEngine.__new__(ChronicleEngine)
    storage = _wire(
        engine,
        existing=[active],
        action=ReconcileAction(op=FactOperation.UPDATE, target=active),
    )

    persisted, errors = await engine._reconcile_facts([newer], ns_id, expertise=None)

    assert errors == 0
    assert storage.supersede_fact.await_args_list, "expected the old fact to be superseded"
    old_id, new_id = storage.supersede_fact.await_args_list[0].args[:2]
    assert old_id == active.id
    assert new_id == newer.id
    assert active.id != newer.id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_event_times_fall_back_to_ingestion_order() -> None:
    """No event_time on either fact → keep prior 'new supersedes old' direction."""
    ns_id = uuid4()
    active = _fact(obj="acme")  # event_time=None, created_at=None
    new_fact = _fact(obj="globex")

    engine = ChronicleEngine.__new__(ChronicleEngine)
    storage = _wire(
        engine,
        existing=[active],
        action=ReconcileAction(op=FactOperation.UPDATE, target=active),
    )

    persisted, errors = await engine._reconcile_facts([new_fact], ns_id, expertise=None)

    assert errors == 0
    old_id, new_id = storage.supersede_fact.await_args_list[0].args[:2]
    assert old_id == active.id
    assert new_id == new_fact.id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tz_naive_event_times_are_normalized_before_comparison() -> None:
    """#1145 class: a tz-naive vs tz-aware comparison must not raise.

    Active fact carries a tz-naive newer event_time; the backfilled fact a
    tz-aware older one. The guard must normalize both to UTC, conclude the new
    fact is older, and refuse to supersede the active fact.
    """
    ns_id = uuid4()
    active = _fact(obj="initech", event_time=datetime(2026, 1, 1))  # tz-naive
    backfilled = _fact(obj="acme", event_time=datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=5))))

    engine = ChronicleEngine.__new__(ChronicleEngine)
    storage = _wire(
        engine,
        existing=[active],
        action=ReconcileAction(op=FactOperation.UPDATE, target=active),
    )

    persisted, errors = await engine._reconcile_facts([backfilled], ns_id, expertise=None)

    assert errors == 0
    superseded_targets = [c.args[0] for c in storage.supersede_fact.await_args_list]
    assert active.id not in superseded_targets


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_time_anchored_on_chunk_occurred_at() -> None:
    """``MemoryFact.event_time`` is stamped from the source chunk's occurred_at."""
    from khora.core.models import Chunk

    ns_id = uuid4()
    occurred = datetime(2025, 3, 4, tzinfo=UTC)
    chunk = Chunk(namespace_id=ns_id, content="alice works at acme", occurred_at=occurred)

    extractor = AsyncMock()
    extractor.extract_facts = AsyncMock(return_value=[_fact(obj="acme")])

    import asyncio

    engine = ChronicleEngine.__new__(ChronicleEngine)
    facts = await engine._extract_facts_for_chunk(chunk, ns_id, extractor, asyncio.Semaphore(1))

    assert facts[0].event_time == occurred
