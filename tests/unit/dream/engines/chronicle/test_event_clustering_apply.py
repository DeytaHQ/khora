"""Unit tests for ``apply_event_clustering`` (#669, Phase 4 apply).

Apply mode for ``chronicle_event_clustering`` is **blocked** in v0.15
because the schema dependency it needs has not yet been migrated:
``chronicle_events`` does not carry the bi-temporal soft-delete columns
(``invalidated_at`` / ``invalidated_by`` / ``merged_into_event_id``)
that the apply path would write to. Migration 033 added those columns
to ``relationships`` and ``memory_facts`` only — landing event-cluster
apply requires a follow-up migration 034 (out of scope for #669).

These tests pin the placeholder behaviour and document the contract
the eventual handler must satisfy:

* the stub raises ``NotImplementedError`` with a message that mentions
  the missing columns (so failures in production surface a clear next
  step, not a vague "not implemented" trace);
* the planner does **not** propose mutating
  ``chronicle_events.chunk_id`` — the architectural promise documented
  in the module docstring becomes a behavioural assertion here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.dream.engines.chronicle import apply_event_clustering
from khora.dream.engines.chronicle.event_clustering import _build_op, _EventRow

pytestmark = pytest.mark.unit


async def test_apply_event_clustering_raises_schema_dependency() -> None:
    """Calling the stub must surface the missing-schema message.

    Production failures should point operators at the next step
    (migration 034) rather than at an opaque ``NotImplementedError``.
    """
    with pytest.raises(NotImplementedError) as exc_info:
        await apply_event_clustering()

    msg = str(exc_info.value)
    assert "invalidated_at" in msg
    assert "merged_into_event_id" in msg
    assert "migration 034" in msg or "follow-up" in msg


def test_planner_inputs_never_target_chunk_id_column() -> None:
    """Architectural invariant: the planner never proposes mutating ``chunk_id``.

    The temporal recall channel dedupes events by ``chunk_id`` and the
    back-pointer is load-bearing — touching it from the dream phase
    would break recall. This test reads the planner's source-level
    contract: ``inputs`` and ``outputs`` keys must never include
    ``chunk_id`` as a mutation target.

    (When migration 034 lands and the apply handler is written, this
    test acquires a second guard at the SQL level: the handler's
    ``UPDATE`` column list must not include ``chunk_id``. That guard
    lands with the handler.)
    """
    ns = uuid4()
    chunk_a = uuid4()
    chunk_b = uuid4()
    now = datetime.now(UTC)
    rows = [
        _EventRow(
            event_id=uuid4(),
            chunk_id=chunk_a,
            subject="Alice",
            referenced_date=now,
            observation_date=now,
            confidence=0.9,
            embedding=[1.0, 0.0],
        ),
        _EventRow(
            event_id=uuid4(),
            chunk_id=chunk_b,
            subject="Alice",
            referenced_date=now,
            observation_date=now,
            confidence=0.7,
            embedding=[1.0, 0.0],
        ),
    ]

    op = _build_op(rows, namespace_id=ns)

    for entry in op.inputs:
        assert "chunk_id" not in entry, f"planner inputs proposed mutating chunk_id: {entry!r}"
    for entry in op.outputs:
        # The output carries an aggregate ``merged_source_chunk_ids_count``
        # (just a count) — never a per-row chunk_id target.
        assert "chunk_id" not in entry, f"planner outputs proposed mutating chunk_id: {entry!r}"
