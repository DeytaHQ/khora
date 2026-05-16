"""Unit tests for the source_chunk_ids GC dream op (#662, Phase 2.3).

The SQL paths (Postgres ``unnest WITH ORDINALITY`` and SQLite JSON
partition) are exercised by integration tests against the live fixture
stacks. These unit tests pin the planner logic by monkeypatching
``_collect_dead_refs`` so every test feeds curated per-entity rows into
the public coroutine.

The GC op is **dry-run only in v0.14** — apply mode raises
``NotImplementedError`` and is tracked in #649 phase 4 / #668.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.engines.vectorcypher import source_chunk_ids_gc
from khora.dream.engines.vectorcypher.source_chunk_ids_gc import (
    plan_vectorcypher_source_chunk_ids_gc,
)
from khora.dream.plan import DreamOp, OpKind

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeCoordinator:
    """Just enough surface for ``plan_vectorcypher_source_chunk_ids_gc``."""

    resolved_id: UUID = field(default_factory=uuid4)
    resolve_calls: list[UUID] = field(default_factory=list)
    transaction_calls: int = 0
    write_calls: list[str] = field(default_factory=list)

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        self.resolve_calls.append(namespace_id)
        return self.resolved_id

    # If the planner ever wrote it would have to go through one of these
    # surfaces. Asserting these stay untouched is the no-writes contract.
    async def create_entity(self, *args: Any, **kwargs: Any) -> None:
        self.write_calls.append("create_entity")
        raise AssertionError("GC dry-run must not call create_entity")

    async def update_entity(self, *args: Any, **kwargs: Any) -> None:
        self.write_calls.append("update_entity")
        raise AssertionError("GC dry-run must not call update_entity")

    async def replace_chunks(self, *args: Any, **kwargs: Any) -> None:
        self.write_calls.append("replace_chunks")
        raise AssertionError("GC dry-run must not call replace_chunks")


def _patch_rows(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    """Replace the SQL collector with one that returns the given rows."""

    async def _fake_collect(coordinator: Any, resolved_namespace_id: UUID) -> list[dict[str, Any]]:
        return rows

    monkeypatch.setattr(source_chunk_ids_gc, "_collect_dead_refs", _fake_collect)


def _row(*, live: int, dead: int) -> dict[str, Any]:
    """Build a fixture row with ``live`` survivor UUIDs + ``dead`` dead ones."""
    live_uuids = [uuid4() for _ in range(live)]
    dead_uuids = [uuid4() for _ in range(dead)]
    return {
        "entity_id": uuid4(),
        "before_length": live + dead,
        "dead_uuids": dead_uuids,
        "after_array": live_uuids,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plans_correct_after_array_for_mixed_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entity with 3 live + 2 dead UUIDs → after_array == survivors, after_length == 3."""
    row = _row(live=3, dead=2)
    _patch_rows(monkeypatch, [row])

    coord = _FakeCoordinator()
    ops = await plan_vectorcypher_source_chunk_ids_gc(uuid4(), coordinator=coord)

    assert len(ops) == 1
    op = ops[0]
    assert isinstance(op, DreamOp)
    assert op.op_type == OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC
    assert op.phase == "mutation"
    assert op.decision == "planned"

    inputs = op.inputs[0]
    outputs = op.outputs[0]
    assert inputs["entity_id"] == str(row["entity_id"])
    assert inputs["before_length"] == 5
    assert inputs["dead_uuids"] == [str(u) for u in row["dead_uuids"]]
    assert outputs["after_length"] == 3
    assert outputs["after_array"] == [str(u) for u in row["after_array"]]


@pytest.mark.asyncio
async def test_excludes_entities_with_zero_dead_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entities surfacing with no dead UUIDs must not produce an op.

    In practice the SQL ``HAVING bool_or(is_dead)`` filter already
    excludes these — this test pins the Python-side guard too so
    upstream changes can't silently emit no-op plans.
    """
    healthy_row = {
        "entity_id": uuid4(),
        "before_length": 4,
        "dead_uuids": [],
        "after_array": [uuid4() for _ in range(4)],
    }
    dirty_row = _row(live=1, dead=1)
    _patch_rows(monkeypatch, [healthy_row, dirty_row])

    coord = _FakeCoordinator()
    ops = await plan_vectorcypher_source_chunk_ids_gc(uuid4(), coordinator=coord)

    assert len(ops) == 1
    assert ops[0].inputs[0]["entity_id"] == str(dirty_row["entity_id"])


@pytest.mark.asyncio
async def test_min_dead_threshold_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entity with fewer than ``min_dead`` dead UUIDs is not emitted."""
    one_dead = _row(live=2, dead=1)
    two_dead = _row(live=2, dead=2)
    five_dead = _row(live=0, dead=5)
    _patch_rows(monkeypatch, [one_dead, two_dead, five_dead])

    coord = _FakeCoordinator()
    ops = await plan_vectorcypher_source_chunk_ids_gc(
        uuid4(),
        coordinator=coord,
        min_dead=2,
    )

    emitted = {op.inputs[0]["entity_id"] for op in ops}
    assert emitted == {str(two_dead["entity_id"]), str(five_dead["entity_id"])}


@pytest.mark.asyncio
async def test_apply_mode_raises_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply mode is blocked in v0.14 — must raise NotImplementedError."""
    _patch_rows(monkeypatch, [_row(live=1, dead=1)])
    coord = _FakeCoordinator()

    with pytest.raises(NotImplementedError, match="apply mode lands in v0.15"):
        await plan_vectorcypher_source_chunk_ids_gc(
            uuid4(),
            coordinator=coord,
            mode="apply",
        )


@pytest.mark.asyncio
async def test_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run planner must not touch any coordinator write surface.

    The fake coordinator raises on every write method; a clean run with
    ``write_calls == []`` is the row-count assertion.
    """
    _patch_rows(monkeypatch, [_row(live=2, dead=3), _row(live=0, dead=1)])

    coord = _FakeCoordinator()
    ops = await plan_vectorcypher_source_chunk_ids_gc(uuid4(), coordinator=coord)

    assert len(ops) == 2
    assert coord.write_calls == []
    for op in ops:
        assert op.decision == "planned"


@pytest.mark.asyncio
async def test_dream_op_round_trips_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-op inputs + outputs must JSON-serialize cleanly for the sinks."""
    row = _row(live=2, dead=3)
    _patch_rows(monkeypatch, [row])

    coord = _FakeCoordinator()
    ops = await plan_vectorcypher_source_chunk_ids_gc(uuid4(), coordinator=coord)
    op = ops[0]

    blob = json.dumps({"inputs": op.inputs[0], "outputs": op.outputs[0]})
    round_tripped = json.loads(blob)

    assert round_tripped["inputs"]["before_length"] == 5
    assert round_tripped["outputs"]["after_length"] == 2
    # entity_id + uuid lists must already be strings.
    UUID(round_tripped["inputs"]["entity_id"])  # parseable
    for raw in round_tripped["inputs"]["dead_uuids"]:
        UUID(raw)
    for raw in round_tripped["outputs"]["after_array"]:
        UUID(raw)


def test_op_kind_is_registered() -> None:
    """The new OpKind value must exist and use the spec'd string."""
    assert OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC.value == "vectorcypher_source_chunk_ids_gc"
