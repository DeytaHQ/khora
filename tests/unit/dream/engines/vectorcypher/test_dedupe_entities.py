"""Tests for the vectorcypher cross-batch dedupe planner (#658, Phase 2.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import numpy as np
import pytest

from khora.core.models.entity import Entity
from khora.dream.engines.vectorcypher import plan_vectorcypher_dedupe_entities
from khora.dream.plan import DreamOp, OpKind


@dataclass
class _FakeCoordinator:
    """Minimal stand-in for :class:`StorageCoordinator` — read-only."""

    entities: list[Entity] = field(default_factory=list)
    list_calls: int = 0
    mutations: list[str] = field(default_factory=list)

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        del namespace_id, limit, offset
        self.list_calls += 1
        rows = list(self.entities)
        if entity_type is not None:
            rows = [e for e in rows if e.entity_type == entity_type]
        return rows

    # Mutation-shaped hooks — every test asserts these stay empty.
    async def upsert_entities_batch(self, *args: object, **kwargs: object) -> None:
        self.mutations.append("upsert_entities_batch")

    async def delete_entities(self, *args: object, **kwargs: object) -> None:
        self.mutations.append("delete_entities")


def _unit(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return vec
    return (arr / norm).astype(np.float32).tolist()


def _make_entity(
    ns: UUID,
    name: str,
    *,
    entity_type: str = "PERSON",
    embedding: list[float] | None = None,
    mention_count: int = 1,
    created_at: datetime | None = None,
    source_document_ids: list[UUID] | None = None,
    source_chunk_ids: list[UUID] | None = None,
) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns,
        name=name,
        entity_type=entity_type,
        embedding=_unit(embedding) if embedding is not None else None,
        mention_count=mention_count,
        created_at=created_at or datetime.now(UTC),
        source_document_ids=source_document_ids or [],
        source_chunk_ids=source_chunk_ids or [],
    )


# ---------------------------------------------------------------------------
# Candidate-pair generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emits_planned_op_for_high_similarity_pair() -> None:
    """Two near-identical PERSON embeddings with overlapping tokens → planned merge."""
    ns = uuid4()
    older = datetime(2026, 1, 1, tzinfo=UTC)
    younger = datetime(2026, 5, 1, tzinfo=UTC)
    a = _make_entity(
        ns,
        "Alice Smith",
        embedding=[1.0, 0.0, 0.0, 0.0],
        mention_count=5,
        created_at=older,
        source_document_ids=[uuid4()],
        source_chunk_ids=[uuid4(), uuid4()],
    )
    b = _make_entity(
        ns,
        "Alice S.",
        embedding=[1.0, 0.001, 0.0, 0.0],
        mention_count=2,
        created_at=younger,
        source_document_ids=[uuid4()],
        source_chunk_ids=[uuid4()],
    )
    coord = _FakeCoordinator(entities=[a, b])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)

    assert len(ops) == 1
    op = ops[0]
    assert isinstance(op, DreamOp)
    assert op.op_type == OpKind.VECTORCYPHER_DEDUPE_ENTITIES
    assert op.decision == "planned"
    payload = op.inputs[0]
    assert payload["op_type"] == "entity_merge"
    assert payload["entity_type"] == "PERSON"
    # Higher mention_count → canonical keeper.
    assert payload["keep_id"] == str(a.id)
    assert payload["drop_ids"] == (str(b.id),)
    assert payload["similarity_score"] >= 0.90
    # #1265: the apply contract lives in outputs[0]["merges"].
    merges = op.outputs[0]["merges"]
    assert len(merges) == 1
    merge = merges[0]
    assert merge["canonical_id"] == str(a.id)
    assert merge["absorbed_id"] == str(b.id)
    # Merged provenance is the union of both entities' source ids.
    assert len(merge["merged_source_document_ids"]) == 2
    assert len(merge["merged_source_chunk_ids"]) == 3
    assert coord.mutations == []


@pytest.mark.asyncio
async def test_skips_below_threshold_pairs() -> None:
    """Pairs whose cosine sits under the threshold do not become ops."""
    ns = uuid4()
    a = _make_entity(ns, "Alice Smith", embedding=[1.0, 0.0, 0.0, 0.0])
    # Orthogonal vector → cosine = 0; even with shared tokens, never crosses.
    b = _make_entity(ns, "Alice Jones", embedding=[0.0, 1.0, 0.0, 0.0])
    coord = _FakeCoordinator(entities=[a, b])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    assert ops == []


# ---------------------------------------------------------------------------
# Per-type threshold overrides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_type_threshold_overrides_default() -> None:
    """A tighter per-type threshold suppresses a pair the default would emit."""
    ns = uuid4()
    # cosine ≈ 0.9285 — above 0.90, below 0.95.
    a = _make_entity(ns, "Bob Jones", embedding=[1.0, 0.4, 0.0, 0.0])
    b = _make_entity(ns, "Bob J.", embedding=[1.0, 0.0, 0.0, 0.0])
    coord = _FakeCoordinator(entities=[a, b])

    # Default 0.90 emits.
    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    assert len(ops) == 1

    # Per-type 0.95 suppresses.
    ops = await plan_vectorcypher_dedupe_entities(
        ns,
        coordinator=coord,
        default_threshold=0.90,
        per_type_thresholds={"PERSON": 0.95},
    )
    assert ops == []


# ---------------------------------------------------------------------------
# UNIQUE-violation prediction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unique_violation_skip_collision() -> None:
    """A predicted survivor whose name collides with a third entity → skip."""
    ns = uuid4()
    # Two near-clones: ``a`` is the predicted keeper (higher mention_count).
    a = _make_entity(ns, "Charlie Brown", embedding=[1.0, 0.0, 0.0, 0.0], mention_count=10)
    b = _make_entity(ns, "Charlie B.", embedding=[1.0, 0.001, 0.0, 0.0], mention_count=1)
    # Third entity in the namespace has the same (name, type) as the
    # predicted survivor — UNIQUE collision.
    collision = _make_entity(ns, "Charlie Brown", embedding=[0.0, 1.0, 0.0, 0.0])
    coord = _FakeCoordinator(entities=[a, b, collision])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)

    # Both the planned-pair op and the collision detection emit *one* op
    # — the same pair (a, b) detected as a UNIQUE collision.
    skip_ops = [op for op in ops if op.decision == "skip_unique_collision"]
    planned_ops = [op for op in ops if op.decision == "planned"]
    assert len(skip_ops) == 1
    assert planned_ops == []
    skip = skip_ops[0]
    payload = skip.inputs[0]
    assert payload["surviving_name"] == "Charlie Brown"
    assert payload["collision_entity_id"] == str(collision.id)
    assert payload["collision_name"] == "Charlie Brown"
    assert payload["keep_id"] == str(a.id)
    assert payload["drop_ids"] == (str(b.id),)


# ---------------------------------------------------------------------------
# Apply mode + read-only guarantees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_mode_raises_not_implemented() -> None:
    """v0.14 must reject apply mode — destructive path lands in v0.15."""
    ns = uuid4()
    coord = _FakeCoordinator(entities=[])
    with pytest.raises(NotImplementedError, match=r"apply mode lands in v0\.15"):
        await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, mode="apply")


@pytest.mark.asyncio
async def test_no_writes_to_coordinator() -> None:
    """The planner must not invoke any mutation method on the coordinator."""
    ns = uuid4()
    entities = [_make_entity(ns, f"Person {i}", embedding=[1.0, float(i) / 100.0, 0.0, 0.0]) for i in range(6)]
    coord = _FakeCoordinator(entities=entities)
    initial_count = len(coord.entities)

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)

    assert coord.mutations == []
    assert len(coord.entities) == initial_count
    # Every op carries the planner phase (not "apply") and the dedupe op type.
    for op in ops:
        assert op.phase == "plan"
        assert op.op_type == OpKind.VECTORCYPHER_DEDUPE_ENTITIES


@pytest.mark.asyncio
async def test_empty_namespace_returns_empty_list() -> None:
    """No entities → no ops, no errors."""
    ns = uuid4()
    coord = _FakeCoordinator(entities=[])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord)
    assert ops == []


# ---------------------------------------------------------------------------
# JSON round-trip — sinks need this
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_op_payloads_are_json_round_trippable() -> None:
    """DreamOp inputs + outputs survive JSON encode/decode."""
    ns = uuid4()
    a = _make_entity(
        ns,
        "Diana Prince",
        embedding=[1.0, 0.0, 0.0, 0.0],
        mention_count=3,
        source_document_ids=[uuid4()],
        source_chunk_ids=[uuid4()],
    )
    b = _make_entity(
        ns,
        "Diana P.",
        embedding=[1.0, 0.0, 0.001, 0.0],
        mention_count=1,
        source_document_ids=[uuid4()],
        source_chunk_ids=[uuid4()],
    )
    coord = _FakeCoordinator(entities=[a, b])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    assert len(ops) == 1

    inputs_payload = json.dumps(list(ops[0].inputs))
    outputs_payload = json.dumps(list(ops[0].outputs))
    restored_inputs = json.loads(inputs_payload)
    restored_outputs = json.loads(outputs_payload)
    assert restored_inputs[0]["op_type"] == "entity_merge"
    assert "keep_id" in restored_inputs[0]
    assert "drop_ids" in restored_inputs[0]
    # #1265: the apply-readable merge payload survives JSON round-trip.
    merge = restored_outputs[0]["merges"][0]
    assert "canonical_id" in merge
    assert "absorbed_id" in merge
    assert "merged_source_document_ids" in merge
    assert "merged_source_chunk_ids" in merge


# ---------------------------------------------------------------------------
# Entities without embeddings are skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_entities_without_embeddings() -> None:
    """Entities with empty/None embedding can't be scored — silently skipped."""
    ns = uuid4()
    a = _make_entity(ns, "Eve A", embedding=[1.0, 0.0, 0.0, 0.0])
    b = _make_entity(ns, "Eve B", embedding=None)
    coord = _FakeCoordinator(entities=[a, b])

    # Only one embedded entity in the bucket → no candidate pairs possible.
    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    assert ops == []


# ---------------------------------------------------------------------------
# Survivor tiebreaker: mention_count then created_at
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_survivor_picked_by_created_at_when_mention_tied() -> None:
    """Equal mention_count → older created_at wins."""
    ns = uuid4()
    older = datetime(2026, 1, 1, tzinfo=UTC)
    younger = older + timedelta(days=30)
    a = _make_entity(
        ns,
        "Frank One",
        embedding=[1.0, 0.0, 0.0, 0.0],
        mention_count=4,
        created_at=younger,
    )
    b = _make_entity(
        ns,
        "Frank O.",
        embedding=[1.0, 0.0001, 0.0, 0.0],
        mention_count=4,
        created_at=older,
    )
    coord = _FakeCoordinator(entities=[a, b])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)

    assert len(ops) == 1
    payload = ops[0].inputs[0]
    # ``b`` is older → keeper.
    assert payload["keep_id"] == str(b.id)
    assert payload["drop_ids"] == (str(a.id),)
