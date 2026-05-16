"""Tests for the vectorcypher schema-drift report (#655, Phase 1.3)."""

from __future__ import annotations

import json
from dataclasses import asdict
from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import Entity, Relationship
from khora.dream import OpKind
from khora.dream.engines.vectorcypher import plan_vectorcypher_schema_drift
from khora.dream.engines.vectorcypher.schema_drift import (
    _frequency_delta,
)
from khora.extraction.skills.base import (
    EntityTypeConfig,
    ExpertiseConfig,
    RelationshipTypeConfig,
)

# ---------------------------------------------------------------------------
# Fake coordinator — paginates over an in-memory list, matching the
# StorageCoordinator.list_entities / list_relationships contract.
# ---------------------------------------------------------------------------


class _FakeCoordinator:
    """Minimal stand-in for ``StorageCoordinator`` for schema-drift tests.

    The op only calls ``list_entities`` / ``list_relationships`` and only
    reads ``entity_type`` / ``relationship_type`` off the rows. We honor
    the offset/limit pagination contract so the planner's loop sees the
    same shape it would in production.
    """

    def __init__(
        self,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
    ) -> None:
        self._entities = entities or []
        self._relationships = relationships or []
        self.entity_calls = 0
        self.relationship_calls = 0
        # Mutation tracking — every test asserts these stay at 0.
        self.mutations: list[str] = []

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        self.entity_calls += 1
        rows = [e for e in self._entities if e.namespace_id == namespace_id]
        if entity_type is not None:
            rows = [e for e in rows if e.entity_type == entity_type]
        return rows[offset : offset + limit]

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        self.relationship_calls += 1
        rows = [r for r in self._relationships if r.namespace_id == namespace_id]
        if relationship_type is not None:
            rows = [r for r in rows if r.relationship_type == relationship_type]
        return rows[offset : offset + limit]

    # Mutation-shaped hooks so a misbehaving op gets caught loudly.
    async def upsert_entities_batch(self, *args: object, **kwargs: object) -> None:
        self.mutations.append("upsert_entities_batch")
        raise AssertionError("schema-drift op must not mutate the store")

    async def delete_entity(self, *args: object, **kwargs: object) -> None:
        self.mutations.append("delete_entity")
        raise AssertionError("schema-drift op must not mutate the store")


def _entity(ns: UUID, name: str, entity_type: str) -> Entity:
    return Entity(namespace_id=ns, name=name, entity_type=entity_type)


def _relationship(ns: UUID, rel_type: str) -> Relationship:
    return Relationship(namespace_id=ns, relationship_type=rel_type)


def _expertise(
    entity_types: list[str] | None = None,
    relationship_types: list[str] | None = None,
) -> ExpertiseConfig:
    return ExpertiseConfig(
        name="test",
        entity_types=[EntityTypeConfig(name=n) for n in (entity_types or [])],
        relationship_types=[RelationshipTypeConfig(name=n) for n in (relationship_types or [])],
    )


# ---------------------------------------------------------------------------
# Bucket coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detects_new_entity_types_not_in_expertise() -> None:
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[
            _entity(ns, "alice", "PERSON"),
            _entity(ns, "bob", "EMPLOYEE"),  # undeclared
            _entity(ns, "carol", "EMPLOYEE"),  # undeclared
        ],
    )
    expertise = _expertise(entity_types=["PERSON"])

    op = await plan_vectorcypher_schema_drift(ns, coordinator=coord, expertise=expertise)

    outputs = op.outputs[0]
    assert outputs["new_entity_types"] == ["EMPLOYEE"]
    assert outputs["unused_entity_types"] == []
    assert outputs["entity_frequencies"]["EMPLOYEE"] == 2
    assert outputs["entity_frequencies"]["PERSON"] == 1
    assert outputs["entity_total"] == 3


@pytest.mark.asyncio
async def test_detects_unused_expertise_types() -> None:
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[_entity(ns, "alice", "PERSON")],
    )
    expertise = _expertise(entity_types=["PERSON", "RECIPE", "ORG"])

    op = await plan_vectorcypher_schema_drift(ns, coordinator=coord, expertise=expertise)

    outputs = op.outputs[0]
    assert outputs["new_entity_types"] == []
    assert outputs["unused_entity_types"] == ["ORG", "RECIPE"]


@pytest.mark.asyncio
async def test_frequency_delta_50pct() -> None:
    ns = uuid4()
    # Now: 200 PERSON entities (was 100 previously — 100% delta, trips).
    # Also: 110 ORG entities (was 100 — 10% delta, does NOT trip).
    coord = _FakeCoordinator(
        entities=[_entity(ns, f"p{i}", "PERSON") for i in range(200)]
        + [_entity(ns, f"o{i}", "ORG") for i in range(110)],
    )
    expertise = _expertise(entity_types=["PERSON", "ORG"])
    previous_run_id = uuid4()

    op = await plan_vectorcypher_schema_drift(
        ns,
        coordinator=coord,
        expertise=expertise,
        previous_run_id=previous_run_id,
        previous_entity_frequencies={"PERSON": 100, "ORG": 100},
    )

    outputs = op.outputs[0]
    assert outputs["entity_frequency_delta"] == {"PERSON": (100, 200)}
    # ORG below the 50% threshold — not in delta bucket.
    assert "ORG" not in outputs["entity_frequency_delta"]
    # previous_run_id is stamped into op.inputs for audit.
    assert op.inputs == (previous_run_id,)


@pytest.mark.asyncio
async def test_no_previous_run() -> None:
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[_entity(ns, "alice", "PERSON")],
    )
    expertise = _expertise(entity_types=["PERSON"])

    op = await plan_vectorcypher_schema_drift(
        ns,
        coordinator=coord,
        expertise=expertise,
        previous_run_id=None,
    )

    outputs = op.outputs[0]
    assert outputs["entity_frequency_delta"] == {}
    assert outputs["relationship_frequency_delta"] == {}
    # previous_run_id=None still stamped into inputs (audit trail).
    assert op.inputs == (None,)


@pytest.mark.asyncio
async def test_relationships_diff() -> None:
    ns = uuid4()
    coord = _FakeCoordinator(
        relationships=[
            _relationship(ns, "WORKS_AT"),
            _relationship(ns, "WORKS_AT"),
            _relationship(ns, "INVENTED_BY"),  # undeclared in expertise
        ],
    )
    expertise = _expertise(
        entity_types=["PERSON"],
        relationship_types=["WORKS_AT", "OWNS"],
    )

    op = await plan_vectorcypher_schema_drift(
        ns,
        coordinator=coord,
        expertise=expertise,
        previous_relationship_frequencies={"WORKS_AT": 1},
    )

    outputs = op.outputs[0]
    assert outputs["new_relationship_types"] == ["INVENTED_BY"]
    assert outputs["unused_relationship_types"] == ["OWNS"]
    assert outputs["relationship_frequencies"]["WORKS_AT"] == 2
    assert outputs["relationship_total"] == 3
    # WORKS_AT doubled (1 → 2) — trips the 50% gate. INVENTED_BY went
    # 0 → 1 since it wasn't in the previous-frequency map, so it also
    # trips (symmetric delta rule).
    assert outputs["relationship_frequency_delta"] == {
        "WORKS_AT": (1, 2),
        "INVENTED_BY": (0, 1),
    }


@pytest.mark.asyncio
async def test_empty_namespace() -> None:
    """Empty namespace returns empty multisets without crashing."""
    ns = uuid4()
    coord = _FakeCoordinator()
    expertise = _expertise(entity_types=["PERSON"], relationship_types=["WORKS_AT"])

    op = await plan_vectorcypher_schema_drift(ns, coordinator=coord, expertise=expertise)

    outputs = op.outputs[0]
    assert outputs["new_entity_types"] == []
    assert outputs["new_relationship_types"] == []
    assert outputs["unused_entity_types"] == ["PERSON"]
    assert outputs["unused_relationship_types"] == ["WORKS_AT"]
    assert outputs["entity_total"] == 0
    assert outputs["relationship_total"] == 0


@pytest.mark.asyncio
async def test_no_config_declared_types() -> None:
    """Bare ExpertiseConfig: every observed type is "new" drift."""
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[_entity(ns, "alice", "PERSON"), _entity(ns, "acme", "ORG")],
    )
    expertise = _expertise()

    op = await plan_vectorcypher_schema_drift(ns, coordinator=coord, expertise=expertise)

    outputs = op.outputs[0]
    assert outputs["new_entity_types"] == ["ORG", "PERSON"]
    assert outputs["unused_entity_types"] == []


@pytest.mark.asyncio
async def test_no_writes() -> None:
    """Op must never trigger any mutation path on the coordinator."""
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[_entity(ns, "alice", "PERSON")],
        relationships=[_relationship(ns, "WORKS_AT")],
    )
    expertise = _expertise(entity_types=["PERSON"], relationship_types=["WORKS_AT"])

    op = await plan_vectorcypher_schema_drift(ns, coordinator=coord, expertise=expertise)

    assert coord.mutations == []
    assert op.decision == "audit_complete"
    assert op.op_type is OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT


@pytest.mark.asyncio
async def test_dream_op_round_trips_json() -> None:
    """DreamOp's output payload must serialize cleanly to JSON for the file sink."""
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[_entity(ns, "alice", "PERSON")],
        relationships=[_relationship(ns, "WORKS_AT")],
    )
    expertise = _expertise(entity_types=["PERSON"], relationship_types=["WORKS_AT"])

    op = await plan_vectorcypher_schema_drift(ns, coordinator=coord, expertise=expertise)

    # Hand-pick the JSON-serializable slice (UUIDs and enums get coerced).
    payload = {
        "op_id": str(op.op_id),
        "phase": op.phase,
        "op_type": str(op.op_type.value),
        "decision": op.decision,
        "namespace_id": str(op.namespace_id),
        "outputs": op.outputs,
    }
    blob = json.dumps(payload, default=str)
    restored = json.loads(blob)
    assert restored["op_type"] == "vectorcypher_schema_drift_report"
    assert restored["decision"] == "audit_complete"
    # outputs is a tuple of one dict — survives the tuple→list coercion.
    assert isinstance(restored["outputs"], list)
    assert restored["outputs"][0]["new_entity_types"] == []
    assert restored["outputs"][0]["entity_total"] == 1


# ---------------------------------------------------------------------------
# Helper-level coverage
# ---------------------------------------------------------------------------


def test_frequency_delta_none_baseline() -> None:
    from collections import Counter

    assert _frequency_delta(None, Counter({"PERSON": 100})) == {}


def test_frequency_delta_handles_new_and_deleted_types() -> None:
    from collections import Counter

    # Fresh type (prev=0, curr=10) trips. Vanished type (prev=10, curr=0) also trips.
    delta = _frequency_delta({"GONE": 10}, Counter({"FRESH": 10}))
    assert delta == {"GONE": (10, 0), "FRESH": (0, 10)}


def test_opkind_enum_member_value() -> None:
    """The OpKind value string is part of the stable wire format."""
    assert OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT.value == "vectorcypher_schema_drift_report"


def test_dream_op_asdict_matches_internal_shape() -> None:
    """DreamOp is a stdlib dataclass — asdict() should round-trip without surprises."""
    op = asdict(
        plan_vectorcypher_schema_drift_op_stub(),
    )
    assert op["op_type"] == OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT
    assert op["decision"] == "audit_complete"
    # outputs survives dataclass introspection — it's the single-dict tuple
    # the file sink will serialize.
    assert len(op["outputs"]) == 1
    assert op["outputs"][0]["new_entity_types"] == []


def plan_vectorcypher_schema_drift_op_stub():
    """Hand-built DreamOp for the dataclass shape test — no async coord needed."""
    from khora.dream.plan import DreamOp

    return DreamOp(
        op_id=uuid4(),
        phase="audit",
        op_type=OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT,
        inputs=(None,),
        outputs=({"new_entity_types": []},),
        decision="audit_complete",
        rationale="stub",
        namespace_id=uuid4(),
    )
