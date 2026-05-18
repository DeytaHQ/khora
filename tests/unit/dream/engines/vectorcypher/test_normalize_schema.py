"""Tests for the vectorcypher schema-drift normalization op (#673, Phase 5.4).

Planner emits zero ops on empty mapping. Apply rewrites entity_type /
relationship_type columns and emits one MemoryEvent per renamed row.
Undo round-trips by swapping the mapping. Idempotent on replay.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import Entity, Relationship
from khora.core.models.event import EventType, MemoryEvent
from khora.dream.config import DreamConfig
from khora.dream.engines.vectorcypher.normalize_schema import (
    apply_vectorcypher_normalize_schema,
    plan_vectorcypher_normalize_schema,
)
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord

# ---------------------------------------------------------------------------
# Fake coordinator (planner side)
# ---------------------------------------------------------------------------


class _FakeCoordinator:
    """Stand-in matching the slice of ``StorageCoordinator`` the planner uses.

    Tracks dispatched MemoryEvents in ``dispatched_events`` so apply-side
    tests can assert the per-row emission contract.
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
        self.mutations: list[str] = []
        self.dispatched_events: list[MemoryEvent] = []

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

    async def dispatch_hook(self, event: MemoryEvent) -> None:
        self.dispatched_events.append(event)

    # Mutation surfaces — assert these stay empty in planner tests.
    async def update_entity(self, *args: Any, **kwargs: Any) -> None:
        self.mutations.append("update_entity")

    async def upsert_entities_batch(self, *args: Any, **kwargs: Any) -> None:
        self.mutations.append("upsert_entities_batch")


# ---------------------------------------------------------------------------
# Fake SQL session for apply tests
# ---------------------------------------------------------------------------


class _FakeSession:
    """Captures executed SQL and serves curated SELECT rows.

    The apply handler issues:

      * ``SELECT id, entity_type FROM entities WHERE namespace_id = :ns
          AND entity_type IN (...) AND valid_until IS NULL``
      * ``SELECT id, relationship_type FROM relationships WHERE
          namespace_id = :ns AND relationship_type IN (...) AND
          invalidated_at IS NULL``
      * ``UPDATE entities SET entity_type = :new ... WHERE id = :id``
      * ``UPDATE relationships SET relationship_type = :new ... WHERE id = :id``
    """

    def __init__(
        self,
        *,
        entity_rows: list[tuple[UUID, str]] | None = None,
        relationship_rows: list[tuple[UUID, str]] | None = None,
        dialect_name: str = "postgresql",
    ) -> None:
        self.entity_rows = entity_rows or []
        self.relationship_rows = relationship_rows or []
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
        self.executed: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        text_str = str(stmt)
        params = params or {}
        self.executed.append((text_str, params))
        upper = text_str.lstrip().upper()
        # SELECTs are id-filtered: return only rows whose id matches params["id"].
        if upper.startswith("SELECT") and "FROM ENTITIES" in upper:
            target = _coerce_uuid(params.get("id"))
            rows = [
                SimpleNamespace(id=eid, entity_type=etype)
                for eid, etype in self.entity_rows
                if target is None or eid == target
            ]
            return _Result(rows)
        if upper.startswith("SELECT") and "FROM RELATIONSHIPS" in upper:
            target = _coerce_uuid(params.get("id"))
            rows = [
                SimpleNamespace(id=rid, relationship_type=rtype)
                for rid, rtype in self.relationship_rows
                if target is None or rid == target
            ]
            return _Result(rows)
        return SimpleNamespace(rowcount=1)


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)

    def fetchall(self):
        return list(self._rows)


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_mapping_emits_single_insufficient_input_op() -> None:
    """An empty operator mapping is treated as a no-op, never silent rewrites."""
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[Entity(name="Alice", entity_type="PERSON_NAME", namespace_id=ns)],
    )
    config = DreamConfig(normalize_schema_enabled=True, normalize_schema_mapping={})

    ops = await plan_vectorcypher_normalize_schema(
        ns,
        coordinator=coord,  # type: ignore[arg-type]
        config=config,
    )

    assert len(ops) == 1
    op = ops[0]
    assert op.op_type is OpKind.VECTORCYPHER_NORMALIZE_SCHEMA
    assert op.decision == "insufficient_input"
    assert "mapping" in op.rationale.lower()
    # No coordinator mutations.
    assert coord.mutations == []


@pytest.mark.asyncio
async def test_disabled_config_returns_empty_plan() -> None:
    """When normalize_schema_enabled=False the op produces no DreamOps."""
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[Entity(name="Alice", entity_type="PERSON_NAME", namespace_id=ns)],
    )
    config = DreamConfig(
        normalize_schema_enabled=False,
        normalize_schema_mapping={"PERSON_NAME": "PERSON"},
    )

    ops = await plan_vectorcypher_normalize_schema(
        ns,
        coordinator=coord,  # type: ignore[arg-type]
        config=config,
    )

    assert ops == []


@pytest.mark.asyncio
async def test_entity_type_rename_emits_planned_op() -> None:
    """A mapping with one entity-type rename produces one planned DreamOp."""
    ns = uuid4()
    alice = Entity(name="Alice", entity_type="PERSON_NAME", namespace_id=ns)
    bob = Entity(name="Bob", entity_type="PERSON_NAME", namespace_id=ns)
    unrelated = Entity(name="Acme", entity_type="ORG", namespace_id=ns)
    coord = _FakeCoordinator(entities=[alice, bob, unrelated])
    config = DreamConfig(
        normalize_schema_enabled=True,
        normalize_schema_mapping={"PERSON_NAME": "PERSON"},
    )

    ops = await plan_vectorcypher_normalize_schema(
        ns,
        coordinator=coord,  # type: ignore[arg-type]
        config=config,
    )

    assert len(ops) == 1
    op = ops[0]
    assert op.op_type is OpKind.VECTORCYPHER_NORMALIZE_SCHEMA
    assert op.decision == "planned"
    payload = op.outputs[0]
    assert payload["entity_renames"]
    target_ids = {UUID(r["id"]) for r in payload["entity_renames"]}
    assert target_ids == {alice.id, bob.id}
    for r in payload["entity_renames"]:
        assert r["old_type"] == "PERSON_NAME"
        assert r["new_type"] == "PERSON"
    # Unrelated entity not touched.
    assert all(UUID(r["id"]) != unrelated.id for r in payload["entity_renames"])
    # Inputs carry the operator-supplied mapping for audit.
    assert op.inputs and op.inputs[0]["mapping"] == {"PERSON_NAME": "PERSON"}


@pytest.mark.asyncio
async def test_relationship_type_rename_emits_planned_op() -> None:
    """A mapping with a relationship-type rename produces a planned op."""
    ns = uuid4()
    src = uuid4()
    tgt = uuid4()
    rel1 = Relationship(
        namespace_id=ns,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type="WORKS_FOR",
    )
    rel2 = Relationship(
        namespace_id=ns,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type="WORKS_FOR",
    )
    other = Relationship(
        namespace_id=ns,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type="KNOWS",
    )
    coord = _FakeCoordinator(relationships=[rel1, rel2, other])
    config = DreamConfig(
        normalize_schema_enabled=True,
        normalize_schema_mapping={"WORKS_FOR": "EMPLOYED_BY"},
    )

    ops = await plan_vectorcypher_normalize_schema(
        ns,
        coordinator=coord,  # type: ignore[arg-type]
        config=config,
    )

    assert len(ops) == 1
    op = ops[0]
    payload = op.outputs[0]
    rids = {UUID(r["id"]) for r in payload["relationship_renames"]}
    assert rids == {rel1.id, rel2.id}
    for r in payload["relationship_renames"]:
        assert r["old_type"] == "WORKS_FOR"
        assert r["new_type"] == "EMPLOYED_BY"


@pytest.mark.asyncio
async def test_planner_never_mutates_coordinator() -> None:
    """Planner is pure SELECT — no writes."""
    ns = uuid4()
    coord = _FakeCoordinator(
        entities=[Entity(name="Alice", entity_type="PERSON_NAME", namespace_id=ns)],
        relationships=[
            Relationship(namespace_id=ns, relationship_type="WORKS_FOR"),
        ],
    )
    config = DreamConfig(
        normalize_schema_enabled=True,
        normalize_schema_mapping={
            "PERSON_NAME": "PERSON",
            "WORKS_FOR": "EMPLOYED_BY",
        },
    )

    await plan_vectorcypher_normalize_schema(
        ns,
        coordinator=coord,  # type: ignore[arg-type]
        config=config,
    )

    assert coord.mutations == []


# ---------------------------------------------------------------------------
# Apply tests
# ---------------------------------------------------------------------------


def _build_apply_op(
    *,
    namespace_id: UUID,
    entity_renames: list[dict[str, Any]] | None = None,
    relationship_renames: list[dict[str, Any]] | None = None,
    mapping: dict[str, str] | None = None,
) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_NORMALIZE_SCHEMA,
        inputs=({"mapping": mapping or {"PERSON_NAME": "PERSON"}},),
        outputs=(
            {
                "entity_renames": entity_renames or [],
                "relationship_renames": relationship_renames or [],
            },
        ),
        decision="planned",
        rationale="op",
        started_at=datetime.now(UTC),
        namespace_id=namespace_id,
    )


@pytest.mark.asyncio
async def test_apply_entity_rename_updates_row_and_emits_event() -> None:
    ns = uuid4()
    eid = uuid4()
    coord = _FakeCoordinator()
    session = _FakeSession(entity_rows=[(eid, "PERSON_NAME")])
    op = _build_apply_op(
        namespace_id=ns,
        entity_renames=[{"id": str(eid), "old_type": "PERSON_NAME", "new_type": "PERSON"}],
        mapping={"PERSON_NAME": "PERSON"},
    )

    undo = await apply_vectorcypher_normalize_schema(
        op,
        coordinator=coord,  # type: ignore[arg-type]
        session=session,
    )

    # SQL: one UPDATE entities.
    sql_blob = " | ".join(s.upper() for s, _ in session.executed)
    assert "UPDATE ENTITIES" in sql_blob
    assert "ENTITY_TYPE" in sql_blob

    # One MemoryEvent dispatched per renamed entity.
    assert len(coord.dispatched_events) == 1
    ev = coord.dispatched_events[0]
    assert ev.event_type is EventType.ENTITY_UPDATED
    assert ev.resource_id == eid
    assert ev.namespace_id == ns
    assert ev.data["old_type"] == "PERSON_NAME"
    assert ev.data["new_type"] == "PERSON"

    # Undo carries the inverse-mapping snapshot.
    assert isinstance(undo, UndoRecord)
    assert undo.before["entity_renames"][0]["id"] == str(eid)
    assert undo.before["entity_renames"][0]["old_type"] == "PERSON_NAME"
    assert undo.before["entity_renames"][0]["new_type"] == "PERSON"


@pytest.mark.asyncio
async def test_apply_relationship_rename_updates_row_and_emits_event() -> None:
    ns = uuid4()
    rid = uuid4()
    coord = _FakeCoordinator()
    session = _FakeSession(relationship_rows=[(rid, "WORKS_FOR")])
    op = _build_apply_op(
        namespace_id=ns,
        relationship_renames=[{"id": str(rid), "old_type": "WORKS_FOR", "new_type": "EMPLOYED_BY"}],
        mapping={"WORKS_FOR": "EMPLOYED_BY"},
    )

    await apply_vectorcypher_normalize_schema(
        op,
        coordinator=coord,  # type: ignore[arg-type]
        session=session,
    )

    sql_blob = " | ".join(s.upper() for s, _ in session.executed)
    assert "UPDATE RELATIONSHIPS" in sql_blob
    assert "RELATIONSHIP_TYPE" in sql_blob

    assert len(coord.dispatched_events) == 1
    ev = coord.dispatched_events[0]
    assert ev.event_type is EventType.RELATIONSHIP_UPDATED
    assert ev.resource_id == rid
    assert ev.namespace_id == ns
    assert ev.data["old_type"] == "WORKS_FOR"
    assert ev.data["new_type"] == "EMPLOYED_BY"


@pytest.mark.asyncio
async def test_apply_emits_one_event_per_renamed_item() -> None:
    """A plan with three entity renames produces three MemoryEvents."""
    ns = uuid4()
    coord = _FakeCoordinator()
    eids = [uuid4(), uuid4(), uuid4()]
    session = _FakeSession(entity_rows=[(eid, "PERSON_NAME") for eid in eids])
    op = _build_apply_op(
        namespace_id=ns,
        entity_renames=[{"id": str(eid), "old_type": "PERSON_NAME", "new_type": "PERSON"} for eid in eids],
    )

    await apply_vectorcypher_normalize_schema(
        op,
        coordinator=coord,  # type: ignore[arg-type]
        session=session,
    )

    assert len(coord.dispatched_events) == 3
    emitted_ids = {ev.resource_id for ev in coord.dispatched_events}
    assert emitted_ids == set(eids)


@pytest.mark.asyncio
async def test_apply_is_idempotent_on_already_renamed_rows() -> None:
    """Rows whose entity_type already matches new_type are skipped.

    A replay scenario: the plan says "rename id=X from PERSON_NAME to
    PERSON", but the SELECT shows the row already has PERSON. The
    handler MUST NOT issue an UPDATE for that row and MUST NOT emit a
    MemoryEvent for it.
    """
    ns = uuid4()
    already_done = uuid4()
    needs_rename = uuid4()
    coord = _FakeCoordinator()
    session = _FakeSession(
        entity_rows=[
            (already_done, "PERSON"),  # already renamed
            (needs_rename, "PERSON_NAME"),
        ]
    )
    op = _build_apply_op(
        namespace_id=ns,
        entity_renames=[
            {"id": str(already_done), "old_type": "PERSON_NAME", "new_type": "PERSON"},
            {"id": str(needs_rename), "old_type": "PERSON_NAME", "new_type": "PERSON"},
        ],
    )

    undo = await apply_vectorcypher_normalize_schema(
        op,
        coordinator=coord,  # type: ignore[arg-type]
        session=session,
    )

    # Only one UPDATE statement should have been issued.
    update_count = sum(1 for s, _ in session.executed if "UPDATE ENTITIES" in s.upper())
    assert update_count == 1
    # Only one event for the row that was actually renamed.
    assert len(coord.dispatched_events) == 1
    assert coord.dispatched_events[0].resource_id == needs_rename
    # Undo carries only the row that was actually rewritten.
    assert len(undo.before["entity_renames"]) == 1
    assert undo.before["entity_renames"][0]["id"] == str(needs_rename)


@pytest.mark.asyncio
async def test_apply_undo_round_trip_restores_original_types() -> None:
    """Swapping the mapping (new->old) on a second plan/apply cycle reverses the rename."""
    ns = uuid4()
    eid = uuid4()
    coord = _FakeCoordinator()
    # First apply: PERSON_NAME -> PERSON.
    session1 = _FakeSession(entity_rows=[(eid, "PERSON_NAME")])
    op1 = _build_apply_op(
        namespace_id=ns,
        entity_renames=[{"id": str(eid), "old_type": "PERSON_NAME", "new_type": "PERSON"}],
        mapping={"PERSON_NAME": "PERSON"},
    )
    undo1 = await apply_vectorcypher_normalize_schema(
        op1,
        coordinator=coord,
        session=session1,  # type: ignore[arg-type]
    )
    # The UPDATE statement carried the new_type, validating direction 1.
    payload1 = [params for sql, params in session1.executed if "UPDATE" in sql.upper()]
    assert any(p.get("new_type") == "PERSON" for p in payload1)
    assert undo1.before["entity_renames"][0]["old_type"] == "PERSON_NAME"
    assert undo1.before["entity_renames"][0]["new_type"] == "PERSON"

    # Second apply: PERSON -> PERSON_NAME (swap).
    session2 = _FakeSession(entity_rows=[(eid, "PERSON")])
    op2 = _build_apply_op(
        namespace_id=ns,
        entity_renames=[{"id": str(eid), "old_type": "PERSON", "new_type": "PERSON_NAME"}],
        mapping={"PERSON": "PERSON_NAME"},
    )
    await apply_vectorcypher_normalize_schema(
        op2,
        coordinator=coord,
        session=session2,  # type: ignore[arg-type]
    )
    payload2 = [params for sql, params in session2.executed if "UPDATE" in sql.upper()]
    assert any(p.get("new_type") == "PERSON_NAME" for p in payload2)

    # Two events total — one per applied rename direction.
    assert len(coord.dispatched_events) == 2
    assert coord.dispatched_events[0].data["new_type"] == "PERSON"
    assert coord.dispatched_events[1].data["new_type"] == "PERSON_NAME"


@pytest.mark.asyncio
async def test_apply_empty_op_returns_empty_undo() -> None:
    """An op with no renames in outputs is a no-op apply (idempotent replay)."""
    ns = uuid4()
    coord = _FakeCoordinator()
    session = _FakeSession()
    op = _build_apply_op(namespace_id=ns, entity_renames=[], relationship_renames=[])

    undo = await apply_vectorcypher_normalize_schema(
        op,
        coordinator=coord,  # type: ignore[arg-type]
        session=session,
    )

    assert undo.before["entity_renames"] == []
    assert undo.before["relationship_renames"] == []
    assert coord.dispatched_events == []
    # No UPDATE statements should fire on empty input.
    upd = [s for s, _ in session.executed if "UPDATE" in s.upper()]
    assert upd == []


@pytest.mark.asyncio
async def test_undo_record_has_no_chunk_id_top_level_key() -> None:
    """Safety floor — _assert_no_chunk_id_mutation must accept the undo record."""
    ns = uuid4()
    eid = uuid4()
    coord = _FakeCoordinator()
    session = _FakeSession(entity_rows=[(eid, "PERSON_NAME")])
    op = _build_apply_op(
        namespace_id=ns,
        entity_renames=[{"id": str(eid), "old_type": "PERSON_NAME", "new_type": "PERSON"}],
    )

    undo = await apply_vectorcypher_normalize_schema(
        op,
        coordinator=coord,  # type: ignore[arg-type]
        session=session,
    )

    assert "chunk_id" not in undo.before
