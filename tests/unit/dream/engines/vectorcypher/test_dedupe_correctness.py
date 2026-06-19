"""Dedupe planner->apply correctness keystone (#1265, #1266, #1267).

These tests pin the contract the apply handler already expects:

  * The planner emits ``outputs[0] = {"merges": [...]}`` (read identically
    by :func:`apply_vectorcypher_dedupe_entities`) — not the stale
    ``merged_source_document_ids`` payload that left apply a silent no-op
    (#1265).
  * Transitive candidate edges (A->B, B->C) collapse into ONE component
    with a single canonical; no merge entry points at a retired
    intermediate (#1265 union-find).
  * Shuffling the input entity order yields byte-identical merge payloads
    and an identical ``plan_hash`` (#1266 stable sort + outputs in hash).
  * Graph-routed entities whose embedding was stripped at the graph
    boundary are re-joined from pgvector before the kernel; the
    absent-embedding case records an ADR-001 degradation rather than
    silently under-consolidating (#1267).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import numpy as np
import pytest

from khora.core.models.entity import Entity
from khora.dream.engines.vectorcypher import plan_vectorcypher_dedupe_entities
from khora.dream.plan import DreamOp, DreamPlan


@dataclass
class _FakeCoordinator:
    """Read-only coordinator stand-in.

    ``graph_entities`` is what ``list_entities`` returns (the graph-prefer
    path strips embeddings). ``pgvector_entities`` is keyed by id and is
    what the embedding re-join (#1267) reads via ``get_entities_batch``.
    """

    entities: list[Entity] = field(default_factory=list)
    pgvector_entities: dict[UUID, Entity] | None = None
    has_vector: bool = False
    has_graph: bool = False
    list_calls: int = 0
    mutations: list[str] = field(default_factory=list)
    get_batch_calls: list[list[UUID]] = field(default_factory=list)

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

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        del namespace_id
        self.get_batch_calls.append(list(entity_ids))
        if self.pgvector_entities is None:
            return {}
        return {eid: self.pgvector_entities[eid] for eid in entity_ids if eid in self.pgvector_entities}

    @property
    def _vector(self) -> object | None:
        return object() if self.has_vector else None

    @property
    def _graph(self) -> object | None:
        return object() if self.has_graph else None

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
    entity_id: UUID | None = None,
) -> Entity:
    return Entity(
        id=entity_id or uuid4(),
        namespace_id=ns,
        name=name,
        entity_type=entity_type,
        embedding=_unit(embedding) if embedding is not None else None,
        mention_count=mention_count,
        created_at=created_at or datetime.now(UTC),
        source_document_ids=source_document_ids or [],
        source_chunk_ids=source_chunk_ids or [],
    )


def _merges_of(op: DreamOp) -> list[dict]:
    """Return the ``outputs[0]['merges']`` list the apply handler reads."""
    assert op.outputs, "planned op must carry an outputs payload"
    payload = op.outputs[0]
    assert isinstance(payload, dict)
    return list(payload.get("merges") or [])


# ---------------------------------------------------------------------------
# #1265 — planner->apply bridge: outputs[0]["merges"] is the read contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_emits_merges_payload_apply_reads() -> None:
    """The planned op must carry ``outputs[0]['merges']`` — not the stale
    ``merged_source_document_ids`` payload that apply never reads (#1265)."""
    ns = uuid4()
    a = _make_entity(ns, "Alice Smith", embedding=[1.0, 0.0, 0.0, 0.0], mention_count=5)
    b = _make_entity(ns, "Alice S.", embedding=[1.0, 0.001, 0.0, 0.0], mention_count=2)
    coord = _FakeCoordinator(entities=[a, b])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)

    planned = [op for op in ops if op.decision == "planned"]
    assert len(planned) == 1
    merges = _merges_of(planned[0])
    assert len(merges) == 1
    entry = merges[0]
    # The exact keys apply consumes.
    assert entry["canonical_id"] == str(a.id)
    assert entry["absorbed_id"] == str(b.id)
    assert "similarity_score" in entry


@pytest.mark.asyncio
async def test_apply_consumes_planner_output_end_to_end() -> None:
    """A planned op fed straight into the apply handler must soft-delete the
    absorbed row and rewrite its edge — proving the bridge is wired (#1265)."""
    from types import SimpleNamespace

    from khora.dream.engines.vectorcypher.dedupe_entities import (
        apply_vectorcypher_dedupe_entities,
    )

    ns = uuid4()
    a = _make_entity(ns, "Acme Corp", entity_type="ORG", embedding=[1.0, 0.0, 0.0, 0.0], mention_count=9)
    b = _make_entity(ns, "Acme Co", entity_type="ORG", embedding=[1.0, 0.001, 0.0, 0.0], mention_count=1)
    coord = _FakeCoordinator(entities=[a, b])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    planned = [op for op in ops if op.decision == "planned"]
    assert len(planned) == 1
    op = planned[0]

    rel_id = uuid4()
    other = uuid4()

    class _Result:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

        def first(self):
            return self._rows[0] if self._rows else None

    class _Session:
        def __init__(self) -> None:
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
            self.executed: list[tuple[str, dict]] = []

        async def execute(self, stmt, params=None):
            params = params or {}
            text_str = str(stmt)
            self.executed.append((text_str, params))
            if text_str.lstrip().upper().startswith("SELECT") and "RELATIONSHIPS" in text_str.upper():
                aid = params.get("aid")
                key = aid if isinstance(aid, UUID) else UUID(str(aid))
                if key == b.id:
                    return _Result(
                        [
                            SimpleNamespace(
                                id=rel_id,
                                source_entity_id=b.id,
                                target_entity_id=other,
                                relationship_type="KNOWS",
                            )
                        ]
                    )
                return _Result([])
            return SimpleNamespace(rowcount=1)

    session = _Session()
    undo = await apply_vectorcypher_dedupe_entities(op, coordinator=coord, session=session)

    merges = undo.before["merges"]
    assert len(merges) == 1
    assert UUID(merges[0]["absorbed_id"]) == b.id
    assert UUID(merges[0]["canonical_id"]) == a.id
    blob = " | ".join(s.upper() for s, _ in session.executed)
    assert "UPDATE RELATIONSHIPS" in blob  # edge rewritten — not a no-op
    assert "VALID_UNTIL" in blob  # absorbed soft-deleted


# ---------------------------------------------------------------------------
# #1265 — union-find: transitive components, single canonical
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transitive_chain_collapses_to_single_component() -> None:
    """A->B, B->C must yield ONE component (one op) with one canonical and
    no merge entry pointing at a retired intermediate (#1265 union-find)."""
    ns = uuid4()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    # Three near-collinear vectors: a~b, b~c, but a..c is still over 0.90.
    a = _make_entity(ns, "Robert Allen", embedding=[1.0, 0.00, 0.0, 0.0], mention_count=9, created_at=base)
    b = _make_entity(
        ns, "Robert A.", embedding=[1.0, 0.02, 0.0, 0.0], mention_count=3, created_at=base + timedelta(days=1)
    )
    c = _make_entity(
        ns, "Rob Allen", embedding=[1.0, 0.04, 0.0, 0.0], mention_count=1, created_at=base + timedelta(days=2)
    )
    coord = _FakeCoordinator(entities=[a, b, c])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    planned = [op for op in ops if op.decision == "planned"]

    # Exactly one op for the whole connected component.
    assert len(planned) == 1, f"expected one component op, got {len(planned)}"
    merges = _merges_of(planned[0])

    canonical_ids = {m["canonical_id"] for m in merges}
    assert len(canonical_ids) == 1, "a component must have exactly one canonical"
    canonical = next(iter(canonical_ids))
    assert canonical == str(a.id), "highest mention_count wins the canonical slot"

    absorbed_ids = {m["absorbed_id"] for m in merges}
    # Every non-canonical member is absorbed directly into the canonical.
    assert absorbed_ids == {str(b.id), str(c.id)}
    # No absorbed id is itself a canonical of another merge (no retired
    # intermediate pointed at by another edge).
    assert absorbed_ids.isdisjoint(canonical_ids)


@pytest.mark.asyncio
async def test_disjoint_components_emit_separate_ops() -> None:
    """Two unrelated duplicate pairs produce two component ops (#1265)."""
    ns = uuid4()
    a = _make_entity(ns, "Alice Smith", embedding=[1.0, 0.0, 0.0, 0.0], mention_count=5)
    b = _make_entity(ns, "Alice S.", embedding=[1.0, 0.001, 0.0, 0.0], mention_count=2)
    c = _make_entity(ns, "Bob Jones", embedding=[0.0, 0.0, 1.0, 0.0], mention_count=5)
    d = _make_entity(ns, "Bob J.", embedding=[0.0, 0.0, 1.0, 0.001], mention_count=2)
    coord = _FakeCoordinator(entities=[a, b, c, d])

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    planned = [op for op in ops if op.decision == "planned"]
    assert len(planned) == 2


# ---------------------------------------------------------------------------
# #1266 — permutation invariance + plan_hash covers the merge payload
# ---------------------------------------------------------------------------


def _build_plan(ns: UUID, ops: list[DreamOp]) -> DreamPlan:
    return DreamPlan(plan_id=uuid4(), namespace_id=ns, ops=tuple(ops))


@pytest.mark.asyncio
async def test_shuffled_input_yields_identical_payloads_and_plan_hash() -> None:
    """Shuffling the entity order must not change the merge payloads or the
    plan_hash (#1266 INVARIANT 0 + outputs in the hash)."""
    from khora.dream.engines.registry import plan_hash

    ns = uuid4()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    ents = [
        _make_entity(ns, "Alice Smith", embedding=[1.0, 0.0, 0.0, 0.0], mention_count=5, created_at=base),
        _make_entity(ns, "Alice S.", embedding=[1.0, 0.001, 0.0, 0.0], mention_count=2, created_at=base),
        _make_entity(ns, "Bob Jones", embedding=[0.0, 0.0, 1.0, 0.0], mention_count=5, created_at=base),
        _make_entity(ns, "Bob J.", embedding=[0.0, 0.0, 1.0, 0.001], mention_count=2, created_at=base),
        _make_entity(ns, "Carol Singh", embedding=[0.0, 1.0, 0.0, 0.0], mention_count=1, created_at=base),
    ]

    coord_a = _FakeCoordinator(entities=list(ents))
    ops_a = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord_a, default_threshold=0.90)

    shuffled = [ents[3], ents[0], ents[4], ents[1], ents[2]]
    coord_b = _FakeCoordinator(entities=shuffled)
    ops_b = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord_b, default_threshold=0.90)

    # Merge payloads (the decision-bearing data) must be byte-identical.
    merges_a = sorted(tuple(sorted(m.items())) for op in ops_a for m in _merges_of(op))
    merges_b = sorted(tuple(sorted(m.items())) for op in ops_b for m in _merges_of(op))
    assert merges_a == merges_b

    # And the plan_hash over the full plan must match.
    assert plan_hash(_build_plan(ns, ops_a)) == plan_hash(_build_plan(ns, ops_b))


@pytest.mark.asyncio
async def test_plan_hash_changes_when_merge_set_drifts() -> None:
    """A different merge set must produce a different plan_hash so a resume
    drift is visible (#1266 — outputs must feed the hash)."""
    from khora.dream.engines.registry import plan_hash

    ns = uuid4()
    a = _make_entity(ns, "Alice Smith", embedding=[1.0, 0.0, 0.0, 0.0], mention_count=5)
    b = _make_entity(ns, "Alice S.", embedding=[1.0, 0.001, 0.0, 0.0], mention_count=2)
    coord = _FakeCoordinator(entities=[a, b])
    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    h1 = plan_hash(_build_plan(ns, ops))

    # Mutate the merge payload (simulate a drifted resume) and re-hash.
    op = ops[0]
    drifted_merges = [{**m, "absorbed_id": str(uuid4())} for m in _merges_of(op)]
    drifted_op = DreamOp(
        op_id=op.op_id,
        phase=op.phase,
        op_type=op.op_type,
        inputs=op.inputs,
        outputs=({"merges": drifted_merges},),
        decision=op.decision,
        rationale=op.rationale,
        started_at=op.started_at,
        namespace_id=op.namespace_id,
    )
    h2 = plan_hash(_build_plan(ns, [drifted_op]))
    assert h1 != h2


# ---------------------------------------------------------------------------
# #1267 — embedding re-join from pgvector on graph stacks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_stack_rejoins_embeddings_from_pgvector() -> None:
    """Graph-listed entities carry no embedding; the planner must re-join
    them by id from pgvector so the plan matches the PG-only plan (#1267)."""
    ns = uuid4()
    a_id, b_id = uuid4(), uuid4()
    # Graph-routed copies have NO embedding (Neo4j _record_to_entity strips it).
    a_graph = _make_entity(ns, "Acme Corp", entity_type="ORG", embedding=None, mention_count=9, entity_id=a_id)
    b_graph = _make_entity(ns, "Acme Co", entity_type="ORG", embedding=None, mention_count=1, entity_id=b_id)
    # The pgvector rows DO carry embeddings.
    a_pg = _make_entity(
        ns, "Acme Corp", entity_type="ORG", embedding=[1.0, 0.0, 0.0, 0.0], mention_count=9, entity_id=a_id
    )
    b_pg = _make_entity(
        ns, "Acme Co", entity_type="ORG", embedding=[1.0, 0.001, 0.0, 0.0], mention_count=1, entity_id=b_id
    )

    coord = _FakeCoordinator(
        entities=[a_graph, b_graph],
        pgvector_entities={a_id: a_pg, b_id: b_pg},
        has_vector=True,
        has_graph=True,
    )

    ops = await plan_vectorcypher_dedupe_entities(ns, coordinator=coord, default_threshold=0.90)
    planned = [op for op in ops if op.decision == "planned"]
    assert len(planned) == 1, "embedding re-join should let the dedupe fire on a graph stack"
    assert coord.get_batch_calls, "planner must consult pgvector for embeddings"


@pytest.mark.asyncio
async def test_absent_embeddings_record_degradation() -> None:
    """When embeddings cannot be sourced (no pgvector, graph strips them),
    the planner must surface an ADR-001 degradation rather than silently
    under-consolidating (#1267)."""
    ns = uuid4()
    a = _make_entity(ns, "Acme Corp", entity_type="ORG", embedding=None, mention_count=9)
    b = _make_entity(ns, "Acme Co", entity_type="ORG", embedding=None, mention_count=1)
    # Graph present, vector absent → cannot re-join.
    coord = _FakeCoordinator(entities=[a, b], has_vector=False, has_graph=True)

    degradations: list[dict] = []
    ops = await plan_vectorcypher_dedupe_entities(
        ns, coordinator=coord, default_threshold=0.90, degradations=degradations
    )

    planned = [op for op in ops if op.decision == "planned"]
    assert planned == []
    assert degradations, "absent embeddings must record a degradation"
    reasons = {d.get("reason") for d in degradations}
    assert any("embedding" in str(r) for r in reasons)
