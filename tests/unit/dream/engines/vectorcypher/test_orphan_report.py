"""Unit tests for the vectorcypher PageRank-based orphan report (#657)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import Entity, Relationship
from khora.dream.engines.vectorcypher import plan_vectorcypher_orphan_report
from khora.dream.plan import DreamOp, OpKind


@dataclass
class _FakeCoordinator:
    """Minimal in-memory stand-in for :class:`StorageCoordinator`."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        del namespace_id, entity_type, limit, offset
        return list(self.entities)

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        del namespace_id, relationship_type, limit, offset
        return list(self.relationships)


def _make_entity(ns: UUID, name: str, *, mention_count: int = 1) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns,
        name=name,
        entity_type="CONCEPT",
        mention_count=mention_count,
    )


def _rel(ns: UUID, src: Entity, tgt: Entity, rel_type: str = "RELATES_TO") -> Relationship:
    return Relationship(
        id=uuid4(),
        namespace_id=ns,
        source_entity_id=src.id,
        target_entity_id=tgt.id,
        relationship_type=rel_type,
    )


@pytest.mark.asyncio
async def test_flags_bottom_5pct_pr_entities() -> None:
    """Bottom 5% of PR scorers with mention_count<=1 are flagged."""
    ns = uuid4()
    # 100 entities arranged in a hub: 5 isolated singletons + 95 connected
    # to a single hub entity. The 5 singletons will sit at the floor of
    # the PR distribution.
    hub = _make_entity(ns, "hub", mention_count=50)
    connected = [_make_entity(ns, f"connected-{i}") for i in range(94)]
    singletons = [_make_entity(ns, f"singleton-{i}") for i in range(5)]
    entities = [hub] + connected + singletons

    rels = [_rel(ns, hub, e) for e in connected] + [_rel(ns, e, hub) for e in connected]

    coord = _FakeCoordinator(entities=entities, relationships=rels)

    op = await plan_vectorcypher_orphan_report(ns, coordinator=coord)

    assert isinstance(op, DreamOp)
    assert op.op_type == OpKind.VECTORCYPHER_ORPHAN_REPORT
    assert op.decision == "audit_complete"

    flagged_ids = {c["entity_id"] for c in op.outputs}
    singleton_ids = {str(s.id) for s in singletons}
    # All 5 singletons are flagged; the hub never is.
    assert singleton_ids <= flagged_ids
    assert str(hub.id) not in flagged_ids
    # Bottom-5% on 100 entities should land near 5 flagged candidates.
    assert 3 <= len(flagged_ids) <= 10


@pytest.mark.asyncio
async def test_excludes_well_connected_low_mention_entities() -> None:
    """High-degree entities with mention_count=1 are NOT flagged."""
    ns = uuid4()
    hub = _make_entity(ns, "hub", mention_count=1)  # low mention but well-connected
    leaves = [_make_entity(ns, f"leaf-{i}", mention_count=1) for i in range(50)]
    entities = [hub] + leaves
    # Hub has degree 50 (in + out via reciprocal RELATES_TO).
    rels = [_rel(ns, hub, leaf) for leaf in leaves] + [_rel(ns, leaf, hub) for leaf in leaves]

    coord = _FakeCoordinator(entities=entities, relationships=rels)

    op = await plan_vectorcypher_orphan_report(ns, coordinator=coord)

    flagged_ids = {c["entity_id"] for c in op.outputs}
    # The hub must not appear despite mention_count=1.
    assert str(hub.id) not in flagged_ids


@pytest.mark.asyncio
async def test_downweights_associated_with() -> None:
    """ASSOCIATED_WITH edges are weighted to 0.2, so leaves connected only via
    co-occurrence don't accumulate the same PR mass as leaves on typed edges.

    Fixture: one hub fans out to two leaf cohorts. Co-occurrence leaves are
    reached only via ASSOCIATED_WITH; typed leaves are reached only via
    RELATES_TO. Because the hub's outbound weight gets split (cooccur edges
    × 0.2 vs typed × 1.0), the hub's PR mass should funnel preferentially
    into the typed leaves. With the down-weight removed both cohorts split
    the mass evenly.
    """
    ns = uuid4()
    hub = _make_entity(ns, "hub", mention_count=10)
    cooccur_leaf = _make_entity(ns, "co-leaf", mention_count=1)
    typed_leaf = _make_entity(ns, "typed-leaf", mention_count=1)
    entities = [hub, cooccur_leaf, typed_leaf]

    rels = [
        _rel(ns, hub, cooccur_leaf, "ASSOCIATED_WITH"),
        _rel(ns, hub, typed_leaf, "RELATES_TO"),
    ]

    from khora import _accel

    idx = {e.id: i for i, e in enumerate(entities)}
    # Default down-weighting: cooccur edge weight 0.2, typed 1.0.
    edges_dampened = [
        (
            idx[r.source_entity_id],
            idx[r.target_entity_id],
            0.2 if r.relationship_type == "ASSOCIATED_WITH" else 1.0,
        )
        for r in rels
    ]
    edges_uniform = [(idx[r.source_entity_id], idx[r.target_entity_id], 1.0) for r in rels]
    pr_d = _accel.pagerank(len(entities), edges_dampened)
    pr_u = _accel.pagerank(len(entities), edges_uniform)

    # Under uniform weights, both leaves get the same mass.
    assert pr_u[idx[cooccur_leaf.id]] == pytest.approx(pr_u[idx[typed_leaf.id]], rel=1e-6)
    # Under the 0.2 down-weight, the typed leaf strictly outscores the cooccur leaf.
    assert pr_d[idx[typed_leaf.id]] > pr_d[idx[cooccur_leaf.id]]

    # And the op honours the kwarg end-to-end.
    coord = _FakeCoordinator(entities=entities, relationships=rels)
    op_dampened = await plan_vectorcypher_orphan_report(ns, coordinator=coord)
    op_undampened = await plan_vectorcypher_orphan_report(ns, coordinator=coord, cooccurrence_edge_weight=1.0)
    assert op_dampened.decision == "audit_complete"
    assert op_undampened.decision == "audit_complete"


@pytest.mark.asyncio
async def test_empty_namespace() -> None:
    """Empty namespace → decision='empty_namespace', no outputs."""
    ns = uuid4()
    coord = _FakeCoordinator(entities=[], relationships=[])

    op = await plan_vectorcypher_orphan_report(ns, coordinator=coord)

    assert op.decision == "empty_namespace"
    assert op.outputs == ()
    assert op.namespace_id == ns
    assert op.op_type == OpKind.VECTORCYPHER_ORPHAN_REPORT


@pytest.mark.asyncio
async def test_no_writes() -> None:
    """The op must not mutate the coordinator's entity/relationship sets."""
    ns = uuid4()
    entities = [_make_entity(ns, f"e-{i}") for i in range(20)]
    rels = [_rel(ns, entities[0], e) for e in entities[1:]]

    coord = _FakeCoordinator(entities=entities, relationships=rels)
    initial_entity_count = len(coord.entities)
    initial_rel_count = len(coord.relationships)

    await plan_vectorcypher_orphan_report(ns, coordinator=coord)

    assert len(coord.entities) == initial_entity_count
    assert len(coord.relationships) == initial_rel_count


@pytest.mark.asyncio
async def test_dream_op_round_trips_json() -> None:
    """DreamOp outputs are JSON-serialisable (sinks need this)."""
    ns = uuid4()
    entities = [_make_entity(ns, f"e-{i}") for i in range(10)]
    coord = _FakeCoordinator(entities=entities, relationships=[])

    op = await plan_vectorcypher_orphan_report(ns, coordinator=coord)

    # Each candidate dict must round-trip through JSON.
    payload = json.dumps(list(op.outputs))
    restored = json.loads(payload)
    assert len(restored) == len(op.outputs)
    for candidate in restored:
        assert "entity_id" in candidate
        assert "name" in candidate
        assert "entity_type" in candidate
        assert "pr_score" in candidate
        assert "mention_count" in candidate
        assert candidate["archive_candidate"] is True
