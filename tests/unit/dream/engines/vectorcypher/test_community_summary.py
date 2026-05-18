"""Unit tests for the vectorcypher community-summary op (#670).

Phase 5.1 of the dream-phase rollout (umbrella #649). The op is the
first LLM-using dream op — pioneers the LLM-budget integration that
other Phase 5 ops will reuse.

Tests cover:

  * Planner: cluster discovery, ``min_size`` threshold, ``ASSOCIATED_WITH``
    down-weighting (0.2), empty-namespace skip.
  * Apply handler: grounding validator drops uncited claims, persists to
    DB via session, idempotent on replay, model is configurable.
  * No real LLM network calls — :func:`khora.config.llm.acompletion` is
    monkeypatched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import Entity, Relationship
from khora.dream.engines.vectorcypher.community_summary import (
    GroundedSummary,
    SummaryClaim,
    apply_vectorcypher_community_summary,
    plan_vectorcypher_community_summary,
    validate_grounding,
)
from khora.dream.plan import OpKind

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeCoordinator:
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


class _FakeSession:
    def __init__(self, dialect_name: str = "postgresql") -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
        self.executed: list[tuple[str, dict[str, Any]]] = []
        # community_id -> existing row tuple (id, namespace_id, valid_to)
        self.existing: dict[UUID, Any] = {}

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        text_str = str(stmt)
        params = params or {}
        upper = text_str.lstrip().upper()
        self.executed.append((text_str, params))
        if upper.startswith("SELECT"):
            cid = params.get("cid")
            try:
                key = cid if isinstance(cid, UUID) else UUID(str(cid))
            except (TypeError, ValueError):
                key = None
            row = self.existing.get(key)
            return _Result([row] if row is not None else [])
        return SimpleNamespace(rowcount=1)


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def first(self) -> Any:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


def _norm(vec: list[float]) -> list[float]:
    import math

    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def _entity(ns: UUID, name: str, *, mention_count: int = 1, entity_type: str = "ORG") -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns,
        name=name,
        entity_type=entity_type,
        embedding=_norm([1.0, 0.0, 0.0]),
        mention_count=mention_count,
    )


def _rel(ns: UUID, src: UUID, tgt: UUID, rel_type: str = "WORKS_WITH") -> Relationship:
    return Relationship(
        id=uuid4(),
        namespace_id=ns,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
    )


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_emits_one_op_per_community_above_min_size() -> None:
    """Two well-connected clusters → two community ops (above min_size=3)."""
    ns = uuid4()
    # Cluster A: 4 entities densely connected
    a1, a2, a3, a4 = (_entity(ns, f"A{i}") for i in range(4))
    # Cluster B: 4 entities densely connected
    b1, b2, b3, b4 = (_entity(ns, f"B{i}") for i in range(4))
    entities = [a1, a2, a3, a4, b1, b2, b3, b4]
    rels = []
    for src, tgt in [(a1, a2), (a2, a3), (a3, a4), (a1, a3), (a2, a4), (a1, a4)]:
        rels.append(_rel(ns, src.id, tgt.id))
        rels.append(_rel(ns, tgt.id, src.id))
    for src, tgt in [(b1, b2), (b2, b3), (b3, b4), (b1, b3), (b2, b4), (b1, b4)]:
        rels.append(_rel(ns, src.id, tgt.id))
        rels.append(_rel(ns, tgt.id, src.id))
    coord = _FakeCoordinator(entities=entities, relationships=rels)

    ops = await plan_vectorcypher_community_summary(ns, coordinator=coord, min_size=3)

    assert len(ops) == 2
    for op in ops:
        assert op.op_type == OpKind.VECTORCYPHER_COMMUNITY_SUMMARY
        assert op.decision == "planned"
        member_ids = op.inputs[0]["member_ids"]
        assert len(member_ids) >= 3


@pytest.mark.asyncio
async def test_planner_skips_communities_below_min_size() -> None:
    """A 2-entity cluster is dropped when min_size=5."""
    ns = uuid4()
    a, b = _entity(ns, "A"), _entity(ns, "B")
    rels = [_rel(ns, a.id, b.id), _rel(ns, b.id, a.id)]
    coord = _FakeCoordinator(entities=[a, b], relationships=rels)

    ops = await plan_vectorcypher_community_summary(ns, coordinator=coord, min_size=5)

    assert ops == ()


@pytest.mark.asyncio
async def test_planner_downweights_associated_with_edges() -> None:
    """A pair of WORKS_WITH-linked entities stays together; ASSOCIATED_WITH
    co-occurrence edges are weighted 0.2 so they cannot single-handedly
    glue otherwise-disconnected nodes into one community.
    """
    ns = uuid4()
    # Cluster A: 3 nodes tightly connected
    a1, a2, a3 = (_entity(ns, f"A{i}") for i in range(3))
    # Cluster B: 3 nodes tightly connected
    b1, b2, b3 = (_entity(ns, f"B{i}") for i in range(3))
    rels = []
    for src, tgt in [(a1, a2), (a2, a3), (a1, a3)]:
        rels.append(_rel(ns, src.id, tgt.id, "WORKS_WITH"))
        rels.append(_rel(ns, tgt.id, src.id, "WORKS_WITH"))
    for src, tgt in [(b1, b2), (b2, b3), (b1, b3)]:
        rels.append(_rel(ns, src.id, tgt.id, "WORKS_WITH"))
        rels.append(_rel(ns, tgt.id, src.id, "WORKS_WITH"))
    # A single ASSOCIATED_WITH bridge — under full-weight Louvain this
    # might merge the two; under 0.2 weighting it shouldn't.
    rels.append(_rel(ns, a1.id, b1.id, "ASSOCIATED_WITH"))
    rels.append(_rel(ns, b1.id, a1.id, "ASSOCIATED_WITH"))
    coord = _FakeCoordinator(entities=[a1, a2, a3, b1, b2, b3], relationships=rels)

    ops = await plan_vectorcypher_community_summary(ns, coordinator=coord, min_size=3)

    # Both clusters should remain distinct → two ops.
    assert len(ops) == 2


@pytest.mark.asyncio
async def test_planner_empty_namespace_returns_no_ops() -> None:
    """Zero entities → empty tuple, no LLM call would fire."""
    ns = uuid4()
    coord = _FakeCoordinator()

    ops = await plan_vectorcypher_community_summary(ns, coordinator=coord, min_size=3)

    assert ops == ()


@pytest.mark.asyncio
async def test_planner_op_inputs_carry_member_ids_and_relationship_modes() -> None:
    """Op carries enough info that the apply handler can re-read raw members."""
    ns = uuid4()
    a1, a2, a3 = (_entity(ns, f"A{i}", mention_count=5 - i) for i in range(3))
    rels = []
    for src, tgt in [(a1, a2), (a2, a3), (a1, a3)]:
        rels.append(_rel(ns, src.id, tgt.id, "WORKS_WITH"))
        rels.append(_rel(ns, tgt.id, src.id, "WORKS_WITH"))
    coord = _FakeCoordinator(entities=[a1, a2, a3], relationships=rels)

    (op,) = await plan_vectorcypher_community_summary(ns, coordinator=coord, min_size=3)

    payload = op.inputs[0]
    assert "community_id" in payload
    assert "member_ids" in payload
    assert "relationship_modes" in payload
    assert payload["relationship_modes"]["WORKS_WITH"] >= 3
    # Member ids are stable strings for JSON sink.
    assert all(isinstance(m, str) for m in payload["member_ids"])


# ---------------------------------------------------------------------------
# Grounding validator tests
# ---------------------------------------------------------------------------


def test_grounding_validator_keeps_claims_with_cited_member_ids() -> None:
    """A claim citing a known member_id is preserved."""
    member_ids = {"e1", "e2"}
    summary = GroundedSummary(
        text="e1 and e2 collaborate often.",
        claims=[
            SummaryClaim(text="e1 collaborates with e2", cited_entity_ids=["e1", "e2"]),
        ],
    )
    kept, dropped = validate_grounding(summary, member_ids)

    assert len(kept) == 1
    assert dropped == []


def test_grounding_validator_drops_uncited_claims() -> None:
    """A claim with no citations is dropped."""
    member_ids = {"e1", "e2"}
    summary = GroundedSummary(
        text="Something unsupported.",
        claims=[
            SummaryClaim(text="e1 won a Nobel Prize", cited_entity_ids=[]),
            SummaryClaim(text="e1 collaborates with e2", cited_entity_ids=["e1", "e2"]),
        ],
    )
    kept, dropped = validate_grounding(summary, member_ids)

    assert len(kept) == 1
    assert kept[0].text == "e1 collaborates with e2"
    assert len(dropped) == 1
    assert dropped[0].text == "e1 won a Nobel Prize"


def test_grounding_validator_drops_claims_citing_unknown_ids() -> None:
    """A claim citing an entity_id not in member_ids is dropped (fabrication)."""
    member_ids = {"e1", "e2"}
    summary = GroundedSummary(
        text="Mixed grounding.",
        claims=[
            SummaryClaim(text="e1 funds e99", cited_entity_ids=["e1", "e99"]),
            SummaryClaim(text="e2 leads e1", cited_entity_ids=["e2", "e1"]),
        ],
    )
    kept, dropped = validate_grounding(summary, member_ids)

    assert len(kept) == 1
    assert kept[0].text == "e2 leads e1"
    assert len(dropped) == 1


# ---------------------------------------------------------------------------
# Apply handler tests
# ---------------------------------------------------------------------------


def _make_planned_op(ns: UUID, member_ids: list[UUID]) -> Any:
    from khora.dream.plan import DreamOp

    return DreamOp(
        op_id=uuid4(),
        phase="mutation",
        op_type=OpKind.VECTORCYPHER_COMMUNITY_SUMMARY,
        inputs=(
            {
                "community_id": str(uuid4()),
                "member_ids": [str(m) for m in member_ids],
                "member_names": [f"e{i}" for i in range(len(member_ids))],
                "relationship_modes": {"WORKS_WITH": 3},
                "cluster_size": len(member_ids),
            },
        ),
        outputs=(),
        decision="planned",
        rationale="cluster ready for summarisation",
        started_at=datetime.now(UTC),
        namespace_id=ns,
    )


@pytest.mark.asyncio
async def test_apply_handler_persists_grounded_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: LLM returns a grounded summary, handler writes one row."""
    ns = uuid4()
    members = [uuid4() for _ in range(3)]
    op = _make_planned_op(ns, members)
    session = _FakeSession()

    captured: dict[str, Any] = {}

    async def fake_acompletion(prompt: str, *args: Any, **kwargs: Any) -> str:
        captured["prompt"] = prompt
        captured["model"] = kwargs.get("model") or (args[0].model if args else None)
        captured["telemetry_op"] = kwargs.get("_telemetry_op")
        return (
            "{"
            f'"text": "Members {members[0]}, {members[1]} and {members[2]} collaborate often.",'
            '"claims": ['
            f'{{"text": "{members[0]} collaborates with {members[1]}",'
            f' "cited_entity_ids": ["{members[0]}", "{members[1]}"]}}'
            "]}"
        )

    import khora.dream.engines.vectorcypher.community_summary as mod

    monkeypatch.setattr(mod, "acompletion", fake_acompletion)

    undo = await apply_vectorcypher_community_summary(
        op,
        coordinator=None,
        session=session,
        model="gpt-4o-mini",
    )

    assert undo.op_type == str(OpKind.VECTORCYPHER_COMMUNITY_SUMMARY)
    # An INSERT/UPSERT against the communities table was issued.
    assert any("UPDATE" in s.upper() or "INSERT" in s.upper() for s, _ in session.executed)
    assert captured["telemetry_op"] == "dream_community_summary"


@pytest.mark.asyncio
async def test_apply_handler_drops_uncited_claims_before_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claims with no citation never make it into the persisted payload."""
    ns = uuid4()
    members = [uuid4() for _ in range(3)]
    op = _make_planned_op(ns, members)
    session = _FakeSession()

    async def fake_acompletion(prompt: str, *args: Any, **kwargs: Any) -> str:
        return (
            "{"
            f'"text": "Summary with fabrication.",'
            '"claims": ['
            f'{{"text": "{members[0]} won a Nobel Prize", "cited_entity_ids": []}},'
            f'{{"text": "{members[0]} collaborates with {members[1]}",'
            f' "cited_entity_ids": ["{members[0]}", "{members[1]}"]}}'
            "]}"
        )

    import khora.dream.engines.vectorcypher.community_summary as mod

    monkeypatch.setattr(mod, "acompletion", fake_acompletion)

    undo = await apply_vectorcypher_community_summary(
        op,
        coordinator=None,
        session=session,
        model="gpt-4o-mini",
    )

    # The undo record carries the persisted (filtered) payload and the
    # dropped-claims count for audit.
    assert undo.before.get("dropped_claims", 0) == 1
    assert undo.before.get("kept_claims", 0) == 1


@pytest.mark.asyncio
async def test_apply_handler_idempotent_on_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running the same op id is a noop on the SQL side."""
    ns = uuid4()
    members = [uuid4() for _ in range(3)]
    op = _make_planned_op(ns, members)
    community_id = UUID(op.inputs[0]["community_id"])

    session = _FakeSession()
    # Pre-existing live row keyed by community_id.
    session.existing[community_id] = SimpleNamespace(id=community_id, namespace_id=ns, valid_to=None)

    async def fake_acompletion(prompt: str, *args: Any, **kwargs: Any) -> str:
        return (
            "{"
            f'"text": "Replay summary",'
            '"claims": ['
            f'{{"text": "{members[0]} works with {members[1]}",'
            f' "cited_entity_ids": ["{members[0]}", "{members[1]}"]}}'
            "]}"
        )

    import khora.dream.engines.vectorcypher.community_summary as mod

    monkeypatch.setattr(mod, "acompletion", fake_acompletion)

    undo = await apply_vectorcypher_community_summary(
        op,
        coordinator=None,
        session=session,
        model="gpt-4o-mini",
    )

    # Idempotent replay → returns a noop UndoRecord without an INSERT.
    assert undo.before.get("noop") is True
    inserts = [s for s, _ in session.executed if "INSERT" in s.upper()]
    assert inserts == []


@pytest.mark.asyncio
async def test_apply_handler_drops_all_claims_when_summary_is_ungrounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every claim is dropped, the handler records a 'no_grounded_claims'
    decision and persists nothing rather than writing an empty summary.
    """
    ns = uuid4()
    members = [uuid4() for _ in range(3)]
    op = _make_planned_op(ns, members)
    session = _FakeSession()

    async def fake_acompletion(prompt: str, *args: Any, **kwargs: Any) -> str:
        return (
            "{"
            '"text": "Hallucinated summary",'
            '"claims": ['
            '{"text": "alpha did beta", "cited_entity_ids": []},'
            '{"text": "gamma met delta", "cited_entity_ids": ["bogus-id"]}'
            "]}"
        )

    import khora.dream.engines.vectorcypher.community_summary as mod

    monkeypatch.setattr(mod, "acompletion", fake_acompletion)

    undo = await apply_vectorcypher_community_summary(
        op,
        coordinator=None,
        session=session,
        model="gpt-4o-mini",
    )

    assert undo.before.get("noop") is True
    assert undo.before.get("reason") == "no_grounded_claims"
    inserts = [s for s, _ in session.executed if "INSERT" in s.upper()]
    assert inserts == []


@pytest.mark.asyncio
async def test_apply_handler_respects_configurable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller can override the LLM model; the prompt is forwarded with that model."""
    ns = uuid4()
    members = [uuid4() for _ in range(3)]
    op = _make_planned_op(ns, members)
    session = _FakeSession()

    captured: dict[str, Any] = {}

    async def fake_acompletion(prompt: str, config: Any = None, **kwargs: Any) -> str:
        captured["model"] = config.model if config is not None else None
        return (
            "{"
            f'"text": "ok",'
            '"claims": ['
            f'{{"text": "{members[0]} collaborates with {members[1]}",'
            f' "cited_entity_ids": ["{members[0]}", "{members[1]}"]}}'
            "]}"
        )

    import khora.dream.engines.vectorcypher.community_summary as mod

    monkeypatch.setattr(mod, "acompletion", fake_acompletion)

    await apply_vectorcypher_community_summary(
        op,
        coordinator=None,
        session=session,
        model="gpt-4o-2025-test",
    )

    assert captured["model"] == "gpt-4o-2025-test"
