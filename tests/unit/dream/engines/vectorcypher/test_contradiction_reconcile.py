"""Unit tests for the vectorcypher contradiction-reconciliation op (#1281).

Two-LLM-judged, opt-in promotion of the report-only contradiction detector:

  * agree -> soft-delete the losing edge (mirror-shaped undo) + triage row
  * disagree / timeout / ungrounded -> defer: NO mutation + triage row + skip_reason
  * keep -> no mutation + triage row

The judge is mocked by patching ``khora.config.llm.acompletion`` with a
model-aware async stub so the two concurrent judges (verifier + auditor) return
deterministic verdicts regardless of ``asyncio.gather`` scheduling order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import Relationship
from khora.dream.config import DreamConfig
from khora.dream.engines.vectorcypher.contradiction_reconcile import (
    apply_vectorcypher_contradiction_reconcile,
    plan_vectorcypher_contradiction_reconcile,
    run_contradiction_judge,
)
from khora.dream.graph_mirror import extract_mirror_targets
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeCoordinator:
    relationships: list[Relationship] = field(default_factory=list)

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
    """Postgres-shaped session double — records every execute() call."""

    def __init__(self, dialect_name: str = "postgresql", soft_delete_rowcount: int = 1) -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self._soft_delete_rowcount = soft_delete_rowcount

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        sql = str(stmt)
        self.executed.append((sql, dict(params or {})))
        if sql.lstrip().upper().startswith("UPDATE RELATIONSHIPS"):
            return SimpleNamespace(rowcount=self._soft_delete_rowcount)
        return SimpleNamespace(rowcount=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rel(
    ns: UUID,
    src: UUID,
    tgt: UUID,
    rel_type: str,
    *,
    description: str = "",
    properties: dict[str, Any] | None = None,
    confidence: float = 1.0,
    source_chunk_ids: list[UUID] | None = None,
    rel_id: UUID | None = None,
    valid_until: datetime | None = None,
) -> Relationship:
    return Relationship(
        id=rel_id or uuid4(),
        namespace_id=ns,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description=description,
        properties=properties or {},
        confidence=confidence,
        source_chunk_ids=source_chunk_ids or [],
        valid_until=valid_until,
    )


def _verdict_json(
    decision: str, *, loser: str = "", confidence: float = 0.9, evidence_ids: list[str] | None = None
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "loser": loser,
            "confidence": confidence,
            "evidence_ids": evidence_ids or [],
            "rationale": "test rationale",
        }
    )


def _install_judge(
    monkeypatch: pytest.MonkeyPatch, by_model: dict[str, str], *, raise_for: set[str] | None = None
) -> None:
    """Patch khora.config.llm.acompletion with a model-aware stub.

    ``by_model`` maps a model id to its raw JSON verdict string. Models in
    ``raise_for`` raise TimeoutError (to exercise the defer-on-timeout path).
    """
    raise_for = raise_for or set()

    async def _fake_acompletion(
        prompt: str, config: Any = None, *, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        model = getattr(config, "model", None)
        if model in raise_for:
            raise TimeoutError("simulated judge timeout")
        return by_model.get(model, _verdict_json("defer"))

    import khora.config.llm as llm_mod

    monkeypatch.setattr(llm_mod, "acompletion", _fake_acompletion)


def _config() -> DreamConfig:
    return DreamConfig(
        contradiction_reconcile_enabled=True,
        contradiction_reconcile_model="gpt-4o-mini",
        contradiction_reconcile_auditor_model="claude-haiku-4.5",
        contradiction_reconcile_min_confidence=0.6,
    )


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_flags_pair_and_enriches_grounding() -> None:
    """The planner reuses the detector logic and enriches each finding with
    both sides' confidence + source ids so the apply-side judge can decide."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    chunk_a, chunk_b = uuid4(), uuid4()
    rels = [
        _rel(
            ns,
            a,
            b,
            "EMPLOYED_BY",
            description="Employed at the company",
            properties={"status": "active"},
            confidence=0.9,
            source_chunk_ids=[chunk_a],
        ),
        _rel(
            ns,
            a,
            b,
            "EMPLOYED_BY",
            description="Employed at the company",
            properties={"status": "ended"},
            confidence=0.3,
            source_chunk_ids=[chunk_b],
        ),
    ]
    coord = _FakeCoordinator(relationships=rels)

    op = await plan_vectorcypher_contradiction_reconcile(ns, coordinator=coord)

    assert op.op_type == OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE
    assert op.decision == "planned"
    assert len(op.outputs) == 1
    finding = op.outputs[0]
    assert "status" in finding["contradicting_keys"]
    # Grounding enrichment present for both sides.
    assert finding["a_confidence"] in (0.9, 0.3)
    assert finding["b_confidence"] in (0.9, 0.3)
    all_chunks = set(finding["a_source_chunk_ids"]) | set(finding["b_source_chunk_ids"])
    assert {str(chunk_a), str(chunk_b)} <= all_chunks


@pytest.mark.asyncio
async def test_plan_makes_no_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The planner is pure — the (budget-gated) judge runs at apply time."""
    called = False

    async def _boom(*a: Any, **k: Any) -> str:
        nonlocal called
        called = True
        return _verdict_json("defer")

    import khora.config.llm as llm_mod

    monkeypatch.setattr(llm_mod, "acompletion", _boom)

    ns = uuid4()
    a, b = uuid4(), uuid4()
    rels = [
        _rel(ns, a, b, "WORKS_AT", description="A", properties={"status": "active"}),
        _rel(ns, a, b, "WORKS_AT", description="B different", properties={"status": "ended"}),
    ]
    await plan_vectorcypher_contradiction_reconcile(ns, coordinator=_FakeCoordinator(relationships=rels))
    assert called is False


# ---------------------------------------------------------------------------
# Judge dispatcher tests
# ---------------------------------------------------------------------------


def _finding(*, a_id: UUID, b_id: UUID, chunk: UUID) -> dict[str, Any]:
    return {
        "relationship_a_id": str(a_id),
        "relationship_b_id": str(b_id),
        "source_entity_id": str(uuid4()),
        "target_entity_id": str(uuid4()),
        "relationship_type": "EMPLOYED_BY",
        "similarity": 0.2,
        "contradicting_keys": ["status"],
        "reason": "both",
        "description_a_hash": "aaaa1111",
        "description_b_hash": "bbbb2222",
        "a_confidence": 0.9,
        "b_confidence": 0.3,
        "a_description": "active",
        "b_description": "ended",
        "a_source_chunk_ids": [str(chunk)],
        "b_source_chunk_ids": [],
        "a_source_document_ids": [],
        "b_source_document_ids": [],
    }


@pytest.mark.asyncio
async def test_judge_agrees_invalidate_grounded(monkeypatch: pytest.MonkeyPatch) -> None:
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    verdict = _verdict_json("invalidate", loser="b", confidence=0.9, evidence_ids=[str(chunk)])
    _install_judge(monkeypatch, {"gpt-4o-mini": verdict, "claude-haiku-4.5": verdict})

    outcome = await run_contradiction_judge(finding, config=_config())

    assert outcome.decision == "invalidate"
    assert outcome.loser == "b"


@pytest.mark.asyncio
async def test_judge_defers_on_ungrounded_citation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both judges say invalidate but cite an id NOT in the candidate's source
    set — the verdict is rejected as ungrounded and degrades to defer."""
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    bogus = str(uuid4())  # not in any source id
    verdict = _verdict_json("invalidate", loser="b", confidence=0.9, evidence_ids=[bogus])
    _install_judge(monkeypatch, {"gpt-4o-mini": verdict, "claude-haiku-4.5": verdict})

    outcome = await run_contradiction_judge(finding, config=_config())

    assert outcome.decision == "defer"


@pytest.mark.asyncio
async def test_judge_defers_on_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    _install_judge(
        monkeypatch,
        {
            "gpt-4o-mini": _verdict_json("invalidate", loser="b", confidence=0.9, evidence_ids=[str(chunk)]),
            "claude-haiku-4.5": _verdict_json("keep", confidence=0.9),
        },
    )

    outcome = await run_contradiction_judge(finding, config=_config())

    assert outcome.decision == "defer"


@pytest.mark.asyncio
async def test_judge_defers_on_loser_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both say invalidate but pick DIFFERENT losers — defer (never guess)."""
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    _install_judge(
        monkeypatch,
        {
            "gpt-4o-mini": _verdict_json("invalidate", loser="a", confidence=0.9, evidence_ids=[str(chunk)]),
            "claude-haiku-4.5": _verdict_json("invalidate", loser="b", confidence=0.9, evidence_ids=[str(chunk)]),
        },
    )

    outcome = await run_contradiction_judge(finding, config=_config())

    assert outcome.decision == "defer"


@pytest.mark.asyncio
async def test_judge_defers_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    verdict = _verdict_json("invalidate", loser="b", confidence=0.9, evidence_ids=[str(chunk)])
    _install_judge(
        monkeypatch,
        {"gpt-4o-mini": verdict, "claude-haiku-4.5": verdict},
        raise_for={"claude-haiku-4.5"},
    )

    outcome = await run_contradiction_judge(finding, config=_config())

    assert outcome.decision == "defer"


@pytest.mark.asyncio
async def test_judge_defers_on_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    verdict = _verdict_json("invalidate", loser="b", confidence=0.4, evidence_ids=[str(chunk)])
    _install_judge(monkeypatch, {"gpt-4o-mini": verdict, "claude-haiku-4.5": verdict})

    outcome = await run_contradiction_judge(finding, config=_config())

    assert outcome.decision == "defer"


@pytest.mark.asyncio
async def test_judge_keep_when_both_keep(monkeypatch: pytest.MonkeyPatch) -> None:
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    keep = _verdict_json("keep", confidence=0.9)
    _install_judge(monkeypatch, {"gpt-4o-mini": keep, "claude-haiku-4.5": keep})

    outcome = await run_contradiction_judge(finding, config=_config())

    assert outcome.decision == "keep"


# ---------------------------------------------------------------------------
# Apply-handler tests
# ---------------------------------------------------------------------------


def _op_from_findings(ns: UUID, findings: list[dict[str, Any]]) -> DreamOp:
    return DreamOp(
        op_id=uuid4(),
        phase="mutation",
        op_type=OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE,
        inputs=({"namespace_id": str(ns)},),
        outputs=tuple(findings),
        decision="planned",
        namespace_id=ns,
    )


@pytest.mark.asyncio
async def test_apply_agree_soft_deletes_loser_and_writes_triage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge agreement soft-deletes the loser (mirror-shaped undo) and UPSERTs
    an 'invalidated' triage row. No hard delete."""
    ns = uuid4()
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    verdict = _verdict_json("invalidate", loser="b", confidence=0.9, evidence_ids=[str(chunk)])
    _install_judge(monkeypatch, {"gpt-4o-mini": verdict, "claude-haiku-4.5": verdict})

    session = _FakeSession()
    undo = await apply_vectorcypher_contradiction_reconcile(
        _op_from_findings(ns, [finding]),
        coordinator=None,
        session=session,
        dream_config=_config(),
    )

    # Exactly one UPDATE relationships ... valid_to (soft-delete of loser b).
    updates = [(sql, p) for sql, p in session.executed if sql.lstrip().upper().startswith("UPDATE RELATIONSHIPS")]
    assert len(updates) == 1
    assert "VALID_TO" in updates[0][0].upper()
    assert updates[0][1]["rid"] == b_id  # the loser, not the winner

    # Never a DELETE.
    for sql, _ in session.executed:
        assert "DELETE" not in sql.upper()

    # A triage row UPSERTed with resolution='invalidated'.
    inserts = [(sql, p) for sql, p in session.executed if "DREAM_CONFLICTS" in sql.upper()]
    assert len(inserts) == 1
    assert "ON CONFLICT" in inserts[0][0].upper()
    assert "DO UPDATE" in inserts[0][0].upper()
    assert inserts[0][1]["resolution"] == "invalidated"
    assert inserts[0][1]["loser_relationship_id"] == b_id
    assert inserts[0][1]["winner_relationship_id"] == a_id

    # Undo is shaped exactly like prune_edges so the #1272 mirror picks it up.
    assert isinstance(undo, UndoRecord)
    assert undo.before["relationships"] == [{"relationship_id": str(b_id)}]
    targets = extract_mirror_targets(OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE.value, undo)
    assert targets["invalidate_relationship_ids"] == [b_id]
    assert targets["retire_entity_ids"] == []


@pytest.mark.asyncio
async def test_apply_defer_does_not_mutate_writes_triage_and_skip_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disagreement -> no soft-delete, a 'deferred' triage row, and an ADR-001
    skip_reason on the undo record."""
    ns = uuid4()
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    _install_judge(
        monkeypatch,
        {
            "gpt-4o-mini": _verdict_json("invalidate", loser="b", confidence=0.9, evidence_ids=[str(chunk)]),
            "claude-haiku-4.5": _verdict_json("keep", confidence=0.9),
        },
    )

    session = _FakeSession()
    undo = await apply_vectorcypher_contradiction_reconcile(
        _op_from_findings(ns, [finding]),
        coordinator=None,
        session=session,
        dream_config=_config(),
    )

    # No relationship mutated.
    updates = [sql for sql, _ in session.executed if sql.lstrip().upper().startswith("UPDATE RELATIONSHIPS")]
    assert updates == []
    assert undo.before["relationships"] == []

    # Triage row written with resolution='deferred' (no loser/winner).
    inserts = [(sql, p) for sql, p in session.executed if "DREAM_CONFLICTS" in sql.upper()]
    assert len(inserts) == 1
    assert inserts[0][1]["resolution"] == "deferred"
    assert inserts[0][1]["loser_relationship_id"] is None

    # ADR-001: a structured skip_reason is recorded.
    skip_reasons = undo.before.get("skip_reasons")
    assert skip_reasons and skip_reasons[0]["reason"] == "reconcile_defer"

    # The mirror is a no-op (nothing to invalidate).
    targets = extract_mirror_targets(OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE.value, undo)
    assert targets["invalidate_relationship_ids"] == []


@pytest.mark.asyncio
async def test_apply_keep_writes_kept_triage_no_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    ns = uuid4()
    a_id, b_id, chunk = uuid4(), uuid4(), uuid4()
    finding = _finding(a_id=a_id, b_id=b_id, chunk=chunk)
    keep = _verdict_json("keep", confidence=0.9)
    _install_judge(monkeypatch, {"gpt-4o-mini": keep, "claude-haiku-4.5": keep})

    session = _FakeSession()
    undo = await apply_vectorcypher_contradiction_reconcile(
        _op_from_findings(ns, [finding]),
        coordinator=None,
        session=session,
        dream_config=_config(),
    )

    updates = [sql for sql, _ in session.executed if sql.lstrip().upper().startswith("UPDATE RELATIONSHIPS")]
    assert updates == []
    inserts = [(sql, p) for sql, p in session.executed if "DREAM_CONFLICTS" in sql.upper()]
    assert inserts[0][1]["resolution"] == "kept"
    assert undo.before["relationships"] == []


@pytest.mark.asyncio
async def test_apply_empty_outputs_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    ns = uuid4()
    _install_judge(monkeypatch, {})
    session = _FakeSession()
    undo = await apply_vectorcypher_contradiction_reconcile(
        _op_from_findings(ns, []),
        coordinator=None,
        session=session,
        dream_config=_config(),
    )
    assert session.executed == []
    assert undo.before["relationships"] == []
    assert undo.before["triage"] == []


# ---------------------------------------------------------------------------
# Config / registry wiring
# ---------------------------------------------------------------------------


def test_opkind_constant_exists() -> None:
    assert OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE.value == "vectorcypher_contradiction_reconcile"


def test_dream_config_reconcile_off_by_default() -> None:
    cfg = DreamConfig()
    assert cfg.contradiction_reconcile_enabled is False
    assert cfg.contradiction_reconcile_model == "gpt-4o-mini"
    assert cfg.contradiction_reconcile_auditor_model == "claude-haiku-4.5"


def test_apply_handler_is_registered() -> None:
    from khora.dream.engines.registry import get_apply_handler

    handler = get_apply_handler(OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE)
    assert handler is apply_vectorcypher_contradiction_reconcile


def test_reconcile_is_budget_gated_and_postgres_only() -> None:
    from khora.dream.orchestrator import _LLM_OP_KINDS, _POSTGRES_ONLY_OP_KINDS

    kind = OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE.value
    assert kind in _LLM_OP_KINDS  # the two-LLM judge respects the dream token budget
    assert kind in _POSTGRES_ONLY_OP_KINDS  # dream_conflicts is PG-only


def test_reconcile_is_mirrorable() -> None:
    from khora.dream.graph_mirror import MIRRORABLE_OP_KINDS

    assert OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE.value in MIRRORABLE_OP_KINDS
