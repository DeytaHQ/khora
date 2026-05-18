"""Unit tests for the vectorcypher contradiction-detection op (#672, Phase 5.3).

Report-only detector — never mutates ``relationships``. Findings are
persisted to ``dream_conflicts`` (PG) so a human triage queue can review
them. Phase 5.4 (#673) consumes the same findings as the natural source
of mapping recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from khora.core.models.entity import Relationship
from khora.dream.engines.vectorcypher.contradiction_detect import (
    apply_vectorcypher_contradiction_detect,
    plan_vectorcypher_contradiction_detect,
)
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

    def __init__(self, dialect_name: str = "postgresql") -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
        self.executed: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        self.executed.append((str(stmt), dict(params or {})))
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
    valid_to: datetime | None = None,
) -> Relationship:
    return Relationship(
        id=uuid4(),
        namespace_id=ns,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description=description,
        properties=properties or {},
        valid_until=valid_to,  # core dataclass uses valid_until
    )


# ---------------------------------------------------------------------------
# Plan-side detection tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flags_low_textual_similarity_pair() -> None:
    """Two live relationships with the same (src,tgt,type) but
    semantically very different ``description`` text are flagged."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    rels = [
        _rel(ns, a, b, "WORKS_AT", description="Permanent senior staff engineer since 2020"),
        _rel(ns, a, b, "WORKS_AT", description="Brief contractor stint in 2018, six weeks"),
    ]
    coord = _FakeCoordinator(relationships=rels)

    op = await plan_vectorcypher_contradiction_detect(ns, coordinator=coord)

    assert isinstance(op, DreamOp)
    assert op.op_type == OpKind.VECTORCYPHER_CONTRADICTION_DETECT
    assert op.decision == "audit_complete"
    assert len(op.outputs) == 1
    finding = op.outputs[0]
    flagged_ids = {finding["relationship_a_id"], finding["relationship_b_id"]}
    assert flagged_ids == {str(rels[0].id), str(rels[1].id)}
    assert finding["reason"] in {"low_similarity", "property_contradiction", "both"}


@pytest.mark.asyncio
async def test_flags_contradicting_properties() -> None:
    """Property contradiction (status=active vs status=ended) is flagged
    even when description text is similar."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    rels = [
        _rel(
            ns,
            a,
            b,
            "EMPLOYED_BY",
            description="Employed at the company",
            properties={"status": "active"},
        ),
        _rel(
            ns,
            a,
            b,
            "EMPLOYED_BY",
            description="Employed at the company",
            properties={"status": "ended"},
        ),
    ]
    coord = _FakeCoordinator(relationships=rels)

    op = await plan_vectorcypher_contradiction_detect(ns, coordinator=coord)

    assert len(op.outputs) == 1
    finding = op.outputs[0]
    assert finding["reason"] in {"property_contradiction", "both"}
    # Contradicting key names are surfaced for triage.
    assert "status" in finding["contradicting_keys"]


@pytest.mark.asyncio
async def test_high_similarity_pair_not_flagged() -> None:
    """Two live relationships with near-identical text + matching props
    are NOT flagged."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    rels = [
        _rel(
            ns,
            a,
            b,
            "WORKS_AT",
            description="Senior engineer at the firm since 2020",
            properties={"status": "active"},
        ),
        _rel(
            ns,
            a,
            b,
            "WORKS_AT",
            description="Senior engineer at the firm since 2020",
            properties={"status": "active"},
        ),
    ]
    coord = _FakeCoordinator(relationships=rels)

    op = await plan_vectorcypher_contradiction_detect(ns, coordinator=coord)

    assert op.outputs == ()
    assert op.decision == "audit_complete"


@pytest.mark.asyncio
async def test_ignores_non_live_relationships() -> None:
    """Relationships with non-NULL valid_to / valid_until are skipped —
    only live relationships participate in contradiction detection."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    closed_at = datetime(2024, 1, 1, tzinfo=UTC)
    rels = [
        _rel(ns, a, b, "WORKS_AT", description="Permanent staff engineer", properties={"status": "active"}),
        _rel(
            ns,
            a,
            b,
            "WORKS_AT",
            description="Brief 2018 contractor stint",
            properties={"status": "ended"},
            valid_to=closed_at,
        ),
    ]
    coord = _FakeCoordinator(relationships=rels)

    op = await plan_vectorcypher_contradiction_detect(ns, coordinator=coord)

    # The closed relationship doesn't participate; the lone live one
    # can't form a pair.
    assert op.outputs == ()


@pytest.mark.asyncio
async def test_singleton_group_not_flagged() -> None:
    """A (src,tgt,type) group with only one live relationship cannot
    contradict itself."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    rels = [_rel(ns, a, b, "WORKS_AT", description="Anything")]
    coord = _FakeCoordinator(relationships=rels)

    op = await plan_vectorcypher_contradiction_detect(ns, coordinator=coord)

    assert op.outputs == ()


@pytest.mark.asyncio
async def test_no_writes_to_coordinator() -> None:
    """The planner must not mutate the coordinator's relationship list."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    rels = [
        _rel(ns, a, b, "WORKS_AT", description="A", properties={"status": "active"}),
        _rel(ns, a, b, "WORKS_AT", description="B different", properties={"status": "ended"}),
    ]
    coord = _FakeCoordinator(relationships=rels)
    snapshot = [r.id for r in coord.relationships]

    await plan_vectorcypher_contradiction_detect(ns, coordinator=coord)

    assert [r.id for r in coord.relationships] == snapshot


@pytest.mark.asyncio
async def test_threshold_kwarg_controls_sensitivity() -> None:
    """A high similarity threshold flags more pairs; a low one flags fewer."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    # Moderately overlapping text; will straddle the threshold range.
    rels = [
        _rel(ns, a, b, "WORKS_AT", description="senior engineer at the firm"),
        _rel(ns, a, b, "WORKS_AT", description="junior consultant at the agency"),
    ]
    coord = _FakeCoordinator(relationships=rels)

    op_high = await plan_vectorcypher_contradiction_detect(ns, coordinator=coord, similarity_threshold=0.95)
    op_low = await plan_vectorcypher_contradiction_detect(ns, coordinator=coord, similarity_threshold=0.05)

    assert len(op_high.outputs) >= len(op_low.outputs)


# ---------------------------------------------------------------------------
# Telemetry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_span_attributes_use_bounded_text_hash() -> None:
    """Free-text attributes on the span MUST be hashed via
    bounded_text_hash, never sent as raw text."""
    ns = uuid4()
    a, b = uuid4(), uuid4()
    raw_desc_a = "Permanent senior engineer since 2020"
    raw_desc_b = "Brief contractor stint in 2018"
    rels = [
        _rel(ns, a, b, "WORKS_AT", description=raw_desc_a, properties={"status": "active"}),
        _rel(ns, a, b, "WORKS_AT", description=raw_desc_b, properties={"status": "ended"}),
    ]
    coord = _FakeCoordinator(relationships=rels)

    with patch("khora.dream.engines.vectorcypher.contradiction_detect.trace_span") as mock_span_factory:
        # Allow span object to record set_attribute / __enter__ / __exit__.
        recorder = _SpanRecorder()
        mock_span_factory.return_value = recorder

        await plan_vectorcypher_contradiction_detect(ns, coordinator=coord)

    # The span itself fired once at op level; each finding adds its own
    # set_attribute calls.
    assert recorder.entered is True
    # No raw description text leaked into attributes.
    for _, value in recorder.attrs.items():
        assert raw_desc_a not in str(value)
        assert raw_desc_b not in str(value)
    # At least one finding-related hashed attribute is present.
    hashed_attrs = {k for k in recorder.attrs if "hash" in k}
    assert hashed_attrs


# ---------------------------------------------------------------------------
# Apply-side persistence tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_persists_finding_rows() -> None:
    """The apply handler inserts one dream_conflicts row per finding and
    issues zero UPDATE/DELETE against ``relationships``."""
    ns = uuid4()
    rel_a, rel_b = uuid4(), uuid4()
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_CONTRADICTION_DETECT,
        inputs=({"namespace_id": str(ns)},),
        outputs=(
            {
                "relationship_a_id": str(rel_a),
                "relationship_b_id": str(rel_b),
                "source_entity_id": str(uuid4()),
                "target_entity_id": str(uuid4()),
                "relationship_type": "WORKS_AT",
                "similarity": 0.12,
                "contradicting_keys": ["status"],
                "reason": "both",
                "description_a_hash": "deadbeef",
                "description_b_hash": "cafebabe",
            },
        ),
        decision="audit_complete",
        namespace_id=ns,
    )

    session = _FakeSession()
    undo = await apply_vectorcypher_contradiction_detect(op, coordinator=None, session=session)

    # All executes must target dream_conflicts; nothing may touch relationships.
    for sql, _ in session.executed:
        upper = sql.upper()
        assert "DREAM_CONFLICTS" in upper
        assert not (upper.lstrip().startswith("UPDATE") and "RELATIONSHIPS" in upper)
        assert not (upper.lstrip().startswith("DELETE") and "RELATIONSHIPS" in upper)

    # Undo carries the inserted finding ids so revert can delete them.
    assert isinstance(undo, UndoRecord)
    assert undo.op_type == OpKind.VECTORCYPHER_CONTRADICTION_DETECT.value
    findings_in_undo = undo.before.get("findings")
    assert findings_in_undo and len(findings_in_undo) == 1
    entry = findings_in_undo[0]
    assert entry["relationship_a_id"] == str(rel_a)
    assert entry["relationship_b_id"] == str(rel_b)


@pytest.mark.asyncio
async def test_apply_idempotent_on_replay() -> None:
    """Re-applying the same op MUST NOT create duplicate findings —
    the persist statement uses ON CONFLICT DO NOTHING on the canonical
    (namespace_id, relationship_a_id, relationship_b_id) tuple."""
    ns = uuid4()
    rel_a, rel_b = uuid4(), uuid4()
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_CONTRADICTION_DETECT,
        inputs=({"namespace_id": str(ns)},),
        outputs=(
            {
                "relationship_a_id": str(rel_a),
                "relationship_b_id": str(rel_b),
                "source_entity_id": str(uuid4()),
                "target_entity_id": str(uuid4()),
                "relationship_type": "WORKS_AT",
                "similarity": 0.1,
                "contradicting_keys": [],
                "reason": "low_similarity",
                "description_a_hash": "11111111",
                "description_b_hash": "22222222",
            },
        ),
        decision="audit_complete",
        namespace_id=ns,
    )
    session = _FakeSession()

    await apply_vectorcypher_contradiction_detect(op, coordinator=None, session=session)

    inserts = [sql for sql, _ in session.executed if "INSERT" in sql.upper()]
    assert inserts, "expected at least one INSERT"
    for sql in inserts:
        assert "ON CONFLICT" in sql.upper()
        assert "DO NOTHING" in sql.upper()


@pytest.mark.asyncio
async def test_apply_empty_outputs_returns_noop_undo() -> None:
    """An op with no findings produces a no-op apply and an empty undo."""
    ns = uuid4()
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_CONTRADICTION_DETECT,
        inputs=({"namespace_id": str(ns)},),
        outputs=(),
        decision="audit_complete",
        namespace_id=ns,
    )
    session = _FakeSession()

    undo = await apply_vectorcypher_contradiction_detect(op, coordinator=None, session=session)

    assert session.executed == []
    assert undo.before == {"findings": []}


# ---------------------------------------------------------------------------
# OpKind / config wiring tests
# ---------------------------------------------------------------------------


def test_opkind_constant_exists() -> None:
    assert OpKind.VECTORCYPHER_CONTRADICTION_DETECT.value == "vectorcypher_contradiction_detect"


def test_dream_config_has_toggles() -> None:
    from khora.dream.config import DreamConfig

    cfg = DreamConfig()
    assert cfg.contradiction_detect_enabled is False
    assert cfg.contradiction_detect_similarity_threshold == 0.5


def test_apply_handler_is_registered() -> None:
    from khora.dream.engines.registry import get_apply_handler

    handler = get_apply_handler(OpKind.VECTORCYPHER_CONTRADICTION_DETECT)
    assert handler is not None
    assert handler is apply_vectorcypher_contradiction_detect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SpanRecorder:
    """Records every set_attribute call so we can assert against attrs."""

    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}
        self.entered = False

    def __enter__(self) -> _SpanRecorder:
        self.entered = True
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value
