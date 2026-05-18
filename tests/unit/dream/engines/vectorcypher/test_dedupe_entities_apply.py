"""Apply-mode unit tests for dedupe_entities handler (#668).

Covers the Phase 4 mutation path:

  * Soft-deletes the absorbed entity by stamping ``valid_until=NOW()``.
  * Rewrites every ``relationships`` row to point at the canonical
    entity_id.
  * Self-loops (an edge whose endpoints both rewrite to the canonical)
    are bi-temporally invalidated via ``invalidated_at=NOW()`` /
    ``invalidated_by=op_id``.
  * Re-keys the absorbed entity's pgvector row under the canonical id by
    deleting the absorbed row (canonical's vector unchanged).
  * Records every prior row in the :class:`UndoRecord` under the
    ``"merges"`` top-level key (never ``"chunk_id"``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.engines.vectorcypher.dedupe_entities import (
    apply_vectorcypher_dedupe_entities,
)
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRow:
    """Lightweight row whose attributes mirror SQLAlchemy ``Row``."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeSession:
    """Captures every SQL call and serves curated SELECT results.

    The handler issues three statement families:
      * ``SELECT * FROM relationships WHERE source_entity_id = :aid OR ...``
        — to enumerate edges that need rewriting.
      * ``UPDATE relationships SET ... WHERE id = :rid`` (one per edge).
      * ``UPDATE entities SET valid_until = ... WHERE id = :aid`` (soft-delete).
      * ``DELETE FROM entities ... `` is NOT issued — entities are
        tombstone-only.

    The test primes ``relationship_rows[absorbed_id]`` with the list of
    edge rows to be rewritten.
    """

    def __init__(self, dialect_name: str = "postgresql") -> None:
        self.dialect_name = dialect_name
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.relationship_rows: dict[UUID, list[_FakeRow]] = {}

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        text_str = str(stmt)
        params = params or {}
        self.executed.append((text_str, params))
        upper = text_str.lstrip().upper()
        if upper.startswith("SELECT") and "RELATIONSHIPS" in upper:
            aid = params.get("aid") or params.get("absorbed_id")
            try:
                key = aid if isinstance(aid, UUID) else UUID(str(aid))
            except (TypeError, ValueError):
                key = None
            rows = self.relationship_rows.get(key, [])
            return _Result(rows)
        return SimpleNamespace(rowcount=1)


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeCoordinator:
    """Marker only — apply handler writes via session."""


# ---------------------------------------------------------------------------
# Op builder
# ---------------------------------------------------------------------------


def _op_dedupe(*, canonical: UUID, absorbed: UUID, entity_type: str = "PERSON") -> DreamOp:
    """Build a planned dedupe op — apply consumes ``op.outputs["merges"]``."""
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        inputs=(
            {
                "op_type": "entity_merge",
                "entity_type": entity_type,
                "threshold": 0.9,
                "similarity_score": 0.95,
                "keep_id": str(canonical),
                "drop_ids": (str(absorbed),),
                "surviving_name": "Canonical",
            },
        ),
        outputs=(
            {
                "merges": [
                    {
                        "canonical_id": str(canonical),
                        "absorbed_id": str(absorbed),
                    }
                ],
            },
        ),
        decision="planned",
        rationale="dedupe",
        started_at=datetime.now(UTC),
        duration_ms=0.1,
        namespace_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_rewrites_edges_and_records_undo() -> None:
    canonical = uuid4()
    absorbed = uuid4()
    other = uuid4()
    rel_id = uuid4()

    session = _FakeSession(dialect_name="postgresql")
    session.relationship_rows[absorbed] = [
        _FakeRow(
            id=rel_id,
            source_entity_id=absorbed,
            target_entity_id=other,
            relationship_type="KNOWS",
        ),
    ]
    op = _op_dedupe(canonical=canonical, absorbed=absorbed)

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    assert "chunk_id" not in undo.before
    merges = undo.before["merges"]
    assert len(merges) == 1
    merge = merges[0]
    assert UUID(merge["absorbed_id"]) == absorbed
    assert UUID(merge["canonical_id"]) == canonical
    # Previous relationships captured for undo.
    assert len(merge["previous_relationships"]) == 1
    prev = merge["previous_relationships"][0]
    assert UUID(prev["id"]) == rel_id
    assert UUID(prev["source_entity_id"]) == absorbed
    assert UUID(prev["target_entity_id"]) == other
    # No self-loop in this scenario.
    assert merge["self_loops_invalidated"] == []

    # Verify the SQL footprint.
    sql_texts = " | ".join(s.upper() for s, _ in session.executed)
    assert "UPDATE RELATIONSHIPS" in sql_texts
    assert "UPDATE ENTITIES" in sql_texts  # soft-delete absorbed
    assert "VALID_UNTIL" in sql_texts


@pytest.mark.asyncio
async def test_self_loops_are_bi_temporally_invalidated() -> None:
    """An edge with both endpoints in {absorbed, canonical} becomes a self-loop after rewrite."""
    canonical = uuid4()
    absorbed = uuid4()
    self_loop_id = uuid4()

    session = _FakeSession(dialect_name="postgresql")
    session.relationship_rows[absorbed] = [
        _FakeRow(
            id=self_loop_id,
            source_entity_id=absorbed,
            target_entity_id=canonical,  # would become canonical -> canonical
            relationship_type="ALIAS_OF",
        ),
    ]
    op = _op_dedupe(canonical=canonical, absorbed=absorbed)

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    merge = undo.before["merges"][0]
    self_loops = merge["self_loops_invalidated"]
    assert len(self_loops) == 1
    assert UUID(self_loops[0]) == self_loop_id

    # The self-loop must trip an UPDATE that sets invalidated_at /
    # invalidated_by — not a rewrite.
    sql_blob = " | ".join(s.upper() for s, _ in session.executed)
    assert "INVALIDATED_AT" in sql_blob
    assert "INVALIDATED_BY" in sql_blob


@pytest.mark.asyncio
async def test_absorbed_entity_is_soft_deleted_via_valid_until() -> None:
    """The absorbed entity row must be soft-deleted via ``valid_until``, never hard-deleted."""
    canonical = uuid4()
    absorbed = uuid4()
    session = _FakeSession(dialect_name="postgresql")
    op = _op_dedupe(canonical=canonical, absorbed=absorbed)

    await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    sql_blob = " | ".join(s.upper() for s, _ in session.executed)
    # Soft-delete path, not hard-delete.
    assert "UPDATE ENTITIES" in sql_blob
    assert "VALID_UNTIL" in sql_blob
    assert "DELETE FROM ENTITIES" not in sql_blob


@pytest.mark.asyncio
async def test_idempotent_replay_no_extra_rewrites() -> None:
    """If absorbed is already soft-deleted and has no live edges, replay is a noop."""
    canonical = uuid4()
    absorbed = uuid4()
    session = _FakeSession(dialect_name="postgresql")
    # No relationship rows returned — already collapsed.
    session.relationship_rows[absorbed] = []
    op = _op_dedupe(canonical=canonical, absorbed=absorbed)

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert isinstance(undo, UndoRecord)
    merge = undo.before["merges"][0]
    assert merge["previous_relationships"] == []
    assert merge["self_loops_invalidated"] == []
    # Soft-delete UPDATE still fires (idempotent UPDATE) — but no edge rewrites.
    rewrite_updates = [s for s, _ in session.executed if "UPDATE RELATIONSHIPS" in s.upper()]
    assert rewrite_updates == []


@pytest.mark.asyncio
async def test_handler_does_not_touch_documents_table() -> None:
    canonical = uuid4()
    absorbed = uuid4()
    session = _FakeSession(dialect_name="postgresql")
    session.relationship_rows[absorbed] = [
        _FakeRow(id=uuid4(), source_entity_id=absorbed, target_entity_id=uuid4(), relationship_type="X")
    ]
    op = _op_dedupe(canonical=canonical, absorbed=absorbed)

    await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    for sql, _ in session.executed:
        upper = sql.upper()
        # CHUNKS is also off-limits — temporal back-pointer.
        assert "DOCUMENTS" not in upper
        assert "FROM CHUNKS" not in upper
        assert "UPDATE CHUNKS" not in upper


@pytest.mark.asyncio
async def test_undo_record_carries_no_chunk_id_top_level_key() -> None:
    """Safety floor: the orchestrator's _assert_no_chunk_id_mutation must accept the undo."""
    canonical = uuid4()
    absorbed = uuid4()
    session = _FakeSession(dialect_name="postgresql")
    session.relationship_rows[absorbed] = []
    op = _op_dedupe(canonical=canonical, absorbed=absorbed)

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    assert "chunk_id" not in undo.before


# ---------------------------------------------------------------------------
# Phase 4.1 — borderline-merge verifier gate (#667)
# ---------------------------------------------------------------------------


def _op_dedupe_with_score(
    *,
    canonical: UUID,
    absorbed: UUID,
    similarity_score: float,
    canonical_name: str = "Canonical",
    absorbed_name: str = "Variant",
    entity_type: str = "PERSON",
) -> DreamOp:
    """Build an op whose merge entry carries the similarity_score the verifier reads."""
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        inputs=(),
        outputs=(
            {
                "merges": [
                    {
                        "canonical_id": str(canonical),
                        "absorbed_id": str(absorbed),
                        "similarity_score": similarity_score,
                        "canonical_name": canonical_name,
                        "absorbed_name": absorbed_name,
                        "entity_type": entity_type,
                    }
                ],
            },
        ),
        decision="planned",
        rationale="dedupe",
        started_at=datetime.now(UTC),
        duration_ms=0.1,
        namespace_id=uuid4(),
    )


class _RecordingVerifier:
    """Fake two-LLM judge that records every call and returns canned verdicts."""

    def __init__(self, decision: str, *, rationale: str = "stubbed") -> None:
        from khora.dream.engines.vectorcypher.verifier import JudgeResult, VerifierVerdict

        self.decision = decision
        self.rationale = rationale
        self.calls: list[Any] = []
        self._JudgeResult = JudgeResult
        self._VerifierVerdict = VerifierVerdict

    async def __call__(self, pair: Any, *, config: Any) -> Any:
        self.calls.append(pair)
        # Both judges agree by construction — the dispatcher would
        # already have collapsed to the decision passed in.
        verdict = self._VerifierVerdict(
            decision=self.decision if self.decision in ("merge", "keep_separate", "defer") else "defer",
            confidence=0.9,
            evidence_ids=[],
            rationale=self.rationale,
        )
        return self._JudgeResult(
            decision=self.decision,  # type: ignore[arg-type]
            verifier_verdict=verdict,
            auditor_verdict=verdict,
            rationale=self.rationale,
        )


@pytest.mark.asyncio
async def test_verifier_skipped_above_band() -> None:
    """Pair at similarity >= dedupe_verifier_band_high must not invoke the verifier."""
    from khora.dream.config import DreamConfig

    canonical = uuid4()
    absorbed = uuid4()
    session = _FakeSession(dialect_name="postgresql")
    session.relationship_rows[absorbed] = []
    op = _op_dedupe_with_score(canonical=canonical, absorbed=absorbed, similarity_score=0.97)
    verifier = _RecordingVerifier(decision="defer")  # would block the merge if called

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
        dream_config=DreamConfig(),
        verifier_fn=verifier,
    )

    # Verifier never ran.
    assert verifier.calls == []
    # Merge applied — soft-delete UPDATE fired.
    sql_blob = " | ".join(s.upper() for s, _ in session.executed)
    assert "VALID_UNTIL" in sql_blob
    # Undo record has applied=True and no verifier payload.
    merge = undo.before["merges"][0]
    assert merge["applied"] is True
    assert "verifier" not in merge


@pytest.mark.asyncio
async def test_verifier_runs_in_band() -> None:
    """Pair at similarity inside [0.78, 0.95) goes through the verifier."""
    from khora.dream.config import DreamConfig

    canonical = uuid4()
    absorbed = uuid4()
    session = _FakeSession(dialect_name="postgresql")
    session.relationship_rows[absorbed] = []
    op = _op_dedupe_with_score(canonical=canonical, absorbed=absorbed, similarity_score=0.85)
    verifier = _RecordingVerifier(decision="merge")

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
        dream_config=DreamConfig(),
        verifier_fn=verifier,
    )

    assert len(verifier.calls) == 1
    # Merge applied (verifier said merge).
    sql_blob = " | ".join(s.upper() for s, _ in session.executed)
    assert "VALID_UNTIL" in sql_blob
    merge = undo.before["merges"][0]
    assert merge["applied"] is True
    assert merge["verifier"]["decision"] == "merge"


@pytest.mark.asyncio
async def test_verifier_defer_blocks_merge() -> None:
    """Verifier decision != merge prevents the soft-delete UPDATE and edge rewrites."""
    from khora.dream.config import DreamConfig

    canonical = uuid4()
    absorbed = uuid4()
    other = uuid4()
    session = _FakeSession(dialect_name="postgresql")
    session.relationship_rows[absorbed] = [
        _FakeRow(id=uuid4(), source_entity_id=absorbed, target_entity_id=other, relationship_type="KNOWS"),
    ]
    op = _op_dedupe_with_score(canonical=canonical, absorbed=absorbed, similarity_score=0.85)
    verifier = _RecordingVerifier(decision="defer", rationale="judges disagreed")

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
        dream_config=DreamConfig(),
        verifier_fn=verifier,
    )

    # No mutation: no UPDATE statements on entities or relationships.
    update_sqls = [s for s, _ in session.executed if s.lstrip().upper().startswith("UPDATE")]
    assert update_sqls == []
    # Undo record records the defer verdict.
    merge = undo.before["merges"][0]
    assert merge["applied"] is False
    assert merge["verifier"]["decision"] == "defer"
    assert merge["verifier"]["rationale"] == "judges disagreed"


@pytest.mark.asyncio
async def test_dispatcher_defers_on_disagreement() -> None:
    """The real dispatcher returns 'defer' when judges disagree — verified at unit level."""
    from khora.dream.engines.vectorcypher.verifier import (
        VerifierVerdict,
        _combine_verdicts,
    )

    verifier = VerifierVerdict(decision="merge", confidence=0.9, evidence_ids=[], rationale="yes")
    auditor = VerifierVerdict(decision="keep_separate", confidence=0.9, evidence_ids=[], rationale="no")
    decision, rationale = _combine_verdicts(verifier, auditor, min_confidence=0.6)
    assert decision == "defer"
    assert "disagree" in rationale


@pytest.mark.asyncio
async def test_dispatcher_defers_on_low_confidence() -> None:
    """Both judges say merge but with low confidence → defer."""
    from khora.dream.engines.vectorcypher.verifier import (
        VerifierVerdict,
        _combine_verdicts,
    )

    verifier = VerifierVerdict(decision="merge", confidence=0.4, evidence_ids=[], rationale="weak")
    auditor = VerifierVerdict(decision="merge", confidence=0.55, evidence_ids=[], rationale="weak")
    decision, _ = _combine_verdicts(verifier, auditor, min_confidence=0.6)
    assert decision == "defer"


def test_verifier_parses_fenced_json() -> None:
    """Models that wrap JSON in ``` fences still parse cleanly."""
    from khora.dream.engines.vectorcypher.verifier import _parse_verdict

    raw = (
        "```json\n"
        '{"decision": "merge", "confidence": 0.92, "evidence_ids": [], '
        '"rationale": "same person, alt spelling"}\n'
        "```"
    )
    verdict = _parse_verdict(raw)
    assert verdict is not None
    assert verdict.decision == "merge"
    assert verdict.confidence == 0.92


def test_verifier_returns_none_on_garbage() -> None:
    from khora.dream.engines.vectorcypher.verifier import _parse_verdict

    assert _parse_verdict("") is None
    assert _parse_verdict("not json at all") is None
    # Wrong schema (missing decision)
    assert _parse_verdict('{"confidence": 0.9}') is None
    # Out-of-range confidence rejected by Pydantic
    assert _parse_verdict('{"decision": "merge", "confidence": 1.5}') is None
    # Invalid decision value
    assert _parse_verdict('{"decision": "yes", "confidence": 0.9}') is None


@pytest.mark.asyncio
async def test_multi_merge_op_handles_every_pair() -> None:
    """An op whose outputs carry N merges produces N undo entries."""
    canonical_a = uuid4()
    absorbed_a = uuid4()
    canonical_b = uuid4()
    absorbed_b = uuid4()

    session = _FakeSession(dialect_name="postgresql")
    session.relationship_rows[absorbed_a] = []
    session.relationship_rows[absorbed_b] = []

    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        inputs=(),
        outputs=(
            {
                "merges": [
                    {"canonical_id": str(canonical_a), "absorbed_id": str(absorbed_a)},
                    {"canonical_id": str(canonical_b), "absorbed_id": str(absorbed_b)},
                ]
            },
        ),
        decision="planned",
        rationale="batch",
        started_at=datetime.now(UTC),
        duration_ms=0.1,
        namespace_id=uuid4(),
    )

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=_FakeCoordinator(),
        session=session,
    )

    merges = undo.before["merges"]
    absorbed_in_undo = {UUID(m["absorbed_id"]) for m in merges}
    assert absorbed_in_undo == {absorbed_a, absorbed_b}
