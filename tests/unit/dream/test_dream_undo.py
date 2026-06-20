"""Unit tests for the ``kb.dream_undo`` public API (#667, Phase 4.1).

Round-trip:

  1. Apply a synthetic dedupe op against a ``_FakeSession`` (the same
     fixture style the dedupe-apply tests use).
  2. Persist the resulting :class:`UndoRecord` to a ``dream-undo/1``
     JSON file under a temp ``base_dir``.
  3. Call :func:`khora.dream.api.dream_undo` and assert it issues the
     reverse SQL: clears the absorbed entity's tombstone, repoints each
     rewritten relationship, and clears the bi-temporal invalidation
     on each self-loop.
  4. Replay the undo — assert it returns ``False`` (idempotent).

Unknown / unsupported op ids return ``False`` without raising.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.dream.api import _locate_undo_op, dream_undo
from khora.dream.engines.vectorcypher.dedupe_entities import (
    apply_vectorcypher_dedupe_entities,
    reverse_vectorcypher_dedupe_entities,
)
from khora.dream.plan import DreamOp, OpKind

# ---------------------------------------------------------------------------
# Fakes (mirror the dedupe-apply fixture style)
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeResult:
    def __init__(self, rows: list[Any], rowcount: int = 0) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def __iter__(self):
        return iter(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self) -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.relationship_rows: dict[UUID, list[_FakeRow]] = {}
        # Simulated row state for the reverse path: every UPDATE returns
        # rowcount=1 by default so the handler treats it as "row touched".
        self.next_update_rowcount: int = 1

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        text_str = str(stmt)
        params = params or {}
        self.executed.append((text_str, params))
        upper = text_str.lstrip().upper()
        if upper.startswith("SELECT") and "RELATIONSHIPS" in upper:
            aid = params.get("aid")
            key = aid if isinstance(aid, UUID) else None
            if not isinstance(aid, UUID):
                try:
                    key = UUID(str(aid))
                except (TypeError, ValueError):
                    key = None
            return _FakeResult(self.relationship_rows.get(key, []) if key is not None else [])
        return _FakeResult([], rowcount=self.next_update_rowcount)


class _FakeCoordinator:
    """Marker only — apply / reverse write via session."""

    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def transaction(self) -> Any:
        return _FakeTxnCtx(self._session)


class _FakeTxnCtx:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return SimpleNamespace(session=self._session)

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeKB:
    def __init__(self, session: _FakeSession) -> None:
        self.storage = _FakeCoordinator(session)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dedupe_op(*, canonical: UUID, absorbed: UUID, similarity: float | None = None) -> DreamOp:
    merge_entry: dict[str, Any] = {
        "canonical_id": str(canonical),
        "absorbed_id": str(absorbed),
    }
    if similarity is not None:
        merge_entry["similarity_score"] = similarity
        merge_entry["canonical_name"] = "Canonical"
        merge_entry["absorbed_name"] = "Variant"
        merge_entry["entity_type"] = "PERSON"
    return DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        inputs=(),
        outputs=({"merges": [merge_entry]},),
        decision="planned",
        rationale="dedupe",
        started_at=datetime.now(UTC),
        duration_ms=0.1,
        namespace_id=uuid4(),
    )


def _write_undo_file(
    *,
    base_dir: Path,
    namespace_id: UUID,
    run_id: UUID,
    op_id: UUID,
    op_type: str,
    before: dict[str, Any],
) -> Path:
    date_dir = base_dir / str(namespace_id) / datetime.now(UTC).strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "dream-undo/1",
        "run_id": str(run_id),
        "namespace_id": str(namespace_id),
        "started_at": datetime.now(UTC).isoformat(),
        "ops": [
            {
                "op_id": str(op_id),
                "op_type": op_type,
                "before": before,
                "applied_at": datetime.now(UTC).isoformat(),
            }
        ],
    }
    path = date_dir / f"{run_id}.undo.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_undo_restores_dedupe_entity_and_relationships(tmp_path: Path) -> None:
    """End-to-end: apply → dream_undo → tombstone cleared + edges re-pointed."""
    canonical = uuid4()
    absorbed = uuid4()
    other = uuid4()
    rel_id = uuid4()

    apply_session = _FakeSession()
    apply_session.relationship_rows[absorbed] = [
        _FakeRow(
            id=rel_id,
            source_entity_id=absorbed,
            target_entity_id=other,
            relationship_type="KNOWS",
        ),
    ]
    op = _build_dedupe_op(canonical=canonical, absorbed=absorbed)

    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=None,  # apply ignores coordinator
        session=apply_session,
    )

    # Persist the run's undo file.
    run_id = uuid4()
    namespace_id = uuid4()
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=namespace_id,
        run_id=run_id,
        op_id=op.op_id,
        op_type=str(op.op_type),
        before=undo.before,
    )

    # Now reverse via the public API on a fresh session.
    reverse_session = _FakeSession()
    kb = _FakeKB(reverse_session)
    ok = await dream_undo(kb, op.op_id, base_dir=tmp_path)
    assert ok is True

    # The reverse path must have issued:
    #   - UPDATE entities ... valid_until = NULL (clear tombstone)
    #   - UPDATE relationships SET source/target = absorbed (re-point)
    sql_blob = " | ".join(s.upper() for s, _ in reverse_session.executed)
    assert "VALID_UNTIL = NULL" in sql_blob
    assert "UPDATE RELATIONSHIPS" in sql_blob

    # Re-pointed back to the absorbed endpoint.
    rewrite_calls = [
        p
        for s, p in reverse_session.executed
        if "UPDATE RELATIONSHIPS" in s.upper() and "SOURCE_ENTITY_ID" in s.upper()
    ]
    assert len(rewrite_calls) >= 1
    params = rewrite_calls[0]
    # source was absorbed before apply.
    assert params["src"] == absorbed
    assert params["tgt"] == other


@pytest.mark.asyncio
async def test_dream_undo_clears_self_loop_invalidation(tmp_path: Path) -> None:
    """A self-loop the apply invalidated must have invalidated_at cleared on undo."""
    canonical = uuid4()
    absorbed = uuid4()
    self_loop_id = uuid4()

    apply_session = _FakeSession()
    apply_session.relationship_rows[absorbed] = [
        _FakeRow(
            id=self_loop_id,
            source_entity_id=absorbed,
            target_entity_id=canonical,
            relationship_type="ALIAS_OF",
        ),
    ]
    op = _build_dedupe_op(canonical=canonical, absorbed=absorbed)
    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=None,
        session=apply_session,
    )
    assert undo.before["merges"][0]["self_loops_invalidated"]

    namespace_id = uuid4()
    run_id = uuid4()
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=namespace_id,
        run_id=run_id,
        op_id=op.op_id,
        op_type=str(op.op_type),
        before=undo.before,
    )

    reverse_session = _FakeSession()
    kb = _FakeKB(reverse_session)
    ok = await dream_undo(kb, op.op_id, base_dir=tmp_path)
    assert ok is True

    sql_blob = " | ".join(s.upper() for s, _ in reverse_session.executed)
    assert "INVALIDATED_AT = NULL" in sql_blob
    assert "INVALIDATED_BY = NULL" in sql_blob


@pytest.mark.asyncio
async def test_dream_undo_is_idempotent(tmp_path: Path) -> None:
    """Re-undoing returns False once the live system has no matching rows."""
    canonical = uuid4()
    absorbed = uuid4()

    apply_session = _FakeSession()
    apply_session.relationship_rows[absorbed] = []
    op = _build_dedupe_op(canonical=canonical, absorbed=absorbed)
    undo = await apply_vectorcypher_dedupe_entities(
        op,
        coordinator=None,
        session=apply_session,
    )

    namespace_id = uuid4()
    run_id = uuid4()
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=namespace_id,
        run_id=run_id,
        op_id=op.op_id,
        op_type=str(op.op_type),
        before=undo.before,
    )

    # Second session — the live DB has nothing tombstoned / nothing to
    # restore, so every UPDATE returns rowcount=0.
    reverse_session = _FakeSession()
    reverse_session.next_update_rowcount = 0
    kb = _FakeKB(reverse_session)
    ok = await dream_undo(kb, op.op_id, base_dir=tmp_path)
    assert ok is False


@pytest.mark.asyncio
async def test_dream_undo_unknown_op_id_returns_false(tmp_path: Path) -> None:
    reverse_session = _FakeSession()
    kb = _FakeKB(reverse_session)
    ok = await dream_undo(kb, uuid4(), base_dir=tmp_path)
    assert ok is False
    # No SQL fired.
    assert reverse_session.executed == []


@pytest.mark.asyncio
async def test_dream_undo_skips_deferred_merges(tmp_path: Path) -> None:
    """A verifier-deferred merge (applied=False) must be a no-op on undo."""
    canonical = uuid4()
    absorbed = uuid4()
    namespace_id = uuid4()
    run_id = uuid4()
    op_id = uuid4()

    deferred_before = {
        "merges": [
            {
                "canonical_id": str(canonical),
                "absorbed_id": str(absorbed),
                "previous_relationships": [],
                "self_loops_invalidated": [],
                "verifier": {"decision": "defer", "rationale": "judges disagreed"},
                "applied": False,
            }
        ]
    }
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=namespace_id,
        run_id=run_id,
        op_id=op_id,
        op_type=str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES),
        before=deferred_before,
    )

    reverse_session = _FakeSession()
    kb = _FakeKB(reverse_session)
    ok = await dream_undo(kb, op_id, base_dir=tmp_path)
    assert ok is False
    # No UPDATE issued for a deferred merge.
    update_sqls = [s for s, _ in reverse_session.executed if s.lstrip().upper().startswith("UPDATE")]
    assert update_sqls == []


@pytest.mark.asyncio
async def test_dream_undo_unsupported_op_type_returns_false(tmp_path: Path) -> None:
    namespace_id = uuid4()
    run_id = uuid4()
    op_id = uuid4()
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=namespace_id,
        run_id=run_id,
        op_id=op_id,
        op_type="some_future_op_kind",
        before={"merges": []},
    )

    reverse_session = _FakeSession()
    kb = _FakeKB(reverse_session)
    ok = await dream_undo(kb, op_id, base_dir=tmp_path)
    assert ok is False


def test_locate_undo_op_walks_layout(tmp_path: Path) -> None:
    """Discovery layer finds the right op record in a multi-run tree."""
    namespace_id = uuid4()
    other_run = uuid4()
    target_run = uuid4()
    target_op = uuid4()
    other_op = uuid4()

    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=namespace_id,
        run_id=other_run,
        op_id=other_op,
        op_type=str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES),
        before={"merges": []},
    )
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=namespace_id,
        run_id=target_run,
        op_id=target_op,
        op_type=str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES),
        before={"merges": [{"canonical_id": str(uuid4()), "absorbed_id": str(uuid4()), "applied": False}]},
    )

    found = _locate_undo_op(target_op, base_dir=tmp_path)
    assert found is not None
    assert UUID(found["op_id"]) == target_op
    assert _locate_undo_op(uuid4(), base_dir=tmp_path) is None


@pytest.mark.asyncio
async def test_reverse_handler_skips_missing_session_rows() -> None:
    """Direct test of the reverse handler when previous_relationships is empty."""
    session = _FakeSession()
    session.next_update_rowcount = 0
    op_id = uuid4()
    undo_op = {
        "op_id": str(op_id),
        "op_type": str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES),
        "before": {
            "merges": [
                {
                    "canonical_id": str(uuid4()),
                    "absorbed_id": str(uuid4()),
                    "previous_relationships": [],
                    "self_loops_invalidated": [],
                    "applied": True,
                }
            ]
        },
    }
    ok = await reverse_vectorcypher_dedupe_entities(undo_op, session=session)
    assert ok is False


# ---------------------------------------------------------------------------
# Graph reverse mirror (#1275)
# ---------------------------------------------------------------------------


class _FakeRestoreGraph:
    """Records the #1275 reverse verb calls; advertises the dedupe op kind."""

    def __init__(self, *, fail: bool = False) -> None:
        self.restored_entities: list[Any] = []
        self.restored_relationships: list[Any] = []
        self.restored_endpoints: list[Any] = []
        self._fail = fail

    def supports_dream_mirror(self):  # noqa: ANN202
        return frozenset({OpKind.VECTORCYPHER_DEDUPE_ENTITIES})

    async def restore_entities_batch(self, entity_ids, *, namespace_id):  # noqa: ANN001, ANN202
        if self._fail:
            raise RuntimeError("neo4j restore boom")
        self.restored_entities.append(list(entity_ids))
        return len(entity_ids)

    async def restore_relationships_batch(self, relationship_ids, *, namespace_id):  # noqa: ANN001, ANN202
        self.restored_relationships.append(list(relationship_ids))
        return len(relationship_ids)

    async def restore_relationship_endpoints_batch(self, rewrites, *, namespace_id):  # noqa: ANN001, ANN202
        self.restored_endpoints.append(list(rewrites))
        return len(rewrites)


class _GraphCoordinator(_FakeCoordinator):
    def __init__(self, session: _FakeSession, graph: Any) -> None:
        super().__init__(session)
        self._graph = graph

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        return namespace_id


class _GraphKB:
    def __init__(self, session: _FakeSession, graph: Any) -> None:
        self.storage = _GraphCoordinator(session, graph)


@pytest.mark.asyncio
async def test_dream_undo_reverses_graph_mirror(tmp_path: Path) -> None:
    """When a graph backend is present, dream_undo also fires the reverse verbs."""
    canonical = uuid4()
    absorbed = uuid4()
    neighbor = uuid4()
    rel_id = uuid4()

    apply_session = _FakeSession()
    apply_session.relationship_rows[absorbed] = [
        _FakeRow(id=rel_id, source_entity_id=absorbed, target_entity_id=neighbor, relationship_type="SUPPLIES"),
    ]
    op = _build_dedupe_op(canonical=canonical, absorbed=absorbed)
    undo = await apply_vectorcypher_dedupe_entities(op, coordinator=None, session=apply_session)

    namespace_id = uuid4()
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=namespace_id,
        run_id=uuid4(),
        op_id=op.op_id,
        op_type=str(op.op_type),
        before=undo.before,
    )

    graph = _FakeRestoreGraph()
    kb = _GraphKB(_FakeSession(), graph)
    ok = await dream_undo(kb, op.op_id, base_dir=tmp_path)
    assert ok is True
    # The absorbed entity is un-retired and the incident edge re-pointed back.
    assert graph.restored_entities == [[absorbed]]
    assert graph.restored_endpoints
    assert graph.restored_endpoints[0][0]["relationship_id"] == str(rel_id)
    assert graph.restored_endpoints[0][0]["source_entity_id"] == str(absorbed)


@pytest.mark.asyncio
async def test_dream_undo_graph_failure_does_not_break_pg_revert(tmp_path: Path) -> None:
    """A reverse-mirror failure records a degradation but the PG revert (True) stands."""
    canonical = uuid4()
    absorbed = uuid4()
    neighbor = uuid4()
    rel_id = uuid4()

    apply_session = _FakeSession()
    apply_session.relationship_rows[absorbed] = [
        _FakeRow(id=rel_id, source_entity_id=absorbed, target_entity_id=neighbor, relationship_type="SUPPLIES"),
    ]
    op = _build_dedupe_op(canonical=canonical, absorbed=absorbed)
    undo = await apply_vectorcypher_dedupe_entities(op, coordinator=None, session=apply_session)

    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=uuid4(),
        run_id=uuid4(),
        op_id=op.op_id,
        op_type=str(op.op_type),
        before=undo.before,
    )

    graph = _FakeRestoreGraph(fail=True)
    kb = _GraphKB(_FakeSession(), graph)
    # The graph reverse raises, but dream_undo returns the PG-revert result and
    # does not propagate the exception (ADR-001: degrade, never half-fail-loudly).
    ok = await dream_undo(kb, op.op_id, base_dir=tmp_path)
    assert ok is True
