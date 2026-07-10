"""Unit tests for the GraphBackend bi-temporal dream verb family (#1271).

Phase-2 foundation: the four verbs dream-apply will mirror to the graph,
plus the ``supports_dream_mirror()`` capability probe. This PR adds the
verb seam + Neo4j native impl only; the mirror wiring into the dream
orchestrator is #1272.

The Neo4j tests are mock-driven (assert the generated Cypher shape +
parameters), matching the existing ``test_neo4j_coverage`` harness. No
real Neo4j is started.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import CommunityNode
from khora.dream.exceptions import DreamBackendUnsupported
from khora.dream.plan import OpKind
from khora.storage.backends.mixins import GraphBackendBase
from khora.storage.backends.neo4j import Neo4jBackend

# ---------------------------------------------------------------------------
# Mock harness — mirrors test_neo4j_coverage._backend_with_session_mock but
# captures the Cypher + params each execute_write unit-of-work passes to
# tx.run, so the tests can assert the generated Cypher shape.
# ---------------------------------------------------------------------------


def _backend_with_write_capture(single: dict[str, Any] | None) -> tuple[Neo4jBackend, MagicMock]:
    """Build a backend whose ``_session().execute_write`` runs the unit-of-work
    against a fake transaction. ``tx.run(...).single()`` returns ``single``.

    Returns ``(backend, tx)`` so callers can assert the Cypher / params passed
    to ``tx.run``.
    """
    result = MagicMock()
    result.single = AsyncMock(return_value=single)
    result.data = AsyncMock(return_value=[single] if single is not None else [])
    tx = MagicMock()
    tx.run = AsyncMock(return_value=result)

    session = AsyncMock()

    async def _execute_write(work: Any, *args: Any, **kwargs: Any) -> Any:
        return await work(tx, *args, **kwargs)

    session.execute_write = AsyncMock(side_effect=_execute_write)

    driver = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    backend = Neo4jBackend.from_driver(driver, query_timeout=1.0)
    return backend, tx


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSupportsDreamMirror:
    def test_neo4j_advertises_all_four_verbs(self) -> None:
        backend = Neo4jBackend("bolt://localhost:7687")
        caps = backend.supports_dream_mirror()
        assert isinstance(caps, frozenset)
        assert OpKind.VECTORCYPHER_PRUNE_EDGES in caps
        assert OpKind.VECTORCYPHER_DEDUPE_ENTITIES in caps
        assert OpKind.VECTORCYPHER_NORMALIZE_SCHEMA in caps
        # #1276: community materialization is also natively mirrored.
        assert OpKind.VECTORCYPHER_COMMUNITY_SUMMARY in caps

    def test_base_backend_advertises_nothing(self) -> None:
        base = GraphBackendBase()
        assert base.supports_dream_mirror() == frozenset()


# ---------------------------------------------------------------------------
# GraphBackendBase default — structured-skip contract (raises, never deletes)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGraphBackendBaseDefaultUnsupported:
    @pytest.mark.asyncio
    async def test_soft_invalidate_relationships_raises(self) -> None:
        base = GraphBackendBase()
        with pytest.raises(DreamBackendUnsupported):
            await base.soft_invalidate_relationships_batch(
                [uuid4()], namespace_id=uuid4(), invalidated_at=datetime.now(UTC)
            )

    @pytest.mark.asyncio
    async def test_soft_retire_entities_raises(self) -> None:
        base = GraphBackendBase()
        with pytest.raises(DreamBackendUnsupported):
            await base.soft_retire_entities_batch([uuid4()], namespace_id=uuid4(), retired_at=datetime.now(UTC))

    @pytest.mark.asyncio
    async def test_rewrite_relationship_endpoints_raises(self) -> None:
        base = GraphBackendBase()
        with pytest.raises(DreamBackendUnsupported):
            await base.rewrite_relationship_endpoints_batch(
                [{"relationship_id": uuid4(), "source_entity_id": uuid4(), "target_entity_id": uuid4()}],
                namespace_id=uuid4(),
                rewritten_at=datetime.now(UTC),
            )

    @pytest.mark.asyncio
    async def test_rename_types_raises(self) -> None:
        base = GraphBackendBase()
        with pytest.raises(DreamBackendUnsupported):
            await base.rename_types_batch(
                [{"old_type": "works for", "new_type": "WORKS_FOR"}],
                namespace_id=uuid4(),
            )

    @pytest.mark.asyncio
    async def test_restore_entities_raises(self) -> None:
        base = GraphBackendBase()
        with pytest.raises(DreamBackendUnsupported):
            await base.restore_entities_batch([uuid4()], namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_restore_relationships_raises(self) -> None:
        base = GraphBackendBase()
        with pytest.raises(DreamBackendUnsupported):
            await base.restore_relationships_batch([uuid4()], namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_restore_relationship_endpoints_raises(self) -> None:
        base = GraphBackendBase()
        with pytest.raises(DreamBackendUnsupported):
            await base.restore_relationship_endpoints_batch(
                [{"relationship_id": uuid4(), "source_entity_id": uuid4(), "target_entity_id": uuid4()}],
                namespace_id=uuid4(),
            )

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits_without_raising(self) -> None:
        """Empty batches are a no-op, not an unsupported error — callers can
        feed an empty plan op to any backend without tripping the gate."""
        base = GraphBackendBase()
        assert (
            await base.soft_invalidate_relationships_batch([], namespace_id=uuid4(), invalidated_at=datetime.now(UTC))
            == 0
        )
        assert await base.soft_retire_entities_batch([], namespace_id=uuid4(), retired_at=datetime.now(UTC)) == 0
        assert (
            await base.rewrite_relationship_endpoints_batch([], namespace_id=uuid4(), rewritten_at=datetime.now(UTC))
            == 0
        )
        assert await base.rename_types_batch([], namespace_id=uuid4()) == 0
        # Reverse verbs (#1275) share the same empty-short-circuit contract.
        assert await base.restore_entities_batch([], namespace_id=uuid4()) == 0
        assert await base.restore_relationships_batch([], namespace_id=uuid4()) == 0
        assert await base.restore_relationship_endpoints_batch([], namespace_id=uuid4()) == 0


# ---------------------------------------------------------------------------
# Neo4j: soft_invalidate_relationships_batch — stamp valid_until by id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jSoftInvalidateRelationships:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        backend, _tx = _backend_with_write_capture(single={"invalidated": 0})
        out = await backend.soft_invalidate_relationships_batch(
            [], namespace_id=uuid4(), invalidated_at=datetime.now(UTC)
        )
        assert out == 0

    @pytest.mark.asyncio
    async def test_stamps_valid_until_by_id_namespace_scoped(self) -> None:
        ns = uuid4()
        rel_id = uuid4()
        ts = datetime(2026, 6, 19, 12, 0, 0, tzinfo=UTC)
        backend, tx = _backend_with_write_capture(single={"invalidated": 1})

        out = await backend.soft_invalidate_relationships_batch([rel_id], namespace_id=ns, invalidated_at=ts)
        assert out == 1

        cypher = tx.run.await_args.args[0]
        kwargs = tx.run.await_args.kwargs
        # Soft-delete stamps valid_until (the column recall honors), not a delete.
        assert "valid_until" in cypher
        assert "DELETE" not in cypher.upper()
        # valid_until is stored native (#1472): the ISO-string param is coerced
        # with datetime() at the write site so the recall filter is index-friendly.
        assert "rel.valid_until = datetime($invalidated_at)" in cypher
        # Idempotent: only stamps edges not already invalidated.
        assert "valid_until IS NULL" in cypher
        # Namespace-scoped + matched by id. The param stays an ISO string; the
        # Cypher datetime() cast produces the native stored value.
        assert kwargs["namespace_id"] == str(ns)
        assert kwargs["invalidated_at"] == ts.isoformat()
        # ids threaded as a batch parameter (not interpolated).
        assert str(rel_id) in str(kwargs)


# ---------------------------------------------------------------------------
# Neo4j: soft_retire_entities_batch — :EntityVersion + [:SUPERSEDES] snapshot
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jSoftRetireEntities:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        backend, _tx = _backend_with_write_capture(single=None)
        out = await backend.soft_retire_entities_batch([], namespace_id=uuid4(), retired_at=datetime.now(UTC))
        assert out == 0

    @pytest.mark.asyncio
    async def test_snapshots_into_entity_version_and_supersedes(self) -> None:
        ns = uuid4()
        entity_id = uuid4()
        ts = datetime(2026, 6, 19, 12, 0, 0, tzinfo=UTC)
        backend, tx = _backend_with_write_capture(single={"id": str(entity_id)})

        out = await backend.soft_retire_entities_batch([entity_id], namespace_id=ns, retired_at=ts)
        assert out == 1

        cypher = tx.run.await_args.args[0]
        kwargs = tx.run.await_args.kwargs
        # Reuse the existing :EntityVersion / [:SUPERSEDES] Cypher shape.
        assert "EntityVersion" in cypher
        assert "SUPERSEDES" in cypher
        # Soft-delete stamps the bi-temporal columns the issue names.
        assert "valid_until" in cypher
        assert "version_valid_to" in cypher
        # valid_until is stored native (#1472); version_valid_to stays an ISO
        # string (its equality match with the snapshot must not cross types).
        assert "current.valid_until = datetime($retired_at)" in cypher
        assert "current.version_valid_to = $retired_at" in cypher
        # Never a hard delete.
        assert "DELETE" not in cypher.upper()
        # Idempotent replay guard: only retire still-live entities.
        assert "valid_until IS NULL" in cypher
        # Namespace-scoped + matched by id.
        assert kwargs["namespace_id"] == str(ns)
        assert kwargs["retired_at"] == ts.isoformat()
        assert str(entity_id) in str(kwargs)

    @pytest.mark.asyncio
    async def test_default_reason_is_dream_keyed_not_document_replaced(self) -> None:
        """The dream retirement reason must be distinct from the
        document-replace primitive's ``document_replaced`` reason."""
        ns = uuid4()
        entity_id = uuid4()
        backend, tx = _backend_with_write_capture(single={"id": str(entity_id)})
        await backend.soft_retire_entities_batch([entity_id], namespace_id=ns, retired_at=datetime.now(UTC))
        # The reason lands in the rows / params, not interpolated into Cypher.
        flat = str(tx.run.await_args.kwargs)
        assert "document_replaced" not in flat


# ---------------------------------------------------------------------------
# Neo4j: rewrite_relationship_endpoints_batch — re-point by id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jRewriteRelationshipEndpoints:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        backend, _tx = _backend_with_write_capture(single={"rewritten": 0})
        out = await backend.rewrite_relationship_endpoints_batch(
            [], namespace_id=uuid4(), rewritten_at=datetime.now(UTC)
        )
        assert out == 0

    @pytest.mark.asyncio
    async def test_repoints_endpoints_by_id(self) -> None:
        ns = uuid4()
        rel_id = uuid4()
        new_src = uuid4()
        new_tgt = uuid4()
        ts = datetime(2026, 6, 19, 12, 0, 0, tzinfo=UTC)
        backend, tx = _backend_with_write_capture(single={"rewritten": 1})

        out = await backend.rewrite_relationship_endpoints_batch(
            [
                {
                    "relationship_id": rel_id,
                    "source_entity_id": new_src,
                    "target_entity_id": new_tgt,
                    "relationship_type": "works for",
                }
            ],
            namespace_id=ns,
            rewritten_at=ts,
        )
        assert out == 1

        cypher = tx.run.await_args.args[0]
        kwargs = tx.run.await_args.kwargs
        # Re-point: the edge is detached from old endpoints and re-attached to
        # the new ones (Neo4j cannot rewrite endpoints in place — delete+create
        # the edge preserving properties is the idiom).
        assert kwargs["namespace_id"] == str(ns)
        # The relationship type is a Cypher label — sanitized + interpolated,
        # never $-parameterized.
        assert "WORKS_FOR" in cypher
        assert "works for" not in cypher
        # ids threaded as batch parameters.
        flat = str(kwargs)
        assert str(rel_id) in flat
        assert str(new_src) in flat
        assert str(new_tgt) in flat


# ---------------------------------------------------------------------------
# Neo4j: rename_types_batch — relabel edge type via the hard sanitizer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jRenameTypes:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        backend, _tx = _backend_with_write_capture(single={"renamed": 0})
        out = await backend.rename_types_batch([], namespace_id=uuid4())
        assert out == 0

    @pytest.mark.asyncio
    async def test_label_routed_through_sanitizer(self) -> None:
        """The new relationship_type is a Cypher edge label — it CANNOT be
        $-parameterized, so it must be interpolated through
        ``_sanitize_neo4j_label`` (Cypher-injection surface)."""
        ns = uuid4()
        backend, tx = _backend_with_write_capture(single={"renamed": 3})

        out = await backend.rename_types_batch(
            [{"old_type": "works for", "new_type": "manages; DROP"}],
            namespace_id=ns,
        )
        assert out == 3

        cypher = tx.run.await_args.args[0]
        kwargs = tx.run.await_args.kwargs
        # The sanitized labels are interpolated, never the raw input. The
        # injection payload "manages; DROP" collapses to a single safe label.
        assert "WORKS_FOR" in cypher
        assert "MANAGES__DROP" in cypher
        # No injection survives: the raw semicolon / space never reaches Cypher,
        # so the payload cannot break out of the edge-label position.
        assert "works for" not in cypher
        assert ";" not in cypher
        # Namespace-scoped.
        assert kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_old_type_also_sanitized_for_match(self) -> None:
        """Both ends of the rename route through the sanitizer — the old type
        is the MATCH label and is equally injection-prone."""
        ns = uuid4()
        backend, tx = _backend_with_write_capture(single={"renamed": 1})
        await backend.rename_types_batch(
            [{"old_type": "at-risk", "new_type": "blocked_by"}],
            namespace_id=ns,
        )
        cypher = tx.run.await_args.args[0]
        assert "AT_RISK" in cypher
        assert "BLOCKED_BY" in cypher


# ---------------------------------------------------------------------------
# Neo4j: community materialization (#1276) — MERGE :Community + [:HAS_MEMBER]
# ---------------------------------------------------------------------------


def _backend_with_read_capture(records: list[dict[str, Any]]) -> tuple[Neo4jBackend, MagicMock]:
    """Build a backend whose ``_session().run(...)`` returns ``records``.

    Returns ``(backend, session)`` so callers can assert the Cypher / params
    passed to ``session.run`` (the read path uses ``session.run`` directly, not
    ``execute_write``).
    """
    result = MagicMock()
    result.data = AsyncMock(return_value=records)
    session = AsyncMock()
    session.run = AsyncMock(return_value=result)

    driver = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    backend = Neo4jBackend.from_driver(driver, query_timeout=1.0)
    return backend, session


@pytest.mark.unit
class TestGraphBackendBaseCommunityDefault:
    @pytest.mark.asyncio
    async def test_materialize_raises_when_unsupported(self) -> None:
        base = GraphBackendBase()
        with pytest.raises(DreamBackendUnsupported):
            await base.materialize_communities_batch(
                [CommunityNode(summary="s")], namespace_id=uuid4(), materialized_at=datetime.now(UTC)
            )

    @pytest.mark.asyncio
    async def test_materialize_empty_short_circuits(self) -> None:
        base = GraphBackendBase()
        assert (
            await base.materialize_communities_batch([], namespace_id=uuid4(), materialized_at=datetime.now(UTC)) == 0
        )

    @pytest.mark.asyncio
    async def test_read_defaults_return_empty(self) -> None:
        base = GraphBackendBase()
        assert await base.get_communities(uuid4()) == []
        assert await base.get_entity_communities([uuid4()], namespace_id=uuid4()) == []


@pytest.mark.unit
class TestNeo4jMaterializeCommunities:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        backend, _tx = _backend_with_write_capture(single={"materialized": 0})
        out = await backend.materialize_communities_batch([], namespace_id=uuid4(), materialized_at=datetime.now(UTC))
        assert out == 0

    @pytest.mark.asyncio
    async def test_merges_community_node_and_member_edges(self) -> None:
        ns = uuid4()
        cid = uuid4()
        m1, m2 = uuid4(), uuid4()
        ts = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)
        backend, tx = _backend_with_write_capture(single={"materialized": 1})

        out = await backend.materialize_communities_batch(
            [CommunityNode(id=cid, summary="Alice + Bob", member_ids=[m1, m2])],
            namespace_id=ns,
            materialized_at=ts,
        )
        assert out == 1

        cypher = tx.run.await_args.args[0]
        kwargs = tx.run.await_args.kwargs
        # Idempotent: MERGE the :Community node (not CREATE) so a re-run is a no-op.
        assert "MERGE (com:Community" in cypher
        assert "CREATE (com:Community" not in cypher
        # HAS_MEMBER edges to member :Entity nodes (also MERGE for idempotence).
        assert "HAS_MEMBER" in cypher
        assert "MATCH (e:Entity" in cypher
        # Never a hard delete.
        assert "DELETE" not in cypher.upper()
        # Namespace-scoped + threaded as params (not interpolated).
        assert kwargs["namespace_id"] == str(ns)
        assert kwargs["materialized_at"] == ts.isoformat()
        flat = str(kwargs)
        assert str(cid) in flat
        assert str(m1) in flat
        assert str(m2) in flat


@pytest.mark.unit
class TestNeo4jReadCommunities:
    @pytest.mark.asyncio
    async def test_get_communities_maps_records(self) -> None:
        ns = uuid4()
        cid = uuid4()
        m1 = uuid4()
        backend, session = _backend_with_read_capture(
            [{"com": {"id": str(cid), "summary": "s", "summary_depth": 1, "member_ids": [str(m1)]}}]
        )
        out = await backend.get_communities(ns, limit=10)
        assert len(out) == 1
        assert out[0].id == cid
        assert out[0].summary == "s"
        assert out[0].member_ids == [m1]
        cypher = session.run.await_args.args[0]
        kwargs = session.run.await_args.kwargs
        assert "MATCH (com:Community" in cypher
        assert kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_entity_communities_empty_input_short_circuits(self) -> None:
        backend, session = _backend_with_read_capture([])
        out = await backend.get_entity_communities([], namespace_id=uuid4())
        assert out == []
        session.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_entity_communities_filters_by_member(self) -> None:
        ns = uuid4()
        cid = uuid4()
        e1 = uuid4()
        backend, session = _backend_with_read_capture(
            [{"com": {"id": str(cid), "summary": "s", "summary_depth": 1, "member_ids": [str(e1)]}}]
        )
        out = await backend.get_entity_communities([e1], namespace_id=ns)
        assert len(out) == 1
        assert out[0].id == cid
        cypher = session.run.await_args.args[0]
        kwargs = session.run.await_args.kwargs
        assert "HAS_MEMBER" in cypher
        assert "e.id IN $entity_ids" in cypher
        assert kwargs["entity_ids"] == [str(e1)]
        assert kwargs["namespace_id"] == str(ns)


# ---------------------------------------------------------------------------
# Neo4j reverse verbs (#1275) — graph-side undo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jRestoreVerbs:
    @pytest.mark.asyncio
    async def test_restore_entities_empty_short_circuits(self) -> None:
        backend, tx = _backend_with_write_capture(single=None)
        out = await backend.restore_entities_batch([], namespace_id=uuid4())
        assert out == 0
        tx.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_restore_entities_clears_valid_until_and_deletes_snapshot(self) -> None:
        ns = uuid4()
        eid = uuid4()
        backend, tx = _backend_with_write_capture(single={"restored": 1})
        out = await backend.restore_entities_batch([eid], namespace_id=ns)
        assert out == 1
        cypher = tx.run.await_args.args[0]
        kwargs = tx.run.await_args.kwargs
        assert "current.valid_until = NULL" in cypher
        assert "current.version_valid_to = NULL" in cypher
        # Idempotent-by-count: only still-retired nodes transition.
        assert "current.valid_until IS NOT NULL OR current.version_valid_to IS NOT NULL" in cypher
        # The :EntityVersion snapshot + [:SUPERSEDES] edge are removed, scoped to
        # THIS op's snapshot. Matched on version_valid_to (both ISO strings) - not
        # valid_until, which is now a native ZONED DATETIME (#1472) and would make
        # the equality a NULL-yielding cross-type comparison.
        assert "DETACH DELETE o" in cypher
        assert "EntityVersion" in cypher
        assert "old.version_valid_to = current.version_valid_to" in cypher
        assert kwargs["entity_ids"] == [str(eid)]
        assert kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_restore_relationships_empty_short_circuits(self) -> None:
        backend, tx = _backend_with_write_capture(single={"restored": 0})
        out = await backend.restore_relationships_batch([], namespace_id=uuid4())
        assert out == 0
        tx.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_restore_relationships_clears_valid_until(self) -> None:
        ns = uuid4()
        rel = uuid4()
        backend, tx = _backend_with_write_capture(single={"restored": 1})
        out = await backend.restore_relationships_batch([rel], namespace_id=ns)
        assert out == 1
        cypher = tx.run.await_args.args[0]
        kwargs = tx.run.await_args.kwargs
        assert "rel.valid_until = NULL" in cypher
        assert "rel.valid_until IS NOT NULL" in cypher
        assert kwargs["relationship_ids"] == [str(rel)]
        assert kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_restore_relationship_endpoints_empty_short_circuits(self) -> None:
        backend, tx = _backend_with_write_capture(single={"rewritten": 0})
        out = await backend.restore_relationship_endpoints_batch([], namespace_id=uuid4())
        assert out == 0
        tx.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_restore_relationship_endpoints_repoints_by_id(self) -> None:
        ns = uuid4()
        rel = uuid4()
        absorbed = uuid4()
        neighbor = uuid4()
        backend, tx = _backend_with_write_capture(single={"rewritten": 1})
        out = await backend.restore_relationship_endpoints_batch(
            [
                {
                    "relationship_id": str(rel),
                    "source_entity_id": str(absorbed),
                    "target_entity_id": str(neighbor),
                    "relationship_type": "SUPPLIES",
                }
            ],
            namespace_id=ns,
        )
        assert out == 1
        cypher = tx.run.await_args.args[0]
        kwargs = tx.run.await_args.kwargs
        # Sanitized label is a static literal; endpoints bind as params.
        assert ":SUPPLIES" in cypher
        assert kwargs["rows"][0]["relationship_id"] == str(rel)
        assert kwargs["rows"][0]["source_entity_id"] == str(absorbed)
        assert kwargs["namespace_id"] == str(ns)
