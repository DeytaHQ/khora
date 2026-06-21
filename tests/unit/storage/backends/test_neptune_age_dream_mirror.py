"""Unit tests for the Neptune + AGE soft-delete-ONLY dream mirror (#1279).

Phase-4 of the dream-on-graph umbrella (#1282). Neptune and AGE lack
entity-versioning primitives, so they mirror ONLY the flat ``valid_until``
SET-by-id used by ``prune_edges``. The entity-version (``dedupe_entities``)
and relabel (``normalize_schema``) op kinds are deliberately NOT advertised so
the dream orchestrator records a structured ``graph_mirror_unsupported_op_kind``
skip for them BEFORE any PG-committed verb runs (a clean pre-commit skip, not a
post-commit partial failure).

Neptune is AWS-only and cannot run locally, so it is covered by mock-driven unit
tests (assert the emitted openCypher + params). AGE needs an AGE-enabled
Postgres the repo's pgvector/pg17 image does not ship, so it is covered by
mock-driven unit tests that assert the emitted Cypher shape + the
one-auto-commit-unit batch behavior. NO real cluster / DB is started.

CRITICAL: neither backend can ``count_relationships`` (both raise
NotImplementedError), so convergence is asserted by id-set / live-set
(``valid_until IS NULL`` guard + returned affected counts), NEVER by a total
edge count.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import CommunityNode
from khora.dream.exceptions import DreamBackendUnsupported
from khora.dream.plan import OpKind
from khora.storage.backends.age import AGEBackend
from khora.storage.backends.neptune import NeptuneBackend

_NS = uuid4()
_TS = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


# ===========================================================================
# Neptune mock plumbing (session.run-based, mirrors test_neptune_coverage)
# ===========================================================================


def _make_session(single: dict[str, Any] | None) -> AsyncMock:
    result = MagicMock()
    result.single = AsyncMock(return_value=single)
    session = AsyncMock()
    session.run = AsyncMock(return_value=result)
    return session


def _neptune(session: AsyncMock) -> NeptuneBackend:
    backend = NeptuneBackend("bolt://cluster:8182")
    driver = MagicMock()

    @asynccontextmanager
    async def _session_ctx():  # type: ignore[no-untyped-def]
        yield session

    driver.session = MagicMock(side_effect=_session_ctx)
    backend._driver = driver
    return backend


# ===========================================================================
# AGE mock plumbing — patch _cypher (the single Cypher chokepoint) + session
# ===========================================================================


def _age_with_cypher_capture(per_call_return: int) -> tuple[AGEBackend, list[str], AsyncMock]:
    """Build an AGE backend whose ``_cypher`` records each Cypher string and
    returns ``[{"<col>": per_call_return}]`` so the affected-count sums.

    Returns ``(backend, captured_cypher_list, session)`` so a caller can assert
    the single-auto-commit-unit batch behavior on the session.
    """
    backend = AGEBackend("postgresql://x/y")

    # Mock the session factory: a session whose .begin() is an async ctx and
    # whose .execute() (the search_path SET) is a no-op.
    session = AsyncMock()

    @asynccontextmanager
    async def _begin_ctx():  # type: ignore[no-untyped-def]
        yield None

    session.begin = MagicMock(side_effect=_begin_ctx)
    session.execute = AsyncMock()

    @asynccontextmanager
    async def _session_ctx():  # type: ignore[no-untyped-def]
        yield session

    factory = MagicMock(side_effect=_session_ctx)
    backend._session_factory = factory  # type: ignore[assignment]

    captured: list[str] = []

    async def _fake_cypher(_session: Any, cypher_query: str, *, columns: list[str] | None = None) -> list[dict]:
        captured.append(cypher_query)
        col = (columns or ["v agtype"])[0].split()[0]
        return [{col: per_call_return}]

    backend._cypher = AsyncMock(side_effect=_fake_cypher)  # type: ignore[method-assign]
    return backend, captured, session


# ===========================================================================
# Capability probe — both advertise ONLY prune_edges
# ===========================================================================


@pytest.mark.unit
class TestSupportsDreamMirror:
    def test_neptune_advertises_flat_soft_delete_kinds(self) -> None:
        caps = NeptuneBackend("bolt://localhost:8182").supports_dream_mirror()
        assert isinstance(caps, frozenset)
        assert caps == frozenset({OpKind.VECTORCYPHER_PRUNE_EDGES, OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE})

    def test_age_advertises_flat_soft_delete_kinds(self) -> None:
        caps = AGEBackend("postgresql://x/y").supports_dream_mirror()
        assert isinstance(caps, frozenset)
        assert caps == frozenset({OpKind.VECTORCYPHER_PRUNE_EDGES, OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE})

    @pytest.mark.parametrize(
        "backend",
        [NeptuneBackend("bolt://localhost:8182"), AGEBackend("postgresql://x/y")],
    )
    def test_version_and_relabel_op_kinds_not_advertised(self, backend: Any) -> None:
        """Entity-version (dedupe) and relabel (normalize_schema) are explicitly
        UNSUPPORTED: not advertised, so the orchestrator records a structured
        skip for those op kinds rather than mirroring them."""
        caps = backend.supports_dream_mirror()
        assert OpKind.VECTORCYPHER_DEDUPE_ENTITIES not in caps
        assert OpKind.VECTORCYPHER_NORMALIZE_SCHEMA not in caps
        assert OpKind.VECTORCYPHER_COMMUNITY_SUMMARY not in caps


# ===========================================================================
# Unsupported verbs keep the raising default (version / relabel / community)
# ===========================================================================


@pytest.mark.unit
class TestUnsupportedVerbsRaise:
    @pytest.mark.parametrize(
        "backend",
        [NeptuneBackend("bolt://localhost:8182"), AGEBackend("postgresql://x/y")],
    )
    @pytest.mark.asyncio
    async def test_soft_retire_entities_raises(self, backend: Any) -> None:
        with pytest.raises(DreamBackendUnsupported):
            await backend.soft_retire_entities_batch([uuid4()], namespace_id=_NS, retired_at=_TS)

    @pytest.mark.parametrize(
        "backend",
        [NeptuneBackend("bolt://localhost:8182"), AGEBackend("postgresql://x/y")],
    )
    @pytest.mark.asyncio
    async def test_rewrite_endpoints_raises(self, backend: Any) -> None:
        with pytest.raises(DreamBackendUnsupported):
            await backend.rewrite_relationship_endpoints_batch(
                [{"relationship_id": uuid4(), "source_entity_id": uuid4(), "target_entity_id": uuid4()}],
                namespace_id=_NS,
                rewritten_at=_TS,
            )

    @pytest.mark.parametrize(
        "backend",
        [NeptuneBackend("bolt://localhost:8182"), AGEBackend("postgresql://x/y")],
    )
    @pytest.mark.asyncio
    async def test_rename_types_raises(self, backend: Any) -> None:
        with pytest.raises(DreamBackendUnsupported):
            await backend.rename_types_batch([{"old_type": "works for", "new_type": "WORKS_FOR"}], namespace_id=_NS)

    @pytest.mark.parametrize(
        "backend",
        [NeptuneBackend("bolt://localhost:8182"), AGEBackend("postgresql://x/y")],
    )
    @pytest.mark.asyncio
    async def test_restore_entities_raises(self, backend: Any) -> None:
        with pytest.raises(DreamBackendUnsupported):
            await backend.restore_entities_batch([uuid4()], namespace_id=_NS)

    @pytest.mark.parametrize(
        "backend",
        [NeptuneBackend("bolt://localhost:8182"), AGEBackend("postgresql://x/y")],
    )
    @pytest.mark.asyncio
    async def test_restore_endpoints_raises(self, backend: Any) -> None:
        with pytest.raises(DreamBackendUnsupported):
            await backend.restore_relationship_endpoints_batch(
                [{"relationship_id": uuid4(), "source_entity_id": uuid4(), "target_entity_id": uuid4()}],
                namespace_id=_NS,
            )

    @pytest.mark.parametrize(
        "backend",
        [NeptuneBackend("bolt://localhost:8182"), AGEBackend("postgresql://x/y")],
    )
    @pytest.mark.asyncio
    async def test_materialize_communities_raises(self, backend: Any) -> None:
        with pytest.raises(DreamBackendUnsupported):
            await backend.materialize_communities_batch(
                [CommunityNode(summary="s")], namespace_id=_NS, materialized_at=_TS
            )


# ===========================================================================
# Neptune: soft_invalidate_relationships_batch — flat valid_until SET by id
# ===========================================================================


@pytest.mark.unit
class TestNeptuneSoftInvalidate:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        session = _make_session(single={"invalidated": 0})
        backend = _neptune(session)
        out = await backend.soft_invalidate_relationships_batch([], namespace_id=_NS, invalidated_at=_TS)
        assert out == 0
        session.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stamps_valid_until_by_id_namespace_scoped(self) -> None:
        rel_id = uuid4()
        session = _make_session(single={"invalidated": 1})
        backend = _neptune(session)

        out = await backend.soft_invalidate_relationships_batch([rel_id], namespace_id=_NS, invalidated_at=_TS)
        assert out == 1

        cypher = session.run.await_args.args[0]
        kwargs = session.run.await_args.kwargs
        # Soft-delete stamps valid_until, never a hard delete.
        assert "valid_until" in cypher
        assert "DELETE" not in cypher.upper()
        # Idempotent replay guard: only edges not already invalidated.
        assert "valid_until IS NULL" in cypher
        # Namespace-scoped + matched by id, threaded as params (not interpolated).
        assert kwargs["namespace_id"] == str(_NS)
        assert kwargs["invalidated_at"] == _TS.isoformat()  # ISO string, matches create_relationship shape
        assert kwargs["relationship_ids"] == [str(rel_id)]


@pytest.mark.unit
class TestNeptuneRestoreRelationships:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        session = _make_session(single={"restored": 0})
        backend = _neptune(session)
        out = await backend.restore_relationships_batch([], namespace_id=_NS)
        assert out == 0
        session.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clears_valid_until_by_id(self) -> None:
        rel_id = uuid4()
        session = _make_session(single={"restored": 1})
        backend = _neptune(session)

        out = await backend.restore_relationships_batch([rel_id], namespace_id=_NS)
        assert out == 1

        cypher = session.run.await_args.args[0]
        kwargs = session.run.await_args.kwargs
        assert "valid_until = null" in cypher
        # updated_at is bumped on restore too (matches the Neo4j reference verb).
        assert "rel.updated_at = $restored_at" in cypher
        assert "restored_at" in kwargs
        # Idempotent: only edges still invalidated transition.
        assert "valid_until IS NOT NULL" in cypher
        assert "DELETE" not in cypher.upper()
        assert kwargs["namespace_id"] == str(_NS)
        assert kwargs["relationship_ids"] == [str(rel_id)]


# ===========================================================================
# AGE: soft_invalidate_relationships_batch — flat valid_until SET by id
# ===========================================================================


@pytest.mark.unit
class TestAGESoftInvalidate:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        backend, captured, _session = _age_with_cypher_capture(per_call_return=1)
        out = await backend.soft_invalidate_relationships_batch([], namespace_id=_NS, invalidated_at=_TS)
        assert out == 0
        assert captured == []
        backend._cypher.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stamps_valid_until_idempotent_namespace_scoped(self) -> None:
        rel_id = uuid4()
        backend, captured, _session = _age_with_cypher_capture(per_call_return=1)

        out = await backend.soft_invalidate_relationships_batch([rel_id], namespace_id=_NS, invalidated_at=_TS)
        assert out == 1
        assert len(captured) == 1
        cypher = captured[0]
        # Soft-delete stamps valid_until (ISO literal), never a hard delete.
        assert "SET r.valid_until" in cypher
        assert _TS.isoformat() in cypher
        assert "DELETE" not in cypher.upper()
        # Idempotent replay guard.
        assert "r.valid_until IS NULL" in cypher
        # Namespace-scoped + matched by id; AGE interpolates literals (no $param).
        assert str(rel_id) in cypher
        assert str(_NS) in cypher

    @pytest.mark.asyncio
    async def test_batch_is_one_auto_commit_unit_and_count_sums(self) -> None:
        """The whole id batch runs inside a SINGLE session.begin() so it is one
        auto-commit unit (a mid-batch failure rolls the batch back -> the
        reconciler retries cleanly). The affected counts sum across ids."""
        ids = [uuid4(), uuid4(), uuid4()]
        backend, captured, session = _age_with_cypher_capture(per_call_return=1)

        out = await backend.soft_invalidate_relationships_batch(ids, namespace_id=_NS, invalidated_at=_TS)
        # One SET per id, summed.
        assert out == 3
        assert len(captured) == 3
        # Exactly one session opened (one auto-commit unit) for the whole batch.
        assert backend._session_factory.call_count == 1
        # session.begin() entered exactly once -> a single transaction wraps all
        # three SETs, so a mid-batch failure rolls the whole batch back.
        assert session.begin.call_count == 1

    @pytest.mark.asyncio
    async def test_uuid_lit_rejects_non_uuid_id(self) -> None:
        """ids route through _uuid_lit (the IDOR / injection boundary): a
        non-UUID payload fails fast at the boundary, never reaching the graph."""
        backend, _captured, _session = _age_with_cypher_capture(per_call_return=1)
        with pytest.raises(ValueError):
            await backend.soft_invalidate_relationships_batch(
                ["x'; MATCH (n) DETACH DELETE n; //"],  # type: ignore[list-item]
                namespace_id=_NS,
                invalidated_at=_TS,
            )


@pytest.mark.unit
class TestAGERestoreRelationships:
    @pytest.mark.asyncio
    async def test_empty_short_circuits(self) -> None:
        backend, captured, _session = _age_with_cypher_capture(per_call_return=1)
        out = await backend.restore_relationships_batch([], namespace_id=_NS)
        assert out == 0
        assert captured == []

    @pytest.mark.asyncio
    async def test_clears_valid_until_idempotent(self) -> None:
        rel_id = uuid4()
        backend, captured, _session = _age_with_cypher_capture(per_call_return=1)

        out = await backend.restore_relationships_batch([rel_id], namespace_id=_NS)
        assert out == 1
        cypher = captured[0]
        assert "SET r.valid_until = null" in cypher
        # updated_at is bumped on restore too (matches the Neo4j reference verb).
        assert "r.updated_at =" in cypher
        assert "r.valid_until IS NOT NULL" in cypher
        assert "DELETE" not in cypher.upper()
        assert str(rel_id) in cypher
        assert str(_NS) in cypher
