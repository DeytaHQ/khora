"""Coverage: ``StorageCoordinator`` not-configured guard paths.

Pre-PR coverage of ``coordinator.py`` was 62%. The missing lines are
mostly the ``if not self.X: raise RuntimeError(...)`` guards on every
delegating method. These tests pin the guard contract so regressions
where someone forgets to add the guard get caught immediately.

These tests do NOT exercise the real backends — they instantiate a
coordinator with all four backends set to None and assert each public
method raises RuntimeError with the right message.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import (
    MemoryNamespace,
)
from khora.storage.coordinator import StorageCoordinator

# ---------------------------------------------------------------------------
# Construction & lifecycle
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction_all_none(self) -> None:
        coord = StorageCoordinator()
        assert coord.relational is None
        assert coord.vector is None
        assert coord.graph is None
        assert coord.event_store is None
        assert coord._connected is False
        assert coord._is_unified_backend is False

    def test_unified_detected_when_graph_vector_share_conn(self) -> None:
        """If graph._conn and vector._conn are the same object → unified."""
        shared_conn = object()
        graph = MagicMock()
        graph._conn = shared_conn
        vector = MagicMock()
        vector._conn = shared_conn
        coord = StorageCoordinator(graph=graph, vector=vector)
        assert coord._is_unified_backend is True

    def test_unified_not_detected_when_conns_differ(self) -> None:
        graph = MagicMock()
        graph._conn = object()
        vector = MagicMock()
        vector._conn = object()
        coord = StorageCoordinator(graph=graph, vector=vector)
        assert coord._is_unified_backend is False

    def test_unified_probe_swallows_property_exceptions(self) -> None:
        """If ``_conn`` is a property that raises, the probe falls back to None."""

        class GraphWithRaisingConn:
            @property
            def _conn(self):  # noqa: D401
                raise RuntimeError("connection not open yet")

        coord = StorageCoordinator(graph=GraphWithRaisingConn(), vector=MagicMock())
        assert coord._is_unified_backend is False


# ---------------------------------------------------------------------------
# connect / disconnect — early-return when no backends
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    @pytest.mark.asyncio
    async def test_connect_with_no_backends_is_noop(self) -> None:
        coord = StorageCoordinator()
        await coord.connect()
        assert coord._connected is True

    @pytest.mark.asyncio
    async def test_double_connect_is_noop(self) -> None:
        rel = AsyncMock()
        coord = StorageCoordinator(relational=rel)
        await coord.connect()
        await coord.connect()  # second call must early-return
        rel.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_is_noop(self) -> None:
        rel = AsyncMock()
        coord = StorageCoordinator(relational=rel)
        await coord.disconnect()
        rel.disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_then_disconnect_calls_both(self) -> None:
        rel = AsyncMock()
        vec = AsyncMock()
        gph = AsyncMock()
        evt = AsyncMock()
        coord = StorageCoordinator(relational=rel, vector=vec, graph=gph, event_store=evt)
        await coord.connect()
        await coord.disconnect()
        rel.connect.assert_awaited_once()
        rel.disconnect.assert_awaited_once()
        vec.connect.assert_awaited_once()
        gph.connect.assert_awaited_once()
        evt.connect.assert_awaited_once()


# ---------------------------------------------------------------------------
# health_check — works with no backends, marks each as healthy/unhealthy
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_no_backends(self) -> None:
        coord = StorageCoordinator()
        health = await coord.health_check()
        # All defaults
        assert health is not None

    @pytest.mark.asyncio
    async def test_health_check_treats_exception_as_unhealthy(self) -> None:
        rel = AsyncMock()
        rel.is_healthy = AsyncMock(side_effect=RuntimeError("boom"))
        coord = StorageCoordinator(relational=rel)
        health = await coord.health_check()
        # Exception → False
        assert health.relational is False

    @pytest.mark.asyncio
    async def test_health_check_records_true(self) -> None:
        rel = AsyncMock()
        rel.is_healthy = AsyncMock(return_value=True)
        coord = StorageCoordinator(relational=rel)
        health = await coord.health_check()
        assert health.relational is True


# ---------------------------------------------------------------------------
# transaction — no SQL backend → RuntimeError
# ---------------------------------------------------------------------------


class TestTransaction:
    @pytest.mark.asyncio
    async def test_transaction_with_no_session_factory_raises(self) -> None:
        coord = StorageCoordinator()  # all None
        with pytest.raises(RuntimeError, match="No SQL backend"):
            async with coord.transaction():
                pass

    @pytest.mark.asyncio
    async def test_transaction_with_no_factory_on_relational_raises(self) -> None:
        rel = MagicMock()
        # no _session_factory attribute → getattr returns None
        del rel._session_factory
        rel.configure_mock(**{"_session_factory": None})
        coord = StorageCoordinator(relational=rel)
        with pytest.raises(RuntimeError, match="No SQL backend"):
            async with coord.transaction():
                pass


# ---------------------------------------------------------------------------
# Delegation guards — every public method that requires a backend
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_coord() -> StorageCoordinator:
    return StorageCoordinator()


class TestRelationalGuards:
    @pytest.mark.asyncio
    async def test_resolve_namespace(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.resolve_namespace(uuid4())

    @pytest.mark.asyncio
    async def test_create_namespace(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.create_namespace(MemoryNamespace())

    @pytest.mark.asyncio
    async def test_get_namespace(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.get_namespace(uuid4())

    @pytest.mark.asyncio
    async def test_list_namespaces(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.list_namespaces()

    @pytest.mark.asyncio
    async def test_update_namespace(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.update_namespace(MemoryNamespace())

    @pytest.mark.asyncio
    async def test_create_namespace_version(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.create_namespace_version()

    @pytest.mark.asyncio
    async def test_deactivate_namespace(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.deactivate_namespace(uuid4())

    @pytest.mark.asyncio
    async def test_create_document(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.create_document(MagicMock())

    @pytest.mark.asyncio
    async def test_get_document(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.get_document(uuid4(), namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_list_documents(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.list_documents(uuid4())

    @pytest.mark.asyncio
    async def test_update_document(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.update_document(MagicMock())

    @pytest.mark.asyncio
    async def test_delete_document(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.delete_document(uuid4())

    @pytest.mark.asyncio
    async def test_get_document_by_checksum(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.get_document_by_checksum(uuid4(), "x")

    @pytest.mark.asyncio
    async def test_get_documents_by_checksums(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.get_documents_by_checksums(uuid4(), ["x"])

    @pytest.mark.asyncio
    async def test_get_document_by_external_id(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.get_document_by_external_id("ext", namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_get_documents_by_external_ids(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.get_documents_by_external_ids(["e"], namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_get_last_activity_at(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.get_last_activity_at(uuid4())

    @pytest.mark.asyncio
    async def test_get_document_stats(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.get_document_stats(uuid4())


class TestVectorGuards:
    @pytest.mark.asyncio
    async def test_create_chunk(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.create_chunk(MagicMock())

    @pytest.mark.asyncio
    async def test_create_chunks_batch(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.create_chunks_batch([MagicMock()])

    @pytest.mark.asyncio
    async def test_get_chunk(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.get_chunk(uuid4(), namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_get_chunks_by_document(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.get_chunks_by_document(uuid4(), namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_get_chunks_batch_empty_returns_empty(self, empty_coord) -> None:
        """Empty list short-circuits before the guard, returning {}."""
        assert await empty_coord.get_chunks_batch([], namespace_id=uuid4()) == {}

    @pytest.mark.asyncio
    async def test_get_chunks_batch_raises_when_unconfigured(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.get_chunks_batch([uuid4()], namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_search_similar_chunks(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.search_similar_chunks(uuid4(), [0.0])

    @pytest.mark.asyncio
    async def test_search_fulltext_chunks(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.search_fulltext_chunks(uuid4(), "q")

    @pytest.mark.asyncio
    async def test_count_chunks(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.count_chunks(uuid4())

    @pytest.mark.asyncio
    async def test_list_chunks(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await empty_coord.list_chunks(uuid4())


class TestEntityCounts:
    @pytest.mark.asyncio
    async def test_count_entities_no_backend_returns_zero(self, empty_coord) -> None:
        """Best-effort behavior: missing backends → 0 (never raises)."""
        assert await empty_coord.count_entities(uuid4()) == 0

    @pytest.mark.asyncio
    async def test_count_relationships_no_graph_returns_zero(self, empty_coord) -> None:
        assert await empty_coord.count_relationships(uuid4()) == 0

    @pytest.mark.asyncio
    async def test_count_entities_prefers_vector(self) -> None:
        vec = AsyncMock()
        vec.count_entities = AsyncMock(return_value=42)
        gph = AsyncMock()
        gph.count_entities = AsyncMock(return_value=99)
        coord = StorageCoordinator(vector=vec, graph=gph)
        assert await coord.count_entities(uuid4()) == 42
        gph.count_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_count_entities_falls_back_to_graph(self) -> None:
        gph = AsyncMock()
        gph.count_entities = AsyncMock(return_value=99)
        coord = StorageCoordinator(graph=gph)
        assert await coord.count_entities(uuid4()) == 99


# ---------------------------------------------------------------------------
# replace_document_extraction — guard rails (full path is integration-tested)
# ---------------------------------------------------------------------------


class TestReplaceDocumentGuards:
    @pytest.mark.asyncio
    async def test_replace_no_relational(self, empty_coord) -> None:
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await empty_coord.replace_document_extraction(
                namespace_id=uuid4(),
                old_document_id=uuid4(),
                new_document=MagicMock(),
                new_chunks=[],
                new_entities=[],
                new_relationships=[],
            )

    @pytest.mark.asyncio
    async def test_replace_no_vector(self) -> None:
        coord = StorageCoordinator(relational=AsyncMock())
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await coord.replace_document_extraction(
                namespace_id=uuid4(),
                old_document_id=uuid4(),
                new_document=MagicMock(),
                new_chunks=[],
                new_entities=[],
                new_relationships=[],
            )

    @pytest.mark.asyncio
    async def test_replace_no_graph(self) -> None:
        coord = StorageCoordinator(relational=AsyncMock(), vector=AsyncMock())
        with pytest.raises(RuntimeError, match="Graph backend not configured"):
            await coord.replace_document_extraction(
                namespace_id=uuid4(),
                old_document_id=uuid4(),
                new_document=MagicMock(),
                new_chunks=[],
                new_entities=[],
                new_relationships=[],
            )


# ---------------------------------------------------------------------------
# dispatch_hook — no dispatcher → no-op
# ---------------------------------------------------------------------------


class TestDispatchHook:
    @pytest.mark.asyncio
    async def test_no_dispatcher_is_noop(self, empty_coord) -> None:
        # Should not raise
        await empty_coord.dispatch_hook(MagicMock())

    @pytest.mark.asyncio
    async def test_dispatcher_with_zero_subscribers_is_noop(self, empty_coord) -> None:
        dispatcher = MagicMock()
        dispatcher.subscription_count = 0
        empty_coord._hook_dispatcher = dispatcher
        await empty_coord.dispatch_hook(MagicMock())
        dispatcher.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatcher_with_subscribers_dispatches(self, empty_coord) -> None:
        dispatcher = MagicMock()
        dispatcher.subscription_count = 2
        dispatcher.dispatch = AsyncMock()
        empty_coord._hook_dispatcher = dispatcher
        event = MagicMock()
        await empty_coord.dispatch_hook(event)
        dispatcher.dispatch.assert_awaited_once_with(event)
