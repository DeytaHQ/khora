"""Tests for ``NamespaceRequiredProxy`` and the coordinator's privatization."""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.storage._namespace_proxy import NamespaceRequiredProxy
from khora.storage.coordinator import StorageCoordinator


@pytest.fixture(autouse=True)
def _reset_warned_roles() -> None:
    """Each test sees a fresh process-wide warning state."""
    NamespaceRequiredProxy._warned_roles.clear()


@pytest.mark.unit
class TestProxyEnforcesNamespaceId:
    @pytest.mark.asyncio
    async def test_read_method_without_namespace_id_raises(self) -> None:
        backend = MagicMock()
        backend.get_entity = AsyncMock(return_value="should not reach")
        proxy = NamespaceRequiredProxy(backend, "graph")

        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError, match="requires keyword argument 'namespace_id'"):
                await proxy.get_entity(uuid4())

        backend.get_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_method_with_namespace_id_dispatches(self) -> None:
        backend = MagicMock()
        backend.get_entity = AsyncMock(return_value="ok")
        proxy = NamespaceRequiredProxy(backend, "graph")

        ns = uuid4()
        with pytest.warns(DeprecationWarning):
            result = await proxy.get_entity(uuid4(), namespace_id=ns)

        assert result == "ok"
        backend.get_entity.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_method_passes_through_without_namespace_id(self) -> None:
        """Write methods aren't in the enforcement set — IGR-226 will tighten them."""
        backend = MagicMock()
        backend.upsert_entity = AsyncMock(return_value="ok")
        proxy = NamespaceRequiredProxy(backend, "graph")

        with pytest.warns(DeprecationWarning):
            result = await proxy.upsert_entity(object())

        assert result == "ok"
        backend.upsert_entity.assert_awaited_once()


@pytest.mark.unit
class TestProxyAttributeForwarding:
    def test_non_callable_public_attr_forwards(self) -> None:
        backend = MagicMock()
        backend.health_status = "ok"
        proxy = NamespaceRequiredProxy(backend, "vector")

        with pytest.warns(DeprecationWarning):
            assert proxy.health_status == "ok"

    def test_underscore_attrs_not_proxied(self) -> None:
        """Internal `_engine`/`_handle` peeks must NOT be reachable via the proxy.

        External callers shouldn't be reaching into backend internals; the
        proxy refuses to forward underscore-prefixed names so this stays a
        compile-time error from outside.
        """
        backend = MagicMock()
        backend._engine = "pg-engine-handle"
        proxy = NamespaceRequiredProxy(backend, "vector")

        with pytest.raises(AttributeError):
            _ = proxy._engine  # noqa: SLF001

    def test_callable_non_read_method_forwards_unchanged(self) -> None:
        backend = MagicMock()
        backend.connect = MagicMock(return_value="connected")
        proxy = NamespaceRequiredProxy(backend, "relational")

        with pytest.warns(DeprecationWarning):
            assert proxy.connect() == "connected"


@pytest.mark.unit
class TestProxyWarnOnce:
    def test_deprecation_warns_only_once_per_role(self) -> None:
        backend = MagicMock()
        backend.connect = MagicMock(return_value=None)
        proxy = NamespaceRequiredProxy(backend, "graph")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            proxy.connect()
            proxy.connect()
            proxy.connect()

        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "graph" in str(deprecations[0].message)


@pytest.mark.unit
class TestCoordinatorPrivatization:
    def test_construction_kwargs_unchanged(self) -> None:
        rel, vec, graph, evt = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        # No "_conn" probe collision — these are pure mocks.
        coord = StorageCoordinator(relational=rel, vector=vec, graph=graph, event_store=evt)

        # Private attrs hold the raw backends.
        assert coord._relational is rel
        assert coord._vector is vec
        assert coord._graph is graph
        assert coord._event_store is evt

    def test_public_attrs_are_proxies(self) -> None:
        rel, vec, graph, evt = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        coord = StorageCoordinator(relational=rel, vector=vec, graph=graph, event_store=evt)

        assert isinstance(coord.relational, NamespaceRequiredProxy)
        assert isinstance(coord.vector, NamespaceRequiredProxy)
        assert isinstance(coord.graph, NamespaceRequiredProxy)
        assert isinstance(coord.event_store, NamespaceRequiredProxy)

    def test_none_backends_stay_none_on_public_attrs(self) -> None:
        coord = StorageCoordinator()
        assert coord._relational is None
        assert coord._vector is None
        assert coord._graph is None
        assert coord._event_store is None
        assert coord.relational is None
        assert coord.vector is None
        assert coord.graph is None
        assert coord.event_store is None

    def test_post_construction_assign_refreshes_proxy(self) -> None:
        """Assigning to coord.relational after __init__ rewraps the proxy."""
        rel = MagicMock()
        coord = StorageCoordinator(relational=rel)
        original_proxy = coord.relational

        new_rel = MagicMock()
        coord.relational = new_rel  # type: ignore[assignment]

        # Private ref updated.
        assert coord._relational is new_rel
        # Public attr is a fresh proxy wrapping the new backend.
        assert coord.relational is not original_proxy
        assert isinstance(coord.relational, NamespaceRequiredProxy)
        assert coord.relational._backend is new_rel  # noqa: SLF001

    def test_post_construction_assign_none_clears_proxy(self) -> None:
        rel = MagicMock()
        coord = StorageCoordinator(relational=rel)
        coord.relational = None  # type: ignore[assignment]
        assert coord._relational is None
        assert coord.relational is None
