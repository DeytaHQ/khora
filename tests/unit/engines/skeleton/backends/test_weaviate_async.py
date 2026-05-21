"""Unit tests for ``WeaviateTemporalStore`` async migration (#783).

These tests exercise the v4 async client surface with mocks - no
real Weaviate instance is required. Integration tests against a live
cluster live behind ``WEAVIATE_INTEGRATION_TEST=1`` (see
``tests/integration/test_weaviate_async_integration.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from khora.engines.skeleton.backends import TemporalChunk
from khora.engines.skeleton.backends.weaviate import (
    COLLECTION_NAME,
    WeaviateBackendConfig,
    WeaviateTemporalStore,
    _coerce_backend_config,
    _coerce_datetime,
    _extract_vector,
    _parse_host_port,
)

# ---------------------------------------------------------------------------
# WeaviateBackendConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWeaviateBackendConfig:
    def test_local_url_only_is_valid(self) -> None:
        cfg = WeaviateBackendConfig(url="http://localhost:8090")
        assert cfg.is_cloud is False
        assert cfg.secret_api_key() is None

    def test_cloud_with_api_key_is_valid(self) -> None:
        cfg = WeaviateBackendConfig(
            cluster_url="https://my.weaviate.network",
            api_key="secret",
        )
        assert cfg.is_cloud is True
        assert cfg.secret_api_key() == "secret"

    def test_cloud_with_secret_str_api_key(self) -> None:
        cfg = WeaviateBackendConfig(
            cluster_url="https://my.weaviate.network",
            api_key=SecretStr("secret"),
        )
        assert cfg.secret_api_key() == "secret"

    def test_url_and_cluster_url_together_raises(self) -> None:
        with pytest.raises(ValueError, match="either `url`.*or `cluster_url`"):
            WeaviateBackendConfig(
                url="http://localhost:8080",
                cluster_url="https://my.weaviate.network",
                api_key="x",
            )

    def test_no_endpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="requires either `url`"):
            WeaviateBackendConfig()

    def test_cloud_without_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="requires an `api_key`"):
            WeaviateBackendConfig(cluster_url="https://my.weaviate.network")

    def test_log_safe_endpoint_redacts(self) -> None:
        cfg = WeaviateBackendConfig(url="http://user:pw@localhost:8080")
        out = cfg.log_safe_endpoint()
        assert "pw" not in out
        assert "localhost" in out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHelpers:
    def test_parse_host_port_defaults_to_8080(self) -> None:
        assert _parse_host_port("http://example.com") == ("example.com", 8080)

    def test_parse_host_port_uses_explicit_port(self) -> None:
        assert _parse_host_port("http://localhost:8090") == ("localhost", 8090)

    def test_parse_host_port_accepts_bare_hostname(self) -> None:
        assert _parse_host_port("localhost:8090") == ("localhost", 8090)

    def test_coerce_string(self) -> None:
        cfg = _coerce_backend_config("http://localhost:8090")
        assert isinstance(cfg, WeaviateBackendConfig)
        assert cfg.url == "http://localhost:8090"

    def test_coerce_config_passthrough(self) -> None:
        original = WeaviateBackendConfig(url="http://localhost:8090")
        assert _coerce_backend_config(original) is original

    def test_coerce_rejects_other_types(self) -> None:
        with pytest.raises(TypeError):
            _coerce_backend_config(8080)  # type: ignore[arg-type]

    # Defensive parsers used by _object_to_chunk — regression coverage
    # for the v4-client shape mismatch that broke #803's first run.

    def test_coerce_datetime_none(self) -> None:
        assert _coerce_datetime(None) is None

    def test_coerce_datetime_passthrough(self) -> None:
        dt = datetime(2026, 5, 21, 12, 30, tzinfo=UTC)
        assert _coerce_datetime(dt) is dt

    def test_coerce_datetime_iso_string(self) -> None:
        out = _coerce_datetime("2026-05-21T12:30:00Z")
        assert out == datetime(2026, 5, 21, 12, 30, tzinfo=UTC)

    def test_coerce_datetime_invalid_returns_none(self) -> None:
        assert _coerce_datetime("not a date") is None
        assert _coerce_datetime(42) is None  # not str / datetime

    def test_extract_vector_none(self) -> None:
        assert _extract_vector(None) is None

    def test_extract_vector_default_keyed_dict(self) -> None:
        assert _extract_vector({"default": [0.1, 0.2, 0.3]}) == [0.1, 0.2, 0.3]

    def test_extract_vector_first_value_when_default_missing(self) -> None:
        # Older / alternate vector names still surface a list
        assert _extract_vector({"main": [0.5, 0.6]}) == [0.5, 0.6]

    def test_extract_vector_empty_dict(self) -> None:
        assert _extract_vector({}) is None

    def test_extract_vector_plain_list(self) -> None:
        assert _extract_vector([0.7, 0.8, 0.9]) == [0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# WeaviateTemporalStore.connect routing (local / cloud / custom)
# ---------------------------------------------------------------------------


def _install_fake_weaviate(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Inject a fake ``weaviate`` module into ``sys.modules``.

    Returns ``(weaviate_module, client_instance)`` so each test can
    assert against the helper used (``use_async_with_local`` /
    ``use_async_with_weaviate_cloud`` / ``use_async_with_custom``) and
    the kwargs threaded through.
    """
    import sys

    fake_weaviate = MagicMock(name="weaviate")

    # A fake client where every method is async-awaitable
    client = MagicMock(name="WeaviateAsyncClient")
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.is_ready = AsyncMock(return_value=True)
    client.collections = MagicMock()
    client.collections.exists = AsyncMock(return_value=True)  # skip create branch
    client.collections.create = AsyncMock()
    client.collections.get = MagicMock(return_value=MagicMock(name="CollectionAsync"))

    fake_weaviate.use_async_with_local = MagicMock(return_value=client)
    fake_weaviate.use_async_with_custom = MagicMock(return_value=client)
    fake_weaviate.use_async_with_weaviate_cloud = MagicMock(return_value=client)

    # weaviate.classes.config + weaviate.classes.init submodules
    classes_mod = ModuleType("weaviate.classes")
    config_mod = ModuleType("weaviate.classes.config")
    init_mod = ModuleType("weaviate.classes.init")
    tenants_mod = ModuleType("weaviate.classes.tenants")
    query_mod = ModuleType("weaviate.classes.query")

    class _Configure:
        @staticmethod
        def multi_tenancy(enabled: bool = True) -> Any:
            return SimpleNamespace(enabled=enabled)

        class Vectorizer:
            @staticmethod
            def none() -> Any:
                return SimpleNamespace(kind="none")

    class _DataType:
        TEXT = "TEXT"
        UUID = "UUID"
        DATE = "DATE"
        TEXT_ARRAY = "TEXT_ARRAY"
        NUMBER = "NUMBER"

    def _Property(*, name: str, data_type: str) -> Any:
        return SimpleNamespace(name=name, data_type=data_type)

    class _Auth:
        @staticmethod
        def api_key(key: str) -> Any:
            return SimpleNamespace(scheme="apikey", key=key)

    class _Tenant:
        def __init__(self, *, name: str) -> None:
            self.name = name

    config_mod.Configure = _Configure  # type: ignore[attr-defined]
    config_mod.DataType = _DataType  # type: ignore[attr-defined]
    config_mod.Property = _Property  # type: ignore[attr-defined]
    init_mod.Auth = _Auth  # type: ignore[attr-defined]
    tenants_mod.Tenant = _Tenant  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "weaviate", fake_weaviate)
    monkeypatch.setitem(sys.modules, "weaviate.classes", classes_mod)
    monkeypatch.setitem(sys.modules, "weaviate.classes.config", config_mod)
    monkeypatch.setitem(sys.modules, "weaviate.classes.init", init_mod)
    monkeypatch.setitem(sys.modules, "weaviate.classes.tenants", tenants_mod)
    monkeypatch.setitem(sys.modules, "weaviate.classes.query", query_mod)

    return fake_weaviate, client


def _build_store(weaviate_config: str | WeaviateBackendConfig) -> WeaviateTemporalStore:
    config = MagicMock(name="KhoraConfig")
    config.llm.embedding_dimension = 1536
    return WeaviateTemporalStore(config, weaviate_config)


@pytest.mark.unit
class TestConnectRouting:
    @pytest.mark.asyncio
    async def test_local_uses_use_async_with_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weaviate, client = _install_fake_weaviate(monkeypatch)
        store = _build_store("http://localhost:8090")

        await store.connect()

        weaviate.use_async_with_custom.assert_called_once()
        call = weaviate.use_async_with_custom.call_args.kwargs
        assert call["http_host"] == "localhost"
        assert call["http_port"] == 8090
        assert call["grpc_port"] == 50051  # default
        assert call["auth_credentials"] is None  # no api_key set
        weaviate.use_async_with_local.assert_not_called()
        weaviate.use_async_with_weaviate_cloud.assert_not_called()
        client.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cloud_uses_use_async_with_weaviate_cloud_with_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weaviate, client = _install_fake_weaviate(monkeypatch)
        store = _build_store(
            WeaviateBackendConfig(
                cluster_url="https://my-cluster.weaviate.network",
                api_key=SecretStr("sk-cloud-key"),
            )
        )

        await store.connect()

        weaviate.use_async_with_weaviate_cloud.assert_called_once()
        call = weaviate.use_async_with_weaviate_cloud.call_args.kwargs
        assert call["cluster_url"] == "https://my-cluster.weaviate.network"
        # Auth credentials object carries the unwrapped api key
        assert call["auth_credentials"].key == "sk-cloud-key"
        weaviate.use_async_with_custom.assert_not_called()
        client.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_ports_threaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weaviate, _client = _install_fake_weaviate(monkeypatch)
        store = _build_store(
            WeaviateBackendConfig(
                url="https://example.internal:8443",
                grpc_port=50061,
                http_secure=True,
                grpc_secure=True,
                api_key="local-key",
            )
        )

        await store.connect()

        call = weaviate.use_async_with_custom.call_args.kwargs
        assert call["http_port"] == 8443
        assert call["grpc_port"] == 50061
        assert call["http_secure"] is True
        assert call["grpc_secure"] is True
        assert call["auth_credentials"].key == "local-key"

    @pytest.mark.asyncio
    async def test_collection_create_skipped_when_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)
        client.collections.exists = AsyncMock(return_value=True)
        store = _build_store("http://localhost:8090")

        await store.connect()

        client.collections.exists.assert_awaited_once_with(COLLECTION_NAME)
        client.collections.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_collection_create_called_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)
        client.collections.exists = AsyncMock(return_value=False)
        store = _build_store("http://localhost:8090")

        await store.connect()

        client.collections.create.assert_awaited_once()
        kwargs = client.collections.create.call_args.kwargs
        assert kwargs["name"] == COLLECTION_NAME
        assert kwargs["multi_tenancy_config"].enabled is True


# ---------------------------------------------------------------------------
# Tenant cache + CRUD smoke
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTenantCache:
    @pytest.mark.asyncio
    async def test_tenants_created_once_per_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)

        collection = MagicMock(name="CollectionAsync")
        collection.tenants.create = AsyncMock()
        collection.with_tenant = MagicMock(return_value=MagicMock(name="TenantScopedCollection"))
        client.collections.get = MagicMock(return_value=collection)

        store = _build_store("http://localhost:8090")
        await store.connect()

        ns_id = uuid4()
        await store._get_collection(ns_id)
        await store._get_collection(ns_id)
        await store._get_collection(ns_id)

        # Tenant creation only fired on the first call - second/third hit the cache.
        assert collection.tenants.create.await_count == 1


@pytest.mark.unit
class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_disconnected_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_weaviate(monkeypatch)
        store = _build_store("http://localhost:8090")
        out = await store.health_check()
        assert out == {"status": "disconnected", "backend": "weaviate"}

    @pytest.mark.asyncio
    async def test_healthy_when_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)
        client.is_ready = AsyncMock(return_value=True)
        store = _build_store("http://localhost:8090")
        await store.connect()
        out = await store.health_check()
        assert out["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_unhealthy_when_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)
        client.is_ready = AsyncMock(return_value=False)
        store = _build_store("http://localhost:8090")
        await store.connect()
        out = await store.health_check()
        assert out["status"] == "unhealthy"
        assert "Not ready" in out["error"]


@pytest.mark.unit
class TestDisconnectIsAsync:
    @pytest.mark.asyncio
    async def test_disconnect_awaits_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)
        store = _build_store("http://localhost:8090")
        await store.connect()

        await store.disconnect()

        client.close.assert_awaited_once()
        assert store._client is None
        assert store._connected is False


@pytest.mark.unit
class TestCRUD:
    @pytest.mark.asyncio
    async def test_create_chunk_awaits_insert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)

        collection = MagicMock(name="CollectionAsync")
        collection.tenants.create = AsyncMock()
        tenant_scoped = MagicMock(name="TenantScopedCollection")
        tenant_scoped.data.insert = AsyncMock()
        collection.with_tenant = MagicMock(return_value=tenant_scoped)
        client.collections.get = MagicMock(return_value=collection)

        store = _build_store("http://localhost:8090")
        await store.connect()

        ns_id = uuid4()
        doc_id = uuid4()
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=doc_id,
            content="hello",
            embedding=[0.1] * 1536,
        )
        out = await store.create_chunk(chunk)

        assert isinstance(out.id, UUID)
        tenant_scoped.data.insert.assert_awaited_once()
        kwargs = tenant_scoped.data.insert.call_args.kwargs
        assert kwargs["properties"]["content"] == "hello"
        assert kwargs["properties"]["document_id"] == str(doc_id)

    @pytest.mark.asyncio
    async def test_create_chunks_batch_fans_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)

        collection = MagicMock(name="CollectionAsync")
        collection.tenants.create = AsyncMock()
        tenant_scoped = MagicMock(name="TenantScopedCollection")
        tenant_scoped.data.insert = AsyncMock()
        collection.with_tenant = MagicMock(return_value=tenant_scoped)
        client.collections.get = MagicMock(return_value=collection)

        store = _build_store("http://localhost:8090")
        await store.connect()

        ns_id = uuid4()
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=ns_id,
                document_id=uuid4(),
                content=f"c{i}",
                embedding=[float(i)] * 1536,
            )
            for i in range(5)
        ]

        out = await store.create_chunks_batch(chunks)

        assert len(out) == 5
        assert tenant_scoped.data.insert.await_count == 5

    @pytest.mark.asyncio
    async def test_delete_chunk_awaits_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _weaviate, client = _install_fake_weaviate(monkeypatch)

        collection = MagicMock(name="CollectionAsync")
        collection.tenants.create = AsyncMock()
        tenant_scoped = MagicMock(name="TenantScopedCollection")
        tenant_scoped.data.delete_by_id = AsyncMock()
        collection.with_tenant = MagicMock(return_value=tenant_scoped)
        client.collections.get = MagicMock(return_value=collection)

        store = _build_store("http://localhost:8090")
        await store.connect()

        ok = await store.delete_chunk(uuid4(), uuid4())
        assert ok is True
        tenant_scoped.data.delete_by_id.assert_awaited_once()
