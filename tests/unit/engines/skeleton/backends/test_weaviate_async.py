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

from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from khora.storage.temporal import TemporalChunk
from khora.storage.temporal.weaviate import (
    _DENORM_TEXT_KEYS,
    _FILTER_OVERFETCH,
    COLLECTION_NAME,
    WeaviateBackendConfig,
    WeaviateTemporalStore,
    _chunk_to_properties,
    _coerce_backend_config,
    _coerce_datetime,
    _denorm_properties,
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

# The collection's pre-denorm column names, in schema order. The denorm names
# (``_DENORM_TEXT_KEYS`` + ``source_timestamp``) are appended to make the
# reconcile-no-op schema.
_BASE_PROPERTY_NAMES: tuple[str, ...] = (
    "content",
    "document_id",
    "namespace_id",
    "occurred_at",
    "created_at",
    "source_system",
    "author",
    "channel",
    "tags",
    "confidence",
    "metadata_json",
)


def _reconcile_noop_collection() -> MagicMock:
    """A collection handle whose schema already carries every denorm property.

    ``connect()``'s reconcile branch reads ``config.get().properties`` and only adds
    a missing denorm prop, so a handle reporting the full schema makes reconcile a
    no-op — what every connect-path test that doesn't exercise reconcile expects.
    """
    collection = MagicMock(name="CollectionAsync")
    full = [*_BASE_PROPERTY_NAMES, *_DENORM_TEXT_KEYS, "source_timestamp"]
    collection.config.get = AsyncMock(return_value=SimpleNamespace(properties=[SimpleNamespace(name=n) for n in full]))
    collection.config.add_property = AsyncMock()
    return collection


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
    # The default existing-collection handle reports a schema that ALREADY carries
    # every denorm property, so connect()'s reconcile branch is a no-op for the
    # tests that don't exercise it (they override collections.get when they do).
    client.collections.get = MagicMock(return_value=_reconcile_noop_collection())

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

    def _Property(*, name: str, data_type: str, **kwargs: Any) -> Any:
        return SimpleNamespace(name=name, data_type=data_type, **kwargs)

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

        collection = _reconcile_noop_collection()
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

        collection = _reconcile_noop_collection()
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

        collection = _reconcile_noop_collection()
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

        collection = _reconcile_noop_collection()
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


# ---------------------------------------------------------------------------
# search() filter_ast wiring (push-down + post-filter)
# ---------------------------------------------------------------------------


class _FakeProperty:
    """Records the property + op a ``Filter.by_property(p).<op>(v)`` call builds.

    The compiler only needs each chained builder to return *some* object the
    wiring can hand to the query; the test inspects the recorded structure to
    confirm a push-down filter was assembled (rather than asserting against the
    real ``weaviate-client`` ``_Filters`` internals, which are not installed in
    the unit-test venv).
    """

    def __init__(self, prop: str) -> None:
        self.prop = prop

    def _leaf(self, op: str, value: Any) -> _FakeFilter:
        return _FakeFilter(kind="leaf", prop=self.prop, op=op, value=value)

    def equal(self, value: Any) -> _FakeFilter:
        return self._leaf("equal", value)

    def greater_than(self, value: Any) -> _FakeFilter:
        return self._leaf("greater_than", value)

    def greater_or_equal(self, value: Any) -> _FakeFilter:
        return self._leaf("greater_or_equal", value)

    def less_than(self, value: Any) -> _FakeFilter:
        return self._leaf("less_than", value)

    def less_or_equal(self, value: Any) -> _FakeFilter:
        return self._leaf("less_or_equal", value)

    def is_none(self, value: bool) -> _FakeFilter:
        return self._leaf("is_none", value)


class _FakeFilter:
    """A minimal stand-in for weaviate v4 ``Filter`` / ``_Filters``.

    Supports the builder surface ``compile_weaviate`` uses
    (``by_property`` / ``all_of`` / ``any_of``) and the ``&`` operator the
    backend wiring uses to AND the pushed-down filter into the legacy temporal
    filter.
    """

    def __init__(
        self,
        *,
        kind: str,
        prop: str | None = None,
        op: str | None = None,
        value: Any = None,
        children: list[_FakeFilter] | None = None,
    ) -> None:
        self.kind = kind
        self.prop = prop
        self.op = op
        self.value = value
        self.children = children or []

    @staticmethod
    def by_property(prop: str) -> _FakeProperty:
        return _FakeProperty(prop)

    @staticmethod
    def all_of(filters: list[_FakeFilter]) -> _FakeFilter:
        return _FakeFilter(kind="all_of", children=list(filters))

    @staticmethod
    def any_of(filters: list[_FakeFilter]) -> _FakeFilter:
        return _FakeFilter(kind="any_of", children=list(filters))

    def __and__(self, other: _FakeFilter) -> _FakeFilter:
        return _FakeFilter(kind="all_of", children=[self, other])

    def props_touched(self) -> set[str]:
        """The set of property names anywhere in the (possibly nested) filter."""
        if self.kind == "leaf":
            return {self.prop} if self.prop else set()
        out: set[str] = set()
        for child in self.children:
            out |= child.props_touched()
        return out


def _install_query_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject ``weaviate.classes.query`` with the fake ``Filter`` + query enums.

    ``compile_weaviate`` lazily imports ``Filter`` from this submodule, and
    ``search`` imports ``HybridFusion`` / ``MetadataQuery`` from it. The base
    fake-weaviate installer leaves ``query`` empty, so this layers the symbols on.
    """
    import sys

    query_mod = sys.modules["weaviate.classes.query"]
    query_mod.Filter = _FakeFilter  # type: ignore[attr-defined]
    query_mod.HybridFusion = SimpleNamespace(RELATIVE_SCORE="relative_score")  # type: ignore[attr-defined]
    query_mod.MetadataQuery = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "weaviate.classes.query", query_mod)


def _fake_object(*, occurred_at: datetime, distance: float = 0.1, metadata: dict[str, Any] | None = None) -> Any:
    """Build a fake Weaviate result object shaped for ``_object_to_chunk``.

    ``metadata`` is serialized into the ``metadata_json`` string property the way
    real chunks store it, so ``_object_to_chunk`` parses it back onto the chunk's
    ``metadata`` dict for the python post-filter to read.
    """
    import json

    return SimpleNamespace(
        uuid=str(uuid4()),
        properties={
            "content": "hello",
            "document_id": str(uuid4()),
            "occurred_at": occurred_at,
            "created_at": occurred_at,
            "source_system": "slack",
            "metadata_json": json.dumps(metadata or {}),
        },
        vector=None,
        metadata=SimpleNamespace(distance=distance, score=1.0 - distance),
    )


async def _store_with_query(
    monkeypatch: pytest.MonkeyPatch, objects: list[Any]
) -> tuple[WeaviateTemporalStore, MagicMock]:
    """A connected store whose tenant-scoped ``query.near_vector`` returns ``objects``."""
    _weaviate, client = _install_fake_weaviate(monkeypatch)
    _install_query_filter(monkeypatch)

    collection = _reconcile_noop_collection()
    collection.tenants.create = AsyncMock()
    tenant_scoped = MagicMock(name="TenantScopedCollection")
    tenant_scoped.query.near_vector = AsyncMock(return_value=SimpleNamespace(objects=objects))
    tenant_scoped.query.hybrid = AsyncMock(return_value=SimpleNamespace(objects=objects))
    collection.with_tenant = MagicMock(return_value=tenant_scoped)
    client.collections.get = MagicMock(return_value=collection)

    store = _build_store("http://localhost:8090")
    await store.connect()
    return store, tenant_scoped


@pytest.mark.unit
class TestSearchFilterAstWiring:
    @pytest.mark.asyncio
    async def test_no_filter_ast_does_not_overfetch_or_post_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Baseline: without filter_ast the query keeps the caller's limit and
        # passes no AST-derived filter (None when there is no temporal filter).
        objs = [_fake_object(occurred_at=datetime(2026, 6, i + 1, tzinfo=UTC)) for i in range(3)]
        store, tenant_scoped = await _store_with_query(monkeypatch, objs)

        results = await store.search(uuid4(), [0.1] * 1536, limit=5)

        tenant_scoped.query.near_vector.assert_awaited_once()
        kwargs = tenant_scoped.query.near_vector.call_args.kwargs
        assert kwargs["limit"] == 5  # no over-fetch
        assert kwargs["filters"] is None
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_filter_ast_pushes_date_filter_and_overfetches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A date predicate on a declared property (occurred_at) must push down to
        # Weaviate (a non-None filter touching occurred_at) and the query must
        # over-fetch because a post-filter runs.
        objs = [_fake_object(occurred_at=datetime(2026, 6, i + 1, tzinfo=UTC)) for i in range(3)]
        store, tenant_scoped = await _store_with_query(monkeypatch, objs)

        ast = parse_to_ast(RecallFilter.model_validate({"occurred_at": {"$gt": "2026-01-01T00:00:00Z"}}))
        await store.search(uuid4(), [0.1] * 1536, limit=10, filter_ast=ast)

        kwargs = tenant_scoped.query.near_vector.call_args.kwargs
        assert kwargs["limit"] == 10 * _FILTER_OVERFETCH
        pushed = kwargs["filters"]
        assert pushed is not None
        assert isinstance(pushed, _FakeFilter)
        assert "occurred_at" in pushed.props_touched()

    @pytest.mark.asyncio
    async def test_post_filter_drops_non_matching_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The post-filter re-checks the whole AST in Python: candidates whose
        # occurred_at falls on/before the cutoff are dropped even though the mock
        # query returned them (the push-down is a superset prefilter).
        cutoff = datetime(2026, 6, 1, tzinfo=UTC)
        passing = [_fake_object(occurred_at=datetime(2026, 6, 5, tzinfo=UTC)) for _ in range(2)]
        failing = [_fake_object(occurred_at=datetime(2026, 5, 1, tzinfo=UTC)) for _ in range(3)]
        store, _tenant = await _store_with_query(monkeypatch, passing + failing)

        ast = parse_to_ast(RecallFilter.model_validate({"occurred_at": {"$gt": cutoff.isoformat()}}))
        results = await store.search(uuid4(), [0.1] * 1536, limit=10, filter_ast=ast)

        assert len(results) == 2  # the 3 failing rows are post-filtered out
        for r in results:
            assert r.chunk.occurred_at is not None
            assert r.chunk.occurred_at > cutoff

    @pytest.mark.asyncio
    async def test_post_filter_trims_survivors_to_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Over-fetch returns more survivors than ``limit``; the result is trimmed
        # back to ``limit`` while preserving the order the query returned them
        # (Weaviate ranks best-first, so trimming keeps the top-``limit`` slice).
        objs = [_fake_object(occurred_at=datetime(2026, 6, 10, tzinfo=UTC), distance=0.05) for _ in range(8)]
        returned_ids = [obj.uuid for obj in objs]
        store, _tenant = await _store_with_query(monkeypatch, objs)

        ast = parse_to_ast(RecallFilter.model_validate({"occurred_at": {"$gt": "2026-01-01T00:00:00Z"}}))
        results = await store.search(uuid4(), [0.1] * 1536, limit=3, filter_ast=ast)

        assert len(results) == 3  # trimmed from 8 survivors
        # Order preserved: the trimmed slice is the first 3 the query returned.
        assert [str(r.chunk.id) for r in results] == returned_ids[:3]

    @pytest.mark.asyncio
    async def test_undeclared_system_key_post_filters_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A predicate on a document-grained key that is NOT a declared Weaviate
        # property (source_name) pushes NOTHING down (filters stays None), but the
        # post-filter still applies it — and since that property is absent on the
        # stored object, compile_python's positive-op semantics exclude every row.
        objs = [_fake_object(occurred_at=datetime(2026, 6, i + 1, tzinfo=UTC)) for i in range(3)]
        store, tenant_scoped = await _store_with_query(monkeypatch, objs)

        ast = parse_to_ast(RecallFilter.model_validate({"source_name": "linear"}))
        results = await store.search(uuid4(), [0.1] * 1536, limit=10, filter_ast=ast)

        # Undeclared key → nothing pushed down (no temporal filter either).
        assert tenant_scoped.query.near_vector.call_args.kwargs["filters"] is None
        # source_name is absent on the chunk → positive $eq excludes all rows.
        assert results == []

    @pytest.mark.asyncio
    async def test_filter_plan_out_reports_split_pushdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The weaviate plan is the most intricate of the five backends (#1069):
        # only declared DATE properties push down; every other leaf defers to the
        # compile_python post-filter (post_filtered_keys), and the always-on
        # full-AST re-check sets defensive_recheck. A date + metadata filter
        # exercises the split. The plan is handed back per-call via the sink (no
        # mutable instance state — race-free under concurrent recalls).
        from khora.filter.report import ChannelPlan

        objs = [_fake_object(occurred_at=datetime(2026, 6, i + 1, tzinfo=UTC)) for i in range(3)]
        store, _tenant = await _store_with_query(monkeypatch, objs)

        ast = parse_to_ast(
            RecallFilter.model_validate({"occurred_at": {"$gt": "2026-01-01T00:00:00Z"}, "metadata.tier": "gold"})
        )
        sink: list[ChannelPlan] = []
        await store.search(uuid4(), [0.1] * 1536, limit=10, filter_ast=ast, filter_plan_out=sink)

        assert len(sink) == 1
        plan = sink[0]
        assert "occurred_at" in plan.pushed_keys  # declared DATE property pushes down
        assert "metadata.tier" in plan.post_filtered_keys  # metadata defers to the post-filter
        assert "occurred_at" not in plan.post_filtered_keys  # NO-DEMOTE: pushed leaf not demoted
        assert plan.defensive_recheck is True  # always-on full-AST re-check

    @pytest.mark.asyncio
    async def test_filter_plan_out_empty_for_no_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A no-filter recall appends the empty (nothing-pushed) plan to the sink.
        from khora.filter.report import ChannelPlan

        objs = [_fake_object(occurred_at=datetime(2026, 6, 1, tzinfo=UTC))]
        store, _tenant = await _store_with_query(monkeypatch, objs)

        sink: list[ChannelPlan] = []
        await store.search(uuid4(), [0.1] * 1536, limit=5, filter_plan_out=sink)

        assert sink == [ChannelPlan()]

    @pytest.mark.asyncio
    async def test_selective_metadata_filter_may_return_fewer_than_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A metadata predicate cannot push down (metadata is not a queryable
        # Weaviate property), so the python post-filter enforces it over the
        # over-fetched candidate pool. When the filter is highly selective the
        # result can be FEWER than ``limit`` — superset-safe (only matching rows,
        # never a wrong one), and the surviving subset keeps the returned order.
        #
        # Shape: limit=3 over-fetches 3 * _FILTER_OVERFETCH = 12 candidates; a
        # selective {"metadata.tag": {"$in": ["rare"]}} matches only 2 of them.
        matching = [
            _fake_object(occurred_at=datetime(2026, 6, i + 1, tzinfo=UTC), distance=0.01 * i, metadata={"tag": "rare"})
            for i in range(2)
        ]
        non_matching = [
            _fake_object(occurred_at=datetime(2026, 6, 10 + i, tzinfo=UTC), distance=0.5, metadata={"tag": "common"})
            for i in range(10)
        ]
        # Interleave so the survivors are not just a contiguous prefix.
        objs = [non_matching[0], matching[0], *non_matching[1:5], matching[1], *non_matching[5:]]
        assert len(objs) == 12  # the full over-fetched pool for limit=3
        match_ids = [matching[0].uuid, matching[1].uuid]
        store, tenant_scoped = await _store_with_query(monkeypatch, objs)

        ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["rare"]}}))
        results = await store.search(uuid4(), [0.1] * 1536, limit=3, filter_ast=ast)

        # Over-fetched the 12-candidate pool; metadata path is undeclared → nothing
        # pushed server-side.
        assert tenant_scoped.query.near_vector.call_args.kwargs["limit"] == 3 * _FILTER_OVERFETCH
        assert tenant_scoped.query.near_vector.call_args.kwargs["filters"] is None
        # Exactly the 2 matching rows survive (< limit=3), in returned order.
        assert [str(r.chunk.id) for r in results] == match_ids
        assert all(r.chunk.metadata.get("tag") == "rare" for r in results)


# ---------------------------------------------------------------------------
# Denormalized document fields: write/read round-trip + reconcile
# ---------------------------------------------------------------------------


def _readback_object(props: dict[str, Any], chunk_id: UUID) -> SimpleNamespace:
    """A fake read-back object whose ``.properties`` is exactly what the store wrote."""
    return SimpleNamespace(uuid=str(chunk_id), properties=props, vector=None)


@pytest.mark.unit
class TestDenormFieldsRoundTrip:
    @pytest.mark.asyncio
    async def test_populated_fields_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # _chunk_to_properties → readback object → _object_to_chunk preserves all
        # eight denormalized document fields.
        _install_fake_weaviate(monkeypatch)
        store = _build_store("http://localhost:8090")

        ns_id = uuid4()
        ts = datetime(2026, 4, 1, 9, 30, tzinfo=UTC)
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=uuid4(),
            content="hello",
            embedding=[0.1] * 1536,
            source_type="slack",
            source_name="general",
            source_url="https://example.com/x",
            external_id="ext-42",
            content_type="text/plain",
            source="slack-export",
            title="A title",
            source_timestamp=ts,
        )

        props = _chunk_to_properties(chunk)
        # The seven strings ride verbatim; source_timestamp serializes to ISO.
        assert props["source_type"] == "slack"
        assert props["title"] == "A title"
        assert props["source_timestamp"] == ts.isoformat()

        back = store._object_to_chunk(_readback_object(props, chunk.id), ns_id)
        assert back.source_type == "slack"
        assert back.source_name == "general"
        assert back.source_url == "https://example.com/x"
        assert back.external_id == "ext-42"
        assert back.content_type == "text/plain"
        assert back.source == "slack-export"
        assert back.title == "A title"
        assert back.source_timestamp == ts

    @pytest.mark.asyncio
    async def test_absent_fields_round_trip_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A chunk with no denorm string keys and no source_timestamp round-trips
        # back to None for each — no crash on absent / None properties.
        _install_fake_weaviate(monkeypatch)
        store = _build_store("http://localhost:8090")

        ns_id = uuid4()
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=uuid4(),
            content="hello",
            embedding=[0.1] * 1536,
        )

        props = _chunk_to_properties(chunk)
        assert props["source_timestamp"] is None
        assert props["source_type"] is None

        # Drop the keys entirely (simulate an older row that never carried them) to
        # also exercise the absent-property read path.
        for key in (*_DENORM_TEXT_KEYS, "source_timestamp"):
            props.pop(key, None)

        back = store._object_to_chunk(_readback_object(props, chunk.id), ns_id)
        for key in _DENORM_TEXT_KEYS:
            assert getattr(back, key) is None
        assert back.source_timestamp is None


@pytest.mark.unit
class TestDenormFieldsReconciliation:
    @pytest.mark.asyncio
    async def test_connect_adds_missing_denorm_properties(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # connect() against a pre-existing collection whose schema predates the denorm
        # properties adds each missing one exactly once, with the exact definitions —
        # and the seven TEXT props carry index_searchable=False.
        _weaviate, client = _install_fake_weaviate(monkeypatch)
        client.collections.exists = AsyncMock(return_value=True)

        # The OLD property set (no denorm props): only the pre-denorm columns.
        old_props = [
            SimpleNamespace(name=name)
            for name in (
                "content",
                "document_id",
                "namespace_id",
                "occurred_at",
                "created_at",
                "source_system",
                "author",
                "channel",
                "tags",
                "confidence",
                "metadata_json",
            )
        ]
        collection = MagicMock(name="CollectionAsync")
        collection.config.get = AsyncMock(return_value=SimpleNamespace(properties=old_props))
        collection.config.add_property = AsyncMock()
        client.collections.get = MagicMock(return_value=collection)

        store = _build_store("http://localhost:8090")
        await store.connect()

        expected = _denorm_properties()
        assert collection.config.add_property.await_count == len(expected)
        added = [call.args[0] for call in collection.config.add_property.await_args_list]
        assert [p.name for p in added] == [p.name for p in expected]
        # The seven TEXT props keep BM25 out (index_searchable=False); the DATE prop
        # (source_timestamp) is not a TEXT prop and carries no such flag.
        text_added = [p for p in added if p.data_type == "TEXT"]
        assert {p.name for p in text_added} == set(_DENORM_TEXT_KEYS)
        assert all(p.index_searchable is False for p in text_added)
        assert all(p.index_filterable is True for p in text_added)
        # The DATE prop (source_timestamp) is typed DATE, not TEXT.
        date_added = [p for p in added if p.name == "source_timestamp"]
        assert [p.data_type for p in date_added] == ["DATE"]

    @pytest.mark.asyncio
    async def test_connect_is_noop_when_denorm_properties_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A second connect() where every denorm property already exists adds nothing.
        _weaviate, client = _install_fake_weaviate(monkeypatch)
        client.collections.exists = AsyncMock(return_value=True)

        existing_names = [
            "content",
            "document_id",
            "namespace_id",
            "occurred_at",
            "created_at",
            "source_system",
            "author",
            "channel",
            "tags",
            "confidence",
            "metadata_json",
            *(p.name for p in _denorm_properties()),
        ]
        full_props = [SimpleNamespace(name=name) for name in existing_names]
        collection = MagicMock(name="CollectionAsync")
        collection.config.get = AsyncMock(return_value=SimpleNamespace(properties=full_props))
        collection.config.add_property = AsyncMock()
        client.collections.get = MagicMock(return_value=collection)

        store = _build_store("http://localhost:8090")
        await store.connect()

        collection.config.add_property.assert_not_awaited()


# ---------------------------------------------------------------------------
# search_fulltext
# ---------------------------------------------------------------------------


async def _store_with_bm25(
    monkeypatch: pytest.MonkeyPatch, objects: list[Any]
) -> tuple[WeaviateTemporalStore, MagicMock]:
    """A connected store whose tenant-scoped ``query.bm25`` returns ``objects``."""
    _weaviate, client = _install_fake_weaviate(monkeypatch)
    _install_query_filter(monkeypatch)

    collection = _reconcile_noop_collection()
    collection.tenants.create = AsyncMock()
    tenant_scoped = MagicMock(name="TenantScopedCollection")
    tenant_scoped.query.bm25 = AsyncMock(return_value=SimpleNamespace(objects=objects))
    collection.with_tenant = MagicMock(return_value=tenant_scoped)
    client.collections.get = MagicMock(return_value=collection)

    store = _build_store("http://localhost:8090")
    await store.connect()
    return store, tenant_scoped


def _fake_bm25_object(content: str = "hello world", score: float = 0.42) -> Any:
    """Minimal Weaviate object returned by a BM25 query."""
    ns_id = uuid4()
    doc_id = uuid4()
    return SimpleNamespace(
        uuid=uuid4(),
        properties={
            "content": content,
            "document_id": str(doc_id),
            "namespace_id": str(ns_id),
            "occurred_at": None,
            "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            "source_system": None,
            "author": None,
            "channel": None,
            "tags": [],
            "confidence": 1.0,
            "metadata_json": "{}",
        },
        metadata=SimpleNamespace(score=score, distance=None),
        vector=None,
    )


@pytest.mark.unit
class TestSearchFulltextWeaviate:
    """Unit tests for WeaviateTemporalStore.search_fulltext."""

    def test_search_fulltext_is_overridden(self) -> None:
        """WeaviateTemporalStore must override search_fulltext (not inherit the []
        default from TemporalVectorStore)."""
        from khora.storage.temporal import TemporalVectorStore

        assert WeaviateTemporalStore.search_fulltext is not TemporalVectorStore.search_fulltext

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store, tenant_scoped = await _store_with_bm25(monkeypatch, [])
        result = await store.search_fulltext(uuid4(), "")
        assert result == []
        tenant_scoped.query.bm25.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_chunks_with_scores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        objs = [_fake_bm25_object("foo bar", score=0.9), _fake_bm25_object("baz", score=0.5)]
        store, tenant_scoped = await _store_with_bm25(monkeypatch, objs)

        ns_id = uuid4()
        results = await store.search_fulltext(ns_id, "foo", limit=5)

        assert len(results) == 2
        chunk0, score0 = results[0]
        assert score0 == pytest.approx(0.9)
        assert "foo bar" in chunk0.content
        tenant_scoped.query.bm25.assert_awaited_once()
        kwargs = tenant_scoped.query.bm25.call_args.kwargs
        assert kwargs["query"] == "foo"
        assert kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_passes_created_after_as_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        objs = [_fake_bm25_object()]
        store, tenant_scoped = await _store_with_bm25(monkeypatch, objs)

        cutoff = datetime(2026, 1, 1, tzinfo=UTC)
        await store.search_fulltext(uuid4(), "query", created_after=cutoff)

        kwargs = tenant_scoped.query.bm25.call_args.kwargs
        # A date filter was pushed down
        assert kwargs["filters"] is not None
