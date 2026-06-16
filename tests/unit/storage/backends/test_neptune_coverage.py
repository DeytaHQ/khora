"""Coverage tests for khora.storage.backends.neptune.

Exercises init, URL/SecretStr handling, lifecycle (with both basic-auth
and IAM SigV4 paths mocked), record-to-domain converters, and the
Cypher-building paths for every CRUD / traversal method using a mocked
neo4j async driver.  No real Neptune cluster.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from khora.config.schema import NeptuneConfig
from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.neptune import NeptuneBackend

# the IDOR family/223: read-side methods now require a kwarg-only ``namespace_id`` so
# the backend can scope every Cypher MATCH to the caller's tenant.  Tests use
# this fixed UUID across the file so assertions against the generated query
# parameters can pin the value.
_NS = uuid4()

# ---------------------------------------------------------------------------
# Mock driver / session plumbing
# ---------------------------------------------------------------------------


def _make_session_with_records(
    records: list[dict[str, Any]] | None = None, single: dict[str, Any] | None = None
) -> AsyncMock:
    result = MagicMock()
    result.data = AsyncMock(return_value=records or [])
    result.single = AsyncMock(return_value=single)
    session = AsyncMock()
    session.run = AsyncMock(return_value=result)
    return session


def _make_driver(session: AsyncMock) -> MagicMock:
    driver = MagicMock()

    @asynccontextmanager
    async def _session_ctx():  # type: ignore[no-untyped-def]
        yield session

    driver.session = MagicMock(side_effect=_session_ctx)
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver


def _connected_backend(session: AsyncMock, **kwargs: Any) -> NeptuneBackend:
    """Skip connect() — bolt the mocked driver directly onto the backend."""
    backend = NeptuneBackend("bolt://cluster:8182", **kwargs)
    backend._driver = _make_driver(session)
    return backend


# ---------------------------------------------------------------------------
# __init__ / from_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_stores_attributes() -> None:
    b = NeptuneBackend(
        "bolt://cluster:8182",
        user="u",
        password="p",
        encrypted=False,
        iam_auth=True,
        aws_region="eu-central-1",
        max_connection_pool_size=200,
    )
    assert b._url == "bolt://cluster:8182"
    assert b._user == "u"
    assert b._password == "p"
    assert b._encrypted is False
    assert b._iam_auth is True
    assert b._aws_region == "eu-central-1"
    assert b._max_connection_pool_size == 200
    assert b._driver is None


@pytest.mark.unit
def test_init_defaults() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    assert b._user == ""
    assert b._password == ""
    assert b._encrypted is True
    assert b._iam_auth is False
    assert b._aws_region == "us-east-1"
    assert b._max_connection_pool_size == 100


@pytest.mark.unit
def test_from_config_plain_values() -> None:
    cfg = NeptuneConfig(
        url="bolt://np:8182",
        user="u",
        password="p",
        iam_auth=False,
        aws_region="us-west-2",
        max_connection_pool_size=50,
    )
    b = NeptuneBackend.from_config(cfg)
    assert b._url == "bolt://np:8182"
    assert b._user == "u"
    assert b._password == "p"
    assert b._iam_auth is False
    assert b._aws_region == "us-west-2"
    assert b._max_connection_pool_size == 50


@pytest.mark.unit
def test_from_config_unwraps_secretstr() -> None:
    cfg = NeptuneConfig(url=SecretStr("bolt://secret:8182"), password=SecretStr("hidden"))
    b = NeptuneBackend.from_config(cfg)
    assert b._url == "bolt://secret:8182"
    assert b._password == "hidden"


@pytest.mark.unit
def test_from_config_url_none_defaults_to_localhost() -> None:
    cfg = NeptuneConfig()
    b = NeptuneBackend.from_config(cfg)
    assert b._url == "bolt://localhost:8182"


# ---------------------------------------------------------------------------
# Lifecycle — basic auth (no IAM)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_connect_basic_auth_initializes_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _make_session_with_records()
    fake_driver = _make_driver(fake_session)

    import neo4j

    captured: dict[str, Any] = {}

    def _factory(*a, **kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        return fake_driver

    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", _factory)

    b = NeptuneBackend("bolt://cluster:8182", user="alice", password="pw")
    await b.connect()
    assert b._driver is fake_driver
    # With user set we pass a tuple auth; encrypted=True by default.
    assert captured["auth"] == ("alice", "pw")
    assert captured["encrypted"] is True
    fake_driver.verify_connectivity.assert_awaited()


@pytest.mark.unit
async def test_connect_no_user_passes_none_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _make_session_with_records()
    fake_driver = _make_driver(fake_session)

    import neo4j

    captured: dict[str, Any] = {}

    def _factory(*a, **kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        return fake_driver

    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", _factory)

    b = NeptuneBackend("bolt://cluster:8182")  # user defaults to ""
    await b.connect()
    assert captured["auth"] is None


@pytest.mark.unit
async def test_connect_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_driver = _make_driver(_make_session_with_records())
    b = NeptuneBackend("bolt://cluster:8182")
    b._driver = fake_driver

    import neo4j

    called = []
    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", lambda *a, **kw: called.append(1) or fake_driver)
    await b.connect()
    assert called == []


# ---------------------------------------------------------------------------
# Lifecycle — IAM SigV4 auth path
# ---------------------------------------------------------------------------


def _stub_iam_signing(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the boto3 + botocore surface used by the IAM signing path.

    Returns the ``SigV4Auth`` class mock.  Its instance's ``add_auth`` stamps
    an incrementing ``X-Amz-Date`` header on the request so each produced
    signature is distinguishable in assertions.
    """
    import itertools
    import sys

    fake_creds = MagicMock()
    fake_session_obj = MagicMock()
    fake_session_obj.get_credentials = MagicMock(return_value=MagicMock(get_frozen_credentials=lambda: fake_creds))

    boto3_mod = MagicMock()
    boto3_mod.Session = MagicMock(return_value=fake_session_obj)

    counter = itertools.count(1)

    def _stamp(request: Any) -> None:
        request.headers["X-Amz-Date"] = f"sig-{next(counter)}"

    sigv4_instance = MagicMock()
    sigv4_instance.add_auth = MagicMock(side_effect=_stamp)
    botocore_auth_mod = MagicMock()
    botocore_auth_mod.SigV4Auth = MagicMock(return_value=sigv4_instance)

    botocore_awsrequest_mod = MagicMock()

    class _FakeAWSRequest:
        def __init__(self, *a, **kw):  # type: ignore[no-untyped-def]
            self.headers = dict(kw.get("headers") or {})

    botocore_awsrequest_mod.AWSRequest = _FakeAWSRequest

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "botocore", MagicMock())
    monkeypatch.setitem(sys.modules, "botocore.auth", botocore_auth_mod)
    monkeypatch.setitem(sys.modules, "botocore.awsrequest", botocore_awsrequest_mod)
    return botocore_auth_mod.SigV4Auth


def _connect_kwargs_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the driver factory and return the dict its kwargs land in."""
    import neo4j

    fake_driver = _make_driver(_make_session_with_records())
    captured: dict[str, Any] = {}

    def _factory(*a, **kw):  # type: ignore[no-untyped-def]
        captured.update(kw)
        return fake_driver

    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", _factory)
    return captured


@pytest.mark.unit
async def test_connect_iam_auth_passes_rotating_auth_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """connect() must install a rotating auth manager, not a token signed once.

    Regression for #1152: SigV4 signatures expire after ~5 minutes, so a
    static token breaks every connection the pool opens after that window.
    Signing must be deferred to connection time, not done eagerly at connect.
    """
    captured = _connect_kwargs_capture(monkeypatch)
    sigv4_cls = _stub_iam_signing(monkeypatch)

    b = NeptuneBackend("bolt://cluster:8182", iam_auth=True, aws_region="us-east-1")
    await b.connect()

    from neo4j.auth_management import AsyncAuthManager

    assert isinstance(captured["auth"], AsyncAuthManager)
    # No signature is computed until the driver opens a connection.
    sigv4_cls.assert_not_called()


@pytest.mark.unit
async def test_connect_iam_auth_resigns_per_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connections established at different times carry different signatures.

    The neo4j driver calls ``get_auth()`` on the auth manager for every new
    Bolt connection; once the cached token has expired the provider must
    produce a fresh signature.  The token TTL is forced to -1 so every fetch
    is past expiry - on main, where the token is signed once at connect(),
    both fetches return the same frozen signature.
    """
    import json

    import khora.storage.backends.neptune as neptune_mod

    captured = _connect_kwargs_capture(monkeypatch)
    _stub_iam_signing(monkeypatch)
    monkeypatch.setattr(neptune_mod, "_IAM_TOKEN_TTL_SECONDS", -1.0)

    b = NeptuneBackend("bolt://cluster:8182", iam_auth=True, aws_region="us-east-1")
    await b.connect()

    manager = captured["auth"]
    auth_first = await manager.get_auth()
    auth_second = await manager.get_auth()

    token_first = json.loads(auth_first.credentials)
    token_second = json.loads(auth_second.credentials)
    assert token_first["X-Amz-Date"] == "sig-1"
    assert token_second["X-Amz-Date"] == "sig-2"


@pytest.mark.unit
async def test_connect_iam_auth_reuses_token_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Within the token TTL the cached signature is reused (no re-sign per fetch)."""
    import json

    captured = _connect_kwargs_capture(monkeypatch)
    sigv4_cls = _stub_iam_signing(monkeypatch)

    b = NeptuneBackend("bolt://cluster:8182", iam_auth=True, aws_region="us-east-1")
    await b.connect()

    manager = captured["auth"]
    auth_first = await manager.get_auth()
    auth_second = await manager.get_auth()

    sigv4_cls.assert_called_once()
    assert json.loads(auth_first.credentials) == json.loads(auth_second.credentials)


@pytest.mark.unit
async def test_connect_iam_auth_token_matches_bolt_iam_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """The token JSON carries the signed headers plus HttpMethod.

    AWS's documented Neptune Bolt-IAM token format requires ``HttpMethod``
    alongside the SigV4-signed headers; without it Neptune rejects the
    connection outright.
    """
    import json

    captured = _connect_kwargs_capture(monkeypatch)
    _stub_iam_signing(monkeypatch)

    b = NeptuneBackend("bolt://cluster:8182", iam_auth=True, aws_region="us-east-1")
    await b.connect()

    auth = await captured["auth"].get_auth()
    token = json.loads(auth.credentials)
    assert token["HttpMethod"] == "GET"
    assert token["Host"] == "cluster"
    assert token["X-Amz-Date"] == "sig-1"


@pytest.mark.unit
async def test_connect_iam_auth_missing_boto3_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ImportError handler re-raises with a helpful pip-install hint."""
    import sys

    # Force ImportError when boto3 is imported.  Use a tombstone in sys.modules
    # so ``import boto3`` raises rather than succeeds.
    class _Boom:
        def __getattr__(self, name: str) -> Any:
            raise ImportError("no boto3")

    monkeypatch.setitem(sys.modules, "boto3", _Boom())

    # ``import boto3`` inside connect() succeeds (returns the tombstone), but
    # ``boto3.Session(...)`` raises.  The except ImportError branch only
    # captures the literal ImportError at import time — we need to actually
    # raise ImportError from the import statement itself.  Easiest path:
    # remove `boto3` from sys.modules and inject a meta-path finder that
    # rejects it.
    sys.modules.pop("boto3", None)

    import importlib.abc
    import importlib.machinery

    class _Reject(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
            if fullname == "boto3":
                raise ImportError("boto3 missing for test")
            return None

    finder = _Reject()
    sys.meta_path.insert(0, finder)
    try:
        b = NeptuneBackend("bolt://cluster:8182", iam_auth=True)
        with pytest.raises(ImportError, match="boto3 is required"):
            await b.connect()
    finally:
        sys.meta_path.remove(finder)


# ---------------------------------------------------------------------------
# Disconnect / health
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_disconnect_closes_and_clears() -> None:
    fake_driver = _make_driver(_make_session_with_records())
    b = NeptuneBackend("bolt://cluster:8182")
    b._driver = fake_driver
    await b.disconnect()
    fake_driver.close.assert_awaited()
    assert b._driver is None


@pytest.mark.unit
async def test_disconnect_noop_when_disconnected() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    await b.disconnect()
    assert b._driver is None


@pytest.mark.unit
async def test_is_healthy_false_when_disconnected() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    assert await b.is_healthy() is False


@pytest.mark.unit
async def test_is_healthy_true_on_success() -> None:
    fake_driver = _make_driver(_make_session_with_records())
    b = NeptuneBackend("bolt://cluster:8182")
    b._driver = fake_driver
    assert await b.is_healthy() is True


@pytest.mark.unit
async def test_is_healthy_false_on_error() -> None:
    fake_driver = _make_driver(_make_session_with_records())
    fake_driver.verify_connectivity = AsyncMock(side_effect=RuntimeError("network"))
    b = NeptuneBackend("bolt://cluster:8182")
    b._driver = fake_driver
    assert await b.is_healthy() is False


@pytest.mark.unit
def test_get_driver_raises_when_disconnected() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    with pytest.raises(RuntimeError, match="not connected"):
        b._get_driver()


@pytest.mark.unit
async def test_create_indexes_is_noop() -> None:
    """Neptune auto-indexes — _create_indexes() should just log and return."""
    b = NeptuneBackend("bolt://cluster:8182")
    # No exception, no driver needed.
    await b._create_indexes()


# ---------------------------------------------------------------------------
# Record-to-domain converters
# ---------------------------------------------------------------------------


def _entity_node(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "name": "Alice",
        "entity_type": "PERSON",
        "description": "",
        "attributes": "{}",
        "source_document_ids": [],
        "source_chunk_ids": [],
        "mention_count": 1,
        "confidence": 1.0,
        "metadata": "{}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_record_to_entity() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    ent = b._record_to_entity(_entity_node(name="Bob"))
    assert isinstance(ent, Entity)
    assert ent.name == "Bob"


@pytest.mark.unit
def test_record_to_entity_with_valid_window() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    ent = b._record_to_entity(
        _entity_node(
            valid_from="2026-01-01T00:00:00+00:00",
            valid_until="2026-12-31T00:00:00+00:00",
        )
    )
    assert ent.valid_from is not None and ent.valid_until is not None


@pytest.mark.unit
def test_record_to_entity_missing_timestamps_uses_now() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    node = _entity_node()
    del node["created_at"]
    del node["updated_at"]
    ent = b._record_to_entity(node)
    assert isinstance(ent.created_at, datetime)


@pytest.mark.unit
def test_record_to_relationship() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    rel = b._record_to_relationship(
        {
            "id": str(uuid4()),
            "namespace_id": str(uuid4()),
            "description": "",
            "properties": "{}",
            "source_document_ids": [],
            "source_chunk_ids": [],
            "confidence": 1.0,
            "weight": 1.0,
            "metadata": "{}",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        str(uuid4()),
        str(uuid4()),
        "WORKS_AT",
    )
    assert isinstance(rel, Relationship)
    assert rel.relationship_type == "WORKS_AT"


@pytest.mark.unit
def test_record_to_episode() -> None:
    b = NeptuneBackend("bolt://cluster:8182")
    ep_node = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "name": "ep",
        "description": "",
        "occurred_at": "2026-01-15T10:00:00+00:00",
        "duration_seconds": 60,
        "entity_ids": [],
        "source_document_ids": [],
        "source_chunk_ids": [],
        "metadata": "{}",
        "created_at": "2026-01-15T10:00:00+00:00",
        "updated_at": "2026-01-15T10:00:00+00:00",
    }
    ep = b._record_to_episode(ep_node)
    assert isinstance(ep, Episode)
    assert ep.duration_seconds == 60


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_entity_sends_expected_params() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    ent = Entity(name="Bob", entity_type="PERSON")
    result = await b.create_entity(ent)
    assert result is ent
    cypher = session.run.await_args.args[0]
    assert "CREATE (e:Entity" in cypher
    kwargs = session.run.await_args.kwargs
    assert kwargs["name"] == "Bob"
    assert kwargs["entity_type"] == "PERSON"


@pytest.mark.unit
async def test_get_entity_missing_returns_none() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    assert await b.get_entity(uuid4(), namespace_id=_NS) is None


@pytest.mark.unit
async def test_get_entity_returns_domain_model() -> None:
    node = _entity_node()
    session = _make_session_with_records(single={"e": node})
    b = _connected_backend(session)
    got = await b.get_entity(UUID(node["id"]), namespace_id=UUID(node["namespace_id"]))
    assert got is not None
    assert got.name == "Alice"


@pytest.mark.unit
async def test_get_entity_by_name_missing() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    assert await b.get_entity_by_name(uuid4(), "x", "y") is None


@pytest.mark.unit
async def test_get_entity_by_name_found() -> None:
    node = _entity_node()
    session = _make_session_with_records(single={"e": node})
    b = _connected_backend(session)
    got = await b.get_entity_by_name(UUID(node["namespace_id"]), node["name"], node["entity_type"])
    assert got is not None


@pytest.mark.unit
async def test_get_entities_batch_empty_returns_empty_dict() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    result = await b.get_entities_batch([], namespace_id=_NS)
    assert result == {}
    session.run.assert_not_called()


@pytest.mark.unit
async def test_get_entities_batch_returns_mapping() -> None:
    node = _entity_node()
    session = _make_session_with_records(records=[{"e": node}])
    b = _connected_backend(session)
    result = await b.get_entities_batch([UUID(node["id"])], namespace_id=UUID(node["namespace_id"]))
    assert UUID(node["id"]) in result


@pytest.mark.unit
async def test_update_entity_emits_set_clause() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    ent = Entity(name="Bob", entity_type="PERSON")
    result = await b.update_entity(ent, namespace_id=ent.namespace_id)
    assert result is ent
    cypher = session.run.await_args.args[0]
    assert "SET" in cypher and "e.name = $name" in cypher


@pytest.mark.unit
async def test_delete_entity_true_when_deleted() -> None:
    session = _make_session_with_records(single={"deleted": 3})
    b = _connected_backend(session)
    assert await b.delete_entity(uuid4(), namespace_id=uuid4()) is True


@pytest.mark.unit
async def test_delete_entity_false_when_missing() -> None:
    session = _make_session_with_records(single={"deleted": 0})
    b = _connected_backend(session)
    assert await b.delete_entity(uuid4(), namespace_id=uuid4()) is False


@pytest.mark.unit
async def test_delete_entity_false_when_no_record() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    assert await b.delete_entity(uuid4(), namespace_id=uuid4()) is False


@pytest.mark.unit
async def test_list_entities_default_query() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.list_entities(uuid4())
    assert out == []
    cypher = session.run.await_args.args[0]
    assert "WHERE" not in cypher
    assert "SKIP $offset" in cypher and "LIMIT $limit" in cypher


@pytest.mark.unit
async def test_list_entities_with_type_filter() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    await b.list_entities(uuid4(), entity_type="PERSON")
    cypher = session.run.await_args.args[0]
    assert "WHERE e.entity_type" in cypher


@pytest.mark.unit
async def test_count_entities_returns_value() -> None:
    session = _make_session_with_records(single={"cnt": 7})
    b = _connected_backend(session)
    assert await b.count_entities(uuid4()) == 7


@pytest.mark.unit
async def test_count_entities_returns_zero_when_no_record() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    assert await b.count_entities(uuid4()) == 0


@pytest.mark.unit
async def test_count_relationships_raises_not_implemented() -> None:
    b = _connected_backend(_make_session_with_records())
    with pytest.raises(NotImplementedError):
        await b.count_relationships(uuid4())


# ---------------------------------------------------------------------------
# Relationship operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_relationship_sanitizes_label() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    rel = Relationship(relationship_type="reports to")
    result = await b.create_relationship(rel)
    assert result is rel
    cypher = session.run.await_args.args[0]
    assert "REPORTS_TO" in cypher


@pytest.mark.unit
async def test_get_relationship_missing_returns_none() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    assert await b.get_relationship(uuid4(), namespace_id=_NS) is None


@pytest.mark.unit
async def test_get_relationship_returns_domain_model() -> None:
    rel_props = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "description": "",
        "properties": "{}",
        "source_document_ids": [],
        "source_chunk_ids": [],
        "confidence": 1.0,
        "weight": 1.0,
        "metadata": "{}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    session = _make_session_with_records(
        single={
            "r": rel_props,
            "source_id": str(uuid4()),
            "target_id": str(uuid4()),
            "rel_type": "KNOWS",
        }
    )
    b = _connected_backend(session)
    got = await b.get_relationship(uuid4(), namespace_id=_NS)
    assert got is not None and got.relationship_type == "KNOWS"


@pytest.mark.unit
async def test_delete_relationship_true_when_deleted() -> None:
    session = _make_session_with_records(single={"deleted": 1})
    b = _connected_backend(session)
    assert await b.delete_relationship(uuid4(), namespace_id=uuid4()) is True


@pytest.mark.unit
async def test_delete_relationship_false_when_missing() -> None:
    session = _make_session_with_records(single={"deleted": 0})
    b = _connected_backend(session)
    assert await b.delete_relationship(uuid4(), namespace_id=uuid4()) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "direction,expected_fragment",
    [
        # Security: each pattern node carries ``{namespace_id: $namespace_id}``
        # so the legacy ``(e)-[r`` form no longer appears.  Pin on the direction
        # arrow instead.
        ("outgoing", "]->(other:Entity"),
        ("incoming", "]->(e:Entity"),
        ("both", "]-(other:Entity"),
    ],
)
async def test_get_entity_relationships_direction(direction: str, expected_fragment: str) -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.get_entity_relationships(uuid4(), namespace_id=_NS, direction=direction)
    assert out == []
    cypher = session.run.await_args.args[0]
    assert expected_fragment in cypher
    # Bound parameter carries the per-tenant namespace.
    assert session.run.await_args.kwargs.get("namespace_id") == str(_NS)


@pytest.mark.unit
async def test_get_entity_relationships_rel_types_join() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    await b.get_entity_relationships(uuid4(), namespace_id=_NS, relationship_types=["a", "b"])
    cypher = session.run.await_args.args[0]
    assert "A|B" in cypher


@pytest.mark.unit
async def test_list_relationships_no_filter() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.list_relationships(uuid4())
    assert out == []
    cypher = session.run.await_args.args[0]
    assert "[r]" in cypher


@pytest.mark.unit
async def test_list_relationships_with_type_filter() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    await b.list_relationships(uuid4(), relationship_type="OWNS")
    cypher = session.run.await_args.args[0]
    assert "[r:OWNS]" in cypher


# ---------------------------------------------------------------------------
# Episode operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_episode_no_entities_single_call() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    ep = Episode(name="x", occurred_at=datetime.now(UTC))
    result = await b.create_episode(ep)
    assert result is ep
    assert session.run.await_count == 1


@pytest.mark.unit
async def test_create_episode_with_entity_ids_emits_involves() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    ep = Episode(name="x", occurred_at=datetime.now(UTC), entity_ids=[uuid4()])
    await b.create_episode(ep)
    assert session.run.await_count == 2
    second_call_cypher = session.run.await_args_list[1].args[0]
    assert "INVOLVES" in second_call_cypher


@pytest.mark.unit
async def test_get_episode_missing_returns_none() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    assert await b.get_episode(uuid4(), namespace_id=_NS) is None


@pytest.mark.unit
async def test_get_episode_returns_domain_model() -> None:
    ep_node = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "name": "ep",
        "description": "",
        "occurred_at": "2026-01-15T10:00:00+00:00",
        "duration_seconds": None,
        "entity_ids": [],
        "source_document_ids": [],
        "source_chunk_ids": [],
        "metadata": "{}",
        "created_at": "2026-01-15T10:00:00+00:00",
        "updated_at": "2026-01-15T10:00:00+00:00",
    }
    session = _make_session_with_records(single={"ep": ep_node})
    b = _connected_backend(session)
    got = await b.get_episode(UUID(ep_node["id"]), namespace_id=UUID(ep_node["namespace_id"]))
    assert got is not None


@pytest.mark.unit
async def test_list_episodes_no_filters() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.list_episodes(uuid4())
    assert out == []
    cypher = session.run.await_args.args[0]
    assert "WHERE" not in cypher


@pytest.mark.unit
async def test_list_episodes_with_time_filters() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    await b.list_episodes(
        uuid4(),
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 12, 31, tzinfo=UTC),
    )
    cypher = session.run.await_args.args[0]
    assert "ep.occurred_at >= $start_time" in cypher
    assert "ep.occurred_at <= $end_time" in cypher


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_find_paths_empty_result() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    paths = await b.find_paths(uuid4(), uuid4(), namespace_id=uuid4())
    assert paths == []


@pytest.mark.unit
async def test_find_paths_with_filter_and_depth() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    await b.find_paths(uuid4(), uuid4(), namespace_id=uuid4(), relationship_types=["KNOWS"], max_depth=4)
    cypher = session.run.await_args.args[0]
    assert ":KNOWS" in cypher
    assert "*1..4" in cypher


@pytest.mark.unit
async def test_get_neighborhood_empty() -> None:
    session = _make_session_with_records(single={"nodes": [], "relationships": []})
    b = _connected_backend(session)
    result = await b.get_neighborhood(uuid4(), namespace_id=_NS)
    assert result == {"entities": [], "relationships": []}


@pytest.mark.unit
async def test_get_neighborhood_returns_none_record() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    result = await b.get_neighborhood(uuid4(), namespace_id=_NS)
    assert result == {"entities": [], "relationships": []}


@pytest.mark.unit
async def test_get_neighborhood_with_rel_types() -> None:
    session = _make_session_with_records(single={"nodes": [], "relationships": []})
    b = _connected_backend(session)
    await b.get_neighborhood(uuid4(), namespace_id=_NS, relationship_types=["RELATES"])
    cypher = session.run.await_args.args[0]
    assert ":RELATES" in cypher


# ---------------------------------------------------------------------------
# search_entities_by_attribute
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_search_entities_by_attribute_prefilters_on_key() -> None:
    """#1153: ``attributes`` is a JSON string, so the query CONTAINS-prefilters
    on the serialized key and the exact value match happens in Python."""
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.search_entities_by_attribute(uuid4(), "role", "admin", limit=20)
    assert out == []
    cypher = session.run.await_args.args[0]
    assert "e.attributes[$attribute_name]" not in cypher
    assert "CONTAINS $key_pattern" in cypher
    assert session.run.await_args.kwargs["key_pattern"] == '"role"'


def _rel_props() -> dict[str, Any]:
    """Minimal post-fix relationship properties dict (the `properties(r)` shape)."""
    return {
        "id": str(uuid4()),
        "namespace_id": str(_NS),
        "description": "edge",
        "properties": "{}",
        "source_document_ids": [],
        "source_chunk_ids": [],
        "confidence": 1.0,
        "weight": 1.0,
        "metadata": "{}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-02T00:00:00+00:00",
    }


@pytest.mark.unit
@pytest.mark.security
class TestNeptuneListRelationshipsScoping:
    """list_relationships endpoint scoping + null-endpoint guard (#1238, port of #1237)."""

    @pytest.mark.asyncio
    async def test_constrains_both_endpoints_to_namespace(self) -> None:
        session = _make_session_with_records(records=[])
        b = _connected_backend(session)
        await b.list_relationships(_NS)
        query = session.run.await_args.args[0]
        assert "(source:Entity {namespace_id: $namespace_id})" in query
        assert "(target:Entity {namespace_id: $namespace_id})" in query
        # Negative check: the pre-fix unlabeled endpoints must be gone.
        assert "MATCH (source)-[r" not in query

    @pytest.mark.asyncio
    async def test_skips_rows_with_null_endpoint_without_raising(self) -> None:
        good = {"rel_props": _rel_props(), "source_id": str(uuid4()), "target_id": str(uuid4()), "rel_type": "KNOWS"}
        bad = {"rel_props": _rel_props(), "source_id": None, "target_id": str(uuid4()), "rel_type": "KNOWS"}
        session = _make_session_with_records(records=[good, bad])
        b = _connected_backend(session)
        rels = await b.list_relationships(_NS)
        assert len(rels) == 1
        assert all(isinstance(r, Relationship) for r in rels)

    def test_record_to_relationship_null_endpoint_returns_none(self) -> None:
        b = NeptuneBackend("bolt://cluster:8182")
        assert b._record_to_relationship(_rel_props(), None, str(uuid4()), "KNOWS") is None
        assert b._record_to_relationship(_rel_props(), str(uuid4()), None, "KNOWS") is None

    def test_record_to_relationship_filters_null_provenance_elements(self) -> None:
        b = NeptuneBackend("bolt://cluster:8182")
        good = str(uuid4())
        rel = b._record_to_relationship(
            dict(_rel_props(), source_document_ids=[good, None], source_chunk_ids=[None]),
            str(uuid4()),
            str(uuid4()),
            "WORKS_FOR",
        )
        assert rel is not None
        assert [str(d) for d in rel.source_document_ids] == [good]
        assert rel.source_chunk_ids == []

    def test_record_to_relationship_synthesizes_missing_id_and_namespace(self) -> None:
        """Missing edge id/namespace_id are synthesized + warned (porting #767), not crashed on.

        Neptune previously did ``UUID(rel["id"])`` (KeyError on a missing id);
        the row is now kept with a synthesized identity.
        """
        b = NeptuneBackend("bolt://cluster:8182")
        rel = b._record_to_relationship(
            dict(_rel_props(), id=None, namespace_id=None),
            str(uuid4()),
            str(uuid4()),
            "KNOWS",
        )
        assert rel is not None

    @pytest.mark.asyncio
    async def test_relationship_type_filter_applied(self) -> None:
        """``relationship_type`` is injected as ``:TYPE`` into the constrained pattern."""
        session = _make_session_with_records(records=[])
        b = _connected_backend(session)
        await b.list_relationships(_NS, relationship_type="KNOWS")
        query = session.run.await_args.args[0]
        assert ":KNOWS]" in query
