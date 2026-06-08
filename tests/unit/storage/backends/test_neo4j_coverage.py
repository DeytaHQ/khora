"""Unit coverage for ``khora.storage.backends.neo4j``.

Most Neo4j paths require a real database (Cypher MERGE, transactions,
session pooling). This file covers the unit-testable surfaces:

- Pure helpers: ``_sanitize_neo4j_label``, ``_derive_version_valid_from``,
  ``_entity_to_cypher_params``, ``_relationship_to_cypher_params``,
  ``BIDIRECTIONAL_TYPES`` table shape.
- ``_safe_url_for_log`` redaction.
- ``_EntityKeyGate`` concurrency primitive.
- ``_InstrumentedSession`` constructor + ``_install_connect_wrap``.
- ``Neo4jBackend.__init__`` defaults / clamps.
- ``from_driver`` / ``from_config`` factory paths.
- ``disconnect()`` paths (owned vs shared driver).
- ``is_healthy()`` disconnected branch.
- ``_get_driver`` raises when ``None``.
- Empty-input short-circuits on ``delete_*_batch`` /
  ``remove_document_from_*_sources_batch`` /
  ``remap_source_document_ids_batch`` / ``upsert_entities_batch``.
- ``_record_to_entity`` / ``_record_to_relationship`` / ``_record_to_episode``
  row converters.

No real Neo4j is started. Driver and session are mocked.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.storage._log_safe import _safe_url_for_log
from khora.storage.backends.neo4j import (
    _DEFAULT_ENTITY_WRITE_CONCURRENCY,
    _DEFAULT_RELATIONSHIP_WRITE_CONCURRENCY,
    _NEO4J_TIMEOUT_CODES,
    BIDIRECTIONAL_TYPES,
    Neo4jBackend,
    _cancel_sampler_task_on_gc,
    _derive_version_valid_from,
    _entity_to_cypher_params,
    _EntityKeyGate,
    _InstrumentedSession,
    _relationship_to_cypher_params,
    _sanitize_neo4j_label,
)

# ---------------------------------------------------------------------------
# _sanitize_neo4j_label
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSanitizeNeo4jLabel:
    def test_clean_label_upper_cased(self) -> None:
        assert _sanitize_neo4j_label("works_for") == "WORKS_FOR"

    def test_non_alphanumeric_replaced(self) -> None:
        assert _sanitize_neo4j_label("at-risk") == "AT_RISK"
        assert _sanitize_neo4j_label("works for") == "WORKS_FOR"

    def test_trims_whitespace(self) -> None:
        assert _sanitize_neo4j_label("  knows  ") == "KNOWS"

    def test_empty_falls_back_to_relates_to(self) -> None:
        assert _sanitize_neo4j_label("") == "RELATES_TO"

    def test_only_special_chars_falls_back(self) -> None:
        assert _sanitize_neo4j_label("!@#$%") == "_____"  # alphanumerics replaced
        # Whitespace-only collapses to empty (after .strip()) -> RELATES_TO
        assert _sanitize_neo4j_label("    ") == "RELATES_TO"

    def test_numbers_preserved(self) -> None:
        assert _sanitize_neo4j_label("v2_works_for") == "V2_WORKS_FOR"


# ---------------------------------------------------------------------------
# BIDIRECTIONAL_TYPES contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBidirectionalTypesTable:
    def test_inverse_pairs_consistent(self) -> None:
        """Each (A, B) inverse pair must also have the (B, A) inverse declared,
        and the round-trip must return A."""
        for forward, reverse in BIDIRECTIONAL_TYPES.items():
            # Self-symmetric (e.g. COLLABORATES_WITH -> COLLABORATES_WITH) is fine.
            assert reverse in BIDIRECTIONAL_TYPES, f"{reverse} missing as forward"
            assert BIDIRECTIONAL_TYPES[reverse] == forward, (
                f"non-involutive: {forward} -> {reverse} -> {BIDIRECTIONAL_TYPES[reverse]}"
            )

    def test_table_is_not_empty(self) -> None:
        # Sanity check the table exists and includes the common cases.
        assert "MANAGES" in BIDIRECTIONAL_TYPES
        assert "WORKS_FOR" in BIDIRECTIONAL_TYPES
        assert "OWNS" in BIDIRECTIONAL_TYPES


# ---------------------------------------------------------------------------
# _safe_url_for_log
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSafeUrlForLog:
    def test_no_credentials_passes_through(self) -> None:
        assert _safe_url_for_log("bolt://localhost:7687") == "bolt://localhost:7687"

    def test_user_only_redacted(self) -> None:
        out = _safe_url_for_log("bolt://neo4j@localhost:7687")
        assert "neo4j" not in out
        assert "<redacted>" in out

    def test_password_redacted(self) -> None:
        out = _safe_url_for_log("bolt://neo4j:secret@localhost:7687")
        assert "secret" not in out
        assert "<redacted>" in out

    def test_host_and_port_preserved(self) -> None:
        out = _safe_url_for_log("neo4j://user:pw@neo4j.example:7687/db")
        assert "neo4j.example:7687" in out


# ---------------------------------------------------------------------------
# _derive_version_valid_from
# ---------------------------------------------------------------------------


def _entity(**kwargs) -> Entity:
    base = {
        "namespace_id": uuid4(),
        "name": "Test",
        "entity_type": "CONCEPT",
    }
    base.update(kwargs)
    return Entity(**base)


@pytest.mark.unit
class TestDeriveVersionValidFrom:
    def test_occurred_at_string_returned_verbatim(self) -> None:
        e = _entity(metadata={"occurred_at": "2024-03-15T00:00:00+00:00"})
        out = _derive_version_valid_from(e)
        assert out == "2024-03-15T00:00:00+00:00"

    def test_occurred_at_datetime_converted_to_iso(self) -> None:
        dt = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
        e = _entity(metadata={"occurred_at": dt})
        out = _derive_version_valid_from(e)
        assert out == dt.isoformat()

    def test_falls_back_to_created_at_metadata_key(self) -> None:
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        e = _entity(metadata={"created_at": dt})
        out = _derive_version_valid_from(e)
        assert out == dt.isoformat()

    def test_falls_back_to_entity_created_at(self) -> None:
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        e = _entity(created_at=dt)
        out = _derive_version_valid_from(e)
        assert out == dt.isoformat()

    def test_occurred_at_wins_over_created_at(self) -> None:
        occ = datetime(2024, 6, 1, tzinfo=UTC)
        cre = datetime(2024, 1, 1, tzinfo=UTC)
        e = _entity(metadata={"occurred_at": occ, "created_at": cre})
        out = _derive_version_valid_from(e)
        assert out == occ.isoformat()


# ---------------------------------------------------------------------------
# _entity_to_cypher_params / _relationship_to_cypher_params smoke (existing
# tests cover the field-by-field shape; these add metadata/version handling).
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityToCypherParamsVersionField:
    def test_version_valid_from_from_metadata(self) -> None:
        e = _entity(metadata={"occurred_at": "2024-04-01T00:00:00+00:00"})
        params = _entity_to_cypher_params(e)
        assert params["version_valid_from"] == "2024-04-01T00:00:00+00:00"
        assert params["version_valid_to"] is None


@pytest.mark.unit
class TestRelationshipToCypherParams:
    def test_basic_fields_present(self) -> None:
        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="WORKS_FOR",
        )
        params = _relationship_to_cypher_params(rel)
        assert "id" in params
        assert "namespace_id" in params
        assert "source_id" in params
        assert "target_id" in params
        # Properties dict is JSON-serialized.
        assert isinstance(params["properties"], str)
        # Empty dict serializes to "{}"
        assert json.loads(params["properties"]) == {}


# ---------------------------------------------------------------------------
# _cancel_sampler_task_on_gc
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCancelSamplerTaskOnGc:
    def test_noop_when_task_already_done(self) -> None:
        task = MagicMock()
        task.done.return_value = True
        # Should not call cancel
        _cancel_sampler_task_on_gc(task)
        task.cancel.assert_not_called()

    def test_cancels_running_task(self) -> None:
        task = MagicMock()
        task.done.return_value = False
        _cancel_sampler_task_on_gc(task)
        task.cancel.assert_called_once()

    def test_swallows_cancel_exception(self) -> None:
        """Finalizer must never raise."""
        task = MagicMock()
        task.done.return_value = False
        task.cancel.side_effect = RuntimeError("loop closed")
        _cancel_sampler_task_on_gc(task)  # no raise


# ---------------------------------------------------------------------------
# _EntityKeyGate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityKeyGate:
    @pytest.mark.asyncio
    async def test_bypass_yields_without_tracking(self) -> None:
        gate = _EntityKeyGate(max_concurrent=1)
        # bypass=True short-circuits all tracking.
        async with gate.acquire([], bypass=True):
            assert gate._active == 0
            assert gate._in_flight == set()

    @pytest.mark.asyncio
    async def test_acquires_and_releases_single_batch(self) -> None:
        gate = _EntityKeyGate(max_concurrent=4)
        ns = uuid4()
        entities = [SimpleNamespace(namespace_id=ns, name="alice", entity_type="PERSON")]
        async with gate.acquire(entities):
            assert gate._active == 1
            assert (str(ns), "alice", "PERSON") in gate._in_flight
        # After release
        assert gate._active == 0
        assert gate._in_flight == set()

    @pytest.mark.asyncio
    async def test_overlapping_keys_serialized(self) -> None:
        """When two coroutines want the same key, the second must wait."""
        gate = _EntityKeyGate(max_concurrent=4)
        ns = uuid4()
        ent = SimpleNamespace(namespace_id=ns, name="alice", entity_type="PERSON")
        evt_first_acquired = asyncio.Event()
        evt_release_first = asyncio.Event()
        order: list[str] = []

        async def first() -> None:
            async with gate.acquire([ent]):
                evt_first_acquired.set()
                order.append("first-in")
                await evt_release_first.wait()
                order.append("first-out")

        async def second() -> None:
            await evt_first_acquired.wait()
            async with gate.acquire([ent]):
                order.append("second-in")

        t1 = asyncio.create_task(first())
        t2 = asyncio.create_task(second())
        # Give second() time to attempt acquire and block.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        evt_release_first.set()
        await asyncio.gather(t1, t2)
        # Second must enter strictly after first exits.
        assert order == ["first-in", "first-out", "second-in"]

    @pytest.mark.asyncio
    async def test_max_concurrency_limit(self) -> None:
        gate = _EntityKeyGate(max_concurrent=1)
        ns = uuid4()
        ent_a = SimpleNamespace(namespace_id=ns, name="alice", entity_type="PERSON")
        ent_b = SimpleNamespace(namespace_id=ns, name="bob", entity_type="PERSON")
        # Non-overlapping keys but the concurrency cap forces serial.
        order: list[str] = []
        first_in = asyncio.Event()
        release = asyncio.Event()

        async def coro(name: str, ent) -> None:
            async with gate.acquire([ent]):
                order.append(f"{name}-in")
                if name == "first":
                    first_in.set()
                    await release.wait()

        t1 = asyncio.create_task(coro("first", ent_a))
        await first_in.wait()
        t2 = asyncio.create_task(coro("second", ent_b))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(t1, t2)
        # second-in only after first-in regardless of (alice, bob) being
        # distinct keys, because max_concurrent=1.
        assert order == ["first-in", "second-in"]


# ---------------------------------------------------------------------------
# _InstrumentedSession — constructor + _install_connect_wrap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInstrumentedSession:
    def test_attrs_set_on_init(self) -> None:
        inner = MagicMock(spec=[])
        # inner has no _connect — wrap installation will no-op.
        hist = MagicMock()
        ctr = MagicMock()
        sess = _InstrumentedSession(inner, acquire_histogram=hist, timeout_counter=ctr)
        assert sess._inner is inner
        assert sess.counted_timeout is False
        assert sess.last_acquire == 0.0
        assert sess.slow_acquire_threshold_exceeded is False

    def test_install_skips_when_no_connect(self) -> None:
        """When the underlying session has no ``_connect`` attribute, the
        installer falls through silently and the histogram stays unused."""
        inner = MagicMock(spec=[])  # explicit empty spec -> no _connect attr
        hist = MagicMock()
        ctr = MagicMock()
        _InstrumentedSession(inner, acquire_histogram=hist, timeout_counter=ctr)
        hist.record.assert_not_called()

    @pytest.mark.asyncio
    async def test_timed_connect_records_histogram_on_success(self) -> None:
        inner = MagicMock()
        # Original _connect — succeeds asynchronously.
        inner._connect = AsyncMock(return_value="ok")
        hist = MagicMock()
        ctr = MagicMock()
        _InstrumentedSession(inner, acquire_histogram=hist, timeout_counter=ctr)
        # After wrapping, calling inner._connect goes through the wrapper.
        result = await inner._connect()
        assert result == "ok"
        hist.record.assert_called_once()
        ctr.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_timed_connect_counts_timeout(self) -> None:
        from neo4j.exceptions import ConnectionAcquisitionTimeoutError

        inner = MagicMock()
        inner._connect = AsyncMock(side_effect=ConnectionAcquisitionTimeoutError("deadline"))
        hist = MagicMock()
        ctr = MagicMock()
        sess = _InstrumentedSession(inner, acquire_histogram=hist, timeout_counter=ctr)
        with pytest.raises(ConnectionAcquisitionTimeoutError):
            await inner._connect()
        ctr.add.assert_called_once_with(1)
        # Histogram MUST NOT be recorded on timeout (keeps p99 tail honest).
        hist.record.assert_not_called()
        assert sess.counted_timeout is True


# ---------------------------------------------------------------------------
# Neo4jBackend.__init__ defaults and clamps
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jBackendInitDefaults:
    def test_default_url_and_creds_stored(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        assert b._url == "bolt://localhost:7687"
        assert b._user == "neo4j"
        assert b._password == ""
        assert b._database == "neo4j"

    def test_pool_size_default(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        assert b._max_connection_pool_size == 100

    def test_entity_concurrency_default(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        assert b._entity_key_gate._max_concurrent == _DEFAULT_ENTITY_WRITE_CONCURRENCY
        assert b._relationship_write_sem._value == _DEFAULT_RELATIONSHIP_WRITE_CONCURRENCY

    def test_sampler_interval_clamped_low(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687", pool_sampler_interval_ms=1)
        assert b._pool_sampler_interval_ms == 50

    def test_sampler_interval_clamped_high(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687", pool_sampler_interval_ms=1_000_000)
        assert b._pool_sampler_interval_ms == 60_000

    def test_sampler_interval_within_range(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687", pool_sampler_interval_ms=2000)
        assert b._pool_sampler_interval_ms == 2000

    def test_owns_driver_true_by_default(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        assert b._owns_driver is True
        assert b._driver is None
        assert b._sampler_task is None

    def test_metrics_initialised(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        # Each metric attribute should be set (real or no-op instrument).
        assert b._acquire_duration_histogram is not None
        assert b._timeout_counter is not None
        assert b._session_duration_histogram is not None
        assert b._pool_metrics_registered is False


# ---------------------------------------------------------------------------
# from_driver — does not own the driver
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromDriver:
    def test_does_not_own_driver(self) -> None:
        driver = MagicMock()
        b = Neo4jBackend.from_driver(driver)
        assert b._driver is driver
        assert b._owns_driver is False
        assert b._database == "neo4j"

    def test_custom_database(self) -> None:
        b = Neo4jBackend.from_driver(MagicMock(), database="custom")
        assert b._database == "custom"

    def test_custom_concurrency(self) -> None:
        b = Neo4jBackend.from_driver(
            MagicMock(),
            entity_write_concurrency=4,
            relationship_write_concurrency=2,
        )
        assert b._entity_key_gate._max_concurrent == 4
        assert b._relationship_write_sem._value == 2

    def test_sampler_explicitly_disabled_on_from_driver(self) -> None:
        b = Neo4jBackend.from_driver(MagicMock())
        # from_driver leaves the sampler off by default (no pool_sampler_enabled arg here).
        assert b._pool_sampler_enabled is False
        assert b._sampler_task is None


# ---------------------------------------------------------------------------
# from_config — SecretStr unwrap + getattr fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromConfig:
    @staticmethod
    def _base_cfg() -> SimpleNamespace:
        """Build a fully-populated config object — SimpleNamespace, not
        MagicMock, so ``int`` and ``float`` fields stay typed (asyncio.Semaphore
        rejects MagicMock values)."""
        return SimpleNamespace(
            url="bolt://localhost:7687",
            user="u",
            password="pw",
            database="neo4j",
            max_connection_pool_size=100,
            connection_acquisition_timeout=60.0,
            retry_delay_jitter_factor=0.5,
            max_connection_lifetime=900,
            liveness_check_timeout=30.0,
            query_timeout=5.0,
            entity_write_concurrency=4,
            relationship_write_concurrency=2,
            pool_sampler_enabled=False,
            pool_sampler_interval_ms=500,
        )

    def test_unwraps_secretstr_password(self) -> None:
        from pydantic import SecretStr

        cfg = self._base_cfg()
        cfg.password = SecretStr("secret")
        b = Neo4jBackend.from_config(cfg)
        assert b._password == "secret"

    def test_unwraps_secretstr_url(self) -> None:
        from pydantic import SecretStr

        cfg = self._base_cfg()
        cfg.url = SecretStr("bolt://localhost:7687")
        b = Neo4jBackend.from_config(cfg)
        assert b._url == "bolt://localhost:7687"

    def test_plain_string_password_passthrough(self) -> None:
        cfg = self._base_cfg()
        cfg.password = "plain"
        b = Neo4jBackend.from_config(cfg)
        assert b._password == "plain"


# ---------------------------------------------------------------------------
# Lifecycle: is_healthy / _get_driver / disconnect
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifecycle:
    @pytest.mark.asyncio
    async def test_is_healthy_false_when_no_driver(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        assert await b.is_healthy() is False

    @pytest.mark.asyncio
    async def test_is_healthy_true_on_successful_verify(self) -> None:
        driver = MagicMock()
        driver.verify_connectivity = AsyncMock()
        b = Neo4jBackend.from_driver(driver)
        assert await b.is_healthy() is True
        driver.verify_connectivity.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_is_healthy_false_on_verify_error(self) -> None:
        driver = MagicMock()
        driver.verify_connectivity = AsyncMock(side_effect=RuntimeError("down"))
        b = Neo4jBackend.from_driver(driver)
        assert await b.is_healthy() is False

    def test_get_driver_raises_when_none(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        with pytest.raises(RuntimeError, match="not connected"):
            b._get_driver()

    def test_get_driver_returns_driver(self) -> None:
        driver = MagicMock()
        b = Neo4jBackend.from_driver(driver)
        assert b._get_driver() is driver

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_noop(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        await b.disconnect()
        assert b._driver is None

    @pytest.mark.asyncio
    async def test_disconnect_closes_owned_driver(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        driver = MagicMock()
        driver.close = AsyncMock()
        b._driver = driver
        b._owns_driver = True
        await b.disconnect()
        driver.close.assert_awaited_once()
        assert b._driver is None

    @pytest.mark.asyncio
    async def test_disconnect_skips_close_on_shared_driver(self) -> None:
        driver = MagicMock()
        driver.close = AsyncMock()
        b = Neo4jBackend.from_driver(driver)
        await b.disconnect()
        driver.close.assert_not_called()
        # _driver still cleared so subsequent ops fail fast.
        assert b._driver is None


# ---------------------------------------------------------------------------
# Pool sampler start/stop without enabled flag
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPoolSampler:
    def test_start_sampler_noop_when_disabled(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687", pool_sampler_enabled=False)
        b._start_pool_sampler()
        assert b._sampler_task is None

    @pytest.mark.asyncio
    async def test_stop_sampler_noop_when_no_task(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        # Should not raise even though _sampler_task is None.
        await b._stop_pool_sampler()


# ---------------------------------------------------------------------------
# Empty-input short-circuits on batch methods
# ---------------------------------------------------------------------------


def _backend_with_session_mock(session: AsyncMock) -> Neo4jBackend:
    driver = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    return Neo4jBackend.from_driver(driver, query_timeout=1.0)


@pytest.mark.unit
class TestEmptyBatchShortCircuits:
    @pytest.mark.asyncio
    async def test_delete_entities_batch_empty_returns_zero(self) -> None:
        b = _backend_with_session_mock(AsyncMock())
        out = await b.delete_entities_batch([], namespace_id=uuid4())
        assert out == 0

    @pytest.mark.asyncio
    async def test_delete_relationships_batch_empty_returns_zero(self) -> None:
        b = _backend_with_session_mock(AsyncMock())
        out = await b.delete_relationships_batch([], namespace_id=uuid4())
        assert out == 0

    @pytest.mark.asyncio
    async def test_remove_document_from_entity_sources_batch_empty(self) -> None:
        b = _backend_with_session_mock(AsyncMock())
        out = await b.remove_document_from_entity_sources_batch([], uuid4(), uuid4())
        assert out == 0

    @pytest.mark.asyncio
    async def test_remove_document_from_relationship_sources_batch_empty(self) -> None:
        b = _backend_with_session_mock(AsyncMock())
        out = await b.remove_document_from_relationship_sources_batch([], uuid4(), uuid4())
        assert out == 0

    @pytest.mark.asyncio
    async def test_get_entities_batch_empty(self) -> None:
        b = _backend_with_session_mock(AsyncMock())
        out = await b.get_entities_batch([], namespace_id=uuid4())
        assert out == {}

    @pytest.mark.asyncio
    async def test_upsert_entities_batch_empty(self) -> None:
        b = _backend_with_session_mock(AsyncMock())
        out = await b.upsert_entities_batch(uuid4(), [])
        assert out == []

    @pytest.mark.asyncio
    async def test_remap_source_document_ids_batch_empty(self) -> None:
        """Both survivor lists empty → no session work."""
        session = AsyncMock()
        b = _backend_with_session_mock(session)
        await b.remap_source_document_ids_batch(entity_survivors=[], relationship_survivors=[], namespace_id=uuid4())
        session.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_neighborhoods_batch_empty(self) -> None:
        b = _backend_with_session_mock(AsyncMock())
        out = await b.get_neighborhoods_batch([], namespace_id=uuid4())
        assert out == {}


# ---------------------------------------------------------------------------
# _record_to_entity / _record_to_relationship / _record_to_episode
# ---------------------------------------------------------------------------


def _entity_node(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "name": "Alice",
        "entity_type": "PERSON",
        "description": "an engineer",
        "attributes": "{}",
        "source_document_ids": [],
        "source_chunk_ids": [],
        "mention_count": 1,
        "valid_from": None,
        "valid_until": None,
        "confidence": 0.9,
        "metadata": "{}",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-02T00:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestRecordToEntity:
    def test_basic_node(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        e = b._record_to_entity(_entity_node())
        assert e.name == "Alice"
        assert e.entity_type == "PERSON"
        assert e.confidence == 0.9

    def test_propagates_version_metadata(self) -> None:
        node = _entity_node(
            version_valid_from="2024-03-01T00:00:00+00:00",
            version_valid_to="2024-04-01T00:00:00+00:00",
        )
        b = Neo4jBackend("bolt://localhost:7687")
        e = b._record_to_entity(node)
        # Bi-temporal version fields land in metadata.
        assert e.metadata["version_valid_from"] == "2024-03-01T00:00:00+00:00"
        assert e.metadata["version_valid_to"] == "2024-04-01T00:00:00+00:00"

    def test_temporal_fields_parsed(self) -> None:
        node = _entity_node(
            valid_from="2024-01-01T00:00:00+00:00",
            valid_until="2024-12-31T00:00:00+00:00",
        )
        b = Neo4jBackend("bolt://localhost:7687")
        e = b._record_to_entity(node)
        assert e.valid_from == datetime(2024, 1, 1, tzinfo=UTC)
        assert e.valid_until == datetime(2024, 12, 31, tzinfo=UTC)


def _rel_node(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "description": "rel",
        "properties": "{}",
        "source_document_ids": [],
        "source_chunk_ids": [],
        "valid_from": None,
        "valid_until": None,
        "confidence": 0.85,
        "weight": 1.0,
        "metadata": "{}",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-02T00:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestRecordToRelationship:
    def test_basic_rel(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        src, tgt = str(uuid4()), str(uuid4())
        r = b._record_to_relationship(_rel_node(), src, tgt, "WORKS_FOR")
        assert r.relationship_type == "WORKS_FOR"
        assert str(r.source_entity_id) == src
        assert str(r.target_entity_id) == tgt
        assert r.confidence == 0.85

    def test_temporal_fields_parsed(self) -> None:
        node = _rel_node(
            valid_from="2024-01-01T00:00:00+00:00",
            valid_until="2024-06-01T00:00:00+00:00",
        )
        b = Neo4jBackend("bolt://localhost:7687")
        r = b._record_to_relationship(node, str(uuid4()), str(uuid4()), "KNOWS")
        assert r.valid_from == datetime(2024, 1, 1, tzinfo=UTC)
        assert r.valid_until == datetime(2024, 6, 1, tzinfo=UTC)

    def test_missing_id_synthesizes_uuid_and_warns(self) -> None:
        # Loguru intercept — capture WARNs via a custom sink.
        from loguru import logger as loguru_logger

        node = _rel_node()
        del node["id"]
        b = Neo4jBackend("bolt://localhost:7687")

        records: list[str] = []
        sink_id = loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
        try:
            r = b._record_to_relationship(node, str(uuid4()), str(uuid4()), "WORKS_FOR")
        finally:
            loguru_logger.remove(sink_id)

        assert isinstance(r.id, UUID)
        assert any("missing id/namespace_id" in m for m in records), records

    def test_missing_namespace_id_synthesizes_uuid_and_warns(self) -> None:
        from loguru import logger as loguru_logger

        node = _rel_node()
        del node["namespace_id"]
        b = Neo4jBackend("bolt://localhost:7687")

        records: list[str] = []
        sink_id = loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
        try:
            r = b._record_to_relationship(node, str(uuid4()), str(uuid4()), "WORKS_FOR")
        finally:
            loguru_logger.remove(sink_id)

        assert isinstance(r.namespace_id, UUID)
        assert any("missing id/namespace_id" in m for m in records), records

    def test_missing_both_synthesizes_uuids_and_warns(self) -> None:
        from loguru import logger as loguru_logger

        node = _rel_node()
        del node["id"]
        del node["namespace_id"]
        b = Neo4jBackend("bolt://localhost:7687")

        records: list[str] = []
        sink_id = loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
        try:
            r = b._record_to_relationship(node, str(uuid4()), str(uuid4()), "WORKS_FOR")
        finally:
            loguru_logger.remove(sink_id)

        assert isinstance(r.id, UUID)
        assert isinstance(r.namespace_id, UUID)
        assert any("missing id/namespace_id" in m for m in records), records

    def test_well_formed_passthrough_no_warn(self) -> None:
        from loguru import logger as loguru_logger

        rel_id, ns_id = uuid4(), uuid4()
        node = _rel_node(id=str(rel_id), namespace_id=str(ns_id))
        b = Neo4jBackend("bolt://localhost:7687")

        records: list[str] = []
        sink_id = loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
        try:
            r = b._record_to_relationship(node, str(uuid4()), str(uuid4()), "WORKS_FOR")
        finally:
            loguru_logger.remove(sink_id)

        assert r.id == rel_id
        assert r.namespace_id == ns_id
        assert not any("missing id/namespace_id" in m for m in records), records


@pytest.mark.unit
class TestRecordToEpisode:
    def test_basic_episode(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        node = {
            "id": str(uuid4()),
            "namespace_id": str(uuid4()),
            "name": "ep1",
            "description": "x",
            "occurred_at": "2024-01-01T00:00:00+00:00",
            "duration_seconds": 60,
            "entity_ids": [],
            "source_document_ids": [],
            "source_chunk_ids": [],
            "metadata": "{}",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-02T00:00:00+00:00",
        }
        ep = b._record_to_episode(node)
        assert ep.name == "ep1"
        assert ep.duration_seconds == 60
        assert ep.occurred_at == datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Sanity: _NEO4J_TIMEOUT_CODES tuple shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeoutCodes:
    def test_codes_present(self) -> None:
        assert "Neo.ClientError.Transaction.TransactionTimedOut" in _NEO4J_TIMEOUT_CODES
        assert all(c.startswith("Neo.ClientError.Transaction.") for c in _NEO4J_TIMEOUT_CODES)


# ---------------------------------------------------------------------------
# Issue #737 — configurable relationship-provenance caps + truncation telemetry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRelationshipProvenanceCapDefaults:
    def test_defaults_preserve_pre_737_behavior(self) -> None:
        """Defaults stay at 100 / 250 — regression gate."""
        b = Neo4jBackend("bolt://localhost:7687")
        assert b._relationship_source_document_ids_max == 100
        assert b._relationship_source_chunk_ids_max == 250

    def test_custom_caps_accepted(self) -> None:
        b = Neo4jBackend(
            "bolt://localhost:7687",
            relationship_source_document_ids_max=500,
            relationship_source_chunk_ids_max=1000,
        )
        assert b._relationship_source_document_ids_max == 500
        assert b._relationship_source_chunk_ids_max == 1000

    def test_from_config_threads_caps(self) -> None:
        cfg = SimpleNamespace(
            url="bolt://localhost:7687",
            user="u",
            password="pw",
            database="neo4j",
            max_connection_pool_size=100,
            connection_acquisition_timeout=60.0,
            retry_delay_jitter_factor=0.5,
            max_connection_lifetime=900,
            liveness_check_timeout=30.0,
            query_timeout=5.0,
            entity_write_concurrency=4,
            relationship_write_concurrency=2,
            pool_sampler_enabled=False,
            pool_sampler_interval_ms=500,
            relationship_source_document_ids_max=750,
            relationship_source_chunk_ids_max=2000,
        )
        b = Neo4jBackend.from_config(cfg)
        assert b._relationship_source_document_ids_max == 750
        assert b._relationship_source_chunk_ids_max == 2000

    def test_from_config_missing_caps_uses_defaults(self) -> None:
        """getattr fallback path — older Neo4jConfig without the new fields."""
        cfg = SimpleNamespace(
            url="bolt://localhost:7687",
            user="u",
            password="pw",
            database="neo4j",
            max_connection_pool_size=100,
            connection_acquisition_timeout=60.0,
            retry_delay_jitter_factor=0.5,
            max_connection_lifetime=900,
            liveness_check_timeout=30.0,
            query_timeout=5.0,
            entity_write_concurrency=4,
            relationship_write_concurrency=2,
            pool_sampler_enabled=False,
            pool_sampler_interval_ms=500,
            # No relationship_source_*_max — exercises the getattr default.
        )
        b = Neo4jBackend.from_config(cfg)
        assert b._relationship_source_document_ids_max == 100
        assert b._relationship_source_chunk_ids_max == 250


@pytest.mark.unit
class TestRecordTruncationHelper:
    def test_zero_dropped_is_noop(self) -> None:
        """No log, no metric increment when nothing was truncated."""
        b = Neo4jBackend("bolt://localhost:7687")
        b._source_id_truncated_counter = MagicMock()
        b._record_truncation(
            field="source_document_ids",
            kind="batch",
            dropped=0,
            rows_affected=0,
            limit=100,
        )
        b._source_id_truncated_counter.add.assert_not_called()

    def test_negative_dropped_is_noop(self) -> None:
        """Defensive: negative sentinel doesn't fire."""
        b = Neo4jBackend("bolt://localhost:7687")
        b._source_id_truncated_counter = MagicMock()
        b._record_truncation(
            field="source_chunk_ids",
            kind="single",
            dropped=-1,
            rows_affected=0,
            limit=250,
        )
        b._source_id_truncated_counter.add.assert_not_called()

    def test_positive_dropped_emits_counter_and_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from loguru import logger as loguru_logger

        b = Neo4jBackend("bolt://localhost:7687")
        b._source_id_truncated_counter = MagicMock()

        # loguru -> stdlib bridge so caplog sees the warning.
        handler_id = loguru_logger.add(
            lambda msg: caplog.handler.emit(
                __import__("logging").LogRecord(
                    name="khora",
                    level=__import__("logging").WARNING,
                    pathname="",
                    lineno=0,
                    msg=msg.record["message"],
                    args=None,
                    exc_info=None,
                )
            ),
            level="WARNING",
        )
        try:
            with caplog.at_level("WARNING"):
                b._record_truncation(
                    field="source_document_ids",
                    kind="batch",
                    dropped=42,
                    rows_affected=3,
                    limit=100,
                    rel_type="WORKS_FOR",
                )
        finally:
            loguru_logger.remove(handler_id)

        b._source_id_truncated_counter.add.assert_called_once_with(
            42, attributes={"field": "source_document_ids", "kind": "batch"}
        )
        # Warning message includes the actionable knob suggestion.
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "dropped 42 source_document_ids" in joined
        assert "WORKS_FOR" in joined
        assert "limit=100" in joined
        assert "RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX" in joined


@pytest.mark.unit
class TestCreateRelationshipsBatchTruncationWiring:
    @pytest.mark.asyncio
    async def test_batch_truncation_drives_counter(self) -> None:
        """When the Cypher RETURN signals dropped > 0, the counter fires."""
        # KNOWS is not in BIDIRECTIONAL_TYPES — single type-group / single tx.
        ns_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="KNOWS",
        )

        record = {"created": 1, "doc_dropped": 15, "chunk_dropped": 30, "doc_rows": 1, "chunk_rows": 1}

        result = MagicMock()
        result.single = AsyncMock(return_value=record)

        async def _run(*_args: Any, **_kwargs: Any) -> Any:
            return result

        async def _exec_write(fn: Any, *args: Any, **kwargs: Any) -> Any:
            tx = MagicMock()
            tx.run = AsyncMock(side_effect=_run)
            return await fn(tx, *args, **kwargs)

        session = AsyncMock()
        session.execute_write = AsyncMock(side_effect=_exec_write)

        b = _backend_with_session_mock(session)
        b._source_id_truncated_counter = MagicMock()

        out = await b.create_relationships_batch([rel])
        assert out == 1
        # Both fields fired with matching label sets.
        b._source_id_truncated_counter.add.assert_any_call(
            15, attributes={"field": "source_document_ids", "kind": "batch"}
        )
        b._source_id_truncated_counter.add.assert_any_call(
            30, attributes={"field": "source_chunk_ids", "kind": "batch"}
        )

    @pytest.mark.asyncio
    async def test_batch_no_truncation_does_not_fire(self) -> None:
        """ON CREATE rows (pre_union always == incoming) → dropped 0 → no emit."""
        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="KNOWS",
        )

        record = {"created": 1, "doc_dropped": 0, "chunk_dropped": 0, "doc_rows": 0, "chunk_rows": 0}
        result = MagicMock()
        result.single = AsyncMock(return_value=record)

        async def _run(*_args: Any, **_kwargs: Any) -> Any:
            return result

        async def _exec_write(fn: Any, *args: Any, **kwargs: Any) -> Any:
            tx = MagicMock()
            tx.run = AsyncMock(side_effect=_run)
            return await fn(tx, *args, **kwargs)

        session = AsyncMock()
        session.execute_write = AsyncMock(side_effect=_exec_write)
        b = _backend_with_session_mock(session)
        b._source_id_truncated_counter = MagicMock()
        await b.create_relationships_batch([rel])
        b._source_id_truncated_counter.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_custom_caps_threaded_into_cypher_params(self) -> None:
        """Custom caps reach the tx.run call as $src_doc_max / $src_chunk_max."""
        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="KNOWS",
        )

        record = {"created": 1, "doc_dropped": 0, "chunk_dropped": 0, "doc_rows": 0, "chunk_rows": 0}
        result = MagicMock()
        result.single = AsyncMock(return_value=record)

        seen_kwargs: dict[str, Any] = {}

        async def _run(*_args: Any, **kwargs: Any) -> Any:
            seen_kwargs.update(kwargs)
            return result

        async def _exec_write(fn: Any, *args: Any, **kwargs: Any) -> Any:
            tx = MagicMock()
            tx.run = AsyncMock(side_effect=_run)
            return await fn(tx, *args, **kwargs)

        session = AsyncMock()
        session.execute_write = AsyncMock(side_effect=_exec_write)

        driver = MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session.return_value = ctx

        b = Neo4jBackend.from_driver(driver, query_timeout=1.0)
        b._relationship_source_document_ids_max = 500
        b._relationship_source_chunk_ids_max = 1000

        await b.create_relationships_batch([rel])
        assert seen_kwargs.get("src_doc_max") == 500
        assert seen_kwargs.get("src_chunk_max") == 1000


@pytest.mark.unit
class TestCreateRelationshipSingleTruncationWiring:
    @pytest.mark.asyncio
    async def test_single_truncation_drives_counter(self) -> None:
        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="OWNS",
        )

        record = {"doc_dropped": 5, "chunk_dropped": 12}
        result = MagicMock()
        result.single = AsyncMock(return_value=record)

        async def _run(*_args: Any, **_kwargs: Any) -> Any:
            return result

        async def _exec_write(fn: Any, *args: Any, **kwargs: Any) -> Any:
            tx = MagicMock()
            tx.run = AsyncMock(side_effect=_run)
            return await fn(tx, *args, **kwargs)

        session = AsyncMock()
        session.execute_write = AsyncMock(side_effect=_exec_write)
        b = _backend_with_session_mock(session)
        b._source_id_truncated_counter = MagicMock()

        out = await b.create_relationship(rel)
        assert out is rel  # Method returns the input row unchanged.
        b._source_id_truncated_counter.add.assert_any_call(
            5, attributes={"field": "source_document_ids", "kind": "single"}
        )
        b._source_id_truncated_counter.add.assert_any_call(
            12, attributes={"field": "source_chunk_ids", "kind": "single"}
        )

    @pytest.mark.asyncio
    async def test_single_no_truncation_does_not_fire(self) -> None:
        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="OWNS",
        )
        record = {"doc_dropped": 0, "chunk_dropped": 0}
        result = MagicMock()
        result.single = AsyncMock(return_value=record)

        async def _run(*_args: Any, **_kwargs: Any) -> Any:
            return result

        async def _exec_write(fn: Any, *args: Any, **kwargs: Any) -> Any:
            tx = MagicMock()
            tx.run = AsyncMock(side_effect=_run)
            return await fn(tx, *args, **kwargs)

        session = AsyncMock()
        session.execute_write = AsyncMock(side_effect=_exec_write)
        b = _backend_with_session_mock(session)
        b._source_id_truncated_counter = MagicMock()

        await b.create_relationship(rel)
        b._source_id_truncated_counter.add.assert_not_called()


# ---------------------------------------------------------------------------
# Issue #777 — configurable entity-provenance caps + truncation telemetry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntityProvenanceCapDefaults:
    def test_defaults_preserve_pre_777_behavior(self) -> None:
        """Defaults stay at 100 / 250 — regression gate."""
        b = Neo4jBackend("bolt://localhost:7687")
        assert b._entity_source_document_ids_max == 100
        assert b._entity_source_chunk_ids_max == 250

    def test_custom_caps_accepted(self) -> None:
        b = Neo4jBackend(
            "bolt://localhost:7687",
            entity_source_document_ids_max=500,
            entity_source_chunk_ids_max=1000,
        )
        assert b._entity_source_document_ids_max == 500
        assert b._entity_source_chunk_ids_max == 1000

    def test_from_config_threads_caps(self) -> None:
        cfg = SimpleNamespace(
            url="bolt://localhost:7687",
            user="u",
            password="pw",
            database="neo4j",
            max_connection_pool_size=100,
            connection_acquisition_timeout=60.0,
            retry_delay_jitter_factor=0.5,
            max_connection_lifetime=900,
            liveness_check_timeout=30.0,
            query_timeout=5.0,
            entity_write_concurrency=4,
            relationship_write_concurrency=2,
            pool_sampler_enabled=False,
            pool_sampler_interval_ms=500,
            relationship_source_document_ids_max=100,
            relationship_source_chunk_ids_max=250,
            entity_source_document_ids_max=750,
            entity_source_chunk_ids_max=2000,
        )
        b = Neo4jBackend.from_config(cfg)
        assert b._entity_source_document_ids_max == 750
        assert b._entity_source_chunk_ids_max == 2000

    def test_from_config_missing_caps_uses_defaults(self) -> None:
        """getattr fallback path — older Neo4jConfig without the new entity fields."""
        cfg = SimpleNamespace(
            url="bolt://localhost:7687",
            user="u",
            password="pw",
            database="neo4j",
            max_connection_pool_size=100,
            connection_acquisition_timeout=60.0,
            retry_delay_jitter_factor=0.5,
            max_connection_lifetime=900,
            liveness_check_timeout=30.0,
            query_timeout=5.0,
            entity_write_concurrency=4,
            relationship_write_concurrency=2,
            pool_sampler_enabled=False,
            pool_sampler_interval_ms=500,
            # No entity_source_*_max — exercises the getattr default.
        )
        b = Neo4jBackend.from_config(cfg)
        assert b._entity_source_document_ids_max == 100
        assert b._entity_source_chunk_ids_max == 250


@pytest.mark.unit
class TestRecordEntityTruncationHelper:
    def test_zero_dropped_is_noop(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        b._entity_source_id_truncated_counter = MagicMock()
        b._record_entity_truncation(
            field="source_document_ids",
            kind="batch",
            dropped=0,
            rows_affected=0,
            limit=100,
        )
        b._entity_source_id_truncated_counter.add.assert_not_called()

    def test_negative_dropped_is_noop(self) -> None:
        b = Neo4jBackend("bolt://localhost:7687")
        b._entity_source_id_truncated_counter = MagicMock()
        b._record_entity_truncation(
            field="source_chunk_ids",
            kind="batch",
            dropped=-1,
            rows_affected=0,
            limit=250,
        )
        b._entity_source_id_truncated_counter.add.assert_not_called()

    def test_positive_dropped_emits_counter_and_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from loguru import logger as loguru_logger

        b = Neo4jBackend("bolt://localhost:7687")
        b._entity_source_id_truncated_counter = MagicMock()

        handler_id = loguru_logger.add(
            lambda msg: caplog.handler.emit(
                __import__("logging").LogRecord(
                    name="khora",
                    level=__import__("logging").WARNING,
                    pathname="",
                    lineno=0,
                    msg=msg.record["message"],
                    args=None,
                    exc_info=None,
                )
            ),
            level="WARNING",
        )
        try:
            with caplog.at_level("WARNING"):
                b._record_entity_truncation(
                    field="source_document_ids",
                    kind="batch",
                    dropped=42,
                    rows_affected=3,
                    limit=100,
                )
        finally:
            loguru_logger.remove(handler_id)

        b._entity_source_id_truncated_counter.add.assert_called_once_with(
            42, attributes={"field": "source_document_ids", "kind": "batch"}
        )
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "dropped 42 source_document_ids" in joined
        assert "limit=100" in joined
        assert "ENTITY_SOURCE_DOCUMENT_IDS_MAX" in joined


@pytest.mark.unit
class TestUpsertEntitiesBatchTruncationWiring:
    @pytest.mark.asyncio
    async def test_batch_truncation_drives_counter(self) -> None:
        """Existing entity with deep provenance + incoming row → drops counted."""
        ns_id = uuid4()
        ent_id = uuid4()
        ent = Entity(
            id=ent_id,
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            source_document_ids=[uuid4() for _ in range(5)],
            source_chunk_ids=[uuid4() for _ in range(10)],
        )

        # Pre-existing entity already at the cap — incoming 5 docs + 10
        # chunks all overflow.
        pre_row = {
            "id": str(ent_id),
            "name": "Alice",
            "entity_type": "PERSON",
            "namespace_id": str(ns_id),
            "attributes": "{}",
            "description": "",
            "source_document_ids": [str(uuid4()) for _ in range(100)],
            "source_chunk_ids": [str(uuid4()) for _ in range(250)],
            "mention_count": 1,
            "confidence": 1.0,
            "metadata": None,
            "version_valid_from": None,
        }
        # MERGE result: ON MATCH (is_new=false), so the input_id maps to
        # the existing entity id.
        merge_row = {
            "id": str(ent_id),
            "name": "Alice",
            "input_id": str(ent_id),
            "is_new": False,
        }

        pre_result = MagicMock()
        pre_result.data = AsyncMock(return_value=[pre_row])
        merge_result = MagicMock()
        merge_result.data = AsyncMock(return_value=[merge_row])

        call_count = {"n": 0}

        async def _run(*_args: Any, **_kwargs: Any) -> Any:
            call_count["n"] += 1
            return pre_result if call_count["n"] == 1 else merge_result

        async def _exec_write(fn: Any, *args: Any, **kwargs: Any) -> Any:
            tx = MagicMock()
            tx.run = AsyncMock(side_effect=_run)
            return await fn(tx, *args, **kwargs)

        session = AsyncMock()
        session.execute_write = AsyncMock(side_effect=_exec_write)
        b = _backend_with_session_mock(session)
        b._entity_source_id_truncated_counter = MagicMock()

        out = await b.upsert_entities_batch(ns_id, [ent])
        assert len(out) == 1

        # Caps 100/250, union 105/260 → drop 5 docs, 10 chunks.
        b._entity_source_id_truncated_counter.add.assert_any_call(
            5, attributes={"field": "source_document_ids", "kind": "batch"}
        )
        b._entity_source_id_truncated_counter.add.assert_any_call(
            10, attributes={"field": "source_chunk_ids", "kind": "batch"}
        )

    @pytest.mark.asyncio
    async def test_batch_no_truncation_does_not_fire(self) -> None:
        """No pre-existing entity → ON CREATE path → no drops."""
        ns_id = uuid4()
        ent_id = uuid4()
        ent = Entity(
            id=ent_id,
            namespace_id=ns_id,
            name="Bob",
            entity_type="PERSON",
            source_document_ids=[uuid4()],
            source_chunk_ids=[uuid4()],
        )

        pre_result = MagicMock()
        pre_result.data = AsyncMock(return_value=[])  # nothing existed
        merge_result = MagicMock()
        merge_result.data = AsyncMock(
            return_value=[
                {
                    "id": str(ent_id),
                    "name": "Bob",
                    "input_id": str(ent_id),
                    "is_new": True,
                }
            ]
        )

        call_count = {"n": 0}

        async def _run(*_args: Any, **_kwargs: Any) -> Any:
            call_count["n"] += 1
            return pre_result if call_count["n"] == 1 else merge_result

        async def _exec_write(fn: Any, *args: Any, **kwargs: Any) -> Any:
            tx = MagicMock()
            tx.run = AsyncMock(side_effect=_run)
            return await fn(tx, *args, **kwargs)

        session = AsyncMock()
        session.execute_write = AsyncMock(side_effect=_exec_write)
        b = _backend_with_session_mock(session)
        b._entity_source_id_truncated_counter = MagicMock()

        await b.upsert_entities_batch(ns_id, [ent])
        b._entity_source_id_truncated_counter.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_custom_caps_threaded_into_cypher_params(self) -> None:
        """Custom caps reach the upsert tx.run call as $ent_doc_max / $ent_chunk_max."""
        ns_id = uuid4()
        ent_id = uuid4()
        ent = Entity(
            id=ent_id,
            namespace_id=ns_id,
            name="Carol",
            entity_type="PERSON",
        )

        pre_result = MagicMock()
        pre_result.data = AsyncMock(return_value=[])
        merge_result = MagicMock()
        merge_result.data = AsyncMock(
            return_value=[
                {
                    "id": str(ent_id),
                    "name": "Carol",
                    "input_id": str(ent_id),
                    "is_new": True,
                }
            ]
        )

        seen_kwargs_list: list[dict[str, Any]] = []

        async def _run(*_args: Any, **kwargs: Any) -> Any:
            seen_kwargs_list.append(dict(kwargs))
            return pre_result if len(seen_kwargs_list) == 1 else merge_result

        async def _exec_write(fn: Any, *args: Any, **kwargs: Any) -> Any:
            tx = MagicMock()
            tx.run = AsyncMock(side_effect=_run)
            return await fn(tx, *args, **kwargs)

        session = AsyncMock()
        session.execute_write = AsyncMock(side_effect=_exec_write)
        b = _backend_with_session_mock(session)
        b._entity_source_document_ids_max = 500
        b._entity_source_chunk_ids_max = 1000

        await b.upsert_entities_batch(ns_id, [ent])
        # First call is _PREFETCH_CYPHER (no ent_*_max), second is _UPSERT_CYPHER.
        assert seen_kwargs_list[1].get("ent_doc_max") == 500
        assert seen_kwargs_list[1].get("ent_chunk_max") == 1000
