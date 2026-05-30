"""Regression for #877: engines that opt out of graph / event store no longer
emit a misleading WARNING when ``neo4j_url`` IS configured (or any time the
engine has explicitly skipped that backend).

Before this fix, ``Khora(engine="skeleton")`` and ``Khora(engine="chronicle")``
produced two confusing WARNINGs per construction:

    Neo4j URL not configured, graph backend disabled
    Event store URL not configured, event store disabled

Both fired even when the user had configured Neo4j - the misleading message
was a symptom of ``build_storage_config(config, skip_graph=True)`` dropping the
URL on the floor and the factory falling into the same code path it uses for
the "operator forgot to configure" case.

The fix adds two boolean sentinels to ``StorageConfig`` (``graph_skipped`` /
``event_store_skipped``) that ``build_storage_config`` sets to record "the
caller opted out". The factory then returns ``None`` silently when the
sentinel is set, and keeps the existing WARNING for the genuine
misconfiguration case.

These tests cover the three engine factories (skeleton, chronicle,
vectorcypher) at the ``build_storage_config`` + ``StorageFactory`` boundary
without spinning up any real backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from loguru import logger

from khora.engines._storage_config import build_storage_config
from khora.storage.factory import StorageConfig, StorageFactory


def _capture_warnings():
    """Return (handler_id, captured) for a loguru WARNING sink."""
    captured: list[str] = []
    handler_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    return handler_id, captured


@pytest.fixture
def base_config() -> MagicMock:
    """Mock KhoraConfig with traditional postgres backend defaults and a
    configured Neo4j URL - mirrors what an end-user with both DBs set up has.
    """
    config = MagicMock()
    config.storage.backend = "postgres"
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.postgresql_pool_pre_ping = False
    config.storage.embedding_dimension = 1536
    config.storage.use_halfvec = True
    config.get_postgresql_url.return_value = "postgresql://localhost/db"
    config.get_neo4j_url.return_value = "bolt://localhost:7687"
    config.get_neo4j_user.return_value = "neo4j"
    config.get_neo4j_password.return_value = "pw"
    config.get_neo4j_database.return_value = "neo4j"
    config.get_graph_config.return_value = MagicMock()
    config.get_vector_config.return_value = MagicMock()
    return config


@pytest.mark.unit
class TestBuildStorageConfigSentinels:
    """``build_storage_config`` must propagate the opt-out sentinels so the
    factory can tell "user forgot to configure" from "engine doesn't want one".
    """

    def test_skip_graph_sets_graph_skipped_sentinel(self, base_config: MagicMock) -> None:
        sc = build_storage_config(base_config, skip_graph=True)
        assert sc.graph_skipped is True

    def test_skip_graph_false_leaves_graph_skipped_false(self, base_config: MagicMock) -> None:
        sc = build_storage_config(base_config, skip_graph=False)
        assert sc.graph_skipped is False

    def test_event_store_skipped_is_always_true_via_builder(self, base_config: MagicMock) -> None:
        """No engine wires the legacy PostgreSQL event store through this
        builder today. Always opt out so the WARNING never fires."""
        sc = build_storage_config(base_config)
        assert sc.event_store_skipped is True

    def test_event_store_skipped_holds_under_skip_graph(self, base_config: MagicMock) -> None:
        sc = build_storage_config(base_config, skip_graph=True)
        assert sc.event_store_skipped is True


@pytest.mark.unit
class TestFactorySilencesWarningsWhenSkipped:
    """``StorageFactory.create_*`` must NOT emit the misleading WARNINGs
    when the caller has opted out via the sentinels.

    These guard the three engine shapes described in the issue:
      - skeleton:  ``build_storage_config(config, skip_graph=True)``
      - chronicle: ``build_storage_config(config, skip_graph=True)``
      - vectorcypher: ``build_storage_config(config)`` (graph IS wanted)
    Plus the legitimate-misconfiguration case (caller did NOT opt out) -
    that one must still warn.
    """

    def test_skeleton_path_no_neo4j_warning(self, base_config: MagicMock) -> None:
        """skeleton-shaped StorageConfig (skip_graph=True) does not warn."""
        sc = build_storage_config(base_config, skip_graph=True)
        # The builder drops the neo4j_url when skip_graph is True; the
        # legacy graph path inside create_graph_backend would otherwise
        # trip the "not configured" warning.
        assert sc.neo4j_url is None
        factory = StorageFactory(config=sc)

        handler_id, captured = _capture_warnings()
        try:
            backend = factory.create_graph_backend()
        finally:
            logger.remove(handler_id)

        assert backend is None
        assert not any("Neo4j URL not configured" in m for m in captured), (
            f"misleading neo4j warning fired on skeleton path: {captured}"
        )

    def test_chronicle_path_no_event_store_warning(self, base_config: MagicMock) -> None:
        """chronicle-shaped StorageConfig does not warn for the event store."""
        sc = build_storage_config(base_config, skip_graph=True)
        assert sc.event_store_url is None
        assert sc.event_store_skipped is True
        factory = StorageFactory(config=sc)

        handler_id, captured = _capture_warnings()
        try:
            store = factory.create_event_store()
        finally:
            logger.remove(handler_id)

        assert store is None
        assert not any("Event store URL not configured" in m for m in captured), (
            f"misleading event-store warning fired on chronicle path: {captured}"
        )

    def test_vectorcypher_path_no_event_store_warning(self, base_config: MagicMock) -> None:
        """vectorcypher does NOT pass skip_graph (it wants the graph). The
        event-store warning must still be silenced because the legacy
        PostgreSQL event store is not wired through this builder."""
        sc = build_storage_config(base_config)
        # vectorcypher: graph IS wanted, so graph_skipped stays False but
        # neo4j_url is set - the graph-backend path will not warn either.
        assert sc.graph_skipped is False
        assert sc.event_store_skipped is True
        factory = StorageFactory(config=sc)

        handler_id, captured = _capture_warnings()
        try:
            store = factory.create_event_store()
        finally:
            logger.remove(handler_id)

        assert store is None
        assert not any("Event store URL not configured" in m for m in captured), (
            f"misleading event-store warning fired on vectorcypher path: {captured}"
        )

    def test_legitimate_neo4j_misconfiguration_still_warns(self) -> None:
        """The "operator forgot to configure" path must still warn -
        ``graph_skipped=False`` + no ``neo4j_url`` is a real misconfiguration.
        """
        sc = StorageConfig(
            postgresql_url="postgresql://localhost/db",
            neo4j_url=None,
            graph_skipped=False,
        )
        factory = StorageFactory(config=sc)

        handler_id, captured = _capture_warnings()
        try:
            backend = factory.create_graph_backend()
        finally:
            logger.remove(handler_id)

        assert backend is None
        assert any("Neo4j URL not configured" in m for m in captured), (
            f"expected legitimate misconfiguration warning; got: {captured}"
        )

    def test_legitimate_event_store_misconfiguration_still_warns(self) -> None:
        """``event_store_skipped=False`` + no URL is still a real
        misconfiguration on the legacy ``StorageConfig.from_dict`` path."""
        sc = StorageConfig(
            postgresql_url="postgresql://localhost/db",
            event_store_url=None,
            event_store_skipped=False,
        )
        factory = StorageFactory(config=sc)

        handler_id, captured = _capture_warnings()
        try:
            store = factory.create_event_store()
        finally:
            logger.remove(handler_id)

        assert store is None
        assert any("Event store URL not configured" in m for m in captured), (
            f"expected legitimate misconfiguration warning; got: {captured}"
        )
