"""Unit tests for Neo4jBackend timeout behavior."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from neo4j.exceptions import ClientError, Neo4jError

from khora.core.models.entity import Relationship
from khora.storage.backends.neo4j import _NEO4J_TIMEOUT_CODES, Neo4jBackend


def _make_neo4j_error(code: str, message: str = "boom") -> ClientError:
    """Build a ClientError instance with a given server-side code."""
    exc = Neo4jError._basic_hydrate(neo4j_code=code, message=message)
    assert isinstance(exc, ClientError), f"expected ClientError for code {code}, got {type(exc).__name__}"
    assert exc.code == code
    return exc


def _make_neo4j_driver() -> tuple[MagicMock, AsyncMock]:
    """Create a mock Neo4j driver with properly mocked session context manager."""
    driver = MagicMock()
    session = AsyncMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx

    return driver, session


@pytest.mark.unit
class TestNeo4jBackendInit:
    """Tests for Neo4jBackend.__init__ query_timeout plumbing."""

    def test_stores_query_timeout(self) -> None:
        """query_timeout is stored on the instance."""
        backend = Neo4jBackend("bolt://localhost:7687", query_timeout=2.5)
        assert backend._query_timeout == 2.5
        assert backend._timed_unit_of_work is not None

    def test_query_timeout_none_disables_wrapper(self) -> None:
        """query_timeout=None means no timed wrapper."""
        backend = Neo4jBackend("bolt://localhost:7687", query_timeout=None)
        assert backend._query_timeout is None
        assert backend._timed_unit_of_work is None

    def test_default_query_timeout_is_5(self) -> None:
        """Default query_timeout is 5.0 seconds."""
        backend = Neo4jBackend("bolt://localhost:7687")
        assert backend._query_timeout == 5.0
        assert backend._timed_unit_of_work is not None


@pytest.mark.unit
class TestNeo4jBackendLogLevelFromEnv:
    """DYT-2625: Neo4jBackend.__init__ applies KHORA_NEO4J_LOG_LEVEL from env."""

    @pytest.fixture
    def _reset_neo4j_logger_level(self):
        neo4j_logger = logging.getLogger("neo4j")
        original = neo4j_logger.level
        yield neo4j_logger
        neo4j_logger.setLevel(original)

    def test_init_applies_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _reset_neo4j_logger_level: logging.Logger,
    ) -> None:
        """Constructor raises the neo4j logger verbosity when env var is set."""
        monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "DEBUG")
        _reset_neo4j_logger_level.setLevel(logging.NOTSET)
        Neo4jBackend("bolt://localhost:7687")
        assert _reset_neo4j_logger_level.level == logging.DEBUG

    def test_init_noop_when_env_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _reset_neo4j_logger_level: logging.Logger,
    ) -> None:
        """Constructor does not touch the neo4j logger when env var is absent."""
        monkeypatch.delenv("KHORA_NEO4J_LOG_LEVEL", raising=False)
        _reset_neo4j_logger_level.setLevel(logging.NOTSET)
        Neo4jBackend("bolt://localhost:7687")
        assert _reset_neo4j_logger_level.level == logging.NOTSET


@pytest.mark.unit
class TestNeo4jBackendFromConfig:
    """Tests for Neo4jBackend.from_config query_timeout passthrough."""

    def test_from_config_reads_query_timeout(self) -> None:
        """from_config passes query_timeout from the config object."""
        config = MagicMock()
        config.url = "bolt://localhost:7687"
        config.user = "neo4j"
        config.password = ""
        config.database = "neo4j"
        config.max_connection_pool_size = 100
        config.connection_acquisition_timeout = 60.0
        config.retry_delay_jitter_factor = 0.5
        config.max_connection_lifetime = 900
        config.liveness_check_timeout = 30.0
        config.query_timeout = 3.5
        config.entity_write_concurrency = 16
        config.relationship_write_concurrency = 8

        backend = Neo4jBackend.from_config(config)
        assert backend._query_timeout == 3.5
        assert backend._timed_unit_of_work is not None


@pytest.mark.unit
class TestNeo4jBackendFromDriver:
    """Tests for Neo4jBackend.from_driver query_timeout kwarg."""

    def test_from_driver_accepts_query_timeout(self) -> None:
        """from_driver stores query_timeout and creates wrapper."""
        driver = MagicMock()
        backend = Neo4jBackend.from_driver(driver, query_timeout=4.0)
        assert backend._query_timeout == 4.0
        assert backend._timed_unit_of_work is not None

    def test_from_driver_none_disables_wrapper(self) -> None:
        """from_driver with query_timeout=None disables the wrapper."""
        driver = MagicMock()
        backend = Neo4jBackend.from_driver(driver, query_timeout=None)
        assert backend._query_timeout is None
        assert backend._timed_unit_of_work is None

    def test_from_driver_default_is_5(self) -> None:
        """from_driver default query_timeout is 5.0."""
        driver = MagicMock()
        backend = Neo4jBackend.from_driver(driver)
        assert backend._query_timeout == 5.0


@pytest.mark.unit
class TestNeo4jBackendGetNeighborhoodTimeout:
    """Tests for get_neighborhood timeout handling."""

    @pytest.mark.parametrize("timeout_code", _NEO4J_TIMEOUT_CODES)
    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self, timeout_code: str) -> None:
        """get_neighborhood degrades to empty dict on timeout."""
        driver, session = _make_neo4j_driver()
        timeout_exc = _make_neo4j_error(timeout_code, message="timed out")
        session.execute_read = AsyncMock(side_effect=timeout_exc)

        backend = Neo4jBackend.from_driver(driver, query_timeout=1.0)

        with patch("khora.storage.backends.neo4j.logger"):
            result = await backend.get_neighborhood(uuid4())

        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_reraises_non_timeout_error(self) -> None:
        """Non-timeout ClientErrors propagate from get_neighborhood."""
        driver, session = _make_neo4j_driver()
        syntax_exc = _make_neo4j_error(
            "Neo.ClientError.Statement.SyntaxError",
            message="Cypher syntax error",
        )
        session.execute_read = AsyncMock(side_effect=syntax_exc)

        backend = Neo4jBackend.from_driver(driver, query_timeout=1.0)

        with pytest.raises(ClientError) as excinfo:
            await backend.get_neighborhood(uuid4())

        assert excinfo.value.code == "Neo.ClientError.Statement.SyntaxError"


@pytest.mark.unit
class TestNeo4jBackendGetNeighborhoodsBatchTimeout:
    """Tests for get_neighborhoods_batch timeout handling."""

    @pytest.mark.parametrize("timeout_code", _NEO4J_TIMEOUT_CODES)
    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self, timeout_code: str) -> None:
        """get_neighborhoods_batch degrades to empty dict on timeout."""
        driver, session = _make_neo4j_driver()
        timeout_exc = _make_neo4j_error(timeout_code, message="timed out")
        session.execute_read = AsyncMock(side_effect=timeout_exc)

        backend = Neo4jBackend.from_driver(driver, query_timeout=1.0)

        with patch("khora.storage.backends.neo4j.logger"):
            result = await backend.get_neighborhoods_batch([uuid4(), uuid4()])

        assert result == {}

    @pytest.mark.asyncio
    async def test_reraises_non_timeout_error(self) -> None:
        """Non-timeout ClientErrors propagate from get_neighborhoods_batch."""
        driver, session = _make_neo4j_driver()
        syntax_exc = _make_neo4j_error(
            "Neo.ClientError.Statement.SyntaxError",
            message="Cypher syntax error",
        )
        session.execute_read = AsyncMock(side_effect=syntax_exc)

        backend = Neo4jBackend.from_driver(driver, query_timeout=1.0)

        with pytest.raises(ClientError) as excinfo:
            await backend.get_neighborhoods_batch([uuid4()])

        assert excinfo.value.code == "Neo.ClientError.Statement.SyntaxError"


def _make_rel_props(**overrides: Any) -> dict[str, Any]:
    """Build a minimal post-fix relationship properties dict, with overrides."""
    props: dict[str, Any] = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "description": "default description",
        "properties": "{}",
        "source_document_ids": [],
        "source_chunk_ids": [],
        "valid_from": None,
        "valid_until": None,
        "confidence": 1.0,
        "weight": 1.0,
        "metadata": "{}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-02T00:00:00+00:00",
    }
    props.update(overrides)
    return props


@pytest.mark.unit
class TestNeo4jBackendGetEntityRelationships:
    """Tests for get_entity_relationships (DYT-2626)."""

    @pytest.mark.asyncio
    async def test_returns_relationships_from_properties_dict(self) -> None:
        """Non-empty path: `r` arrives as a properties dict (post-fix shape)."""
        driver, session = _make_neo4j_driver()

        entity_id = uuid4()
        target_id = uuid4()
        rel_id = uuid4()
        ns_id = uuid4()

        rel_props = _make_rel_props(
            id=str(rel_id),
            namespace_id=str(ns_id),
            description="knows each other",
            properties='{"since": "2024"}',
            confidence=0.9,
            metadata='{"k": "v"}',
        )
        records = [
            {
                "r": rel_props,
                "source_id": str(entity_id),
                "target_id": str(target_id),
                "rel_type": "KNOWS",
            }
        ]

        result = MagicMock()
        result.data = AsyncMock(return_value=records)
        session.run = AsyncMock(return_value=result)

        backend = Neo4jBackend.from_driver(driver, query_timeout=None)

        got = await backend.get_entity_relationships(entity_id, direction="outgoing")

        assert isinstance(got, list)
        assert len(got) == 1
        rel = got[0]
        assert isinstance(rel, Relationship)
        assert rel.id == rel_id
        assert rel.namespace_id == ns_id
        assert rel.source_entity_id == entity_id
        assert rel.target_entity_id == target_id
        assert rel.relationship_type == "KNOWS"
        assert rel.description == "knows each other"
        assert rel.properties == {"since": "2024"}
        assert rel.metadata == {"k": "v"}
        assert rel.confidence == 0.9
        assert rel.weight == 1.0

    @pytest.mark.asyncio
    async def test_raises_typeerror_on_raw_relationship_tuple(self) -> None:
        """Regression lock: raw relationship 3-tuple (pre-fix shape) must fail.

        If the Cypher query ever regresses from ``RETURN properties(r) as r`` back
        to ``RETURN r``, ``result.data()`` serializes the Relationship value as a
        3-tuple (start_dict, rel_type, end_dict) and ``_record_to_relationship``
        indexes it with string keys — raising TypeError. Pinning this prevents
        the DYT-2626 regression from silently returning.
        """
        driver, session = _make_neo4j_driver()

        entity_id = uuid4()

        # Simulate the pre-fix shape: `r` is a 3-tuple, not a dict.
        raw_tuple = ({"id": str(uuid4())}, "KNOWS", {"id": str(uuid4())})
        records = [
            {
                "r": raw_tuple,
                "source_id": str(entity_id),
                "target_id": str(uuid4()),
                "rel_type": "KNOWS",
            }
        ]

        result = MagicMock()
        result.data = AsyncMock(return_value=records)
        session.run = AsyncMock(return_value=result)

        backend = Neo4jBackend.from_driver(driver, query_timeout=None)

        with pytest.raises(TypeError, match="tuple indices"):
            await backend.get_entity_relationships(entity_id, direction="outgoing")

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_relationships(self) -> None:
        """Empty-result path: ``.data()`` returns ``[]`` → method returns ``[]``."""
        driver, session = _make_neo4j_driver()

        result = MagicMock()
        result.data = AsyncMock(return_value=[])
        session.run = AsyncMock(return_value=result)

        backend = Neo4jBackend.from_driver(driver, query_timeout=None)

        got = await backend.get_entity_relationships(uuid4(), direction="outgoing")

        assert got == []

    @pytest.mark.asyncio
    async def test_direction_incoming_swaps_cypher_pattern(self) -> None:
        """``direction="incoming"`` generates ``(other)-[r]->(e)`` pattern."""
        driver, session = _make_neo4j_driver()

        result = MagicMock()
        result.data = AsyncMock(return_value=[])
        session.run = AsyncMock(return_value=result)

        backend = Neo4jBackend.from_driver(driver, query_timeout=None)

        await backend.get_entity_relationships(uuid4(), direction="incoming")

        query = session.run.call_args.args[0]
        assert "(other)-[r]->(e)" in query
        # Negative check: outgoing arrow must not be present.
        assert "(e)-[r]->(other)" not in query

    @pytest.mark.asyncio
    async def test_direction_both_uses_undirected_pattern(self) -> None:
        """``direction="both"`` generates ``(e)-[r]-(other)`` (no arrow)."""
        driver, session = _make_neo4j_driver()

        result = MagicMock()
        result.data = AsyncMock(return_value=[])
        session.run = AsyncMock(return_value=result)

        backend = Neo4jBackend.from_driver(driver, query_timeout=None)

        await backend.get_entity_relationships(uuid4(), direction="both")

        query = session.run.call_args.args[0]
        assert "(e)-[r]-(other)" in query
        # Negative checks: neither arrow form should appear for "both".
        assert "(e)-[r]->(other)" not in query
        assert "(other)-[r]->(e)" not in query

    @pytest.mark.asyncio
    async def test_relationship_types_filter_is_applied(self) -> None:
        """``relationship_types`` is injected as ``:A|B`` into the Cypher pattern."""
        driver, session = _make_neo4j_driver()

        result = MagicMock()
        result.data = AsyncMock(return_value=[])
        session.run = AsyncMock(return_value=result)

        backend = Neo4jBackend.from_driver(driver, query_timeout=None)

        await backend.get_entity_relationships(
            uuid4(),
            direction="outgoing",
            relationship_types=["KNOWS", "WORKS_WITH"],
        )

        query = session.run.call_args.args[0]
        assert ":KNOWS|WORKS_WITH" in query

    @pytest.mark.asyncio
    async def test_handles_missing_optional_fields(self) -> None:
        """Records missing valid_from/valid_until/metadata fall back to defaults."""
        driver, session = _make_neo4j_driver()

        entity_id = uuid4()
        target_id = uuid4()
        rel_id = uuid4()
        ns_id = uuid4()

        # Only required fields — no valid_from, valid_until, or metadata.
        rel_props = {
            "id": str(rel_id),
            "namespace_id": str(ns_id),
            "description": "minimal",
            "properties": "{}",
            "source_document_ids": [],
            "source_chunk_ids": [],
            "confidence": 0.8,
            "weight": 0.7,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-02T00:00:00+00:00",
        }
        records = [
            {
                "r": rel_props,
                "source_id": str(entity_id),
                "target_id": str(target_id),
                "rel_type": "KNOWS",
            }
        ]

        result = MagicMock()
        result.data = AsyncMock(return_value=records)
        session.run = AsyncMock(return_value=result)

        backend = Neo4jBackend.from_driver(driver, query_timeout=None)

        got = await backend.get_entity_relationships(entity_id, direction="outgoing")

        assert len(got) == 1
        rel = got[0]
        assert rel.valid_from is None
        assert rel.valid_until is None
        assert rel.metadata == {}

    @pytest.mark.asyncio
    async def test_returns_multiple_relationships(self) -> None:
        """Multiple records each materialize into a distinct ``Relationship``."""
        driver, session = _make_neo4j_driver()

        entity_id = uuid4()
        rel_ids = [uuid4(), uuid4(), uuid4()]
        target_ids = [uuid4(), uuid4(), uuid4()]

        records = [
            {
                "r": _make_rel_props(id=str(rel_ids[i])),
                "source_id": str(entity_id),
                "target_id": str(target_ids[i]),
                "rel_type": "KNOWS",
            }
            for i in range(3)
        ]

        result = MagicMock()
        result.data = AsyncMock(return_value=records)
        session.run = AsyncMock(return_value=result)

        backend = Neo4jBackend.from_driver(driver, query_timeout=None)

        got = await backend.get_entity_relationships(entity_id, direction="outgoing")

        assert len(got) == 3
        assert all(isinstance(r, Relationship) for r in got)
        assert {r.id for r in got} == set(rel_ids)
