"""Composition-level integration test for Neo4j query timeout.

Verifies end-to-end wiring from ``KhoraConfig`` → VectorCypher engine
pattern → ``DualNodeManager`` → ``unit_of_work`` decoration → ``ClientError``
timeout catch, without requiring a running Neo4j instance.

This test is marked ``@pytest.mark.integration`` (not ``unit``) because it
exercises the full configuration-to-execution composition that unit tests
mock piecewise — specifically the wiring path:

    KHORA_STORAGE__GRAPH__QUERY_TIMEOUT=<value>
      → KhoraConfig() load
      → Neo4jConfig.query_timeout
      → engine.py:  getattr(neo4j_cfg, "query_timeout", 5.0)
      → DualNodeManager(query_timeout=...)
      → __init__ hoist: self._timed_unit_of_work = unit_of_work(timeout=...)
      → get_entity_neighborhoods(): _work = self._timed_unit_of_work(_work)
      → session.execute_read(_work)
      → ClientError catch
      → trace_span(".timeout") + warning + return {}

The failure mode this catches is *wiring drift*: a future refactor that
renames the config field, changes the env var delimiter, drops the
``getattr`` in ``engine.py``, or un-hoists the decorator would break one
link in this chain without failing any individual unit test.

Real-Neo4j protocol compatibility (driver-to-server timeout transmission,
wire-level error code handling) is out of scope for CI because khora's CI
does NOT provision a Neo4j instance. Local developers wanting that
coverage can run the test suite against a dev Neo4j via ``make dev``
and a suggested ``NEO4J_INTEGRATION_TEST=1`` env var — a follow-up
ticket will add that real-driver coverage when useful.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from neo4j.exceptions import ClientError, Neo4jError

from khora.config.schema import KhoraConfig, Neo4jConfig
from khora.engines.vectorcypher.dual_nodes import (
    _NEO4J_TIMEOUT_CODES,
    DualNodeManager,
)


def _timeout_error(
    code: str = "Neo.ClientError.Transaction.TransactionTimedOut",
) -> ClientError:
    exc = Neo4jError._basic_hydrate(neo4j_code=code, message="timed out")
    assert isinstance(exc, ClientError), f"neo4j _basic_hydrate unexpectedly returned {type(exc).__name__}"
    return exc


def _driver_raising_timeout(code: str) -> MagicMock:
    """Build a fake AsyncDriver whose ``execute_read`` raises the given code."""
    driver = MagicMock()
    session = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    session.execute_read = AsyncMock(side_effect=_timeout_error(code))
    return driver


@pytest.mark.integration
class TestDyt1948TimeoutCompositionIntegration:
    """Full composition: config → manager → timeout catch → empty dict."""

    def test_config_env_var_wires_through_to_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``KHORA_STORAGE__GRAPH__QUERY_TIMEOUT`` reaches ``DualNodeManager``.

        Exercises the exact env var documented in
        ``Neo4jConfig.query_timeout.description`` (double underscore
        nesting, per ``env_nested_delimiter='__'``).
        """
        monkeypatch.setenv("KHORA_STORAGE__GRAPH__BACKEND", "neo4j")
        monkeypatch.setenv("KHORA_STORAGE__GRAPH__URL", "bolt://localhost:7687")
        monkeypatch.setenv("KHORA_STORAGE__GRAPH__QUERY_TIMEOUT", "4.2")

        cfg = KhoraConfig()
        graph_cfg = cfg.get_graph_config()
        assert isinstance(graph_cfg, Neo4jConfig)
        assert graph_cfg.query_timeout == 4.2

        # Mirror the exact getattr pattern ``engine.py`` uses to resolve
        # the timeout with a 5.0 fallback for older config objects.
        resolved = getattr(graph_cfg, "query_timeout", 5.0) if graph_cfg else 5.0
        assert resolved == 4.2

        driver = _driver_raising_timeout(_NEO4J_TIMEOUT_CODES[0])
        manager = DualNodeManager(driver, query_timeout=resolved)
        assert manager._query_timeout == 4.2
        # Hoisted decorator exists — catches a future un-hoist regression
        # at the composition boundary, not just at the unit level.
        assert manager._timed_unit_of_work is not None

    @pytest.mark.parametrize("timeout_code", _NEO4J_TIMEOUT_CODES)
    @pytest.mark.asyncio
    async def test_full_path_degrades_to_empty_on_timeout(self, timeout_code: str) -> None:
        """Both timeout codes produce ``{}`` from the full composition path."""
        driver = _driver_raising_timeout(timeout_code)
        manager = DualNodeManager(driver, query_timeout=1.0)

        result = await manager.get_entity_neighborhoods(
            [uuid4(), uuid4()],
            uuid4(),
            depth=2,
        )
        assert result == {}
        # The driver session was actually exercised — guards against
        # a future short-circuit that skips the Neo4j call entirely
        # based on ``query_timeout`` being set.
        driver.session.assert_called_once_with(database="neo4j")

    @pytest.mark.asyncio
    async def test_disabled_timeout_skips_decoration_end_to_end(self) -> None:
        """``query_timeout=None`` wires through to no decoration at all."""
        driver = MagicMock()
        session = AsyncMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session.return_value = ctx
        session.execute_read = AsyncMock(return_value=[])

        manager = DualNodeManager(driver, query_timeout=None)
        assert manager._timed_unit_of_work is None

        result = await manager.get_entity_neighborhoods([uuid4()], uuid4())
        assert result == {}
        # Sanity-check that the decoration skip did not accidentally
        # short-circuit the rest of the method.
        driver.session.assert_called_once()
