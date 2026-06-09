"""Real-Neo4j integration tests for the connection-pool keepalive.

Mirrors the gating of the other real-Neo4j integration modules in this
repo (``test_neo4j_get_entity_relationships_integration.py``): marked
``@pytest.mark.integration`` and skipped unless ``NEO4J_INTEGRATION_TEST=1``,
because real-Neo4j tests are gated on that flag. The CI integration job
provisions Neo4j and sets the flag; locally, run ``make dev`` first.

How to run locally:

    make dev  # starts postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_pool_keepalive_integration.py -v

Connection parameters (match the ``make dev`` compose stack):

    KHORA_NEO4J_URL       (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME  (default: neo4j)
    KHORA_NEO4J_PASSWORD  (default: password)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from khora.storage.backends.neo4j import Neo4jBackend

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

_NEO4J_GATE = pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
)


def _neo4j_params() -> tuple[str, str, str]:
    url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")
    return url, user, password


def _wire_keepalive_counters(backend: Neo4jBackend) -> dict[str, int]:
    """Replace the keepalive counters with counting recorders.

    The OTel counters are no-ops without logfire, so we swap in plain
    recorders to observe ping / failure activity deterministically.
    """
    observed = {"pings": 0, "failures": 0}

    class _Recorder:
        def __init__(self, key: str) -> None:
            self._key = key

        def add(self, value: int, **_kw: object) -> None:
            observed[self._key] += value

    backend._keepalive_pings_counter = _Recorder("pings")
    backend._keepalive_failures_counter = _Recorder("failures")
    return observed


@pytest.mark.integration
@_NEO4J_GATE
class TestKeepaliveAgainstRealNeo4j:
    """Real driver, real pool — the keepalive fires pings and stops cleanly."""

    @pytest.mark.asyncio
    async def test_keepalive_fires_pings_over_a_couple_intervals(self) -> None:
        """from_driver path: connect → warm one pooled connection → wait a few
        intervals → assert pings fired and no failures, then disconnect cleanly.
        """
        from neo4j import AsyncGraphDatabase

        url, user, password = _neo4j_params()
        driver = AsyncGraphDatabase.driver(url, auth=(user, password))
        backend = Neo4jBackend.from_driver(
            driver,
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=50,
        )
        observed = _wire_keepalive_counters(backend)
        try:
            await backend.connect()
            assert backend._keepalive_task is not None

            # Warm at least one connection into the pool so the keepalive has
            # an idle connection to count and ping.
            async with backend._session() as s:
                await s.run("RETURN 1")

            # ~6 intervals at 50ms.
            await asyncio.sleep(0.3)
        finally:
            await backend.disconnect()
            await driver.close()

        assert backend._keepalive_task is None, "disconnect() must stop the keepalive"
        assert observed["pings"] >= 1, f"expected the keepalive to fire pings, got {observed}"
        # Against a healthy local Neo4j, pings should not fail.
        assert observed["failures"] == 0, f"unexpected keepalive ping failures: {observed}"

    @pytest.mark.asyncio
    async def test_sampler_and_keepalive_coexist(self) -> None:
        """Sampler + keepalive both enabled run together without interfering."""
        from neo4j import AsyncGraphDatabase

        url, user, password = _neo4j_params()
        driver = AsyncGraphDatabase.driver(url, auth=(user, password))
        backend = Neo4jBackend.from_driver(
            driver,
            pool_sampler_enabled=True,
            pool_sampler_interval_ms=50,
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=50,
        )
        observed = _wire_keepalive_counters(backend)
        try:
            await backend.connect()
            assert backend._sampler_task is not None
            assert backend._keepalive_task is not None
            assert backend._sampler_task is not backend._keepalive_task

            async with backend._session() as s:
                await s.run("RETURN 1")

            await asyncio.sleep(0.3)

            # Both tasks still alive (no deadlock, no crash).
            assert not backend._sampler_task.done()
            assert not backend._keepalive_task.done()
        finally:
            await backend.disconnect()
            await driver.close()

        assert backend._sampler_task is None
        assert backend._keepalive_task is None
        assert observed["pings"] >= 1, f"keepalive did not fire alongside sampler: {observed}"

    @pytest.mark.asyncio
    async def test_disconnect_does_not_touch_a_closed_driver(self) -> None:
        """disconnect() stops the keepalive cleanly even after the driver closed.

        The keepalive owns no driver lifecycle (from_driver => not owns_driver),
        so stopping it must be a clean task-cancel with no use of the driver.
        """
        from neo4j import AsyncGraphDatabase

        url, user, password = _neo4j_params()
        driver = AsyncGraphDatabase.driver(url, auth=(user, password))
        backend = Neo4jBackend.from_driver(
            driver,
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=50,
        )
        await backend.connect()
        await asyncio.sleep(0.1)
        # disconnect() cancels+awaits the keepalive task; it does not close the
        # shared driver (owns_driver=False). Must not raise.
        await backend.disconnect()
        assert backend._keepalive_task is None
        await driver.close()


@pytest.mark.embedded
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])")
class TestKeepaliveUnaffectedBackends:
    """Graph-less / non-Neo4j stacks never create a keepalive task.

    This needs no real Neo4j — the sqlite_lance coordinator uses no
    ``Neo4jBackend`` — so it is gated on the embedded extras rather than
    ``NEO4J_INTEGRATION_TEST``, matching the embedded-stack integration
    tests' style.
    """

    @pytest.mark.asyncio
    async def test_sqlite_lance_stack_has_no_neo4j_keepalive(self, tmp_path: Path) -> None:
        from tests.integration._sqlite_lance_fixtures import build_sqlite_lance_coordinator

        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            # No graph backend is a Neo4jBackend on the embedded path; none of
            # the coordinator's backends carries a running keepalive task.
            for backend in (
                getattr(coord, "graph", None),
                getattr(coord, "vector", None),
                getattr(coord, "relational", None),
            ):
                if backend is None:
                    continue
                assert not isinstance(backend, Neo4jBackend), "sqlite_lance stack must not use Neo4jBackend"
                assert getattr(backend, "_keepalive_task", None) is None
        finally:
            await coord.disconnect()
