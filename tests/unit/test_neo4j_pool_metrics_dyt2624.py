"""DYT-2624 — tests for the correctness fixes to Neo4j pool metrics.

Covers the three problems identified in the ticket:
1. ``khora.neo4j.pool.acquire_duration`` records *real* pool acquisition
   time (first ``run``/``execute_read``/``execute_write``), not session
   construction.
2. ``khora.neo4j.pool.timeout`` increments on every
   ``ConnectionAcquisitionTimeoutError`` — from any entry path, via both
   ``Neo4jBackend`` and ``DualNodeManager``.
3. The high-frequency pool sampler produces many samples per unit time
   when enabled, and is fully inert when disabled.

Also includes a static-source guard ensuring no raw ``driver.session(`` /
``_driver.session(`` calls sneak back into the Neo4j code paths in
``src/khora/``.
"""

from __future__ import annotations

import asyncio
import re as _re
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import ConnectionAcquisitionTimeoutError

from khora.engines.vectorcypher.dual_nodes import DualNodeManager
from khora.storage.backends.neo4j import Neo4jBackend

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_session_mock(
    *,
    run_side_effect: Any = None,
    execute_read_side_effect: Any = None,
    execute_write_side_effect: Any = None,
    run_async: bool = True,
) -> AsyncMock:
    """Build a mock AsyncSession with configurable entry-method behaviour."""
    session = AsyncMock()
    if run_side_effect is not None:
        session.run = AsyncMock(side_effect=run_side_effect)
    else:
        session.run = AsyncMock(return_value=MagicMock())
    if execute_read_side_effect is not None:
        session.execute_read = AsyncMock(side_effect=execute_read_side_effect)
    else:
        session.execute_read = AsyncMock(return_value=[])
    if execute_write_side_effect is not None:
        session.execute_write = AsyncMock(side_effect=execute_write_side_effect)
    else:
        session.execute_write = AsyncMock(return_value=None)
    return session


def _make_driver_with_session(session: AsyncMock) -> MagicMock:
    driver = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    return driver


# ---------------------------------------------------------------------------
# Problem 1: acquire_duration measures real pool acquisition time
# ---------------------------------------------------------------------------


def _make_session_with_connect_hook(connect_delay: float = 0.0) -> AsyncMock:
    """Build a mock AsyncSession whose ``_connect`` mimics a pool bind.

    Each ``_connect`` call sleeps for ``connect_delay``, marks the session
    "bound" by setting ``_connection`` to a sentinel, and returns None.
    ``run``/``execute_*`` check ``_connection is None`` and invoke
    ``_connect`` when needed (mirroring the real driver at
    ``neo4j/_async/work/session.py:304``).
    """
    session = AsyncMock()
    session._connection = None

    async def fake_connect(*_args: Any, **_kwargs: Any) -> None:
        if connect_delay > 0:
            await asyncio.sleep(connect_delay)
        session._connection = object()  # mark bound

    session._connect = fake_connect

    async def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        if session._connection is None:
            await session._connect()
        return MagicMock()

    async def fake_execute_read(fn: Any = None, *_a: Any, **_kw: Any) -> Any:
        if session._connection is None:
            await session._connect()
        return None

    async def fake_execute_write(fn: Any = None, *_a: Any, **_kw: Any) -> Any:
        if session._connection is None:
            await session._connect()
        return None

    session.run = AsyncMock(side_effect=fake_run)
    session.execute_read = AsyncMock(side_effect=fake_execute_read)
    session.execute_write = AsyncMock(side_effect=fake_execute_write)
    return session


@pytest.mark.unit
class TestAcquireDurationMetricIsReal:
    """_InstrumentedSession records acquire_duration per pool acquire via ``_connect``."""

    @pytest.mark.asyncio
    async def test_not_recorded_when_session_is_entered_but_never_used(self) -> None:
        """No acquire_duration sample if no run/execute_* was called."""
        session = _make_session_mock()

        # Give the mock a _connect that simply no-ops; nothing should call it.
        async def _noop(*_a: Any, **_k: Any) -> None:
            return None

        session._connect = _noop
        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        samples: list[tuple[float, dict[str, Any] | None]] = []
        backend._acquire_duration_histogram = MagicMock()
        backend._acquire_duration_histogram.record = lambda v, attributes=None, **_: samples.append((v, attributes))

        async with backend._session():
            pass

        assert samples == [], "acquire_duration should not record without a pool-binding call"

    @pytest.mark.asyncio
    async def test_records_connect_time_not_total_run_time(self) -> None:
        """acquire_duration reflects ``_connect`` duration, not full run wall-clock."""
        connect_delay = 0.05  # 50 ms
        session = _make_session_with_connect_hook(connect_delay=connect_delay)
        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        samples: list[float] = []
        backend._acquire_duration_histogram = MagicMock()
        backend._acquire_duration_histogram.record = lambda v, attributes=None, **_: samples.append(v)

        async with backend._session() as s:
            await s.run("RETURN 1")

        assert len(samples) == 1
        assert samples[0] >= 0.04, f"acquire_duration too low: {samples[0]}"

    @pytest.mark.asyncio
    async def test_records_once_per_connect_not_per_run(self) -> None:
        """Multiple ``run`` calls on the same bound connection record exactly once."""
        session = _make_session_with_connect_hook()
        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        samples: list[float] = []
        backend._acquire_duration_histogram = MagicMock()
        backend._acquire_duration_histogram.record = lambda v, attributes=None, **_: samples.append(v)

        async with backend._session() as s:
            await s.run("RETURN 1")
            await s.run("RETURN 2")
            await s.execute_read(lambda _tx: None)

        assert len(samples) == 1, f"expected 1 record per bind, got {len(samples)}"

    @pytest.mark.asyncio
    async def test_records_again_after_reconnect(self) -> None:
        """If the session rebinds (``_connection = None`` then another call),
        the histogram records a second observation."""
        session = _make_session_with_connect_hook()
        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        samples: list[float] = []
        backend._acquire_duration_histogram = MagicMock()
        backend._acquire_duration_histogram.record = lambda v, attributes=None, **_: samples.append(v)

        async with backend._session() as s:
            await s.run("RETURN 1")  # bind #1
            # simulate retry / reconnect scenario: release the connection.
            s._inner._connection = None
            await s.run("RETURN 2")  # bind #2

        assert len(samples) == 2, f"expected 2 records across 2 binds, got {len(samples)}"

    @pytest.mark.asyncio
    async def test_excludes_query_and_retry_time(self) -> None:
        """Acquire is 10 ms; the query sleeps 100 ms after. The histogram must
        record ~10 ms, not ~110 ms — slow-query time must not be conflated
        with acquire time."""
        session = AsyncMock()
        session._connection = None

        async def fake_connect(*_a: Any, **_k: Any) -> None:
            await asyncio.sleep(0.01)  # 10ms "acquire"
            session._connection = object()

        session._connect = fake_connect

        async def fake_run(*_a: Any, **_k: Any) -> Any:
            if session._connection is None:
                await session._connect()
            await asyncio.sleep(0.1)  # 100ms "query"
            return MagicMock()

        session.run = AsyncMock(side_effect=fake_run)
        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        samples: list[float] = []
        backend._acquire_duration_histogram = MagicMock()
        backend._acquire_duration_histogram.record = lambda v, attributes=None, **_: samples.append(v)

        async with backend._session() as s:
            await s.run("RETURN 1")

        assert len(samples) == 1
        # Must be closer to 10ms than to 110ms. Give generous CI slack: <60ms.
        assert 0.005 <= samples[0] < 0.06, (
            f"acquire_duration ({samples[0] * 1000:.1f} ms) conflates query time — "
            f"should be near 10 ms, saw {samples[0] * 1000:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Problem 2: timeout counter increments from every entry path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeoutCounterUniversality:
    """Every method that can surface a ConnectionAcquisitionTimeoutError bumps the counter."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entry_method", ["run", "execute_read", "execute_write"])
    async def test_counter_increments_from_each_entry_method(self, entry_method: str) -> None:
        """Pool timeout raised from the session's internal ``_connect`` (invoked by
        ``run``/``execute_read``/``execute_write``) increments the counter exactly once."""
        err = ConnectionAcquisitionTimeoutError("pool exhausted")
        session = AsyncMock()
        session._connection = None

        async def failing_connect(*_a: Any, **_k: Any) -> None:
            raise err

        session._connect = failing_connect

        async def run_through_connect(*_a: Any, **_k: Any) -> Any:
            if session._connection is None:
                await session._connect()
            return MagicMock()

        for name in ("run", "execute_read", "execute_write"):
            setattr(session, name, AsyncMock(side_effect=run_through_connect))

        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        adds: list[int] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **_: adds.append(v)

        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with backend._session() as s:
                await getattr(s, entry_method)("RETURN 1")

        assert adds == [1]

    @pytest.mark.asyncio
    async def test_counter_increments_on_aenter_timeout(self) -> None:
        """Timeouts raised from __aenter__ (lazy driver variants) also bump the counter."""
        driver = MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=ConnectionAcquisitionTimeoutError("x"))
        ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session.return_value = ctx

        backend = Neo4jBackend.from_driver(driver)
        adds: list[int] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **_: adds.append(v)

        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with backend._session():
                pass

        assert adds == [1]

    @pytest.mark.asyncio
    async def test_counter_increments_via_dualnode_manager_when_wired(self) -> None:
        """When DualNodeManager has session_factory=backend._session, its entry paths bump the counter."""
        err = ConnectionAcquisitionTimeoutError("pool exhausted")
        session = AsyncMock()
        session._connection = None

        async def failing_connect(*_a: Any, **_k: Any) -> None:
            raise err

        session._connect = failing_connect

        async def execute_read_through_connect(*_a: Any, **_k: Any) -> Any:
            await session._connect()
            return []

        session.run = AsyncMock(return_value=MagicMock())
        session.execute_read = AsyncMock(side_effect=execute_read_through_connect)
        session.execute_write = AsyncMock()

        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        adds: list[int] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **_: adds.append(v)

        manager = DualNodeManager(driver, session_factory=backend._session)

        async def _work(_tx: Any) -> list[Any]:
            return []

        # Exercises: DualNodeManager._session -> backend._session
        # -> _InstrumentedSession -> session._connect raises.
        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with manager._session() as s:
                await s.execute_read(_work)

        assert adds == [1]

    @pytest.mark.asyncio
    async def test_counter_increments_twice_for_two_burst_timeouts(self) -> None:
        """Two timeouts across both entry paths -> counter == 2.

        Each ``driver.session()`` call returns a fresh session in the real
        driver, so the proxy's ``_connect`` wrap does not nest. We emulate
        that here by returning a new AsyncMock per ``driver.session()``.
        """
        err = ConnectionAcquisitionTimeoutError("pool exhausted")

        def make_failing_session() -> AsyncMock:
            session = AsyncMock()
            session._connection = None

            async def failing_connect(*_a: Any, **_k: Any) -> None:
                raise err

            session._connect = failing_connect

            async def entry_through_connect(*_a: Any, **_k: Any) -> Any:
                await session._connect()
                return MagicMock()

            session.run = AsyncMock(side_effect=entry_through_connect)
            session.execute_read = AsyncMock(side_effect=entry_through_connect)
            session.execute_write = AsyncMock()
            return session

        driver = MagicMock()

        def new_session_ctx(*_a: Any, **_kw: Any) -> MagicMock:
            session = make_failing_session()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        driver.session.side_effect = new_session_ctx

        backend = Neo4jBackend.from_driver(driver)
        manager = DualNodeManager(driver, session_factory=backend._session)

        adds: list[int] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **_: adds.append(v)

        # Path 1: Neo4jBackend direct (representing get_entity-style calls).
        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with backend._session() as s:
                await s.run("MATCH (n) RETURN n")

        # Path 2: DualNodeManager wired to backend._session.
        async def _work(_tx: Any) -> list[Any]:
            return []

        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with manager._session() as s:
                await s.execute_read(_work)

        assert sum(adds) == 2


# ---------------------------------------------------------------------------
# Timeout histogram exclusion — pool-acquire p99 must not be polluted by
# the connection-acquisition-timeout deadline.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeoutHistogramExclusion:
    """Timeouts count on the ``pool.timeout`` counter but never on the acquire histogram."""

    @pytest.mark.asyncio
    async def test_timeout_does_not_record_acquire_duration_histogram(self) -> None:
        """Raising ConnectionAcquisitionTimeoutError from _connect bumps the counter
        exactly once; the acquire_duration histogram records zero observations."""
        err = ConnectionAcquisitionTimeoutError("pool exhausted")
        session = AsyncMock()
        session._connection = None

        async def failing_connect(*_a: Any, **_k: Any) -> None:
            await asyncio.sleep(0.04)
            raise err

        session._connect = failing_connect

        async def run_through_connect(*_a: Any, **_k: Any) -> Any:
            await session._connect()
            return MagicMock()

        session.run = AsyncMock(side_effect=run_through_connect)
        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        counter_adds: list[int] = []
        hist_records: list[tuple[float, dict[str, Any] | None]] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **_: counter_adds.append(v)
        backend._acquire_duration_histogram = MagicMock()
        backend._acquire_duration_histogram.record = lambda v, attributes=None, **_: hist_records.append(
            (v, attributes)
        )

        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with backend._session() as s:
                await s.run("RETURN 1")

        assert sum(counter_adds) == 1, "timeout counter should increment exactly once"
        assert hist_records == [], f"acquire_duration histogram must not record on timeout; got {hist_records}"


# ---------------------------------------------------------------------------
# Problem 3: high-frequency pool sampler
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPoolSampler:
    def test_sampler_not_started_when_disabled(self) -> None:
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {}
        driver._pool.connections_reservations = {}
        driver._pool.in_use_connection_count = MagicMock(return_value=0)

        backend = Neo4jBackend.from_driver(driver)
        assert backend._pool_sampler_enabled is False

        backend._start_pool_sampler()
        assert backend._sampler_task is None

    def test_sample_pool_once_reads_driver_internals(self) -> None:
        driver = MagicMock()
        pool = MagicMock()
        pool.connections = {
            "addr1": deque([MagicMock(), MagicMock(), MagicMock()]),
        }
        pool.connections_reservations = {"addr1": 2}
        pool.in_use_connection_count = MagicMock(return_value=2)
        pool.pool_config = MagicMock()
        pool.pool_config.max_connection_pool_size = 10
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        sample = backend._sample_pool_once()

        assert sample is not None
        assert sample["active"] == 2.0
        assert sample["total"] == 3.0
        assert sample["idle"] == 1.0  # total - active, clamped
        assert sample["creating"] == 2.0
        assert sample["utilization"] == pytest.approx(0.2)

    def test_sample_pool_once_clamps_idle_when_inconsistent(self) -> None:
        """Transient inconsistency in driver state must not produce negative idle counts."""
        driver = MagicMock()
        pool = MagicMock()
        pool.connections = {"addr1": deque([MagicMock()])}  # 1 conn total
        pool.connections_reservations = {}
        pool.in_use_connection_count = MagicMock(return_value=5)  # lies
        pool.pool_config = MagicMock()
        pool.pool_config.max_connection_pool_size = 10
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        sample = backend._sample_pool_once()
        assert sample is not None
        assert sample["idle"] == 0.0

    def test_sample_pool_once_returns_none_when_driver_detached(self) -> None:
        backend = Neo4jBackend.from_driver(MagicMock())
        backend._driver = None
        assert backend._sample_pool_once() is None

    @pytest.mark.asyncio
    async def test_sampler_produces_many_samples_in_synthetic_window(self) -> None:
        """Simulated short window: a 10ms interval over 200ms should yield ≥ 10 observations per histogram."""
        driver = MagicMock()
        pool = MagicMock()
        pool.connections = {"addr1": deque([MagicMock()])}
        pool.connections_reservations = {"addr1": 0}
        pool.in_use_connection_count = MagicMock(return_value=1)
        pool.pool_config = MagicMock()
        pool.pool_config.max_connection_pool_size = 10
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        backend._pool_sampler_enabled = True
        backend._pool_sampler_interval_ms = 10  # fast

        observed: dict[str, int] = {k: 0 for k in ("active", "idle", "total", "creating", "utilization")}
        for key in observed:
            attr = f"_sampled_{key}_histogram"
            setattr(backend, attr, MagicMock())

            def make_recorder(k: str):
                def record(_v: float, attributes: Any = None, **_kw: Any) -> None:
                    observed[k] += 1

                return record

            getattr(backend, attr).record = make_recorder(key)

        backend._start_pool_sampler()
        assert backend._sampler_task is not None

        # Let sampler tick for ~200 ms.
        await asyncio.sleep(0.2)
        await backend._stop_pool_sampler()

        for key, count in observed.items():
            assert count >= 10, f"{key}: expected >=10 samples, got {count}"

    @pytest.mark.asyncio
    async def test_sampler_tolerates_driver_internal_errors(self) -> None:
        """Read errors in _sample_pool_once log a single warning and return None (no crash loop)."""
        driver = MagicMock()

        class BrokenPool:
            @property
            def connections(self) -> dict[str, Any]:
                raise RuntimeError("internal driver broke")

        driver._pool = BrokenPool()

        backend = Neo4jBackend.from_driver(driver)

        # First call triggers the single warning, returns None.
        with patch("khora.storage.backends.neo4j.logger") as mock_logger:
            assert backend._sample_pool_once() is None
            mock_logger.warning.assert_called_once()

        # Subsequent calls are silent — we don't log-storm on every tick.
        with patch("khora.storage.backends.neo4j.logger") as mock_logger:
            assert backend._sample_pool_once() is None
            mock_logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_sampler_is_idempotent_and_safe_when_not_running(self) -> None:
        backend = Neo4jBackend.from_driver(MagicMock())
        assert backend._sampler_task is None
        await backend._stop_pool_sampler()  # must not raise
        assert backend._sampler_task is None


# ---------------------------------------------------------------------------
# Config: pool_sampler_enabled / pool_sampler_interval_ms plumbing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSamplerConfig:
    def test_interval_ms_clamped_low(self) -> None:
        backend = Neo4jBackend(
            "bolt://localhost:7687",
            pool_sampler_enabled=True,
            pool_sampler_interval_ms=1,
        )
        assert backend._pool_sampler_interval_ms == 50

    def test_interval_ms_clamped_high(self) -> None:
        backend = Neo4jBackend(
            "bolt://localhost:7687",
            pool_sampler_enabled=True,
            pool_sampler_interval_ms=10**9,
        )
        assert backend._pool_sampler_interval_ms == 60_000

    def test_defaults_off(self) -> None:
        backend = Neo4jBackend("bolt://localhost:7687")
        assert backend._pool_sampler_enabled is False
        assert backend._pool_sampler_interval_ms == 500

    def test_from_config_reads_new_fields(self) -> None:
        # Use the real Pydantic Neo4jConfig so all numeric fields are real ints
        # (MagicMock would leak into asyncio.Semaphore's validation).
        from khora.config.schema import Neo4jConfig

        cfg = Neo4jConfig(
            url="bolt://localhost:7687",
            pool_sampler_enabled=True,
            pool_sampler_interval_ms=750,
        )
        backend = Neo4jBackend.from_config(cfg)
        assert backend._pool_sampler_enabled is True
        assert backend._pool_sampler_interval_ms == 750


# ---------------------------------------------------------------------------
# Static guard: no raw driver.session(...) calls outside the allowlisted
# session-helper bodies. This is the "lint/test that fails if a raw
# driver.session( call sneaks in" the ticket asks for.
# ---------------------------------------------------------------------------


# Files allowed to contain the raw `driver.session(` / `_driver.session(`
# string because they own the session-acquisition helpers themselves or
# target non-Neo4j backends (Memgraph/Neptune/maintenance utilities).
_ALLOWED_RAW_SESSION_FILES = {
    # Owns Neo4jBackend._session helper.
    Path("src/khora/storage/backends/neo4j.py"),
    # Owns DualNodeManager._session fallback when no session_factory is wired.
    Path("src/khora/engines/vectorcypher/dual_nodes.py"),
    # Non-Neo4j Bolt backends — share the neo4j driver package but are NOT
    # scoped by the khora.neo4j.pool.* metric suite.
    Path("src/khora/storage/backends/memgraph.py"),
    Path("src/khora/storage/backends/neptune.py"),
    # One-shot Neo4j maintenance utility used by operators via CLI, not
    # part of the read/write hot path the metric suite is measuring.
    Path("src/khora/storage/optimize.py"),
}


@pytest.mark.unit
class TestNoRawDriverSessionInHotPath:
    """Catch regressions where someone re-introduces a raw driver.session(...)."""

    _RAW_SESSION_RE = _re.compile(r"(?<![\w.])(?:_driver|driver)\.session\(")

    def test_hot_paths_have_no_raw_driver_session(self) -> None:
        src_root = Path(__file__).resolve().parents[2] / "src" / "khora"
        assert src_root.is_dir(), f"src/khora not found from test dir: {src_root}"

        offenders: list[str] = []
        for py_file in src_root.rglob("*.py"):
            rel = py_file.relative_to(src_root.parents[1])
            if rel in _ALLOWED_RAW_SESSION_FILES:
                continue
            for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
                # Skip comments / docstrings — the regex can match ``driver.session(...)``
                # in a docstring describing the API, which is fine.
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if self._RAW_SESSION_RE.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

        assert not offenders, (
            "Raw driver.session(...) calls found outside the allowlist. "
            "Route them through Neo4jBackend._session (or DualNodeManager._session "
            "with session_factory=backend._session) so khora.neo4j.pool.* metrics "
            "observe them:\n" + "\n".join(offenders)
        )
