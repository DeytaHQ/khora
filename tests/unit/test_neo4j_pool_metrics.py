"""Unit tests for Neo4j pool metrics instrumentation.

Validates that:
- _NoOpCounter and _NoOpHistogram silently discard calls
- metric_counter/metric_histogram return no-ops when logfire absent
- metric_gauge_callback is a no-op when logfire absent
- When logfire is present, helpers delegate to logfire APIs
- Neo4jBackend._session() records acquisition time and session duration
- Neo4jBackend._session() increments timeout counter on acquisition timeout
- Neo4jBackend._session() re-raises the original exception
- Neo4jBackend._register_pool_metrics() registers gauge callbacks
- Gauge callbacks return correct Observations from mock pool state
- Neo4jBackend._init_metrics() creates histogram and counter instruments
"""

from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import ConnectionAcquisitionTimeoutError

from khora.storage.backends.neo4j import Neo4jBackend
from khora.telemetry.metrics import (
    metric_counter,
    metric_gauge_callback,
    metric_histogram,
)

# ---------------------------------------------------------------------------
# telemetry.metrics no-op fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricHelpers:
    """Helpers create OTel instruments via khora's meter scope.

    When no MeterProvider is configured, OTel's proxy returns no-op
    instruments — ``.add`` / ``.record`` are silent. We just assert
    the helpers return *something* callable.
    """

    def test_metric_counter_returns_addable_instrument(self) -> None:
        c = metric_counter("test.counter", unit="1", description="A counter")
        c.add(1)
        c.add(3.14, attributes={"key": "val"})

    def test_metric_histogram_returns_recordable_instrument(self) -> None:
        h = metric_histogram("test.hist", unit="s", description="Latency")
        h.record(42)
        h.record(0.5, attributes={"op": "read"})

    def test_metric_gauge_callback_does_not_raise(self) -> None:
        """metric_gauge_callback registers an observable gauge with the meter."""
        metric_gauge_callback("test.gauge", [lambda _: iter([])], unit="1", description="A gauge")


# ---------------------------------------------------------------------------
# Neo4jBackend._init_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jInitMetrics:
    """The backend creates real OTel instruments via metric_*; when no
    MeterProvider is configured, those are OTel proxy instruments that
    silently swallow calls. We assert the instruments are callable —
    duck-typed against the OTel Counter / Histogram protocol."""

    def test_init_creates_metric_instruments(self) -> None:
        backend = Neo4jBackend("bolt://localhost:7687")
        backend._acquisition_histogram.record(0.1)
        backend._session_duration_histogram.record(0.2)
        backend._timeout_counter.add(1)

    def test_from_driver_creates_metric_instruments(self) -> None:
        driver = MagicMock()
        backend = Neo4jBackend.from_driver(driver)
        backend._acquisition_histogram.record(0.1)
        backend._session_duration_histogram.record(0.2)
        backend._timeout_counter.add(1)


# ---------------------------------------------------------------------------
# Neo4jBackend._session context manager
# ---------------------------------------------------------------------------


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
class TestSessionContextManager:
    @pytest.mark.asyncio
    async def test_records_acquisition_time(self) -> None:
        """_session records the legacy construction-time histogram.

        NOTE: as of the session is wrapped in ``_InstrumentedSession``;
        the inner session is exposed via ``_inner`` for tests that need to
        assert identity.
        """
        driver, session = _make_neo4j_driver()
        backend = Neo4jBackend.from_driver(driver)

        recorded: list[float] = []
        backend._acquisition_histogram = MagicMock()
        backend._acquisition_histogram.record = lambda v, **kw: recorded.append(v)
        backend._session_duration_histogram = MagicMock()

        async with backend._session() as s:
            assert s._inner is session  # proxy wraps the raw AsyncSession

        assert len(recorded) == 1
        assert recorded[0] >= 0.0

    @pytest.mark.asyncio
    async def test_records_session_duration(self) -> None:
        """_session records total session duration on close."""
        driver, session = _make_neo4j_driver()
        backend = Neo4jBackend.from_driver(driver)

        durations: list[float] = []
        backend._acquisition_histogram = MagicMock()
        backend._session_duration_histogram = MagicMock()
        backend._session_duration_histogram.record = lambda v, **kw: durations.append(v)

        async with backend._session():
            pass

        assert len(durations) == 1
        assert durations[0] >= 0.0

    @pytest.mark.asyncio
    async def test_counts_acquisition_timeout(self) -> None:
        """_session increments timeout counter on ConnectionAcquisitionTimeoutError."""
        driver = MagicMock()

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=ConnectionAcquisitionTimeoutError("pool exhausted"))
        ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session.return_value = ctx

        backend = Neo4jBackend.from_driver(driver)

        timeout_adds: list[int] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **kw: timeout_adds.append(v)

        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with backend._session():
                pass

        assert timeout_adds == [1]

    @pytest.mark.asyncio
    async def test_does_not_count_other_exceptions(self) -> None:
        """_session does not increment timeout counter for non-timeout exceptions."""
        driver = MagicMock()

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("something else"))
        ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session.return_value = ctx

        backend = Neo4jBackend.from_driver(driver)

        timeout_adds: list[int] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **kw: timeout_adds.append(v)

        with pytest.raises(RuntimeError):
            async with backend._session():
                pass

        assert timeout_adds == []

    @pytest.mark.asyncio
    async def test_reraises_original_exception(self) -> None:
        """_session re-raises ConnectionAcquisitionTimeoutError (doesn't swallow it)."""
        driver = MagicMock()

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=ConnectionAcquisitionTimeoutError("pool exhausted"))
        ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session.return_value = ctx

        backend = Neo4jBackend.from_driver(driver)
        backend._timeout_counter = MagicMock()

        with pytest.raises(ConnectionAcquisitionTimeoutError, match="pool exhausted"):
            async with backend._session():
                pass

    @pytest.mark.asyncio
    async def test_session_duration_gte_acquisition_time(self) -> None:
        """Session duration should be >= acquisition time."""
        driver, session = _make_neo4j_driver()
        backend = Neo4jBackend.from_driver(driver)

        acq_values: list[float] = []
        dur_values: list[float] = []
        backend._acquisition_histogram = MagicMock()
        backend._acquisition_histogram.record = lambda v, **kw: acq_values.append(v)
        backend._session_duration_histogram = MagicMock()
        backend._session_duration_histogram.record = lambda v, **kw: dur_values.append(v)

        async with backend._session():
            pass

        assert len(acq_values) == 1
        assert len(dur_values) == 1
        assert dur_values[0] >= acq_values[0]

    @pytest.mark.asyncio
    async def test_slow_acquisition_logs_warning(self) -> None:
        """_session logs a warning when a real pool acquire takes > 5s.

        As of the slow-acquire threshold is evaluated inside the
        ``session._connect`` wrap (one observation per real pool bind), not
        from session construction.
        """
        driver = MagicMock()
        session = AsyncMock()
        session._connection = None

        async def slow_connect(*_a, **_k):  # type: ignore[no-untyped-def]
            session._connection = object()

        session._connect = slow_connect

        async def run_through_connect(*_a, **_k):  # type: ignore[no-untyped-def]
            if session._connection is None:
                await session._connect()
            return MagicMock()

        session.run = AsyncMock(side_effect=run_through_connect)

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        driver.session.return_value = ctx

        backend = Neo4jBackend.from_driver(driver)
        backend._acquisition_histogram = MagicMock()
        backend._acquire_duration_histogram = MagicMock()
        backend._session_duration_histogram = MagicMock()

        # Patch time.monotonic so the wrap measures a slow connect (6s).
        # Call sequence inside backend._session + _timed_connect:
        #   t0 = 0 (outer), legacy = 0 (proxy construction),
        #   t0 in _timed_connect = 0, post-connect = 6, then final duration = 6.
        values = [0.0, 0.0, 0.0, 6.0, 6.0]
        call_count = 0

        def fake_monotonic():  # type: ignore[no-untyped-def]
            nonlocal call_count
            val = values[min(call_count, len(values) - 1)]
            call_count += 1
            return val

        with patch("khora.storage.backends.neo4j._time") as mock_time:
            mock_time.monotonic = fake_monotonic
            with patch("khora.storage.backends.neo4j.logger") as mock_logger:
                async with backend._session() as s:
                    await s.run("RETURN 1")

                mock_logger.warning.assert_called_once()
                assert "6.0s" in mock_logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# Neo4jBackend._register_pool_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegisterPoolMetrics:
    def test_skips_when_pool_not_accessible(self) -> None:
        """_register_pool_metrics logs debug when _pool is None."""
        driver = MagicMock(spec=[])  # No _pool attribute
        backend = Neo4jBackend.from_driver(driver)

        with patch("khora.storage.backends.neo4j.logger") as mock_logger:
            backend._register_pool_metrics()
            mock_logger.debug.assert_called_once()

    def test_registers_five_gauges(self) -> None:
        """_register_pool_metrics registers 5 gauge callbacks."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {}
        driver._pool.connections_reservations = {}

        backend = Neo4jBackend.from_driver(driver)

        with patch("khora.telemetry.metrics.metric_gauge_callback") as mock_gauge:
            backend._register_pool_metrics()
            assert mock_gauge.call_count == 5

            names = [call.args[0] for call in mock_gauge.call_args_list]
            assert "khora.neo4j.pool.connections.active" in names
            assert "khora.neo4j.pool.connections.idle" in names
            assert "khora.neo4j.pool.connections.total" in names
            assert "khora.neo4j.pool.connections.creating" in names
            assert "khora.neo4j.pool.utilization" in names

    def test_idempotent_on_repeated_connect(self) -> None:
        """_register_pool_metrics registers gauges only once even if called twice."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {}
        driver._pool.connections_reservations = {}

        backend = Neo4jBackend.from_driver(driver)

        with patch("khora.telemetry.metrics.metric_gauge_callback") as mock_gauge:
            backend._register_pool_metrics()
            backend._register_pool_metrics()  # second call should be a no-op
            assert mock_gauge.call_count == 5  # not 10


def _make_mock_connection(in_use: bool = False) -> MagicMock:
    conn = MagicMock()
    conn.in_use = in_use
    return conn


def _capture_gauge_callbacks(driver: MagicMock, backend: Neo4jBackend) -> dict:
    """Register pool metrics and capture the callback functions by name."""
    captured: dict[str, object] = {}

    def capture(name, callbacks, **kwargs):
        captured[name] = callbacks[0]

    with patch("khora.telemetry.metrics.metric_gauge_callback", side_effect=capture):
        backend._register_pool_metrics()

    return captured


class _FakeObservation:
    """Lightweight stand-in for opentelemetry.metrics.Observation."""

    def __init__(self, value):
        self.value = value


def _invoke_gauge(callback) -> list[_FakeObservation]:
    """Invoke a gauge callback with a mock Observation class injected."""
    import sys
    import types

    # Build a minimal fake opentelemetry.metrics module
    fake_mod = types.ModuleType("opentelemetry.metrics")
    fake_mod.Observation = _FakeObservation  # type: ignore[attr-defined]

    # Also need opentelemetry package module
    fake_pkg = types.ModuleType("opentelemetry")

    saved = {}
    for key in ("opentelemetry", "opentelemetry.metrics"):
        saved[key] = sys.modules.get(key)

    sys.modules["opentelemetry"] = fake_pkg
    sys.modules["opentelemetry.metrics"] = fake_mod
    try:
        return list(callback(None))
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


@pytest.mark.unit
class TestGaugeCallbackObservations:
    """Tests that gauge callbacks return correct Observations from mock pool state."""

    def test_active_count(self) -> None:
        """Active gauge yields Observation with count of in_use=True connections."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {
            "addr1": deque([_make_mock_connection(in_use=True), _make_mock_connection(in_use=False)]),
            "addr2": deque([_make_mock_connection(in_use=True)]),
        }
        driver._pool.connections_reservations = {}

        backend = Neo4jBackend.from_driver(driver)
        callbacks = _capture_gauge_callbacks(driver, backend)

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.connections.active"])
        assert len(obs) == 1
        assert obs[0].value == 2

    def test_idle_count(self) -> None:
        """Idle gauge yields Observation with count of in_use=False connections."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {
            "addr1": deque(
                [
                    _make_mock_connection(in_use=True),
                    _make_mock_connection(in_use=False),
                    _make_mock_connection(in_use=False),
                ]
            ),
        }
        driver._pool.connections_reservations = {}

        backend = Neo4jBackend.from_driver(driver)
        callbacks = _capture_gauge_callbacks(driver, backend)

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.connections.idle"])
        assert len(obs) == 1
        assert obs[0].value == 2

    def test_total_count(self) -> None:
        """Total gauge yields Observation with count of all connections."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {
            "addr1": deque([_make_mock_connection(), _make_mock_connection()]),
            "addr2": deque([_make_mock_connection(), _make_mock_connection()]),
        }
        driver._pool.connections_reservations = {}

        backend = Neo4jBackend.from_driver(driver)
        callbacks = _capture_gauge_callbacks(driver, backend)

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.connections.total"])
        assert len(obs) == 1
        assert obs[0].value == 4

    def test_creating_count(self) -> None:
        """Creating gauge yields Observation with sum of connections_reservations."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {}
        driver._pool.connections_reservations = {"addr1": 2, "addr2": 3}

        backend = Neo4jBackend.from_driver(driver)
        callbacks = _capture_gauge_callbacks(driver, backend)

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.connections.creating"])
        assert len(obs) == 1
        assert obs[0].value == 5

    def test_utilization_ratio(self) -> None:
        """Utilization gauge yields Observation(active / max_pool_size)."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {
            "addr1": deque(
                [
                    _make_mock_connection(in_use=True),
                    _make_mock_connection(in_use=True),
                    _make_mock_connection(in_use=False),
                ]
            ),
        }
        driver._pool.connections_reservations = {}

        backend = Neo4jBackend.from_driver(driver)
        backend._max_connection_pool_size = 50
        callbacks = _capture_gauge_callbacks(driver, backend)

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.utilization"])
        assert len(obs) == 1
        assert obs[0].value == pytest.approx(0.04)  # 2 active / 50 max

    def test_empty_pool_returns_zeros(self) -> None:
        """All gauge callbacks return zero observations when pool is empty."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {}
        driver._pool.connections_reservations = {}

        backend = Neo4jBackend.from_driver(driver)
        # Pin the denominator to a real int so the utilization gauge yields a
        # numeric 0.0 here (a bare MagicMock pool_config would otherwise seed
        # _max_connection_pool_size with a MagicMock).
        backend._max_connection_pool_size = 100
        callbacks = _capture_gauge_callbacks(driver, backend)

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.connections.active"])
        assert obs[0].value == 0

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.connections.idle"])
        assert obs[0].value == 0

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.connections.total"])
        assert obs[0].value == 0

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.utilization"])
        assert obs[0].value == 0.0

    def test_utilization_zero_max_pool_size_falls_back(self) -> None:
        """Utilization gauge falls back to max_size=100 when _max_connection_pool_size is 0."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver._pool.connections = {
            "addr1": deque([_make_mock_connection(in_use=True)]),
        }
        driver._pool.connections_reservations = {}

        backend = Neo4jBackend.from_driver(driver)
        # Force the falsy denominator so the gauge's own `or 100` fallback applies.
        # (from_driver now seeds the denominator from the driver's pool_config; a
        # zero/unreadable ceiling still degrades to the 100 default here.)
        backend._max_connection_pool_size = 0
        callbacks = _capture_gauge_callbacks(driver, backend)

        obs = _invoke_gauge(callbacks["khora.neo4j.pool.utilization"])
        assert len(obs) == 1
        assert obs[0].value == pytest.approx(0.01)  # 1 active / 100 fallback
