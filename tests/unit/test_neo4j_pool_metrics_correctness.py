"""Tests for the correctness fixes to Neo4j pool metrics.

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
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ConnectionAcquisitionTimeoutError, ServiceUnavailable

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


class FakeAsyncSession:
    """Lightweight stand-in for ``neo4j._async.work.session.AsyncSession``.

    Faithfully implements the subset of ``AsyncSession`` that
    :class:`_InstrumentedSession` and our tests exercise:

    * ``_connect``: the real pool-bind hook. ``_InstrumentedSession`` patches
      this on the instance to record ``khora.neo4j.pool.acquire_duration``.
    * ``_connection``: set to ``None`` when no bolt is bound and to a
      non-None sentinel after ``_connect`` succeeds. The real driver sets
      this back to ``None`` in ``_disconnect`` (see
      ``neo4j/_async/work/session.py:144`` — ``self._connection = None``)
      and our tests flip it to ``None`` to simulate a retry-triggered
      disconnect before the next ``_connect``.
    * ``run(...)``: connects if needed, then returns a stub result.
    * ``execute_read(fn)`` / ``execute_write(fn)``: connects (every retry),
      then awaits ``fn(tx)``; on retriable errors, disconnects and retries
      up to ``max_retries`` times, exactly matching
      ``AsyncSession._run_transaction`` (``neo4j/_async/work/session.py:494``),
      which calls ``_open_transaction`` → ``_connect`` on every attempt.

    Using a real object (not an ``AsyncMock``) keeps method attribute
    assignments (``_connect = _timed_connect``) from being intercepted by
    the mock framework.
    """

    def __init__(
        self,
        *,
        connect_delay: float = 0.0,
        connect_error: BaseException | None = None,
        query_delay: float = 0.0,
        max_retries: int = 6,
    ) -> None:
        self._connection: Any = None
        self._connect_delay = connect_delay
        self._connect_error = connect_error
        self._query_delay = query_delay
        self._max_retries = max_retries
        self.connect_calls = 0

    async def _connect(self, *_args: Any, **_kwargs: Any) -> None:
        """Simulate pool-bind.

        The ``_InstrumentedSession._install_connect_wrap`` logic replaces
        this method on the instance, so tests that assert histogram
        observations exercise the wrap, not this bare hook.
        """
        self.connect_calls += 1
        if self._connect_delay > 0:
            await asyncio.sleep(self._connect_delay)
        if self._connect_error is not None:
            raise self._connect_error
        self._connection = object()  # mark bound

    async def _disconnect(self) -> None:
        """Mirror ``AsyncSession._disconnect`` — drops the bolt reference.

        The real driver does this at ``neo4j/_async/work/session.py:144``
        (``self._connection = None``). ``execute_read`` / ``execute_write``
        invoke this between retries so the next attempt has to re-bind via
        ``_connect`` again.
        """
        self._connection = None

    async def run(self, *_args: Any, **_kwargs: Any) -> Any:
        if self._connection is None:
            await self._connect()
        if self._query_delay > 0:
            await asyncio.sleep(self._query_delay)
        return MagicMock()

    async def _execute_tx(self, fn: Any) -> Any:
        """Shared retry-loop mirroring ``AsyncSession._run_transaction``."""
        errors: list[BaseException] = []
        for _attempt in range(self._max_retries):
            try:
                # Each attempt re-binds — matches ``_open_transaction``
                # calling ``_connect`` inside ``_run_transaction``'s loop.
                if self._connection is None:
                    await self._connect()
                tx = MagicMock()  # fake AsyncManagedTransaction
                return await fn(tx)
            except ConnectionAcquisitionTimeoutError:
                # Real driver does NOT retry ConnectionAcquisitionTimeoutError
                # (pool saturation is not transient). Propagate immediately.
                raise
            except Exception as exc:  # noqa: BLE001 — intentional test surface
                errors.append(exc)
                # Match ``_run_transaction`` which disconnects between attempts.
                await self._disconnect()
                if not _is_retriable(exc):
                    raise
                continue
        # Exhausted retries — surface the last error.
        raise errors[-1]

    async def execute_read(self, fn: Any, *_a: Any, **_kw: Any) -> Any:
        return await self._execute_tx(fn)

    async def execute_write(self, fn: Any, *_a: Any, **_kw: Any) -> Any:
        return await self._execute_tx(fn)


def _is_retriable(exc: BaseException) -> bool:
    """Mimic ``Neo4jError.is_retryable`` for the exceptions our tests raise."""
    # ServiceUnavailable / SessionExpired / TransientError are retriable;
    # ConnectionAcquisitionTimeoutError is NOT (handled above).
    return isinstance(exc, ServiceUnavailable)


def _make_session_with_connect_hook(connect_delay: float = 0.0) -> FakeAsyncSession:
    """Build a FakeAsyncSession whose ``_connect`` mimics a pool bind.

    Kept for backward compatibility with existing tests. Prefer
    :class:`FakeAsyncSession` directly for new tests.
    """
    return FakeAsyncSession(connect_delay=connect_delay)


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

        async def _work(_tx: Any) -> None:
            return None

        async with backend._session() as s:
            await s.run("RETURN 1")
            await s.run("RETURN 2")
            await s.execute_read(_work)

        assert len(samples) == 1, f"expected 1 record per bind, got {len(samples)}"

    @pytest.mark.asyncio
    async def test_records_again_after_reconnect(self) -> None:
        """If the session rebinds (``_connection = None`` then another call),
        the histogram records a second observation. Simulates the real
        ``AsyncSession._disconnect`` (``neo4j/_async/work/session.py:144``,
        ``self._connection = None``) that precedes each retry inside
        ``_run_transaction``.
        """
        session = _make_session_with_connect_hook()
        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        samples: list[float] = []
        backend._acquire_duration_histogram = MagicMock()
        backend._acquire_duration_histogram.record = lambda v, attributes=None, **_: samples.append(v)

        async with backend._session() as s:
            await s.run("RETURN 1")  # bind #1
            # simulate retry / reconnect scenario: release the connection,
            # matching ``AsyncSession._disconnect`` which sets
            # ``self._connection = None`` before the next ``_connect``.
            s._inner._connection = None
            await s.run("RETURN 2")  # bind #2

        assert len(samples) == 2, f"expected 2 records across 2 binds, got {len(samples)}"

    @pytest.mark.asyncio
    async def test_excludes_query_and_retry_time(self) -> None:
        """Acquire is 10 ms; the query sleeps 100 ms after. The histogram must
        record ~10 ms, not ~110 ms — slow-query time must not be conflated
        with acquire time."""
        session = FakeAsyncSession(connect_delay=0.01, query_delay=0.1)
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

    @pytest.mark.asyncio
    async def test_execute_read_retry_records_multiple_acquires(self) -> None:
        """``execute_read`` retries on ``ServiceUnavailable`` — each attempt re-acquires
        a fresh connection (``_open_transaction`` → ``_connect`` in
        ``AsyncSession._run_transaction``). The histogram must record *each* acquire,
        not fold them into a single observation.
        """
        session = FakeAsyncSession()
        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        samples: list[float] = []
        backend._acquire_duration_histogram = MagicMock()
        backend._acquire_duration_histogram.record = lambda v, attributes=None, **_: samples.append(v)

        attempts: list[int] = []

        async def _work(_tx: Any) -> str:
            attempts.append(1)
            if len(attempts) < 2:
                # Raise a retriable error on the first attempt. FakeAsyncSession's
                # ``_execute_tx`` will call ``_disconnect`` (setting
                # ``_connection = None``) and loop, re-invoking ``_connect``.
                raise ServiceUnavailable("transient network blip")
            return "ok"

        async with backend._session() as s:
            result = await s.execute_read(_work)

        assert result == "ok"
        assert len(attempts) == 2, f"expected 2 attempts, got {len(attempts)}"
        assert len(samples) == 2, f"expected 2 acquire observations (one per attempt), got {len(samples)}"


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
        ``run``/``execute_read``/``execute_write``) increments the counter exactly once.

        Uses a :class:`FakeAsyncSession` that honours the real
        ``execute_read`` / ``execute_write`` contract (calls ``_connect``
        then invokes the callable).
        """
        err = ConnectionAcquisitionTimeoutError("pool exhausted")
        session = FakeAsyncSession(connect_error=err)

        driver = _make_driver_with_session(session)
        backend = Neo4jBackend.from_driver(driver)

        adds: list[int] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **_: adds.append(v)

        async def _work(_tx: Any) -> Any:
            return None

        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with backend._session() as s:
                if entry_method == "run":
                    await s.run("RETURN 1")
                else:
                    await getattr(s, entry_method)(_work)

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
        """Real ``AsyncDriver`` with a zero-size pool deterministically raises
        ``ConnectionAcquisitionTimeoutError`` on the very first acquire.
        Routing a ``DualNodeManager`` through ``Neo4jBackend._session`` must
        bump the counter exactly once — proves the instrumentation observes
        real ``execute_read`` invocations, not a mock shortcut.

        No real Neo4j server is needed: ``max_connection_pool_size=0`` fails
        the acquire before any network I/O happens.
        """
        driver = AsyncGraphDatabase.driver(
            "bolt://127.0.0.1:1",
            auth=("neo4j", "password"),
            max_connection_pool_size=0,
            connection_acquisition_timeout=0.05,
        )
        try:
            backend = Neo4jBackend.from_driver(driver)

            adds: list[int] = []
            backend._timeout_counter = MagicMock()
            backend._timeout_counter.add = lambda v, **_: adds.append(v)

            manager = DualNodeManager(driver, pool_backend=backend)

            async def _work(tx: Any) -> list[Any]:
                await tx.run("RETURN 1")
                return []

            # Exercises: DualNodeManager._session -> backend._session
            # -> _InstrumentedSession -> real session._connect times out.
            with pytest.raises(ConnectionAcquisitionTimeoutError):
                async with manager._session() as s:
                    await s.execute_read(_work)

            assert sum(adds) == 1
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_counter_increments_twice_for_two_burst_timeouts(self) -> None:
        """Two timeouts across both entry paths -> counter == 2.

        Each ``driver.session()`` call returns a fresh session in the real
        driver, so the proxy's ``_connect`` wrap does not nest. We emulate
        that by returning a fresh :class:`FakeAsyncSession` per call.
        """
        err = ConnectionAcquisitionTimeoutError("pool exhausted")

        driver = MagicMock()

        def new_session_ctx(*_a: Any, **_kw: Any) -> MagicMock:
            session = FakeAsyncSession(connect_error=err)
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        driver.session.side_effect = new_session_ctx

        backend = Neo4jBackend.from_driver(driver)
        manager = DualNodeManager(driver, pool_backend=backend)

        adds: list[int] = []
        backend._timeout_counter = MagicMock()
        backend._timeout_counter.add = lambda v, **_: adds.append(v)

        # Path 1: Neo4jBackend direct (representing get_entity-style calls).
        with pytest.raises(ConnectionAcquisitionTimeoutError):
            async with backend._session() as s:
                await s.run("MATCH (n) RETURN n")

        # Path 2: DualNodeManager wired to backend (via pool_backend).
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
        session = FakeAsyncSession(connect_delay=0.04, connect_error=err)
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

    def test_sample_pool_once_without_pool_lock(self) -> None:
        """Pools that do not expose ``.lock`` must take the unlocked fallback path
        without crashing, and produce a valid snapshot.
        """

        class LocklessPool:
            """Pool without a ``lock`` attribute — simulates a driver-shape drift."""

            def __init__(self) -> None:
                self.connections = {"addr1": deque([MagicMock(), MagicMock()])}
                self.connections_reservations = {"addr1": 1}
                self.pool_config = MagicMock()
                self.pool_config.max_connection_pool_size = 10

            def in_use_connection_count(self, _addr: Any) -> int:
                return 1

        driver = MagicMock()
        driver._pool = LocklessPool()
        assert not hasattr(driver._pool, "lock"), "test needs a pool without .lock"

        backend = Neo4jBackend.from_driver(driver)
        sample = backend._sample_pool_once()

        # Unlocked branch produced a valid snapshot, no exception.
        assert sample is not None
        assert sample["active"] == 1.0
        assert sample["total"] == 2.0
        assert sample["idle"] == 1.0
        assert sample["creating"] == 1.0
        assert sample["utilization"] == pytest.approx(0.1)

    def test_sample_pool_once_returns_none_when_unlocked_read_raises(self) -> None:
        """When there is no ``lock`` AND reading ``connections`` raises, the sampler
        must still degrade gracefully to ``None`` (no crash, no infinite warnings)."""

        class BrokenLocklessPool:
            """No ``lock``, and ``connections`` access raises on the unlocked path."""

            @property
            def connections(self) -> dict[str, Any]:
                raise RuntimeError("driver shape drifted mid-read")

        driver = MagicMock()
        driver._pool = BrokenLocklessPool()
        backend = Neo4jBackend.from_driver(driver)

        with patch("khora.storage.backends.neo4j.logger") as mock_logger:
            assert backend._sample_pool_once() is None
            mock_logger.warning.assert_called_once()

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

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sampler_ac3_fifteen_second_burst_at_default_interval(self) -> None:
        """AC3-faithful: 15-second burst at the default 500 ms interval must
        yield ≥ 10 samples on each of the 5 sampled histograms.

        This doubles the synthetic 200 ms/10 ms test by running the sampler
        loop for the full AC3 window at production cadence, catching
        cadence regressions that the compressed test can mask. No real
        Neo4j is needed: ``_sample_pool_once`` is patched to return a
        deterministic dict.

        Slack: 15s / 500ms = 30 ticks nominal; we allow ≥ 28 to tolerate
        scheduler jitter (two dropped ticks).

        Skipped by default (``@pytest.mark.slow``). To run:
        ``uv run pytest -m slow tests/unit/test_neo4j_pool_metrics_correctness.py``.
        """
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
        # Default production cadence — do NOT compress.
        backend._pool_sampler_interval_ms = 500

        # Deterministic snapshot — we're testing the cadence, not the read logic.
        backend._sample_pool_once = lambda: {
            "active": 1.0,
            "idle": 0.0,
            "total": 1.0,
            "creating": 0.0,
            "utilization": 0.1,
        }

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
        try:
            # 15-second AC3 window.
            await asyncio.sleep(15.0)
        finally:
            await backend._stop_pool_sampler()

        # 15s / 500ms = 30 ticks. Allow 2 dropped ticks for scheduler jitter.
        for key, count in observed.items():
            assert count >= 28, f"{key}: expected >=28 samples over 15s, got {count}"

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

    def test_sampler_acquires_pool_lock(self) -> None:
        """Each sample must take pool.lock so the snapshot is consistent with
        driver mutations (pool.acquire / pool.release mutate inside the lock)."""
        driver = MagicMock()
        pool = MagicMock()

        # Track lock enter/exit counts.
        lock_enters = 0
        lock_exits = 0

        class TrackingLock:
            def __enter__(self_lock) -> Any:  # noqa: N805
                nonlocal lock_enters
                lock_enters += 1
                return self_lock

            def __exit__(self_lock, *exc: Any) -> bool:  # noqa: N805
                nonlocal lock_exits
                lock_exits += 1
                return False

        pool.lock = TrackingLock()
        pool.connections = {"addr1": deque([MagicMock()])}
        pool.connections_reservations = {}
        pool.in_use_connection_count = MagicMock(return_value=0)
        pool.pool_config = MagicMock()
        pool.pool_config.max_connection_pool_size = 10
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        sample = backend._sample_pool_once()
        assert sample is not None
        assert lock_enters == 1, f"expected 1 lock enter, got {lock_enters}"
        assert lock_exits == 1, f"expected 1 lock exit, got {lock_exits}"

    @pytest.mark.asyncio
    async def test_sampler_task_does_not_leak_on_double_start(self) -> None:
        """Calling _start_pool_sampler twice must not spawn a second task."""
        driver = MagicMock()
        pool = MagicMock()
        pool.connections = {}
        pool.connections_reservations = {}
        pool.lock = MagicMock()
        pool.lock.__enter__ = MagicMock(return_value=pool.lock)
        pool.lock.__exit__ = MagicMock(return_value=False)
        pool.pool_config = MagicMock()
        pool.pool_config.max_connection_pool_size = 10
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        backend._pool_sampler_enabled = True
        backend._pool_sampler_interval_ms = 50

        backend._start_pool_sampler()
        first_task = backend._sampler_task
        assert first_task is not None

        backend._start_pool_sampler()  # second call — must be a no-op
        assert backend._sampler_task is first_task, "second _start_pool_sampler must reuse the first task"

        await backend._stop_pool_sampler()
        assert backend._sampler_task is None


# ---------------------------------------------------------------------------
# connect() on the shared-driver path starts/skips the sampler per the
# from_driver kwarg, and emits the khora.neo4j.pool.sampled.* histograms.
# ---------------------------------------------------------------------------


def _make_sampler_driver() -> MagicMock:
    """Shared-driver mock with a pool the sampler can read deterministically."""
    driver = MagicMock()
    pool = MagicMock()
    pool.connections = {"addr1": deque([MagicMock()])}
    pool.connections_reservations = {"addr1": 0}
    pool.in_use_connection_count = MagicMock(return_value=1)
    pool.lock = MagicMock()
    pool.lock.__enter__ = MagicMock(return_value=pool.lock)
    pool.lock.__exit__ = MagicMock(return_value=False)
    pool.pool_config = MagicMock()
    pool.pool_config.max_connection_pool_size = 10
    driver._pool = pool
    return driver


def _wire_sampled_recorders(backend: Neo4jBackend) -> dict[str, int]:
    """Replace the 5 sampled histograms with counting recorders."""
    observed: dict[str, int] = {k: 0 for k in ("active", "idle", "total", "creating", "utilization")}
    for key in observed:
        attr = f"_sampled_{key}_histogram"
        setattr(backend, attr, MagicMock())

        def make_recorder(k: str):
            def record(_v: float, attributes: Any = None, **_kw: Any) -> None:
                observed[k] += 1

            return record

        getattr(backend, attr).record = make_recorder(key)
    return observed


@pytest.mark.unit
class TestConnectStartsSampler:
    """connect() on the shared-driver path honors the from_driver sampler kwarg."""

    @pytest.mark.asyncio
    async def test_connect_starts_sampler_and_emits_when_enabled(self) -> None:
        driver = _make_sampler_driver()
        backend = Neo4jBackend.from_driver(driver, pool_sampler_enabled=True, pool_sampler_interval_ms=10)

        # Isolate the sampler: stub out the other connect() side effects.
        backend._create_indexes = AsyncMock()
        backend._register_pool_metrics = MagicMock()
        observed = _wire_sampled_recorders(backend)

        await backend.connect()
        try:
            assert backend._sampler_task is not None, "connect() must start the sampler when enabled"
            await asyncio.sleep(0.2)  # ~20 ticks at 10ms
        finally:
            await backend.disconnect()

        assert backend._sampler_task is None, "disconnect() must stop the sampler"
        for key, count in observed.items():
            assert count >= 5, f"{key}: expected the sampler to emit, got {count}"

    @pytest.mark.asyncio
    async def test_connect_does_not_start_sampler_when_disabled(self) -> None:
        driver = _make_sampler_driver()
        backend = Neo4jBackend.from_driver(driver)  # default: pool_sampler_enabled=False

        backend._create_indexes = AsyncMock()
        backend._register_pool_metrics = MagicMock()
        observed = _wire_sampled_recorders(backend)

        await backend.connect()
        try:
            assert backend._sampler_task is None, "default connect() must NOT start the sampler"
            await asyncio.sleep(0.05)
        finally:
            await backend.disconnect()

        assert all(count == 0 for count in observed.values()), f"disabled sampler must emit nothing; got {observed}"


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

    def test_from_driver_honors_sampler_kwargs(self) -> None:
        """from_driver wires its sampler kwargs into the instance fields
        (was previously hardcoded to False / 500)."""
        driver = MagicMock()
        backend = Neo4jBackend.from_driver(
            driver,
            pool_sampler_enabled=True,
            pool_sampler_interval_ms=250,
        )
        assert backend._pool_sampler_enabled is True
        assert backend._pool_sampler_interval_ms == 250

    def test_from_driver_sampler_defaults_off(self) -> None:
        """from_driver leaves the sampler off by default."""
        backend = Neo4jBackend.from_driver(MagicMock())
        assert backend._pool_sampler_enabled is False
        assert backend._pool_sampler_interval_ms == 500


# ---------------------------------------------------------------------------
# from_driver: gauge/utilization denominator seeded from the shared driver's
# real pool ceiling (so the gauge denominator equals the sampler's).
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromDriverMaxPoolSize:
    def test_seeds_max_pool_size_from_driver_pool_config(self) -> None:
        """from_driver reads ``_pool.pool_config.max_connection_pool_size`` so the
        gauge denominator matches the sampler's observed ceiling (not the 100
        fallback)."""
        driver = MagicMock()
        pool = MagicMock()
        pool.pool_config = MagicMock()
        pool.pool_config.max_connection_pool_size = 37  # != 100 default
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        assert backend._max_connection_pool_size == 37

    def test_falls_back_to_100_when_pool_config_missing(self) -> None:
        """A driver whose pool exposes no readable ceiling degrades to the
        neo4j default (100), matching the gauge's own ``or 100`` fallback."""
        driver = MagicMock(spec=[])  # no _pool attribute at all
        backend = Neo4jBackend.from_driver(driver)
        assert backend._max_connection_pool_size == 100

    def test_falls_back_to_100_when_ceiling_is_zero(self) -> None:
        """A zero ceiling is falsy → degrades to 100 (never a 0 denominator)."""
        driver = MagicMock()
        pool = MagicMock()
        pool.pool_config = MagicMock()
        pool.pool_config.max_connection_pool_size = 0
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        assert backend._max_connection_pool_size == 100


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
    # Owns DualNodeManager._session fallback when no pool_backend is wired.
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
            "with pool_backend=backend) so khora.neo4j.pool.* metrics "
            "observe them:\n" + "\n".join(offenders)
        )
