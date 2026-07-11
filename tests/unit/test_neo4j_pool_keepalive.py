"""Unit tests for the opt-in Neo4j connection-pool keepalive.

Mirrors ``tests/unit/test_neo4j_pool_metrics_correctness.py``'s
``TestPoolSampler`` / ``TestConnectStartsSampler`` / ``TestSamplerConfig``
structure for the keepalive feature: a default-OFF background task that
fires ``RETURN 1`` pings on idle pooled connections so they are never
idle-dropped before the driver's liveness check would catch a stale
connection.

No real Neo4j here — the driver / pool are mocked. The keepalive's
defining property is **metric isolation**: a ping uses a RAW
``driver.session(...)`` (never ``Neo4jBackend._session`` /
``_InstrumentedSession``), so it must never touch ``acquire_duration`` /
``session.duration`` / the timeout counter.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.storage.backends.neo4j import Neo4jBackend, _cancel_keepalive_task_on_gc

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_keepalive_driver(*, idle: int = 1) -> MagicMock:
    """Shared-driver mock with a pool the keepalive can read deterministically.

    ``idle`` connections are exposed as a single-address deque with
    ``in_use_connection_count`` reporting zero in-use, so
    ``_count_free_connections`` returns exactly ``idle``.
    """
    driver = MagicMock()
    pool = MagicMock()
    pool.connections = {"addr1": deque(MagicMock() for _ in range(idle))}
    pool.connections_reservations = {"addr1": 0}
    pool.in_use_connection_count = MagicMock(return_value=0)
    pool.lock = MagicMock()
    pool.lock.__enter__ = MagicMock(return_value=pool.lock)
    pool.lock.__exit__ = MagicMock(return_value=False)
    pool.pool_config = MagicMock()
    pool.pool_config.max_connection_pool_size = 10
    driver._pool = pool

    # Raw-session path the keepalive ping uses: driver.session(...) returns
    # an AsyncSession with execute_read + close. NOT an async context manager
    # here — the keepalive opens it directly and closes it in a finally.
    def _new_session(*_a: Any, **_kw: Any) -> AsyncMock:
        session = AsyncMock()
        session.execute_read = AsyncMock(return_value=[1])
        session.close = AsyncMock(return_value=None)
        return session

    driver.session.side_effect = _new_session
    return driver


# ===========================================================================
# Default-OFF: zero cost when the keepalive is not enabled.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveDefaultOff:
    @pytest.mark.asyncio
    async def test_connect_does_not_start_keepalive_when_disabled(self) -> None:
        """connect() with pool_keepalive_enabled=False must not spawn a task."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver)  # default: disabled
        assert backend._pool_keepalive_enabled is False

        backend._create_indexes = AsyncMock()
        backend._backfill_native_valid_datetimes = AsyncMock()  # #1472: isolate keepalive from connect() side effects
        backend._register_pool_metrics = MagicMock()

        await backend.connect()
        try:
            assert backend._keepalive_task is None, "disabled keepalive must not start a task"
            await asyncio.sleep(0.05)
            assert driver.session.call_count == 0, "disabled keepalive must fire zero pings"
        finally:
            await backend.disconnect()

    def test_start_pool_keepalive_noop_when_disabled(self) -> None:
        """Calling _start_pool_keepalive directly is inert when disabled."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver)
        backend._start_pool_keepalive()
        assert backend._keepalive_task is None


# ===========================================================================
# Enabled via __init__ AND via from_driver — connect() starts the task.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveStarts:
    @pytest.mark.asyncio
    async def test_connect_starts_keepalive_via_from_driver(self) -> None:
        """from_driver honors pool_keepalive_enabled; connect() starts the task."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(
            driver,
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=50,
        )
        assert backend._pool_keepalive_enabled is True
        assert backend._pool_keepalive_interval_ms == 50

        backend._create_indexes = AsyncMock()
        backend._backfill_native_valid_datetimes = AsyncMock()  # #1472: isolate keepalive from connect() side effects
        backend._register_pool_metrics = MagicMock()

        await backend.connect()
        try:
            assert backend._keepalive_task is not None, "connect() must start the keepalive when enabled"
        finally:
            await backend.disconnect()
        assert backend._keepalive_task is None, "disconnect() must stop the keepalive"

    @pytest.mark.asyncio
    async def test_init_kwargs_set_instance_flags(self) -> None:
        """The public constructor honors the keepalive kwargs on the instance."""
        backend = Neo4jBackend(
            "bolt://localhost:7687",
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=200,
        )
        assert backend._pool_keepalive_enabled is True
        assert backend._pool_keepalive_interval_ms == 200
        # No event loop / task yet — connect() owns task creation.
        assert backend._keepalive_task is None

    @pytest.mark.asyncio
    async def test_start_pool_keepalive_spawns_when_enabled(self) -> None:
        """_start_pool_keepalive creates a task when enabled."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True, pool_keepalive_interval_ms=50)
        backend._start_pool_keepalive()
        try:
            assert backend._keepalive_task is not None
            assert not backend._keepalive_task.done()
        finally:
            await backend._stop_pool_keepalive()


# ===========================================================================
# Idempotent start.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveIdempotentStart:
    @pytest.mark.asyncio
    async def test_double_start_does_not_leak_a_second_task(self) -> None:
        """Calling _start_pool_keepalive twice reuses the first task."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True, pool_keepalive_interval_ms=50)

        backend._start_pool_keepalive()
        first_task = backend._keepalive_task
        assert first_task is not None

        backend._start_pool_keepalive()  # second call — must be a no-op
        assert backend._keepalive_task is first_task, "second _start_pool_keepalive must reuse the first task"

        await backend._stop_pool_keepalive()
        assert backend._keepalive_task is None

    @pytest.mark.asyncio
    async def test_double_connect_does_not_spawn_a_second_task(self) -> None:
        """connect() twice on a shared driver does not spawn a second keepalive."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True, pool_keepalive_interval_ms=50)
        backend._create_indexes = AsyncMock()
        backend._backfill_native_valid_datetimes = AsyncMock()  # #1472: isolate keepalive from connect() side effects
        backend._register_pool_metrics = MagicMock()

        await backend.connect()
        try:
            first_task = backend._keepalive_task
            assert first_task is not None
            await backend.connect()  # second connect — keepalive must be reused
            assert backend._keepalive_task is first_task
        finally:
            await backend.disconnect()


# ===========================================================================
# Clean stop.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_and_clears_task_and_detaches_finalizer(self) -> None:
        """_stop_pool_keepalive cancels + awaits, nulls the task, detaches finalizer."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True, pool_keepalive_interval_ms=50)
        backend._start_pool_keepalive()
        task = backend._keepalive_task
        assert task is not None
        assert backend._keepalive_finalizer is not None

        await backend._stop_pool_keepalive()

        assert backend._keepalive_task is None
        assert task.done(), "the keepalive task must be done after stop"
        assert backend._keepalive_finalizer is None, "stop must detach the finalizer"

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_no_task_running(self) -> None:
        """_stop_pool_keepalive must not raise when nothing is running."""
        backend = Neo4jBackend.from_driver(MagicMock())
        assert backend._keepalive_task is None
        await backend._stop_pool_keepalive()  # must not raise
        assert backend._keepalive_task is None


# ===========================================================================
# Metric isolation (critical): a ping uses a RAW driver.session(...) and
# does NOT go through _session() / _InstrumentedSession.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveMetricIsolation:
    @pytest.mark.asyncio
    async def test_ping_uses_raw_driver_session_not_instrumented_session(self) -> None:
        """A keepalive ping opens driver.session(...) directly, never _session()."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True)

        # Spy on the instrumented helper — it must NOT be touched by the ping.
        session_spy = MagicMock(side_effect=AssertionError("keepalive must not call _session()"))
        backend._session = session_spy

        await backend._keepalive_ping_once()

        session_spy.assert_not_called()
        driver.session.assert_called_once()
        # The ping runs against the configured database via the raw session.
        _, kwargs = driver.session.call_args
        assert kwargs.get("database") == backend._database

    @pytest.mark.asyncio
    async def test_ping_does_not_touch_acquire_or_session_or_timeout_instruments(self) -> None:
        """The keepalive ping records nothing on the read/write hot-path instruments."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True)

        # Wire spies onto every hot-path instrument the keepalive must avoid.
        backend._acquire_duration_histogram = MagicMock()
        backend._acquisition_histogram = MagicMock()
        backend._session_duration_histogram = MagicMock()
        backend._timeout_counter = MagicMock()
        # And the keepalive's own counters, which it SHOULD touch.
        backend._keepalive_pings_counter = MagicMock()
        backend._keepalive_failures_counter = MagicMock()

        await backend._keepalive_ping_once()

        backend._acquire_duration_histogram.record.assert_not_called()
        backend._acquisition_histogram.record.assert_not_called()
        backend._session_duration_histogram.record.assert_not_called()
        backend._timeout_counter.add.assert_not_called()
        # The ping path increments pings (attempt) and, on success, not failures.
        backend._keepalive_pings_counter.add.assert_called_once_with(1)
        backend._keepalive_failures_counter.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_ping_uses_execute_read_return_1_and_closes_session(self) -> None:
        """The ping does a single execute_read(RETURN 1) and closes the session."""
        opened: list[AsyncMock] = []

        driver = MagicMock()
        driver._pool = MagicMock()

        def _new_session(*_a: Any, **_kw: Any) -> AsyncMock:
            session = AsyncMock()
            session.execute_read = AsyncMock(return_value=[1])
            session.close = AsyncMock(return_value=None)
            opened.append(session)
            return session

        driver.session.side_effect = _new_session
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True)

        await backend._keepalive_ping_once()

        assert len(opened) == 1
        opened[0].execute_read.assert_awaited_once()
        opened[0].close.assert_awaited_once()


# ===========================================================================
# Cap: at most min(N, 16) pings per round.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveCap:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("free", "expected"),
        [
            (0, 0),
            (5, 5),  # N < 16
            (16, 16),  # N == 16
            (40, 16),  # N > 16 — capped
        ],
    )
    async def test_one_round_fires_min_n_16_pings(self, free: int, expected: int) -> None:
        """_run_keepalive fires exactly min(free, 16) pings in a single round."""
        backend = Neo4jBackend.from_driver(MagicMock(), pool_keepalive_enabled=True)
        backend._pool_keepalive_interval_ms = 10_000  # long sleep — we cancel after round 1
        backend._count_free_connections = MagicMock(return_value=free)

        pinged = 0

        async def _fake_ping() -> None:
            nonlocal pinged
            pinged += 1

        backend._keepalive_ping_once = _fake_ping

        task = asyncio.create_task(backend._run_keepalive())
        # Let the first round dispatch its ping tasks, then cancel before the
        # next interval. A short sleep lets the created ping tasks run.
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert pinged == expected, f"free={free}: expected {expected} pings, fired {pinged}"


# ===========================================================================
# Failure resilience: a raising ping increments failures, logs (not ERROR),
# and does NOT stall / kill the loop.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveFailureResilience:
    @pytest.mark.asyncio
    async def test_failing_ping_increments_failures_and_logs_debug_not_error(self) -> None:
        """A ping whose session raises bumps the failures counter and logs at DEBUG."""
        driver = MagicMock()
        driver._pool = MagicMock()
        driver.session.side_effect = RuntimeError("connection reset by peer")

        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True)
        backend._keepalive_pings_counter = MagicMock()
        backend._keepalive_failures_counter = MagicMock()

        with patch("khora.storage.backends.neo4j.logger") as mock_logger:
            await backend._keepalive_ping_once()  # must not raise

        backend._keepalive_pings_counter.add.assert_called_once_with(1)
        backend._keepalive_failures_counter.add.assert_called_once_with(1)
        mock_logger.debug.assert_called_once()
        mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_keeps_firing_after_a_ping_raises(self) -> None:
        """A raising ping in round 1 must not stop round 2 from firing."""
        backend = Neo4jBackend.from_driver(MagicMock(), pool_keepalive_enabled=True)
        backend._pool_keepalive_interval_ms = 20
        backend._count_free_connections = MagicMock(return_value=1)

        rounds = 0

        async def _ping() -> None:
            nonlocal rounds
            rounds += 1
            raise RuntimeError("ping blew up")

        backend._keepalive_ping_once = _ping

        task = asyncio.create_task(backend._run_keepalive())
        # ~3 intervals at 20ms — expect multiple rounds despite each raising.
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert rounds >= 2, f"loop stalled after a failing ping; only {rounds} round(s) fired"


# ===========================================================================
# No asyncio.wait_for in the ping path — a slow ping must not be cancelled
# by the loop (the half-closed connection must be allowed to self-heal).
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveNoWaitFor:
    @pytest.mark.asyncio
    async def test_ping_path_does_not_wrap_in_wait_for(self) -> None:
        """Patch asyncio.wait_for and assert the keepalive never calls it."""
        driver = _make_keepalive_driver()
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True)

        with patch("khora.storage.backends.neo4j.asyncio.wait_for") as mock_wait_for:
            await backend._keepalive_ping_once()
            mock_wait_for.assert_not_called()

    @pytest.mark.asyncio
    async def test_slow_ping_runs_to_completion_uncancelled(self) -> None:
        """A slow ping is awaited to completion — not deadline-cancelled."""
        completed = asyncio.Event()

        driver = MagicMock()
        driver._pool = MagicMock()

        def _new_session(*_a: Any, **_kw: Any) -> AsyncMock:
            session = AsyncMock()

            async def _slow_execute_read(*_a: Any, **_kw: Any) -> list[int]:
                await asyncio.sleep(0.1)
                completed.set()
                return [1]

            session.execute_read = _slow_execute_read
            session.close = AsyncMock(return_value=None)
            return session

        driver.session.side_effect = _new_session
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True)

        await backend._keepalive_ping_once()
        assert completed.is_set(), "slow ping was cut short — it must run to completion"


# ===========================================================================
# Detached-task hygiene: the in-flight ping set holds hard refs so pings are
# not GC'd mid-flight, and the done-callback drains the set as pings finish.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveDetachedTaskHygiene:
    @pytest.mark.asyncio
    async def test_in_flight_pings_held_then_drained_by_done_callback(self) -> None:
        """A round holds hard refs to its ping tasks, then the done-callback
        discards each when it completes — no leak, no GC mid-flight.

        Runs the REAL ``_run_keepalive`` (only the ping body is gated) so the
        ``pings`` set + ``add_done_callback(pings.discard)`` wiring is exercised.
        """
        backend = Neo4jBackend.from_driver(MagicMock(), pool_keepalive_enabled=True)
        backend._pool_keepalive_interval_ms = 10_000  # one round, then long sleep
        backend._count_free_connections = MagicMock(return_value=3)

        release = asyncio.Event()
        started = 0

        async def _gated_ping() -> None:
            nonlocal started
            started += 1
            await release.wait()  # stay in-flight until released

        backend._keepalive_ping_once = _gated_ping

        task = asyncio.create_task(backend._run_keepalive())
        # Let the round dispatch all 3 ping tasks and block them in-flight.
        while started < 3:
            await asyncio.sleep(0)
        await asyncio.sleep(0)

        # All 3 pings are alive: the loop must be holding hard refs to them
        # (otherwise they'd be GC-eligible). Count keepalive ping tasks on the
        # loop that are not yet done.
        live_pings = [
            t
            for t in asyncio.all_tasks()
            if t is not task and not t.done() and t.get_coro().__qualname__.endswith("_gated_ping")
        ]
        assert len(live_pings) == 3, f"expected 3 in-flight pings held, found {len(live_pings)}"

        # Release them — the done-callback (pings.discard) drains the set.
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        for t in live_pings:
            await t
        assert all(t.done() for t in live_pings)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ===========================================================================
# Shutdown race: stopping the loop while pings are airborne is clean — the
# loop cancels without an exception escaping and pings are not awaited by it.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveShutdownRace:
    @pytest.mark.asyncio
    async def test_stop_while_pings_in_flight_is_clean(self) -> None:
        """Fire a full round of pings, then stop the loop mid-flight.

        ``_stop_pool_keepalive`` cancels + awaits the loop without raising,
        even though detached ping tasks are still airborne (the loop never
        awaits them, so cancelling the loop cannot surface a ping error).
        """
        backend = Neo4jBackend.from_driver(MagicMock(), pool_keepalive_enabled=True)
        backend._pool_keepalive_interval_ms = 10_000
        backend._count_free_connections = MagicMock(return_value=16)

        release = asyncio.Event()
        started = 0

        async def _gated_ping() -> None:
            nonlocal started
            started += 1
            await release.wait()

        backend._keepalive_ping_once = _gated_ping

        backend._start_pool_keepalive()
        while started < 16:
            await asyncio.sleep(0)

        # Stop the loop while all 16 pings are airborne — must be clean.
        await backend._stop_pool_keepalive()
        assert backend._keepalive_task is None

        # Drain the orphaned ping tasks so the test leaves no pending tasks.
        release.set()
        pending = [t for t in asyncio.all_tasks() if not t.done() and t.get_coro().__qualname__.endswith("_gated_ping")]
        for t in pending:
            await t

    @pytest.mark.asyncio
    async def test_disconnect_while_pings_in_flight_does_not_raise(self) -> None:
        """A real-shaped ping that errors mid-shutdown never escapes disconnect().

        The ping body raises (driver going away), but because pings are
        fire-and-forget and the failure is caught + logged at DEBUG inside
        ``_keepalive_ping_once``, ``disconnect()`` completes cleanly.
        """
        driver = _make_keepalive_driver(idle=4)
        backend = Neo4jBackend.from_driver(driver, pool_keepalive_enabled=True, pool_keepalive_interval_ms=20)
        backend._create_indexes = AsyncMock()
        backend._backfill_native_valid_datetimes = AsyncMock()  # #1472: isolate keepalive from connect() side effects
        backend._register_pool_metrics = MagicMock()

        await backend.connect()
        await asyncio.sleep(0.03)  # let at least one round dispatch pings
        # Simulate the driver going away under in-flight pings.
        driver.session.side_effect = RuntimeError("driver closed")
        await asyncio.sleep(0.03)

        # disconnect() must stop the keepalive cleanly despite ping errors.
        await backend.disconnect()
        assert backend._keepalive_task is None


# ===========================================================================
# _count_free_connections: idle math + getattr fallbacks.
# ===========================================================================


@pytest.mark.unit
class TestCountFreeConnections:
    def test_idle_math_total_minus_active(self) -> None:
        """idle = total - active from a well-formed pool."""
        driver = MagicMock()
        pool = MagicMock()
        pool.connections = {"addr1": deque([MagicMock(), MagicMock(), MagicMock()])}
        pool.in_use_connection_count = MagicMock(return_value=1)
        pool.lock = MagicMock()
        pool.lock.__enter__ = MagicMock(return_value=pool.lock)
        pool.lock.__exit__ = MagicMock(return_value=False)
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        assert backend._count_free_connections() == 2  # 3 total - 1 active

    def test_idle_clamped_when_active_exceeds_total(self) -> None:
        """Transient inconsistency must not yield a negative free count."""
        driver = MagicMock()
        pool = MagicMock()
        pool.connections = {"addr1": deque([MagicMock()])}  # 1 total
        pool.in_use_connection_count = MagicMock(return_value=5)  # lies
        pool.lock = MagicMock()
        pool.lock.__enter__ = MagicMock(return_value=pool.lock)
        pool.lock.__exit__ = MagicMock(return_value=False)
        driver._pool = pool

        backend = Neo4jBackend.from_driver(driver)
        assert backend._count_free_connections() == 0

    def test_returns_zero_when_driver_detached(self) -> None:
        backend = Neo4jBackend.from_driver(MagicMock())
        backend._driver = None
        assert backend._count_free_connections() == 0

    def test_returns_zero_when_no_pool(self) -> None:
        driver = MagicMock(spec=[])  # no _pool attribute
        backend = Neo4jBackend.from_driver(driver)
        assert backend._count_free_connections() == 0

    def test_unlocked_pool_takes_fallback_path(self) -> None:
        """A pool with no ``.lock`` reads via the unlocked branch without crashing."""

        class LocklessPool:
            def __init__(self) -> None:
                self.connections = {"addr1": deque([MagicMock(), MagicMock()])}

            def in_use_connection_count(self, _addr: Any) -> int:
                return 1

        driver = MagicMock()
        driver._pool = LocklessPool()
        assert not hasattr(driver._pool, "lock"), "test needs a pool without .lock"

        backend = Neo4jBackend.from_driver(driver)
        assert backend._count_free_connections() == 1  # 2 total - 1 active

    def test_warns_once_and_returns_zero_on_malformed_pool(self) -> None:
        """A pool whose reads raise warns exactly once and returns 0 thereafter."""

        class BrokenPool:
            @property
            def connections(self) -> dict[str, Any]:
                raise RuntimeError("driver shape drifted mid-read")

        driver = MagicMock()
        driver._pool = BrokenPool()
        backend = Neo4jBackend.from_driver(driver)

        with patch("khora.storage.backends.neo4j.logger") as mock_logger:
            assert backend._count_free_connections() == 0
            mock_logger.warning.assert_called_once()

        # Subsequent failures are silent — no log-storm on every tick.
        with patch("khora.storage.backends.neo4j.logger") as mock_logger:
            assert backend._count_free_connections() == 0
            mock_logger.warning.assert_not_called()


# ===========================================================================
# GC finalizer cancel helper — mirrors _cancel_sampler_task_on_gc coverage.
# ===========================================================================


@pytest.mark.unit
class TestCancelKeepaliveTaskOnGc:
    @pytest.mark.asyncio
    async def test_cancels_a_live_task(self) -> None:
        async def _runs_forever() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(_runs_forever())
        await asyncio.sleep(0)  # let it start
        _cancel_keepalive_task_on_gc(task)
        assert task.cancelled() or task.cancelling() > 0
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_noop_on_done_task(self) -> None:
        async def _instant() -> None:
            return None

        task = asyncio.create_task(_instant())
        await task
        _cancel_keepalive_task_on_gc(task)  # must not raise


# ===========================================================================
# Config: defaults / validation / clamp / env aliases.
# ===========================================================================


@pytest.mark.unit
class TestKeepaliveConfig:
    def test_init_clamps_interval_low(self) -> None:
        backend = Neo4jBackend(
            "bolt://localhost:7687",
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=1,
        )
        assert backend._pool_keepalive_interval_ms == 50

    def test_init_clamps_interval_high(self) -> None:
        backend = Neo4jBackend(
            "bolt://localhost:7687",
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=10**9,
        )
        assert backend._pool_keepalive_interval_ms == 60_000

    def test_init_defaults_off_with_15s_interval(self) -> None:
        backend = Neo4jBackend("bolt://localhost:7687")
        assert backend._pool_keepalive_enabled is False
        assert backend._pool_keepalive_interval_ms == 15000

    def test_from_driver_keepalive_defaults_off(self) -> None:
        backend = Neo4jBackend.from_driver(MagicMock())
        assert backend._pool_keepalive_enabled is False
        assert backend._pool_keepalive_interval_ms == 15000

    def test_from_driver_honors_keepalive_kwargs(self) -> None:
        backend = Neo4jBackend.from_driver(
            MagicMock(),
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=250,
        )
        assert backend._pool_keepalive_enabled is True
        assert backend._pool_keepalive_interval_ms == 250

    def test_config_defaults(self) -> None:
        from khora.config.schema import Neo4jConfig

        cfg = Neo4jConfig(url="bolt://localhost:7687")
        assert cfg.pool_keepalive_enabled is False
        assert cfg.pool_keepalive_interval_ms == 15000

    def test_config_rejects_interval_below_range(self) -> None:
        from pydantic import ValidationError

        from khora.config.schema import Neo4jConfig

        with pytest.raises(ValidationError):
            Neo4jConfig(url="bolt://localhost:7687", pool_keepalive_interval_ms=49)

    def test_config_rejects_interval_above_range(self) -> None:
        from pydantic import ValidationError

        from khora.config.schema import Neo4jConfig

        with pytest.raises(ValidationError):
            Neo4jConfig(url="bolt://localhost:7687", pool_keepalive_interval_ms=60_001)

    def test_config_accepts_in_range(self) -> None:
        from khora.config.schema import Neo4jConfig

        cfg = Neo4jConfig(url="bolt://localhost:7687", pool_keepalive_enabled=True, pool_keepalive_interval_ms=750)
        assert cfg.pool_keepalive_enabled is True
        assert cfg.pool_keepalive_interval_ms == 750

    def test_from_config_reads_keepalive_fields(self) -> None:
        """from_config must forward the keepalive fields into the backend.

        Mirrors ``TestSamplerConfig.test_from_config_reads_new_fields``. If
        from_config drops the keepalive kwargs, a config-driven keepalive
        silently never starts.
        """
        from khora.config.schema import Neo4jConfig

        cfg = Neo4jConfig(
            url="bolt://localhost:7687",
            pool_keepalive_enabled=True,
            pool_keepalive_interval_ms=750,
        )
        backend = Neo4jBackend.from_config(cfg)
        assert backend._pool_keepalive_enabled is True
        assert backend._pool_keepalive_interval_ms == 750
