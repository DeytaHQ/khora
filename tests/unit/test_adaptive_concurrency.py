"""Unit tests for the adaptive concurrency controller (AIMD) in neo4j.py."""

from __future__ import annotations

import asyncio
import time

import pytest

from khora.storage.backends.neo4j import (
    _AIMD_DECREASE_FACTOR,
    _CALM_WINDOW_SEC,
    _CONTENTION_THRESHOLD,
    _MIN_ENTITY_CONCURRENCY,
    _MIN_RELATIONSHIP_CONCURRENCY,
    _AdaptiveConcurrencyController,
    _AdaptiveRelationshipSemaphore,
    _ContentionTracker,
    _EntityKeyGate,
    _LatencyTracker,
)

# ---------------------------------------------------------------------------
# _ContentionTracker
# ---------------------------------------------------------------------------


class TestContentionTracker:
    def test_empty_tracker_has_zero_count(self) -> None:
        tracker = _ContentionTracker(window=1.0)
        assert tracker.count() == 0

    def test_record_and_count(self) -> None:
        tracker = _ContentionTracker(window=10.0)
        tracker.record()
        tracker.record()
        assert tracker.count() == 2

    def test_old_events_expire(self) -> None:
        tracker = _ContentionTracker(window=0.05)
        tracker.record()
        assert tracker.count() == 1
        time.sleep(0.06)
        assert tracker.count() == 0


# ---------------------------------------------------------------------------
# _LatencyTracker (EWMA)
# ---------------------------------------------------------------------------


class TestLatencyTracker:
    def test_first_update_never_signals_contention(self) -> None:
        tracker = _LatencyTracker()
        assert tracker.update(1.0) is False
        assert tracker.mean == 1.0

    def test_stable_latencies_no_contention(self) -> None:
        tracker = _LatencyTracker()
        # Feed stable latencies — none should trigger contention
        for _ in range(20):
            assert tracker.update(0.2) is False

    def test_spike_triggers_contention(self) -> None:
        tracker = _LatencyTracker()
        # Build up a baseline of 0.2s
        for _ in range(20):
            tracker.update(0.2)
        # A 5x spike should trigger contention
        assert tracker.update(1.0) is True

    def test_gradual_increase_no_contention(self) -> None:
        tracker = _LatencyTracker()
        # Gradually increasing latencies — EWMA adapts, no spike
        for i in range(50):
            latency = 0.1 + i * 0.01
            tracker.update(latency)
        # After gradual ramp, a slightly higher value should NOT trigger
        assert tracker.update(0.62) is False

    def test_mean_tracks_input(self) -> None:
        tracker = _LatencyTracker(alpha=1.0)  # alpha=1.0 means instant tracking
        tracker.update(5.0)
        assert tracker.mean == 5.0
        tracker.update(10.0)
        assert tracker.mean == 10.0


# ---------------------------------------------------------------------------
# _AdaptiveConcurrencyController
# ---------------------------------------------------------------------------


class TestAdaptiveConcurrencyController:
    def test_starts_at_ceiling(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        assert ctrl.entity_limit == 16
        assert ctrl.relationship_limit == 8

    @pytest.mark.asyncio
    async def test_multiplicative_decrease_on_entity_contention(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        # Directly record contention events (simulating EWMA spike detection)
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl._entity_contention.record()
        await ctrl.maybe_adjust()
        assert ctrl.entity_limit == max(_MIN_ENTITY_CONCURRENCY, int(16 * _AIMD_DECREASE_FACTOR))
        assert ctrl.relationship_limit == max(_MIN_RELATIONSHIP_CONCURRENCY, int(8 * _AIMD_DECREASE_FACTOR))

    @pytest.mark.asyncio
    async def test_multiplicative_decrease_on_relationship_contention(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl._relationship_contention.record()
        await ctrl.maybe_adjust()
        # Both limits decrease when either category has contention
        assert ctrl.entity_limit == max(_MIN_ENTITY_CONCURRENCY, int(16 * _AIMD_DECREASE_FACTOR))
        assert ctrl.relationship_limit == max(_MIN_RELATIONSHIP_CONCURRENCY, int(8 * _AIMD_DECREASE_FACTOR))

    @pytest.mark.asyncio
    async def test_ewma_signals_contention(self) -> None:
        """record_entity_write with a latency spike should produce contention events."""
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        # Build baseline
        for _ in range(20):
            ctrl.record_entity_write(0.2)
        assert ctrl._entity_contention.count() == 0
        # Spike — should record contention
        ctrl.record_entity_write(2.0)
        assert ctrl._entity_contention.count() >= 1

    @pytest.mark.asyncio
    async def test_floor_is_respected(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=4, relationship_ceiling=4)
        # Force many decreases
        for _ in range(10):
            for _ in range(_CONTENTION_THRESHOLD):
                ctrl._entity_contention.record()
            # Reset the last_decrease time to allow another decrease
            ctrl._last_decrease = 0.0
            await ctrl.maybe_adjust()
        assert ctrl.entity_limit >= _MIN_ENTITY_CONCURRENCY
        assert ctrl.relationship_limit >= _MIN_RELATIONSHIP_CONCURRENCY

    @pytest.mark.asyncio
    async def test_additive_increase_after_calm(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        # First decrease
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl._entity_contention.record()
        await ctrl.maybe_adjust()
        decreased_entity = ctrl.entity_limit
        decreased_rel = ctrl.relationship_limit
        # Clear contention events (simulate them expiring) and simulate calm window
        ctrl._entity_contention._events.clear()
        ctrl._relationship_contention._events.clear()
        ctrl._last_decrease = time.monotonic() - _CALM_WINDOW_SEC - 1
        ctrl._last_increase = ctrl._last_decrease
        await ctrl.maybe_adjust()
        # Should have increased by 1
        assert ctrl.entity_limit == decreased_entity + 1
        assert ctrl.relationship_limit == decreased_rel + 1

    @pytest.mark.asyncio
    async def test_no_increase_above_ceiling(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        # Already at ceiling, simulate calm
        ctrl._last_decrease = time.monotonic() - _CALM_WINDOW_SEC - 1
        ctrl._last_increase = ctrl._last_decrease
        await ctrl.maybe_adjust()
        assert ctrl.entity_limit == 16
        assert ctrl.relationship_limit == 8

    @pytest.mark.asyncio
    async def test_decrease_is_rate_limited(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl._entity_contention.record()
        await ctrl.maybe_adjust()
        first_limit = ctrl.entity_limit
        # Immediately try again — should NOT decrease further (rate-limited)
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl._entity_contention.record()
        await ctrl.maybe_adjust()
        assert ctrl.entity_limit == first_limit


# ---------------------------------------------------------------------------
# _EntityKeyGate with adaptive controller
# ---------------------------------------------------------------------------


class TestEntityKeyGateAdaptive:
    @pytest.mark.asyncio
    async def test_static_gate_respects_max_concurrent(self) -> None:
        """Without controller, gate uses static max_concurrent."""
        gate = _EntityKeyGate(max_concurrent=2)
        assert gate._effective_limit == 2

    @pytest.mark.asyncio
    async def test_adaptive_gate_follows_controller(self) -> None:
        """With controller, gate uses dynamic entity_limit."""
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        gate = _EntityKeyGate(max_concurrent=16, controller=ctrl)
        assert gate._effective_limit == 16
        # Trigger decrease via direct contention recording
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl._entity_contention.record()
        await ctrl.maybe_adjust()
        assert gate._effective_limit == max(_MIN_ENTITY_CONCURRENCY, int(16 * _AIMD_DECREASE_FACTOR))


# ---------------------------------------------------------------------------
# _AdaptiveRelationshipSemaphore
# ---------------------------------------------------------------------------


class TestAdaptiveRelationshipSemaphore:
    @pytest.mark.asyncio
    async def test_acquire_release(self) -> None:
        """Basic acquire/release works."""
        sem = _AdaptiveRelationshipSemaphore(max_concurrent=2)
        async with sem.acquire():
            assert sem._active == 1
        assert sem._active == 0

    @pytest.mark.asyncio
    async def test_respects_limit(self) -> None:
        """Can't exceed effective limit."""
        sem = _AdaptiveRelationshipSemaphore(max_concurrent=1)
        acquired = []

        async def _worker(idx: int) -> None:
            async with sem.acquire():
                acquired.append(idx)
                assert sem._active <= 1
                await asyncio.sleep(0.01)

        await asyncio.gather(_worker(0), _worker(1))
        assert len(acquired) == 2

    @pytest.mark.asyncio
    async def test_follows_controller(self) -> None:
        """Semaphore uses controller's dynamic limit."""
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        sem = _AdaptiveRelationshipSemaphore(max_concurrent=8, controller=ctrl)
        assert sem._effective_limit == 8
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl._relationship_contention.record()
        await ctrl.maybe_adjust()
        assert sem._effective_limit == max(_MIN_RELATIONSHIP_CONCURRENCY, int(8 * _AIMD_DECREASE_FACTOR))
