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
        # Record enough contention events to trigger decrease
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl.record_entity_contention()
        await ctrl.maybe_adjust()
        assert ctrl.entity_limit == max(_MIN_ENTITY_CONCURRENCY, int(16 * _AIMD_DECREASE_FACTOR))
        assert ctrl.relationship_limit == max(_MIN_RELATIONSHIP_CONCURRENCY, int(8 * _AIMD_DECREASE_FACTOR))

    @pytest.mark.asyncio
    async def test_multiplicative_decrease_on_relationship_contention(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=16, relationship_ceiling=8)
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl.record_relationship_contention()
        await ctrl.maybe_adjust()
        # Both limits decrease when either category has contention
        assert ctrl.entity_limit == max(_MIN_ENTITY_CONCURRENCY, int(16 * _AIMD_DECREASE_FACTOR))
        assert ctrl.relationship_limit == max(_MIN_RELATIONSHIP_CONCURRENCY, int(8 * _AIMD_DECREASE_FACTOR))

    @pytest.mark.asyncio
    async def test_floor_is_respected(self) -> None:
        ctrl = _AdaptiveConcurrencyController(entity_ceiling=4, relationship_ceiling=4)
        # Force many decreases
        for _ in range(10):
            for _ in range(_CONTENTION_THRESHOLD):
                ctrl.record_entity_contention()
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
            ctrl.record_entity_contention()
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
            ctrl.record_entity_contention()
        await ctrl.maybe_adjust()
        first_limit = ctrl.entity_limit
        # Immediately try again — should NOT decrease further (rate-limited)
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl.record_entity_contention()
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
        # Trigger decrease
        for _ in range(_CONTENTION_THRESHOLD):
            ctrl.record_entity_contention()
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
            ctrl.record_relationship_contention()
        await ctrl.maybe_adjust()
        assert sem._effective_limit == max(_MIN_RELATIONSHIP_CONCURRENCY, int(8 * _AIMD_DECREASE_FACTOR))
