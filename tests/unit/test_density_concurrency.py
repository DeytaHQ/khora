"""Unit tests for density-based concurrency optimizations in neo4j.py."""

from __future__ import annotations

from khora.storage.backends.neo4j import (
    _HIGH_DENSITY_ENTITY_BATCH_SIZE,
    _HIGH_DENSITY_ENTITY_THRESHOLD,
    _HIGH_DENSITY_REL_BATCH_SIZE,
    _HIGH_DENSITY_REL_THRESHOLD,
    _HUB_OVERLAP_THRESHOLD,
)


class TestDensityConstants:
    """Verify density thresholds are sane."""

    def test_entity_threshold_above_low_density(self) -> None:
        """Slack/Linear produce 0-30 entities — threshold must be well above."""
        assert _HIGH_DENSITY_ENTITY_THRESHOLD >= 50

    def test_entity_batch_size_smaller_than_default(self) -> None:
        """High-density sub-batch must be smaller than default (100)."""
        assert _HIGH_DENSITY_ENTITY_BATCH_SIZE < 100

    def test_rel_threshold_above_low_density(self) -> None:
        """Low-density sources produce <100 relationships."""
        assert _HIGH_DENSITY_REL_THRESHOLD >= 200

    def test_rel_batch_size_smaller_than_default(self) -> None:
        """High-density sub-batch must be smaller than default (200)."""
        assert _HIGH_DENSITY_REL_BATCH_SIZE < 200

    def test_hub_overlap_threshold_is_reasonable(self) -> None:
        """Jaccard threshold should be between 0.1 and 0.5."""
        assert 0.1 <= _HUB_OVERLAP_THRESHOLD <= 0.5
