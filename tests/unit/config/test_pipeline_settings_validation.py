"""Validation coverage for ``PipelineSettings`` numeric bounds (Issue #933)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from khora.config.schema import PipelineSettings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [0, -5])
def test_pending_processor_max_concurrent_rejects_non_positive(value: int) -> None:
    """0 / negative would spin a zero-worker pool that drains nothing (#933)."""
    with pytest.raises(ValidationError):
        PipelineSettings(pending_processor_max_concurrent=value)


def test_pending_processor_max_concurrent_accepts_one() -> None:
    assert PipelineSettings(pending_processor_max_concurrent=1).pending_processor_max_concurrent == 1
