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


def test_extraction_second_pass_defaults_off() -> None:
    """#1420: the batch second-pass relationship extraction is a cost opt-in."""
    assert PipelineSettings().extraction_second_pass is False


def test_extraction_second_pass_env_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KHORA_PIPELINES_EXTRACTION_SECOND_PASS", "true")
    assert PipelineSettings().extraction_second_pass is True
