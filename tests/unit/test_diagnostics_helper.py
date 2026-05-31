"""Tests for the ADR-001 diagnostics test helper.

The helper lives at ``tests/test_helpers/diagnostics.py``; this test
file just exercises ``assert_no_silent_degradation`` against the result
shapes the convention covers (``metadata`` dict, ``engine_info`` dict,
and a bare dict).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.test_helpers.diagnostics import assert_no_silent_degradation


@dataclass
class _ResultWithMetadata:
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ResultWithEngineInfo:
    engine_info: dict[str, Any] = field(default_factory=dict)


@pytest.mark.unit
class TestAssertNoSilentDegradation:
    def test_passes_on_empty_metadata(self) -> None:
        result = _ResultWithMetadata(metadata={})
        # Must not raise.
        assert_no_silent_degradation(result)

    def test_passes_on_empty_engine_info(self) -> None:
        result = _ResultWithEngineInfo(engine_info={})
        assert_no_silent_degradation(result)

    def test_passes_on_empty_lists(self) -> None:
        result = _ResultWithMetadata(metadata={"degradations": [], "errors": []})
        assert_no_silent_degradation(result)

    def test_passes_when_only_skipped_is_present(self) -> None:
        # SkipReason is a declared choice, not a silent failure.
        result = _ResultWithMetadata(metadata={"skipped": [{"op_kind": "foo", "reason": "op_not_supported_by_engine"}]})
        assert_no_silent_degradation(result)

    def test_raises_on_non_empty_degradations(self) -> None:
        result = _ResultWithEngineInfo(
            engine_info={
                "degradations": [{"component": "chronicle.bm25", "reason": "channel_exception"}],
            }
        )
        with pytest.raises(AssertionError, match="silent degradation"):
            assert_no_silent_degradation(result)

    def test_raises_on_non_empty_errors(self) -> None:
        result = _ResultWithMetadata(
            metadata={
                "errors": [
                    {
                        "component": "chronicle.temporal_channel",
                        "reason": "boom",
                        "exception": "RuntimeError",
                    }
                ]
            }
        )
        with pytest.raises(AssertionError, match="silent degradation"):
            assert_no_silent_degradation(result)

    def test_works_on_bare_dict(self) -> None:
        # Caller might hand the bag in directly.
        assert_no_silent_degradation({})
        with pytest.raises(AssertionError):
            assert_no_silent_degradation({"degradations": [{"component": "x", "reason": "y"}]})

    def test_metadata_wins_over_engine_info(self) -> None:
        # Hybrid object: metadata is checked first, engine_info ignored if
        # metadata is a dict (matching dream / remember result shapes).
        class _Both:
            metadata: dict[str, Any] = {}
            engine_info: dict[str, Any] = {
                "degradations": [{"component": "x", "reason": "y"}],
            }

        # Must not raise because metadata is the canonical bag and is empty.
        assert_no_silent_degradation(_Both())
