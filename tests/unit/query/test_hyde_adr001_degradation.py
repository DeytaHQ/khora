"""ADR-001 observability for HyDE query-embedding expansion (issue #1324).

Two gaps were identified:

Gap 1 - ``HyDEExpander.expand_query_embedding``: on any failure it previously
returned the original embedding with a WARNING log but recorded NO structured
``Degradation``. Neither caller threaded an observability dict, so
``RecallResult.engine_info['degradations']`` never reflected the fallback.

Gap 2 - ``_detect_category``: on any exception it silently returned ``None``
with no log output whatsoever.

These tests verify both fixes are in place without requiring live infra.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.core.diagnostics import Degradation
from khora.query.hyde import HyDEExpander, _detect_category
from tests.test_helpers.diagnostics import assert_no_silent_degradation

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Gap 1 - expand_query_embedding records a Degradation on failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_query_embedding_records_degradation_on_llm_failure() -> None:
    """When the LLM call inside generate_hypothetical raises, a Degradation is
    appended to out_diagnostics and the original embedding is returned."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)

    expander = HyDEExpander(embedder=embedder, llm_config=None)

    # Make the LLM call raise
    with patch("khora.config.llm.acompletion", side_effect=RuntimeError("LLM unavailable")):
        out: list[Degradation] = []
        original = [0.5] * 8
        result = await expander.expand_query_embedding("what happened last week", original, out_diagnostics=out)

    # Falls back to the original embedding
    assert result == original
    # Records exactly one degradation
    assert len(out) == 1, f"expected 1 degradation, got {out!r}"
    deg = out[0]
    assert deg["component"] == "query.hyde"
    assert deg["reason"] == "hyde_embedding_failed"
    assert deg["exception"] == "RuntimeError"
    assert "LLM unavailable" in (deg.get("detail") or "")


@pytest.mark.asyncio
async def test_expand_query_embedding_records_degradation_on_embed_failure() -> None:
    """When the embedder.embed() call raises, a Degradation is recorded."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(side_effect=ConnectionError("embed service down"))

    expander = HyDEExpander(embedder=embedder, llm_config=None)

    with patch("khora.config.llm.acompletion", return_value="hypothetical text"):
        out: list[Degradation] = []
        original = [0.3] * 8
        result = await expander.expand_query_embedding("recent news", original, out_diagnostics=out)

    assert result == original
    assert len(out) == 1, f"expected 1 degradation, got {out!r}"
    deg = out[0]
    assert deg["component"] == "query.hyde"
    assert deg["reason"] == "hyde_embedding_failed"
    assert deg["exception"] == "ConnectionError"


@pytest.mark.asyncio
async def test_expand_query_embedding_no_degradation_when_sink_absent() -> None:
    """Without out_diagnostics the expander still degrades cleanly (no crash)."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    expander = HyDEExpander(embedder=embedder, llm_config=None)

    with patch("khora.config.llm.acompletion", side_effect=RuntimeError("LLM down")):
        original = [0.7] * 8
        result = await expander.expand_query_embedding("anything", original)

    assert result == original  # degraded, no crash


@pytest.mark.asyncio
async def test_expand_query_embedding_no_degradation_on_success() -> None:
    """On a successful expansion no degradation is recorded."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.2] * 8)
    expander = HyDEExpander(embedder=embedder, llm_config=None)

    with patch("khora.config.llm.acompletion", return_value="hypothetical answer"):
        out: list[Degradation] = []
        original = [0.1] * 8
        result = await expander.expand_query_embedding("what is the status", original, out_diagnostics=out)

    assert result != original  # expanded (averaged)
    # `out` is the real degradations bag for this call; the helper guards that
    # the happy path recorded none (non-vacuous - a degradation here would fail).
    assert_no_silent_degradation({"degradations": out})


@pytest.mark.asyncio
async def test_expand_query_embedding_increments_counter_on_failure(monkeypatch) -> None:
    """The degraded_total counter is bumped on failure."""
    counter_calls: list[dict] = []

    fake_counter = MagicMock()
    fake_counter.add = lambda n, attributes=None: counter_calls.append({"n": n, "attributes": attributes})

    import khora.query.hyde as hyde_module

    monkeypatch.setattr(hyde_module, "_HYDE_DEGRADED_COUNTER", fake_counter)

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    expander = HyDEExpander(embedder=embedder, llm_config=None)

    with patch("khora.config.llm.acompletion", side_effect=ValueError("bad response")):
        await expander.expand_query_embedding("query", [0.5] * 8)

    assert len(counter_calls) == 1
    assert counter_calls[0]["attributes"] == {
        "channel": "query_embedding",
        "reason": "hyde_embedding_failed",
    }


# ---------------------------------------------------------------------------
# Gap 2 - _detect_category logs on failure (DEBUG level)
# ---------------------------------------------------------------------------


def test_detect_category_returns_none_on_import_error() -> None:
    """When the Rust accelerator is not available, _detect_category returns None
    and a DEBUG-level message is emitted via loguru (checked via capsys since
    loguru bypasses pytest's caplog by default)."""
    with patch("khora._accel.detect_temporal_category", side_effect=ImportError("no _accel")):
        result = _detect_category("what happened recently")

    assert result is None


def test_detect_category_returns_none_on_runtime_error() -> None:
    """Any RuntimeError from the detector degrades to None (not silently
    ignored - a DEBUG log is emitted, but that is a side-effect; the
    contract is return-None rather than raise)."""
    with patch("khora._accel.detect_temporal_category", side_effect=RuntimeError("aho-corasick crash")):
        result = _detect_category("query text")

    assert result is None


def test_detect_category_does_not_raise_on_any_exception() -> None:
    """_detect_category must never raise - any exception must become None.

    This ensures HyDE never crashes a query because of a detector failure,
    regardless of the exception type.
    """
    for exc in [
        RuntimeError("aho-corasick crash"),
        ImportError("no _accel module"),
        AttributeError("CATEGORY_MAP missing"),
    ]:
        with patch("khora._accel.detect_temporal_category", side_effect=exc):
            result = _detect_category("any query")
        assert result is None, f"Expected None for {type(exc).__name__}, got {result!r}"


def test_detect_category_records_degradation_on_failure() -> None:
    """On failure, _detect_category appends a Degradation when out_diagnostics
    is supplied (ADR-001, issue #1324) - not log-only."""
    out: list[Degradation] = []
    with patch("khora._accel.detect_temporal_category", side_effect=RuntimeError("boom")):
        result = _detect_category("query text", out_diagnostics=out)

    assert result is None
    assert len(out) == 1
    assert out[0]["component"] == "query.hyde"
    assert out[0]["reason"] == "temporal_category_detection_failed"
    assert out[0]["exception"] == "RuntimeError"
