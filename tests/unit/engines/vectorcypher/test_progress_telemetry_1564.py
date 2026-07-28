"""#1564: extraction progress telemetry (engine side).

Covers the ``remember_batch`` window loop half:

- an unconditional per-window INFO fires even when extraction produced no
  entities - the case where the pre-existing "Streaming pipeline batch store"
  INFO (nested under ``if all_entities:``) goes silent;
- a ``khora.vectorcypher.remember_batch.window`` span carries the window
  progress attributes;
- the ``min_extraction_tokens`` gate, previously a bare ``continue``, now
  counts the dropped chunks under ``outcome=skipped_short``.

Reuses the streaming-engine harness from ``test_batch_diagnostics_1410``.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

import pytest
from loguru import logger
from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from khora.engines.vectorcypher import engine as engine_mod
from khora.telemetry import _otel as _otel_module
from tests.test_helpers.otel import reset_khora_telemetry
from tests.unit.engines.vectorcypher.test_batch_diagnostics_1410 import (
    _LONG_CONTENT,
    _make_streaming_engine,
    _patch_extract_multi,
)

pytestmark = pytest.mark.unit


class _RecordingCounter:
    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


class _CaptureLogs:
    def __init__(self, level: str) -> None:
        self._level = level
        self.messages: list[str] = []

    def __enter__(self) -> _CaptureLogs:
        self._sink_id = logger.add(lambda m: self.messages.append(str(m)), level=self._level)
        return self

    def __exit__(self, *exc: object) -> None:
        logger.remove(self._sink_id)

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


class TestWindowProgressLogging:
    @pytest.mark.asyncio
    async def test_window_info_emitted_when_extraction_fails(self, monkeypatch) -> None:
        """The window INFO fires even though extraction yielded zero entities."""
        engine = _make_streaming_engine()
        _patch_extract_multi(monkeypatch, fail=True)

        with _CaptureLogs("INFO") as logs:
            result = await engine.remember_batch(
                [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
            )

        assert result.processed == 1
        assert result.entities == 0
        # The unconditional window line surfaced progress + the error count ...
        assert "extraction window 1/1" in logs.text
        assert "extraction_errors=" in logs.text
        # ... where the nested "batch store" INFO stayed silent (no entities).
        assert "Streaming pipeline batch store" not in logs.text

    @pytest.mark.asyncio
    async def test_window_info_reports_chunk_totals_on_success(self, monkeypatch) -> None:
        engine = _make_streaming_engine()
        _patch_extract_multi(monkeypatch, fail=False)

        with _CaptureLogs("INFO") as logs:
            await engine.remember_batch(
                [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
            )

        # Single window: chunks-done equals chunks-total, and the count is the
        # chunker's output (not hard-coded, so a chunker tweak won't break this).
        m = re.search(r"extraction window 1/1: chunks (\d+)/(\d+)", logs.text)
        assert m is not None, logs.text
        done, total = int(m.group(1)), int(m.group(2))
        assert done == total and done > 0


class TestSkippedShortCounter:
    @pytest.mark.asyncio
    async def test_skipped_short_counted_at_min_token_gate(self, monkeypatch) -> None:
        engine = _make_streaming_engine()
        # A non-conversation document whose (single) chunk sits below the gate.
        engine._vc_config.min_extraction_tokens = 50
        rec = _RecordingCounter()
        monkeypatch.setattr(engine_mod, "_EXTRACTION_CHUNK_OUTCOME_COUNTER", rec)
        _patch_extract_multi(monkeypatch, fail=False)

        result = await engine.remember_batch(
            [{"content": "a short quarterly note about the alpha project review"}],
            uuid4(),
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        assert result.processed == 1
        skipped = [(v, a) for v, a in rec.adds if a.get("outcome") == "skipped_short"]
        assert skipped, f"no skipped_short add recorded; got {rec.adds}"
        assert skipped[0][0] >= 1


class TestWindowSpan:
    @pytest.fixture(autouse=True)
    def _install_exporter(self):
        reset_khora_telemetry()
        tp = TracerProvider()
        exporter = InMemorySpanExporter()
        tp.add_span_processor(SimpleSpanProcessor(exporter))
        _otel_trace.set_tracer_provider(tp)
        # Rebind khora's cached tracer so trace_span picks up the new provider.
        _otel_module._TRACER = _otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION)
        yield exporter
        exporter.shutdown()
        reset_khora_telemetry()

    @pytest.mark.asyncio
    async def test_window_span_carries_progress_attributes(self, _install_exporter, monkeypatch) -> None:
        exporter = _install_exporter
        engine = _make_streaming_engine()
        _patch_extract_multi(monkeypatch, fail=False)

        await engine.remember_batch(
            [{"content": _LONG_CONTENT}], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"]
        )

        windows = [s for s in exporter.get_finished_spans() if s.name == "khora.vectorcypher.remember_batch.window"]
        assert len(windows) == 1
        attrs = windows[0].attributes
        assert attrs["window_index"] == 1
        assert attrs["window_count"] == 1
        # Single window with one document: every chunk is done and accounted for.
        wc = attrs["window_chunks"]
        assert wc > 0
        assert attrs["chunks_done"] == wc
        assert attrs["chunks_total"] == wc
        assert attrs["docs_processed"] == 1
        assert attrs["extraction_errors"] == 0
