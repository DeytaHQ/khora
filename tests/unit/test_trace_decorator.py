"""Tests for the @trace decorator."""

from __future__ import annotations

import enum
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from khora.telemetry.logfire_integration import Span
from khora.telemetry.trace_decorator import trace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingSpan(Span):
    """Span that records all attribute writes for assertion."""

    __slots__ = ("attributes",)

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)


def _make_recording_trace_span():
    """Create a mock trace_span that yields a recording span and records calls."""
    recording_span = _RecordingSpan()
    calls: list[tuple[str, dict]] = []

    from contextlib import contextmanager

    @contextmanager
    def mock_trace_span(name, /, **attributes):
        calls.append((name, attributes))
        yield recording_span

    return mock_trace_span, recording_span, calls


# ---------------------------------------------------------------------------
# Tests: basic decorator forms
# ---------------------------------------------------------------------------


class TestTraceDecoratorForms:
    """Test the different @trace invocation forms."""

    @pytest.mark.asyncio
    async def test_bare_trace_async(self) -> None:
        """@trace on async fn uses khora.{fn.__name__} as span name."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace
            async def my_operation(namespace_id: UUID, limit: int = 10):
                return ["a", "b"]

            result = await my_operation(uuid4(), limit=5)

        assert result == ["a", "b"]
        assert len(calls) == 1
        assert calls[0][0] == "khora.my_operation"
        assert calls[0][1]["limit"] == 5

    @pytest.mark.asyncio
    async def test_explicit_name(self) -> None:
        """@trace("custom.name") uses the explicit span name."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("custom.span.name")
            async def my_func(x: int = 1):
                return x

            await my_func(42)

        assert calls[0][0] == "custom.span.name"
        assert calls[0][1]["x"] == 42

    def test_sync_function(self) -> None:
        """@trace works on non-async functions."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.sync_op")
            def compute(a: int, b: int) -> int:
                return a + b

            result = compute(3, 7)

        assert result == 10
        assert calls[0][0] == "khora.sync_op"
        assert calls[0][1] == {"a": 3, "b": 7}

    @pytest.mark.asyncio
    async def test_empty_parens(self) -> None:
        """@trace() works the same as @trace — auto-derives name."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace()
            async def another_func():
                return True

            await another_func()

        assert calls[0][0] == "khora.another_func"


# ---------------------------------------------------------------------------
# Tests: argument capture rules
# ---------------------------------------------------------------------------


class TestArgumentCapture:
    """Test argument extraction and type handling."""

    @pytest.mark.asyncio
    async def test_self_skipped(self) -> None:
        """self parameter is not captured as a span attribute."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            class MyClass:
                @trace("khora.test")
                async def method(self, entity_id: UUID):
                    return None

            obj = MyClass()
            test_id = uuid4()
            await obj.method(test_id)

        assert "self" not in calls[0][1]
        assert calls[0][1]["entity_id"] == str(test_id)

    @pytest.mark.asyncio
    async def test_uuid_conversion(self) -> None:
        """UUID args are converted to strings."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test")
            async def fn(namespace_id: UUID, entity_id: UUID):
                return None

            ns = uuid4()
            eid = uuid4()
            await fn(ns, eid)

        assert calls[0][1]["namespace_id"] == str(ns)
        assert calls[0][1]["entity_id"] == str(eid)

    @pytest.mark.asyncio
    async def test_complex_object_skipped(self) -> None:
        """Dict and other complex objects are silently excluded."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test")
            async def fn(config: dict, name: str):
                return None

            await fn({"key": "value"}, "test")

        assert "config" not in calls[0][1]
        assert calls[0][1]["name"] == "test"

    @pytest.mark.asyncio
    async def test_list_tuple_length(self) -> None:
        """Lists and tuples are captured as {name}_count."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test")
            async def fn(items: list, ids: tuple):
                return None

            await fn([1, 2, 3], (4, 5))

        assert "items" not in calls[0][1]
        assert calls[0][1]["items_count"] == 3
        assert "ids" not in calls[0][1]
        assert calls[0][1]["ids_count"] == 2

    @pytest.mark.asyncio
    async def test_set_frozenset_length(self) -> None:
        """Sets and frozensets are captured as {name}_count."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test")
            async def fn(tags: set, ids: frozenset):
                return None

            await fn({"a", "b", "c"}, frozenset([1, 2]))

        assert "tags" not in calls[0][1]
        assert calls[0][1]["tags_count"] == 3
        assert "ids" not in calls[0][1]
        assert calls[0][1]["ids_count"] == 2

    @pytest.mark.asyncio
    async def test_enum_value(self) -> None:
        """Enum args are converted to their .value."""
        mock_ts, span, calls = _make_recording_trace_span()

        class Direction(enum.Enum):
            INCOMING = "incoming"
            OUTGOING = "outgoing"

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test")
            async def fn(direction: Direction):
                return None

            await fn(Direction.OUTGOING)

        assert calls[0][1]["direction"] == "outgoing"


# ---------------------------------------------------------------------------
# Tests: include / exclude
# ---------------------------------------------------------------------------


class TestIncludeExclude:
    """Test parameter filtering."""

    @pytest.mark.asyncio
    async def test_exclude(self) -> None:
        """Excluded params are not captured."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test", exclude={"content"})
            async def fn(content: str, namespace_id: UUID):
                return None

            ns = uuid4()
            await fn("sensitive text", ns)

        assert "content" not in calls[0][1]
        assert calls[0][1]["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_include(self) -> None:
        """Only included params are captured."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test", include={"namespace_id"})
            async def fn(namespace_id: UUID, query_embedding: list, config: dict):
                return None

            ns = uuid4()
            await fn(ns, [0.1, 0.2], {"key": "val"})

        assert calls[0][1] == {"namespace_id": str(ns)}

    def test_include_and_exclude_raises(self) -> None:
        """Providing both include and exclude raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            trace("khora.test", include={"a"}, exclude={"b"})

    def test_include_unknown_param_raises(self) -> None:
        """include with param names not in the function signature raises ValueError."""
        with pytest.raises(ValueError, match="include param.*not_a_param.*not found"):

            @trace("khora.test", include={"not_a_param"})
            async def fn(x: int) -> int:
                return x

    def test_exclude_unknown_param_raises(self) -> None:
        """exclude with param names not in the function signature raises ValueError."""
        with pytest.raises(ValueError, match="exclude param.*bogus.*not found"):

            @trace("khora.test", exclude={"bogus"})
            async def fn(x: int) -> int:
                return x

    def test_include_valid_params_accepted(self) -> None:
        """include with valid param names does not raise."""

        @trace("khora.test", include={"x"})
        async def fn(x: int, y: int) -> int:
            return x + y

        assert fn.__name__ == "fn"

    def test_exclude_valid_params_accepted(self) -> None:
        """exclude with valid param names does not raise."""

        @trace("khora.test", exclude={"y"})
        async def fn(x: int, y: int) -> int:
            return x + y

        assert fn.__name__ == "fn"


# ---------------------------------------------------------------------------
# Tests: result extractor
# ---------------------------------------------------------------------------


class TestResultExtractor:
    """Test return value interception."""

    @pytest.mark.asyncio
    async def test_result_extractor_list(self) -> None:
        """Result extractor adds attributes from return value."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test", result=lambda r: {"result_count": len(r)})
            async def fn():
                return [1, 2, 3]

            result = await fn()

        assert result == [1, 2, 3]
        assert span.attributes["result_count"] == 3

    @pytest.mark.asyncio
    async def test_result_extractor_tuple(self) -> None:
        """Result extractor works with tuple returns."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test", result=lambda r: {"chunks": r[0], "entities": r[1]})
            async def fn():
                return (10, 5, 3)

            result = await fn()

        assert result == (10, 5, 3)
        assert span.attributes["chunks"] == 10
        assert span.attributes["entities"] == 5

    @pytest.mark.asyncio
    async def test_result_extractor_error_silenced(self) -> None:
        """Result extractor errors are silently ignored."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test", result=lambda r: {"count": len(r)})
            async def fn():
                return None  # len(None) will raise

            result = await fn()

        assert result is None
        assert "count" not in span.attributes  # error silenced


# ---------------------------------------------------------------------------
# Tests: exception propagation and short-circuit
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test exception handling and no-op behavior."""

    @pytest.mark.asyncio
    async def test_exception_propagated(self) -> None:
        """Exceptions from the decorated function are not swallowed."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test")
            async def fn():
                raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                await fn()

    @pytest.mark.asyncio
    async def test_short_circuit_when_no_logfire(self) -> None:
        """When _HAS_LOGFIRE is False, function is called directly with no span."""
        call_count = 0

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):

            @trace("khora.test")
            async def fn(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            result = await fn(21)

        assert result == 42
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_functools_wraps_preserved(self) -> None:
        """Decorated function preserves __name__ and __doc__."""

        @trace("khora.test")
        async def my_documented_function():
            """My docstring."""
            return True

        assert my_documented_function.__name__ == "my_documented_function"
        assert my_documented_function.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_defaults_captured(self) -> None:
        """Default argument values are captured as span attributes."""
        mock_ts, span, calls = _make_recording_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.trace_decorator.trace_span", mock_ts),
        ):

            @trace("khora.test")
            async def fn(limit: int = 10, offset: int = 0):
                return []

            await fn()

        assert calls[0][1]["limit"] == 10
        assert calls[0][1]["offset"] == 0
