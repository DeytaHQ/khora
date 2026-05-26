"""Unit tests for the request-observability helpers in
``khora.extraction.embedders._request_telemetry``.

These helpers attach best-effort span attributes around upstream LLM /
embedding requests:

* ``parse_rate_limit_headers`` / ``set_rate_limit_attributes`` read provider
  rate-limit headers off the response *after* it returns.
* ``connector_snapshot`` / ``set_connector_attributes`` read aiohttp connector
  contention counters off the shared session *before* the request awaits.

The contract every test below pins down: these functions are defensive and
must never raise into the request path, and the setters must record exactly
the expected attribute keys/values on a span.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from khora.extraction.embedders._request_telemetry import (
    connector_snapshot,
    parse_rate_limit_headers,
    set_connector_attributes,
    set_rate_limit_attributes,
)


class _FakeSpan:
    """Minimal span stand-in recording set_attribute() calls into a dict."""

    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


def _response_with(*, hidden: dict | None = None, headers: dict | None = None) -> Any:
    """Build a fake litellm response carrying the given header sources.

    ``hidden`` populates ``_hidden_params["additional_headers"]``; ``headers``
    populates ``_headers``. Either may be omitted.
    """

    class _Resp:
        pass

    resp = _Resp()
    if hidden is not None:
        resp._hidden_params = {"additional_headers": hidden}
    if headers is not None:
        resp._headers = headers
    return resp


# ---------------------------------------------------------------------------
# parse_rate_limit_headers
# ---------------------------------------------------------------------------


class TestParseRateLimitHeaders:
    def test_headers_via_hidden_params(self) -> None:
        """Headers from _hidden_params['additional_headers'] are coerced."""
        resp = _response_with(
            hidden={
                "x-ratelimit-remaining-requests": "59",
                "x-ratelimit-remaining-tokens": "149000",
                "retry-after": "12",
                "x-ratelimit-reset-requests": "1s",
                "x-ratelimit-reset-tokens": "6m0s",
            }
        )

        attrs = parse_rate_limit_headers(resp)

        assert attrs == {
            "ratelimit.remaining_requests": 59,
            "ratelimit.remaining_tokens": 149000,
            "retry_after": 12,
            "ratelimit.reset_requests": "1s",
            "ratelimit.reset_tokens": "6m0s",
        }
        # remaining_*/retry_after coerced to int; reset_* kept as str.
        assert isinstance(attrs["ratelimit.remaining_requests"], int)
        assert isinstance(attrs["ratelimit.remaining_tokens"], int)
        assert isinstance(attrs["retry_after"], int)
        assert isinstance(attrs["ratelimit.reset_requests"], str)
        assert isinstance(attrs["ratelimit.reset_tokens"], str)

    def test_headers_via_underscore_headers(self) -> None:
        """Headers from response._headers are coerced identically."""
        resp = _response_with(
            headers={
                "x-ratelimit-remaining-requests": "5",
                "x-ratelimit-remaining-tokens": "100",
                "retry-after": "3",
                "x-ratelimit-reset-requests": "20ms",
                "x-ratelimit-reset-tokens": "1m",
            }
        )

        attrs = parse_rate_limit_headers(resp)

        assert attrs == {
            "ratelimit.remaining_requests": 5,
            "ratelimit.remaining_tokens": 100,
            "retry_after": 3,
            "ratelimit.reset_requests": "20ms",
            "ratelimit.reset_tokens": "1m",
        }

    def test_provider_prefixed_keys_matched_after_strip(self) -> None:
        """``llm_provider-`` prefixed header names match after the prefix strip."""
        resp = _response_with(
            headers={
                "llm_provider-x-ratelimit-remaining-requests": "42",
                "llm_provider-x-ratelimit-reset-tokens": "7s",
            }
        )

        attrs = parse_rate_limit_headers(resp)

        assert attrs["ratelimit.remaining_requests"] == 42
        assert attrs["ratelimit.reset_tokens"] == "7s"

    def test_uppercase_header_names_lowercased(self) -> None:
        """Header keys are matched case-insensitively (lowercased on collect)."""
        resp = _response_with(headers={"X-RateLimit-Remaining-Requests": "7"})

        attrs = parse_rate_limit_headers(resp)

        assert attrs["ratelimit.remaining_requests"] == 7

    def test_non_numeric_retry_after_falls_back_to_string(self) -> None:
        """An HTTP-date retry-after that won't int() falls back to the raw string."""
        resp = _response_with(headers={"retry-after": "Mon, 01 Jan 2035 00:00:00 GMT"})

        attrs = parse_rate_limit_headers(resp)

        assert attrs["retry_after"] == "Mon, 01 Jan 2035 00:00:00 GMT"
        assert isinstance(attrs["retry_after"], str)

    def test_non_numeric_remaining_falls_back_to_string(self) -> None:
        """A non-numeric remaining-requests value falls back to the raw string."""
        resp = _response_with(headers={"x-ratelimit-remaining-requests": "n/a"})

        attrs = parse_rate_limit_headers(resp)

        assert attrs["ratelimit.remaining_requests"] == "n/a"

    def test_both_sources_merged(self) -> None:
        """Both header sources contribute; later (._headers) wins on collision."""
        resp = _response_with(
            hidden={"x-ratelimit-remaining-requests": "1"},
            headers={"x-ratelimit-remaining-tokens": "2"},
        )

        attrs = parse_rate_limit_headers(resp)

        assert attrs["ratelimit.remaining_requests"] == 1
        assert attrs["ratelimit.remaining_tokens"] == 2

    def test_absent_sources_returns_empty(self) -> None:
        """A response with no header sources returns {} and does not raise."""
        assert parse_rate_limit_headers(_response_with()) == {}

    def test_empty_sources_returns_empty(self) -> None:
        """Empty/None header containers yield {} without raising."""
        assert parse_rate_limit_headers(_response_with(hidden={}, headers={})) == {}
        resp = _response_with()
        resp._hidden_params = {"additional_headers": None}
        resp._headers = None
        assert parse_rate_limit_headers(resp) == {}

    def test_none_response_returns_empty(self) -> None:
        """A None response is tolerated and yields {}."""
        assert parse_rate_limit_headers(None) == {}

    def test_non_dict_hidden_params_tolerated(self) -> None:
        """A non-dict ``_hidden_params`` is ignored, not crashed on."""
        resp = _response_with(headers={"x-ratelimit-remaining-requests": "7"})
        resp._hidden_params = "not-a-dict"  # litellm should never do this, but be safe
        # The _headers source still parses; the bad _hidden_params is skipped.
        assert parse_rate_limit_headers(resp) == {"ratelimit.remaining_requests": 7}

    def test_unrelated_headers_ignored(self) -> None:
        """Headers not in the spec are dropped."""
        resp = _response_with(headers={"content-type": "application/json", "x-request-id": "abc"})

        assert parse_rate_limit_headers(resp) == {}

    def test_limit_requests_surfaced_in_dict(self) -> None:
        """``x-ratelimit-limit-requests`` round-trips into the dict, coerced to int.

        This key feeds the DEBUG ``{remaining}/{limit}`` log line; it is NOT a
        span attribute (see the span-allowlist test below).
        """
        resp = _response_with(
            headers={
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "59",
            }
        )

        attrs = parse_rate_limit_headers(resp)

        assert attrs["ratelimit.limit_requests"] == 60
        assert isinstance(attrs["ratelimit.limit_requests"], int)
        assert attrs["ratelimit.remaining_requests"] == 59

    def test_limit_requests_non_numeric_falls_back_to_string(self) -> None:
        """A non-numeric limit-requests value falls back to the raw string."""
        resp = _response_with(headers={"x-ratelimit-limit-requests": "unlimited"})

        attrs = parse_rate_limit_headers(resp)

        assert attrs["ratelimit.limit_requests"] == "unlimited"


# ---------------------------------------------------------------------------
# set_rate_limit_attributes
# ---------------------------------------------------------------------------


class TestSetRateLimitAttributes:
    def test_records_expected_keys_and_values(self) -> None:
        span = _FakeSpan()
        resp = _response_with(
            headers={
                "x-ratelimit-remaining-requests": "59",
                "retry-after": "12",
                "x-ratelimit-reset-tokens": "6m0s",
            }
        )

        set_rate_limit_attributes(span, resp)

        assert span.attrs == {
            "ratelimit.remaining_requests": 59,
            "retry_after": 12,
            "ratelimit.reset_tokens": "6m0s",
        }

    def test_no_headers_sets_nothing(self) -> None:
        span = _FakeSpan()

        set_rate_limit_attributes(span, _response_with())

        assert span.attrs == {}

    def test_limit_requests_excluded_from_span_attrs(self) -> None:
        """``ratelimit.limit_requests`` round-trips through the parser but the
        span-attribute allowlist excludes it."""
        span = _FakeSpan()
        resp = _response_with(
            headers={
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "59",
            }
        )

        # Sanity: the parser surfaces limit_requests...
        assert "ratelimit.limit_requests" in parse_rate_limit_headers(resp)

        set_rate_limit_attributes(span, resp)

        # ...but it must never reach the span.
        assert "ratelimit.limit_requests" not in span.attrs
        assert span.attrs == {"ratelimit.remaining_requests": 59}

    def test_does_not_raise_on_bad_span(self) -> None:
        """A span whose set_attribute raises is swallowed, not propagated."""

        class _ExplodingSpan:
            def set_attribute(self, key: str, value: Any) -> None:
                raise RuntimeError("boom")

        # Must not raise even though set_attribute blows up.
        set_rate_limit_attributes(_ExplodingSpan(), _response_with(headers={"retry-after": "1"}))


# ---------------------------------------------------------------------------
# connector_snapshot
# ---------------------------------------------------------------------------


class _FakeConnector:
    def __init__(
        self,
        *,
        acquired: Any = None,
        conns: dict | None = None,
        waiters: dict | None = None,
        limit: int | None = None,
        limit_per_host: int | None = None,
    ) -> None:
        if acquired is not None:
            self._acquired = acquired
        if conns is not None:
            self._conns = conns
        if waiters is not None:
            self._waiters = waiters
        if limit is not None:
            self.limit = limit
        if limit_per_host is not None:
            self.limit_per_host = limit_per_host


class _FakeSession:
    def __init__(self, connector: Any) -> None:
        self.connector = connector


class TestConnectorSnapshot:
    def test_full_snapshot(self) -> None:
        connector = _FakeConnector(
            acquired={"c1", "c2", "c3"},  # in_use = 3
            conns={"keyA": [1, 2], "keyB": [3]},  # available = 3
            waiters={"keyA": deque([1, 2]), "keyB": deque()},  # queued = 2
            limit=100,
            limit_per_host=10,
        )
        snapshot = connector_snapshot(_FakeSession(connector))

        assert snapshot == {
            "connector.in_use": 3,
            "connector.available": 3,
            "connector.queued": 2,
            "connector.limit": 100,
            "connector.limit_per_host": 10,
        }

    def test_acquired_as_list(self) -> None:
        connector = _FakeConnector(acquired=["c1", "c2"], conns={}, waiters={})
        snapshot = connector_snapshot(_FakeSession(connector))

        assert snapshot["connector.in_use"] == 2
        assert snapshot["connector.available"] == 0
        assert snapshot["connector.queued"] == 0

    def test_none_session_returns_empty(self) -> None:
        assert connector_snapshot(None) == {}

    def test_session_with_none_connector_returns_empty(self) -> None:
        assert connector_snapshot(_FakeSession(None)) == {}

    def test_missing_private_attrs_degrade(self) -> None:
        """A connector missing the private counters reports zeros, not a crash."""
        connector = _FakeConnector(limit=50)  # no _acquired/_conns/_waiters
        snapshot = connector_snapshot(_FakeSession(connector))

        assert snapshot["connector.in_use"] == 0
        assert snapshot["connector.available"] == 0
        assert snapshot["connector.queued"] == 0
        assert snapshot["connector.limit"] == 50
        # limit_per_host was never set -> omitted (value is None).
        assert "connector.limit_per_host" not in snapshot

    def test_none_limits_omitted(self) -> None:
        """limit / limit_per_host of None are dropped from the snapshot."""
        connector = _FakeConnector(acquired=[], conns={}, waiters={})
        snapshot = connector_snapshot(_FakeSession(connector))

        assert "connector.limit" not in snapshot
        assert "connector.limit_per_host" not in snapshot
        # The three counters are always present (zero, not None).
        assert snapshot == {
            "connector.in_use": 0,
            "connector.available": 0,
            "connector.queued": 0,
        }

    def test_zero_limit_per_host_kept(self) -> None:
        """A 0-valued limit_per_host is kept; the omit rule is None-based, not falsy."""
        connector = _FakeConnector(acquired=[], conns={}, waiters={}, limit_per_host=0)
        snapshot = connector_snapshot(_FakeSession(connector))

        assert snapshot["connector.limit_per_host"] == 0

    def test_does_not_raise_on_bad_connector(self) -> None:
        """A connector whose attribute access raises degrades to {}."""

        class _ExplodingConnector:
            @property
            def _acquired(self):  # noqa: ANN202
                raise RuntimeError("boom")

        # Must return {} rather than propagate.
        assert connector_snapshot(_FakeSession(_ExplodingConnector())) == {}


# ---------------------------------------------------------------------------
# set_connector_attributes
# ---------------------------------------------------------------------------


class TestSetConnectorAttributes:
    def test_records_snapshot_on_span(self) -> None:
        span = _FakeSpan()
        connector = _FakeConnector(acquired=["c1"], conns={"k": [1]}, waiters={}, limit=20)

        set_connector_attributes(span, _FakeSession(connector))

        assert span.attrs == {
            "connector.in_use": 1,
            "connector.available": 1,
            "connector.queued": 0,
            "connector.limit": 20,
        }

    def test_none_session_sets_nothing(self) -> None:
        span = _FakeSpan()

        set_connector_attributes(span, None)

        assert span.attrs == {}

    def test_does_not_raise_on_bad_span(self) -> None:
        class _ExplodingSpan:
            def set_attribute(self, key: str, value: Any) -> None:
                raise RuntimeError("boom")

        connector = _FakeConnector(acquired=["c1"], conns={}, waiters={})
        # Must not raise.
        set_connector_attributes(_ExplodingSpan(), _FakeSession(connector))


# ---------------------------------------------------------------------------
# End-to-end: the embedder records both attribute families on its request span.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedder_sets_rate_limit_and_connector_attrs_on_span(monkeypatch) -> None:
    """Drive one embed through and assert the litellm_request span carries the
    rate-limit + connector attributes, using a real recording TracerProvider."""
    from unittest.mock import AsyncMock, MagicMock

    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from khora.extraction.embedders.litellm import LiteLLMEmbedder
    from khora.telemetry import _otel as _otel_module

    # Install a real recording provider and rebind the cached tracer.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _otel_trace.set_tracer_provider(provider)
    _otel_module._TRACER = _otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION)

    # Response carries rate-limit headers via _headers.
    mock_response = MagicMock()
    mock_response.data = [{"embedding": [1.0, 0.0]}]
    mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)
    mock_response._headers = {
        "x-ratelimit-remaining-requests": "58",
        "retry-after": "4",
    }
    mock_response._hidden_params = {}

    # Fake shared session with a connector exposing the contention counters.
    fake_connector = _FakeConnector(acquired=["c1", "c2"], conns={"k": [1]}, waiters={}, limit=64)
    fake_session = _FakeSession(fake_connector)

    embedder = LiteLLMEmbedder(model="test-model", batch_size=100, max_retries=1)

    with (
        monkeypatch.context() as m,
    ):
        import litellm

        m.setattr(litellm, "aembedding", AsyncMock(return_value=mock_response))
        m.setattr(
            "khora.extraction.embedders.litellm.get_shared_session",
            lambda: fake_session,
        )
        m.setattr("khora.telemetry.get_collector", lambda: MagicMock())
        await embedder.embed_batch(["hello"])

    exporter.shutdown()
    spans = exporter.get_finished_spans()
    req_spans = [s for s in spans if s.name == "khora.embedder.litellm_request"]
    assert len(req_spans) == 1
    attrs = dict(req_spans[0].attributes)

    # Rate-limit attributes (recorded AFTER the response).
    assert attrs["ratelimit.remaining_requests"] == 58
    assert attrs["retry_after"] == 4
    # Connector attributes (recorded BEFORE the await).
    assert attrs["connector.in_use"] == 2
    assert attrs["connector.available"] == 1
    assert attrs["connector.queued"] == 0
    assert attrs["connector.limit"] == 64
