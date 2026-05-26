"""Best-effort span attributes for LLM/embedding request observability.

Two diagnostic dimensions are captured around each upstream request, both
recorded as plain span attributes (OTel scalar types only):

* **Rate-limit headers** read off the response *after* it returns — these
  are immutable once the response lands and tell us how much provider
  budget remains.
* **Connector contention** read off the shared aiohttp session *before* the
  request awaits — this reflects how many connections are in use / queued at
  the moment this request asks for one, i.e. whether connection acquisition
  is likely to block.

Every read here is defensive. None of these functions may raise into the
request path — they mirror the "best-effort, unlocked reads" resilience
pattern used for the Neo4j pool gauges: any unexpected error is swallowed and
logged at DEBUG. A header-shape change or a connector internals change must
degrade gracefully, never crash an ingestion.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Provider headers sometimes arrive prefixed by litellm; strip it before matching.
_PROVIDER_PREFIX = "llm_provider-"

# Header name -> dict key, with how to coerce the value.
# "int" tries int() and falls back to the raw string; "str" keeps it verbatim.
_HEADER_SPECS: list[tuple[str, str, str]] = [
    ("x-ratelimit-remaining-requests", "ratelimit.remaining_requests", "int"),
    ("x-ratelimit-remaining-tokens", "ratelimit.remaining_tokens", "int"),
    ("retry-after", "retry_after", "int"),
    ("x-ratelimit-reset-requests", "ratelimit.reset_requests", "str"),
    ("x-ratelimit-reset-tokens", "ratelimit.reset_tokens", "str"),
    # Surfaced for the DEBUG `{remaining}/{limit}` log line only — NOT a span
    # attribute (excluded by _RATE_LIMIT_SPAN_KEYS below).
    ("x-ratelimit-limit-requests", "ratelimit.limit_requests", "int"),
]

# The subset of parse_rate_limit_headers keys that are approved span attributes.
# limit_requests is intentionally absent: it feeds the DEBUG log, not a span.
_RATE_LIMIT_SPAN_KEYS: frozenset[str] = frozenset(
    {
        "ratelimit.remaining_requests",
        "ratelimit.remaining_tokens",
        "retry_after",
        "ratelimit.reset_requests",
        "ratelimit.reset_tokens",
    }
)


def _coerce_int(value: Any) -> int | str:
    """Return ``int(value)`` if it parses cleanly, else the raw string."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return str(value)


def _collect_headers(response: Any) -> dict[str, Any]:
    """Merge candidate header sources off *response*, tolerating missing/None.

    Reads from both ``_hidden_params["additional_headers"]`` and ``_headers``
    (litellm populates either depending on provider/path) and strips a leading
    ``llm_provider-`` prefix off each key so provider-prefixed headers match
    the canonical names.
    """
    merged: dict[str, Any] = {}
    hidden = getattr(response, "_hidden_params", None) or {}
    additional = hidden.get("additional_headers") if isinstance(hidden, dict) else None
    for source in (additional, getattr(response, "_headers", None)):
        if not source:
            continue
        try:
            items = source.items()
        except AttributeError:
            continue
        for key, value in items:
            if not isinstance(key, str):
                continue
            normalized = key[len(_PROVIDER_PREFIX) :] if key.startswith(_PROVIDER_PREFIX) else key
            merged[normalized.lower()] = value
    return merged


def parse_rate_limit_headers(response: Any) -> dict[str, int | str]:
    """Extract rate-limit headers from *response* into a dict.

    Returns a ``{key: value}`` dict for the headers that are present (empty
    dict if none). Most keys are span-ready attributes; ``ratelimit.limit_requests``
    is included too but feeds only the DEBUG ``{remaining}/{limit}`` log line —
    callers writing span attributes filter it out via :func:`set_rate_limit_attributes`.
    Never raises — on any unexpected error it logs at DEBUG and returns whatever
    was collected so far.
    """
    attrs: dict[str, int | str] = {}
    try:
        headers = _collect_headers(response)
        for header_name, attr_key, kind in _HEADER_SPECS:
            if header_name not in headers:
                continue
            raw = headers[header_name]
            attrs[attr_key] = _coerce_int(raw) if kind == "int" else str(raw)
    except Exception as exc:  # pragma: no cover - defensive, never break the request
        logger.debug("parse_rate_limit_headers failed: {!s}", exc)
    return attrs


def set_rate_limit_attributes(span: Any, response: Any) -> None:
    """Set rate-limit header attributes on *span*. Call AFTER the response returns.

    Writes only the approved span-attribute keys (``_RATE_LIMIT_SPAN_KEYS``);
    ``ratelimit.limit_requests`` is parsed for the DEBUG log but never set as a
    span attribute.
    """
    try:
        for key, value in parse_rate_limit_headers(response).items():
            if key in _RATE_LIMIT_SPAN_KEYS:
                span.set_attribute(key, value)
    except Exception as exc:  # pragma: no cover - defensive, never break the request
        logger.debug("set_rate_limit_attributes failed: {!s}", exc)


def connector_snapshot(session: Any) -> dict[str, int]:
    """Read aiohttp connector contention counters off *session*.

    SYNCHRONOUS with NO ``await`` anywhere inside. That is what makes the read
    atomic under the single asyncio event loop: a non-awaiting ``len()`` /
    ``sum()`` read cannot be preempted by another coroutine mutating the
    connector, so no lock is needed (same rationale as the Neo4j pool gauges'
    unlocked reads). Capture this IMMEDIATELY BEFORE the request awaits so it
    reflects in-use/queued contention at the moment this request asks for a
    connection — read after the response returns it would just show idle
    post-release state.

    Returns a ``{attr_key: int}`` dict; keys whose underlying value is None are
    omitted. Returns ``{}`` if there is no session or connector. Never raises.
    """
    try:
        if session is None:
            return {}
        connector = getattr(session, "connector", None)
        if connector is None:
            return {}

        # Materialize the dict-of-lists values before summing so the read is a
        # true point-in-time snapshot (and stays correct if an await is ever
        # introduced near this call).
        snapshot: dict[str, int | None] = {
            "connector.in_use": len(getattr(connector, "_acquired", ())),
            "connector.available": sum(len(conns) for conns in list(getattr(connector, "_conns", {}).values())),
            "connector.queued": sum(len(waiters) for waiters in list(getattr(connector, "_waiters", {}).values())),
            "connector.limit": getattr(connector, "limit", None),
            "connector.limit_per_host": getattr(connector, "limit_per_host", None),
        }
        return {key: value for key, value in snapshot.items() if value is not None}
    except Exception as exc:  # pragma: no cover - defensive, never break the request
        logger.debug("connector_snapshot failed: {!s}", exc)
        return {}


def set_connector_attributes(span: Any, session: Any) -> None:
    """Set connector contention attributes on *span*. Call BEFORE the request awaits."""
    try:
        for key, value in connector_snapshot(session).items():
            span.set_attribute(key, value)
    except Exception as exc:  # pragma: no cover - defensive, never break the request
        logger.debug("set_connector_attributes failed: {!s}", exc)
