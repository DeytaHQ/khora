"""@trace decorator for automatic OTEL span creation with argument capture.

Eliminates boilerplate for the common pattern of opening a span, passing
function arguments as attributes, running the function, and extracting
result attributes before returning.

When no real ``TracerProvider`` is installed, the OTel API's
``NonRecordingSpan`` makes the wrapper effectively free.

Usage::

    from khora.telemetry import trace

    @trace("khora.search_entities", result=lambda r: {"result_count": len(r)})
    async def search_entities(self, namespace_id: UUID, *, limit: int = 10):
        ...

    @trace
    async def get_neighborhood(self, entity_id: UUID, depth: int = 1):
        ...
"""

from __future__ import annotations

import enum
import functools
import inspect
from collections.abc import Callable
from typing import Any, overload
from uuid import UUID

from loguru import logger

from ._otel import trace_span

# Types safe to pass directly as span attributes
_SAFE_TYPES = (str, int, float, bool, type(None))


def _extract_span_attributes(
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    include: frozenset[str] | None,
    exclude: frozenset[str],
) -> dict[str, Any]:
    """Extract safe span attributes from function arguments."""
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    attrs: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name in ("self", "cls"):
            continue
        if include is not None and name not in include:
            continue
        if name in exclude:
            continue

        if isinstance(value, UUID):
            attrs[name] = str(value)
        elif isinstance(value, _SAFE_TYPES):
            attrs[name] = value
        elif isinstance(value, enum.Enum):
            attrs[name] = value.value
        elif isinstance(value, (list, tuple, set, frozenset)):
            attrs[f"{name}_count"] = len(value)
        # else: skip complex objects silently

    return attrs


def _make_wrapper(
    fn: Callable,
    span_name: str,
    *,
    include: frozenset[str] | None,
    exclude: frozenset[str],
    result_extractor: Callable[[Any], dict[str, Any]] | None,
) -> Callable:
    """Build the sync or async wrapper around the trace_span context."""
    sig = inspect.signature(fn)
    param_names = set(sig.parameters.keys()) - {"self", "cls"}

    if include is not None:
        unknown = set(include) - param_names
        if unknown:
            raise ValueError(f"include param(s) {unknown} not found in {fn.__name__}() signature")
    if exclude:
        unknown = set(exclude) - param_names
        if unknown:
            raise ValueError(f"exclude param(s) {unknown} not found in {fn.__name__}() signature")

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            attrs = _extract_span_attributes(sig, args, kwargs, include=include, exclude=exclude)
            with trace_span(span_name, **attrs) as span:
                ret = await fn(*args, **kwargs)
                if result_extractor is not None:
                    try:
                        span.set_attributes(result_extractor(ret))
                    except Exception as e:
                        logger.debug(f"Trace result extractor failed for {span_name}: {e}")
                return ret

        return async_wrapper
    else:

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            attrs = _extract_span_attributes(sig, args, kwargs, include=include, exclude=exclude)
            with trace_span(span_name, **attrs) as span:
                ret = fn(*args, **kwargs)
                if result_extractor is not None:
                    try:
                        span.set_attributes(result_extractor(ret))
                    except Exception as e:
                        logger.debug(f"Trace result extractor failed for {span_name}: {e}")
                return ret

        return sync_wrapper


@overload
def trace(fn: Callable) -> Callable: ...


@overload
def trace(
    name: str,
    *,
    result: Callable[[Any], dict[str, Any]] | None = ...,
    exclude: set[str] | None = ...,
    include: set[str] | None = ...,
) -> Callable[[Callable], Callable]: ...


def trace(
    fn_or_name=None,
    *,
    result: Callable[[Any], dict[str, Any]] | None = None,
    exclude: set[str] | None = None,
    include: set[str] | None = None,
):
    """Decorator that creates an OTEL span around a function call.

    Captures function arguments as span attributes (auto-skipping self/cls,
    converting UUIDs to strings, skipping complex objects).  Optionally
    extracts attributes from the return value.

    When no real ``TracerProvider`` is installed, the underlying span
    is OTel's ``NonRecordingSpan`` and attribute writes are dropped.

    Argument capture rules:
        - ``self`` / ``cls`` parameters are always skipped.
        - ``UUID`` values are converted to ``str``.
        - ``Enum`` values are converted to their ``.value``.
        - ``list``, ``tuple``, ``set``, and ``frozenset`` are captured as
          ``{name}_count`` (the length, not the contents).
        - ``str``, ``int``, ``float``, ``bool``, and ``None`` pass through
          unchanged.
        - All other types (dict, custom objects, etc.) are silently skipped.
        - Default parameter values are captured when the caller omits them.
        - The ``result`` extractor receives the return value and should return
          a ``dict[str, Any]`` of extra span attributes.  If it raises, the
          error is silently suppressed and the return value is still returned.

    Decorator order:
        Place ``@trace`` as the **outermost** (topmost) decorator so that
        the span wraps the fully-decorated function.  Inner decorators
        (e.g. ``@retry``, ``@cache``) will execute inside the span::

            @trace("khora.my_op")      # outermost — creates the span first
            @retry(max_attempts=3)
            async def my_op(...):
                ...

    Args:
        fn_or_name: The function (bare ``@trace``) or span name string.
        result: Callable taking the return value and returning a dict of
            extra span attributes.
        exclude: Parameter names to exclude from span attributes.
            Validated at decoration time — unknown names raise ``ValueError``.
        include: Parameter names to include (allowlist, mutually exclusive
            with *exclude*).  Validated at decoration time — unknown names
            raise ``ValueError``.
    """
    if include is not None and exclude is not None:
        raise ValueError("Cannot specify both 'include' and 'exclude'")

    _exclude = frozenset(exclude) if exclude else frozenset()
    _include = frozenset(include) if include else None

    if callable(fn_or_name):
        # @trace (bare decorator, no arguments)
        fn = fn_or_name
        span_name = f"khora.{fn.__name__}"
        return _make_wrapper(fn, span_name, include=_include, exclude=_exclude, result_extractor=result)

    # @trace("name", ...) or @trace(exclude=...) or @trace()
    name = fn_or_name

    def decorator(fn: Callable) -> Callable:
        span = name or f"khora.{fn.__name__}"
        return _make_wrapper(fn, span, include=_include, exclude=_exclude, result_extractor=result)

    return decorator
