"""@trace decorator for automatic OTEL span creation with argument capture.

Eliminates boilerplate for the common pattern of opening a span, passing
function arguments as attributes, running the function, and extracting
result attributes before returning.

When logfire is not installed, the wrapper short-circuits to a direct
function call with zero overhead.

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

from .logfire_integration import _HAS_LOGFIRE, trace_span

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
        elif isinstance(value, (list, tuple)):
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
    """Build the sync or async wrapper with short-circuit when logfire is absent."""
    sig = inspect.signature(fn)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _HAS_LOGFIRE:
                return await fn(*args, **kwargs)
            attrs = _extract_span_attributes(sig, args, kwargs, include=include, exclude=exclude)
            with trace_span(span_name, **attrs) as span:
                ret = await fn(*args, **kwargs)
                if result_extractor is not None:
                    try:
                        span.set_attributes(result_extractor(ret))
                    except Exception:
                        pass
                return ret

        return async_wrapper
    else:

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _HAS_LOGFIRE:
                return fn(*args, **kwargs)
            attrs = _extract_span_attributes(sig, args, kwargs, include=include, exclude=exclude)
            with trace_span(span_name, **attrs) as span:
                ret = fn(*args, **kwargs)
                if result_extractor is not None:
                    try:
                        span.set_attributes(result_extractor(ret))
                    except Exception:
                        pass
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

    When logfire is not installed, the wrapper short-circuits to a direct
    function call with zero overhead — no span created, no argument
    extraction.

    Args:
        fn_or_name: The function (bare ``@trace``) or span name string.
        result: Callable taking the return value and returning a dict of
            extra span attributes.
        exclude: Parameter names to exclude from span attributes.
        include: Parameter names to include (allowlist, mutually exclusive
            with *exclude*).
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
