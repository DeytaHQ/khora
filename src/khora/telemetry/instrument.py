"""Decorators and context managers for telemetry instrumentation."""

from __future__ import annotations

import functools
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from .logfire_integration import logfire_span


def instrument_llm(operation: str):
    """Decorator for async functions that make LLM calls.

    Extracts model, prompt_tokens, completion_tokens, and total_tokens
    from LiteLLM ``ModelResponse.usage``.  Automatically populates
    trace_id and parent_event_id from context vars.

    Args:
        operation: Logical operation name (e.g. "entity_extraction", "embedding").
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from . import get_collector

            collector = get_collector()
            start = time.perf_counter()
            status = "success"
            error_msg: str | None = None
            result = None
            try:
                with logfire_span(f"khora.llm.{operation}") as span:
                    result = await fn(*args, **kwargs)
                    # Set span attributes after call completes
                    latency_ms = (time.perf_counter() - start) * 1000
                    model = ""
                    total_tokens = 0
                    if result is not None:
                        usage = getattr(result, "usage", None)
                        if usage is not None:
                            total_tokens = getattr(usage, "total_tokens", 0) or 0
                        model = getattr(result, "model", "") or ""
                    if span is not None:
                        span.set_attribute("model", model)
                        span.set_attribute("total_tokens", total_tokens)
                        span.set_attribute("latency_ms", latency_ms)
                return result
            except Exception as exc:
                status = "error"
                error_msg = str(exc)[:500]
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                # Extract model and token counts from LiteLLM response
                model = ""
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if result is not None:
                    usage = getattr(result, "usage", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                        total_tokens = getattr(usage, "total_tokens", 0) or 0
                    model = getattr(result, "model", "") or ""
                collector.record_llm_call(
                    operation=operation,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_msg,
                )

        return wrapper

    return decorator


def instrument_storage(backend: str, operation: str):
    """Decorator for async storage methods.

    Extracts namespace_id from kwargs or first UUID positional arg.
    Automatically populates trace_id and parent_event_id from context vars.

    Args:
        backend: Storage backend name (e.g. "postgresql", "pgvector", "neo4j").
        operation: Operation name (e.g. "create_document").
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from . import get_collector

            collector = get_collector()
            start = time.perf_counter()
            status = "success"
            error_msg: str | None = None
            record_count = 0
            namespace_id: UUID | None = kwargs.get("namespace_id")

            # Try to extract namespace_id from positional args if not in kwargs
            if namespace_id is None:
                for arg in args:
                    if isinstance(arg, UUID):
                        namespace_id = arg
                        break

            try:
                with logfire_span(f"khora.storage.{operation}", backend=backend) as span:
                    result = await fn(*args, **kwargs)
                    # Try to guess record count from result
                    if isinstance(result, list):
                        record_count = len(result)
                    elif result is not None:
                        record_count = 1
                    latency_ms = (time.perf_counter() - start) * 1000
                    if span is not None:
                        span.set_attribute("status", "success")
                        span.set_attribute("latency_ms", latency_ms)
                        span.set_attribute("record_count", record_count)
                return result
            except Exception as exc:
                status = "error"
                error_msg = str(exc)[:500]
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                collector.record_storage_op(
                    backend=backend,
                    operation=operation,
                    latency_ms=latency_ms,
                    record_count=record_count,
                    status=status,
                    error_message=error_msg,
                    namespace_id=namespace_id,
                )

        return wrapper

    return decorator


@asynccontextmanager
async def pipeline_stage(
    pipeline_name: str,
    stage: str,
    run_id: UUID | None = None,
    *,
    namespace_id: UUID | None = None,
    input_count: int = 0,
    extra_metadata: dict[str, Any] | None = None,
):
    """Async context manager that records a pipeline stage to telemetry.

    Sets parent_event_id in context so that child LLM/storage calls
    auto-link to this pipeline stage.

    Usage::

        async with pipeline_stage("ingestion", "chunking", run_id, input_count=1) as ctx:
            chunks = await chunk_document(doc)
            ctx["output_count"] = len(chunks)

    Args:
        pipeline_name: Pipeline identifier (e.g. "ingestion", "query").
        stage: Stage name (e.g. "chunking", "embedding").
        run_id: UUID grouping stages in one pipeline execution.
        namespace_id: Optional namespace ID for the event.
        input_count: Number of items going into this stage.
        extra_metadata: Optional extra metadata dict.
    """
    from . import get_collector
    from .context import get_parent_event_id, set_parent_event_id

    collector = get_collector()
    # Save previous parent so we can restore it
    prev_parent = get_parent_event_id()
    # Use a simple incrementing ID for parent linking (based on buffer position)
    # Since we don't have the DB-assigned id yet, use the run_id hash as a stable reference
    stage_id = hash((pipeline_name, stage, run_id)) & 0x7FFFFFFFFFFFFFFF  # positive int64
    set_parent_event_id(stage_id)

    ctx: dict[str, Any] = {"output_count": 0}
    start = time.perf_counter()
    status = "success"
    error_msg: str | None = None
    try:
        with logfire_span(f"khora.{pipeline_name}.{stage}", input_count=input_count) as span:
            yield ctx
            latency_ms = (time.perf_counter() - start) * 1000
            if span is not None:
                span.set_attribute("output_count", ctx.get("output_count", 0))
                span.set_attribute("status", "success")
                span.set_attribute("latency_ms", latency_ms)
    except Exception as exc:
        status = "error"
        error_msg = str(exc)[:500]
        raise
    finally:
        set_parent_event_id(prev_parent)
        latency_ms = (time.perf_counter() - start) * 1000
        collector.record_pipeline_stage(
            pipeline=pipeline_name,
            stage=stage,
            run_id=run_id,
            latency_ms=latency_ms,
            input_count=input_count,
            output_count=ctx.get("output_count", 0),
            status=status,
            error_message=error_msg,
            namespace_id=namespace_id,
            metadata=extra_metadata or {},
        )
