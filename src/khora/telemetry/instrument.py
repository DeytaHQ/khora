"""Decorators and context managers for telemetry instrumentation."""

from __future__ import annotations

import functools
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID


def instrument_llm(operation: str):
    """Decorator for async functions that make LLM calls.

    The wrapped function must return a LiteLLM response object with a
    ``usage`` attribute (prompt_tokens, completion_tokens, total_tokens).

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
            try:
                result = await fn(*args, **kwargs)
                return result
            except Exception as exc:
                status = "error"
                error_msg = str(exc)[:500]
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                collector.record_llm_call(
                    operation=operation,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_msg,
                )

        return wrapper

    return decorator


def instrument_storage(backend: str, operation: str):
    """Decorator for async storage methods.

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
            try:
                result = await fn(*args, **kwargs)
                # Try to guess record count from result
                if isinstance(result, list):
                    record_count = len(result)
                elif result is not None:
                    record_count = 1
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
    extra_metadata: dict[str, Any] | None = None,
):
    """Async context manager that records a pipeline stage to telemetry.

    Usage::

        async with pipeline_stage("ingestion", "chunking", run_id):
            chunks = await chunk_document(doc)

    Args:
        pipeline_name: Pipeline identifier (e.g. "ingestion", "query").
        stage: Stage name (e.g. "chunking", "embedding").
        run_id: UUID grouping stages in one pipeline execution.
        namespace_id: Optional namespace ID for the event.
        extra_metadata: Optional extra metadata dict.
    """
    from . import get_collector

    collector = get_collector()
    start = time.perf_counter()
    status = "success"
    error_msg: str | None = None
    try:
        yield
    except Exception as exc:
        status = "error"
        error_msg = str(exc)[:500]
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        collector.record_pipeline_stage(
            pipeline=pipeline_name,
            stage=stage,
            run_id=run_id,
            latency_ms=latency_ms,
            status=status,
            error_message=error_msg,
            namespace_id=namespace_id,
            metadata=extra_metadata or {},
        )
