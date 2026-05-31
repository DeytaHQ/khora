"""Shared namespace-stats counting helper (ADR-001 failure observability).

All three engines (vectorcypher, chronicle, skeleton) gather the same four
counters for ``stats()``. Before #878 each engine coerced a raising counter
to ``0`` and dropped the exception on the floor, so a routing crash looked
identical to an empty namespace. This helper does the gather once and, on any
failure, logs at WARNING, appends an ``ErrorRecord`` to a degradations dict,
and bumps a metric - without raising (stats() is a health-check path).

See ``docs/architecture/failure-observability-contract.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.telemetry.metrics import metric_counter

if TYPE_CHECKING:
    from khora.core.diagnostics import ErrorRecord
    from khora.storage.coordinator import StorageCoordinator

# Counter for stats() counters that could not run. Labels: engine, counter.
# NO namespace_id label - cardinality rule.
_STATS_COUNTER_FAILED = metric_counter(
    "khora.stats.counter_failed_total",
    description=(
        "stats() counters that could not run (backend lacks the method or it "
        "raised). Labels: engine, counter. The int field stays 0 and an "
        "ErrorRecord is appended to Stats.metadata['errors']. NO namespace_id "
        "label (cardinality rule). See docs/architecture/failure-observability-contract.md."
    ),
)


async def gather_counts(
    storage: StorageCoordinator,
    namespace_id: UUID,
    *,
    engine: str,
) -> tuple[int, int, int, dict[str, Any]]:
    """Count chunks/entities/relationships, surfacing failures per ADR-001.

    Returns ``(chunk_count, entity_count, relationship_count, metadata)``.
    Each count is 0 when its counter raised; the ``metadata`` dict carries an
    ``errors`` list of ``ErrorRecord`` when any counter failed (empty dict
    when all succeeded).
    """
    errors: list[ErrorRecord] = []
    metadata: dict[str, Any] = {}

    async def _run(counter: str, coro: Any) -> int:
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001 - stats() must not raise
            logger.warning(
                "stats: {}.count_{} failed, reporting 0",
                engine,
                counter,
                exc_info=True,
            )
            errors.append(
                {
                    "component": f"{engine}.stats.count_{counter}",
                    "reason": "counter_unavailable",
                    "exception": type(exc).__name__,
                    "detail": str(exc) or None,
                }
            )
            _STATS_COUNTER_FAILED.add(1, attributes={"engine": engine, "counter": counter})
            return 0

    chunk_count = await _run("chunks", storage.count_chunks(namespace_id))
    entity_count = await _run("entities", storage.count_entities(namespace_id))
    relationship_count = await _run("relationships", storage.count_relationships(namespace_id))

    if errors:
        metadata["errors"] = errors

    return chunk_count, entity_count, relationship_count, metadata


__all__ = ["gather_counts"]
