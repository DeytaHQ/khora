"""Entity key gate for the SQLite + LanceDB graph adapter.

Ported verbatim from ``surrealdb.graph._SurrealDBEntityKeyGate`` — the
mechanism is engine-agnostic: an :class:`asyncio.Condition` protecting
an ``in_flight`` set of entity keys that serializes concurrent
``upsert_entities_batch`` calls whose keys overlap.  Prevents the
prefetch-compare-update race while permitting up to ``max_concurrent``
disjoint batches to run in parallel.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from khora.core.models import Entity


class _SQLiteLanceEntityKeyGate:
    """Serialize access to entities by ``(namespace_id, name, entity_type)`` key.

    Mirrors the Neo4j and SurrealDB gates: two concurrent batches that share
    any entity key wait on each other so that the prefetch (SELECT) and the
    write (INSERT/UPDATE) remain atomic per key.
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self._condition = asyncio.Condition()
        self._in_flight: set[tuple[str, str, str]] = set()
        self._active = 0
        self._max_concurrent = max_concurrent

    @asynccontextmanager
    async def acquire(self, entities: list[Entity]) -> AsyncIterator[None]:
        """Acquire exclusive access for the given batch of entity keys."""
        keys = {(str(e.namespace_id), e.name, str(e.entity_type)) for e in entities}

        async with self._condition:
            while (keys & self._in_flight) or self._active >= self._max_concurrent:
                await self._condition.wait()
            self._in_flight |= keys
            self._active += 1

        try:
            yield
        finally:
            async with self._condition:
                self._in_flight -= keys
                self._active -= 1
                self._condition.notify_all()
