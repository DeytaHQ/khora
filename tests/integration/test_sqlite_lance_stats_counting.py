"""Headline repro for #878: stats counts are silently zero on sqlite_lance.

Before the fix, ``StorageCoordinator.count_entities`` preferred the vector
backend, but the sqlite_lance vector adapter has no ``count_entities`` method,
so the call raised ``AttributeError``. Each engine's ``stats()`` coerced that
to ``0``, making a populated namespace look empty. The fix routes
``count_entities`` to the backend that OWNS entities (the graph backend), so
the count is correct on every topology.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Entity, MemoryNamespace
from tests.integration._sqlite_lance_fixtures import build_sqlite_lance_coordinator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed",
    ),
]


async def test_count_entities_nonzero_on_sqlite_lance(tmp_path: Path) -> None:
    """count_entities returns the real count, not 0/AttributeError (#878)."""
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())

        # The vector adapter has no count_entities on this topology.
        assert not hasattr(coord._vector, "count_entities")

        await coord.upsert_entities_batch(
            ns.id,
            [
                Entity(namespace_id=ns.id, name="Ada Lovelace", entity_type="PERSON"),
                Entity(namespace_id=ns.id, name="Analytical Engine", entity_type="CONCEPT"),
            ],
        )

        count = await coord.count_entities(ns.id)
        assert count == 2, "count_entities must route to the graph backend that owns entities"
    finally:
        with contextlib.suppress(Exception):
            await coord.disconnect()
