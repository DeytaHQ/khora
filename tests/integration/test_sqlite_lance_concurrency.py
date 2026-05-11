"""Concurrency integration tests for sqlite_lance entity key gate (DYT-2734).

Verifies ``_SQLiteLanceEntityKeyGate`` serializes overlapping
``(namespace_id, name, entity_type)`` upserts so N concurrent tasks
that all want the same entity converge on a single row — not N rows,
no exceptions, no partial writes.

Adapter-level test (not Khora.remember) per the ticket rules:
remember() would drag Neo4j and LLM calls into scope, which is
irrelevant to the entity-gate contract this test protects.
"""

from __future__ import annotations

import asyncio
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
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


class TestSQLiteLanceConcurrency:
    """Entity key gate under concurrent fan-in."""

    async def test_concurrent_same_key_upserts_collapse_to_one_row(self, tmp_path: Path) -> None:
        """10 concurrent upserts of (Alice, PERSON) yield exactly one row."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())

            async def upsert_alice() -> None:
                alice = Entity(
                    namespace_id=ns.id,
                    name="Alice",
                    entity_type="PERSON",
                    description="concurrent writer",
                )
                await coord.upsert_entities_batch(ns.id, [alice])

            # 10 racing tasks, all targeting the same (ns, name, type) key.
            await asyncio.gather(*(upsert_alice() for _ in range(10)))

            # Gate must have serialized writes — only one Alice row persists.
            alice_count = await coord.graph.count_entities(ns.id)  # type: ignore[union-attr]
            assert alice_count == 1, f"expected 1 Alice row, got {alice_count}"

            # And mention_count must reflect the 10 merged writes.
            entities = await coord.graph.list_entities(  # type: ignore[union-attr]
                ns.id, entity_type="PERSON", limit=10
            )
            assert len(entities) == 1
            assert entities[0].mention_count >= 10
        finally:
            await coord.disconnect()

    async def test_concurrent_mixed_keys_preserves_distinct_rows(self, tmp_path: Path) -> None:
        """Distinct keys racing in parallel keep their own rows (no false sharing)."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())

            names = ["Alice", "Bob", "Carol", "Dan", "Eve"]
            # 5 tasks per distinct name = 25 parallel upserts.
            tasks = []
            for _ in range(5):
                for name in names:
                    ent = Entity(
                        namespace_id=ns.id,
                        name=name,
                        entity_type="PERSON",
                    )
                    tasks.append(coord.upsert_entities_batch(ns.id, [ent]))

            await asyncio.gather(*tasks)

            # Exactly one row per distinct name — gate serializes per-key.
            assert await coord.graph.count_entities(ns.id) == len(names)  # type: ignore[union-attr]

            persisted = await coord.graph.list_entities(  # type: ignore[union-attr]
                ns.id, entity_type="PERSON", limit=100
            )
            assert {e.name for e in persisted} == set(names)
            for e in persisted:
                assert e.mention_count >= 5
        finally:
            await coord.disconnect()

    async def test_concurrent_partial_overlap_no_exceptions(self, tmp_path: Path) -> None:
        """Interleaved batches that share some keys and not others stay consistent."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())

            # Batch A: {shared, unique_a}. Batch B: {shared, unique_b}.
            # Run 5 copies of each batch in parallel.
            async def run_batch(unique_name: str) -> None:
                batch = [
                    Entity(namespace_id=ns.id, name="shared", entity_type="CONCEPT"),
                    Entity(namespace_id=ns.id, name=unique_name, entity_type="CONCEPT"),
                ]
                await coord.upsert_entities_batch(ns.id, batch)

            tasks = []
            for _ in range(5):
                tasks.append(run_batch("unique_a"))
                tasks.append(run_batch("unique_b"))

            # Gate must serialize the "shared" key across all 10 tasks
            # without blocking the unique keys.
            await asyncio.gather(*tasks)

            ents = await coord.graph.list_entities(  # type: ignore[union-attr]
                ns.id, entity_type="CONCEPT", limit=50
            )
            by_name = {e.name: e for e in ents}
            assert set(by_name) == {"shared", "unique_a", "unique_b"}
            # "shared" merged 10 times, each unique 5 times.
            assert by_name["shared"].mention_count >= 10
            assert by_name["unique_a"].mention_count >= 5
            assert by_name["unique_b"].mention_count >= 5
        finally:
            await coord.disconnect()
