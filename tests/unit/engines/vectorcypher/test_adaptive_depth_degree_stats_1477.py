"""#1477 - retriever wires the epoch-cached degree histogram into depth.

Covers ``VectorCypherRetriever._get_degree_stats``: lazy build on a miss,
reuse under the same write-epoch, rebuild after an epoch bump, and graceful
None (fall back to the count rule) when the cache / epoch-reader / storage
are absent.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.retriever import VectorCypherRetriever
from khora.query.degree_stats import DegreeStatsCache

pytestmark = pytest.mark.unit


def _entity(eid):
    return SimpleNamespace(id=eid)


def _rel(src, tgt):
    return SimpleNamespace(source_entity_id=src, target_entity_id=tgt)


def _make_storage(entities, relationships):
    storage = AsyncMock()
    storage.list_entities = AsyncMock(return_value=entities)
    storage.list_relationships = AsyncMock(return_value=relationships)
    return storage


def _make_retriever(storage, cache, epoch_holder):
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=AsyncMock(),
        embedder=AsyncMock(),
        storage=storage,
        degree_stats_cache=cache,
        epoch_reader=lambda _ns: epoch_holder["epoch"],
    )


@pytest.mark.asyncio
async def test_builds_and_caches_on_miss() -> None:
    ns = uuid4()
    hub, leaf = uuid4(), uuid4()
    storage = _make_storage([_entity(hub), _entity(leaf)], [_rel(hub, leaf)])
    cache = DegreeStatsCache()
    epoch = {"epoch": 1}
    retriever = _make_retriever(storage, cache, epoch)

    stats = await retriever._get_degree_stats(ns)

    assert stats is not None
    assert stats.degree_by_entity[hub] == 1
    assert stats.degree_by_entity[leaf] == 1
    # Cached under the current epoch.
    assert cache.get(ns, 1) is stats
    storage.list_relationships.assert_awaited_once()


@pytest.mark.asyncio
async def test_reuses_cache_under_same_epoch() -> None:
    ns = uuid4()
    a, b = uuid4(), uuid4()
    storage = _make_storage([_entity(a), _entity(b)], [_rel(a, b)])
    cache = DegreeStatsCache()
    epoch = {"epoch": 1}
    retriever = _make_retriever(storage, cache, epoch)

    first = await retriever._get_degree_stats(ns)
    second = await retriever._get_degree_stats(ns)

    assert first is second
    # Only one storage round-trip - the second call hit the cache.
    storage.list_relationships.assert_awaited_once()


@pytest.mark.asyncio
async def test_rebuilds_after_epoch_bump() -> None:
    ns = uuid4()
    a, b = uuid4(), uuid4()
    storage = _make_storage([_entity(a), _entity(b)], [_rel(a, b)])
    cache = DegreeStatsCache()
    epoch = {"epoch": 1}
    retriever = _make_retriever(storage, cache, epoch)

    await retriever._get_degree_stats(ns)
    epoch["epoch"] = 2  # a write bumped the namespace write-epoch
    await retriever._get_degree_stats(ns)

    # Two builds: the epoch bump invalidated the first cached entry.
    assert storage.list_relationships.await_count == 2


@pytest.mark.asyncio
async def test_returns_none_without_cache_or_reader() -> None:
    ns = uuid4()
    storage = _make_storage([_entity(uuid4())], [])
    # No cache / epoch_reader supplied -> the depth rule falls back to counts.
    retriever = VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=AsyncMock(),
        embedder=AsyncMock(),
        storage=storage,
    )
    assert await retriever._get_degree_stats(ns) is None
    storage.list_relationships.assert_not_awaited()


@pytest.mark.asyncio
async def test_storage_failure_degrades_to_none() -> None:
    ns = uuid4()
    storage = AsyncMock()
    storage.list_entities = AsyncMock(side_effect=RuntimeError("boom"))
    cache = DegreeStatsCache()
    retriever = _make_retriever(storage, cache, {"epoch": 1})
    # A fetch failure must not blow up recall - it returns None (count fallback).
    assert await retriever._get_degree_stats(ns) is None


@pytest.mark.asyncio
async def test_concurrent_misses_single_flight_one_build() -> None:
    # A burst of concurrent recalls after a write-epoch bump must collapse onto
    # a single histogram build (single-flight), not one full scan per recall.
    ns = uuid4()
    a, b = uuid4(), uuid4()
    storage = _make_storage([_entity(a), _entity(b)], [_rel(a, b)])
    cache = DegreeStatsCache()
    retriever = _make_retriever(storage, cache, {"epoch": 1})

    results = await asyncio.gather(*[retriever._get_degree_stats(ns) for _ in range(10)])

    assert all(r is results[0] for r in results)
    # Exactly one scan despite 10 concurrent misses.
    assert storage.list_relationships.await_count == 1
    assert storage.list_entities.await_count == 1
