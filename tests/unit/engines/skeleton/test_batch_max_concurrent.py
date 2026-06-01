"""Skeleton ``remember_batch`` honors ``max_concurrent`` (#935).

Before the fix, the chunking stage fanned out one ``asyncio.to_thread`` per
document under a single unbounded ``asyncio.gather`` and never read the
documented ``max_concurrent`` kwarg. These tests patch the chunker so each
chunk call records the number of concurrently in-flight chunk operations, then
assert the observed peak never exceeds ``max_concurrent``.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import khora.extraction.chunkers as chunkers_mod
from khora.engines.skeleton.engine import SkeletonConstructionEngine


def _mock_config(*, backend: str = "pgvector") -> MagicMock:
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.storage.embedding_dimension = 1536
    config.storage.backend = backend
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _connected(backend: str = "pgvector") -> SkeletonConstructionEngine:
    eng = SkeletonConstructionEngine(_mock_config(backend=backend), backend=backend)
    eng._connected = True
    eng._storage = AsyncMock()
    eng._embedder = AsyncMock()
    eng._temporal_store = AsyncMock()

    async def _create_document(doc):
        if doc.id is None:
            doc.id = uuid4()
        return doc

    eng._storage.create_document.side_effect = _create_document
    eng._storage.get_documents_by_checksums.return_value = {}
    eng._storage.update_document.return_value = None
    eng._embedder.embed_batch.return_value = []
    eng._temporal_store.create_chunks_batch.return_value = []
    return eng


class _ConcurrencyTrackingChunker:
    """Chunker whose ``chunk`` blocks briefly while recording peak in-flight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def chunk(self, content: str):  # noqa: ANN001 - mirrors real chunker signature
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(0.02)
        finally:
            with self._lock:
                self.in_flight -= 1
        return []  # no chunks -> skip embed/store stages, keeps the test pure


@pytest.mark.asyncio
@pytest.mark.parametrize("max_concurrent", [1, 2, 5])
async def test_remember_batch_respects_max_concurrent(monkeypatch, max_concurrent):
    eng = _connected()
    tracker = _ConcurrencyTrackingChunker()
    monkeypatch.setattr(chunkers_mod, "create_chunker", lambda **kwargs: tracker)

    documents = [{"content": f"doc {i}"} for i in range(20)]

    await eng.remember_batch(
        documents,
        namespace_id=uuid4(),
        max_concurrent=max_concurrent,
        entity_types=[],
        relationship_types=[],
    )

    assert tracker.max_in_flight <= max_concurrent
    # sanity: with 20 docs and a small bound we should actually saturate it
    assert tracker.max_in_flight == min(max_concurrent, 20)
