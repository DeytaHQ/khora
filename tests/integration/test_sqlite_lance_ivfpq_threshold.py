"""Real-data IVF-PQ retrain at the 5,000-row threshold (sqlite_lance).

``tests/unit/storage/backends/sqlite_lance/test_index_retrain.py`` is
fully mocked: it patches ``adapter._chunks_table`` so ``create_index``
never runs against a real LanceDB. That covers the threshold logic but
not the real-LanceDB interaction. The first user to cross 5,000 rows
in production is the canary — IVF-PQ training with insufficient
partition data or a dim mismatch surfaces only against actual data.

This test ingests 5,100 synthetic embeddings via the vector adapter
and asserts a search returns results post-threshold. Marked
``integration`` so it doesn't slow the unit suite; runtime ~10s.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Chunk, ChunkMetadata, Document, DocumentMetadata, MemoryNamespace
from khora.storage.backends.sqlite_lance.vector import _ANN_INDEX_THRESHOLD
from tests.integration._sqlite_lance_fixtures import build_sqlite_lance_coordinator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed",
    ),
]

EMBED_DIM = 32


def _synthetic_embedding(i: int) -> list[float]:
    """Distinct, L2-normalised 32-d vector for index entry ``i``.

    Mixes sine + cosine of the index across dims so the resulting cloud
    has enough variance for IVF-PQ partition training to converge.
    """
    raw = [math.sin(i * (k + 1) * 0.013) + math.cos(i * (k + 1) * 0.017) for k in range(EMBED_DIM)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


async def test_ivfpq_index_trains_at_5000_threshold(tmp_path: Path) -> None:
    """5,100 chunks → first search after the threshold trains the index.

    Assertions:
    1. The search call doesn't raise.
    2. The search returns at least one result.
    3. ``_chunks_at_last_index`` is populated post-search (the inline
       train-on-first-need path fired).
    """
    coord = await build_sqlite_lance_coordinator(tmp_path, embed_dim=EMBED_DIM)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        doc = Document(
            namespace_id=ns.id,
            content="ivfpq-threshold",
            external_id="ivfpq-test",
            metadata=DocumentMetadata(source="test", title="ivfpq"),
        )
        await coord.create_document(doc)

        # Insert just over the threshold. Batches of 500 keep the executemany
        # round-trip cost reasonable; total runtime ~5-10s.
        total = _ANN_INDEX_THRESHOLD + 100
        for batch_start in range(0, total, 500):
            batch = [
                Chunk(
                    namespace_id=ns.id,
                    document_id=doc.id,
                    content=f"chunk-{i}",
                    metadata=ChunkMetadata(document_id=doc.id, chunk_index=i),
                    embedding=_synthetic_embedding(i),
                    embedding_model="synthetic",
                )
                for i in range(batch_start, min(batch_start + 500, total))
            ]
            await coord.create_chunks_batch(batch)

        # Pre-search: index has never been trained (no search has run yet).
        assert coord.vector._chunks_at_last_index is None  # type: ignore[union-attr]

        # First search after crossing the threshold trains IVF-PQ inline.
        query = _synthetic_embedding(42)
        results = await coord.search_similar_chunks(ns.id, query, limit=10)
        assert results, "search returned no hits across 5,100 indexed chunks"

        # Index must have been built — the inline train-on-first-need path
        # records the row count at training time.
        trained_at = coord.vector._chunks_at_last_index  # type: ignore[union-attr]
        assert trained_at is not None, (
            f"crossed the {_ANN_INDEX_THRESHOLD}-row IVF-PQ threshold ({total} chunks "
            f"inserted) but the index didn't train on first search"
        )
        assert trained_at >= _ANN_INDEX_THRESHOLD, (
            f"index trained on {trained_at} rows, expected >= {_ANN_INDEX_THRESHOLD}"
        )
    finally:
        await coord.disconnect()
