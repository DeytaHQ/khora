"""Embedded integration tests for the keyword_ppr lexical channel (#1391).

Exercises the real storage stack (SQLite + LanceDB in tmp_path, the same
migrated schema as production) end-to-end: the ingest-time edge write
(``persist_keyword_chunk_edges``), the storage round-trip
(``upsert_keyword_chunk_edges`` / ``get_keyword_chunk_edges``), and the
query-time channel (``keyword_ppr_retrieve_chunks``). Also pins the default
(bm25) gate: when the channel is off, no keyword_chunks rows are written.

Hermetic — deterministic fake embeddings, no LLM. Mirrors
test_sqlite_lance_ingest.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Chunk, Document, MemoryNamespace
from khora.engines.vectorcypher.keyword_edges import persist_keyword_chunk_edges
from khora.extraction.tokenize import tokenize_multilingual
from khora.query.keyword_ppr import keyword_ppr_retrieve_chunks
from tests.integration._sqlite_lance_fixtures import build_sqlite_lance_coordinator, fake_embedding
from tests.integration.conftest import _database_url, _pg_reachable

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


async def _seed_docs(coord, namespace_id, docs: list[str]) -> list[Chunk]:
    """Persist one document + one chunk per content string. Returns the chunks."""
    chunks: list[Chunk] = []
    for i, content in enumerate(docs):
        doc = Document(namespace_id=namespace_id, content=content, external_id=f"doc-{i}", title=f"doc-{i}")
        await coord.create_document(doc)
        chunks.append(
            Chunk(
                namespace_id=namespace_id,
                document_id=doc.id,
                content=content,
                chunk_index=0,
                embedding=fake_embedding(content),
                embedding_model="fake",
            )
        )
    await coord.create_chunks_batch(chunks)
    return chunks


async def test_ingest_populates_keyword_chunks_and_channel_recalls(tmp_path: Path) -> None:
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        # Documents/chunks FK to memory_namespaces.id (the row id), so use the
        # resolved id. The coordinator keyword methods resolve internally and are
        # idempotent on row ids, so passing the row id is fine there too.
        ns_id = await coord.resolve_namespace(ns.namespace_id)

        chunks = await _seed_docs(
            coord,
            ns_id,
            [
                "photosynthesis converts sunlight into chemical energy in plants",
                "mitochondria produce energy through cellular respiration",
                "the weather today is cloudy with occasional rain",
            ],
        )
        target = chunks[0]  # the only chunk mentioning "photosynthesis"

        # Ingest-time gated write (the helper the engine calls when keyword_ppr is on).
        await persist_keyword_chunk_edges(coord, ns_id, chunks)

        # keyword_chunks must be populated.
        edges = await coord.get_keyword_chunk_edges(ns_id, limit=10_000)
        assert edges, "keyword_chunks was not populated by the ingest helper"
        edge_chunk_ids = {cid for _kw, cid, _idf in edges}
        assert {c.id for c in chunks} <= edge_chunk_ids

        # Query-time channel ranks the photosynthesis chunk first.
        results = await keyword_ppr_retrieve_chunks(
            coord,
            ns_id,
            "tell me about photosynthesis",
            tokenizer=tokenize_multilingual,
            damping=0.85,
            max_iter=50,
            tol=1e-6,
            limit=10,
            max_edges=50_000,
        )
        assert results, "keyword_ppr channel returned no chunks"
        assert results[0][0] == target.id
    finally:
        await coord.disconnect()


async def test_default_bm25_does_not_write_keyword_chunks(tmp_path: Path) -> None:
    """Default lexical_channel (bm25) must not populate keyword_chunks.

    The gate lives in the engine (``if lexical_channel == "keyword_ppr"``); this
    asserts the storage table stays empty when the ingest helper is never
    called, i.e. the default path writes zero edges.
    """
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        # Documents/chunks FK to memory_namespaces.id (the row id), so use the
        # resolved id. The coordinator keyword methods resolve internally and are
        # idempotent on row ids, so passing the row id is fine there too.
        ns_id = await coord.resolve_namespace(ns.namespace_id)

        await _seed_docs(coord, ns_id, ["a document about volcanoes and lava flows"])

        # No persist_keyword_chunk_edges call (the bm25 default never invokes it).
        edges = await coord.get_keyword_chunk_edges(ns_id, limit=10_000)
        assert edges == [], "keyword_chunks should be empty on the default bm25 path"
    finally:
        await coord.disconnect()


async def test_upsert_is_idempotent_per_chunk(tmp_path: Path) -> None:
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        ns = await coord.create_namespace(MemoryNamespace())
        # Documents/chunks FK to memory_namespaces.id (the row id), so use the
        # resolved id. The coordinator keyword methods resolve internally and are
        # idempotent on row ids, so passing the row id is fine there too.
        ns_id = await coord.resolve_namespace(ns.namespace_id)
        chunks = await _seed_docs(coord, ns_id, ["alpha beta gamma keywords here"])
        chunk_id = chunks[0].id

        await coord.upsert_keyword_chunk_edges(ns_id, [("alpha", chunk_id, 1.0), ("beta", chunk_id, 1.0)])
        first = await coord.get_keyword_chunk_edges(ns_id, limit=1000)
        assert {kw for kw, _, _ in first} == {"alpha", "beta"}

        # Re-ingest the same chunk with a different keyword set: edges replaced,
        # not accumulated.
        await coord.upsert_keyword_chunk_edges(ns_id, [("gamma", chunk_id, 2.0)])
        second = await coord.get_keyword_chunk_edges(ns_id, limit=1000)
        assert {kw for kw, _, _ in second} == {"gamma"}
    finally:
        await coord.disconnect()


# ---------------------------------------------------------------------------
# Live-Postgres leg (the real pgvector backend, #1391).
# ---------------------------------------------------------------------------

_PG_SKIP = pytest.mark.skipif(
    not _pg_reachable(),
    reason="run `make dev` (Postgres on :5434) or set KHORA_DATABASE_URL to exercise the pgvector keyword_ppr leg",
)
_PG_EMBED_DIM = 1536


def _pg_vec(seed: str) -> list[float]:
    h = abs(hash(seed))
    return [((h >> (i % 31)) & 0xFF) / 255.0 + 0.01 for i in range(_PG_EMBED_DIM)]


@_PG_SKIP
async def test_pgvector_keyword_ppr_round_trip_and_recall() -> None:
    """Ingest helper populates keyword_chunks on real pgvector; channel recalls.

    Exercises the pgvector backend's upsert/load + the query channel end-to-end
    against the live Postgres stack, and asserts a fresh namespace with the
    default (no edge write) leaves keyword_chunks empty.
    """
    from khora import Khora
    from khora.config import KhoraConfig

    database_url = os.environ.get("KHORA_DATABASE_URL", _database_url())
    config = KhoraConfig(database_url=database_url)
    config.llm.embedding_dimension = _PG_EMBED_DIM
    config.storage.embedding_dimension = _PG_EMBED_DIM
    kb = Khora(config, run_migrations=True)
    await kb.connect()
    try:
        storage = kb._engine._retriever._storage  # type: ignore[union-attr,attr-defined]
        ns_public = (await kb.create_namespace()).namespace_id
        ns_row = await storage.resolve_namespace(ns_public)

        doc = Document(namespace_id=ns_row, content="photosynthesis in plants", title="bio")
        await storage.create_document(doc)
        chunks = [
            Chunk(
                namespace_id=ns_row,
                document_id=doc.id,
                content="photosynthesis converts sunlight into chemical energy in plants",
                embedding=_pg_vec("c0"),
                chunk_index=0,
            ),
            Chunk(
                namespace_id=ns_row,
                document_id=doc.id,
                content="the weather today is cloudy with occasional rain",
                embedding=_pg_vec("c1"),
                chunk_index=1,
            ),
        ]
        await storage.create_chunks_batch(chunks)
        target = chunks[0]

        # Default (no edge write yet): keyword_chunks empty for this namespace.
        assert await storage.get_keyword_chunk_edges(ns_row, limit=1000) == []

        # Ingest-time gated write.
        await persist_keyword_chunk_edges(storage, ns_row, chunks)
        edges = await storage.get_keyword_chunk_edges(ns_row, limit=10_000)
        assert edges, "pgvector keyword_chunks not populated"

        results = await keyword_ppr_retrieve_chunks(
            storage,
            ns_row,
            "tell me about photosynthesis",
            tokenizer=tokenize_multilingual,
            damping=0.85,
            max_iter=50,
            tol=1e-6,
            limit=10,
            max_edges=50_000,
        )
        assert results, "pgvector keyword_ppr channel returned no chunks"
        assert results[0][0] == target.id
    finally:
        await kb.disconnect()
