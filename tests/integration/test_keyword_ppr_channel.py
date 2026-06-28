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

import ast
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


async def test_default_bm25_ingest_writes_no_keyword_chunks(tmp_path: Path) -> None:
    """A REAL default-config (bm25) ingest through the engine writes zero edges.

    Unlike asserting "the helper was never called", this drives an actual
    ``VectorCypherEngine.remember()`` with the default ``lexical_channel="bm25"``
    so the engine's ``if lexical_channel == "keyword_ppr"`` gate is exercised and
    evaluates False. If the default ingest path ever started persisting edges
    unconditionally, keyword_chunks would be non-empty and this fails.

    Hermetic: sqlite_lance backend (graph-less, no Neo4j), a fake embedder (no
    LiteLLM), and ``extract_entities=False`` (no LLM extraction).
    """
    from khora.config import KhoraConfig
    from khora.config.schema import SQLiteLanceConfig
    from khora.db.session import run_migrations
    from khora.engines.vectorcypher.engine import VectorCypherEngine

    db_path = str(tmp_path / "khora.db")
    lance_path = str(tmp_path / "khora.lance")
    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    assert result.success, f"migration failed: {result.error}"

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(db_path=db_path, lance_path=lance_path, embedding_dimension=32)
    config.storage.embedding_dimension = 32
    config.llm.embedding_dimension = 32
    config.pipeline.extract_entities = False
    assert config.query.lexical_channel == "bm25", "default lexical_channel must be bm25 for this test"

    class _FakeEmbedder:
        async def embed(self, text: str) -> list[float]:
            return fake_embedding(text)

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [fake_embedding(t) for t in texts]

    engine = VectorCypherEngine(config)
    await engine.connect()
    try:
        engine._embedder = _FakeEmbedder()  # type: ignore[assignment]  # no LiteLLM in the hermetic suite
        ns = await engine.create_namespace()
        # The engine's remember() works in row-id space (the public Khora wrapper
        # does the stable->row resolution); documents FK to memory_namespaces.id.
        ns_row = ns.id
        result = await engine.remember(
            "a document about volcanoes and lava flows and magma chambers",
            ns_row,
            entity_types=[],
            relationship_types=[],
        )
        # The ingest really ran (so the no-edges assertion below is not vacuous).
        assert result.chunks_created > 0, "default ingest produced no chunks"
        edges = await engine._storage.get_keyword_chunk_edges(ns_row, limit=10_000)  # type: ignore[union-attr]
        assert edges == [], "default bm25 ingest must not populate keyword_chunks"
    finally:
        await engine.disconnect()


def test_every_engine_edge_write_is_gated_on_keyword_ppr() -> None:
    """Pin the ingest gate to source: no UNGATED ``persist_keyword_chunk_edges``.

    Belt-and-braces with the behavioral test above: every
    ``persist_keyword_chunk_edges*(...)`` call in the engine is lexically inside
    an ``if ... lexical_channel == "keyword_ppr"`` guard, so default bm25
    deployments never write keyword_chunks. AST source-scan (no DB), so it fails
    the instant a NEW call site is added without the gate — catching a regression
    statically even before the behavioral test runs.
    """
    edge_write_fns = {"persist_keyword_chunk_edges", "persist_keyword_chunk_edges_from_keywords"}
    engine_src = Path(__file__).resolve().parents[2] / "src" / "khora" / "engines" / "vectorcypher" / "engine.py"
    tree = ast.parse(engine_src.read_text())

    def _guards_keyword_ppr(test: ast.expr) -> bool:
        # True if any Compare in the test is `... == "keyword_ppr"`.
        return any(
            isinstance(cmp.comparators[0], ast.Constant) and cmp.comparators[0].value == "keyword_ppr"
            for cmp in ast.walk(test)
            if isinstance(cmp, ast.Compare) and cmp.comparators
        )

    # Map each call line -> whether it sits under a keyword_ppr-guarded `if`.
    guarded_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _guards_keyword_ppr(node.test):
            for child in ast.walk(node):
                guarded_lines.add(getattr(child, "lineno", -1))

    ungated = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in edge_write_fns
        and node.lineno not in guarded_lines
    ]
    assert not ungated, (
        f"a keyword_chunk edge write was called WITHOUT a `lexical_channel == 'keyword_ppr'` "
        f"guard at engine.py line(s) {ungated} — the default bm25 path would write "
        "keyword_chunks. Wrap the call in the gate."
    )


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

    Uses a graph-LESS coordinator (PostgreSQL relational + pgvector on one shared
    engine), not a full ``Khora(...).connect()``: the channel only touches the
    storage layer + the query helper, and the full engine's ``connect()`` would
    require Neo4j credentials this PG-only leg has no business depending on
    (mirrors test_chronicle_filter_pgvector.py).
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from khora.db.session import run_migrations
    from khora.storage.backends.pgvector import PgVectorBackend
    from khora.storage.backends.postgresql import PostgreSQLBackend
    from khora.storage.coordinator import StorageCoordinator

    database_url = os.environ.get("KHORA_DATABASE_URL", _database_url())
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    result = await run_migrations(database_url)
    assert result.success, f"migrations failed: {result.error}"

    engine = create_async_engine(database_url)
    relational = PostgreSQLBackend(database_url, engine=engine)
    vector = PgVectorBackend(database_url, embedding_dimension=_PG_EMBED_DIM, engine=engine)
    storage = StorageCoordinator(relational=relational, vector=vector)
    await storage.connect()
    try:
        ns = await storage.create_namespace(MemoryNamespace())
        ns_row = await storage.resolve_namespace(ns.namespace_id)

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
        await storage.disconnect()
        await engine.dispose()
