"""End-to-end tests for ChronicleEngine with the LanceDB (sqlite_lance) backend.

These tests use real aiosqlite + lancedb under ``tmp_path`` — no mocks at the
storage layer. The LiteLLM embedder is bypassed entirely: chunks are written
through the coordinator with deterministic fake embeddings so the suite is
hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import LLMSettings, SQLiteLanceConfig, StorageSettings
from khora.core.models import Chunk, ChunkMetadata, Document, DocumentMetadata
from tests.integration._sqlite_lance_fixtures import EMBED_DIM, fake_embedding

pytestmark = [
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


def _make_config(db_path: Path) -> KhoraConfig:
    """KhoraConfig pinned to the sqlite_lance backend at ``db_path``."""
    return KhoraConfig(
        storage=StorageSettings(
            backend="sqlite_lance",
            sqlite_lance=SQLiteLanceConfig(
                db_path=str(db_path),
                lance_path=str(db_path.parent / "khora.lance"),
                embedding_dimension=EMBED_DIM,
            ),
        ),
        llm=LLMSettings(embedding_dimension=EMBED_DIM),
    )


async def _seed(coord, namespace_id, items: list[tuple[str, str]]) -> list[Chunk]:
    """Insert (title, content) pairs as one document + one chunk each."""
    chunks: list[Chunk] = []
    for idx, (title, content) in enumerate(items):
        doc = Document(
            namespace_id=namespace_id,
            content=content,
            metadata=DocumentMetadata(title=title, source="test"),
        )
        await coord.create_document(doc)
        chunk = Chunk(
            namespace_id=namespace_id,
            document_id=doc.id,
            content=content,
            metadata=ChunkMetadata(document_id=doc.id, chunk_index=0),
            embedding=fake_embedding(content),
            embedding_model="fake",
        )
        chunks.append(chunk)
    await coord.create_chunks_batch(chunks)
    return chunks


class TestChronicleLanceDBBackend:
    """Chronicle engine wired against the LanceDB-backed coordinator."""

    @pytest.mark.asyncio
    async def test_storage_backend_lancedb_routes_to_sqlite_lance(self, tmp_path: Path) -> None:
        """Passing ``storage_backend='lancedb'`` selects the sqlite_lance coordinator."""
        from khora.engines.chronicle.engine import ChronicleEngine

        config = _make_config(tmp_path / "chronicle.db")
        engine = ChronicleEngine(
            config,
            storage_backend="lancedb",
            lancedb_path=str(tmp_path / "chronicle.db"),
        )
        assert engine._storage_config.backend == "sqlite_lance"
        assert engine._storage_config.sqlite_lance_config is not None
        assert engine._storage_config.sqlite_lance_config.db_path == str(tmp_path / "chronicle.db")

    @pytest.mark.asyncio
    async def test_storage_backend_pgvector_unchanged(self, tmp_path: Path) -> None:
        """Default / pgvector path stays on the traditional StorageConfig."""
        from khora.engines.chronicle.engine import ChronicleEngine

        config = KhoraConfig(database_url="postgresql://localhost/test")
        engine = ChronicleEngine(config)  # default — no override
        assert engine._storage_config.backend != "sqlite_lance"
        assert engine._storage_config.postgresql_url == "postgresql://localhost/test"

    @pytest.mark.asyncio
    async def test_inherits_sqlite_lance_from_config(self, tmp_path: Path) -> None:
        """When ``config.storage.backend == 'sqlite_lance'``, no engine-level override is needed."""
        from khora.engines.chronicle.engine import ChronicleEngine

        config = _make_config(tmp_path / "chronicle.db")
        engine = ChronicleEngine(config)
        assert engine._storage_config.backend == "sqlite_lance"

    @pytest.mark.asyncio
    async def test_recall_returns_top_k_via_lancedb(self, tmp_path: Path) -> None:
        """Ingest 5 chunks, recall top 3 — verifies semantic + BM25 channels run on LanceDB."""
        from khora.engines.chronicle.engine import ChronicleEngine

        config = _make_config(tmp_path / "chronicle.db")
        engine = ChronicleEngine(
            config,
            storage_backend="lancedb",
            lancedb_path=str(tmp_path / "chronicle.db"),
        )
        await engine.connect()
        try:
            # Stub the embedder so we don't reach LiteLLM
            engine._embedder = _StubEmbedder()

            ns = await engine.create_namespace()
            await _seed(
                engine._get_storage(),
                ns.id,
                [
                    ("doc1", "neural networks learn from data"),
                    ("doc2", "cryogenic storage preserves cells"),
                    ("doc3", "graph databases model connections"),
                    ("doc4", "machine learning powers recommendation"),
                    ("doc5", "vector search uses embeddings"),
                ],
            )

            result = await engine.recall("neural networks", ns.id, limit=3)
            assert result.chunks, "recall should return chunks via LanceDB"
            assert len(result.chunks) <= 3
            # Top hit must come from a real chunk, not an empty fallback.
            top_chunk, top_score = result.chunks[0]
            assert top_chunk.content
            assert top_score > 0.0
        finally:
            await engine.disconnect()

    @pytest.mark.asyncio
    async def test_temporal_filter_pushdown(self, tmp_path: Path) -> None:
        """Temporal filter limits results to chunks within the requested window."""
        from datetime import UTC, datetime, timedelta

        from khora.engines.chronicle.engine import ChronicleEngine
        from khora.query.temporal import TemporalFilter

        config = _make_config(tmp_path / "chronicle.db")
        engine = ChronicleEngine(
            config,
            storage_backend="lancedb",
            lancedb_path=str(tmp_path / "chronicle.db"),
        )
        await engine.connect()
        try:
            engine._embedder = _StubEmbedder()

            ns = await engine.create_namespace()
            now = datetime.now(UTC)
            old_cutoff = now - timedelta(days=10)
            old_doc = Document(
                namespace_id=ns.id,
                content="old chunk content about neural networks",
                metadata=DocumentMetadata(title="old", source="test"),
            )
            new_doc = Document(
                namespace_id=ns.id,
                content="new chunk content about neural networks",
                metadata=DocumentMetadata(title="new", source="test"),
            )
            for d in (old_doc, new_doc):
                await engine._get_storage().create_document(d)

            old_chunk = Chunk(
                namespace_id=ns.id,
                document_id=old_doc.id,
                content="old chunk content about neural networks",
                metadata=ChunkMetadata(document_id=old_doc.id, chunk_index=0),
                embedding=fake_embedding("old chunk content about neural networks"),
                embedding_model="fake",
                created_at=old_cutoff,
            )
            new_chunk = Chunk(
                namespace_id=ns.id,
                document_id=new_doc.id,
                content="new chunk content about neural networks",
                metadata=ChunkMetadata(document_id=new_doc.id, chunk_index=0),
                embedding=fake_embedding("new chunk content about neural networks"),
                embedding_model="fake",
                created_at=now,
            )
            await engine._get_storage().create_chunks_batch([old_chunk, new_chunk])

            # Window: only the last 3 days — old chunk must be excluded.
            # Query against the new chunk's own embedding and accept negative
            # cosine similarity so the adapter's similarity floor doesn't drop
            # otherwise-temporally-valid hits.
            tf = TemporalFilter(start_time=now - timedelta(days=3), end_time=now + timedelta(days=1))
            results = await engine._get_storage().search_similar_chunks(
                ns.id,
                fake_embedding("new chunk content about neural networks"),
                limit=10,
                min_similarity=-1.0,
                created_after=tf.start_time,
                created_before=tf.end_time,
            )
            ids = {c.id for c, _ in results}
            assert new_chunk.id in ids, "new chunk must be inside the window"
            assert old_chunk.id not in ids, "old chunk must be excluded by temporal filter"

            # Sanity check: without the filter, both chunks come back.
            unfiltered = await engine._get_storage().search_similar_chunks(
                ns.id,
                fake_embedding("new chunk content about neural networks"),
                limit=10,
                min_similarity=-1.0,
            )
            unfiltered_ids = {c.id for c, _ in unfiltered}
            assert {old_chunk.id, new_chunk.id}.issubset(unfiltered_ids)
        finally:
            await engine.disconnect()

    @pytest.mark.asyncio
    async def test_entity_channel_works_against_lancedb(self, tmp_path: Path) -> None:
        """search_similar_entities + get_entities_batch must work on the sqlite_lance graph."""
        from khora.core.models import Entity
        from khora.engines.chronicle.engine import ChronicleEngine

        config = _make_config(tmp_path / "chronicle.db")
        engine = ChronicleEngine(
            config,
            storage_backend="lancedb",
            lancedb_path=str(tmp_path / "chronicle.db"),
        )
        await engine.connect()
        try:
            engine._embedder = _StubEmbedder()
            ns = await engine.create_namespace()
            doc = Document(
                namespace_id=ns.id,
                content="alice and bob",
                metadata=DocumentMetadata(title="people", source="test"),
            )
            await engine._get_storage().create_document(doc)

            alice = Entity(
                namespace_id=ns.id,
                name="Alice",
                entity_type="PERSON",
                description="protagonist",
                source_document_ids=[doc.id],
                embedding=fake_embedding("Alice PERSON"),
            )
            bob = Entity(
                namespace_id=ns.id,
                name="Bob",
                entity_type="PERSON",
                source_document_ids=[doc.id],
                embedding=fake_embedding("Bob PERSON"),
            )
            await engine._get_storage().upsert_entities_batch(ns.id, [alice, bob])
            # Mirror the chronicle ingest pipeline: entity rows go through the
            # graph upsert; their embeddings are written separately through the
            # vector adapter via update_entity_embeddings_batch.
            await engine._get_storage().update_entity_embeddings_batch(
                [
                    (alice.id, fake_embedding("Alice PERSON"), "fake"),
                    (bob.id, fake_embedding("Bob PERSON"), "fake"),
                ]
            )

            # Vector adapter writes embeddings to LanceDB; entity search should
            # surface both rows.
            hits = await engine._get_storage().search_similar_entities(
                ns.id,
                fake_embedding("Alice PERSON"),
                limit=5,
                min_similarity=-1.0,
            )
            entity_ids = [eid for eid, _ in hits]
            assert alice.id in entity_ids

            # The N+1 fallback in GraphBackendBase handles batch fetch.
            fetched = await engine._get_storage().get_entities_batch([alice.id, bob.id])
            assert alice.id in fetched and bob.id in fetched
            assert fetched[alice.id].name == "Alice"
        finally:
            await engine.disconnect()


class _StubEmbedder:
    """Returns deterministic fake embeddings, no LiteLLM round-trip."""

    async def embed(self, text: str) -> list[float]:
        return fake_embedding(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [fake_embedding(t) for t in texts]
