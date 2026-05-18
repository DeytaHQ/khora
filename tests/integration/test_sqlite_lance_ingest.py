"""End-to-end ingest + search integration tests for sqlite_lance.

Exercises the full ``StorageCoordinator`` stack against a live SQLite +
LanceDB pair in ``tmp_path``.  No mocks at the storage layer; LLM +
embedder are replaced with deterministic fakes so the suite runs
hermetically.
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

from khora.core.models import Chunk, ChunkMetadata, Document, DocumentMetadata, Entity, MemoryNamespace
from tests.integration._sqlite_lance_fixtures import (
    build_sqlite_lance_coordinator,
    fake_embedding,
)

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


def _make_document(namespace_id, *, idx: int, topic: str) -> Document:
    content = f"Document {idx}: {topic} — details about the {topic} and its uses."
    return Document(
        namespace_id=namespace_id,
        content=content,
        external_id=f"doc-{idx}",
        metadata=DocumentMetadata(source="test", title=f"doc-{idx}"),
    )


def _make_chunk(namespace_id, document_id, *, idx: int, content: str) -> Chunk:
    return Chunk(
        namespace_id=namespace_id,
        document_id=document_id,
        content=content,
        metadata=ChunkMetadata(document_id=document_id, chunk_index=idx),
        embedding=fake_embedding(content),
        embedding_model="fake",
    )


async def _seed_ingest(coord, namespace_id, topics):
    """Insert one document + one chunk per topic.  Returns list of (doc, chunk)."""
    pairs: list[tuple[Document, Chunk]] = []
    for i, topic in enumerate(topics):
        doc = _make_document(namespace_id, idx=i, topic=topic)
        await coord.create_document(doc)
        chunk = _make_chunk(namespace_id, doc.id, idx=0, content=doc.content)
        pairs.append((doc, chunk))
    chunks = [c for _, c in pairs]
    await coord.create_chunks_batch(chunks)
    return pairs


class TestSQLiteLanceIngest:
    """Full ingest + recall lifecycle through the coordinator."""

    async def test_ingest_100_documents_counts_and_fetch(self, tmp_path: Path) -> None:
        """100 docs ingested via create_document + create_chunks_batch are queryable."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())

            topics = [f"topic-{i}" for i in range(100)]
            pairs = await _seed_ingest(coord, ns.id, topics)

            assert await coord.count_documents(ns.id) == 100
            assert await coord.count_chunks(ns.id) == 100

            # Spot-check one document survives a round-trip.
            sample_doc, sample_chunk = pairs[42]
            fetched = await coord.get_document(sample_doc.id)
            assert fetched is not None
            assert fetched.external_id == "doc-42"

            chunk_fetched = await coord.vector.get_chunk(sample_chunk.id, namespace_id=ns.id)  # type: ignore[union-attr]
            assert chunk_fetched is not None
            assert chunk_fetched.content == sample_chunk.content
        finally:
            await coord.disconnect()

    async def test_vector_search_orders_by_similarity(self, tmp_path: Path) -> None:
        """Vector ANN search returns the seed chunk first when queried with its own embedding."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            topics = ["apples", "bananas", "cherries", "dates", "elderberries"]
            pairs = await _seed_ingest(coord, ns.id, topics)

            # Query with the exact embedding of the "cherries" document —
            # that chunk must come back first (similarity = 1.0).
            target_doc, target_chunk = pairs[2]
            results = await coord.search_similar_chunks(
                ns.id,
                target_chunk.embedding,  # type: ignore[arg-type]
                limit=5,
            )
            assert results, "expected non-empty vector search results"
            top_chunk, top_score = results[0]
            assert top_chunk.id == target_chunk.id
            assert top_score == pytest.approx(1.0, abs=1e-3)
            # Scores must be monotonically non-increasing (ordering invariant).
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            await coord.disconnect()

    async def test_fulltext_bm25_finds_relevant_chunk(self, tmp_path: Path) -> None:
        """FTS5 BM25 full-text search matches tokens in chunk content."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            # Distinct tokens per doc so BM25 can pick one out unambiguously.
            topics = ["quantum", "mesoscopic", "zettabyte", "isotropic", "tungsten"]
            pairs = await _seed_ingest(coord, ns.id, topics)

            results = await coord.search_fulltext_chunks(ns.id, "zettabyte", limit=5)
            assert results, "expected FTS5 to return at least one match"
            top_chunk, score = results[0]
            assert top_chunk.id == pairs[2][1].id
            assert "zettabyte" in top_chunk.content.lower()
            assert score > 0
        finally:
            await coord.disconnect()

    async def test_chronicle_events_and_facts_persist(self, tmp_path: Path) -> None:
        """Regression for issue #529: chronicle write/query on sqlite_lance.

        Before the fix the coordinator dispatched chronicle methods only to
        the vector adapter (LanceDB), which doesn't implement them — every
        write was silently dropped and queries raised. After the fix the
        coordinator falls back to the relational adapter, which writes to
        the chronicle_events / memory_facts tables migration 024 already
        created on SQLite.
        """
        from datetime import UTC, datetime

        from khora.engines.chronicle.compression import MemoryFact
        from khora.engines.chronicle.events import ChronicleEvent

        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            # Seed a document + chunk so the chronicle FK target exists.
            doc = _make_document(ns.id, idx=0, topic="curie")
            await coord.create_document(doc)
            chunk = _make_chunk(ns.id, doc.id, idx=0, content=doc.content)
            await coord.create_chunks_batch([chunk])

            # --- write_events / query_events round-trip ---------------------
            ev1 = ChronicleEvent(
                chunk_id=chunk.id,
                namespace_id=ns.id,
                subject="Marie Curie",
                verb="won",
                object="Nobel Prize",
                observation_date=datetime.now(UTC),
                referenced_date=datetime(1903, 12, 10, tzinfo=UTC),
                confidence=0.95,
                source_text="Marie Curie won the Nobel Prize in Physics in 1903.",
            )
            ev2 = ChronicleEvent(
                chunk_id=chunk.id,
                namespace_id=ns.id,
                subject="Marie Curie",
                verb="discovered",
                object="radium",
                observation_date=datetime.now(UTC),
                confidence=0.9,
                source_text="Curie discovered radium.",
            )
            event_ids = await coord.write_events([ev1, ev2], namespace_id=ns.id)
            assert len(event_ids) == 2
            assert event_ids == [ev1.id, ev2.id]

            events = await coord.query_events(ns.id, subject="Marie Curie", limit=10)
            assert len(events) == 2
            verbs = {e.verb for e in events}
            assert verbs == {"won", "discovered"}

            # --- write_facts / query_active_facts_for_subject ---------------
            f1 = MemoryFact(
                namespace_id=ns.id,
                subject="Marie Curie",
                predicate="won",
                object_="Nobel Prize",
                fact_text="Marie Curie won the Nobel Prize.",
                confidence=0.95,
                source_chunk_ids=[chunk.id],
            )
            f2 = MemoryFact(
                namespace_id=ns.id,
                subject="Marie Curie",
                predicate="discovered",
                object_="radium",
                fact_text="Marie Curie discovered radium.",
                confidence=0.9,
                source_chunk_ids=[chunk.id],
            )
            fact_ids = await coord.write_facts([f1, f2], namespace_id=ns.id)
            assert len(fact_ids) == 2

            facts = await coord.query_active_facts_for_subject(ns.id, "Marie Curie")
            assert len(facts) == 2
            assert all(f.is_active for f in facts)
            assert {f.predicate for f in facts} == {"won", "discovered"}
            # source_chunk_ids round-trip as UUIDs even though SQLite stores them as JSON.
            for f in facts:
                assert chunk.id in f.source_chunk_ids

            # --- supersede_fact flips is_active ----------------------------
            # FK constraint: superseded_by must reference an existing fact;
            # write a replacement first.
            replacement = MemoryFact(
                namespace_id=ns.id,
                subject="Marie Curie",
                predicate="won",
                object_="Nobel Prize in Physics",
                fact_text="Marie Curie won the Nobel Prize in Physics.",
                confidence=0.99,
                source_chunk_ids=[chunk.id],
            )
            await coord.write_facts([replacement], namespace_id=ns.id)
            await coord.supersede_fact(f1.id, replacement.id)

            active = await coord.query_active_facts_for_subject(ns.id, "Marie Curie")
            # f2 + replacement should remain active; f1 was superseded.
            assert {f.id for f in active} == {f2.id, replacement.id}

            # --- namespace isolation: another namespace sees nothing ------
            other = await coord.create_namespace(MemoryNamespace())
            assert await coord.query_events(other.id, subject="Marie Curie") == []
            assert await coord.query_active_facts_for_subject(other.id, "Marie Curie") == []

            # --- empty inputs short-circuit (no SQL emitted) ----------------
            assert await coord.write_events([], namespace_id=ns.id) == []
            assert await coord.write_facts([], namespace_id=ns.id) == []
        finally:
            await coord.disconnect()

    async def test_fulltext_bm25_handles_punctuation_in_query(self, tmp_path: Path) -> None:
        """Regression for issue #526: natural-language queries with `?`, `:`, etc.

        Before the fix, ``search_fulltext_chunks(ns, "What did Curie win?")``
        raised ``sqlite3.OperationalError: fts5: syntax error near "?"``. The
        primary assertion here is "no crash"; the secondary assertion is that
        recall is preserved for a single token surrounded by FTS5 metachars.
        """
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            topics = ["quantum", "mesoscopic", "zettabyte", "isotropic", "tungsten"]
            await _seed_ingest(coord, ns.id, topics)

            # These inputs would have raised fts5 syntax errors prior to the fix.
            # Single-token queries each contain "zettabyte" wrapped in FTS5 metachars;
            # post-fix the chunk should still surface.
            for query in (
                "zettabyte?",  # `?` was the original crash
                "zettabyte:",  # `:` is FTS5 column filter
                "(zettabyte)",  # `(` `)` are grouping operators
                "zettabyte*",  # `*` is prefix operator
                "-zettabyte",  # leading `-` is FTS5 NOT in some contexts
            ):
                results = await coord.search_fulltext_chunks(ns.id, query, limit=5)
                assert results, f"expected ≥1 hit for query {query!r}"
                assert any("zettabyte" in c.content.lower() for c, _ in results), (
                    f"expected the zettabyte chunk to surface for {query!r}"
                )

            # Multi-word natural-language queries containing FTS5 keywords/punctuation
            # must not raise. Recall depends on the seeded content; we only assert
            # that the call returns (a list, possibly empty).
            for query in (
                "What did Curie win?",  # the issue #526 repro
                "zettabyte AND mesoscopic",  # `AND` as bareword operator
                "zettabyte OR mesoscopic",  # `OR` as bareword operator
                "zettabyte NEAR mesoscopic",  # `NEAR` as bareword operator
                'say "hello"',  # embedded double quotes
            ):
                results = await coord.search_fulltext_chunks(ns.id, query, limit=5)
                assert isinstance(results, list), f"call must return a list (no crash) for {query!r}"

            # Empty / whitespace-only query short-circuits (no crash, no hits).
            for empty_query in ("", "   ", "\t\n"):
                results = await coord.search_fulltext_chunks(ns.id, empty_query, limit=5)
                assert results == [], f"expected empty result for {empty_query!r}"
        finally:
            await coord.disconnect()

    async def test_hybrid_both_modalities_contribute(self, tmp_path: Path) -> None:
        """Vector + BM25 each return the right chunk; a naive merge covers both."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            topics = ["neural", "quantum", "relativity", "photonic", "cryogenic"]
            pairs = await _seed_ingest(coord, ns.id, topics)

            # Semantic channel: query by "neural" chunk's embedding.
            neural_chunk = pairs[0][1]
            vec_hits = await coord.search_similar_chunks(ns.id, neural_chunk.embedding, limit=3)

            # Keyword channel: query FTS for "cryogenic".
            cryo_hits = await coord.search_fulltext_chunks(ns.id, "cryogenic", limit=3)

            vec_ids = {c.id for c, _ in vec_hits}
            fts_ids = {c.id for c, _ in cryo_hits}

            assert neural_chunk.id in vec_ids, "vector channel must surface the neural chunk"
            assert pairs[4][1].id in fts_ids, "BM25 channel must surface the cryogenic chunk"
            # The two modalities contribute distinct top-hits ⇒ a hybrid
            # fusion (RRF, weighted sum, etc.) would see both.
            assert vec_ids != fts_ids
        finally:
            await coord.disconnect()

    async def test_entities_upsert_appears_in_graph(self, tmp_path: Path) -> None:
        """Coordinator upsert_entities_batch writes through the graph adapter."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            doc = _make_document(ns.id, idx=0, topic="alice")
            await coord.create_document(doc)

            alice = Entity(
                namespace_id=ns.id,
                name="Alice",
                entity_type="PERSON",
                description="protagonist",
                source_document_ids=[doc.id],
            )
            bob = Entity(
                namespace_id=ns.id,
                name="Bob",
                entity_type="PERSON",
                source_document_ids=[doc.id],
            )
            results = await coord.upsert_entities_batch(ns.id, [alice, bob])

            # Both rows are new inserts.
            assert all(is_new for _, is_new in results)
            assert await coord.graph.count_entities(ns.id) == 2  # type: ignore[union-attr]

            # Same key upsert merges — mention_count bumps, row count stays at 2.
            alice_again = Entity(
                namespace_id=ns.id,
                name="Alice",
                entity_type="PERSON",
                source_document_ids=[doc.id],
            )
            await coord.upsert_entities_batch(ns.id, [alice_again])
            assert await coord.graph.count_entities(ns.id) == 2  # type: ignore[union-attr]
        finally:
            await coord.disconnect()

    async def test_chunk_insert_with_unknown_document_id_fails(self, tmp_path: Path) -> None:
        """FKs must be enforced: a chunk pointing at a non-existent document
        cannot be inserted.  After the drift fix, UUIDs on both
        sides (``chunks.document_id`` and ``documents.id``) share the same
        format (32-char hex) and SQLite's FK checker can compare them.
        """
        from uuid import uuid4

        import aiosqlite

        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())

            # Craft a chunk row that references a document that was never
            # created.  Use the same storage format the adapter uses so
            # the failure must come from FK enforcement, not a format
            # mismatch that silently passes.
            bogus_doc = uuid4()
            chunk = _make_chunk(ns.id, bogus_doc, idx=0, content="orphan")

            with pytest.raises((aiosqlite.IntegrityError, Exception)) as excinfo:
                await coord.create_chunks_batch([chunk])
            # Make sure the failure is FK-related, not a generic error.
            assert "FOREIGN KEY" in str(excinfo.value).upper() or "foreign key" in str(excinfo.value).lower()
        finally:
            await coord.disconnect()
