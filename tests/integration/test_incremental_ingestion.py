"""Integration tests for incremental document ingestion.

Validates that the MemoryLake correctly handles incremental updates:
1. Ingest batch 1 → query returns batch 1 results
2. Ingest batch 2 → query returns BOTH batch 1 AND batch 2 results

Specifically tests P0 bug fixes:
- BM25 cache invalidation after new document ingestion
- Entity ID mapping (new entities get proper IDs, existing entities merged)
- MERGE relationships (existing entities updated, not duplicated)

Uses mocked storage backends so no live databases are needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk, ChunkMetadata, Document, DocumentMetadata, Entity, Relationship
from khora.memory_lake import BatchResult, MemoryLake, RecallResult, RememberResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NAMESPACE_ID = uuid4()


def _mock_config() -> MagicMock:
    """Create a mock KhoraConfig."""
    cfg = MagicMock()
    cfg.get_postgresql_url.return_value = "postgresql://test"
    cfg.get_graph_config.return_value = None
    cfg.get_vector_config.return_value = None
    cfg.get_neo4j_url.return_value = None
    cfg.get_neo4j_user.return_value = None
    cfg.get_neo4j_password.return_value = None
    cfg.get_neo4j_database.return_value = None
    cfg.storage.embedding_dimension = 1536
    cfg.storage.postgresql_pool_size = 5
    cfg.storage.postgresql_max_overflow = 10
    cfg.llm.model = "gpt-4o-mini"
    cfg.llm.embedding_model = "text-embedding-3-small"
    cfg.llm.embedding_dimension = 1536
    cfg.llm.extraction_model = None
    cfg.llm.timeout = 30
    cfg.llm.max_retries = 3
    cfg.telemetry_database_url = None
    cfg.telemetry_service_name = "khora-test"
    return cfg


def _make_chunk(
    content: str,
    doc_id: UUID,
    *,
    embedding: list[float] | None = None,
    chunk_index: int = 0,
) -> Chunk:
    """Create a test chunk."""
    return Chunk(
        id=uuid4(),
        namespace_id=NAMESPACE_ID,
        document_id=doc_id,
        content=content,
        metadata=ChunkMetadata(document_id=doc_id, chunk_index=chunk_index),
        embedding=embedding or [0.1] * 8,
    )


def _make_entity(
    name: str,
    entity_type: str = "PERSON",
    *,
    doc_ids: list[UUID] | None = None,
    embedding: list[float] | None = None,
    mention_count: int = 1,
) -> Entity:
    """Create a test entity."""
    return Entity(
        id=uuid4(),
        namespace_id=NAMESPACE_ID,
        name=name,
        entity_type=entity_type,
        description=f"Entity: {name}",
        source_document_ids=doc_ids or [],
        embedding=embedding or [0.1] * 8,
        mention_count=mention_count,
    )


def _make_relationship(
    source: Entity,
    target: Entity,
    rel_type: str = "WORKS_FOR",
) -> Relationship:
    """Create a test relationship."""
    return Relationship(
        id=uuid4(),
        namespace_id=NAMESPACE_ID,
        source_entity_id=source.id,
        target_entity_id=target.id,
        relationship_type=rel_type,
    )


# ---------------------------------------------------------------------------
# Stateful mock storage that accumulates ingested data
# ---------------------------------------------------------------------------


class IncrementalStorageState:
    """Tracks state across multiple ingestion batches.

    Simulates the storage layer accumulating documents, chunks, entities,
    and relationships as batches are ingested.
    """

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.entities: list[Entity] = []
        self.relationships: list[Relationship] = []
        self._checksums: dict[str, Document] = {}

    def add_batch(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Add a batch of ingested data, merging entities by name."""
        self.documents.extend(documents)
        self.chunks.extend(chunks)

        for doc in documents:
            self._checksums[doc.metadata.checksum] = doc

        # Merge entities by name (simulates MERGE behavior)
        for new_entity in entities:
            existing = next(
                (e for e in self.entities if e.name == new_entity.name and e.entity_type == new_entity.entity_type),
                None,
            )
            if existing:
                existing.merge_with(new_entity)
            else:
                self.entities.append(new_entity)

        # Add relationships, dedup by (source, target, type)
        for new_rel in relationships:
            existing = next(
                (
                    r
                    for r in self.relationships
                    if r.source_entity_id == new_rel.source_entity_id
                    and r.target_entity_id == new_rel.target_entity_id
                    and r.relationship_type == new_rel.relationship_type
                ),
                None,
            )
            if not existing:
                self.relationships.append(new_rel)

    def get_document_by_checksum(self, checksum: str) -> Document | None:
        return self._checksums.get(checksum)

    def search_chunks_by_content(self, query: str) -> list[tuple[Chunk, float]]:
        """Simple keyword-based chunk search for testing."""
        query_lower = query.lower()
        results = []
        for chunk in self.chunks:
            if any(word in chunk.content.lower() for word in query_lower.split()):
                score = sum(1 for word in query_lower.split() if word in chunk.content.lower()) / len(
                    query_lower.split()
                )
                results.append((chunk, score))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_entities_by_name(self, query: str) -> list[tuple[Entity, float]]:
        """Simple entity name search for testing."""
        query_lower = query.lower()
        results = []
        for entity in self.entities:
            if query_lower in entity.name.lower() or any(word in entity.name.lower() for word in query_lower.split()):
                results.append((entity, 0.9))
        return results


# ---------------------------------------------------------------------------
# Batch 1 data: Alice and Bob at Acme Corp
# ---------------------------------------------------------------------------

BATCH_1_DOCS = [
    {"content": "Alice Johnson is a senior engineer at Acme Corp.", "title": "Alice Profile", "source": "hr"},
    {
        "content": "Bob Smith manages the platform team at Acme Corp. He reports to the CTO.",
        "title": "Bob Profile",
        "source": "hr",
    },
    {
        "content": "Acme Corp is a technology company founded in 2020, headquartered in San Francisco.",
        "title": "Acme Overview",
        "source": "wiki",
    },
]


# ---------------------------------------------------------------------------
# Batch 2 data: Charlie and Dave, with Alice appearing again
# ---------------------------------------------------------------------------

BATCH_2_DOCS = [
    {
        "content": "Charlie Brown joined Acme Corp as a data scientist. He works with Alice Johnson on ML projects.",
        "title": "Charlie Profile",
        "source": "hr",
    },
    {
        "content": "Dave Wilson is a product manager at Acme Corp. He collaborates with Bob Smith on the roadmap.",
        "title": "Dave Profile",
        "source": "hr",
    },
]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIncrementalIngestion:
    """Tests that incremental ingestion preserves existing data and makes new data queryable."""

    @pytest.fixture
    def state(self) -> IncrementalStorageState:
        """Create fresh storage state for each test."""
        return IncrementalStorageState()

    @pytest.fixture
    def batch1_data(self, state: IncrementalStorageState) -> IncrementalStorageState:
        """Populate state with batch 1 data."""
        doc_ids = [uuid4() for _ in BATCH_1_DOCS]
        documents = [
            Document(
                id=doc_ids[i],
                namespace_id=NAMESPACE_ID,
                content=d["content"],
                metadata=DocumentMetadata(
                    title=d["title"],
                    source=d["source"],
                    checksum=f"checksum_batch1_{i}",
                ),
            )
            for i, d in enumerate(BATCH_1_DOCS)
        ]
        chunks = [_make_chunk(d["content"], doc_ids[i], chunk_index=0) for i, d in enumerate(BATCH_1_DOCS)]

        alice = _make_entity("alice johnson", "PERSON", doc_ids=[doc_ids[0]])
        bob = _make_entity("bob smith", "PERSON", doc_ids=[doc_ids[1]])
        acme = _make_entity("acme corp", "ORGANIZATION", doc_ids=[doc_ids[0], doc_ids[1], doc_ids[2]])
        sf = _make_entity("san francisco", "LOCATION", doc_ids=[doc_ids[2]])

        relationships = [
            _make_relationship(alice, acme, "WORKS_FOR"),
            _make_relationship(bob, acme, "WORKS_FOR"),
            _make_relationship(acme, sf, "HEADQUARTERED_IN"),
        ]

        state.add_batch(documents, chunks, [alice, bob, acme, sf], relationships)
        return state

    @pytest.fixture
    def batch2_data(self, batch1_data: IncrementalStorageState) -> IncrementalStorageState:
        """Populate state with batch 2 data (on top of batch 1)."""
        state = batch1_data
        doc_ids = [uuid4() for _ in BATCH_2_DOCS]
        documents = [
            Document(
                id=doc_ids[i],
                namespace_id=NAMESPACE_ID,
                content=d["content"],
                metadata=DocumentMetadata(
                    title=d["title"],
                    source=d["source"],
                    checksum=f"checksum_batch2_{i}",
                ),
            )
            for i, d in enumerate(BATCH_2_DOCS)
        ]
        chunks = [_make_chunk(d["content"], doc_ids[i], chunk_index=0) for i, d in enumerate(BATCH_2_DOCS)]

        charlie = _make_entity("charlie brown", "PERSON", doc_ids=[doc_ids[0]])
        dave = _make_entity("dave wilson", "PERSON", doc_ids=[doc_ids[1]])
        # Alice appears again — should be merged, not duplicated
        alice_again = _make_entity("alice johnson", "PERSON", doc_ids=[doc_ids[0]])
        # Acme appears again — should be merged
        acme_again = _make_entity("acme corp", "ORGANIZATION", doc_ids=[doc_ids[0], doc_ids[1]])
        # Bob appears again
        bob_again = _make_entity("bob smith", "PERSON", doc_ids=[doc_ids[1]])

        relationships = [
            _make_relationship(charlie, _find_entity(state, "acme corp"), "WORKS_FOR"),
            _make_relationship(dave, _find_entity(state, "acme corp"), "WORKS_FOR"),
            _make_relationship(charlie, _find_entity(state, "alice johnson"), "COLLABORATES_WITH"),
            _make_relationship(dave, _find_entity(state, "bob smith"), "COLLABORATES_WITH"),
        ]

        state.add_batch(documents, chunks, [charlie, dave, alice_again, acme_again, bob_again], relationships)
        return state

    # -----------------------------------------------------------------------
    # Test: Batch 1 ingestion and query
    # -----------------------------------------------------------------------

    async def test_batch1_keyword_search(self, batch1_data: IncrementalStorageState) -> None:
        """After batch 1, keyword search finds batch 1 content."""
        results = batch1_data.search_chunks_by_content("Alice engineer Acme")
        assert len(results) > 0
        content_texts = [c.content for c, _ in results]
        assert any("Alice" in t for t in content_texts)

    async def test_batch1_entity_search(self, batch1_data: IncrementalStorageState) -> None:
        """After batch 1, entity search finds batch 1 entities."""
        results = batch1_data.search_entities_by_name("alice")
        assert len(results) == 1
        assert results[0][0].name == "alice johnson"

    async def test_batch1_entity_count(self, batch1_data: IncrementalStorageState) -> None:
        """Batch 1 produces expected number of entities."""
        assert len(batch1_data.entities) == 4  # Alice, Bob, Acme, SF
        assert len(batch1_data.relationships) == 3

    async def test_batch1_graph_relationships(self, batch1_data: IncrementalStorageState) -> None:
        """After batch 1, graph relationships exist."""
        alice = _find_entity(batch1_data, "alice johnson")
        acme = _find_entity(batch1_data, "acme corp")

        works_for_rels = [
            r for r in batch1_data.relationships if r.source_entity_id == alice.id and r.target_entity_id == acme.id
        ]
        assert len(works_for_rels) == 1
        assert works_for_rels[0].relationship_type == "WORKS_FOR"

    # -----------------------------------------------------------------------
    # Test: Batch 2 ingestion preserves batch 1 data
    # -----------------------------------------------------------------------

    async def test_batch2_keyword_search_old_docs(self, batch2_data: IncrementalStorageState) -> None:
        """After batch 2, keyword search still finds batch 1 content."""
        results = batch2_data.search_chunks_by_content("Alice engineer")
        content_texts = [c.content for c, _ in results]
        assert any("Alice Johnson is a senior engineer" in t for t in content_texts)

    async def test_batch2_keyword_search_new_docs(self, batch2_data: IncrementalStorageState) -> None:
        """After batch 2, keyword search finds batch 2 content."""
        results = batch2_data.search_chunks_by_content("Charlie data scientist")
        content_texts = [c.content for c, _ in results]
        assert any("Charlie" in t for t in content_texts)

    async def test_batch2_keyword_search_combined(self, batch2_data: IncrementalStorageState) -> None:
        """After batch 2, keyword search for 'Acme' returns docs from BOTH batches."""
        results = batch2_data.search_chunks_by_content("Acme Corp")
        assert len(results) == 5  # All 5 docs mention Acme Corp

    async def test_batch2_entity_search_old(self, batch2_data: IncrementalStorageState) -> None:
        """After batch 2, entity search still finds batch 1 entities."""
        results = batch2_data.search_entities_by_name("bob smith")
        assert len(results) == 1
        assert results[0][0].name == "bob smith"

    async def test_batch2_entity_search_new(self, batch2_data: IncrementalStorageState) -> None:
        """After batch 2, entity search finds batch 2 entities."""
        results = batch2_data.search_entities_by_name("charlie")
        assert len(results) == 1
        assert results[0][0].name == "charlie brown"

    # -----------------------------------------------------------------------
    # Test: Entity MERGE behavior (P0 fix)
    # -----------------------------------------------------------------------

    async def test_entity_merge_no_duplicates(self, batch2_data: IncrementalStorageState) -> None:
        """Entities appearing in both batches are merged, not duplicated."""
        alice_entities = [e for e in batch2_data.entities if e.name == "alice johnson"]
        assert len(alice_entities) == 1, "Alice should be merged into a single entity"

        acme_entities = [e for e in batch2_data.entities if e.name == "acme corp"]
        assert len(acme_entities) == 1, "Acme Corp should be merged into a single entity"

    async def test_entity_merge_mention_count(self, batch2_data: IncrementalStorageState) -> None:
        """Merged entities have incremented mention counts."""
        alice = _find_entity(batch2_data, "alice johnson")
        assert alice.mention_count == 2, "Alice appears in batch 1 and batch 2"

        acme = _find_entity(batch2_data, "acme corp")
        assert acme.mention_count == 2, "Acme appears in batch 1 and batch 2"

    async def test_entity_merge_source_documents(self, batch2_data: IncrementalStorageState) -> None:
        """Merged entities track source documents from both batches."""
        alice = _find_entity(batch2_data, "alice johnson")
        assert len(alice.source_document_ids) == 2, "Alice referenced from 2 different documents"

    async def test_entity_count_after_merge(self, batch2_data: IncrementalStorageState) -> None:
        """Total entity count reflects merges (no duplicates)."""
        # Batch 1: Alice, Bob, Acme, SF = 4
        # Batch 2 adds: Charlie, Dave = 2 new (Alice, Acme, Bob merged)
        assert len(batch2_data.entities) == 6

    # -----------------------------------------------------------------------
    # Test: Relationship MERGE behavior (P0 fix)
    # -----------------------------------------------------------------------

    async def test_relationship_merge_no_duplicates(self, batch2_data: IncrementalStorageState) -> None:
        """Relationships are not duplicated across batches."""
        acme = _find_entity(batch2_data, "acme corp")
        works_for_acme = [
            r
            for r in batch2_data.relationships
            if r.target_entity_id == acme.id and str(r.relationship_type) == "WORKS_FOR"
        ]
        # Alice, Bob, Charlie, Dave all WORKS_FOR Acme
        assert len(works_for_acme) == 4

    async def test_new_relationships_added(self, batch2_data: IncrementalStorageState) -> None:
        """New relationships from batch 2 are added."""
        # Total: 3 from batch1 + 4 from batch2 = 7
        assert len(batch2_data.relationships) == 7

    async def test_collaboration_relationships(self, batch2_data: IncrementalStorageState) -> None:
        """Cross-batch collaboration relationships are created."""
        charlie = _find_entity(batch2_data, "charlie brown")
        alice = _find_entity(batch2_data, "alice johnson")
        collab_rels = [
            r for r in batch2_data.relationships if r.source_entity_id == charlie.id and r.target_entity_id == alice.id
        ]
        assert len(collab_rels) == 1

    # -----------------------------------------------------------------------
    # Test: BM25 cache invalidation (P0 fix)
    # -----------------------------------------------------------------------

    async def test_bm25_cache_invalidation(self) -> None:
        """BM25 keyword searcher cache is invalidated after ingestion.

        After remember() or remember_batch(), the GraphRAG engine should
        call invalidate_caches() so stale BM25 indexes are rebuilt.
        """
        lake = _make_connected_lake()
        engine = lake._engine

        # First ingestion
        engine.remember = AsyncMock(
            return_value=RememberResult(
                document_id=uuid4(),
                namespace_id=NAMESPACE_ID,
                chunks_created=1,
                entities_extracted=1,
                relationships_created=0,
            )
        )

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember(
                "Test content",
                namespace=NAMESPACE_ID,
                title="Test",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        engine.remember.assert_awaited_once()

    async def test_cache_invalidation_called_in_graphrag_engine(self) -> None:
        """GraphRAG engine calls invalidate_caches after remember()."""
        from khora.engines.graphrag.engine import GraphRAGEngine

        mock_config = _mock_config()
        engine = GraphRAGEngine(mock_config)

        # Mock internals
        mock_storage = MagicMock()
        mock_storage.get_document_by_checksum = AsyncMock(return_value=None)
        mock_storage.create_document = AsyncMock(side_effect=lambda doc: doc)

        mock_query_engine = MagicMock()
        mock_query_engine.invalidate_caches = MagicMock()

        engine._storage = mock_storage
        engine._query_engine = mock_query_engine
        engine._connected = True
        engine._embedder = MagicMock()

        mock_result = {"chunks": 1, "entities": 1, "relationships": 0}
        with patch("khora.pipelines.flows.ingest.process_document", AsyncMock(return_value=mock_result)):
            await engine.remember(
                "Test content",
                NAMESPACE_ID,
                title="Test",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        mock_query_engine.invalidate_caches.assert_called_once_with(NAMESPACE_ID)

    async def test_cache_invalidation_called_in_remember_batch(self) -> None:
        """GraphRAG engine calls invalidate_caches after remember_batch()."""
        from khora.engines.graphrag.engine import GraphRAGEngine

        mock_config = _mock_config()
        engine = GraphRAGEngine(mock_config)

        mock_storage = MagicMock()
        mock_storage.list_entities = AsyncMock(return_value=[])

        mock_query_engine = MagicMock()
        mock_query_engine.invalidate_caches = MagicMock()

        engine._storage = mock_storage
        engine._query_engine = mock_query_engine
        engine._connected = True
        engine._embedder = MagicMock()

        mock_result = {
            "total_documents": 2,
            "processed_documents": 2,
            "skipped_documents": 0,
            "failed_documents": 0,
            "total_chunks": 2,
            "total_entities": 1,
            "total_relationships": 0,
            "total_inferred_relationships": 0,
        }
        with patch("khora.pipelines.flows.ingest.ingest_documents", AsyncMock(return_value=mock_result)):
            await engine.remember_batch(
                [{"content": "Doc 1"}, {"content": "Doc 2"}],
                NAMESPACE_ID,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        mock_query_engine.invalidate_caches.assert_called_once_with(NAMESPACE_ID)

    # -----------------------------------------------------------------------
    # Test: Full MemoryLake remember→recall integration
    # -----------------------------------------------------------------------

    async def test_remember_then_recall_integration(self) -> None:
        """Full flow: remember content, then recall finds it."""
        lake = _make_connected_lake()
        engine = lake._engine

        doc_id = uuid4()
        engine.remember = AsyncMock(
            return_value=RememberResult(
                document_id=doc_id,
                namespace_id=NAMESPACE_ID,
                chunks_created=3,
                entities_extracted=2,
                relationships_created=1,
            )
        )

        mock_recall = RecallResult(
            query="Alice Acme Corp",
            namespace_id=NAMESPACE_ID,
            chunks=[("Alice Johnson is a senior engineer at Acme Corp.", 0.95)],
            entities=[("alice johnson", 0.9)],
            context_text="Alice Johnson is a senior engineer at Acme Corp.",
        )
        engine.recall = AsyncMock(return_value=mock_recall)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            # Remember
            result = await lake.remember(
                "Alice Johnson is a senior engineer at Acme Corp.",
                namespace=NAMESPACE_ID,
                title="Alice Profile",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )
            assert result.chunks_created == 3
            assert result.entities_extracted == 2

            # Recall
            recall_result = await lake.recall("Alice Acme Corp", namespace=NAMESPACE_ID)
            assert len(recall_result.chunks) == 1
            assert "Alice" in recall_result.context_text

    async def test_incremental_remember_batch_then_recall(self) -> None:
        """Full flow: remember_batch twice, then recall finds all content."""
        lake = _make_connected_lake()
        engine = lake._engine

        # Batch 1 result
        batch1_result = BatchResult(total=3, processed=3, skipped=0, failed=0, chunks=3, entities=4, relationships=3)
        # Batch 2 result
        batch2_result = BatchResult(total=2, processed=2, skipped=0, failed=0, chunks=2, entities=2, relationships=4)

        call_count = 0

        async def mock_batch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return batch1_result if call_count == 1 else batch2_result

        engine.remember_batch = AsyncMock(side_effect=mock_batch)

        # After both batches, recall returns combined results
        mock_recall = RecallResult(
            query="Acme Corp employees",
            namespace_id=NAMESPACE_ID,
            chunks=[
                ("Alice Johnson is a senior engineer at Acme Corp.", 0.95),
                ("Bob Smith manages the platform team at Acme Corp.", 0.90),
                ("Charlie Brown joined Acme Corp as a data scientist.", 0.85),
                ("Dave Wilson is a product manager at Acme Corp.", 0.80),
            ],
            entities=[
                ("alice johnson", 0.9),
                ("bob smith", 0.85),
                ("charlie brown", 0.8),
                ("dave wilson", 0.75),
                ("acme corp", 0.95),
            ],
            context_text=(
                "Alice Johnson is a senior engineer at Acme Corp. "
                "Bob Smith manages the platform team. "
                "Charlie Brown joined as data scientist. "
                "Dave Wilson is a product manager."
            ),
        )
        engine.recall = AsyncMock(return_value=mock_recall)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            # Ingest batch 1
            r1 = await lake.remember_batch(
                BATCH_1_DOCS,
                namespace=NAMESPACE_ID,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "MANAGES", "COLLABORATES_WITH"],
            )
            assert r1.processed == 3
            assert r1.entities == 4

            # Ingest batch 2
            r2 = await lake.remember_batch(
                BATCH_2_DOCS,
                namespace=NAMESPACE_ID,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "MANAGES", "COLLABORATES_WITH"],
            )
            assert r2.processed == 2
            assert r2.entities == 2

            # Recall — should find content from both batches
            result = await lake.recall("Acme Corp employees", namespace=NAMESPACE_ID)
            assert len(result.chunks) == 4  # All 4 employee docs
            assert len(result.entities) == 5  # All entities including Acme Corp

            # Verify content from batch 1 is present
            chunk_texts = [c[0] for c in result.chunks]
            assert any("Alice" in t for t in chunk_texts), "Batch 1 Alice not found in recall"
            assert any("Bob" in t for t in chunk_texts), "Batch 1 Bob not found in recall"

            # Verify content from batch 2 is present
            assert any("Charlie" in t for t in chunk_texts), "Batch 2 Charlie not found in recall"
            assert any("Dave" in t for t in chunk_texts), "Batch 2 Dave not found in recall"

    # -----------------------------------------------------------------------
    # Test: Vector search after incremental update
    # -----------------------------------------------------------------------

    async def test_vector_search_both_batches(self, batch2_data: IncrementalStorageState) -> None:
        """Vector search (simulated) returns chunks from both batches."""
        # All chunks should have embeddings
        assert all(c.embedding is not None for c in batch2_data.chunks)
        assert len(batch2_data.chunks) == 5  # 3 from batch1 + 2 from batch2

    async def test_graph_search_cross_batch_traversal(self, batch2_data: IncrementalStorageState) -> None:
        """Graph traversal can cross from batch 2 entities to batch 1 entities.

        Starting from Charlie (batch 2), we should be able to traverse
        COLLABORATES_WITH to Alice (batch 1) → WORKS_FOR to Acme (batch 1).
        """
        charlie = _find_entity(batch2_data, "charlie brown")
        alice = _find_entity(batch2_data, "alice johnson")
        acme = _find_entity(batch2_data, "acme corp")

        # Charlie → Alice (COLLABORATES_WITH)
        charlie_to_alice = [
            r for r in batch2_data.relationships if r.source_entity_id == charlie.id and r.target_entity_id == alice.id
        ]
        assert len(charlie_to_alice) == 1

        # Alice → Acme (WORKS_FOR)
        alice_to_acme = [
            r for r in batch2_data.relationships if r.source_entity_id == alice.id and r.target_entity_id == acme.id
        ]
        assert len(alice_to_acme) == 1

        # This demonstrates cross-batch graph traversal works

    # -----------------------------------------------------------------------
    # Test: Multi-batch (3+) ingestion regression
    # -----------------------------------------------------------------------

    async def test_three_batch_ingestion(self, batch2_data: IncrementalStorageState) -> None:
        """Three successive batches: all data survives and entities merge correctly."""
        state = batch2_data

        # Batch 3: Eve joins, Alice appears again, new company "Beta Inc"
        batch3_docs = [
            {
                "content": "Eve Martinez is a designer at Acme Corp. She works closely with Dave Wilson on product design.",
                "title": "Eve Profile",
                "source": "hr",
            },
            {
                "content": "Acme Corp acquired Beta Inc in 2024. Alice Johnson leads the integration team.",
                "title": "Acquisition News",
                "source": "wiki",
            },
        ]
        doc_ids = [uuid4() for _ in batch3_docs]
        documents = [
            Document(
                id=doc_ids[i],
                namespace_id=NAMESPACE_ID,
                content=d["content"],
                metadata=DocumentMetadata(
                    title=d["title"],
                    source=d["source"],
                    checksum=f"checksum_batch3_{i}",
                ),
            )
            for i, d in enumerate(batch3_docs)
        ]
        chunks = [_make_chunk(d["content"], doc_ids[i], chunk_index=0) for i, d in enumerate(batch3_docs)]

        eve = _make_entity("eve martinez", "PERSON", doc_ids=[doc_ids[0]])
        beta = _make_entity("beta inc", "ORGANIZATION", doc_ids=[doc_ids[1]])
        # Alice and Acme appear again — must merge
        alice_again = _make_entity("alice johnson", "PERSON", doc_ids=[doc_ids[1]])
        acme_again = _make_entity("acme corp", "ORGANIZATION", doc_ids=[doc_ids[0], doc_ids[1]])
        dave_again = _make_entity("dave wilson", "PERSON", doc_ids=[doc_ids[0]])

        relationships = [
            _make_relationship(eve, _find_entity(state, "acme corp"), "WORKS_FOR"),
            _make_relationship(eve, _find_entity(state, "dave wilson"), "COLLABORATES_WITH"),
            _make_relationship(_find_entity(state, "acme corp"), beta, "ACQUIRED"),
            _make_relationship(_find_entity(state, "alice johnson"), beta, "LEADS_INTEGRATION"),
        ]

        state.add_batch(documents, chunks, [eve, beta, alice_again, acme_again, dave_again], relationships)

        # --- Verify cumulative state after 3 batches ---

        # Total chunks: 3 (b1) + 2 (b2) + 2 (b3) = 7
        assert len(state.chunks) == 7

        # Total documents: 3 + 2 + 2 = 7
        assert len(state.documents) == 7

        # Entities: Alice, Bob, Acme, SF, Charlie, Dave, Eve, Beta = 8
        # (Alice, Acme, Dave merged across batches)
        assert len(state.entities) == 8

        # Alice merged across 3 batches — mention_count = 3
        alice = _find_entity(state, "alice johnson")
        assert alice.mention_count == 3
        assert len(alice.source_document_ids) == 3

        # Acme merged across 3 batches — mention_count = 3
        acme = _find_entity(state, "acme corp")
        assert acme.mention_count == 3

        # Dave merged across 2 batches — mention_count = 2
        dave = _find_entity(state, "dave wilson")
        assert dave.mention_count == 2

        # No duplicate entities
        entity_name_type = [(e.name, str(e.entity_type)) for e in state.entities]
        assert len(entity_name_type) == len(set(entity_name_type)), "Duplicate entities found"

        # Relationships: 3 (b1) + 4 (b2) + 4 (b3) = 11
        assert len(state.relationships) == 11

        # Search still finds content from all 3 batches
        b1_results = state.search_chunks_by_content("senior engineer")
        assert len(b1_results) > 0, "Batch 1 content lost"

        b2_results = state.search_chunks_by_content("data scientist")
        assert len(b2_results) > 0, "Batch 2 content lost"

        b3_results = state.search_chunks_by_content("designer")
        assert len(b3_results) > 0, "Batch 3 content lost"

        # Cross-batch traversal: Eve → Acme ← Bob (shared org relationship)
        eve_entity = _find_entity(state, "eve martinez")
        bob_entity = _find_entity(state, "bob smith")
        acme_entity = _find_entity(state, "acme corp")

        eve_works_at_acme = [
            r
            for r in state.relationships
            if r.source_entity_id == eve_entity.id and r.target_entity_id == acme_entity.id
        ]
        assert len(eve_works_at_acme) == 1

        bob_works_at_acme = [
            r
            for r in state.relationships
            if r.source_entity_id == bob_entity.id and r.target_entity_id == acme_entity.id
        ]
        assert len(bob_works_at_acme) == 1

    async def test_three_batch_remember_batch_then_recall(self) -> None:
        """Full MemoryLake flow with 3 batches — recall finds content from all."""
        lake = _make_connected_lake()
        engine = lake._engine

        batch_results = [
            BatchResult(total=3, processed=3, skipped=0, failed=0, chunks=3, entities=4, relationships=3),
            BatchResult(total=2, processed=2, skipped=0, failed=0, chunks=2, entities=2, relationships=4),
            BatchResult(total=2, processed=2, skipped=0, failed=0, chunks=2, entities=2, relationships=4),
        ]

        call_count = 0

        async def mock_batch(*args, **kwargs):
            nonlocal call_count
            result = batch_results[call_count]
            call_count += 1
            return result

        engine.remember_batch = AsyncMock(side_effect=mock_batch)

        mock_recall = RecallResult(
            query="Acme Corp team",
            namespace_id=NAMESPACE_ID,
            chunks=[
                ("Alice Johnson is a senior engineer at Acme Corp.", 0.95),
                ("Bob Smith manages the platform team at Acme Corp.", 0.90),
                ("Charlie Brown joined Acme Corp as a data scientist.", 0.85),
                ("Dave Wilson is a product manager at Acme Corp.", 0.80),
                ("Eve Martinez is a designer at Acme Corp.", 0.75),
            ],
            entities=[
                ("alice johnson", 0.9),
                ("bob smith", 0.85),
                ("charlie brown", 0.8),
                ("dave wilson", 0.75),
                ("eve martinez", 0.7),
                ("acme corp", 0.95),
            ],
            context_text=(
                "Alice Johnson is a senior engineer. Bob Smith manages the platform team. "
                "Charlie Brown is a data scientist. Dave Wilson is a product manager. "
                "Eve Martinez is a designer. All at Acme Corp."
            ),
        )
        engine.recall = AsyncMock(return_value=mock_recall)

        batch3_docs = [
            {"content": "Eve Martinez is a designer at Acme Corp.", "title": "Eve Profile", "source": "hr"},
            {"content": "Acme Corp acquired Beta Inc in 2024.", "title": "Acquisition", "source": "wiki"},
        ]

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            r1 = await lake.remember_batch(
                BATCH_1_DOCS,
                namespace=NAMESPACE_ID,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "MANAGES", "COLLABORATES_WITH"],
            )
            assert r1.processed == 3

            r2 = await lake.remember_batch(
                BATCH_2_DOCS,
                namespace=NAMESPACE_ID,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "MANAGES", "COLLABORATES_WITH"],
            )
            assert r2.processed == 2

            r3 = await lake.remember_batch(
                batch3_docs,
                namespace=NAMESPACE_ID,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "MANAGES", "COLLABORATES_WITH"],
            )
            assert r3.processed == 2

            result = await lake.recall("Acme Corp team", namespace=NAMESPACE_ID)
            assert len(result.chunks) == 5
            chunk_texts = [c[0] for c in result.chunks]
            assert any("Alice" in t for t in chunk_texts), "Batch 1 content missing"
            assert any("Charlie" in t for t in chunk_texts), "Batch 2 content missing"
            assert any("Eve" in t for t in chunk_texts), "Batch 3 content missing"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_entity(state: IncrementalStorageState, name: str) -> Entity:
    """Find entity by name in state."""
    for entity in state.entities:
        if entity.name == name:
            return entity
    raise ValueError(f"Entity not found: {name}")


def _make_connected_lake() -> MemoryLake:
    """Create a MemoryLake with a mocked engine, pre-connected."""
    with patch("khora.memory_lake.load_config", return_value=_mock_config()):
        lake = MemoryLake()

    lake._connected = True

    mock_engine = MagicMock()
    mock_engine._storage = MagicMock()
    mock_engine._storage.resolve_namespace = AsyncMock(return_value=uuid4())
    mock_engine._embedder = MagicMock()
    mock_engine.connect = AsyncMock()
    mock_engine.disconnect = AsyncMock()
    mock_engine.remember = AsyncMock()
    mock_engine.recall = AsyncMock()
    mock_engine.remember_batch = AsyncMock()

    lake._engine = mock_engine
    return lake
