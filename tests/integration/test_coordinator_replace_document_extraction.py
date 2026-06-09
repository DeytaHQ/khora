"""Real-backend integration tests for ``StorageCoordinator.replace_document_extraction``.

Exercises the full document-replacement lifecycle against a running Postgres
+ Neo4j stack:

- Happy path with mixed retire / survive / net-new entity and relationship sets
- Graph-side failure → document lands in ``FAILED``; next successful replace
  heals it back to ``COMPLETED`` (self-heal)

Gated by ``NEO4J_INTEGRATION_TEST=1``; the CI integration job sets that flag.

How to run locally::

    make dev  # postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \\
        tests/integration/test_coordinator_replace_document_extraction.py -v

Connection parameters (env overrides, sensible ``make dev`` defaults)::

    KHORA_NEO4J_URL          (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME     (default: neo4j)
    KHORA_NEO4J_PASSWORD     (default: password)
    KHORA_DATABASE_URL       (default: postgresql+asyncpg://khora:khora@localhost:5432/khora)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document, Entity, MemoryNamespace, Relationship
from khora.core.models.document import DocumentStatus
from khora.storage.backends.neo4j import Neo4jBackend
from khora.storage.backends.pgvector import PgVectorBackend
from khora.storage.backends.postgresql import PostgreSQLBackend
from khora.storage.coordinator import ReplaceResult, StorageCoordinator

EMBED_DIM = 1536


def _chunks(namespace_id, document_id, count=2) -> list[Chunk]:
    return [
        Chunk(
            namespace_id=namespace_id,
            document_id=document_id,
            content=f"chunk-{i}",
            chunk_index=i,
            embedding=[0.1 * (i + 1)] * EMBED_DIM,
            embedding_model="test",
        )
        for i in range(count)
    ]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real backends (requires make dev)",
)
class TestReplaceDocumentExtractionIntegration:
    """End-to-end replace lifecycle against live Postgres + Neo4j."""

    @pytest.fixture
    async def coord(self) -> AsyncIterator[StorageCoordinator]:
        database_url = os.environ.get(
            "KHORA_DATABASE_URL",
            "postgresql+asyncpg://khora:khora@localhost:5432/khora",
        )
        neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        rel = PostgreSQLBackend(database_url=database_url)
        vec = PgVectorBackend(database_url=database_url, embedding_dimension=EMBED_DIM)
        graph = Neo4jBackend(neo4j_url, user=neo4j_user, password=neo4j_password)

        coord = StorageCoordinator(relational=rel, vector=vec, graph=graph)
        await coord.connect()
        try:
            yield coord
        finally:
            await coord.disconnect()

    @pytest.fixture
    async def namespace_id(self, coord: StorageCoordinator):
        ns = MemoryNamespace()
        created = await coord.create_namespace(ns)
        # Coordinator primitives (create_document, upsert_entities_batch, ...)
        # write the passed id verbatim into FK columns referencing
        # memory_namespaces.id (the row PK), so resolve the stable
        # namespace_id to the active version's row id first. Production
        # Khora.remember() does this resolution before reaching the
        # coordinator; these coordinator-level tests must do it explicitly.
        return await coord.resolve_namespace(created.namespace_id)

    @pytest.mark.asyncio
    async def test_happy_path_mixed_retire_survive_net_new(self, coord: StorageCoordinator, namespace_id) -> None:
        """Full lifecycle: orphan retires, survivor remaps, net-new is created."""
        # Seed: old document with 2 chunks, a survivor entity (alice), an orphan entity (bob)
        old_doc = Document(
            namespace_id=namespace_id,
            content="old body",
            external_id=f"replace-happy-{uuid4().hex[:8]}",
            source="test",
        )
        await coord.create_document(old_doc)
        old_chunks = _chunks(namespace_id, old_doc.id, count=2)
        await coord.create_chunks_batch(old_chunks)

        alice = Entity(
            namespace_id=namespace_id,
            name=f"alice-{uuid4().hex[:6]}",
            entity_type="PERSON",
            source_document_ids=[old_doc.id],
        )
        bob = Entity(
            namespace_id=namespace_id,
            name=f"bob-{uuid4().hex[:6]}",
            entity_type="PERSON",
            source_document_ids=[old_doc.id],
        )
        await coord.upsert_entities_batch(namespace_id, [alice, bob])
        # Relationship: alice -KNOWS-> bob, sole-sourced from old_doc
        rel_alice_bob = Relationship(
            namespace_id=namespace_id,
            source_entity_id=alice.id,
            target_entity_id=bob.id,
            relationship_type="KNOWS",
            source_document_ids=[old_doc.id],
        )
        await coord.create_relationships_batch([rel_alice_bob])

        # New extraction: alice survives, bob is orphaned (not in new_entities),
        # carol is net-new. No new relationship (KNOWS is orphaned).
        new_doc = Document(
            id=old_doc.id,  # update in place
            namespace_id=namespace_id,
            content="new body",
            external_id=old_doc.external_id,
            source="test",
        )
        new_doc.mark_processing()
        new_chunks = _chunks(namespace_id, new_doc.id, count=3)
        alice_new = Entity(
            namespace_id=namespace_id,
            name=alice.name,  # same key ⇒ survivor
            entity_type="PERSON",
            source_document_ids=[new_doc.id],
        )
        carol = Entity(
            namespace_id=namespace_id,
            name=f"carol-{uuid4().hex[:6]}",
            entity_type="PERSON",
            source_document_ids=[new_doc.id],
        )

        result: ReplaceResult = await coord.replace_document_extraction(
            namespace_id=namespace_id,
            old_document_id=old_doc.id,
            new_document=new_doc,
            new_chunks=new_chunks,
            new_entities=[alice_new, carol],
            new_relationships=[],
        )

        assert result.chunks_deleted == 2
        assert result.chunks_created == 3
        assert result.entities_retired == 1  # bob
        assert result.entities_created == 1  # carol
        assert result.entities_updated >= 1  # alice (survivor remap)
        assert result.relationships_retired == 1  # alice-KNOWS-bob
        assert result.relationships_created == 0

        # Document is COMPLETED
        persisted = await coord.get_document(new_doc.id, namespace_id=namespace_id)
        assert persisted is not None
        assert persisted.status == DocumentStatus.COMPLETED

        # Chunk count matches the new chunks
        chunks = await coord.get_chunks_by_document(new_doc.id, namespace_id=namespace_id)
        assert len(chunks) == 3

        # Graph state: alice survives the replace, bob is retired (valid_until
        # set), carol is net-new. This is an in-place replace
        # (new_doc.id == old_doc.id), so the survivor remap leaves the shared
        # doc id present rather than swapping one UUID for another; assert
        # alice is still associated with the surviving doc id.
        alice_fetched = await coord.get_entity(alice.id, namespace_id=namespace_id)
        assert alice_fetched is not None
        assert new_doc.id in alice_fetched.source_document_ids

        bob_fetched = await coord.get_entity(bob.id, namespace_id=namespace_id)
        assert bob_fetched is not None
        assert bob_fetched.valid_until is not None  # retired

        # KNOWS edge was retired (valid_until stamped)
        alice_edges = await coord.get_entity_relationships(alice.id, namespace_id=namespace_id)
        knows = [r for r in alice_edges if r.relationship_type == "KNOWS"]
        assert len(knows) == 1
        assert knows[0].valid_until is not None

    @pytest.mark.asyncio
    async def test_graph_failure_keeps_document_completed_then_next_replace_succeeds(
        self, coord: StorageCoordinator, namespace_id, monkeypatch
    ) -> None:
        """Graph-side failure -> doc stays COMPLETED (PG data is durable, #887);
        next successful replace overwrites cleanly.
        """
        old_doc = Document(
            namespace_id=namespace_id,
            content="seed",
            external_id=f"replace-heal-{uuid4().hex[:8]}",
            source="test",
        )
        await coord.create_document(old_doc)
        old_chunks = _chunks(namespace_id, old_doc.id, count=1)
        await coord.create_chunks_batch(old_chunks)

        new_doc = Document(
            id=old_doc.id,
            namespace_id=namespace_id,
            content="new",
            external_id=old_doc.external_id,
            source="test",
        )
        new_doc.mark_processing()
        new_chunks = _chunks(namespace_id, new_doc.id, count=2)

        # Monkey-patch retire_orphaned_entities_batch to fail — but since we
        # have no pre-existing entities, provide one so the retire code path
        # is triggered.  Create a sole-sourced entity first.
        sole = Entity(
            namespace_id=namespace_id,
            name=f"doomed-{uuid4().hex[:6]}",
            entity_type="PERSON",
            source_document_ids=[old_doc.id],
        )
        await coord.upsert_entities_batch(namespace_id, [sole])

        # ``coord.graph`` is a NamespaceRequiredProxy (the coordinator wraps
        # every public backend attr in __setattr__ regardless of how it was
        # constructed); it has no __dict__ and rejects setattr, so patch the
        # real backend behind it.
        graph_backend = getattr(coord.graph, "_backend", coord.graph)
        orig_retire = graph_backend.retire_orphaned_entities_batch  # type: ignore[union-attr]
        call_count = {"n": 0}

        async def flaky_retire(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("injected graph failure")
            return await orig_retire(*args, **kwargs)

        monkeypatch.setattr(
            graph_backend,
            "retire_orphaned_entities_batch",
            flaky_retire,
        )

        # First replace: PG tx commits (chunks + status stamp persisted), then
        # graph retire raises.  Per #887, the document row remains COMPLETED -
        # PG data is durable, so marking FAILED would diverge status from the
        # fully-written data.  Per #884, the underlying graph exception is
        # wrapped in GraphMirrorFailedAfterPGCommitError so the caller can
        # record the divergence on a user-facing result; the original
        # exception is preserved via __cause__.
        from khora.exceptions import GraphMirrorFailedAfterPGCommitError

        with pytest.raises(GraphMirrorFailedAfterPGCommitError) as exc_info:
            await coord.replace_document_extraction(
                namespace_id=namespace_id,
                old_document_id=old_doc.id,
                new_document=new_doc,
                new_chunks=new_chunks,
                new_entities=[],  # orphans the sole entity
                new_relationships=[],
            )
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "injected graph failure" in str(exc_info.value.__cause__)

        after_graph_fail = await coord.get_document(new_doc.id, namespace_id=namespace_id)
        assert after_graph_fail is not None
        assert after_graph_fail.status == DocumentStatus.COMPLETED
        assert after_graph_fail.error_message is None

        # Second replace (retire will succeed this time) - document stays
        # COMPLETED with the fresh extraction footprint.
        new_doc2 = Document(
            id=new_doc.id,
            namespace_id=namespace_id,
            content="heal",
            external_id=new_doc.external_id,
            source="test",
        )
        new_doc2.mark_processing()
        new_chunks2 = _chunks(namespace_id, new_doc2.id, count=1)

        result = await coord.replace_document_extraction(
            namespace_id=namespace_id,
            old_document_id=new_doc2.id,
            new_document=new_doc2,
            new_chunks=new_chunks2,
            new_entities=[],
            new_relationships=[],
        )
        assert result.document_id == new_doc2.id

        healed = await coord.get_document(new_doc2.id, namespace_id=namespace_id)
        assert healed is not None
        assert healed.status == DocumentStatus.COMPLETED
