"""Integration test reproducing IGR-202: ``forget()`` leaks orphan entities
and relationships into ``entity_search`` / ``entity_explore``.

Before the DYT-4164 cascade, ``khora.forget(document_id, namespace)`` deleted
the document row and its chunks but never touched the entities or
relationships extracted from that document. Both pgvector and Neo4j retained
the orphan, so:

* ``search_entities`` (pgvector) still surfaced the orphan entity.
* ``find_related_entities`` (Neo4j graph traversal) still walked to the
  orphan via leftover edges.

This test exercises the public ``Khora.remember()`` / ``Khora.forget()`` API
against a real Postgres + Neo4j stack and asserts:

1. Orphan entities (single-source = the forgotten doc) disappear from both
   backends after ``forget()``.
2. Orphan relationships likewise disappear.
3. Survivor entities/relationships (multi-source = co-mentioned in another
   surviving doc) remain queryable; only the forgotten ``document_id`` is
   stripped from their ``source_document_ids`` array.

Gated by ``NEO4J_INTEGRATION_TEST=1`` (CI does not provision Neo4j).

How to run locally::

    make dev  # postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \\
        tests/integration/test_forget_cascade_orphans_integration.py -v

Connection parameters (env overrides, sensible ``make dev`` defaults)::

    KHORA_NEO4J_URL          (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME     (default: neo4j)
    KHORA_NEO4J_PASSWORD     (default: password)
    KHORA_DATABASE_URL       (default: postgresql+asyncpg://khora:khora@localhost:5432/khora)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.khora import Khora

EMBED_DIM = 4

_EXTRACTION_REGISTRY: dict[str, ExtractionResult] = {}


def _plan_extraction(
    marker: str,
    entities: list[tuple[str, str]],
    relationships: list[tuple[str, str, str]],
) -> None:
    _EXTRACTION_REGISTRY[marker] = ExtractionResult(
        entities=[ExtractedEntity(name=n, entity_type=t, confidence=0.99) for n, t in entities],
        relationships=[
            ExtractedRelationship(
                source_entity=s,
                target_entity=t,
                relationship_type=rt,
                confidence=0.99,
            )
            for s, t, rt in relationships
        ],
    )


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    out: list[ExtractionResult] = []
    for text in texts:
        matched = next(
            (result for marker, result in _EXTRACTION_REGISTRY.items() if marker in text),
            None,
        )
        out.append(matched if matched is not None else ExtractionResult())
    return out


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    unit = [1.0] + [0.0] * (EMBED_DIM - 1)
    return [unit[:] for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return [1.0] + [0.0] * (EMBED_DIM - 1)


async def _run_cypher(driver: Any, query: str, **params: Any) -> list[dict[str, Any]]:
    async with driver.session() as session:
        result = await session.run(query, **params)
        return await result.data()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real backends (requires make dev)",
)
class TestForgetCascadeOrphansIntegration:
    """End-to-end IGR-202 reproduction against real Postgres + Neo4j."""

    @pytest.fixture(autouse=True)
    def _stub_extractor_and_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _EXTRACTION_REGISTRY.clear()
        monkeypatch.setattr(
            "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
            _stub_extract_multi,
        )
        monkeypatch.setattr(
            "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
            _stub_embed_batch,
        )
        monkeypatch.setattr(
            "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
            _stub_embed,
        )

    @pytest.fixture(scope="class")
    async def kb(self) -> AsyncIterator[Khora]:
        database_url = os.environ.get(
            "KHORA_DATABASE_URL",
            "postgresql+asyncpg://khora:khora@localhost:5432/khora",
        )
        neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        config = KhoraConfig(database_url=database_url, neo4j_url=neo4j_url)
        config.storage.neo4j_user = neo4j_user
        config.storage.neo4j_password = neo4j_password
        config.llm.embedding_dimension = EMBED_DIM
        config.storage.embedding_dimension = EMBED_DIM
        config.pipeline.chunk_size = 1024
        config.pipeline.extract_entities = True
        config.pipeline.selective_extraction = False

        kb = Khora(config, run_migrations=False)
        await kb.connect()
        try:
            yield kb
        finally:
            await kb.disconnect()

    @pytest.fixture
    async def namespace_id(self, kb: Khora) -> UUID:
        ns = await kb.create_namespace()
        return ns.namespace_id

    def _graph_driver(self, kb: Khora) -> Any:
        graph = kb.storage.graph
        assert graph is not None, "graph backend must be configured"
        driver = getattr(graph, "_driver", None)
        assert driver is not None, "Neo4j driver must be connected"
        return driver

    async def _remember(
        self,
        kb: Khora,
        *,
        namespace_id: UUID,
        content: str,
    ) -> Any:
        return await kb.remember(
            content=content,
            namespace=namespace_id,
            entity_types=["PERSON", "CONCEPT"],
            relationship_types=["KNOWS", "RELATES_TO"],
        )

    @pytest.mark.asyncio
    async def test_forget_removes_orphan_entity_from_search_and_explore(self, kb: Khora, namespace_id: UUID) -> None:
        """IGR-202 core reproduction: orphan entity disappears from entity_search
        and find_related_entities after forget().

        Setup: 2 docs, each mentioning a unique entity.
        - doc_a: (alice, mallory, KNOWS) — alice will survive via doc_b
        - doc_b: (alice, charlie, KNOWS) — charlie & bob from doc_a unrelated

        After forget(doc_a):
          - mallory (single-source = doc_a) must be GONE from both backends.
          - alice (multi-source) must REMAIN, but doc_a stripped from sdids.
          - The (alice, mallory, KNOWS) edge must be GONE.
          - The (alice, charlie, KNOWS) edge must REMAIN.
        """
        driver = self._graph_driver(kb)
        ns_uuid = await kb.storage.resolve_namespace(namespace_id)
        ns_str = str(ns_uuid)

        alice = f"alice-{uuid4().hex[:6]}"
        mallory = f"mallory-{uuid4().hex[:6]}"
        charlie = f"charlie-{uuid4().hex[:6]}"
        marker_a = f"forget-a-{uuid4().hex[:6]}"
        marker_b = f"forget-b-{uuid4().hex[:6]}"

        _plan_extraction(
            marker_a,
            entities=[(alice, "PERSON"), (mallory, "PERSON")],
            relationships=[(alice, mallory, "KNOWS")],
        )
        _plan_extraction(
            marker_b,
            entities=[(alice, "PERSON"), (charlie, "PERSON")],
            relationships=[(alice, charlie, "KNOWS")],
        )

        doc_a = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"{marker_a} chronicle naming {alice} and {mallory}",
        )
        doc_b = await self._remember(
            kb,
            namespace_id=namespace_id,
            content=f"{marker_b} chronicle naming {alice} and {charlie}",
        )

        # Sanity: pre-forget — all 3 entities surface via entity_search.
        results = await kb.search_entities(namespace_id=namespace_id, query="anyone", limit=50)
        names_before = {e.name for e in results}
        assert alice in names_before
        assert mallory in names_before, (
            "pre-forget: mallory expected to be surfaced by entity search (extraction did not persist)"
        )
        assert charlie in names_before

        # Pre-forget: alice's entity row in Neo4j carries both doc ids.
        alice_row_before = await _run_cypher(
            driver,
            "MATCH (e:Entity {namespace_id: $ns, name: $name}) RETURN e.id AS id, e.source_document_ids AS sdids",
            ns=ns_str,
            name=alice,
        )
        assert len(alice_row_before) == 1
        sdids_before = set(alice_row_before[0]["sdids"])
        assert str(doc_a.document_id) in sdids_before
        assert str(doc_b.document_id) in sdids_before

        # ----- Act: forget doc_a -----
        forgot = await kb.forget(doc_a.document_id, namespace=namespace_id)
        assert forgot is True

        # ----- Assertions: orphan entity gone, survivor remains -----

        # 1. mallory (single-source on doc_a) gone from entity_search.
        results_after = await kb.search_entities(namespace_id=namespace_id, query="anyone", limit=50)
        names_after = {e.name for e in results_after}
        assert mallory not in names_after, (
            f"IGR-202 regression: orphan entity {mallory!r} still surfaced "
            f"by entity_search after forget(doc_a). Got names: {names_after}"
        )
        # Survivors still visible.
        assert alice in names_after
        assert charlie in names_after

        # 2. mallory's Entity node fully gone from Neo4j.
        mallory_rows = await _run_cypher(
            driver,
            "MATCH (e:Entity {namespace_id: $ns, name: $name}) RETURN e.id AS id",
            ns=ns_str,
            name=mallory,
        )
        assert mallory_rows == [], (
            f"IGR-202 regression: orphan entity {mallory!r} still in Neo4j after "
            f"forget(doc_a). Found rows: {mallory_rows}"
        )

        # 3. mallory's pgvector row also gone.
        mallory_pg_ids = [e.id for e in results_after if e.name == mallory]
        assert mallory_pg_ids == []

        # 4. The (alice, mallory, KNOWS) edge is gone.
        alice_mallory_edge = await _run_cypher(
            driver,
            """
            MATCH (s:Entity {namespace_id: $ns, name: $a})
                  -[r]->(t:Entity {namespace_id: $ns, name: $m})
            RETURN type(r) AS type
            """,
            ns=ns_str,
            a=alice,
            m=mallory,
        )
        assert alice_mallory_edge == []

        # 5. Survivor entity alice: still present, with doc_a STRIPPED from sdids.
        alice_row_after = await _run_cypher(
            driver,
            "MATCH (e:Entity {namespace_id: $ns, name: $name}) RETURN e.id AS id, e.source_document_ids AS sdids",
            ns=ns_str,
            name=alice,
        )
        assert len(alice_row_after) == 1
        sdids_after = set(alice_row_after[0]["sdids"])
        assert str(doc_a.document_id) not in sdids_after, (
            f"survivor entity {alice!r} should have {doc_a.document_id} stripped "
            f"from source_document_ids; got {sdids_after}"
        )
        assert str(doc_b.document_id) in sdids_after, (
            f"survivor entity {alice!r} lost reference to surviving doc; sdids={sdids_after}"
        )

        # 6. The (alice, charlie, KNOWS) edge (only on doc_b) still walks via
        #    find_related_entities — entity_explore must still surface charlie
        #    from alice's neighborhood.
        alice_id = UUID(alice_row_after[0]["id"])
        related = await kb.find_related_entities(
            entity_id=alice_id,
            namespace_id=namespace_id,
            max_depth=1,
            limit=20,
        )
        related_names = {e.name for e, _ in related}
        assert charlie in related_names, (
            f"survivor edge (alice -> charlie) lost from find_related_entities "
            f"after forget(doc_a); related={related_names}"
        )
        assert mallory not in related_names, (
            f"IGR-202 regression: find_related_entities still walks to orphan "
            f"{mallory!r} after forget(doc_a); related={related_names}"
        )

        # Keep doc_b around long enough for later debugging — explicit cleanup:
        await kb.forget(doc_b.document_id, namespace=namespace_id)

    @pytest.mark.asyncio
    async def test_forget_strips_doc_from_survivor_relationship_sources(self, kb: Khora, namespace_id: UUID) -> None:
        """Co-sourced relationship: same (alice, bob, KNOWS) extracted from two
        documents. After forget(doc_a), the edge still exists with doc_b in
        its source_document_ids, but doc_a is stripped."""
        driver = self._graph_driver(kb)
        ns_uuid = await kb.storage.resolve_namespace(namespace_id)
        ns_str = str(ns_uuid)

        alice = f"alice-{uuid4().hex[:6]}"
        bob = f"bob-{uuid4().hex[:6]}"
        marker_a = f"rel-a-{uuid4().hex[:6]}"
        marker_b = f"rel-b-{uuid4().hex[:6]}"

        _plan_extraction(
            marker_a,
            entities=[(alice, "PERSON"), (bob, "PERSON")],
            relationships=[(alice, bob, "KNOWS")],
        )
        _plan_extraction(
            marker_b,
            entities=[(alice, "PERSON"), (bob, "PERSON")],
            relationships=[(alice, bob, "KNOWS")],
        )

        doc_a = await self._remember(kb, namespace_id=namespace_id, content=f"{marker_a} {alice} and {bob}")
        doc_b = await self._remember(kb, namespace_id=namespace_id, content=f"{marker_b} {alice} and {bob}")

        # Sanity: pre-forget the KNOWS edge carries both doc ids.
        edges_before = await _run_cypher(
            driver,
            """
            MATCH (s:Entity {namespace_id: $ns, name: $a})
                  -[r:KNOWS]->(t:Entity {namespace_id: $ns, name: $b})
            RETURN r.id AS id, r.source_document_ids AS sdids
            """,
            ns=ns_str,
            a=alice,
            b=bob,
        )
        assert len(edges_before) == 1
        sdids_before = set(edges_before[0]["sdids"])
        assert {str(doc_a.document_id), str(doc_b.document_id)} <= sdids_before

        # ----- Act -----
        assert await kb.forget(doc_a.document_id, namespace=namespace_id) is True

        # The edge survives because doc_b still references it.
        edges_after = await _run_cypher(
            driver,
            """
            MATCH (s:Entity {namespace_id: $ns, name: $a})
                  -[r:KNOWS]->(t:Entity {namespace_id: $ns, name: $b})
            RETURN r.id AS id, r.source_document_ids AS sdids
            """,
            ns=ns_str,
            a=alice,
            b=bob,
        )
        assert len(edges_after) == 1, (
            f"survivor relationship should still exist (doc_b sources it); got edges={edges_after}"
        )
        sdids_after = set(edges_after[0]["sdids"])
        assert str(doc_a.document_id) not in sdids_after
        assert str(doc_b.document_id) in sdids_after

        # Cleanup
        await kb.forget(doc_b.document_id, namespace=namespace_id)
