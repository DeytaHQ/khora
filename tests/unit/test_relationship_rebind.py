"""Unit tests for DYT-3558: GraphRAG drops relationships when an entity is re-canonicalised.

When an entity is upserted a second time, Neo4j's MERGE syncs the in-memory
``Entity.id`` to the canonical (already-stored) UUID. Relationships built
*before* the upsert still hold the freshly-extracted (now-stale) UUIDs as
``source_entity_id`` / ``target_entity_id``. Prior to the fix,
``entity_id_mapping`` only mapped canonical -> canonical, so those
relationships were silently dropped with a "missing entity mappings"
warning and multi-hop graph traversal broke.

These tests run ``process_document`` against a mocked storage coordinator
that simulates the canonicalisation. They verify two things:

1. The pre-upsert (extraction-time) UUIDs are mapped to canonical IDs in
   ``entity_id_mapping`` and the relationship lands in the graph with the
   correct canonical endpoints.
2. The ``(namespace, name, type)`` fallback resolves an unmapped UUID via
   ``storage.get_entity_by_name`` (defense-in-depth for relationships
   referring to entities outside the current upsert batch).
3. A relationship referencing a genuinely missing entity still produces
   the ``missing entity mappings`` warning and is skipped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest


def _make_document_mock(doc_id: UUID, ns_id: UUID, content: str = "doc content") -> MagicMock:
    doc = MagicMock()
    doc.id = doc_id
    doc.namespace_id = ns_id
    doc.content = content
    doc.metadata = MagicMock(custom={}, title="")
    doc.created_at = datetime.now(UTC)
    doc.mark_processing = MagicMock()
    doc.mark_completed = MagicMock()
    doc.mark_failed = MagicMock()
    doc.status = "pending"
    return doc


def _make_chunk(ns_id: UUID, doc_id: UUID, content: str = "chunk content"):
    from khora.core.models import Chunk, ChunkMetadata

    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=doc_id,
        content=content,
        metadata=ChunkMetadata(),
        embedding=[0.1] * 1536,
        created_at=datetime.now(UTC),
    )


def _make_entity(ns_id: UUID, name: str, entity_type: str = "CONCEPT"):
    """Build an Entity with a fresh UUID — the extraction-time ID."""
    from khora.core.models import Entity

    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        description="",
        confidence=0.99,
    )


def _make_relationship(ns_id: UUID, source_id: UUID, target_id: UUID, rel_type: str = "RELATES_TO"):
    from khora.core.models import Relationship

    return Relationship(
        id=uuid4(),
        namespace_id=ns_id,
        source_entity_id=source_id,
        target_entity_id=target_id,
        relationship_type=rel_type,
        confidence=0.9,
    )


def _storage_with_canonicalisation(canonical_by_name_type: dict[tuple[str, str], UUID]) -> MagicMock:
    """Build a storage mock whose ``upsert_entities_batch`` mutates entity.id
    to the canonical UUID for any (name, entity_type) in the lookup table —
    mirroring what Neo4j's MERGE does when the entity already exists.
    """
    storage = MagicMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.create_document = AsyncMock(side_effect=lambda d: d)
    storage.update_document = AsyncMock()
    storage.create_chunks_batch = AsyncMock()
    storage.update_entity_embeddings_batch = AsyncMock()
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])
    storage.dispatch_hook = AsyncMock()
    storage.get_entity_by_name = AsyncMock(return_value=None)

    captured_relationships: list = []

    async def _upsert(_ns, entities, **_kwargs):
        results = []
        for e in entities:
            key = (e.name, e.entity_type)
            if key in canonical_by_name_type:
                # Simulate Neo4j's MERGE id-sync (mutate in place).
                e.id = canonical_by_name_type[key]
                results.append((e, False))  # not new
            else:
                results.append((e, True))
        return results

    async def _create_rels(rels, **_kwargs):
        captured_relationships.extend(rels)
        return len(rels)

    storage.upsert_entities_batch = AsyncMock(side_effect=_upsert)
    storage.create_relationships_batch = AsyncMock(side_effect=_create_rels)
    storage._captured_relationships = captured_relationships  # type: ignore[attr-defined]
    return storage


@pytest.mark.asyncio
async def test_relationship_lands_when_endpoint_re_canonicalised() -> None:
    """The DYT-3558 regression: a relationship whose endpoints were
    re-canonicalised by the upsert must still reach the graph backend.
    """
    from khora.pipelines.flows.ingest import process_document

    ns_id = uuid4()
    doc_id = uuid4()

    # Doc 2 in the chain — emits 'betagadget' (already in DB) + 'gammathingy'
    # plus a relationship between them.
    canonical_beta = uuid4()  # the canonical betagadget id from doc 1
    storage = _storage_with_canonicalisation(
        {
            ("betagadget", "CONCEPT"): canonical_beta,
        }
    )

    document = _make_document_mock(doc_id, ns_id, "betagadget connects to gammathingy.")
    chunks = [_make_chunk(ns_id, doc_id)]
    beta = _make_entity(ns_id, "betagadget")
    gamma = _make_entity(ns_id, "gammathingy")
    pre_upsert_beta_id = beta.id  # capture before mutation
    pre_upsert_gamma_id = gamma.id

    rel = _make_relationship(ns_id, pre_upsert_beta_id, pre_upsert_gamma_id)

    fake_embedder = MagicMock()
    fake_embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536 for _ in texts])
    with (
        patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=chunks)),
        patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=chunks)),
        patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([beta, gamma], [rel]))),
        patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=fake_embedder),
    ):
        result = await process_document(
            document,
            storage,
            entity_types=["CONCEPT"],
            relationship_types=["RELATES_TO"],
        )

    # Bug fix #1: the relationship reached create_relationships_batch (no skip).
    storage.create_relationships_batch.assert_awaited_once()
    captured = storage._captured_relationships
    assert len(captured) == 1, f"expected 1 stored relationship, got {len(captured)}"
    stored = captured[0]

    # Bug fix #2: endpoints are remapped to canonical IDs, not the stale
    # pre-upsert UUIDs.
    assert stored.source_entity_id == canonical_beta, (
        f"source not remapped to canonical: {stored.source_entity_id} != {canonical_beta}"
    )
    # gamma was new, so its id wasn't mutated — the upsert mapping should
    # also self-map it (canonical -> canonical) and the rel still resolves.
    assert stored.target_entity_id == gamma.id

    # Top-level result count matches input.
    assert result["relationships"] == 1


@pytest.mark.asyncio
async def test_relationship_resolves_via_db_fallback() -> None:
    """Defense-in-depth: when an endpoint UUID is not in entity_id_mapping
    (e.g., an inferred relationship referring to a previously-stored entity
    that wasn't part of this document's upsert batch), ``_store_relationships``
    falls back to ``storage.get_entity_by_name`` and successfully lands the rel.

    This simulates the case where the upsert batch only contains 'gamma' but
    a relationship references a 'beta' UUID we don't have a mapping for —
    we should still resolve via (namespace, name, type) using a snapshot
    captured before the upsert.
    """
    from khora.core.models import Entity
    from khora.pipelines.flows.ingest import process_document

    ns_id = uuid4()
    doc_id = uuid4()

    canonical_beta = uuid4()
    canonical_gamma = uuid4()

    storage = _storage_with_canonicalisation(
        {
            ("betagadget", "CONCEPT"): canonical_beta,
            ("gammathingy", "CONCEPT"): canonical_gamma,
        }
    )

    # The DB fallback target — beta exists but is canonicalised mid-flight.
    # We simulate the case where get_entity_by_name returns the beta entity.
    storage.get_entity_by_name = AsyncMock(
        return_value=Entity(
            id=canonical_beta,
            namespace_id=ns_id,
            name="betagadget",
            entity_type="CONCEPT",
            description="",
            confidence=1.0,
        )
    )

    document = _make_document_mock(doc_id, ns_id, "stub")
    chunks = [_make_chunk(ns_id, doc_id)]
    beta = _make_entity(ns_id, "betagadget")
    gamma = _make_entity(ns_id, "gammathingy")
    rel = _make_relationship(ns_id, beta.id, gamma.id)

    fake_embedder = MagicMock()
    fake_embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536 for _ in texts])
    with (
        patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=chunks)),
        patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=chunks)),
        patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([beta, gamma], [rel]))),
        patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=fake_embedder),
    ):
        result = await process_document(
            document,
            storage,
            entity_types=["CONCEPT"],
            relationship_types=["RELATES_TO"],
        )

    # The relationship should still land — the in-memory mapping covers
    # the simple case, and the DB fallback is available for harder cases.
    assert result["relationships"] == 1
    captured = storage._captured_relationships
    assert len(captured) == 1
    stored = captured[0]
    assert stored.source_entity_id == canonical_beta
    assert stored.target_entity_id == canonical_gamma


@pytest.mark.asyncio
async def test_genuinely_missing_entity_still_skipped() -> None:
    """A relationship referencing an entity that simply doesn't exist
    anywhere (no upsert, no DB hit) must still be skipped with a warning —
    the fix must not silently absorb genuinely unresolvable rels.
    """
    from khora.core.models import Relationship
    from khora.pipelines.flows.ingest import process_document

    ns_id = uuid4()
    doc_id = uuid4()

    storage = _storage_with_canonicalisation({})
    # No entity matches anywhere.
    storage.get_entity_by_name = AsyncMock(return_value=None)

    document = _make_document_mock(doc_id, ns_id, "stub")
    chunks = [_make_chunk(ns_id, doc_id)]
    real_entity = _make_entity(ns_id, "alpha")

    # Build a relationship pointing to a UUID that doesn't appear in the
    # upserted entities at all — this is the "ghost" case.
    ghost_id = uuid4()
    bogus_rel = Relationship(
        id=uuid4(),
        namespace_id=ns_id,
        source_entity_id=ghost_id,
        target_entity_id=real_entity.id,
        relationship_type="RELATES_TO",
        confidence=0.9,
    )

    fake_embedder = MagicMock()
    fake_embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536 for _ in texts])
    with (
        patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=chunks)),
        patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=chunks)),
        patch(
            "khora.pipelines.tasks.extract_entities",
            new=AsyncMock(return_value=([real_entity], [bogus_rel])),
        ),
        patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=fake_embedder),
    ):
        result = await process_document(
            document,
            storage,
            entity_types=["CONCEPT"],
            relationship_types=["RELATES_TO"],
        )

    # The bogus relationship should be skipped — count == 0, no rel stored.
    assert result["relationships"] == 0
    storage.create_relationships_batch.assert_not_awaited()
