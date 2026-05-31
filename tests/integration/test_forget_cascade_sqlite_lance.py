"""Integration test for the #923 vector-anchored forget cascade on sqlite_lance.

Pre-#923 ``_cascade_forget_extraction`` early-returned whenever the graph
backend lacked the Neo4j-only ``fetch_document_extraction_state``, so on the
graph-less / non-Neo4j stacks (sqlite_lance, SurrealDB, Memgraph, Neptune,
AGE) the entity/relationship cleanup was a silent no-op: ``forget()`` dropped
the document + chunks but left orphan entities behind and reported success.

On sqlite_lance entities and relationships live in the sqlite_lance graph
adapter's SQLite tables. This test seeds two documents that SHARE one entity
and each carry one unique entity, then drives the real fixed cascade against
the real sqlite_lance coordinator and asserts refcounting:

- doc A's unique entity is GONE (orphan: sole source was doc A).
- the SHARED entity SURVIVES, with doc A stripped from source_document_ids.
- doc B's unique entity is untouched.
- the cascade reports NO degradation (it actually ran, not a silent no-op).

Runs without Docker - sqlite_lance is fully embedded.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from khora.core.models import Entity, MemoryNamespace, Relationship
from khora.engines._forget_cascade import cascade_forget_extraction

from ._sqlite_lance_fixtures import build_sqlite_lance_coordinator


@pytest.mark.integration
@pytest.mark.asyncio
async def test_forget_cascade_cleans_entities_on_sqlite_lance(tmp_path: Path) -> None:
    coord = await build_sqlite_lance_coordinator(tmp_path)
    try:
        namespace_id = uuid4()
        # entities.namespace_id FKs memory_namespaces.id, so the namespace row
        # must exist with id == namespace_id.
        await coord.create_namespace(MemoryNamespace(id=namespace_id, namespace_id=namespace_id))
        doc_a = uuid4()
        doc_b = uuid4()

        shared = Entity(namespace_id=namespace_id, name="Shared", entity_type="CONCEPT")
        shared.source_document_ids = [doc_a, doc_b]
        unique_a = Entity(namespace_id=namespace_id, name="UniqueA", entity_type="CONCEPT")
        unique_a.source_document_ids = [doc_a]
        unique_b = Entity(namespace_id=namespace_id, name="UniqueB", entity_type="CONCEPT")
        unique_b.source_document_ids = [doc_b]

        graph = coord._graph
        assert graph is not None
        for ent in (shared, unique_a, unique_b):
            await graph.create_entity(ent)

        # An edge sourced solely by doc A must be deleted; an edge shared by
        # both docs must survive with doc A stripped.
        orphan_edge = Relationship(
            namespace_id=namespace_id,
            source_entity_id=shared.id,
            target_entity_id=unique_a.id,
            relationship_type="RELATES_TO",
        )
        orphan_edge.source_document_ids = [doc_a]
        shared_edge = Relationship(
            namespace_id=namespace_id,
            source_entity_id=shared.id,
            target_entity_id=unique_b.id,
            relationship_type="RELATES_TO",
        )
        shared_edge.source_document_ids = [doc_a, doc_b]
        await graph.create_relationships_batch([orphan_edge, shared_edge])

        # ----- Act: forget doc A via the real fixed cascade -----
        degradations = await cascade_forget_extraction(
            graph=coord._graph,
            vector=coord._vector,
            document_id=doc_a,
            namespace_id=namespace_id,
            engine="test",
        )

        # The cascade actually ran (no silent no-op / degradation).
        assert degradations == []

        names = {e.name: e for e in await graph.list_entities(namespace_id, limit=1000)}

        # (a) doc A's unique entity is gone.
        assert "UniqueA" not in names, f"orphan entity UniqueA survived forget(doc_a); got {set(names)}"

        # (b) the shared entity survives with doc A stripped from its sources.
        assert "Shared" in names
        assert doc_a not in names["Shared"].source_document_ids
        assert doc_b in names["Shared"].source_document_ids

        # (c) doc B's unique entity is untouched.
        assert "UniqueB" in names
        assert names["UniqueB"].source_document_ids == [doc_b]

        # Relationships: orphan edge gone, shared edge survives with doc A stripped.
        rels = await graph.list_relationships(namespace_id, limit=1000)
        by_target = {r.target_entity_id: r for r in rels}
        assert unique_a.id not in by_target, "orphan relationship survived forget(doc_a)"
        assert unique_b.id in by_target, "shared relationship was wrongly deleted"
        survivor = by_target[unique_b.id]
        assert doc_a not in survivor.source_document_ids
        assert doc_b in survivor.source_document_ids
    finally:
        await coord.disconnect()
