"""Real PG+Neo4j regression for #1373: PPR seed-anchored graph augmentation.

Pins the actual bug the mock-only unit tests cannot catch. The VectorCypher
PPR channel builds its graph from a query-independent global slice
(``list_entities(limit=_MAX_ENTITIES_FOR_PPR)`` ordered ``BY name``,
``list_relationships(limit=_MAX_RELATIONSHIPS_FOR_PPR)``). On a namespace
larger than the slice cap, the alphabetical slice excludes the seed entities
the query resolved to, so ``build_personalization_vector`` sums to zero and the
graph channel silently returns nothing.

This test reproduces the exact mechanism of ``probe_ppr_global_slice.py`` from
the issue against a live PG+Neo4j stack, but with the cap monkeypatched low so
only a handful of fillers are needed (no 5200-node write):

1. Write a document + 2 chunks + 2 seed entities (``marie curie`` / ``radium``,
   each with a 1536-dim embedding + source_chunk_ids) + 1 relationship to the
   real coordinator (pgvector + Neo4j).
2. Pad Neo4j with ``cap``-many filler entities named ``aaa_filler_*`` so they
   sort *before* the seeds and fill the entire alphabetical slice.
3. Prove the bug: building the PPR graph from the global slice alone gives a
   zero personalization vector (0/2 seeds survive).
4. Prove the fix: ``ppr_retrieve_chunks`` (which now augments the at-cap slice
   with the seeds + their 1-hop neighborhood) keeps both seeds, returns a
   non-zero PR mass for them, returns chunks, and records no degradation.

Gated by ``NEO4J_INTEGRATION_TEST=1`` + a reachable Postgres, matching the
other live-stack integration suites in this directory.

How to run locally::

    make dev  # postgres + neo4j via docker compose
    KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora \
    KHORA_NEO4J_URL=bolt://neo4j:pleaseletmein@localhost:7688 \
    NEO4J_INTEGRATION_TEST=1 \
    uv run pytest tests/integration/test_ppr_seed_anchored_graph_integration.py -v
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest

from khora import Khora
from khora.config import KhoraConfig
from khora.core.diagnostics import Degradation
from khora.core.models import Chunk, Document, Entity, Relationship
from khora.engines.vectorcypher import ppr_retrieval
from khora.engines.vectorcypher.ppr_retrieval import (
    build_personalization_vector,
    build_ppr_graph,
    ppr_retrieve_chunks,
)
from tests.integration.conftest import _pg_reachable

_PG_EMBED_DIM = 1536
# Small cap so the alphabetical slice fills with a handful of fillers instead of
# the production 5000. The seed-exclusion mechanism is identical.
_TEST_CAP = 6
_N_FILLERS = _TEST_CAP + 2

_SKIP = pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST") or not _pg_reachable(),
    reason="set NEO4J_INTEGRATION_TEST=1 and run `make dev` (PG+Neo4j) to exercise the #1373 PPR regression",
)


def _vec(seed: str) -> list[float]:
    """Deterministic non-zero unit-ish vector; pgvector requires dim 1536."""
    h = abs(hash(seed))
    base = [((h >> (i % 31)) & 0xFF) / 255.0 + 0.01 for i in range(_PG_EMBED_DIM)]
    return base


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    database_url = os.environ.get("KHORA_DATABASE_URL", "postgresql+asyncpg://khora:khora@localhost:5434/khora")
    neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    config = KhoraConfig(database_url=database_url, neo4j_url=neo4j_url)
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")
    config.llm.embedding_dimension = _PG_EMBED_DIM
    config.storage.embedding_dimension = _PG_EMBED_DIM
    instance = Khora(config, run_migrations=False)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.disconnect()


async def _pad_neo4j_fillers(driver, ns_row: UUID, n: int) -> None:
    """Insert ``n`` filler :Entity nodes named ``aaa_filler_*`` (sort first)."""
    async with driver.session() as session:
        rows = [{"id": str(uuid4()), "name": f"aaa_filler_{i:05d}"} for i in range(n)]
        await session.run(
            "UNWIND $rows AS r CREATE (e:Entity {id: r.id, namespace_id: $ns, name: r.name, entity_type: 'FILLER'})",
            rows=rows,
            ns=str(ns_row),
        )


@_SKIP
@pytest.mark.integration
@pytest.mark.asyncio
async def test_seed_anchored_ppr_survives_large_namespace(monkeypatch: pytest.MonkeyPatch, kb: Khora) -> None:
    monkeypatch.setattr(ppr_retrieval, "_MAX_ENTITIES_FOR_PPR", _TEST_CAP)

    storage = kb._engine._retriever._storage  # type: ignore[union-attr,attr-defined]
    ns_public = (await kb.create_namespace()).namespace_id
    ns_row = await storage.resolve_namespace(ns_public)

    # --- Document + 2 chunks (real pgvector rows) ---------------------------
    doc = Document(namespace_id=ns_row, content="Marie Curie discovered radium.", title="curie")
    await storage.create_document(doc)
    chunk_a = Chunk(
        namespace_id=ns_row,
        document_id=doc.id,
        content="Marie Curie discovered radium and polonium in Paris.",
        embedding=_vec("chunk_a"),
        chunk_index=0,
    )
    chunk_b = Chunk(
        namespace_id=ns_row,
        document_id=doc.id,
        content="Radium is a radioactive element isolated by the Curies.",
        embedding=_vec("chunk_b"),
        chunk_index=1,
    )
    await storage.create_chunks_batch([chunk_a, chunk_b])

    # --- 2 seed entities (named to sort AFTER the fillers) + 1 relationship --
    seed_curie = Entity(
        namespace_id=ns_row,
        name="marie curie",
        entity_type="PERSON",
        source_chunk_ids=[chunk_a.id],
        embedding=_vec("marie curie"),
    )
    seed_radium = Entity(
        namespace_id=ns_row,
        name="radium",
        entity_type="CONCEPT",
        source_chunk_ids=[chunk_a.id, chunk_b.id],
        embedding=_vec("radium"),
    )
    await storage.create_entity(seed_curie)
    await storage.create_entity(seed_radium)
    await storage.create_relationship(
        Relationship(
            namespace_id=ns_row,
            source_entity_id=seed_curie.id,
            target_entity_id=seed_radium.id,
            relationship_type="DISCOVERED",
            weight=1.0,
        )
    )

    # --- Pad Neo4j so the alphabetical limit slice is all fillers ----------
    driver = storage._graph._driver  # type: ignore[union-attr,attr-defined]
    await _pad_neo4j_fillers(driver, ns_row, _N_FILLERS)

    entry_entities = [(seed_curie.id, 1.0), (seed_radium.id, 0.9)]

    # --- (1) Prove the bug: global slice alone excludes the seeds ----------
    slice_entities = await storage.list_entities(ns_row, limit=ppr_retrieval._MAX_ENTITIES_FOR_PPR)
    assert len(slice_entities) >= _TEST_CAP, "slice did not hit the cap; padding failed"
    slice_only_graph = build_ppr_graph(slice_entities, [])
    bug_personalization = build_personalization_vector(entry_entities, slice_only_graph.entity_id_to_idx)
    assert sum(bug_personalization) == 0.0, (
        "expected the global slice to exclude both seeds (the #1373 bug) — "
        f"got personalization sum {sum(bug_personalization)}"
    )

    # --- (2) Prove the fix: seed-anchored augmentation keeps the seeds -----
    degradations: list[Degradation] = []
    results, entity_scores = await ppr_retrieve_chunks(
        storage=storage,
        namespace_id=ns_row,
        entry_entities=entry_entities,
        damping=0.85,
        max_iter=50,
        tol=1e-5,
        top_entities=30,
        limit=10,
        out_degradations=degradations,
    )

    # The invariant that matters: every resolvable seed survives into the graph.
    assert seed_curie.id in entity_scores
    assert seed_radium.id in entity_scores
    assert entity_scores[seed_curie.id] > 0.0
    assert entity_scores[seed_radium.id] > 0.0
    # Graph channel now returns chunks instead of silently nothing.
    assert results, "seed-anchored PPR returned no chunks on a >cap namespace (the #1373 regression)"
    returned_chunk_ids = {cid for cid, _, _ in results}
    assert returned_chunk_ids & {chunk_a.id, chunk_b.id}
    # Happy path: no degradation recorded.
    assert degradations == []
