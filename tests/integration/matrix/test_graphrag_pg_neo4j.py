"""GraphRAG PostgreSQL+Neo4j integration tests (DYT-3545).

GraphRAG is khora's default production engine but had **zero dedicated
integration tests** prior to this file (per the DYT-3545 audit, §5).
These tests wire up ``MemoryLake(engine="graphrag")`` against the real
production stack: ``khora-postgres`` (compose.yaml, port 5434) +
``khora-neo4j`` (compose.yaml, bolt port 7688), with stubbed LLM calls
to keep the suite hermetic and offline.

Embedded-stack GraphRAG coverage (SQLite + LanceDB + Kuzu) is
deliberately deferred to a follow-up PR — that path is waiting on the
R1/R2 fixes called out in the audit.

How LLM calls are mocked:
* ``LLMEntityExtractor.extract_multi`` is replaced with a registry-based
  stub that emits a fixed entity list per content marker. The chronicle
  test pattern carries over verbatim.
* ``LiteLLMEmbedder.embed_batch`` and ``embed`` return deterministic
  unit vectors of dimension 1536 (matches the ``chunks.embedding``
  ``Vector(1536)`` column hard-coded in migration 000).

How to run locally::

    make dev    # postgres on :5434, neo4j on bolt :7688
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
    KHORA_NEO4J_URL=bolt://neo4j:pleaseletmein@localhost:7688 \\
        uv run pytest tests/integration/matrix/test_graphrag_pg_neo4j.py \\
            -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from khora.config import KhoraConfig
from khora.db.session import run_migrations
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.skills import ExpertiseConfig
from khora.memory_lake import MemoryLake

EMBED_DIM = 1536  # matches the chunks.embedding Vector(1536) column from migrations

# Hardcoded to the values from this repo's compose.yaml. We intentionally do
# NOT read KHORA_DATABASE_URL / KHORA_NEO4J_URL from the surrounding shell —
# developers very often have these env vars pointing at other projects' DBs
# (genesis, anima, etc.) and an integration test that drops + recreates the
# public schema must never touch an unrelated database. To override the
# target DBs, set GRAPHRAG_INTEGRATION_DATABASE_URL / GRAPHRAG_INTEGRATION_NEO4J_URL
# explicitly when running this file.
DATABASE_URL = os.environ.get(
    "GRAPHRAG_INTEGRATION_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

NEO4J_URL = os.environ.get(
    "GRAPHRAG_INTEGRATION_NEO4J_URL",
    "bolt://neo4j:pleaseletmein@localhost:7688",
)


# ---------------------------------------------------------------------------
# Fixtures: skip-if-no-stack, run-migrations-once, extraction stub
# ---------------------------------------------------------------------------


def _tcp_reachable(url: str, default_port: int) -> bool:
    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _tcp_reachable(DATABASE_URL, 5432),
        reason="PostgreSQL not reachable (run `make dev` first)",
    ),
    pytest.mark.skipif(
        not _tcp_reachable(NEO4J_URL, 7687),
        reason="Neo4j not reachable (run `make dev` first)",
    ),
]


# Module-level extraction registry. ``_plan_extraction`` stages the
# ExtractionResult to return for documents containing ``marker``.
_EXTRACTION_REGISTRY: dict[str, ExtractionResult] = {}


def _plan_extraction(
    marker: str,
    entities: list[tuple[str, str]],
    relationships: list[tuple[str, str, str]] | None = None,
) -> None:
    """Stage an extraction result for texts containing ``marker``."""
    _EXTRACTION_REGISTRY[marker] = ExtractionResult(
        entities=[ExtractedEntity(name=n, entity_type=t, confidence=0.99) for n, t in entities],
        relationships=[
            ExtractedRelationship(
                source_entity=s,
                target_entity=t,
                relationship_type=rt,
                confidence=0.99,
            )
            for s, t, rt in (relationships or [])
        ],
    )


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    """Registry-based stub for ``LLMEntityExtractor.extract_multi``."""
    out: list[ExtractionResult] = []
    for text_in in texts:
        matched = next(
            (result for marker, result in _EXTRACTION_REGISTRY.items() if marker in text_in),
            None,
        )
        out.append(matched if matched is not None else ExtractionResult())
    return out


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    """Deterministic unit vector embedder stub (length EMBED_DIM)."""
    unit = [1.0] + [0.0] * (EMBED_DIM - 1)
    return [unit[:] for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    """Single-string variant for query-time ``embedder.embed(query)``."""
    return [1.0] + [0.0] * (EMBED_DIM - 1)


@pytest.fixture(scope="module")
async def _stack_ready() -> None:
    """Reset and migrate PG once for the module, and wipe Neo4j.

    PG reset workaround mirrors ``test_chronicle_pg.py`` verbatim — needed
    when an earlier integration run left ``khora_alembic_version`` with
    the default ``VARCHAR(32)`` (DYT-3546). Once #471's column-widen is
    rolled out everywhere, this can be simplified.

    Neo4j is wiped in the same module-level fixture so that prior runs
    (or other integration files) don't leak entity nodes that would
    confuse the namespace-isolation and 2-hop-traversal assertions.
    """
    # --- PostgreSQL: reset schema, ensure pgvector, prep alembic table ---
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.begin() as conn:
            r = await conn.execute(
                text("SELECT typname FROM pg_type WHERE typnamespace = 'public'::regnamespace AND typtype = 'e'")
            )
            for (typname,) in r.fetchall():
                await conn.execute(text(f"DROP TYPE IF EXISTS public.{typname} CASCADE"))
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    "CREATE TABLE khora_alembic_version ("
                    "  version_num VARCHAR(64) NOT NULL,"
                    "  CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)"
                    ")"
                )
            )
    finally:
        await eng.dispose()

    result = await run_migrations(DATABASE_URL)
    assert result.success, f"Migrations failed: {result.error}"

    # --- Neo4j: wipe the entire graph ---
    from neo4j import AsyncGraphDatabase

    parsed = urlparse(NEO4J_URL)
    user = parsed.username or "neo4j"
    password = parsed.password or "pleaseletmein"
    bolt_url = f"bolt://{parsed.hostname}:{parsed.port or 7687}"
    driver = AsyncGraphDatabase.driver(bolt_url, auth=(user, password))
    try:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    finally:
        await driver.close()


def _expertise() -> ExpertiseConfig:
    """Default expertise config for GraphRAG tests.

    GraphRAG (unlike Chronicle) has no per-chunk event/fact extraction
    pass, so we just return a named ExpertiseConfig — entity/relationship
    extraction itself is what we want to exercise here.
    """
    return ExpertiseConfig(name="graphrag-pg-neo4j-integ")


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the extractor + embedder so no real LLM is called."""
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


@pytest.fixture
async def lake(_stack_ready: None, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[MemoryLake]:
    """Per-test GraphRAG MemoryLake bound to live PG + Neo4j.

    Function-scoped because the storage coordinator caches engine pools
    by URL; sharing across tests was tripping the autouse monkeypatch
    reset (the engine instance wires the embedder reference at
    ``connect()`` time).

    KhoraConfig is a Pydantic-settings model — process env vars override
    constructor kwargs. Developers running these tests typically have
    KHORA_DATABASE_URL / KHORA_NEO4J_URL exported pointing at *other*
    project databases (genesis, etc.). We monkeypatch those env vars to
    our test target before instantiating the config so the test never
    accidentally clobbers an unrelated dev database.
    """
    monkeypatch.setenv("KHORA_DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("KHORA_NEO4J_URL", NEO4J_URL)
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=NEO4J_URL)
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    # Single-chunk documents keep the test deterministic.
    config.pipelines.chunk_size = 1024
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False

    lake = MemoryLake(config, engine="graphrag", run_migrations=False)
    await lake.connect()
    try:
        yield lake
    finally:
        await lake.disconnect()


@pytest.fixture
async def namespace_id(lake: MemoryLake) -> UUID:
    ns = await lake.create_namespace()
    return ns.namespace_id


async def _remember(
    lake: MemoryLake,
    *,
    namespace_id: UUID,
    content: str,
    title: str = "",
) -> Any:
    return await lake.remember(
        content=content,
        namespace=namespace_id,
        title=title,
        entity_types=["PERSON", "CONCEPT", "EVENT", "LOCATION"],
        relationship_types=["KNOWS", "RELATES_TO", "PART_OF", "ATTENDED"],
        expertise=_expertise(),
    )


async def _neo4j_query(query: str, **params: Any) -> list[dict[str, Any]]:
    """Run a Cypher query directly against Neo4j and return all records."""
    from neo4j import AsyncGraphDatabase

    parsed = urlparse(NEO4J_URL)
    user = parsed.username or "neo4j"
    password = parsed.password or "pleaseletmein"
    bolt_url = f"bolt://{parsed.hostname}:{parsed.port or 7687}"
    driver = AsyncGraphDatabase.driver(bolt_url, auth=(user, password))
    try:
        async with driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return records
    finally:
        await driver.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_graphrag_remember_recall_roundtrip(lake: MemoryLake, namespace_id: UUID) -> None:
    """Ingest 3 docs, recall, assert ingested doc text appears in context."""
    contents = [
        "Alice met Bob at the Python conference in Berlin on March 15th.",
        "Carol presented her research on graph databases at the same event.",
        "Dan organized the after-party that lasted until midnight.",
    ]
    for c in contents:
        await _remember(lake, namespace_id=namespace_id, content=c)

    result = await lake.recall("Python conference Berlin", namespace=namespace_id, limit=10)

    assert len(result.chunks) >= 1, "expected at least one chunk back"
    assert "Python conference" in result.context_text


async def test_graphrag_namespace_isolation(lake: MemoryLake) -> None:
    """Two namespaces, queries don't cross-bleed."""
    ns_a = (await lake.create_namespace()).namespace_id
    ns_b = (await lake.create_namespace()).namespace_id

    await _remember(lake, namespace_id=ns_a, content="alpha document about kangaroos")
    await _remember(lake, namespace_id=ns_b, content="bravo document about penguins")

    result_a = await lake.recall("animals", namespace=ns_a, limit=10)
    result_b = await lake.recall("animals", namespace=ns_b, limit=10)

    a_text = " ".join(c.content for c, _ in result_a.chunks)
    b_text = " ".join(c.content for c, _ in result_b.chunks)

    assert "kangaroos" in a_text
    assert "penguins" not in a_text, "namespace_b content leaked into namespace_a"
    assert "penguins" in b_text
    assert "kangaroos" not in b_text, "namespace_a content leaked into namespace_b"


async def test_graphrag_entity_extraction_via_neo4j(lake: MemoryLake) -> None:
    """Ingest a doc with a known entity, assert it lands as a node in Neo4j.

    This exercises the full extraction → entity-resolution → graph-write
    path. We register the LLM extractor stub to emit a deterministic
    ``Marie Curie`` PERSON entity, then verify via direct Cypher that
    the node is present in Neo4j with the correct namespace and type.

    Notes:
    * khora's ingest pipeline normalizes entity names to lowercase
      (see ``normalize_entity_names_batch`` in ``_accel.py``), so we
      look up by ``"marie curie"`` even though the LLM emits
      ``"Marie Curie"``.
    * MemoryNamespace has dual IDs (ADR-024): ``namespace_id`` is the
      stable public identifier passed across the API surface;
      ``id`` is the row-level FK that backends use internally. The
      entity rows in Neo4j carry the row-level ``id`` in their
      ``namespace_id`` property, so a direct Cypher lookup must match
      against ``ns.id``, not ``ns.namespace_id``.
    """
    ns = await lake.create_namespace()
    _plan_extraction(
        "Marie Curie",
        entities=[("Marie Curie", "PERSON"), ("radium", "CONCEPT")],
    )
    await _remember(
        lake,
        namespace_id=ns.namespace_id,
        content="Marie Curie discovered radium in 1898 while working in Paris.",
    )

    records = await _neo4j_query(
        """
        MATCH (e:Entity {namespace_id: $ns, name: $name})
        RETURN e.name AS name, e.entity_type AS entity_type
        """,
        ns=str(ns.id),
        name="marie curie",
    )
    assert records, "marie curie entity not found in Neo4j after ingest"
    assert records[0]["entity_type"] == "PERSON"


async def test_graphrag_temporal_filter(lake: MemoryLake, namespace_id: UUID) -> None:
    """Backdate one doc 20 days, leave another at 5 days, query last 7 days.

    Asserts the ``start_time`` parameter on ``recall()`` excludes the
    20-day-old chunk from the fused result. Backdating is done via
    direct SQL UPDATE of ``chunks.created_at`` (the column GraphRAG's
    vector channel pushes ``created_after`` against — see
    ``query/engine.py::_vector_search``).
    """
    r_recent = await _remember(lake, namespace_id=namespace_id, content="recent Falcon launch report.")
    r_old = await _remember(lake, namespace_id=namespace_id, content="old Falcon launch report.")

    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.begin() as conn:
            await conn.execute(
                text("UPDATE chunks SET created_at = NOW() - INTERVAL '20 days' WHERE document_id = :doc_id"),
                {"doc_id": r_old.document_id},
            )
            await conn.execute(
                text("UPDATE chunks SET created_at = NOW() - INTERVAL '5 days' WHERE document_id = :doc_id"),
                {"doc_id": r_recent.document_id},
            )
    finally:
        await eng.dispose()

    seven_days_ago = datetime.now(UTC) - timedelta(days=7)
    result = await lake.recall(
        "Falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=seven_days_ago,
    )
    returned_doc_ids = {c.document_id for c, _ in result.chunks}
    assert r_recent.document_id in returned_doc_ids, "recent doc missing from result"
    assert r_old.document_id not in returned_doc_ids, (
        "20-day-old doc returned despite start_time=7 days ago — temporal filter not applied"
    )


async def test_graphrag_recall_metadata(lake: MemoryLake, namespace_id: UUID) -> None:
    """RecallResult.metadata carries the expected GraphRAG keys."""
    await _remember(lake, namespace_id=namespace_id, content="A simple sentence about apples.")

    result = await lake.recall("apples", namespace=namespace_id, limit=5)

    md = result.metadata
    # query/mode/namespace_id are unconditionally populated by the
    # HybridQueryEngine; search_methods + graph_traversal + temporal +
    # metrics are added near the end of HybridQueryEngine.query().
    expected = {"query", "mode", "namespace_id", "search_methods", "graph_traversal", "temporal", "metrics"}
    missing = expected - md.keys()
    assert not missing, f"missing GraphRAG metadata keys: {missing} (got: {sorted(md.keys())})"

    # search_methods reports per-channel chunk/entity counts under by_method
    sm = md["search_methods"]
    assert "by_method" in sm, f"search_methods.by_method missing: {sm.keys()}"
    by_method = sm["by_method"]
    # GraphRAG fans out to vector + graph + keyword channels in HYBRID mode
    for channel in ("vector", "graph", "keyword"):
        assert channel in by_method, f"by_method.{channel} missing: {by_method.keys()}"
    # graph_traversal reports the configured depth
    assert "neighborhood_depth" in md["graph_traversal"]


async def test_graphrag_concurrent_remember(lake: MemoryLake, namespace_id: UUID) -> None:
    """5 concurrent ingests in one namespace, no integrity errors."""
    contents = [f"document number {i} mentions widget-{i}" for i in range(5)]
    results = await asyncio.gather(
        *(_remember(lake, namespace_id=namespace_id, content=c) for c in contents),
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"concurrent remember raised: {errors}"

    doc_ids = {r.document_id for r in results}  # type: ignore[union-attr]
    assert len(doc_ids) == 5, f"expected 5 distinct documents, got {doc_ids}"

    result = await lake.recall("widget", namespace=namespace_id, limit=20)
    contents_returned = {c.content for c, _ in result.chunks}
    assert len(contents_returned) >= 5


async def test_graphrag_recall_empty_namespace(lake: MemoryLake, namespace_id: UUID) -> None:
    """Recall against a freshly-created namespace returns an empty result cleanly."""
    result = await lake.recall("anything at all", namespace=namespace_id, limit=10)

    assert result.chunks == [], f"expected no chunks, got {len(result.chunks)}"
    assert result.entities == [], f"expected no entities, got {len(result.entities)}"
    # Even on empty corpora, metadata should still be well-formed.
    assert result.metadata.get("mode") in {"HYBRID", "VECTOR", "GRAPH", "ALL"}


async def test_graphrag_two_hop_traversal(lake: MemoryLake) -> None:
    """Ingest A→B→C, query about A, assert C surfaces via 2-hop graph traversal.

    Stages a 3-document chain where extraction emits explicit relationships:
        doc1: alphawidget RELATES_TO betagadget
        doc2: betagadget  RELATES_TO gammathingy
        doc3: gammathingy is the terminal node

    GraphRAG's HybridQueryEngine is configured to traverse the graph by
    ``max_graph_depth`` (default 2), so a query starting from
    ``alphawidget`` should reach ``gammathingy`` via the 2-hop path.

    Caveat (surfaced loudly via xfail, *not* fixed in this PR):
    when ``betagadget`` is upserted a second time (in doc 2), its
    canonical ID is reassigned by the entity-resolver. The relationship
    extracted from doc 2 still references doc 2's freshly-minted UUIDs,
    which are no longer in ``entity_id_mapping`` after the upsert
    canonicalises them. ``_store_relationships`` then skips the
    relationship with a "missing entity mappings" warning, so the
    ``betagadget→gammathingy`` edge never lands in Neo4j and the
    2-hop traversal can never succeed. Filed as **DYT-3558**.

    Same dual-ID consideration as ``test_graphrag_entity_extraction_via_neo4j``:
    direct Cypher uses ``ns.id`` while ``lake.recall(namespace=...)``
    takes the stable ``ns.namespace_id``.
    """
    ns = await lake.create_namespace()
    _plan_extraction(
        "alphawidget",
        entities=[("alphawidget", "CONCEPT"), ("betagadget", "CONCEPT")],
        relationships=[("alphawidget", "betagadget", "RELATES_TO")],
    )
    _plan_extraction(
        "betagadget connects",
        entities=[("betagadget", "CONCEPT"), ("gammathingy", "CONCEPT")],
        relationships=[("betagadget", "gammathingy", "RELATES_TO")],
    )
    _plan_extraction(
        "gammathingy is",
        entities=[("gammathingy", "CONCEPT")],
    )

    await _remember(lake, namespace_id=ns.namespace_id, content="alphawidget pairs with betagadget.")
    await _remember(lake, namespace_id=ns.namespace_id, content="betagadget connects to gammathingy.")
    await _remember(lake, namespace_id=ns.namespace_id, content="gammathingy is the terminal node.")

    # Inspect Neo4j directly to surface the gap loudly.
    rel_records = await _neo4j_query(
        """
        MATCH (a:Entity {namespace_id: $ns})-[r:RELATES_TO]->(b:Entity {namespace_id: $ns})
        RETURN a.name AS source, b.name AS target
        """,
        ns=str(ns.id),
    )
    rel_pairs = {(r["source"], r["target"]) for r in rel_records}
    # First hop always lands — first-time upserts hit the happy path.
    assert ("alphawidget", "betagadget") in rel_pairs, f"alpha→beta relationship missing in Neo4j: {rel_pairs}"

    # Second hop reveals the DYT-3558 gap. xfail captures the regression
    # without failing the suite, and writes the diagnostic into the
    # output so it's visible in CI logs.
    if ("betagadget", "gammathingy") not in rel_pairs:
        pytest.xfail(
            "DYT-3558: GraphRAG drops a relationship when it references an "
            "entity that gets re-canonicalised by the entity-resolver. "
            "Pipeline log shows: '_store_relationships ... skipped due to "
            "missing entity mappings'. "
            f"Neo4j has only: {sorted(rel_pairs)}"
        )

    # If we get here the bug is fixed — assert the 2-hop traversal then
    # succeeds end-to-end via lake.recall.
    result = await lake.recall("alphawidget", namespace=ns.namespace_id, limit=10)
    entity_names = {e.name for e, _ in result.entities}
    assert "gammathingy" in entity_names, f"2-hop traversal did not surface gammathingy: {sorted(entity_names)}"
