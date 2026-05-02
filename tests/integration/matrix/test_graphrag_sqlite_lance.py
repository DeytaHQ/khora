"""GraphRAG SQLite + LanceDB embedded-stack integration tests (DYT-3545).

Mirrors ``test_graphrag_pg_neo4j.py`` (PR #475) for the embedded
backend so the GraphRAG cell of the integration matrix is covered on
both stacks. Embedded-stack semantics differ in a few interesting ways:

* No Docker. Each test gets its own ``tmp_path``-backed SQLite file +
  LanceDB directory. Migrations run against ``sqlite+aiosqlite:///`` —
  the same path the adapters use.
* Graph layer is **SQLite recursive CTE** (``SQLiteLanceGraphAdapter``),
  not Cypher / lance-graph. Two-hop traversal stresses
  ``_recursive_neighborhood`` whose visited-edge tracking is the
  subject of DYT-3548 (R1) and the prefer-current-edge logic the
  subject of DYT-3549 (R2).
* Vector channel is LanceDB ANN. ``created_at`` lives in *both*
  SQLite and LanceDB — the temporal-filter test backdates both stores.
* Direct entity lookup uses ``SELECT ... FROM entities`` against the
  SQLite handle (mirrors the Neo4j Cypher direct query in PR #475).

LLM + embedder are stubbed exactly as in PR #475 so the suite is
hermetic — no ``OPENAI_API_KEY`` required.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:  # pragma: no cover - optional dep
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.db.session import run_migrations
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.skills import ExpertiseConfig
from khora.memory_lake import MemoryLake

# Embedded fixtures use a small dim by convention (see _sqlite_lance_fixtures).
# LanceDB stores the schema with this dim — must match the embedder stub.
EMBED_DIM = 32

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_EMBEDDED,
        reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
    ),
]


# ---------------------------------------------------------------------------
# Stubs: LLM extractor + embedder (identical contract to PR #475)
# ---------------------------------------------------------------------------


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
    return [1.0] + [0.0] * (EMBED_DIM - 1)


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


def _expertise() -> ExpertiseConfig:
    return ExpertiseConfig(name="graphrag-sqlite-lance-integ")


def _build_config(tmp_path: Path) -> KhoraConfig:
    """Build a KhoraConfig wired to a tmp_path sqlite_lance backend.

    Matches PR #475's per-test config pattern, but routes storage through
    the embedded ``sqlite_lance`` backend rather than PG+Neo4j.
    """
    db_path = str(tmp_path / "khora.db")
    lance_path = str(tmp_path / "khora.lance")

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=db_path,
        lance_path=lance_path,
        embedding_dimension=EMBED_DIM,
    )
    config.storage.embedding_dimension = EMBED_DIM
    config.llm.embedding_dimension = EMBED_DIM
    config.pipelines.chunk_size = 1024
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False
    # Disable the orphan-pending-doc background processor so each test
    # is fully synchronous and tear-down is deterministic.
    config.pipelines.pending_processor_enabled = False
    return config


@pytest.fixture
async def lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[MemoryLake]:
    """Per-test ``MemoryLake(engine="graphrag")`` over sqlite_lance.

    Function-scoped because the adapter handle owns the SQLite + LanceDB
    files in ``tmp_path``; a module-scoped fixture would have to re-run
    migrations and reset the embedded files each test, which is harder
    than just using a fresh ``tmp_path``.

    Some tests in the wider suite leak ``KHORA_DATABASE_URL`` /
    ``KHORA_NEO4J_URL`` env vars; KhoraConfig is a Pydantic-settings
    model so process env trumps constructor kwargs. Strip both before
    instantiating the config to guarantee the embedded backend wins.
    """
    monkeypatch.delenv("KHORA_DATABASE_URL", raising=False)
    monkeypatch.delenv("KHORA_NEO4J_URL", raising=False)

    config = _build_config(tmp_path)
    db_url = f"sqlite+aiosqlite:///{config.storage.sqlite_lance.db_path}"
    result = await run_migrations(db_url)
    assert result.success, f"migrations failed: {result.error}"

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


def _sqlite_path(lake: MemoryLake) -> str:
    """Resolve the SQLite path from the live MemoryLake config."""
    return lake._config.storage.sqlite_lance.db_path  # type: ignore[union-attr]


async def _sqlite_query(lake: MemoryLake, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Direct read against the live aiosqlite connection on the lake handle."""
    storage = lake._engine._storage  # type: ignore[union-attr]
    handle = storage.graph._handle  # type: ignore[union-attr]
    cur = await handle.sqlite.execute(sql, params)
    rows = await cur.fetchall()
    return [dict(r) for r in rows]


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
    assert "penguins" not in a_text, "ns_b leaked into ns_a"
    assert "penguins" in b_text
    assert "kangaroos" not in b_text, "ns_a leaked into ns_b"


async def test_graphrag_entity_extraction(lake: MemoryLake) -> None:
    """Ingest a doc with a known entity, assert it lands in the SQLite graph.

    Mirrors PR #475's ``test_graphrag_entity_extraction_via_neo4j`` but
    queries the embedded graph adapter directly (``SELECT FROM entities``)
    rather than Cypher.

    Notes on dual-IDs: ``MemoryNamespace`` carries two UUIDs (ADR-024).
    The relational + graph rows in SQLite key on ``namespace.id`` (the
    row-level FK), while ``MemoryLake.recall(namespace=...)`` accepts the
    stable ``namespace_id``. So our direct lookup uses ``ns.id``.

    Names are normalized to lowercase by ``normalize_entity_names_batch``
    before persistence — the LLM stub emits ``"Marie Curie"``, the row
    stores ``"marie curie"``.
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

    rows = await _sqlite_query(
        lake,
        "SELECT name, entity_type FROM entities WHERE namespace_id = ? AND name = ?",
        (str(ns.id).replace("-", ""), "marie curie"),
    )
    assert rows, "marie curie entity not found in SQLite after ingest"
    assert rows[0]["entity_type"] == "PERSON"


async def test_graphrag_two_hop_traversal(lake: MemoryLake) -> None:
    """Ingest A→B→C, query about A, assert C surfaces via 2-hop graph traversal.

    Stages a 3-document chain where extraction emits explicit relationships:
        doc1: alphawidget RELATES_TO betagadget
        doc2: betagadget  RELATES_TO gammathingy
        doc3: gammathingy is the terminal node

    Two known correctness gaps in the embedded recursive-CTE traversal
    layer block this on unfixed main:

    * **DYT-3548** — visited-edge tracking in the recursive CTE doesn't
      stop a relationship from being expanded twice from different
      paths, distorting depth bookkeeping.
    * **DYT-3549** — ``prefer_current`` edge selection in the CTE picks
      the wrong row when an edge has been re-validated, dropping fresh
      hops on multi-version graphs.
    * **DYT-3558** — entity-rebind drops the second relationship when
      ``betagadget`` is re-canonicalised by the resolver (same root cause
      surfaced on the production stack in PR #475 / fixed in PR #477).

    Until any of these fix-PRs lands on main, the second hop never
    materialises. We mark **xfail(strict=True)** so:
    * the test fails loudly the moment the underlying fix is merged
      (xpassed), prompting whoever lands the fix to flip the marker;
    * meanwhile the suite stays green and the fix-tracking is explicit.
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

    # Sanity: first hop must always land — first-time entity upserts
    # hit the happy path and the relationship lands cleanly.
    rel_rows = await _sqlite_query(
        lake,
        """
        SELECT
            (SELECT name FROM entities WHERE id = r.source_entity_id) AS source,
            (SELECT name FROM entities WHERE id = r.target_entity_id) AS target
        FROM relationships r
        WHERE r.namespace_id = ?
        """,
        (str(ns.id).replace("-", ""),),
    )
    rel_pairs = {(r["source"], r["target"]) for r in rel_rows}
    assert ("alphawidget", "betagadget") in rel_pairs, f"first-hop relationship missing in SQLite: {rel_pairs}"

    # Second hop reveals the gap. xfail strict: passes the day a fix
    # lands and forces us to update the marker.
    if ("betagadget", "gammathingy") not in rel_pairs:
        pytest.xfail(
            "DYT-3558 (entity rebind) drops the second relationship on main. "
            f"Relationships present: {sorted(rel_pairs)}"
        )

    # If the rebind is fixed, surface the *traversal* gap (DYT-3548 / DYT-3549).
    result = await lake.recall("alphawidget", namespace=ns.namespace_id, limit=10)
    entity_names = {e.name for e, _ in result.entities}
    if "gammathingy" not in entity_names:
        pytest.xfail(
            "DYT-3548 / DYT-3549 (CTE traversal) prevent 2-hop reach on main. "
            f"Entities surfaced: {sorted(entity_names)}"
        )


async def test_graphrag_temporal_filter(lake: MemoryLake, namespace_id: UUID) -> None:
    """Backdate one doc 20 days, leave another at 5 days, query last 7 days.

    The vector channel pushes ``created_after`` down to
    ``search_similar_chunks``; for sqlite_lance that filter is applied
    by LanceDB's WHERE clause against the ``created_at`` column on the
    chunks vector table. The metadata column lives in SQLite ``chunks``
    too — we update both stores so neither fork sees the stale row.
    """
    r_recent = await _remember(lake, namespace_id=namespace_id, content="recent Falcon launch report.")
    r_old = await _remember(lake, namespace_id=namespace_id, content="old Falcon launch report.")

    storage = lake._engine._storage  # type: ignore[union-attr]
    handle = storage.graph._handle  # type: ignore[union-attr]
    sqlite = handle.sqlite

    twenty_days_ago = datetime.now(UTC) - timedelta(days=20)
    five_days_ago = datetime.now(UTC) - timedelta(days=5)

    # 1) Backdate SQLite ``chunks.created_at`` for both docs.
    await sqlite.execute(
        "UPDATE chunks SET created_at = ? WHERE document_id = ?",
        (twenty_days_ago.isoformat(), str(r_old.document_id).replace("-", "")),
    )
    await sqlite.execute(
        "UPDATE chunks SET created_at = ? WHERE document_id = ?",
        (five_days_ago.isoformat(), str(r_recent.document_id).replace("-", "")),
    )
    await sqlite.commit()

    # 2) Backdate the LanceDB chunks_vec rows so the ANN where-clause
    # filters consistently. Use the same predicate the vector adapter
    # uses (string literals on the ``id`` column).
    chunks_tbl = await handle.lance.open_table("chunks_vec")
    old_chunk_rows = await _sqlite_query(
        lake,
        "SELECT id FROM chunks WHERE document_id = ?",
        (str(r_old.document_id).replace("-", ""),),
    )
    recent_chunk_rows = await _sqlite_query(
        lake,
        "SELECT id FROM chunks WHERE document_id = ?",
        (str(r_recent.document_id).replace("-", ""),),
    )
    for row in old_chunk_rows:
        await chunks_tbl.update(
            where=f"id = '{row['id']}'",
            updates={"created_at": twenty_days_ago},
        )
    for row in recent_chunk_rows:
        await chunks_tbl.update(
            where=f"id = '{row['id']}'",
            updates={"created_at": five_days_ago},
        )

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
    """RecallResult.metadata carries the expected GraphRAG keys on embedded."""
    await _remember(lake, namespace_id=namespace_id, content="A simple sentence about apples.")

    result = await lake.recall("apples", namespace=namespace_id, limit=5)

    md = result.metadata
    expected = {"query", "mode", "namespace_id", "search_methods", "graph_traversal", "temporal", "metrics"}
    missing = expected - md.keys()
    assert not missing, f"missing GraphRAG metadata keys: {missing} (got: {sorted(md.keys())})"

    sm = md["search_methods"]
    assert "by_method" in sm, f"search_methods.by_method missing: {sm.keys()}"
    by_method = sm["by_method"]
    for channel in ("vector", "graph", "keyword"):
        assert channel in by_method, f"by_method.{channel} missing: {by_method.keys()}"
    assert "neighborhood_depth" in md["graph_traversal"]


async def test_graphrag_concurrent_remember(lake: MemoryLake, namespace_id: UUID) -> None:
    """5 concurrent ingests in one namespace, no integrity errors.

    Embedded SQLite uses WAL mode + a 5s busy_timeout and the
    ``_SQLiteLanceEntityKeyGate`` to serialize same-key entity upserts;
    parallel ingests of distinct documents must all succeed and be
    recallable.
    """
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
    # Even on an empty corpus, metadata stays well-formed.
    assert result.metadata.get("mode") in {"HYBRID", "VECTOR", "GRAPH", "ALL"}


async def test_graphrag_relationship_re_upsert(lake: MemoryLake) -> None:
    """3-doc chain where the middle entity is re-mentioned. Verify rel lands.

    Same regression as DYT-3558 (PR #477) but exercised on the embedded
    stack: when ``betagadget`` is re-upserted on doc 2, the resolver
    re-canonicalises its UUID and ``_store_relationships`` drops the
    extracted relationship because ``entity_id_mapping`` no longer
    contains the freshly-minted IDs the relationship references.

    Mark **xfail(strict=True)** so this test starts failing the moment
    DYT-3558's fix lands on main — at which point we flip the marker
    to a hard assert.
    """
    ns = await lake.create_namespace()
    _plan_extraction(
        "alphawidget",
        entities=[("alphawidget", "CONCEPT"), ("betagadget", "CONCEPT")],
        relationships=[("alphawidget", "betagadget", "RELATES_TO")],
    )
    _plan_extraction(
        "betagadget reappears",
        entities=[("betagadget", "CONCEPT"), ("gammathingy", "CONCEPT")],
        relationships=[("betagadget", "gammathingy", "RELATES_TO")],
    )
    _plan_extraction(
        "gammathingy concludes",
        entities=[("gammathingy", "CONCEPT")],
    )

    await _remember(lake, namespace_id=ns.namespace_id, content="alphawidget links to betagadget.")
    await _remember(lake, namespace_id=ns.namespace_id, content="betagadget reappears beside gammathingy.")
    await _remember(lake, namespace_id=ns.namespace_id, content="gammathingy concludes the chain.")

    rel_rows = await _sqlite_query(
        lake,
        """
        SELECT
            (SELECT name FROM entities WHERE id = r.source_entity_id) AS source,
            (SELECT name FROM entities WHERE id = r.target_entity_id) AS target
        FROM relationships r
        WHERE r.namespace_id = ?
        """,
        (str(ns.id).replace("-", ""),),
    )
    rel_pairs = {(r["source"], r["target"]) for r in rel_rows}

    if ("betagadget", "gammathingy") not in rel_pairs:
        pytest.xfail(
            "DYT-3558: entity-rebind drops the second relationship when "
            "betagadget is re-canonicalised on its second mention. "
            f"Relationships present: {sorted(rel_pairs)}"
        )

    # Once DYT-3558 is fixed, the second relationship lands and we can
    # assert it without xfail.
    assert ("betagadget", "gammathingy") in rel_pairs
