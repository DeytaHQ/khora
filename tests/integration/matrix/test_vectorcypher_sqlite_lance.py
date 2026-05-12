"""VectorCypher SQLite+LanceDB integration tests (DYT-3545 / PR-D).

VectorCypher is one of khora's two production-ready engines and v0.9.0
declares **SQLite + LanceDB** the default *embedded* stack. These tests
wire up ``Khora(engine="vectorcypher")`` against a fully-embedded
sqlite_lance coordinator (per-test ``tmp_path``) and exercise the same
remember/recall behaviour we already cover for the production stack.

How LLM calls are stubbed:
* ``LiteLLMEmbedder.embed_batch`` and ``embed`` return content-derived
  unit vectors of dimension ``EMBED_DIM=32`` (the embedded backend's
  default) so similar text shares an embedding and recall ordering is
  deterministic. ``OPENAI_API_KEY`` is **not** required.
* ``LLMEntityExtractor.extract_multi`` is replaced with a registry stub
  identical to the chronicle-pg pattern — register entities per content
  marker before calling ``_remember()``.

How to run locally::

    uv run pytest tests/integration/matrix/test_vectorcypher_sqlite_lance.py \\
        -v -m integration --no-cov

No Docker / Postgres / Neo4j needed — the embedded stack is pure
in-process SQLite (``aiosqlite``) and LanceDB (``lancedb``).

## State

VectorCypher's embedded ``sqlite_lance`` path is wired and 7/10 tests
in this module pass cleanly. Three tests remain ``xfail`` for known
backend gaps (each xfail carries an explanatory string):

* ``test_vc_two_hop_traversal`` — multi-hop CTE traversal correctness
  on the SQL-emulated graph.
* ``test_vc_temporal_filter`` — temporal pushdown on the embedded path.
* ``test_vc_prefer_current_via_cte`` — ``prefer_current`` honoring on
  CTE traversal.

The remaining xfails track concrete behavioural gaps, not a wiring
issue. They are written end-to-end so that when the underlying paths
land, the same tests serve as the acceptance suite without rewriting.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:  # Module-level import gate matches existing sqlite_lance suites.
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.skills import ExpertiseConfig
from khora.khora import Khora

EMBED_DIM = 32  # matches the sqlite_lance default and the existing fixture helper

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


# ---------------------------------------------------------------------------
# Deterministic embedder + extractor stubs (no OPENAI_API_KEY needed)
# ---------------------------------------------------------------------------


def _embed_for(text_in: str) -> list[float]:
    """Return a deterministic L2-normalised ``EMBED_DIM`` vector for ``text_in``.

    Mirrors ``tests/integration/_sqlite_lance_fixtures.fake_embedding`` —
    SHA-256 the text, expand to ``EMBED_DIM`` floats, normalise.  Same
    text ⇒ same vector, different text ⇒ different vector.  Suitable for
    top-k ordering assertions but NOT for semantic similarity (the hash
    has no notion of meaning).
    """
    seed = hashlib.sha256(text_in.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(EMBED_DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


_EXTRACTION_REGISTRY: dict[str, ExtractionResult] = {}


def _plan_extraction(
    marker: str,
    entities: list[tuple[str, str]],
    relationships: list[tuple[str, str, str]] | None = None,
) -> None:
    """Stage an ``ExtractionResult`` for documents containing ``marker``."""
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
    for t in texts:
        matched = next(
            (result for marker, result in _EXTRACTION_REGISTRY.items() if marker in t),
            None,
        )
        out.append(matched if matched is not None else ExtractionResult())
    return out


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub embedder + extractor so no real LLM is called.

    This is the explicit ``OPENAI_API_KEY``-not-required guarantee — VC
    calls ``LiteLLMEmbedder`` on every remember/recall, so without this
    patch the tests would attempt real network calls.
    """
    _EXTRACTION_REGISTRY.clear()
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )
    monkeypatch.setattr(
        "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
        _stub_extract_multi,
    )


# ---------------------------------------------------------------------------
# Per-test embedded Khora fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Khora]:
    """Per-test VectorCypher Khora on a fresh embedded stack.

    Each test gets its own ``tmp_path`` for isolation — sqlite_lance
    caches engine pools by URL inside ``StorageFactory``, so reusing a
    path across tests would leak state.
    """
    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "khora.db"),
        lance_path=str(tmp_path / "khora.lance"),
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    # No Neo4j — the whole point of the embedded path.
    config.neo4j_url = None
    # Single-chunk documents keep the test deterministic.
    config.pipelines.chunk_size = 1024
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False

    kb = Khora(config, engine="vectorcypher", run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        try:
            await kb.disconnect()
        except Exception:
            # Disconnect can throw if connect partially succeeded; the
            # xfail marker is what matters at the test boundary.
            pass


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    ns = await kb.create_namespace()
    return ns.namespace_id


def _no_event_extraction() -> ExpertiseConfig:
    """ExpertiseConfig that runs entity extraction but skips event/fact extraction."""
    return ExpertiseConfig(name="vc-sqlite-lance-integ")


async def _remember(
    kb: Khora,
    *,
    namespace_id: UUID,
    content: str,
    title: str = "",
) -> Any:
    return await kb.remember(
        content=content,
        namespace=namespace_id,
        title=title,
        entity_types=["PERSON", "CONCEPT", "EVENT", "ORG"],
        relationship_types=["KNOWS", "RELATES_TO", "MENTIONS"],
        expertise=_no_event_extraction(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_vc_remember_recall_roundtrip(kb: Khora, namespace_id: UUID) -> None:
    """Ingest 3 docs, recall, assert ingested text appears in ``context_text``."""
    contents = [
        "Alice met Bob at the Python conference in Berlin.",
        "Carol presented research on graph databases at the same event.",
        "Dan organized the after-party that lasted until midnight.",
    ]
    for c in contents:
        await _remember(kb, namespace_id=namespace_id, content=c)

    result = await kb.recall("Python conference Berlin", namespace=namespace_id, limit=10)

    assert result.metadata.get("engine") == "vectorcypher"
    assert len(result.chunks) >= 1, "expected at least one chunk back"
    # The most-relevant ingested text must be visible in the LLM context.
    assert "Python conference" in result.context_text


async def test_vc_namespace_isolation(kb: Khora) -> None:
    """Two namespaces, recall does not cross-bleed."""
    ns_a = (await kb.create_namespace()).namespace_id
    ns_b = (await kb.create_namespace()).namespace_id

    await _remember(kb, namespace_id=ns_a, content="alpha document about kangaroos")
    await _remember(kb, namespace_id=ns_b, content="bravo document about penguins")

    result_a = await kb.recall("animals", namespace=ns_a, limit=10)
    result_b = await kb.recall("animals", namespace=ns_b, limit=10)

    a_text = " ".join(c.content for c, _ in result_a.chunks)
    b_text = " ".join(c.content for c, _ in result_b.chunks)

    assert "kangaroos" in a_text
    assert "penguins" not in a_text, "namespace_b leaked into namespace_a"
    assert "penguins" in b_text
    assert "kangaroos" not in b_text, "namespace_a leaked into namespace_b"


async def test_vc_entity_extraction(kb: Khora, namespace_id: UUID) -> None:
    """Ingest a doc with a known entity → entity persists in the embedded graph.

    Queries the SQLite-CTE graph adapter directly via
    ``coord.graph.list_entities`` to confirm the entity was actually
    written (not just returned by the LLM stub).

    Note on dual-IDs: ``MemoryNamespace`` carries two UUIDs —
    ``namespace_id`` (stable) is what ``Khora.recall`` accepts, but
    the graph rows key on ``namespace.id`` (row-level FK), so we resolve
    before the direct lookup. Names are lowercased by
    ``normalize_entity_names_batch`` before persistence.
    """
    _plan_extraction(
        "Ada Lovelace",
        entities=[("Ada Lovelace", "PERSON"), ("Analytical Engine", "CONCEPT")],
        relationships=[("Ada Lovelace", "Analytical Engine", "WORKED_ON")],
    )
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Ada Lovelace wrote the first algorithm intended for the Analytical Engine.",
    )

    coord = kb._engine._storage  # type: ignore[union-attr]
    row_ns_id = await coord.resolve_namespace(namespace_id)
    entities = await coord.graph.list_entities(row_ns_id, limit=100)
    names = {e.name for e in entities}
    assert "ada lovelace" in names, f"expected ada lovelace in graph, got {names}"
    assert "analytical engine" in names


@pytest.mark.xfail(
    strict=False,
    reason=(
        "VC's sync remember path does not rebind extraction-time entity IDs "
        "to upsert-resolved IDs (DYT-3558 fixed this only in "
        "pipelines/flows/ingest.py, not in vectorcypher/engine._run_skeleton_extraction). "
        "On sqlite_lance, that produces FOREIGN KEY failures when a second "
        "document re-mentions an entity from the first doc."
    ),
    raises=Exception,
)
async def test_vc_two_hop_traversal(kb: Khora, namespace_id: UUID) -> None:
    """3 connected docs (A→B→C), query about A surfaces C via 2-hop traversal."""
    _plan_extraction(
        "Alice",
        entities=[("Alice", "PERSON"), ("Bob", "PERSON")],
        relationships=[("Alice", "Bob", "KNOWS")],
    )
    _plan_extraction(
        "Bob and Carol",
        entities=[("Bob", "PERSON"), ("Carol", "PERSON")],
        relationships=[("Bob", "Carol", "KNOWS")],
    )
    _plan_extraction(
        "Carol presented",
        entities=[("Carol", "PERSON"), ("graph databases", "CONCEPT")],
        relationships=[("Carol", "graph databases", "RESEARCHES")],
    )

    await _remember(kb, namespace_id=namespace_id, content="Alice knows Bob from college.")
    await _remember(kb, namespace_id=namespace_id, content="Bob and Carol collaborate on research projects.")
    await _remember(kb, namespace_id=namespace_id, content="Carol presented findings on graph databases.")

    # Force ≥ 2-hop expansion via graph_depth=2.
    result = await kb.recall(
        "Alice",
        namespace=namespace_id,
        limit=10,
        graph_depth=2,
    )

    text_blob = result.context_text + " ".join(c.content for c, _ in result.chunks)
    # The 2-hop reachable concept ("graph databases" via Bob→Carol) must
    # surface in the recall result. If this fails the CTE traversal is
    # not crossing 2 hops — see DYT-3548.
    assert "graph databases" in text_blob.lower(), f"2-hop entity not surfaced; got context={text_blob[:300]!r}"


@pytest.mark.xfail(
    strict=True,
    reason="DYT-3562: VectorCypher embedded path doesn't push temporal filter to LanceDB query",
)
async def test_vc_temporal_filter(kb: Khora, namespace_id: UUID) -> None:
    """Two docs at different ``occurred_at``; recall with ``last 7 days`` filter
    only returns the recent one.

    ``Khora.remember()`` does not surface ``occurred_at`` on its public
    API (engines accept it but the kb-level wrapper does not forward it),
    so we ingest both docs at ``now`` and then back-date one chunk's
    ``source_timestamp`` directly in SQLite — same pattern as
    ``test_skeleton_pg`` for the production stack.
    """
    now = datetime.now(UTC)
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="recent doc about Falcon launch in May 2026.",
    )
    r_old = await _remember(
        kb,
        namespace_id=namespace_id,
        content="old doc about Falcon launch in 2024.",
    )

    # Back-date the "old" doc's chunks via the embedded SQLite handle.
    # The sqlite_lance backend stores UUIDs as 32-char hex (no dashes),
    # so we must match that form when filtering on ``document_id``.
    coord = kb._engine._storage  # type: ignore[union-attr]
    handle = coord.vector._handle  # type: ignore[union-attr]
    backdated_iso = (now - timedelta(days=400)).isoformat()
    await handle.sqlite.execute(
        "UPDATE chunks SET source_timestamp = ? WHERE document_id = ?",
        (backdated_iso, str(r_old.document_id).replace("-", "")),
    )
    await handle.sqlite.commit()

    seven_days_ago = now - timedelta(days=7)
    result = await kb.recall(
        "Falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=seven_days_ago,
    )

    returned_doc_ids = {c.document_id for c, _ in result.chunks}
    assert r_old.document_id not in returned_doc_ids, f"old document leaked through temporal filter: {returned_doc_ids}"


async def test_vc_recall_metadata_keys(kb: Khora, namespace_id: UUID) -> None:
    """``RecallResult.metadata`` exposes the keys VC documents on every recall."""
    await _remember(kb, namespace_id=namespace_id, content="A simple sentence about apples.")

    result = await kb.recall("apples", namespace=namespace_id, limit=5)

    md = result.metadata
    # Engine identifier is the only key VC promises across every code
    # path; routing/timing keys depend on the router being enabled. We
    # assert the floor.
    assert md.get("engine") == "vectorcypher"
    # ``RecallResult.metadata`` must be a dict so downstream consumers
    # can ``.get()`` keys safely.
    assert isinstance(md, dict)
    # When routing fires, ``routing`` will be present; we treat it as
    # informational rather than load-bearing on the embedded path.


async def test_vc_concurrent_remember(kb: Khora, namespace_id: UUID) -> None:
    """5 concurrent ingests in one namespace, no integrity errors."""
    contents = [f"document number {i} mentions widget-{i}" for i in range(5)]
    results = await asyncio.gather(
        *(_remember(kb, namespace_id=namespace_id, content=c) for c in contents),
        return_exceptions=True,
    )
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"concurrent remember raised: {errors}"
    doc_ids = {r.document_id for r in results}  # type: ignore[union-attr]
    assert len(doc_ids) == 5, f"expected 5 distinct documents, got {doc_ids}"


async def test_vc_recall_empty_namespace(kb: Khora) -> None:
    """Recall on a fresh empty namespace returns an empty (but well-formed) result."""
    ns = (await kb.create_namespace()).namespace_id
    result = await kb.recall("anything", namespace=ns, limit=5)
    assert result.chunks == []
    assert result.metadata.get("engine") == "vectorcypher"


async def test_vc_recall_handles_punctuated_query(kb: Khora, namespace_id: UUID) -> None:
    """Regression for issue #526 at the **vectorcypher engine layer**.

    PR #528's escape_fts5_query fix was verified at the storage adapter
    layer. This test routes punctuated / FTS5-operator queries through
    the full Khora.recall() → vectorcypher engine (which fuses vector +
    BM25). Catches a future regression that introduces a fusion path
    bypassing the escape.
    """
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Marie Curie won the Nobel Prize in Physics in 1903.",
    )
    for query in (
        "What did Curie win?",
        "Curie: Nobel",
        "Curie (Nobel)",
        "Curie AND Physics",
        'say "hello" Curie',
        "Curie*",
    ):
        result = await kb.recall(query, namespace=namespace_id, limit=3)
        assert isinstance(result.chunks, list), f"recall must not raise on {query!r}"


async def test_vc_dual_node_persistence(kb: Khora, namespace_id: UUID) -> None:
    """Ingest one doc with an entity → confirm the dual-node markers VectorCypher
    writes (chunk-node + entity-node + MENTIONED_IN-style edge) are persisted in
    the embedded graph.

    On Neo4j the markers are ``(:Chunk)``, ``(:Entity)``, ``[:MENTIONED_IN]``
    nodes/edges (see ``dual_nodes.py``). Translated to the SQLite-CTE
    adapter, we expect:
      * a chunk row in the chunks table,
      * an entity row with a ``source_chunk_ids`` link back to that chunk,
      * a relationship the engine emits to mirror MENTIONED_IN (today
        VectorCypher writes ASSOCIATED_WITH co-occurrence edges via
        ``_build_cooccurrence_relationships``).
    """
    _plan_extraction(
        "Marie Curie",
        entities=[("Marie Curie", "PERSON"), ("radium", "CONCEPT")],
        relationships=[("Marie Curie", "radium", "DISCOVERED")],
    )
    r = await _remember(
        kb,
        namespace_id=namespace_id,
        content="Marie Curie discovered radium and polonium in 1898.",
    )

    coord = kb._engine._storage  # type: ignore[union-attr]
    row_ns_id = await coord.resolve_namespace(namespace_id)

    # Entity persistence: both names land in the graph (lowercased).
    entities = await coord.graph.list_entities(row_ns_id, limit=100)
    name_to_entity = {e.name: e for e in entities}
    assert "marie curie" in name_to_entity, f"missing marie curie: {list(name_to_entity)}"
    assert "radium" in name_to_entity, f"missing radium: {list(name_to_entity)}"

    # Chunk-node ↔ entity-node link: ``source_chunk_ids`` (or
    # ``source_document_ids``) must include this document's chunk.
    marie = name_to_entity["marie curie"]
    assert r.document_id in (marie.source_document_ids or []) or any(marie.source_chunk_ids or []), (
        f"marie curie has no source link back to the ingest doc: {marie!r}"
    )

    # Edge persistence: at least one relationship survives the ingest.
    rels = await coord.graph.list_relationships(row_ns_id, limit=100)
    assert rels, "expected at least one relationship after dual-node write"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Same root cause as test_vc_two_hop_traversal: VC's sync remember "
        "does not rebind extraction-time entity IDs after upsert resolution, "
        "so a second doc re-mentioning a prior entity hits FOREIGN KEY on "
        "create_relationships_batch. Tracked alongside DYT-3548/DYT-3549/DYT-3558."
    ),
    raises=Exception,
)
async def test_vc_prefer_current_via_cte(kb: Khora, namespace_id: UUID) -> None:
    """3-hop A→B→C with B's outgoing edge expired; ``prefer_current=True`` must
    NOT return C's content.

    This tests Graphiti-style bi-temporal invalidation pushed through the
    SQLite-CTE traversal. R2 (DYT-3549) is the fix that makes
    ``prefer_current`` honor expired/invalidated edges in the CTE. Until
    that lands, the CTE happily walks expired edges, so this test is
    expected to fail even after DYT-3560 wires the VC embedded path.

    The chain: B→C edge has ``valid_to`` in the past → with
    ``prefer_current=True`` it must be skipped → C ("polonium") is
    unreachable from A ("Marie") via 2-hop traversal. We assert
    ``"polonium"`` is NOT in the recall context.
    """
    _plan_extraction(
        "Marie",
        entities=[("Marie", "PERSON"), ("Pierre", "PERSON")],
        relationships=[("Marie", "Pierre", "KNOWS")],
    )
    _plan_extraction(
        "Pierre",
        entities=[("Pierre", "PERSON"), ("polonium", "CONCEPT")],
        relationships=[("Pierre", "polonium", "STUDIES")],
    )

    await _remember(kb, namespace_id=namespace_id, content="Marie collaborated with Pierre.")
    await _remember(kb, namespace_id=namespace_id, content="Pierre researched polonium extensively.")

    # Manually expire the Pierre→polonium edge in the embedded graph.
    coord = kb._engine._storage  # type: ignore[union-attr]
    row_ns_id = await coord.resolve_namespace(namespace_id)
    rels = await coord.graph.list_relationships(row_ns_id, limit=100)
    target_rel = next(
        (r for r in rels if r.relationship_type == "STUDIES"),
        None,
    )
    assert target_rel is not None, f"expected STUDIES relationship, got {rels!r}"
    # Expire the edge by setting valid_to in the past. The exact field
    # name lives on Relationship; CTE traversal R2 reads it for filtering.
    target_rel.valid_to = datetime.now(UTC) - timedelta(days=1)
    await coord.graph.create_relationships_batch(row_ns_id, [target_rel])

    result = await kb.recall(
        "Marie",
        namespace=namespace_id,
        limit=10,
        graph_depth=2,
        prefer_current=True,
    )

    text_blob = result.context_text.lower() + " ".join(c.content for c, _ in result.chunks).lower()
    assert "polonium" not in text_blob, "expired edge leaked through prefer_current=True (DYT-3549 R2 fix not in main)"
