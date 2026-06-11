"""VectorCypher SQLite+LanceDB integration tests (PR-D).

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

VectorCypher's embedded ``sqlite_lance`` path is wired. Occurred-bounds
temporal recall (``start_time`` / ``end_time``) now works on the embedded
path — the filter pushes down to ``khora_chunks.occurred_at`` and the
retriever skips the unsupported entity-version narrowing with a structured
degradation rather than failing. ``prefer_current`` now honors expired edges
on the embedded CTE path (#1087) — see ``test_vc_prefer_current_via_cte``.
Multi-hop graph-channel recall now surfaces a graph-reached chunk on the
embedded path (#1086) — the recursive CTE was already correct, and three
wiring fixes (the missing ``namespace_id`` on the coordinator
``get_neighborhoods_batch`` call, the ``Entity``-object normalization in
``_cypher_expand``, and the ``get_chunks_batch`` ``khora_chunks`` fallback)
let the surfaced entity's chunk reach the result. See
``test_vc_graph_mode_surfaces_chunk_via_graph_channel`` (engine-level, via
``mode=SearchMode.GRAPH``, asserting the chunk arrives through the graph
channel with the vector channel proven empty) and
``test_vc_two_hop_neighborhood_batch`` (coordinator-level CTE pin).

No tests in this module remain ``xfail`` — the embedded ``sqlite_lance``
behaviours they tracked have all landed.
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
from khora.query import SearchMode

EMBED_DIM = 32  # matches the sqlite_lance default and the existing fixture helper

pytestmark = [
    pytest.mark.embedded,
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
            # per-test assertions are what matter at the test boundary.
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
    source_timestamp: datetime | None = None,
) -> Any:
    return await kb.remember(
        content=content,
        namespace=namespace_id,
        title=title,
        entity_types=["PERSON", "CONCEPT", "EVENT", "ORG"],
        relationship_types=["KNOWS", "RELATES_TO", "MENTIONS"],
        expertise=_no_event_extraction(),
        source_timestamp=source_timestamp,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_vc_remember_recall_roundtrip(kb: Khora, namespace_id: UUID) -> None:
    """Ingest 3 docs, recall, assert ingested text appears in ``result.chunks[*].content``."""
    contents = [
        "Alice met Bob at the Python conference in Berlin.",
        "Carol presented research on graph databases at the same event.",
        "Dan organized the after-party that lasted until midnight.",
    ]
    for c in contents:
        await _remember(kb, namespace_id=namespace_id, content=c)

    result = await kb.recall("Python conference Berlin", namespace=namespace_id, limit=10)

    assert result.engine_info.get("engine") == "vectorcypher"
    assert len(result.chunks) >= 1, "expected at least one chunk back"
    # The most-relevant ingested text must be visible in the LLM context.
    assert any("Python conference" in c.content for c in result.chunks)


async def test_vc_namespace_isolation(kb: Khora) -> None:
    """Two namespaces, recall does not cross-bleed."""
    ns_a = (await kb.create_namespace()).namespace_id
    ns_b = (await kb.create_namespace()).namespace_id

    await _remember(kb, namespace_id=ns_a, content="alpha document about kangaroos")
    await _remember(kb, namespace_id=ns_b, content="bravo document about penguins")

    result_a = await kb.recall("animals", namespace=ns_a, limit=10)
    result_b = await kb.recall("animals", namespace=ns_b, limit=10)

    a_text = " ".join(c.content for c in result_a.chunks)
    b_text = " ".join(c.content for c in result_b.chunks)

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


async def test_vc_second_ingest_sharing_entity_does_not_crash(kb: Khora, namespace_id: UUID) -> None:
    """Regression for issue #806.

    The second ``remember()`` into the same namespace that re-mentions
    an entity from the first ingest was crashing with
    ``sqlite3.IntegrityError: FOREIGN KEY constraint failed``. Root cause:
    ``VectorCypherEngine._run_skeleton_extraction`` passed relationships
    to ``create_relationships_batch`` whose ``source_entity_id`` /
    ``target_entity_id`` still carried the extraction-time UUIDs while
    ``upsert_entities_batch`` had canonicalised the entity's id to the
    persisted row's UUID. The FK on the relationships table then bit.
    On PG + Neo4j the same shape silently dropped the relationships
    inside Neo4j's ``MATCH``-by-id.

    Plant the same entity on both ingests and assert neither raises.
    """
    _plan_extraction(
        "Alice",
        entities=[("Alice", "PERSON")],
        relationships=[],
    )
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Alice's dietary preferences: vegetarian, no meat, no dairy.",
    )
    # Second ingest re-mentions Alice. Pre-fix this raised
    # IntegrityError; the assertion is "does not raise".
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Alice's dietary preferences: keto, eats meat and dairy, no carbs.",
    )

    coord = kb._engine._storage  # type: ignore[union-attr]
    row_ns_id = await coord.resolve_namespace(namespace_id)
    entities = await coord.graph.list_entities(row_ns_id, limit=100)
    alice_rows = [e for e in entities if e.name == "alice"]
    # Exactly one Alice persisted (deduped by (namespace, name, type)),
    # carrying mention_count >= 2 to prove both ingests landed.
    assert len(alice_rows) == 1, f"expected one canonical Alice row, got {len(alice_rows)}: {alice_rows!r}"


def _plan_two_hop_chain() -> None:
    """Stage the Alice→Bob→Carol→"graph databases" extraction chain.

    Three docs wire a directed chain so the concept ``graph databases`` sits
    two hops out from Alice (Alice→Bob, Bob→Carol, Carol→graph databases).
    """
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


async def _remember_two_hop_chain(kb: Khora, namespace_id: UUID) -> None:
    await _remember(kb, namespace_id=namespace_id, content="Alice knows Bob from college.")
    await _remember(kb, namespace_id=namespace_id, content="Bob and Carol collaborate on research projects.")
    await _remember(kb, namespace_id=namespace_id, content="Carol presented findings on graph databases.")


async def test_vc_two_hop_neighborhood_batch(kb: Khora, namespace_id: UUID) -> None:
    """Coordinator-level pin: the recursive CTE crosses the second hop (#1086).

    Asserts at the exact sink ``recall``'s graph channel drives —
    ``StorageCoordinator.get_neighborhoods_batch`` — that anchoring on Alice
    with ``depth=2`` reaches Carol (2 hops out: Alice→Bob→Carol). This pins the
    SQLite recursive-CTE traversal independently of the recall fusion layer,
    so a future regression in the CTE is caught even if the engine path changes.

    ``depth=1`` is the control: it must surface Bob (1 hop) but NOT Carol —
    proving the ``depth=2`` result is the second hop being walked, not the
    whole graph being returned regardless of depth.
    """
    _plan_two_hop_chain()
    await _remember_two_hop_chain(kb, namespace_id)

    coord = kb._engine._storage  # type: ignore[union-attr]
    row_ns_id = await coord.resolve_namespace(namespace_id)

    entities = await coord.graph.list_entities(row_ns_id, limit=100)
    name_to_id = {e.name: e.id for e in entities}
    assert "alice" in name_to_id, f"expected alice in graph, got {sorted(name_to_id)}"
    alice_id = name_to_id["alice"]

    # depth=1 control: Bob reachable, Carol not yet.
    nb_one = await coord.get_neighborhoods_batch([alice_id], namespace_id=row_ns_id, depth=1)
    one_hop_names = {e.name for e in nb_one[alice_id]["entities"]}
    assert "bob" in one_hop_names, f"1-hop neighbor bob missing: {one_hop_names}"
    assert "carol" not in one_hop_names, f"carol reached at depth=1 — traversal ignored depth: {one_hop_names}"

    # depth=2: the second hop must now include Carol.
    nb_two = await coord.get_neighborhoods_batch([alice_id], namespace_id=row_ns_id, depth=2)
    two_hop_names = {e.name for e in nb_two[alice_id]["entities"]}
    assert "carol" in two_hop_names, f"2-hop neighbor carol not surfaced by the recursive CTE: {two_hop_names}"


async def test_vc_graph_mode_surfaces_chunk_via_graph_channel(kb: Khora, namespace_id: UUID) -> None:
    """Engine-level non-vacuous recall: ``mode=SearchMode.GRAPH`` surfaces a
    graph-reached chunk through the graph channel alone (#1086).

    This guards the full recall wiring of the three #1086 fixes end-to-end:
    ``_cypher_expand`` must (a) call ``get_neighborhoods_batch`` with the
    required ``namespace_id`` and (b) map the embedded backend's ``Entity``
    domain objects to the dict shape the scoring loop reads, and
    ``get_chunks_batch`` must fall back to ``khora_chunks`` so the surfaced
    entity's chunk is actually fetched. Pre-fix, the graph channel produced
    zero chunks (the ``namespace_id`` ``TypeError`` collapsed expansion, the
    ``Entity`` entries were dropped, and the chunk ids resolved to nothing).

    Non-vacuity is enforced two ways rather than trusting ``mode``:

    * The query text is deliberately DISSIMILAR to the chunk content (it does
      not contain ``graph databases``), so the fake content-hash embedder's
      vector channel cannot rank that chunk in — and we assert
      ``vector_chunk_count == 0`` to prove the vector channel surfaced nothing.
      This is the masking the sibling ``test_vc_prefer_current_via_cte``
      documents: at the recall layer the vector channel can otherwise smuggle a
      chunk in independent of the graph edge.
    * We assert the chunk arrived via the graph channel (``graph_chunk_count``
      > 0 and the chunk id is in the graph-channel id set), so a present chunk
      is unambiguously the graph traversal's doing.
    """
    _plan_two_hop_chain()
    await _remember_two_hop_chain(kb, namespace_id)

    result = await kb.recall(
        "Describe the multi-hop chain spanning Alice through her collaborators downstream.",
        namespace=namespace_id,
        mode=SearchMode.GRAPH,
        limit=10,
    )

    assert result.engine_info.get("engine") == "vectorcypher"

    # Anti-vacuity guard 1: the vector channel surfaced nothing, so any chunk
    # present is not vector-channel leakage on the fake embedder.
    assert result.engine_info.get("vector_chunk_count") == 0, (
        f"vector channel surfaced chunks (count={result.engine_info.get('vector_chunk_count')!r}); the "
        "graph-channel assertion below would be masked by vector leakage"
    )
    # Anti-vacuity guard 2: the graph channel actually produced chunks.
    assert (result.engine_info.get("graph_chunk_count") or 0) > 0, (
        f"graph channel produced no chunks (count={result.engine_info.get('graph_chunk_count')!r}) — "
        "the #1086 wiring did not carry the traversal through to fetched chunks"
    )

    # The graph-reached chunk ("graph databases", via Carol) must be present,
    # and it must be one the graph channel surfaced.
    by_content = {c.content.lower(): c.id for c in result.chunks}
    target = next((cid for content, cid in by_content.items() if "graph databases" in content), None)
    assert target is not None, f"graph-reached chunk not surfaced via GRAPH mode; got {sorted(by_content)!r}"
    graph_ids = {
        UUID(i)
        for i in result.engine_info.get("search_methods", {}).get("by_method", {}).get("graph", {}).get("chunk_ids", [])
    }
    assert target in graph_ids, f"target chunk {target} not attributed to the graph channel: {graph_ids}"


async def test_vc_temporal_filter(kb: Khora, namespace_id: UUID) -> None:
    """Two docs at different ``occurred_at``; recall with a ``start_time`` floor
    returns the recent one and drops the old one.

    The chunk ``occurred_at`` (the column the occurred-bounds filter reads,
    ``khora_chunks.occurred_at``) is set AT INGEST from the ``source_timestamp``
    kwarg — exercising the real ``source_timestamp -> occurred_at`` path (#859)
    rather than a raw post-hoc SQL UPDATE. The recent doc is stamped now; the
    old doc now-400d.

    The assertion is non-vacuous: the recent doc must be PRESENT and the old
    doc ABSENT, proving the ``start_time`` floor narrows on the right column
    rather than dropping (or keeping) both.
    """
    now = datetime.now(UTC)
    r_recent = await _remember(
        kb,
        namespace_id=namespace_id,
        content="recent doc about Falcon launch in May 2026.",
        source_timestamp=now,
    )
    r_old = await _remember(
        kb,
        namespace_id=namespace_id,
        content="old doc about Falcon launch in 2024.",
        source_timestamp=now - timedelta(days=400),
    )

    seven_days_ago = now - timedelta(days=7)
    result = await kb.recall(
        "Falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=seven_days_ago,
    )

    returned_doc_ids = {c.document_id for c in result.chunks}
    assert r_old.document_id not in returned_doc_ids, f"old document leaked through temporal filter: {returned_doc_ids}"
    assert r_recent.document_id in returned_doc_ids, (
        f"recent document was dropped by the temporal filter (vacuous pass guard): {returned_doc_ids}"
    )


async def test_vc_temporal_filter_excludes_only_old_doc(kb: Khora, namespace_id: UUID) -> None:
    """Control: seed ONLY the back-dated doc, recall with a ``start_time`` floor
    above it → result is EMPTY.

    Proves the temporal bound is the narrowing force — not vector top-k or
    routing. If recall still returned the back-dated chunk here, the floor
    would not be doing the filtering and ``test_vc_temporal_filter`` above
    would be passing for the wrong reason.
    """
    now = datetime.now(UTC)
    r_old = await _remember(
        kb,
        namespace_id=namespace_id,
        content="old doc about Falcon launch in 2024.",
        source_timestamp=now - timedelta(days=400),
    )

    result = await kb.recall(
        "Falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=now - timedelta(days=7),
    )

    assert result.chunks == [], (
        f"back-dated doc leaked past the start_time floor: {[c.document_id for c in result.chunks]}"
    )
    # Sanity: the doc was actually ingested (the floor is what drops it, not an
    # empty corpus). Without the floor the same recall returns the old chunk.
    unfiltered = await kb.recall("Falcon launch", namespace=namespace_id, limit=10)
    assert r_old.document_id in {c.document_id for c in unfiltered.chunks}, (
        "control corpus was empty even without the floor — test would pass vacuously"
    )


async def test_vc_recall_metadata_keys(kb: Khora, namespace_id: UUID) -> None:
    """``RecallResult.metadata`` exposes the keys VC documents on every recall."""
    await _remember(kb, namespace_id=namespace_id, content="A simple sentence about apples.")

    result = await kb.recall("apples", namespace=namespace_id, limit=5)

    md = result.engine_info
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
    assert result.engine_info.get("engine") == "vectorcypher"


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


async def test_vc_cooccurrence_carries_source_document_ids(kb: Khora, namespace_id: UUID) -> None:
    """ASSOCIATED_WITH co-occurrence edges round-trip with ``source_document_ids``.

    Ingest a single doc that yields ≥2 entities in one chunk → VC's
    ``_build_cooccurrence_relationships`` synthesizes an
    ``ASSOCIATED_WITH`` edge between them.  Read the edge back from the
    embedded graph and assert its provenance is populated — both
    ``source_chunk_ids`` (which chunk produced the pair) and
    ``source_document_ids`` (which document that chunk belongs to).
    """
    _plan_extraction(
        "Marie Curie",
        entities=[("Marie Curie", "PERSON"), ("radium", "CONCEPT")],
    )
    r = await _remember(
        kb,
        namespace_id=namespace_id,
        content="Marie Curie discovered radium and polonium in 1898.",
    )

    coord = kb._engine._storage  # type: ignore[union-attr]
    row_ns_id = await coord.resolve_namespace(namespace_id)
    rels = await coord.graph.list_relationships(row_ns_id, limit=100)

    cooccurrence_rels = [rel for rel in rels if rel.relationship_type == "ASSOCIATED_WITH"]
    assert cooccurrence_rels, (
        f"expected at least one ASSOCIATED_WITH co-occurrence edge, got types: "
        f"{sorted({rel.relationship_type for rel in rels})}"
    )
    for rel in cooccurrence_rels:
        assert rel.source_document_ids, (
            f"ASSOCIATED_WITH edge {rel.id} has empty source_document_ids — provenance regression"
        )
        assert r.document_id in rel.source_document_ids, f"expected {r.document_id} in {rel.source_document_ids}"
        assert rel.source_chunk_ids, (
            f"ASSOCIATED_WITH edge {rel.id} has empty source_chunk_ids — "
            f"co-occurrence edges should know which chunk they came from"
        )


async def test_vc_prefer_current_via_cte(kb: Khora, namespace_id: UUID) -> None:
    """A 1-hop edge whose validity window has closed is skipped when
    ``prefer_current`` is active (#1087).

    Mechanism: re-saving the ``Pierre→polonium`` edge with ``valid_until`` in
    the past UPSERTs the stored row — the fix, since
    ``create_relationships_batch`` was INSERT-only and the re-save previously
    collided on the primary key instead of closing the window. The recursive
    CTE traversal then drops the edge wherever ``prefer_current`` requires
    ``valid_until > now``.

    The scenario is deliberately **1-hop** — anchor on Pierre and expire the
    single edge out of Pierre — so the result isolates the expiry from the
    orthogonal multi-hop traversal gap (``test_vc_two_hop_traversal``).

    The traversal is asserted at the coordinator path ``recall`` drives
    (``get_neighborhoods_batch`` — the ``prefer_current`` sink) rather than on
    ``kb.recall``'s chunks DELIBERATELY: the embedded stack uses fake
    content-hash embeddings, so the vector channel surfaces the polonium chunk
    independently of the graph edge and would mask the edge filter at the
    recall layer. One layer down, ``prefer_current`` is the only thing acting,
    so the signal is deterministic. Three cases prove the filter is the cause:

    * BEFORE expiry, ``prefer_current=True`` → polonium present (the live edge
      is returned even under ``prefer_current`` — not a vacuous always-empty).
    * AFTER expiry, ``prefer_current=True`` → polonium gone (the closed window
      is filtered — the fix).
    * AFTER expiry, ``prefer_current=False`` → polonium present (the row still
      exists; its absence above is caused by ``prefer_current`` + expiry).

    We also confirm the front door: a "currently"-phrased query classifies as
    STATE_QUERY, the temporal category that switches ``prefer_current`` on
    (it is not a public ``recall`` kwarg).
    """
    _plan_extraction(
        "Pierre worked",
        entities=[("Pierre", "PERSON")],
    )
    _plan_extraction(
        "polonium",
        entities=[("Pierre", "PERSON"), ("polonium", "CONCEPT")],
        relationships=[("Pierre", "polonium", "STUDIES")],
    )

    await _remember(kb, namespace_id=namespace_id, content="Pierre worked at the institute.")
    await _remember(kb, namespace_id=namespace_id, content="Pierre researched polonium extensively.")

    coord = kb._engine._storage  # type: ignore[union-attr]
    row_ns_id = await coord.resolve_namespace(namespace_id)

    rels = await coord.graph.list_relationships(row_ns_id, limit=100)
    target_rel = next((r for r in rels if r.relationship_type == "STUDIES"), None)
    assert target_rel is not None, f"expected STUDIES relationship, got {rels!r}"
    pierre_id = target_rel.source_entity_id  # Pierre→polonium

    # The "currently"-phrased query classifies as STATE_QUERY, the category
    # that carries prefer_current=True — the only way recall turns it on.
    result = await kb.recall("What is Pierre currently studying?", namespace=namespace_id, limit=10)
    assert result.engine_info.get("temporal_category") == "state_query", (
        f"temporal query did not trigger the prefer_current category: {result.engine_info.get('temporal_category')!r}"
    )

    # BEFORE expiry, prefer_current=True still returns polonium: the live edge
    # is walked under prefer_current, so a later empty result is the filter
    # acting, not prefer_current trivially returning nothing.
    nb_live = await coord.get_neighborhoods_batch([pierre_id], namespace_id=row_ns_id, depth=1, prefer_current=True)
    live_names = {e.name for e in nb_live[pierre_id]["entities"]}
    assert "polonium" in live_names, f"live edge dropped under prefer_current before any expiry: {live_names}"

    # Expire the single Pierre→polonium edge by re-saving it with valid_until
    # in the past. This exercises the UPSERT (pre-fix the re-save collided on
    # the primary key instead of closing the window).
    target_rel.valid_until = datetime.now(UTC) - timedelta(days=1)
    await coord.graph.create_relationships_batch([target_rel])

    # prefer_current=True now filters the closed-window edge → polonium gone.
    nb_current = await coord.get_neighborhoods_batch([pierre_id], namespace_id=row_ns_id, depth=1, prefer_current=True)
    current_names = {e.name for e in nb_current[pierre_id]["entities"]}
    assert "polonium" not in current_names, f"expired edge leaked through prefer_current=True: {current_names}"

    # And without prefer_current the expired edge is still walked — proving the
    # absence above is caused by prefer_current + expiry, not the edge vanishing.
    nb_default = await coord.get_neighborhoods_batch([pierre_id], namespace_id=row_ns_id, depth=1, prefer_current=False)
    default_names = {e.name for e in nb_default[pierre_id]["entities"]}
    assert "polonium" in default_names, (
        f"polonium unreachable even without prefer_current after expiry — control is vacuous: {default_names}"
    )
