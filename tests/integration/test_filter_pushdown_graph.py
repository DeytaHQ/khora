"""Filter-pushdown spies — VectorCypher paths that need the live graph stack.

White-box spies proving the recall filter reaches every graph-side channel
UNCHANGED (same ``canonical_hash`` at each pushdown boundary). These paths only
fire on the full PG+Neo4j VectorCypher config. Two reasons they cannot run on the
embedded ``sqlite_lance`` stack: (a) session fan-out and CHANGE version-history are
Neo4j-gated (``_dual_nodes`` / ``_neo4j_driver`` are ``None`` there); and (b) on the
embedded stack with the deterministic test embedder, entity-vector similarity falls
below the entity-similarity floor (``min_entity_similarity=0.3``), so entry-entity
search returns 0 and every graph path short-circuits to ``_simple_retrieve`` before
its channel can fire. The live stack lowers the floor in the fixture so real
entry entities surface (see ``_retriever``). The embedded counterparts (vector /
BM25 / recency / restrictive-fallback / EXPLICIT synthesis) live in the no-Docker
module owned by qa-embedded.

Paths covered here:

* (3) Cypher channel — the caller filter is compiled to a ``c.<key>`` predicate
  via ``compile_cypher`` at the graph chunk-fetch boundary.
* (4) session fan-out — when entry entities span >= 2 sessions, the per-session
  ``_vector_search_chunks`` calls + the unscoped fallback each carry the filter.
* (5) restrictive-fallback re-run guard — under a caller filter, the unfiltered
  re-run that normally fires on sparse temporal results is SUPPRESSED (it drops
  the filter); exercised on PG, where the point-in-time entity-version path the
  re-run pairs with is honored (embedded skips entity-version narrowing).
* (7) CHANGE decomposition — a CHANGE query with version history runs a second
  ``_vector_search_chunks`` over the decomposed current-state sub-query, which
  carries the filter.
* (8) graph over-fetch + metadata post-filter — a residual (metadata) predicate
  widens the graph fetch and is applied as an in-memory post-filter; the filter
  threads to ``_fetch_chunks_from_entities``.

Spy contract (shared with the embedded suite via ``tests.test_helpers.filter_spy``):
``len(captures) >= N`` vacuity guard + every capture's ``canonical_hash`` equals
the facade-built expected hash. NEVER an exact count — several channels compile
the same filter at more than one site (``compile_cypher`` fires from both the
over-fetch probe and ``dual_nodes`` chunk fetch; ``compile_python`` from both the
graph and recency post-filters), so an exact-count assertion is a vacuity trap.
Each ``>= N`` spy is paired with a CONTROL recall under the inverse condition
(no filter, or a mode that skips the channel) proving the spied call COULD fire
but did NOT — so the positive assertion cannot pass vacuously. NO result/ranking
inspection.

Self-skip: the whole module is gated on ``NEO4J_INTEGRATION_TEST`` + Postgres
reachability, so a no-Docker ``make test`` collects-and-skips it cleanly. Run it
with THIS repo's live stack — note ``make dev`` exposes Neo4j Bolt on 7688 and
Postgres on 5434 with the Neo4j password ``pleaseletmein`` (see ``compose.yaml``),
so the env overrides below are required (the defaults match the CI service ports
7687 / ``password``, not the local ``make dev`` ports)::

    make dev
    NEO4J_INTEGRATION_TEST=1 KHORA_PG_REQUIRED=1 \\
        KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
        KHORA_NEO4J_URL=bolt://localhost:7688 \\
        KHORA_NEO4J_PASSWORD=pleaseletmein \\
        uv run pytest tests/integration/test_filter_pushdown_graph.py
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse
from uuid import UUID

import pytest

from khora import Khora
from khora.config import KhoraConfig
from khora.query import SearchMode
from tests.test_helpers.filter_spy import (
    assert_filter_threaded,
    plan_extraction,
    seed_corpus,
    spy_on,
    stub_llm,
)

pytestmark = [pytest.mark.integration, pytest.mark.filter_enforcement]

# The Postgres pgvector column is fixed at 1536 (KhoraConfig rejects any other
# dimension on the PG backend), so the live-DB suite sizes its deterministic
# vectors at 1536 rather than the small embedded ``EMBED_DIM``.
_PG_EMBED_DIM = 1536


# --------------------------------------------------------------------------- #
# Self-skip guards — mirror the existing tests/integration skip pattern so the
# default no-Docker run collects-and-skips cleanly.
# --------------------------------------------------------------------------- #


def _pg_reachable() -> bool:
    url = os.environ.get("KHORA_DATABASE_URL", "postgresql://khora:khora@localhost:5434/khora")
    parsed = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 5432), timeout=2):
            return True
    except OSError:
        return False


_SKIP = pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST") or not _pg_reachable(),
    reason="set NEO4J_INTEGRATION_TEST=1 and run `make dev` (PG+Neo4j) to exercise the graph-side filter spies",
)


# --------------------------------------------------------------------------- #
# Live PG+Neo4j kb fixture (mirrors tests/integration/test_*_integration.py).
# --------------------------------------------------------------------------- #


@pytest.fixture
async def kb(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Khora]:
    # Deterministic extractor + embedder so the same content seeds the same
    # entities + vectors as the embedded suite (shared via filter_spy.stub_llm).
    # Sized at 1536 because the Postgres pgvector column requires it.
    stub_llm(monkeypatch, dim=_PG_EMBED_DIM)

    database_url = os.environ.get(
        "KHORA_DATABASE_URL",
        "postgresql+asyncpg://khora:khora@localhost:5434/khora",
    )
    neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    config = KhoraConfig(database_url=database_url, neo4j_url=neo4j_url)
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")
    config.llm.embedding_dimension = _PG_EMBED_DIM
    config.storage.embedding_dimension = _PG_EMBED_DIM
    config.pipeline.extract_entities = True
    config.pipeline.selective_extraction = False

    instance = Khora(config, run_migrations=False)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.disconnect()


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    ns = await kb.create_namespace()
    return ns.namespace_id


def _retriever(kb: Khora):
    """Reach the live VectorCypher retriever the engine drives, floor lowered.

    The shared seed uses a deterministic HASH embedder (``fake_embedding``): its
    vectors carry no semantic meaning, so a query↔entity cosine similarity sits
    well below the default ``min_entity_similarity=0.3`` floor (retriever.py:270)
    and ``_vector_search_entities`` would return 0 entry entities — short-circuiting
    every graph path to ``_simple_retrieve`` and making these spies pass vacuously.
    Lowering the floor to 0.0 lets the deterministic-embedder entities clear it so
    ``entry_entities > 0`` on the live pgvector stack. This is a test-fixture knob,
    not a product change.
    """
    retriever = kb._engine._retriever  # type: ignore[union-attr,attr-defined]
    retriever._config.min_entity_similarity = 0.0
    return retriever


async def _assert_entry_entities_present(retriever, query: str, namespace_id: UUID) -> None:
    """Sanity gate: prove the seed yields entry entities on the live stack.

    Every graph path lives PAST the ``if not entry_entities: return
    _simple_retrieve(...)`` early-exit in ``_vectorcypher_retrieve`` (retriever.py:1149).
    If entity vector search returns 0, the recall short-circuits and the graph
    spies pass VACUOUSLY. Asserting > 0 here (with the floor lowered) closes that
    hole before any channel assertion.

    ``recall()`` resolves the public namespace_id to the active row-level id
    before searching entity vectors (entities are keyed by the row id), so we
    resolve here too — passing the public id straight to
    ``_vector_search_entities`` matches nothing and fails this gate spuriously.
    """
    resolved = await retriever._storage.resolve_namespace(namespace_id)
    embedding = await retriever._embedder.embed(query)
    entry = await retriever._vector_search_entities(embedding, resolved, limit=10)
    assert entry, (
        "entry_entities == 0 on the live stack even with min_entity_similarity=0.0 — "
        "the seed did not surface; graph spies would pass vacuously. Escalate (seed problem)."
    )


# Filters: a pushable system-key leaf (consumed by compile_cypher → path 3) and a
# residual metadata SUB-PATH leaf (unpushable to Cypher → over-fetch + in-memory
# post-filter → path 8). The dotted form yields path ('metadata', 'tier') — a
# genuine sub-path the Cypher compiler defers, vs whole-blob ('metadata',) equality.
_SYSTEM_FILTER = {"source_type": {"$eq": "slack"}}
_METADATA_FILTER = {"metadata.tier": {"$eq": "gold"}}


# --------------------------------------------------------------------------- #
# (3) Cypher channel — caller filter reaches compile_cypher unchanged.
# --------------------------------------------------------------------------- #


@_SKIP
@pytest.mark.asyncio
async def test_cypher_channel_threads_filter(kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pushable system-key filter reaches ``compile_cypher`` with its hash intact.

    ``compile_cypher`` fires from BOTH the over-fetch probe and the dual_nodes
    chunk fetch, so we assert ``>= 1`` and that EVERY capture matches — never an
    exact count.
    """
    plan_extraction(
        "Ada Lovelace",
        entities=[("Ada Lovelace", "PERSON"), ("Analytical Engine", "CONCEPT")],
        relationships=[("Ada Lovelace", "Analytical Engine", "WORKED_ON")],
    )
    await seed_corpus(
        lambda **kw: kb.remember(entity_types=["PERSON", "CONCEPT"], relationship_types=["WORKED_ON"], **kw),
        namespace_id,
        ["Ada Lovelace wrote the first algorithm for the Analytical Engine."],
    )

    retriever = _retriever(kb)
    await _assert_entry_entities_present(retriever, "Ada Lovelace", namespace_id)

    # compile_cypher is a SYNC module-level function imported function-locally in
    # both call sites; spy the SOURCE module so both sites are captured.
    import khora.filter.compilers.cypher as cypher_mod

    captures = spy_on(monkeypatch, cypher_mod, "compile_cypher")

    await kb.recall("Ada Lovelace", namespace=namespace_id, filter=_SYSTEM_FILTER)
    assert_filter_threaded(captures, _SYSTEM_FILTER, min_calls=1)

    # CONTROL: a no-filter recall must NOT compile any filter at this boundary —
    # proves the spied call is driven by the caller filter, not always-on.
    control = spy_on(monkeypatch, cypher_mod, "compile_cypher")
    await kb.recall("Ada Lovelace", namespace=namespace_id)
    assert control == [], "control: compile_cypher must not run with no caller filter"


# --------------------------------------------------------------------------- #
# (8) graph over-fetch + metadata post-filter.
# --------------------------------------------------------------------------- #


@_SKIP
@pytest.mark.asyncio
async def test_graph_overfetch_post_filter_threads_filter(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A residual metadata filter widens the graph fetch and threads to the post-filter.

    The metadata predicate cannot push to Cypher, so the engine over-fetches and
    applies an in-memory ``compile_python`` post-filter. We assert the filter
    reaches ``_fetch_chunks_from_entities`` (graph fetch) unchanged; ``>= 1`` +
    all-match (compile_python also fires from the recency post-filter site).
    """
    import khora.filter.compilers.cypher as cypher_mod
    import khora.filter.compilers.python as python_mod

    plan_extraction(
        "Ada Lovelace",
        entities=[("Ada Lovelace", "PERSON"), ("Analytical Engine", "CONCEPT")],
        relationships=[("Ada Lovelace", "Analytical Engine", "WORKED_ON")],
    )
    # The seeded chunk CARRIES metadata {"tier": "gold"} so the residual metadata
    # predicate post-filters something real and graph_chunks are non-empty.
    await seed_corpus(
        lambda **kw: kb.remember(entity_types=["PERSON", "CONCEPT"], relationship_types=["WORKED_ON"], **kw),
        namespace_id,
        [
            {
                "content": "Ada Lovelace wrote the first algorithm for the Analytical Engine.",
                "metadata": {"tier": "gold"},
            }
        ],
    )

    retriever = _retriever(kb)
    await _assert_entry_entities_present(retriever, "Ada Lovelace", namespace_id)

    # 8a: the over-fetch residual probe + dual_nodes fetch both compile the filter
    # to Cypher; 8b: the in-memory post-filter compiles it to Python. Both must
    # receive the UNCHANGED full AST. Spy the graph fetch (filter threading) AND
    # the compile_python post-filter (the metadata backstop).
    fetch_caps = spy_on(monkeypatch, retriever, "_fetch_chunks_from_entities")
    cypher_caps = spy_on(monkeypatch, cypher_mod, "compile_cypher")
    python_caps = spy_on(monkeypatch, python_mod, "compile_python")

    await kb.recall("Ada Lovelace", namespace=namespace_id, filter=_METADATA_FILTER)

    # The graph fetch carries the filter; the Cypher probe + Python post-filter
    # both receive the same unchanged AST. ``>= 1`` + ALL-match (each compiler
    # also fires at a second site — path-3 cypher fetch / recency post-filter —
    # so an exact count is a vacuity trap).
    assert_filter_threaded(fetch_caps, _METADATA_FILTER, min_calls=1)
    assert_filter_threaded(cypher_caps, _METADATA_FILTER, min_calls=1)
    assert_filter_threaded(python_caps, _METADATA_FILTER, min_calls=1)

    # CONTROL: no-filter recall — the graph fetch still runs, but with no
    # filter_ast, so a capture with no FilterNode proves the spy observes the
    # boundary and the positive case is driven by the caller filter.
    control = spy_on(monkeypatch, retriever, "_fetch_chunks_from_entities")
    await kb.recall("Ada Lovelace", namespace=namespace_id)
    assert control, "control precondition: the graph fetch must run on a no-filter recall too"
    assert all(c.canonical_hash is None for c in control), "control: no-filter recall must carry no filter_ast"


# --------------------------------------------------------------------------- #
# (4) session fan-out — per-session searches each carry the filter.
# --------------------------------------------------------------------------- #


@_SKIP
@pytest.mark.asyncio
async def test_session_fanout_threads_filter(kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch) -> None:
    """When entry entities span >= 2 channels, every per-channel search threads the filter.

    Vacuity N is DYNAMIC: one ``_vector_search_chunks`` per fanned-out channel +
    one unscoped fallback. We assert ``>= 2`` (fan-out requires >= 2 channels)
    and that every capture matches — not a hardcoded exact count.
    """
    plan_extraction(
        "Ada Lovelace",
        entities=[("Ada Lovelace", "PERSON")],
    )
    # Two docs in TWO DISTINCT channels naming the same entity, so the entry
    # entity is MENTIONED_IN chunks spanning >= 2 distinct ``c.channel`` values
    # and the session-aware fan-out activates. ``get_entity_channels`` reads the
    # Chunk ``channel`` column, which is derived from document ``metadata["channel"]``
    # (engine.py: ``doc_metadata.get("channel")``) — NOT session_id — so the
    # discriminator to seed is the channel, set via per-doc metadata.
    await seed_corpus(
        lambda **kw: kb.remember(entity_types=["PERSON"], relationship_types=[], **kw),
        namespace_id,
        [
            {"content": "Ada Lovelace published her notes.", "metadata": {"channel": "chan-a"}},
            {"content": "Ada Lovelace presented again later.", "metadata": {"channel": "chan-b"}},
        ],
    )

    retriever = _retriever(kb)
    await _assert_entry_entities_present(retriever, "Ada Lovelace", namespace_id)
    captures = spy_on(monkeypatch, retriever, "_vector_search_chunks")

    # A temporal query is required for session-aware fan-out to engage.
    await kb.recall("what did Ada Lovelace do recently", namespace=namespace_id, filter=_SYSTEM_FILTER)
    assert_filter_threaded(captures, _SYSTEM_FILTER, min_calls=2)

    # CONTROL: mode=VECTOR skips the graph entry-entity path that drives fan-out,
    # so the per-session fan-out does not engage — proves the >= 2 above is the
    # fan-out firing, not an unconditional channel.
    control = spy_on(monkeypatch, retriever, "_vector_search_chunks")
    await kb.recall(
        "what did Ada Lovelace do recently",
        namespace=namespace_id,
        filter=_SYSTEM_FILTER,
        mode=SearchMode.VECTOR,
    )
    assert len(control) < 2, "control: mode=VECTOR must not fan out per-session searches"


# --------------------------------------------------------------------------- #
# (7) CHANGE decomposition — the current-state sub-query carries the filter.
# --------------------------------------------------------------------------- #


@_SKIP
@pytest.mark.asyncio
async def test_change_decomposition_threads_filter(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CHANGE query with version history runs a 2nd vector search that carries the filter.

    SUPERSEDES version history is written automatically by ``upsert_entities_batch``
    when an existing entity's attributes change across ingests: two ``remember()``
    calls naming the SAME entity with a CHANGED description close the old version
    (``version_valid_to``) and create the ``[:SUPERSEDES]`` edge, so
    ``_fetch_version_history`` returns rows. With a CHANGE-classified query, the
    decomposition fires a 2nd ``_vector_search_chunks`` for the current-state
    sub-query — so N >= 2 (original + decomposed), every capture matching.
    """

    # Same entity, CHANGED attrs across two ingests → SUPERSEDES edge in Neo4j.
    def rem(**kw):
        return kb.remember(entity_types=["ORG"], relationship_types=[], **kw)

    plan_extraction("Acme Corp", entities=[("Acme Corp", "ORG")])
    await seed_corpus(rem, namespace_id, ["Acme Corp is a hardware company."])
    plan_extraction("Acme Corp", entities=[("Acme Corp", "ORG")])
    await seed_corpus(rem, namespace_id, ["Acme Corp is now a cloud software company."])

    retriever = _retriever(kb)
    await _assert_entry_entities_present(retriever, "Acme Corp", namespace_id)

    # Sanity: the SUPERSEDES seed must produce version history, else the CHANGE
    # decomposition never fires and the >= 2 assertion is vacuous. Resolve the
    # public namespace_id to the row id (entities/version history are keyed by it).
    resolved = await retriever._storage.resolve_namespace(namespace_id)
    entry = await retriever._vector_search_entities(await retriever._embedder.embed("Acme Corp"), resolved, limit=10)
    version_history = await retriever._fetch_version_history([e[0] for e in entry], resolved)
    assert version_history, "SUPERSEDES seed produced no version history — CHANGE decomp would not fire"

    captures = spy_on(monkeypatch, retriever, "_vector_search_chunks")
    # "how has Acme Corp changed" → CHANGE category; _decompose_change_query
    # rewrites it, triggering the second vector search.
    await kb.recall("how has Acme Corp changed", namespace=namespace_id, filter=_SYSTEM_FILTER)
    assert_filter_threaded(captures, _SYSTEM_FILTER, min_calls=2)

    # CONTROL: a non-CHANGE query must not run the decomposition sub-search, so
    # the capture count is strictly lower — proves the extra sub-search above is
    # the CHANGE decomposition, not the always-on primary vector channel.
    control = spy_on(monkeypatch, retriever, "_vector_search_chunks")
    await kb.recall("who is Acme Corp", namespace=namespace_id, filter=_SYSTEM_FILTER)
    assert len(control) < len(captures), "control: non-CHANGE query must not add a decomposition sub-search"


# --------------------------------------------------------------------------- #
# (5) restrictive-fallback re-run guard — SUPPRESSED under a caller filter.
# --------------------------------------------------------------------------- #


def _is_unfiltered_rerun(record: object) -> bool:
    """Whether a ``_vector_search_chunks`` capture is the unfiltered re-run.

    The restrictive-fallback re-run is the only ``_vector_search_chunks`` call
    issued with BOTH ``temporal_filter=None`` AND no ``filter_ast`` — it drops
    both, which is precisely why it must not fire under a caller filter (it would
    smuggle filter-violating chunks into RRF). The primary vector channel always
    carries the caller filter, so this predicate isolates the re-run.
    """
    kwargs = record.kwargs  # type: ignore[attr-defined]
    return kwargs.get("temporal_filter") is None and record.canonical_hash is None  # type: ignore[attr-defined]


@_SKIP
@pytest.mark.asyncio
async def test_restrictive_fallback_rerun_suppressed_under_filter(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller filter SUPPRESSES the unfiltered restrictive-fallback re-run.

    When a temporal filter yields sparse results, the engine normally re-runs the
    vector channel WITHOUT the temporal filter (and without ``filter_ast``). That
    re-run is gated ``and filter_ast is None`` (retriever ``_vectorcypher_retrieve``):
    under a caller filter it must NOT fire, because re-searching unfiltered would
    leak filter-violating chunks. This is a NEGATIVE (suppression) assertion, so
    it is paired with a DIFFERENTIAL control proving the re-run IS reachable when
    no caller filter is present — otherwise "no re-run fired" is vacuously true.

    PG-specific: the embedded sqlite_lance stack skips point-in-time
    entity-version narrowing (it lacks the bi-temporal columns), so the
    version-filter-paired re-run path is exercised on the live PG+Neo4j stack —
    which is why this spy lives here, not in the embedded suite.

    IMPORTANT — arming the fallback. The re-run is gated on a populated
    ``temporal_filter`` (retriever ``_vectorcypher_retrieve``). The public
    ``filter=`` kwarg does NOT populate ``temporal_filter`` — a date predicate
    there stays in ``filter_ast`` and leaves ``temporal_filter`` None (see
    ``khora.recall``), so the fallback never arms. The engine populates
    ``temporal_filter`` by SYNTHESIS from a RECENCY signal. So the trigger is a
    recency QUERY (``"... recently"``), not a date filter; the seed is a single
    old chunk so the synthesized recency window is sparse. The caller filter
    (the thing whose presence must suppress the re-run) comes from ``filter=``,
    which is orthogonal to the synthesized ``temporal_filter`` — both can be set,
    unlike ``filter=`` vs the deprecated ``start_time/end_time`` (mutually
    exclusive).
    """
    plan_extraction("Ada Lovelace", entities=[("Ada Lovelace", "PERSON")])
    # One OLD chunk: a synthesized recency window then has < limit//2 in range,
    # arming the restrictive-fallback.
    await seed_corpus(
        lambda **kw: kb.remember(entity_types=["PERSON"], relationship_types=[], **kw),
        namespace_id,
        ["Ada Lovelace published her notes long ago."],
    )

    retriever = _retriever(kb)
    await _assert_entry_entities_present(retriever, "Ada Lovelace", namespace_id)

    # POSITIVE: a recency query SYNTHESIZES temporal_filter; with a caller filter
    # present, the unfiltered re-run must be SUPPRESSED (gate `filter_ast is None`).
    captures = spy_on(monkeypatch, retriever, "_vector_search_chunks")
    await kb.recall("what did Ada Lovelace do recently", namespace=namespace_id, filter=_SYSTEM_FILTER)
    assert not any(_is_unfiltered_rerun(c) for c in captures), (
        "restrictive-fallback re-run fired under a caller filter — it drops the "
        "filter and would leak filter-violating chunks (gate is `filter_ast is None`)"
    )

    # CONTROL: same recency query, NO caller filter → the re-run IS reachable, so
    # at least one unfiltered re-run capture appears. This proves the positive
    # assertion above is the guard working, not the path being dead.
    control = spy_on(monkeypatch, retriever, "_vector_search_chunks")
    await kb.recall("what did Ada Lovelace do recently", namespace=namespace_id)
    assert any(_is_unfiltered_rerun(c) for c in control), (
        "control: with no caller filter, the restrictive-fallback re-run MUST be "
        "reachable on the live PG stack — else the suppression test is vacuous"
    )
