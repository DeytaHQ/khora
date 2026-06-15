"""Live-PG proof for the VectorCypher recency channel (GitHub issue #1182).

Two layers of coverage:

(a) STORE-LEVEL — ``PgVectorTemporalStore.search_recent_chunks`` against live
    Postgres. Proves the recency SQL orders on the 3-way
    ``COALESCE(occurred_at, source_timestamp, created_at) DESC`` axis, honors
    ``limit`` and the ``created_after`` floor, isolates by namespace, and — the
    LOAD-BEARING regression guard — selects the embedding column so every
    returned ``Chunk`` carries a non-None ``.embedding`` (the recency channel
    drops embedding-less chunks before the cosine gate, so a missing embedding
    column would silently empty the channel).

(b) CHANNEL-LEVEL — a full ``Khora(engine="vectorcypher")`` recall with
    ``temporal_recency_channel_enabled=True`` and a RECENCY-classified query
    under a caller filter. Asserts ``engine_info["filter"]`` records the
    ``"recency"`` channel (pushing every filter leaf into the khora_chunks SQL,
    GitHub issue #1223) and that every returned chunk satisfies the filter. The
    recency ChannelPlan is recorded ONLY when surviving recency candidates GATE
    in RRF, so a green assertion proves the channel fired end-to-end (never
    vacuous).

(c) CHANNEL-LEVEL no-leak — a full recall under a ``source_name`` ``$ne``
    filter that a RECENT chunk VIOLATES. Proves the violating recent chunk is
    absent from results and ``engine_info["filter"]`` is honest (``source_name``
    pushed, ``unenforced_keys == []``) — the GitHub issue #1223 regression guard
    at the engine boundary.

    PATH CHOSEN: the full ``Khora.recall()`` path. ``VectorCypherEngine.connect``
    verifies Neo4j connectivity for the PG backend, so the recency channel only
    runs inside ``_vectorcypher_retrieve`` — which needs a graph-seeded entry
    entity. Hence (b) gates on BOTH Postgres AND Neo4j reachability; (a) needs
    only Postgres. ``mode=SearchMode.GRAPH`` forces ``_vectorcypher_retrieve``
    deterministically (HYBRID would let the router classify a short query as
    SIMPLE and fall to the recency-less ``_simple_retrieve``).

How to run locally::

    make dev   # postgres (5434) + neo4j (bolt 7688) via docker compose
    NEO4J_INTEGRATION_TEST=1 KHORA_NEO4J_URL=bolt://localhost:7688 \\
        uv run pytest tests/integration/test_vectorcypher_recency_channel_pg.py \\
        -v -m integration --no-cov
"""

from __future__ import annotations

import math
import os
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from khora.config import KhoraConfig
from khora.db.session import run_migrations
from khora.engines.skeleton.backends import TemporalChunk
from khora.engines.skeleton.backends.pgvector import PgVectorTemporalStore
from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult
from khora.khora import Khora
from khora.query import SearchMode

EMBED_DIM = 1536  # matches the khora_chunks.embedding Vector(1536) column

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


# ---------------------------------------------------------------------------
# Reachability gates (mirrors tests/integration/matrix/test_skeleton_pg.py)
# ---------------------------------------------------------------------------


def _pg_reachable() -> bool:
    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _pg_reachable(),
        reason="PostgreSQL not reachable (run `make dev` first)",
    ),
]


# ---------------------------------------------------------------------------
# Deterministic keyword embedder stub (mirrors test_skeleton_pg.py).
#
# Each keyword maps to a fixed slot in the 1536-dim vector; chunks containing a
# keyword get a unit component there, so cosine similarity reflects keyword
# overlap. The query embedding shares the freshest chunk's keyword, so the
# fresh chunk clears ``temporal_query_relevance_floor`` in the cosine gate.
# ---------------------------------------------------------------------------
_KEYWORD_SLOTS: dict[str, int] = {
    "falcon": 0,
    "launch": 1,
    "rocket": 2,
    "recent": 3,
    "old": 4,
    "alpha": 5,
    "bravo": 6,
    "charlie": 7,
}


def _embed_for(text_in: str) -> list[float]:
    """Deterministic 1536-dim unit vector derived from ``text_in``."""
    vec = [0.0] * EMBED_DIM
    vec[EMBED_DIM - 1] = 0.01  # baseline so empty inputs aren't zero vectors
    lower = text_in.lower()
    for kw, slot in _KEYWORD_SLOTS.items():
        if kw in lower:
            vec[slot] = 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


# ---------------------------------------------------------------------------
# Migrations-once fixture (mirrors test_skeleton_pg.py). Needed for the
# channel-level test (full Khora wires the alembic-managed core schema); the
# PgVectorTemporalStore creates its own ``khora_chunks`` table imperatively on
# connect(), so the store-level test does not depend on this.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def _migrations_once() -> None:
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


# ===========================================================================
# (a) STORE-LEVEL — PgVectorTemporalStore.search_recent_chunks on live PG
# ===========================================================================


@pytest.fixture
async def store() -> AsyncIterator[PgVectorTemporalStore]:
    """A connected PgVectorTemporalStore.

    ``connect()`` runs ``metadata.create_all`` for ``khora_chunks`` plus the
    recency / HNSW / GIN indexes, so this fixture does NOT need the alembic
    chain — the temporal store owns its table.
    """
    config = KhoraConfig(database_url=DATABASE_URL)
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.storage.postgresql_url = DATABASE_URL
    store = PgVectorTemporalStore(config)
    await store.connect()
    try:
        yield store
    finally:
        await store.disconnect()


def _temporal_chunk(
    *,
    namespace_id: UUID,
    content: str,
    occurred_at: datetime | None = None,
    source_timestamp: datetime | None = None,
    created_at: datetime | None = None,
) -> TemporalChunk:
    return TemporalChunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=content,
        embedding=_embed_for(content),
        occurred_at=occurred_at,
        source_timestamp=source_timestamp,
        created_at=created_at,
    )


async def test_store_search_recent_chunks_recency_order_and_embedding(
    store: PgVectorTemporalStore,
) -> None:
    """Recency-ordered on the COALESCE axis; every Chunk carries an embedding."""
    ns = uuid4()
    now = datetime.now(UTC)

    # Three chunks exercising each leg of COALESCE(occurred_at, source_timestamp,
    # created_at): the recent one uses occurred_at, the middle uses
    # source_timestamp (occurred_at NULL), the old one falls back to created_at.
    recent = _temporal_chunk(namespace_id=ns, content="recent falcon launch", occurred_at=now - timedelta(days=1))
    middle = _temporal_chunk(
        namespace_id=ns,
        content="middle falcon note",
        occurred_at=None,
        source_timestamp=now - timedelta(days=10),
    )
    old = _temporal_chunk(
        namespace_id=ns,
        content="old falcon archive",
        occurred_at=None,
        source_timestamp=None,
        created_at=now - timedelta(days=30),
    )
    await store.create_chunks_batch([old, middle, recent])

    results = await store.search_recent_chunks(ns, limit=10)

    # Recency-ordered: recent before middle before old, on the COALESCE axis.
    returned_ids = [chunk.id for chunk, _sim in results]
    assert returned_ids == [recent.id, middle.id, old.id], (
        f"recency order violated on the COALESCE axis: {returned_ids}"
    )
    # Every tuple carries the None similarity sentinel.
    assert all(sim is None for _chunk, sim in results)
    # LOAD-BEARING: the embedding column was selected, so .embedding survives.
    for chunk, _sim in results:
        assert chunk.embedding is not None, f"chunk {chunk.id} returned with embedding=None"
        assert len(chunk.embedding) == EMBED_DIM


async def test_store_search_recent_chunks_honors_limit(store: PgVectorTemporalStore) -> None:
    """``limit`` caps the returned rows to the most-recent N."""
    ns = uuid4()
    now = datetime.now(UTC)
    chunks = [
        _temporal_chunk(namespace_id=ns, content=f"falcon entry {i}", occurred_at=now - timedelta(days=i))
        for i in range(5)
    ]
    await store.create_chunks_batch(chunks)

    results = await store.search_recent_chunks(ns, limit=2)

    assert len(results) == 2
    # The two most-recent (smallest day offset) come back first.
    assert [c.id for c, _ in results] == [chunks[0].id, chunks[1].id]


async def test_store_search_recent_chunks_created_after_excludes_old(
    store: PgVectorTemporalStore,
) -> None:
    """``created_after`` narrows on the same COALESCE axis, excluding old rows."""
    ns = uuid4()
    now = datetime.now(UTC)
    recent = _temporal_chunk(namespace_id=ns, content="recent falcon", occurred_at=now - timedelta(days=2))
    old = _temporal_chunk(namespace_id=ns, content="old falcon", occurred_at=now - timedelta(days=40))
    await store.create_chunks_batch([recent, old])

    cutoff = now - timedelta(days=7)
    results = await store.search_recent_chunks(ns, limit=10, created_after=cutoff)

    returned_ids = {c.id for c, _ in results}
    assert recent.id in returned_ids
    assert old.id not in returned_ids, "created_after floor did not exclude the 40-day-old chunk"


async def test_store_search_recent_chunks_namespace_isolation(store: PgVectorTemporalStore) -> None:
    """Chunks in a second namespace never appear in the first namespace's results."""
    ns_a = uuid4()
    ns_b = uuid4()
    now = datetime.now(UTC)

    a_chunks = [
        _temporal_chunk(namespace_id=ns_a, content=f"falcon alpha {i}", occurred_at=now - timedelta(hours=i))
        for i in range(3)
    ]
    b_chunks = [
        _temporal_chunk(namespace_id=ns_b, content=f"falcon bravo {i}", occurred_at=now - timedelta(hours=i))
        for i in range(3)
    ]
    await store.create_chunks_batch(a_chunks + b_chunks)

    results_a = await store.search_recent_chunks(ns_a, limit=50)
    returned_a_ids = {c.id for c, _ in results_a}
    b_ids = {c.id for c in b_chunks}

    assert returned_a_ids == {c.id for c in a_chunks}, "namespace A results are not exactly A's chunks"
    assert not (returned_a_ids & b_ids), "namespace B chunks leaked into namespace A results"


# ===========================================================================
# (b) CHANNEL-LEVEL — full Khora.recall() proves the recency channel fires
#                     end-to-end and records its ChannelPlan under a filter.
# ===========================================================================


def _neo4j_url() -> str:
    return os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688")


def _neo4j_reachable() -> bool:
    parsed = urlparse(_neo4j_url())
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


_GRAPH_ENTITY_NAME = "Falcon"
_GRAPH_MARKER = "graphdoc"


async def _stub_extract_multi_with_entity(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    """Emit the shared entity for marker-carrying docs (real MENTIONED_IN edges)."""
    out: list[ExtractionResult] = []
    for text_in in texts:
        if _GRAPH_MARKER in text_in:
            out.append(
                ExtractionResult(
                    entities=[ExtractedEntity(name=_GRAPH_ENTITY_NAME, entity_type="PERSON", confidence=0.99)]
                )
            )
        else:
            out.append(ExtractionResult())
    return out


@pytest.fixture
def _patch_llm_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub embedder + extractor for the channel-level test (no external LLM)."""
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
        _stub_extract_multi_with_entity,
    )


@pytest.fixture
async def kb_vc(_migrations_once: None, _patch_llm_channel: None) -> AsyncIterator[Khora]:
    """Connected VectorCypher Khora (live PG + Neo4j) with the recency channel ON."""
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=_neo4j_url())
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.storage.postgresql_url = DATABASE_URL
    config.pipeline.extract_entities = True
    config.pipeline.selective_extraction = False
    config.query.enable_reranking = False
    config.query.enable_llm_reranking = False
    config.query.min_entity_similarity = 0.0
    # The feature under test. Floor synthesis stays OFF (default) so the recency
    # channel runs with temporal_filter=None and is not vetoed.
    config.query.temporal_recency_channel_enabled = True
    config.query.temporal_query_relevance_floor = 0.30
    config.pipelines.chunk_size = 1024  # single-chunk docs keep the test deterministic

    instance = Khora(config, engine="vectorcypher", run_migrations=False)
    await instance.connect()
    try:
        yield instance
    finally:
        try:
            await instance.disconnect()
        except Exception:
            pass


@pytest.mark.skipif(
    not _neo4j_reachable() or not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 and run `make dev` (needs Neo4j for the full vectorcypher path)",
)
async def test_recency_channel_records_plan_end_to_end(kb_vc: Khora) -> None:
    """A RECENCY recall under a caller filter records the ``"recency"`` channel
    in ``engine_info["filter"]`` and every returned chunk satisfies the filter.

    Non-vacuous by construction: the recency ChannelPlan is recorded ONLY when
    post-filtered recency candidates SURVIVE. The freshest doc carries the
    filter-matching ``tag="urgent"``; older docs carry ``tag="noise"``. All docs
    mention the shared entity (real MENTIONED_IN edges) and share the query's
    ``falcon`` keyword (cosine 1.0, clearing the relevance floor), so the
    recency channel post-filters the urgent doc through and gates it in RRF.
    """
    ns = await kb_vc.create_namespace()
    namespace_id: UUID = ns.namespace_id

    # Older "noise" docs — relevant + recent enough to enter the recency channel,
    # but excluded by the filter (proves the post-filter is real).
    for i in range(3):
        await kb_vc.remember(
            content=f"{_GRAPH_ENTITY_NAME} {_GRAPH_MARKER} old launch note {i}: alpha bravo charlie.",
            namespace=namespace_id,
            title=f"noise-doc-{i}",
            metadata={"tag": "noise"},
            entity_types=["PERSON"],
            relationship_types=[],
        )
    # The freshest, filter-matching doc.
    r_fresh = await kb_vc.remember(
        content=f"{_GRAPH_ENTITY_NAME} {_GRAPH_MARKER} recent launch update: alpha bravo charlie.",
        namespace=namespace_id,
        title="urgent-doc",
        metadata={"tag": "urgent"},
        entity_types=["PERSON"],
        relationship_types=[],
    )

    # Backdate occurred_at on the temporal store table so the freshest doc is
    # unambiguously the newest on the recency axis (single-doc remember does not
    # stamp occurred_at; mirror the skeleton-pg pattern via direct SQL).
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.begin() as conn:
            await conn.execute(
                text("UPDATE khora_chunks SET occurred_at = NOW() - INTERVAL '20 days' WHERE namespace_id = :ns"),
                {"ns": str(namespace_id)},
            )
            await conn.execute(
                text(
                    "UPDATE khora_chunks SET occurred_at = NOW() - INTERVAL '1 day' "
                    "WHERE namespace_id = :ns AND document_id = :doc"
                ),
                {"ns": str(namespace_id), "doc": str(r_fresh.document_id)},
            )
    finally:
        await eng.dispose()

    # mode=GRAPH forces _vectorcypher_retrieve (where the recency channel lives);
    # the RECENCY-classified query drives the recency path; the caller filter
    # restricts to the urgent doc.
    result = await kb_vc.recall(
        "latest falcon launch update",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.GRAPH,
        filter={"metadata.tag": {"$in": ["urgent"]}},
    )

    filter_report = result.engine_info["filter"]
    channels = filter_report["channels"]
    assert "recency" in channels, (
        "the recency channel did not record a ChannelPlan — it never produced "
        f"surviving post-filtered candidates; channels={list(channels)}, "
        f"engine_info.filter={filter_report}"
    )
    # The recency channel now compiles the filter into the khora_chunks WHERE
    # (the SAME raise-mode pushdown the vector path uses — GitHub issue #1223),
    # so every leaf is pushed and nothing is re-checked in memory.
    assert channels["recency"]["pushed_keys"] == ["metadata.tag"]
    assert channels["recency"]["post_filtered_keys"] == []

    # Every returned chunk must satisfy the filter (tag == "urgent"). The tag
    # is carried on the chunk's parent document (``DocumentProjection.metadata``),
    # reached via ``RecallChunk.document_id`` into ``result.documents`` — the
    # response projection does not duplicate metadata onto each chunk.
    assert result.chunks, "expected at least the urgent chunk to survive the filter"
    docs_by_id = {doc.id: doc for doc in result.documents}
    for chunk in result.chunks:
        doc = docs_by_id.get(chunk.document_id)
        assert doc is not None, (
            f"chunk {chunk.id} references document {chunk.document_id} missing from result.documents"
        )
        tag = (doc.metadata or {}).get("tag")
        assert tag == "urgent", f"filter-violating chunk leaked through: tag={tag!r}"


# ===========================================================================
# (c) CHANNEL-LEVEL no-leak — GitHub issue #1223 regression guard.
#     A RECENT chunk whose ``source_name`` VIOLATES a ``$ne`` filter must never
#     reach results, and the report must credit ``source_name`` as pushed.
# ===========================================================================


_LEAK_SOURCE = "leakdoc"
_CLEAN_SOURCE = "cleandoc"


@pytest.mark.skipif(
    not _neo4j_reachable() or not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 and run `make dev` (needs Neo4j for the full vectorcypher path)",
)
async def test_recency_channel_source_name_filter_no_leak(kb_vc: Khora) -> None:
    """A RECENT chunk whose ``source_name`` violates ``$ne`` must be ABSENT.

    Regression guard for GitHub issue #1223 at the engine boundary: the
    freshest doc carries ``source_name=="leakdoc"`` (newest on the recency
    axis, so it would top the recency candidate list) but VIOLATES the caller
    filter ``{"source_name": {"$ne": "leakdoc"}}``. A slightly-older doc carries
    ``source_name=="cleandoc"`` and satisfies the filter. Both mention the
    shared entity and share the query's ``falcon`` keyword.

    Pre-fix the recency channel post-filtered a provenance-blank ``Chunk`` (no
    ``source_name``), so ``$ne`` matched-all and the leak doc surfaced. Post-fix
    the filter compiles into the khora_chunks WHERE, so the leak doc is never
    fetched. Asserts: (a) no leak-doc chunk in results, (b) the report is honest
    — ``source_name`` pushed and ``unenforced_keys == []``, (c) the recency
    channel actually fired (the clean doc survives and records the plan).
    """
    ns = await kb_vc.create_namespace()
    namespace_id: UUID = ns.namespace_id

    # The clean doc satisfies the filter; seed it first (older on the axis).
    r_clean = await kb_vc.remember(
        content=f"{_GRAPH_ENTITY_NAME} {_GRAPH_MARKER} clean launch note: alpha bravo charlie.",
        namespace=namespace_id,
        title="clean-doc",
        source_name=_CLEAN_SOURCE,
        entity_types=["PERSON"],
        relationship_types=[],
    )
    # The leak doc VIOLATES the filter and is the freshest on the recency axis.
    r_leak = await kb_vc.remember(
        content=f"{_GRAPH_ENTITY_NAME} {_GRAPH_MARKER} recent launch update: alpha bravo charlie.",
        namespace=namespace_id,
        title="leak-doc",
        source_name=_LEAK_SOURCE,
        entity_types=["PERSON"],
        relationship_types=[],
    )

    # Backdate occurred_at so the leak doc is unambiguously the newest (top of
    # the recency axis) and the clean doc is recent but second — mirroring the
    # direct-SQL pattern test (b) uses.
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.begin() as conn:
            await conn.execute(
                text("UPDATE khora_chunks SET occurred_at = NOW() - INTERVAL '5 days' WHERE namespace_id = :ns"),
                {"ns": str(namespace_id)},
            )
            await conn.execute(
                text(
                    "UPDATE khora_chunks SET occurred_at = NOW() - INTERVAL '1 day' "
                    "WHERE namespace_id = :ns AND document_id = :doc"
                ),
                {"ns": str(namespace_id), "doc": str(r_leak.document_id)},
            )
    finally:
        await eng.dispose()

    result = await kb_vc.recall(
        "latest falcon launch update",
        namespace=namespace_id,
        limit=20,
        mode=SearchMode.GRAPH,
        filter={"source_name": {"$ne": _LEAK_SOURCE}},
    )

    # (a) The violating leak doc must be absent from results.
    docs_by_id = {doc.id: doc for doc in result.documents}
    for chunk in result.chunks:
        doc = docs_by_id.get(chunk.document_id)
        assert doc is not None, (
            f"chunk {chunk.id} references document {chunk.document_id} missing from result.documents"
        )
        assert doc.source_name != _LEAK_SOURCE, (
            f"GitHub issue #1223 regression: filter-violating recent chunk leaked through "
            f"(source_name={doc.source_name!r}, doc={chunk.document_id})"
        )
    assert r_leak.document_id not in {c.document_id for c in result.chunks}

    # (b) The report is honest: source_name pushed, nothing unenforced.
    filter_report = result.engine_info["filter"]
    assert "source_name" in filter_report["pushed_keys"], f"source_name not credited as pushed: {filter_report}"
    assert filter_report["unenforced_keys"] == [], f"a filter leaf went unenforced — honesty violation: {filter_report}"

    # (c) The recency channel actually fired (non-vacuous): the clean doc
    # survived and the recency channel recorded its pushed-down ChannelPlan.
    channels = filter_report["channels"]
    assert "recency" in channels, (
        f"recency channel did not fire — no surviving candidate gated in RRF; channels={list(channels)}"
    )
    assert channels["recency"]["pushed_keys"] == ["source_name"]
    assert channels["recency"]["post_filtered_keys"] == []
    assert r_clean.document_id in {c.document_id for c in result.chunks}
