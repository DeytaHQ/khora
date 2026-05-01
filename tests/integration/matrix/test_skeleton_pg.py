"""Skeleton PostgreSQL integration tests (DYT-3545).

Skeleton is the second of khora's two production-ready engines and (per the
DB-prod audit) had zero dedicated integration coverage. These tests wire up
``MemoryLake(engine="skeleton")`` against ``khora-postgres`` (compose.yaml)
with stubbed LLM calls — no Neo4j, no OpenAI.

Why no Neo4j: ``SkeletonConstructionEngine.__init__`` builds its storage
config with ``skip_graph=True`` (engine.py:88-89). The engine's only
backends are ``pgvector`` (default), ``weaviate``, and ``surrealdb`` — none
of them require a graph store. So the production stack subset for Skeleton
is **PostgreSQL + pgvector only**, and the ``_pg.py`` filename is correct.

How LLM calls are mocked:
* ``LiteLLMEmbedder.embed_batch`` and ``embed`` return content-derived
  unit vectors of dimension 1536 (matches the ``khora_chunks.embedding``
  ``Vector(1536)`` column hard-coded in ``backends/pgvector.py``). The
  vectors are derived from a simple keyword scheme so different documents
  have different similarities — this lets us exercise top-k ordering.
* Skeleton does **no** entity extraction (it is the cost-optimised engine
  by design), so no extractor stub is required.

How to run locally::

    make dev    # only postgres needed (compose.yaml uses port 5434)
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/matrix/test_skeleton_pg.py -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
import math
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
from khora.engines.skeleton.backends import TemporalFilter
from khora.memory_lake import MemoryLake
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
# Fixtures: skip-if-no-PG, run-migrations-once, embedder stub
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


# Keyword vocabulary used to build deterministic, content-aware embeddings.
# Each keyword maps to a slot in the 1536-dim vector; documents containing
# the keyword get a non-zero component there, so cosine similarity reflects
# keyword overlap. Slots stay well below 1536 so we don't risk index drift.
_KEYWORD_SLOTS: dict[str, int] = {
    # Test corpus keywords
    "alpha": 0,
    "bravo": 1,
    "charlie": 2,
    "delta": 3,
    "echo": 4,
    "kangaroo": 5,
    "kangaroos": 5,
    "penguin": 6,
    "penguins": 6,
    "widget": 7,
    "falcon": 8,
    "launch": 9,
    "rocket": 10,
    "tag": 11,
    "metadata": 12,
    "filter": 13,
    "concurrent": 14,
    "batch": 15,
    "bulk": 16,
    "recent": 17,
    "old": 18,
    "first": 19,
    "second": 20,
    "third": 21,
    "fourth": 22,
    "fifth": 23,
}


def _embed_for(text_in: str) -> list[float]:
    """Return a deterministic 1536-dim unit vector derived from ``text_in``.

    The embedding has a small constant baseline component so all-zero edge
    cases (queries that match no vocabulary) still get a defined vector.
    Matched keywords contribute equal weight, then the vector is L2-normalised
    so cosine similarity = ratio of shared keywords to ``sqrt(|A|*|B|)``.
    """
    vec = [0.0] * EMBED_DIM
    # Small constant component so we never produce a true zero vector.
    vec[EMBED_DIM - 1] = 0.01

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


@pytest.fixture(scope="module")
async def _migrations_once() -> None:
    """Reset and migrate the live PG once for the module.

    Mirrors the workaround documented in ``test_chronicle_pg.py``: alembic
    creates ``khora_alembic_version`` with the default ``VARCHAR(32)`` but
    migration revision IDs are wider. We pre-create the version table with
    ``VARCHAR(64)`` and wipe ``public`` (including ENUM types) before
    running migrations so the schema is in a known-good state.

    Skeleton ALSO creates its own ``khora_chunks`` table imperatively in
    ``PgVectorTemporalStore.connect()`` via ``metadata.create_all`` — that
    happens lazily on first ``lake.connect()``, so this fixture only needs
    to handle the alembic-managed core schema.
    """
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


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the embedder so no real LLM is called.

    Skeleton does no entity extraction, so the extractor doesn't need a stub.
    """
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )


@pytest.fixture
async def lake(_migrations_once: None) -> AsyncIterator[MemoryLake]:
    """Per-test Skeleton MemoryLake bound to live PG.

    Function-scoped to match the chronicle-pg pattern — the storage
    coordinator caches engine pools by URL but the engine instance wires
    the embedder reference at ``connect()`` time, which is incompatible
    with module-scoped autouse monkeypatching.
    """
    config = KhoraConfig(database_url=DATABASE_URL)
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    # Skeleton's pgvector backend is PG-only; no graph URL needed.
    config.neo4j_url = None
    # Belt-and-braces: explicitly set storage.postgresql_url too — environments
    # that have ``KHORA_DATABASE_URL`` already exported get sidestepped by
    # ``database_url=...``, but tests should also tolerate stray
    # ``KHORA_STORAGE_POSTGRESQL_URL`` env vars from local dev shells.
    config.storage.postgresql_url = DATABASE_URL
    # Single-chunk documents keep the test deterministic.
    config.pipelines.chunk_size = 1024

    lake = MemoryLake(config, engine="skeleton", run_migrations=False)
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
    metadata: dict[str, Any] | None = None,
) -> Any:
    return await lake.remember(
        content=content,
        namespace=namespace_id,
        title=title,
        metadata=metadata,
        # Skeleton accepts these for protocol compliance but doesn't use them.
        entity_types=["PERSON", "CONCEPT"],
        relationship_types=["RELATES_TO"],
    )


async def _recall(lake: MemoryLake, query: str, **kwargs: Any) -> Any:
    """Recall wrapper that pins ``mode=SearchMode.VECTOR`` to dodge DYT-3555.

    ``SkeletonConstructionEngine.recall`` references the non-existent
    ``SearchMode.KEYWORD`` member when ``hybrid_alpha is None and
    mode != SearchMode.VECTOR`` (engine.py:441). Until DYT-3555 is fixed,
    every Skeleton recall path exposed by ``MemoryLake`` blows up with
    ``AttributeError`` on its default ``HYBRID`` mode. Pinning mode to
    VECTOR short-circuits the buggy ``elif`` branch — the test
    ``test_skeleton_recall_default_hybrid_mode_bug`` documents the bug
    itself, the rest of the suite uses this workaround so we still
    exercise the real Skeleton ingest+retrieval surface.
    """
    kwargs.setdefault("mode", SearchMode.VECTOR)
    return await lake.recall(query, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_skeleton_remember_recall_roundtrip(lake: MemoryLake, namespace_id: UUID) -> None:
    """Ingest 3 docs, recall, assert ingested text appears in context."""
    contents = [
        "alpha document mentions the falcon launch in detail.",
        "bravo document covers a different rocket programme entirely.",
        "charlie document is a side note unrelated to anything else.",
    ]
    for c in contents:
        await _remember(lake, namespace_id=namespace_id, content=c)

    result = await _recall(lake, "falcon launch", namespace=namespace_id, limit=10)

    assert result.metadata.get("backend") == "pgvector"
    assert len(result.chunks) >= 1, "expected at least one chunk back"
    # The most-relevant ingested text should be visible in the context block.
    assert "falcon" in result.context_text.lower()


async def test_skeleton_namespace_isolation(lake: MemoryLake) -> None:
    """Two namespaces, queries don't cross-bleed."""
    ns_a = (await lake.create_namespace()).namespace_id
    ns_b = (await lake.create_namespace()).namespace_id

    await _remember(lake, namespace_id=ns_a, content="alpha document about kangaroos in the outback.")
    await _remember(lake, namespace_id=ns_b, content="bravo document about penguins on the ice.")

    result_a = await _recall(lake, "animals", namespace=ns_a, limit=10)
    result_b = await _recall(lake, "animals", namespace=ns_b, limit=10)

    a_text = " ".join(c.content for c, _ in result_a.chunks)
    b_text = " ".join(c.content for c, _ in result_b.chunks)

    assert "kangaroos" in a_text
    assert "penguins" not in a_text, "namespace_b content leaked into namespace_a"
    assert "penguins" in b_text
    assert "kangaroos" not in b_text, "namespace_a content leaked into namespace_b"


async def test_skeleton_recall_top_k_ordering(lake: MemoryLake, namespace_id: UUID) -> None:
    """Results ordered by descending similarity (combined_score)."""
    # Each doc shares a different number of keywords with the query, so
    # the deterministic embedder produces strictly decreasing similarities.
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha bravo charlie delta echo (high overlap with query)",
    )
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha bravo charlie (medium overlap with query)",
    )
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha (low overlap with query)",
    )

    result = await _recall(lake, "alpha bravo charlie delta echo", namespace=namespace_id, limit=10)

    assert len(result.chunks) >= 3
    scores = [score for _, score in result.chunks]
    # Strictly non-increasing — pgvector returns ORDER BY similarity DESC.
    for prev, curr in zip(scores, scores[1:]):
        assert prev >= curr, f"similarity ordering violated: {prev} < {curr} in {scores}"


@pytest.mark.xfail(
    strict=True,
    reason="DYT-3556: tags column is VARCHAR[] but filter literal lands as TEXT[]",
)
async def test_skeleton_recall_with_metadata_filter(lake: MemoryLake, namespace_id: UUID) -> None:
    """Tag filter restricts recall to chunks carrying the requested tag.

    NB: ``MemoryLake.recall()`` only forwards ``start_time``/``end_time`` to
    the engine (not arbitrary structured filters), so we drop down to the
    engine layer directly — the same way a downstream caller would if they
    needed metadata filtering today. If/when MemoryLake grows a ``filters``
    parameter, this test should switch to using it.

    See DYT-3556 — ``ARRAY(String).contains(...)`` compiles to a SQL clause
    that asyncpg can't execute against the ``character varying[]`` column,
    so the query effectively returns no rows. Test is xfail until fixed.
    """
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha document tagged group A",
        metadata={"tags": ["group-A"]},
    )
    await _remember(
        lake,
        namespace_id=namespace_id,
        content="alpha document tagged group B",
        metadata={"tags": ["group-B"]},
    )

    engine = lake._get_engine()  # type: ignore[attr-defined]
    # ``hybrid_alpha=1.0`` (pure vector) is set explicitly so we skip the
    # ``mode``-based defaulting at engine.py:438-444 that would otherwise
    # trip DYT-3555 (SearchMode.KEYWORD AttributeError).
    result = await engine.recall(
        "alpha document",
        namespace_id,
        limit=10,
        temporal_filter=TemporalFilter(tags=["group-A"]),
        hybrid_alpha=1.0,
    )

    assert len(result.chunks) >= 1
    for chunk, _score in result.chunks:
        assert "group A" in chunk.content, f"non-group-A leaked in: {chunk.content!r}"
    assert all("group B" not in c.content for c, _ in result.chunks), "group-B chunk leaked through tag filter"


async def test_skeleton_temporal_filter(lake: MemoryLake, namespace_id: UUID) -> None:
    """Two docs with ``occurred_at`` 5d vs 20d apart, "last 7 days" → recent only.

    Skeleton.remember (single-doc) ignores ``metadata['occurred_at']``
    (DYT-3557 — only ``remember_batch`` reads it), so we backdate via direct
    SQL after ingest, mirroring the chronicle-pg pattern. This isolates the
    test to the question we actually care about: does
    ``TemporalFilter.occurred_after`` reach the storage layer and gate the
    result correctly?
    """
    r_recent = await _remember(
        lake,
        namespace_id=namespace_id,
        content="recent document about the falcon launch",
    )
    r_old = await _remember(
        lake,
        namespace_id=namespace_id,
        content="old document about the falcon launch",
    )

    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.begin() as conn:
            await conn.execute(
                text("UPDATE khora_chunks SET occurred_at = NOW() - INTERVAL '20 days' WHERE document_id = :doc_id"),
                {"doc_id": r_old.document_id},
            )
            await conn.execute(
                text("UPDATE khora_chunks SET occurred_at = NOW() - INTERVAL '5 days' WHERE document_id = :doc_id"),
                {"doc_id": r_recent.document_id},
            )
    finally:
        await eng.dispose()

    seven_days_ago = datetime.now(UTC) - timedelta(days=7)
    result = await _recall(
        lake,
        "falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=seven_days_ago,
    )

    returned_doc_ids = {c.document_id for c, _ in result.chunks}
    assert r_old.document_id not in returned_doc_ids, (
        f"20-day-old document leaked through occurred_after filter; returned doc_ids={returned_doc_ids}"
    )
    # The recent doc should still surface.
    assert any("recent" in c.content for c, _ in result.chunks), "recent doc not returned"


async def test_skeleton_remember_batch(lake: MemoryLake, namespace_id: UUID) -> None:
    """Bulk-ingest 20 docs in a single ``remember_batch`` call."""
    documents = [
        {
            "content": f"batch document number {i} contains widget-{i} content",
            "title": f"doc-{i}",
        }
        for i in range(20)
    ]
    batch = await lake.remember_batch(
        documents,
        namespace=namespace_id,
        entity_types=["PERSON"],
        relationship_types=["RELATES_TO"],
    )

    assert batch.processed == 20, f"expected 20 processed, got {batch}"
    assert batch.failed == 0, f"unexpected failures: {batch}"
    assert batch.chunks >= 20, f"expected ≥20 chunks (one per doc), got {batch.chunks}"

    # All 20 should be queryable.
    result = await _recall(lake, "widget batch document", namespace=namespace_id, limit=25)
    contents_returned = {c.content for c, _ in result.chunks}
    assert len(contents_returned) >= 20, f"expected ≥20 distinct chunks returned, got {len(contents_returned)}"


async def test_skeleton_recall_empty_namespace(lake: MemoryLake) -> None:
    """Recall against an empty namespace returns an empty chunks list."""
    ns = (await lake.create_namespace()).namespace_id

    result = await _recall(lake, "anything at all", namespace=ns, limit=10)

    assert result.chunks == []
    assert result.entities == []  # Skeleton never returns entities anyway.
    # Metadata should still be populated (engine identity, backend, etc.).
    assert result.metadata.get("backend") == "pgvector"


async def test_skeleton_recall_metadata_keys(lake: MemoryLake, namespace_id: UUID) -> None:
    """RecallResult.metadata exposes the keys the Skeleton engine documents."""
    await _remember(lake, namespace_id=namespace_id, content="alpha simple sentence")

    result = await _recall(lake, "alpha", namespace=namespace_id, limit=5)

    md = result.metadata
    # Skeleton populates these three keys at engine.py:485-489.
    expected = {"backend", "hybrid_alpha", "temporal_filter"}
    missing = expected - md.keys()
    assert not missing, f"missing skeleton metadata keys: {missing}"
    assert md["backend"] == "pgvector"
    # ``mode=VECTOR`` makes the engine default ``hybrid_alpha`` to 1.0.
    assert md["hybrid_alpha"] == 1.0
    # No temporal filter was applied here.
    assert md["temporal_filter"] is None


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="DYT-3555: Skeleton.recall references SearchMode.KEYWORD, which doesn't exist",
)
async def test_skeleton_recall_default_hybrid_mode_bug(lake: MemoryLake, namespace_id: UUID) -> None:
    """Default ``MemoryLake.recall(...)`` against Skeleton crashes — see DYT-3555.

    ``SkeletonConstructionEngine.recall`` (engine.py:441) does
    ``elif mode == SearchMode.KEYWORD`` but the enum has no such member,
    so any non-VECTOR mode (HYBRID is the MemoryLake default) raises
    ``AttributeError`` before we ever reach the temporal store. This test
    documents the regression so the fix can flip it from xfail to pass.
    """
    await _remember(lake, namespace_id=namespace_id, content="alpha simple sentence")
    # Note: NOT calling ``_recall`` — this exercises the buggy default path.
    await lake.recall("alpha", namespace=namespace_id, limit=5)


async def test_skeleton_concurrent_remember(lake: MemoryLake, namespace_id: UUID) -> None:
    """5 concurrent ingests in one namespace, no integrity errors."""
    contents = [f"document number {i} mentions widget-{i}" for i in range(5)]
    results = await asyncio.gather(
        *(_remember(lake, namespace_id=namespace_id, content=c) for c in contents),
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"concurrent remember raised: {errors}"

    # Five distinct documents persisted.
    doc_ids = {r.document_id for r in results}  # type: ignore[union-attr]
    assert len(doc_ids) == 5, f"expected 5 distinct documents, got {doc_ids}"

    # All five recoverable via recall.
    result = await _recall(lake, "widget", namespace=namespace_id, limit=20)
    contents_returned = {c.content for c, _ in result.chunks}
    assert len(contents_returned) >= 5
