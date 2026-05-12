"""Chronicle PostgreSQL integration tests.

Chronicle is one of khora's two production-ready engines but had zero
integration coverage on a real PostgreSQL stack. These tests wire up
``Khora(engine="chronicle")`` against ``khora-postgres`` (compose.yaml)
with stubbed LLM calls — no Neo4j, no OpenAI.

Why no Neo4j: Chronicle's four channels (semantic / BM25 / temporal / entity)
all live on pgvector + SQL columns. There is no graph backend in its storage
config, so it is correct that this file is PG-only.

How LLM calls are mocked:
* ``LLMEntityExtractor.extract_multi`` is replaced with a registry-based
  stub that emits a fixed entity list per content marker.
* ``LiteLLMEmbedder.embed_batch`` and ``embed`` return deterministic
  unit vectors of dimension 1536 (matches the ``chunks.embedding``
  ``Vector(1536)`` column hard-coded in migration 000).

How to run locally::

    make dev    # only postgres needed (compose.yaml uses port 5434)
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/matrix/test_chronicle_pg.py -v -m integration --no-cov
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
from khora.extraction.skills.base import EventExtractionConfig, FactExtractionConfig
from khora.khora import Khora

EMBED_DIM = 1536  # matches the chunks.embedding Vector(1536) column from migrations

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


# ---------------------------------------------------------------------------
# Fixtures: skip-if-no-PG, run-migrations-once, extraction stub
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


# Module-level extraction registry. ``plan_extraction`` stages the
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
async def _migrations_once() -> None:
    """Reset and migrate the live PG once for the module.

    Workaround for a pre-existing bug: alembic creates ``khora_alembic_version``
    with the default ``VARCHAR(32)``, but migration revision IDs like
    ``022_promote_external_id_index_unique`` are 38 chars and break the
    UPDATE that records each completed step. To compound the problem,
    migration 022 uses ``autocommit_block()`` for ``CREATE INDEX CONCURRENTLY``,
    so when the UPDATE fails the DB is left in a half-applied state with
    types and tables from migrations 000–021 committed. Recovering from
    that partial state requires a full schema reset.

    This fixture reproduces what ``DROP DATABASE`` would do but without
    needing superuser privileges: drops every relation and ENUM type in
    ``public``, then pre-creates ``khora_alembic_version`` with
    ``VARCHAR(64)``. The pre-existing bug is tracked in the PR description
    — the equivalent reset would also be needed by every other integration
    test on a half-applied DB.
    """
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.begin() as conn:
            # Drop every public ENUM type (their backing tables go with the
            # schema drop, but the types themselves survive a schema drop
            # only if their type-namespace is non-public; ours are public).
            r = await conn.execute(
                text("SELECT typname FROM pg_type WHERE typnamespace = 'public'::regnamespace AND typtype = 'e'")
            )
            for (typname,) in r.fetchall():
                await conn.execute(text(f"DROP TYPE IF EXISTS public.{typname} CASCADE"))
            # Wipe the schema. ``CASCADE`` removes tables, sequences, indexes,
            # functions — everything bound to it.
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # Pre-create the alembic version table with a wider column so
            # migration 022's UPDATE fits. Alembic's ``checkfirst=True``
            # honors the existing definition and won't shrink the column.
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


def _no_extraction_expertise() -> ExpertiseConfig:
    """Disable Chronicle's per-chunk event/fact extraction (extra LLM calls)."""
    return ExpertiseConfig(
        name="chronicle-pg-integ",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=False),
    )


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
async def kb(_migrations_once: None) -> AsyncIterator[Khora]:
    """Per-test Chronicle Khora bound to live PG.

    Function-scoped because the storage coordinator caches engine pools by
    URL; sharing across tests was tripping the autouse monkeypatch reset
    (the engine instance wires the embedder reference at ``connect()`` time).
    """
    config = KhoraConfig(database_url=DATABASE_URL)
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    # Chronicle's 4 channels are PG-only; no graph URL needed.
    config.neo4j_url = None
    # Single-chunk documents keep the test deterministic.
    config.pipelines.chunk_size = 1024
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False

    kb = Khora(config, engine="chronicle", run_migrations=False)
    await kb.connect()
    try:
        yield kb
    finally:
        await kb.disconnect()


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    ns = await kb.create_namespace()
    return ns.namespace_id


async def _remember(
    kb: Khora,
    *,
    namespace_id: UUID,
    content: str,
    title: str = "",
    expertise: ExpertiseConfig | None = None,
) -> Any:
    return await kb.remember(
        content=content,
        namespace=namespace_id,
        title=title,
        entity_types=["PERSON", "CONCEPT", "EVENT"],
        relationship_types=["KNOWS", "ATTENDED", "RELATES_TO"],
        expertise=expertise or _no_extraction_expertise(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_chronicle_remember_recall_roundtrip(kb: Khora, namespace_id: UUID) -> None:
    """Ingest 3 docs, recall, assert ingested text appears in context."""
    contents = [
        "Alice met Bob at the Python conference in Berlin on March 15th.",
        "Carol presented her research on graph databases at the same event.",
        "Dan organized the after-party that lasted until midnight.",
    ]
    for c in contents:
        await _remember(kb, namespace_id=namespace_id, content=c)

    result = await kb.recall("Python conference Berlin", namespace=namespace_id, limit=10)

    assert result.metadata.get("engine") == "chronicle"
    assert len(result.chunks) >= 1, "expected at least one chunk back"
    # The most-relevant ingested text should be visible to the LLM context.
    assert "Python conference" in result.context_text


async def test_chronicle_four_channels_contribute(kb: Khora, namespace_id: UUID) -> None:
    """Each applicable channel reports a non-zero hit count for a hybrid query."""
    _plan_extraction("Alice", entities=[("Alice", "PERSON"), ("Berlin", "LOCATION")])
    _plan_extraction("Carol", entities=[("Carol", "PERSON")])

    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Alice met Bob in Berlin at the Python conference.",
    )
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Carol gave a lightning talk on graph databases.",
    )

    result = await kb.recall("Alice Berlin", namespace=namespace_id, limit=10)

    channels = result.metadata["channels"]
    # Semantic + temporal always run on HYBRID. BM25 + entity also run for
    # MODERATE/COMPLEX/ENTITY_ANCHORED routing, which "Alice Berlin" triggers.
    assert channels["semantic"] >= 1, f"semantic channel empty: {channels}"
    assert channels["bm25"] >= 1, f"bm25 channel empty: {channels}"
    # Temporal is always invoked (chronicle's differentiator); count counts hits.
    # Even a fallback chunk-search path counts here.
    assert channels["temporal"] >= 1, f"temporal channel empty: {channels}"
    # Entity channel needs at least one extracted entity in the corpus —
    # we registered Alice/Berlin/Carol, and "Alice" is similar to the query.
    assert channels["entity"] >= 1, f"entity channel empty: {channels}"


async def test_chronicle_abstention_signals_on_topic(kb: Khora, namespace_id: UUID) -> None:
    """Query matches corpus → ``should_abstain`` is False, combined < 0.5."""
    _plan_extraction("Alice", entities=[("Alice", "PERSON")])
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Alice gave a keynote on distributed databases at the conference.",
    )

    result = await kb.recall("Alice keynote conference", namespace=namespace_id, limit=5)

    sig = result.metadata["abstention_signals"]
    assert sig["chunks_empty"] is False
    assert sig["should_abstain"] is False, f"unexpected abstention: {sig}"
    assert sig["combined_score"] < 0.5, f"combined too high: {sig}"


async def test_chronicle_abstention_signals_off_topic(kb: Khora, namespace_id: UUID) -> None:
    """Query unrelated to corpus → at least one weak-signal flag fires.

    With deterministic identical embeddings every chunk has cosine=1.0, so
    ``top_score_low`` cannot fire on score alone. The reliable always-on
    signal in this fixture is ``entities_empty`` (no extracted entities
    when the registry stub matches nothing). We assert that the engine
    populates the signals dict and that the weak retrieval surfaces in
    at least one of the three weighted flags.
    """
    # Deliberately don't register any extraction → no entities surface.
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="The weather report mentioned rain on Saturday.",
    )

    result = await kb.recall("quantum chromodynamics gauge symmetry", namespace=namespace_id, limit=5)

    sig = result.metadata["abstention_signals"]
    # Required keys present.
    for key in (
        "entities_empty",
        "chunks_empty",
        "chunks_below_min",
        "top_score_low",
        "combined_score",
        "should_abstain",
    ):
        assert key in sig, f"missing abstention signal: {key}"
    # Without entity extraction the entities side is empty → weak signal fires.
    assert sig["entities_empty"] is True


async def test_chronicle_temporal_filter_pushdown(
    kb: Khora, namespace_id: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backdate one chunk by 20 days, leave another at 5 days, query "last 7 days".

    Asserts the temporal filter reaches the SQL layer (via captured
    ``created_after`` kwarg on ``search_similar_chunks``) — this is the
    SQL-pushdown contract referenced in ADR/CLAUDE.md "Temporal SQL pushdown".

    NB: only the temporal channel forwards ``created_after`` today —
    the semantic + BM25 + entity channels still scan the full namespace,
    so the 20-day-old chunk leaks into the fused result. We assert the
    SQL pushdown contract here, and assert the 20-day exclusion as
    ``xfail(strict=True)`` to surface the gap loudly. See PR description.
    """
    r_recent = await _remember(kb, namespace_id=namespace_id, content="recent doc about Falcon launch.")
    r_old = await _remember(kb, namespace_id=namespace_id, content="old doc about Falcon launch.")

    # Backdate "old" chunks 20 days; "recent" stays at now (~5 days for test
    # determinism — though "now" already satisfies the 7-day filter).
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

    # Spy on coordinator.search_similar_chunks to confirm SQL-level pushdown.
    captured: list[dict[str, Any]] = []
    coord = kb._engine._storage  # type: ignore[union-attr]
    real = coord.search_similar_chunks

    async def _spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        return await real(*args, **kwargs)

    monkeypatch.setattr(coord, "search_similar_chunks", _spy)

    # Query with start_time = 7 days ago → Khora constructs a
    # SkeletonTemporalFilter with occurred_after, which Chronicle's temporal
    # channel forwards as created_after to search_similar_chunks (engine.py:1700-1733).
    seven_days_ago = datetime.now(UTC) - timedelta(days=7)
    await kb.recall(
        "Falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=seven_days_ago,
    )

    # Pushdown contract: at least one search_similar_chunks call carried
    # the created_after kwarg with our threshold.
    pushdown_calls = [c for c in captured if c.get("created_after") is not None]
    assert pushdown_calls, f"expected created_after to be pushed to SQL, captured kwargs={captured!r}"
    pushed = pushdown_calls[0]["created_after"]
    # Allow tz-naive (defensive coercion happens elsewhere); just compare epoch.
    if pushed.tzinfo is None:
        pushed = pushed.replace(tzinfo=UTC)
    assert pushed >= seven_days_ago - timedelta(seconds=1), (
        f"pushdown threshold drifted: pushed={pushed} expected>={seven_days_ago}"
    )

    # The pushdown contract is the load-bearing assertion here. Whether
    # the 20-day-old doc survives in the fused result depends on whether
    # all four channels honor the filter — see test_chronicle_temporal_old_doc_excluded.


async def test_chronicle_temporal_old_doc_excluded(kb: Khora, namespace_id: UUID) -> None:
    """Document older than the temporal_filter window must not appear in chunks."""
    r_recent = await _remember(kb, namespace_id=namespace_id, content="recent doc about Falcon launch.")
    r_old = await _remember(kb, namespace_id=namespace_id, content="old doc about Falcon launch.")

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
    result = await kb.recall(
        "Falcon launch",
        namespace=namespace_id,
        limit=10,
        start_time=seven_days_ago,
    )
    returned_doc_ids = {c.document_id for c, _ in result.chunks}
    assert r_old.document_id not in returned_doc_ids


async def test_chronicle_entity_anchored_routing(kb: Khora, namespace_id: UUID) -> None:
    """ "Who is Alice?" → ENTITY_ANCHORED routing classification.

    The router's classification depends on the heuristic match against the
    query string — "Who is X?" is the canonical entity-anchored pattern
    (see DYT-3147). When entity_anchored fires, the entity-channel RRF
    weight is doubled (engine.py:1232-1233). We can't easily inspect the
    weight without patching, so we assert the routing label only — the
    weight-doubling is covered by ``test_router_and_fusion.py``.
    """
    _plan_extraction("Alice", entities=[("Alice", "PERSON")])
    await _remember(
        kb,
        namespace_id=namespace_id,
        content="Alice is a senior researcher specializing in graph databases.",
    )

    result = await kb.recall("Who is Alice?", namespace=namespace_id, limit=5)

    assert result.metadata["routing"] == "entity_anchored", (
        f"expected entity_anchored, got {result.metadata['routing']!r}"
    )


async def test_chronicle_namespace_isolation(kb: Khora) -> None:
    """Two namespaces, queries don't cross-bleed."""
    ns_a = (await kb.create_namespace()).namespace_id
    ns_b = (await kb.create_namespace()).namespace_id

    await _remember(kb, namespace_id=ns_a, content="alpha document about kangaroos")
    await _remember(kb, namespace_id=ns_b, content="bravo document about penguins")

    result_a = await kb.recall("animals", namespace=ns_a, limit=10)
    result_b = await kb.recall("animals", namespace=ns_b, limit=10)

    a_text = " ".join(c.content for c, _ in result_a.chunks)
    b_text = " ".join(c.content for c, _ in result_b.chunks)

    assert "kangaroos" in a_text
    assert "penguins" not in a_text, "namespace_b content leaked into namespace_a"
    assert "penguins" in b_text
    assert "kangaroos" not in b_text, "namespace_a content leaked into namespace_b"


async def test_chronicle_recall_metadata_completeness(kb: Khora, namespace_id: UUID) -> None:
    """All RecallResult.metadata keys documented in CLAUDE.md must be present."""
    await _remember(kb, namespace_id=namespace_id, content="A simple sentence about apples.")

    result = await kb.recall("apples", namespace=namespace_id, limit=5)

    md = result.metadata
    expected_top_level = {
        "engine",
        "channels",
        "routing",
        "decay_weight",
        "max_raw_vector_score",
        "abstention_signals",
        "timings",
    }
    missing = expected_top_level - md.keys()
    assert not missing, f"missing top-level metadata keys: {missing}"

    # channels sub-dict structure
    assert set(md["channels"].keys()) == {"semantic", "bm25", "temporal", "entity"}

    # abstention_signals sub-dict structure (CLAUDE.md / contract)
    expected_signals = {
        "entities_empty",
        "chunks_empty",
        "chunks_below_min",
        "top_score_low",
        "combined_score",
        "should_abstain",
    }
    assert set(md["abstention_signals"].keys()) == expected_signals


async def test_chronicle_concurrent_remember(kb: Khora, namespace_id: UUID) -> None:
    """5 concurrent ingests in one namespace, no integrity errors."""
    contents = [f"document number {i} mentions widget-{i}" for i in range(5)]
    results = await asyncio.gather(
        *(_remember(kb, namespace_id=namespace_id, content=c) for c in contents),
        return_exceptions=True,
    )

    # No exceptions surfaced.
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"concurrent remember raised: {errors}"

    # Five distinct documents persisted.
    doc_ids = {r.document_id for r in results}  # type: ignore[union-attr]
    assert len(doc_ids) == 5, f"expected 5 distinct documents, got {doc_ids}"

    # All five recoverable via recall.
    result = await kb.recall("widget", namespace=namespace_id, limit=20)
    contents_returned = {c.content for c, _ in result.chunks}
    assert len(contents_returned) >= 5
