"""Live PG+Neo4j proof for #1459: VectorCypher chunk counts are non-zero.

On the default VectorCypher engine, chunks are written to the temporal store's
``khora_chunks`` table, not the relational ``chunks`` table. Before the fix,
``Khora.stats(ns).chunks`` and ``kb.storage.count_chunks(ns)`` both counted the
empty relational ``chunks`` table and reported ``0`` for a namespace whose
chunks were present and fully recallable — ``stats().documents`` was correct;
only the chunk count was wrong.

The fix attaches the engine's temporal store to the ``StorageCoordinator`` at
connect time and routes ``StorageCoordinator.count_chunks`` through it, so BOTH
``kb.storage.count_chunks()`` (the coordinator IS ``kb.storage``) and
``stats().chunks`` (via ``engines/_stats.gather_counts`` → ``count_chunks``)
report the true count.

This is the VectorCypher half of #1070 (which fixed the same symptom for the
Skeleton engine only). The sqlite_lance embedded half lives in
``tests/integration/matrix/test_vectorcypher_sqlite_lance.py``
(``test_vc_stats_chunk_count_nonzero_matches_temporal_store``).

How to run locally::

    make dev   # postgres (5434) + neo4j (bolt 7688) via docker compose
    NEO4J_INTEGRATION_TEST=1 KHORA_NEO4J_URL=bolt://localhost:7688 \\
        uv run pytest tests/integration/test_vectorcypher_stats_chunk_count_1459.py \\
        -v -m integration --no-cov
"""

from __future__ import annotations

import hashlib
import os
import socket
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import pytest

from khora.config import KhoraConfig
from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult
from khora.khora import Khora

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
# Reachability gates (mirrors test_vectorcypher_recency_channel_pg.py)
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


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _pg_reachable(),
        reason="PostgreSQL not reachable (run `make dev` first)",
    ),
]


# ---------------------------------------------------------------------------
# Deterministic embedder + extractor stubs (no OPENAI_API_KEY needed)
# ---------------------------------------------------------------------------


def _embed_for(text_in: str) -> list[float]:
    """Deterministic L2-normalised ``EMBED_DIM`` vector derived from SHA-256."""
    seed = hashlib.sha256(text_in.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(EMBED_DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    """One entity per chunk keeps extraction non-empty without an LLM."""
    return [
        ExtractionResult(entities=[ExtractedEntity(name="Photosynthesis", entity_type="CONCEPT", confidence=0.99)])
        for _ in texts
    ]


@pytest.fixture
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.fixture
async def kb_vc(_patch_llm: None) -> AsyncIterator[Khora]:
    """Connected VectorCypher Khora on live PG + Neo4j with a small chunk_size.

    ``chunk_size=30`` / ``chunk_overlap=0`` splits the probe document into
    several chunks so the count assertion rules out an off-by-one.

    Migrations are applied via ``run_migrations=True`` (advisory-locked and
    idempotent) rather than a destructive ``DROP SCHEMA`` reset — this test is
    fully namespace-isolated (it creates its own namespace and counts only its
    own chunks), so it is safe to run against a shared khora stack.
    """
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=_neo4j_url())
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.storage.postgresql_url = DATABASE_URL
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False
    config.pipelines.chunk_size = 30
    config.pipelines.chunk_overlap = 0

    instance = Khora(config, engine="vectorcypher", run_migrations=True)
    await instance.connect()
    try:
        yield instance
    finally:
        # This test exercises the connect/disconnect lifecycle the fix changes,
        # so a teardown failure is signal — do not swallow it (#1459 review).
        await instance.disconnect()


@pytest.mark.skipif(
    not _neo4j_reachable() or not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 and run `make dev` (needs Neo4j for the full vectorcypher path)",
)
async def test_vc_stats_chunk_count_nonzero_on_pg_neo4j(kb_vc: Khora) -> None:
    """stats().chunks and kb.storage.count_chunks() equal the true count (#1459).

    Ingests one multi-chunk document, confirms the chunks recall, then asserts
    BOTH counters equal ``RememberResult.chunks_created`` (the ground truth) —
    where before the fix both reported 0.
    """
    ns = await kb_vc.create_namespace()
    stable: UUID = ns.namespace_id

    doc = (
        "Photosynthesis converts light energy into chemical energy in plants. "
        "The light-dependent reactions occur in the thylakoid membranes. "
        "The Calvin cycle fixes carbon dioxide into glucose in the stroma. "
        "Chlorophyll absorbs light in the blue and red wavelengths. "
        "Oxygen is released as a byproduct when water molecules are split."
    )
    rem = await kb_vc.remember(
        content=doc,
        namespace=stable,
        entity_types=["CONCEPT"],
        relationship_types=[],
    )
    n_created = rem.chunks_created
    assert n_created > 1, f"expected a multi-chunk document, got chunks_created={n_created}"

    recalled = await kb_vc.recall("photosynthesis light energy", namespace=stable, limit=50)
    assert len(recalled.chunks) >= 1, "chunks must be recallable for the bug premise to hold"

    st = await kb_vc.stats(namespace=stable)
    assert st.documents == 1
    assert st.chunks == n_created, f"stats().chunks={st.chunks} != created {n_created} (#1459)"
    # ADR-001: a correct count must NOT have degraded through gather_counts.
    assert "errors" not in st.metadata

    resolved = await kb_vc.storage.resolve_namespace(stable)
    assert await kb_vc.storage.count_chunks(resolved) == n_created, (
        "kb.storage.count_chunks() must equal the true chunk count (#1459)"
    )
