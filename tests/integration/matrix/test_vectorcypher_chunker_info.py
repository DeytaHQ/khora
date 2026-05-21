"""End-to-end ``chunker_info`` propagation on the VectorCypher embedded path.

Pins the contract that the chunker's strategy identifier survives the
full remember → store → recall round-trip on the production embedded
stack (sqlite_lance + VectorCypher). The matching unit test in
``tests/unit/engines/vectorcypher/test_chunker_info_propagation.py``
verifies the engine-layer wiring; this test exercises the storage write
+ retriever read against a real (in-process) database and vector index.

Why this test exists: a regression that drops ``chunker_info`` would not
surface in the unit suite if the storage adapter quietly returns
``None`` or fails to project the new column on read. End-to-end gives
us confidence the entire chain — chunker → engine → temporal store →
LanceDB/SQLite → retriever → ``RecallChunk.chunker_info`` — is intact.

Shares no infrastructure with sibling tests beyond ``tmp_path``.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.extraction.extractors.base import ExtractionResult
from khora.extraction.skills import ExpertiseConfig
from khora.khora import Khora

EMBED_DIM = 32

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


# ---------------------------------------------------------------------------
# Deterministic embedder + no-op extractor stubs (no OPENAI_API_KEY needed)
# ---------------------------------------------------------------------------


def _embed_for(text_in: str) -> list[float]:
    seed = hashlib.sha256(text_in.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(EMBED_DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    # Entity extraction is irrelevant to this test; return empty results.
    return [ExtractionResult() for _ in texts]


@pytest.fixture(autouse=True)
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


# ---------------------------------------------------------------------------
# Per-test embedded Khora fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Khora]:
    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "khora.db"),
        lance_path=str(tmp_path / "khora.lance"),
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.neo4j_url = None
    config.pipelines.chunk_size = 1024
    # Disable extraction — chunker_info is independent of entity extraction.
    config.pipelines.extract_entities = False
    config.pipelines.selective_extraction = False

    kb = Khora(config, engine="vectorcypher", run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        try:
            await kb.disconnect()
        except Exception:
            pass


@pytest.fixture
async def namespace_id(kb: Khora) -> UUID:
    ns = await kb.create_namespace()
    return ns.namespace_id


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


async def test_chunker_info_survives_remember_recall_roundtrip(kb: Khora, namespace_id: UUID) -> None:
    """Every recalled chunk must carry a non-empty ``chunker_info`` with the strategy name."""
    await kb.remember(
        content="hello world from the VectorCypher embedded path",
        namespace=namespace_id,
        title="",
        entity_types=[],
        relationship_types=[],
        expertise=ExpertiseConfig(name="chunker-info-roundtrip"),
    )

    result = await kb.recall("hello", namespace=namespace_id, limit=10)

    assert result.chunks, "recall must return at least one chunk"
    for chunk in result.chunks:
        assert chunk.chunker_info, f"empty chunker_info on chunk {chunk.id}: {chunk.chunker_info!r}"
        assert "chunker" in chunk.chunker_info, (
            f"chunker_info missing 'chunker' key on chunk {chunk.id}: {chunk.chunker_info!r}"
        )
        # The configured default strategy is "recursive" (from KhoraConfig
        # defaults), but we accept any registered chunker — this test is
        # agnostic to which strategy the engine picked, only that the
        # identifier propagated.
        assert isinstance(chunk.chunker_info["chunker"], str)
        assert chunk.chunker_info["chunker"], "chunker name must be a non-empty string"
