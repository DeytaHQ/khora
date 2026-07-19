"""Embedded (sqlite_lance) coverage for #1528: NUL-byte sanitization.

PostgreSQL text/jsonb columns reject ``0x00`` and abort the INSERT with
``asyncpg.CharacterNotInRepertoireError``. The fix strips NUL bytes at the
ingestion boundary (document content) and from extracted entity/relationship
text as it is staged into the storage models, so every backend receives clean
data.

The embedded SQLite/LanceDB stack does not crash on a NUL the way PostgreSQL
does, so this leg is a *coverage* proof that the sanitization runs regardless
of backend: after ``remember()`` of a document whose content AND whose
extracted entity (name/description/attributes) both carry ``\\x00``, the stored
chunk text and entity fields must be NUL-free and the document recallable. The
crash-repro half lives in ``tests/integration/test_nul_sanitization_pg_neo4j.py``.

How to run locally::

    uv run pytest tests/integration/matrix/test_nul_sanitization_sqlite_lance.py \\
        -v -m integration --no-cov

No Docker / Postgres / Neo4j needed — pure in-process SQLite + LanceDB.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
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
from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult
from khora.khora import Khora

EMBED_DIM = 32  # sqlite_lance default


def _assert_no_nul(value: Any) -> None:
    """Recursively assert no string in ``value`` contains a NUL byte."""
    if isinstance(value, str):
        assert "\x00" not in value, f"NUL byte found in {value!r}"
    elif isinstance(value, dict):
        for k, v in value.items():
            _assert_no_nul(k)
            _assert_no_nul(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_nul(item)


pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


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
    """One entity per chunk whose name/description/attributes carry NUL bytes."""
    return [
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Ac\x00me Corp",
                    entity_type="ORG",
                    description="A wid\x00get maker",
                    attributes={"ali\x00as": "AC\x00ME", "ok": "plain"},
                    confidence=0.99,
                )
            ]
        )
        for _ in texts
    ]


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
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False

    instance = Khora(config, engine="vectorcypher", run_migrations=True)
    await instance.connect()
    try:
        yield instance
    finally:
        # A teardown failure is signal — do not swallow it (mirrors the pg fixture).
        await instance.disconnect()


async def test_remember_strips_nul_embedded(kb: Khora) -> None:
    """NUL in document content AND in an extracted entity is stripped on store."""
    ns = await kb.create_namespace()
    stable: UUID = ns.namespace_id

    content = "Ac\x00me Corp makes wid\x00gets in a fac\x00tory."
    rem = await kb.remember(
        content=content,
        namespace=stable,
        title="Acme\x00 Annual Report",
        source="scr\x00aper",
        metadata={"au\x00thor": "Bo\x00b", "tags": ["fin\x00ance"], "nested": {"k\x00": "v\x00"}},
        entity_types=["ORG"],
        relationship_types=[],
    )
    assert rem.chunks_created >= 1

    # Document must still be recallable.
    recalled = await kb.recall("Acme Corp widgets", namespace=stable, limit=10)
    assert recalled.chunks, "chunk must be recallable"
    for ch in recalled.chunks:
        assert "\x00" not in ch.content

    # Stored document content, title, source, and metadata (jsonb) must be NUL-free.
    resolved = await kb.storage.resolve_namespace(stable)
    doc = await kb.storage.get_document(rem.document_id, namespace_id=resolved)
    assert doc is not None
    assert "\x00" not in doc.content
    assert "\x00" not in (doc.title or "")
    assert "\x00" not in (doc.source or "")
    _assert_no_nul(doc.metadata)

    # Stored chunk content + metadata (metadata is a copy of the sanitized
    # document metadata) must be NUL-free.
    chunks = await kb.storage.get_chunks_by_document(rem.document_id, namespace_id=resolved)
    assert chunks, "chunks must be stored"
    for ch in chunks:
        assert "\x00" not in ch.content
        _assert_no_nul(ch.metadata)
        _assert_no_nul(ch.chunker_info)

    # Extracted entity name/description/attributes must be NUL-free.
    entities = await kb.storage.list_entities(stable, limit=50)
    assert entities, "entity must have been stored"
    for ent in entities:
        assert "\x00" not in ent.name
        assert "\x00" not in (ent.description or "")
        for k, v in (ent.attributes or {}).items():
            assert "\x00" not in k
            if isinstance(v, str):
                assert "\x00" not in v
