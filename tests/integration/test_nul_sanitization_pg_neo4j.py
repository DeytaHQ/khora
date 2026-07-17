"""Live PG+Neo4j proof for #1528: ingestion survives NUL bytes (0x00).

Before the fix, a document whose content (or whose extracted entity
name/description/attributes) carried a NUL byte aborted the entity INSERT with
``asyncpg.CharacterNotInRepertoireError: invalid byte sequence for encoding
"UTF8": 0x00`` and crashed the whole ingestion run. The fix strips NUL bytes at
the ingestion boundary (document content on ``remember``) and from extracted
entity/relationship text as it is staged into the storage models.

This test drives the real crash path: ``remember()`` a document with ``\\x00``
in its content and a stub extractor that emits an entity with ``\\x00`` in its
name / description / attributes, then asserts the call succeeds, the stored text
is NUL-free, and the document/entity are recallable.

How to run locally::

    make dev   # postgres (5434) + neo4j (bolt 7688) via docker compose
    NEO4J_INTEGRATION_TEST=1 KHORA_NEO4J_URL=bolt://localhost:7688 \\
        uv run pytest tests/integration/test_nul_sanitization_pg_neo4j.py \\
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


DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


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
    """Emit one entity per chunk whose name/description/attributes carry NUL."""
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
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=_neo4j_url())
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.storage.postgresql_url = DATABASE_URL
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False

    instance = Khora(config, engine="vectorcypher", run_migrations=True)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.disconnect()


@pytest.mark.skipif(
    not _neo4j_reachable() or not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 and run `make dev` (needs Neo4j for the full vectorcypher path)",
)
async def test_remember_survives_nul_on_pg_neo4j(kb_vc: Khora) -> None:
    """remember() of NUL-bearing content + entity does not crash and stores clean text (#1528)."""
    ns = await kb_vc.create_namespace()
    stable: UUID = ns.namespace_id

    content = "Ac\x00me Corp makes wid\x00gets in a fac\x00tory near the river."
    # Pre-fix this raised asyncpg.CharacterNotInRepertoireError during the
    # documents / entities INSERT; post-fix it succeeds. NUL in title (a
    # String(512) column) and metadata (jsonb) are the gap #1529 surfaced.
    rem = await kb_vc.remember(
        content=content,
        namespace=stable,
        title="Acme\x00 Annual Report",
        source="scr\x00aper",
        metadata={"au\x00thor": "Bo\x00b", "tags": ["fin\x00ance"], "nested": {"k\x00": "v\x00"}},
        entity_types=["ORG"],
        relationship_types=[],
    )
    assert rem.chunks_created >= 1
    assert rem.entities_extracted >= 1

    # Document must still be recallable and its chunk text NUL-free. On the
    # VectorCypher path chunks live in the temporal ``khora_chunks`` table (its
    # content + metadata are jsonb/text columns), so a NUL in chunk metadata —
    # which is a copy of the document metadata — would itself have raised
    # CharacterNotInRepertoireError above; the successful remember() is the
    # proof. The embedded leg asserts stored chunk metadata is NUL-free directly.
    recalled = await kb_vc.recall("Acme Corp widgets", namespace=stable, limit=10)
    assert recalled.chunks, "chunk must be recallable"
    for ch in recalled.chunks:
        assert "\x00" not in ch.content

    resolved = await kb_vc.storage.resolve_namespace(stable)
    doc = await kb_vc.storage.get_document(rem.document_id, namespace_id=resolved)
    assert doc is not None
    assert "\x00" not in doc.content
    assert "\x00" not in (doc.title or "")
    assert "\x00" not in (doc.source or "")
    _assert_no_nul(doc.metadata)

    # Extracted entity name/description/attributes are NUL-free in storage.
    entities = await kb_vc.storage.list_entities(stable, limit=50)
    assert entities, "entity must have been stored"
    for ent in entities:
        assert "\x00" not in ent.name
        assert "\x00" not in (ent.description or "")
        for k, v in (ent.attributes or {}).items():
            assert "\x00" not in k
            if isinstance(v, str):
                assert "\x00" not in v
