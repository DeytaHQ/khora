"""End-to-end contract: chunk position bookkeeping lives in ``chunker_info``.

Pins the split between the two chunk-level JSON columns on the persisted
``khora_chunks`` row:

* ``metadata`` carries USER / DOCUMENT metadata ONLY. The four position
  bookkeeping keys (``chunk_index`` / ``start_char`` / ``end_char`` /
  ``token_count``) must be ABSENT from it.
* ``chunker_info`` carries the chunker identifier (``chunker``) PLUS the
  four position bookkeeping keys with their correct values.

This guards the write side of the refactor that moved chunk bookkeeping
out of ``metadata`` and into ``chunker_info`` across every persisting
writer. It is exercised across all three write paths that stamp the
temporal-store chunk row:

1. Default ingest (VectorCypher create path).
2. The Skeleton engine chunk-document path.
3. The VectorCypher replace / update-document path — a re-``remember`` of
   a document under the same ``external_id`` re-writes every chunk.

Scope: sqlite_lance only — lightweight, no Docker / Postgres / Neo4j
required. The embedded ``khora_chunks`` table is created at runtime from
the same column definition the migration adds on Postgres, so the
write/read round-trip exercised here matches the production schema.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
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

# The four chunk position bookkeeping keys that MUST live in chunker_info,
# never in the user/document metadata column.
_BOOKKEEPING_KEYS = ("chunk_index", "start_char", "end_char", "token_count")

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


# ---------------------------------------------------------------------------
# Deterministic embedder + no-op extractor stubs (no OPENAI_API_KEY needed)
# ---------------------------------------------------------------------------


def _embed_for(text_in: str) -> list[float]:
    """SHA-256 -> 32-dim unit vector. Same text yields the same vector."""
    seed = hashlib.sha256(text_in.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(EMBED_DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    # Entity extraction is irrelevant to this contract; return empty results.
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
# Per-engine Khora fixture (sqlite_lance)
# ---------------------------------------------------------------------------


async def _build_kb(tmp_path: Path, engine: str) -> Khora:
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
    # Bookkeeping is independent of entity extraction — keep it off so the
    # test stays hermetic (no LLM entity calls).
    config.pipelines.extract_entities = False
    config.pipelines.selective_extraction = False

    kb = Khora(config, engine=engine, run_migrations=True)
    await kb.connect()
    return kb


@pytest.fixture
async def kb(tmp_path: Path, request: pytest.FixtureRequest) -> AsyncIterator[Khora]:
    """Per-test Khora bound to a sqlite_lance stack on the requested engine."""
    engine: str = request.param
    instance = await _build_kb(tmp_path, engine)
    try:
        yield instance
    finally:
        try:
            await instance.disconnect()
        except Exception:
            pass


def _types_and_expertise(engine: str) -> tuple[list[str], list[str], ExpertiseConfig | None]:
    """Skeleton refuses non-empty types and non-None expertise (#1431)."""
    if engine == "skeleton":
        return [], [], None
    return [], [], ExpertiseConfig(name=f"bookkeeping-{engine}")


_CONTENT = "Marie Curie won the Nobel Prize in Physics in 1903 for her work on radioactivity."


# ---------------------------------------------------------------------------
# Raw khora_chunks reader
# ---------------------------------------------------------------------------


def _khora_chunks_json(db_path: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return ``(metadata, chunker_info)`` decoded from every khora_chunks row."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cols = [r["name"] for r in con.execute("PRAGMA table_info(khora_chunks)").fetchall()]
        assert "metadata" in cols, f"khora_chunks missing metadata column: {cols}"
        assert "chunker_info" in cols, f"khora_chunks missing chunker_info column: {cols}"
        rows = con.execute("SELECT metadata, chunker_info FROM khora_chunks").fetchall()
    finally:
        con.close()

    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        md = json.loads(row["metadata"]) if row["metadata"] else {}
        ci = json.loads(row["chunker_info"]) if row["chunker_info"] else {}
        out.append((md, ci))
    return out


def _assert_bookkeeping_split(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    """Assert the metadata/chunker_info split for every persisted chunk row.

    metadata: user/document metadata only ("team":"alpha" present, the four
    bookkeeping keys ABSENT). chunker_info: a non-empty ``chunker`` name plus
    all four bookkeeping keys with plausible (non-negative int) values.
    """
    assert rows, "expected at least one khora_chunks row"
    for md, ci in rows:
        # metadata carries the user tag and NONE of the bookkeeping keys.
        assert md.get("team") == "alpha", f"user metadata not preserved: {md!r}"
        leaked = [k for k in _BOOKKEEPING_KEYS if k in md]
        assert not leaked, f"bookkeeping keys leaked into metadata: {leaked} in {md!r}"

        # chunker_info carries the chunker identifier + the four keys.
        assert ci.get("chunker"), f"chunker_info missing non-empty 'chunker': {ci!r}"
        assert isinstance(ci["chunker"], str), f"chunker must be a str: {ci!r}"
        missing = [k for k in _BOOKKEEPING_KEYS if k not in ci]
        assert not missing, f"chunker_info missing bookkeeping keys: {missing} in {ci!r}"
        for k in _BOOKKEEPING_KEYS:
            assert isinstance(ci[k], int), f"chunker_info[{k!r}] must be int, got {ci[k]!r}"
            assert ci[k] >= 0, f"chunker_info[{k!r}] must be >= 0, got {ci[k]!r}"


# ---------------------------------------------------------------------------
# 1 + 2. Default ingest (vectorcypher) and skeleton engine write paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kb",
    ["vectorcypher", "skeleton"],
    indirect=True,
    ids=["vectorcypher", "skeleton"],
)
async def test_bookkeeping_in_chunker_info_not_metadata(kb: Khora) -> None:
    """Persisted chunk metadata is user-only; bookkeeping is in chunker_info.

    Covers the default VectorCypher create path and the Skeleton engine
    chunk-document path. After a single ``remember`` with a user tag, every
    ``khora_chunks`` row must carry ``{"team": "alpha"}`` in metadata (and
    none of the four bookkeeping keys), and ``{"chunker": <name>}`` plus the
    four bookkeeping keys in chunker_info.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    engine_name: str = kb._engine_name  # type: ignore[attr-defined]
    entity_types, relationship_types, expertise = _types_and_expertise(engine_name)

    await kb.remember(
        content=_CONTENT,
        namespace=namespace_id,
        title="curie",
        metadata={"team": "alpha"},
        entity_types=entity_types,
        relationship_types=relationship_types,
        expertise=expertise,
    )

    db_path = str(kb._config.storage.sqlite_lance.db_path)  # type: ignore[attr-defined]
    _assert_bookkeeping_split(_khora_chunks_json(db_path))


# ---------------------------------------------------------------------------
# 3. VectorCypher replace / update-document write path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kb", ["vectorcypher"], indirect=True, ids=["vectorcypher"])
async def test_bookkeeping_split_survives_replace_document(kb: Khora) -> None:
    """The replace path re-writes chunk rows keeping the metadata split.

    Re-``remember``-ing a document under the same ``external_id`` routes the
    VectorCypher engine through the replace / update-document write path,
    which re-stamps every chunk row. After the replace the persisted rows
    must still carry user-only metadata and bookkeeping-in-chunker_info.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    expertise = ExpertiseConfig(name="bookkeeping-replace")

    # First remember creates the document (VectorCypher create path).
    await kb.remember(
        content=_CONTENT,
        namespace=namespace_id,
        title="curie",
        external_id="ext-curie-replace",
        metadata={"team": "alpha"},
        entity_types=[],
        relationship_types=[],
        expertise=expertise,
    )

    # Second remember under the same external_id routes to the replace path,
    # re-writing every chunk row from scratch.
    await kb.remember(
        content=_CONTENT + " She also won the 1911 Nobel Prize in Chemistry.",
        namespace=namespace_id,
        title="curie",
        external_id="ext-curie-replace",
        metadata={"team": "alpha"},
        entity_types=[],
        relationship_types=[],
        expertise=expertise,
    )

    db_path = str(kb._config.storage.sqlite_lance.db_path)  # type: ignore[attr-defined]
    _assert_bookkeeping_split(_khora_chunks_json(db_path))
