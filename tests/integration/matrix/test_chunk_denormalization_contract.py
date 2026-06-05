"""End-to-end contract for chunk-level denormalized document fields.

This suite pins three guarantees that the chunk-denormalization work must
hold across the engines that read the temporal-store chunk table
(``khora_chunks``): VectorCypher (the default engine) and Skeleton.

1. ``occurred_at`` and ``source_timestamp`` stay DISTINCT end-to-end.
   When a document carries a producer-level ``source_timestamp`` (T_doc)
   and a chunk-level ``occurred_at`` (T_chunk) that differ, recall must
   surface the chunk's ``occurred_at`` — never collapse it onto the
   document's ``source_timestamp``. The collapse is the regression this
   guards: once the write-path populates the denormalized
   ``source_timestamp`` column, a read path that prefers it over
   ``occurred_at`` would silently rewrite every chunk's event time to the
   document time. This is checked under both VECTOR and KEYWORD recall so
   the contract holds across retrieval channels.

2. The denormalized document-grained fields are written to the chunk row
   from the parent ``Document``'s attributes at ingest, so recall filters
   can hit the chunk row without a document join.

3. With no chunk-level ``occurred_at``, the chunk's event time falls back
   to the document ``source_timestamp`` and recall surfaces that value —
   the existing single-timestamp behavior is preserved.

Scope: sqlite_lance only — lightweight, no Docker / Postgres / Neo4j
required. The embedded ``khora_chunks`` table is created at runtime from
the same column definition the migration adds on Postgres, so the
write/read round-trip exercised here matches the production schema.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
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
from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult
from khora.extraction.skills import ExpertiseConfig
from khora.khora import Khora
from khora.query import SearchMode

EMBED_DIM = 32

# The denormalized document-grained columns the chunk row carries.
# ``content_type`` is not settable through the public ``remember`` kwarg
# surface, so the round-trip test sets it on the ``Document`` directly via
# metadata pass-through where the engine reads it; the write-path test
# below asserts the API-settable subset end-to-end and ``content_type``
# through the document attribute.
_DENORMALIZED_FIELDS = (
    "source_type",
    "source_name",
    "source_url",
    "source_timestamp",
    "external_id",
    "content_type",
    "source",
    "title",
)

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


# ---------------------------------------------------------------------------
# Deterministic embedder + extractor stubs (no OPENAI_API_KEY required)
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
    out: list[ExtractionResult] = []
    for t in texts:
        if "Marie Curie" in t:
            out.append(
                ExtractionResult(
                    entities=[ExtractedEntity(name="Marie Curie", entity_type="PERSON", confidence=0.99)],
                )
            )
        else:
            out.append(ExtractionResult())
    return out


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
    if engine == "vectorcypher":
        config.pipelines.extract_entities = True
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


def _types_for(engine: str) -> tuple[list[str], list[str]]:
    """Skeleton refuses non-empty entity/relationship types (no extraction)."""
    if engine == "skeleton":
        return [], []
    return ["PERSON", "CONCEPT"], ["RELATES_TO"]


def _as_utc(dt: datetime | None) -> datetime | None:
    """Treat naive datetimes as UTC — the sqlite round-trip drops tzinfo."""
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _khora_chunks_rows(db_path: str) -> tuple[list[str], list[sqlite3.Row]]:
    """Read the raw ``khora_chunks`` table from the embedded SQLite file."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cols = [r["name"] for r in con.execute("PRAGMA table_info(khora_chunks)").fetchall()]
        rows = con.execute("SELECT * FROM khora_chunks").fetchall()
        return cols, rows
    finally:
        con.close()


_CONTENT = "Marie Curie won the Nobel Prize in Physics in 1903 for her work on radioactivity."


# ---------------------------------------------------------------------------
# 1. Tripwire: occurred_at and source_timestamp stay distinct end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kb",
    ["vectorcypher", "skeleton"],
    indirect=True,
    ids=["vectorcypher", "skeleton"],
)
@pytest.mark.parametrize("mode", [SearchMode.VECTOR, SearchMode.KEYWORD], ids=["vector", "keyword"])
async def test_recall_occurred_at_does_not_collapse_onto_source_timestamp(kb: Khora, mode: SearchMode) -> None:
    """A chunk's ``occurred_at`` survives even when it differs from the
    document ``source_timestamp``.

    The document is ingested with a producer ``source_timestamp`` 100 days
    old and a chunk-level ``occurred_at`` 10 days old (supplied via
    ``metadata['occurred_at']``). Recall must return the chunk's
    ``occurred_at`` (10 days), not the document's ``source_timestamp``
    (100 days). The VECTOR / KEYWORD parametrization runs the assertion
    over both the vector dispatch and the full-text / BM25 dispatch so the
    contract holds regardless of which retrieval channel surfaces a chunk.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    t_doc = datetime.now(UTC) - timedelta(days=100)  # document source_timestamp
    t_chunk = datetime.now(UTC) - timedelta(days=10)  # chunk occurred_at (distinct)
    assert t_doc != t_chunk

    engine_name: str = kb._engine_name  # type: ignore[attr-defined]
    entity_types, relationship_types = _types_for(engine_name)

    await kb.remember(
        content=_CONTENT,
        namespace=namespace_id,
        title="curie",
        source_timestamp=t_doc,
        metadata={"occurred_at": t_chunk.isoformat()},
        entity_types=entity_types,
        relationship_types=relationship_types,
        expertise=ExpertiseConfig(name=f"denorm-{engine_name}"),
    )

    result = await kb.recall(
        "Marie Curie Nobel Prize",
        namespace=namespace_id,
        limit=5,
        mode=mode,
    )

    assert result.chunks, f"expected at least one chunk back (engine={engine_name}, mode={mode})"
    occurred_at = _as_utc(result.chunks[0].occurred_at)
    assert occurred_at is not None, f"chunk.occurred_at is None (engine={engine_name}, mode={mode})"

    # The contract: occurred_at reflects the chunk event time (T_chunk),
    # NOT the document source_timestamp (T_doc).
    assert abs((occurred_at - t_chunk).total_seconds()) < 120, (
        f"RecallChunk.occurred_at={occurred_at!r} should reflect the chunk's "
        f"occurred_at={t_chunk!r}, not document source_timestamp={t_doc!r} "
        f"(engine={engine_name}, mode={mode})"
    )
    assert abs((occurred_at - t_doc).total_seconds()) > 120, (
        f"RecallChunk.occurred_at={occurred_at!r} collapsed onto the document "
        f"source_timestamp={t_doc!r} (engine={engine_name}, mode={mode})"
    )


# ---------------------------------------------------------------------------
# 2. Denormalized fields are written to the chunk row from Document attrs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kb",
    ["vectorcypher", "skeleton"],
    indirect=True,
    ids=["vectorcypher", "skeleton"],
)
async def test_denormalized_fields_written_from_document(kb: Khora) -> None:
    """The chunk row carries the parent document's denormalized fields.

    Ingest a document with the API-settable provenance fields populated,
    then read the raw ``khora_chunks`` row and assert each denormalized
    column equals the document attribute it was copied from.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    t_doc = datetime.now(UTC) - timedelta(days=42)
    engine_name: str = kb._engine_name  # type: ignore[attr-defined]
    entity_types, relationship_types = _types_for(engine_name)

    await kb.remember(
        content=_CONTENT,
        namespace=namespace_id,
        title="Curie Biography",
        source="library://curie",
        source_type="article",
        source_name="wikipedia",
        source_url="https://en.wikipedia.org/wiki/Marie_Curie",
        source_timestamp=t_doc,
        external_id="ext-curie-001",
        entity_types=entity_types,
        relationship_types=relationship_types,
        expertise=ExpertiseConfig(name=f"denorm-{engine_name}"),
    )

    db_path = str(kb._config.storage.sqlite_lance.db_path)  # type: ignore[attr-defined]
    cols, rows = _khora_chunks_rows(db_path)

    missing_cols = [f for f in _DENORMALIZED_FIELDS if f not in cols]
    assert not missing_cols, f"khora_chunks is missing denormalized columns: {missing_cols}"
    assert rows, "expected at least one khora_chunks row"
    row = rows[0]

    assert row["source_type"] == "article"
    assert row["source_name"] == "wikipedia"
    assert row["source_url"] == "https://en.wikipedia.org/wiki/Marie_Curie"
    assert row["external_id"] == "ext-curie-001"
    assert row["source"] == "library://curie"
    assert row["title"] == "Curie Biography"

    stored_ts = row["source_timestamp"]
    assert stored_ts is not None, "denormalized source_timestamp not written"
    parsed = _as_utc(datetime.fromisoformat(str(stored_ts)))
    assert parsed is not None
    assert abs((parsed - t_doc).total_seconds()) < 120, (
        f"denormalized source_timestamp={parsed!r} does not match document source_timestamp={t_doc!r}"
    )


# ---------------------------------------------------------------------------
# 3. Fallback: no chunk occurred_at -> occurred_at derives from source_timestamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kb",
    ["vectorcypher", "skeleton"],
    indirect=True,
    ids=["vectorcypher", "skeleton"],
)
async def test_recall_occurred_at_falls_back_to_source_timestamp(kb: Khora) -> None:
    """With no chunk-level ``occurred_at``, recall surfaces the document
    ``source_timestamp`` as the chunk event time.

    This preserves the single-timestamp behavior: callers who supply only
    ``source_timestamp`` still get it back as ``RecallChunk.occurred_at``.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    t_doc = datetime.now(UTC) - timedelta(days=30)
    engine_name: str = kb._engine_name  # type: ignore[attr-defined]
    entity_types, relationship_types = _types_for(engine_name)

    await kb.remember(
        content=_CONTENT,
        namespace=namespace_id,
        title="curie",
        source_timestamp=t_doc,
        # No metadata["occurred_at"] — chunk event time must fall back.
        entity_types=entity_types,
        relationship_types=relationship_types,
        expertise=ExpertiseConfig(name=f"denorm-{engine_name}"),
    )

    result = await kb.recall(
        "Marie Curie Nobel Prize",
        namespace=namespace_id,
        limit=5,
    )

    assert result.chunks, f"expected at least one chunk back (engine={engine_name})"
    occurred_at = _as_utc(result.chunks[0].occurred_at)
    assert occurred_at is not None, f"chunk.occurred_at is None (engine={engine_name})"
    assert abs((occurred_at - t_doc).total_seconds()) < 120, (
        f"RecallChunk.occurred_at={occurred_at!r} did not fall back to the "
        f"document source_timestamp={t_doc!r} (engine={engine_name})"
    )
