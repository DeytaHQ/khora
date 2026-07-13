"""End-to-end cross-channel invariant: the stored metadata blob is what recall
filters see, on every retrieval channel.

The recall read paths carry the STORED chunk metadata blob verbatim — the
first-class ``occurred_at`` column is never folded back into the blob at read
time. This suite pins the user-observable consequence through a REAL recall:

1. A chunk ingested with a producer ``source_timestamp`` (so its ``occurred_at``
   column is populated) but a CLEAN metadata blob (no ``occurred_at`` key) is
   selected identically by a ``metadata.occurred_at $exists false`` filter across
   the vector, keyword/BM25, and hybrid (graph-bearing) recall channels. Before
   the read paths stopped injecting the column into the blob, the graph channel's
   in-memory post-filter saw an ``occurred_at`` key the SQL-pushdown channels
   never saw — so the same filter kept the chunk on one channel and dropped it on
   another. This is that cross-channel divergence, guarded end-to-end.

2. The mirror ``metadata.occurred_at $exists true`` selects NOTHING — the blob
   has no such key on any channel.

3. A whole-blob ``$eq`` (the exact stored blob) selects the chunk — the in-memory
   full-AST post-filter compares against the same stored blob the SQL pushdown
   does.

4. The raw ``khora_chunks`` blob is exactly what was written: user keys only, no
   ``occurred_at`` / ``connected_entities`` / ``ppr_score`` injected. This asserts
   the WRITE path stores a clean blob — the precondition every recall channel then
   reads back. It does NOT observe the skeleton engine's rebuilt chunk: skeleton
   drops chunk metadata into an internal ``Chunk`` that never reaches the caller
   (``RecallChunk`` carries no metadata field), so the skeleton constructor's
   stored-blob line is a pure dead write with no observable recall surface —
   covered only indirectly, by this clean-write precondition plus the vector/BM25
   recall assertions above (which skeleton also serves).

Scope: sqlite_lance only — lightweight, no Docker. The embedded ``khora_chunks``
table matches the production schema, so the write/read round-trip exercised here
is faithful. VectorCypher's HYBRID recall fires the graph channel (an entity is
extracted from the content); Skeleton is graph-less, so its HYBRID recall is the
vector + BM25 legs.
"""

from __future__ import annotations

import hashlib
import json
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

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

# The clean stored blob and the injection keys no channel may add to it.
_STORED_BLOB: dict[str, Any] = {"author": "alice", "score": 3}
_INJECTION_KEYS = ("occurred_at", "connected_entities", "ppr_score")
_CONTENT = "Marie Curie won the Nobel Prize in Physics in 1903 for her work on radioactivity."


# ---------------------------------------------------------------------------
# Deterministic embedder + extractor stubs (no OPENAI_API_KEY required)
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
# Per-engine Khora fixture (sqlite_lance) — seeded once with the clean-blob chunk
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


def _types_for(engine: str) -> tuple[list[str], list[str]]:
    if engine == "skeleton":
        return [], []
    return ["PERSON", "CONCEPT"], ["RELATES_TO"]


def _expertise_for(engine: str) -> ExpertiseConfig | None:
    if engine == "skeleton":
        return None
    return ExpertiseConfig(name=f"blob-invariant-{engine}")


@pytest.fixture
async def seeded_kb(tmp_path: Path, request: pytest.FixtureRequest) -> AsyncIterator[tuple[Khora, UUID]]:
    """A Khora seeded with one clean-blob chunk whose occurred_at column is set.

    ``source_timestamp`` populates the ``occurred_at`` column (via the document
    fallback), while ``metadata`` is the CLEAN blob — no ``occurred_at`` key. That
    is exactly the "populated column + clean blob" shape the invariant is about.
    """
    engine: str = request.param
    kb = await _build_kb(tmp_path, engine)
    namespace_id = (await kb.create_namespace()).namespace_id
    entity_types, relationship_types = _types_for(engine)
    await kb.remember(
        content=_CONTENT,
        namespace=namespace_id,
        title="curie",
        source_timestamp=datetime.now(UTC) - timedelta(days=20),
        metadata=dict(_STORED_BLOB),
        entity_types=entity_types,
        relationship_types=relationship_types,
        expertise=_expertise_for(engine),
    )
    try:
        yield kb, namespace_id
    finally:
        try:
            await kb.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. The raw stored blob is clean (no injected keys) + occurred_at column set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seeded_kb", ["vectorcypher", "skeleton"], indirect=True, ids=["vectorcypher", "skeleton"])
async def test_stored_khora_chunks_blob_is_clean(seeded_kb: tuple[Khora, UUID]) -> None:
    """The persisted ``khora_chunks`` blob holds user keys only; the event-time is
    on the ``occurred_at`` column, not in the blob."""
    kb, _ns = seeded_kb
    db_path = str(kb._config.storage.sqlite_lance.db_path)  # type: ignore[attr-defined]
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("SELECT metadata, occurred_at FROM khora_chunks").fetchall()
    finally:
        con.close()

    assert rows, "expected at least one khora_chunks row"
    for row in rows:
        blob = json.loads(row["metadata"]) if row["metadata"] else {}
        assert blob == _STORED_BLOB
        for key in _INJECTION_KEYS:
            assert key not in blob, f"stored blob carries injected key {key!r}"
        # The event-time lives on the first-class column.
        assert row["occurred_at"] is not None


# ---------------------------------------------------------------------------
# 2. metadata.occurred_at $exists false selects the chunk on EVERY channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seeded_kb", ["vectorcypher", "skeleton"], indirect=True, ids=["vectorcypher", "skeleton"])
@pytest.mark.parametrize(
    "mode",
    [SearchMode.VECTOR, SearchMode.KEYWORD, SearchMode.HYBRID],
    ids=["vector", "keyword", "hybrid"],
)
async def test_absent_occurred_at_key_matches_across_channels(seeded_kb: tuple[Khora, UUID], mode: SearchMode) -> None:
    """A ``metadata.occurred_at $exists false`` filter keeps the clean-blob chunk
    identically across vector, keyword/BM25, and hybrid (graph-bearing) recall.

    The chunk's ``occurred_at`` COLUMN is populated, but the blob has no such key,
    so the filter must MATCH. VectorCypher HYBRID fires the graph channel (whose
    in-memory post-filter reads the rebuilt blob); the SQL-pushed vector/keyword
    channels read the stored blob directly. All must agree.
    """
    kb, ns = seeded_kb
    result = await kb.recall(
        "Marie Curie Nobel Prize",
        namespace=ns,
        limit=5,
        mode=mode,
        filter={"metadata.occurred_at": {"$exists": False}},
    )
    assert len(result.chunks) == 1, (
        f"clean-blob chunk with a populated occurred_at column must survive "
        f"'metadata.occurred_at $exists false' on mode={mode}"
    )


@pytest.mark.parametrize("seeded_kb", ["vectorcypher", "skeleton"], indirect=True, ids=["vectorcypher", "skeleton"])
@pytest.mark.parametrize(
    "mode",
    [SearchMode.VECTOR, SearchMode.KEYWORD, SearchMode.HYBRID],
    ids=["vector", "keyword", "hybrid"],
)
async def test_present_occurred_at_key_matches_nothing_across_channels(
    seeded_kb: tuple[Khora, UUID], mode: SearchMode
) -> None:
    """The mirror: ``metadata.occurred_at $exists true`` selects nothing — the
    blob carries no such key on any channel (a column-into-blob leak would make
    the graph channel surface the chunk here)."""
    kb, ns = seeded_kb
    result = await kb.recall(
        "Marie Curie Nobel Prize",
        namespace=ns,
        limit=5,
        mode=mode,
        filter={"metadata.occurred_at": {"$exists": True}},
    )
    assert result.chunks == [], f"no chunk should carry a 'metadata.occurred_at' key on mode={mode}"


# ---------------------------------------------------------------------------
# 3. Whole-blob $eq selects the chunk (incl. the graph channel post-filter)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seeded_kb", ["vectorcypher", "skeleton"], indirect=True, ids=["vectorcypher", "skeleton"])
@pytest.mark.parametrize(
    "mode",
    [SearchMode.VECTOR, SearchMode.KEYWORD, SearchMode.HYBRID],
    ids=["vector", "keyword", "hybrid"],
)
async def test_whole_blob_eq_matches_across_channels(seeded_kb: tuple[Khora, UUID], mode: SearchMode) -> None:
    """A whole-blob ``$eq`` of the exact stored blob selects the chunk on every
    channel — the graph channel's full-AST in-memory post-filter compares against
    the same stored blob the SQL pushdown does."""
    kb, ns = seeded_kb
    result = await kb.recall(
        "Marie Curie Nobel Prize",
        namespace=ns,
        limit=5,
        mode=mode,
        filter={"metadata": dict(_STORED_BLOB)},
    )
    assert len(result.chunks) == 1, f"whole-blob $eq must match the stored blob on mode={mode}"
