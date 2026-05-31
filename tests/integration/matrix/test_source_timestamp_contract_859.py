"""Cross-engine ``source_timestamp`` round-trip contract (#859).

Damir's report (#859) flagged ``Khora.remember(source_timestamp=T)`` →
``Khora.recall(...).chunks[0].occurred_at == None`` on VectorCypher. The
analysis team verified the Skeleton fix in #856 (commit 68cd4ee9) and
identified two distinct slices in VectorCypher:

* Slice A (ingest): ``engines/vectorcypher/engine.py`` dropped
  ``source_timestamp`` from the ``occurred_at`` resolution chain, so the
  chunk landed at ``datetime.now(UTC)`` instead of the supplied
  timestamp.
* Slice B (recall): three ``Chunk(...)`` construction sites in
  ``engines/vectorcypher/retriever.py`` dropped the persisted column
  when building the in-memory ``Chunk``, so
  ``RecallChunk.occurred_at`` read ``None`` even when the row
  contained a value.

This test pins the cross-engine contract: for every registered engine,
``Khora.remember(content, source_timestamp=T)`` followed by
``Khora.recall(...)`` returns a ``RecallChunk`` whose ``occurred_at``
equals ``T`` (within seconds). Skeleton + Chronicle were already correct
(positive controls); VectorCypher would fail this test against
``v0.17.4`` and pass after the fix.

Scope: sqlite_lance only - lightweight, no Docker / Postgres / Neo4j
required. TODO(#859): extend to the pg + neo4j matrix once the
infrastructure-isolation guard in CLAUDE.md is wired into CI selectors.
"""

from __future__ import annotations

import hashlib
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
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractionResult,
)
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import EventExtractionConfig, FactExtractionConfig
from khora.khora import Khora

EMBED_DIM = 32

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


# ---------------------------------------------------------------------------
# Deterministic embedder + extractor stubs (no OPENAI_API_KEY required)
# ---------------------------------------------------------------------------


def _embed_for(text_in: str) -> list[float]:
    """SHA-256 → 32-dim unit vector. Same text yields same vector."""
    seed = hashlib.sha256(text_in.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(EMBED_DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    # Both VectorCypher and Chronicle run entity extraction by default.
    # Return one canonical PERSON entity for any text that mentions
    # "Marie Curie" so the recall path has something to surface;
    # otherwise return empty.
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
# Per-engine Khora fixture
# ---------------------------------------------------------------------------


def _expertise_for(engine: str) -> ExpertiseConfig:
    """Disable per-chunk event/fact extraction on Chronicle to keep the
    test fast. The ``source_timestamp`` round-trip is independent of
    those channels.
    """
    if engine == "chronicle":
        return ExpertiseConfig(
            name=f"contract-859-{engine}",
            events=EventExtractionConfig(enabled=False),
            facts=FactExtractionConfig(enabled=False),
        )
    return ExpertiseConfig(name=f"contract-859-{engine}")


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
    if engine in {"vectorcypher", "chronicle"}:
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


# ---------------------------------------------------------------------------
# Contract test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kb",
    ["skeleton", "vectorcypher", "chronicle"],
    indirect=True,
    ids=["skeleton", "vectorcypher", "chronicle"],
)
async def test_remember_source_timestamp_round_trips_to_recall(kb: Khora) -> None:
    """``Khora.remember(source_timestamp=T)`` round-trips to ``recall().chunks[*].occurred_at``.

    Pinned per engine - the bug Damir reported lived in VectorCypher only,
    but the contract belongs to every engine. Future engines registered
    via ``register_engine`` will fail this parametrized cell if they
    introduce the same drop.
    """
    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    # 30 days in the past - far enough that "now()" fallback would
    # produce a clearly different value.
    intended = datetime.now(UTC) - timedelta(days=30)
    content = "Marie Curie won the Nobel Prize in Physics in 1903 for her work on radioactivity."

    # #890: Skeleton refuses non-empty entity_types / relationship_types
    # because it has no entity extraction. VectorCypher and Chronicle
    # still use the typed extraction whitelist.
    engine_name: str = kb._engine_name  # type: ignore[attr-defined]
    if engine_name == "skeleton":
        entity_types_for_engine: list[str] = []
        relationship_types_for_engine: list[str] = []
    else:
        entity_types_for_engine = ["PERSON", "CONCEPT"]
        relationship_types_for_engine = ["RELATES_TO"]

    await kb.remember(
        content=content,
        namespace=namespace_id,
        title="curie-1903",
        source_timestamp=intended,
        entity_types=entity_types_for_engine,
        relationship_types=relationship_types_for_engine,
        expertise=_expertise_for(engine_name),
    )

    result = await kb.recall(
        "Marie Curie Nobel Prize",
        namespace=namespace_id,
        limit=5,
    )

    assert result.chunks, "expected at least one chunk back"
    occurred_at = result.chunks[0].occurred_at
    # Pre-fix on VectorCypher this was ``None`` (recall-side dropout in
    # retriever.py) or ``datetime.now(UTC)`` (ingest-side dropout in
    # engine.py - depends on whether the recall path went through the
    # buggy Chunk-construction site).
    assert occurred_at is not None, (
        f"chunk.occurred_at is None on engine={kb._engine_name!r}; "  # type: ignore[attr-defined]
        "either ingest-side dropped source_timestamp or recall-side "
        "dropped the persisted column (#859)"
    )
    # Normalize tz: Chronicle's sqlite_lance round-trip drops the UTC
    # tzinfo on read (the ``DateTime`` SQLAlchemy column is naive by
    # default on SQLite). Treat naive datetimes as UTC for the diff -
    # the contract is "same instant in time", not "same tzinfo object".
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=UTC)
    # Allow seconds of slop for storage round-trip / timezone normalization.
    assert abs((occurred_at - intended).total_seconds()) < 60, (
        f"chunk.occurred_at={occurred_at!r} does not match "
        f"source_timestamp={intended!r} on engine={kb._engine_name!r} (#859)"  # type: ignore[attr-defined]
    )
