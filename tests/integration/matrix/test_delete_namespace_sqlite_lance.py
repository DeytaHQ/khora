"""Embedded ``delete_namespace`` cascade test on the sqlite_lance stack (#1460).

Adapts the issue's deterministic stub probe: create a namespace, ingest one
document (chunks + 2 entities + 1 relationship), then ``delete_namespace`` and
assert EVERY namespace-scoped direct-store count hits 0 (SQLite
documents/khora_chunks/entities/relationships + LanceDB khora_chunks_vec/
entities_vec) AND the namespace no longer lists (active or not) / resolves.

Also proves the stranded-namespace recovery path: ``deactivate_namespace``
first (which strands the data with no cascade), THEN ``delete_namespace`` still
reclaims everything.

No Docker / Postgres / Neo4j needed — pure in-process SQLite (``aiosqlite``) +
LanceDB (``lancedb``).  Stubs the embedder + extractor so no real LLM is
called (no ``OPENAI_API_KEY`` required)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

try:  # Module-level import gate matches the other sqlite_lance suites.
    import aiosqlite
    import lancedb

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.config import KhoraConfig
from khora.config.schema import SQLiteLanceConfig
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import EventExtractionConfig, FactExtractionConfig
from khora.khora import Khora
from khora.storage.backends.sqlite_lance._helpers import uuid_to_text

EMBED_DIM = 32  # sqlite_lance default; direct-store counts are dimension-agnostic

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

DOC = "Sarah Chen is the CFO at Globex Corp. She approves expense reimbursements over 2000 USD."


# ---------------------------------------------------------------------------
# Deterministic embedder + extractor stubs (no OPENAI_API_KEY needed)
# ---------------------------------------------------------------------------
def _embed_for(text_in: str) -> list[float]:
    seed = hashlib.sha256(text_in.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(EMBED_DIM)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


_REGISTRY: dict[str, ExtractionResult] = {}


async def _stub_extract_multi(self: Any, texts: list[str], **_kw: Any) -> list[ExtractionResult]:
    return [next((r for marker, r in _REGISTRY.items() if marker in t), ExtractionResult()) for t in texts]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    _REGISTRY.clear()
    _REGISTRY[DOC] = ExtractionResult(
        entities=[
            ExtractedEntity(name="Sarah Chen", entity_type="PERSON", confidence=0.99),
            ExtractedEntity(name="Globex Corp", entity_type="ORGANIZATION", confidence=0.99),
        ],
        relationships=[
            ExtractedRelationship(
                source_entity="Sarah Chen",
                target_entity="Globex Corp",
                relationship_type="WORKS_AT",
                confidence=0.99,
            )
        ],
    )
    monkeypatch.setattr("khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch", _stub_embed_batch)
    monkeypatch.setattr("khora.extraction.embedders.litellm.LiteLLMEmbedder.embed", _stub_embed)
    monkeypatch.setattr("khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi", _stub_extract_multi)


def _no_ef() -> ExpertiseConfig:
    return ExpertiseConfig(
        name="ns-delete-probe",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Per-test embedded Khora fixture (paths exposed for direct-store probing)
# ---------------------------------------------------------------------------
@pytest.fixture
async def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "khora.db", tmp_path / "khora.lance"


@pytest.fixture
async def kb(paths: tuple[Path, Path]) -> AsyncIterator[Khora]:
    db_path, lance_path = paths
    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(db_path),
        lance_path=str(lance_path),
        embedding_dimension=EMBED_DIM,
    )
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.neo4j_url = None
    config.pipelines.chunk_size = 1024
    config.pipelines.extract_entities = True
    config.pipelines.selective_extraction = False

    kb = Khora(config, engine="vectorcypher", run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        await kb.disconnect()


# ---------------------------------------------------------------------------
# Direct-store row counters (bypass the facade — prove PHYSICAL removal)
# ---------------------------------------------------------------------------
async def _sqlite_direct_counts(db_path: Path, lance_path: Path, row_id: UUID) -> dict[str, int]:
    out: dict[str, int] = {}
    row_txt = uuid_to_text(row_id)
    db = await aiosqlite.connect(str(db_path))
    try:
        for tbl in ("documents", "khora_chunks", "entities", "relationships"):
            cur = await db.execute(
                f"SELECT count(*) FROM {tbl} WHERE namespace_id = ?",  # noqa: S608 - tbl is a hardcoded literal
                (row_txt,),
            )
            out[f"sqlite.{tbl}"] = (await cur.fetchone())[0]
    finally:
        await db.close()

    ldb = await lancedb.connect_async(str(lance_path))
    for tname in ("khora_chunks_vec", "entities_vec"):
        tbl = await ldb.open_table(tname)
        out[f"lance.{tname}"] = await tbl.count_rows(f"namespace_id = '{row_txt}'")
    return out


async def _lists_namespace(kb: Khora, stable: UUID, row_id: UUID, *, active_only: bool) -> bool:
    page = await kb.storage.list_namespaces(active_only=active_only, limit=1000)
    ids = {n.id for n in page.items} | {n.namespace_id for n in page.items}
    return stable in ids or row_id in ids


async def _ingest_one(kb: Khora, stable: UUID) -> None:
    await kb.remember(
        content=DOC,
        namespace=stable,
        entity_types=["PERSON", "ORGANIZATION"],
        relationship_types=["WORKS_AT"],
        expertise=_no_ef(),
    )


@pytest.mark.asyncio
async def test_delete_namespace_reclaims_all_stores(kb: Khora, paths: tuple[Path, Path]) -> None:
    db_path, lance_path = paths
    ns = await kb.create_namespace()
    stable, row_id = ns.namespace_id, ns.id

    await _ingest_one(kb, stable)

    before = await _sqlite_direct_counts(db_path, lance_path, row_id)
    # Everything the namespace owns is physically present before deletion.
    assert before["sqlite.documents"] == 1
    assert before["sqlite.khora_chunks"] >= 1
    assert before["sqlite.entities"] == 2
    assert before["sqlite.relationships"] == 1
    assert before["lance.khora_chunks_vec"] >= 1
    assert before["lance.entities_vec"] == 2

    result = await kb.delete_namespace(stable)

    assert not result.partial_failure, result.degradations
    assert result.namespaces_removed == 1
    assert result.documents_removed == 1
    assert row_id in result.removed_row_ids

    after = await _sqlite_direct_counts(db_path, lance_path, row_id)
    assert after == {
        "sqlite.documents": 0,
        "sqlite.khora_chunks": 0,
        "sqlite.entities": 0,
        "sqlite.relationships": 0,
        "lance.khora_chunks_vec": 0,
        "lance.entities_vec": 0,
    }

    # The namespace no longer lists (active or not) and no longer resolves.
    assert not await _lists_namespace(kb, stable, row_id, active_only=True)
    assert not await _lists_namespace(kb, stable, row_id, active_only=False)
    assert await kb.get_namespace(row_id) is None
    with pytest.raises(ValueError):
        await kb.storage.resolve_namespace(stable)


@pytest.mark.asyncio
async def test_delete_reclaims_a_deactivated_stranded_namespace(kb: Khora, paths: tuple[Path, Path]) -> None:
    """deactivate_namespace strands the data (no cascade, recall/forget raise);
    delete_namespace must still reclaim it — the recovery path."""
    db_path, lance_path = paths
    ns = await kb.create_namespace()
    stable, row_id = ns.namespace_id, ns.id

    await _ingest_one(kb, stable)
    assert (await _sqlite_direct_counts(db_path, lance_path, row_id))["sqlite.entities"] == 2

    # Strand it: is_active=False, no cascade. resolve/recall now dead-end.
    await kb.storage.deactivate_namespace(row_id)
    with pytest.raises(ValueError):
        await kb.storage.resolve_namespace(stable)

    # Reclaim it anyway — delete resolves WITHOUT the is_active filter.
    result = await kb.delete_namespace(stable)
    assert not result.partial_failure, result.degradations
    assert result.namespaces_removed == 1

    after = await _sqlite_direct_counts(db_path, lance_path, row_id)
    assert sum(after.values()) == 0, after
    assert not await _lists_namespace(kb, stable, row_id, active_only=False)


@pytest.mark.asyncio
async def test_delete_unknown_namespace_is_a_noop(kb: Khora) -> None:
    from uuid import uuid4

    result = await kb.delete_namespace(uuid4())
    assert result.namespaces_removed == 0
    assert result.removed_row_ids == []
    assert not result.partial_failure
