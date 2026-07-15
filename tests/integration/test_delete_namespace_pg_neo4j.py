"""Integration test for ``Khora.delete_namespace`` on the Postgres + Neo4j stack (#1460).

Exercises the public ``Khora.remember()`` / ``Khora.delete_namespace()`` API
against a real Postgres + Neo4j stack and asserts every namespace-scoped
direct-store count is 0 after deletion:

* PG: ``documents`` / ``chunks`` / ``khora_chunks`` / ``entities`` /
  ``relationships`` (queried directly via asyncpg, bypassing the facade);
* Neo4j: ``:Entity`` nodes, relationship edges, and ``:Chunk`` nodes.

Also proves the stranded-namespace recovery path: ``deactivate_namespace``
first (no cascade, ``resolve``/``recall``/``forget`` then raise), THEN
``delete_namespace`` still reclaims everything.

Gated by ``NEO4J_INTEGRATION_TEST=1`` (set by the CI integration job).

How to run locally::

    make dev  # postgres :5434 + neo4j :7688 via docker compose
    NEO4J_INTEGRATION_TEST=1 \\
    KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora \\
    KHORA_NEO4J_URL=bolt://localhost:7688 \\
    KHORA_NEO4J_PASSWORD=pleaseletmein \\
    uv run pytest tests/integration/test_delete_namespace_pg_neo4j.py -v -m integration --no-cov
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.skills import ExpertiseConfig
from khora.extraction.skills.base import EventExtractionConfig, FactExtractionConfig
from khora.khora import Khora

EMBED_DIM = 1536

DOC = "Sarah Chen is the CFO at Globex Corp. She approves expense reimbursements over 2000 USD."

_REGISTRY: dict[str, ExtractionResult] = {}


async def _stub_extract_multi(self: Any, texts: list[str], **_kw: Any) -> list[ExtractionResult]:
    return [next((r for marker, r in _REGISTRY.items() if marker in t), ExtractionResult()) for t in texts]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    unit = [1.0] + [0.0] * (EMBED_DIM - 1)
    return [unit[:] for _ in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return [1.0] + [0.0] * (EMBED_DIM - 1)


def _no_ef() -> ExpertiseConfig:
    return ExpertiseConfig(
        name="ns-delete-probe",
        events=EventExtractionConfig(enabled=False),
        facts=FactExtractionConfig(enabled=False),
    )


def _asyncpg_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real backends (requires make dev)",
)
class TestDeleteNamespacePgNeo4j:
    """End-to-end delete_namespace cascade against real Postgres + Neo4j."""

    @pytest.fixture(autouse=True)
    def _stub_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        monkeypatch.setattr("khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi", _stub_extract_multi)
        monkeypatch.setattr("khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch", _stub_embed_batch)
        monkeypatch.setattr("khora.extraction.embedders.litellm.LiteLLMEmbedder.embed", _stub_embed)

    @pytest.fixture
    def _pg_url(self) -> str:
        return os.environ.get("KHORA_DATABASE_URL", "postgresql://khora:khora@localhost:5434/khora")

    @pytest.fixture
    async def kb(self, _pg_url: str) -> AsyncIterator[Khora]:
        neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        config = KhoraConfig(database_url=_pg_url, neo4j_url=neo4j_url)
        config.storage.neo4j_user = neo4j_user
        config.storage.neo4j_password = neo4j_password
        config.llm.embedding_dimension = EMBED_DIM
        config.storage.embedding_dimension = EMBED_DIM
        config.pipelines.chunk_size = 1024
        config.pipelines.extract_entities = True
        config.pipelines.selective_extraction = False

        kb = Khora(config, run_migrations=True)
        await kb.connect()
        try:
            yield kb
        finally:
            await kb.disconnect()

    def _graph_driver(self, kb: Khora) -> Any:
        graph = kb.storage.graph
        assert graph is not None, "graph backend must be configured"
        backend = getattr(graph, "_backend", graph)
        driver = getattr(backend, "_driver", None)
        assert driver is not None, "Neo4j driver must be connected"
        return driver

    async def _pg_counts(self, pg_url: str, row_id: UUID) -> dict[str, int]:
        import asyncpg

        out: dict[str, int] = {}
        conn = await asyncpg.connect(_asyncpg_url(pg_url))
        try:
            for tbl in ("documents", "chunks", "khora_chunks", "entities", "relationships"):
                out[f"pg.{tbl}"] = await conn.fetchval(
                    f"SELECT count(*) FROM {tbl} WHERE namespace_id = $1::uuid",  # noqa: S608 - tbl is a hardcoded literal
                    row_id,
                )
        finally:
            await conn.close()
        return out

    async def _neo4j_counts(self, kb: Khora, row_id: UUID) -> dict[str, int]:
        driver = self._graph_driver(kb)
        ns = str(row_id)
        out: dict[str, int] = {}
        async with driver.session() as s:
            rec = await (await s.run("MATCH (e:Entity {namespace_id:$ns}) RETURN count(e) AS n", ns=ns)).single()
            out["neo4j.Entity_nodes"] = rec["n"] if rec else 0
            rec = await (
                await s.run(
                    "MATCH (:Entity {namespace_id:$ns})-[r]->(:Entity {namespace_id:$ns}) RETURN count(r) AS n",
                    ns=ns,
                )
            ).single()
            out["neo4j.rel_edges"] = rec["n"] if rec else 0
            rec = await (await s.run("MATCH (c:Chunk {namespace_id:$ns}) RETURN count(c) AS n", ns=ns)).single()
            out["neo4j.Chunk_nodes"] = rec["n"] if rec else 0
        return out

    async def _ingest_one(self, kb: Khora, stable: UUID) -> None:
        await kb.remember(
            content=DOC,
            namespace=stable,
            entity_types=["PERSON", "ORGANIZATION"],
            relationship_types=["WORKS_AT"],
            expertise=_no_ef(),
        )

    async def _lists(self, kb: Khora, stable: UUID, row_id: UUID, *, active_only: bool) -> bool:
        page = await kb.storage.list_namespaces(active_only=active_only, limit=1000)
        ids = {n.id for n in page.items} | {n.namespace_id for n in page.items}
        return stable in ids or row_id in ids

    @pytest.mark.asyncio
    async def test_delete_namespace_reclaims_pg_and_neo4j(self, kb: Khora, _pg_url: str) -> None:
        ns = await kb.create_namespace()
        stable, row_id = ns.namespace_id, ns.id

        await self._ingest_one(kb, stable)

        pg_before = await self._pg_counts(_pg_url, row_id)
        neo_before = await self._neo4j_counts(kb, row_id)
        assert pg_before["pg.documents"] == 1
        assert pg_before["pg.khora_chunks"] >= 1
        assert pg_before["pg.entities"] == 2
        assert neo_before["neo4j.Entity_nodes"] == 2
        assert neo_before["neo4j.rel_edges"] == 1

        result = await kb.delete_namespace(stable)
        assert not result.partial_failure, result.degradations
        assert result.namespaces_removed == 1
        assert result.documents_removed == 1
        assert row_id in result.removed_row_ids

        pg_after = await self._pg_counts(_pg_url, row_id)
        neo_after = await self._neo4j_counts(kb, row_id)
        assert pg_after == {
            "pg.documents": 0,
            "pg.chunks": 0,
            "pg.khora_chunks": 0,
            "pg.entities": 0,
            "pg.relationships": 0,
        }
        assert neo_after == {"neo4j.Entity_nodes": 0, "neo4j.rel_edges": 0, "neo4j.Chunk_nodes": 0}

        assert not await self._lists(kb, stable, row_id, active_only=True)
        assert not await self._lists(kb, stable, row_id, active_only=False)
        assert await kb.get_namespace(row_id) is None
        with pytest.raises(ValueError):
            await kb.storage.resolve_namespace(stable)

    @pytest.mark.asyncio
    async def test_delete_reclaims_deactivated_stranded_namespace(self, kb: Khora, _pg_url: str) -> None:
        ns = await kb.create_namespace()
        stable, row_id = ns.namespace_id, ns.id

        await self._ingest_one(kb, stable)
        # Strand it: is_active=False, no cascade. resolve/recall now dead-end.
        await kb.storage.deactivate_namespace(row_id)
        with pytest.raises(ValueError):
            await kb.storage.resolve_namespace(stable)

        # Reclaim it anyway — delete resolves WITHOUT the is_active filter.
        result = await kb.delete_namespace(stable)
        assert not result.partial_failure, result.degradations
        assert result.namespaces_removed == 1

        pg_after = await self._pg_counts(_pg_url, row_id)
        neo_after = await self._neo4j_counts(kb, row_id)
        assert sum(pg_after.values()) == 0, pg_after
        assert sum(neo_after.values()) == 0, neo_after
        assert not await self._lists(kb, stable, row_id, active_only=False)

    @pytest.mark.asyncio
    async def test_delete_unknown_namespace_is_a_noop(self, kb: Khora) -> None:
        result = await kb.delete_namespace(uuid4())
        assert result.namespaces_removed == 0
        assert result.removed_row_ids == []
        assert not result.partial_failure
