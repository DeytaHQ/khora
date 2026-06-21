"""Cross-store live-set invariant for the contradiction-reconcile mirror (#1281).

Phase 5 (final) of the dream-on-graph umbrella (#1282). After a two-LLM-judged
contradiction reconcile + post-commit mirror on a real pg+Neo4j stack, the
judge-invalidated losing edge must be invisible to BOTH the PG ground-truth live
set and graph-preferring recall, and a triage row must land in dream_conflicts.
A judge that DEFERS (disagreement) must mutate nothing in either store but still
write a triage row.

The two-LLM judge is mocked by patching ``khora.config.llm.acompletion`` (the
chokepoint both judges call) with a model-aware async stub, so NO real LLM key
is needed and the agree / defer outcomes are deterministic.

How to run locally::

    make dev   # starts postgres (5434) + neo4j (7688) via compose
    KHORA_DATABASE_URL="postgresql://khora:khora@localhost:5434/khora" \\
    KHORA_NEO4J_URL="bolt://localhost:7688" KHORA_NEO4J_PASSWORD="pleaseletmein" \\
    NEO4J_INTEGRATION_TEST=1 UV_NO_SYNC=1 \\
        uv run pytest tests/integration/dream/test_neo4j_contradiction_reconcile_integration.py \\
        -o addopts="" -q
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from khora.config.schema import KhoraConfig
from khora.core.models.entity import Entity, Relationship
from khora.dream.config import DreamConfig
from khora.dream.plan import DreamScope, OpKind
from khora.khora import Khora

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
NEO4J_URL = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")


def _reachable(url: str, default_port: int) -> bool:
    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_reachable(DATABASE_URL, 5432) and _reachable(NEO4J_URL, 7687)),
        reason="pg+neo4j not reachable (run `make dev`)",
    ),
]

EMBED_DIM = 4

_VERIFIER_MODEL = "gpt-4o-mini"
_AUDITOR_MODEL = "claude-haiku-4.5"


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=NEO4J_URL)
    config.storage.neo4j_user = NEO4J_USER
    config.storage.neo4j_password = NEO4J_PASSWORD
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.dream = DreamConfig(
        enabled=True,
        contradiction_reconcile_enabled=True,
        contradiction_reconcile_model=_VERIFIER_MODEL,
        contradiction_reconcile_auditor_model=_AUDITOR_MODEL,
        contradiction_reconcile_min_confidence=0.6,
    )
    kb = Khora(config, run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        await kb.disconnect()


# ---------------------------------------------------------------------------
# Seeding + assertion helpers (mirror the #1272 mirror test shapes)
# ---------------------------------------------------------------------------


async def _seed_entity_both(kb: Khora, ns_row_id: UUID, name: str) -> UUID:
    ent = Entity(namespace_id=ns_row_id, name=name, entity_type="PERSON", description=name)
    await kb.storage.create_entity(ent)
    return ent.id


async def _seed_pg_edge(
    kb: Khora,
    ns_row_id: UUID,
    src: UUID,
    tgt: UUID,
    *,
    description: str,
    confidence: float,
    chunk_id: UUID,
    properties: dict[str, str] | None = None,
) -> UUID:
    """Insert one EMPLOYED_BY relationship row into PG only (distinct id).

    A contradiction is two *distinct* PG rows in the same (src, tgt, type)
    bucket; the graph (Neo4j) MERGEs edges on endpoints+type so it can hold at
    most ONE edge for the pair. We seed the graph edge separately
    (:func:`_seed_graph_edge`) so the test controls which row id the single
    graph edge carries.
    """
    rel_id = uuid4()
    async with kb.storage.transaction() as txn:
        await txn.session.execute(
            text(
                "INSERT INTO relationships (id, namespace_id, source_entity_id, target_entity_id, "
                "relationship_type, description, properties, source_document_ids, source_chunk_ids, "
                "confidence, weight, metadata, created_at, updated_at) "
                "VALUES (:id, :ns, :src, :tgt, 'EMPLOYED_BY', :desc, CAST(:props AS jsonb), '{}', "
                "ARRAY[:chunk]::uuid[], :conf, 1.0, '{}'::jsonb, now(), now())"
            ),
            {
                "id": rel_id,
                "ns": ns_row_id,
                "src": src,
                "tgt": tgt,
                "desc": description,
                "props": json.dumps(properties or {}),
                "chunk": chunk_id,
                "conf": confidence,
            },
        )
    return rel_id


async def _seed_graph_edge(kb: Khora, ns_row_id: UUID, rel_id: UUID, src: UUID, tgt: UUID) -> None:
    """Create the single graph edge for the pair carrying ``rel_id``.

    The #1271 ``soft_invalidate_relationships_batch`` verb matches by edge id, so
    pinning the graph edge's id lets the test assert the mirror invalidated the
    exact judge-chosen loser.
    """
    rel = Relationship(
        id=rel_id,
        namespace_id=ns_row_id,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type="EMPLOYED_BY",
        description="graph edge",
        confidence=0.9,
    )
    await kb.storage.create_relationship(rel)


async def _live_pg_relationship_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    async with kb.storage.transaction() as txn:
        rows = (
            await txn.session.execute(
                text(
                    "SELECT id FROM relationships WHERE namespace_id = :ns "
                    "AND valid_to IS NULL AND invalidated_at IS NULL "
                    "AND (valid_until IS NULL OR valid_until > now())"
                ),
                {"ns": ns_row_id},
            )
        ).all()
    return {str(r.id) for r in rows}


async def _graph_relationship_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    rels = await kb.storage.list_relationships(ns_row_id, limit=1000)
    return {str(r.id) for r in rels}


async def _triage_row(kb: Khora, ns_id: UUID, rel_a: UUID, rel_b: UUID) -> dict[str, Any] | None:
    """Read the triage row for a pair. ``ns_id`` is the namespace id the op
    stamped onto ``dream_conflicts`` (the stable id from ``kb.dream``)."""
    a, b = (rel_a, rel_b) if str(rel_a) < str(rel_b) else (rel_b, rel_a)
    async with kb.storage.transaction() as txn:
        row = (
            await txn.session.execute(
                text(
                    "SELECT resolution, loser_relationship_id, winner_relationship_id, resolved_by_op_id "
                    "FROM dream_conflicts WHERE namespace_id = :ns "
                    "AND relationship_a_id = :a AND relationship_b_id = :b"
                ),
                {"ns": ns_id, "a": a, "b": b},
            )
        ).first()
    if row is None:
        return None
    return {
        "resolution": row.resolution,
        "loser_relationship_id": str(row.loser_relationship_id) if row.loser_relationship_id else None,
        "winner_relationship_id": str(row.winner_relationship_id) if row.winner_relationship_id else None,
    }


def _install_judge(monkeypatch: pytest.MonkeyPatch, by_model: dict[str, str]) -> None:
    """Patch the LLM chokepoint both judges call with a model-aware stub."""

    async def _fake_acompletion(
        prompt: str, config: Any = None, *, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        model = getattr(config, "model", None)
        return by_model.get(model, json.dumps({"decision": "defer", "confidence": 0.5, "evidence_ids": []}))

    import khora.config.llm as llm_mod

    monkeypatch.setattr(llm_mod, "acompletion", _fake_acompletion)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_agree_soft_deletes_loser_in_both_stores(kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both judges agree -> the losing (lower-confidence) edge is soft-deleted in
    PG AND mirrored to Neo4j; the winner survives; a triage row records it."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    a = await _seed_entity_both(kb, ns_row_id, f"alice-{uuid4().hex[:8]}")
    company = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")

    chunk_winner, chunk_loser = uuid4(), uuid4()
    # Two distinct PG rows in the same (src, tgt, type) bucket = a contradiction.
    winner_id = await _seed_pg_edge(
        kb,
        ns_row_id,
        a,
        company,
        description="Currently employed, permanent",
        confidence=0.9,
        chunk_id=chunk_winner,
        properties={"status": "active"},
    )
    loser_id = await _seed_pg_edge(
        kb,
        ns_row_id,
        a,
        company,
        description="Employment ended in 2019",
        confidence=0.3,
        chunk_id=chunk_loser,
        properties={"status": "ended"},
    )
    # The graph holds exactly one edge for the pair (Neo4j MERGEs on
    # endpoints+type); pin its id to the LOSER so the mirror's invalidate-by-id
    # has a graph edge to act on - proving the soft-delete reaches Neo4j.
    await _seed_graph_edge(kb, ns_row_id, loser_id, a, company)

    # Pre-apply: both rows live in PG; the single graph edge (the loser) is live.
    pg_pre = await _live_pg_relationship_ids(kb, ns_row_id)
    assert {str(winner_id), str(loser_id)} <= pg_pre
    assert str(loser_id) in await _graph_relationship_ids(kb, ns_row_id)

    # Drive both judges to agree: invalidate the lower-confidence edge, citing its
    # grounding chunk. The planner orders the pair canonically by id, so 'a' is the
    # lexicographically-smaller id - resolve which side the loser landed on.
    loser_side = "a" if (str(loser_id) < str(winner_id)) else "b"
    verdict = json.dumps(
        {
            "decision": "invalidate",
            "loser": loser_side,
            "confidence": 0.95,
            "evidence_ids": [str(chunk_loser), str(chunk_winner)],
            "rationale": "ended contradicts active; lower confidence loses",
        }
    )
    _install_judge(monkeypatch, {_VERIFIER_MODEL: verdict, _AUDITOR_MODEL: verdict})

    result = await kb.dream(
        ns_stable,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE,)),
    )
    assert not result.metadata.get("degradations"), result.metadata.get("degradations")

    pg_live = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_live = await _graph_relationship_ids(kb, ns_row_id)

    # The loser is gone from BOTH stores (PG soft-delete + mirrored to Neo4j).
    assert str(loser_id) not in pg_live
    assert str(loser_id) not in graph_live
    # The winner survives in PG.
    assert str(winner_id) in pg_live

    # The losing row is soft-deleted, NOT hard-deleted (still present, valid_to set).
    async with kb.storage.transaction() as txn:
        row = (
            await txn.session.execute(
                text("SELECT valid_to FROM relationships WHERE id = :rid"),
                {"rid": loser_id},
            )
        ).first()
    assert row is not None, "the losing row must still exist (soft-delete, never hard-delete)"
    assert row.valid_to is not None

    # A triage row recorded the resolution.
    triage = await _triage_row(kb, ns_stable, winner_id, loser_id)
    assert triage is not None
    assert triage["resolution"] == "invalidated"
    assert triage["loser_relationship_id"] == str(loser_id)
    assert triage["winner_relationship_id"] == str(winner_id)


@pytest.mark.asyncio
async def test_reconcile_defer_mutates_nothing_but_writes_triage(kb: Khora, monkeypatch: pytest.MonkeyPatch) -> None:
    """Judges disagree -> no mutation in either store; a 'deferred' triage row."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    a = await _seed_entity_both(kb, ns_row_id, f"bob-{uuid4().hex[:8]}")
    company = await _seed_entity_both(kb, ns_row_id, f"globex-{uuid4().hex[:8]}")

    chunk_x, chunk_y = uuid4(), uuid4()
    rel_x = await _seed_pg_edge(
        kb,
        ns_row_id,
        a,
        company,
        description="Permanent staff",
        confidence=0.9,
        chunk_id=chunk_x,
        properties={"status": "active"},
    )
    rel_y = await _seed_pg_edge(
        kb,
        ns_row_id,
        a,
        company,
        description="Short contract",
        confidence=0.3,
        chunk_id=chunk_y,
        properties={"status": "ended"},
    )
    # One graph edge for the pair, pinned to rel_y.
    await _seed_graph_edge(kb, ns_row_id, rel_y, a, company)

    # Judges disagree -> defer.
    _install_judge(
        monkeypatch,
        {
            _VERIFIER_MODEL: json.dumps(
                {"decision": "invalidate", "loser": "b", "confidence": 0.9, "evidence_ids": [str(chunk_y)]}
            ),
            _AUDITOR_MODEL: json.dumps({"decision": "keep", "confidence": 0.9, "evidence_ids": [str(chunk_x)]}),
        },
    )

    result = await kb.dream(
        ns_stable,
        mode="apply",
        scope=DreamScope(op_kinds=(OpKind.VECTORCYPHER_CONTRADICTION_RECONCILE,)),
    )
    assert not result.metadata.get("degradations"), result.metadata.get("degradations")

    pg_live = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_live = await _graph_relationship_ids(kb, ns_row_id)
    # Nothing was mutated: both PG rows live; the graph edge live.
    assert {str(rel_x), str(rel_y)} <= pg_live
    assert str(rel_y) in graph_live

    # A 'deferred' triage row was still written (no loser).
    triage = await _triage_row(kb, ns_stable, rel_x, rel_y)
    assert triage is not None
    assert triage["resolution"] == "deferred"
    assert triage["loser_relationship_id"] is None
