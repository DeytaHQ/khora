"""Cross-store live-set invariant for graph-side dream undo (#1275).

Phase 3 of the dream-on-graph umbrella (#1282). Before this leg, ``dream_undo``
reversed only the PG soft-deletes; the forward graph mirror (#1272/#1273) was
never undone, so undo was a HALF-REVERT that re-diverged the two stores.

This module asserts the convergence invariant: after apply + post-commit mirror
on a real pg+neo4j stack, ``dream_undo`` restores BOTH stores to byte-identical
pre-apply live sets - entities un-retired, the :EntityVersion/[:SUPERSEDES]
snapshot deleted, self-loops un-invalidated, and incident edges re-pointed back
onto the absorbed entity.

This FAILS on origin/main (PG reverts, graph keeps the merged shape): the
absorbed entity stays retired in graph recall, the self-loop stays invalidated,
and the incident edges stay on the canonical. It PASSES once ``dream_undo``
also reverses the graph mirror.

How to run locally::

    make dev   # starts postgres (5434) + neo4j (7688) via compose
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
    KHORA_NEO4J_URL=bolt://localhost:7688 \\
    KHORA_NEO4J_USERNAME=neo4j KHORA_NEO4J_PASSWORD=pleaseletmein \\
        uv run pytest tests/integration/dream/test_neo4j_dream_undo_integration.py -v
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from khora.config.schema import KhoraConfig
from khora.core.models.entity import Entity, Relationship
from khora.dream.api import dream_undo
from khora.dream.config import DreamConfig
from khora.dream.engines.vectorcypher.dedupe_entities import apply_vectorcypher_dedupe_entities
from khora.dream.plan import DreamOp, OpKind
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


@pytest.fixture
async def kb() -> AsyncIterator[Khora]:
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=NEO4J_URL)
    config.storage.neo4j_user = NEO4J_USER
    config.storage.neo4j_password = NEO4J_PASSWORD
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.dream = DreamConfig(enabled=True)
    kb = Khora(config, run_migrations=True)
    await kb.connect()
    try:
        yield kb
    finally:
        await kb.disconnect()


def _orchestrator(kb: Khora):
    from khora.dream.orchestrator import DreamOrchestrator

    return DreamOrchestrator(kb, kb._config.dream, sinks=[])


async def _seed_entity_both(kb: Khora, ns_row_id: UUID, name: str) -> UUID:
    ent = Entity(namespace_id=ns_row_id, name=name, entity_type="PERSON", description=name)
    await kb.storage.create_entity(ent)
    return ent.id


async def _insert_pg_relationship(
    kb: Khora, ns_row_id: UUID, rel_id: UUID, src: UUID, tgt: UUID, rel_type: str
) -> None:
    async with kb.storage.transaction() as txn:
        await txn.session.execute(
            text(
                "INSERT INTO relationships (id, namespace_id, source_entity_id, target_entity_id, "
                "relationship_type, description, properties, source_document_ids, source_chunk_ids, "
                "confidence, weight, metadata, created_at, updated_at) "
                "VALUES (:id, :ns, :src, :tgt, :rt, '', '{}'::jsonb, '{}', '{}', "
                "0.9, 1.0, '{}'::jsonb, now(), now())"
            ),
            {"id": rel_id, "ns": ns_row_id, "src": src, "tgt": tgt, "rt": rel_type},
        )


async def _seed_edge_both(kb: Khora, ns_row_id: UUID, src: UUID, tgt: UUID, rel_type: str) -> UUID:
    rel = Relationship(
        namespace_id=ns_row_id,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description="incident",
        confidence=0.9,
    )
    await kb.storage.create_relationship(rel)
    await _insert_pg_relationship(kb, ns_row_id, rel.id, src, tgt, rel_type)
    return rel.id


async def _live_pg_entity_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    async with kb.storage.transaction() as txn:
        rows = (
            await txn.session.execute(
                text(
                    "SELECT id FROM entities WHERE namespace_id = :ns AND (valid_until IS NULL OR valid_until > now())"
                ),
                {"ns": ns_row_id},
            )
        ).all()
    return {str(r.id) for r in rows}


async def _graph_entity_ids(kb: Khora, ns_row_id: UUID) -> set[str]:
    ents = await kb.storage.list_entities(ns_row_id, limit=1000)
    return {str(e.id) for e in ents}


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


async def _pg_relationship_endpoints(kb: Khora, ns_row_id: UUID) -> dict[str, tuple[str, str]]:
    async with kb.storage.transaction() as txn:
        rows = (
            await txn.session.execute(
                text(
                    "SELECT id, source_entity_id, target_entity_id FROM relationships "
                    "WHERE namespace_id = :ns AND valid_to IS NULL AND invalidated_at IS NULL "
                    "AND (valid_until IS NULL OR valid_until > now())"
                ),
                {"ns": ns_row_id},
            )
        ).all()
    return {str(r.id): (str(r.source_entity_id), str(r.target_entity_id)) for r in rows}


async def _graph_relationship_endpoints(kb: Khora, ns_row_id: UUID) -> dict[str, tuple[str, str]]:
    rels = await kb.storage.list_relationships(ns_row_id, limit=1000)
    return {str(r.id): (str(r.source_entity_id), str(r.target_entity_id)) for r in rels}


async def _entity_version_ids(kb: Khora, ns_row_id: UUID, absorbed: UUID) -> set[str]:
    """Return the :EntityVersion snapshot ids superseded by the absorbed node (graph)."""
    graph = getattr(kb.storage.graph, "_backend", kb.storage.graph)
    async with graph._session() as session:  # noqa: SLF001 - test introspection

        async def _ids(tx):  # noqa: ANN001, ANN202
            result = await tx.run(
                "MATCH (current:Entity {id: $aid, namespace_id: $ns})-[:SUPERSEDES]->(old:EntityVersion) "
                "RETURN old.id AS id",
                aid=str(absorbed),
                ns=str(ns_row_id),
            )
            return [r["id"] async for r in result]

        return set(await session.execute_read(_ids))


async def _entity_version_count(kb: Khora, ns_row_id: UUID, absorbed: UUID) -> int:
    """Count :EntityVersion snapshots referencing the absorbed node (graph)."""
    return len(await _entity_version_ids(kb, ns_row_id, absorbed))


async def _seed_prior_entity_version(kb: Khora, ns_row_id: UUID, entity_id: UUID) -> str:
    """Attach a pre-existing :EntityVersion snapshot to a live node (e.g. a prior
    document-replace version), stamped with a DIFFERENT version_valid_to than the
    dream retire will use. Returns the snapshot id."""
    snapshot_id = str(uuid4())
    graph = getattr(kb.storage.graph, "_backend", kb.storage.graph)
    async with graph._session() as session:  # noqa: SLF001 - test introspection

        async def _seed(tx):  # noqa: ANN001, ANN202
            await tx.run(
                "MATCH (current:Entity {id: $eid, namespace_id: $ns}) "
                "CREATE (old:EntityVersion {id: $sid, namespace_id: $ns, "
                "version_valid_to: '2000-01-01T00:00:00+00:00'}) "
                "CREATE (current)-[:SUPERSEDES {superseded_at: '2000-01-01T00:00:00+00:00'}]->(old)",
                eid=str(entity_id),
                ns=str(ns_row_id),
                sid=snapshot_id,
            )

        await session.execute_write(_seed)
    return snapshot_id


def _write_undo_file(
    *,
    base_dir: Path,
    namespace_id: UUID,
    run_id: UUID,
    op_id: UUID,
    op_type: str,
    before: dict,
) -> Path:
    date_dir = base_dir / str(namespace_id) / datetime.now(UTC).strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "dream-undo/1",
        "run_id": str(run_id),
        "namespace_id": str(namespace_id),
        "started_at": datetime.now(UTC).isoformat(),
        "ops": [
            {
                "op_id": str(op_id),
                "op_type": op_type,
                "before": before,
                "applied_at": datetime.now(UTC).isoformat(),
            }
        ],
    }
    path = date_dir / f"{run_id}.undo.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


@pytest.mark.asyncio
async def test_dedupe_undo_converges_pg_and_graph(kb: Khora, tmp_path: Path) -> None:
    """apply -> mirror -> dream_undo restores BOTH stores to the pre-apply live sets.

    Covers all three reverse legs: un-retire the absorbed entity (and delete its
    :EntityVersion snapshot), re-point the incident edge back onto the absorbed
    entity, and un-invalidate the self-loop.
    """
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme-corp-{uuid4().hex[:8]}")
    neighbor = await _seed_entity_both(kb, ns_row_id, f"vendor-{uuid4().hex[:8]}")

    # An incident edge (absorbed -> neighbor) that gets re-pointed, and an edge
    # (canonical -> absorbed) that becomes a self-loop after the merge.
    incident = await _seed_edge_both(kb, ns_row_id, absorbed, neighbor, "SUPPLIES")
    self_loop = await _seed_edge_both(kb, ns_row_id, canonical, absorbed, "RELATES_TO")

    # Snapshot the pre-apply live sets / endpoints in BOTH stores.
    pg_ent_before = await _live_pg_entity_ids(kb, ns_row_id)
    graph_ent_before = await _graph_entity_ids(kb, ns_row_id)
    pg_rel_before = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_rel_before = await _graph_relationship_ids(kb, ns_row_id)
    pg_eps_before = await _pg_relationship_endpoints(kb, ns_row_id)
    graph_eps_before = await _graph_relationship_endpoints(kb, ns_row_id)
    assert pg_ent_before == graph_ent_before
    assert pg_rel_before == graph_rel_before
    assert pg_eps_before == graph_eps_before

    # --- Apply + post-commit mirror ---
    orch = _orchestrator(kb)
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=({"merges": [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}]},),
        namespace_id=ns_row_id,
    )
    async with kb.storage.transaction() as txn:
        undo = await apply_vectorcypher_dedupe_entities(op, coordinator=kb.storage, session=txn.session)
    assert await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo) is None

    # Sanity: the apply diverged the live sets from the snapshot in both stores.
    assert str(absorbed) not in await _graph_entity_ids(kb, ns_row_id)
    assert str(self_loop) not in await _graph_relationship_ids(kb, ns_row_id)
    assert (await _graph_relationship_endpoints(kb, ns_row_id))[str(incident)] == (str(canonical), str(neighbor))
    assert await _entity_version_count(kb, ns_row_id, absorbed) == 1

    # --- Persist the undo file the way the orchestrator file sink does ---
    run_id = uuid4()
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=ns_stable,
        run_id=run_id,
        op_id=op.op_id,
        op_type=str(op.op_type),
        before=undo.before,
    )

    # --- Undo (reverses PG + graph) ---
    ok = await dream_undo(kb, op.op_id, base_dir=tmp_path)
    assert ok is True

    # --- Converged back to the pre-apply live sets in BOTH stores ---
    pg_ent_after = await _live_pg_entity_ids(kb, ns_row_id)
    graph_ent_after = await _graph_entity_ids(kb, ns_row_id)
    pg_rel_after = await _live_pg_relationship_ids(kb, ns_row_id)
    graph_rel_after = await _graph_relationship_ids(kb, ns_row_id)
    pg_eps_after = await _pg_relationship_endpoints(kb, ns_row_id)
    graph_eps_after = await _graph_relationship_endpoints(kb, ns_row_id)

    assert pg_ent_after == pg_ent_before
    assert graph_ent_after == graph_ent_before
    assert pg_ent_after == graph_ent_after
    assert pg_rel_after == pg_rel_before
    assert graph_rel_after == graph_rel_before
    assert pg_rel_after == graph_rel_after
    # Endpoints restored: the incident edge points back at the absorbed entity.
    assert pg_eps_after == pg_eps_before
    assert graph_eps_after == graph_eps_before
    assert pg_eps_after == graph_eps_after
    assert graph_eps_after[str(incident)] == (str(absorbed), str(neighbor))
    # The :EntityVersion snapshot is deleted on un-retire.
    assert await _entity_version_count(kb, ns_row_id, absorbed) == 0


@pytest.mark.asyncio
async def test_dedupe_undo_is_idempotent_on_graph(kb: Khora, tmp_path: Path) -> None:
    """A second dream_undo is a no-op and does not re-diverge the graph."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme2-{uuid4().hex[:8]}")
    neighbor = await _seed_entity_both(kb, ns_row_id, f"vendor-{uuid4().hex[:8]}")
    incident = await _seed_edge_both(kb, ns_row_id, absorbed, neighbor, "SUPPLIES")

    orch = _orchestrator(kb)
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=({"merges": [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}]},),
        namespace_id=ns_row_id,
    )
    async with kb.storage.transaction() as txn:
        undo = await apply_vectorcypher_dedupe_entities(op, coordinator=kb.storage, session=txn.session)
    assert await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo) is None

    run_id = uuid4()
    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=ns_stable,
        run_id=run_id,
        op_id=op.op_id,
        op_type=str(op.op_type),
        before=undo.before,
    )

    assert await dream_undo(kb, op.op_id, base_dir=tmp_path) is True
    graph_eps_first = await _graph_relationship_endpoints(kb, ns_row_id)
    graph_ent_first = await _graph_entity_ids(kb, ns_row_id)
    # The incident edge is re-pointed back onto the absorbed entity.
    assert graph_eps_first[str(incident)] == (str(absorbed), str(neighbor))

    # Second undo: the graph-side restore verbs are idempotent (match nothing
    # on replay), so the graph does NOT re-diverge - same endpoints, same live
    # entity set, no duplicate edge. (The PG reverse handler's rowcount-based
    # bool is non-idempotent for the edge re-point and is out of scope for
    # #1275, which guards the GRAPH convergence.)
    await dream_undo(kb, op.op_id, base_dir=tmp_path)
    graph_eps_second = await _graph_relationship_endpoints(kb, ns_row_id)
    graph_ent_second = await _graph_entity_ids(kb, ns_row_id)
    assert graph_eps_first == graph_eps_second
    assert graph_ent_first == graph_ent_second
    assert graph_eps_second[str(incident)] == (str(absorbed), str(neighbor))


@pytest.mark.asyncio
async def test_undo_preserves_prior_entity_version_chain(kb: Khora, tmp_path: Path) -> None:
    """Un-retiring the absorbed entity deletes ONLY the dream snapshot, not a
    pre-existing :EntityVersion chain (e.g. from an earlier document replace)."""
    ns = await kb.create_namespace()
    ns_stable = ns.namespace_id
    ns_row_id = await kb.storage.resolve_namespace(ns_stable)

    canonical = await _seed_entity_both(kb, ns_row_id, f"acme-{uuid4().hex[:8]}")
    absorbed = await _seed_entity_both(kb, ns_row_id, f"acme2-{uuid4().hex[:8]}")

    # The absorbed entity already carries a prior version snapshot (still live).
    prior_snapshot = await _seed_prior_entity_version(kb, ns_row_id, absorbed)
    assert prior_snapshot in await _entity_version_ids(kb, ns_row_id, absorbed)

    orch = _orchestrator(kb)
    op = DreamOp(
        op_id=uuid4(),
        phase="apply",
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        outputs=({"merges": [{"canonical_id": str(canonical), "absorbed_id": str(absorbed)}]},),
        namespace_id=ns_row_id,
    )
    async with kb.storage.transaction() as txn:
        undo = await apply_vectorcypher_dedupe_entities(op, coordinator=kb.storage, session=txn.session)
    assert await orch._mirror_dream_op(uuid4(), 0, ns_row_id, op, undo) is None
    # Now there are two snapshots: the prior one + the dream one.
    assert len(await _entity_version_ids(kb, ns_row_id, absorbed)) == 2

    _write_undo_file(
        base_dir=tmp_path,
        namespace_id=ns_stable,
        run_id=uuid4(),
        op_id=op.op_id,
        op_type=str(op.op_type),
        before=undo.before,
    )
    assert await dream_undo(kb, op.op_id, base_dir=tmp_path) is True

    # The absorbed entity is live again, and ONLY the prior snapshot remains.
    assert str(absorbed) in await _graph_entity_ids(kb, ns_row_id)
    remaining = await _entity_version_ids(kb, ns_row_id, absorbed)
    assert remaining == {prior_snapshot}
