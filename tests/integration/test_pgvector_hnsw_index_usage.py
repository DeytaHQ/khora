"""Regression tests for #1407: pgvector similarity search must be HNSW
index-backed and correct under selective filters.

Bug: all SQL vector search ordered by the wrapped similarity expression
descending - ``ORDER BY (1 - (embedding <=> :q)) DESC`` - which pgvector's
HNSW index cannot satisfy (it only serves ``ORDER BY embedding <=> :q ASC``).
Every query seq-scanned the whole table: ~1-3 ms at benchmark scale
(invisible), 10-60 s at 10M rows (fatal).

These tests capture the REAL SQL the backend emits (via a
``before_cursor_execute`` listener), re-run it under ``EXPLAIN (FORMAT
JSON)``, and assert the plan uses the HNSW index - plus:

* namespace-filtered search returns a full result set (the
  ``hnsw.iterative_scan = relaxed_order`` guarantee - without it, a
  namespace holding a minority of rows can starve below ``limit``);
* score parity: on a namespace fully covered by ``limit``, the new
  ascending-distance path returns identical ids, ordering, and similarity
  values to the old ``(1 - distance) DESC`` formula, including the
  ``min_similarity`` floor.

Gated on PostgreSQL reachability (``KHORA_DATABASE_URL``, defaults to the
``make dev`` Postgres). Requires migrations applied (HNSW indexes from
migrations 007/018).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy import event

from khora.storage.backends.pgvector import PgVectorBackend

EMBED_DIM = 1536
# Majority namespace: enough rows that the planner picks the HNSW index on
# its own (verified empirically: at ~1.5k rows the ascending-distance form
# gets an Index Scan while the legacy DESC form still seq-scans).
NS_BIG_ROWS = 1200
# Minority namespace: small share of the table, exercises post-index
# filtering.
NS_SMALL_ROWS = 300
# Parity namespace: fully covered by a single ``limit`` so exact old-formula
# SQL and the ANN path must agree on the complete row set.
NS_PARITY_ROWS = 40


def _database_url() -> str:
    return os.environ.get(
        "KHORA_DATABASE_URL",
        "postgresql+asyncpg://khora:khora@localhost:5432/khora",
    )


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(_database_url().replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _pg_reachable(), reason="PostgreSQL not reachable (run `make dev`)"),
]


def _unit_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    vecs = rng.standard_normal((n, EMBED_DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def _vec_str(vec: np.ndarray) -> str:
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


@pytest.fixture
async def backend() -> AsyncIterator[PgVectorBackend]:
    be = PgVectorBackend(database_url=_database_url(), embedding_dimension=EMBED_DIM)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


@pytest.fixture
async def seeded(backend: PgVectorBackend) -> AsyncIterator[dict]:
    """Seed three namespaces of chunks + one of entities; tear down by
    deleting the namespaces (FK cascade removes chunks; entities are removed
    explicitly since they don't FK to memory_namespaces)."""
    rng = np.random.default_rng(1407)
    ns_big, ns_small, ns_parity = uuid4(), uuid4(), uuid4()
    ns_ids = [ns_big, ns_small, ns_parity]
    doc_ids = {ns: uuid4() for ns in ns_ids}

    engine = backend._engine
    assert engine is not None
    async with engine.begin() as conn:
        for ns in ns_ids:
            await conn.execute(
                sa.text(
                    "INSERT INTO memory_namespaces (id, namespace_id, version, "
                    "is_active, tenancy_mode, created_at, updated_at) "
                    "VALUES (:id, :id, 1, true, 'shared', NOW(), NOW())"
                ),
                {"id": ns},
            )
            await conn.execute(
                sa.text("INSERT INTO documents (id, namespace_id, content) VALUES (:id, :ns, 'seed doc')"),
                {"id": doc_ids[ns], "ns": ns},
            )

        chunk_insert = sa.text(
            "INSERT INTO chunks (id, namespace_id, document_id, content, chunk_index, embedding) "
            "VALUES (:id, :ns, :doc, :content, :idx, CAST(:emb AS vector))"
        )
        parity_vecs: list[np.ndarray] = []
        for ns, n_rows in ((ns_big, NS_BIG_ROWS), (ns_small, NS_SMALL_ROWS), (ns_parity, NS_PARITY_ROWS)):
            vecs = _unit_rows(rng, n_rows)
            if ns is ns_parity:
                parity_vecs = list(vecs)
            await conn.execute(
                chunk_insert,
                [
                    {
                        "id": uuid4(),
                        "ns": ns,
                        "doc": doc_ids[ns],
                        "content": f"chunk {i}",
                        "idx": i,
                        "emb": _vec_str(vecs[i]),
                    }
                    for i in range(n_rows)
                ],
            )

        entity_insert = sa.text(
            "INSERT INTO entities (id, namespace_id, name, entity_type, embedding) "
            "VALUES (:id, :ns, :name, 'CONCEPT', CAST(:emb AS vector))"
        )
        for ns, n_rows in ((ns_big, NS_BIG_ROWS), (ns_parity, NS_PARITY_ROWS)):
            vecs = _unit_rows(rng, n_rows)
            await conn.execute(
                entity_insert,
                [
                    {"id": uuid4(), "ns": ns, "name": f"entity-{ns}-{i}", "emb": _vec_str(vecs[i])}
                    for i in range(n_rows)
                ],
            )

    # ANALYZE must COMMIT - pg_statistic updates are MVCC-transactional, so
    # an autobegun-then-rolled-back connection would silently discard them.
    async with engine.begin() as conn:
        await conn.execute(sa.text("ANALYZE chunks"))
        await conn.execute(sa.text("ANALYZE entities"))

    try:
        yield {
            "ns_big": ns_big,
            "ns_small": ns_small,
            "ns_parity": ns_parity,
            "query_vec": list(map(float, _unit_rows(rng, 1)[0])),
            "parity_vecs": parity_vecs,
        }
    finally:
        async with engine.begin() as conn:
            for ns in ns_ids:
                await conn.execute(sa.text("DELETE FROM entities WHERE namespace_id = :ns"), {"ns": ns})
                await conn.execute(sa.text("DELETE FROM memory_namespaces WHERE id = :id"), {"id": ns})


async def _explain_last_search(backend: PgVectorBackend, run_search, table: str) -> str:
    """Run *run_search* while capturing the SQL the backend actually emits,
    then re-run the captured similarity SELECT under EXPLAIN (FORMAT JSON)
    and return the plan as a string."""
    engine = backend._engine
    assert engine is not None
    captured: list[tuple[str, tuple]] = []

    def _capture(conn, cursor, statement, parameters, context, executemany) -> None:
        if "ORDER BY" in statement and table in statement:
            captured.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        await run_search()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)

    assert captured, f"no similarity SELECT against {table} was captured"
    statement, parameters = captured[-1]

    async with engine.connect() as conn:
        result = await conn.exec_driver_sql(f"EXPLAIN (FORMAT JSON) {statement}", parameters)
        plan = result.scalar()
    return plan if isinstance(plan, str) else json.dumps(plan)


@pytest.mark.asyncio
async def test_chunk_search_uses_hnsw_index(backend: PgVectorBackend, seeded: dict) -> None:
    """The EXACT query ``search_similar`` emits must be planned as an HNSW
    Index Scan - the legacy ``(1 - distance) DESC`` form seq-scans."""
    ns = seeded["ns_big"]

    async def _run() -> None:
        results = await backend.search_similar(ns, seeded["query_vec"], limit=10)
        assert len(results) == 10

    plan = await _explain_last_search(backend, _run, "chunks")
    assert "Index Scan" in plan, f"expected HNSW Index Scan, got plan: {plan}"
    expected_index = "ix_chunks_embedding_halfvec_hnsw" if backend.halfvec_enabled else "ix_chunks_embedding_hnsw"
    assert expected_index in plan, f"expected {expected_index} in plan: {plan}"


@pytest.mark.asyncio
async def test_entity_search_uses_hnsw_index(backend: PgVectorBackend, seeded: dict) -> None:
    """``search_similar_entities`` (hit on every VectorCypher recall) must be
    index-backed too."""
    ns = seeded["ns_big"]

    async def _run() -> None:
        results = await backend.search_similar_entities(ns, seeded["query_vec"], limit=10)
        assert len(results) == 10

    plan = await _explain_last_search(backend, _run, "entities")
    assert "Index Scan" in plan, f"expected HNSW Index Scan, got plan: {plan}"
    expected_index = "ix_entities_embedding_halfvec_hnsw" if backend.halfvec_enabled else "ix_entities_embedding_hnsw"
    assert expected_index in plan, f"expected {expected_index} in plan: {plan}"


@pytest.mark.asyncio
async def test_minority_namespace_returns_full_result_set(backend: PgVectorBackend, seeded: dict) -> None:
    """A namespace holding a minority of table rows must still fill ``limit``
    - pgvector post-filters HNSW candidates, so without relaxed_order
    iterative scan the result set can starve below ``limit``."""
    ns = seeded["ns_small"]
    results = await backend.search_similar(ns, seeded["query_vec"], limit=20)
    assert len(results) == 20
    assert all(chunk.namespace_id == ns for chunk, _score in results)
    sims = [score for _chunk, score in results]
    assert sims == sorted(sims, reverse=True), "results must be similarity-DESC ordered"


@pytest.mark.asyncio
async def test_score_parity_with_legacy_formula(backend: PgVectorBackend, seeded: dict) -> None:
    """On a namespace fully covered by ``limit``, the index-backed path must
    return identical ids, ordering, and similarity values to the legacy
    ``ORDER BY (1 - distance) DESC`` SQL (which the planner executes as an
    exact seq scan + sort)."""
    ns = seeded["ns_parity"]
    query_vec = seeded["query_vec"]
    limit = NS_PARITY_ROWS

    if backend.halfvec_enabled:
        legacy_sim = f"1 - (CAST(embedding AS halfvec({EMBED_DIM})) <=> CAST(:q AS halfvec({EMBED_DIM})))"
    else:
        legacy_sim = f"1 - (embedding <=> CAST(:q AS vector({EMBED_DIM})))"

    engine = backend._engine
    assert engine is not None
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.text(
                # noqa-justification: legacy_sim is a static expression built
                # from the EMBED_DIM constant; all values bind as parameters.
                f"SELECT id, {legacy_sim} AS similarity FROM chunks "  # noqa: S608
                "WHERE namespace_id = :ns AND embedding IS NOT NULL "
                "ORDER BY similarity DESC LIMIT :lim"
            ),
            {"q": _vec_str(np.asarray(query_vec)), "ns": ns, "lim": limit},
        )
        legacy = [(row.id, row.similarity) for row in result.all()]

    new = await backend.search_similar(ns, query_vec, limit=limit)
    new_pairs = [(chunk.id, score) for chunk, score in new]

    assert [i for i, _s in new_pairs] == [i for i, _s in legacy], "id ordering must match the legacy formula"
    for (_nid, nscore), (_lid, lscore) in zip(new_pairs, legacy, strict=True):
        assert nscore == pytest.approx(lscore, abs=1e-9)

    # min_similarity floor parity: the new path post-filters on the returned
    # similarity; the legacy path expressed it as WHERE similarity >= floor.
    floor = legacy[len(legacy) // 2][1]  # median similarity - splits the set
    legacy_floored = [(i, s) for i, s in legacy if s >= floor]
    new_floored = await backend.search_similar(ns, query_vec, limit=limit, min_similarity=floor)
    assert [c.id for c, _s in new_floored] == [i for i, _s in legacy_floored]
