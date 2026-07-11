"""Live-PostgreSQL integration test for per-namespace partial HNSW indexes (#1470).

Verifies on real Postgres + pgvector that the policy-gated promotion mechanism
in ``khora.storage.optimize`` works end-to-end:

1. ``promote_namespace_hnsw`` builds a partial HNSW index
   ``... WHERE namespace_id = <ns>`` on ``chunks`` (and ``entities``).
2. The PLANNER picks that partial index for a namespace-scoped vector query —
   the whole point of the feature (an EXPLAIN regression guard). pgvector
   estimates the partial and shared HNSW indexes at near-equal cost, so with
   both present the choice flutters run to run; the test therefore drops the
   shared float32 + halfvec indexes inside a rolled-back transaction so the
   partial index is the only vector index the planner can use, and asserts the
   plan is a clean ``Index Scan using ix_chunks_embedding_hnsw_ns_<hex>`` with
   no ``Rows Removed by Filter`` and no ``Seq Scan``.
3. ``demote_namespace_hnsw`` drops it (idempotent).
4. The POLICY gate ``maybe_promote_namespace`` is default-OFF and only promotes
   when enabled + over the row threshold + under the index ceiling, and retries
   a half-promoted namespace instead of reporting it already done.

Postgres-only (``CREATE INDEX CONCURRENTLY`` + HNSW). Skips when unreachable.

The test seeds its own throwaway namespace / document / chunks and removes them
(``ON DELETE CASCADE`` from ``memory_namespaces``) in a ``finally`` block, so it
never leaves rows or indexes behind in the shared dev DB.

Run with an explicit DB URL (the shell leaks a different one)::

    KHORA_DATABASE_URL="postgresql://khora:khora@localhost:5434/khora" \
        UV_NO_SYNC=1 uv run pytest \
        tests/integration/storage/test_partial_hnsw_promotion_pg.py \
        -o addopts="" -q
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from khora.storage.optimize import (
    _partial_hnsw_index_name,
    demote_namespace_hnsw,
    list_partial_hnsw_indexes,
    maybe_promote_namespace,
    promote_namespace_hnsw,
)

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


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _pg_reachable(), reason="PostgreSQL not reachable for partial-HNSW test"),
]

_DIM = 1536
# Hot-namespace row count for the EXPLAIN test. Empirically (compose PG 17 /
# pgvector 0.8) the planner only prefers an HNSW index over a top-N sort of the
# namespace's rows once the candidate set is large enough; below ~5k rows the
# sort is cheaper. 6000 sits comfortably above that knee, so — with the shared
# indexes dropped in-tx — the planner deterministically picks the partial index.
_HOT_ROWS = 6000
# The policy-gate tests only exercise the row-count / ceiling logic, not the
# planner, so they seed a small, fast namespace.
_POLICY_ROWS = 50
# Query vector: a fixed literal so the EXPLAIN plan is deterministic.
_QUERY_VEC = "[" + ",".join(["0.5"] * _DIM) + "]"


def _rand_vec_sql() -> str:
    """SQL fragment producing a random ``vector(1536)`` literal.

    ``_DIM`` is a module int constant (1536), not user input, so the f-string is
    injection-safe.
    """
    return f"(SELECT array_agg(random())::vector FROM generate_series(1, {_DIM}))"  # noqa: S608


async def _seed(engine, namespace_id, *, hot_rows: int) -> None:
    """Seed a namespace with ``hot_rows`` chunks (and its FK parents).

    ``chunks.namespace_id`` FKs to ``memory_namespaces.id`` and
    ``chunks.document_id`` FKs to ``documents.id``, so the namespace and a
    document are registered first.
    """
    doc_id = uuid4()
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(
            sa.text(
                "INSERT INTO memory_namespaces (id, namespace_id, version, is_active, tenancy_mode) "
                "VALUES (:id, :nid, 1, true, 'shared')"
            ),
            {"id": namespace_id, "nid": namespace_id},
        )
        await conn.execute(
            sa.text("INSERT INTO documents (id, namespace_id, content) VALUES (:id, :ns, :c)"),
            {"id": doc_id, "ns": namespace_id, "c": "seed doc"},
        )
        await conn.execute(
            sa.text(
                "INSERT INTO chunks (id, namespace_id, document_id, content, embedding) "
                "SELECT gen_random_uuid(), :ns, :doc, 'seed chunk', " + _rand_vec_sql() + " "
                "FROM generate_series(1, :n)"
            ),
            {"ns": namespace_id, "doc": doc_id, "n": hot_rows},
        )
        await conn.execute(sa.text("ANALYZE chunks"))


async def _cleanup(engine, namespace_id) -> None:
    # Drop partial indexes first (best-effort), then cascade-delete the seed
    # namespace (removes its document + chunks via ON DELETE CASCADE).
    try:
        await demote_namespace_hnsw(engine, namespace_id)
    except Exception as exc:  # noqa: BLE001
        # Best-effort cleanup; log rather than swallow so a flaky drop is
        # visible. (S110 is repo-ignored under tests/**, so this is not a lint
        # requirement — it just aids debugging.)
        print(f"best-effort demote_namespace_hnsw cleanup failed: {exc}")
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(
            sa.text("DELETE FROM memory_namespaces WHERE id = :id"),
            {"id": namespace_id},
        )
        await conn.execute(sa.text("ANALYZE chunks"))


async def _explain_scoped_query(engine, namespace_id) -> str:
    """EXPLAIN the scoped query with the SHARED HNSW indexes made unavailable.

    pgvector estimates the partial and the shared HNSW indexes at near-equal
    cost, so when both are present the planner's choice between them is a
    coin-flip that flutters run to run — not a property this test can assert
    deterministically. What the mechanism actually *guarantees* is that the
    partial index is a valid, chosen access path for the scoped query that
    serves it WITHOUT post-filtering out-of-namespace rows. To assert exactly
    that, we drop the shared float32 + halfvec HNSW indexes inside a
    transaction (so the partial index is the only vector index the planner can
    use for this ORDER BY), run EXPLAIN, then ROLLBACK — the shared indexes are
    restored untouched. This is fully deterministic and does not depend on
    background-noise volume.
    """
    async with engine.connect() as conn:
        # Do NOT wrap in `async with conn.begin()`: rolling back inside that
        # block closes the transaction the context manager still owns. Instead
        # let the connection autobegin on the first execute, then roll back
        # explicitly so the in-tx DROPs are undone and the shared indexes
        # survive. Mirror khora's shipped query-time settings (pgvector.py
        # search_similar); SET LOCAL scopes them to this transaction.
        try:
            await conn.execute(sa.text("DROP INDEX ix_chunks_embedding_hnsw"))
            await conn.execute(sa.text("DROP INDEX IF EXISTS ix_chunks_embedding_halfvec_hnsw"))
            await conn.execute(sa.text("SET LOCAL hnsw.ef_search = 100"))
            await conn.execute(sa.text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            rows = await conn.execute(
                sa.text(
                    "EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) "
                    "SELECT id FROM chunks WHERE namespace_id = :ns "
                    "ORDER BY embedding <=> :qv LIMIT 10"
                ),
                {"ns": namespace_id, "qv": _QUERY_VEC},
            )
            plan = "\n".join(r[0] for r in rows)
        finally:
            await conn.rollback()
    return plan


@pytest.mark.asyncio
async def test_planner_uses_partial_hnsw_index_for_scoped_query() -> None:
    """EXPLAIN regression guard: a scoped query uses the partial HNSW index."""
    engine = create_async_engine(DATABASE_URL)
    ns = uuid4()
    try:
        await _seed(engine, ns, hot_rows=_HOT_ROWS)

        result = await promote_namespace_hnsw(engine, ns)
        assert not result["errors"], result["errors"]
        chunk_idx = _partial_hnsw_index_name("chunks", ns)
        entity_idx = _partial_hnsw_index_name("entities", ns)
        assert chunk_idx in result["indexes"]
        assert entity_idx in result["indexes"]

        listed = await list_partial_hnsw_indexes(engine)
        assert chunk_idx in listed
        assert entity_idx in listed
        # The discovery filter must not match the shared index.
        assert "ix_chunks_embedding_hnsw" not in listed

        plan = await _explain_scoped_query(engine, ns)
        # The load-bearing assertion: with the shared indexes dropped in-tx, the
        # planner must choose the partial index by name and serve the scoped
        # query WITHOUT post-filtering out-of-namespace rows. A shared-index
        # fallback would show "ix_chunks_embedding_hnsw" + "Rows Removed by
        # Filter"; a seq-scan fallback would show "Seq Scan".
        assert chunk_idx in plan, f"expected partial index in plan, got:\n{plan}"
        assert "Rows Removed by Filter" not in plan, f"unexpected post-filter:\n{plan}"
        assert "Seq Scan" not in plan, f"unexpected seq scan:\n{plan}"

        # Idempotent re-promote is a no-op (index already exists).
        again = await promote_namespace_hnsw(engine, ns)
        assert again["indexes_created"] == 0
        assert not again["errors"]

        # Demote drops both indexes; re-listing no longer finds them.
        dropped = await demote_namespace_hnsw(engine, ns)
        assert dropped["indexes_dropped"] == 2
        listed_after = await list_partial_hnsw_indexes(engine)
        assert chunk_idx not in listed_after
        assert entity_idx not in listed_after
    finally:
        await _cleanup(engine, ns)
        await engine.dispose()


@pytest.mark.asyncio
async def test_policy_gate_default_off_and_thresholds() -> None:
    """maybe_promote_namespace is default-OFF and honours the row threshold."""
    engine = create_async_engine(DATABASE_URL)
    ns = uuid4()
    try:
        await _seed(engine, ns, hot_rows=_POLICY_ROWS)

        # Disabled: no-op regardless of row count.
        r = await maybe_promote_namespace(engine, ns, enabled=False, min_rows=1, max_indexes=64)
        assert r == {"promoted": False, "reason": "disabled", "row_count": 0}
        assert _partial_hnsw_index_name("chunks", ns) not in await list_partial_hnsw_indexes(engine)

        # Enabled but below threshold: refused with row_count reported.
        r = await maybe_promote_namespace(engine, ns, enabled=True, min_rows=_POLICY_ROWS + 1, max_indexes=64)
        assert r["promoted"] is False
        assert r["reason"] == "below_min_rows"
        assert r["row_count"] == _POLICY_ROWS

        # Enabled and over threshold: promotes.
        r = await maybe_promote_namespace(engine, ns, enabled=True, min_rows=_POLICY_ROWS, max_indexes=64)
        assert r["promoted"] is True
        assert r["reason"] == "promoted"
        assert r["row_count"] == _POLICY_ROWS
        assert _partial_hnsw_index_name("chunks", ns) in await list_partial_hnsw_indexes(engine)

        # Second call is idempotent (already promoted, not double-counted).
        r = await maybe_promote_namespace(engine, ns, enabled=True, min_rows=_POLICY_ROWS, max_indexes=64)
        assert r["promoted"] is False
        assert r["reason"] == "already_promoted"
    finally:
        await _cleanup(engine, ns)
        await engine.dispose()


@pytest.mark.asyncio
async def test_policy_gate_respects_index_ceiling() -> None:
    """The ceiling refuses promotion once max_indexes partial indexes exist."""
    engine = create_async_engine(DATABASE_URL)
    ns = uuid4()
    try:
        await _seed(engine, ns, hot_rows=_POLICY_ROWS)
        # max_indexes=0 makes any promotion hit the ceiling immediately, even
        # with zero existing partial indexes — a clean unit of the guard that
        # does not depend on other namespaces' indexes on the shared table.
        r = await maybe_promote_namespace(engine, ns, enabled=True, min_rows=_POLICY_ROWS, max_indexes=0)
        assert r["promoted"] is False
        assert r["reason"] == "ceiling_reached"
        assert r["row_count"] == _POLICY_ROWS
        assert _partial_hnsw_index_name("chunks", ns) not in await list_partial_hnsw_indexes(engine)
    finally:
        await _cleanup(engine, ns)
        await engine.dispose()


@pytest.mark.asyncio
async def test_policy_gate_retries_half_promoted_namespace() -> None:
    """A namespace with only the chunks index (entities failed) is retried.

    Reproduces the partial-failure case: promote_namespace_hnsw captures a
    per-table error instead of raising, so a namespace can end up with the
    chunks partial index but not the entities one. maybe_promote_namespace must
    NOT short-circuit as ``already_promoted`` — it must fall through and build
    the missing entities index.
    """
    engine = create_async_engine(DATABASE_URL)
    ns = uuid4()
    chunk_idx = _partial_hnsw_index_name("chunks", ns)
    entity_idx = _partial_hnsw_index_name("entities", ns)
    try:
        await _seed(engine, ns, hot_rows=_POLICY_ROWS)

        # Simulate a half-promotion: build only the chunks partial index.
        async with engine.connect() as conn:
            await conn.execution_options(isolation_level="AUTOCOMMIT")
            await conn.execute(
                sa.text(
                    f"CREATE INDEX {chunk_idx} ON chunks USING hnsw (embedding vector_cosine_ops) "
                    f"WHERE namespace_id = '{ns}'::uuid"
                )
            )
        assert chunk_idx in await list_partial_hnsw_indexes(engine, table="chunks")
        assert entity_idx not in await list_partial_hnsw_indexes(engine, table="entities")

        # Must retry, not report already-promoted.
        r = await maybe_promote_namespace(engine, ns, enabled=True, min_rows=_POLICY_ROWS, max_indexes=64)
        assert r["promoted"] is True
        assert r["reason"] == "promoted"
        # The missing entities index now exists.
        assert entity_idx in await list_partial_hnsw_indexes(engine, table="entities")

        # Now fully promoted — a further call short-circuits.
        r = await maybe_promote_namespace(engine, ns, enabled=True, min_rows=_POLICY_ROWS, max_indexes=64)
        assert r["promoted"] is False
        assert r["reason"] == "already_promoted"
    finally:
        await _cleanup(engine, ns)
        await engine.dispose()
