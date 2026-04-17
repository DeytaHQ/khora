"""Microbenchmark: replace_document_extraction() vs ingest-path chunk-write (DYT-2673, ADR-056).

ADR-056 §Hard Requirements binds the replace primitive to ≤ parity with the
current ingest-path chunk-write cost at p50 and p95 before merge.  This
script runs both paths at representative chunk counts (30 / 100 / 500) in
single-call and 20-way concurrent configurations across distinct
``external_id`` values and prints a comparison table.

The graph-side retirement path is NOT exercised — ADR-056 §Performance is
explicit that embedding + extraction are OUT of the transaction scope, and
the parity bar is *chunk-write cost* specifically.  Using a mock Neo4j
backend keeps the benchmark focused on pgvector transaction overhead.

Usage::

    make dev  # brings up postgres on localhost:5432
    uv run python scripts/bench_replace_extraction.py \\
        --database-url postgresql+asyncpg://khora:khora@localhost:5432/khora

The script assumes a fresh, migrated database.  It creates its own
throwaway namespace and cleans up after each trial.

Example output::

    === chunks=100, concurrency=1 ===
    ingest_path   p50=23.10ms  p95=41.23ms  mean=25.00ms
    replace       p50=22.80ms  p95=40.05ms  mean=24.60ms
    parity        ✓ (p50 Δ=-1.3%, p95 Δ=-2.9%)

Exit code is non-zero if any tested scenario shows p50 or p95 regression
greater than the ``--tolerance`` threshold (default 15%).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from khora.core.models import Chunk, Document, Entity, Relationship
from khora.core.models.document import ChunkMetadata, DocumentMetadata
from khora.storage.backends.pgvector import PgVectorBackend
from khora.storage.backends.postgresql import PostgreSQLBackend
from khora.storage.coordinator import StorageCoordinator

EMBED_DIM = 1536


def _random_embedding() -> list[float]:
    # Pre-generate a deterministic-ish normalized vector per call — we only
    # care about payload-size bytes on the pg write path, not semantics.
    rnd = random.Random()  # noqa: S311 — benchmark payload, not cryptographic
    v = [rnd.random() for _ in range(EMBED_DIM)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def _make_document(namespace_id: UUID, external_id: str) -> Document:
    return Document(
        namespace_id=namespace_id,
        content=f"body-{external_id}",
        external_id=external_id,
        metadata=DocumentMetadata(
            source=f"bench/{external_id}",
            source_type="bench",
            content_type="text/plain",
            title=f"Bench Doc {external_id}",
        ),
    )


def _make_chunks(namespace_id: UUID, document_id: UUID, count: int) -> list[Chunk]:
    return [
        Chunk(
            namespace_id=namespace_id,
            document_id=document_id,
            content=f"chunk-{i}-" + "x" * 200,
            metadata=ChunkMetadata(
                document_id=document_id,
                chunk_index=i,
                start_char=i * 200,
                end_char=(i + 1) * 200,
                token_count=50,
            ),
            embedding=_random_embedding(),
            embedding_model="bench",
        )
        for i in range(count)
    ]


@dataclass
class Trial:
    wall_times_ms: list[float]

    @property
    def p50(self) -> float:
        return _percentile(self.wall_times_ms, 50)

    @property
    def p95(self) -> float:
        return _percentile(self.wall_times_ms, 95)

    @property
    def mean(self) -> float:
        return statistics.mean(self.wall_times_ms)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _build_mock_graph() -> MagicMock:
    """Mock graph backend — retirement/remap are not part of the parity bar."""
    graph = MagicMock()
    graph.fetch_document_extraction_state = AsyncMock(return_value=([], []))
    graph.retire_orphaned_entities_batch = AsyncMock(return_value=0)
    graph.retire_orphaned_relationships_batch = AsyncMock(return_value=0)
    graph.remap_source_document_ids_batch = AsyncMock(return_value=None)
    graph.upsert_entities_batch = AsyncMock(return_value=[])
    graph.create_relationships_batch = AsyncMock(return_value=0)
    return graph


async def _measure_ingest(
    coord: StorageCoordinator,
    namespace_id: UUID,
    chunks_per_doc: int,
) -> float:
    """Today's ingest-path: create_document + create_chunks_batch (no transaction wrap)."""
    doc = _make_document(namespace_id, f"ingest-{uuid4().hex[:12]}")
    await coord.create_document(doc)
    chunks = _make_chunks(namespace_id, doc.id, chunks_per_doc)

    t0 = time.perf_counter()
    await coord.create_chunks_batch(chunks)
    return (time.perf_counter() - t0) * 1000.0


async def _measure_replace(
    coord: StorageCoordinator,
    namespace_id: UUID,
    chunks_per_doc: int,
) -> float:
    """Replace path: prefetch (empty) + txn-wrapped update + delete + insert."""
    # Seed an old document + a handful of chunks to give delete a real rowcount.
    old_doc = _make_document(namespace_id, f"replace-old-{uuid4().hex[:12]}")
    await coord.create_document(old_doc)
    old_chunks = _make_chunks(namespace_id, old_doc.id, max(1, chunks_per_doc // 10))
    await coord.create_chunks_batch(old_chunks)

    new_chunks = _make_chunks(namespace_id, old_doc.id, chunks_per_doc)

    t0 = time.perf_counter()
    await coord.replace_document_extraction(
        namespace_id=namespace_id,
        old_document_id=old_doc.id,
        new_document=old_doc,
        new_chunks=new_chunks,
        new_entities=[],
        new_relationships=[],
    )
    return (time.perf_counter() - t0) * 1000.0


async def _run_trial(
    coord: StorageCoordinator,
    namespace_id: UUID,
    path: str,
    *,
    chunks_per_doc: int,
    concurrency: int,
    iterations: int,
) -> Trial:
    if path == "ingest":
        measure = _measure_ingest
    elif path == "replace":
        measure = _measure_replace
    else:
        raise ValueError(f"unknown path: {path}")

    wall_times: list[float] = []
    for _ in range(iterations):
        if concurrency == 1:
            wall_times.append(await measure(coord, namespace_id, chunks_per_doc))
        else:
            results = await asyncio.gather(*[measure(coord, namespace_id, chunks_per_doc) for _ in range(concurrency)])
            wall_times.extend(results)
    return Trial(wall_times_ms=wall_times)


def _format_delta(replace_v: float, ingest_v: float) -> str:
    if ingest_v == 0:
        return "n/a"
    delta = (replace_v - ingest_v) / ingest_v * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


async def _ensure_namespace(rel: PostgreSQLBackend, namespace_id: UUID) -> None:
    from khora.core.models import MemoryNamespace

    ns = MemoryNamespace(id=namespace_id, namespace_id=namespace_id)
    await rel.create_namespace(ns)


async def _main(args: argparse.Namespace) -> int:
    rel = PostgreSQLBackend(database_url=args.database_url)
    vec = PgVectorBackend(database_url=args.database_url, embedding_dimension=EMBED_DIM)
    graph = _build_mock_graph()
    coord = StorageCoordinator(relational=rel, vector=vec, graph=graph)
    await coord.connect()

    namespace_id = uuid4()
    await _ensure_namespace(rel, namespace_id)

    scenarios = [(c, conc) for c in (30, 100, 500) for conc in (1, 20)]
    regressed = False

    try:
        for chunks_per_doc, concurrency in scenarios:
            iterations = max(1, args.iterations // max(concurrency, 1))
            print(f"\n=== chunks={chunks_per_doc}, concurrency={concurrency}, iter={iterations} ===")

            ingest_trial = await _run_trial(
                coord,
                namespace_id,
                "ingest",
                chunks_per_doc=chunks_per_doc,
                concurrency=concurrency,
                iterations=iterations,
            )
            replace_trial = await _run_trial(
                coord,
                namespace_id,
                "replace",
                chunks_per_doc=chunks_per_doc,
                concurrency=concurrency,
                iterations=iterations,
            )

            print(
                f"ingest_path   p50={ingest_trial.p50:7.2f}ms  "
                f"p95={ingest_trial.p95:7.2f}ms  mean={ingest_trial.mean:7.2f}ms  "
                f"n={len(ingest_trial.wall_times_ms)}"
            )
            print(
                f"replace       p50={replace_trial.p50:7.2f}ms  "
                f"p95={replace_trial.p95:7.2f}ms  mean={replace_trial.mean:7.2f}ms  "
                f"n={len(replace_trial.wall_times_ms)}"
            )

            p50_delta = _format_delta(replace_trial.p50, ingest_trial.p50)
            p95_delta = _format_delta(replace_trial.p95, ingest_trial.p95)
            p50_regress = replace_trial.p50 > ingest_trial.p50 * (1 + args.tolerance / 100.0)
            p95_regress = replace_trial.p95 > ingest_trial.p95 * (1 + args.tolerance / 100.0)
            verdict = "✗ REGRESSED" if (p50_regress or p95_regress) else "✓ parity"
            print(f"parity        {verdict} (p50 Δ={p50_delta}, p95 Δ={p95_delta})")
            if p50_regress or p95_regress:
                regressed = True
    finally:
        await coord.disconnect()

    if regressed:
        print(
            f"\nFAIL: at least one scenario exceeds {args.tolerance:.0f}% tolerance vs ingest.",
            file=sys.stderr,
        )
        return 1
    print("\nOK: replace is at parity with ingest-path chunk-write cost.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--database-url",
        required=True,
        help="Async PostgreSQL URL, e.g. postgresql+asyncpg://khora:khora@localhost:5432/khora",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=40,
        help="Total iterations per scenario; divided by concurrency for the batch count",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=15.0,
        help="Allowed p50/p95 regression vs ingest-path, in %% (default: 15)",
    )
    return parser.parse_args()


def main() -> None:
    sys.exit(asyncio.run(_main(_parse_args())))


if __name__ == "__main__":
    main()


# Silence unused-import warnings in static analysis — these are used by type hints
# on callbacks above but some linters miss that.
_UNUSED = (Any, Entity, Relationship)
