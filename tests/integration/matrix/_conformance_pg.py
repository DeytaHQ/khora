"""Shared live-Postgres wiring for the recall-filter conformance leg.

The conformance harness (``khora.filter.conformance``) is storage-agnostic: it
compiles every case through the real ``compile_postgres`` and hands the predicate
to an injected ``PostgresRunner``. This module is that injected seam for the live
Postgres CI leg — it constructs a coordinator whose vector backend is the
skeleton ``khora_chunks`` temporal store (the production target ``compile_postgres``
emits column refs against), seeds through ``seed_case`` verbatim, and runs the
compiled ``WHERE`` against the seeded namespace.

Why the temporal store (not the factory's default vector backend): the Postgres
compiler targets ``khora_chunks`` with its denormalized document columns. The
default ``StorageFactory`` coordinator wires the legacy ``chunks`` table instead,
so seeded rows would never appear under the compiled predicate. Wiring the
``PgVectorTemporalStore`` as ``_vector`` makes ``coord.create_chunks_batch`` land
rows in ``khora_chunks`` — the exact table the predicate reads.

Seed/read split (write-once, read-many). The DB is seeded EXACTLY ONCE by the
out-of-band entrypoint :mod:`tests.integration.matrix._conformance_seed`, which
also writes a JSON ``seed map`` (``case_id -> {seed_id: chunk_uuid}``) to the path
in ``KHORA_CONFORMANCE_SEED_MAP``. ``seed_case`` assigns random chunk UUIDs, so
that map is the only bridge across the seed-step/pytest-step process boundary. The
test then runs READ-ONLY: it loads the map and runs each compiled ``WHERE`` against
the pre-seeded ``khora_chunks`` rows — no ``seed_case`` call, so xdist workers only
read (no write contention).

Kept out of ``conftest.py`` and named ``_conformance_pg`` (leading underscore, not
a ``test_`` module) so it is a plain helper shared by both the seed entrypoint
(:mod:`tests.integration.matrix._conformance_seed`) and the test module
(``test_filter_conformance.py``), never collected as tests itself.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from khora.config import KhoraConfig
from khora.core.models import Chunk
from khora.engines.skeleton.backends import TemporalChunk
from khora.engines.skeleton.backends.pgvector import (
    PgVectorTemporalStore,
    khora_chunks_table,
)
from khora.filter.conformance import (
    ConformanceCase,
    f_array_cases,
    f_coerce_cases,
    f_dates_cases,
    f_dotkey_cases,
    f_exists_cases,
    f_impossible_cases,
    f_logic_cases,
    f_nullval_cases,
    f_objeq_cases,
    f_op_cases,
    f_polarity_cases,
    f_sel_cases,
    f_sugar_cases,
    f_unsup_cases,
)
from khora.storage.backends.postgresql import PostgreSQLBackend
from khora.storage.coordinator import StorageCoordinator
from tests.integration._sqlite_lance_fixtures import fake_embedding

# Same default + normalization as the sibling skeleton/chronicle PG modules.
DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# matches the khora_chunks.embedding Vector(1536) column the temporal store hard-codes
EMBED_DIM = 1536

# Path of the JSON seed-map artifact the seed entrypoint writes and the test reads.
# Shared via env so the workflow's seed step and pytest step agree on one file.
SEED_MAP_PATH = os.environ.get("KHORA_CONFORMANCE_SEED_MAP", ".conformance_seed_map.json")


def _to_temporal_chunk(chunk: Chunk) -> TemporalChunk:
    """Adapt a core ``Chunk`` (what ``seed_case`` builds) to the ``TemporalChunk``
    the skeleton ``khora_chunks`` store writes.

    ``PgVectorTemporalStore.create_chunks_batch`` reads ``TemporalChunk``-only
    columns (``source_system``/``author``/``channel``/the denormalized document
    keys). ``seed_case`` sets only the fields both models share — ids, content,
    embedding, the three ``_DATE_KEYS`` (occurred_at/created_at/source_timestamp),
    and metadata — so copy those and let the skeleton-only columns default to
    ``None``. Cases that value-read a string document key (across every family) are
    pruned from the postgres ``backends`` for exactly this reason — those columns are
    not seeded here — so postgres only ever asserts metadata / occurred_at /
    source_timestamp cases, which this adapter carries faithfully.

    The embedding is regenerated at ``EMBED_DIM`` (1536): ``seed_case`` builds it at
    the sqlite_lance fixture's small dimension, but ``khora_chunks.embedding`` is a
    fixed ``Vector(1536)`` column. The value is irrelevant to filter conformance
    (the compiled predicate never touches the vector channel), only its dimension.
    """
    return TemporalChunk(
        id=chunk.id,
        namespace_id=chunk.namespace_id,
        document_id=chunk.document_id,
        content=chunk.content,
        embedding=fake_embedding(chunk.content, dim=EMBED_DIM),
        occurred_at=chunk.occurred_at,
        created_at=chunk.created_at,
        source_timestamp=chunk.source_timestamp,
        metadata=dict(chunk.metadata or {}),
        chunker_info=dict(chunk.chunker_info or {}),
    )


class _CoreChunkTemporalStore(PgVectorTemporalStore):
    """``PgVectorTemporalStore`` that accepts ``seed_case``'s core ``Chunk`` objects.

    The conformance seeder writes through the coordinator with core ``Chunk``
    instances, but the skeleton store's batch insert reads ``TemporalChunk``-only
    attributes. Convert at this boundary so the harness stays storage-agnostic and
    the production store is untouched.
    """

    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[TemporalChunk]:  # type: ignore[override]
        return await super().create_chunks_batch([_to_temporal_chunk(c) for c in chunks])


# The 14 family generators — the full conformance corpus. The seed entrypoint and
# the test module both pull from this through ``postgres_conformance_cases`` so they
# never drift on what the postgres leg seeds vs asserts.
_FAMILY_GENERATORS = (
    f_op_cases,
    f_coerce_cases,
    f_polarity_cases,
    f_array_cases,
    f_exists_cases,
    f_logic_cases,
    f_sugar_cases,
    f_dates_cases,
    f_nullval_cases,
    f_objeq_cases,
    f_dotkey_cases,
    f_sel_cases,
    f_unsup_cases,
    f_impossible_cases,
)


def postgres_conformance_cases() -> list[ConformanceCase]:
    """Every corpus case whose ``backends`` includes ``postgres``.

    The single source of truth for *what the postgres leg seeds and asserts* — the
    seed entrypoint and the test module both call this so they never drift. Pulls
    from all 14 families and keeps the postgres-targeting subset (a family prunes
    postgres only with a documented capability reason — e.g. the string document
    keys and ``created_at`` are not seeded onto the live chunk row).
    """
    cases: list[ConformanceCase] = []
    for generator in _FAMILY_GENERATORS:
        cases.extend(generator())
    return [c for c in cases if "postgres" in c.backends]


@asynccontextmanager
async def conformance_pg_coordinator() -> AsyncIterator[StorageCoordinator]:
    """Yield a connected coordinator whose vector backend is ``khora_chunks``.

    Relational (namespaces/documents) and the skeleton temporal vector store
    (``khora_chunks``) share a single engine so ``seed_case``'s
    ``create_namespace`` / ``create_document`` / ``create_chunks_batch`` writes
    all land in the same database the compiled ``WHERE`` later reads.
    """
    # The temporal store reads only its embedding dimension off the config; the
    # connection URL is irrelevant because we inject a shared engine below (the
    # store skips its own pool creation when given one).
    config = KhoraConfig(database_url=DATABASE_URL)
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM

    engine = create_async_engine(DATABASE_URL)
    relational = PostgreSQLBackend(DATABASE_URL, engine=engine)
    vector = _CoreChunkTemporalStore(config, engine=engine)
    coord = StorageCoordinator(relational=relational, vector=vector)
    await coord.connect()
    try:
        yield coord
    finally:
        await coord.disconnect()
        await engine.dispose()


# --------------------------------------------------------------------------- #
# Seed-map persistence (write-once by the entrypoint, read-many by the test).
# --------------------------------------------------------------------------- #


async def build_seed_map() -> dict[str, dict[str, str]]:
    """Seed every postgres case ONCE and return ``case_id -> {seed_id: chunk_uuid}``.

    Reuses ``seed_case`` verbatim against the skeleton ``khora_chunks`` coordinator.
    Chunk UUIDs are stringified for JSON. Called only by the one-time seed
    entrypoint — never by the test.
    """
    from khora.filter.conformance import seed_case

    seed_map: dict[str, dict[str, str]] = {}
    async with conformance_pg_coordinator() as coord:
        for case in postgres_conformance_cases():
            id_map = await seed_case(coord, case)
            seed_map[case.id] = {seed_id: str(chunk_id) for seed_id, chunk_id in id_map.items()}
    return seed_map


def write_seed_map(seed_map: Mapping[str, Mapping[str, str]]) -> None:
    """Write the seed map to ``SEED_MAP_PATH`` (the one-time artifact write)."""
    with open(SEED_MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump(seed_map, fh, sort_keys=True, indent=2)


@lru_cache(maxsize=1)
def load_seed_map() -> dict[str, dict[str, UUID]]:
    """Load the seed map written by the entrypoint; ``case_id -> {seed_id: chunk UUID}``.

    Cached so every xdist worker parses the JSON at most once. Chunk ids are parsed
    back to ``UUID`` for the ``id = ANY(:ids)`` bind. Raises a clear, actionable
    error if the map is absent — the postgres leg is read-only and depends on the
    one-time seed step (``python -m tests.integration.matrix._conformance_seed``)
    having run first; an opaque ``FileNotFoundError`` would obscure that contract.
    """
    if not os.path.exists(SEED_MAP_PATH):
        raise FileNotFoundError(
            f"conformance seed map not found at {SEED_MAP_PATH!r}; the postgres leg is "
            f"read-only and requires the one-time seed step to run first: "
            f"`python -m tests.integration.matrix._conformance_seed` "
            f"(set KHORA_CONFORMANCE_SEED_MAP to the same path for both steps)"
        )
    with open(SEED_MAP_PATH, encoding="utf-8") as fh:
        raw: dict[str, dict[str, str]] = json.load(fh)
    return {case_id: {seed_id: UUID(cid) for seed_id, cid in m.items()} for case_id, m in raw.items()}


# --------------------------------------------------------------------------- #
# Read-only predicate execution.
# --------------------------------------------------------------------------- #


async def run_predicate(
    id_map: Mapping[str, UUID],
    predicate: Any,
    records: Sequence[tuple[str, Mapping[str, Any]]],
) -> frozenset[str]:
    """Run a compiled ``khora_chunks`` ``WHERE`` against ONE case's pre-seeded rows.

    Read-only: scopes the query to exactly this case's chunk ids (``id = ANY(:ids)``,
    from the persisted seed map) so the per-case namespace never has to be re-derived
    and two cases never alias. Returns the ``SeedRecord`` ids whose chunk survived.
    ``records`` is part of the ``PostgresRunner`` contract but unused here — the live
    rows are the source of truth.
    """
    chunk_to_seed = {chunk_id: seed_id for seed_id, chunk_id in id_map.items()}
    case_chunk_ids = list(chunk_to_seed)
    stmt = sa.select(khora_chunks_table.c.id).where(khora_chunks_table.c.id.in_(case_chunk_ids)).where(predicate)
    engine = create_async_engine(DATABASE_URL, connect_args={"server_settings": {"TimeZone": "UTC"}})
    try:
        async with engine.connect() as conn:
            result = await conn.execute(stmt)
            survivors = [row[0] for row in result.fetchall()]
    finally:
        await engine.dispose()
    return frozenset(chunk_to_seed[cid] for cid in survivors if cid in chunk_to_seed)
