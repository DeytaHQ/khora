"""Live-Postgres wiring for the documents-target conformance leg.

The store is :class:`~khora.storage.backends.postgresql.PostgreSQLBackend` — the only
documents tier whose ``created_at`` / ``source_timestamp`` are real ``timestamptz``
columns, so it is the only one whose ``_documents_compile_context`` declares all NINE
enumerable system keys pushable. Every other leg withholds the two date keys because
their stored text does not order against a datetime bind. That asymmetry is the reason
this leg is worth a live server: the date-key F-OP cases are the ones that reach SQL
here and reach the post-filter everywhere else.

**Seed/read split (write-once, read-many).** Unlike the three embedded legs, the
Postgres store outlives the process that seeds it, so the seed runs out-of-band exactly
once via :mod:`tests.integration.matrix._conformance_seed` and the pytest step is
strictly READ-ONLY — under ``-n auto`` every xdist worker only reads.

The map that bridges the two processes is narrower than the chunk leg's. ``seed_case``
assigns random chunk UUIDs, so its map is the only way to find those rows again; here
the *namespace* is what the reader needs, and ``seed_documents_case`` pins it
deterministically to ``_documents_case_namespace_id(case)``. The persisted map is
therefore needed only to translate surviving ``Document.id`` values back to
:class:`~khora.filter.conformance.SeedRecord` ids.

Kept out of ``conftest.py`` and named with a leading underscore (not a ``test_``
module) so it is a plain helper shared by the seed entrypoint and the test module,
never collected as tests itself.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from functools import lru_cache
from uuid import UUID

from sqlalchemy.ext.asyncio import create_async_engine

from khora.filter.conformance import (
    ConformanceCase,
    DocumentsExecutor,
    _documents_case_namespace_id,
    documents_conformance_cases,
    seed_documents_case,
)
from khora.storage.backends.postgresql import PostgreSQLBackend
from khora.storage.coordinator import StorageCoordinator
from tests.integration.matrix._conformance_docs_common import documents_executor, run_async

# Same default + normalization as the sibling conformance / skeleton PG modules.
DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# Path of the JSON seed-map artifact the seed entrypoint writes and the test reads.
# Distinct from the chunk leg's ``KHORA_CONFORMANCE_SEED_MAP`` — the two corpora seed
# different tables and are written by separate invocations of the seed entrypoint.
SEED_MAP_PATH = os.environ.get("KHORA_CONFORMANCE_DOCS_SEED_MAP", ".conformance_docs_seed_map.json")


def reachable() -> bool:
    """Whether the Postgres server accepts a TCP connection (the local-dev skip gate)."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@asynccontextmanager
async def conformance_docs_pg_coordinator() -> AsyncIterator[StorageCoordinator]:
    """Yield a connected relational-only coordinator over the live Postgres database.

    No vector backend: document enumeration reads ``documents`` alone, so — unlike the
    chunk leg, which has to wire the skeleton ``khora_chunks`` temporal store to land
    rows in the table its predicate reads — the plain relational backend is the
    production target here.
    """
    engine = create_async_engine(DATABASE_URL)
    coord = StorageCoordinator(relational=PostgreSQLBackend(DATABASE_URL, engine=engine))
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
    """Seed every documents case ONCE and return ``case_id -> {seed_id: document_uuid}``.

    Reuses ``seed_documents_case`` verbatim, so the rows the test reads are
    byte-for-byte the rows the harness's own seeder writes. Called only by the one-time
    seed entrypoint — never by the test. Re-running against an already-seeded database
    fails on the namespace primary key rather than duplicating the corpus, because
    ``seed_documents_case`` pins that id deterministically.
    """
    seed_map: dict[str, dict[str, str]] = {}
    async with conformance_docs_pg_coordinator() as coord:
        for case in documents_conformance_cases():
            id_map = await seed_documents_case(coord, case)
            seed_map[case.id] = {seed_id: str(doc_id) for seed_id, doc_id in id_map.items()}
    return seed_map


def write_seed_map(seed_map: Mapping[str, Mapping[str, str]]) -> None:
    """Write the seed map to ``SEED_MAP_PATH`` (the one-time artifact write)."""
    with open(SEED_MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump(seed_map, fh, sort_keys=True, indent=2)


@lru_cache(maxsize=1)
def load_seed_map() -> dict[str, dict[str, UUID]]:
    """Load the seed map written by the entrypoint; ``case_id -> {seed_id: document UUID}``.

    Cached so every xdist worker parses the JSON at most once. Raises a clear,
    actionable error if the map is absent — this leg is read-only and depends on the
    one-time seed step having run first; an opaque ``FileNotFoundError`` would obscure
    that contract.
    """
    if not os.path.exists(SEED_MAP_PATH):
        raise FileNotFoundError(
            f"documents conformance seed map not found at {SEED_MAP_PATH!r}; the postgres "
            f"documents leg is read-only and requires the one-time seed step to run first: "
            f"`python -m tests.integration.matrix._conformance_seed documents-postgres` "
            f"(set KHORA_CONFORMANCE_DOCS_SEED_MAP to the same path for both steps)"
        )
    with open(SEED_MAP_PATH, encoding="utf-8") as fh:
        raw: dict[str, dict[str, str]] = json.load(fh)
    return {case_id: {seed_id: UUID(doc_id) for seed_id, doc_id in m.items()} for case_id, m in raw.items()}


# --------------------------------------------------------------------------- #
# Read-only executor over the pre-seeded store.
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _coordinator() -> StorageCoordinator:
    """A process-wide connected coordinator on the shared documents loop thread.

    The coordinator is opened on the same loop every walk is submitted to (see
    ``_conformance_docs_common``), so the asyncpg pool is never used across loops. It
    is READ-ONLY: this module's ``executor_for`` never seeds.
    """

    async def build() -> StorageCoordinator:
        engine = create_async_engine(DATABASE_URL, connect_args={"server_settings": {"TimeZone": "UTC"}})
        coord = StorageCoordinator(relational=PostgreSQLBackend(DATABASE_URL, engine=engine))
        await coord.connect()
        return coord

    return run_async(build())


def executor_for(case: ConformanceCase, *, forced_residual: bool) -> DocumentsExecutor:
    """Wire a READ-ONLY :class:`DocumentsExecutor` over ``case``'s pre-seeded rows.

    Looks the case up in the persisted seed map (loaded once, cached) and binds the
    walk to the case's deterministic namespace. No seeding — the rows already exist.
    """
    import pytest

    seed_map = load_seed_map()
    if case.id not in seed_map:
        pytest.fail(
            f"case {case.id!r} missing from the documents seed map "
            f"(re-run `python -m tests.integration.matrix._conformance_seed documents-postgres`)"
        )
    return documents_executor(
        _coordinator(),
        _documents_case_namespace_id(case),
        seed_map[case.id],
        forced_residual=forced_residual,
    )
