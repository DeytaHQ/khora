"""Shared embedded-sqlite_lance wiring for the recall-filter conformance leg.

The conformance harness (``khora.filter.conformance``) is storage-agnostic: it
lowers every case through the real validator + ``parse_to_ast`` and runs the result
through an injected runner. This module is that injected seam for the sqlite_lance
leg.

It exposes the three callables ``test_filter_conformance.py`` injects (the same seam
shape as the live-Postgres leg):

* ``reachable() -> bool`` — the local-dev skip gate. The embedded SQLite + LanceDB
  stack is in-process, so it is always available — returns ``True``.
* ``load_seed_map() -> dict[str, dict[str, UUID]]`` — seeds every sqlite_lance-targeted
  case ONCE into a process-wide coordinator (a tmp SQLite + LanceDB pair) and returns
  ``case_id -> {seed_id: chunk UUID}``. The embedded stack is per-PROCESS, so (unlike
  the docker legs) there is NO shared artifact file — the seed lives in the same
  pytest process that later runs ``run_live``, kept alive in a module-level singleton.
  Cached, so under ``-n auto`` each xdist worker seeds its own store exactly once.
* ``run_live(id_map, compiled, filter_ast, post_filter, records)`` — the
  ``LiveRunner``-shaped executor. sqlite_lance is a *split* pushdown: the
  :class:`~khora.filter.conformance.LanceExecutor` already compiled ``compiled`` with
  the real ``compile_lance`` and built ``post_filter``. This runner runs the SQLite
  fragment (the prefilter) scoped to the case's chunk ids, then MUST apply
  ``post_filter`` to the candidate rows — the prefilter defers metadata-without-JSON1
  / ``$date`` / bare-blob / dict-or-null ``$in`` leaves, so the SQL row-set is a
  superset the oracle narrows.

``executor_for(case)`` is also exposed: it seeds the case in-process and returns a
ready :class:`LanceExecutor` bound to a runner closed over the seeded ``id_map``.

Why the skeleton temporal store (not the factory's unified adapter): the lance
compiler targets ``khora_chunks`` with its denormalized document columns. The unified
``SQLiteLanceVectorAdapter`` writes the legacy ``chunks`` table instead, so seeded
rows would never appear under the compiled predicate. Wiring the skeleton
``SQLiteLanceTemporalStore`` as ``_vector`` makes ``coord.create_chunks_batch`` land
rows in ``khora_chunks`` — the exact table the predicate reads. The store's
``connect()`` issues its own ``khora_chunks`` DDL (the Alembic chain only creates the
legacy ``chunks`` table on SQLite), so the table exists before the first seed.

Kept out of ``conftest.py`` and named ``_conformance_lance`` (leading underscore, not
a ``test_`` module) so it is a plain helper the test module imports, never collected
as tests itself.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

from khora.core.models import Chunk
from khora.db.session import run_migrations
from khora.filter import CompiledFilter
from khora.filter.ast import FilterNode
from khora.filter.conformance import _DOC_STRING_KEYS, ConformanceCase, LanceExecutor, seed_case
from khora.storage.backends.sqlite_lance import SQLiteLanceRelationalAdapter
from khora.storage.backends.sqlite_lance._helpers import uuid_to_text
from khora.storage.backends.sqlite_lance.connection import (
    EmbeddedStorageHandle,
    EmbeddedStorageHandleConfig,
)
from khora.storage.coordinator import StorageCoordinator
from khora.storage.temporal import TemporalChunk
from khora.storage.temporal.sqlite_lance import SQLiteLanceTemporalStore
from tests.integration._sqlite_lance_fixtures import EMBED_DIM

# The 14 corpus family generators, by name (resolved off ``khora.filter.conformance``
# at call time). Mirrors the postgres leg so the sqlite_lance leg seeds/asserts the
# same corpus, filtered to the sqlite_lance-targeting subset.
_FAMILY_GENERATORS: tuple[str, ...] = (
    "f_op_cases",
    "f_coerce_cases",
    "f_polarity_cases",
    "f_array_cases",
    "f_exists_cases",
    "f_logic_cases",
    "f_sugar_cases",
    "f_dates_cases",
    "f_nullval_cases",
    "f_objeq_cases",
    "f_dotkey_cases",
    "f_sel_cases",
    "f_unsup_cases",
    "f_impossible_cases",
)


def reachable() -> bool:
    """Whether the sqlite_lance store is reachable — always ``True`` (embedded in-process)."""
    return True


def lance_conformance_cases() -> list[ConformanceCase]:
    """Every corpus case whose ``backends`` includes ``sqlite_lance`` (all 14 families)."""
    from khora.filter import conformance

    cases: list[ConformanceCase] = []
    for generator in _FAMILY_GENERATORS:
        cases.extend(getattr(conformance, generator)())
    return [c for c in cases if "sqlite_lance" in c.backends]


def _to_temporal_chunk(chunk: Chunk, doc_keys: Mapping[str, Any]) -> TemporalChunk:
    """Adapt a core ``Chunk`` (what ``seed_case`` builds) to the ``TemporalChunk``
    the skeleton ``khora_chunks`` store writes, stamping the parent document's
    denormalized string keys.

    ``SQLiteLanceTemporalStore.create_chunks_batch`` reads ``TemporalChunk``-only
    columns (``source_system`` / ``author`` / ``channel`` / the denormalized document
    keys). ``seed_case`` sets only the fields both models share — ids, content, the
    three date keys (occurred_at / created_at / source_timestamp), and metadata — so
    copy those. The seven string document keys
    (``source_type`` / ``source_name`` / ``source_url`` / ``external_id`` /
    ``content_type`` / ``source`` / ``title``) live on the parent ``Document``, not
    the core ``Chunk``; ``doc_keys`` carries them off that document so the chunk row
    is queryable on them, mirroring the production denormalization the
    ``SQLiteLanceTemporalStore`` write path performs (the same move the postgres twin
    ``_conformance_pg.py`` makes). The remaining skeleton-only columns default to
    ``None``.

    ``embedding`` is dropped (``None``): filter conformance never touches the vector
    channel, and a ``None`` embedding makes the store skip the LanceDB write entirely
    — the seed only needs the ``khora_chunks`` SQLite row the predicate reads.
    """
    return TemporalChunk(
        id=chunk.id,
        namespace_id=chunk.namespace_id,
        document_id=chunk.document_id,
        content=chunk.content,
        embedding=None,
        occurred_at=chunk.occurred_at,
        created_at=chunk.created_at,
        source_timestamp=chunk.source_timestamp,
        metadata=dict(chunk.metadata or {}),
        chunker_info=dict(chunk.chunker_info or {}),
        **doc_keys,
    )


class _CoreChunkTemporalStore(SQLiteLanceTemporalStore):
    """``SQLiteLanceTemporalStore`` that accepts ``seed_case``'s core ``Chunk`` objects.

    The conformance seeder writes through the coordinator with core ``Chunk``
    instances, but the skeleton store's batch insert reads ``TemporalChunk``-only
    attributes. Convert at this boundary so the harness stays storage-agnostic and the
    production store is untouched. The parent document's seven string keys are
    denormalized onto each chunk (one batched SELECT on the shared SQLite file), so the
    sqlite_lance leg carries them on the queryable row exactly as production does —
    mirroring the postgres twin's ``_CoreChunkTemporalStore`` in ``_conformance_pg.py``.
    """

    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[TemporalChunk]:  # type: ignore[override]
        doc_keys_by_id = await self._fetch_document_keys({c.document_id for c in chunks})
        temporal = [_to_temporal_chunk(c, doc_keys_by_id.get(c.document_id, {})) for c in chunks]
        return await super().create_chunks_batch(temporal)

    async def _fetch_document_keys(self, document_ids: set[UUID]) -> dict[UUID, dict[str, Any]]:
        """Read the seven string keys off each parent document (one batched query).

        Mirrors production denormalization: the keys live on ``documents`` and are
        copied onto the chunk row. The documents were written by ``seed_case`` through
        the relational adapter's SQLAlchemy engine immediately before — onto the same
        physical SQLite file this raw-aiosqlite handle reads, in the shared
        hex-no-dashes UUID encoding (``uuid_to_text``) — so the rows are present and
        the ``id IN (...)`` bind matches. This is the sqlite_lance counterpart to the
        postgres twin's ``_fetch_document_keys`` in ``_conformance_pg.py``.
        """
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        columns = ", ".join(("id", *_DOC_STRING_KEYS))
        sql = f"SELECT {columns} FROM documents WHERE id IN ({placeholders})"  # noqa: S608 - ids bind positionally
        ids = [uuid_to_text(did) for did in document_ids]
        cur = await self._sqlite.execute(sql, ids)
        rows = await cur.fetchall()
        return {UUID(row["id"]): {key: row[key] for key in _DOC_STRING_KEYS} for row in rows}


# --------------------------------------------------------------------------- #
# Dedicated event loop owning the seeded coordinator.
# --------------------------------------------------------------------------- #
#
# The SQLite + LanceDB stack is per-PROCESS, so the seed must live in the same
# process that runs the queries. Crucially, an aiosqlite connection is bound to the
# event loop it was opened on — querying it from a different loop deadlocks. So a
# single long-lived background thread runs ONE event loop that owns the seeded
# coordinator; every async call (the one-time seed AND each ``run_live`` query) is
# submitted to that same loop via ``run_coroutine_threadsafe``. The sync seam
# (``load_seed_map`` / the ``run_live`` bridge) blocks on the returned future. This
# replaces a per-call ``asyncio.run`` (which would open a fresh loop the connection
# cannot be used from).


class _LoopThread:
    """A daemon thread running one asyncio loop; submit coroutines, block for results."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


@lru_cache(maxsize=1)
def _loop_thread() -> _LoopThread:
    """The process-wide loop thread that owns the seeded coordinator."""
    return _LoopThread()


def _run_async(coro: Any) -> Any:
    """Run ``coro`` on the dedicated loop that owns the embedded coordinator."""
    return _loop_thread().run(coro)


class _SeededCoordinator:
    """A connected embedded coordinator seeded with every sqlite_lance case."""

    def __init__(
        self,
        coord: StorageCoordinator,
        handle: EmbeddedStorageHandle,
        seed_map: dict[str, dict[str, UUID]],
    ) -> None:
        self.coord = coord
        self.handle = handle
        self.seed_map = seed_map


async def _build_seeded_coordinator() -> _SeededCoordinator:
    """Migrate a tmp SQLite file, wire the skeleton ``khora_chunks`` store, seed all cases."""
    tmp_path = Path(tempfile.mkdtemp(prefix="khora-conformance-lance-"))
    db_path = str(tmp_path / "khora.db")
    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    if not result.success:
        raise RuntimeError(f"migration failed: {result.error}")

    handle = EmbeddedStorageHandle(
        EmbeddedStorageHandleConfig(
            db_path=db_path,
            lance_path=str(tmp_path / "khora.lance"),
            embedding_dimension=EMBED_DIM,
        )
    )
    await handle.connect()
    coord = StorageCoordinator(
        relational=SQLiteLanceRelationalAdapter(handle),
        vector=_CoreChunkTemporalStore(handle),
    )
    await coord.connect()

    seed_map: dict[str, dict[str, UUID]] = {}
    for case in lance_conformance_cases():
        seed_map[case.id] = await seed_case(coord, case)
    return _SeededCoordinator(coord, handle, seed_map)


@lru_cache(maxsize=1)
def _seeded_coordinator() -> _SeededCoordinator:
    """The process-wide seeded embedded coordinator (built + seeded exactly once)."""
    return _run_async(_build_seeded_coordinator())


def load_seed_map() -> dict[str, dict[str, UUID]]:
    """Seed every sqlite_lance case once (in-process) and return ``case_id -> {seed_id: chunk UUID}``.

    The embedded stack is per-process, so there is no artifact file: the seed lives in
    this worker's process via the cached :func:`_seeded_coordinator` singleton, which
    ``run_live`` reuses.
    """
    return _seeded_coordinator().seed_map


async def run_live(
    id_map: Mapping[str, UUID],
    compiled: CompiledFilter[str],
    filter_ast: FilterNode,
    post_filter: Callable[[Mapping[str, Any]], bool],
    records: Sequence[tuple[str, Mapping[str, Any]]],
) -> frozenset[str]:
    """Run the compiled lance ``WHERE`` against ONE case's seeded ``khora_chunks`` rows.

    The :class:`~khora.filter.conformance.LanceExecutor` already compiled ``compiled``
    with the real ``compile_lance`` (split mode, JSON1 advertised) and built
    ``post_filter`` (the ``compile_python`` full-AST oracle). This runner only
    executes: it runs the SQLite fragment scoped to exactly this case's chunk ids
    (``id IN (...)``, from ``id_map``) so two cases never alias, then applies
    ``post_filter`` over the FULL AST to the per-row record mappings — MANDATORY,
    because the lance compiler is a superset-safe *split* pushdown (it defers
    metadata-without-JSON1 / ``$date`` / bare-blob / dict-or-null ``$in`` leaves), so
    the pushed SQL row-set is a superset the oracle narrows.

    ``records`` carries the ``(seed_id, mapping)`` pairs ``post_filter`` reads; they
    are keyed back to the surviving live rows by chunk id. Returns the ``SeedRecord``
    ids whose chunk survived both the pushdown and the post-filter.
    """
    record_map = dict(records)
    chunk_to_seed = {chunk_id: seed_id for seed_id, chunk_id in id_map.items()}

    placeholders = ",".join("?" for _ in chunk_to_seed)
    sql = f"SELECT id FROM khora_chunks WHERE id IN ({placeholders})"  # noqa: S608 - ids bind positionally
    args: list[Any] = [uuid_to_text(cid) for cid in chunk_to_seed]
    # ``compile_lance`` emits the literal ``"1"`` for a match-everything filter — skip
    # AND-ing it so the scoped read returns every seeded row for the post-filter pass.
    if compiled.predicate and compiled.predicate != "1":
        sql += f" AND ({compiled.predicate})"
        args.extend(compiled.params["args"])

    handle = _seeded_coordinator().handle
    cur = await handle.sqlite.execute(sql, args)
    rows = await cur.fetchall()

    survivors: set[str] = set()
    for row in rows:
        chunk_id = UUID(row[0])
        seed_id = chunk_to_seed.get(chunk_id)
        if seed_id is None:
            continue
        if post_filter(record_map[seed_id]):
            survivors.add(seed_id)
    return frozenset(survivors)


def executor_for(case: ConformanceCase) -> LanceExecutor:
    """Seed ``case`` (in-process) and return a ready :class:`LanceExecutor`.

    The one-call seed-and-wire entry point: looks the case up in the process-wide
    seeded coordinator's map and closes a sync ``LiveRunner`` (over that case's
    ``id_map``) that bridges to the async :func:`run_live` on a worker-thread loop —
    mirroring ``_conformance_pg``'s ``_postgres_executor_for``. The ``LanceExecutor``
    invokes the REAL ``compile_lance`` (this is what conformance checks); the runner
    only executes.
    """
    id_map = load_seed_map()[case.id]

    def runner(compiled, filter_ast, post_filter, records):  # noqa: ANN001, ANN202 - matches LiveRunner
        return _run_async(run_live(id_map, compiled, filter_ast, post_filter, records))

    return LanceExecutor(runner)
