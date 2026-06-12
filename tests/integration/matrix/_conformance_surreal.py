"""Shared embedded-SurrealDB wiring for the recall-filter conformance leg.

The conformance harness (``khora.filter.conformance``) is storage-agnostic: it
lowers every case through the real validator + ``parse_to_ast`` and runs the result
through an injected runner. This module is that injected seam for the SurrealDB leg.

It exposes the three callables ``test_filter_conformance.py`` injects (the same seam
shape as the live-Postgres leg):

* ``reachable() -> bool`` — the local-dev skip gate. SurrealDB embedded ``memory://``
  is in-process, so the store is always available — returns ``True``.
* ``load_seed_map() -> dict[str, dict[str, UUID]]`` — seeds every surrealdb-targeted
  case ONCE into a process-wide embedded store and returns ``case_id -> {seed_id:
  chunk UUID}``. Embedded ``memory://`` is per-PROCESS, so (unlike the docker legs)
  there is NO shared artifact file — the seed lives in the same pytest process that
  later runs ``run_live``, kept alive in a module-level singleton. Cached, so under
  ``-n auto`` each xdist worker seeds its own in-process store exactly once.
* ``run_live(id_map, compiled, filter_ast, post_filter, records)`` — the
  ``LiveRunner``-shaped executor. SurrealDB is TOTAL-exact: the
  :class:`~khora.filter.conformance.SurrealExecutor` already compiled ``compiled``
  with the real ``compile_surrealdb`` (``on_unsupported="raise"``), so the compiled
  ``WHERE`` alone decides the row-set — ``post_filter`` is ignored (applying it is
  oracle-equivalent). Runs the predicate server-side scoped to the case's chunk ids.

``executor_for(case)`` is also exposed: it seeds the case in-process and returns a
ready :class:`SurrealExecutor` bound to a runner closed over the seeded ``id_map`` —
the one-call seed-and-wire entry point.

Why a trimmed ``temporal_chunk`` schema (not the skeleton store's full SCHEMAFULL
DDL): the production schema declares ``namespace`` / ``document`` as required
``record<...>`` links, but filter conformance scopes by chunk id and reads only the
date system columns, the seven string document keys, and the FLEXIBLE ``metadata_``
object. Seeding the trimmed table (mirroring
``tests/integration/filter/test_compile_surrealdb_embedded.py``) writes EXACTLY the
surface the compiler reads — dates as real ``option<datetime>`` values (so the
compiler's native datetime compares apply), the doc-keys as ``option<string>``, and
``metadata_`` as the FLEXIBLE blob — without record-link FKs the corpus never uses.

Kept out of ``conftest.py`` and named ``_conformance_surreal`` (leading underscore,
not a ``test_`` module) so it is a plain helper the test module imports, never
collected as tests itself.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

from khora.filter import CompiledFilter
from khora.filter.ast import FilterNode
from khora.filter.conformance import ConformanceCase, SeedRecord, SurrealExecutor
from khora.storage.backends.surrealdb._helpers import _rid
from khora.storage.backends.surrealdb.connection import SurrealDBConnection

# The three date system keys + the seven denormalized document string keys the
# conformance corpus stamps on a chunk.
_DATE_KEYS: tuple[str, ...] = ("occurred_at", "created_at", "source_timestamp")
_STRING_KEYS: tuple[str, ...] = (
    "source_type",
    "source_name",
    "source_url",
    "external_id",
    "content_type",
    "source",
    "title",
)

# A trimmed ``temporal_chunk`` shape — only the columns the compiled predicate reads
# for the conformance corpus: the three date columns as real ``option<datetime>``
# (native datetime compares apply), the seven string document keys as
# ``option<string>`` (the corpus tags some string-key cases for surrealdb), and the
# FLEXIBLE ``metadata_`` blob. No namespace/document record links or HNSW index,
# which the id-scoped filter conformance never touches.
_SCHEMA = """
DEFINE TABLE IF NOT EXISTS temporal_chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS content ON temporal_chunk TYPE string;
DEFINE FIELD IF NOT EXISTS occurred_at ON temporal_chunk TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS created_at ON temporal_chunk TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS source_timestamp ON temporal_chunk TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS source_type ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS source_name ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS source_url ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS external_id ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content_type ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS source ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS title ON temporal_chunk TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata_ ON temporal_chunk FLEXIBLE TYPE option<object>;
"""

# The 14 corpus family generators, by name (resolved off ``khora.filter.conformance``
# at call time). Mirrors the postgres leg so the surrealdb leg seeds/asserts the
# same corpus, filtered to the surrealdb-targeting subset.
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
    """Whether the SurrealDB store is reachable — always ``True`` (embedded in-process)."""
    return True


def surreal_conformance_cases() -> list[ConformanceCase]:
    """Every corpus case whose ``backends`` includes ``surrealdb`` (all 14 families)."""
    from khora.filter import conformance

    cases: list[ConformanceCase] = []
    for generator in _FAMILY_GENERATORS:
        cases.extend(getattr(conformance, generator)())
    return [c for c in cases if "surrealdb" in c.backends]


def _seed_row(record: SeedRecord, chunk_id: UUID) -> dict[str, Any]:
    """Build one ``temporal_chunk`` row carrying EXACTLY the compiler-read surface.

    Every date key is set explicitly (``None`` when absent), mirroring the production
    store's ``create_chunks_batch`` which always writes ``occurred_at=chunk.occurred_at``
    into the ``option<datetime>`` column. The seven string document keys are set
    verbatim (``None`` when absent for the six nullable keys; ``source_type`` is
    non-null and always carries its value — ``"library"`` by default). ``metadata_``
    carries the chunk's metadata blob (the remapped ``metadata`` root).
    """
    row: dict[str, Any] = {
        "id": _rid("temporal_chunk", chunk_id),
        "content": record.content,
        "metadata_": dict(record.metadata or {}),
    }
    for key in _DATE_KEYS:
        row[key] = getattr(record, key)
    for key in _STRING_KEYS:
        row[key] = getattr(record, key)
    return row


# --------------------------------------------------------------------------- #
# Dedicated event loop owning the seeded store.
# --------------------------------------------------------------------------- #
#
# ``memory://`` is per-PROCESS, and the embedded SurrealDB connection is bound to the
# event loop it was opened on. So a single long-lived background thread runs ONE
# event loop that owns the seeded connection; every async call (the one-time seed AND
# each ``run_live`` query) is submitted to that same loop via
# ``run_coroutine_threadsafe``, and the sync seam blocks on the returned future. A
# per-call ``asyncio.run`` would open a fresh loop the connection cannot be used from.


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
    """The process-wide loop thread that owns the seeded connection."""
    return _LoopThread()


def _run_async(coro: Any) -> Any:
    """Run ``coro`` on the dedicated loop that owns the embedded connection."""
    return _loop_thread().run(coro)


class _SeededStore:
    """A connected embedded SurrealDB connection seeded with every surrealdb case."""

    def __init__(self, connection: SurrealDBConnection, seed_map: dict[str, dict[str, UUID]]) -> None:
        self.connection = connection
        self.seed_map = seed_map


async def _build_seeded_store() -> _SeededStore:
    """Open one embedded connection, define the schema, seed every surrealdb case."""
    connection = SurrealDBConnection(mode="memory")
    await connection.connect()
    await connection.execute(_SCHEMA)

    seed_map: dict[str, dict[str, UUID]] = {}
    rows: list[dict[str, Any]] = []
    for case in surreal_conformance_cases():
        id_map: dict[str, UUID] = {}
        for record in case.seed_records:
            chunk_id = uuid4()
            id_map[record.id] = chunk_id
            rows.append(_seed_row(record, chunk_id))
        seed_map[case.id] = id_map
    if rows:
        await connection.execute("INSERT INTO temporal_chunk $records", {"records": rows})
    return _SeededStore(connection, seed_map)


@lru_cache(maxsize=1)
def _seeded_store() -> _SeededStore:
    """The process-wide seeded embedded store (built + seeded exactly once)."""
    return _run_async(_build_seeded_store())


def load_seed_map() -> dict[str, dict[str, UUID]]:
    """Seed every surrealdb case once (in-process) and return ``case_id -> {seed_id: chunk UUID}``.

    Embedded ``memory://`` is per-process, so there is no artifact file: the seed
    lives in this worker's process via the cached :func:`_seeded_store` singleton,
    which ``run_live`` reuses.
    """
    return _seeded_store().seed_map


async def run_live(
    id_map: Mapping[str, UUID],
    compiled: CompiledFilter[str],
    filter_ast: FilterNode,
    post_filter: Callable[[Mapping[str, Any]], bool],
    records: Sequence[tuple[str, Mapping[str, Any]]],
) -> frozenset[str]:
    """Run the compiled SurrealQL ``WHERE`` against ONE case's seeded rows (total-exact).

    The :class:`~khora.filter.conformance.SurrealExecutor` already compiled ``compiled``
    with the real ``compile_surrealdb`` (``on_unsupported="raise"``). This runner only
    executes: it runs ``compiled.predicate`` server-side scoped to exactly this case's
    chunk ids (``id IN $case_ids``, from ``id_map``) so two cases never alias. SurrealDB
    is total, so the surviving rows ARE the answer — ``post_filter`` and ``records`` are
    part of the shared ``LiveRunner`` contract but unused here (the live rows are the
    source of truth). Returns the surviving ``SeedRecord`` ids.
    """
    connection = _seeded_store().connection
    chunk_to_seed = {chunk_id: seed_id for seed_id, chunk_id in id_map.items()}

    bindings: dict[str, Any] = dict(compiled.params)
    # ``case_ids`` cannot collide with a compiled bind: ``compile_surrealdb`` names
    # every bind ``{param_namespace}_{n}`` (default namespace ``f`` → ``f_0`` ...).
    bindings["case_ids"] = [_rid("temporal_chunk", cid) for cid in chunk_to_seed]
    sql = f"SELECT id FROM temporal_chunk WHERE id IN $case_ids AND ({compiled.predicate})"  # noqa: S608 - predicate is compiler-emitted, values bind
    rows = await connection.query(sql, bindings)

    survivors: set[str] = set()
    for row in rows:
        chunk_id = UUID(str(row["id"].id))
        seed_id = chunk_to_seed.get(chunk_id)
        if seed_id is not None:
            survivors.add(seed_id)
    return frozenset(survivors)


def executor_for(case: ConformanceCase) -> SurrealExecutor:
    """Seed ``case`` (in-process) and return a ready :class:`SurrealExecutor`.

    The one-call seed-and-wire entry point: looks the case up in the process-wide
    seeded store's map and closes a sync ``LiveRunner`` (over that case's
    ``id_map``) that bridges to the async :func:`run_live` on a worker-thread loop —
    exactly mirroring ``_conformance_pg``'s ``_postgres_executor_for``. The
    ``SurrealExecutor`` invokes the REAL ``compile_surrealdb`` (this is what
    conformance checks); the runner only executes.
    """
    id_map = load_seed_map()[case.id]

    def runner(compiled, filter_ast, post_filter, records):  # noqa: ANN001, ANN202 - matches LiveRunner
        return _run_async(run_live(id_map, compiled, filter_ast, post_filter, records))

    return SurrealExecutor(runner)
