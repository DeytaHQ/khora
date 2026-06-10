"""Shared live-Neo4j wiring for the recall-filter conformance leg.

The conformance harness (``khora.filter.conformance``) is storage-agnostic: it
lowers every case through the real validator + ``parse_to_ast`` and runs the
result through an injected runner. This module is that injected seam for the
live-Neo4j (cypher) leg — it seeds ``Chunk`` nodes carrying the system-key
properties the compiler reads, runs the compiled Cypher ``WHERE`` server-side
scoped to the case's chunk ids, and then applies the ``compile_python`` post-filter.

Why the post-filter: ``compile_cypher`` is a *split* pushdown. It pushes the
system-key predicates (``c.occurred_at`` ...) into Cypher but defers metadata (a
serialized JSON property on the chunk node, not pushable to Cypher) to the engine's
in-memory post-filter — exactly as the VectorCypher retriever does. So the Cypher
candidate set is a superset the ``compile_python`` oracle narrows.

Why the production write path: the seed goes through
:meth:`DualNodeManager.create_chunk_nodes_batch` — the exact code VectorCypher runs
to land ``Chunk`` nodes — so the conformance leg exercises the real property mapping,
including the production ``serialize_dict`` form of ``metadata`` landing on the node,
not a hand-rolled approximation of it. The compiled Cypher predicate then reads only
the pushed-down system-key properties (per the split-pushdown note above); ``metadata``
is never in the WHERE — it stays in the in-memory post-filter — so the production write
form is what's under test on the write side. Each :class:`SeedRecord` is adapted to a
:class:`TemporalChunk` (mirroring how ``_conformance_pg`` adapts to the skeleton
store for the postgres leg) so both legs feed their production writer the same corpus.

One corpus-fidelity adjustment: ``create_chunk_nodes_batch`` stamps an absent
``created_at`` to ``now()`` (the production default), but the conformance corpus
keeps ``created_at`` cases on the cypher leg *because* cypher is expected to leave an
absent ``created_at`` NULL (an absent row that gets a ``now()`` value would satisfy a
lower-bound pushdown and break the by-construction ``expected_ids``). So after the
batch write, the seeder ``REMOVE``s ``created_at`` from exactly the nodes whose
``SeedRecord`` left it ``None``, restoring the missing-property semantics the compiler
relies on. ``occurred_at`` / ``source_timestamp`` are user-supplied and the production
writer already leaves them absent when ``None``, so they need no fixup.

Seed/read split (write-once, read-many), mirroring the live-Postgres leg. The graph
is seeded ONCE by ``_conformance_seed`` (the workflow's one-time step), which also
persists a JSON ``seed map`` (``case_id -> {seed_id: chunk_uuid}``) to the path in
``KHORA_CONFORMANCE_NEO4J_SEED_MAP``. The pytest step then runs READ-ONLY: it loads
the map and runs each compiled ``WHERE`` against the pre-seeded nodes — no seeding,
so under ``-n auto`` every xdist worker only reads (no write contention against the
shared Neo4j container).

Kept out of ``conftest.py`` and named ``_conformance_neo4j`` (leading underscore,
not a ``test_`` module) so it is a plain helper shared by both the seed entrypoint
and the test module, never collected as tests itself.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

from khora.engines.skeleton.backends import TemporalChunk
from khora.engines.vectorcypher.dual_nodes import DualNodeManager
from khora.filter import CompiledFilter
from khora.filter.ast import FilterNode
from khora.filter.conformance import ConformanceCase, CypherExecutor, SeedRecord

# The node alias the ``CypherExecutor`` compiles against (``Chunk`` node, alias
# ``c`` — see ``VectorCypherRetriever`` graph-channel filter pushdown). The executor
# owns the compile; this module only seeds + executes the compiled predicate.
_NODE_ALIAS = "c"

# Connection parameters match the ``make dev`` compose stack and the sibling
# ``tests/integration/test_neo4j_*_integration.py`` modules.
NEO4J_URL = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

# Path of the JSON seed-map artifact the seed entrypoint writes and the test reads.
SEED_MAP_PATH = os.environ.get("KHORA_CONFORMANCE_NEO4J_SEED_MAP", ".conformance_neo4j_seed_map.json")

# The 14 corpus family generators, by name (resolved off ``khora.filter.conformance``
# at call time so this module imports no family generator eagerly). Mirrors the
# postgres leg's ``_FAMILY_GENERATORS`` so the cypher leg seeds/asserts the same
# corpus, filtered to the cypher-targeting subset.
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


def _to_temporal_chunk(record: SeedRecord, chunk_id: UUID, namespace_id: UUID) -> TemporalChunk:
    """Adapt a :class:`SeedRecord` to the :class:`TemporalChunk` the production
    graph writer (:meth:`DualNodeManager.create_chunk_nodes_batch`) consumes.

    Mirrors ``_conformance_pg._to_temporal_chunk``: copies the fields the corpus
    addresses — the three date keys (``occurred_at`` / ``created_at`` /
    ``source_timestamp``), the seven denormalized document string keys, and the
    ``metadata`` blob — onto a ``TemporalChunk`` and lets the production writer own
    the property mapping (``.isoformat()`` on the dates, ``serialize_dict`` on
    ``metadata``, absent string/date keys written as Cypher nulls and so omitted from
    the node). The ``id`` is assigned here so it round-trips through the seed map: the
    writer stores it as ``str(chunk_id)``, the read query scopes on ``c.id IN
    $case_ids``. No embedding — ``create_chunk_nodes_batch`` does not write the vector
    onto the node, and the compiled predicate never touches the vector channel.
    """
    return TemporalChunk(
        id=chunk_id,
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=record.content,
        occurred_at=record.occurred_at,
        created_at=record.created_at,
        source_timestamp=record.source_timestamp,
        metadata=dict(record.metadata or {}),
        source_type=record.source_type,
        source_name=record.source_name,
        source_url=record.source_url,
        external_id=record.external_id,
        content_type=record.content_type,
        source=record.source,
        title=record.title,
    )


def _connect() -> Any:
    """Return a connected Neo4j async driver. Lazy import — neo4j is an optional extra."""
    from neo4j import AsyncGraphDatabase

    return AsyncGraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# --------------------------------------------------------------------------- #
# One-time seeding (write-once by the entrypoint).
# --------------------------------------------------------------------------- #


def neo4j_conformance_cases() -> list[ConformanceCase]:
    """Every corpus case whose ``backends`` includes ``cypher``.

    The single source of truth for *what the cypher leg seeds and asserts* — the
    seed entrypoint and the test module both call this so they never drift. Pulls
    from all 14 corpus families (mirroring ``postgres_conformance_cases``) and keeps
    the cypher-targeting subset; a family prunes cypher only with a documented reason.
    """
    from khora.filter import conformance

    cases: list[ConformanceCase] = []
    for generator in _FAMILY_GENERATORS:
        cases.extend(getattr(conformance, generator)())
    return [c for c in cases if "cypher" in c.backends]


async def build_seed_map() -> dict[str, dict[str, str]]:
    """Seed every cypher case ONCE and return ``case_id -> {seed_id: chunk_uuid}``.

    Writes one ``Chunk`` node per :class:`SeedRecord` through the production graph
    writer (:meth:`DualNodeManager.create_chunk_nodes_batch`), so the compiled
    predicate later reads the real property mapping and ``serialize_dict`` metadata
    form. Each case owns its own ``namespace_id`` (the read query scopes by chunk id,
    not namespace, but a per-case namespace keeps the nodes corpus-faithful). Chunk
    UUIDs are stringified for JSON. Called only by the one-time seed entrypoint —
    never by the test.

    The production writer stamps an absent ``created_at`` to ``now()``; the corpus
    expects cypher to leave it NULL (see module docstring), so after the batch write
    every node whose ``SeedRecord`` left ``created_at`` ``None`` has the property
    removed, restoring the missing-value semantics the compiler relies on.
    """
    driver = _connect()
    seed_map: dict[str, dict[str, str]] = {}
    try:
        database = os.environ.get("KHORA_NEO4J_DATABASE", "neo4j")
        manager = DualNodeManager(driver, database=database)
        for case in neo4j_conformance_cases():
            namespace_id = uuid4()
            id_map: dict[str, str] = {}
            chunks: list[TemporalChunk] = []
            absent_created_at: list[str] = []
            for record in case.seed_records:
                chunk_id = uuid4()
                id_map[record.id] = str(chunk_id)
                chunks.append(_to_temporal_chunk(record, chunk_id, namespace_id))
                if record.created_at is None:
                    absent_created_at.append(str(chunk_id))
            await manager.create_chunk_nodes_batch(chunks, namespace_id)
            await _strip_stamped_created_at(driver, database, absent_created_at)
            seed_map[case.id] = id_map
    finally:
        await driver.close()
    return seed_map


async def _strip_stamped_created_at(driver: Any, database: str, chunk_ids: Sequence[str]) -> None:
    """Remove the writer-stamped ``created_at`` from nodes whose seed left it absent.

    ``create_chunk_nodes_batch`` defaults an absent ``created_at`` to ``now()`` (the
    production behavior). The conformance corpus keeps ``created_at`` cases on the
    cypher leg precisely because cypher is expected to leave an absent ``created_at``
    NULL, so a ``now()`` value would satisfy a lower-bound pushdown and break the
    by-construction ``expected_ids``. Re-establish the missing-property semantics by
    removing the stamped value on exactly those nodes.
    """
    if not chunk_ids:
        return
    async with driver.session(database=database) as session:
        await session.run(
            f"MATCH ({_NODE_ALIAS}:Chunk) WHERE {_NODE_ALIAS}.id IN $ids REMOVE {_NODE_ALIAS}.created_at",
            ids=list(chunk_ids),
        )


def write_seed_map(seed_map: Mapping[str, Mapping[str, str]]) -> None:
    """Write the seed map to ``SEED_MAP_PATH`` (the one-time artifact write)."""
    with open(SEED_MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump(seed_map, fh, sort_keys=True, indent=2)


@lru_cache(maxsize=1)
def load_seed_map() -> dict[str, dict[str, UUID]]:
    """Load the seed map written by the entrypoint; ``case_id -> {seed_id: chunk UUID}``.

    Cached so every xdist worker parses the JSON at most once. Raises a clear,
    actionable error if the map is absent — the cypher leg is read-only and depends
    on the one-time seed step having run first.
    """
    if not os.path.exists(SEED_MAP_PATH):
        raise FileNotFoundError(
            f"conformance neo4j seed map not found at {SEED_MAP_PATH!r}; the cypher leg is "
            f"read-only and requires the one-time seed step to run first: "
            f"`python -m tests.integration.matrix._conformance_seed neo4j` "
            f"(set KHORA_CONFORMANCE_NEO4J_SEED_MAP to the same path for both steps)"
        )
    with open(SEED_MAP_PATH, encoding="utf-8") as fh:
        raw: dict[str, dict[str, str]] = json.load(fh)
    return {case_id: {seed_id: UUID(cid) for seed_id, cid in m.items()} for case_id, m in raw.items()}


# --------------------------------------------------------------------------- #
# Reachability gate + read-only predicate execution.
# --------------------------------------------------------------------------- #


def reachable() -> bool:
    """Whether the live Neo4j store is reachable (the local-dev skip gate).

    A cheap TCP connect to the bolt host/port — mirrors ``test_filter_conformance``'s
    ``_pg_reachable``. CI's own conftest still aborts RED when the store is required
    but down; this is the local-dev convenience gate the dispatch consults.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(NEO4J_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


async def run_live(
    id_map: Mapping[str, UUID],
    compiled: CompiledFilter[str],
    filter_ast: FilterNode,
    post_filter: Callable[[Mapping[str, Any]], bool],
    records: Sequence[tuple[str, Mapping[str, Any]]],
) -> frozenset[str]:
    """Run the compiled Cypher ``WHERE`` against ONE case's pre-seeded ``Chunk`` nodes.

    The :class:`~khora.filter.conformance.CypherExecutor` already compiled ``compiled``
    with the real ``compile_cypher`` and built ``post_filter`` (the ``compile_python``
    full-AST oracle). This runner only executes: it runs ``compiled.predicate`` (the
    server-side prefilter — cypher pushes system keys, leaves metadata to the
    post-filter) scoped to exactly this case's chunk ids (``c.id IN $case_ids``, from
    the seed map) so two cases never alias, then applies ``post_filter`` to the
    candidate rows by their record mapping.

    Read-only: no seeding here. ``records`` carries the ``(seed_id, mapping)`` pairs
    ``post_filter`` reads; they are keyed back to the surviving nodes by chunk id.
    Returns the ``SeedRecord`` ids whose node survived the prefilter and post-filter.
    """
    record_map = dict(records)
    chunk_to_seed = {chunk_id: seed_id for seed_id, chunk_id in id_map.items()}

    params: dict[str, Any] = dict(compiled.params)
    # ``case_ids`` cannot collide with a compiled bind: ``compile_cypher`` names every
    # bind ``{param_namespace}_{n}`` (default namespace ``f`` → ``f_0`` ...).
    params["case_ids"] = [str(cid) for cid in chunk_to_seed]
    query = (
        f"MATCH ({_NODE_ALIAS}:Chunk) WHERE {_NODE_ALIAS}.id IN $case_ids "
        f"AND ({compiled.predicate}) RETURN {_NODE_ALIAS}.id AS id"
    )

    driver = _connect()
    try:
        async with driver.session(database=os.environ.get("KHORA_NEO4J_DATABASE", "neo4j")) as session:
            result = await session.run(query, **params)
            rows = await result.data()
    finally:
        await driver.close()

    survivors: set[str] = set()
    for row in rows:
        chunk_id = UUID(row["id"])
        seed_id = chunk_to_seed.get(chunk_id)
        if seed_id is None:
            continue
        if post_filter(record_map[seed_id]):
            survivors.add(seed_id)
    return frozenset(survivors)


def executor_for(case: ConformanceCase) -> CypherExecutor:
    """Load ``case``'s pre-seeded entry and return a ready :class:`CypherExecutor`.

    The docker leg is seeded ONCE out-of-band (the seed step), so this only looks the
    case up in the persisted seed map and closes a sync ``LiveRunner`` (over that
    case's ``id_map``) that bridges to the async :func:`run_live` on a worker-thread
    loop — mirroring ``_conformance_pg``'s ``_postgres_executor_for``. The
    ``CypherExecutor`` invokes the REAL ``compile_cypher`` (this is what conformance
    checks); the runner only executes.
    """
    id_map = load_seed_map()[case.id]

    def runner(compiled, filter_ast, post_filter, records):  # noqa: ANN001, ANN202 - matches LiveRunner
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                lambda: asyncio.run(run_live(id_map, compiled, filter_ast, post_filter, records))
            ).result()

    return CypherExecutor(runner)
