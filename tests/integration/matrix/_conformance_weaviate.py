"""Shared live-Weaviate wiring for the recall-filter conformance leg.

The conformance harness (``khora.filter.conformance``) is storage-agnostic: it
lowers every case through the real validator + ``parse_to_ast`` and runs the
result through an injected runner. This module is that injected seam for the
live-Weaviate leg — it seeds ``KhoraChunk`` objects, applies the compiled
``_Filters`` tree as a server-side PREFILTER, reads back the candidates, and then
applies the ``compile_python`` post-filter.

Why the post-filter is MANDATORY: ``compile_weaviate`` is a deliberately
superset-safe partial pushdown — it pushes ONLY monotone-narrowing predicates
(``$eq`` / range / ``$in`` on the two declared date properties) and drops
everything else (negations, ``$exists`` / null, ALL metadata, ``source_timestamp``)
to the post-filter, because a server-side negation would false-exclude null/absent
rows. So the Weaviate prefilter always OVER-returns and the ``compile_python``
oracle over the FULL AST is what narrows to the exact answer — exactly as
``WeaviateTemporalStore.search`` does. A ``None`` prefilter means "no server-side
filter, post-filter everything".

Why ``WeaviateTemporalStore`` (not a hand-rolled client): the store owns the
``KhoraChunk`` collection definition (the two ``DATE`` properties the compiler's
``field_mapping`` declares, plus the ``metadata_json`` text property the post-filter
decodes) and the per-namespace tenant lifecycle. Seeding through it writes EXACTLY
the property surface the compiler reads, with dates stored as ``.isoformat()``
strings (the form ``compile_weaviate`` binds).

Seed/read split (write-once, read-many), mirroring the live-Postgres leg. The
collection is seeded ONCE by ``_conformance_seed`` (the workflow's one-time step),
which also persists a JSON ``seed map`` (``case_id -> {seed_id: chunk_uuid}``) to
the path in ``KHORA_CONFORMANCE_WEAVIATE_SEED_MAP``. The pytest step then runs
READ-ONLY: it loads the map and runs each compiled prefilter against the pre-seeded
objects — no seeding, so under ``-n auto`` every xdist worker only reads (no write
contention against the shared Weaviate container). Each case is seeded under its own
tenant (the per-case conformance namespace), so cases are tenant-isolated.

Kept out of ``conftest.py`` and named ``_conformance_weaviate`` (leading underscore,
not a ``test_`` module) so it is a plain helper shared by both the seed entrypoint
and the test module, never collected as tests itself.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

from khora.config import KhoraConfig
from khora.engines.skeleton.backends import TemporalChunk
from khora.engines.skeleton.backends.weaviate import (
    WeaviateBackendConfig,
    WeaviateTemporalStore,
    _coerce_datetime,
)
from khora.filter import CompiledFilter
from khora.filter.ast import FilterNode
from khora.filter.conformance import _DOC_STRING_KEYS, ConformanceCase, SeedRecord, WeaviateExecutor

# One fixed tenant for the whole conformance corpus. Chunk ids are globally unique
# (a fresh ``uuid4`` per seed record), so scoping a read by chunk id inside this one
# tenant never aliases across cases — and it frees ``run_live`` from needing the
# per-case namespace (the ``LiveRunner`` signature carries only ``id_map``).
_CONFORMANCE_NAMESPACE = UUID("00000000-0000-0000-0000-0000000c0fee")

# Connection parameters match the compose ``weaviate`` profile and the sibling
# ``tests/integration/test_weaviate_async_integration.py`` module.
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8090")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")  # optional - empty for anonymous
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50061"))

# Path of the JSON seed-map artifact the seed entrypoint writes and the test reads.
SEED_MAP_PATH = os.environ.get("KHORA_CONFORMANCE_WEAVIATE_SEED_MAP", ".conformance_weaviate_seed_map.json")

# The 14 corpus family generators, by name (resolved off ``khora.filter.conformance``
# at call time so this module imports no family generator eagerly). Mirrors the
# postgres leg's ``_FAMILY_GENERATORS`` so the weaviate leg seeds/asserts the same
# corpus, filtered to the weaviate-targeting subset.
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

# A small fixed embedding — the KhoraChunk collection requires a vector on write,
# but filter conformance never queries the vector channel (the prefilter + the
# id-scope are the only narrowing forces). The dimension only needs to be stable.
_EMBED_DIM = 8
_EMBED = [0.1] * _EMBED_DIM


def _to_chunk(record: SeedRecord, namespace_id: UUID, chunk_id: UUID) -> TemporalChunk:
    """Adapt a :class:`SeedRecord` to a :class:`TemporalChunk` for the store writer.

    Carries the fields the weaviate property surface exposes: the three date keys
    (the store stores ``occurred_at`` / ``created_at`` as ISO strings; the compiler
    declares only those two pushable), the seven denormalized document string keys
    (``source_type`` … ``title``, stored as filterable-only TEXT properties the
    post-filter reads back), and the ``metadata`` blob (serialized to the
    ``metadata_json`` text property the post-filter decodes). A document id is
    minted per chunk; it is irrelevant to the filter. ``source_timestamp`` rides on
    the chunk so the post-filter (which re-checks the full AST) sees it.
    """
    return TemporalChunk(
        id=chunk_id,
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=record.content,
        embedding=_EMBED,
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


def _store() -> WeaviateTemporalStore:
    """Build a (not-yet-connected) ``WeaviateTemporalStore`` against the live cluster.

    Uses a minimal ``KhoraConfig`` — the backend only touches Weaviate, not the
    storage coordinator.
    """
    from pydantic_settings import BaseSettings

    config = KhoraConfig.__new__(KhoraConfig)
    BaseSettings.__init__(config)
    backend_config = WeaviateBackendConfig(
        url=WEAVIATE_URL,
        api_key=WEAVIATE_API_KEY if WEAVIATE_API_KEY else None,
        grpc_port=WEAVIATE_GRPC_PORT,
    )
    return WeaviateTemporalStore(config, backend_config)


# --------------------------------------------------------------------------- #
# One-time seeding (write-once by the entrypoint).
# --------------------------------------------------------------------------- #


def weaviate_conformance_cases() -> list[ConformanceCase]:
    """Every corpus case whose ``backends`` includes ``weaviate``.

    The single source of truth for *what the weaviate leg seeds and asserts* — the
    seed entrypoint and the test module both call this so they never drift. Pulls
    from all 14 corpus families (mirroring ``postgres_conformance_cases``) and keeps
    the weaviate-targeting subset; a family prunes weaviate only with a documented
    reason.
    """
    from khora.filter import conformance

    cases: list[ConformanceCase] = []
    for generator in _FAMILY_GENERATORS:
        cases.extend(getattr(conformance, generator)())
    return [c for c in cases if "weaviate" in c.backends]


async def build_seed_map() -> dict[str, dict[str, str]]:
    """Seed every weaviate case ONCE and return ``case_id -> {seed_id: chunk_uuid}``.

    Writes one ``KhoraChunk`` object per :class:`SeedRecord` under one fixed
    conformance tenant. Chunk ids are globally unique (a fresh ``uuid4`` per record),
    so the id-scoped read in :func:`run_live` never aliases across cases. Chunk UUIDs
    are stringified for JSON. Called only by the one-time seed entrypoint — never by
    the test.
    """
    store = _store()
    await store.connect()
    seed_map: dict[str, dict[str, str]] = {}
    try:
        for case in weaviate_conformance_cases():
            id_map: dict[str, str] = {}
            chunks: list[TemporalChunk] = []
            for record in case.seed_records:
                chunk_id = uuid4()
                id_map[record.id] = str(chunk_id)
                chunks.append(_to_chunk(record, _CONFORMANCE_NAMESPACE, chunk_id))
            await store.create_chunks_batch(chunks)
            seed_map[case.id] = id_map
    finally:
        await store.disconnect()
    return seed_map


def write_seed_map(seed_map: Mapping[str, Mapping[str, str]]) -> None:
    """Write the seed map to ``SEED_MAP_PATH`` (the one-time artifact write)."""
    with open(SEED_MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump(seed_map, fh, sort_keys=True, indent=2)


@lru_cache(maxsize=1)
def load_seed_map() -> dict[str, dict[str, UUID]]:
    """Load the seed map written by the entrypoint; ``case_id -> {seed_id: chunk UUID}``.

    Cached so every xdist worker parses the JSON at most once. Raises a clear,
    actionable error if the map is absent — the weaviate leg is read-only and
    depends on the one-time seed step having run first.
    """
    if not os.path.exists(SEED_MAP_PATH):
        raise FileNotFoundError(
            f"conformance weaviate seed map not found at {SEED_MAP_PATH!r}; the weaviate leg is "
            f"read-only and requires the one-time seed step to run first: "
            f"`python -m tests.integration.matrix._conformance_seed weaviate` "
            f"(set KHORA_CONFORMANCE_WEAVIATE_SEED_MAP to the same path for both steps)"
        )
    with open(SEED_MAP_PATH, encoding="utf-8") as fh:
        raw: dict[str, dict[str, str]] = json.load(fh)
    return {case_id: {seed_id: UUID(cid) for seed_id, cid in m.items()} for case_id, m in raw.items()}


# --------------------------------------------------------------------------- #
# Reachability gate + read-only predicate execution.
# --------------------------------------------------------------------------- #


def reachable() -> bool:
    """Whether the live Weaviate store is reachable (the local-dev skip gate).

    A cheap TCP connect to the HTTP host/port parsed from ``WEAVIATE_URL`` — mirrors
    ``test_filter_conformance``'s ``_pg_reachable``. CI's own conftest still aborts RED
    when the store is required but down; this is the local-dev convenience gate.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _object_to_record(obj: Any) -> dict[str, Any]:
    """Rebuild the ``_record_mapping``-shaped dict the post-filter reads from a read-back object.

    Mirrors the harness's ``_record_mapping``: ``occurred_at`` coalesces to
    ``source_timestamp`` (the corpus's ``expected_ids`` assume that fold), the literal
    ``created_at`` / ``source_timestamp`` columns ride through, ``metadata`` decodes
    from the ``metadata_json`` text property, and each populated denormalized string
    key is read straight off the object surface. Re-applying the coalesce here is
    required because the weaviate seeder bypasses the engine, so the read-back record
    must reproduce the shape ``_record_mapping`` would have produced.
    """
    props = obj.properties
    occurred_at = _coerce_datetime(props.get("occurred_at"))
    source_timestamp = _coerce_datetime(props.get("source_timestamp"))
    mapping: dict[str, Any] = {
        "occurred_at": occurred_at if occurred_at is not None else source_timestamp,
        "created_at": _coerce_datetime(props.get("created_at")),
        "source_timestamp": source_timestamp,
        "metadata": json.loads(props["metadata_json"]) if props.get("metadata_json") else {},
    }
    for key in _DOC_STRING_KEYS:
        value = props.get(key)
        if value is not None:
            mapping[key] = value
    return mapping


async def run_live(
    id_map: Mapping[str, UUID],
    compiled: CompiledFilter[Any],
    filter_ast: FilterNode,
    post_filter: Callable[[Mapping[str, Any]], bool],
    records: Sequence[tuple[str, Mapping[str, Any]]],
) -> frozenset[str]:
    """Run the compiled Weaviate prefilter against ONE case's pre-seeded objects.

    The :class:`~khora.filter.conformance.WeaviateExecutor` already compiled ``compiled``
    with the real ``compile_weaviate`` (split mode, the two declared date properties)
    and built ``post_filter`` (the ``compile_python`` full-AST oracle). This runner only
    executes: it AND-s the resulting superset prefilter (``compiled.predicate``, which
    may be ``None`` ⇒ no server-side narrowing) with an id-scope (so two cases never
    alias), reads back the candidate objects from the fixed conformance tenant, then
    applies ``post_filter`` over the FULL AST — MANDATORY, because the prefilter always
    over-returns (it pushes only monotone-narrowing date predicates and defers
    negations / ``$exists`` / ``source_timestamp`` / all metadata).

    Read-only: no seeding here. The ``records`` parameter is part of the shared
    :class:`LiveRunner` signature but is intentionally unused on this leg: the
    post-filter now evaluates the read-back OBJECT surface (rebuilt by
    :func:`_object_to_record`) rather than the harness's in-memory mapping, so the leg
    verifies the store actually round-trips every filtered field (the eight
    denormalized document keys included), not just that the in-memory record would
    have matched. Returns the ``SeedRecord`` ids whose object survived prefilter and
    post-filter.
    """
    from weaviate.classes.query import Filter

    chunk_to_seed = {chunk_id: seed_id for seed_id, chunk_id in id_map.items()}

    # Scope to exactly this case's chunk ids; AND the superset prefilter when present.
    id_scope = Filter.by_id().contains_any([str(cid) for cid in chunk_to_seed])
    prefilter = id_scope if compiled.predicate is None else (id_scope & compiled.predicate)

    store = _store()
    await store.connect()
    try:
        collection = await store._get_collection(_CONFORMANCE_NAMESPACE)
        result = await collection.query.fetch_objects(filters=prefilter, limit=len(chunk_to_seed) or 1)
    finally:
        await store.disconnect()

    survivors: set[str] = set()
    for obj in result.objects:
        chunk_id = UUID(str(obj.uuid))
        seed_id = chunk_to_seed.get(chunk_id)
        if seed_id is None:
            continue
        if post_filter(_object_to_record(obj)):
            survivors.add(seed_id)
    return frozenset(survivors)


def executor_for(case: ConformanceCase) -> WeaviateExecutor:
    """Load ``case``'s pre-seeded entry and return a ready :class:`WeaviateExecutor`.

    The docker leg is seeded ONCE out-of-band (the seed step), so this only looks the
    case up in the persisted seed map and closes a sync ``LiveRunner`` (over that
    case's ``id_map``) that bridges to the async :func:`run_live` on a worker-thread
    loop — mirroring ``_conformance_pg``'s ``_postgres_executor_for``. The
    ``WeaviateExecutor`` invokes the REAL ``compile_weaviate`` (this is what
    conformance checks); the runner only executes.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    id_map = load_seed_map()[case.id]

    def runner(compiled, filter_ast, post_filter, records):  # noqa: ANN001, ANN202 - matches LiveRunner
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                lambda: asyncio.run(run_live(id_map, compiled, filter_ast, post_filter, records))
            ).result()

    return WeaviateExecutor(runner)
