"""Recall-filter conformance harness — ``@internal``.

The conformance harness drives one **catalog of filter cases** through every
backend compiler and asserts they all agree with the in-memory Python oracle
(:func:`~khora.filter.compilers.python.compile_python`). It is the machinery a
sibling catalog ticket fills with hand-authored cases and a sibling CI ticket
wires into a marked pytest job; this module owns the *machinery* only — the
case schema, the corpus runner, the three backend executors, the live-store
seeder, and the fully-generated ``F-OP`` (system-key operator-coverage) family.

The oracle contract is the whole point: a backend compiler is *conformant* iff,
for every case, the set of records its predicate keeps equals the set the Python
oracle keeps (or it raises :class:`RecallFilterUnsupportedError` on exactly the
backends a case marks unsupported). The runner never reconstructs a predicate —
it always lowers through the **real** validator
(:meth:`RecallFilter.model_validate`), the **real** :func:`parse_to_ast`, and the
**real** per-backend compiler, so a harness pass is evidence about production code,
not about a parallel re-implementation.

**Layout of a case.** A :class:`ConformanceCase` carries a filter (wire dict or a
constructed :class:`RecallFilter`), a small deterministic ``seed_records`` set,
the ``expected_ids`` survivors (:class:`SeedRecord` ids, resolved to live UUIDs by
the seeder), the ``backends`` it applies to, the ``expect_unsupported`` backends,
and an ``exercises`` tag tuple the coverage meta-test reads.

**The three executors.**

* :class:`PythonExecutor` — compiles the AST with :func:`compile_python` and runs
  the resulting callable against each record. This is the **oracle**.
* :class:`ChronicleExecutor` — delegates to the Chronicle plan/run seam in
  :mod:`khora.filter.execute` (the date-bound pushdown + Python post-filter).
* :class:`PostgresExecutor` — invokes the **real** :func:`compile_postgres` and
  hands the compiled predicate to an injected callable that runs it against a
  seeded coordinator. The live-Postgres wiring is the CI ticket's concern; this
  module defines only the seam.

``@internal``. Reachable as ``khora.filter.conformance`` for khora's own test
suite; **not** re-exported from :mod:`khora.__init__` or :mod:`khora.filter`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid5

import pytest

from khora.core.models import Chunk, Document, MemoryNamespace
from khora.filter import (
    RecallFilter,
    RecallFilterUnsupportedError,
)
from khora.filter.ast import FilterNode, parse_to_ast
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.compilers.python import compile_python

# Cross-file import boundary: conformance.py depends on execute.py (the production
# compile/execute seam); execute.py imports NOTHING back (one-way). ``build_compile_context``
# is the one production CompileContext builder; ``run_chronicle_filter`` is the
# Chronicle date-bound-pushdown + full-AST post-filter applied to in-memory records.
from khora.filter.execute import build_compile_context, run_chronicle_filter
from khora.storage.coordinator import StorageCoordinator

__all__ = [
    "BackendExecutor",
    "ChronicleExecutor",
    "ConformanceCase",
    "PostgresExecutor",
    "PostgresRunner",
    "PythonExecutor",
    "SeedRecord",
    "assert_case",
    "f_array_cases",
    "f_coerce_cases",
    "f_dates_cases",
    "f_dotkey_cases",
    "f_exists_cases",
    "f_impossible_cases",
    "f_logic_cases",
    "f_nullval_cases",
    "f_objeq_cases",
    "f_op_cases",
    "f_polarity_cases",
    "f_sel_cases",
    "f_sugar_cases",
    "f_unsup_cases",
    "oracle_survivors",
    "run_case_for_backend",
    "seed_case",
]

# The three backend names a case may target / a runner may dispatch on.
BACKENDS: frozenset[str] = frozenset({"python", "postgres", "chronicle"})

# The eight denormalized document system keys carried on the seed Document, in
# the order ``parse_to_ast`` lowers them (the three date keys live on the chunk).
_DOC_STRING_KEYS: tuple[str, ...] = (
    "source_type",
    "source_name",
    "source_url",
    "external_id",
    "content_type",
    "source",
    "title",
)
# The three date system keys, carried as real datetime columns on the chunk.
_DATE_KEYS: tuple[str, ...] = ("occurred_at", "created_at", "source_timestamp")


# --------------------------------------------------------------------------- #
# Case schema.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """One record to seed (one Document + one Chunk) and its filterable fields.

    ``@internal``. ``id`` is a stable, human-readable handle (NOT a UUID) the
    case's ``expected_ids`` reference; the seeder maps it to the live chunk UUID.
    ``content`` defaults to a fixed anchor so every chunk shares an embedding (the
    vector channel returns the whole set and the filter is the only narrowing
    force). The remaining fields populate the eight denormalized document system
    keys, the three date columns, and the chunk ``metadata`` blob — the surface a
    filter can address. A field left ``None`` / empty is simply absent from that
    record (the missing-value semantics each compiler must agree on).
    """

    id: str
    content: str = "conformance anchor"
    metadata: dict[str, Any] = field(default_factory=dict)
    source_timestamp: datetime | None = None
    occurred_at: datetime | None = None
    created_at: datetime | None = None
    source_type: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    external_id: str | None = None
    content_type: str | None = None
    source: str | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    """One filter + seed + expectation, run against a set of backends.

    ``@internal``.

    * ``id`` — stable, unique case handle (also the per-case namespace key, so the
      seeder is xdist-safe: each case owns its own namespace).
    * ``filter`` — the filter under test, as a wire ``dict`` (validated through
      :meth:`RecallFilter.model_validate`) or an already-constructed
      :class:`RecallFilter`.
    * ``seed_records`` — the records to seed; the filter selects a subset.
    * ``expected_ids`` — the :class:`SeedRecord` ids that must survive the filter,
      or ``None`` when the case only asserts an unsupported outcome.
    * ``backends`` — the backends this case applies to (subset of
      :data:`BACKENDS`).
    * ``expect_unsupported`` — the backends on which the filter must raise
      :class:`RecallFilterUnsupportedError` rather than return survivors.
    * ``exercises`` — free-form coverage tags (e.g. ``("F-OP", "occurred_at",
      "$gte")``); the coverage meta-test asserts the union of these over the
      generated corpus covers every :data:`SYSTEM_KEYS` member.
    """

    id: str
    filter: dict[str, Any] | RecallFilter
    seed_records: tuple[SeedRecord, ...]
    expected_ids: frozenset[str] | None
    backends: frozenset[str]
    expect_unsupported: frozenset[str] = frozenset()
    exercises: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# AST resolution.
# --------------------------------------------------------------------------- #


def _resolve_ast(filter_: dict[str, Any] | RecallFilter) -> FilterNode:
    """Lower a case filter to its canonical AST through the real pipeline.

    A wire ``dict`` is validated with :meth:`RecallFilter.model_validate`; an
    already-constructed :class:`RecallFilter` is used as-is. Both then lower via
    the real :func:`parse_to_ast`. Never reconstructs or mocks the AST.
    """
    model = filter_ if isinstance(filter_, RecallFilter) else RecallFilter.model_validate(filter_)
    return parse_to_ast(model)


def _record_mapping(record: SeedRecord) -> dict[str, Any]:
    """Build the in-memory record mapping the Python oracle reads.

    Mirrors the engine's ``_chunk_to_record`` shape so the oracle sees the same
    surface a real recall post-filter would: the effective event time
    ``COALESCE(occurred_at, source_timestamp)``, the literal ``created_at`` /
    ``source_timestamp`` columns, the ``metadata`` blob, and any populated
    denormalized document key. A document key left ``None`` stays absent (the
    missing-value semantics the compilers must agree on).
    """
    mapping: dict[str, Any] = {
        "occurred_at": record.occurred_at if record.occurred_at is not None else record.source_timestamp,
        "created_at": record.created_at,
        "source_timestamp": record.source_timestamp,
        "metadata": record.metadata or {},
    }
    for key in _DOC_STRING_KEYS:
        value = getattr(record, key)
        if value is not None:
            mapping[key] = value
    return mapping


# --------------------------------------------------------------------------- #
# Backend executors.
# --------------------------------------------------------------------------- #


class BackendExecutor(Protocol):
    """The seam each backend implements: AST + records -> surviving ids.

    ``@internal``. ``records`` is a sequence of ``(seed_id, mapping)`` pairs;
    ``survivors`` returns the subset of ``seed_id`` values the backend's predicate
    keeps. An executor that cannot express a clause raises
    :class:`RecallFilterUnsupportedError` (the harness asserts that against
    ``ConformanceCase.expect_unsupported``).
    """

    def survivors(
        self,
        filter_ast: FilterNode,
        records: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> frozenset[str]: ...


class PythonExecutor:
    """The oracle: run :func:`compile_python` against each record.

    ``@internal``. Compiles the AST with the real
    :func:`~khora.filter.compilers.python.compile_python` (``on_unsupported``
    ``"raise"`` — the oracle must express the whole filter or surface the gap) and
    applies the resulting callable to each record mapping. The survivor set this
    returns is the reference every other backend is checked against.
    """

    def survivors(
        self,
        filter_ast: FilterNode,
        records: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> frozenset[str]:
        ctx = build_compile_context("chunks", on_unsupported="raise")
        predicate = compile_python(filter_ast, ctx).predicate
        return frozenset(seed_id for seed_id, mapping in records if predicate(mapping))


class ChronicleExecutor:
    """Run a filter through the Chronicle plan/run seam.

    ``@internal``. Delegates to :func:`khora.filter.execute.run_chronicle_filter`,
    which applies the Chronicle ``source_timestamp`` date-bound pushdown and the
    :func:`compile_python` post-filter (the full-AST safety net) — the same path
    the Chronicle engine drives. The harness passes the in-memory record mappings
    so the executor needs no live store; ``run_chronicle_filter`` returns the
    surviving record mappings (the same dict objects, in order), which this maps
    back to their ``seed_id`` by object identity.
    """

    def survivors(
        self,
        filter_ast: FilterNode,
        records: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> frozenset[str]:
        # ``run_chronicle_filter`` filters by reference and returns the same mapping
        # objects, so identity (``id(...)``) recovers the seed_id unambiguously even
        # when two records carry equal field values.
        by_identity = {id(mapping): seed_id for seed_id, mapping in records}
        survivors = run_chronicle_filter(filter_ast, [mapping for _, mapping in records])
        return frozenset(by_identity[id(mapping)] for mapping in survivors)


class PostgresRunner(Protocol):
    """Run a compiled Postgres predicate against a seeded store -> surviving ids.

    ``@internal``. The injected seam :class:`PostgresExecutor` calls after it has
    compiled the AST with the real :func:`compile_postgres`. Wiring this to a live
    coordinator (seed → emit ``WHERE`` → collect surviving chunk ids → map back to
    ``seed_id``) is the CI ticket's concern; the harness defines only the contract.
    ``predicate`` is the SQLAlchemy ``ColumnElement[bool]`` and ``params`` its bind
    parameters; ``records`` is the ``(seed_id, mapping)`` list under test.
    """

    def __call__(
        self,
        predicate: Any,
        params: Mapping[str, Any],
        records: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> frozenset[str]: ...


class PostgresExecutor:
    """Compile with the real :func:`compile_postgres`; run via an injected runner.

    ``@internal``. The real compiler IS invoked (this is what conformance checks);
    execution against a live Postgres is delegated to the injected
    :class:`PostgresRunner` so the harness stays storage-agnostic. ``compile_postgres``
    is driven with ``on_unsupported`` ``"raise"`` so a clause Postgres cannot push
    down surfaces as :class:`RecallFilterUnsupportedError` (matched against
    ``expect_unsupported``), rather than being silently dropped.
    """

    def __init__(self, runner: PostgresRunner) -> None:
        self._runner = runner

    def survivors(
        self,
        filter_ast: FilterNode,
        records: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> frozenset[str]:
        # ``"khora_chunks"`` is the production Postgres target the skeleton pgvector
        # backend compiles against (build_compile_context("khora_chunks",
        # on_unsupported="raise")). Matching it here means the harness exercises the
        # exact compiler invocation prod runs, not a context-only shim.
        ctx = build_compile_context("khora_chunks", on_unsupported="raise")
        compiled = compile_postgres(filter_ast, ctx)
        return self._runner(compiled.predicate, compiled.params, records)


# --------------------------------------------------------------------------- #
# Corpus runner.
# --------------------------------------------------------------------------- #


def run_case_for_backend(
    case: ConformanceCase,
    backend: str,
    *,
    executor: BackendExecutor,
) -> frozenset[str]:
    """Resolve a case's filter to an AST and run it through one backend.

    ``@internal``. Lowers ``case.filter`` via the real validator + ``parse_to_ast``,
    builds the ``(seed_id, mapping)`` record list from ``case.seed_records``, runs
    it through ``executor``, and returns the surviving :class:`SeedRecord` ids.
    Raises :class:`RecallFilterUnsupportedError` straight through when the executor
    cannot express the filter (the caller asserts that against
    ``expect_unsupported``).
    """
    filter_ast = _resolve_ast(case.filter)
    records = [(rec.id, _record_mapping(rec)) for rec in case.seed_records]
    return executor.survivors(filter_ast, records)


def assert_case(case: ConformanceCase, backend: str, executor: BackendExecutor) -> None:
    """Assert a case's outcome on one backend (the per-(case, backend) check).

    ``@internal``. When ``backend`` is in ``case.expect_unsupported`` the filter
    must raise :class:`RecallFilterUnsupportedError`; otherwise the surviving id
    set must equal ``case.expected_ids``. ``expected_ids`` must be set for a
    survivor assertion (a case that only asserts an unsupported outcome lists the
    backend in ``expect_unsupported`` instead).
    """
    if backend in case.expect_unsupported:
        with pytest.raises(RecallFilterUnsupportedError):
            run_case_for_backend(case, backend, executor=executor)
        return
    if case.expected_ids is None:
        raise ValueError(f"case {case.id!r} on backend {backend!r}: expected_ids is required for a survivor assertion")
    assert run_case_for_backend(case, backend, executor=executor) == case.expected_ids


def oracle_survivors(case: ConformanceCase) -> frozenset[str]:
    """Run the Python oracle against a case — an **authoring-time** sanity helper.

    ``@internal``. Returns the :class:`SeedRecord` ids the
    :class:`PythonExecutor` keeps for ``case.filter`` over ``case.seed_records``.
    A case author can compare this against the ids they *declared* in
    ``expected_ids`` to catch a hand-counting slip while authoring a case.

    **This is deliberately NOT used inside** :func:`assert_case`. The assertion
    target is always the hand-declared ``case.expected_ids`` — never the oracle's
    live output — so a wrong :func:`compile_python` fails its own ``"python"``
    case (the oracle is itself falsifiable) instead of silently redefining every
    expectation to whatever it currently computes.
    """
    return run_case_for_backend(case, "python", executor=PythonExecutor())


# --------------------------------------------------------------------------- #
# Live-store seeder.
# --------------------------------------------------------------------------- #


# Fixed namespace root for deriving a deterministic per-case namespace_id. Any
# constant UUID works — it only needs to be stable so the same case.id always maps
# to the same namespace (xdist-safe) and distinct case ids never collide.
_CONFORMANCE_NS_ROOT = UUID("00000000-0000-0000-0000-0000000c0fee")


def _case_namespace_id(case: ConformanceCase) -> UUID:
    """Derive a deterministic namespace_id from ``case.id`` (xdist-safe).

    Same ``case.id`` → same namespace on every worker; distinct ``case.id`` →
    distinct namespace → no cross-worker collision. Never a random ``uuid4``.
    """
    return uuid5(_CONFORMANCE_NS_ROOT, case.id)


async def seed_case(coord: StorageCoordinator, case: ConformanceCase) -> dict[str, UUID]:
    """Seed a case's records into a live coordinator; return ``seed_id -> chunk UUID``.

    ``@internal``. Writes one :class:`Document` + one :class:`Chunk` per
    :class:`SeedRecord` through the coordinator's **write API only**
    (:meth:`create_namespace`, :meth:`create_document`, :meth:`create_chunks_batch`)
    — never raw SQL/Cypher. The namespace ``namespace_id`` is derived
    deterministically from ``case.id`` (:func:`_case_namespace_id`), so the same
    case maps to the same namespace on every xdist worker and distinct cases never
    collide — no random ``uuid4`` that could clash across workers. The namespace is
    write-once and read-only after seeding.

    Every chunk shares one embedding derived from its content (all records use the
    same default content, so the vector channel returns the whole seed set and the
    filter is the only narrowing force). The Document carries the eight
    denormalized document system keys; the Chunk carries the ``metadata`` blob and
    the three date columns. Returns the ``seed_id -> chunk.id`` map the runner uses
    to translate ``expected_ids`` (SeedRecord ids) into live chunk UUIDs.
    """
    # Lazy import: the seeder runs only under the live-store integration job, and
    # the fixture helper lives under ``tests/`` (not importable in a bare install).
    from tests.integration._sqlite_lance_fixtures import EMBED_DIM, fake_embedding

    # Deterministic, per-case namespace (xdist-safe): the stable namespace_id is
    # derived from case.id, so the same case maps to the same namespace on every
    # worker and distinct cases never collide. Child rows (documents/chunks) FK to
    # ``memory_namespaces.id`` — the row-level id, NOT the stable namespace_id
    # (MemoryNamespace: "use id for ... child-table FK lookups") — so seed every
    # child under ``ns.id``, the persisted row id create_namespace returns.
    ns = await coord.create_namespace(
        MemoryNamespace(
            namespace_id=_case_namespace_id(case),
            metadata={"conformance_case": case.id},
        )
    )
    namespace_id = ns.id

    id_map: dict[str, UUID] = {}
    chunks: list[Chunk] = []
    for record in case.seed_records:
        doc = Document(
            namespace_id=namespace_id,
            content=record.content,
            source_type=record.source_type if record.source_type is not None else "library",
            source_name=record.source_name,
            source_url=record.source_url,
            external_id=record.external_id,
            content_type=record.content_type,
            source=record.source,
            title=record.title,
        )
        await coord.create_document(doc)

        chunk_kwargs: dict[str, Any] = {
            "namespace_id": namespace_id,
            "document_id": doc.id,
            "content": record.content,
            "chunk_index": 0,
            "embedding": fake_embedding(record.content, dim=EMBED_DIM),
            "embedding_model": "fake",
            "metadata": dict(record.metadata),
        }
        for date_key in _DATE_KEYS:
            value = getattr(record, date_key)
            if value is not None:
                chunk_kwargs[date_key] = value
        chunk = Chunk(**chunk_kwargs)
        id_map[record.id] = chunk.id
        chunks.append(chunk)

    await coord.create_chunks_batch(chunks)
    return id_map


# --------------------------------------------------------------------------- #
# Case-family generators.
# --------------------------------------------------------------------------- #
#
# The catalog ticket fills the F-COERCE / F-EXISTS / F-LOGIC / F-SUGAR / F-DATES
# / F-NULLVAL / F-SEL / F-OBJEQ / F-DOTKEY / F-UNSUP families with hand-authored
# expected_ids. This module fully implements only the F-OP system-key
# operator-coverage family (every SYSTEM_KEYS member × its operators), with
# expected_ids computed BY CONSTRUCTION from a known seed so the runner's
# python-oracle cross-check can confirm — never define — them.


# Fixed seed anchors for the date F-OP cases. ``_DATE_HIT`` is in range of every
# generated date bound, ``_DATE_MISS`` out of range; ``_DATE_MID`` sits between the
# two bounds so the boundary ops separate cleanly, and the ``-5`` record carries no
# date at all (the NULL row, exercised by F1). UTC, matching the validator's
# normalization. The bound ``_DATE_BOUND`` is below the hit and above the miss.
_DATE_HIT = datetime(2026, 6, 1, tzinfo=UTC)
_DATE_MID = datetime(2026, 3, 15, tzinfo=UTC)
_DATE_MISS = datetime(2020, 1, 1, tzinfo=UTC)
_DATE_BOUND = "2026-01-01T00:00:00Z"

# All three backends are oracle-comparable for the F-OP family on the embedded
# date/metadata surface; the catalog/CI tickets prune per-backend as needed (the
# eight denorm document keys are not carried on the legacy chronicle DTO, so a
# positive predicate on them is chronicle-empty — flagged via ``backends`` there).
_OP_BACKENDS: frozenset[str] = frozenset({"python", "postgres", "chronicle"})

# String-key F-OP cases run on python + chronicle only — NOT postgres. The postgres
# leg targets ``khora_chunks`` (the denormalized single-table target), but
# ``seed_case`` denormalizes only ``_DATE_KEYS`` onto the seeded Chunk — the core
# Chunk model carries none of the seven string document keys (only the skeleton DTO
# does), so a positive string predicate is postgres-empty until the seeder
# denormalizes the doc-keys onto the chunk row (a follow-up harness concern).
# Pruning postgres here keeps ``ConformanceCase.backends`` an honest single source of
# truth; the date-key F-OP cases keep postgres. python (the oracle) + chronicle still
# validate every string case.
_STRING_OP_BACKENDS: frozenset[str] = _OP_BACKENDS - frozenset({"postgres"})

# The one non-null string system key — its column defaults to ``"library"`` at the
# SQL layer, so a ``-5`` row with the key unset would be ``"library"`` on the SQL
# backends but ``None`` in the in-memory oracle. To keep ``expected_ids`` backend-
# consistent that key is seeded with five populated rows (no NULL row); every other
# string key is nullable (unset ⇒ NULL on the SQL backends, ``None`` in the oracle —
# consistent), so it carries the ``-5`` NULL row that F1 (Mongo-faithful negation)
# keeps under ``$ne``/``$nin``. (String-key cases run on python + chronicle today —
# see ``_STRING_OP_BACKENDS`` — but the seed stays SQL-consistent for when the
# seeder denormalizes the doc-keys onto the chunk row and they gain a SQL leg.)
_NON_NULL_STRING_KEY = "source_type"


def _date_field_seed(key: str) -> tuple[SeedRecord, ...]:
    """Five records for a date-key F-OP case: two ties, a mid, a miss, and a NULL row.

    The date is stamped on ``key`` (one of the three date columns). ``-3``/``-4``
    are tied at the hit instant (so ``$eq`` keeps a pair), ``-2`` sits at the mid
    instant (in range of the lower bound, exercising boundary inclusion), ``-1`` is
    the out-of-range miss, and ``-5`` carries no date at all — the NULL row F1
    keeps under ``$ne``/``$nin``. All share the default content so they share an
    embedding.
    """
    return (
        SeedRecord(id=f"{key}-1", **{key: _DATE_MISS}),
        SeedRecord(id=f"{key}-2", **{key: _DATE_MID}),
        SeedRecord(id=f"{key}-3", **{key: _DATE_HIT}),
        SeedRecord(id=f"{key}-4", **{key: _DATE_HIT}),
        SeedRecord(id=f"{key}-5"),
    )


def _string_field_seed(key: str, *, nullable: bool) -> tuple[SeedRecord, ...]:
    """Five records for a string-key F-OP case.

    ``-1`` is ``"library"``, ``-2``/``-3`` are tied at ``"connection"`` (so ``$eq``
    keeps a pair), ``-4`` is ``"direct"``, and ``-5`` is either a fifth distinct
    value (``"kafka"``, for the non-null key) or the NULL row (key unset, for the
    nullable keys). The fixed value vocabulary mirrors the catalog's string seed so
    each operator separates a known subset.
    """
    fifth = SeedRecord(id=f"{key}-5") if nullable else SeedRecord(id=f"{key}-5", **{key: "kafka"})
    return (
        SeedRecord(id=f"{key}-1", **{key: "library"}),
        SeedRecord(id=f"{key}-2", **{key: "connection"}),
        SeedRecord(id=f"{key}-3", **{key: "connection"}),
        SeedRecord(id=f"{key}-4", **{key: "direct"}),
        fifth,
    )


def _date_op_cases(key: str) -> list[ConformanceCase]:
    """Every date-key F-OP case: range + set ops over one date key (+ empty-set, F1).

    ``expected_ids`` is computed by construction from the known five-record seed
    (``-1`` miss @ 2020-01-01, ``-2`` mid @ 2026-03-15, ``-3``/``-4`` tied @
    2026-06-01, ``-5`` NULL). The bound is 2026-01-01, so ``$gt``/``$gte`` keep the
    mid + the tied pair, the upper-bound ops keep only the miss, ``$eq`` against the
    hit instant keeps the tied pair, and ``$lte`` against the hit keeps everything
    dated. Negations follow F1: ``$ne``/``$nin``/``$nin:[]`` over the nullable date
    column **include the NULL ``-5`` row**; ``$in:[]`` keeps nothing.

    Date keys deliberately have **no** ``$exists`` case: ``DateOps`` carries no
    ``$exists`` field, so ``$exists`` on a date key is a *validation* failure (not
    a compile-time unsupported outcome) — the presence operator is exercised on the
    string keys instead. The operand is a plain ISO-8601 string (the system-key
    ``DateOps`` form), not the ``{"$date": ...}`` typed literal (that form is the
    metadata grammar's).
    """
    r1, r2, r3, r4, r5 = (r.id for r in _date_field_seed(key))
    seed = _date_field_seed(key)
    bound = _DATE_BOUND
    hit_iso = _DATE_HIT.isoformat().replace("+00:00", "Z")

    # ``created_at`` is stamped by the store on insert (the khora_chunks writer
    # coalesces a missing created_at to ``now()``), so the "absent" ``-5`` record
    # cannot be represented as NULL on the postgres leg — its ``now()`` value
    # satisfies the lower-bound ops and breaks them. ``created_at`` is therefore
    # validated on python (the oracle) + chronicle only, which keep an absent value
    # NULL. ``occurred_at`` / ``source_timestamp`` are user-supplied and stay NULL
    # when absent, so they keep postgres.
    backends = _OP_BACKENDS - frozenset({"postgres"}) if key == "created_at" else _OP_BACKENDS

    def case(suffix: str, predicate: dict[str, Any], expected: frozenset[str], op_tag: str) -> ConformanceCase:
        return ConformanceCase(
            id=f"F-OP-{key}-{suffix}",
            filter={key: predicate},
            seed_records=seed,
            expected_ids=expected,
            backends=backends,
            exercises=("F-OP", key, op_tag),
        )

    return [
        case("gt", {"$gt": bound}, frozenset({r2, r3, r4}), "$gt"),
        case("gte", {"$gte": bound}, frozenset({r2, r3, r4}), "$gte"),
        case("lt", {"$lt": bound}, frozenset({r1}), "$lt"),
        case("lte", {"$lte": bound}, frozenset({r1}), "$lte"),
        case("eq", {"$eq": hit_iso}, frozenset({r3, r4}), "$eq"),
        # F1: $ne over a nullable date column includes the NULL -5 row.
        case("ne", {"$ne": hit_iso}, frozenset({r1, r2, r5}), "$ne"),
        case("in", {"$in": [hit_iso, _DATE_MISS.isoformat().replace("+00:00", "Z")]}, frozenset({r1, r3, r4}), "$in"),
        case("in-empty", {"$in": []}, frozenset(), "$in"),
        case("nin", {"$nin": [hit_iso]}, frozenset({r1, r2, r5}), "$nin"),
        case("nin-empty", {"$nin": []}, frozenset({r1, r2, r3, r4, r5}), "$nin"),
    ]


def _string_op_cases(key: str) -> list[ConformanceCase]:
    """Every string-key F-OP case: ``$eq``/``$ne``/``$in``/``$nin``/``$exists`` (+ empty-set, F1).

    ``expected_ids`` is computed by construction from the known five-record seed
    (``-1`` ``"library"``, ``-2``/``-3`` ``"connection"``, ``-4`` ``"direct"``,
    ``-5`` ``"kafka"`` for the non-null key or NULL for the nullable keys). The
    eight denormalized document keys are always present columns at the SQL layer, so
    ``$exists:true`` is trivially all-records — its coverage is the presence-operator
    tag, not a narrowing assertion.

    Negations follow F1: for the **nullable** string keys, ``$ne``/``$nin``/
    ``$nin:[]`` include the NULL ``-5`` row; the **non-null** key (``source_type``)
    has no NULL row so its negations only flip among populated values. The bare-list
    short-circuit (``{key: [a, b]}``) lowers to an ``$eq`` *exact-array* operand,
    which can never equal a scalar column — a constant-false predicate keeping
    nothing (§4 rule #3).
    """
    nullable = key != _NON_NULL_STRING_KEY
    seed = _string_field_seed(key, nullable=nullable)
    r1, r2, r3, r4, r5 = (r.id for r in seed)
    # The ``-5`` row is kept by every negation here regardless of nullability: for a
    # nullable key it is the NULL row F1 includes; for the non-null key it is
    # ``"kafka"``, which is unequal to every negation operand below ("direct" /
    # "connection"). The two reasons converge on the same survivor set, so the
    # expected_ids stay consistent across the python oracle and chronicle (the two
    # backends these string cases run on — see ``_STRING_OP_BACKENDS``).

    def case(suffix: str, predicate: Any, expected: frozenset[str], op_tag: str) -> ConformanceCase:
        return ConformanceCase(
            id=f"F-OP-{key}-{suffix}",
            filter={key: predicate},
            seed_records=seed,
            expected_ids=expected,
            backends=_STRING_OP_BACKENDS,
            exercises=("F-OP", key, op_tag),
        )

    return [
        case("eq", {"$eq": "connection"}, frozenset({r2, r3}), "$eq"),
        case("ne", {"$ne": "direct"}, frozenset({r1, r2, r3, r5}), "$ne"),
        case("in", {"$in": ["library", "direct"]}, frozenset({r1, r4}), "$in"),
        case("in-empty", {"$in": []}, frozenset(), "$in"),
        case("nin", {"$nin": ["connection"]}, frozenset({r1, r4, r5}), "$nin"),
        case("nin-empty", {"$nin": []}, frozenset({r1, r2, r3, r4, r5}), "$nin"),
        case("exists-true", {"$exists": True}, frozenset({r1, r2, r3, r4, r5}), "$exists"),
        # Bare-list ⇒ $eq exact-array ⇒ constant-false against a scalar column.
        case("eq-barelist", ["library", "direct"], frozenset(), "$eq"),
    ]


def _metadata_op_case() -> ConformanceCase:
    """A representative metadata-scalar F-OP case (``metadata.tier == "gold"``).

    The system-key families above cover every :data:`SYSTEM_KEYS` member;
    ``metadata`` is intentionally excluded from ``SYSTEM_KEYS`` (it is the free-form
    blob, not a system key), so this single representative scalar case rounds out
    the F-OP family's surface without enumerating the metadata grammar (that is the
    catalog ticket's F-OBJEQ / F-DOTKEY territory).
    """
    seed = (
        SeedRecord(id="meta-gold", metadata={"tier": "gold"}),
        SeedRecord(id="meta-silver", metadata={"tier": "silver"}),
        SeedRecord(id="meta-absent"),
    )
    return ConformanceCase(
        id="F-OP-metadata-tier-eq",
        filter={"metadata.tier": "gold"},
        seed_records=seed,
        expected_ids=frozenset({"meta-gold"}),
        backends=_OP_BACKENDS,
        exercises=("F-OP", "metadata.tier", "$eq"),
    )


def f_op_cases() -> list[ConformanceCase]:
    """The fully-generated ``F-OP`` family: every system key × its operators.

    ``@internal``. Iterates every :data:`SYSTEM_KEYS` member — the three date keys
    (``occurred_at`` / ``created_at`` / ``source_timestamp``) across range / set
    ops with the empty-set ``$in:[]`` / ``$nin:[]`` variants, and the seven string
    keys (``source_type`` / ``source_name`` / ``source_url`` / ``external_id`` /
    ``content_type`` / ``source`` / ``title``) across ``$eq`` / ``$ne`` / ``$in`` /
    ``$nin`` / ``$exists``, the empty-set variants, and the bare-list ``$eq``
    exact-array constant-false short-circuit — plus one representative metadata
    scalar. Each key is seeded with five records; the nullable keys carry a NULL
    ``-5`` row that F1 (Mongo-faithful negation) keeps under ``$ne`` / ``$nin``,
    while the one non-null key (``source_type``) is seeded fully populated so its
    ``expected_ids`` stay consistent across the in-memory oracle and chronicle (the
    string-key cases run on python + chronicle only — see ``_STRING_OP_BACKENDS``).
    Every case tags the system key it exercises in ``exercises`` so the coverage
    meta-test can assert the union covers :data:`SYSTEM_KEYS`. ``expected_ids`` is
    computed by construction from each case's known seed (the runner confirms,
    never defines, them via the oracle).
    """
    cases: list[ConformanceCase] = []
    for key in _DATE_KEYS:
        cases.extend(_date_op_cases(key))
    for key in _DOC_STRING_KEYS:
        cases.extend(_string_op_cases(key))
    cases.append(_metadata_op_case())
    return cases


# --------------------------------------------------------------------------- #
# Hand-authored case-family generators.
# --------------------------------------------------------------------------- #
#
# Each generator yields a list of :class:`ConformanceCase` with HAND-AUTHORED
# ``expected_ids`` — counted from each seed against the ADR §4 match-mode rules,
# never copied from the oracle's output (``oracle_survivors`` is an authoring aid
# the harness's own meta-tests cross-check the declared sets against, keeping the
# oracle falsifiable). All cases declare ``backends = _OP_BACKENDS`` and an empty
# ``expect_unsupported``: routing is equivalence-only across the three in-memory
# executors, which read the same record mapping. The corpus is append-only per
# family; ids are family-prefixed and namespace-local.


def _case(
    cid: str,
    filter_: Any,
    seed: tuple[SeedRecord, ...],
    expected: frozenset[str],
    exercises: tuple[str, ...],
) -> ConformanceCase:
    """Build a ConformanceCase scoped to the three in-memory backends.

    A thin constructor so the family generators read as a flat table of
    ``id · filter · expected_ids · exercises`` rows.
    """
    return ConformanceCase(
        id=cid,
        filter=filter_,
        seed_records=seed,
        expected_ids=expected,
        backends=_OP_BACKENDS,
        exercises=exercises,
    )


# F-COERCE / F-POLARITY share one six-record seed per operand type-block so the
# polarity negation is an exact set-complement (subtlety a). Layout per block:
# match / nomatch / wrongtype / array / object / absent. The ``array`` record is
# seeded NOT to contain the ``$eq`` operand (subtlety b) so array-containment
# (#21) does not mask the type-gate.
def _coerce_seed(key: str, match: Any, nomatch: Any, wrongtype: Any, array: list[Any]) -> tuple[SeedRecord, ...]:
    """A six-record type-gate seed under ``metadata.<key>`` (shared by F-COERCE / F-POLARITY)."""
    return (
        SeedRecord(id="match", metadata={key: match}),
        SeedRecord(id="nomatch", metadata={key: nomatch}),
        SeedRecord(id="wrongtype", metadata={key: wrongtype}),
        SeedRecord(id="array", metadata={key: array}),
        SeedRecord(id="object", metadata={key: {"k": 1}}),
        SeedRecord(id="absent", metadata={}),
    )


# The four operand type-blocks (number / bool / string / $date). Each pairs a
# typed seed with the value its positive comparison singles out.
_COERCE_NUM = _coerce_seed("score", 15, 5, "20", [99])
_COERCE_BOOL = _coerce_seed("flag", True, False, 1, [False])
_COERCE_STR = _coerce_seed("code", "m", "a", 5, ["z"])
_COERCE_DATE = _coerce_seed("due", "2026-06-01T00:00:00Z", "2020-01-01T00:00:00Z", 5, ["2099-01-01T00:00:00Z"])

_DATE_LIT = "2026-01-01T00:00:00Z"


def f_coerce_cases() -> list[ConformanceCase]:
    """F-COERCE: §4 rule #1 type-gate — every positive op keeps only the typed satisfier.

    A comparison whose operand type differs from the stored value never compares
    lexicographically and never aborts (Rule 1): it type-gates and excludes the
    mismatch. So a numeric ``$gt`` against a numeric-string excludes it, a bool
    gate excludes the ``int`` ``1`` (``isinstance(True, int)`` trap), a string gate
    excludes a stored number, and a ``$date`` operand parses-or-excludes. Each case
    keeps exactly the one correctly-typed record the bound singles out (``match``
    for the upper-bound / equality ops, ``nomatch`` for the lower-bound ops — both
    are the only correctly-typed satisfier of their predicate).
    """
    n, b, s, d = _COERCE_NUM, _COERCE_BOOL, _COERCE_STR, _COERCE_DATE
    only = frozenset
    return [
        _case("F-COERCE-num-gt", {"metadata.score": {"$gt": 10}}, n, only({"match"}), ("F-COERCE", "number", "$gt")),
        _case("F-COERCE-num-gte", {"metadata.score": {"$gte": 15}}, n, only({"match"}), ("F-COERCE", "number", "$gte")),
        _case("F-COERCE-num-lt", {"metadata.score": {"$lt": 10}}, n, only({"nomatch"}), ("F-COERCE", "number", "$lt")),
        _case(
            "F-COERCE-num-lte", {"metadata.score": {"$lte": 5}}, n, only({"nomatch"}), ("F-COERCE", "number", "$lte")
        ),
        _case("F-COERCE-num-eq", {"metadata.score": {"$eq": 15}}, n, only({"match"}), ("F-COERCE", "number", "$eq")),
        _case(
            "F-COERCE-date-gt",
            {"metadata.due": {"$gt": {"$date": _DATE_LIT}}},
            d,
            only({"match"}),
            ("F-COERCE", "$date", "$gt"),
        ),
        _case(
            "F-COERCE-date-lt",
            {"metadata.due": {"$lt": {"$date": _DATE_LIT}}},
            d,
            only({"nomatch"}),
            ("F-COERCE", "$date", "$lt"),
        ),
        _case(
            "F-COERCE-date-eq",
            {"metadata.due": {"$date": "2026-06-01T00:00:00Z"}},
            d,
            only({"match"}),
            ("F-COERCE", "$date", "$eq"),
        ),
        # bool gate: True > False keeps only the bool match (the int 1 is excluded).
        _case("F-COERCE-bool-gt", {"metadata.flag": {"$gt": False}}, b, only({"match"}), ("F-COERCE", "bool", "$gt")),
        _case("F-COERCE-bool-eq", {"metadata.flag": {"$eq": True}}, b, only({"match"}), ("F-COERCE", "bool", "$eq")),
        # string gate: a stored number is excluded; the array record does not contain "m".
        _case("F-COERCE-str-gt", {"metadata.code": {"$gt": "c"}}, s, only({"match"}), ("F-COERCE", "string", "$gt")),
        _case("F-COERCE-str-lt", {"metadata.code": {"$lt": "c"}}, s, only({"nomatch"}), ("F-COERCE", "string", "$lt")),
        _case("F-COERCE-str-eq", {"metadata.code": {"$eq": "m"}}, s, only({"match"}), ("F-COERCE", "string", "$eq")),
    ]


def f_polarity_cases() -> list[ConformanceCase]:
    """F-POLARITY: §4 rule #2 — a negation INCLUDES the mismatch / wrong-type / absent rows.

    Reuses the F-COERCE seeds (identical record ids), so each negation's survivor
    set is the exact set-complement of the matching positive op: everything **but**
    the record the positive predicate singled out. ``$ne`` / ``$nin`` / a field
    ``$not`` of a range all flip the same way — a record whose value mismatches, is
    wrong-typed, an array, an object, or absent satisfies the negation. The
    ``$not($gt 5)`` and ``$not($lt 6)`` cases are the pointed traps: the negation is
    ``NOT(gate AND compare)``, so a wrong-typed record (which fails the gate) is
    **included**, not excluded.
    """
    n, b, s, d = _COERCE_NUM, _COERCE_BOOL, _COERCE_STR, _COERCE_DATE
    all_but_match = frozenset({"nomatch", "wrongtype", "array", "object", "absent"})
    all_but_nomatch = frozenset({"match", "wrongtype", "array", "object", "absent"})
    return [
        _case("F-POLARITY-num-ne", {"metadata.score": {"$ne": 15}}, n, all_but_match, ("F-POLARITY", "number", "$ne")),
        _case(
            "F-POLARITY-num-nin",
            {"metadata.score": {"$nin": [15, 20]}},
            n,
            all_but_match,
            ("F-POLARITY", "number", "$nin"),
        ),
        _case(
            "F-POLARITY-num-not-gt",
            {"metadata.score": {"$not": {"$gt": 5}}},
            n,
            all_but_match,
            ("F-POLARITY", "number", "$not"),
        ),
        _case(
            "F-POLARITY-date-ne",
            {"metadata.due": {"$ne": {"$date": "2026-06-01T00:00:00Z"}}},
            d,
            all_but_match,
            ("F-POLARITY", "$date", "$ne"),
        ),
        _case(
            "F-POLARITY-date-not-gt",
            {"metadata.due": {"$not": {"$gt": {"$date": _DATE_LIT}}}},
            d,
            all_but_match,
            ("F-POLARITY", "$date", "$not"),
        ),
        _case("F-POLARITY-bool-ne", {"metadata.flag": {"$ne": True}}, b, all_but_match, ("F-POLARITY", "bool", "$ne")),
        _case("F-POLARITY-str-ne", {"metadata.code": {"$ne": "m"}}, s, all_but_match, ("F-POLARITY", "string", "$ne")),
        _case(
            "F-POLARITY-str-nin", {"metadata.code": {"$nin": ["m"]}}, s, all_but_match, ("F-POLARITY", "string", "$nin")
        ),
        # Pointed trap: $not($lt 6) — nomatch (5) satisfies $lt 6, so $not excludes
        # ONLY nomatch; every other row (incl. wrong-type / absent) is included.
        _case(
            "F-POLARITY-num-not-lt",
            {"metadata.score": {"$not": {"$lt": 6}}},
            n,
            all_but_nomatch,
            ("F-POLARITY", "number", "$not"),
        ),
    ]


# F-ARRAY array-containment seed (#21). ``tags`` is variously an array, a scalar,
# absent, or empty; the scalar / $in / $nin / $ne cases probe array-aware match.
_ARRAY_SEED = (
    SeedRecord(id="arr-list", metadata={"tags": ["urgent", "release"]}),
    SeedRecord(id="arr-scalar", metadata={"tags": "urgent"}),
    SeedRecord(id="arr-other-list", metadata={"tags": ["okrs"]}),
    SeedRecord(id="arr-empty", metadata={"tags": []}),
    SeedRecord(id="arr-absent", metadata={}),
    SeedRecord(id="arr-okrs-scalar", metadata={"tags": "blocker"}),
)
# F-ARRAY exact-array seed (order-sensitive): a reversed list must NOT match.
_ARRAY_EXACT_SEED = (
    SeedRecord(id="ex-ordered", metadata={"tags": ["urgent", "release"]}),
    SeedRecord(id="ex-reversed", metadata={"tags": ["release", "urgent"]}),
    SeedRecord(id="ex-single", metadata={"tags": ["urgent"]}),
    SeedRecord(id="ex-absent", metadata={}),
)
# F-ARRAY range-vs-array seed: a range op is scalar-only (an array value excluded).
_ARRAY_RANGE_SEED = (
    SeedRecord(id="rng-scalar", metadata={"scores": 30}),
    SeedRecord(id="rng-array", metadata={"scores": [30, 40]}),
    SeedRecord(id="rng-low", metadata={"scores": 10}),
    SeedRecord(id="rng-absent", metadata={}),
)


def f_array_cases() -> list[ConformanceCase]:
    """F-ARRAY: §4 array-aware containment (#21), exact-array, and range scalar-only.

    A scalar operand matches both a scalar field and an array field that CONTAINS
    it; ``$in`` is contains-any; ``$nin`` / ``$ne`` exclude a containing field but
    INCLUDE an absent one (Rule 2). A bare-list operand is exact-array equality —
    order-sensitive (a reversed list does not match) — while ``$in`` over a list is
    element membership, not whole-list equality. A range op (``$gt``) is
    scalar-only: an array value is excluded. ``$exists`` treats an empty array as
    present.
    """
    a, e, r = _ARRAY_SEED, _ARRAY_EXACT_SEED, _ARRAY_RANGE_SEED
    return [
        # scalar matches array-containing + scalar.
        _case(
            "F-ARRAY-scalar-contains",
            {"metadata.tags": "urgent"},
            a,
            frozenset({"arr-list", "arr-scalar"}),
            ("F-ARRAY", "metadata.tags", "contains"),
        ),
        # $in contains-any: urgent (list+scalar) OR okrs (other-list).
        _case(
            "F-ARRAY-in-any",
            {"metadata.tags": {"$in": ["urgent", "okrs"]}},
            a,
            frozenset({"arr-list", "arr-scalar", "arr-other-list"}),
            ("F-ARRAY", "metadata.tags", "$in"),
        ),
        # $nin: exclude if contains either; INCLUDE empty + absent (Rule 2).
        _case(
            "F-ARRAY-nin",
            {"metadata.tags": {"$nin": ["urgent", "okrs"]}},
            a,
            frozenset({"arr-empty", "arr-absent", "arr-okrs-scalar"}),
            ("F-ARRAY", "metadata.tags", "$nin"),
        ),
        # $ne urgent: not-contains + absent + empty (Rule 2).
        _case(
            "F-ARRAY-ne",
            {"metadata.tags": {"$ne": "urgent"}},
            a,
            frozenset({"arr-other-list", "arr-empty", "arr-absent", "arr-okrs-scalar"}),
            ("F-ARRAY", "metadata.tags", "$ne"),
        ),
        # bare-list exact-array, order-sensitive: reversed must NOT match.
        _case(
            "F-ARRAY-exact",
            {"metadata.tags": ["urgent", "release"]},
            e,
            frozenset({"ex-ordered"}),
            ("F-ARRAY", "metadata.tags", "$eq", "exact-array"),
        ),
        # $in over a list is element membership: every list containing "urgent".
        _case(
            "F-ARRAY-in-element",
            {"metadata.tags": {"$in": ["urgent"]}},
            e,
            frozenset({"ex-ordered", "ex-reversed", "ex-single"}),
            ("F-ARRAY", "metadata.tags", "$in", "element"),
        ),
        # range $gt is scalar-only: the array value is excluded.
        _case(
            "F-ARRAY-range-scalar",
            {"metadata.scores": {"$gt": 25}},
            r,
            frozenset({"rng-scalar"}),
            ("F-ARRAY", "metadata.scores", "$gt"),
        ),
        # $exists: an empty array is present.
        _case(
            "F-ARRAY-exists",
            {"metadata.tags": {"$exists": True}},
            a,
            frozenset({"arr-list", "arr-scalar", "arr-other-list", "arr-empty", "arr-okrs-scalar"}),
            ("F-ARRAY", "metadata.tags", "$exists"),
        ),
        # $exists:false is the complement — only the absent record.
        _case(
            "F-ARRAY-exists-false",
            {"metadata.tags": {"$exists": False}},
            a,
            frozenset({"arr-absent"}),
            ("F-ARRAY", "metadata.tags", "$exists"),
        ),
    ]


# F-EXISTS truth-table seed: system NULL / value, metadata absent / present-null /
# present, nested absent / present-null / present. The metadata-absent and nested
# cases are the load-bearing absent-vs-present-null distinction.
_EXISTS_SEED = (
    SeedRecord(id="sys-null"),  # source_name unset → NULL
    SeedRecord(id="sys-value", source_name="linear"),
    SeedRecord(id="md-absent", metadata={}),
    SeedRecord(id="md-null", metadata={"mk": None}),  # present JSON-null
    SeedRecord(id="md-value", metadata={"mk": "v"}),
)
_EXISTS_NESTED_SEED = (
    SeedRecord(id="nest-seg-absent", metadata={"a": {}}),
    SeedRecord(id="nest-root-absent", metadata={}),
    SeedRecord(id="nest-null", metadata={"a": {"b": None}}),  # present JSON-null
    SeedRecord(id="nest-value", metadata={"a": {"b": "v"}}),
)


def f_exists_cases() -> list[ConformanceCase]:
    """F-EXISTS: presence across system NULL, metadata absent / present-null / present.

    ``$exists`` is allowed on string + metadata keys only (a date key raises — a
    validator concern). A system key is always a present column, so ``$exists:true``
    is all-rows and ``$exists:false`` is empty on it. On a metadata path, presence
    distinguishes absent from present-JSON-null: an explicit ``None`` value is
    PRESENT (``$exists:true`` includes it; only a genuinely absent path satisfies
    ``$exists:false``). The present-and-exactly-null composition (``$exists:true``
    AND ``{k: null}``) isolates the present-null record from the absent one — the v1
    way to express ``$type:"null"``.
    """
    e, ne = _EXISTS_SEED, _EXISTS_NESTED_SEED
    all_e = frozenset({r.id for r in e})
    return [
        # System key: always present → $exists:true is all, $exists:false is none.
        _case(
            "F-EXISTS-sys-true", {"source_name": {"$exists": True}}, e, all_e, ("F-EXISTS", "source_name", "$exists")
        ),
        _case(
            "F-EXISTS-sys-false",
            {"source_name": {"$exists": False}},
            e,
            frozenset(),
            ("F-EXISTS", "source_name", "$exists"),
        ),
        # Metadata: present (incl. JSON-null) → true; only absent → false.
        _case(
            "F-EXISTS-md-true",
            {"metadata.mk": {"$exists": True}},
            e,
            frozenset({"md-null", "md-value"}),
            ("F-EXISTS", "metadata.mk", "$exists"),
        ),
        _case(
            "F-EXISTS-md-false",
            {"metadata.mk": {"$exists": False}},
            e,
            frozenset({"sys-null", "sys-value", "md-absent"}),
            ("F-EXISTS", "metadata.mk", "$exists"),
        ),
        # Nested path: every segment must resolve; a present-null leaf is present.
        _case(
            "F-EXISTS-nested-true",
            {"metadata.a.b": {"$exists": True}},
            ne,
            frozenset({"nest-null", "nest-value"}),
            ("F-EXISTS", "metadata.a.b", "$exists"),
        ),
        _case(
            "F-EXISTS-nested-false",
            {"metadata.a.b": {"$exists": False}},
            ne,
            frozenset({"nest-seg-absent", "nest-root-absent"}),
            ("F-EXISTS", "metadata.a.b", "$exists"),
        ),
        # Present-and-exactly-null: isolates md-null from md-absent.
        _case(
            "F-EXISTS-present-and-null",
            {"$and": [{"metadata.mk": {"$exists": True}}, {"metadata.mk": None}]},
            e,
            frozenset({"md-null"}),
            ("F-EXISTS", "metadata.mk", "present-and-null"),
        ),
        # Contrast: bare {k: null} is null-OR-missing → the present-null row plus every
        # row where mk is absent (md-absent and both system-key rows carry no mk).
        _case(
            "F-EXISTS-null-or-missing",
            {"metadata.mk": None},
            e,
            frozenset({"md-absent", "md-null", "sys-null", "sys-value"}),
            ("F-EXISTS", "metadata.mk", "null-or-missing"),
        ),
    ]


def f_logic_cases() -> list[ConformanceCase]:
    """F-LOGIC: boolean composition — implicit-AND, ``$or``/``$in``, ``$nor``, ``$not``, De Morgan.

    Each ``equivalent-to`` pair returns byte-identical row sets (the canonical hash
    may or may not coincide — the desugar identities like ``$nor ≡ $not($or)`` and
    implicit-AND ``≡ $and`` share a hash, while ``$or ≡ $in`` and the De Morgan pair
    are row-equivalent but hash-distinct; the harness asserts rows). NB-mixed cases
    span system + metadata so the whole op must post-filter-combine, never pushing
    only the system disjunct. Missing-key rows exercise the Mongo-faithful negation
    polarity under composition.
    """
    # System+metadata composition seed.
    s = (
        SeedRecord(id="L-linear-conn", source_name="linear", source_type="connection"),
        SeedRecord(id="L-slack-lib", source_name="slack", source_type="library"),
        SeedRecord(id="L-linear-lib", source_name="linear", source_type="library"),
        SeedRecord(id="L-linear-gold", source_name="linear", metadata={"tier": "gold"}),
        SeedRecord(id="L-silver", metadata={"tier": "silver"}),
    )
    # De Morgan seed.
    dm = (
        SeedRecord(id="dm-linear-conn", source_name="linear", source_type="connection"),
        SeedRecord(id="dm-linear-lib", source_name="linear", source_type="library"),
        SeedRecord(id="dm-slack-lib", source_name="slack", source_type="library"),
    )
    # Field-position $not over a date range seed.
    nr = (
        SeedRecord(id="nr-recent", source_timestamp=_DATE_HIT),
        SeedRecord(id="nr-old", source_timestamp=_DATE_MISS),
        SeedRecord(id="nr-undated"),
    )
    # Multi-op-one-key range bracket seed (a date between two bounds).
    br = (
        SeedRecord(id="br-in", occurred_at=_DATE_MID),
        SeedRecord(id="br-low", occurred_at=_DATE_MISS),
        SeedRecord(id="br-high", occurred_at=datetime(2099, 1, 1, tzinfo=UTC)),
    )
    # Multi-op-one metadata key ($ne + $nin), with a missing-key row (Rule 2).
    mk = (
        SeedRecord(id="mk-a", metadata={"tag": "a"}),
        SeedRecord(id="mk-b", metadata={"tag": "b"}),
        SeedRecord(id="mk-c", metadata={"tag": "c"}),
        SeedRecord(id="mk-absent", metadata={}),
    )
    # Distributivity + depth-3 seed (system + metadata).
    dist = (
        SeedRecord(id="ds-linear-conn-gold", source_name="linear", source_type="connection", metadata={"tier": "gold"}),
        SeedRecord(id="ds-linear-lib-silver", source_name="linear", source_type="library", metadata={"tier": "silver"}),
        SeedRecord(id="ds-slack-conn-gold", source_name="slack", source_type="connection", metadata={"tier": "gold"}),
    )
    # Presence-negation seed ($not($exists)).
    pe = (
        SeedRecord(id="pe-has", metadata={"k": "v"}),
        SeedRecord(id="pe-absent", metadata={}),
    )
    return [
        # implicit-AND ≡ $and (system).
        _case(
            "F-LOGIC-implicit-and",
            {"source_name": "linear", "source_type": "connection"},
            s,
            frozenset({"L-linear-conn"}),
            ("F-LOGIC", "implicit-and"),
        ),
        _case(
            "F-LOGIC-explicit-and",
            {"$and": [{"source_name": "linear"}, {"source_type": "connection"}]},
            s,
            frozenset({"L-linear-conn"}),
            ("F-LOGIC", "$and"),
        ),
        # $or ≡ $in (single key) — row-equivalent.
        _case(
            "F-LOGIC-or",
            {"$or": [{"source_name": "linear"}, {"source_name": "slack"}]},
            s,
            frozenset({"L-linear-conn", "L-slack-lib", "L-linear-lib", "L-linear-gold"}),
            ("F-LOGIC", "$or"),
        ),
        _case(
            "F-LOGIC-in-equiv",
            {"source_name": {"$in": ["linear", "slack"]}},
            s,
            frozenset({"L-linear-conn", "L-slack-lib", "L-linear-lib", "L-linear-gold"}),
            ("F-LOGIC", "$in"),
        ),
        # $nor ≡ $not($or) — desugar identity (shares a hash).
        _case(
            "F-LOGIC-nor",
            {"$nor": [{"source_name": "linear"}]},
            s,
            frozenset({"L-slack-lib", "L-silver"}),
            ("F-LOGIC", "$nor"),
        ),
        _case(
            "F-LOGIC-not-or",
            {"$not": {"$or": [{"source_name": "linear"}]}},
            s,
            frozenset({"L-slack-lib", "L-silver"}),
            ("F-LOGIC", "$not"),
        ),
        # mixed system+metadata $or — whole op combines, no system-only pushdown.
        _case(
            "F-LOGIC-mixed-or",
            {"$or": [{"source_name": "linear"}, {"metadata.tier": "gold"}]},
            s,
            frozenset({"L-linear-conn", "L-linear-lib", "L-linear-gold"}),
            ("F-LOGIC", "mixed-or"),
        ),
        # 1-arg $and ≡ bare (arity boundary).
        _case(
            "F-LOGIC-and-arity1",
            {"$and": [{"source_name": "linear"}]},
            s,
            frozenset({"L-linear-conn", "L-linear-lib", "L-linear-gold"}),
            ("F-LOGIC", "$and", "arity"),
        ),
        # De Morgan: NOT(a AND b) ≡ (NOT a) OR (NOT b) — row-equivalent.
        _case(
            "F-LOGIC-demorgan-not-and",
            {"$not": {"$and": [{"source_name": "linear"}, {"source_type": "connection"}]}},
            dm,
            frozenset({"dm-linear-lib", "dm-slack-lib"}),
            ("F-LOGIC", "demorgan"),
        ),
        _case(
            "F-LOGIC-demorgan-or-ne",
            {"$or": [{"source_name": {"$ne": "linear"}}, {"source_type": {"$ne": "connection"}}]},
            dm,
            frozenset({"dm-linear-lib", "dm-slack-lib"}),
            ("F-LOGIC", "demorgan"),
        ),
        # field-position $not over a date range: includes the undated row (Rule 2).
        _case(
            "F-LOGIC-not-range",
            {"source_timestamp": {"$not": {"$gt": _DATE_LIT}}},
            nr,
            frozenset({"nr-old", "nr-undated"}),
            ("F-LOGIC", "$not", "range"),
        ),
        # multi-op one date key (range bracket): only the in-range row survives.
        _case(
            "F-LOGIC-range-bracket",
            {"occurred_at": {"$gt": _DATE_LIT, "$lt": "2099-01-01T00:00:00Z"}},
            br,
            frozenset({"br-in"}),
            ("F-LOGIC", "range-bracket"),
        ),
        # multi-op one metadata key ($ne + $nin): the conjunction excludes a, b;
        # the absent row is included (both negations are missing-inclusive).
        _case(
            "F-LOGIC-meta-ne-nin",
            {"metadata.tag": {"$ne": "a", "$nin": ["b"]}},
            mk,
            frozenset({"mk-c", "mk-absent"}),
            ("F-LOGIC", "multi-op"),
        ),
        # double-negation: $not($not(eq)) ≡ eq.
        _case(
            "F-LOGIC-double-negation",
            {"$not": {"$not": {"source_name": "linear"}}},
            dist,
            frozenset({"ds-linear-conn-gold", "ds-linear-lib-silver"}),
            ("F-LOGIC", "double-negation"),
        ),
        # distributivity: a AND (b OR c) ≡ (a AND b) OR (a AND c) — row-equivalent.
        _case(
            "F-LOGIC-distrib-and-or",
            {"$and": [{"source_name": "linear"}, {"$or": [{"source_type": "connection"}, {"source_type": "library"}]}]},
            dist,
            frozenset({"ds-linear-conn-gold", "ds-linear-lib-silver"}),
            ("F-LOGIC", "distributivity"),
        ),
        _case(
            "F-LOGIC-distrib-or-and",
            {
                "$or": [
                    {"$and": [{"source_name": "linear"}, {"source_type": "connection"}]},
                    {"$and": [{"source_name": "linear"}, {"source_type": "library"}]},
                ]
            },
            dist,
            frozenset({"ds-linear-conn-gold", "ds-linear-lib-silver"}),
            ("F-LOGIC", "distributivity"),
        ),
        # depth-3 nesting, all ops: linear AND (connection OR NOT(tier=silver)).
        _case(
            "F-LOGIC-depth3",
            {
                "$and": [
                    {"source_name": "linear"},
                    {"$or": [{"source_type": "connection"}, {"$not": {"metadata.tier": "silver"}}]},
                ]
            },
            dist,
            frozenset({"ds-linear-conn-gold"}),
            ("F-LOGIC", "depth-3"),
        ),
        # $not($exists): the negation of presence keeps only the absent row.
        _case(
            "F-LOGIC-not-exists",
            {"metadata.k": {"$not": {"$exists": True}}},
            pe,
            frozenset({"pe-absent"}),
            ("F-LOGIC", "$not", "$exists"),
        ),
    ]


def f_sugar_cases() -> list[ConformanceCase]:
    """F-SUGAR: bare-value ``$eq`` sugar, exact-array, subdoc, and the ``$in`` negative guards.

    A bare scalar matches its ``$eq`` form; a bare metadata-path scalar is
    array-aware containment; a bare list is ``$eq`` EXACT-ARRAY equality (NOT
    ``$in`` membership); a bare subdocument is whole-subdoc ``object_equal`` (a
    reordered operand is order-insensitive, matching both rows). S7/S8 are negative
    guards: an explicit ``$in`` is membership / contains-any, deliberately NOT the
    bare-list exact-array form. Each ``a``/``b`` pair declares the same survivor set
    (the desugaring is row-transparent).
    """
    seed = (
        SeedRecord(id="sug-linear", source_name="linear"),
        SeedRecord(id="sug-slack", source_name="slack"),
        SeedRecord(id="sug-tag-list", metadata={"tag": ["urgent", "x"]}),
        SeedRecord(id="sug-tag-scalar", metadata={"tag": "urgent"}),
        SeedRecord(id="sug-tags-ab", metadata={"tags": ["a", "b"]}),
        SeedRecord(id="sug-tags-a", metadata={"tags": ["a"]}),
        SeedRecord(id="sug-labels", metadata={"labels": {"team": "ingest", "tier": "gold"}}),
        SeedRecord(id="sug-labels-rev", metadata={"labels": {"tier": "gold", "team": "ingest"}}),
    )
    return [
        # S1 bare scalar ≡ $eq (system key).
        _case(
            "F-SUGAR-S1a-bare",
            {"source_name": "linear"},
            seed,
            frozenset({"sug-linear"}),
            ("F-SUGAR", "source_name", "bare"),
        ),
        _case(
            "F-SUGAR-S1b-eq",
            {"source_name": {"$eq": "linear"}},
            seed,
            frozenset({"sug-linear"}),
            ("F-SUGAR", "source_name", "$eq"),
        ),
        # S3 bare metadata scalar ≡ $eq (array-aware containment).
        _case(
            "F-SUGAR-S3a-bare",
            {"metadata.tag": "urgent"},
            seed,
            frozenset({"sug-tag-list", "sug-tag-scalar"}),
            ("F-SUGAR", "metadata.tag", "bare"),
        ),
        _case(
            "F-SUGAR-S3b-eq",
            {"metadata.tag": {"$eq": "urgent"}},
            seed,
            frozenset({"sug-tag-list", "sug-tag-scalar"}),
            ("F-SUGAR", "metadata.tag", "$eq"),
        ),
        # S4 bare list ≡ $eq EXACT-ARRAY (order matters; NOT $in).
        _case(
            "F-SUGAR-S4-exact-array",
            {"metadata.tags": ["a", "b"]},
            seed,
            frozenset({"sug-tags-ab"}),
            ("F-SUGAR", "metadata.tags", "exact-array"),
        ),
        # S5 bare subdoc ≡ $eq object_equal (order-insensitive on operand keys).
        _case(
            "F-SUGAR-S5a-bare",
            {"metadata.labels": {"team": "ingest", "tier": "gold"}},
            seed,
            frozenset({"sug-labels", "sug-labels-rev"}),
            ("F-SUGAR", "metadata.labels", "subdoc"),
        ),
        _case(
            "F-SUGAR-S5b-reordered",
            {"metadata.labels": {"tier": "gold", "team": "ingest"}},
            seed,
            frozenset({"sug-labels", "sug-labels-rev"}),
            ("F-SUGAR", "metadata.labels", "subdoc"),
        ),
        # S7 explicit $in is membership (NOT a bare list).
        _case(
            "F-SUGAR-S7-in-membership",
            {"source_name": {"$in": ["linear", "slack"]}},
            seed,
            frozenset({"sug-linear", "sug-slack"}),
            ("F-SUGAR", "source_name", "$in"),
        ),
        # S8 explicit $in on a metadata array is contains-any (NOT exact-array).
        _case(
            "F-SUGAR-S8-in-contains-any",
            {"metadata.tags": {"$in": ["a", "b"]}},
            seed,
            frozenset({"sug-tags-ab", "sug-tags-a"}),
            ("F-SUGAR", "metadata.tags", "$in"),
        ),
    ]


def f_dates_cases() -> list[ConformanceCase]:
    """F-DATES: ``$date`` typed literal, timezone normalization, lexicographic control, AND-compose.

    A ``$date`` literal is metadata-grammar only (a system date key takes a plain
    ISO/datetime, so ``{"$date": ...}`` on a system key is a validator concern). A
    bare string operand on a metadata path is LEXICOGRAPHIC, not date-parsed (the
    negative control). Naive and tz-aware operands at the same instant compare
    equal (UTC normalization, by-instant). The three system date keys are seeded
    DISTINCT (collapse tripwire) so an AND over two of them narrows correctly. A
    ``$date`` op on an unparseable value parses-or-excludes (never raises).
    """
    md = (
        SeedRecord(id="dt-hit", metadata={"due": "2026-06-01T00:00:00Z"}),
        SeedRecord(id="dt-miss", metadata={"due": "2020-01-01T00:00:00Z"}),
        SeedRecord(id="dt-bad", metadata={"due": "not-a-date"}),
        SeedRecord(id="dt-absent", metadata={}),
    )
    # Distinct instant per key (collapse tripwire): occurred 2026-06, created 2026-03, source_ts 2026-01.
    three = (
        SeedRecord(
            id="dk-all-recent",
            occurred_at=_DATE_HIT,
            created_at=_DATE_MID,
            source_timestamp=datetime(2026, 1, 15, tzinfo=UTC),
        ),
        SeedRecord(
            id="dk-old-occurred",
            occurred_at=_DATE_MISS,
            created_at=_DATE_MID,
            source_timestamp=datetime(2026, 1, 15, tzinfo=UTC),
        ),
    )
    # Single instant for the tz-normalization probes.
    tz = (SeedRecord(id="tz-noon", metadata={"due": "2026-06-01T12:00:00Z"}),)
    # Boundary probe on a system date key (plain ISO operand).
    bnd = (
        SeedRecord(id="bd-on", occurred_at=_DATE_HIT),
        SeedRecord(id="bd-after", occurred_at=datetime(2026, 7, 1, tzinfo=UTC)),
        SeedRecord(id="bd-before", occurred_at=datetime(2026, 5, 1, tzinfo=UTC)),
    )
    after2025 = "2025-01-01T00:00:00Z"
    return [
        # $date range literal on a metadata path.
        _case(
            "F-DATES-md-date-gt",
            {"metadata.due": {"$gt": {"$date": _DATE_LIT}}},
            md,
            frozenset({"dt-hit"}),
            ("F-DATES", "metadata.due", "$date"),
        ),
        _case(
            "F-DATES-md-date-lt",
            {"metadata.due": {"$lt": {"$date": _DATE_LIT}}},
            md,
            frozenset({"dt-miss"}),
            ("F-DATES", "metadata.due", "$date"),
        ),
        # negative control: a bare string is lexicographic, not date-parsed — so the
        # ISO hit AND the non-date string "not-a-date" both sort above "2025".
        _case(
            "F-DATES-lexicographic",
            {"metadata.due": {"$gt": "2025"}},
            md,
            frozenset({"dt-hit", "dt-bad"}),
            ("F-DATES", "metadata.due", "lexicographic"),
        ),
        # $date eq ≡ ISO on a metadata path.
        _case(
            "F-DATES-md-date-eq",
            {"metadata.due": {"$date": "2026-06-01T00:00:00Z"}},
            md,
            frozenset({"dt-hit"}),
            ("F-DATES", "metadata.due", "$date"),
        ),
        # parse-or-exclude: the unparseable + absent rows drop, never raise.
        _case(
            "F-DATES-parse-or-exclude",
            {"metadata.due": {"$gte": {"$date": _DATE_LIT}}},
            md,
            frozenset({"dt-hit"}),
            ("F-DATES", "metadata.due", "$date"),
        ),
        # naive operand normalizes to UTC → equals the same instant.
        _case(
            "F-DATES-naive-utc",
            {"metadata.due": {"$eq": {"$date": "2026-06-01T12:00:00"}}},
            tz,
            frozenset({"tz-noon"}),
            ("F-DATES", "metadata.due", "naive-utc"),
        ),
        # tz-aware non-UTC operand compares by instant (14:00+02:00 == 12:00Z).
        _case(
            "F-DATES-tz-by-instant",
            {"metadata.due": {"$eq": {"$date": "2026-06-01T14:00:00+02:00"}}},
            tz,
            frozenset({"tz-noon"}),
            ("F-DATES", "metadata.due", "tz-instant"),
        ),
        # system date-key boundary: $gt excludes the boundary, $gte includes it.
        _case(
            "F-DATES-boundary-gt",
            {"occurred_at": {"$gt": "2026-06-01T00:00:00Z"}},
            bnd,
            frozenset({"bd-after"}),
            ("F-DATES", "occurred_at", "boundary"),
        ),
        _case(
            "F-DATES-boundary-gte",
            {"occurred_at": {"$gte": "2026-06-01T00:00:00Z"}},
            bnd,
            frozenset({"bd-after", "bd-on"}),
            ("F-DATES", "occurred_at", "boundary"),
        ),
        # three-keys-distinct AND-compose: only the all-recent row clears both bounds.
        _case(
            "F-DATES-and-compose",
            {"$and": [{"occurred_at": {"$gt": after2025}}, {"created_at": {"$gt": after2025}}]},
            three,
            frozenset({"dk-all-recent"}),
            ("F-DATES", "and-compose"),
        ),
    ]


def f_nullval_cases() -> list[ConformanceCase]:
    """F-NULLVAL: explicit-``null`` operand — null-or-missing match and its complement.

    ``{k: null}`` is an ACTIVE match (no drop-if-null): it keeps a present-JSON-null
    value AND an absent path (and a NULL system column). ``$ne null`` is present-AND-non-null
    (absent EXCLUDED). A negation over a non-null operand follows F1: ``$ne`` / ``$nin``
    over a NULL system column or an absent metadata path INCLUDES that row. ``$in``
    with a literal ``null`` member preserves it. Omitting a key entirely (``{}``) is
    "no filter" (all rows) — distinct from ``{k: null}`` (active null-or-missing).
    """
    md = (
        SeedRecord(id="nv-urgent", metadata={"tag": "urgent"}),
        SeedRecord(id="nv-okrs", metadata={"tag": "okrs"}),
        SeedRecord(id="nv-absent", metadata={}),
        SeedRecord(id="nv-jsonnull", metadata={"tag": None}),
    )
    sysn = (
        SeedRecord(id="sn-linear", source_name="linear"),
        SeedRecord(id="sn-slack", source_name="slack"),
        SeedRecord(id="sn-null"),  # NULL source_name
    )
    mx = (
        SeedRecord(id="mx-jsonnull", metadata={"x": None}),
        SeedRecord(id="mx-value", metadata={"x": "v"}),
        SeedRecord(id="mx-other", metadata={"x": "other"}),
        SeedRecord(id="mx-absent", metadata={}),
    )
    return [
        # metadata $ne scalar: Mongo missing-incl (absent + JSON-null + non-equal).
        _case(
            "F-NULLVAL-md-ne",
            {"metadata.tag": {"$ne": "urgent"}},
            md,
            frozenset({"nv-okrs", "nv-absent", "nv-jsonnull"}),
            ("F-NULLVAL", "metadata.tag", "$ne"),
        ),
        # metadata $nin: non-member + absent + JSON-null.
        _case(
            "F-NULLVAL-md-nin",
            {"metadata.tag": {"$nin": ["urgent", "blocker"]}},
            md,
            frozenset({"nv-okrs", "nv-absent", "nv-jsonnull"}),
            ("F-NULLVAL", "metadata.tag", "$nin"),
        ),
        # F1 system-col NULL incl under $ne / $nin.
        _case(
            "F-NULLVAL-sys-ne",
            {"source_name": {"$ne": "linear"}},
            sysn,
            frozenset({"sn-slack", "sn-null"}),
            ("F-NULLVAL", "source_name", "$ne"),
        ),
        _case(
            "F-NULLVAL-sys-nin",
            {"source_name": {"$nin": ["linear", "x"]}},
            sysn,
            frozenset({"sn-slack", "sn-null"}),
            ("F-NULLVAL", "source_name", "$nin"),
        ),
        # {x: null} ≡ {$eq: null} — null-or-missing (JSON-null + absent).
        _case(
            "F-NULLVAL-md-eq-null",
            {"metadata.x": {"$eq": None}},
            mx,
            frozenset({"mx-jsonnull", "mx-absent"}),
            ("F-NULLVAL", "metadata.x", "$eq-null"),
        ),
        _case(
            "F-NULLVAL-md-bare-null",
            {"metadata.x": None},
            mx,
            frozenset({"mx-jsonnull", "mx-absent"}),
            ("F-NULLVAL", "metadata.x", "bare-null"),
        ),
        # $in with a literal null member preserves it (JSON-null + value match).
        _case(
            "F-NULLVAL-md-in-null",
            {"metadata.x": {"$in": [None, "v"]}},
            mx,
            frozenset({"mx-jsonnull", "mx-value"}),
            ("F-NULLVAL", "metadata.x", "$in-null"),
        ),
        # $ne null: present-AND-non-null (absent EXCLUDED).
        _case(
            "F-NULLVAL-md-ne-null",
            {"metadata.x": {"$ne": None}},
            mx,
            frozenset({"mx-value", "mx-other"}),
            ("F-NULLVAL", "metadata.x", "$ne-null"),
        ),
        # omit = no filter (all rows) vs {k: null} = active (only NULL).
        _case("F-NULLVAL-omit-all", {}, sysn, frozenset({"sn-linear", "sn-slack", "sn-null"}), ("F-NULLVAL", "omit")),
        _case(
            "F-NULLVAL-sys-bare-null",
            {"source_name": None},
            sysn,
            frozenset({"sn-null"}),
            ("F-NULLVAL", "source_name", "bare-null"),
        ),
    ]


def f_objeq_cases() -> list[ConformanceCase]:
    """F-OBJEQ: whole-subdoc ``object_equal`` (``=``) + opaque-literal operand.

    A metadata sub-path dict operand is EXACT object equality, NOT ``@>``
    containment — a stored object with EXTRA keys does NOT survive. Equality is
    order-insensitive (reordered operand keys, recursively). An operand that itself
    LOOKS like an operator-expression (``{"$gt": 5}`` / ``{"$or": [1, 2]}``) is
    carried as an OPAQUE LITERAL — recursion stops at the operand, so it matches a
    stored value that equals that literal object, not a range/disjunction. The
    dot-path form (``metadata.labels.team``) descends and matches within the
    sub-object.
    """
    seed = (
        SeedRecord(id="oe-exact", metadata={"labels": {"team": "ingest"}}),
        SeedRecord(id="oe-exact-dup", metadata={"labels": {"team": "ingest"}}),
        SeedRecord(id="oe-two-key", metadata={"labels": {"team": "ingest", "tier": "gold"}}),
        SeedRecord(id="oe-two-key-rev", metadata={"labels": {"tier": "gold", "team": "ingest"}}),
        SeedRecord(id="oe-nested", metadata={"labels": {"a": {"x": 1, "y": 2}}}),
        SeedRecord(id="oe-nested-rev", metadata={"labels": {"a": {"y": 2, "x": 1}}}),
        SeedRecord(id="oe-superset", metadata={"labels": {"team": "ingest", "extra": 1}}),
        SeedRecord(id="oe-other", metadata={"labels": {"team": "other"}}),
        SeedRecord(id="oe-absent"),
    )
    lit = (
        SeedRecord(id="lit-gt", metadata={"x": {"$gt": 5}}),  # literal {$gt:5}
        SeedRecord(id="lit-num", metadata={"x": 7}),  # numeric 7 (would match a real range)
        SeedRecord(id="lit-or", metadata={"y": {"$or": [1, 2]}}),  # literal {$or:[1,2]}
    )
    return [
        # exact subdoc: extra-key / scalar / absent excluded.
        _case(
            "F-OBJEQ-exact",
            {"metadata.labels": {"team": "ingest"}},
            seed,
            frozenset({"oe-exact", "oe-exact-dup"}),
            ("F-OBJEQ", "metadata.labels", "exact"),
        ),
        # order-insensitive on operand keys (2-key).
        _case(
            "F-OBJEQ-reordered",
            {"metadata.labels": {"team": "ingest", "tier": "gold"}},
            seed,
            frozenset({"oe-two-key", "oe-two-key-rev"}),
            ("F-OBJEQ", "metadata.labels", "reordered"),
        ),
        # recursive order-insensitive (nested object).
        _case(
            "F-OBJEQ-nested-reordered",
            {"metadata.labels": {"a": {"x": 1, "y": 2}}},
            seed,
            frozenset({"oe-nested", "oe-nested-rev"}),
            ("F-OBJEQ", "metadata.labels", "nested"),
        ),
        # superset (extra key) must NOT match the exact form.
        _case(
            "F-OBJEQ-superset-excluded",
            {"metadata.labels": {"team": "ingest", "extra": 1}},
            seed,
            frozenset({"oe-superset"}),
            ("F-OBJEQ", "metadata.labels", "superset"),
        ),
        # opaque literal {$gt:5}: matches the literal, not the numeric 7.
        _case(
            "F-OBJEQ-literal-gt",
            {"metadata.x": {"$eq": {"$gt": 5}}},
            lit,
            frozenset({"lit-gt"}),
            ("F-OBJEQ", "metadata.x", "literal"),
        ),
        # opaque literal {$or:[1,2]}: recursion stops at the operand.
        _case(
            "F-OBJEQ-literal-or",
            {"metadata.y": {"$eq": {"$or": [1, 2]}}},
            lit,
            frozenset({"lit-or"}),
            ("F-OBJEQ", "metadata.y", "literal"),
        ),
        # $ne of the exact form: complement incl. absent (Rule 2).
        _case(
            "F-OBJEQ-ne",
            {"metadata.labels": {"$ne": {"team": "ingest"}}},
            seed,
            frozenset(
                {"oe-two-key", "oe-two-key-rev", "oe-nested", "oe-nested-rev", "oe-superset", "oe-other", "oe-absent"}
            ),
            ("F-OBJEQ", "metadata.labels", "$ne"),
        ),
        # dot-path descends and matches within the sub-object.
        _case(
            "F-OBJEQ-path",
            {"metadata.labels.team": "ingest"},
            seed,
            frozenset({"oe-exact", "oe-exact-dup", "oe-two-key", "oe-two-key-rev", "oe-superset"}),
            ("F-OBJEQ", "metadata.labels.team", "path"),
        ),
        ConformanceCase(
            id="F-OBJEQ-metadata-labels-in",
            # A dict $in element is EXACT object_equal per element, NOT @>
            # containment: only the record whose subdocument EQUALS the operand
            # survives — the superset (extra key) must NOT match.
            filter={"metadata.labels": {"$in": [{"team": "x"}]}},
            seed_records=seed,
            expected_ids=frozenset({"exact"}),
            backends=_OP_BACKENDS,
            exercises=("F-OBJEQ", "metadata.labels", "$in", "dict"),
        ),
        ConformanceCase(
            id="F-OBJEQ-metadata-labels-nin",
            # $nin negates the per-element exact form: the complement (superset,
            # other) plus the absent record survive (Rule 2 polarity — absent
            # satisfies $nin).
            filter={"metadata.labels": {"$nin": [{"team": "x"}]}},
            seed_records=seed,
            expected_ids=frozenset({"superset", "other", "absent"}),
            backends=_OP_BACKENDS,
            exercises=("F-OBJEQ", "metadata.labels", "$nin", "dict"),
        ),
    ]


def f_dotkey_cases() -> list[ConformanceCase]:
    """F-DOTKEY: dot-path descent vs literal-dotted / ``$``-prefixed / whole-blob.

    A folded ``metadata.a.b`` key DESCENDS into nested objects (a stored flat
    literal ``"a.b"`` key is unreachable by descent). The bare ``{"metadata": {...}}``
    form is whole-metadata-blob ``$eq`` equality (Mongo-A), so it matches the row
    whose ENTIRE metadata equals the operand — the flat-literal-key row, NOT the
    nested one. A ``$``-prefixed final segment is a valid descent key in the
    in-memory oracle (it resolves the literal ``"$ref"`` member). Deep descent walks
    every segment; nested presence requires every segment to resolve.
    """
    seed = (
        SeedRecord(id="dk-nested", metadata={"a": {"b": "v"}}),
        SeedRecord(id="dk-literal", metadata={"a.b": "v"}),  # flat literal dotted key
        SeedRecord(id="dk-deep", metadata={"a": {"b": {"c": {"d": 42}}}}),
        SeedRecord(id="dk-ref", metadata={"$ref": "x"}),  # $-prefixed member
    )
    return [
        # descent reaches the nested value, not the flat literal key.
        _case(
            "F-DOTKEY-descent",
            {"metadata.a.b": "v"},
            seed,
            frozenset({"dk-nested"}),
            ("F-DOTKEY", "metadata.a.b", "descent"),
        ),
        # bare {"metadata": {"a.b": "v"}} = whole-blob eq → the flat-literal-key row.
        _case(
            "F-DOTKEY-whole-blob",
            {"metadata": {"a.b": "v"}},
            seed,
            frozenset({"dk-literal"}),
            ("F-DOTKEY", "metadata", "whole-blob"),
        ),
        # $-prefixed final segment descends to the literal "$ref" member.
        _case(
            "F-DOTKEY-dollar-key",
            {"metadata.$ref": "x"},
            seed,
            frozenset({"dk-ref"}),
            ("F-DOTKEY", "metadata.$ref", "dollar"),
        ),
        # 4-segment deep descent.
        _case(
            "F-DOTKEY-deep",
            {"metadata.a.b.c.d": 42},
            seed,
            frozenset({"dk-deep"}),
            ("F-DOTKEY", "metadata.a.b.c.d", "deep"),
        ),
        # nested presence: every segment must resolve.
        _case(
            "F-DOTKEY-nested-exists",
            {"metadata.a.b": {"$exists": True}},
            seed,
            frozenset({"dk-nested", "dk-deep"}),
            ("F-DOTKEY", "metadata.a.b", "$exists"),
        ),
    ]


def f_sel_cases() -> list[ConformanceCase]:
    """F-SEL: selectivity / multi-record narrowing across mixed predicates.

    A larger seed where predicates narrow a five-record set: an impossible bound
    keeps nothing, an all-match keeps everything, a single equality keeps one, a
    date range pre-filters, a ``$in`` keeps a chosen subset, and a composite AND of
    a system key and a metadata range intersects to the overlap. (The harness asserts
    fixed survivor ids; the limit-clause selectivity cases — ``len`` assertions —
    are a recall-engine concern outside the compiled-predicate seam.)
    """
    seed = tuple(
        SeedRecord(
            id=f"sel-{i}",
            metadata={"n": i},
            source_type="library" if i <= 3 else "connection",
            occurred_at=datetime(2026, i, 1, tzinfo=UTC),
        )
        for i in range(1, 6)
    )
    return [
        _case("F-SEL-impossible", {"metadata.n": {"$gt": 100}}, seed, frozenset(), ("F-SEL", "impossible")),
        _case(
            "F-SEL-all-match",
            {"metadata.n": {"$exists": True}},
            seed,
            frozenset({"sel-1", "sel-2", "sel-3", "sel-4", "sel-5"}),
            ("F-SEL", "all-match"),
        ),
        _case("F-SEL-single", {"metadata.n": 3}, seed, frozenset({"sel-3"}), ("F-SEL", "single")),
        _case(
            "F-SEL-date-range",
            {"occurred_at": {"$gte": "2026-03-01T00:00:00Z"}},
            seed,
            frozenset({"sel-3", "sel-4", "sel-5"}),
            ("F-SEL", "date-range"),
        ),
        _case("F-SEL-in-multi", {"metadata.n": {"$in": [2, 4]}}, seed, frozenset({"sel-2", "sel-4"}), ("F-SEL", "$in")),
        # composite AND: source_type=library (sel-1..3) ∧ n>=2 → {sel-2, sel-3}.
        _case(
            "F-SEL-composite",
            {"$and": [{"source_type": "library"}, {"metadata.n": {"$gte": 2}}]},
            seed,
            frozenset({"sel-2", "sel-3"}),
            ("F-SEL", "composite"),
        ),
    ]


def f_unsup_cases() -> list[ConformanceCase]:
    """F-UNSUP: pure routing-equivalence — every backend returns the same rows, none raise.

    The undeclared-property and nested-path questions are routing concerns: a
    backend that cannot push a predicate down POST-FILTERS it (same rows, different
    path) rather than raising. ``expect_unsupported`` is empty corpus-wide; these
    cases confirm a metadata predicate, a projected-system-key predicate, and a
    deep nested path all resolve to identical survivor sets across the three
    executors (the routing-equivalence contract).
    """
    seed = (
        SeedRecord(id="us-meta", metadata={"undeclared": "v"}),
        SeedRecord(id="us-meta-other", metadata={"undeclared": "w"}),
        SeedRecord(id="us-sys", source_name="linear"),
        SeedRecord(id="us-nested", metadata={"a": {"b": {"c": "deep"}}}),
        SeedRecord(id="us-absent"),
    )
    return [
        # undeclared metadata property → post-filter, same rows.
        _case(
            "F-UNSUP-undeclared-meta",
            {"metadata.undeclared": "v"},
            seed,
            frozenset({"us-meta"}),
            ("F-UNSUP", "metadata.undeclared", "post-filter"),
        ),
        # projected system key → same rows on every route.
        _case(
            "F-UNSUP-projected-system",
            {"source_name": "linear"},
            seed,
            frozenset({"us-sys"}),
            ("F-UNSUP", "source_name", "post-filter"),
        ),
        # deep nested path → post-filter descent, same rows.
        _case(
            "F-UNSUP-nested-path",
            {"metadata.a.b.c": "deep"},
            seed,
            frozenset({"us-nested"}),
            ("F-UNSUP", "metadata.a.b.c", "post-filter"),
        ),
    ]


def f_impossible_cases() -> list[ConformanceCase]:
    """F-IMPOSSIBLE: §4 rule #3 — type-gated / constant-false predicates that keep nothing or are type-gated.

    A bare-list operand on a scalar system column lowers to ``$eq`` exact-array,
    which a scalar value can never equal — a constant-false predicate keeping
    nothing (the Postgres ``varchar = text[]`` hazard, expressed safely). A metadata
    range op type-gates: a numeric ``$gt`` excludes numeric-strings, an array value,
    and a bool value; a string ``$gt`` is lexicographic (VALID, not impossible); a
    ``$date`` op parses-or-excludes a non-timestamp. None of these raise — they
    type-gate and exclude.

    (The catalog's explicit-form variants — ``{"$eq": [list]}`` on a string key, a
    bare list on a date key, a numeric ``$in`` on a string column — fail VALIDATION
    pre-compile, so they live in the validator family, not here.)
    """
    # Bare-list-on-scalar seeds (one populated row; the predicate keeps nothing).
    s_name = (SeedRecord(id="im-name", source_name="linear"),)
    s_type = (SeedRecord(id="im-type", source_type="library"),)
    # Metadata type-gate seed.
    md = (
        SeedRecord(id="im-code-2027", metadata={"code": "2027"}),  # string > "2026" lexico
        SeedRecord(id="im-code-2024", metadata={"code": "2024"}),
        SeedRecord(id="im-score-num", metadata={"score": 30}),  # numeric > 25
        SeedRecord(id="im-score-str", metadata={"score": "30"}),  # numeric-string excluded
        SeedRecord(id="im-scores-array", metadata={"scores": [30]}),  # array excluded under range
        SeedRecord(id="im-scores-scalar", metadata={"scores": 30}),
        SeedRecord(id="im-due-ts", metadata={"due": "2026-06-01T00:00:00Z"}),
        SeedRecord(id="im-due-bad", metadata={"due": "xyz"}),  # unparseable
        SeedRecord(id="im-flag-bool", metadata={"flag": True}),  # bool excluded under numeric
        SeedRecord(id="im-flag-num", metadata={"flag": 10}),
    )
    return [
        # bare-list on a scalar column → exact-array $eq → constant-false.
        _case(
            "F-IMPOSSIBLE-name-barelist",
            {"source_name": ["linear", "slack"]},
            s_name,
            frozenset(),
            ("F-IMPOSSIBLE", "source_name", "barelist"),
        ),
        _case(
            "F-IMPOSSIBLE-type-barelist",
            {"source_type": ["library", "connection"]},
            s_type,
            frozenset(),
            ("F-IMPOSSIBLE", "source_type", "barelist"),
        ),
        # string $gt is lexicographic (VALID): keeps the lexicographically-greater row.
        _case(
            "F-IMPOSSIBLE-string-gt-valid",
            {"metadata.code": {"$gt": "2026"}},
            md,
            frozenset({"im-code-2027"}),
            ("F-IMPOSSIBLE", "metadata.code", "lexico"),
        ),
        # numeric $gt excludes numeric-strings (type-gate).
        _case(
            "F-IMPOSSIBLE-num-vs-string",
            {"metadata.score": {"$gt": 25}},
            md,
            frozenset({"im-score-num"}),
            ("F-IMPOSSIBLE", "metadata.score", "type-gate"),
        ),
        # range over an array value is excluded (#1); only the scalar survives.
        _case(
            "F-IMPOSSIBLE-range-array",
            {"metadata.scores": {"$gt": 25}},
            md,
            frozenset({"im-scores-scalar"}),
            ("F-IMPOSSIBLE", "metadata.scores", "array"),
        ),
        # $date op parses-or-excludes a non-timestamp value.
        _case(
            "F-IMPOSSIBLE-date-parse",
            {"metadata.due": {"$gte": {"$date": _DATE_LIT}}},
            md,
            frozenset({"im-due-ts"}),
            ("F-IMPOSSIBLE", "metadata.due", "$date"),
        ),
        # numeric $gt excludes a bool value (bool is not a number).
        _case(
            "F-IMPOSSIBLE-num-vs-bool",
            {"metadata.flag": {"$gt": 5}},
            md,
            frozenset({"im-flag-num"}),
            ("F-IMPOSSIBLE", "metadata.flag", "bool"),
        ),
    ]
