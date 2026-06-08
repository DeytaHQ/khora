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
    "f_op_cases",
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

    # Deterministic, per-case namespace_id (xdist-safe). create_namespace honors a
    # caller-supplied namespace_id (it passes the model straight to the relational
    # backend). Seed every row under ns.namespace_id — the stable external id the
    # recall read path scopes on.
    ns = await coord.create_namespace(
        MemoryNamespace(
            namespace_id=_case_namespace_id(case),
            metadata={"conformance_case": case.id},
        )
    )
    namespace_id = ns.namespace_id

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


# Fixed seed anchors for the date F-OP cases. ``_HIT`` is in range of every
# generated date bound, ``_MISS`` out of range; the third record carries no date
# at all (the absent-value record). UTC, matching the validator's normalization.
_DATE_HIT = datetime(2026, 6, 1, tzinfo=UTC)
_DATE_MISS = datetime(2020, 1, 1, tzinfo=UTC)
_DATE_BOUND = "2026-01-01T00:00:00Z"

# The string F-OP cases compare against this value; ``_STR_OTHER`` is a distinct
# value and the third record leaves the key unset (the absent-value record).
_STR_HIT = "match-me"
_STR_OTHER = "other-value"

# All three backends are oracle-comparable for the F-OP family on the embedded
# date/metadata surface; the catalog/CI tickets prune per-backend as needed (the
# eight denorm document keys are not carried on the legacy chronicle DTO, so a
# positive predicate on them is chronicle-empty — flagged via ``backends`` there).
_OP_BACKENDS: frozenset[str] = frozenset({"python", "postgres", "chronicle"})

# String-key F-OP cases run on python + chronicle only — NOT postgres. The
# postgres leg targets ``khora_chunks`` (the denormalized single-table target),
# but ``seed_case`` denormalizes only ``_DATE_KEYS`` onto the seeded Chunk — the
# core Chunk model carries none of the seven string document keys (only the
# skeleton DTO does), so a positive string predicate is postgres-empty until the
# seeder denormalizes the doc-keys onto the chunk row (a follow-up harness
# concern). Pruning postgres here keeps ``ConformanceCase.backends`` an honest
# single source of truth; the date-key F-OP cases keep postgres. python (the
# oracle) + chronicle still validate every string case.
_STRING_OP_BACKENDS: frozenset[str] = _OP_BACKENDS - frozenset({"postgres"})


def _date_field_seed(key: str) -> tuple[SeedRecord, SeedRecord, SeedRecord]:
    """Three records for a date-key F-OP case: in-range, out-of-range, absent.

    The date is stamped on ``key`` (one of the three date columns). All three
    share the default content so they share an embedding.
    """
    return (
        SeedRecord(id=f"{key}-hit", **{key: _DATE_HIT}),
        SeedRecord(id=f"{key}-miss", **{key: _DATE_MISS}),
        SeedRecord(id=f"{key}-absent"),
    )


def _string_field_seed(key: str) -> tuple[SeedRecord, SeedRecord, SeedRecord]:
    """Three records for a string-key F-OP case: matching, other, absent."""
    return (
        SeedRecord(id=f"{key}-hit", **{key: _STR_HIT}),
        SeedRecord(id=f"{key}-other", **{key: _STR_OTHER}),
        SeedRecord(id=f"{key}-absent"),
    )


def _date_op_cases(key: str) -> list[ConformanceCase]:
    """Every date-key F-OP case: range + set ops over one date key.

    ``expected_ids`` is computed by construction from the known three-record seed
    (hit @ 2026-06-01, miss @ 2020-01-01, absent). The bound is 2026-01-01, so the
    lower-bound ops (``$gt``/``$gte``) keep only the hit; the upper-bound ops keep
    only the miss; ``$eq`` against the bound keeps neither; ``$ne`` keeps the
    complement (negations include the absent record); ``$in`` keeps the hit, ``$nin``
    its complement.

    Date keys deliberately have **no** ``$exists`` case: ``DateOps`` carries no
    ``$exists`` field, so ``$exists`` on a date key is a *validation* failure (not
    a compile-time unsupported outcome) — the presence operator is exercised on the
    string keys instead. The operand is a plain ISO-8601 string (the system-key
    ``DateOps`` form), not the ``{"$date": ...}`` typed literal (that form is the
    metadata grammar's).
    """
    seed = _date_field_seed(key)
    hit, miss, absent = (r.id for r in seed)
    bound = _DATE_BOUND

    def case(suffix: str, predicate: dict[str, Any], expected: frozenset[str], op_tag: str) -> ConformanceCase:
        return ConformanceCase(
            id=f"F-OP-{key}-{suffix}",
            filter={key: predicate},
            seed_records=seed,
            expected_ids=expected,
            backends=_OP_BACKENDS,
            exercises=("F-OP", key, op_tag),
        )

    return [
        case("gt", {"$gt": bound}, frozenset({hit}), "$gt"),
        case("gte", {"$gte": bound}, frozenset({hit}), "$gte"),
        case("lt", {"$lt": bound}, frozenset({miss}), "$lt"),
        case("lte", {"$lte": bound}, frozenset({miss}), "$lte"),
        case("eq", {"$eq": bound}, frozenset(), "$eq"),
        case("ne", {"$ne": bound}, frozenset({hit, miss, absent}), "$ne"),
        case("in", {"$in": [bound]}, frozenset(), "$in"),
        case("nin", {"$nin": [bound]}, frozenset({hit, miss, absent}), "$nin"),
    ]


def _string_op_cases(key: str) -> list[ConformanceCase]:
    """Every string-key F-OP case: ``$eq``/``$ne``/``$in``/``$nin``/``$exists``.

    ``expected_ids`` is computed by construction from the known three-record seed
    (hit == ``match-me``, other == ``other-value``, absent == key unset). The eight
    denormalized document keys are always present columns at the SQL layer (NULL when
    unset), so ``$exists`` is trivially all-records — its coverage is the
    presence-operator tag, not a narrowing assertion.
    """
    seed = _string_field_seed(key)
    hit, other, absent = (r.id for r in seed)

    def case(suffix: str, predicate: dict[str, Any], expected: frozenset[str], op_tag: str) -> ConformanceCase:
        return ConformanceCase(
            id=f"F-OP-{key}-{suffix}",
            filter={key: predicate},
            seed_records=seed,
            expected_ids=expected,
            backends=_STRING_OP_BACKENDS,
            exercises=("F-OP", key, op_tag),
        )

    return [
        case("eq", {"$eq": _STR_HIT}, frozenset({hit}), "$eq"),
        case("ne", {"$ne": _STR_HIT}, frozenset({other, absent}), "$ne"),
        case("in", {"$in": [_STR_HIT]}, frozenset({hit}), "$in"),
        case("nin", {"$nin": [_STR_HIT]}, frozenset({other, absent}), "$nin"),
        case("exists-true", {"$exists": True}, frozenset({hit, other, absent}), "$exists"),
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
    (``occurred_at`` / ``created_at`` / ``source_timestamp``) across range / set /
    ``$exists`` ops, and the seven string keys (``source_type`` / ``source_name`` /
    ``source_url`` / ``external_id`` / ``content_type`` / ``source`` / ``title``)
    across ``$eq`` / ``$ne`` / ``$in`` / ``$nin`` / ``$exists`` — plus one
    representative metadata scalar. Every case tags the system key it exercises in
    ``exercises`` so the coverage meta-test can assert the union covers
    :data:`SYSTEM_KEYS`. ``expected_ids`` is computed by construction from each
    case's known seed (the runner confirms, never defines, them via the oracle).
    """
    cases: list[ConformanceCase] = []
    for key in _DATE_KEYS:
        cases.extend(_date_op_cases(key))
    for key in _DOC_STRING_KEYS:
        cases.extend(_string_op_cases(key))
    cases.append(_metadata_op_case())
    return cases


# --------------------------------------------------------------------------- #
# Case-family generator stubs (filled by the catalog ticket).
# --------------------------------------------------------------------------- #
#
# Each stub returns ``[]`` so the corpus assembler can already iterate every
# family; the catalog ticket replaces the body with hand-authored cases (filter +
# seed + expected_ids) per the family's intent described in each docstring.


def f_coerce_cases() -> list[ConformanceCase]:
    """F-COERCE: cross-type operand coercion (e.g. numeric string vs. number).

    Filled by the catalog ticket. Asserts the compilers agree on how an operand
    whose type differs from the stored value is compared (or excluded).
    """
    return []


def f_exists_cases() -> list[ConformanceCase]:
    """F-EXISTS: ``$exists`` presence semantics across absent / present-null / present.

    Filled by the catalog ticket. The absent-vs-present-null distinction on a
    metadata path is the load-bearing case.
    """
    return []


def f_logic_cases() -> list[ConformanceCase]:
    """F-LOGIC: ``$and`` / ``$or`` / ``$not`` / ``$nor`` composition and nesting.

    Filled by the catalog ticket. Includes de Morgan equivalences and the
    negation-includes-absent polarity rule under logical composition.
    """
    return []


def f_sugar_cases() -> list[ConformanceCase]:
    """F-SUGAR: bare-value / bare-list / implicit-AND desugaring equivalence.

    Filled by the catalog ticket. A bare scalar must match its ``$eq`` form; a
    bare list its ``$eq`` exact-array form (NOT ``$in``); sibling keys their
    explicit ``$and``.
    """
    return []


def f_dates_cases() -> list[ConformanceCase]:
    """F-DATES: ``$date`` typed-literal handling and timezone normalization.

    Filled by the catalog ticket. Covers naive-vs-aware operands and the
    cross-axis date-key pushdown rules (only ``source_timestamp`` pushes down).
    """
    return []


def f_nullval_cases() -> list[ConformanceCase]:
    """F-NULLVAL: explicit-``null`` operand (null-or-missing match) semantics.

    Filled by the catalog ticket. A ``{key: null}`` matches absent-or-present-null;
    a ``$ne null`` its complement.
    """
    return []


def f_sel_cases() -> list[ConformanceCase]:
    """F-SEL: selectivity / multi-record narrowing across mixed predicates.

    Filled by the catalog ticket. Larger seeds where several predicates intersect,
    stress-testing the conjunction narrowing.
    """
    return []


def f_objeq_cases() -> list[ConformanceCase]:
    """F-OBJEQ: whole-subdocument / whole-blob object equality (exact ``=``).

    Filled by the catalog ticket. A metadata sub-path dict operand is EXACT
    equality, NOT ``@>`` containment.
    """
    return []


def f_dotkey_cases() -> list[ConformanceCase]:
    """F-DOTKEY: folded ``metadata.<path>`` multi-segment path addressing.

    Filled by the catalog ticket. Nested path descent and array-aware containment
    on metadata sub-paths.
    """
    return []


def f_unsup_cases() -> list[ConformanceCase]:
    """F-UNSUP: predicates a backend cannot push down (``expect_unsupported``).

    Filled by the catalog ticket. Each case lists the backends that must raise
    :class:`RecallFilterUnsupportedError` under ``on_unsupported='raise'``.
    """
    return []
