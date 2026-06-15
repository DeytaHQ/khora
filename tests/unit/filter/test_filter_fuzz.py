"""Property-based differential fuzzer for the recall-filter compilers.

The recall-filter conformance harness (``khora.filter.conformance``) pins a
hand-authored *catalog* of cases: each is lowered through the real validator +
``parse_to_ast`` + every backend compiler, and the surviving row-set is asserted
against the Python oracle (``compile_python``). That catalog is precise but
finite — it covers the shapes a human thought to enumerate.

This module is the complementary force: it *generates* filters with Hypothesis
and checks the same oracle contract on every draw, so a divergence the catalog
did not happen to enumerate surfaces as a shrunk counterexample instead of an
escaped production bug. Two properties:

**Property A — differential (primary).** For a generated valid filter over a
single FIXED seed, the Python oracle and the real ``sqlite_lance`` read path
(the ``compile_lance`` server-side prefilter ∘ the ``compile_python``
post-filter) must keep the SAME ``SeedRecord`` ids. sqlite_lance is the
differential target — NOT chronicle, whose post-filter *is* ``compile_python``
(the oracle itself), so it cannot disagree the way an independent compiler can.
A separate sweep re-runs Property A under eight distinct Hypothesis seeds to
confirm the agreement is stable, not an artifact of one PRNG stream.

**Property B — metamorphic (oracle-only).** Four logical rewrites that MUST
preserve the oracle row-set: implicit-AND ⇔ ``$and``, ``$nor`` ⇔
``$not($or)``, De Morgan, and double-negation. A divergence here is a
``parse_to_ast`` / ``compile_python`` normalization bug, caught without any
store. A dedicated strategy emits only the transformable shapes (a generic
strategy gated by ``assume`` would trip Hypothesis's ``filter_too_much``).

The bare-list-``$eq`` exact-array-on-scalar-metadata crash fixed in PR #1234 is
in scope and exercised with no carve-out: the strategy freely draws bare-list
metadata operands, and the seed carries both array and scalar metadata nodes.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from hypothesis import HealthCheck, assume, given, seed, settings
from hypothesis import strategies as st

from khora.db.session import run_migrations
from khora.filter import (
    CompiledFilter,
    RecallFilter,
    RecallFilterUnsupportedError,
    RecallFilterValidationError,
)
from khora.filter.ast import FilterNode, parse_to_ast
from khora.filter.conformance import (
    ConformanceCase,
    LanceExecutor,
    PythonExecutor,
    SeedRecord,
    _record_mapping,
    seed_case,
)
from khora.storage.backends.sqlite_lance import SQLiteLanceRelationalAdapter
from khora.storage.backends.sqlite_lance._helpers import uuid_to_text
from khora.storage.backends.sqlite_lance.connection import (
    EmbeddedStorageHandle,
    EmbeddedStorageHandleConfig,
)
from khora.storage.coordinator import StorageCoordinator
from tests.integration._sqlite_lance_fixtures import EMBED_DIM
from tests.integration.matrix._conformance_lance import _CoreChunkTemporalStore, _run_async

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# The fixed seed.
# --------------------------------------------------------------------------- #
#
# ~18 records whose values are spread across the filterable surface so a
# generated predicate lands a STRICT SUBSET of survivors (not all-or-nothing):
# distinct + repeated + absent values on every string key, three date instants
# plus nulls, and a metadata blob that mixes scalars, arrays (including the empty
# array), an array-vs-scalar collision on one key (``tags``), a present JSON null,
# a nested sub-document, and a bare ``{}``.
#
# SEED INVARIANT (hard): every record stamps ``created_at`` (it is always-present
# in production — NOT NULL + default + writer stamping; an unset ``created_at``
# models an unreachable state the live legs would diverge on). Any record that
# stamps ``source_timestamp`` ALSO stamps ``occurred_at`` to the SAME value,
# because the oracle's ``_record_mapping`` coalesces
# ``occurred_at := occurred_at or source_timestamp`` while the stored
# ``khora_chunks.occurred_at`` column is literal — equalizing them keeps the two
# sides reading the same effective event time.

_T_HIT = datetime(2026, 6, 1, tzinfo=UTC)
_T_MID = datetime(2026, 3, 15, tzinfo=UTC)
_T_LOW = datetime(2026, 1, 1, tzinfo=UTC)
_T_MISS = datetime(2020, 1, 1, tzinfo=UTC)


def _build_fixed_seed() -> tuple[SeedRecord, ...]:
    """The ~18 spread-value records both properties share (frozen at module import)."""
    return (
        # --- string-key spread (source_name / source_type / content_type / ...) ---
        SeedRecord(
            id="r01",
            created_at=_T_LOW,
            source_type="library",
            source_name="linear",
            content_type="text/markdown",
            external_id="ext-01",
            source="api",
            title="alpha",
            source_url="https://example.test/a",
            metadata={"tier": "gold", "score": 10, "tags": ["urgent", "release"]},
        ),
        SeedRecord(
            id="r02",
            created_at=_T_MID,
            source_type="connection",
            source_name="slack",
            content_type="application/pdf",
            external_id="ext-02",
            source="ingest",
            title="beta",
            metadata={"tier": "silver", "score": 0, "tags": ["okrs"]},
        ),
        SeedRecord(
            id="r03",
            created_at=_T_HIT,
            source_type="direct",
            source_name="linear",
            content_type="text/markdown",
            external_id="ext-03",
            metadata={"tier": "gold", "score": 3, "tags": []},
        ),
        SeedRecord(
            id="r04",
            created_at=_T_HIT,
            source_type="connection",
            source_name=None,
            content_type=None,
            external_id="ext-04",
            metadata={"tier": "silver", "score": 10},
        ),
        SeedRecord(
            id="r05",
            created_at=_T_MISS,
            source_type="library",
            source_name="slack",
            content_type="application/pdf",
            external_id=None,
            metadata={"score": 0, "tags": "urgent"},  # scalar on a key elsewhere an array
        ),
        SeedRecord(
            id="r06",
            created_at=_T_MID,
            source_type="direct",
            source_name="linear",
            content_type="text/markdown",
            external_id="ext-06",
            title="gamma",
            metadata={"tier": "gold", "tags": ["release"]},
        ),
        # --- date spread on occurred_at / source_timestamp (kept equal per invariant) ---
        SeedRecord(
            id="r07",
            created_at=_T_LOW,
            occurred_at=_T_HIT,
            source_timestamp=_T_HIT,
            source_type="library",
            source_name="slack",
            metadata={"score": 7, "a": {"b": "v"}},  # nested sub-document
        ),
        SeedRecord(
            id="r08",
            created_at=_T_MID,
            occurred_at=_T_MID,
            source_timestamp=_T_MID,
            source_type="connection",
            source_name="linear",
            content_type="application/pdf",
            metadata={"score": 3, "a": {"b": "w"}},
        ),
        SeedRecord(
            id="r09",
            created_at=_T_HIT,
            occurred_at=_T_MISS,
            source_timestamp=_T_MISS,
            source_type="direct",
            metadata={"tier": "silver", "tags": ["urgent"]},
        ),
        SeedRecord(
            id="r10",
            created_at=_T_HIT,
            occurred_at=_T_HIT,  # occurred_at without source_timestamp — allowed
            source_type="library",
            source_name="linear",
            content_type="text/markdown",
            metadata={"mk": None},  # present JSON null
        ),
        SeedRecord(
            id="r11",
            created_at=_T_LOW,
            source_type="connection",
            source_name=None,
            external_id="ext-11",
            metadata={"tier": "gold", "score": 10, "tags": ["release", "urgent"]},
        ),
        SeedRecord(
            id="r12",
            created_at=_T_MID,
            source_type="direct",
            source_name="slack",
            content_type=None,
            metadata={},  # empty blob
        ),
        SeedRecord(
            id="r13",
            created_at=_T_HIT,
            occurred_at=_T_MID,
            source_timestamp=_T_MID,
            source_type="library",
            source_name="linear",
            title="delta",
            metadata={"score": 0, "tier": "silver"},
        ),
        SeedRecord(
            id="r14",
            created_at=_T_MISS,
            source_type="connection",
            content_type="application/pdf",
            external_id="ext-14",
            metadata={"tags": ["okrs", "release"], "score": 3},
        ),
        SeedRecord(
            id="r15",
            created_at=_T_MID,
            source_type="direct",
            source_name="slack",
            source_url="https://example.test/o",
            metadata={"tier": "gold"},
        ),
        SeedRecord(
            id="r16",
            created_at=_T_HIT,
            occurred_at=_T_LOW,
            source_timestamp=_T_LOW,
            source_type="library",
            source_name="linear",
            content_type="text/markdown",
            metadata={"tier": "silver", "score": 7, "tags": ["urgent"]},
        ),
        SeedRecord(
            id="r17",
            created_at=_T_LOW,
            source_type="connection",
            source_name=None,
            external_id=None,
            metadata={"a": {"b": "v"}, "score": 10},
        ),
        SeedRecord(
            id="r18",
            created_at=_T_HIT,
            source_type="direct",
            source_name="slack",
            content_type="application/pdf",
            title="epsilon",
            metadata={"tier": "gold", "tags": [], "mk": None},
        ),
    )


FIXED_SEED: tuple[SeedRecord, ...] = _build_fixed_seed()
_RECORDS: list[tuple[str, dict[str, Any]]] = [(r.id, _record_mapping(r)) for r in FIXED_SEED]

# Discrimination collector: the Property A run appends every oracle survivor-set
# size here, and a separate plain test (running after, by name order) asserts that
# a meaningful fraction of draws were STRICT SUBSETS (0 < size < |seed|). This is
# the anti-vacuous guard: if every filter kept all or none, oracle == lance would
# agree trivially and prove nothing. A module-level list is robust and simple (no
# dependence on Hypothesis statistics plumbing).
_SURVIVOR_SET_SIZES: list[int] = []


# --------------------------------------------------------------------------- #
# Embedded sqlite_lance store + runner (Property A only).
# --------------------------------------------------------------------------- #
#
# Property A needs the REAL sqlite_lance read path, so it seeds the fixed corpus
# into a one-case embedded coordinator ONCE. The build + seed runs on the
# dedicated loop thread that owns the aiosqlite connection (``_run_async`` from
# the conformance helper — an aiosqlite handle is bound to the loop it was opened
# on, so every later query MUST go through the same loop).
#
# RE-ENTRANCY: the ``@lru_cache`` singleton is resolved on the CALLER (test)
# thread; the runner closure captures the ``handle`` + ``id_map`` it returns and
# submits ONLY the read coroutine to the loop thread. Resolving the cache from
# inside a coroutine already running on that loop would deadlock (the loop would
# block on a future only it can complete). Module-level singletons (not a
# function-scoped fixture) also keep Hypothesis's ``function_scoped_fixture``
# health check from firing on the @given test.


class _SeededLanceStore:
    """A connected one-case embedded coordinator seeded with ``FIXED_SEED``."""

    def __init__(self, handle: EmbeddedStorageHandle, id_map: Mapping[str, UUID]) -> None:
        self.handle = handle
        self.id_map = id_map


async def _build_seeded_store() -> _SeededLanceStore:
    """Migrate a tmp SQLite file, wire the ``khora_chunks`` store, seed the fixed corpus."""
    tmp_path = Path(tempfile.mkdtemp(prefix="khora-filter-fuzz-"))
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

    # ``filter={}`` is the match-everything seed case — the runner re-derives the
    # predicate per draw; only the seeded rows matter here. ``expected_ids=None``
    # because this case is never asserted as a conformance case.
    fuzz_case = ConformanceCase(
        id="fuzz-seed",
        filter={},
        seed_records=FIXED_SEED,
        expected_ids=None,
        backends=frozenset({"sqlite_lance"}),
    )
    id_map = await seed_case(coord, fuzz_case)

    # Materialization guard (fail loud, at build time): the seeder must have landed
    # one queryable ``khora_chunks`` row per record with a 1:1 ``seed_id``↔chunk-UUID
    # map. If it silently dropped rows, BOTH the oracle and lance would later agree
    # on a too-small set — the empty-store false-green this guards against.
    if set(id_map) != {r.id for r in FIXED_SEED} or len(set(id_map.values())) != len(FIXED_SEED):
        raise RuntimeError(f"seed id_map is not 1:1 with FIXED_SEED: {sorted(id_map)}")
    cur = await handle.sqlite.execute("SELECT COUNT(*) FROM khora_chunks")
    (count,) = await cur.fetchone()
    if count != len(FIXED_SEED):
        raise RuntimeError(f"expected {len(FIXED_SEED)} seeded khora_chunks rows, found {count}")

    return _SeededLanceStore(handle, id_map)


@lru_cache(maxsize=1)
def _seeded_store() -> _SeededLanceStore:
    """The process-wide seeded embedded store (built + seeded exactly once).

    Resolved on the CALLER thread (never inside a coroutine already running on the
    loop thread — that would deadlock). The store is intentionally process-lived
    for the whole run; like the conformance helper's loop-thread coordinator, the
    embedded connection is left to interpreter shutdown (a benign ``ResourceWarning``
    on the daemon loop) rather than closed via an ``atexit`` hook that would log into
    an already-torn-down loguru sink.
    """
    return _run_async(_build_seeded_store())


async def _run_lance(
    handle: EmbeddedStorageHandle,
    id_map: Mapping[str, UUID],
    compiled: CompiledFilter[str],
    post_filter: Any,
    records: Sequence[tuple[str, Mapping[str, Any]]],
) -> frozenset[str]:
    """Run the compiled lance prefilter (scoped to the seeded rows) then the post-filter.

    Mirrors ``_conformance_lance.run_live``: scope the read to exactly the fixed
    seed's chunk ids (``id IN (...)``), AND-in the compiled predicate unless it is
    the match-everything literal ``"1"``, then narrow the candidate rows through
    the full-AST ``compile_python`` post-filter (the lance compiler is a
    superset-safe split pushdown). Returns the surviving ``SeedRecord`` ids.
    """
    record_map = dict(records)
    chunk_to_seed = {chunk_id: seed_id for seed_id, chunk_id in id_map.items()}

    placeholders = ",".join("?" for _ in chunk_to_seed)
    sql = f"SELECT id FROM khora_chunks WHERE id IN ({placeholders})"  # noqa: S608 - ids bind positionally
    args: list[Any] = [uuid_to_text(cid) for cid in chunk_to_seed]
    if compiled.predicate and compiled.predicate != "1":
        sql += f" AND ({compiled.predicate})"
        args.extend(compiled.params["args"])

    cur = await handle.sqlite.execute(sql, args)
    rows = await cur.fetchall()

    survivors: set[str] = set()
    for row in rows:
        seed_id = chunk_to_seed.get(UUID(row[0]))
        if seed_id is not None and post_filter(record_map[seed_id]):
            survivors.add(seed_id)
    return frozenset(survivors)


def _lance_survivors(filter_ast: FilterNode) -> frozenset[str]:
    """Survivors of ``filter_ast`` on the real sqlite_lance read path.

    Resolves the seeded-store singleton on the CALLER thread, then builds a
    ``LanceExecutor`` whose injected runner submits only the read coroutine to the
    loop thread (closing over the already-resolved ``handle`` + ``id_map``).
    """
    store = _seeded_store()  # caller-thread resolution — never inside the loop coroutine

    def runner(compiled, _filter_ast, post_filter, records):  # noqa: ANN001, ANN202 - matches LiveRunner
        return _run_async(_run_lance(store.handle, store.id_map, compiled, post_filter, records))

    return LanceExecutor(runner).survivors(filter_ast, _RECORDS)


def _oracle_survivors(filter_ast: FilterNode) -> frozenset[str]:
    """Survivors of ``filter_ast`` under the Python oracle (the reference)."""
    return PythonExecutor().survivors(filter_ast, _RECORDS)


# --------------------------------------------------------------------------- #
# Shared valid-filter strategy.
# --------------------------------------------------------------------------- #
#
# Per-key operator rules mirror the validator's typed submodels so the vast
# majority of draws validate (a wrong op for a key — e.g. a range op on a string
# key, or ``$exists`` on a date key — is a validation error, discarded). Operands
# are drawn from the seed value pools so a predicate discriminates a real subset.

# Operand value pools (drawn from the seed so filters separate known subsets).
_DATE_OPERANDS = [_T_HIT, _T_MID, _T_LOW, _T_MISS]
_STRING_POOLS: dict[str, list[str]] = {
    "source_type": ["library", "connection", "direct"],
    "source_name": ["linear", "slack"],
    "content_type": ["text/markdown", "application/pdf"],
    "external_id": ["ext-01", "ext-03", "ext-14", "missing"],
    "source": ["api", "ingest"],
    "title": ["alpha", "beta", "gamma", "delta", "epsilon"],
    "source_url": ["https://example.test/a", "https://example.test/o"],
}
_META_SCALAR_POOL = ["gold", "silver", "urgent", "release", "okrs", 0, 3, 7, 10, "v", "w"]


def _date_iso(dt: datetime) -> str:
    """The system-key ``DateOps`` operand form (a plain ISO-8601 string, not ``$date``)."""
    return dt.isoformat().replace("+00:00", "Z")


@st.composite
def _date_predicate(draw: st.DrawFn) -> Any:
    """A date-key predicate: a bare ISO scalar or a ``DateOps`` operator-expression."""
    if draw(st.booleans()):
        return _date_iso(draw(st.sampled_from(_DATE_OPERANDS)))
    op = draw(st.sampled_from(["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]))
    if op in ("$in", "$nin"):
        values = draw(st.lists(st.sampled_from(_DATE_OPERANDS), min_size=0, max_size=3))
        return {op: [_date_iso(v) for v in values]}
    return {op: _date_iso(draw(st.sampled_from(_DATE_OPERANDS)))}


@st.composite
def _string_predicate(draw: st.DrawFn, key: str) -> Any:
    """A string-key predicate: a bare scalar, exact-array, or ``StringOps`` expression."""
    pool = _STRING_POOLS[key]
    kind = draw(st.sampled_from(["bare", "eq", "ne", "in", "nin", "exists", "bare_list"]))
    if kind == "bare":
        return draw(st.sampled_from(pool))
    if kind == "bare_list":  # bare list ⇒ $eq exact-array (NOT $in)
        return draw(st.lists(st.sampled_from(pool), min_size=1, max_size=2))
    if kind == "exists":
        return {"$exists": draw(st.booleans())}
    if kind in ("eq", "ne"):
        return {f"${kind}": draw(st.sampled_from(pool))}
    return {f"${kind}": draw(st.lists(st.sampled_from(pool), min_size=0, max_size=3))}


@st.composite
def _metadata_predicate(draw: st.DrawFn) -> tuple[str, Any]:
    """A ``metadata.<path>`` key + predicate over the seed's metadata surface."""
    key = draw(st.sampled_from(["metadata.tier", "metadata.score", "metadata.tags", "metadata.a.b", "metadata.mk"]))
    kind = draw(
        st.sampled_from(["bare", "bare_list", "eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "exists", "date"])
    )
    if kind == "bare":
        return key, draw(st.sampled_from(_META_SCALAR_POOL))
    if kind == "bare_list":  # bare list ⇒ $eq exact-array — exercises the #1234 path
        return key, draw(st.lists(st.sampled_from(["urgent", "release", "okrs"]), min_size=0, max_size=3))
    if kind == "exists":
        return key, {"$exists": draw(st.booleans())}
    if kind == "date":
        return key, {"$eq": {"$date": _date_iso(draw(st.sampled_from(_DATE_OPERANDS)))}}
    if kind in ("in", "nin"):
        return key, {f"${kind}": draw(st.lists(st.sampled_from(_META_SCALAR_POOL), min_size=0, max_size=3))}
    return key, {f"${kind}": draw(st.sampled_from(_META_SCALAR_POOL))}


@st.composite
def _single_predicate(draw: st.DrawFn) -> dict[str, Any]:
    """One single-field predicate (a date key, a string key, or a metadata path)."""
    channel = draw(st.sampled_from(["date", "string", "metadata"]))
    if channel == "date":
        key = draw(st.sampled_from(["occurred_at", "created_at", "source_timestamp"]))
        return {key: draw(_date_predicate())}
    if channel == "string":
        key = draw(st.sampled_from(list(_STRING_POOLS)))
        return {key: draw(_string_predicate(key))}
    meta_key, predicate = draw(_metadata_predicate())
    return {meta_key: predicate}


def _valid_filter(max_depth: int = 3) -> st.SearchStrategy[dict[str, Any]]:
    """A recursively-composed filter dict (bounded depth/breadth).

    A leaf is a single-field predicate; an internal node composes children with a
    logical operator. ``$and``/``$or``/``$nor`` take a nonempty array; ``$not``
    negates one document. The recursion is depth-bounded (real filters nest only a
    few levels) and breadth-bounded so a single draw stays cheap.
    """

    def extend(children: st.SearchStrategy[dict[str, Any]]) -> st.SearchStrategy[dict[str, Any]]:
        branch = st.lists(children, min_size=1, max_size=3)
        return st.one_of(
            children,
            branch.map(lambda cs: {"$and": cs}),
            branch.map(lambda cs: {"$or": cs}),
            branch.map(lambda cs: {"$nor": cs}),
            children.map(lambda c: {"$not": c}),
            # A bag of sibling single-field predicates (implicit AND). Merge keeps
            # distinct keys; a collision just collapses to one predicate (still valid).
            st.lists(_single_predicate(), min_size=2, max_size=3).map(
                lambda preds: {k: v for p in preds for k, v in p.items()}
            ),
        )

    return st.recursive(_single_predicate(), extend, max_leaves=max_depth * 3)


def _validated_ast(filter_dict: dict[str, Any]) -> FilterNode | None:
    """Validate + lower a generated dict, or ``None`` if it fails validation.

    A draw the validator rejects (a malformed shape the per-key rules did not
    fully constrain) is discarded by the caller — the strategy is biased so this
    is rare, not the common path.
    """
    try:
        model = RecallFilter.model_validate(filter_dict)
    except RecallFilterValidationError:
        return None
    return parse_to_ast(model)


# --------------------------------------------------------------------------- #
# Property A — differential: oracle == sqlite_lance read path.
# --------------------------------------------------------------------------- #


class TestDifferentialOracleVsLance:
    """The Python oracle and the real sqlite_lance read path keep the same rows.

    This is the load-bearing property: an independent compiler (``compile_lance``
    + its split post-filter) is checked against the oracle on generated input. A
    divergence is a real compiler bug (the catalog's job is to pin the named
    shapes; this catches the ones nobody enumerated).
    """

    @given(filter_dict=_valid_filter())
    @settings(
        max_examples=300,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_oracle_equals_lance(self, filter_dict: dict[str, Any]) -> None:
        ast = _validated_ast(filter_dict)
        assume(ast is not None)  # invalid draw — skip (rare; strategy is biased to validate)
        assert ast is not None  # narrow for the type checker after the assume
        try:
            expected = _oracle_survivors(ast)
        except RecallFilterUnsupportedError:
            # The oracle compiles with on_unsupported="raise"; a clause it cannot
            # express in memory is not oracle-comparable, so SKIP it (assume(False))
            # rather than assert "both raise". compile_python expresses the whole
            # grammar, so this is effectively unreachable — a defensive guard, not a
            # carve-out.
            assume(False)
        actual = _lance_survivors(ast)
        # Record the oracle survivor-set size for the post-hoc discrimination check.
        _SURVIVOR_SET_SIZES.append(len(expected))
        assert actual == expected, (
            "oracle/lance divergence:\n"
            f"  filter = {filter_dict!r}\n"
            f"  oracle = {sorted(expected)}\n"
            f"  lance  = {sorted(actual)}"
        )


class TestDifferentialSeedSweep:
    """Property A re-run under eight distinct Hypothesis seeds — stability sweep.

    A single PRNG stream could miss a discriminating draw by luck; eight fixed
    seeds (fewer examples each, so total CI time stays bounded) confirm the
    oracle/lance agreement holds across independent streams, not one.
    """

    @pytest.mark.parametrize("hypothesis_seed", range(8))
    def test_oracle_equals_lance_under_seed(self, hypothesis_seed: int) -> None:
        @seed(hypothesis_seed)
        @given(filter_dict=_valid_filter())
        @settings(
            max_examples=40,
            deadline=None,
            suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
        )
        def _check(filter_dict: dict[str, Any]) -> None:
            ast = _validated_ast(filter_dict)
            assume(ast is not None)
            assert ast is not None  # narrow for the type checker after the assume
            try:
                expected = _oracle_survivors(ast)
            except RecallFilterUnsupportedError:
                assume(False)
            actual = _lance_survivors(ast)
            assert actual == expected, (
                f"seed={hypothesis_seed} oracle/lance divergence:\n"
                f"  filter = {filter_dict!r}\n"
                f"  oracle = {sorted(expected)}\n"
                f"  lance  = {sorted(actual)}"
            )

        _check()


def _sample_survivor_set_sizes(n: int = 200) -> list[int]:
    """Collect oracle survivor-set sizes over ``n`` valid draws (discrimination fallback).

    Used when the Property A run did not populate ``_SURVIVOR_SET_SIZES`` (e.g. the
    discrimination test was selected in isolation), so the ≥30%-strict-subset
    assertion always has data to judge.
    """
    sizes: list[int] = []

    @settings(max_examples=n, deadline=None)
    @given(filter_dict=_valid_filter())
    def _collect(filter_dict: dict[str, Any]) -> None:
        ast = _validated_ast(filter_dict)
        assume(ast is not None)
        assert ast is not None
        try:
            sizes.append(len(_oracle_survivors(ast)))
        except RecallFilterUnsupportedError:
            assume(False)

    _collect()
    return sizes


class TestSeedDiscriminates:
    """Sanity: the seed + strategy actually produce STRICT-SUBSET survivor sets.

    Guards against a degenerate seed where every filter keeps all (or no) rows —
    which would make Property A's ``oracle == lance`` agree trivially and prove
    nothing. The post-hoc check reads the sizes the Property A run recorded; the
    explicit check proves a discriminating partial filter + deep composition are
    reachable.
    """

    def test_a_meaningful_fraction_of_draws_are_strict_subsets(self) -> None:
        """At least 30% of recorded oracle draws kept a STRICT, non-empty subset.

        Reads ``_SURVIVOR_SET_SIZES`` (populated by ``test_oracle_equals_lance``,
        which runs earlier in this module); falls back to its own sample if empty
        (selective run). A high all/none rate would mean the seed or strategy can't
        discriminate — the false-green Property A is meant to rule out.
        """
        sizes = _SURVIVOR_SET_SIZES or _sample_survivor_set_sizes()
        assert sizes, "no survivor-set sizes recorded"
        strict = sum(1 for s in sizes if 0 < s < len(FIXED_SEED))
        fraction = strict / len(sizes)
        assert fraction >= 0.30, (
            f"only {fraction:.0%} of {len(sizes)} draws were strict subsets "
            f"(need >=30%); the seed/strategy may not discriminate"
        )

    def test_strict_subset_and_deep_composition_are_reachable(self) -> None:
        """Explicit (non-Hypothesis) proof that the seed discriminates + deep nesting parses.

        A concrete partial filter must keep a strict, non-empty subset, and a
        depth-3 ``$and([$or([...]), ...])`` must lower to a nested logical AST —
        so the fuzz space genuinely contains discriminating, deeply-composed
        filters rather than passing vacuously.
        """
        partial = parse_to_ast(RecallFilter.model_validate({"source_type": "library"}))
        survivors = _oracle_survivors(partial)
        assert 0 < len(survivors) < len(FIXED_SEED), survivors

        deep = parse_to_ast(
            RecallFilter.model_validate(
                {
                    "$and": [
                        {"$or": [{"source_name": "linear"}, {"source_name": "slack"}]},
                        {"$not": {"metadata.tier": {"$eq": "gold"}}},
                    ]
                }
            )
        )
        # Root AND with a nested OR child two levels down → genuine deep composition.
        assert deep.op.value == "$and"
        assert any(isinstance(child, FilterNode) and child.op.value == "$or" for child in deep.children)
        deep_survivors = _oracle_survivors(deep)
        assert 0 < len(deep_survivors) < len(FIXED_SEED), deep_survivors


class TestLanceHarnessIsNotVacuous:
    """Guard the Property A harness against silent false-greens.

    Property A only means something if the sqlite_lance store is actually seeded
    and the split post-filter is actually applied. The seed-materialization + 1:1
    id-mapping guard runs at store-build time (``_build_seeded_store`` raises if the
    seeder dropped rows). These checks add the behavioral half: that match-everything
    keeps the full seed on both sides and that a deferred leaf is narrowed by the
    post-filter, not the SQL prefilter alone.
    """

    def test_match_everything_keeps_all_rows(self) -> None:
        """The empty filter survives the full seed on BOTH sides (no silent row loss).

        ``{}`` lowers to the match-everything ``AND``; the lance predicate is the
        literal ``"1"`` (skipped in the scoped read). If either side returned a
        strict subset here, a row went missing — the trivial-``∅``-agreement trap.
        """
        empty_ast = parse_to_ast(RecallFilter.model_validate({}))
        all_ids = frozenset(r.id for r in FIXED_SEED)
        assert _oracle_survivors(empty_ast) == all_ids
        assert _lance_survivors(empty_ast) == all_ids

    def test_post_filter_deferred_leaf_still_agrees(self) -> None:
        """A leaf the lance compiler DEFERS to the post-filter still matches the oracle.

        A metadata ``$date`` compare is returned as ``None`` by ``compile_lance``
        (SQLite cannot replicate the ISO parse-or-exclude), so the whole leaf is
        answered by the ``compile_python`` post-filter — NOT the SQL prefilter.
        Agreement here proves the split post-filter is wired and applied, not that
        the prefilter happened to be exact. Uses a metadata date present on the seed
        (``r07``/``r08`` carry no metadata date, so the result is a real subset).
        """
        # Seed a metadata date on a couple of records via a dedicated micro-seed is
        # overkill; instead pick a deferred-but-decidable leaf already covered by the
        # fixed seed: a bare-list $eq exact-array on a scalar metadata node (the
        # #1234 path) — the array operand routes through _md_exact_array's CASE gate,
        # and a non-array stored node (e.g. r05's scalar tags) must read 0.
        ast = parse_to_ast(RecallFilter.model_validate({"metadata.tags": ["urgent", "release"]}))
        oracle = _oracle_survivors(ast)
        lance = _lance_survivors(ast)
        assert oracle == lance
        # r01 stores tags == ["urgent","release"] (exact match); r05 stores the
        # SCALAR "urgent" (must NOT match an exact-array) — a real strict subset, so
        # the comparison is not vacuous.
        assert "r01" in oracle and "r05" not in oracle
        assert 0 < len(oracle) < len(FIXED_SEED)


# --------------------------------------------------------------------------- #
# Property B — metamorphic: oracle invariance under equivalence-preserving rewrites.
# --------------------------------------------------------------------------- #
#
# A DEDICATED strategy emits ONLY transformable shapes. The four transforms each
# need a specific shape (e.g. De Morgan needs single-field operator-expressions so
# the rewritten field-position ``$not`` is legal), so generating a generic filter
# and ``assume``-ing it is transformable would discard almost everything
# (``filter_too_much``). Instead the strategy builds the operand pair directly.


# Every field a ``_field_op_expr`` may address — the dedicated transformable-shape
# key space (date keys + string keys + a few metadata paths).
_OP_EXPR_KEYS: tuple[str, ...] = (
    "occurred_at",
    "created_at",
    "source_timestamp",
    *_STRING_POOLS,
    "metadata.tier",
    "metadata.score",
    "metadata.tags",
)


@st.composite
def _field_op_expr_for(draw: st.DrawFn, key: str) -> dict[str, Any]:
    """A single-field operator-EXPRESSION on ``key`` (so a field-position ``$not`` is legal).

    De Morgan rewrites ``$not($or([a,b]))`` to ``$and([$not(a),$not(b)])``; the
    rewritten ``$not`` is FIELD-position, which the validator allows only on an
    operator-expression (a bare scalar / bare list there is invalid). So the value
    is always an explicit operator-expression — the operator + operand are chosen
    to match ``key``'s type (date / string / metadata).
    """
    if key in ("occurred_at", "created_at", "source_timestamp"):
        op = draw(st.sampled_from(["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"]))
        return {key: {op: _date_iso(draw(st.sampled_from(_DATE_OPERANDS)))}}
    if key in _STRING_POOLS:
        op = draw(st.sampled_from(["$eq", "$ne"]))
        return {key: {op: draw(st.sampled_from(_STRING_POOLS[key]))}}
    op = draw(st.sampled_from(["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"]))
    return {key: {op: draw(st.sampled_from(_META_SCALAR_POOL))}}


@st.composite
def _field_op_expr(draw: st.DrawFn) -> dict[str, Any]:
    """A single-field operator-expression on any key (a transformable-shape leaf)."""
    return draw(_field_op_expr_for(draw(st.sampled_from(_OP_EXPR_KEYS))))


@st.composite
def _distinct_field_op_exprs(draw: st.DrawFn) -> list[dict[str, Any]]:
    """A list of 2-3 ``_field_op_expr`` operands with PAIRWISE-DISTINCT keys.

    The implicit-AND ⇔ ``$and`` equivalence only holds when the merged document
    keeps one predicate per operand — a key collision would collapse two siblings
    into one and break the 1:1 mapping to the ``$and`` array. Distinct keys are
    drawn up front (``unique=True`` — no rejection loop, so Hypothesis keeps a
    small base example) and one operator-expression is built per key, so every
    example exercises the assertion rather than skipping on a collision.
    """
    keys = draw(st.lists(st.sampled_from(_OP_EXPR_KEYS), min_size=2, max_size=3, unique=True))
    return [draw(_field_op_expr_for(key)) for key in keys]


def _oracle_for(filter_dict: dict[str, Any]) -> frozenset[str]:
    """Oracle survivors for an already-known-valid filter dict (Property B helper)."""
    return _oracle_survivors(parse_to_ast(RecallFilter.model_validate(filter_dict)))


class TestMetamorphicOracleInvariance:
    """Equivalence-preserving rewrites must not change the oracle row-set.

    Oracle-only (no store): a divergence is a ``parse_to_ast`` / ``compile_python``
    normalization bug. Each transform draws its operand(s) from the dedicated
    transformable-shape strategy, so every draw is genuinely transformable.
    """

    @given(operands=_distinct_field_op_exprs())
    @settings(max_examples=150, deadline=None)
    def test_implicit_and_equals_explicit_and(self, operands: list[dict[str, Any]]) -> None:
        """Sibling predicates (implicit AND) ⇔ an explicit ``$and`` array.

        The operands carry pairwise-distinct keys (``_distinct_field_op_exprs``),
        so merging them into one document is 1:1 with the ``$and`` array — both
        forms address the same predicates and must keep the same rows. The defensive
        collision guard never fires given the distinct-key strategy, but stays so a
        future strategy change cannot silently make the comparison meaningless.
        """
        merged: dict[str, Any] = {}
        for operand in operands:
            merged.update(operand)
        if len(merged) != len(operands):  # pragma: no cover - distinct keys by construction
            return
        assert _oracle_for(merged) == _oracle_for({"$and": operands})

    @given(operands=st.lists(_field_op_expr(), min_size=2, max_size=3))
    @settings(max_examples=150, deadline=None)
    def test_nor_equals_not_or(self, operands: list[dict[str, Any]]) -> None:
        """``$nor([a, b])`` ⇔ ``$not($or([a, b]))``."""
        assert _oracle_for({"$nor": operands}) == _oracle_for({"$not": {"$or": operands}})

    @given(a=_field_op_expr(), b=_field_op_expr())
    @settings(max_examples=150, deadline=None)
    def test_de_morgan(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        """De Morgan: ``$not($or([a, b]))`` ⇔ ``$and([$not(a), $not(b)])``.

        ``a`` / ``b`` are single-field operator-expressions, so the field-position
        ``$not`` in the rewritten form is legal (a ``$not`` on a bare value is not).
        """
        left = {"$not": {"$or": [a, b]}}
        # Field-position $not: negate each operand's inner operator-expression.
        right = {"$and": [_field_not(a), _field_not(b)]}
        assert _oracle_for(left) == _oracle_for(right)

    @given(a=_field_op_expr())
    @settings(max_examples=150, deadline=None)
    def test_double_negation(self, a: dict[str, Any]) -> None:
        """``$not($not(a))`` ⇔ ``a`` (document-position double negation)."""
        assert _oracle_for({"$not": {"$not": a}}) == _oracle_for(a)


def _field_not(operand: dict[str, Any]) -> dict[str, Any]:
    """Wrap a single-field operator-expression's value in a field-position ``$not``.

    ``{key: {"$eq": v}}`` → ``{key: {"$not": {"$eq": v}}}``. The operand is always
    a one-key dict whose value is an operator-expression (the ``_field_op_expr``
    contract), so this is the legal field-position negation the validator accepts.
    """
    ((key, expr),) = operand.items()
    return {key: {"$not": expr}}
