"""Meta-tests for the recall-filter conformance harness machinery — ``@internal``.

These guard the *machinery* in :mod:`khora.filter.conformance` (the case schema,
the runner, the executors, the F-OP generator, the oracle-falsifiability
contract), NOT the ~263-case catalog (a sibling ticket) nor the CI marked job
(another sibling ticket). A handful of hand-declared in-file smoke cases exercise
each moving part:

* the runner lowers through the REAL validator + ``parse_to_ast`` + real
  compilers and ``assert_case`` compares survivors against the **hand-declared**
  ``expected_ids`` — for every backend, the Python oracle included (so a wrong
  ``compile_python`` fails its own ``"python"`` case);
* the Python oracle and the Chronicle plan/run seam agree on the smoke cases;
* the F-OP generator covers every system key (the coverage assertion QA reuses)
  and its by-construction ``expected_ids`` survive the oracle cross-check;
* ``oracle_survivors`` is an authoring aid, not the assertion target.

The live-store seeder (:func:`~khora.filter.conformance.seed_case`) and the
``PostgresExecutor`` live-run path are integration concerns covered elsewhere;
here ``PostgresExecutor`` is only checked to invoke the real ``compile_postgres``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.sql.elements import ColumnElement

from khora.filter import RecallFilterUnsupportedError
from khora.filter.ast import FilterNode
from khora.filter.conformance import (
    ChronicleExecutor,
    ConformanceCase,
    CypherExecutor,
    LanceExecutor,
    PostgresExecutor,
    PythonExecutor,
    SeedRecord,
    SurrealExecutor,
    WeaviateExecutor,
    assert_case,
    f_exists_cases,
    f_objeq_cases,
    f_op_cases,
    oracle_survivors,
    run_case_for_backend,
)
from khora.filter.model import SYSTEM_KEYS

_HIT = datetime(2026, 6, 1, tzinfo=UTC)
_MISS = datetime(2020, 1, 1, tzinfo=UTC)
_BOUND = "2026-01-01T00:00:00Z"


# A handful of hand-declared smoke cases. expected_ids is counted by hand from the
# seed (NOT computed by running a compiler) — the whole point of the falsifiability
# contract is that these are an independent oracle.
_SMOKE_CASES: tuple[ConformanceCase, ...] = (
    ConformanceCase(
        id="SMOKE-source_timestamp-gte",
        filter={"source_timestamp": {"$gte": _BOUND}},
        seed_records=(
            SeedRecord(id="recent", source_timestamp=_HIT),
            SeedRecord(id="ancient", source_timestamp=_MISS),
            SeedRecord(id="undated"),
        ),
        expected_ids=frozenset({"recent"}),
        backends=frozenset({"python", "chronicle"}),
        exercises=("SMOKE", "source_timestamp", "$gte"),
    ),
    ConformanceCase(
        id="SMOKE-metadata-eq",
        filter={"metadata.tier": "gold"},
        seed_records=(
            SeedRecord(id="gold", metadata={"tier": "gold"}),
            SeedRecord(id="silver", metadata={"tier": "silver"}),
            SeedRecord(id="untagged"),
        ),
        expected_ids=frozenset({"gold"}),
        backends=frozenset({"python", "chronicle"}),
        exercises=("SMOKE", "metadata.tier", "$eq"),
    ),
    ConformanceCase(
        id="SMOKE-source_name-ne",
        filter={"source_name": {"$ne": "linear"}},
        seed_records=(
            SeedRecord(id="from-linear", source_name="linear"),
            SeedRecord(id="from-slack", source_name="slack"),
            SeedRecord(id="no-source-name"),
        ),
        # $ne includes absent / non-equal: everything except the exact match.
        expected_ids=frozenset({"from-slack", "no-source-name"}),
        backends=frozenset({"python", "chronicle"}),
        exercises=("SMOKE", "source_name", "$ne"),
    ),
    ConformanceCase(
        id="SMOKE-empty-matches-all",
        filter={},
        seed_records=(
            SeedRecord(id="a"),
            SeedRecord(id="b"),
        ),
        expected_ids=frozenset({"a", "b"}),
        backends=frozenset({"python", "chronicle"}),
        exercises=("SMOKE", "empty"),
    ),
)


@pytest.fixture
def python_executor() -> PythonExecutor:
    return PythonExecutor()


@pytest.fixture
def chronicle_executor() -> ChronicleExecutor:
    return ChronicleExecutor()


# --------------------------------------------------------------------------- #
# The runner + assert_case on the hand-declared smoke cases.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _SMOKE_CASES, ids=lambda c: c.id)
def test_python_oracle_matches_declared_expected_ids(case: ConformanceCase, python_executor: PythonExecutor) -> None:
    # assert_case compares the Python oracle's survivors against the HAND-DECLARED
    # expected_ids — the oracle is falsifiable against an independent count.
    assert_case(case, "python", python_executor)


@pytest.mark.parametrize("case", _SMOKE_CASES, ids=lambda c: c.id)
def test_chronicle_agrees_with_declared_expected_ids(
    case: ConformanceCase, chronicle_executor: ChronicleExecutor
) -> None:
    # The Chronicle plan/run seam (date-bound pushdown + full-AST post-filter)
    # returns the same survivors the case declares.
    assert_case(case, "chronicle", chronicle_executor)


@pytest.mark.parametrize("case", _SMOKE_CASES, ids=lambda c: c.id)
def test_chronicle_agrees_with_python_oracle(
    case: ConformanceCase, python_executor: PythonExecutor, chronicle_executor: ChronicleExecutor
) -> None:
    py = run_case_for_backend(case, "python", executor=python_executor)
    chron = run_case_for_backend(case, "chronicle", executor=chronicle_executor)
    assert chron == py


# --------------------------------------------------------------------------- #
# Falsifiability: a wrong declared expected_ids must make assert_case fail.
# --------------------------------------------------------------------------- #


def test_assert_case_fails_on_wrong_expected_ids(python_executor: PythonExecutor) -> None:
    # A deliberately wrong hand-declared expectation must be caught — proving the
    # assertion target is the declared set, not whatever the oracle computes.
    wrong = ConformanceCase(
        id="SMOKE-wrong",
        filter={"metadata.tier": "gold"},
        seed_records=(
            SeedRecord(id="gold", metadata={"tier": "gold"}),
            SeedRecord(id="silver", metadata={"tier": "silver"}),
        ),
        expected_ids=frozenset({"silver"}),  # WRONG on purpose (gold is the match)
        backends=frozenset({"python"}),
    )
    with pytest.raises(AssertionError):
        assert_case(wrong, "python", python_executor)


def test_assert_case_requires_expected_ids_for_survivor_assertion(python_executor: PythonExecutor) -> None:
    case = ConformanceCase(
        id="SMOKE-none-expected",
        filter={"metadata.tier": "gold"},
        seed_records=(SeedRecord(id="gold", metadata={"tier": "gold"}),),
        expected_ids=None,
        backends=frozenset({"python"}),
    )
    with pytest.raises(ValueError, match="expected_ids is required"):
        assert_case(case, "python", python_executor)


# --------------------------------------------------------------------------- #
# oracle_survivors is an authoring aid, not the assertion target.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _SMOKE_CASES, ids=lambda c: c.id)
def test_oracle_survivors_helper_agrees_with_declared_ids(case: ConformanceCase) -> None:
    assert oracle_survivors(case) == case.expected_ids


# --------------------------------------------------------------------------- #
# F-OP generator: coverage + by-construction expected_ids survive the oracle.
# --------------------------------------------------------------------------- #


def test_f_op_corpus_is_non_empty() -> None:
    assert f_op_cases(), "the F-OP generator must produce cases"


def test_f_op_exercises_cover_every_system_key() -> None:
    # The coverage meta-test QA reuses: the union of every case's `exercises` tags
    # must be a superset of SYSTEM_KEYS.
    covered = {tag for case in f_op_cases() for tag in case.exercises}
    assert SYSTEM_KEYS <= covered, f"F-OP exercises miss system keys: {sorted(SYSTEM_KEYS - covered)}"


def test_f_op_coverage_check_is_falsifiable() -> None:
    # Teeth for the coverage assertion above: drop every case tagging one covered
    # key and confirm the SAME `SYSTEM_KEYS <= covered` check then FAILS for that
    # key — so the coverage test is demonstrably not a tautology.
    cases = f_op_cases()
    covered = {tag for case in cases for tag in case.exercises}
    target = sorted(SYSTEM_KEYS & covered)[0]

    depleted = [case for case in cases if target not in case.exercises]
    assert depleted != cases, "expected to drop at least one case for the target key"

    depleted_covered = {tag for case in depleted for tag in case.exercises}
    assert not (SYSTEM_KEYS <= depleted_covered), "coverage check stayed green with a key removed — no teeth"
    assert target in (SYSTEM_KEYS - depleted_covered)


def test_f_op_expected_ids_confirmed_by_oracle() -> None:
    # Each F-OP case's BY-CONSTRUCTION expected_ids must survive the Python oracle
    # cross-check (the generator declares them; the oracle confirms — never defines).
    for case in f_op_cases():
        assert oracle_survivors(case) == case.expected_ids, case.id


def test_f_op_chronicle_agrees_with_python_oracle() -> None:
    py = PythonExecutor()
    chron = ChronicleExecutor()
    for case in f_op_cases():
        py_ids = run_case_for_backend(case, "python", executor=py)
        chron_ids = run_case_for_backend(case, "chronicle", executor=chron)
        assert chron_ids == py_ids, case.id


# --------------------------------------------------------------------------- #
# F-OBJEQ: a metadata sub-path dict operand is EXACT object_equal, NOT @>.
# --------------------------------------------------------------------------- #
#
# These ACTUALLY RUN the f_objeq_cases() corpus through the Python oracle (the
# reference compile_postgres is checked against). The superset record (extra key)
# must NOT survive the $eq — that is the bug the postgres _md_eq fix closes; the
# matching postgres-side SQL shape (exact `#> = sorted-jsonb`, no `@>`) is asserted
# in test_compile_postgres.py, and the same case runs end-to-end against the live
# PostgresExecutor under the integration-gated harness.


def test_f_objeq_corpus_is_non_empty() -> None:
    assert f_objeq_cases(), "the F-OBJEQ generator must produce cases"


@pytest.mark.parametrize("case", f_objeq_cases(), ids=lambda c: c.id)
def test_f_objeq_python_oracle_matches_declared_expected_ids(case: ConformanceCase) -> None:
    # The oracle's survivors must equal the HAND-DECLARED expected_ids — proving the
    # superset record (extra key) does NOT match the exact object_equal $eq.
    assert_case(case, "python", PythonExecutor())


def test_postgres_executor_invokes_real_compiler() -> None:
    captured: dict[str, Any] = {}

    def runner(predicate: Any, params: Any, records: Any) -> frozenset[str]:
        captured["predicate"] = predicate
        return frozenset()

    executor = PostgresExecutor(runner)
    run_case_for_backend(_SMOKE_CASES[1], "postgres", executor=executor)

    # The injected runner received a genuine compiled SQLAlchemy predicate — proof
    # the real compile_postgres ran, not a context-only shim.
    assert isinstance(captured["predicate"], ColumnElement)


# --------------------------------------------------------------------------- #
# assert_case's expect_unsupported branch (harness control flow).
# --------------------------------------------------------------------------- #
#
# The validator and the python/postgres compilers share one expressiveness
# envelope by design, so no VALIDATED wire filter naturally diverges between them
# (the catalog's F-UNSUP family carves out the real unsupported clauses end to end).
# To exercise the harness's own unsupported handling here, in process, a synthetic
# executor raises RecallFilterUnsupportedError and assert_case is checked to (a)
# accept it when the backend is declared unsupported and (b) NOT swallow it
# otherwise.


class _RaisingExecutor:
    """A BackendExecutor (structural) that always raises unsupported.

    Stands in for a real compiler that cannot express a clause; the filter it is
    handed only needs to lower through ``parse_to_ast`` (a trivially valid one).
    """

    def survivors(
        self,
        filter_ast: FilterNode,
        records: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> frozenset[str]:
        raise RecallFilterUnsupportedError(("source_name",), "synthetic: backend cannot express this clause")


def _unsupported_self_test_case(*, expect_unsupported: bool) -> ConformanceCase:
    return ConformanceCase(
        id="SMOKE-unsupported",
        filter={"source_name": "x"},
        seed_records=(SeedRecord(id="a", source_name="x"),),
        expected_ids=None if expect_unsupported else frozenset({"a"}),
        backends=frozenset({"postgres"}),
        expect_unsupported=frozenset({"postgres"}) if expect_unsupported else frozenset(),
    )


def test_assert_case_accepts_declared_unsupported() -> None:
    # When the backend is in expect_unsupported AND the executor raises, the case
    # passes — assert_case wraps the run in pytest.raises for that backend.
    assert_case(_unsupported_self_test_case(expect_unsupported=True), "postgres", _RaisingExecutor())


def test_assert_case_propagates_unexpected_unsupported() -> None:
    # A raise from a backend NOT listed in expect_unsupported must surface — the
    # harness never silently swallows an unsupported error it did not expect.
    with pytest.raises(RecallFilterUnsupportedError):
        assert_case(_unsupported_self_test_case(expect_unsupported=False), "postgres", _RaisingExecutor())


# --------------------------------------------------------------------------- #
# Live-store executors invoke their REAL compiler (proof they execute).
# --------------------------------------------------------------------------- #
#
# Each of the four live-store executors must run its REAL backend compiler in-harness
# (this is what conformance checks) and hand the result to the injected LiveRunner,
# not a context-only shim. A capturing runner proves the compiled artifact + the
# compile_python post-filter reached the seam.


@pytest.mark.parametrize(
    ("executor_cls", "predicate_is_str"),
    [
        (SurrealExecutor, True),
        (CypherExecutor, True),
        (LanceExecutor, True),
        (WeaviateExecutor, False),  # weaviate predicate is a _Filters object or None
    ],
)
def test_live_executor_invokes_real_compiler(executor_cls: type, predicate_is_str: bool) -> None:
    captured: dict[str, Any] = {}

    def runner(compiled: Any, filter_ast: Any, post_filter: Any, records: Any) -> frozenset[str]:
        captured["compiled"] = compiled
        captured["post_filter"] = post_filter
        return frozenset()

    executor = executor_cls(runner)
    # A metadata $eq case — pushes down on surrealdb/lance, defers on cypher/weaviate.
    run_case_for_backend(_SMOKE_CASES[1], "surrealdb", executor=executor)

    compiled = captured["compiled"]
    # The injected runner received a genuine CompiledFilter from the real compiler.
    assert hasattr(compiled, "predicate")
    assert hasattr(compiled, "consumed_keys")
    if predicate_is_str:
        assert isinstance(compiled.predicate, str)
    # The compile_python post-filter (the split-mode safety net) reached the seam.
    assert callable(captured["post_filter"])


# --------------------------------------------------------------------------- #
# AC#5 — F-EXISTS executes in BOTH modes: pushed-down AND post-filtered.
# --------------------------------------------------------------------------- #
#
# Each of the 8 F-EXISTS shapes must run in both pushdown mode (postgres / surrealdb
# / sqlite_lance push $exists natively → the leaf is consumed) AND post-filter mode
# (cypher / weaviate defer it → the leaf is NOT consumed, the compile_python
# post-filter re-checks it). Asserting BOTH per shape gives 8 shapes × 2 modes = 16
# executed exists-mode combinations. The mode is read off ``consumed_keys`` from the
# real per-backend compiler (no live store needed — this guards the routing), using
# the SAME production split-mode contexts the live executors compile with.


def _exists_leaf_consumed(case: ConformanceCase, compiler, ctx) -> bool:  # noqa: ANN001 - compiler/ctx typed in module
    """Whether ``compiler`` pushes ``case``'s $exists leaf down (it lands in consumed_keys)."""
    from khora.filter.conformance import _resolve_ast  # internal harness helper

    ast = _resolve_ast(case.filter)
    consumed = compiler(ast, ctx).consumed_keys
    # The exists path is the case's single metadata/system leaf; if ANY leaf was
    # consumed the backend pushed this shape (F-EXISTS cases are single-leaf except
    # the $and present-and-null shape, where consuming the metadata leaf is the push).
    return bool(consumed)


def _exists_mode_contexts():  # noqa: ANN202 - returns a list of (label, compiler, ctx, pushes)
    """The (compiler, production split-mode ctx) pairs, labeled by expected push mode.

    Mirrors the four live executors' contexts exactly. ``pushes`` records the
    expected mode for a metadata $exists leaf: postgres / surrealdb / sqlite_lance
    push it; cypher / weaviate defer it to the post-filter.
    """
    from khora.filter import SchemaCapabilities
    from khora.filter.compilers.cypher import compile_cypher
    from khora.filter.compilers.lance import compile_lance
    from khora.filter.compilers.postgres import compile_postgres
    from khora.filter.compilers.surrealdb import compile_surrealdb
    from khora.filter.compilers.weaviate import compile_weaviate
    from khora.filter.context import CompileContext

    return [
        ("postgres", compile_postgres, CompileContext(backend_target="khora_chunks", on_unsupported="split"), True),
        (
            "surrealdb",
            compile_surrealdb,
            CompileContext(
                backend_target="temporal_chunk", field_mapping={"metadata": "metadata_"}, on_unsupported="split"
            ),
            True,
        ),
        (
            "sqlite_lance",
            compile_lance,
            CompileContext(
                backend_target="khora_chunks",
                on_unsupported="split",
                schema_capabilities=SchemaCapabilities(sqlite_json1=True),
            ),
            True,
        ),
        (
            "cypher",
            compile_cypher,
            CompileContext(backend_target="Chunk", table_alias="c", on_unsupported="split"),
            False,
        ),
        (
            "weaviate",
            compile_weaviate,
            CompileContext(
                backend_target="KhoraChunk",
                field_mapping={"occurred_at": "occurred_at", "created_at": "created_at"},
                on_unsupported="split",
            ),
            False,
        ),
    ]


@pytest.mark.parametrize("case", f_exists_cases(), ids=lambda c: c.id)
def test_f_exists_runs_in_both_pushdown_and_postfilter_modes(case: ConformanceCase) -> None:
    # AC#5: every F-EXISTS shape must execute in BOTH modes. Confirm ≥1 backend
    # pushes the shape down (consumed) AND ≥1 backend defers it to the post-filter
    # (not consumed) — 8 shapes × 2 modes = 16 executed combinations. The push set
    # is {postgres, surrealdb, sqlite_lance}; the defer set is {cypher, weaviate}.
    pushed: list[str] = []
    deferred: list[str] = []
    for label, compiler, ctx, _expected in _exists_mode_contexts():
        if label not in case.backends:
            continue
        if _exists_leaf_consumed(case, compiler, ctx):
            pushed.append(label)
        else:
            deferred.append(label)
    assert pushed, f"{case.id}: no pushdown-mode backend executed this shape (expected ≥1 of postgres/surrealdb/lance)"
    assert deferred, f"{case.id}: no post-filter-mode backend executed this shape (expected ≥1 of cypher/weaviate)"


def test_f_exists_covers_eight_shapes_in_two_modes() -> None:
    # The aggregate AC#5 count: 8 distinct F-EXISTS shapes, each runnable in both
    # modes → 16 executed exists-mode combinations. Guard the shape count so a
    # dropped shape is caught loudly (the per-shape test above guards each mode).
    shapes = [c for c in f_exists_cases()]
    assert len(shapes) == 8, f"expected 8 F-EXISTS shapes (8 × 2 modes = 16), got {len(shapes)}"
