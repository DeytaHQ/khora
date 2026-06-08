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
    PostgresExecutor,
    PythonExecutor,
    SeedRecord,
    assert_case,
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
# PostgresExecutor invokes the REAL compile_postgres.
# --------------------------------------------------------------------------- #


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
