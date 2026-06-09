"""Catalog tests for the recall-filter conformance corpus — ``@internal``.

This module is the *catalog* gate: it runs every hand-authored case the family
generators in :mod:`khora.filter.conformance` produce through the in-suite Python
oracle and the Chronicle plan/run seam, and asserts each backend agrees with the
case's **hand-declared** ``expected_ids``. It complements
:mod:`tests.unit.filter.test_conformance_harness` (which guards the *machinery* on
a handful of in-file smoke cases) by exercising the full ~263-case corpus.

The contract this gate pins, per family:

* every family generator returns a **non-empty** list (a stub that still
  ``return []`` is a corpus gap, caught loudly here, not a silent green);
* for every case, the Python oracle's survivors equal the case's declared
  ``expected_ids`` — :func:`assert_case` compares against the **declared** set,
  never the oracle's live output, so a wrong ``compile_python`` fails its own
  ``"python"`` case (the oracle is itself falsifiable);
* for every case whose ``backends`` includes ``"chronicle"``, the Chronicle
  executor agrees with the oracle (same survivor set, different path: the
  ``source_timestamp`` date-bound pushdown + full-AST post-filter).

``expected_ids`` is NEVER re-derived from the oracle here — the generators carry
the hand-declared sets, and a disagreement is the corpus author's bug to fix in
``conformance.py`` (this module only asserts; it does not author).
"""

from __future__ import annotations

import pytest

from khora.filter.conformance import (
    ChronicleExecutor,
    ConformanceCase,
    PythonExecutor,
    assert_case,
    f_array_cases,
    f_coerce_cases,
    f_dates_cases,
    f_dotkey_cases,
    f_exists_cases,
    f_impossible_cases,
    f_logic_cases,
    f_nullval_cases,
    f_objeq_cases,
    f_op_cases,
    f_polarity_cases,
    f_sel_cases,
    f_sugar_cases,
    f_unsup_cases,
    run_case_for_backend,
)

pytestmark = pytest.mark.unit


# The 14 family generators that make up the conformance corpus. Each entry is a
# (family-name, generator) pair: the name labels failures, the generator produces
# that family's hand-authored cases. F-VALIDATE is intentionally absent — it is a
# single-run validator family (tests/recall/test_filter_validator.py), not a
# parametrized-over-backends conformance family.
_FAMILY_GENERATORS = (
    ("F-OP", f_op_cases),
    ("F-SUGAR", f_sugar_cases),
    ("F-IMPOSSIBLE", f_impossible_cases),
    ("F-EXISTS", f_exists_cases),
    ("F-COERCE", f_coerce_cases),
    ("F-POLARITY", f_polarity_cases),
    ("F-OBJEQ", f_objeq_cases),
    ("F-DOTKEY", f_dotkey_cases),
    ("F-ARRAY", f_array_cases),
    ("F-LOGIC", f_logic_cases),
    ("F-DATES", f_dates_cases),
    ("F-NULLVAL", f_nullval_cases),
    ("F-SEL", f_sel_cases),
    ("F-UNSUP", f_unsup_cases),
)


def _all_cases() -> list[ConformanceCase]:
    """Every case across every family, in family order (catalog-wide corpus)."""
    cases: list[ConformanceCase] = []
    for _name, generator in _FAMILY_GENERATORS:
        cases.extend(generator())
    return cases


def _chronicle_cases() -> list[ConformanceCase]:
    """The subset of cases that target the Chronicle backend."""
    return [case for case in _all_cases() if "chronicle" in case.backends]


# --------------------------------------------------------------------------- #
# Every family generator is non-empty (no remaining `return []` stub).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,generator", _FAMILY_GENERATORS, ids=[n for n, _ in _FAMILY_GENERATORS])
def test_family_generator_is_non_empty(name: str, generator) -> None:
    cases = generator()
    assert isinstance(cases, list)
    assert cases, f"family {name} produced no cases — a generator stub is unfilled"


def test_corpus_case_ids_are_unique() -> None:
    # Each case.id is also its per-case namespace key (xdist-safe seeding), so a
    # collision would silently make two cases share a namespace. Guard uniqueness.
    ids = [case.id for case in _all_cases()]
    dupes = sorted({cid for cid in ids if ids.count(cid) > 1})
    assert not dupes, f"duplicate conformance case ids: {dupes}"


# --------------------------------------------------------------------------- #
# Python oracle: every case's survivors equal its HAND-DECLARED expected_ids.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _all_cases(), ids=lambda c: c.id)
def test_python_oracle_matches_declared_expected_ids(case: ConformanceCase) -> None:
    # assert_case compares the Python oracle's survivors against the case's
    # hand-declared expected_ids — never the oracle's live output. A disagreement
    # is the corpus author's bug (a wrong declared set OR a wrong compile_python),
    # surfaced here per case id.
    assert_case(case, "python", PythonExecutor())


# --------------------------------------------------------------------------- #
# Chronicle: for every chronicle-targeting case, agree with the oracle.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _chronicle_cases(), ids=lambda c: c.id)
def test_chronicle_matches_declared_expected_ids(case: ConformanceCase) -> None:
    # The Chronicle plan/run seam (source_timestamp date-bound pushdown + full-AST
    # post-filter) returns the same survivors the case declares.
    assert_case(case, "chronicle", ChronicleExecutor())


@pytest.mark.parametrize("case", _chronicle_cases(), ids=lambda c: c.id)
def test_chronicle_agrees_with_python_oracle(case: ConformanceCase) -> None:
    # Cross-check the two execution paths directly: Chronicle and the Python oracle
    # must keep the identical row set (the routing-equivalence the harness asserts
    # for the smoke cases, here over the whole chronicle-targeting corpus).
    py = run_case_for_backend(case, "python", executor=PythonExecutor())
    chron = run_case_for_backend(case, "chronicle", executor=ChronicleExecutor())
    assert chron == py, case.id
