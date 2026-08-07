"""Coverage meta-test: every system key has at least one generated operator case.

The generated ``F-OP`` family is meant to exercise *every* system key the filter
grammar whitelists (:data:`~khora.filter.SYSTEM_KEYS`) across its operators. This
meta-test reads the ``exercises`` tag tuple each generated case carries and asserts
their union is a superset of :data:`SYSTEM_KEYS` — so the day a new system key is
added without a matching operator case, this goes red.

A meta-test that can never fail is worthless, so the second test proves the teeth:
it drops every case tagging one key, then asserts the same coverage check *would*
fail for the depleted corpus. That makes the falsifiability explicit rather than
assumed.

No DB, no infra — this runs in the fast unit suite.
"""

from __future__ import annotations

import pytest

from khora.filter import SYSTEM_KEYS
from khora.filter.conformance import ConformanceCase, f_op_cases, f_op_documents_cases

pytestmark = pytest.mark.unit

# The document-enumeration surface accepts nine of the ten system keys. ``occurred_at``
# is the chunk event-time axis and no ``documents`` column backs it, so the documents
# corpus encodes it as a REJECTION (``_DOCUMENTS_REJECTED_FILTERS``, asserted in
# ``test_documents_conformance.py``) rather than as a row-set case — which is why the
# coverage target below is a strict subset and why its absence is asserted, not
# tolerated.
_ENUMERABLE_KEYS = SYSTEM_KEYS - {"occurred_at"}


def _covered_keys(cases: list[ConformanceCase]) -> frozenset[str]:
    """The union of every ``exercises`` tag across ``cases``.

    The ``exercises`` tuple is free-form (``("F-OP", "occurred_at", "$gte")``); the
    system-key member is whichever tag also lives in :data:`SYSTEM_KEYS`. Returning
    the whole tag union and intersecting against ``SYSTEM_KEYS`` at the assertion
    site keeps this helper agnostic to tag position.
    """
    return frozenset(tag for case in cases for tag in case.exercises)


def test_every_system_key_has_an_operator_case() -> None:
    """Each :data:`SYSTEM_KEYS` member is exercised by at least one generated case."""
    cases = f_op_cases()
    covered = _covered_keys(cases)
    missing = SYSTEM_KEYS - covered
    assert not missing, f"system keys with no F-OP case: {sorted(missing)}"


def test_coverage_check_is_falsifiable() -> None:
    """The coverage assertion has teeth: deplete one key and it must fail.

    Removing every case that tags one arbitrary system key leaves that key
    uncovered; the same superset check must then report it as missing. If this
    *passed* on the depleted corpus, the real coverage test above would be vacuous.
    """
    cases = f_op_cases()
    # Pick a key the corpus genuinely covers, so dropping its cases is a real change.
    target = sorted(SYSTEM_KEYS & _covered_keys(cases))[0]

    depleted = [case for case in cases if target not in case.exercises]
    assert depleted != cases, "expected to remove at least one case for the target key"

    missing = SYSTEM_KEYS - _covered_keys(depleted)
    assert target in missing, f"coverage check failed to notice {target!r} went uncovered — the meta-test has no teeth"


def test_every_enumerable_key_has_a_documents_op_case() -> None:
    """The documents corpus covers all nine enumerable keys — and NOT ``occurred_at``.

    Two assertions, both load-bearing. The superset check is the same contract as the
    chunk corpus's, narrowed to the enumerable subset. The second is the tighter one:
    an ``occurred_at`` tag reappearing would mean a case filtering a key the
    enumeration surface rejects had slipped back into the corpus as a row-set case,
    which ``_documents_eligible`` exists to prevent.
    """
    covered = _covered_keys(f_op_documents_cases())
    missing = _ENUMERABLE_KEYS - covered
    assert not missing, f"enumerable keys with no documents F-OP case: {sorted(missing)}"
    assert "occurred_at" not in covered, (
        "occurred_at is not an enumerable document key — it must be encoded as a "
        "rejection, never as a documents row-set case"
    )


def test_documents_coverage_is_falsifiable() -> None:
    """The documents coverage assertion has teeth: deplete one key and it must fail."""
    cases = f_op_documents_cases()
    # Pick a key the corpus genuinely covers, so dropping its cases is a real change.
    target = sorted(_ENUMERABLE_KEYS & _covered_keys(cases))[0]

    depleted = [case for case in cases if target not in case.exercises]
    assert depleted != cases, "expected to remove at least one case for the target key"

    missing = _ENUMERABLE_KEYS - _covered_keys(depleted)
    assert target in missing, f"coverage check failed to notice {target!r} went uncovered — the meta-test has no teeth"
