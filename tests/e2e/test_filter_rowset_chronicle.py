"""Row-set recall-filter proof on the live Chronicle (PG-only) stack.

The Chronicle leg of the row-set reconciliation: it drives the same
``Khora.remember()`` -> ``Khora.recall(filter=...)`` proof through the Chronicle
engine over Postgres (no graph backend — Chronicle returns ``relationships=[]``).
It proves the deterministic recall filter narrows the same row set on the
Chronicle date-bound-pushdown + post-filter read path that it does on the embedded
and VectorCypher lanes.

Self-skip: gated on Postgres reachability via the ``chronicle_kb`` fixture's guard,
so a no-Docker run collects and skips this module cleanly. Run under ``make dev``.
"""

from __future__ import annotations

import pytest

from khora.filter import conformance
from khora.query import SearchMode
from tests.e2e import _harness

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _harness._pg_reachable(),
        reason="start Postgres (make dev) to exercise the live Chronicle lane",
    ),
]


def _chronicle_rowset_cases() -> list[conformance.ConformanceCase]:
    """Metadata-predicate cases that target the chronicle backend.

    The metadata-filtering families restricted to ``chronicle``-targeting cases with
    no duplicate ``external_id`` (the reconciliation key) and a dotted ``metadata``
    path predicate (whole-blob equality is curated out, as on the embedded lane).
    """
    cases: list[conformance.ConformanceCase] = []
    for family in (conformance.f_coerce_cases, conformance.f_objeq_cases, conformance.f_dotkey_cases):
        cases.extend(c for c in family() if "chronicle" in c.backends)
    return [
        c for c in cases if not _harness._has_duplicate_external_id(c.seed_records) and c.exercises[1] != "metadata"
    ]


@pytest.mark.parametrize("case", _chronicle_rowset_cases(), ids=lambda c: c.id)
async def test_rowset_reconciliation_chronicle(chronicle_kb, case) -> None:
    """The filter survivors reconcile to the case's expected ids on the Chronicle stack."""
    kb = chronicle_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)
