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
    _harness.lane_skip("chronicle"),
    pytest.mark.skipif(
        not _harness._pg_reachable(),
        reason="start Postgres (make dev) to exercise the live Chronicle lane",
    ),
]


def _chronicle_rowset_cases() -> list[conformance.ConformanceCase]:
    """The row-set cases for the live Chronicle (``chronicle``) lane.

    Chronicle hydrates the denormalized document keys on the recall path, so this
    PG-only lane narrows on the system keys as well as the dotted-``metadata``
    families (``include_system_keys=True``): the ``remember``-threadable system-key
    ``F-OP`` families and the two ``source_name`` ``F-EXISTS`` presence states the
    embedded lane defers here. See ``_harness.rowset_cases``.
    """
    return _harness.rowset_cases("chronicle", include_system_keys=True)


@pytest.mark.parametrize("case", _chronicle_rowset_cases(), ids=lambda c: c.id)
async def test_rowset_reconciliation_chronicle(chronicle_kb, case) -> None:
    """The filter survivors reconcile to the case's expected ids on the Chronicle stack."""
    kb = chronicle_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


def test_lane_selection_matches_shipped_chronicle() -> None:
    """The resolver lane selection equals this module's shipped precedent (Layer-3).

    The Layer-1 empty-raise in ``lane_rowset_cases`` catches a token that selects
    NOTHING, but not a valid-but-wrong token that happens to select a different
    non-empty set. This pins the resolver path (``lane_rowset_cases("chronicle")``)
    to the shipped selection this module has always parametrized over
    (``rowset_cases("chronicle", include_system_keys=True)``), so a future
    ``_E2E_BACKEND_MAP`` drift that re-points ``chronicle`` at a different token or
    flips its system-key flag fails LOUD here instead of silently under/over-covering.
    """
    resolver_ids = {c.id for c in _harness.lane_rowset_cases("chronicle")}
    shipped_ids = {c.id for c in _chronicle_rowset_cases()}
    assert resolver_ids == shipped_ids
