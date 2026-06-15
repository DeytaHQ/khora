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

    # The F-LOGIC lane selection is pinned the same way (Layer-3): the resolver path
    # (``lane_logic_cases("chronicle")``) must equal the threadable subset this module
    # parametrizes ``test_logic_reconciliation_chronicle`` over, so a token / corpus
    # drift re-pointing ``chronicle`` at a different non-empty F-LOGIC set fails LOUD.
    logic_resolver_ids = {c.id for c in _harness.lane_logic_cases("chronicle")}
    logic_shipped_ids = {c.id for c in _chronicle_logic_cases()}
    assert logic_resolver_ids == logic_shipped_ids


# --------------------------------------------------------------------------- #
# F-LOGIC boolean-composition reconciliation (system + metadata compositions).
# --------------------------------------------------------------------------- #


def _chronicle_logic_cases() -> list[conformance.ConformanceCase]:
    """The threadable F-LOGIC boolean-composition cases for the live (``chronicle``) lane.

    Chronicle hydrates the denormalized document keys on the recall path, so this
    PG-only lane narrows on the system-key compositions (``$and`` / ``$or`` / ``$not``
    / De Morgan / distributivity over ``source_name`` / ``source_type`` /
    ``source_timestamp``) as well as the ``metadata`` ones. See
    ``_harness.engine_logic_cases``. The empty-raise is the same Layer-1 anti-vacuity
    guard the row-set helpers carry — a corpus shrink that drops every threadable
    F-LOGIC case for this lane fails RED rather than parametrizing a vacuous lane.
    """
    cases = [c for c in _harness.engine_logic_cases(conformance.f_logic_cases()) if "chronicle" in c.backends]
    if not cases:
        raise RuntimeError(
            "engine_logic_cases selected zero F-LOGIC cases for the chronicle lane — refusing a vacuously-green lane."
        )
    return cases


@pytest.mark.parametrize("case", _chronicle_logic_cases(), ids=lambda c: c.id)
async def test_logic_reconciliation_chronicle(chronicle_kb, case) -> None:
    """A boolean-composition filter's survivors reconcile to the case's expected ids on the Chronicle stack."""
    kb = chronicle_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)
