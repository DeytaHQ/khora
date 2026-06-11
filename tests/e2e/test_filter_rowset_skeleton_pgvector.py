"""Row-set recall-filter proof on the live Skeleton + pgvector (PG-only) stack.

The Skeleton-engine pgvector leg of the row-set reconciliation: the same
``Khora.remember()`` -> ``Khora.recall(filter=...)`` proof the other lanes run,
through the cost-optimised Skeleton engine over Postgres + pgvector (no graph
backend — Skeleton has no graph channel, ``supported_modes={VECTOR, HYBRID,
KEYWORD}``). The live pgvector chunk row carries the denormalized document
system keys, so this lane narrows on them as well as the dotted-``metadata``
families (the lane resolver turns ``include_system_keys`` on). All proofs run
under ``mode=HYBRID`` only — the AC2 graph-fires proof stays on the vc_full lane
(``test_filter_rowset_graph.py`` / ``test_graph_contribution.py``).

The node ids carry the ``skeleton_pgvector`` token so the e2e workflow selects
exactly this lane with ``-k skeleton_pgvector`` (matching its
``KHORA_E2E_BACKEND``). Self-skip: gated on Postgres reachability via the module
``skipif`` so a no-Docker run collects-and-skips it cleanly; the CI leg
provisions Postgres and sets ``KHORA_E2E_PG_REQUIRED=1`` to convert a skip into a
hard red. Run under ``make dev``.
"""

from __future__ import annotations

import pytest

from khora.filter import conformance
from khora.query import SearchMode
from tests.e2e import _harness

_BACKEND = "skeleton_pgvector"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    _harness.lane_skip(_BACKEND),
    pytest.mark.skipif(
        not _harness._pg_reachable(),
        reason="start Postgres (make dev) to exercise the live Skeleton pgvector lane",
    ),
]


# --------------------------------------------------------------------------- #
# Row-set reconciliation through the live Skeleton pgvector read path (HYBRID only).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _harness.lane_rowset_cases(_BACKEND), ids=lambda c: c.id)
async def test_rowset_reconciliation_skeleton_pgvector(skeleton_pgvector_kb, case) -> None:
    """The filter survivors reconcile to the case's expected ids on the live Skeleton pgvector stack."""
    kb = skeleton_pgvector_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


# --------------------------------------------------------------------------- #
# F-EXISTS presence reachability on the live Skeleton pgvector read path.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _harness.lane_exists_cases(_BACKEND), ids=lambda c: c.id)
async def test_f_exists_reachability_skeleton_pgvector(skeleton_pgvector_kb, case) -> None:
    """Each presence state reconciles to its hand-declared survivor set on Skeleton pgvector."""
    kb = skeleton_pgvector_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


# --------------------------------------------------------------------------- #
# Layer-2 visibility — a named green signal that this lane ran >= 1 case.
# --------------------------------------------------------------------------- #


def test_lane_corpus_is_nonempty_skeleton_pgvector() -> None:
    """The Skeleton pgvector lane selected at least one row-set and one F-EXISTS case.

    A green leg must never advertise coverage it did not run: this asserts the
    selector handed this lane a non-empty corpus, so a future change that drops
    every seedable family on the live pgvector path fails LOUD here instead of
    passing vacuously with zero parametrized cases.
    """
    assert len(_harness.lane_rowset_cases(_BACKEND)) > 0
    assert len(_harness.lane_exists_cases(_BACKEND)) > 0
