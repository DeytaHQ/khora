"""Row-set recall-filter proof on the embedded Skeleton + SurrealDB stack.

The Skeleton-engine SurrealDB leg of the row-set reconciliation: it drives the
same ``Khora.remember()`` -> ``Khora.recall(filter=...)`` proof the other lanes
do, but through the cost-optimised Skeleton engine over an embedded
``memory://`` SurrealDB store. Skeleton has no graph channel
(``supported_modes={VECTOR, HYBRID, KEYWORD}``), so this lane runs the row-set +
F-EXISTS proofs under ``mode=HYBRID`` only — the AC2 graph-fires proof stays on
the vc_full lane (``test_filter_rowset_graph.py`` / ``test_graph_contribution.py``).

The node ids carry the ``skeleton_surrealdb`` token so the e2e workflow selects
exactly this lane with ``-k skeleton_surrealdb`` (matching its
``KHORA_E2E_BACKEND``). Embedded ``memory://`` is in-process and container-free,
so this lane runs without Docker on the slow e2e job; it self-skips when the
embedded SurrealDB SDK is unavailable.
"""

from __future__ import annotations

import pytest

from khora.filter import conformance
from khora.query import SearchMode
from tests.e2e import _harness

_BACKEND = "skeleton_surrealdb"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    _harness.lane_skip(_BACKEND),
    pytest.mark.skipif(
        not _harness._surreal_embedded_available(),
        reason="embedded SurrealDB SDK not installed (pip install khora[surrealdb])",
    ),
]


# --------------------------------------------------------------------------- #
# Row-set reconciliation through the Skeleton SurrealDB read path (HYBRID only).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _harness.lane_rowset_cases(_BACKEND), ids=lambda c: c.id)
async def test_rowset_reconciliation_skeleton_surrealdb(skeleton_surrealdb_kb, case) -> None:
    """The filter survivors reconcile to the case's expected ids on the Skeleton SurrealDB stack."""
    kb = skeleton_surrealdb_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


# --------------------------------------------------------------------------- #
# F-EXISTS presence reachability on the Skeleton SurrealDB read path.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _harness.lane_exists_cases(_BACKEND), ids=lambda c: c.id)
async def test_f_exists_reachability_skeleton_surrealdb(skeleton_surrealdb_kb, case) -> None:
    """Each presence state reconciles to its hand-declared survivor set on Skeleton SurrealDB."""
    kb = skeleton_surrealdb_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


# --------------------------------------------------------------------------- #
# Layer-2 visibility — a named green signal that this lane ran >= 1 case.
# --------------------------------------------------------------------------- #


def test_lane_corpus_is_nonempty_skeleton_surrealdb() -> None:
    """The Skeleton SurrealDB lane selected at least one row-set and one F-EXISTS case.

    A green leg must never advertise coverage it did not run: this asserts the
    selector handed this lane a non-empty corpus, so a future change that drops
    every seedable family on SurrealDB fails LOUD here instead of passing
    vacuously with zero parametrized cases.
    """
    assert len(_harness.lane_rowset_cases(_BACKEND)) > 0
    assert len(_harness.lane_exists_cases(_BACKEND)) > 0
