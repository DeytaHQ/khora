"""Row-set recall-filter proof on the embedded VectorCypher + sqlite_lance stack.

The VectorCypher-engine embedded leg of the matrix row-set reconciliation: the
same ``Khora.remember()`` -> ``Khora.recall(filter=...)`` proof the other lanes
run, through the default VectorCypher engine over an embedded SQLite + LanceDB
store. This is the matrix counterpart to ``test_filter_rowset_embedded.py``
(which carries the AC1/AC4/AC5 corpus on the main no-Docker job) — here the same
embedded VC stack is driven under the matrix lane selector so the e2e workflow
can run it as its own leg with ``-k vc_embedded``.

VectorCypher has a graph channel, but on the embedded stack the row-set proof
runs under ``mode=HYBRID`` like every matrix lane (the AC2 graph-fires proof
stays on the live vc_full lane, ``test_filter_rowset_graph.py`` /
``test_graph_contribution.py``). Because the embedded ``chunks`` row carries no
denormalized document system keys, the lane resolver leaves
``include_system_keys`` off, so only the dotted-``metadata`` families are in
scope here (the system-key families run on the live lanes).

The node ids carry the ``vc_embedded`` token so the e2e workflow selects exactly
this lane with ``-k vc_embedded`` (matching its ``KHORA_E2E_BACKEND``). The
embedded SQLite+LanceDB stack is container-free, so this lane runs without Docker
on the slow e2e job; it self-skips when the embedded deps are unavailable.
"""

from __future__ import annotations

import pytest

from khora.filter import conformance
from khora.query import SearchMode
from tests.e2e import _harness

_BACKEND = "vc_embedded"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    _harness.lane_skip(_BACKEND),
    pytest.mark.skipif(
        not _harness._embedded_available(),
        reason="aiosqlite/lancedb not installed (pip install khora[sqlite_lance])",
    ),
]


# --------------------------------------------------------------------------- #
# Row-set reconciliation through the embedded VectorCypher read path (HYBRID).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _harness.lane_rowset_cases(_BACKEND), ids=lambda c: c.id)
async def test_rowset_reconciliation_vc_embedded(sqlite_lance_kb, case) -> None:
    """The filter survivors reconcile to the case's expected ids on the embedded VectorCypher stack."""
    kb = sqlite_lance_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


# --------------------------------------------------------------------------- #
# F-EXISTS presence reachability on the embedded VectorCypher read path.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _harness.lane_exists_cases(_BACKEND), ids=lambda c: c.id)
async def test_f_exists_reachability_vc_embedded(sqlite_lance_kb, case) -> None:
    """Each presence state reconciles to its hand-declared survivor set on embedded VectorCypher."""
    kb = sqlite_lance_kb
    namespace_id = (await kb.create_namespace()).namespace_id
    survivors = await _harness.recall_survivors(kb, case, namespace_id, mode=SearchMode.HYBRID)

    assert survivors == case.expected_ids
    assert survivors == conformance.oracle_survivors(case)


# --------------------------------------------------------------------------- #
# Layer-2 visibility — a named green signal that this lane ran >= 1 case.
# --------------------------------------------------------------------------- #


def test_lane_corpus_is_nonempty_vc_embedded() -> None:
    """The embedded VectorCypher lane selected at least one row-set and one F-EXISTS case.

    A green leg must never advertise coverage it did not run: this asserts the
    selector handed this lane a non-empty corpus, so a future change that drops
    every seedable family on the embedded VC path fails LOUD here instead of
    passing vacuously with zero parametrized cases.
    """
    assert len(_harness.lane_rowset_cases(_BACKEND)) > 0
    assert len(_harness.lane_exists_cases(_BACKEND)) > 0
