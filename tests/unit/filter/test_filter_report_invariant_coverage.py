"""Drift guard for the cross-engine filter-report invariant gate.

The gate (``tests/e2e/test_filter_report_invariant.py``) asserts every engine
lane's emitted ``engine_info["filter"]`` obeys the engine-independent invariants.
It rides the e2e recall matrix because the report is produced only on a real
recall (the compile-only filter-conformance matrix never calls ``.recall()`` and
emits no report). For the gate to actually cover the matrix, four things must
stay in lockstep:

  * the conformance backend tokens (``khora.filter.conformance.BACKENDS``) — every
    backend the filter subsystem compiles for,
  * the e2e lanes (``_harness._E2E_BACKEND_MAP`` keys) — the real-recall configs,
  * the gate's covered lanes (``test_filter_report_invariant.GATE_LANES``),
  * the e2e workflow ``matrix.include`` backends — what CI actually runs.

This hermetic test (PyYAML parse + import of the pure constants — no DB, no
network) maps each conformance token to its representative e2e lane and asserts
the chain. A new conformance backend, a new e2e lane, a new workflow leg, or a
gate that drops a lane — any of these without the others — fails RED here.

The ``python`` conformance token is the one legitimate exclusion: it is the
in-process compile oracle (``compile_python``), not an engine — no engine
``.recall()`` emits a ``FilterPushdownReport`` for it, so it maps to ``None`` and
is never expected on a lane. ``turbopuffer`` is deliberately NOT here: it is not
in ``BACKENDS`` (it raises on constraint filters), so it never reaches the table.

Falsifiability: two tests prove the chain has teeth — a synthetic extra
conformance token must break the BACKENDS-equality assertion, and a synthetic
extra workflow leg must break the matrix-equality assertion. Precedent:
``test_verification_coverage_gate.py::test_gate_is_falsifiable``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from khora.filter.conformance import BACKENDS
from tests.e2e._harness import _E2E_BACKEND_MAP, assert_filter_report_invariants
from tests.e2e.test_filter_report_invariant import GATE_EXCLUSIONS, GATE_LANES

pytestmark = [pytest.mark.unit]

# tests/unit/filter/<this file> → parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_E2E_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "e2e.yml"

# Each conformance backend token → the e2e lane that exercises it on a real recall
# (or None when no engine recall emits a report for it). EVERY token in BACKENDS
# must appear here (assertion 1), so a new conformance backend without a lane
# mapping fails RED. The trailing comment on each line names any additional lanes
# that also exercise the token (the table picks ONE representative lane per token).
_BACKEND_TOKEN_TO_LANE: dict[str, str | None] = {
    "postgres": "skeleton_pgvector",  # also exercised by chronicle (PG) + vc_full
    "chronicle": "chronicle",
    "surrealdb": "skeleton_surrealdb",
    "cypher": "vc_full",  # the VC graph lane reconciles against the cypher token
    "weaviate": "skeleton_weaviate",
    "sqlite_lance": "vc_embedded",  # also skeleton_sqlite_lance
    "python": None,  # EXCLUDED: in-process compile oracle; no engine .recall() emits a report for it.
}


def _e2e_matrix_backends() -> set[str]:
    """The ``matrix.include`` backend tokens from the e2e workflow (hermetic parse)."""
    doc = yaml.safe_load(_E2E_WORKFLOW.read_text())
    backends = {entry["backend"] for entry in doc["jobs"]["e2e"]["strategy"]["matrix"]["include"]}
    assert backends, "e2e workflow matrix.include yielded no backends — workflow shape changed"
    return backends


def _mapped_lanes() -> set[str]:
    """The non-None lanes the backend-token table maps to."""
    return {lane for lane in _BACKEND_TOKEN_TO_LANE.values() if lane}


def test_backend_token_table_covers_every_conformance_backend() -> None:
    """Assertion 1: the token table's keys equal the conformance ``BACKENDS`` set.

    A new conformance backend token without a table entry fails here — forcing a
    conscious decision (map it to its real-recall lane, or to ``None`` with a
    ``# EXCLUDED: <reason>`` comment). ``turbopuffer`` is not in ``BACKENDS`` (it
    raises on constraint filters), so it never reaches this table.
    """
    assert set(_BACKEND_TOKEN_TO_LANE) == set(BACKENDS), (
        "the filter-report-gate backend-token table drifted from conformance.BACKENDS:\n"
        f"  only in conformance.BACKENDS (add a lane mapping or None): {sorted(set(BACKENDS) - set(_BACKEND_TOKEN_TO_LANE))}\n"
        f"  only in the table (stale token):                          {sorted(set(_BACKEND_TOKEN_TO_LANE) - set(BACKENDS))}"
    )


def test_mapped_lanes_are_real_e2e_lanes() -> None:
    """Assertion 2: every mapped lane is a real ``_E2E_BACKEND_MAP`` lane."""
    stray = _mapped_lanes() - set(_E2E_BACKEND_MAP)
    assert not stray, f"backend-token table maps to lanes not in _E2E_BACKEND_MAP: {sorted(stray)}"


def test_mapped_lanes_are_gate_lanes() -> None:
    """Assertion 3: every mapped lane is covered by the gate (``GATE_LANES``)."""
    uncovered = _mapped_lanes() - set(GATE_LANES)
    assert not uncovered, f"backend-token table maps to lanes the gate does not cover: {sorted(uncovered)}"


def test_e2e_matrix_equals_backend_map() -> None:
    """Assertion 4: the e2e workflow matrix backends equal the ``_E2E_BACKEND_MAP`` keys.

    A new e2e.yml leg without a map key (or a new map key without a leg) fails
    here — the gate would otherwise silently not run on (or claim) a lane.
    """
    e2e_backends = _e2e_matrix_backends()
    map_keys = set(_E2E_BACKEND_MAP)
    assert e2e_backends == map_keys, (
        "e2e workflow matrix backends and _E2E_BACKEND_MAP keys diverged:\n"
        f"  only in e2e.yml:          {sorted(e2e_backends - map_keys)}\n"
        f"  only in _E2E_BACKEND_MAP: {sorted(map_keys - e2e_backends)}"
    )


def test_gate_covers_every_real_recall_lane() -> None:
    """Assertion 5: the gate covers every real-recall lane (``GATE_LANES`` == map keys).

    With no exclusions today this is plain equality; the structure tolerates a
    future compile-only lane via ``GATE_EXCLUSIONS`` (asserted empty below), so the
    general contract is ``GATE_LANES | GATE_EXCLUSIONS == set(_E2E_BACKEND_MAP)``.
    """
    assert set(GATE_LANES) | set(GATE_EXCLUSIONS) == set(_E2E_BACKEND_MAP), (
        "the gate's covered lanes (plus documented exclusions) drifted from _E2E_BACKEND_MAP:\n"
        f"  uncovered lanes (add to GATE_LANES or GATE_EXCLUSIONS): "
        f"{sorted(set(_E2E_BACKEND_MAP) - set(GATE_LANES) - set(GATE_EXCLUSIONS))}\n"
        f"  gate lanes with no e2e map entry: {sorted((set(GATE_LANES) | set(GATE_EXCLUSIONS)) - set(_E2E_BACKEND_MAP))}"
    )


def test_no_exclusions_today() -> None:
    """Every engine emits a report on recall, so there are NO gate exclusions today.

    The ``GATE_EXCLUSIONS`` structure exists for a future compile-only engine.
    Until one lands the set MUST be empty — an exclusion here would hide a real
    coverage gap rather than record a genuine compile-only lane.
    """
    assert GATE_EXCLUSIONS == frozenset(), (
        "GATE_EXCLUSIONS is non-empty, but every current engine emits engine_info['filter'] on "
        "recall — an exclusion would hide a coverage gap. Remove it, or, if a genuine compile-only "
        "engine landed, update this test and document the lane."
    )


# --------------------------------------------------------------------------- #
# Falsifiability — the chain must FAIL on a synthetic drift, or it is theater.
# --------------------------------------------------------------------------- #


def test_backend_equality_is_falsifiable() -> None:
    """A conformance token absent from the table must break assertion 1.

    Proves assertion 1 has teeth: a synthetic extra ``BACKENDS`` member with no
    table entry makes the key-set equality fail, mirroring the real RED a new
    conformance backend would trigger.
    """
    synthetic_backends = set(BACKENDS) | {"newbackend"}
    assert set(_BACKEND_TOKEN_TO_LANE) != synthetic_backends, (
        "a token not in the table did NOT break BACKENDS-equality — the drift guard has no teeth"
    )


def test_matrix_equality_is_falsifiable() -> None:
    """A workflow leg absent from ``_E2E_BACKEND_MAP`` must break assertion 4.

    Proves assertion 4 has teeth: a synthetic extra matrix backend with no map key
    makes the equality fail, mirroring the real RED a new e2e.yml leg would trigger.
    """
    synthetic_matrix = _e2e_matrix_backends() | {"newlane"}
    assert synthetic_matrix != set(_E2E_BACKEND_MAP), (
        "an extra matrix leg did NOT break matrix-equality — the drift guard has no teeth"
    )


# --------------------------------------------------------------------------- #
# Helper falsifiability — assert_filter_report_invariants must REJECT a malformed
# report (a helper that can't fail on a bad report is theater). A canonical valid
# report is the control: it must PASS so the rejections aren't a degenerate
# always-raise. Each malformed variant violates exactly one invariant.
# --------------------------------------------------------------------------- #

_TWO_LEAVES = frozenset({"source_name", "metadata.tier"})
_VALID_REPORT: dict = {
    "pushed_down": True,
    "post_filtered": False,
    "pushed_keys": ["metadata.tier", "source_name"],
    "post_filtered_keys": [],
    "channels": {"vector": {"pushed_keys": ["metadata.tier", "source_name"], "post_filtered_keys": []}},
}


def test_helper_accepts_a_valid_report() -> None:
    """The control: a canonical fully-pushed two-leaf report passes (no degenerate always-raise)."""
    assert_filter_report_invariants(_VALID_REPORT, _TWO_LEAVES)


@pytest.mark.parametrize(
    ("report", "why"),
    [
        # pushed_down True while a leaf is post-filtered (violates (d) list-form).
        (
            {
                "pushed_down": True,
                "post_filtered": True,
                "pushed_keys": ["source_name"],
                "post_filtered_keys": ["metadata.tier"],
                "channels": {"vector": {"pushed_keys": ["source_name"], "post_filtered_keys": ["metadata.tier"]}},
            },
            "pushed_down True with non-empty post_filtered_keys",
        ),
        # Top-level pushed_keys disagrees with the channel fold (violates (e)):
        # the channel re-checked metadata.tier but the top level lists it as pushed.
        (
            {
                "pushed_down": True,
                "post_filtered": True,
                "pushed_keys": ["metadata.tier", "source_name"],
                "post_filtered_keys": [],
                "channels": {"vector": {"pushed_keys": ["source_name"], "post_filtered_keys": ["metadata.tier"]}},
            },
            "top-level pushed_keys disagrees with the channel fold",
        ),
        # A key not in the leaf set (violates (c) subset + (e) channel subset).
        (
            {
                "pushed_down": False,
                "post_filtered": False,
                "pushed_keys": ["not_a_leaf"],
                "post_filtered_keys": [],
                "channels": {"vector": {"pushed_keys": ["not_a_leaf"], "post_filtered_keys": []}},
            },
            "pushed_keys names a key outside the leaf set",
        ),
        # pushed_keys and post_filtered_keys overlap (violates (c) disjoint).
        (
            {
                "pushed_down": False,
                "post_filtered": True,
                "pushed_keys": ["source_name"],
                "post_filtered_keys": ["source_name"],
                "channels": {"vector": {"pushed_keys": ["source_name"], "post_filtered_keys": ["source_name"]}},
            },
            "pushed_keys and post_filtered_keys overlap",
        ),
        # Unsorted top-level pushed_keys (violates (b)).
        (
            {
                "pushed_down": True,
                "post_filtered": False,
                "pushed_keys": ["source_name", "metadata.tier"],
                "post_filtered_keys": [],
                "channels": {"vector": {"pushed_keys": ["metadata.tier", "source_name"], "post_filtered_keys": []}},
            },
            "top-level pushed_keys not sorted",
        ),
        # Non-empty post_filtered_keys with the post_filtered bool False (violates (f)).
        (
            {
                "pushed_down": False,
                "post_filtered": False,
                "pushed_keys": ["source_name"],
                "post_filtered_keys": ["metadata.tier"],
                "channels": {"vector": {"pushed_keys": ["source_name"], "post_filtered_keys": ["metadata.tier"]}},
            },
            "non-empty post_filtered_keys but post_filtered flag False",
        ),
        # An extra top-level key (violates (a) schema).
        (
            {
                "pushed_down": True,
                "post_filtered": False,
                "pushed_keys": ["metadata.tier", "source_name"],
                "post_filtered_keys": [],
                "channels": {"vector": {"pushed_keys": ["metadata.tier", "source_name"], "post_filtered_keys": []}},
                "supported": True,
            },
            "extra top-level key (schema)",
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_helper_rejects_a_malformed_report(report: dict, why: str) -> None:
    """Each malformed report violates exactly one invariant and must raise."""
    with pytest.raises(AssertionError):
        assert_filter_report_invariants(report, _TWO_LEAVES)


def test_helper_rejects_a_leafless_report_claiming_pushed_down() -> None:
    """A leafless report that claims pushed_down=True must raise (the (d) else branch)."""
    bad = {"pushed_down": True, "post_filtered": False, "pushed_keys": [], "post_filtered_keys": [], "channels": {}}
    with pytest.raises(AssertionError):
        assert_filter_report_invariants(bad, frozenset())
