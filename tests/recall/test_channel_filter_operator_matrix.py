"""Channel × filter-operator coverage matrix — registry + meta-tests + cell driver.

GitHub issue #1223 follow-up. The #1223 fix stopped a provenance ``$ne`` filter
leaking through the recency channel; this matrix turns that one-off leak into a
SYSTEMATIC channel × operator-class grid so a regression on ANY post-filter-prone
channel × any operator class fails RED.

This file is the HERMETIC half (no Postgres, no Neo4j): it holds

* the **registry** (the single source of truth both halves key off):
  ``_POST_FILTER_CHANNELS`` / ``_EXCLUDED_CHANNELS`` / ``_MATRIX_ROWS`` /
  ``_OPERATOR_CLASSES`` / ``_CELLS`` / ``_RECIPES``;
* the **registry / partition META-TESTS** (``pytest.mark.unit``) — pure set
  arithmetic over the source. They are the new-channel backstop: if someone adds a
  filter-aware channel to ``retriever.py`` and does not classify it (post-filter
  seam vs pushdown/dispatcher), the partition meta-test fails until they do. They
  run in CI's ``test-unit`` job (``pytest tests/recall/`` — no infra), so the leak
  backstop is ALWAYS exercised. Modelled on
  ``tests/recall/test_filter_enforcement_audit_gate.py`` in this same directory.
* the shared **cell driver** ``_assert_channel_cell`` (+ ``_SeedDoc`` / ``_violates``
  / the per-row ``_RECIPES``). The driver is a function DEFINITION here — it is only
  EXECUTED by the live cells in ``tests/integration/test_channel_filter_operator_matrix.py``
  (RECENCY + GRAPH) and ``tests/integration/test_channel_filter_operator_matrix_change.py``
  (SESSION + CHANGE), which run in CI's ``test-integration`` job with live PG+Neo4j
  and ``NEO4J_INTEGRATION_TEST=1``. Defining it here (hermetic) keeps the registry +
  helper in one importable module both live modules share.

WHY two directories: CI's ``e2e.yml`` ``vc_full`` leg (the only leg with Neo4j)
path-pins its pytest selection to specific modules, so a NEW ``tests/e2e/`` module
would never be selected there and its cells would only ever SKIP (vacuous green).
``tests/integration/`` is run wholesale (``pytest tests/integration/ -m integration``)
with PG+Neo4j provisioned, so the live cells actually execute there — no workflow
change needed.

The matrix axes:

* **Rows = the candidate-gating channels the matrix VERIFIES BEHAVIORALLY** — the
  post-filter-PRONE / historically-post-filter seams where a filter-violating chunk
  could reach RRF fusion if enforcement regressed. NOTE this is NOT "channels that
  post-filter in memory today": post-#1236 the recency channel PUSHES the filter
  into the khora_chunks SQL, and ``_fetch_chunks_from_entities`` is HYBRID (pushes
  the system/provenance slice via compile_cypher, post-filters metadata via
  compile_python). They stay rows because the per-cell assertion is mechanism-
  AGNOSTIC (violating chunk absent + honest report + pinned key in
  ``pushed_keys ∪ post_filtered_keys``), so a row is valid whether the channel
  pushes or post-filters — and the recency row is the #1223 behavioral guard that
  catches any regression back to in-memory post-filtering. Exactly two such seams
  exist in ``retriever.py`` (``_recency_channel_chunks`` +
  ``_fetch_chunks_from_entities``), proven by the partition meta-test below. Two
  more rows (``session`` / ``change``) cover behavioral fan-out / decomposition that
  enforce VIA the pushdown ``_vector_search_chunks`` channel — NOT a separate
  post-filter seam, so they are ``is_partition_member=False`` and their cells live in
  ``tests/integration/test_channel_filter_operator_matrix_change.py``.

* **Columns = operator classes** — the filter shapes most likely to mis-enforce at
  a post-filter boundary: relational provenance comparisons ($eq/$in/$ne/$nin),
  $exists presence states, the ``$ne``/``$nin`` MISSING-INCLUSION trap (the #1223
  vector — a provenance-blank chunk must SURVIVE a ``$ne`` because the key is ABSENT,
  never matched-against), and present-JSON-null-vs-absent metadata.

The shared symbols the live cell modules import from here: ``_assert_channel_cell``,
``_SeedDoc``, ``_MATRIX_ROWS``, ``_OPERATOR_CLASSES``, ``_CELLS``, ``_content``,
``_lower_entity_floor``, ``_PG_EMBED_DIM``, ``_violating`` / ``_satisfying``.

Run the meta-tests (no infra)::

    uv run pytest tests/recall/test_channel_filter_operator_matrix.py -m unit -p no:cacheprovider
"""

from __future__ import annotations

import ast
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from khora import Khora
from khora.filter import RecallFilter, parse_to_ast
from khora.filter.execute import iter_leaf_clauses
from khora.search_mode import SearchMode
from tests.e2e import _harness
from tests.recall.test_filter_enforcement_audit_gate import _FILTER_AWARE_METHODS
from tests.test_helpers.filter_spy import plan_extraction, spy_on

# The Postgres pgvector column is fixed at 1536, so the live-DB cell fixtures size
# their deterministic vectors at 1536. Exported for the integration cell modules.
_PG_EMBED_DIM = 1536

# --------------------------------------------------------------------------- #
# REGISTRY — the single source of truth this whole matrix (both live modules)
# keys off. Declared ONLY here.
# --------------------------------------------------------------------------- #

# The candidate-gating channels the matrix VERIFIES BEHAVIORALLY — the
# post-filter-PRONE / historically-post-filter seams where a filter-violating
# chunk could reach RRF fusion if enforcement regressed. This is NOT "channels
# that post-filter in memory TODAY": post-#1236 the recency channel PUSHES the
# filter into the khora_chunks SQL, and the graph channel is HYBRID (pushes the
# system/provenance slice via compile_cypher, post-filters metadata via
# compile_python). The matrix keeps both as rows because its per-cell assertion is
# mechanism-AGNOSTIC — violating chunk absent + honest report + pinned key in
# ``pushed_keys ∪ post_filtered_keys`` — so a row stays valid whether the channel
# pushes down or post-filters, and the recency row in particular is the #1223
# behavioral guard that catches any regression back to in-memory post-filtering
# (what the sibling change module's recency pre-fix proof demonstrates). Each
# member carries an inline mechanism note in ``_POST_FILTER_CHANNEL_NOTES``. Every
# OTHER ``_FILTER_AWARE_METHODS`` member is a pushdown channel, a dispatcher, or a
# transient-fault arm — classified (with a reason) in ``_EXCLUDED_CHANNELS``.
_POST_FILTER_CHANNEL_NOTES: dict[str, str] = {
    "_recency_channel_chunks": (
        "The #1223 channel. NOW PUSHES (post-#1236): it compiles filter_ast into the "
        "khora_chunks WHERE (the same raise-mode compile the vector path uses), so the "
        "recency cell asserts channels['recency'].pushed_keys == [leaf] and "
        "post_filtered_keys == []. It WAS an in-memory post-filter over a provenance-blank "
        "Chunk (the leak); the matrix's recency row is the behavioral guard against a "
        "regression back to that — the per-cell assertion is mechanism-agnostic, so it "
        "stays green while the channel pushes and goes red the instant a violating chunk "
        "leaks again."
    ),
    "_fetch_chunks_from_entities": (
        "HYBRID over-fetch seam. The system/provenance slice PUSHES via compile_cypher "
        "(those leaves land in the graph channel's pushed_keys); metadata sub-path leaves "
        "are unpushable to Cypher and POST-FILTER via compile_python (they land in "
        "post_filtered_keys). So the graph cell asserts the pinned key in "
        "pushed_keys ∪ post_filtered_keys (pushed for A/B/C provenance columns, "
        "post-filtered for the D metadata column) — never hard-requiring one bucket."
    ),
}
_POST_FILTER_CHANNELS: frozenset[str] = frozenset(_POST_FILTER_CHANNEL_NOTES)

# The remaining ``_FILTER_AWARE_METHODS`` members, each mapped to the reason it
# is NOT a post-filter seam. Together with ``_POST_FILTER_CHANNELS`` this is a
# total, disjoint partition of ``_FILTER_AWARE_METHODS`` (asserted below). A new
# filter-aware channel that lands in neither set fails the partition meta-test
# until it is consciously classified.
_EXCLUDED_CHANNELS: dict[str, str] = {
    "_vector_search_chunks": (
        "Pushdown channel: forwards filter_ast to self._vector_store.search("
        "filter_ast=, filter_plan_out=) — a pure khora_chunks WHERE pushdown, no "
        "in-memory over-fetch. It is also the vehicle for the session fan-out + "
        "CHANGE-decomposition behaviors (tested as behavioral rows in _MATRIX_ROWS), "
        "but those enforce VIA the same pushdown, so the method is not a post-filter seam."
    ),
    "_bm25_search_chunks": (
        "Pushdown channel: search_fulltext compiles filter_ast to a khora_chunks WHERE "
        "(on_unsupported='raise') — no in-memory over-fetch."
    ),
    "_typed_entity_recent_retrieve": (
        "Gated OFF whenever a caller filter is present: the dispatch guard is "
        "`... and filter_ast is None`. A filtered recall never enters it, so it can "
        "never be the seam a filter-violating chunk slips through."
    ),
    "_vector_only_fallback": (
        "Fires only inside the `except _NEO4J_TRANSIENT_ERRORS` arm — a Neo4j transient "
        "fault, not a normal candidate channel. Documented known gap; not a steady-state "
        "post-filter seam."
    ),
    "retrieve": ("Top-level orchestrator/dispatcher — routes to the sub-paths; not a candidate-gating channel itself."),
    "_vectorcypher_retrieve": (
        "Complex-path dispatcher that fans out to the real channels (vector / bm25 / "
        "recency / graph); not a candidate-gating channel itself."
    ),
    "_simple_retrieve": (
        "Simple/VECTOR/KEYWORD dispatcher — delegates to the pushdown vector / BM25 channels; not a post-filter seam."
    ),
}


@dataclass(frozen=True)
class _MatrixRow:
    """One row of the channel × operator matrix.

    ``enforcing_method`` is the ``retriever.py`` method whose seam this row drives
    a filter-violating chunk through. ``is_partition_member`` is True iff that
    method is a post-filter seam (a member of :data:`_POST_FILTER_CHANNELS` — the
    post-filter-PRONE / historically-post-filter sense, NOT "post-filters in memory
    today": recency now pushes, graph is hybrid — see ``_POST_FILTER_CHANNEL_NOTES``).
    The two behavioral rows (session / change) enforce via the pushdown
    ``_vector_search_chunks`` and so are False.
    """

    enforcing_method: str
    is_partition_member: bool


# The four matrix rows. The two post-filter-seam rows (recency / graph) have their
# cells in tests/integration/test_channel_filter_operator_matrix.py; the two
# behavioral rows (session / change) in the ``_change`` sibling there — but the
# registry is declared once, here, so both modules and the meta-tests share one
# definition.
_MATRIX_ROWS: dict[str, _MatrixRow] = {
    "recency": _MatrixRow(enforcing_method="_recency_channel_chunks", is_partition_member=True),
    "graph": _MatrixRow(enforcing_method="_fetch_chunks_from_entities", is_partition_member=True),
    "session": _MatrixRow(enforcing_method="_vector_search_chunks", is_partition_member=False),
    "change": _MatrixRow(enforcing_method="_vector_search_chunks", is_partition_member=False),
}


@dataclass(frozen=True)
class _OperatorClass:
    """One operator-class column of the matrix.

    ``key`` is the short column id (A/B/C/D). ``description`` names the filter
    shape and the enforcement trap it probes. The cells construct the concrete
    filter spec inline (per row), so the class itself carries no spec — only the
    contract a cell of this column must honor.
    """

    key: str
    description: str


# The operator-class columns. A/B/C/D — chosen as the filter shapes most likely
# to mis-enforce at a post-filter boundary.
_OPERATOR_CLASSES: dict[str, _OperatorClass] = {
    "A": _OperatorClass(
        key="A",
        description=(
            "Provenance-key relational comparison ($eq / $in / $ne / $nin) over a "
            "denormalized document column (source_name / source_type / source_url / "
            "external_id / content_type / source / title). The #1223 vector is $ne over "
            "source_name."
        ),
    ),
    "B": _OperatorClass(
        key="B",
        description="$exists true / false over a provenance key (presence vs absence).",
    ),
    "C": _OperatorClass(
        key="C",
        description=(
            "$ne / $nin MISSING-INCLUSION: the violating chunk has the key SET to the "
            "excluded value; the satisfying chunk has the key ABSENT and MUST SURVIVE "
            "(a $ne never matches an absent key — the exact #1223 leak shape)."
        ),
    ),
    "D": _OperatorClass(
        key="D",
        description=(
            "metadata present-JSON-null vs absent: metadata.<k>: null present on one "
            "doc vs the key omitted on the other (a metadata sub-path the graph channel "
            "post-filters, not pushes)."
        ),
    ),
}


# Per-row column coverage — which operator-class columns each row declares a cell
# for. "Present" in the coverage meta-test means "declared here AND collected",
# never the full all-rows × all-columns cross-product (most cells of which are
# not meaningful — e.g. session/change cover A as the representative relational
# column only). The recency + graph rows cover the full A/B/C/D set; the
# behavioral session/change rows cover A.
_CELLS: dict[str, frozenset[str]] = {
    "recency": frozenset({"A", "B", "C", "D"}),
    "graph": frozenset({"A", "B", "C", "D"}),
    "session": frozenset({"A"}),
    "change": frozenset({"A"}),
}


# --------------------------------------------------------------------------- #
# REGISTRY / PARTITION META-TESTS — pytest.mark.unit, hermetic. Run in CI's
# ``test-unit`` job (``pytest tests/recall/``). The always-runs leak backstop.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_partition_total_and_disjoint() -> None:
    """The two seam-sets totally and disjointly partition the filter-aware methods.

    ``_POST_FILTER_CHANNELS`` ∪ ``_EXCLUDED_CHANNELS.keys()`` == the audit gate's
    ``_FILTER_AWARE_METHODS`` (currently 2 + 7 = 9), and the two sets are disjoint.
    A new filter-aware channel added to ``retriever.py`` (which ``test_param_set_is_current``
    in the audit gate forces into ``_FILTER_AWARE_METHODS``) lands in NEITHER set
    until it is classified, failing this — the new-channel backstop.
    """
    post = set(_POST_FILTER_CHANNELS)
    excluded = set(_EXCLUDED_CHANNELS)
    assert not (post & excluded), (
        f"a channel is BOTH a post-filter seam and excluded: {sorted(post & excluded)} — classify it as exactly one"
    )
    assert post | excluded == set(_FILTER_AWARE_METHODS), (
        "the seam classification drifted from the audit gate's filter-aware method set.\n"
        f"  in _FILTER_AWARE_METHODS but unclassified: {sorted(set(_FILTER_AWARE_METHODS) - post - excluded)}\n"
        f"  classified here but not filter-aware: {sorted(post | excluded - set(_FILTER_AWARE_METHODS))}\n"
        "Add the new channel to _POST_FILTER_CHANNELS (it is a post-filter seam) or to "
        "_EXCLUDED_CHANNELS with a reason (it is a pushdown / dispatcher / fault arm)."
    )


@pytest.mark.unit
def test_post_filter_members_are_partition_rows() -> None:
    """The partition-member rows map onto the post-filter seams exactly.

    Every ``_MATRIX_ROWS`` row flagged ``is_partition_member`` has an
    ``enforcing_method`` in ``_POST_FILTER_CHANNELS``, and collectively the
    partition-member rows' methods cover ``_POST_FILTER_CHANNELS`` exactly — so a
    seam without a matrix row (or a partition row that is not a seam) fails.
    """
    member_methods = {row.enforcing_method for row in _MATRIX_ROWS.values() if row.is_partition_member}
    assert member_methods == set(_POST_FILTER_CHANNELS), (
        "partition-member rows do not map onto the post-filter seams exactly.\n"
        f"  seams with no partition-member row: {sorted(set(_POST_FILTER_CHANNELS) - member_methods)}\n"
        f"  partition rows whose method is not a seam: {sorted(member_methods - set(_POST_FILTER_CHANNELS))}"
    )
    # And the non-member rows must NOT name a post-filter seam (they enforce via
    # the pushdown channel) — keeps the behavioral/seam split unambiguous.
    non_member_methods = {row.enforcing_method for row in _MATRIX_ROWS.values() if not row.is_partition_member}
    assert not (non_member_methods & set(_POST_FILTER_CHANNELS)), (
        f"a behavioral (non-partition) row names a post-filter seam: "
        f"{sorted(non_member_methods & set(_POST_FILTER_CHANNELS))} — it should enforce via the pushdown channel"
    )


@pytest.mark.unit
def test_excluded_have_reasons() -> None:
    """Every excluded (non-seam) channel carries a non-empty justification."""
    for method, reason in _EXCLUDED_CHANNELS.items():
        assert reason and reason.strip(), f"excluded channel {method!r} has no justification"


@pytest.mark.unit
def test_post_filter_channels_have_mechanism_notes() -> None:
    """Every post-filter seam carries a non-empty inline mechanism note.

    The seam set is defined as the KEYS of ``_POST_FILTER_CHANNEL_NOTES``, so this
    cannot drift — but the note must be non-empty (the same honesty bar as
    ``test_excluded_have_reasons``). The note records the channel's CURRENT pushdown
    vs post-filter mechanism (recency PUSHES post-#1236; graph is HYBRID), which is
    what keeps the per-cell ``pushed_keys ∪ post_filtered_keys`` assertion honest.
    """
    assert set(_POST_FILTER_CHANNEL_NOTES) == set(_POST_FILTER_CHANNELS), (
        "the mechanism-note keys drifted from the post-filter seam set"
    )
    for method, note in _POST_FILTER_CHANNEL_NOTES.items():
        assert note and note.strip(), f"post-filter seam {method!r} has no mechanism note"


@pytest.mark.unit
def test_param_set_is_current() -> None:
    """Re-derive the filter-aware method set from ``retriever.py`` source.

    Independent of the audit gate's own copy: walk ``retriever.py`` for every
    method whose signature declares a ``filter_ast`` parameter and assert that set
    equals ``_FILTER_AWARE_METHODS``. This is the new-channel backstop AT THE
    SOURCE — a method gaining a ``filter_ast`` param (a new filter-aware channel)
    fails here until ``_FILTER_AWARE_METHODS`` (and therefore the partition above)
    is updated. Mirrors the audit gate's ``test_param_set_is_current``.
    """
    retriever_src = _project_root() / "src" / "khora" / "engines" / "vectorcypher" / "retriever.py"
    tree = ast.parse(retriever_src.read_text())
    declared_in_source: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            params = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
            if "filter_ast" in params:
                declared_in_source.add(node.name)
    assert declared_in_source == set(_FILTER_AWARE_METHODS), (
        "filter-aware method set drifted from retriever.py source.\n"
        f"  in source but not in _FILTER_AWARE_METHODS: {sorted(declared_in_source - set(_FILTER_AWARE_METHODS))}\n"
        f"  in _FILTER_AWARE_METHODS but not in source: {sorted(set(_FILTER_AWARE_METHODS) - declared_in_source)}\n"
        "A new filter-aware channel must be threaded, audited, AND classified in this module's registry."
    )


# Which live integration module owns each row's cells. The recency / graph rows'
# cells live in ``tests/integration/test_channel_filter_operator_matrix.py``; the
# behavioral session / change rows' cells live in its ``_change`` sibling. The
# coverage meta-test (:func:`test_every_matrix_row_has_a_cell`) AST-scans these
# files for the convention-named ``test_<row>_<col>_cell`` functions — so a
# declared ``_CELLS`` column with no collected cell fails RED in the hermetic
# ``test-unit`` job. It is a source scan, not an import (the live modules pull in DB
# fixtures + an ``integration`` skipif, so they can't be imported here), mirroring
# the audit gate's approach. ALL FOUR rows are gated HERE — there is no separate
# per-module presence test — so this hermetic file is the single always-on
# cell-presence backstop for the whole matrix.
_CELL_MODULES: dict[str, str] = {
    "recency": "tests/integration/test_channel_filter_operator_matrix.py",
    "graph": "tests/integration/test_channel_filter_operator_matrix.py",
    "session": "tests/integration/test_channel_filter_operator_matrix_change.py",
    "change": "tests/integration/test_channel_filter_operator_matrix_change.py",
}


def _async_def_names(path) -> set[str]:
    """The set of ``async def`` function names declared at module scope in ``path``."""
    tree = ast.parse(path.read_text())
    return {n.name for n in tree.body if isinstance(n, ast.AsyncFunctionDef)}


@pytest.mark.unit
def test_registry_is_coherent() -> None:
    """The registry maps rows ↔ columns ↔ recipes ↔ cell-modules consistently.

    (a) every ``_MATRIX_ROWS`` key has a ``_CELLS`` entry; (b) every column a row
    declares is a known ``_OPERATOR_CLASSES`` key; (c) every row names a known
    recipe in ``_RECIPES`` and a live cell module in ``_CELL_MODULES``. The
    per-(row, column) cell-function presence is checked in
    :func:`test_every_matrix_row_has_a_cell`.
    """
    assert set(_CELLS) == set(_MATRIX_ROWS), f"_CELLS rows {sorted(_CELLS)} != _MATRIX_ROWS rows {sorted(_MATRIX_ROWS)}"
    assert set(_CELL_MODULES) == set(_MATRIX_ROWS), (
        f"_CELL_MODULES rows {sorted(_CELL_MODULES)} != _MATRIX_ROWS rows {sorted(_MATRIX_ROWS)}"
    )
    for row, cols in _CELLS.items():
        assert cols, f"row {row!r} declares no operator-class columns"
        unknown = cols - set(_OPERATOR_CLASSES)
        assert not unknown, f"row {row!r} declares unknown operator-class column(s): {sorted(unknown)}"
        assert row in _RECIPES, f"row {row!r} has no recipe in _RECIPES"


@pytest.mark.unit
def test_every_matrix_row_has_a_cell() -> None:
    """Every declared (row, column) cell has a real ``async def`` in its live module.

    Self-contained coverage gate for ALL FOUR rows (recency / graph in the live
    cell module, session / change in the ``_change`` sibling). It AST-scans both
    integration cell-module files for the convention-named ``test_<row>_<col>_cell``
    ``async def`` — a SOURCE scan, not an import, because the live modules carry DB
    fixtures + an ``integration`` skipif and so can't be imported in this hermetic
    ``test-unit`` job (mirrors the audit gate's source-scan approach). A declared
    ``_CELLS`` column with no collected cell fails RED here (the silent
    under-coverage guard), so the whole matrix's coverage is gated in one always-on
    unit test.
    """
    root = _project_root()
    cell_fns_by_module: dict[str, set[str]] = {}
    for row, cols in _CELLS.items():
        module_rel = _CELL_MODULES[row]
        module_path = root / module_rel
        assert module_path.exists(), f"live cell module {module_rel!r} for row {row!r} does not exist"
        if module_rel not in cell_fns_by_module:
            cell_fns_by_module[module_rel] = _async_def_names(module_path)
        present = cell_fns_by_module[module_rel]
        for col in sorted(cols):
            fn_name = f"test_{row}_{col.lower()}_cell"
            assert fn_name in present, (
                f"declared cell ({row!r}, column {col!r}) has no ``async def {fn_name}`` in "
                f"{module_rel!r} — declare the cell or drop the column from _CELLS[{row!r}]"
            )


def _project_root():
    """Repo root (``…/khora``) from this test file's location."""
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# CELL DRIVER — the shared per-cell assertion helper. DEFINED here (hermetic);
# EXECUTED only by the live cell modules under tests/integration/.
# --------------------------------------------------------------------------- #


# The external_id keys both docs carry. The survivor reconciliation + the recency
# prepare-hook key off these.
_VIOLATING_EXTERNAL_ID = "violating"
_SATISFYING_EXTERNAL_ID = "satisfying"


@dataclass(frozen=True)
class _SeedDoc:
    """One doc a cell seeds — content + the provenance / metadata surface.

    ``remember_kwargs`` are threaded straight into ``Khora.remember`` (e.g.
    ``source_name=`` / ``metadata=``). ``external_id`` is the survivor-reconciliation
    key — the helper asserts a cell's survivors / absences by this id, and the
    recency prepare-hook backdates the doc keyed ``"violating"`` to be the freshest.
    ``content`` carries the shared entity marker + query keyword so the doc is
    recallable on the channel under test; cells differentiate it per-doc where the
    row needs it (e.g. CHANGE needs a CHANGED attribute across the two docs).
    """

    external_id: str
    content: str
    remember_kwargs: dict[str, Any]


@dataclass(frozen=True)
class _Recipe:
    """Per-row execution recipe consumed by :func:`_assert_channel_cell`.

    * ``mode`` — the ``SearchMode`` that deterministically routes to the row's
      channel (GRAPH for all four rows — every row needs the graph entry-entity
      path; HYBRID would let the router fall to ``_simple_retrieve``).
    * ``configure`` — a callable applied to the connected ``kb`` to turn the
      channel ON / lower floors (e.g. enable the recency channel).
    * ``spy_method`` — the retriever method to install the non-vacuity spy on.
    * ``assert_fired`` — the per-recipe non-vacuity signal (spy fired AND the
      channel contributed), given the spy record list + the recall result.
    * ``prepare`` — an optional async hook run AFTER seeding (e.g. backdate
      ``occurred_at`` for the recency axis), given ``(kb, namespace_id, seeded)``
      where ``seeded`` is the ``{external_id: document_id}`` map.

    The row-specific FIRING (session needs the entity across >= 2 channels; CHANGE
    needs a SUPERSEDES edge) is encoded by the CELL in what it puts on the two
    :class:`_SeedDoc`\\ s (distinct ``metadata.channel`` / changed content) — NOT by
    a per-recipe seed strategy. The helper seeds both docs uniformly (once each).
    """

    mode: SearchMode
    configure: Callable[[Khora], None]
    spy_method: str
    assert_fired: Callable[[list[Any], Any], None]
    prepare: Callable[[Khora, UUID, dict[str, str]], Any] | None = None


def _violates(doc: Any, leaf: Any) -> bool:
    """Whether ``doc`` VIOLATES the single pinned filter leaf — cell-shaped, not an engine.

    ``leaf`` is the one :class:`~khora.filter.ast.FilterClause` the cell pinned
    (``path`` / ``op`` / ``operand``). This is a tiny per-cell check of exactly that
    leaf against the surviving document's projection — NOT a general filter engine
    (the engine-independent report invariants are checked separately). It mirrors
    the operand semantics the conformance oracle uses for the handful of operator
    classes the matrix exercises.

    A document SURVIVES (returns False) when the leaf's predicate holds for the
    document's value at ``leaf.path``; it VIOLATES (returns True) otherwise. Missing
    keys read as ``None`` — the MISSING-INCLUSION contract: ``$ne`` / ``$nin`` over an
    absent key SURVIVE (a missing value is never equal to an excluded one), and a
    metadata ``$eq null`` matches an absent key.
    """
    value = _doc_value_at(doc, leaf.path)
    op = leaf.op.value if hasattr(leaf.op, "value") else str(leaf.op)
    operand = leaf.operand

    if op == "$eq":
        return not (value == operand)
    if op == "$ne":
        return not (value != operand)
    if op == "$in":
        return value not in tuple(operand)
    if op == "$nin":
        return value in tuple(operand)
    if op == "$exists":
        present = value is not None
        return present is not bool(operand)
    raise AssertionError(f"_violates does not model operator {op!r}; the cell pinned an unsupported leaf")


def _doc_value_at(doc: Any, path: tuple[str, ...]) -> Any:
    """Read the document projection's value at a filter leaf ``path``.

    A single-segment provenance path (``("source_name",)``) reads the attribute off
    the :class:`DocumentProjection`. A ``("metadata", <k>, …)`` path walks the
    document ``metadata`` blob. A missing attribute / key reads as ``None`` so the
    MISSING-INCLUSION semantics in :func:`_violates` apply uniformly.
    """
    if path and path[0] == "metadata":
        cursor: Any = doc.metadata or {}
        for segment in path[1:]:
            if not isinstance(cursor, dict) or segment not in cursor:
                return None
            cursor = cursor[segment]
        return cursor
    return getattr(doc, path[0], None)


def _pinned_leaf(filter_spec: dict[str, Any] | RecallFilter):
    """The single constraint leaf a cell pinned — asserts the spec has exactly one."""
    model = filter_spec if isinstance(filter_spec, RecallFilter) else RecallFilter.model_validate(filter_spec)
    leaves = list(iter_leaf_clauses(parse_to_ast(model)))
    assert len(leaves) == 1, f"a cell filter must pin exactly one constraint leaf; got {len(leaves)}: {filter_spec!r}"
    return leaves[0]


async def _assert_channel_cell(
    kb: Khora,
    *,
    row: str,
    filter_spec: dict[str, Any] | RecallFilter,
    violating_doc: _SeedDoc,
    satisfying_doc: _SeedDoc,
    query: str,
    monkeypatch: pytest.MonkeyPatch,
    expect_satisfying_present: bool,
) -> None:
    """Drive one channel × operator-class cell and assert the four contracts.

    ``row`` is the matrix-row STRING key (``"recency"`` / ``"graph"`` / ``"session"``
    / ``"change"``) — it selects the row's :class:`_Recipe`. ``violating_doc`` /
    ``satisfying_doc`` are :class:`_SeedDoc`\\ s carrying each doc's content +
    ``external_id`` + extra ``remember`` kwargs (the cell's provenance / metadata
    surface). ``filter_spec`` must pin EXACTLY ONE constraint leaf (the cell's
    operator-class probe).

    Steps:

    1. Mint a fresh namespace and select the row's :class:`_Recipe`.
    2. Configure ``kb`` for the channel, lower the entity-similarity floor, and
       stage the shared extractor plan so the seeded docs populate the graph.
    3. Seed the satisfying then violating doc once each (the row-specific firing is
       encoded by the cell in the two ``_SeedDoc``\\ s).
    4. Run the recipe's ``prepare`` hook (e.g. recency backdating).
    5. Pre-flight: a no-filter GRAPH recall must return entities (else the channel
       never fires and every assertion is vacuous).
    6. Install the non-vacuity spy at the row's seam and run the FILTERED recall.

    Then assert, against the result:

    * **(4) NON-VACUITY** — the recipe's ``assert_fired`` (spy fired + channel
      contributed). A green cell can never be a channel that did not run.
    * **(1) ENFORCEMENT** — no surviving chunk's document violates the pinned leaf;
      if ``expect_satisfying_present`` the satisfying doc is among the survivors.
    * **(2) NO PRIVATE LEAK** — ``"_filter_channel_plans"`` is absent from
      ``engine_info`` (the private carrier never escapes).
    * **(3) REPORT INVARIANTS** — ``engine_info["filter"]`` obeys the
      engine-independent invariants for the spec's leaves, and the pinned leaf is
      accounted for (pushed OR post-filtered — never silently unenforced).
    """
    recipe = _RECIPES[row]
    leaf = _pinned_leaf(filter_spec)

    ns = await kb.create_namespace()
    namespace_id: UUID = ns.namespace_id

    recipe.configure(kb)
    _lower_entity_floor(kb)

    # Stage the shared entity for the marker so the seeded docs yield real
    # MENTIONED_IN edges (the graph entry-entity path every row needs).
    plan_extraction(_MARKER, entities=[(_ENTITY_NAME, "PERSON")])

    # Seed both docs once each (satisfying first, so it is older on the recency
    # axis before the violating doc is backdated to be freshest). The survivor map
    # is {external_id: document_id}.
    seeded: dict[str, str] = {}
    for doc in (satisfying_doc, violating_doc):
        remembered = await kb.remember(
            content=doc.content,
            namespace=namespace_id,
            external_id=doc.external_id,
            entity_types=["PERSON"],
            relationship_types=[],
            **doc.remember_kwargs,
        )
        seeded[doc.external_id] = str(remembered.document_id)

    if recipe.prepare is not None:
        await recipe.prepare(kb, namespace_id, seeded)

    # Pre-flight entry-entity gate — a GRAPH recall for the entity surface must
    # return entities, else the channel never fires and the cell passes vacuously.
    await _harness.assert_graph_contributes(kb, namespace_id, _ENTITY_NAME)

    # Install the non-vacuity spy at the row's enforcing seam.
    retriever = kb._engine._retriever  # type: ignore[union-attr]
    spy = spy_on(monkeypatch, retriever, recipe.spy_method)

    result = await kb.recall(
        query,
        namespace=namespace_id,
        limit=20,
        mode=recipe.mode,
        filter=filter_spec,
    )

    # (4) NON-VACUITY — the channel under test actually fired and contributed.
    recipe.assert_fired(spy, result)

    # (1) ENFORCEMENT — no surviving chunk's document violates the pinned leaf.
    docs_by_id = {doc.id: doc for doc in result.documents}
    survivor_external_ids: set[str | None] = set()
    for chunk in result.chunks:
        doc = docs_by_id.get(chunk.document_id)
        assert doc is not None, (
            f"chunk {chunk.id} references document {chunk.document_id} missing from result.documents"
        )
        assert not _violates(doc, leaf), (
            f"filter-violating chunk leaked through the {row!r} channel: "
            f"doc external_id={doc.external_id!r}, value at {leaf.path}={_doc_value_at(doc, leaf.path)!r}, "
            f"leaf op={leaf.op}, operand={leaf.operand!r}"
        )
        survivor_external_ids.add(doc.external_id)

    if expect_satisfying_present:
        assert satisfying_doc.external_id in survivor_external_ids, (
            f"the satisfying doc {satisfying_doc.external_id!r} was dropped — over-filtering. "
            f"survivors={sorted(x for x in survivor_external_ids if x is not None)}"
        )

    # (2) NO PRIVATE LEAK — the private channel-plan carrier never escapes.
    assert "_filter_channel_plans" not in result.engine_info, (
        "the private _filter_channel_plans carrier leaked into public engine_info"
    )

    # (3) REPORT INVARIANTS — the emitted filter report is honest for the spec's
    # leaves, and the pinned leaf is accounted for (pushed OR post-filtered —
    # mechanism-AGNOSTIC: never hard-requires a non-empty post_filtered_keys, so a
    # pushed-down cell is just as valid as a post-filtered one).
    report = result.engine_info["filter"]
    leaves = _harness.filter_spec_leaves(filter_spec)
    _harness.assert_filter_report_invariants(report, leaves)
    pinned_key = ".".join(leaf.path)
    assert pinned_key in (set(report["pushed_keys"]) | set(report["post_filtered_keys"])), (
        f"the pinned leaf {pinned_key!r} is neither pushed nor post-filtered in the report "
        f"(pushed={report['pushed_keys']}, post_filtered={report['post_filtered_keys']}, "
        f"unenforced={report['unenforced_keys']}) — it slipped past every channel"
    )

    # MECHANISM — the channel under test gated the pinned leaf in the bucket its
    # CURRENT mechanism implies (the per-cell counterpart to the registry's
    # ``_POST_FILTER_CHANNEL_NOTES``). For recency this is the post-#1236 PUSH guard
    # (channels["recency"].pushed_keys == leaves, post_filtered_keys == []); for the
    # hybrid graph channel it asserts the channel ADDRESSED every leaf (pushed ∪
    # post-filtered == leaves) without pinning the bucket (provenance push vs
    # metadata post-filter is column-dependent — the cell's column already knows
    # which). Skipped on rows whose channel name is not in the report (the
    # behavioral session/change rows enforce via the unnamed pushdown vector
    # channel, already covered by the top-level partition check above).
    channel_name = {"recency": "recency", "graph": "graph"}.get(row)
    if channel_name is not None and channel_name in report["channels"]:
        chan = report["channels"][channel_name]
        chan_keys = set(chan["pushed_keys"]) | set(chan["post_filtered_keys"])
        assert chan_keys == leaves, (
            f"the {channel_name!r} channel addressed {sorted(chan_keys)} but the filter leaves are "
            f"{sorted(leaves)} — the channel under test did not gate exactly the cell's leaf"
        )
        if row == "recency":
            assert set(chan["pushed_keys"]) == leaves and chan["post_filtered_keys"] == [], (
                "post-#1236 recency PUSH mechanism: expected "
                f"pushed_keys == {sorted(leaves)} / post_filtered_keys == [], got "
                f"pushed={chan['pushed_keys']} / post_filtered={chan['post_filtered_keys']}"
            )


# --------------------------------------------------------------------------- #
# Shared seed constants — one entity surface + marker drives the graph path for
# every row; one query keyword keeps both docs recallable.
# --------------------------------------------------------------------------- #

_ENTITY_NAME = "Falcon"
_MARKER = "graphdoc"
# Both docs share this keyword so they clear the recency cosine floor and surface
# on the entity-driven graph path.
_QUERY = "latest falcon launch update"


def _content(suffix: str) -> str:
    """Doc content carrying the entity, the graph marker, and the query keyword."""
    return f"{_ENTITY_NAME} {_MARKER} falcon launch update: {suffix}."


def _violating(suffix: str, **remember_kwargs: Any) -> _SeedDoc:
    """A :class:`_SeedDoc` for the doc that VIOLATES the cell's filter (the leak vector).

    Carries the ``"violating"`` external id (the recency prepare-hook backdates this
    doc to be the freshest on the recency axis, so a dropped filter would surface it).
    """
    return _SeedDoc(external_id=_VIOLATING_EXTERNAL_ID, content=_content(suffix), remember_kwargs=remember_kwargs)


def _satisfying(suffix: str, **remember_kwargs: Any) -> _SeedDoc:
    """A :class:`_SeedDoc` for the doc that SATISFIES the cell's filter (must survive)."""
    return _SeedDoc(external_id=_SATISFYING_EXTERNAL_ID, content=_content(suffix), remember_kwargs=remember_kwargs)


# --------------------------------------------------------------------------- #
# Channel configuration + floor helpers.
# --------------------------------------------------------------------------- #


def _lower_entity_floor(kb: Khora) -> None:
    """Lower the VectorCypher entity-similarity floor to 0 on a connected kb.

    The deterministic hash embedder's vectors carry no semantic meaning, so a
    query↔entity cosine sits below the default floor and entity vector search
    returns nothing — short-circuiting every graph path. Lowering the floor lets
    the seeded entities clear it. Test-fixture knob, not a product change (mirrors
    the integration fixtures' ``_retriever`` floor-lowering).
    """
    retriever = getattr(kb._engine, "_retriever", None)
    if retriever is not None and getattr(retriever, "_config", None) is not None:
        retriever._config.min_entity_similarity = 0.0


def _configure_recency(kb: Khora) -> None:
    """Turn the recency channel ON for the recency-row cells.

    The recency channel is OFF by default; enable it and set a relevance floor low
    enough that both keyword-sharing docs clear the cosine gate, with reranking off
    so the deterministic ordering is undisturbed. Mirrors the integration recency
    fixture (``test_vectorcypher_recency_channel_pg.py``). The recency cell fixture
    already sets these at build time, so this is idempotent there.
    """
    cfg = kb._engine._retriever._config  # type: ignore[union-attr]
    cfg.temporal_recency_channel_enabled = True
    cfg.temporal_query_relevance_floor = 0.30


def _configure_graph(kb: Khora) -> None:
    """No extra channel flag for the graph row — the over-fetch path is always on.

    The graph channel runs on every ``mode=GRAPH`` complex recall; only the entity
    floor (lowered in ``_assert_channel_cell``) gates it. Kept as an explicit no-op
    configure so every row has a uniform recipe shape.
    """
    return None


# --------------------------------------------------------------------------- #
# Non-vacuity signals — per-row "the channel fired AND contributed".
# --------------------------------------------------------------------------- #


def _assert_recency_fired(spy: list[Any], result: Any) -> None:
    """Recency NON-VACUITY: the spy fired AND the recency channel gated in RRF.

    Two things, so a "pushed-not-post-filtered" cell still proves the channel FIRED
    (answering the Devil's-Advocate concern with a real fired-signal, not by
    dropping the row):

    1. ``_recency_channel_chunks`` ran (spy captured >= 1).
    2. The recency channel recorded a ChannelPlan in the report — which it does ONLY
       when surviving recency candidates GATE in RRF (retriever.py). So this is a
       genuine "the channel produced a gating, filter-honoring result", never an
       always-on artifact.

    The post-#1236 PUSH mechanism guard (pushed_keys == leaves, post_filtered_keys
    == []) lives in :func:`_assert_channel_cell`'s MECHANISM block (it has the
    filter leaves there), so a regression back to in-memory post-filtering fails
    there OR leaks a violating chunk into the enforcement assertion.
    """
    assert len(spy) >= 1, "non-vacuity: _recency_channel_chunks was never called — the recency channel did not run"
    channels = result.engine_info["filter"]["channels"]
    assert "recency" in channels, (
        "non-vacuity: the recency channel produced no surviving gating candidate, so it "
        f"recorded no ChannelPlan; channels={list(channels)}"
    )


def _assert_graph_fired(spy: list[Any], result: Any) -> None:
    """Graph NON-VACUITY: the fetch spy fired AND the graph channel contributed.

    ``_fetch_chunks_from_entities`` ran (spy captured >= 1), AND the graph channel
    held candidates this recall (the ``"graph"`` channel is recorded in the report
    only when it had candidates to post-filter — retriever.py). So a green here
    proves the over-fetch seam both ran and the graph channel contributed — a real
    fired-signal even when the cell's leaf is PUSHED (provenance columns A/B/C),
    answering the Devil's-Advocate concern without dropping the row.

    The HYBRID mechanism accounting (the graph channel addressed exactly the leaf,
    via pushed_keys for provenance / post_filtered_keys for metadata) lives in
    :func:`_assert_channel_cell`'s MECHANISM block, which has the filter leaves.
    """
    assert len(spy) >= 1, "non-vacuity: _fetch_chunks_from_entities was never called — the graph over-fetch did not run"
    channels = result.engine_info["filter"]["channels"]
    assert "graph" in channels, (
        f"non-vacuity: the graph channel held no candidates, so it recorded no ChannelPlan; channels={list(channels)}"
    )


def _assert_session_fired(spy: list[Any], result: Any) -> None:
    """Session non-vacuity: fan-out issued >= 2 per-channel vector searches.

    ``_vector_search_chunks`` ran at least twice (one per fanned-out channel + the
    unscoped fallback) — fan-out requires >= 2 channels, so a single capture means
    the fan-out never engaged and the cell would be vacuous.
    """
    assert len(spy) >= 2, (
        f"non-vacuity: session fan-out issued {len(spy)} _vector_search_chunks call(s), expected >= 2 "
        "(per-channel + fallback) — the fan-out did not engage"
    )


def _assert_change_fired(spy: list[Any], result: Any) -> None:
    """CHANGE non-vacuity: the decomposition issued a 2nd vector search.

    ``_vector_search_chunks`` ran at least twice (the original + the decomposed
    current-state sub-search). A single capture means the CHANGE decomposition never
    fired (no version history), so the cell would be vacuous.
    """
    assert len(spy) >= 2, (
        f"non-vacuity: CHANGE decomposition issued {len(spy)} _vector_search_chunks call(s), expected >= 2 "
        "(original + decomposed sub-search) — the decomposition did not fire"
    )


# --------------------------------------------------------------------------- #
# Recency backdating prepare-hook — make the violating doc the freshest on the
# recency axis so a dropped filter WOULD surface it.
# --------------------------------------------------------------------------- #

_DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if _DATABASE_URL.startswith("postgresql://"):
    _DATABASE_URL = _DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _DATABASE_URL.startswith("postgres://"):
    _DATABASE_URL = _DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


async def _backdate_recency(kb: Khora, namespace_id: UUID, seeded: dict[str, str]) -> None:
    """Backdate ``occurred_at`` so the VIOLATING doc is the newest on the recency axis.

    Single-doc ``remember`` does not stamp ``occurred_at``, so the recency order is
    ambiguous. Mirror the integration recency test's direct-SQL pattern: push the
    whole namespace back, then bring the violating doc to 1 day ago (freshest). If
    the recency channel dropped the filter, the violating doc — being the freshest —
    would top the recency candidate list and leak. The satisfying doc stays
    slightly older but still inside the relevance window.
    """
    eng = create_async_engine(_DATABASE_URL)
    try:
        async with eng.begin() as conn:
            await conn.execute(
                text("UPDATE khora_chunks SET occurred_at = NOW() - INTERVAL '20 days' WHERE namespace_id = :ns"),
                {"ns": str(namespace_id)},
            )
            await conn.execute(
                text(
                    "UPDATE khora_chunks SET occurred_at = NOW() - INTERVAL '1 day' "
                    "WHERE namespace_id = :ns AND document_id = :doc"
                ),
                {"ns": str(namespace_id), "doc": seeded[_VIOLATING_EXTERNAL_ID]},
            )
    finally:
        await eng.dispose()


# --------------------------------------------------------------------------- #
# Recipe registry — one per row. The live cell modules reach these by calling
# ``_assert_channel_cell(row=..., ...)``.
# --------------------------------------------------------------------------- #

_RECIPES: dict[str, _Recipe] = {
    "recency": _Recipe(
        mode=SearchMode.GRAPH,
        configure=_configure_recency,
        spy_method="_recency_channel_chunks",
        assert_fired=_assert_recency_fired,
        prepare=_backdate_recency,
    ),
    "graph": _Recipe(
        mode=SearchMode.GRAPH,
        configure=_configure_graph,
        spy_method="_fetch_chunks_from_entities",
        assert_fired=_assert_graph_fired,
        prepare=None,
    ),
    "session": _Recipe(
        mode=SearchMode.GRAPH,
        configure=_configure_graph,
        spy_method="_vector_search_chunks",
        assert_fired=_assert_session_fired,
        prepare=None,
    ),
    "change": _Recipe(
        mode=SearchMode.GRAPH,
        configure=_configure_graph,
        spy_method="_vector_search_chunks",
        assert_fired=_assert_change_fired,
        prepare=None,
    ),
}


# Re-exported so the live cell modules can build their RECENCY query / GRAPH entity
# surface from the same constants the driver seeds with.
_RECENCY_QUERY = _QUERY
_GRAPH_QUERY = _ENTITY_NAME
