"""Documents-target conformance: the oracle leg, the rejected key, and an embedded smoke.

Three things live here, in increasing cost:

1. **The oracle leg** — every case in :func:`documents_conformance_cases` is run
   through :func:`documents_oracle_survivors` and asserted against the ``expected_ids``
   it was authored with on the CHUNK surface. The whole documents corpus is carried
   over verbatim on the claim that a filter which never reads ``occurred_at`` selects
   the same records off a document-row mapping as off a chunk one; this leg is what
   makes that claim falsifiable rather than assumed. Pure Python, no store.
2. **The rejected key** — ``occurred_at`` is the one system key document enumeration
   does not accept, and the corpus encodes it as a *validation outcome* instead of a
   row-set case. These tests drive the real
   :func:`khora.khora._reject_non_enumerable_keys` and pin the structured error,
   including the substitute keys the message names.
3. **The embedded smoke leg** — a representative slice of the corpus driven through
   the REAL ``scan_documents_page`` walk against three in-process stores (raw
   ``backend: sqlite``, embedded ``sqlite_lance``, embedded SurrealDB), in BOTH modes.
   No Docker, no LLM, no embeddings.

The smoke leg is a *slice*, not the corpus: the full 186-case × 2-mode × 4-backend
matrix belongs to ``tests/integration/matrix/test_documents_conformance.py`` behind the
``filter_conformance`` marker. What this leg buys is that the seam itself — seeder,
walk, both modes, all three embedded stores — cannot rot unnoticed between runs of
that job.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from khora.filter import SYSTEM_KEYS
from khora.filter.conformance import (
    _DOCUMENTS_REJECTED_FILTERS,
    ConformanceCase,
    _documents_surreal_excluded,
    documents_conformance_cases,
    documents_oracle_survivors,
    run_case_for_backend,
)

pytestmark = pytest.mark.unit

_CASES = documents_conformance_cases()


# --------------------------------------------------------------------------- #
# 1. The oracle leg.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.id)
def test_documents_oracle_agrees_with_the_declared_expectations(case: ConformanceCase) -> None:
    """Every carried-over case selects the same records off a document-row mapping.

    ``expected_ids`` is hand-authored against the chunk mapping, which synthesizes
    ``occurred_at`` as ``COALESCE(occurred_at, source_timestamp)``; the documents
    mapping has no such key. A case that quietly depended on that synthesis would show
    up here as a mismatch — not as a mysterious failure on a live leg hours later.
    """
    assert documents_oracle_survivors(case) == case.expected_ids


def test_the_corpus_has_unique_case_ids() -> None:
    """Case ids double as namespace keys, so a duplicate would alias two seeds."""
    ids = [case.id for case in _CASES]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    assert not duplicates, f"duplicate documents case ids: {duplicates}"


def test_no_case_filters_the_non_enumerable_key() -> None:
    """``occurred_at`` never appears as a row-set case — only as a rejection.

    The complement of the corpus-assembly rule. ``_documents_eligible`` drops such a
    case at assembly time; this asserts the result rather than trusting the filter.
    """
    from khora.filter.conformance import _resolve_ast
    from khora.filter.execute import iter_leaf_clauses

    offenders = [
        case.id
        for case in _CASES
        if any(
            clause.path and clause.path[0] == "occurred_at" for clause in iter_leaf_clauses(_resolve_ast(case.filter))
        )
    ]
    assert not offenders, f"documents cases filtering the non-enumerable key occurred_at: {offenders}"


# --------------------------------------------------------------------------- #
# 2. The rejected key.
# --------------------------------------------------------------------------- #


def _reject(wire: dict[str, Any]) -> Any:
    """Lower a wire filter through the real validator + ``parse_to_ast``, then reject."""
    from khora.filter import RecallFilter, parse_to_ast
    from khora.khora import _reject_non_enumerable_keys

    return _reject_non_enumerable_keys(parse_to_ast(RecallFilter.model_validate(wire)))


@pytest.mark.parametrize("wire", _DOCUMENTS_REJECTED_FILTERS, ids=("bare", "nested-in-and"))
def test_occurred_at_is_rejected_at_any_depth(wire: dict[str, Any]) -> None:
    """An ``occurred_at`` leaf is a structured validation failure, not a zero-match query.

    Both shapes matter: a bare leaf, and one buried inside an ``$and`` next to a key
    that IS enumerable. The rejection walks every leaf, so the second must fail exactly
    like the first — a depth-1-only check would let the composed form through, where it
    would silently match nothing (no document row backs the column).
    """
    from khora.filter.model import RecallFilterValidationError

    with pytest.raises(RecallFilterValidationError) as excinfo:
        _reject(wire)

    error = excinfo.value.errors[0]
    assert error.path == "occurred_at"
    assert error.code == "key_not_enumerable"
    assert error.allowed == sorted(SYSTEM_KEYS - {"occurred_at"})


def test_the_rejection_names_both_substitutes_and_denies_equivalence() -> None:
    """The message must offer the two time axes AND deny they are the same thing.

    Naming ``source_timestamp`` / ``created_at`` without the denial would read as
    "use this instead", which is wrong: they are the source-provided and ingest times,
    neither of which is the event time. The denial is the load-bearing half, so it is
    asserted rather than left to the reader of the docstring.
    """
    from khora.filter.model import RecallFilterValidationError

    with pytest.raises(RecallFilterValidationError) as excinfo:
        _reject(_DOCUMENTS_REJECTED_FILTERS[0])

    message = excinfo.value.errors[0].message
    assert "source_timestamp" in message
    assert "created_at" in message
    assert "different time axes, not equivalent substitutes" in message


def test_an_enumerable_filter_is_not_rejected() -> None:
    """The gate has to let the nine enumerable keys through, or it proves nothing."""
    assert _reject({"$and": [{"source_name": "linear"}, {"created_at": {"$gt": "2026-01-01T00:00:00Z"}}]}) is None


# --------------------------------------------------------------------------- #
# 3. The embedded smoke leg.
# --------------------------------------------------------------------------- #
#
# One case per distinct shape the documents read path can take, chosen so that a
# regression in any single mechanism turns at least one of them red:
#
# * pushed vs. post-filtered SYSTEM keys — a string key (pushed everywhere) and the two
#   date keys (pushed only on Postgres; TEXT-serialization-unsafe everywhere else, so
#   they land on the post-filter on all three stores here);
# * the F1 negation polarity over a NULL column, and the bare-``null`` match;
# * the documents-only ``external_id`` re-seed (the UNIQUE-constraint variant);
# * metadata: a scalar, a ``$``-prefixed segment (unrenderable as a SurrealQL
#   identifier — under ``"split"`` it must DEFER, which is precisely the bucket the
#   documents surreal prune drops), array containment, and whole-subdoc equality;
# * the all-or-nothing boolean gate — a mixed system+metadata ``$or`` and a
#   ``$not($exists)``, the two shapes a compiler must defer WHOLE or invert;
# * the constant-false bare-list, and a multi-predicate AND.
_SMOKE_CASE_IDS: tuple[str, ...] = (
    "F-OP-source_type-eq",
    "F-OP-source_name-ne",
    "F-OP-title-exists-true",
    "F-OP-external_id-documents-eq",
    "F-OP-external_id-documents-nin",
    "F-OP-created_at-gte",
    "F-OP-source_timestamp-lt",
    "F-OP-metadata-tier-eq",
    "F-EXISTS-md-false",
    "F-LOGIC-mixed-or",
    "F-LOGIC-not-exists",
    "F-DOTKEY-dollar-key",
    "F-NULLVAL-sys-bare-null",
    "F-OBJEQ-exact",
    "F-ARRAY-in-any",
    "F-IMPOSSIBLE-name-barelist",
    "F-SEL-composite",
)

_BY_ID = {case.id: case for case in _CASES}

# backend name -> runner-module import path. Imported LAZILY inside the test so a
# missing optional extra skips its own leg instead of erroring collection for all.
_EMBEDDED_BACKENDS: dict[str, str] = {
    "sqlite": "tests.integration.matrix._conformance_docs_sqlite",
    "sqlite_lance": "tests.integration.matrix._conformance_docs_lance",
    "surrealdb": "tests.integration.matrix._conformance_docs_surreal",
}


def test_every_smoke_case_id_resolves() -> None:
    """A renamed or dropped case must break loudly, not shrink the smoke set silently."""
    missing = [cid for cid in _SMOKE_CASE_IDS if cid not in _BY_ID]
    assert not missing, f"smoke case ids not present in the documents corpus: {missing}"


@pytest.mark.parametrize("backend", sorted(_EMBEDDED_BACKENDS), ids=sorted(_EMBEDDED_BACKENDS))
@pytest.mark.parametrize("mode", ["natural", "residual"])
@pytest.mark.parametrize("case_id", _SMOKE_CASE_IDS)
def test_embedded_documents_smoke(case_id: str, mode: str, backend: str) -> None:
    """One case, one mode, one embedded store: the read output must equal the oracle.

    ``natural`` hands the AST to ``scan_documents_page`` (backend pushdown + the
    coordinator's own post-filter); ``residual`` withholds it and applies the identical
    post-filter over the returned documents. Both drive the same real keyset primitive,
    and both must land on ``expected_ids`` — the surface is split + post-filter on every
    backend, so no case is expected to raise.
    """
    try:
        module = importlib.import_module(_EMBEDDED_BACKENDS[backend])
    except ImportError as exc:  # optional extra absent (lancedb / surrealdb)
        pytest.skip(f"{backend} documents leg unavailable: {exc}")
    if not module.reachable():
        pytest.skip(f"{backend} documents store not reachable")

    case = _BY_ID[case_id]
    if backend == "surrealdb" and _documents_surreal_excluded(case.filter, case.seed_records):
        pytest.skip(
            f"{case.id} is pruned from the surreal documents leg by a documented storage-"
            f"representation quirk (see _documents_surreal_excluded)"
        )

    executor = module.executor_for(case, forced_residual=(mode == "residual"))
    assert run_case_for_backend(case, backend, executor=executor) == case.expected_ids


# --------------------------------------------------------------------------- #
# 4. The Postgres seed-map artifact (DB-free).
# --------------------------------------------------------------------------- #
#
# The three embedded legs above seed in-process, so their whole path is exercised by
# this module. The Postgres leg cannot be: it is read-only over a store seeded by a
# separate process, bridged by a JSON artifact. That artifact is therefore the one
# piece of the documents seam no live-store run in this file can reach — and if its
# round-trip drifts (the ``{case_id: {seed_id: document_id}}`` shape, the
# ``str``↔``UUID`` coercion, the missing-file contract) the leg silently mis-maps every
# survivor. These two checks are pure ``tmp_path`` file I/O, so they run here rather
# than being deferred to a job that needs a container. Same guard the chunk leg has in
# ``tests/integration/matrix/test_seed_map_roundtrip.py``, for the documents map.


@pytest.fixture
def _docs_seed_map_at(tmp_path, monkeypatch):  # noqa: ANN001, ANN202 - pytest fixture
    """Point the documents seed-map helpers at a ``tmp_path`` file; reset the load cache."""
    from tests.integration.matrix import _conformance_docs_pg

    monkeypatch.setattr(_conformance_docs_pg, "SEED_MAP_PATH", str(tmp_path / "docs_seed_map.json"))
    _conformance_docs_pg.load_seed_map.cache_clear()
    try:
        yield _conformance_docs_pg
    finally:
        _conformance_docs_pg.load_seed_map.cache_clear()


def test_documents_seed_map_round_trip_preserves_ids_and_uuid_type(_docs_seed_map_at) -> None:  # noqa: ANN001
    """``write_seed_map`` -> ``load_seed_map`` preserves case ids, seed ids, and UUID identity."""
    from uuid import UUID, uuid4

    doc_a, doc_b = uuid4(), uuid4()
    original = {"F-OP-source_name-eq": {"source_name-1": str(doc_a), "source_name-2": str(doc_b)}}

    _docs_seed_map_at.write_seed_map(original)
    loaded = _docs_seed_map_at.load_seed_map()

    assert loaded.keys() == original.keys()
    assert loaded["F-OP-source_name-eq"].keys() == {"source_name-1", "source_name-2"}
    assert loaded["F-OP-source_name-eq"]["source_name-1"] == doc_a
    assert isinstance(loaded["F-OP-source_name-eq"]["source_name-1"], UUID)
    assert loaded["F-OP-source_name-eq"]["source_name-2"] == doc_b


def test_documents_seed_map_missing_file_raises_an_actionable_error(_docs_seed_map_at) -> None:  # noqa: ANN001
    """A missing map names the documents seed step, not just the absent path.

    The chunk and documents corpora have separate seed invocations and separate
    artifact paths, so a generic "no such file" would send an operator to the wrong
    command.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        _docs_seed_map_at.load_seed_map()

    message = str(excinfo.value)
    assert "_conformance_seed documents-postgres" in message
    assert "read-only" in message
    assert "KHORA_CONFORMANCE_DOCS_SEED_MAP" in message
