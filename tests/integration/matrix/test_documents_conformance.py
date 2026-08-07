"""Documents-target conformance corpus — parametrized matrix leg.

The sibling of ``test_filter_conformance.py``, for the document-ENUMERATION surface
rather than the chunk-recall one. Every case in
:func:`~khora.filter.conformance.documents_conformance_cases` is lowered through the
real validator + ``parse_to_ast``, seeded as documents through the coordinator's write
API, and enumerated back through the REAL
:meth:`~khora.storage.coordinator.StorageCoordinator.scan_documents_page` keyset walk.
A backend is *conformant* iff the walk's output equals the Python oracle's, case for
case.

**Why this is a separate module rather than another backend of the chunk one.** The
two corpora differ (``occurred_at`` is not enumerable, and ``external_id`` needs a
distinct-value re-seed), the seeder differs (one document per record, no chunks), and
— most of all — the *shape of the claim* differs. On the chunk surface, postgres and
surrealdb are TOTAL-exact and a filter they cannot express RAISES; the corpus carries
``expect_unsupported`` legs to assert that. Document enumeration is uniformly split +
post-filter on all four backends: the coordinator compiles the whole AST once with
``build_compile_context("documents", on_unsupported="split")`` and re-checks it over
every scanned row, and each store's ``scan_documents`` compiles its prefilter with the
same mode. **Nothing raises here**, so every case is a plain row-set comparison and
there is no ``expect_unsupported`` leg to assert.

Which backend this leg runs is selected by ``KHORA_CONFORMANCE_DOC_BACKEND`` (default
``sqlite``), so the CI matrix runs one leg per backend from one module:

* ``sqlite`` — the raw ``backend: sqlite`` tier. In-process, no Docker. **The first
  conformance coverage of that backend at all** (the chunk corpus reaches SQLite only
  through ``sqlite_lance``, a different ``documents`` schema).
* ``sqlite_lance`` — the shared Alembic ``documents`` model over embedded SQLite.
  In-process, no Docker.
* ``surrealdb`` — embedded ``memory://`` with the production schema. In-process; needs
  the ``surrealdb`` extra. Drops the two storage-representation buckets
  :func:`~khora.filter.conformance._documents_surreal_excluded` names — a documented
  capability prune, never a silent skip.
* ``postgres`` — the live server, read-only over a store seeded ONCE out-of-band
  (``python -m tests.integration.matrix._conformance_seed documents-postgres``), so
  every xdist worker only reads. The only tier with real ``timestamptz`` columns, and
  therefore the only one that pushes the two date keys into SQL.

Every case runs in BOTH modes:

* ``natural`` — the AST reaches ``scan_documents_page``, so the backend pushes what it
  can and the coordinator post-filters the rest. This is the production path, and the
  only mode that exercises the coordinator's post-filter at all (including its
  no-pushdown extreme, which any case whose leaves the backend defers wholesale
  already reaches).
* ``residual`` — the AST is WITHHELD, which makes ``scan_documents_page`` a RAW
  ENUMERATION (no AST ⇒ no compiled post-filter ⇒ no narrowing), and the identical
  ``compile_python`` predicate is applied in-harness over the returned documents.
  **This is not a coordinator code path** — the filtering happens outside it. What it
  pins is ``compile_python`` over ``"documents"`` against each store's ROUND-TRIP
  document shape — raw-sqlite's TEXT datetimes, SurrealDB's ``metadata_`` remap,
  SQLAlchemy's offset-discarding ``DATETIME`` — rather than against whatever SQL
  happened to accept, so a mangled round-trip cannot hide behind a pushdown that
  rejected the row for the right reason by accident. A ``SchemaCapabilities`` override
  cannot express this: ``scan_documents`` takes no context override, so withholding
  the AST is how the mode is realized.

The corpus is excluded from the main test/integration jobs via the
``filter_conformance`` marker (see ``pyproject.toml`` addopts); it runs only in its own
CI job.
"""

from __future__ import annotations

import importlib
import os

import pytest

from khora.filter.conformance import (
    ConformanceCase,
    _documents_surreal_excluded,
    documents_conformance_cases,
    run_case_for_backend,
)

pytestmark = [pytest.mark.integration, pytest.mark.filter_conformance]

# backend name -> runner-module import path. Every module exposes the same two
# callables: ``reachable() -> bool`` and ``executor_for(case, *, forced_residual)``.
# Imports are LAZY (inside the dispatch) so the sqlite leg never imports the lancedb /
# surrealdb / asyncpg stacks.
_DOC_BACKENDS: dict[str, str] = {
    "sqlite": "tests.integration.matrix._conformance_docs_sqlite",
    "sqlite_lance": "tests.integration.matrix._conformance_docs_lance",
    "surrealdb": "tests.integration.matrix._conformance_docs_surreal",
    "postgres": "tests.integration.matrix._conformance_docs_pg",
}

# Default ``sqlite``: in-process and dependency-free, so a bare
# ``pytest -m filter_conformance`` on the no-Docker path still runs the whole corpus.
SELECTED_BACKEND = os.environ.get("KHORA_CONFORMANCE_DOC_BACKEND", "sqlite")

if SELECTED_BACKEND not in _DOC_BACKENDS:
    pytest.skip(
        f"backend {SELECTED_BACKEND!r} is not a documents-conformance backend (available: {sorted(_DOC_BACKENDS)})",
        allow_module_level=True,
    )


def _runner_module():  # noqa: ANN202 - the runner module
    """Lazily import the selected backend's ``_conformance_docs_<backend>`` module."""
    return importlib.import_module(_DOC_BACKENDS[SELECTED_BACKEND])


# Gate the leg on its store being reachable. LOCAL-DEV CONVENIENCE ONLY for the
# postgres leg: in CI the parent ``tests/integration/conftest.py::pytest_configure``
# aborts the session RED first when ``KHORA_PG_REQUIRED=1`` and PG is unreachable, so a
# PG-down leg fails loudly rather than silently skipping — keep this module under
# ``tests/integration/`` so that parent conftest loads.
if not _runner_module().reachable():
    pytest.skip(
        f"{SELECTED_BACKEND} documents store not reachable "
        f"(postgres: run `make dev`; surrealdb: install the `surrealdb` extra)",
        allow_module_level=True,
    )


def _cases_for(backend: str) -> list[ConformanceCase]:
    """The corpus this leg runs — the whole thing, minus the surreal capability prune.

    Only ``surrealdb`` prunes, and only the two storage-representation buckets
    :func:`_documents_surreal_excluded` names (a metadata datetime stored as a string;
    an explicit JSON ``null`` dropped from a FLEXIBLE object on WRITE). Both corrupt
    the stored row before any filter runs, so no post-filter can recover them — they
    are capability facts about the store, not gaps in the compiler.
    """
    cases = documents_conformance_cases()
    if backend != "surrealdb":
        return cases
    return [c for c in cases if not _documents_surreal_excluded(c.filter, c.seed_records)]


_SELECTED_CASES = _cases_for(SELECTED_BACKEND)


@pytest.mark.parametrize("mode", ["natural", "residual"])
@pytest.mark.parametrize("case", _SELECTED_CASES, ids=lambda c: c.id)
def test_documents_conformance_case(case: ConformanceCase, mode: str) -> None:
    """Assert one case's row-set on the selected backend, in one mode, vs the oracle."""
    executor = _runner_module().executor_for(case, forced_residual=(mode == "residual"))
    assert run_case_for_backend(case, SELECTED_BACKEND, executor=executor) == case.expected_ids
