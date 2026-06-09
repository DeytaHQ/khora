"""Recall-filter conformance corpus — parametrized matrix leg.

Drives the existing harness in :mod:`khora.filter.conformance`: every corpus case
is lowered through the real validator + ``parse_to_ast`` + the real per-backend
compiler, and the surviving record set is asserted against the case's declared
``expected_ids`` (or an unsupported outcome). A backend compiler is *conformant*
iff it agrees with the Python oracle on every case.

Which backend this leg runs is selected by ``KHORA_CONFORMANCE_BACKEND`` (default
``python``), so the CI matrix runs one leg per backend with the same test module:

* ``python`` / ``chronicle`` — in-memory executors, NO database. Every case runs.
* ``postgres`` — gated behind ``_pg_reachable``; compiles the real Postgres
  ``WHERE`` and runs it against a live ``khora_chunks`` store. The store is seeded
  ONCE, out-of-band, by ``_conformance_seed`` (the workflow's one-time step);
  this module is strictly READ-ONLY, so every xdist worker only reads (no write
  contention under ``-n auto``).
* anything else (surreal / weaviate / cypher / sqlite_lance — NOT in the harness
  ``BACKENDS``) collects ZERO conformance cases; only the registry guard runs on
  that leg. A module-level skip states the reason rather than erroring.

The corpus is excluded from the main test/integration jobs via the
``filter_conformance`` marker (see ``pyproject.toml`` addopts); it runs only in
its own CI job.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

from khora.filter.conformance import (
    BACKENDS,
    ChronicleExecutor,
    ConformanceCase,
    PostgresExecutor,
    PythonExecutor,
    assert_case,
    f_op_cases,
)

pytestmark = [pytest.mark.integration, pytest.mark.filter_conformance]

# Which backend's cases this leg runs. Default "python" keeps a bare
# `pytest -m filter_conformance` on the no-Docker path meaningful.
SELECTED_BACKEND = os.environ.get("KHORA_CONFORMANCE_BACKEND", "python")


# --------------------------------------------------------------------------- #
# Corpus assembly.
# --------------------------------------------------------------------------- #


def _all_cases() -> list[ConformanceCase]:
    """Every case from every non-empty corpus family.

    Today only ``f_op_cases`` is populated; the catalog families return ``[]``
    until filled, and iterating them here means new families join the matrix
    with no change to this module.
    """
    return list(f_op_cases())


def _cases_for(backend: str) -> list[ConformanceCase]:
    """Cases whose ``backends`` includes ``backend`` (empty if not a harness backend)."""
    if backend not in BACKENDS:
        return []
    return [c for c in _all_cases() if backend in c.backends]


_SELECTED_CASES = _cases_for(SELECTED_BACKEND)

# A backend outside the harness ``BACKENDS`` (surreal/weaviate/cypher/sqlite_lance)
# has no in-corpus cases. Skip the module with a clear reason rather than erroring
# — the registry guard still runs on that leg (it is a separate module).
if SELECTED_BACKEND not in BACKENDS:
    pytest.skip(
        f"backend {SELECTED_BACKEND!r} is not a conformance backend "
        f"(harness BACKENDS = {sorted(BACKENDS)}); only the registry guard runs on this leg",
        allow_module_level=True,
    )


# --------------------------------------------------------------------------- #
# Postgres leg: live-store gate + READ-ONLY runner over the pre-seeded store.
# --------------------------------------------------------------------------- #
#
# The DB is seeded ONCE, out-of-band, by ``_conformance_seed`` (the workflow's
# one-time step), which also persists the ``case_id -> {seed_id: chunk_uuid}`` map.
# This module is strictly READ-ONLY: it loads that map and runs each compiled
# ``WHERE`` against the already-seeded ``khora_chunks`` rows — no ``seed_case``
# call here, so under ``-n auto`` every worker only reads (no write contention).


def _pg_reachable() -> bool:
    from tests.integration.matrix._conformance_pg import DATABASE_URL

    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# Only the postgres leg needs a live DB; gate just that leg so the python /
# chronicle legs always run. This skip is LOCAL-DEV CONVENIENCE ONLY: in CI the
# parent ``tests/integration/conftest.py::pytest_configure`` aborts the session RED
# first when ``KHORA_PG_REQUIRED=1`` and PG is unreachable, so a PG-down postgres leg
# fails loudly rather than silently skipping — keep this module under
# ``tests/integration/`` so that parent conftest loads.
if SELECTED_BACKEND == "postgres" and not _pg_reachable():
    pytest.skip(
        "PostgreSQL not reachable (run `make dev` first)",
        allow_module_level=True,
    )


def _postgres_executor_for(case: ConformanceCase) -> PostgresExecutor:
    """Wire a READ-ONLY ``PostgresExecutor`` over ``case``'s pre-seeded rows.

    Looks the case up in the persisted seed map (loaded once, cached) and closes a
    ``PostgresRunner`` over that case's ``seed_id -> chunk UUID`` map. No seeding —
    the rows already exist. ``PostgresExecutor`` invokes the real ``compile_postgres``
    (this is what conformance checks); the runner only executes the predicate.

    The runner is *synchronous* (the ``PostgresRunner`` contract) but the query is
    async and the test already owns the running event loop, so it runs the query on
    a fresh loop in a worker thread (``asyncio.run`` cannot nest in a running loop).
    """
    from concurrent.futures import ThreadPoolExecutor

    from tests.integration.matrix._conformance_pg import (
        load_seed_map,
        run_predicate,
    )

    seed_map = load_seed_map()
    if case.id not in seed_map:
        pytest.fail(
            f"case {case.id!r} missing from the seed map "
            f"(re-run `python -m tests.integration.matrix._conformance_seed`)"
        )
    id_map = seed_map[case.id]

    def runner(predicate, params, records):  # noqa: ANN001, ANN202 - matches PostgresRunner
        import asyncio

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(run_predicate(id_map, predicate, records))).result()

    return PostgresExecutor(runner)


# --------------------------------------------------------------------------- #
# The parametrized assertion.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _SELECTED_CASES, ids=lambda c: c.id)
def test_conformance_case(case: ConformanceCase) -> None:
    """Assert one case's outcome on the selected backend, vs the Python oracle."""
    if SELECTED_BACKEND == "python":
        assert_case(case, "python", PythonExecutor())
    elif SELECTED_BACKEND == "chronicle":
        assert_case(case, "chronicle", ChronicleExecutor())
    elif SELECTED_BACKEND == "postgres":
        assert_case(case, "postgres", _postgres_executor_for(case))
    else:  # pragma: no cover - module-level skip already guards this
        pytest.fail(f"unexpected conformance backend {SELECTED_BACKEND!r}")
