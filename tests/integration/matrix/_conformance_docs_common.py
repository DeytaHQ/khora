"""Backend-agnostic wiring shared by the four documents-target conformance legs.

The chunk-surface conformance seam gives each backend its own ``run_live`` because
each one executes a *different compiled artifact* (a SQLAlchemy expression, a SurrealQL
string, a Cypher string, a weaviate ``Filter``). The documents surface is not shaped
that way: every leg drives the same public coordinator method,
:meth:`~khora.storage.coordinator.StorageCoordinator.scan_documents_page`, and the
backend compiles its own prefilter *inside* ``scan_documents``. So the walk is
genuinely one piece of code, and it lives here rather than in four near-copies that
could drift.

What stays per-backend (in the sibling ``_conformance_docs_<backend>`` modules) is only
the part that really differs: standing up a relational-only coordinator for that store,
and — for the read-only Postgres leg — the seed-map artifact.

Kept out of ``conftest.py`` and named with a leading underscore (not a ``test_``
module) so it is a plain helper the test modules import, never collected as tests.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any
from uuid import UUID

from khora.filter.ast import FilterNode
from khora.filter.conformance import DocumentsExecutor
from khora.storage.backends.base import DocumentScanKey
from khora.storage.coordinator import StorageCoordinator

# One page is sized to swallow any corpus case whole (the largest seeds ten records),
# so a green run does not depend on multi-page resume — that is
# ``tests/**/test_*scan_documents*.py``'s subject, not conformance's. The walk below
# still loops on ``exhausted``, so a smaller page would only cost round-trips.
PAGE_LIMIT = 100
# Raw rows the page may scan. Comfortably above any case's seed so the bound is never
# the reason a walk stops: a bound-limited page reports ``exhausted=False`` with a
# non-``None`` position, which the loop would follow, but silent truncation of the
# LAST page is the failure mode worth designing out entirely.
SCAN_BOUND = 100_000

# A walk that neither advances its cursor nor reports exhaustion would spin forever.
# No store does that today; the guard converts a future regression into a named
# failure instead of a hung CI job.
_MAX_PAGES = 1_000

# Upper bound on any single submitted coroutine. The ``_MAX_PAGES`` guard only covers
# a non-advancing walk, not a coroutine that stalls inside a store call; a bounded
# wait converts a hung CI job into a named ``TimeoutError`` there too. Generous enough
# for a full corpus seed on a cold store, short enough that a stalled store fails.
_RUN_TIMEOUT_S = 600


class _LoopThread:
    """A daemon thread running one asyncio loop; submit coroutines, block for results."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=_RUN_TIMEOUT_S)


@lru_cache(maxsize=1)
def _loop_thread() -> _LoopThread:
    """The ONE process-wide loop thread that owns every seeded documents store.

    Deliberately shared across all three embedded backends rather than one apiece: an
    aiosqlite connection and an embedded SurrealDB connection are each bound to the
    loop they were opened on, and the unit smoke leg drives all three in a single
    process. One loop for all of them keeps every open-and-use pair on the same loop
    with no coordination between the modules.
    """
    return _LoopThread()


def run_async(coro: Any) -> Any:
    """Run ``coro`` on the dedicated loop that owns the seeded documents stores."""
    return _loop_thread().run(coro)


async def walk_documents(
    coord: StorageCoordinator,
    namespace_id: UUID,
    *,
    filter_ast: FilterNode,
    post_filter: Callable[[Any], bool],
    forced_residual: bool,
    id_map: Mapping[str, UUID],
) -> frozenset[str]:
    """Enumerate a case's namespace to exhaustion; return the surviving seed ids.

    The :class:`~khora.filter.conformance.DocumentsRunner` body, shared by all four
    legs. Drives the REAL ``scan_documents_page`` — which drives the store's real
    ``scan_documents`` keyset primitive and its own ``_documents_compile_context()``
    pushdown — and translates surviving ``Document.id`` values back to
    :class:`~khora.filter.conformance.SeedRecord` ids.

    The two modes differ in exactly one place, which is the point of having them:

    * **natural** — ``filter_ast`` goes to the coordinator, which pushes what the
      backend can take and applies its OWN ``compile_python`` post-filter to the rest.
      ``post_filter`` is not applied here; doing so would be a second, redundant copy
      and would mask a coordinator that forgot to run its own.
    * **residual** — the coordinator gets ``filter_ast=None``, which makes the call a
      RAW ENUMERATION: with no AST it compiles no post-filter and does no filtering at
      all. ``post_filter`` (the identical predicate, built by the executor) is the only
      narrowing, and it runs HERE — outside the coordinator, not through any branch of
      it. With SQL out of the picture entirely, every leaf is forced to evaluate
      against each store's ROUND-TRIP document shape rather than against whatever SQL
      happened to accept.

    ``exhausted`` is the only termination signal read — a short page means nothing
    under a selective filter, and an empty one means nothing at all.
    """
    doc_to_seed = {doc_id: seed_id for seed_id, doc_id in id_map.items()}
    survivors: set[str] = set()
    after: DocumentScanKey | None = None

    for _ in range(_MAX_PAGES):
        page = await coord.scan_documents_page(
            namespace_id,
            filter_ast=None if forced_residual else filter_ast,
            limit=PAGE_LIMIT,
            after=after,
            scan_bound=SCAN_BOUND,
        )
        for document in page:
            seed_id = doc_to_seed.get(document.id)
            if seed_id is None:
                # Unreachable: each case owns a private namespace. Guarding rather
                # than indexing blindly keeps a namespace-scope regression a failed
                # assertion downstream instead of a KeyError here.
                continue
            if forced_residual and not post_filter(document):
                continue
            survivors.add(seed_id)
        if page.exhausted:
            return frozenset(survivors)
        assert page.next_after is not None, "a non-exhausted page must carry a resume position"
        after = (page.next_after.created_at, page.next_after.id)

    raise AssertionError(f"documents walk did not reach exhaustion within {_MAX_PAGES} pages")


def documents_executor(
    coord: StorageCoordinator,
    namespace_id: UUID,
    id_map: Mapping[str, UUID],
    *,
    forced_residual: bool,
) -> DocumentsExecutor:
    """Bind :func:`walk_documents` to one seeded case and wrap it in the executor.

    The runner is *synchronous* (the ``DocumentsRunner`` contract) while the walk is
    async and the seeded store belongs to :func:`_loop_thread`'s loop, so the bridge
    submits the coroutine there and blocks — the same shape the chunk legs' embedded
    runners use.
    """

    def runner(filter_ast, post_filter, *, forced_residual):  # noqa: ANN001, ANN202 - matches DocumentsRunner
        return run_async(
            walk_documents(
                coord,
                namespace_id,
                filter_ast=filter_ast,
                post_filter=post_filter,
                forced_residual=forced_residual,
                id_map=id_map,
            )
        )

    return DocumentsExecutor(runner, forced_residual=forced_residual)
