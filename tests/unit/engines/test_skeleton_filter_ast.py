"""Regression tests: Skeleton recall must push ``filter_ast`` to the store.

The Skeleton engine's ``recall`` accepted a ``filter_ast`` (the canonical
recall-filter AST) for protocol parity but historically dropped it on the
floor — it never forwarded the node to ``temporal_store.search(...)``. This
file guards two distinct regressions:

1. **Engine drop (the primary regression).** ``recall(..., filter_ast=node)``
   MUST forward ``filter_ast=node`` in the kwargs of the ``temporal_store.search``
   call. ``test_recall_forwards_filter_ast_to_store`` drives ``recall`` against
   an ``AsyncMock`` store and asserts the spy received the exact node. If the
   engine ever again drops ``filter_ast=filter_ast`` from the call, this fails.

2. **Backend signature drift (the turbopuffer regression).** Every concrete
   backend ``search()`` must accept a ``filter_ast`` parameter, otherwise the
   engine's forwarded kwarg raises ``TypeError`` at runtime on that backend.
   ``test_all_backend_search_methods_accept_filter_ast`` walks the protocol +
   all five concrete stores via ``inspect.signature``. PgVector already had the
   param (#1008); the other four were added when the engine was wired.

Both tests are NO-Postgres: they use shallow ``AsyncMock`` stubs (mirroring
``test_skeleton_kwarg_drift`` / ``test_skeleton_search_mode``) and module-level
``inspect`` reads. The Postgres-dependent end-to-end filter test lives on a
separate ticket.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.skeleton.engine import SkeletonConstructionEngine
from khora.filter import FilterClause, FilterNode, FilterOp
from khora.query import SearchMode


def _build_engine_with_stubs() -> tuple[SkeletonConstructionEngine, AsyncMock]:
    """Construct an engine with embedder + temporal store stubs.

    Mirrors ``test_skeleton_search_mode._build_engine_with_stubs``: the
    engine's ``__init__`` does no network I/O until ``connect()``, so we
    bypass ``connect()`` via ``__new__`` and inject the two collaborators
    ``recall()`` actually touches (``_embedder`` and ``_temporal_store``).
    """
    cfg = MagicMock()
    cfg.storage.backend = "pgvector"

    engine = SkeletonConstructionEngine.__new__(SkeletonConstructionEngine)
    engine._config = cfg
    engine._backend_type = "pgvector"
    engine._weaviate_url = None
    engine._storage_config = MagicMock()
    engine._storage = None
    engine._connected = True

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    engine._embedder = embedder

    temporal_store = AsyncMock()
    temporal_store.search = AsyncMock(return_value=[])
    engine._temporal_store = temporal_store

    return engine, temporal_store


def _sample_filter_ast() -> FilterNode:
    """A small, real ``FilterNode`` — ``AND([author $eq "alice"])``.

    Built directly (no ``RecallFilter`` round-trip) so the test depends only
    on the AST layer, not the wire-model validator. The exact shape is
    irrelevant to the assertion — what matters is that this *specific object*
    is the one handed back to the store.
    """
    return FilterNode(
        op=FilterOp.AND,
        children=(FilterClause(path=("author",), op=FilterOp.EQ, operand="alice"),),
    )


# ---------------------------------------------------------------------------
# Regression 1: the engine must forward filter_ast to temporal_store.search.
# ---------------------------------------------------------------------------


async def test_recall_forwards_filter_ast_to_store() -> None:
    """``recall(..., filter_ast=node)`` forwards ``filter_ast=node`` to search.

    This is the core regression guard. If a future edit drops
    ``filter_ast=filter_ast`` from the ``temporal_store.search(...)`` call in
    ``engine.recall`` (reverting to the "ignored for protocol parity"
    behavior), the kwarg vanishes from ``await_args.kwargs`` and this fails.
    """
    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()
    node = _sample_filter_ast()

    await engine.recall("alpha", namespace_id, mode=SearchMode.VECTOR, filter_ast=node)

    temporal_store.search.assert_awaited_once()
    kwargs = temporal_store.search.await_args.kwargs
    assert "filter_ast" in kwargs, (
        "engine.recall must forward filter_ast as a kwarg to temporal_store.search; "
        "it was dropped (the pre-wiring 'ignored for protocol parity' regression)"
    )
    # Identity, not just equality: the exact node passed in must reach the store.
    assert kwargs["filter_ast"] is node


async def test_recall_filter_ast_none_forwards_none() -> None:
    """The default ``filter_ast=None`` still reaches the store as ``None``.

    Forwarding must be unconditional — the engine should pass ``filter_ast``
    through on every call, not only when it is non-None. A ``None`` here means
    "no filter", and the store's compiler treats it as no constraint.
    """
    engine, temporal_store = _build_engine_with_stubs()
    namespace_id = uuid4()

    await engine.recall("alpha", namespace_id, mode=SearchMode.VECTOR)

    temporal_store.search.assert_awaited_once()
    kwargs = temporal_store.search.await_args.kwargs
    assert "filter_ast" in kwargs
    assert kwargs["filter_ast"] is None


# ---------------------------------------------------------------------------
# Regression 2: every backend search() must accept the filter_ast parameter.
# ---------------------------------------------------------------------------


def _all_search_owners() -> list[tuple[str, object]]:
    """The protocol + the five concrete temporal stores, by class.

    Importing the concrete stores is cheap — each defers its framework SDK
    (weaviate / turbopuffer / surreal client) to a lazy in-method import, so
    the class object is reachable without the optional extra installed.
    """
    from khora.engines.skeleton.backends import TemporalVectorStore
    from khora.engines.skeleton.backends.pgvector import PgVectorTemporalStore
    from khora.engines.skeleton.backends.sqlite_lance import SQLiteLanceTemporalStore
    from khora.engines.skeleton.backends.surrealdb import SurrealDBTemporalStore
    from khora.engines.skeleton.backends.turbopuffer import TurbopufferTemporalStore
    from khora.engines.skeleton.backends.weaviate import WeaviateTemporalStore

    return [
        ("TemporalVectorStore", TemporalVectorStore),
        ("PgVectorTemporalStore", PgVectorTemporalStore),
        ("WeaviateTemporalStore", WeaviateTemporalStore),
        ("TurbopufferTemporalStore", TurbopufferTemporalStore),
        ("SurrealDBTemporalStore", SurrealDBTemporalStore),
        ("SQLiteLanceTemporalStore", SQLiteLanceTemporalStore),
    ]


@pytest.mark.parametrize("name,store_cls", _all_search_owners())
def test_all_backend_search_methods_accept_filter_ast(name: str, store_cls: object) -> None:
    """Every backend ``search()`` exposes a ``filter_ast`` keyword parameter.

    Guards the turbopuffer-style regression: the engine forwards
    ``filter_ast=...`` to whichever backend is configured, so a backend whose
    ``search`` lacks the parameter raises ``TypeError`` at recall time. The
    ``inspect.signature`` walk catches the drift statically across all five
    backends + the protocol they implement.
    """
    sig = inspect.signature(store_cls.search)  # type: ignore[attr-defined]
    assert "filter_ast" in sig.parameters, (
        f"{name}.search() is missing the 'filter_ast' parameter; the engine "
        f"forwards filter_ast= to it and would raise TypeError at recall time"
    )
    # It must be keyword-acceptable (keyword-only or positional-or-keyword),
    # because the engine forwards it by keyword. A VAR_KEYWORD (**kwargs) sink
    # would technically swallow it but is not the contract here.
    param = sig.parameters["filter_ast"]
    assert param.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), f"{name}.search() 'filter_ast' must be passable by keyword, got kind={param.kind}"


# ---------------------------------------------------------------------------
# engine_info['filter']['pushed_down'] derivation (engine-side, no Postgres).
# ---------------------------------------------------------------------------
#
# The skeleton engine derives the pushdown flag from the filter_ast + backend
# without recompiling: ``filter_ast is not None and bool(filter_ast.children)
# and backend_type == "pgvector"``. The live-pg end-to-end proof of this flag
# was the retired interim smoke test; this drives the same derivation directly
# off a stubbed engine (no database), and the facade-side carrier threading is
# pinned separately by tests/unit/test_khora.py.


async def _pushed_down_for(filter_ast: FilterNode | None) -> bool:
    """Run ``recall`` with ``filter_ast`` and read the derived pushdown flag.

    Uses the same no-Postgres stub harness as the forwarding tests: the engine
    is on the pgvector backend with a stub store returning no rows, so recall
    reaches the engine_info derivation and we read it off the result.
    """
    engine, _ = _build_engine_with_stubs()
    result = await engine.recall("alpha", uuid4(), mode=SearchMode.VECTOR, filter_ast=filter_ast)
    return result.engine_info["filter"]["pushed_down"]


async def test_pushed_down_true_for_constrained_ast_on_pgvector() -> None:
    """A non-None AST WITH constraints on the pgvector backend → pushed_down True.

    The pgvector compiler is all-or-nothing (``on_unsupported="raise"``), so a
    recall that returns with a constrained filter_ast means every leaf pushed
    down — the engine reports that as ``pushed_down=True``.
    """
    assert await _pushed_down_for(_sample_filter_ast()) is True


async def test_pushed_down_false_for_none_ast() -> None:
    """No filter (``filter_ast is None``) → pushed_down False (nothing to push)."""
    assert await _pushed_down_for(None) is False


async def test_pushed_down_false_for_empty_ast() -> None:
    """A constraint-free AST (empty-AND root) → pushed_down False.

    ``filter={}`` / ``RecallFilter()`` normalize to ``FilterNode(op=AND,
    children=())`` — a match-everything root carrying zero leaves. "All leaves
    consumed" is vacuous, so it narrows nothing and reports ``pushed_down=False``,
    matching the no-filter case rather than claiming an empty pushdown.
    """
    empty_ast = FilterNode(op=FilterOp.AND, children=())
    assert await _pushed_down_for(empty_ast) is False
