"""Store-level recall-filter contract on BOTH read channels — embedded SurrealDB.

The compiler-level tests (``tests/unit/filter/test_compile_surrealdb.py``) and the
predicate row-set tests (``tests/integration/filter/test_compile_surrealdb_embedded.py``)
prove the SurrealDB compiler raises on an unbacked system key and pushes a backed
one. This module proves the SAME contract holds at the STORE seam — through
:class:`~khora.engines.skeleton.backends.surrealdb.SurrealDBTemporalStore`'s two
public read channels:

* the **vector** channel — ``store.search(..., hybrid_alpha=1.0, filter_ast=…)``
  (the ``SearchMode.VECTOR`` route compiles ``filter_ast`` in ``on_unsupported="raise"``
  mode before the cosine scan);
* the **BM25** channel — ``store.search_fulltext(..., filter_ast=…)`` (the
  ``SearchMode.KEYWORD`` route, same raise-mode compile before the full-text scan).

Both channels compile the filter through the live recall context, so an unbacked
system key (one of the eight denormalized document keys the ``temporal_chunk`` table
does not back with a column) must FAIL LOUD on EITHER channel rather than silently
returning a wrong row-set, and a backed predicate (``occurred_at`` range,
``metadata.<key>``) must still narrow correctly on EITHER channel.

Embedded ``memory://`` runs in-process — no server, no Docker. The module self-skips
when the embedded SurrealDB SDK is unavailable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.config import KhoraConfig  # noqa: E402
from khora.engines.skeleton.backends import TemporalChunk  # noqa: E402
from khora.engines.skeleton.backends.surrealdb import _BACKED_SYSTEM_KEYS, SurrealDBTemporalStore  # noqa: E402
from khora.filter import RecallFilter  # noqa: E402
from khora.filter.ast import FilterNode, parse_to_ast  # noqa: E402
from khora.filter.context import RecallFilterUnsupportedError  # noqa: E402
from khora.filter.model import SYSTEM_KEYS  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402

pytestmark = pytest.mark.integration

# Embedding width matches the HNSW index the store's schema defines (DIMENSION 1536).
_DIM = 1536
_EMBED = [0.1] * _DIM

# The eight unbacked system keys = SYSTEM_KEYS minus the store's declared backed set.
# Derived from the source of truth so the corpus tracks any change to the backed set.
_UNBACKED_KEYS = tuple(sorted(SYSTEM_KEYS - _BACKED_SYSTEM_KEYS))
_DATE_TYPED_UNBACKED = "source_timestamp"
_STRING_UNBACKED_KEYS = tuple(k for k in _UNBACKED_KEYS if k != _DATE_TYPED_UNBACKED)
# Substring the backend gate raises with (the message also names the offending key).
_GATE_REASON = "does not back"


def _unbacked_wire(key: str, *, exists: bool = False) -> dict:
    """A valid wire filter over an unbacked key, respecting its operand type.

    ``source_timestamp`` is datetime-typed (DateOps) — it takes a date operand and
    forbids ``$exists``; the seven string keys take a string and accept ``$exists``.
    """
    if exists:
        return {key: {"$exists": True}}
    if key == _DATE_TYPED_UNBACKED:
        return {key: "2026-01-01T00:00:00Z"}
    return {key: "v"}


# Two seed rows: a recent, gold-tier chunk and an ancient, silver-tier one. Both
# share the BM25 token "pagerduty" so the full-text channel returns both before the
# filter narrows. ``content`` is the row's stable label.
_NS = uuid4()
_DOC = uuid4()
_RECENT = datetime(2026, 6, 1, tzinfo=UTC)
_ANCIENT = datetime(2020, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset_surrealdb_schema_init_lock() -> None:
    """Reset the module-level schema-init lock per test (loop-local, see issue #718)."""
    from khora.storage.backends.surrealdb import connection as _conn_mod

    _conn_mod._schema_init_lock = asyncio.Lock()


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


@pytest.fixture
async def store() -> AsyncIterator[tuple[SurrealDBTemporalStore, UUID]]:
    """An embedded SurrealDB store seeded with the two chunks + BM25 index built.

    Yields ``(store, namespace_id)``. ``ensure_search_indexes()`` builds the BM25
    full-text index so the ``search_fulltext`` channel actually runs (without it the
    BM25 path degrades to ``[]`` on "no suitable index").
    """
    conn = SurrealDBConnection(mode="memory")
    await conn.connect()
    store = SurrealDBTemporalStore(KhoraConfig(), connection=conn)
    await store.connect()
    await store.ensure_search_indexes()
    try:
        await store.create_chunks_batch(
            [
                TemporalChunk(
                    id=uuid4(),
                    namespace_id=_NS,
                    document_id=_DOC,
                    content="recent pagerduty alert",
                    embedding=_EMBED,
                    occurred_at=_RECENT,
                    metadata={"tier": "gold"},
                ),
                TemporalChunk(
                    id=uuid4(),
                    namespace_id=_NS,
                    document_id=_DOC,
                    content="ancient pagerduty alert",
                    embedding=_EMBED,
                    occurred_at=_ANCIENT,
                    metadata={"tier": "silver"},
                ),
            ]
        )
        yield store, _NS
    finally:
        await conn.disconnect()


async def _vector_contents(store: SurrealDBTemporalStore, ns: UUID, filter_ast: FilterNode) -> list[str]:
    """Run the vector channel (pure-vector) with ``filter_ast``; return sorted contents."""
    results = await store.search(ns, _EMBED, hybrid_alpha=1.0, filter_ast=filter_ast)
    return sorted(r.chunk.content for r in results)


async def _bm25_contents(store: SurrealDBTemporalStore, ns: UUID, filter_ast: FilterNode) -> list[str]:
    """Run the BM25 channel (search_fulltext) with ``filter_ast``; return sorted contents."""
    results = await store.search_fulltext(ns, "pagerduty", filter_ast=filter_ast)
    return sorted(chunk.content for chunk, _score in results)


# ===========================================================================
# (1) Unbacked system keys FAIL LOUD on BOTH channels — every key.
# ===========================================================================


@pytest.mark.parametrize("key", _UNBACKED_KEYS)
async def test_unbacked_key_raises_on_vector_channel(store: tuple[SurrealDBTemporalStore, UUID], key: str) -> None:
    # The vector channel compiles ``filter_ast`` in raise mode before the cosine
    # scan, so an unbacked system key raises rather than returning a wrong row-set.
    st, ns = store
    with pytest.raises(RecallFilterUnsupportedError, match=_GATE_REASON) as exc:
        await st.search(ns, _EMBED, hybrid_alpha=1.0, filter_ast=_ast(_unbacked_wire(key)))
    # Load-bearing invariant: the gate named the RIGHT key (robust to re-wording).
    assert key in str(exc.value)


@pytest.mark.parametrize("key", _UNBACKED_KEYS)
async def test_unbacked_key_raises_on_bm25_channel(store: tuple[SurrealDBTemporalStore, UUID], key: str) -> None:
    # The BM25 channel (search_fulltext / SearchMode.KEYWORD) compiles in the same
    # raise mode, so the identical fail-loud contract holds there.
    st, ns = store
    with pytest.raises(RecallFilterUnsupportedError, match=_GATE_REASON) as exc:
        await st.search_fulltext(ns, "pagerduty", filter_ast=_ast(_unbacked_wire(key)))
    assert key in str(exc.value)


@pytest.mark.parametrize("key", _STRING_UNBACKED_KEYS)
async def test_unbacked_string_key_exists_raises_on_both_channels(
    store: tuple[SurrealDBTemporalStore, UUID], key: str
) -> None:
    # The gate fires for EVERY operator, $exists included: on a backed key $exists is
    # a constant, but on an unbacked (non-column) key it raises on both channels.
    # (source_timestamp is datetime-typed and DateOps forbids $exists upstream, so the
    # $exists probe is the seven string keys.)
    st, ns = store
    ast = _ast(_unbacked_wire(key, exists=True))
    with pytest.raises(RecallFilterUnsupportedError, match=_GATE_REASON) as exc_vec:
        await st.search(ns, _EMBED, hybrid_alpha=1.0, filter_ast=ast)
    assert key in str(exc_vec.value)
    with pytest.raises(RecallFilterUnsupportedError, match=_GATE_REASON) as exc_bm25:
        await st.search_fulltext(ns, "pagerduty", filter_ast=ast)
    assert key in str(exc_bm25.value)


async def test_unbacked_key_ne_raises_on_both_channels(store: tuple[SurrealDBTemporalStore, UUID]) -> None:
    # The rejection is op-independent: a $ne over an unbacked key raises on both
    # channels too (pre-fix it silently kept every row).
    st, ns = store
    ast = _ast({"source_name": {"$ne": "x"}})
    with pytest.raises(RecallFilterUnsupportedError, match=_GATE_REASON) as exc_vec:
        await st.search(ns, _EMBED, hybrid_alpha=1.0, filter_ast=ast)
    assert "source_name" in str(exc_vec.value)
    with pytest.raises(RecallFilterUnsupportedError, match=_GATE_REASON) as exc_bm25:
        await st.search_fulltext(ns, "pagerduty", filter_ast=ast)
    assert "source_name" in str(exc_bm25.value)


# ===========================================================================
# (2) Positive controls — a backed predicate still narrows on BOTH channels.
# ===========================================================================


async def test_occurred_at_range_narrows_on_both_channels(store: tuple[SurrealDBTemporalStore, UUID]) -> None:
    # occurred_at is a backed datetime column, so a range filter pushes down and
    # keeps only the recent row on both the vector and the BM25 channel.
    st, ns = store
    ast = _ast({"occurred_at": {"$gte": "2026-01-01T00:00:00Z"}})
    assert await _vector_contents(st, ns, ast) == ["recent pagerduty alert"]
    assert await _bm25_contents(st, ns, ast) == ["recent pagerduty alert"]


async def test_metadata_filter_narrows_on_both_channels(store: tuple[SurrealDBTemporalStore, UUID]) -> None:
    # A metadata.<key> filter descends into the FLEXIBLE metadata_ column and keeps
    # only the gold-tier row on both channels — the backed-predicate counterpart to
    # the unbacked fail-loud cases above.
    st, ns = store
    ast = _ast({"metadata.tier": "gold"})
    assert await _vector_contents(st, ns, ast) == ["recent pagerduty alert"]
    assert await _bm25_contents(st, ns, ast) == ["recent pagerduty alert"]


async def test_no_filter_returns_both_rows_on_both_channels(store: tuple[SurrealDBTemporalStore, UUID]) -> None:
    # Control for the controls: with no filter both channels see both rows, so the
    # narrowing above is a real cut, not an empty corpus.
    st, ns = store
    vector = sorted(r.chunk.content for r in await st.search(ns, _EMBED, hybrid_alpha=1.0))
    bm25 = sorted(chunk.content for chunk, _ in await st.search_fulltext(ns, "pagerduty"))
    assert vector == ["ancient pagerduty alert", "recent pagerduty alert"]
    assert bm25 == ["ancient pagerduty alert", "recent pagerduty alert"]
