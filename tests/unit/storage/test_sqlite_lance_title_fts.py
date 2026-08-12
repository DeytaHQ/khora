"""#1574 — chunk ``title`` folded into the embedded (sqlite_lance) FTS index.

``khora_chunks.title`` has been a denormalized column since migration 041, but
``khora_chunks_fts`` only ever indexed ``content``.  A chunk whose *title* was
the only place a term appeared was invisible to the lexical channel — the
"0-hit repro" this module pins first.

Four groups, in the order the change has to hold up:

1. :class:`TestTitleIsIndexed` — the repro probe.  Body tokens AND title tokens
   both find the chunk.
2. :class:`TestTriggerSync` — the FTS mirror tracks ``title`` through UPDATE and
   DELETE.  The delete leg is the one worth having: FTS5 external-content
   ``'delete'`` rows must carry the SAME column list the index was built with,
   or terms are left behind and the index silently rots.  Asserted twice — no
   match, and a clean ``'integrity-check'``.
3. :class:`TestTitleWeight` — ``title_weight`` actually reorders results, and
   ``1.0`` emits the bare, byte-identical-to-before ``bm25()`` call.
4. :class:`TestLegacyOneColumnShape` — a database built before this change keeps
   its 1-column FTS table (the DDL is ``IF NOT EXISTS``; nothing alters it), so
   ``title_weight`` is inert there.  Verified on SQLite 3.53.4: a weighted
   ``bm25(fts, 1.0, w)`` against that shape does **not** raise — SQLite ignores
   the surplus weight argument.  The store therefore cannot rely on an
   exception; it probes the shape and warns.  These tests pin the probe, the
   no-crash return, and the warning, and deliberately assert NO exception.

Embedded only: aiosqlite + LanceDB in ``tmp_path``.  No Docker, no LLM.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:  # pragma: no cover - optional extra
    _HAS_EMBEDDED = False

from loguru import logger

from khora.core.temporal import TemporalChunk
from khora.storage.backends.sqlite_lance.connection import (
    EmbeddedStorageHandle,
    EmbeddedStorageHandleConfig,
)
from khora.storage.temporal.sqlite_lance import SQLiteLanceTemporalStore

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

EMBED_DIM = 8

#: The field name from the #1574 repro.  Kept verbatim because it is a
#: tokenizer regression guard as much as a fixture: unicode61 (which ``porter``
#: wraps) treats ``_`` as a separator, so this must index as four tokens —
#: ``floor``, ``panel`` (stemmed), ``dimens`` (stemmed), ``20260213``.  A
#: tokenizer change that swallowed the underscores would make the whole title
#: one unmatchable token and every assertion below would fail loudly.
TITLE = "Floor Panels_Dimensioned_20260213"

#: Body text sharing NO vocabulary with :data:`TITLE`.  Load-bearing: if the
#: body contained title words, a title hit and a content hit would be
#: indistinguishable and the repro probe would pass without the change.
BODY = "the assembly drawing revision notes for the north wing"

#: A replacement title with disjoint vocabulary, for the UPDATE-trigger leg.
NEW_TITLE = "Ceiling Tiles_Revised_20260314"


def _chunk(namespace_id: UUID, *, content: str, title: str | None) -> TemporalChunk:
    return TemporalChunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=content,
        embedding=[0.0] * (EMBED_DIM - 1) + [1.0],
        occurred_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
        title=title,
    )


async def _make_handle(tmp_path: Path) -> EmbeddedStorageHandle:
    handle = EmbeddedStorageHandle(
        EmbeddedStorageHandleConfig(
            db_path=str(tmp_path / "khora.db"),
            lance_path=str(tmp_path / "khora.lance"),
            embedding_dimension=EMBED_DIM,
        )
    )
    await handle.connect()
    return handle


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SQLiteLanceTemporalStore]:
    """A connected temporal store on a throwaway embedded database.

    ``khora_chunks`` is runtime-managed (the store's own ``connect()`` issues
    its DDL), so no Alembic run is needed here.
    """
    handle = await _make_handle(tmp_path)
    temporal = SQLiteLanceTemporalStore(handle)
    await temporal.connect()
    try:
        yield temporal
    finally:
        with contextlib.suppress(Exception):
            await temporal.disconnect()
        with contextlib.suppress(Exception):
            await handle.disconnect()


async def _contents(store: SQLiteLanceTemporalStore, ns: UUID, query: str, **kwargs) -> list[str]:
    """Chunk contents returned by the lexical channel, best-ranked first."""
    return [chunk.content for chunk, _score in await store.search_fulltext(ns, query, limit=10, **kwargs)]


# ---------------------------------------------------------------------------
# 1. Repro probe — the 0-hit bug
# ---------------------------------------------------------------------------


class TestTitleIsIndexed:
    async def test_body_and_title_tokens_both_match(self, store: SQLiteLanceTemporalStore) -> None:
        """The #1574 repro: title vocabulary finds the chunk.

        Three queries against ONE chunk. The body query is the control — it
        passed before the change and must still pass. The two title queries are
        the bug: each returned zero rows because ``khora_chunks_fts`` indexed
        only ``content``.
        """
        ns = uuid4()
        await store.create_chunk(_chunk(ns, content=BODY, title=TITLE))

        # Control: content is still indexed (no regression on the old path).
        assert await _contents(store, ns, "assembly drawing") == [BODY]

        # The repro. Words that exist ONLY in the title.
        assert await _contents(store, ns, "floor panels dimensioned") == [BODY], (
            "title words must reach the FTS index (#1574 0-hit repro)"
        )
        # The bare numeric token, separately: it is the half a naive tokenizer
        # would lose to the underscores around it.
        assert await _contents(store, ns, "20260213") == [BODY]

    async def test_title_only_chunk_is_reachable(self, store: SQLiteLanceTemporalStore) -> None:
        """A title query must not just match *something* — it must match the
        chunk that carries the title, and not its title-less neighbour."""
        ns = uuid4()
        titled = _chunk(ns, content=BODY, title=TITLE)
        untitled = _chunk(ns, content="wholly unrelated procurement paperwork", title=None)
        await store.create_chunks_batch([titled, untitled])

        assert await _contents(store, ns, "floor panels") == [BODY]

    async def test_title_search_is_namespace_scoped(self, store: SQLiteLanceTemporalStore) -> None:
        """Indexing the title must not leak it across namespaces."""
        mine, theirs = uuid4(), uuid4()
        await store.create_chunk(_chunk(theirs, content=BODY, title=TITLE))

        assert await _contents(store, mine, "floor panels dimensioned") == []


# ---------------------------------------------------------------------------
# 2. Trigger sync — UPDATE and DELETE keep the 2-column mirror honest
# ---------------------------------------------------------------------------


async def _fts_integrity_ok(store: SQLiteLanceTemporalStore) -> bool:
    """True when the FTS index agrees with the ``khora_chunks`` rows it mirrors.

    ``rank = 1`` is load-bearing and not decoration. The bare
    ``VALUES('integrity-check')`` form checks only that the index is internally
    well-formed, and an asymmetric ``'delete'`` leaves a *well-formed* index
    that merely disagrees with the content table — verified: against a 2-column
    index with a 1-column delete trigger, the bare form returns OK while the
    ``rank = 1`` form raises ``database disk image is malformed``. Only the
    latter re-derives terms from ``khora_chunks`` and compares.

    Kept alongside the behavioural "no longer matches" assertions rather than
    instead of them: those catch residue on the deleted row, this catches
    residue anywhere in the table.
    """
    try:
        await store._sqlite.execute("INSERT INTO khora_chunks_fts(khora_chunks_fts, rank) VALUES('integrity-check', 1)")
    except Exception:
        return False
    return True


class TestTriggerSync:
    async def test_update_swaps_the_indexed_title(self, store: SQLiteLanceTemporalStore) -> None:
        """AFTER UPDATE deletes the OLD title terms and inserts the new ones.

        Written as a raw ``UPDATE`` because that is what the trigger is for —
        the store has no update-chunk method, and any future one would go
        through the same statement.
        """
        ns = uuid4()
        chunk = _chunk(ns, content=BODY, title=TITLE)
        await store.create_chunk(chunk)
        assert await _contents(store, ns, "floor panels") == [BODY]

        await store._sqlite.execute(
            "UPDATE khora_chunks SET title = ? WHERE id = ?",
            (NEW_TITLE, chunk.id.hex),
        )
        await store._sqlite.commit()

        # Old title terms are gone ...
        assert await _contents(store, ns, "floor panels") == []
        assert await _contents(store, ns, "20260213") == []
        # ... the new ones are present ...
        assert await _contents(store, ns, "ceiling tiles") == [BODY]
        assert await _contents(store, ns, "20260314") == [BODY]
        # ... and content is untouched by the title swap.
        assert await _contents(store, ns, "assembly drawing") == [BODY]
        assert await _fts_integrity_ok(store)

    async def test_delete_removes_both_columns_symmetrically(self, store: SQLiteLanceTemporalStore) -> None:
        """AFTER DELETE must pass the same ``(content, title)`` pair the insert did.

        An external-content FTS5 table stores no copy of the row; the ``'delete'``
        command re-tokenizes the values it is handed to work out which terms to
        remove. Hand it fewer columns than the index has and the title terms are
        never removed.
        """
        ns = uuid4()
        chunk = _chunk(ns, content=BODY, title=TITLE)
        await store.create_chunk(chunk)

        assert await store.delete_chunk(chunk.id, ns) is True

        assert await _contents(store, ns, "floor panels dimensioned") == []
        assert await _contents(store, ns, "assembly drawing") == []
        assert await _fts_integrity_ok(store), (
            "the AFTER DELETE trigger left title terms in khora_chunks_fts — "
            "its column list must match the CREATE VIRTUAL TABLE column list"
        )

    async def test_delete_leaves_sibling_rows_searchable(self, store: SQLiteLanceTemporalStore) -> None:
        """Deleting one chunk must not disturb another's title terms."""
        ns = uuid4()
        doomed = _chunk(ns, content=BODY, title=TITLE)
        keeper = _chunk(ns, content="separate body prose entirely", title=NEW_TITLE)
        await store.create_chunks_batch([doomed, keeper])

        await store.delete_chunk(doomed.id, ns)

        assert await _contents(store, ns, "floor panels") == []
        assert await _contents(store, ns, "ceiling tiles") == ["separate body prose entirely"]
        assert await _fts_integrity_ok(store)


# ---------------------------------------------------------------------------
# 3. title_weight — reordering, and the untouched default
# ---------------------------------------------------------------------------

#: Filler that shares no vocabulary with the query. The two fixture chunks below
#: differ in length on purpose: at the neutral weight the SHORTER body-only
#: chunk must win on BM25's length normalization, so a weight that flips the
#: order is doing real work rather than confirming a pre-existing tie.
_FILLER = " ".join(f"pad{i}" for i in range(40))
_EXTRA_FILLER = " ".join(f"extra{i}" for i in range(20))

#: Query terms live ONLY in this chunk's title; its body is the longer one.
_TITLED_BODY = f"{_FILLER} {_EXTRA_FILLER}"
#: Query terms live ONLY in this chunk's body, which is shorter.
_BODY_ONLY = f"floor panels {_FILLER}"


class TestTitleWeight:
    async def _seed_pair(self, store: SQLiteLanceTemporalStore) -> UUID:
        ns = uuid4()
        await store.create_chunks_batch(
            [
                _chunk(ns, content=_TITLED_BODY, title="Floor Panels"),
                _chunk(ns, content=_BODY_ONLY, title=None),
            ]
        )
        return ns

    async def test_neutral_weight_ranks_the_body_match_first(self, store: SQLiteLanceTemporalStore) -> None:
        """The baseline the reordering test below is measured against.

        With every column weighted 1.0 the shorter body-only chunk outscores the
        longer title-only one on BM25's length normalization. This is not the
        pre-change behavior — before the change the titled chunk would not have
        matched at all — it is the fixture's neutral-weight order. If it ever
        stops holding, the flip asserted next proves nothing, so it is pinned
        rather than assumed.
        """
        ns = await self._seed_pair(store)
        assert await _contents(store, ns, "floor panels") == [_BODY_ONLY, _TITLED_BODY]

    async def test_raised_weight_floats_the_titled_chunk(self, store: SQLiteLanceTemporalStore) -> None:
        """``title_weight > 1`` reorders the SAME two chunks, title first."""
        ns = await self._seed_pair(store)
        assert await _contents(store, ns, "floor panels", title_weight=4.0) == [_TITLED_BODY, _BODY_ONLY]

    async def test_weight_is_the_only_difference(self, store: SQLiteLanceTemporalStore) -> None:
        """Both weights return the same result SET — only the order differs.

        Guards against a weighted call accidentally becoming a filter.
        """
        ns = await self._seed_pair(store)
        neutral = await _contents(store, ns, "floor panels")
        weighted = await _contents(store, ns, "floor panels", title_weight=4.0)
        assert sorted(neutral) == sorted(weighted)
        assert neutral != weighted

    def test_default_weight_emits_the_bare_bm25_call(self, store: SQLiteLanceTemporalStore) -> None:
        """``title_weight=1.0`` must produce the pre-#1574 SQL verbatim.

        The default path is the one every existing deployment takes, so the
        scoring expression has to be byte-identical rather than merely
        equivalent — an all-ones weight vector would compute the same numbers
        but is a different string, and this is the cheapest place to pin it.
        """
        assert store._bm25_expr(1.0) == "bm25(khora_chunks_fts)"

    def test_non_default_weight_emits_the_per_column_call(self, store: SQLiteLanceTemporalStore) -> None:
        """Content is column 0 (weight pinned at 1.0), title is column 1."""
        assert store.fts_has_title is True
        assert store._bm25_expr(2.0) == "bm25(khora_chunks_fts, 1.0, 2)"
        assert store._bm25_expr(0.5) == "bm25(khora_chunks_fts, 1.0, 0.5)"

    def test_rendered_weight_carries_no_injection_surface(self, store: SQLiteLanceTemporalStore) -> None:
        """The weight is inlined, not bound — so its rendering IS the argument.

        FTS5 auxiliary-function arguments must be literals, which rules out a
        bind parameter. What makes that safe is that the value is a
        Pydantic-clamped ``float`` rendered through ``{:g}``, which can only
        emit digits, ``.``, ``e`` and a sign. Checked across the config range's
        ends and a fractional value rather than argued in a comment.
        """
        allowed = set("0123456789.e+-")
        for weight in (0.0, 0.25, 3.0, 10.0):
            expr = store._bm25_expr(weight)
            rendered = expr.removeprefix("bm25(khora_chunks_fts, 1.0, ").removesuffix(")")
            assert set(rendered) <= allowed, expr

    async def test_zero_weight_mutes_the_title_channel(self, store: SQLiteLanceTemporalStore) -> None:
        """``title_weight=0.0`` is a legal config value (Pydantic's floor).

        A title-only match still MATCHes — the predicate is weight-independent —
        but contributes nothing to the score. Worth pinning because "0 disables
        the column" and "0 hides the row" are easy to confuse, and the recall
        semantics differ sharply.
        """
        ns = uuid4()
        await store.create_chunk(_chunk(ns, content=_TITLED_BODY, title="Floor Panels"))

        results = await store.search_fulltext(ns, "floor panels", limit=10, title_weight=0.0)
        assert [c.content for c, _ in results] == [_TITLED_BODY]
        assert results[0][1] == pytest.approx(0.0, abs=1e-12), "a zeroed column contributes nothing to bm25"


# ---------------------------------------------------------------------------
# 4. Legacy 1-column FTS shape — degrade, do not crash
# ---------------------------------------------------------------------------

#: The pre-#1574 embedded schema, replayed verbatim. Recreating it by hand is
#: the only way to get one: the live DDL is ``CREATE ... IF NOT EXISTS``, so a
#: database built before the change is never migrated to the 2-column shape.
_LEGACY_FTS_DDL = (
    "DROP TRIGGER IF EXISTS khora_chunks_ai",
    "DROP TRIGGER IF EXISTS khora_chunks_ad",
    "DROP TRIGGER IF EXISTS khora_chunks_au",
    "DROP TABLE IF EXISTS khora_chunks_fts",
    """
    CREATE VIRTUAL TABLE khora_chunks_fts USING fts5(
        content, content='khora_chunks', content_rowid='rowid', tokenize='porter'
    )
    """,
    """
    CREATE TRIGGER khora_chunks_ai AFTER INSERT ON khora_chunks BEGIN
        INSERT INTO khora_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
    """
    CREATE TRIGGER khora_chunks_ad AFTER DELETE ON khora_chunks BEGIN
        INSERT INTO khora_chunks_fts(khora_chunks_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    END
    """,
    """
    CREATE TRIGGER khora_chunks_au AFTER UPDATE ON khora_chunks BEGIN
        INSERT INTO khora_chunks_fts(khora_chunks_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
        INSERT INTO khora_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
)


async def _downgrade_fts_to_legacy(handle: EmbeddedStorageHandle) -> None:
    for statement in _LEGACY_FTS_DDL:
        await handle.sqlite.execute(statement)
    await handle.sqlite.commit()


@pytest.fixture
async def legacy_store(tmp_path: Path) -> AsyncIterator[SQLiteLanceTemporalStore]:
    """A store whose live FTS table has the old 1-column shape.

    Built the way a real one is: the current DDL runs first (creating
    ``khora_chunks``), the FTS table and its triggers are swapped back to the
    legacy shape, and then a FRESH store connects — so its probe sees exactly
    what a pre-change database presents. The second ``connect()`` re-runs the
    ``IF NOT EXISTS`` DDL and, correctly, changes nothing.
    """
    handle = await _make_handle(tmp_path)
    bootstrap = SQLiteLanceTemporalStore(handle)
    await bootstrap.connect()
    await _downgrade_fts_to_legacy(handle)

    temporal = SQLiteLanceTemporalStore(handle)
    await temporal.connect()
    try:
        yield temporal
    finally:
        with contextlib.suppress(Exception):
            await temporal.disconnect()
        with contextlib.suppress(Exception):
            await handle.disconnect()


class TestLegacyOneColumnShape:
    async def test_probe_reports_no_title_column(self, legacy_store: SQLiteLanceTemporalStore) -> None:
        assert legacy_store.fts_has_title is False

    async def test_title_is_genuinely_not_indexed(self, legacy_store: SQLiteLanceTemporalStore) -> None:
        """The degradation is real, not cosmetic: there is nothing to weight.

        This is why the store cannot treat SQLite's tolerance of the surplus
        weight argument as "it worked".
        """
        ns = uuid4()
        await legacy_store.create_chunk(_chunk(ns, content=BODY, title=TITLE))

        assert await _contents(legacy_store, ns, "assembly drawing") == [BODY]
        assert await _contents(legacy_store, ns, "floor panels dimensioned") == []

    async def test_weighted_search_returns_rows_without_raising(self, legacy_store: SQLiteLanceTemporalStore) -> None:
        """A non-neutral weight on the legacy shape degrades to content-only.

        Deliberately NOT a ``pytest.raises``: measured on SQLite 3.53.4, a
        weighted ``bm25(fts, 1.0, w)`` against a 1-column table silently ignores
        the extra argument. The guard exists to make the no-op *visible*, not to
        prevent a crash — so the contract under test is "results, no exception".
        """
        ns = uuid4()
        await legacy_store.create_chunk(_chunk(ns, content=BODY, title=TITLE))

        results = await legacy_store.search_fulltext(ns, "assembly drawing", limit=10, title_weight=2.0)

        assert [chunk.content for chunk, _ in results] == [BODY]

    async def test_fallback_expression_is_the_bare_call(self, legacy_store: SQLiteLanceTemporalStore) -> None:
        """Even asked for a weight, the legacy shape gets the shape-agnostic SQL."""
        assert legacy_store._bm25_expr(2.0) == "bm25(khora_chunks_fts)"

    async def test_fallback_warns_once_then_throttles(self, legacy_store: SQLiteLanceTemporalStore) -> None:
        """ADR-001 throttle: WARNING on the first ignored weight, DEBUG after.

        Both halves matter. Without the warning the no-op is invisible; without
        the throttle a per-query warning would flood the log of any deployment
        that sets the knob and never upgraded its database.
        """
        captured: list[str] = []
        sink = logger.add(lambda message: captured.append(str(message)), level="WARNING")
        try:
            legacy_store._bm25_expr(2.0)
            legacy_store._bm25_expr(2.0)
            legacy_store._bm25_expr(3.0)
        finally:
            logger.remove(sink)

        assert len(captured) == 1, f"expected exactly one WARNING, got {captured!r}"
        assert "title_weight=2" in captured[0]
        assert "khora_chunks_fts lacks the 'title' column" in captured[0]

    async def test_neutral_weight_never_warns(self, legacy_store: SQLiteLanceTemporalStore) -> None:
        """The default asks for nothing, so nothing is being ignored.

        A warning here would fire on every recall of every un-upgraded embedded
        deployment, none of which requested title weighting.
        """
        captured: list[str] = []
        sink = logger.add(lambda message: captured.append(str(message)), level="WARNING")
        try:
            assert legacy_store._bm25_expr(1.0) == "bm25(khora_chunks_fts)"
        finally:
            logger.remove(sink)

        assert captured == []
