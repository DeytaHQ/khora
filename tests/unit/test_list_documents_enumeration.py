"""Unit coverage for the document-enumeration facade surface.

Covers the pieces that need no live backend: the ``DocumentPage`` sequence
contract, the ``occurred_at`` enumeration key-scope rejection (structured error
+ asserted copy), the ``status`` enum validation, and the coordinator's
``post_filtered_keys`` derivation. The multi-page cursor walk and facade/storage
parity live in the integration lane (they need a seeded namespace on the
sqlite_lance stack).

The ``post_filtered_keys`` class runs the real coordinator over a real store, and
still needs no service: ``SQLiteRelationalBackend(":memory:")`` builds its own
schema at ``connect()`` with no Alembic chain behind it. That is the cheapest
store that can answer the question honestly — the derivation under test is a set
difference over a compiler's answer, and a mock backend would let the test assert
its own stub's arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from khora.core.models import Document, MemoryNamespace
from khora.core.models.document import DocumentCursor, DocumentPage
from khora.core.models.tenancy import TenancyMode
from khora.filter import CompilerRegistry, RecallFilter, RecallFilterValidationError
from khora.filter.ast import parse_to_ast
from khora.filter.execute import filter_leaf_keys
from khora.khora import _reject_non_enumerable_keys
from khora.storage.backends import sqlite as sqlite_module
from khora.storage.backends.sqlite import SQLiteRelationalBackend, _documents_compile_context
from khora.storage.coordinator import StorageCoordinator
from tests.test_helpers.document_page_oracle import (
    assert_page_compliant,
    assert_walk_compliant,
    assert_walk_matches_expected,
)
from tests.test_helpers.document_scan import WHOLE_SECOND, scan_seed, to_filter_ast, write_document
from tests.test_helpers.document_scan_spy import (
    PROBE_VALUE,
    assert_split_honest,
    force_residual,
    method_sql_log,
)


def _ast(wire: dict) -> object:
    return parse_to_ast(RecallFilter.model_validate(wire))


class TestDocumentPageSequence:
    """DocumentPage is a Sequence[Document]: iterate / len / truthiness / index."""

    def _page(self, docs: list[Document], **kw: object) -> DocumentPage:
        kw.setdefault("next_after", None)
        kw.setdefault("exhausted", True)
        return DocumentPage(docs, **kw)  # type: ignore[arg-type]

    def test_is_sequence(self) -> None:
        assert isinstance(self._page([]), Sequence)

    def test_len_iter_index(self) -> None:
        docs = [Document(content="a"), Document(content="b")]
        page = self._page(docs)
        assert len(page) == 2
        assert list(page) == docs
        assert page[0] is docs[0]
        assert page[-1] is docs[1]
        assert list(page[0:1]) == [docs[0]]

    def test_truthiness_tracks_length(self) -> None:
        # khora-benchmarks-style `if await kb.list_documents(...)` usage.
        assert not self._page([])
        assert self._page([Document(content="x")])

    def test_carries_walk_metadata(self) -> None:
        cur = DocumentCursor(created_at=datetime.now(UTC), id=uuid4())
        page = DocumentPage(
            [Document(content="x")],
            next_after=cur,
            exhausted=False,
            post_filtered_keys=("metadata.tier",),
        )
        assert page.next_after is cur
        assert page.exhausted is False
        assert page.post_filtered_keys == ("metadata.tier",)


class TestOccurredAtRejection:
    """occurred_at is the chunk event-time axis — not enumerable on a document row."""

    @pytest.mark.parametrize(
        "wire",
        [
            {"occurred_at": {"$gte": "2020-01-01T00:00:00Z"}},
            {"$or": [{"occurred_at": {"$gte": "2020-01-01T00:00:00Z"}}, {"source_type": "report"}]},
            {"$not": {"occurred_at": {"$lte": "2020-01-01T00:00:00Z"}}},
        ],
    )
    def test_rejected_with_structured_error(self, wire: dict) -> None:
        with pytest.raises(RecallFilterValidationError) as exc_info:
            _reject_non_enumerable_keys(_ast(wire))
        errors = exc_info.value.errors
        assert len(errors) == 1
        fe = errors[0]
        assert fe.path == "occurred_at"
        assert fe.code == "key_not_enumerable"
        # The nine enumerable keys (the ten system keys minus occurred_at).
        assert fe.allowed is not None and len(fe.allowed) == 9
        assert "occurred_at" not in fe.allowed
        # Copy names both substitutes and refuses to call the axes equivalent.
        assert "source_timestamp" in fe.message
        assert "created_at" in fe.message
        assert "equivalent" in fe.message

    @pytest.mark.parametrize(
        "wire",
        [
            {"source_type": "report"},
            {"created_at": {"$gte": "2020-01-01T00:00:00Z"}},
            {"source_timestamp": {"$gte": "2020-01-01T00:00:00Z"}},
            {"metadata.tier": "gold"},
            {"title": {"$in": ["a", "b"]}},
        ],
    )
    def test_enumerable_keys_pass(self, wire: dict) -> None:
        # Does not raise.
        _reject_non_enumerable_keys(_ast(wire))


# The shapes the page-level split is checked over. Each carries three things: the
# wire filter, the residual the coordinator must REPORT, and an independent
# plain-Python matcher over the seeded rows giving the id set the walk must RETURN.
#
# The matcher is hand-written arithmetic on purpose. Deriving the expectation with
# ``compile_python`` would reproduce the very predicate the coordinator applies, so
# the comparison could not fail — the tautology this triple exists to escape.
#
# The residual is pinned as well as recomputed against the compiler, because a
# recompute alone is satisfied by two computations agreeing on the wrong answer,
# including the degenerate all-empty case that makes every comparison ``() == ()``.
#
# The date bound is ``WHOLE_SECOND`` rather than some far-past date so it actually
# BITES: the seed puts one row a second below the tie instant, so a ``$gte`` cutoff
# there keeps 5 of 6. A bound every row satisfies would make the completeness check
# "returns everything", which a dropped row would still fail but which cannot catch
# an over-broad one.
_CUTOFF = WHOLE_SECOND.isoformat()

_PAGE_SPLIT_SHAPES: dict[str, tuple[dict[str, Any], tuple[str, ...], Any]] = {
    "no_residual": (
        {"source_type": {"$eq": PROBE_VALUE}},
        (),
        lambda r: r["source_type"] == PROBE_VALUE,
    ),
    "date_key_residual": (
        {"created_at": {"$gte": _CUTOFF}},
        ("created_at",),
        lambda r: r["created_at"] >= WHOLE_SECOND,
    ),
    "mixed": (
        {"source_type": {"$eq": PROBE_VALUE}, "created_at": {"$gte": _CUTOFF}},
        ("created_at",),
        lambda r: r["source_type"] == PROBE_VALUE and r["created_at"] >= WHOLE_SECOND,
    ),
    "or_wholly_deferred": (
        {"$or": [{"created_at": {"$gte": _CUTOFF}}, {"source_type": {"$eq": PROBE_VALUE}}]},
        ("created_at", "source_type"),
        lambda r: r["created_at"] >= WHOLE_SECOND or r["source_type"] == PROBE_VALUE,
    ),
}


class TestPagePostFilteredKeys:
    """``DocumentPage.post_filtered_keys`` is the leaf keys the backend did NOT push.

    The coordinator derives it as ``sorted(filter_leaf_keys(ast) - consumed_keys)``
    unioned across the steps of a page. Two failures it has to be pinned against,
    and neither shows up on the row set (the coordinator re-checks the full AST
    either way, so the rows are right in both):

    * **understating the residual** — a key the backend never pushed, omitted from
      the report. A caller sizing its own over-fetch off this signal, or auditing
      which leaves SQL enforced, is told SQL owns a predicate it does not.
    * **overstating it** — a pushed key listed as residual, which makes the
      pushdown look useless and invites a "fix" that removes it.

    Recomputed with the same compiler and context the store scans with, so a
    difference can only be the coordinator's own arithmetic.
    """

    @pytest.fixture
    async def store(self):
        backend = SQLiteRelationalBackend(":memory:")
        await backend.connect()
        try:
            yield backend
        finally:
            await backend.disconnect()

    @pytest.fixture
    async def seeded(self, store):
        """A coordinator over a varied six-row namespace.

        Returns ``(coordinator, ns_id, rows)``. ``rows`` is the seed as plain dicts
        — the INDEPENDENT record of what was written, which the completeness
        matchers evaluate against. Nothing here reads it back out of the store.
        """
        nid = uuid4()
        ns = await store.create_namespace(MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED))
        seed = scan_seed(6)
        rows: list[dict[str, Any]] = []
        for i, (doc_id, created_at) in enumerate(seed.writes):
            fields = {"title": f"doc-{i}", "source_type": PROBE_VALUE if i % 2 == 0 else "library"}
            await write_document(store, ns.id, doc_id, created_at, **fields)
            rows.append({"id": doc_id, "created_at": created_at, **fields})
        return StorageCoordinator(relational=store), ns.id, rows

    @pytest.mark.parametrize(
        ("wire", "expected", "matcher"), _PAGE_SPLIT_SHAPES.values(), ids=_PAGE_SPLIT_SHAPES.keys()
    )
    async def test_page_reports_its_residual_and_returns_exactly_the_matches(
        self, seeded, wire, expected, matcher
    ) -> None:
        coordinator, namespace_id, rows = seeded
        ast = to_filter_ast(wire)
        expected_ids = [r["id"] for r in rows if matcher(r)]
        assert 0 < len(expected_ids) < len(rows), "the shape must narrow, or completeness proves nothing"

        page = await coordinator.scan_documents_page(namespace_id, filter_ast=ast, limit=100)

        # 1. The reported residual, pinned and recomputed against the compiler.
        assert page.post_filtered_keys == expected
        compiler = CompilerRegistry.get("relational.sqlite", "documents")
        consumed = compiler(ast, _documents_compile_context()).consumed_keys
        assert page.post_filtered_keys == tuple(sorted(filter_leaf_keys(ast) - consumed))
        # 2. COMPLETENESS — the assertion that can actually fail. Compared against
        #    the hand-written matcher over the seed, never against another
        #    enumeration call, and never against ``compile_python`` (which is the
        #    predicate the coordinator itself applied).
        assert_walk_matches_expected(page, expected_ids)
        # 3. Order + the walk-control pair. Soundness rides along and is
        #    tautological here by construction; see the helper's docstring.
        assert_page_compliant(page, ast)

    async def test_no_filter_reports_no_residual(self, seeded) -> None:
        """Empty, and specifically not "every key, none of them pushed"."""
        coordinator, namespace_id, rows = seeded

        page = await coordinator.scan_documents_page(namespace_id, limit=100)

        assert page.post_filtered_keys == ()
        assert_walk_matches_expected(page, [r["id"] for r in rows])
        assert_page_compliant(page)

    async def test_a_residual_only_filter_still_narrows_the_page(self, seeded) -> None:
        """The residual is enforced by the coordinator's post-filter, on rows.

        ``created_at`` is unpushable on this store, so SQL narrows nothing and the
        whole filter is the coordinator's to enforce. A page that reported the
        residual honestly and then *skipped* it would return the full corpus, and
        every reporting assertion above would still pass. Narrowed to a SINGLE row
        so the completeness comparison is at its tightest.
        """
        coordinator, namespace_id, rows = seeded
        # The seed puts exactly one row above its tie instant (see ``scan_seed``),
        # so a cutoff one second past ``WHOLE_SECOND`` keeps that row and no other.
        cutoff = WHOLE_SECOND + timedelta(seconds=1)
        ast = to_filter_ast({"created_at": {"$gte": cutoff.isoformat()}})
        expected_ids = [r["id"] for r in rows if r["created_at"] >= cutoff]
        assert len(expected_ids) == 1, "the seed must put exactly one row above the cutoff"

        page = await coordinator.scan_documents_page(namespace_id, filter_ast=ast, limit=100)

        assert page.post_filtered_keys == ("created_at",)
        assert_walk_matches_expected(page, expected_ids)
        assert_page_compliant(page, ast)

    async def test_a_multi_page_walk_loses_no_match_across_cursor_boundaries(self, seeded) -> None:
        """Completeness across a REAL walk, with the page boundary inside a tie block.

        The single-page tests above cannot see a cursor bug: one page never resumes.
        Here ``limit=2`` over a 5-row match set forces three pages whose boundaries
        land inside the seed's ``created_at`` tie block, which is where a cursor that
        mis-serializes its timestamp or its id silently skips a tie-mate — the exact
        failure no per-returned-row assertion can detect, because the skipped row
        simply never appears.
        """
        coordinator, namespace_id, rows = seeded
        cutoff = WHOLE_SECOND
        ast = to_filter_ast({"created_at": {"$gte": cutoff.isoformat()}})
        expected_ids = [r["id"] for r in rows if r["created_at"] >= cutoff]
        assert len(expected_ids) == 5, "5 of 6 rows sit at or above the tie instant"

        walked = []
        after = None
        while True:
            page = await coordinator.scan_documents_page(namespace_id, filter_ast=ast, limit=2, after=after)
            walked.append(page)
            assert len(walked) < 10, "walk did not terminate"
            if page.exhausted:
                break
            after = (page.next_after.created_at, page.next_after.id)

        assert len(walked) >= 3, "limit=2 over 5 matches must span multiple pages"
        assert_walk_compliant(walked, ast, expected_ids=expected_ids)

    async def test_the_same_filter_both_ways_round_reaches_the_same_rows(self, seeded, store, monkeypatch) -> None:
        """One AST, two splits, one answer — with the seam watched in both modes.

        The end-to-end form of the split contract, and the only place both modes of
        it are compared *for the same filter*. ``source_type`` normally pushes here,
        so the natural mode reports no residual and the operand is bound in the SQL;
        ``force_residual`` drops the key from this store's ``field_mapping`` (the
        pushdown whitelist), and the SAME filter must then report the key as the
        residual, bind nothing for it, and still return exactly the same three rows
        because the coordinator's post-filter picked it up.

        A page that reported the residual honestly and then failed to enforce it
        would return all six rows here; one that enforced it in SQL while reporting
        it deferred would bind the value and fail the seam check. The two modes
        pin each other, which is what neither can do alone.
        """
        coordinator, namespace_id, rows = seeded
        ast = to_filter_ast({"source_type": {"$eq": PROBE_VALUE}})
        expected_ids = [r["id"] for r in rows if r["source_type"] == PROBE_VALUE]
        assert len(expected_ids) == 3, "the filter must narrow, or neither mode proves anything"

        with method_sql_log(store._conn, "execute") as pushed_log:  # noqa: SLF001 — no public driver seam
            pushed_page = await coordinator.scan_documents_page(namespace_id, filter_ast=ast, limit=100)
        assert pushed_page.post_filtered_keys == ()
        assert_split_honest(pushed_log, pushed_values=[PROBE_VALUE])

        force_residual(monkeypatch, sqlite_module)
        with method_sql_log(store._conn, "execute") as residual_log:  # noqa: SLF001
            residual_page = await coordinator.scan_documents_page(namespace_id, filter_ast=ast, limit=100)
        assert residual_page.post_filtered_keys == ("source_type",)
        assert_split_honest(residual_log, residual_values=[PROBE_VALUE])

        # Both modes hit the independently-known match set, and each other.
        assert_walk_matches_expected(pushed_page, expected_ids)
        assert_walk_matches_expected(residual_page, expected_ids)
        assert [d.id for d in residual_page] == [d.id for d in pushed_page]
        assert_page_compliant(pushed_page, ast)
        assert_page_compliant(residual_page, ast)
