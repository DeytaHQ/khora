"""Facade document-enumeration wiring on the embedded stack.

The keyset ``list_documents`` facade returns a ``DocumentPage`` walked by
``next_after`` until ``exhausted``. These co-located tests prove the three
things the assembly ticket calls for against a real sqlite_lance namespace:

* a multi-page cursor walk is exactly-once, correctly ordered, and terminates
  only on ``exhausted`` (with ``next_after is None``), and the page is a
  ``Sequence`` at every step;
* the facade result set equals the offset-based ``kb.storage.list_documents``
  for the same ``status`` / ``updated_before`` constraints (closes khora#1527's
  documented workaround);
* a filter narrows the enumeration exactly, and ``occurred_at`` is refused.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Document
from khora.core.models.document import DocumentPage, DocumentStatus
from khora.filter import RecallFilter, RecallFilterValidationError
from khora.filter.ast import parse_to_ast
from tests.test_helpers.document_page_oracle import assert_page_compliant, assert_walk_compliant

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

_BASE = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)


def _ast(wire: dict):
    """The canonical AST for a wire filter — what the oracle recompiles from."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _config(tmp_path: Path):
    from khora.config import KhoraConfig
    from khora.config.schema import SQLiteLanceConfig

    config = KhoraConfig()
    config.storage.backend = "sqlite_lance"
    config.storage.sqlite_lance = SQLiteLanceConfig(
        db_path=str(tmp_path / "k.db"),
        lance_path=str(tmp_path / "k.lance"),
        embedding_dimension=8,
    )
    config.storage.embedding_dimension = 8
    config.llm.embedding_dimension = 8
    return config


async def _seed(kb, row_ns: UUID, specs: list[dict]) -> list[Document]:
    """Write documents directly through the storage tier (no extraction)."""
    docs: list[Document] = []
    for spec in specs:
        doc = Document(
            namespace_id=row_ns,
            content=spec.get("content", "seed"),
            checksum=f"chk-{uuid4().hex}",
            created_at=spec["created_at"],
            updated_at=spec.get("updated_at", spec["created_at"]),
            status=spec.get("status", DocumentStatus.COMPLETED),
            source_type=spec.get("source_type", "library"),
            metadata=spec.get("metadata", {}),
        )
        await kb.storage.create_document(doc)
        docs.append(doc)
    return docs


def _expected_order(docs: list[Document]) -> list[UUID]:
    """The single correct enumeration order: created_at DESC, then id DESC."""
    return [d.id for d in sorted(docs, key=lambda d: (d.created_at, d.id), reverse=True)]


async def test_basic_walk_exactly_once_and_sequence_compat(tmp_path: Path) -> None:
    from khora import Khora

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        row_ns = await kb._resolve_namespace(ns.namespace_id)
        seeded = await _seed(
            kb,
            row_ns,
            [{"created_at": _BASE + timedelta(minutes=i)} for i in range(7)],
        )
        expected = _expected_order(seeded)

        collected: list[UUID] = []
        walked: list[DocumentPage] = []
        pages = 0
        after = None
        while True:
            page = await kb.list_documents(namespace=ns.namespace_id, limit=3, after=after)
            pages += 1
            walked.append(page)
            assert isinstance(page, DocumentPage)
            assert len(page) <= 3
            collected.extend(d.id for d in page)  # Sequence iteration
            if page.exhausted:
                assert page.next_after is None
                break
            assert page.next_after is not None
            after = page.next_after
            assert pages < 20, "walk did not terminate"

        assert pages >= 3, "limit=3 over 7 rows must span multiple pages"
        assert collected == expected, "walk must be ordered and exactly-once"
        assert len(set(collected)) == len(collected), "no document returned twice"
        # The reusable page-level oracle over the same walk: per-page total order,
        # the next_after/exhausted pair, exactly-once, one descending run across the
        # concatenation, and — via ``expected_ids`` — COMPLETENESS against the seed.
        # That last one is the leg that matters: a dropped row never reaches the
        # surface, so nothing else here (nor any per-returned-row predicate) can see
        # it. The point of wiring it in is that the SAME helper is what the
        # property-based walk fuzzer asserts with, so a divergence surfaces now.
        flat = assert_walk_compliant(walked, expected_ids=expected)
        assert [d.id for d in flat] == expected


async def test_resume_from_returned_document(tmp_path: Path) -> None:
    """``after`` also accepts a Document — its (created_at, id) is the cursor."""
    from khora import Khora

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        row_ns = await kb._resolve_namespace(ns.namespace_id)
        seeded = await _seed(kb, row_ns, [{"created_at": _BASE + timedelta(minutes=i)} for i in range(5)])
        expected = _expected_order(seeded)

        first = await kb.list_documents(namespace=ns.namespace_id, limit=2)
        assert [d.id for d in first] == expected[:2]

        # Resume from the last returned Document object rather than next_after.
        rest = await kb.list_documents(namespace=ns.namespace_id, limit=10, after=first[-1])
        assert [d.id for d in rest] == expected[2:]
        assert rest.exhausted


async def test_facade_storage_parity_status_and_updated_before(tmp_path: Path) -> None:
    from khora import Khora

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        row_ns = await kb._resolve_namespace(ns.namespace_id)
        cutoff = _BASE + timedelta(hours=1)
        await _seed(
            kb,
            row_ns,
            [
                {"created_at": _BASE + timedelta(minutes=0), "status": DocumentStatus.COMPLETED, "updated_at": _BASE},
                {
                    "created_at": _BASE + timedelta(minutes=1),
                    "status": DocumentStatus.PENDING,
                    "updated_at": _BASE + timedelta(minutes=1),
                },
                {
                    "created_at": _BASE + timedelta(minutes=2),
                    "status": DocumentStatus.COMPLETED,
                    "updated_at": cutoff + timedelta(minutes=5),
                },
                {
                    "created_at": _BASE + timedelta(minutes=3),
                    "status": DocumentStatus.FAILED,
                    "updated_at": _BASE + timedelta(minutes=3),
                },
            ],
        )

        # status parity
        facade = await kb.list_documents(namespace=ns.namespace_id, status="completed", limit=100)
        offset = await kb.storage.list_documents(row_ns, status="completed", limit=100)
        assert [d.id for d in facade] == [d.id for d in offset]
        assert facade.exhausted

        # status enum object parity (same as the string form)
        facade_enum = await kb.list_documents(namespace=ns.namespace_id, status=DocumentStatus.COMPLETED, limit=100)
        assert [d.id for d in facade_enum] == [d.id for d in offset]

        # updated_before parity
        facade_ub = await kb.list_documents(namespace=ns.namespace_id, updated_before=cutoff, limit=100)
        offset_ub = await kb.storage.list_documents(row_ns, updated_before=cutoff, limit=100)
        assert [d.id for d in facade_ub] == [d.id for d in offset_ub]


async def test_filter_narrows_and_occurred_at_rejected(tmp_path: Path) -> None:
    from khora import Khora

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        row_ns = await kb._resolve_namespace(ns.namespace_id)
        seeded = await _seed(
            kb,
            row_ns,
            [
                {"created_at": _BASE + timedelta(minutes=0), "source_type": "report", "metadata": {"tier": "gold"}},
                {"created_at": _BASE + timedelta(minutes=1), "source_type": "library", "metadata": {"tier": "silver"}},
                {"created_at": _BASE + timedelta(minutes=2), "source_type": "report", "metadata": {"tier": "gold"}},
            ],
        )
        by_id = {d.id: d for d in seeded}

        # System-key filter narrows to the matching rows, still ordered.
        page = await kb.list_documents(namespace=ns.namespace_id, filter={"source_type": "report"}, limit=100)
        assert all(by_id[d.id].source_type == "report" for d in page)
        # Compare ordered lists, not sets: a post-filter ordering regression must fail.
        assert [d.id for d in page] == _expected_order([d for d in seeded if d.source_type == "report"])
        assert isinstance(page.post_filtered_keys, tuple)
        # The oracle recompiles the filter and re-checks it against the RETURNED
        # rows, so it fails on a returned row the filter excludes — independently of
        # what ``post_filtered_keys`` claims about who enforced which leaf.
        assert_page_compliant(page, _ast({"source_type": "report"}))

        # Metadata filter — exercises the in-memory post-filter path.
        gold = await kb.list_documents(namespace=ns.namespace_id, filter={"metadata.tier": "gold"}, limit=100)
        assert [d.id for d in gold] == _expected_order([d for d in seeded if d.metadata.get("tier") == "gold"])
        assert_page_compliant(gold, _ast({"metadata.tier": "gold"}))

        # occurred_at is not enumerable on a document row.
        with pytest.raises(RecallFilterValidationError) as exc_info:
            await kb.list_documents(namespace=ns.namespace_id, filter={"occurred_at": {"$gte": "2020-01-01T00:00:00Z"}})
        assert exc_info.value.errors[0].code == "key_not_enumerable"

        # Unknown status is a hard error, never a silent zero-match query.
        with pytest.raises(ValueError, match="bogus"):
            await kb.list_documents(namespace=ns.namespace_id, status="bogus")


async def test_multistep_page_fill_and_short_page_not_exhausted(tmp_path: Path) -> None:
    """A ``created_at`` filter is post-filtered (not pushed) on sqlite_lance, so
    the coordinator re-steps ``scan_documents`` within a page to skip rejected
    rows; a tiny ``scan_bound`` yields a short, NOT-exhausted page whose cursor
    still completes the set exactly-once."""
    from khora import Khora
    from khora.filter import RecallFilter
    from khora.filter.ast import parse_to_ast

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        row_ns = await kb._resolve_namespace(ns.namespace_id)
        seeded = await _seed(kb, row_ns, [{"created_at": _BASE + timedelta(minutes=i)} for i in range(20)])
        cutoff = _BASE + timedelta(minutes=15)  # keep the 5 newest (minutes 15..19)
        match_order = _expected_order([d for d in seeded if d.created_at >= cutoff])
        assert len(match_order) == 5
        ast = parse_to_ast(RecallFilter.model_validate({"created_at": {"$gte": cutoff.isoformat()}}))

        # created_at is withheld from pushdown on sqlite_lance, so it is the
        # in-memory residual — reported, and correctly narrowing the result.
        probe = await kb.storage.scan_documents_page(row_ns, filter_ast=ast, limit=100, scan_bound=1000)
        assert probe.post_filtered_keys == ("created_at",)
        assert [d.id for d in probe] == match_order
        assert probe.exhausted and probe.next_after is None

        # limit=3 forces a page whose window straddles the cutoff. bound=1000
        # exercises intra-page re-stepping (skip rejected rows); bound=3 exercises
        # the scan-bound-consumed short-page branch. Both complete exactly-once.
        for bound in (1000, 3):
            collected: list[UUID] = []
            walked: list[DocumentPage] = []
            after = None
            exhausted = False
            steps = 0
            while not exhausted:
                steps += 1
                assert steps < 60, "walk did not terminate"
                pg = await kb.storage.scan_documents_page(
                    row_ns, filter_ast=ast, limit=3, scan_bound=bound, after=after
                )
                walked.append(pg)
                assert len(pg) <= 3
                collected.extend(d.id for d in pg)
                exhausted = pg.exhausted
                if not exhausted:
                    assert pg.next_after is not None
                    after = (pg.next_after.created_at, pg.next_after.id)
            assert collected == match_order, f"bound={bound}"
            assert len(set(collected)) == len(collected), f"bound={bound}"
            # Same oracle, on the walk that actually exercises the residual: this
            # filter is post-filtered rather than pushed, and the short-page branch
            # (bound=3) is where the next_after/exhausted pair is easiest to get
            # wrong. ``expected_ids`` is the load-bearing argument — it asserts the
            # walk lost none of the 5 matches while re-stepping past rejected rows,
            # which is precisely what a mis-sized intra-page step would break.
            flat = assert_walk_compliant(walked, ast, expected_ids=match_order)
            assert [d.id for d in flat] == match_order, f"bound={bound}"

        # scan_bound must be positive (guards the next_after/exhausted invariant).
        with pytest.raises(ValueError, match="scan_bound"):
            await kb.storage.scan_documents_page(row_ns, limit=10, scan_bound=0)


async def test_walk_resumes_across_a_created_at_tie(tmp_path: Path) -> None:
    """Page boundaries landing inside a created_at tie block stay exactly-once —
    the id-tiebreak half of the (created_at DESC, id DESC) keyset."""
    from khora import Khora

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        row_ns = await kb._resolve_namespace(ns.namespace_id)
        tie = _BASE
        specs = [{"created_at": tie} for _ in range(5)]  # tie block
        specs.append({"created_at": tie + timedelta(minutes=1)})  # newest, outside tie
        specs.append({"created_at": tie - timedelta(minutes=1)})  # oldest, outside tie
        seeded = await _seed(kb, row_ns, specs)
        expected = _expected_order(seeded)

        collected: list[UUID] = []
        walked: list[DocumentPage] = []
        after = None
        pages = 0
        while True:
            pages += 1
            assert pages < 20
            page = await kb.list_documents(namespace=ns.namespace_id, limit=2, after=after)  # boundary inside tie
            walked.append(page)
            collected.extend(d.id for d in page)
            if page.exhausted:
                break
            after = page.next_after
        assert collected == expected
        assert len(set(collected)) == len(collected)
        # Both directions of the tie hazard, which is why this walk gets both legs:
        # a boundary that RE-SERVED a tie-mate fails the strict total-order check and
        # the exactly-once check, and one that SKIPPED a tie-mate fails only
        # ``expected_ids`` — nothing about a page whose rows are all present and
        # ordered reveals the row that is missing from it.
        flat = assert_walk_compliant(walked, expected_ids=expected)
        assert [d.id for d in flat] == expected


async def test_empty_namespace_and_resume_past_end(tmp_path: Path) -> None:
    from khora import Khora

    async with Khora(_config(tmp_path), run_migrations=True) as kb:
        ns = await kb.create_namespace()
        empty = await kb.list_documents(namespace=ns.namespace_id, limit=10)
        assert list(empty) == [] and empty.exhausted and empty.next_after is None

        row_ns = await kb._resolve_namespace(ns.namespace_id)
        await _seed(kb, row_ns, [{"created_at": _BASE + timedelta(minutes=i)} for i in range(3)])
        full = await kb.list_documents(namespace=ns.namespace_id, limit=10)
        assert full.exhausted and len(full) == 3

        # Resuming from the oldest returned document yields an empty, exhausted page.
        past = await kb.list_documents(namespace=ns.namespace_id, limit=10, after=full[-1])
        assert list(past) == [] and past.exhausted and past.next_after is None
