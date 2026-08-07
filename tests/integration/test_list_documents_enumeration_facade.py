"""Facade document-enumeration wiring on the embedded stack (DYT-5537).

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
from khora.filter import RecallFilterValidationError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

_BASE = datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)


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
        pages = 0
        after = None
        while True:
            page = await kb.list_documents(namespace=ns.namespace_id, limit=3, after=after)
            pages += 1
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
        assert {d.id for d in page} == {d.id for d in seeded if d.source_type == "report"}
        assert isinstance(page.post_filtered_keys, tuple)

        # Metadata filter — exercises the in-memory post-filter path.
        gold = await kb.list_documents(namespace=ns.namespace_id, filter={"metadata.tier": "gold"}, limit=100)
        assert {d.id for d in gold} == {d.id for d in seeded if d.metadata.get("tier") == "gold"}

        # occurred_at is not enumerable on a document row.
        with pytest.raises(RecallFilterValidationError) as exc_info:
            await kb.list_documents(namespace=ns.namespace_id, filter={"occurred_at": {"$gte": "2020-01-01T00:00:00Z"}})
        assert exc_info.value.errors[0].code == "key_not_enumerable"

        # Unknown status is a hard error, never a silent zero-match query.
        with pytest.raises(ValueError, match="bogus"):
            await kb.list_documents(namespace=ns.namespace_id, status="bogus")
